"""
yy_mask_gt_field_mlp.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
在 field 特征 + 三视图图像监督下预测每个面片 0/1 标签。
batch_size 固定 1，梯度累积等效放大批量。

【本版改动】
1. 不再使用 `torch.amp.GradScaler(device_type=...)`，
   统一回退到向后兼容的 `torch.cuda.amp.*` API。
2. `autocast()` 同步修改；
   这样无论 PyTorch 1.13 / 2.0 / 2.1 / 2.2 都能跑。
"""

import os
import re
import glob
import argparse
import random
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ------------------------- argparse & seed ------------------------- #

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base_dir",    type=str, required=True,
                   help="根目录 /hy-tmp/PartField_Sketch_simpleMLP")
    p.add_argument("--feature_dim", type=int, default=448)
    p.add_argument("--img_size",    type=int, default=512)
    p.add_argument("--hidden_dim",  type=int, default=256)
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--accum",       type=int, default=4,
                   help="梯度累积步数 (等效 batch=accum)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pin_mem",     action="store_true", default=True)
    p.add_argument("--prefetch",    type=int, default=4)
    return p.parse_args()

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------------- 数据扫描 ------------------------- #

Sample = Tuple[str, str, str, str, str, str, str]    # mid, cat, jid, view, asp, dep, nor

def collect_samples(img_dir: str, feature_dir: str) -> List[Sample]:
    arrow_imgs = glob.glob(os.path.join(img_dir, "*_arrow-sketch_*_joint_*.png"))
    samples: List[Sample] = []

    for asp in arrow_imgs:
        fname = os.path.basename(asp)
        m = re.match(r"(.+?)_(\d+)_arrow-sketch_(.+?)_joint_(\d+)\.png", fname)
        if not m:
            continue
        cat, mid, view, jid = m.groups()

        feat_path  = os.path.join(feature_dir, mid, "feature", f"{mid}.npy")
        lbl_path   = os.path.join(feature_dir, mid, "yy_visualization", f"labels_{jid}.npy")
        dep_path   = os.path.join(img_dir, f"{cat}_{mid}_depth_{view}.png")
        nor_path   = os.path.join(img_dir, f"{cat}_{mid}_normal_{view}.png")

        if not (os.path.exists(feat_path) and os.path.exists(lbl_path) and
                os.path.exists(dep_path) and os.path.exists(nor_path)):
            continue

        samples.append((mid, cat, jid, view, asp, dep_path, nor_path))

    print(f"扫描完成，找到 {len(samples)} 条可用样本。")
    if samples:
        mid0, _, jid0, _, *_ = samples[0]
        f0 = np.load(os.path.join(feature_dir, mid0, "feature", f"{mid0}.npy"))
        l0 = np.load(os.path.join(feature_dir, mid0, "yy_visualization", f"labels_{jid0}.npy"))
        print(f"示例: id={mid0}, joint={jid0}, feature.shape={f0.shape}, label.shape={l0.shape}")
    return samples

# ------------------------- Dataset ------------------------- #

class PartFieldDataset(Dataset):
    def __init__(self, samples: List[Sample], feature_dir: str, img_size: int):
        super().__init__()
        self.samples = samples
        self.feature_dir = feature_dir
        self.rgb_tf  = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        self.gray_tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        mid, cat, jid, view, asp_path, dep_path, nor_path = self.samples[idx]
        feat_path  = os.path.join(self.feature_dir, mid, "feature", f"{mid}.npy")
        label_path = os.path.join(self.feature_dir, mid, "yy_visualization", f"labels_{jid}.npy")

        features = np.load(feat_path).astype(np.float32)
        labels   = np.load(label_path).astype(np.float32)
        if len(features) != len(labels):
            m = min(len(features), len(labels))
            features, labels = features[:m], labels[:m]

        return dict(
            features=torch.from_numpy(features),
            labels=torch.from_numpy(labels),
            arrow=self.rgb_tf(Image.open(asp_path).convert("RGB")),
            depth=self.gray_tf(Image.open(dep_path).convert("L")),
            normal=self.rgb_tf(Image.open(nor_path).convert("RGB")),
            meta=dict(id=mid, joint=jid, view=view)
        )

# ------------------------- 模型 ------------------------- #

class MultiImageFieldMLP(nn.Module):
    def __init__(self, feature_dim: int, img_size: int, hidden_dim: int):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
        )

        def _img_enc(in_c):
            return nn.Sequential(
                nn.Conv2d(in_c, 16, 3, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(16, 32, 3, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 3, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(0.2),
                nn.Flatten(),
                nn.Linear(256 * (img_size // 32) * (img_size // 32), hidden_dim),
                nn.LeakyReLU(0.2),
            )

        self.arrow_enc  = _img_enc(3)
        self.depth_enc  = _img_enc(1)
        self.normal_enc = _img_enc(3)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),     nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feats, arrow, depth, normal):
        b, n, fd = feats.shape
        feat_e = self.feature_encoder(feats.reshape(-1, fd)).reshape(b, n, -1)
        arr_e  = self.arrow_enc(arrow)
        dep_e  = self.depth_enc(depth)
        nor_e  = self.normal_enc(normal)
        arr_e  = arr_e.unsqueeze(1).expand(-1, n, -1)
        dep_e  = dep_e.unsqueeze(1).expand(-1, n, -1)
        nor_e  = nor_e.unsqueeze(1).expand(-1, n, -1)
        fused  = self.fusion(torch.cat([feat_e, arr_e, dep_e, nor_e], dim=-1))
        return self.decoder(fused).squeeze(-1)    # (b,n)

# ------------------------- Train / Val ------------------------- #

def train_val_loop(args, model, train_ld, val_ld, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler    = torch.cuda.amp.GradScaler()           # 统一用旧 API，所有版本可用
    autocast  = torch.cuda.amp.autocast

    best_val = float("inf")
    tr_hist, va_hist = [], []

    for epoch in range(1, args.epochs + 1):
        # ---------------- train ----------------
        model.train()
        run_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_ld, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for step, batch in enumerate(pbar, 1):
            feats  = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            arrow  = batch["arrow"].to(device, non_blocking=True)
            depth  = batch["depth"].to(device, non_blocking=True)
            normal = batch["normal"].to(device, non_blocking=True)

            with autocast():
                loss = criterion(model(feats, arrow, depth, normal), labels) / args.accum
            scaler.scale(loss).backward()

            if step % args.accum == 0 or step == len(train_ld):
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
            run_loss += loss.item() * args.accum
            pbar.set_postfix(loss=run_loss / step)
        tr_hist.append(run_loss / len(train_ld))

        # ---------------- val ----------------
        model.eval(); val_loss = 0.0
        with torch.no_grad(), autocast():
            for batch in tqdm(val_ld, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                feats  = batch["features"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                arrow  = batch["arrow"].to(device, non_blocking=True)
                depth  = batch["depth"].to(device, non_blocking=True)
                normal = batch["normal"].to(device, non_blocking=True)
                val_loss += criterion(model(feats, arrow, depth, normal), labels).item()
        val_loss /= len(val_ld); va_hist.append(val_loss)

        print(f"[{epoch:03d}] train {tr_hist[-1]:.4f} | val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  * 保存最优模型 best_model.pth")

    # 绘制曲线
    plt.figure(figsize=(8,4))
    plt.plot(range(1, args.epochs+1), tr_hist, label="train")
    plt.plot(range(1, args.epochs+1), va_hist, label="val")
    plt.xlabel("epoch"); plt.ylabel("BCE loss"); plt.legend(); plt.tight_layout()
    plt.savefig("loss_curve.png")
    print("训练完成，loss_curve.png 已保存。")

# ------------------------- main ------------------------- #

def main():
    args = get_args(); set_seed(42)
    torch.backends.cudnn.benchmark = True
    feature_dir = os.path.join(args.base_dir, "data/urdf")
    img_dir     = os.path.join(args.base_dir, "data/img")

    samples = collect_samples(img_dir, feature_dir)
    if not samples:
        print("无有效样本，结束。"); return

    train_s, val_s = train_test_split(samples, test_size=0.2, random_state=0)
    train_ds = PartFieldDataset(train_s, feature_dir, args.img_size)
    val_ds   = PartFieldDataset(val_s,   feature_dir, args.img_size)

    train_ld = DataLoader(train_ds, batch_size=1, shuffle=True,
                          num_workers=args.num_workers, pin_memory=args.pin_mem,
                          prefetch_factor=args.prefetch, persistent_workers=True)
    val_ld   = DataLoader(val_ds, batch_size=1, shuffle=False,
                          num_workers=args.num_workers, pin_memory=args.pin_mem,
                          prefetch_factor=args.prefetch, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    model = MultiImageFieldMLP(args.feature_dim, args.img_size, args.hidden_dim).to(device)
    if torch.__version__.startswith("2"):
        model = torch.compile(model)

    # 样本检查
    sample0 = train_ds[0]
    print("\n样本检查:")
    print(f"id={sample0['meta']['id']}, joint={sample0['meta']['joint']}, view={sample0['meta']['view']}")
    print("features:", sample0['features'].shape,
          "arrow:",   sample0['arrow'].shape,
          "depth:",   sample0['depth'].shape,
          "normal:",  sample0['normal'].shape,
          "labels:",  sample0['labels'].shape)

    train_val_loop(args, model, train_ld, val_ld, device)

if __name__ == "__main__":
    main()
