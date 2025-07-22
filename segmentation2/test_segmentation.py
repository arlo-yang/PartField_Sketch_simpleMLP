# filename: test_segmentation.py
import os
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

from loader import SimplifiedArticulatedDataset, simplified_collate_fn, default_transforms
from network import SketchSegmentUNet

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# --------------------------- 参数解析 ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Sketch‑to‑Segmentation Testing (mask‑relative confidence可视化 + 原始prob保存)")

    parser.add_argument("--data_dir",        type=str, default="data/img",
                        help="Dataset root (必须含 test 子目录).")
    parser.add_argument("--out_dir",         type=str, default="./outputs_sketch2seg",
                        help="日志 / 可视化输出目录.")
    parser.add_argument("--batch_size",      type=int, default=12)
    parser.add_argument("--num_workers",     type=int, default=20)
    parser.add_argument("--resume",          type=str, default="",
                        help="模型 checkpoint (.pth). 默认尝试 out_dir/best_model.pth")
    parser.add_argument("--features",        type=int, default=128)
    parser.add_argument("--visualize_count", type=int, default=150,
                        help="保存多少张可视化.")
    parser.add_argument("--save_confidence", action="store_true",
                        help="保存原始 prob 为 .npy")
    return parser.parse_args()


# --------------------------- DataLoader ---------------------------
def create_dataloader(data_dir, split, batch_size, num_workers, shuffle=False):
    transform = default_transforms(False)
    dataset = SimplifiedArticulatedDataset(
        root_dir=data_dir,
        split=split,
        transform=transform,
        verbose=False
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=simplified_collate_fn
    )
    return loader


# --------------------------- 损失 & 指标 ---------------------------
def dice_loss(prob, target, eps=1e-6):
    inter = torch.sum(prob * target, dim=(1, 2, 3))
    union = torch.sum(prob, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) + eps
    dice = 2.0 * inter / union
    return 1.0 - dice.mean()


def combined_loss(logits, target, alpha=0.5):
    bce  = nn.BCEWithLogitsLoss()(logits, target)
    prob = torch.sigmoid(logits)
    return alpha * bce + (1 - alpha) * dice_loss(prob, target)


def tensor_bool(logits, thr=0.5):
    return (torch.sigmoid(logits) > thr).float()


def compute_iou(logits, target, thr=0.5):
    pred = tensor_bool(logits, thr)
    inter = torch.sum(pred * target, dim=(1, 2, 3))
    union = torch.sum((pred + target) > 0, dim=(1, 2, 3))
    return (inter / (union + 1e-6)).mean().item()


def compute_pixel_accuracy(logits, target, thr=0.5):
    pred = tensor_bool(logits, thr)
    correct = torch.sum(pred == target, dim=(1, 2, 3))
    total = torch.numel(pred[0])
    return (correct / total).mean().item()


def compute_precision(logits, target, thr=0.5):
    pred = tensor_bool(logits, thr)
    tp = torch.sum(pred * target, dim=(1, 2, 3))
    pp = torch.sum(pred, dim=(1, 2, 3))
    return (tp / (pp + 1e-6)).mean().item()


def compute_recall(logits, target, thr=0.5):
    pred = tensor_bool(logits, thr)
    tp = torch.sum(pred * target, dim=(1, 2, 3))
    ap = torch.sum(target, dim=(1, 2, 3))
    return (tp / (ap + 1e-6)).mean().item()


# --------------------------- 测试流程 ---------------------------
@torch.no_grad()
def test_model(model,
               dataloader,
               device,
               writer=None,
               visualize_count=150,
               out_dir="./outputs_sketch2seg",
               save_confidence=False):
    model.eval()

    criterion = combined_loss
    total_loss = total_iou = total_acc = total_prec = total_rec = 0.0
    sample_cnt = 0

    vis_samples = []
    vis_saved = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc="Testing", ncols=120)

    for step, (inp, tgt, meta) in pbar:
        inp, tgt = inp.to(device), tgt.to(device)
        bs = inp.size(0)

        logits = model(inp)
        loss   = criterion(logits, tgt)

        total_loss += loss.item() * bs
        total_iou  += compute_iou(logits, tgt)  * bs
        total_acc  += compute_pixel_accuracy(logits, tgt) * bs
        total_prec += compute_precision(logits, tgt) * bs
        total_rec  += compute_recall(logits, tgt) * bs
        sample_cnt += bs

        for b in range(bs):
            if vis_saved < visualize_count:
                vis_samples.append({
                    "prob": torch.sigmoid(logits[b]).cpu(),  # [1,H,W]
                    "gt":   tgt[b].cpu(),                    # [1,H,W]
                    "meta": meta[b]
                })
                vis_saved += 1
            else:
                break

        pbar.set_postfix({
            "loss": f"{total_loss / sample_cnt:.4f}",
            "iou":  f"{total_iou  / sample_cnt:.4f}",
            "acc":  f"{total_acc  / sample_cnt:.4f}",
            "prec": f"{total_prec / sample_cnt:.4f}",
            "rec":  f"{total_rec  / sample_cnt:.4f}"
        })

    vis_dir = os.path.join(out_dir, "test_composite")
    os.makedirs(vis_dir, exist_ok=True)
    conf_dir = os.path.join(out_dir, "confidence_npy")
    if save_confidence:
        os.makedirs(conf_dir, exist_ok=True)

    cmap = cm.get_cmap('jet')

    for idx, item in enumerate(vis_samples):
        prob_np = item["prob"].numpy()[0]           # (H,W) 原始概率
        gt_np   = item["gt"].numpy()[0]             # (H,W)
        pred_bin = (prob_np > 0.5).astype(np.uint8) # (H,W) 0/1

        # -------- 可视化映射（不影响 prob_np） --------
        mask_vals = prob_np[pred_bin == 1]
        if mask_vals.size > 0:
            v_max = mask_vals.max()
            if v_max > 0.5:
                vis_vals = (prob_np - 0.5) / (v_max - 0.5)
                vis_vals = np.clip(vis_vals, 0.0, 1.0)
            else:
                vis_vals = np.zeros_like(prob_np)
        else:
            vis_vals = np.zeros_like(prob_np)

        color_rgba = cmap(vis_vals)
        color_rgba[..., 3] = pred_bin  # alpha = mask

        meta = item["meta"]
        cat  = meta.get("category", "unk")
        j_id = meta.get("joint",    "unk")

        # -------- 三联图 --------
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(color_rgba)
        axes[0].set_title("Masked Rel‑Confidence")
        axes[1].imshow(pred_bin, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Pred Mask")
        axes[2].imshow(gt_np,   cmap='gray', vmin=0, vmax=1)
        axes[2].set_title("GT Mask")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()

        save_path = os.path.join(vis_dir, f"sample_{idx:03d}_{cat}_j{j_id}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        if save_confidence:
            np.save(os.path.join(conf_dir, f"conf_{idx:03d}_{cat}_j{j_id}.npy"),
                    prob_np.astype(np.float32))

    avg_loss = total_loss / sample_cnt
    avg_iou  = total_iou  / sample_cnt
    avg_acc  = total_acc  / sample_cnt
    avg_prec = total_prec / sample_cnt
    avg_rec  = total_rec  / sample_cnt

    if writer:
        writer.add_scalar("Test/Loss", avg_loss, 0)
        writer.add_scalar("Test/IoU",  avg_iou,  0)
        writer.add_scalar("Test/Acc",  avg_acc,  0)
        writer.add_scalar("Test/Prec", avg_prec, 0)
        writer.add_scalar("Test/Rec",  avg_rec,  0)
        writer.close()

    with open(os.path.join(out_dir, "test_results.txt"), "w") as f:
        f.write(f"Loss: {avg_loss:.6f}\nIoU: {avg_iou:.6f}\nAcc: {avg_acc:.6f}\n")
        f.write(f"Prec:{avg_prec:.6f}\nRec: {avg_rec:.6f}\n")

    return avg_loss, avg_iou, avg_acc, avg_prec, avg_rec


# --------------------------- 主入口 ---------------------------
def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"Testing with args: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    writer = SummaryWriter(os.path.join(args.out_dir, "tb_logs_test")) \
             if TENSORBOARD_AVAILABLE else None

    # ---------- checkpoint ----------
    if not args.resume:
        default_ckpt = os.path.join(args.out_dir, "best_model.pth")
        if os.path.isfile(default_ckpt):
            args.resume = default_ckpt
        else:
            raise FileNotFoundError("Checkpoint not found.")
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint '{args.resume}' does not exist.")

    # ---------- Data ----------
    test_loader = create_dataloader(
        args.data_dir, 'test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    # ---------- Model ----------
    model = SketchSegmentUNet(in_channels=3,
                              out_channels=1,
                              base_features=args.features,
                              bilinear=False)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    ckpt = torch.load(args.resume, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt["model_state"])

    # ---------- Run Test ----------
    test_model(model,
               test_loader,
               device,
               writer,
               visualize_count=args.visualize_count,
               out_dir=args.out_dir,
               save_confidence=args.save_confidence)


if __name__ == "__main__":
    main()
