# filename: test_segmentation.py
import os
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------------
# 导入模型和数据加载器
# -----------------------------------
from loader import SimplifiedArticulatedDataset, simplified_collate_fn, default_transforms
from network import SketchSegmentUNet

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# --------------------------- 参数解析 ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Sketch‑to‑Segmentation Testing")

    parser.add_argument("--data_dir",        type=str, default="data/img",
                        help="Root directory; should contain 'test' subfolder.")
    parser.add_argument("--out_dir",         type=str, default="./outputs_sketch2seg",
                        help="Directory to save logs / visualizations.")
    parser.add_argument("--batch_size",      type=int, default=12,
                        help="Batch size for testing.")
    parser.add_argument("--num_workers",     type=int, default=20,
                        help="Number of DataLoader workers.")
    parser.add_argument("--resume",          type=str, default="",
                        help="Checkpoint path; default uses best_model.pth in out_dir.")
    parser.add_argument("--features",        type=int, default=128,
                        help="Base features of UNet (match training).")
    parser.add_argument("--visualize_count", type=int, default=50,
                        help="Number of samples to save as PNG.")
    return parser.parse_args()


# --------------------------- DataLoader ---------------------------
def create_dataloader(data_dir, split, batch_size, num_workers, shuffle=False):
    transform = default_transforms(False)  # 测试集无增强
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
    bce = nn.BCEWithLogitsLoss()(logits, target)
    prob = torch.sigmoid(logits)
    dsc = dice_loss(prob, target)
    return alpha * bce + (1 - alpha) * dsc


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
               visualize_count=50,
               out_dir="./outputs_sketch2seg"):
    model.eval()

    criterion = combined_loss
    total_loss = total_iou = total_acc = total_prec = total_rec = 0.0
    sample_cnt = 0

    images_to_vis = []
    vis_saved = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc="Testing", ncols=120)

    for step, (inp, tgt, meta) in pbar:
        inp, tgt = inp.to(device), tgt.to(device)
        bs = inp.size(0)

        logits = model(inp)
        loss   = criterion(logits, tgt)

        # 评估
        batch_iou  = compute_iou(logits, tgt)
        batch_acc  = compute_pixel_accuracy(logits, tgt)
        batch_prec = compute_precision(logits, tgt)
        batch_rec  = compute_recall(logits, tgt)

        total_loss += loss.item() * bs
        total_iou  += batch_iou  * bs
        total_acc  += batch_acc  * bs
        total_prec += batch_prec * bs
        total_rec  += batch_rec  * bs
        sample_cnt += bs

        # 收集可视化样本
        for b in range(bs):
            if vis_saved < visualize_count:
                images_to_vis.append({
                    "inp":  inp[b].cpu(),
                    "pred": logits[b].cpu(),
                    "gt":   tgt[b].cpu(),
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

    avg_loss = total_loss / sample_cnt if sample_cnt else 0
    avg_iou  = total_iou  / sample_cnt if sample_cnt else 0
    avg_acc  = total_acc  / sample_cnt if sample_cnt else 0
    avg_prec = total_prec / sample_cnt if sample_cnt else 0
    avg_rec  = total_rec  / sample_cnt if sample_cnt else 0

    print(f"[Test] Loss={avg_loss:.4f} | IoU={avg_iou:.4f} | "
          f"Acc={avg_acc:.4f} | Precision={avg_prec:.4f} | Recall={avg_rec:.4f}")

    if writer:
        writer.add_scalar("Test/Loss", avg_loss, 0)
        writer.add_scalar("Test/IoU",  avg_iou,  0)
        writer.add_scalar("Test/Acc",  avg_acc,  0)
        writer.add_scalar("Test/Prec", avg_prec, 0)
        writer.add_scalar("Test/Rec",  avg_rec,  0)

    # ---------- 保存可视化 ----------
    vis_dir = os.path.join(out_dir, "test_vis")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Saving visualizations to {vis_dir} ...")

    for idx, item in enumerate(images_to_vis):
        inp  = item["inp"].numpy().transpose(1, 2, 0)
        pred = torch.sigmoid(item["pred"]).numpy()[0]
        gt   = item["gt"].numpy()[0]
        meta = item["meta"]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(np.clip(inp, 0, 1))
        axes[0].set_title("Arrow‑Sketch")
        axes[1].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Pred")
        axes[2].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("GT")

        overlay = np.clip(inp, 0, 1)
        mask = np.zeros_like(overlay)
        mask[:, :, 0] = (pred > 0.5).astype(np.float32)
        overlay = overlay * 0.7 + mask * 0.3
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay")

        for ax in axes:
            ax.axis("off")

        cat  = meta.get("category", "unknown")
        j_id = meta.get("joint",     "unknown")
        fig.suptitle(f"Category: {cat} | Joint: {j_id}", fontsize=16)

        save_path = os.path.join(vis_dir, f"sample_{idx:03d}_{cat}_joint{j_id}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    # ---------- 保存数值结果 ----------
    res_file = os.path.join(out_dir, "test_results.txt")
    with open(res_file, "w") as f:
        f.write("Test Results\n")
        f.write(f"Loss:      {avg_loss:.6f}\n")
        f.write(f"IoU:       {avg_iou:.6f}\n")
        f.write(f"Accuracy:  {avg_acc:.6f}\n")
        f.write(f"Precision: {avg_prec:.6f}\n")
        f.write(f"Recall:    {avg_rec:.6f}\n")
    print(f"Metrics written to {res_file}")

    if writer:
        writer.close()

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

    # ---------- 检查 checkpoint ----------
    if not args.resume:
        default_ckpt = os.path.join(args.out_dir, "best_model.pth")
        if os.path.isfile(default_ckpt):
            args.resume = default_ckpt
            logger.info(f"No --resume given, using {args.resume}")
        else:
            raise FileNotFoundError(
                "Checkpoint not specified and default best_model.pth not found."
            )
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint '{args.resume}' does not exist.")

    # ---------- Data ----------
    test_loader = create_dataloader(
        data_dir=args.data_dir,
        split="test",
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
        logger.info(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    model = model.to(device)

    ckpt = torch.load(args.resume, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded checkpoint '{args.resume}' (epoch={ckpt.get('epoch','?')})")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # ---------- Run Test ----------
    test_loss, test_iou, test_acc, test_prec, test_rec = test_model(
        model, test_loader, device, writer,
        visualize_count=args.visualize_count,
        out_dir=args.out_dir
    )

    logger.info("[Final Test Results]")
    logger.info(f"Loss:      {test_loss:.6f}")
    logger.info(f"IoU:       {test_iou:.6f}")
    logger.info(f"Accuracy:  {test_acc:.6f}")
    logger.info(f"Precision: {test_prec:.6f}")
    logger.info(f"Recall:    {test_rec:.6f}")


if __name__ == "__main__":
    main()
