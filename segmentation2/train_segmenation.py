# train_segmenation.py
import os
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

# --------- 修改导入路径 -----------
from loader import SimplifiedArticulatedDataset, simplified_collate_fn, default_transforms
from network import SketchSegmentUNet

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from torch.amp import GradScaler, autocast


# --------------------------- 参数解析 ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Sketch‑to‑Segmentation Training")
    parser.add_argument("--data_dir",   type=str, default="data/img")
    parser.add_argument("--out_dir",    type=str, default="./outputs_sketch2seg")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=70)
    parser.add_argument("--num_workers",type=int, default=20)
    parser.add_argument("--log_interval",type=int, default=10)
    parser.add_argument("--val_interval",type=int, default=1)
    parser.add_argument("--resume",     type=str, default="")
    parser.add_argument("--features",   type=int, default=128)
    return parser.parse_args()


# --------------------------- DataLoader ---------------------------
def create_dataloader(data_dir, split, batch_size, num_workers, shuffle=True):
    transform = default_transforms(split == 'train')
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


# --------------------------- 损失 & 评估 ---------------------------
def dice_loss(prob, target, eps=1e-6):
    """
    prob 已 sigmoid；target ∈ {0,1}
    """
    inter = torch.sum(prob * target, dim=(1, 2, 3))
    union = torch.sum(prob, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) + eps
    dice = 2.0 * inter / union
    return 1.0 - dice.mean()


def combined_loss(logits, target, alpha=0.5):
    """
    BCEWithLogits + Dice
    """
    bce = nn.BCEWithLogitsLoss()(logits, target)
    prob = torch.sigmoid(logits)
    dice = dice_loss(prob, target)
    return alpha * bce + (1 - alpha) * dice


def tensor_bool(pred_logits, thr=0.5):
    return (torch.sigmoid(pred_logits) > thr).float()


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


# --------------------------- Epoch 训练 ---------------------------
def train_one_epoch(model, loader, optimizer, device, epoch, max_epochs,
                   logger, writer=None, log_interval=10, scaler=None):
    model.train()
    total_loss = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f"Epoch {epoch+1}/{max_epochs} (train)", ncols=120)

    for step, (inp, tgt, _) in pbar:
        inp  = inp.to(device)
        tgt  = tgt.to(device)

        optimizer.zero_grad()

        if scaler:
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                logits = model(inp)
                loss   = combined_loss(logits, tgt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inp)
            loss   = combined_loss(logits, tgt)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        if (step + 1) % log_interval == 0:
            logger.info(f"[Train] Ep[{epoch+1}/{max_epochs}] "
                        f"Step[{step+1}/{len(loader)}] Loss:{avg_loss:.4f}")

        if writer:
            global_iter = epoch * len(loader) + step
            writer.add_scalar("Train/Loss", loss.item(), global_iter)

    return total_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(model, loader, device, epoch, max_epochs,
                       logger, writer=None):
    model.eval()
    total_loss = total_iou = total_acc = total_prec = total_rec = 0.0
    sample_cnt = 0

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f"Epoch {epoch+1}/{max_epochs} (val)", ncols=120)

    for step, (inp, tgt, _) in pbar:
        inp, tgt = inp.to(device), tgt.to(device)
        bs       = inp.size(0)
        logits   = model(inp)
        loss     = combined_loss(logits, tgt)

        # 统计
        total_loss += loss.item() * bs
        total_iou  += compute_iou(logits, tgt) * bs
        total_acc  += compute_pixel_accuracy(logits, tgt) * bs
        total_prec += compute_precision(logits, tgt) * bs
        total_rec  += compute_recall(logits, tgt) * bs
        sample_cnt += bs

        pbar.set_postfix({
            "val_loss": f"{total_loss / sample_cnt:.4f}",
            "val_iou":  f"{total_iou  / sample_cnt:.4f}",
            "val_acc":  f"{total_acc  / sample_cnt:.4f}"
        })

    avg_loss = total_loss / sample_cnt
    avg_iou  = total_iou  / sample_cnt
    avg_acc  = total_acc  / sample_cnt
    avg_prec = total_prec / sample_cnt
    avg_rec  = total_rec  / sample_cnt

    logger.info(f"[Val] Ep[{epoch+1}/{max_epochs}] "
                f"Loss:{avg_loss:.4f} IoU:{avg_iou:.4f} "
                f"Acc:{avg_acc:.4f} Prec:{avg_prec:.4f} Rec:{avg_rec:.4f}")

    if writer:
        writer.add_scalar("Val/Loss", avg_loss, epoch + 1)
        writer.add_scalar("Val/IoU",  avg_iou,  epoch + 1)
        writer.add_scalar("Val/Acc",  avg_acc,  epoch + 1)
        writer.add_scalar("Val/Prec", avg_prec, epoch + 1)
        writer.add_scalar("Val/Rec",  avg_rec,  epoch + 1)

        # 可视化：最多 4 张
        vis_cnt = min(4, inp.size(0))
        vis_inp = inp[:vis_cnt]
        vis_gt  = tgt[:vis_cnt]
        vis_prb = torch.sigmoid(logits[:vis_cnt])

        writer.add_image("Val/Input", make_grid(vis_inp, nrow=2, normalize=True),
                         global_step=epoch + 1)
        writer.add_image("Val/GT",    make_grid(vis_gt,  nrow=2, normalize=True),
                         global_step=epoch + 1)
        writer.add_image("Val/Pred",  make_grid(vis_prb, nrow=2, normalize=True),
                         global_step=epoch + 1)

    return avg_loss, avg_iou, avg_acc, avg_prec, avg_rec


# --------------------------- 主程序 ---------------------------
def main():
    args = parse_args()

    # ---------- 日志 ----------
    os.makedirs(args.out_dir, exist_ok=True)
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    log_file   = os.path.join(args.out_dir, f"train_log_{time_stamp}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_file),
                  logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Start training with args: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    writer = SummaryWriter(os.path.join(args.out_dir, f"tb_logs_{time_stamp}")) \
             if TENSORBOARD_AVAILABLE else None

    # ---------- Data ----------
    train_loader = create_dataloader(
        args.data_dir, 'train', args.batch_size, args.num_workers, shuffle=True)
    val_loader = create_dataloader(
        args.data_dir, 'val', args.batch_size, args.num_workers, shuffle=False)

    # ---------- Model ----------
    logger.info(f"Using SketchSegmentUNet with base_features={args.features}")
    model = SketchSegmentUNet(in_channels=3,
                              out_channels=1,
                              base_features=args.features,
                              bilinear=False)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer  = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=5)
    scaler     = GradScaler() if torch.cuda.is_available() else None
    best_model = os.path.join(args.out_dir, "best_model.pth")
    best_loss  = float("inf")
    start_ep   = 0

    # ---------- Resume ----------
    ckpt_path = args.resume if args.resume else best_model
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_ep  = ckpt["epoch"] + 1
        best_loss = ckpt.get("val_loss", best_loss)
        logger.info(f"Resumed from '{ckpt_path}' (ep {start_ep}, val_loss={best_loss:.4f})")
    else:
        logger.info("No checkpoint found; training from scratch.")

    # ---------- Training Loop ----------
    early_patience = 15
    no_improve_ep  = 0

    for epoch in range(start_ep, args.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     epoch, args.num_epochs, logger, writer,
                                     args.log_interval, scaler)

        if (epoch + 1) % args.val_interval == 0:
            val_loss, val_iou, val_acc, val_prec, val_rec = validate_one_epoch(
                model, val_loader, device, epoch, args.num_epochs, logger, writer)

            scheduler.step(val_loss)

            # 保存最优
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state": (model.module.state_dict()
                                    if isinstance(model, nn.DataParallel) else model.state_dict()),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss
                }, best_model)
                logger.info(f"Saved best model (ep {epoch+1}, val_loss {val_loss:.4f})")
                no_improve_ep = 0
            else:
                no_improve_ep += 1
                logger.info(f"No improvement for {no_improve_ep} epoch(s).")
                if no_improve_ep >= early_patience:
                    logger.info("Early stopping triggered.")
                    break

        if (epoch + 1) % 10 == 0:
            cp_path = os.path.join(args.out_dir, f"checkpoint_ep{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": (model.module.state_dict()
                                if isinstance(model, nn.DataParallel) else model.state_dict()),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_loss
            }, cp_path)
            logger.info(f"Checkpoint saved to {cp_path}")

    logger.info("Training finished!")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
