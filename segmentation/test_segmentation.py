# filename: test_segmentation.py

import os
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------------
# 更新导入部分以匹配训练脚本
# -----------------------------------
from loader import NewArticulatedDataset, new_collate_fn, default_transforms
from network import FlexibleUNet

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Testing")

    default_data_dir = "data/img"
    parser.add_argument("--data_dir", type=str, default=default_data_dir,
                        help="Root directory of your dataset with 'test' subfolder.")
    parser.add_argument("--out_dir", type=str, default="./outputs",
                        help="Directory to save test logs or visualization results.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for testing.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of worker threads.")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to the checkpoint (.pth) for model weights. If not set, will use best_model.pth in out_dir.")
    # 默认启用TensorBoard
    parser.add_argument("--visualize_count", type=int, default=400,
                        help="Number of samples to visualize (save or show).")

    args = parser.parse_args()
    return args


def create_dataloader(data_dir, split, batch_size, num_workers, shuffle=False):
    """
    创建 NewArticulatedDataset 对象和 DataLoader。
    """
    # 测试集不需要数据增强
    transform = default_transforms(False)
    
    dataset = NewArticulatedDataset(
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
        collate_fn=new_collate_fn
    )
    return loader


def dice_loss(pred, target, epsilon=1e-6):
    pred = pred.float()
    target = target.float()
    intersection = torch.sum(pred * target, dim=(1,2,3))
    union = torch.sum(pred, dim=(1,2,3)) + torch.sum(target, dim=(1,2,3)) + epsilon
    dice = 2.0 * intersection / union
    loss = 1.0 - dice
    return loss.mean()


def compute_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target = target.float()
    intersection = torch.sum(pred_bin * target, dim=(1,2,3))
    union = torch.sum((pred_bin + target) > 0, dim=(1,2,3))
    iou = intersection / (union + 1e-6)
    return iou.mean().item()


def compute_pixel_accuracy(pred, target, threshold=0.5):
    pred_bin   = (pred > threshold).float()
    target_bin = (target > threshold).float()
    correct = torch.sum(pred_bin == target_bin, dim=(1,2,3))
    total   = pred_bin.shape[0] * pred_bin.shape[2] * pred_bin.shape[3]
    accuracy = correct / total
    return accuracy.mean().item()


def compute_precision(pred, target, threshold=0.5):
    """
    计算精确率。
    Args:
        pred: 预测概率，[B,1,H,W]
        target: 真实标签，[B,1,H,W]
        threshold: 二值化阈值
    Returns:
        平均精确率
    """
    pred_bin = (pred > threshold).float()
    target_bin = target.float()

    true_positive = torch.sum(pred_bin * target_bin, dim=(1, 2, 3))
    predicted_positive = torch.sum(pred_bin, dim=(1, 2, 3))
    precision = true_positive / (predicted_positive + 1e-6)
    return precision.mean().item()


def compute_recall(pred, target, threshold=0.5):
    """
    计算召回率。
    Args:
        pred: 预测概率，[B,1,H,W]
        target: 真实标签，[B,1,H,W]
        threshold: 二值化阈值
    Returns:
        平均召回率
    """
    pred_bin = (pred > threshold).float()
    target_bin = target.float()

    true_positive = torch.sum(pred_bin * target_bin, dim=(1, 2, 3))
    actual_positive = torch.sum(target_bin, dim=(1, 2, 3))
    recall = true_positive / (actual_positive + 1e-6)
    return recall.mean().item()


@torch.no_grad()
def test_model(model, dataloader, device, writer=None, visualize_count=4, out_dir="./outputs"):
    model.eval()

    criterion = dice_loss
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_precision = 0.0
    total_recall = 0.0
    sample_count = 0

    images_to_visualize = []
    count_visualized = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing", ncols=120)
    for step, batch_data in pbar:
        if isinstance(batch_data, tuple):
            # 单一类型输入
            inp_batch, lab_batch, meta_batch = batch_data
            inp_batch = inp_batch.to(device)
            lab_batch = lab_batch.to(device)
            batch_size = inp_batch.size(0)

            # 前向传播
            out_batch = model(inp_batch)
            out_batch = torch.sigmoid(out_batch)
            
            loss = criterion(out_batch, lab_batch)
            
            # 性能评估
            batch_iou = compute_iou(out_batch, lab_batch)
            batch_acc = compute_pixel_accuracy(out_batch, lab_batch)
            batch_precision = compute_precision(out_batch, lab_batch)
            batch_recall = compute_recall(out_batch, lab_batch)
            
            # 累加加权统计量
            total_loss += loss.item() * batch_size
            total_iou += batch_iou * batch_size
            total_acc += batch_acc * batch_size
            total_precision += batch_precision * batch_size
            total_recall += batch_recall * batch_size
            sample_count += batch_size
            
            # 可视化样本收集
            for b_idx in range(inp_batch.size(0)):
                if count_visualized < visualize_count:
                    images_to_visualize.append({
                        "inp": inp_batch[b_idx, :3, :, :].cpu(),  # 只取RGB通道
                        "pred": out_batch[b_idx].cpu(),
                        "gt": lab_batch[b_idx].cpu(),
                        "meta": meta_batch[b_idx]
                    })
                    count_visualized += 1
                else:
                    break
            
        else:
            # 混合通道输入
            tensors_list, meta_batch = batch_data
            
            for inp_batch, lab_batch in tensors_list:
                inp_batch = inp_batch.to(device)
                lab_batch = lab_batch.to(device)
                batch_size = inp_batch.size(0)
                
                # 前向传播
                out_batch = model(inp_batch)
                out_batch = torch.sigmoid(out_batch)
                
                loss = criterion(out_batch, lab_batch)
                
                # 性能评估
                batch_iou = compute_iou(out_batch, lab_batch)
                batch_acc = compute_pixel_accuracy(out_batch, lab_batch)
                batch_precision = compute_precision(out_batch, lab_batch)
                batch_recall = compute_recall(out_batch, lab_batch)
                
                # 累加加权统计量
                total_loss += loss.item() * batch_size
                total_iou += batch_iou * batch_size
                total_acc += batch_acc * batch_size
                total_precision += batch_precision * batch_size
                total_recall += batch_recall * batch_size
                sample_count += batch_size
                
                # 可视化样本收集
                for b_idx in range(inp_batch.size(0)):
                    if count_visualized < visualize_count:
                        meta_idx = meta_batch[min(b_idx, len(meta_batch)-1)]  # 避免索引越界
                        images_to_visualize.append({
                            "inp": inp_batch[b_idx, :3, :, :].cpu(),  # 只取RGB通道
                            "pred": out_batch[b_idx].cpu(),
                            "gt": lab_batch[b_idx].cpu(),
                            "meta": meta_idx
                        })
                        count_visualized += 1
                    else:
                        break
        
        # 更新进度条信息
        if sample_count > 0:
            pbar.set_postfix({
                "loss": f"{(total_loss / sample_count):.4f}",
                "iou": f"{(total_iou / sample_count):.4f}",
                "acc": f"{(total_acc / sample_count):.4f}",
                "precision": f"{(total_precision / sample_count):.4f}",
                "recall": f"{(total_recall / sample_count):.4f}"
            })

    # 计算平均值
    avg_loss = total_loss / sample_count if sample_count > 0 else 0
    avg_iou = total_iou / sample_count if sample_count > 0 else 0
    avg_acc = total_acc / sample_count if sample_count > 0 else 0
    avg_precision = total_precision / sample_count if sample_count > 0 else 0
    avg_recall = total_recall / sample_count if sample_count > 0 else 0

    print(f"[Test] Loss={avg_loss:.4f} | IoU={avg_iou:.4f} | Acc={avg_acc:.4f} | Precision={avg_precision:.4f} | Recall={avg_recall:.4f}")

    if writer is not None:
        writer.add_scalar("Test/Loss", avg_loss, 0)
        writer.add_scalar("Test/IoU", avg_iou, 0)
        writer.add_scalar("Test/PixelAcc", avg_acc, 0)
        writer.add_scalar("Test/Precision", avg_precision, 0)
        writer.add_scalar("Test/Recall", avg_recall, 0)

    visualize_save_dir = os.path.join(out_dir, "test_vis")
    os.makedirs(visualize_save_dir, exist_ok=True)

    for idx, item in enumerate(images_to_visualize):
        inp = item["inp"]
        pred = item["pred"]
        gt = item["gt"]
        meta = item["meta"]

        # 转换为numpy，准备可视化
        inp_np = inp.numpy()
        pred_np = pred.numpy()[0]
        gt_np = gt.numpy()[0]

        # 创建4子图：输入、预测、GT和叠加预测
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # 输入图像（RGB通道）
        inp_show = np.transpose(inp_np, (1, 2, 0))
        inp_show = (inp_show - inp_show.min()) / (inp_show.max() - inp_show.min() + 1e-8)
        axes[0].imshow(inp_show)
        axes[0].set_title("Input")

        # 预测
        axes[1].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Prediction")

        # 真实分割
        axes[2].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title("Ground Truth")
        
        # 叠加在输入上的预测
        overlay = inp_show.copy()
        pred_mask = np.stack([pred_np, np.zeros_like(pred_np), np.zeros_like(pred_np)], axis=-1)
        pred_mask[..., 0] = pred_np > 0.5  # 二值化并设为红色
        overlay = 0.7 * overlay + 0.3 * pred_mask
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay")

        # 设置标题包含元数据
        category = meta.get('category', 'unknown')
        joint = meta.get('joint', 'none')
        fig.suptitle(f"Category: {category}, Joint: {joint}")

        for ax in axes:
            ax.axis("off")

        save_path = os.path.join(visualize_save_dir, f"sample_{idx}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    if writer is not None:
        writer.close()

    return avg_loss, avg_iou, avg_acc, avg_precision, avg_recall


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Testing with args: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 默认启用TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb_logs_test"))
    else:
        logger.warning("TensorBoard is not available in your environment!")

    if args.resume == "":
        default_ckpt = os.path.join(args.out_dir, "best_model.pth")
        if os.path.isfile(default_ckpt):
            args.resume = default_ckpt
            logger.info(f"No --resume specified, using default: {args.resume}")
        else:
            raise FileNotFoundError(
                f"No --resume specified and {default_ckpt} does not exist. "
                "Please provide a valid checkpoint via --resume."
            )

    test_loader = create_dataloader(
        data_dir=args.data_dir,
        split='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    # 使用FlexibleUNet而不是UNet
    model = FlexibleUNet(base_features=64, out_channels=1, bilinear=True)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    model = model.to(device)

    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"No checkpoint found at {args.resume}")

    ckpt = torch.load(args.resume, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded checkpoint from '{args.resume}' (epoch={ckpt['epoch']})")

    test_loss, test_iou, test_acc, test_precision, test_recall = test_model(
        model=model,
        dataloader=test_loader,
        device=device,
        writer=writer,
        visualize_count=args.visualize_count,
        out_dir=args.out_dir
    )

    logger.info(f"[Final Test] Loss={test_loss:.4f}, IoU={test_iou:.4f}, Acc={test_acc:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}")


if __name__ == "__main__":
    main()
