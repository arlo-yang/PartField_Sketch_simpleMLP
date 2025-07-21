# filename: train_segmentation.py

import os
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from tqdm import tqdm  # 用于进度条显示

# --------- 修改导入路径 -----------
from loader import NewArticulatedDataset, new_collate_fn, default_transforms
from network import FlexibleUNet

try:
    from torch.utils.tensorboard import SummaryWriter  # PyTorch >= 1.2
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from torch.cuda.amp import GradScaler, autocast  # 混合精度训练


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training Example")

    # 1) 如果用户没指定 --data_dir，就用一个你想要的默认路径
    default_data_dir = "data/img"

    parser.add_argument("--data_dir", type=str, default=default_data_dir,
                        help="Root directory of your dataset containing 'train', 'val', 'test' subdirectories.")

    # 2) 其他超参数给定更大默认值
    parser.add_argument("--out_dir", type=str, default="./outputs",
                        help="Directory to save checkpoints/log.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default=16).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default=1e-4).")
    parser.add_argument("--num_epochs", type=int, default=70,
                        help="Number of training epochs (default=70).")
    parser.add_argument("--num_workers", type=int, default=8,  # 调低 num_workers
                        help="Number of worker threads for DataLoader.")

    # 移除 --use_tensorboard 参数，默认启用 TensorBoard
    # parser.add_argument("--use_tensorboard", action="store_true",
    #                     help="Use TensorBoard to log losses/visuals.")
    # 默认启用 TensorBoard，不需要参数
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Print/log every N steps inside an epoch.")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="Run validation every N epochs.")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from.")

    args = parser.parse_args()
    return args


def create_dataloader(data_dir, split, batch_size, num_workers, shuffle=True):
    """
    创建 NewArticulatedDataset 对象和 DataLoader。
    Args:
        data_dir (str): 数据集根目录。
        split (str): 'train', 'val', 或 'test'。
        batch_size (int): 每批次大小。
        num_workers (int): DataLoader 的 worker 数量。
        shuffle (bool): 是否打乱数据。
    """
    # 在训练集上应用数据增强，验证和测试集不应用
    transform = default_transforms(split=='train')
    
    dataset = NewArticulatedDataset(
        root_dir=data_dir,
        split=split,
        transform=transform,
        verbose=False  # 不想打印太多信息时设为 False
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
    """
    用 Dice Loss 作分割损失。
    pred, target: [B,1,H,W], 取值范围 0~1
    """
    pred = pred.float()
    target = target.float()

    intersection = torch.sum(pred * target, dim=(1, 2, 3))
    union = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) + epsilon
    dice = 2.0 * intersection / union
    loss = 1.0 - dice
    return loss.mean()


def combined_loss(pred, target, alpha=0.5, epsilon=1e-6):
    """
    组合 BCEWithLogitsLoss 和 Dice Loss。
    Args:
        pred: 模型输出的 logits，[B,1,H,W]
        target: 真实标签，[B,1,H,W]
        alpha: BCE 和 Dice 的权重
        epsilon: 防止除零
    Returns:
        组合损失值
    """
    bce = nn.BCEWithLogitsLoss()(pred, target)
    pred_sigmoid = torch.sigmoid(pred)
    dice = dice_loss(pred_sigmoid, target, epsilon)
    return alpha * bce + (1 - alpha) * dice


def compute_iou(pred, target, threshold=0.5):
    """
    计算 IoU。
    Args:
        pred: 预测概率，[B,1,H,W]
        target: 真实标签，[B,1,H,W]
        threshold: 二值化阈值
    Returns:
        平均 IoU
    """
    pred_bin = (pred > threshold).float()
    target_bin = target.float()

    intersection = torch.sum(pred_bin * target_bin, dim=(1, 2, 3))
    union = torch.sum((pred_bin + target_bin) > 0, dim=(1, 2, 3))
    iou = intersection / (union + 1e-6)
    return iou.mean().item()


def compute_pixel_accuracy(pred, target, threshold=0.5):
    """
    计算像素准确率。
    Args:
        pred: 预测概率，[B,1,H,W]
        target: 真实标签，[B,1,H,W]
        threshold: 二值化阈值
    Returns:
        平均像素准确率
    """
    pred_bin = (pred > threshold).float()
    target_bin = target.float()
    correct = torch.sum(pred_bin == target_bin, dim=(1, 2, 3))
    total = pred_bin.shape[0] * pred_bin.shape[2] * pred_bin.shape[3]
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


def train_one_epoch(model, dataloader, optimizer, device, epoch, max_epochs, logger,
                   writer=None, log_interval=10, scaler=None):
    """
    单个 epoch 的训练流程，使用 tqdm 可视化进度条。
    修改以处理可能的不同通道数输入
    """
    model.train()
    criterion = combined_loss  # 使用组合损失

    total_loss = 0.0
    num_steps = len(dataloader)

    # 用 tqdm 包裹 dataloader，显示进度
    pbar = tqdm(enumerate(dataloader), total=num_steps,
                desc=f"Epoch {epoch+1}/{max_epochs} (train)", ncols=120)
    for step, batch_data in pbar:
        # 处理batch_data，现在它可能是元组(inp, lab, meta)或列表[(inp1, lab1), (inp2, lab2)]和meta
        if isinstance(batch_data, tuple):
            inp_batch, lab_batch, meta_batch = batch_data
            # 单一类型输入，直接处理
            inp_batch = inp_batch.to(device)
            lab_batch = lab_batch.to(device)
            
            optimizer.zero_grad()

            if scaler:
                with autocast():
                    out_batch = model(inp_batch)
                    loss = criterion(out_batch, lab_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_batch = model(inp_batch)
                loss = criterion(out_batch, lab_batch)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
        else:
            # 混合通道输入，分开处理
            tensors_list, meta_batch = batch_data
            batch_loss = 0.0
            
            optimizer.zero_grad()
            
            # 分别处理不同通道类型的批次
            for inp_batch, lab_batch in tensors_list:
                inp_batch = inp_batch.to(device)
                lab_batch = lab_batch.to(device)
                
                if scaler:
                    with autocast():
                        out_batch = model(inp_batch)
                        loss = criterion(out_batch, lab_batch)
                    scaler.scale(loss).backward()
                    batch_loss += loss.item()
                else:
                    out_batch = model(inp_batch)
                    loss = criterion(out_batch, lab_batch)
                    loss.backward()
                    batch_loss += loss.item()
            
            # 更新参数
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # 计算平均损失（批次大小加权）
            total_loss += batch_loss / len(tensors_list)

        avg_loss = total_loss / (step + 1)

        # 用 tqdm 显示当前 loss
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # 每隔 log_interval 步，打到日志
        if (step + 1) % log_interval == 0:
            logger.info(f"[Train] Epoch [{epoch+1}/{max_epochs}] Step [{step+1}/{num_steps}] Loss: {avg_loss:.4f}")

        # TensorBoard 记录
        if writer is not None:
            global_iter = epoch * num_steps + step
            writer.add_scalar("Train/Loss", avg_loss, global_iter)

    return total_loss / num_steps


@torch.no_grad()
def validate_one_epoch(model, dataloader, device, epoch, max_epochs, logger, writer=None):
    """
    单个 epoch 的验证流程，使用 tqdm 进度条。
    修改以处理可能的不同通道数输入
    """
    model.eval()
    criterion = combined_loss  # 使用组合损失

    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_precision = 0.0
    total_recall = 0.0
    sample_count = 0

    # 可以用 tqdm 看验证进度，也可以省略
    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch+1}/{max_epochs} (val)", ncols=120)
    
    for step, batch_data in pbar:
        if isinstance(batch_data, tuple):
            # 单一类型输入
            inp_batch, lab_batch, meta_batch = batch_data
            inp_batch = inp_batch.to(device)
            lab_batch = lab_batch.to(device)
            batch_size = inp_batch.size(0)

            # 前向传播
            out_batch = model(inp_batch)
            loss = criterion(out_batch, lab_batch)
            
            # 性能评估
            pred = torch.sigmoid(out_batch)
            batch_iou = compute_iou(pred, lab_batch)
            batch_acc = compute_pixel_accuracy(pred, lab_batch)
            batch_precision = compute_precision(pred, lab_batch)
            batch_recall = compute_recall(pred, lab_batch)
            
            # 累加加权统计量
            total_loss += loss.item() * batch_size
            total_iou += batch_iou * batch_size
            total_acc += batch_acc * batch_size
            total_precision += batch_precision * batch_size
            total_recall += batch_recall * batch_size
            sample_count += batch_size
            
        else:
            # 混合通道输入
            tensors_list, meta_batch = batch_data
            
            for inp_batch, lab_batch in tensors_list:
                inp_batch = inp_batch.to(device)
                lab_batch = lab_batch.to(device)
                batch_size = inp_batch.size(0)
                
                # 前向传播
                out_batch = model(inp_batch)
                loss = criterion(out_batch, lab_batch)
                
                # 性能评估
                pred = torch.sigmoid(out_batch)
                batch_iou = compute_iou(pred, lab_batch)
                batch_acc = compute_pixel_accuracy(pred, lab_batch)
                batch_precision = compute_precision(pred, lab_batch)
                batch_recall = compute_recall(pred, lab_batch)
                
                # 累加加权统计量
                total_loss += loss.item() * batch_size
                total_iou += batch_iou * batch_size
                total_acc += batch_acc * batch_size
                total_precision += batch_precision * batch_size
                total_recall += batch_recall * batch_size
                sample_count += batch_size
        
        # 更新进度条信息
        if sample_count > 0:
            pbar.set_postfix({
                "val_loss": f"{(total_loss / sample_count):.4f}",
                "val_iou": f"{(total_iou / sample_count):.4f}",
                "val_acc": f"{(total_acc / sample_count):.4f}",
                "val_precision": f"{(total_precision / sample_count):.4f}",
                "val_recall": f"{(total_recall / sample_count):.4f}"
            })

    # 计算平均值
    avg_loss = total_loss / sample_count if sample_count > 0 else 0
    avg_iou = total_iou / sample_count if sample_count > 0 else 0
    avg_acc = total_acc / sample_count if sample_count > 0 else 0
    avg_precision = total_precision / sample_count if sample_count > 0 else 0
    avg_recall = total_recall / sample_count if sample_count > 0 else 0

    logger.info(f"[Val] Epoch [{epoch+1}/{max_epochs}] Average Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | PixelAcc: {avg_acc:.4f} | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f}")

    # 若要记录到 TensorBoard
    if writer is not None:
        writer.add_scalar("Val/Loss", avg_loss, epoch + 1)
        writer.add_scalar("Val/IoU", avg_iou, epoch + 1)
        writer.add_scalar("Val/PixelAcc", avg_acc, epoch + 1)
        writer.add_scalar("Val/Precision", avg_precision, epoch + 1)
        writer.add_scalar("Val/Recall", avg_recall, epoch + 1)

        # 也可以可视化部分推理结果，这里简化处理方式
        if isinstance(batch_data, tuple):
            # 取一组样本可视化
            vis_inp = inp_batch[:2, :3, :, :]  # 仅使用RGB通道
            vis_pred = pred[:2, ...]
            vis_gt = lab_batch[:2, ...]
            
            # make_grid: (N, C, H, W) -> (C, H, W) for TB
            grid_inp = make_grid(vis_inp, nrow=2, normalize=True)
            grid_pred = make_grid(vis_pred, nrow=2, normalize=True)
            grid_gt = make_grid(vis_gt, nrow=2, normalize=True)
            
            writer.add_image("Val/Input", grid_inp, global_step=epoch + 1)
            writer.add_image("Val/Pred", grid_pred, global_step=epoch + 1)
            writer.add_image("Val/GT", grid_gt, global_step=epoch + 1)

    return avg_loss, avg_iou, avg_acc, avg_precision, avg_recall


def main():
    args = parse_args()

    # 0) 输出目录 & 日志
    os.makedirs(args.out_dir, exist_ok=True)
    time_suffix = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.out_dir, f"train_log_{time_suffix}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Start training with args: {args}")

    # 默认用 GPU (如有) 否则 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 默认启用 TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb_logs_" + time_suffix))
    else:
        logger.warning("TensorBoard is not available in your environment!")

    # 1) DataLoader
    train_loader = create_dataloader(
        data_dir=args.data_dir,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    val_loader = create_dataloader(
        data_dir=args.data_dir,
        split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    # 2) 使用FlexibleUNet模型
    model = FlexibleUNet(base_features=64, out_channels=1, bilinear=True)

    # 检查是否有多个 GPU 可用
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    # 移动模型到设备（自动处理多 GPU 的情况）
    model = model.to(device)

    # 优化器，添加权重衰减
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=5)

    # 初始化混合精度训练的 GradScaler
    scaler = GradScaler() if torch.cuda.is_available() else None

    # 定义最佳模型路径
    best_model_path = os.path.join(args.out_dir, "best_model.pth")

    # 若要恢复 checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        logger.info(f"Resumed from checkpoint '{args.resume}' (epoch {start_epoch}, val_loss={best_val_loss:.4f})")
    else:
        # 如果没有指定 resume，尝试加载已有的 best_model.pth
        if os.path.isfile(best_model_path):
            ckpt = torch.load(best_model_path, map_location=device)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(ckpt["model_state"])
            else:
                model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            logger.info(f"Loaded existing best checkpoint '{best_model_path}' (epoch {start_epoch}, val_loss={best_val_loss:.4f})")
        else:
            logger.info("No checkpoint specified or found. Starting training from scratch.")

    # 3) 训练循环
    # 添加早停机制（可选）
    early_stopping_patience = 10  # 连续 10 个 epoch 验证损失未改善则停止
    early_stopping_counter = 0

    for epoch in range(start_epoch, args.num_epochs):
        # -- train
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_epochs=args.num_epochs,
            logger=logger,
            writer=writer,
            log_interval=args.log_interval,
            scaler=scaler
        )

        # -- val
        if (epoch + 1) % args.val_interval == 0:
            val_loss, val_iou, val_acc, val_precision, val_recall = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                epoch=epoch,
                max_epochs=args.num_epochs,
                logger=logger,
                writer=writer
            )

            # 调度器步进
            scheduler.step(val_loss)

            # 如果更优，则保存 best_model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存新的 best_model.pth，会覆盖旧的
                torch.save({
                    "epoch": epoch + 1,
                    "model_state": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss
                }, best_model_path)
                logger.info(f"Saved best checkpoint at epoch {epoch+1}, val_loss={val_loss:.4f}")
                early_stopping_counter = 0  # 重置计数器
            else:
                early_stopping_counter += 1
                logger.info(f"No improvement in validation loss for {early_stopping_counter} epoch(s).")
                if early_stopping_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {early_stopping_patience} epochs without improvement.")
                    break  # 提前停止训练

    logger.info("Training finished!")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
