#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存高效SOTA模型训练脚本

功能：
1. 使用我们的loader.py加载数据
2. 训练内存高效SOTA模型
3. 支持梯度检查点和混合精度
4. 实时监控和可视化
5. 模型保存和恢复
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# 导入我们的模块
from loader import PartSegmentationDataset, get_dataloader
from model import create_model, ConfidenceWeightedLoss, FocalLoss

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """内存高效SOTA模型训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        set_seed(config.get('seed', 42))
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
        
        # 初始化混合精度
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # 初始化最佳指标
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        
        print(f"训练设备: {self.device}")
        print(f"混合精度: {'启用' if self.use_amp else '禁用'}")
        print(f"输出目录: {self.output_dir}")
    
    def build_datasets(self):
        """构建数据集"""
        print("构建数据集...")
        
        data_config = self.config['data']
        
        # 创建数据集
        self.train_dataset = PartSegmentationDataset(
            data_dir=data_config['data_dir'],
            split='train',
            config=data_config
        )
        
        self.val_dataset = PartSegmentationDataset(
            data_dir=data_config['data_dir'],
            split='val',
            config=data_config
        )
        
        self.test_dataset = PartSegmentationDataset(
            data_dir=data_config['data_dir'],
            split='test',
            config=data_config
        )
        
        # 打印数据集统计信息
        self.train_dataset.print_statistics()
        
        print(f"训练集: {len(self.train_dataset)} 样本")
        print(f"验证集: {len(self.val_dataset)} 样本")
        print(f"测试集: {len(self.test_dataset)} 样本")
    
    def build_dataloaders(self):
        """构建数据加载器"""
        print("构建数据加载器...")
        
        dataloader_config = self.config['dataloader']
        
        self.train_loader = get_dataloader(self.train_dataset, dataloader_config)
        
        # 验证和测试不需要shuffle
        val_config = dataloader_config.copy()
        val_config['shuffle'] = False
        self.val_loader = get_dataloader(self.val_dataset, val_config)
        self.test_loader = get_dataloader(self.test_dataset, val_config)
        
        print(f"训练批次: {len(self.train_loader)}")
        print(f"验证批次: {len(self.val_loader)}")
        print(f"测试批次: {len(self.test_loader)}")
    
    def build_model(self):
        """构建模型"""
        print("构建模型...")
        
        model_config = self.config['model']
        self.model = create_model(model_config)
        self.model.to(self.device)
        
        # 损失函数
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'confidence_weighted')
        
        if loss_type == 'confidence_weighted':
            self.criterion = ConfidenceWeightedLoss(
                base_loss=loss_config.get('base_loss', 'focal'),
                conf_weight=loss_config.get('conf_weight', 0.3),
                alpha=loss_config.get('alpha', 0.25),
                gamma=loss_config.get('gamma', 2.0)
            )
        elif loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=loss_config.get('alpha', 0.25),
                gamma=loss_config.get('gamma', 2.0)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        print(f"损失函数: {loss_type}")
    
    def build_optimizer(self):
        """构建优化器和调度器"""
        print("构建优化器...")
        
        opt_config = self.config['optimizer']
        
        # 优化器
        if opt_config['type'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.01),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        
        # 学习率调度器
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
        
        print(f"优化器: {opt_config['type']}")
        print(f"学习率调度: {scheduler_type}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (features, labels, sample_info) in enumerate(pbar):
            features = features.to(self.device)
            labels = labels.to(self.device)
            confidence = features[:, 448:449]  # 提取confidence
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    if self.config['training'].get('use_checkpointing', False):
                        logits = self.model.forward_with_checkpointing(features)
                    else:
                        logits = self.model(features)
                    
                    if isinstance(self.criterion, ConfidenceWeightedLoss):
                        loss = self.criterion(logits, labels, confidence)
                    else:
                        loss = self.criterion(logits, labels)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config['training'].get('max_grad_norm', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['max_grad_norm']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.config['training'].get('use_checkpointing', False):
                    logits = self.model.forward_with_checkpointing(features)
                else:
                    logits = self.model(features)
                
                if isinstance(self.criterion, ConfidenceWeightedLoss):
                    loss = self.criterion(logits, labels, confidence)
                else:
                    loss = self.criterion(logits, labels)
                
                loss.backward()
                
                # 梯度裁剪
                if self.config['training'].get('max_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['max_grad_norm']
                    )
                
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += labels.size(0)
            
            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = total_correct / total_samples
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{avg_acc:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # 记录到TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/Accuracy', avg_acc, global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)
        
        return total_loss / len(self.train_loader), total_correct / total_samples
    
    @torch.no_grad()
    def validate(self, dataloader, split_name='Val'):
        """验证模型"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc=f'{split_name}')
        
        for features, labels, sample_info in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)
            confidence = features[:, 448:449]
            
            if self.use_amp:
                with autocast():
                    logits = self.model(features)
                    if isinstance(self.criterion, ConfidenceWeightedLoss):
                        loss = self.criterion(logits, labels, confidence)
                    else:
                        loss = self.criterion(logits, labels)
            else:
                logits = self.model(features)
                if isinstance(self.criterion, ConfidenceWeightedLoss):
                    loss = self.criterion(logits, labels, confidence)
                else:
                    loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += labels.size(0)
            
            # 收集预测结果
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            avg_loss = total_loss / len(all_preds) * labels.size(0)
            avg_acc = total_correct / total_samples
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{avg_acc:.4f}'
            })
        
        # 计算详细指标
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / 'latest_checkpoint.pth')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
            print(f"💾 保存最佳模型 (Val F1: {metrics['val_f1']:.4f})")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']
    
    def train(self):
        """主训练循环"""
        print("开始训练...")
        
        # 构建所有组件
        self.build_datasets()
        self.build_dataloaders()
        self.build_model()
        self.build_optimizer()
        
        # 训练配置
        epochs = self.config['training']['epochs']
        start_epoch = 0
        
        # 恢复训练（如果需要）
        if self.config['training'].get('resume', False):
            checkpoint_path = self.output_dir / 'latest_checkpoint.pth'
            if checkpoint_path.exists():
                start_epoch, _ = self.load_checkpoint(checkpoint_path)
                print(f"从epoch {start_epoch} 恢复训练")
        
        # 训练历史
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        val_f1s = []
        
        # 训练循环
        for epoch in range(start_epoch, epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(self.val_loader, 'Val')
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 记录指标
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_metrics['loss'])
            val_accs.append(val_metrics['accuracy'])
            val_f1s.append(val_metrics['f1'])
            
            # TensorBoard记录
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_Acc', train_acc, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/Val_Acc', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Epoch/Val_F1', val_metrics['f1'], epoch)
            
            # 打印结果
            print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}")
            
            # 保存检查点
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1']
            }
            
            is_best = val_metrics['f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1']
                self.best_val_acc = val_metrics['accuracy']
            
            self.save_checkpoint(epoch, metrics, is_best)
            
            # 绘制训练曲线
            if (epoch + 1) % 5 == 0:
                self.plot_training_curves(train_losses, train_accs, val_losses, val_accs, val_f1s)
        
        # 最终测试
        print(f"\n{'='*60}")
        print("最终测试...")
        print(f"{'='*60}")
        
        # 加载最佳模型
        best_checkpoint_path = self.output_dir / 'best_model.pth'
        if best_checkpoint_path.exists():
            self.load_checkpoint(best_checkpoint_path)
        
        test_metrics = self.validate(self.test_loader, 'Test')
        
        print(f"\n最终测试结果:")
        print(f"Loss: {test_metrics['loss']:.4f}")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"F1: {test_metrics['f1']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        
        # 保存最终结果
        final_results = {
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'test_metrics': test_metrics
        }
        
        with open(self.output_dir / 'final_results.yaml', 'w') as f:
            yaml.dump(final_results, f)
        
        self.writer.close()
        print(f"\n训练完成！结果保存在: {self.output_dir}")
    
    def plot_training_curves(self, train_losses, train_accs, val_losses, val_accs, val_f1s):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss曲线
        axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy曲线
        axes[0, 1].plot(train_accs, label='Train Acc', color='blue')
        axes[0, 1].plot(val_accs, label='Val Acc', color='red')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1曲线
        axes[1, 0].plot(val_f1s, label='Val F1', color='green')
        axes[1, 0].set_title('F1 Score Curve')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率曲线（简化版本）
        if self.scheduler:
            # 简单显示当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            axes[1, 1].axhline(y=current_lr, color='orange', linestyle='--', label=f'Current LR: {current_lr:.6f}')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'No LR Scheduler', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="训练内存高效SOTA模型")
    parser.add_argument("--config", type=str, default="train_config.yaml", help="配置文件路径")
    parser.add_argument("--output_dir", type=str, help="输出目录（覆盖配置文件）")
    parser.add_argument("--resume", action="store_true", help="恢复训练")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.resume:
        config['training']['resume'] = True
    
    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 