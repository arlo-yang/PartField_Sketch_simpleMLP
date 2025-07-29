# ===========================
# file: training.py
# ===========================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版训练脚本（Attention-MLP）
- 使用 SimpleAttentionMLP（PMA 全局上下文 + MLP）
- AMP（可选 bf16）+ GradScaler
- ReduceLROnPlateau / Cosine / OneCycle
- EMA、早停、完整指标（Acc/Precision/Recall/F1/AUROC）
- 全链路 NaN/Inf 防护（清洗输入 & 检测 loss）
"""

import os
import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from model import create_model, ConfidenceAwareLoss


# 你现有的数据接口（保持不变）
from loader import PartSegmentationDataset, get_dataloader


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _sanitize_tensor(x: torch.Tensor, clamp_val: float = 1e4) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
    if clamp_val is not None:
        x = x.clamp(min=-clamp_val, max=clamp_val)
    return x


class ModelEMA:
    """轻量 EMA"""
    def __init__(self, model: nn.Module, model_config: dict, decay: float = 0.999, device=None):
        self.device = device if device is not None else next(model.parameters()).device
        self.ema = create_model(model_config)
        self.ema.to(self.device)
        self.ema.load_state_dict(model.state_dict(), strict=True)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].to(v.dtype) * (1.0 - self.decay))

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.ema.state_dict(), strict=True)


class Trainer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seed(cfg.get('seed', 42))

        # 可开启 TF32（对 A100/3090 等有效）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # 路径
        self.out_dir = Path(cfg['output_dir'])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # 日志
        self.writer = SummaryWriter(log_dir=self.out_dir / 'tb')

        # AMP
        self.use_amp = cfg.get('use_amp', True) and torch.cuda.is_available()
        amp_dtype = cfg.get('training', {}).get('amp_dtype', 'fp16').lower()
        if amp_dtype not in ['fp16', 'bf16']:
            amp_dtype = 'fp16'
        self.amp_dtype = torch.float16 if amp_dtype == 'fp16' else torch.bfloat16

        scaler_cfg = cfg.get('training', {}).get('grad_scaler', {})
        self.scaler = GradScaler(
            enabled=self.use_amp and self.amp_dtype == torch.float16,
            init_scale=scaler_cfg.get('init_scale', 2.0 ** 10),
            growth_factor=scaler_cfg.get('growth_factor', 2.0),
            backoff_factor=scaler_cfg.get('backoff_factor', 0.5),
            growth_interval=scaler_cfg.get('growth_interval', 2000)
        )

        # 数据
        self._build_datasets_and_loaders()

        # 模型/损失/优化
        self._build_model()
        self._build_optim()

        # EMA
        self.use_ema = cfg['training'].get('use_ema', True)
        self.ema = ModelEMA(self.model, self.cfg['model'], decay=cfg['training'].get('ema_decay', 0.999), device=self.device) if self.use_ema else None

        # 早停
        es_cfg = cfg['training'].get('early_stopping', {})
        self.es_patience = es_cfg.get('patience', 0)
        self.es_min_delta = es_cfg.get('min_delta', 1e-4)
        self.es_monitor = es_cfg.get('monitor', 'val_f1')
        self.es_best = -float('inf')
        self.es_wait = 0

        self.skip_bad_batches = cfg['training'].get('skip_bad_batches', True)

    def _build_datasets_and_loaders(self):
        data_cfg = self.cfg['data']
        self.train_dataset = PartSegmentationDataset(data_dir=data_cfg['data_dir'], split='train', config=data_cfg)
        self.val_dataset = PartSegmentationDataset(data_dir=data_cfg['data_dir'], split='val', config=data_cfg)
        self.test_dataset = PartSegmentationDataset(data_dir=data_cfg['data_dir'], split='test', config=data_cfg)

        dl_cfg = self.cfg['dataloader']
        self.train_loader = get_dataloader(self.train_dataset, dl_cfg)

        dl_val = dict(dl_cfg)
        dl_val['shuffle'] = False
        self.val_loader = get_dataloader(self.val_dataset, dl_val)
        self.test_loader = get_dataloader(self.test_dataset, dl_val)

        print(f"Train/Val/Test sizes: {len(self.train_dataset)}/{len(self.val_dataset)}/{len(self.test_dataset)}")
        print(f"Batches: {len(self.train_loader)}/{len(self.val_loader)}/{len(self.test_loader)}")

    def _build_model(self):
        mcfg = self.cfg['model']
        self.model = create_model(mcfg).to(self.device)
        # 损失
        lcfg = self.cfg.get('loss', {})
        self.criterion = ConfidenceAwareLoss(
            alpha=lcfg.get('alpha', 0.25),
            gamma=lcfg.get('gamma', 1.5),
            conf_weight=lcfg.get('conf_weight', 0.1),
            reduction='mean'
        )

    def _build_optim(self):
        ocfg = self.cfg['optimizer']
        if ocfg['type'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=ocfg['lr'],
                weight_decay=ocfg.get('weight_decay', 1e-3),
                betas=ocfg.get('betas', (0.9, 0.999)),
                eps=ocfg.get('eps', 1e-8)
            )
        elif ocfg['type'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=ocfg['lr'],
                weight_decay=ocfg.get('weight_decay', 0.0),
                eps=ocfg.get('eps', 1e-8)
            )
        else:
            raise ValueError(f"Unknown optimizer {ocfg['type']}")

        scfg = self.cfg.get('scheduler', {'type': 'reduce_on_plateau'})
        stype = scfg.get('type', 'reduce_on_plateau').lower()
        epochs = self.cfg['training']['epochs']

        if stype in ['plateau', 'reduce_on_plateau']:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                factor=scfg.get('factor', 0.5), patience=scfg.get('patience', 5),
                min_lr=scfg.get('min_lr', 1e-6), threshold=scfg.get('threshold', 1e-3), verbose=True
            )
        elif stype == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=scfg.get('eta_min', 1e-6)
            )
        elif stype == 'onecycle':
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=ocfg['lr'],
                epochs=epochs, steps_per_epoch=len(self.train_loader),
                pct_start=scfg.get('pct_start', 0.1),
                div_factor=scfg.get('div_factor', 25.0),
                final_div_factor=scfg.get('final_div_factor', 1e4),
                three_phase=scfg.get('three_phase', False),
                anneal_strategy=scfg.get('anneal_strategy', 'cos')
            )
        else:
            self.scheduler = None

    def _step_scheduler(self, val_loss: float, epoch_end: bool = False):
        if self.scheduler is None:
            return
        stype = self.cfg.get('scheduler', {}).get('type', 'reduce_on_plateau').lower()
        if stype in ['plateau', 'reduce_on_plateau']:
            self.scheduler.step(val_loss)
        elif stype == 'onecycle':
            pass
        else:
            if epoch_end:
                self.scheduler.step()

    def _is_bad_tensor(self, t: torch.Tensor) -> bool:
        return torch.isnan(t).any() or torch.isinf(t).any()

    def _clean_features(self, features: torch.Tensor) -> torch.Tensor:
        return _sanitize_tensor(features)

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        max_grad_norm = self.cfg['training'].get('max_grad_norm', 0.0)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for it, (features, labels, sample_info) in enumerate(pbar):
            features = self._clean_features(features.to(self.device, non_blocking=True))
            labels = labels.to(self.device, non_blocking=True)

            if self._is_bad_tensor(features):
                if self.skip_bad_batches:
                    print(f"⚠️  Warning: Bad features (NaN/Inf) at epoch {epoch+1}, iter {it+1}, skipping batch...")
                    continue
                else:
                    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            confidence = torch.clamp(torch.nan_to_num(features[:, 448:449], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            features = features.clone()
            features[:, 448:449] = confidence

            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    logits = self.model(features)
                    loss = self.criterion(logits, labels, confidence)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  Warning: NaN/Inf loss detected at epoch {epoch+1}, iter {it+1}, skipping...")
                    continue

                self.scaler.scale(loss).backward()
                if max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(features)
                loss = self.criterion(logits, labels, confidence)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  Warning: NaN/Inf loss detected at epoch {epoch+1}, iter {it+1}, skipping...")
                    continue

                loss.backward()
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

            if self.ema is not None:
                self.ema.update(self.model)

            total_loss += float(loss.item()) * labels.size(0)
            pred = logits.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += labels.size(0)

            # 进度条
            cur_loss = total_loss / max(1, total_samples)
            cur_acc = total_correct / max(1, total_samples)
            lr0 = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{cur_loss:.4f}',
                'Acc': f'{cur_acc:.4f}',
                'LR': f'{lr0:.6f}'
            })

            # TensorBoard（按迭代）
            if (it + 1) % self.cfg['training'].get('log_interval', 50) == 0:
                step = epoch * len(self.train_loader) + it
                self.writer.add_scalar("Train/Loss_iter", cur_loss, step)
                self.writer.add_scalar("Train/Acc_iter", cur_acc, step)
                self.writer.add_scalar("Train/LR", lr0, step)

        avg_loss = total_loss / max(1, total_samples)
        avg_acc = total_correct / max(1, total_samples)
        self.writer.add_scalar("Train/Loss_epoch", avg_loss, epoch)
        self.writer.add_scalar("Train/Acc_epoch", avg_acc, epoch)
        return avg_loss, avg_acc

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, split: str = "Val"):
        model = self.model
        model.eval()

        total_loss = 0.0
        all_probs = []
        all_preds = []
        all_labels = []

        pbar = tqdm(loader, desc=f'{split}')
        for features, labels, sample_info in pbar:
            features = self._clean_features(features.to(self.device, non_blocking=True))
            labels = labels.to(self.device, non_blocking=True)
            confidence = torch.clamp(torch.nan_to_num(features[:, 448:449], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

            logits = model(features)
            loss = self.criterion(logits, labels, confidence)

            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=1)

            total_loss += float(loss.item()) * labels.size(0)
            all_probs.append(torch.clamp(probs, 0.0, 1.0).cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            cur_loss = total_loss / max(1, labels.size(0))
            pbar.set_postfix({'Loss': f'{cur_loss:.4f}'})

        probs = torch.cat(all_probs, dim=0).numpy()
        preds = torch.cat(all_preds, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        avg_loss = total_loss / max(1, labels.shape[0])

        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)

        try:
            auroc = roc_auc_score(labels, probs)
        except Exception:
            auroc = float('nan')

        print(f"[{split}] Loss {avg_loss:.4f} | Acc {acc:.4f} | F1 {f1:.4f} | AUROC {auroc:.4f} | P {precision:.4f} | R {recall:.4f}")
        return {
            'loss': avg_loss,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auroc': auroc
        }

    def _should_early_stop(self, metric_value: float) -> bool:
        if self.es_patience <= 0:
            return False
        improved = metric_value > self.es_best + self.es_min_delta
        if improved:
            self.es_best = metric_value
            self.es_wait = 0
            return False
        else:
            self.es_wait += 1
            print(f"[EarlyStop] wait {self.es_wait}/{self.es_patience}")
            return self.es_wait >= self.es_patience

    def train(self):
        epochs = self.cfg['training']['epochs']
        best_f1 = -1.0
        best_metrics = None

        start_epoch = 0
        if self.cfg['training'].get('resume', False):
            ckpt = self.out_dir / 'latest.pth'
            if ckpt.exists():
                print(f"Resuming from {ckpt}")
                data = torch.load(ckpt, map_location=self.device)
                self.model.load_state_dict(data['model'])
                self.optimizer.load_state_dict(data['optim'])
                if self.scheduler is not None and data.get('sched', None) is not None:
                    self.scheduler.load_state_dict(data['sched'])
                if self.use_amp and data.get('scaler', None) is not None:
                    self.scaler.load_state_dict(data['scaler'])
                if self.ema is not None and data.get('ema', None) is not None:
                    self.ema.ema.load_state_dict(data['ema'])
                start_epoch = data.get('epoch', 0)

        for epoch in range(start_epoch, epochs):
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_metrics = self._evaluate(self.val_loader, split="Val")

            # 调度
            self._step_scheduler(val_metrics['loss'], epoch_end=True)

            # 记录
            self.writer.add_scalar("Val/Loss", val_metrics['loss'], epoch)
            self.writer.add_scalar("Val/Acc", val_metrics['accuracy'], epoch)
            self.writer.add_scalar("Val/F1", val_metrics['f1'], epoch)
            if not np.isnan(val_metrics['auroc']):
                self.writer.add_scalar("Val/AUROC", val_metrics['auroc'], epoch)

            # 保存
            is_best = val_metrics['f1'] > best_f1
            if is_best:
                best_f1 = val_metrics['f1']
                best_metrics = val_metrics
                self._save_checkpoint(epoch, best=True)
            self._save_checkpoint(epoch, best=False)

            # 早停
            monitor_value = val_metrics['f1'] if self.es_monitor.endswith('f1') else val_metrics['accuracy']
            if self._should_early_stop(monitor_value):
                print(f"Early stopping at epoch {epoch+1}")
                break

        # 测试（使用 EMA）
        if self.ema is not None:
            print("Evaluating EMA weights...")
            ema_backup = create_model(self.cfg['model']).to(self.device)
            ema_backup.load_state_dict(self.model.state_dict(), strict=True)
            self.ema.copy_to(self.model)

        test_metrics = self._evaluate(self.test_loader, split="Test")
        print("\nFinal Test Metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v}")

        # 保存最终结果（转基础类型）
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating, np.ndarray)):
                v = float(obj)
                if np.isnan(v) or np.isinf(v):
                    return None
                return v
            else:
                return obj
        
        results = {
            'best_val': convert_numpy(best_metrics),
            'test': convert_numpy(test_metrics)
        }
        with open(self.out_dir / "final_results.yaml", 'w', encoding='utf-8') as f:
            yaml.safe_dump(results, f, allow_unicode=True)

        if self.ema is not None:
            self.model.load_state_dict(ema_backup.state_dict(), strict=True)

        print(f"\n训练完成！输出目录：{self.out_dir}")

    def _save_checkpoint(self, epoch: int, best: bool = False):
        data = {
            'epoch': epoch + 1,
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'sched': self.scheduler.state_dict() if self.scheduler is not None else None,
            'scaler': self.scaler.state_dict() if (self.use_amp and self.amp_dtype == torch.float16) else None,
            'ema': self.ema.ema.state_dict() if self.ema is not None else None,
            'config': self.cfg
        }
        path = self.out_dir / ('best.pth' if best else 'latest.pth')
        torch.save(data, path)
        if best:
            print(f"✅ Saved BEST checkpoint to {path}")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="训练简化版 Attention-MLP（PMA + MLP）")
    parser.add_argument("--config", type=str, default="train_config.yaml", help="配置文件路径")
    parser.add_argument("--output_dir", type=str, help="输出目录（覆盖配置）")
    parser.add_argument("--resume", action="store_true", help="从 latest.pth 恢复")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.output_dir:
        cfg['output_dir'] = args.output_dir
    if args.resume:
        cfg.setdefault('training', {})['resume'] = True

    # 默认值填充
    cfg.setdefault('model', {})
    cfg['model'].setdefault('embed_dim', 128)
    cfg['model'].setdefault('num_heads', 4)
    cfg['model'].setdefault('num_seeds', 8)
    cfg['model'].setdefault('dropout', 0.1)

    cfg.setdefault('optimizer', {'type': 'adamw', 'lr': 1e-4, 'weight_decay': 1e-3, 'eps': 1e-8})
    cfg.setdefault('scheduler', {'type': 'reduce_on_plateau', 'factor': 0.5, 'patience': 5, 'min_lr': 1e-6, 'threshold': 1e-3})
    cfg.setdefault('training', {'epochs': 80, 'use_ema': True, 'ema_decay': 0.999, 'log_interval': 50, 'skip_bad_batches': True, 'amp_dtype': 'fp16'})
    cfg.setdefault('loss', {'alpha': 0.25, 'gamma': 1.5, 'conf_weight': 0.1})
    cfg.setdefault('data', {})
    cfg.setdefault('dataloader', {'batch_size': 1, 'shuffle': True, 'num_workers': 4, 'pin_memory': True})

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
