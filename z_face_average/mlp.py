#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mlp.py · 448‑d PartField 特征 + 1‑d 置信度 → 面片二分类

✅ 数据文件
──────────
▸ PartField 特征
    /home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/urdf/{id}/feature/{id}.npy
      └─ 形状 = (N_faces, 448)

▸ 置信度（**每个面片都有**，无需补零）
    /home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/result/
      {class}_{id}_segmentation_{view}_joint_{joint_id}/pred_face_confidence.txt
      └─ 每行: "face_id confidence"

▸ 标签
    /home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/urdf/{id}/yy_visualization/
      moveable_ids_{joint_id}.txt
      └─ 面片 id（列出的 = 1，可动；未列出的 = 0）

⚙ 任务说明
──────────
• **所有视角文件都视为独立样本**，没有“数据增广”概念——就是更多样本。  
• 单 MLP 处理全部 joint_id；joint 信息已隐含在标签文件中。  
• (id, joint_id) 为单位随机划分 **train/val/test = 8:1:1**，保存 splits.json。  
• 支持两种模式：
    • train —— 训练并在训练过程中用 val 集监控；最终同时输出 best & final 权重  
    • infer —— 加载权重，对 test 集全部样本推理，保存预测与标签

"""

# ----------------------------------------------------------------------------- 
#                                    Imports
# -----------------------------------------------------------------------------
import os, json, glob, random, math, argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------------------------------------------------------- 
#                               Utility — seeds
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------------------------- 
#                              MLP Architecture
# -----------------------------------------------------------------------------
class FaceMLP(nn.Module):
    def __init__(self,
                 input_dim: int = 449,
                 hidden_dims: List[int] = [512, 256, 128],
                 output_dim: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                # 使用LayerNorm替代BatchNorm，LayerNorm不受批大小限制
                nn.LayerNorm(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ----------------------------------------------------------------------------- 
#                             Dataset Definition
# -----------------------------------------------------------------------------
class ConfidencePartFieldDataset(Dataset):
    """
    每条样本:
        • (N_faces, 449) 特征矩阵 - 整个几何体的所有面片
        • (N_faces,) 标签向量 - 对应的标签
    按 (id, joint_id, view) 读取置信度文件；使用所有视角的数据。
    批次大小恒为1，因为每个几何体的面片数不同。
    """
    def __init__(self,
                 data_root: str,
                 split_json: Dict[str, List[List[str]]] | None,
                 split: str,
                 view_filter: str = None,  # 修改为None，表示不过滤视角
                 max_samples: int = None):     # 可选：限制样本数量
        assert split in {'train', 'val', 'test', 'all'}
        self.data_root = Path(data_root)
        self.urdf_dir = self.data_root / 'urdf'
        self.result_dir = self.data_root / 'result'
        self.view_filter = view_filter
        
        view_msg = f"只使用视角={view_filter}" if view_filter else "使用所有视角"
        print(f"\n初始化数据集: split={split}, {view_msg}")
        print(f"data_root: {self.data_root}")
        print(f"urdf_dir: {self.urdf_dir} (存在: {os.path.exists(self.urdf_dir)})")
        print(f"result_dir: {self.result_dir} (存在: {os.path.exists(self.result_dir)})")
        
        if split != 'all' and split_json is not None:
            print(f"该拆分中的(id, joint)对数量: {len(split_json[split])}")
            if len(split_json[split]) > 0:
                print(f"前5个(id, joint)对: {split_json[split][:5]}")

        # ---------------------------------------------------- 
        #      收集样本索引列表 - 每个几何体一条记录
        # ----------------------------------------------------
        self.items: List[Dict] = []
        glob_pattern = str(self.result_dir / '*/pred_face_confidence.txt')
        print(f"搜索文件: {glob_pattern}")
        conf_paths = glob.glob(glob_pattern)
        print(f"找到 {len(conf_paths)} 个置信度文件")
        
        for cpath in conf_paths:
            dir_name = Path(cpath).parent.name
            
            # 解析目录名获取信息
            parts = dir_name.split('_')
            if len(parts) < 5 or "segmentation" not in parts or "joint" not in parts:
                print(f"警告: 无法解析目录 {dir_name}，跳过")
                continue
                
            # 提取类别和ID
            class_idx = 0
            id_idx = 1
            cls = parts[class_idx]
            mid = parts[id_idx]
            
            # 提取joint_id（通常是最后一个元素）
            joint = parts[-1]
            
            # 提取view信息
            try:
                seg_idx = parts.index("segmentation")
                joint_idx = parts.index("joint")
                view = '_'.join(parts[seg_idx+1:joint_idx])
            except ValueError:
                print(f"警告: 目录名 {dir_name} 格式异常，尝试备用解析")
                try:
                    view = '_'.join(parts[3:-2])  # 尝试原始方式
                except:
                    view = "unknown"
                    print(f"警告: 无法确定视角名称，使用默认值 'unknown'")
            
            # 只有设置了视角过滤器时才过滤视角
            if self.view_filter and view != self.view_filter:
                continue
                
            tag = [mid, joint]
            if split != 'all' and split_json is not None:
                if tag not in split_json[split]:
                    continue
            
            # 检查特征文件是否存在
            feat_path = self.urdf_dir / mid / 'feature' / f"{mid}.npy"
            if not os.path.exists(feat_path):
                print(f"警告: 特征文件不存在 {feat_path}")
                continue
                
            # 检查标签文件是否存在
            label_path = (self.urdf_dir / mid / 'yy_visualization' / f"moveable_ids_{joint}.txt")
            if not os.path.exists(label_path) and split != 'infer':
                print(f"警告: 标签文件不存在 {label_path}")
                continue
            
            self.items.append({
                'class': cls,
                'id': mid,
                'joint': int(joint),
                'view': view,
                'conf_path': cpath,
                'feature_path': str(feat_path),
                'label_path': str(label_path)
            })
        
        # 如果需要限制样本数量（用于调试或快速实验）
        if max_samples and max_samples < len(self.items):
            self.items = random.sample(self.items, max_samples)
            
        print(f"为拆分 {split} 收集了 {len(self.items)} 个有效几何体")
        if not self.items:
            print("错误: 没有找到任何有效样本!")
        
        assert self.items, f'No data for split={split}'

    def __len__(self) -> int:
        return len(self.items)
        
    def __getitem__(self, idx: int) -> Dict:
        """返回一个完整几何体的所有面片特征和标签"""
        item = self.items[idx]
        
        # 加载特征矩阵 - (N_faces, 448)
        feat448 = np.load(item['feature_path'])
        n_faces = feat448.shape[0]
        
        # 加载置信度 - (N_faces,)
        conf_vec = np.zeros(n_faces, dtype=np.float32)
        with open(item['conf_path'], 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:  # 确保行格式正确
                    fid, val = int(parts[0]), float(parts[1])
                    if 0 <= fid < n_faces:
                        conf_vec[fid] = val
        
        # 加载标签 - (N_faces,)
        label_vec = np.zeros(n_faces, dtype=np.int64)
        if os.path.exists(item['label_path']):
            with open(item['label_path'], 'r') as f:
                for line in f:
                    try:
                        fid = int(line.strip())
                        if 0 <= fid < n_faces:
                            label_vec[fid] = 1
                    except ValueError:
                        pass  # 忽略无法解析的行
        
        # 组合特征 - (N_faces, 449)
        feat449 = np.concatenate([feat448, conf_vec[:, None]], axis=1)
        
        return {
            'feature': torch.from_numpy(feat449.astype(np.float32)),
            'label': torch.from_numpy(label_vec),
            'model_id': item['id'],
            'joint_id': item['joint'],
            'n_faces': n_faces
        }

# ----------------------------------------------------------------------------- 
#                            Split‑file Generation
# -----------------------------------------------------------------------------
def build_split_json(data_root: str, out_json: str, seed: int = 42) -> Dict:
    if os.path.isfile(out_json):
        print(f"加载已有splits文件: {out_json}")
        with open(out_json, 'r') as f:
            splits = json.load(f)
            print(f"拆分统计: 训练={len(splits['train'])}, 验证={len(splits['val'])}, 测试={len(splits['test'])}")
            return splits

    print(f"创建新的splits文件...")
    result_dir = Path(data_root) / 'result'
    print(f"正在搜索: {result_dir}")
    
    # 先检查目录是否存在
    if not os.path.exists(result_dir):
        print(f"错误: 目录 {result_dir} 不存在")
        return {'train': [], 'val': [], 'test': []}
    
    # 使用与ConfidencePartFieldDataset中相同的路径模式
    conf_paths = glob.glob(str(result_dir / '*/pred_face_confidence.txt'))
    print(f"找到 {len(conf_paths)} 个置信度文件")
    
    combos: List[Tuple[str, str]] = []
    
    for cpath in conf_paths:
        # 目录名形如 "Dishwasher_11622_segmentation_center_joint_0"
        dir_name = Path(cpath).parent.name
        
        # 解析目录名获取信息
        parts = dir_name.split('_')
        if len(parts) < 5 or "segmentation" not in parts or "joint" not in parts:
            print(f"跳过格式不匹配的目录: {dir_name}")
            continue
            
        # 提取ID和关节ID
        mid = parts[1]
        joint = parts[-1]
        
        print(f"解析目录: {dir_name} -> id={mid}, joint={joint}")
        combos.append((mid, joint))
    
    # 去重
    combos = list(set(combos))
    print(f"找到 {len(combos)} 个唯一(id, joint)组合")
    
    if not combos:
        print("错误: 没有找到有效的(id, joint)组合")
        return {'train': [], 'val': [], 'test': []}
    
    random.Random(seed).shuffle(combos)

    n = len(combos)
    train_n = int(0.8 * n)
    val_n = int(0.1 * n)
    splits = {
        'train': [list(c) for c in combos[:train_n]],
        'val':   [list(c) for c in combos[train_n:train_n+val_n]],
        'test':  [list(c) for c in combos[train_n+val_n:]]
    }
    
    print(f"拆分完成: 训练={len(splits['train'])}, 验证={len(splits['val'])}, 测试={len(splits['test'])}")
    print(f"训练集前5个: {splits['train'][:5]}")
    print(f"验证集前5个: {splits['val'][:5]}")
    print(f"测试集前5个: {splits['test'][:5]}")
    
    with open(out_json, 'w') as f:
        json.dump(splits, f, indent=2)
        
    print(f"分割文件已保存到: {out_json}")
    return splits

# ----------------------------------------------------------------------------- 
#                           Train / Validate helpers
# -----------------------------------------------------------------------------
def loop(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
        desc = "Training"
    else:
        model.eval()
        desc = "Validating"
    total_loss, correct, total = 0.0, 0, 0
    
    # 添加tqdm进度条
    pbar = tqdm(loader, desc=f"{desc}", ncols=100)
    for batch in pbar:
        # 特征和标签形状为 [1, N_faces, 449] 和 [1, N_faces]
        # 因为每个几何体的面片数不同，每个批次只包含一个几何体
        x = batch['feature'].to(device)  # [1, N_faces, 449]
        y = batch['label'].to(device)    # [1, N_faces]
        
        # 重塑为 [N_faces, 449] 和 [N_faces]
        x = x.squeeze(0)
        y = y.squeeze(0)
        
        if train:
            optimizer.zero_grad()
            
        out = model(x)  # [N_faces, 2]
        loss = criterion(out, y)
        
        if train:
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        
        # 更新进度条显示当前损失和准确率
        if total > 0:
            pbar.set_postfix({
                'loss': f"{total_loss/total:.4f}", 
                'acc': f"{correct/total:.4f}"
            })
    
    return total_loss / total, correct / total

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    results = []
    
    # 添加tqdm进度条
    pbar = tqdm(loader, desc="Predicting", ncols=100)
    for batch in pbar:
        x = batch['feature'].to(device).squeeze(0)  # [N_faces, 449]
        y = batch['label'].to(device).squeeze(0)    # [N_faces]
        model_id = batch['model_id'][0]
        joint_id = batch['joint_id'].item()
        n_faces = batch['n_faces'].item()
        
        # 如果有view信息，也获取
        view = batch.get('view', ['unknown'])[0]
        
        out = model(x)  # [N_faces, 2]
        pred_labels = out.argmax(1).cpu().numpy()
        true_labels = y.cpu().numpy()
        
        # 收集结果
        results.append({
            'model_id': model_id,
            'joint_id': joint_id,
            'view': view,
            'n_faces': n_faces,
            'predictions': pred_labels,
            'ground_truth': true_labels
        })
        
        pbar.set_postfix({'model': f"{model_id}_joint_{joint_id}"})
    
    return results

# ----------------------------------------------------------------------------- 
#                                     Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'infer'], default='train')
    parser.add_argument('--data_root', required=True,
                        help='data_small 根目录')
    parser.add_argument('--output_dir', required=True,
                        help='输出目录')
    parser.add_argument('--model_path', default=None,
                        help='infer 模式下的模型权重')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='恒为1，因为每个几何体的面片数不同')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--hidden_dims', type=int, nargs='+',
                        default=[512, 256, 128])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='限制每个集合的最大样本数（用于调试）')
    parser.add_argument('--view_filter', type=str, default=None,
                        help='指定视角过滤器，如"center"。不指定则使用所有视角')
    args = parser.parse_args()

    # 强制batch_size=1，因为每个几何体的面片数不同
    args.batch_size = 1
    
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    split_json = build_split_json(args.data_root,
                                  Path(args.output_dir) / 'splits.json',
                                  seed=args.seed)

    # -------------------------------------------------------- 
    #                         TRAIN
    # --------------------------------------------------------
    if args.mode == 'train':
        ds_train = ConfidencePartFieldDataset(args.data_root, split_json, 'train', 
                                             view_filter=args.view_filter, max_samples=args.max_samples)
        ds_val = ConfidencePartFieldDataset(args.data_root, split_json, 'val', 
                                           view_filter=args.view_filter, max_samples=args.max_samples)
        
        dl_train = DataLoader(ds_train, batch_size=1, shuffle=True, 
                             num_workers=4, pin_memory=True)
        dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, 
                           num_workers=4, pin_memory=True)

        model = FaceMLP(input_dim=449,
                        hidden_dims=args.hidden_dims,
                        dropout=args.dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.5, patience=5, verbose=True)

        hist = {'train_loss': [], 'val_loss': [],
                'train_acc': [], 'val_acc': []}
        best_val_loss = math.inf

        for epoch in range(1, args.epochs+1):
            tr_loss, tr_acc = loop(model, dl_train, criterion, optimizer, device, train=True)
            val_loss, val_acc = loop(model, dl_val,   criterion, optimizer, device, train=False)
            scheduler.step(val_loss)

            hist['train_loss'].append(tr_loss)
            hist['val_loss'].append(val_loss)
            hist['train_acc'].append(tr_acc)
            hist['val_acc'].append(val_acc)

            print(f"[{epoch:02d}/{args.epochs}] "
                  f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), Path(args.output_dir) / 'best_model.pth')
                print("✓ best model updated")

        torch.save(model.state_dict(), Path(args.output_dir) / 'final_model.pth')
        print("✓ final model saved")

        # 绘制曲线
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(hist['train_loss'], label='train')
        plt.plot(hist['val_loss'], label='val')
        plt.title('Loss'); plt.legend()
        plt.subplot(1,2,2)
        plt.plot(hist['train_acc'], label='train')
        plt.plot(hist['val_acc'], label='val')
        plt.title('Accuracy'); plt.legend()
        plt.tight_layout()
        plt.savefig(Path(args.output_dir) / 'training_curves.png')
        plt.close()

    # -------------------------------------------------------- 
    #                         INFER
    # --------------------------------------------------------
    if args.mode == 'infer':
        assert args.model_path, "--model_path 必须提供"
        model = FaceMLP(input_dim=449,
                        hidden_dims=args.hidden_dims,
                        dropout=0.0).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))

        ds_test = ConfidencePartFieldDataset(args.data_root, split_json, 'test', 
                                           view_filter=args.view_filter, max_samples=args.max_samples)
        dl_test = DataLoader(ds_test, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True)
        
        # 获取详细预测结果
        results = predict(model, dl_test, device)
        
        # 统计整体指标
        all_preds = []
        all_gts = []
        
        # 为每个样本创建结果目录和文件
        for res in results:
            model_id = res['model_id']
            joint_id = res['joint_id']
            view = res['view']
            preds = res['predictions']
            gts = res['ground_truth']
            
            # 添加到全局结果
            all_preds.extend(preds)
            all_gts.extend(gts)
            
            # 构建目录名称
            # 注意：尝试构建与原始数据集相同结构的目录名称
            if view == 'center':  # 最常见的情况
                # 获取类别前缀
                class_prefix = None
                for item in ds_test.items:
                    if item['id'] == model_id and int(item['joint']) == joint_id:
                        class_prefix = item['class']
                        break
                
                if class_prefix is None:
                    class_prefix = "Unknown"  # 如果无法确定类别，使用默认值
                
                result_dir_name = f"{class_prefix}_{model_id}_segmentation_{view}_joint_{joint_id}"
            else:
                # 为其他视角使用相同格式
                class_prefix = None
                for item in ds_test.items:
                    if item['id'] == model_id and int(item['joint']) == joint_id:
                        class_prefix = item['class']
                        break
                
                if class_prefix is None:
                    class_prefix = "Unknown"
                    
                result_dir_name = f"{class_prefix}_{model_id}_segmentation_{view}_joint_{joint_id}"
            
            # 创建结果目录
            result_dir = Path(args.output_dir) / result_dir_name
            os.makedirs(result_dir, exist_ok=True)
            
            # 保存预测结果：只保存预测为1（可动）的面片ID
            pred_moveable_ids = np.where(preds == 1)[0]
            pred_result_path = result_dir / "pred_moveable_ids.txt"
            with open(pred_result_path, 'w') as f:
                for face_id in pred_moveable_ids:
                    f.write(f"{face_id}\n")
            
            # 保存真实标签：只保存真实为1（可动）的面片ID
            gt_moveable_ids = np.where(gts == 1)[0]
            gt_result_path = result_dir / "gt_moveable_ids.txt"
            with open(gt_result_path, 'w') as f:
                for face_id in gt_moveable_ids:
                    f.write(f"{face_id}\n")
            
            print(f"已保存结果到: {result_dir}")
            print(f"  - 预测可动面片数: {len(pred_moveable_ids)}")
            print(f"  - 真实可动面片数: {len(gt_moveable_ids)}")
        
        # 同时保存整体指标和结果
        all_preds = np.array(all_preds)
        all_gts = np.array(all_gts)
        
        # 计算整体指标
        acc = accuracy_score(all_gts, all_preds)
        prec = precision_score(all_gts, all_preds, zero_division=0)
        rec = recall_score(all_gts, all_preds, zero_division=0)
        f1 = f1_score(all_gts, all_preds, zero_division=0)
        
        # 保存整体指标
        summary_path = Path(args.output_dir) / "summary_metrics.txt"
        with open(summary_path, 'w') as f:
            f.write(f"总样本数: {len(results)}\n")
            f.write(f"总面片数: {len(all_preds)}\n")
            f.write(f"精度: {acc:.4f}\n")
            f.write(f"精确率: {prec:.4f}\n")
            f.write(f"召回率: {rec:.4f}\n")
            f.write(f"F1分数: {f1:.4f}\n")
        
        print("\n总体评估指标:")
        print(f"精度: {acc:.4f}, 精确率: {prec:.4f}, 召回率: {rec:.4f}, F1: {f1:.4f}")
        print(f"详细指标已保存至: {summary_path}")

if __name__ == '__main__':
    main()
