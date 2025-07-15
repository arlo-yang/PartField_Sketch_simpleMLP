"""
yy_mask_gt_field_mlp_inference.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
使用训练好的模型在验证集上进行推理和可视化。
- 加载训练好的模型权重
- 在验证集上执行推理
- 生成可视化结果（彩色PLY文件）
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
import matplotlib.pyplot as plt
import trimesh

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# 复用训练代码中的定义
# ------------------------- argparse & seed ------------------------- #

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base_dir",    type=str, required=True,
                   help="根目录 /hy-tmp/PartField_Sketch_simpleMLP")
    p.add_argument("--model_path",  type=str, default="best_model.pth",
                   help="训练好的模型权重路径")
    p.add_argument("--feature_dim", type=int, default=448)
    p.add_argument("--img_size",    type=int, default=512)
    p.add_argument("--hidden_dim",  type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pin_mem",     action="store_true", default=True)
    p.add_argument("--prefetch",    type=int, default=4)
    p.add_argument("--output_dir",  type=str, default="inference_results",
                   help="结果保存目录")
    p.add_argument("--threshold",   type=float, default=0.5,
                   help="预测的二值化阈值")
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

        # 加载原始数据用于可视化
        vertices_path = os.path.join(self.feature_dir, mid, "yy_merged.obj")
        has_mesh = os.path.exists(vertices_path)
        
        features = np.load(feat_path).astype(np.float32)
        labels   = np.load(label_path).astype(np.float32)
        if len(features) != len(labels):
            m = min(len(features), len(labels))
            features, labels = features[:m], labels[:m]

        result = {
            'features': torch.from_numpy(features),
            'labels': torch.from_numpy(labels),
            'arrow': self.rgb_tf(Image.open(asp_path).convert("RGB")),
            'depth': self.gray_tf(Image.open(dep_path).convert("L")),
            'normal': self.rgb_tf(Image.open(nor_path).convert("RGB")),
            'meta': {
                'id': mid,
                'category': cat,
                'joint': jid,
                'view': view,
                'asp_path': asp_path,
                'vertices_path': vertices_path if has_mesh else None,
                'has_mesh': has_mesh
            }
        }
        return result

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

# ------------------------- 推理和可视化 ------------------------- #

def inference(args, model, dataloader, device, output_dir):
    """执行推理并生成可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备自动混合精度
    autocast = torch.cuda.amp.autocast
    
    # 保存结果的字典
    results = {}
    
    model.eval()
    with torch.no_grad(), autocast():
        for batch in tqdm(dataloader, desc="执行推理"):
            feats  = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)  # 用于评估
            arrow  = batch["arrow"].to(device, non_blocking=True)
            depth  = batch["depth"].to(device, non_blocking=True)
            normal = batch["normal"].to(device, non_blocking=True)
            meta   = batch["meta"]
            
            # 获取批次信息
            model_id = meta["id"][0]
            joint_id = meta["joint"][0]
            view = meta["view"][0]
            category = meta["category"][0]
            has_mesh = meta["has_mesh"][0]
            vertices_path = meta["vertices_path"][0] if has_mesh else None
            
            # 模型预测
            logits = model(feats, arrow, depth, normal)
            probs = torch.sigmoid(logits)
            preds = (probs > args.threshold).float()
            
            # 计算指标
            accuracy = (preds == labels).float().mean().item()
            
            # 创建模型特定的输出目录
            model_output_dir = os.path.join(output_dir, f"{model_id}_joint_{joint_id}")
            os.makedirs(model_output_dir, exist_ok=True)
            
            # 转为NumPy
            probs_np = probs[0].cpu().numpy()
            preds_np = preds[0].cpu().numpy()
            labels_np = labels[0].cpu().numpy()
            
            # 保存结果
            np.save(os.path.join(model_output_dir, f"{view}_probs.npy"), probs_np)
            np.save(os.path.join(model_output_dir, f"{view}_preds.npy"), preds_np)
            
            # 记录结果
            key = f"{model_id}_{joint_id}_{view}"
            results[key] = {
                "model_id": model_id,
                "joint_id": joint_id,
                "view": view,
                "category": category,
                "accuracy": accuracy,
                "probs": probs_np,
                "preds": preds_np,
                "has_mesh": has_mesh
            }
            
            # 可视化: 生成彩色PLY文件
            if has_mesh:
                visualize_predictions(model_id, joint_id, view, vertices_path, preds_np, model_output_dir)
                visualize_comparison(model_id, joint_id, view, vertices_path, preds_np, labels_np, model_output_dir)
    
    # 计算总体指标
    accuracies = [item["accuracy"] for item in results.values()]
    avg_accuracy = np.mean(accuracies)
    
    print(f"\n推理完成! 平均准确率: {avg_accuracy:.4f}")
    print(f"结果保存在: {output_dir}")
    
    return results

def visualize_predictions(model_id, joint_id, view, mesh_path, predictions, output_dir):
    """将预测结果可视化为彩色PLY文件"""
    try:
        # 加载网格
        mesh = trimesh.load(mesh_path)
        
        # 确保预测数量与面片数量一致
        if len(predictions) != len(mesh.faces):
            print(f"警告: 预测数量 ({len(predictions)}) 与面片数量 ({len(mesh.faces)}) 不一致!")
            # 使用最小的数量
            min_len = min(len(predictions), len(mesh.faces))
            predictions = predictions[:min_len]
        
        # 为面片着色
        # 预测=1的面片为红色，预测=0的面片为绿色
        colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
        colors[predictions > 0.5] = [255, 0, 0, 255]  # 红色 - 预测为部件
        colors[predictions <= 0.5] = [0, 255, 0, 255]  # 绿色 - 预测为非部件
        
        mesh.visual.face_colors = colors
        
        # 保存PLY文件
        output_path = os.path.join(output_dir, f"{view}_prediction.ply")
        mesh.export(output_path)
        print(f"已保存预测可视化: {output_path}")
        
    except Exception as e:
        print(f"可视化失败: {e}")

def visualize_comparison(model_id, joint_id, view, mesh_path, predictions, ground_truth, output_dir):
    """比较预测和真实标签的可视化"""
    try:
        # 加载网格
        mesh = trimesh.load(mesh_path)
        
        # 确保长度一致
        min_len = min(len(predictions), len(ground_truth), len(mesh.faces))
        predictions = predictions[:min_len]
        ground_truth = ground_truth[:min_len]
        
        # 设置颜色 - 简化方案:
        # 红色: 预测为部件
        # 绿色: 预测为非部件
        # 蓝色: 真实标签为部件
        colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
        
        # 先绘制预测结果
        colors[predictions > 0.5] = [255, 0, 0, 255]    # 红色 - 预测为部件
        colors[predictions <= 0.5] = [0, 255, 0, 255]   # 绿色 - 预测为非部件
        
        # 保存PLY文件
        output_path = os.path.join(output_dir, f"{view}_prediction.ply")
        mesh.visual.face_colors = colors
        mesh.export(output_path)
        
        # 绘制真实标签
        colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
        colors[ground_truth > 0.5] = [0, 0, 255, 255]   # 蓝色 - 真实标签为部件
        colors[ground_truth <= 0.5] = [0, 255, 0, 255]  # 绿色 - 真实标签为非部件
        
        # 保存PLY文件
        output_path = os.path.join(output_dir, f"{view}_ground_truth.ply")
        mesh.visual.face_colors = colors
        mesh.export(output_path)
        
        # 计算并保存指标
        accuracy = np.mean(predictions == ground_truth)
        tp = (predictions == 1) & (ground_truth == 1)  # 真正例
        fp = (predictions == 1) & (ground_truth == 0)  # 假正例
        fn = (predictions == 0) & (ground_truth == 1)  # 假负例
        tn = (predictions == 0) & (ground_truth == 0)  # 真负例
        precision = np.sum(tp) / (np.sum(tp) + np.sum(fp) + 1e-6)
        recall = np.sum(tp) / (np.sum(tp) + np.sum(fn) + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": np.sum(tp),
            "fp": np.sum(fp),
            "fn": np.sum(fn),
            "tn": np.sum(tn)
        }
        
        # 保存指标
        with open(os.path.join(output_dir, f"{view}_metrics.txt"), "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        
        print(f"已保存预测和真实标签可视化")
        
    except Exception as e:
        print(f"比较可视化失败: {e}")

# ------------------------- 主函数 ------------------------- #

def main():
    args = get_args()
    set_seed(42)
    torch.backends.cudnn.benchmark = True
    
    # 设置路径
    feature_dir = os.path.join(args.base_dir, "data/urdf")
    img_dir = os.path.join(args.base_dir, "data/img")
    output_dir = args.output_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model = MultiImageFieldMLP(args.feature_dim, args.img_size, args.hidden_dim).to(device)
    
    # 加载预训练权重
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在 - {args.model_path}")
        return
    
    # 加载权重并处理可能存在的"_orig_mod."前缀
    state_dict = torch.load(args.model_path, map_location=device)
    
    # 检查是否存在"_orig_mod."前缀，如果存在则移除
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("检测到编译后的模型权重，正在移除'_orig_mod.'前缀...")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_key = k[len("_orig_mod."):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    print(f"成功加载模型: {args.model_path}")
    
    if torch.__version__.startswith("2"):
        model = torch.compile(model)
        print("已启用 PyTorch 2.0+ 模型编译加速")
    
    # 收集样本
    samples = collect_samples(img_dir, feature_dir)
    if not samples:
        print("无有效样本，结束。")
        return
    
    # 划分训练和验证集（只使用验证集）
    _, val_samples = train_test_split(samples, test_size=0.2, random_state=0)
    print(f"使用 {len(val_samples)} 个验证样本进行推理")
    
    # 创建数据集和加载器
    val_dataset = PartFieldDataset(val_samples, feature_dir, args.img_size)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                          num_workers=args.num_workers, pin_memory=args.pin_mem,
                          prefetch_factor=args.prefetch, persistent_workers=True)
    
    # 检查数据
    print("\n验证数据检查:")
    sample0 = val_dataset[0]
    print(f"id={sample0['meta']['id']}, joint={sample0['meta']['joint']}, view={sample0['meta']['view']}")
    print(f"特征: {sample0['features'].shape}, 标签: {sample0['labels'].shape}")
    
    # 运行推理
    results = inference(args, model, val_loader, device, output_dir)
    
    # 总结报告
    print("\n推理完成!")

if __name__ == "__main__":
    main()
