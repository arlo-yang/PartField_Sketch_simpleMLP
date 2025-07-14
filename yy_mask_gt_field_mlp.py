# """
# 任务：
# 我们训练一个encoder-decoder的模型，
# 输入的是一个field，比如/hy-tmp/PartField_Sketch_simpleMLP/data/urdf/11622/feature/11622.npy (命名规则：/hy-tmp/PartField_Sketch_simpleMLP/data/urdf/{id}/feature/{id}.npy)。urdf里面有很多这样的id
# 然后我们还输入3种图片：
# 1. 比如/hy-tmp/PartField_Sketch_simpleMLP/data/img/Dishwasher_11622_arrow-sketch_bottom_center_joint_0.png。
# 2. /hy-tmp/PartField_Sketch_simpleMLP/data/img/Dishwasher_11622_depth_bottom_center.png
# 3. /hy-tmp/PartField_Sketch_simpleMLP/data/img/Dishwasher_11622_normal_bottom_center.png
# 512*512大小

# 命名规则：/hy-tmp/PartField_Sketch_simpleMLP/data/img/{类别}_{id}_{render mode}_{view}.png 如果是arrow-sketch的话还有_joint_{joint_id}。



# 我们用id去match，找到对应的图片和npy文件。一个npy对应很多个不同view的png。

# 然后我们去找label来监督，在/hy-tmp/PartField_Sketch_simpleMLP/data/urdf/{id}/yy_visualization/labels_{joint_id}.npy
# 对于有多个joint_id的，比如/hy-tmp/PartField_Sketch_simpleMLP/data/urdf/28164/yy_visualization/labels_0.npy，/hy-tmp/PartField_Sketch_simpleMLP/data/urdf/28164/yy_visualization/labels_1.npy，
# 我们是按照joint_id去匹配对应的png，然后用id配上npy。

# 每一次给encoder-decoder输入的是同一个id的npy，三种render mode并且是同一个view下的png，和png同样joint_id的label。

# 因为我们有很多的view，所以每次都是这样搭配。注意，不是一次性输入多个view，而是每次输入一个view下的三种图，npy，然后和arrow-sketch mode文件名一样的joint_id的label用来监督。

# 如果可以，加载了数据后打印一个数据看看。我看看名字，看看你加载对了没。

# 然后训练一个encoder-decoder的模型，输入是field，输出的是对label的预测。label其实就是每一个面片写上0或者1。


# 我会给你一个参考的脚本yy_mlp.py 他当时是接受一个field生成预测。现在我们只是在这个基础上做了修改，
# 让它接受指定图片，field，然后预测label。

# 你能理解吗？写一下你的理解和代码。

# """

# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import glob
# import re
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

# # 设置随机种子以确保可重复性
# torch.manual_seed(42)
# np.random.seed(42)

# # 定义常量
# BASE_DIR = "/hy-tmp/PartField_Sketch_simpleMLP"
# FEATURE_DIR = os.path.join(BASE_DIR, "data/urdf")
# IMAGE_DIR = os.path.join(BASE_DIR, "data/img")
# BATCH_SIZE = 1  # 改为1
# EPOCHS = 100
# LEARNING_RATE = 0.0001
# FEATURE_DIM = 448  # 面片特征维度
# IMAGE_SIZE = 512   # 图像大小

# # 数据集类，负责加载和预处理数据
# class PartFieldMultiImageDataset(Dataset):
#     def __init__(self, data_list, transform=None):
#         """
#         初始化数据集
        
#         参数:
#             data_list: 包含(id, category, joint_id, view, arrow_sketch_path, depth_path, normal_path)元组的列表
#             transform: 图像转换函数
#         """
#         self.data_list = data_list
#         self.transform = transform
    
#     def __len__(self):
#         return len(self.data_list)
    
#     def __getitem__(self, idx):
#         model_id, category, joint_id, view, arrow_sketch_path, depth_path, normal_path = self.data_list[idx]
        
#         # 加载特征场
#         feature_path = os.path.join(FEATURE_DIR, model_id, "feature", f"{model_id}.npy")
#         features = np.load(feature_path)
        
#         # 加载三种图像
#         arrow_sketch_img = Image.open(arrow_sketch_path).convert('RGB')  # 保持为3通道RGB图
#         depth_img = Image.open(depth_path).convert('L')  # 转为1通道灰度图
#         normal_img = Image.open(normal_path).convert('RGB')  # 保持为3通道RGB图
        
#         if self.transform:
#             arrow_sketch_img = transforms.ToTensor()(arrow_sketch_img)  # 3通道图像
#             arrow_sketch_img = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(arrow_sketch_img)
            
#             depth_img = self.transform(depth_img)  # 1通道图像
            
#             normal_img = transforms.ToTensor()(normal_img)  # 3通道图像
#             normal_img = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(normal_img)
        
#         # 加载标签
#         label_path = os.path.join(FEATURE_DIR, model_id, "yy_visualization", f"labels_{joint_id}.npy")
#         labels = np.load(label_path)
        
#         # 确保标签与特征的数量匹配
#         if len(labels) != len(features):
#             print(f"警告: 标签数量 ({len(labels)}) 与特征数量 ({len(features)}) 不匹配! 模型ID: {model_id}, Joint ID: {joint_id}")
#             # 如果不匹配，进行裁剪或填充
#             min_len = min(len(labels), len(features))
#             labels = labels[:min_len]
#             features = features[:min_len]
        
#         # 转换为PyTorch张量
#         features = torch.FloatTensor(features)
#         labels = torch.FloatTensor(labels)
        
#         return {
#             'features': features,
#             'arrow_sketch': arrow_sketch_img,
#             'depth': depth_img,
#             'normal': normal_img,
#             'labels': labels,
#             'model_id': model_id,
#             'category': category,
#             'joint_id': joint_id,
#             'view': view,
#             'arrow_sketch_path': arrow_sketch_path,
#             'depth_path': depth_path,
#             'normal_path': normal_path
#         }

# # 自定义MLP模型，用于处理特征场和三种图像
# class MultiImageFieldMLP(nn.Module):
#     def __init__(self, feature_dim, hidden_dim=256):
#         super(MultiImageFieldMLP, self).__init__()
        
#         # 特征场编码器
#         self.feature_encoder = nn.Sequential(
#             nn.Linear(feature_dim, hidden_dim),
#             nn.LeakyReLU(0.2),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LeakyReLU(0.2)
#         )
        
#         # arrow-sketch图像编码器 (3通道输入)
#         self.arrow_sketch_encoder = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 256x256
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 128x128
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x64
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 16x16
#             nn.LeakyReLU(0.2),
#             nn.Flatten(),
#             nn.Linear(256 * 16 * 16, hidden_dim),
#             nn.LeakyReLU(0.2)
#         )
        
#         # depth图像编码器 (1通道输入)
#         self.depth_encoder = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 256x256
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 128x128
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x64
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 16x16
#             nn.LeakyReLU(0.2),
#             nn.Flatten(),
#             nn.Linear(256 * 16 * 16, hidden_dim),
#             nn.LeakyReLU(0.2)
#         )
        
#         # normal图像编码器 (3通道输入)
#         self.normal_encoder = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 256x256
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 128x128
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x64
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 16x16
#             nn.LeakyReLU(0.2),
#             nn.Flatten(),
#             nn.Linear(256 * 16 * 16, hidden_dim),
#             nn.LeakyReLU(0.2)
#         )
        
#         # 融合层
#         self.fusion = nn.Sequential(
#             nn.Linear(hidden_dim * 4, hidden_dim * 2),
#             nn.LeakyReLU(0.2),
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LeakyReLU(0.2)
#         )
        
#         # 解码器 (预测每个面片的标签)
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LeakyReLU(0.2),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()  # 输出0-1之间的值
#         )
    
#     def forward(self, features, arrow_sketch, depth, normal):
#         # 编码特征场
#         batch_size, num_faces, feat_dim = features.shape
#         features_flat = features.view(-1, feat_dim)
#         feature_encoding = self.feature_encoder(features_flat)
#         feature_encoding = feature_encoding.view(batch_size, num_faces, -1)
        
#         # 编码三种图像
#         arrow_sketch_encoding = self.arrow_sketch_encoder(arrow_sketch)
#         depth_encoding = self.depth_encoder(depth)
#         normal_encoding = self.normal_encoder(normal)
        
#         # 对每个面片应用相同的图像编码
#         arrow_sketch_encoding = arrow_sketch_encoding.unsqueeze(1).expand(-1, num_faces, -1)
#         depth_encoding = depth_encoding.unsqueeze(1).expand(-1, num_faces, -1)
#         normal_encoding = normal_encoding.unsqueeze(1).expand(-1, num_faces, -1)
        
#         # 融合所有编码
#         combined = torch.cat([
#             feature_encoding, 
#             arrow_sketch_encoding, 
#             depth_encoding, 
#             normal_encoding
#         ], dim=2)
        
#         combined_flat = combined.view(-1, combined.size(-1))
#         fused_flat = self.fusion(combined_flat)
#         fused = fused_flat.view(batch_size, num_faces, -1)
        
#         # 解码预测标签
#         fused_flat = fused.view(-1, fused.size(-1))
#         predictions_flat = self.decoder(fused_flat)
#         predictions = predictions_flat.view(batch_size, num_faces)
        
#         return predictions

# # 扫描并收集数据
# def collect_data():
#     print("扫描数据...")
#     data_list = []
    
#     # 查找所有arrow-sketch图像
#     arrow_sketch_images = glob.glob(os.path.join(IMAGE_DIR, "*arrow-sketch*joint*.png"))
    
#     for arrow_sketch_path in tqdm(arrow_sketch_images):
#         # 从文件名中提取信息
#         filename = os.path.basename(arrow_sketch_path)
#         # 解析文件名: {类别}_{id}_arrow-sketch_{view}_joint_{joint_id}.png
#         match = re.match(r'(.+)_(\d+)_arrow-sketch_(.+)_joint_(\d+)\.png', filename)
        
#         if match:
#             category, model_id, view, joint_id = match.groups()
            
#             # 检查特征文件是否存在
#             feature_path = os.path.join(FEATURE_DIR, model_id, "feature", f"{model_id}.npy")
#             if not os.path.exists(feature_path):
#                 continue
                
#             # 检查标签文件是否存在
#             label_path = os.path.join(FEATURE_DIR, model_id, "yy_visualization", f"labels_{joint_id}.npy")
#             if not os.path.exists(label_path):
#                 continue
            
#             # 构建depth和normal图像的路径
#             depth_path = os.path.join(IMAGE_DIR, f"{category}_{model_id}_depth_{view}.png")
#             normal_path = os.path.join(IMAGE_DIR, f"{category}_{model_id}_normal_{view}.png")
            
#             # 检查这两种图像是否存在
#             if not os.path.exists(depth_path) or not os.path.exists(normal_path):
#                 continue
                
#             # 添加到数据列表
#             data_list.append((model_id, category, joint_id, view, arrow_sketch_path, depth_path, normal_path))
    
#     print(f"找到 {len(data_list)} 个有效样本")
    
#     # 打印一个样本用于验证
#     if data_list:
#         sample = data_list[0]
#         print("\n数据样本示例:")
#         print(f"模型ID: {sample[0]}")
#         print(f"类别: {sample[1]}")
#         print(f"关节ID: {sample[2]}")
#         print(f"视角: {sample[3]}")
#         print(f"Arrow-sketch 路径: {sample[4]}")
#         print(f"Depth 路径: {sample[5]}")
#         print(f"Normal 路径: {sample[6]}")
        
#         # 加载特征和标签进行验证
#         feature_path = os.path.join(FEATURE_DIR, sample[0], "feature", f"{sample[0]}.npy")
#         label_path = os.path.join(FEATURE_DIR, sample[0], "yy_visualization", f"labels_{sample[2]}.npy")
        
#         features = np.load(feature_path)
#         labels = np.load(label_path)
        
#         print(f"特征形状: {features.shape}")
#         print(f"标签形状: {labels.shape}")
        
#     return data_list

# # 自定义collate函数，处理不同长度的特征和标签
# def custom_collate_fn(batch):
#     # 获取批次中最小的面片数量
#     min_faces = min([item['features'].shape[0] for item in batch])
    
#     # 裁剪所有样本到相同数量的面片
#     for item in batch:
#         item['features'] = item['features'][:min_faces]
#         item['labels'] = item['labels'][:min_faces]
    
#     # 为不同键组合批次
#     batch_dict = {}
#     for key in batch[0].keys():
#         if key in ['features', 'labels']:
#             batch_dict[key] = torch.stack([item[key] for item in batch])
#         elif key in ['arrow_sketch', 'depth', 'normal']:
#             batch_dict[key] = torch.stack([item[key] for item in batch])
#         else:
#             batch_dict[key] = [item[key] for item in batch]
            
#     return batch_dict

# # 训练函数
# def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备: {device}")
    
#     model.to(device)
    
#     best_val_loss = float('inf')
#     train_losses = []
#     val_losses = []
    
#     for epoch in range(num_epochs):
#         # 训练阶段
#         model.train()
#         train_loss = 0.0
        
#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
#         for batch in progress_bar:
#             features = batch['features'].to(device)
#             arrow_sketch = batch['arrow_sketch'].to(device)
#             depth = batch['depth'].to(device)
#             normal = batch['normal'].to(device)
#             labels = batch['labels'].to(device)
            
#             # 前向传播
#             outputs = model(features, arrow_sketch, depth, normal)
#             loss = criterion(outputs, labels)
            
#             # 反向传播和优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item() * features.size(0)
#             progress_bar.set_postfix({"loss": loss.item()})
        
#         train_loss = train_loss / len(train_loader.dataset)
#         train_losses.append(train_loss)
        
#         # 验证阶段
#         model.eval()
#         val_loss = 0.0
        
#         with torch.no_grad():
#             for batch in val_loader:
#                 features = batch['features'].to(device)
#                 arrow_sketch = batch['arrow_sketch'].to(device)
#                 depth = batch['depth'].to(device)
#                 normal = batch['normal'].to(device)
#                 labels = batch['labels'].to(device)
                
#                 outputs = model(features, arrow_sketch, depth, normal)
#                 loss = criterion(outputs, labels)
                
#                 val_loss += loss.item() * features.size(0)
        
#         val_loss = val_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)
        
#         print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
#         # 保存最佳模型
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), "best_model.pth")
#             print("保存最佳模型")
    
#     # 绘制损失曲线
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
#     plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.savefig('loss_curve.png')
#     plt.close()
    
#     return train_losses, val_losses

# def main():
#     # 收集数据
#     data_list = collect_data()
    
#     if not data_list:
#         print("没有找到有效数据，退出程序")
#         return
    
#     # 分割训练集和验证集
#     train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)
    
#     # 定义图像转换
#     transform = transforms.Compose([
#         transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#         transforms.ToTensor(),
#     ])
    
#     # 创建数据集和数据加载器
#     train_dataset = PartFieldMultiImageDataset(train_data, transform=transform)
#     val_dataset = PartFieldMultiImageDataset(val_data, transform=transform)
    
#     # 验证一个样本
#     sample_data = train_dataset[0]
#     print("\n加载的数据样本:")
#     print(f"模型ID: {sample_data['model_id']}")
#     print(f"类别: {sample_data['category']}")
#     print(f"关节ID: {sample_data['joint_id']}")
#     print(f"视角: {sample_data['view']}")
#     print(f"特征形状: {sample_data['features'].shape}")
#     print(f"Arrow-sketch 形状: {sample_data['arrow_sketch'].shape}")
#     print(f"Depth 形状: {sample_data['depth'].shape}")
#     print(f"Normal 形状: {sample_data['normal'].shape}")
#     print(f"标签形状: {sample_data['labels'].shape}")
    
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
#     # 初始化模型
#     model = MultiImageFieldMLP(feature_dim=FEATURE_DIM)
    
#     # 定义损失函数和优化器
#     criterion = nn.BCELoss()  # 二元交叉熵损失
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
#     # 训练模型
#     train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
    
#     print("训练完成!")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
