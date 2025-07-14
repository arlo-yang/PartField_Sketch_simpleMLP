"""
基于PartField特征的3D模型可动部分分类

该脚本包含以下功能：
1. 数据处理：将base.obj和moveable{n}.obj映射到whole.obj面片，生成标签
2. 特征处理：加载PartField提取的特征向量
3. MLP模型：实现简单有效的多层感知机进行二分类
4. 训练和评估：包括数据划分、模型训练和性能评估

使用方法:
- 训练模型: python yy_mlp.py --mode train
- 推理模型: python yy_mlp.py --mode infer --model_path [PATH_TO_MODEL] --feature_dir [PATH_TO_FEATURES] --geometry_dir [PATH_TO_GEOMETRY]
- 评估模型: python yy_mlp.py --mode eval --model_path [PATH_TO_MODEL]
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import trimesh
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random
import glob
from pathlib import Path
import time

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 定义MLP模型
class PartMoveabilityMLP(nn.Module):
    """
    基于PartField特征的可动部分分类MLP模型
    """
    def __init__(self, input_dim=448, hidden_dims=[256, 128, 64], output_dim=2, dropout=0.3):
        super(PartMoveabilityMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 自定义数据集
class MoveabilityDataset(Dataset):
    """
    可动部分分类数据集
    加载PartField特征和从base.obj/moveable.obj生成的标签
    """
    def __init__(self, base_dir="PartField", ids=None, mode="train", feature_dir=None, geometry_dir=None):
        self.base_dir = base_dir
        self.feature_dir = feature_dir if feature_dir else os.path.join(base_dir, "yy_field")
        self.geometry_dir = geometry_dir if geometry_dir else os.path.join(base_dir, "yy_finetune_data")
        self.mode = mode
        
        print(f"使用特征目录: {self.feature_dir}")
        print(f"使用几何数据目录: {self.geometry_dir}")
        
        # 获取所有ID或使用提供的ID列表
        if ids is None:
            # 从特征目录获取ID列表
            self.ids = [d for d in os.listdir(self.feature_dir) 
                      if os.path.isdir(os.path.join(self.feature_dir, d))]
        else:
            self.ids = ids
            
        # 过滤掉没有对应数据的ID
        self.valid_ids = []
        for model_id in self.ids:
            field_path = os.path.join(self.feature_dir, model_id, f"{model_id}.npy")
            
            # 如果是训练或评估模式，需要检查监督数据是否存在
            if mode in ["train", "eval"]:
                whole_path = os.path.join(self.geometry_dir, model_id, "whole.obj")
                base_path = os.path.join(self.geometry_dir, model_id, "base.obj")
                
                # 检查必要的文件是否存在
                if (os.path.exists(field_path) and os.path.exists(whole_path) and 
                    os.path.exists(base_path)):
                    self.valid_ids.append(model_id)
            else:  # 推理模式
                whole_path = os.path.join(self.geometry_dir, model_id, "whole.obj")
                
                # 推理模式只需要特征和whole.obj
                if os.path.exists(field_path) and os.path.exists(whole_path):
                    self.valid_ids.append(model_id)
        
        print(f"找到 {len(self.valid_ids)} 个有效ID用于{mode}模式")
        
        # 预处理每个模型的数据
        self.processed_data = []
        self._preprocess_data()
        
    def _load_obj(self, file_path):
        """加载OBJ文件并返回trimesh对象"""
        try:
            mesh = trimesh.load(file_path, force='mesh')
            return mesh
        except Exception as e:
            print(f"加载 {file_path} 时出错: {e}")
            return None
    
    def _match_faces(self, whole_mesh, part_mesh):
        """
        将部件网格的面匹配到完整网格的面
        通过计算面的质心距离进行匹配
        """
        # 提取质心
        whole_centroids = np.mean(whole_mesh.vertices[whole_mesh.faces], axis=1)
        part_centroids = np.mean(part_mesh.vertices[part_mesh.faces], axis=1)
        
        # 初始化匹配结果
        matched_indices = []
        
        # 对部件中的每个面，找到whole中最近的面
        for part_centroid in part_centroids:
            # 计算到whole中所有面的距离
            distances = np.linalg.norm(whole_centroids - part_centroid, axis=1)
            # 找到最近的面的索引
            closest_face_idx = np.argmin(distances)
            # 如果距离小于阈值，认为匹配成功
            min_distance = distances[closest_face_idx]
            if min_distance < 0.05:  # 距离阈值，可根据数据特性调整
                matched_indices.append(closest_face_idx)
        
        return matched_indices
    
    def _preprocess_data(self):
        """预处理数据，生成特征和标签对"""
        print("开始预处理数据...")
        
        for model_id in tqdm(self.valid_ids):
            # 加载PartField特征
            feature_path = os.path.join(self.feature_dir, model_id, f"{model_id}.npy")
            features = np.load(feature_path)
            
            # 加载whole网格
            whole_path = os.path.join(self.geometry_dir, model_id, "whole.obj")
            whole_mesh = self._load_obj(whole_path)
            
            if whole_mesh is None:
                continue
            
            # 初始化标签，默认值取决于模式
            if self.mode == "infer":
                # 推理模式下，只需要特征，可以设置为默认值
                labels = np.zeros(len(whole_mesh.faces), dtype=np.int64)
            else:
                # 训练和评估模式需要通过监督信息生成标签
                # 初始化标签，默认全部为可动部分(1)
                labels = np.ones(len(whole_mesh.faces), dtype=np.int64)
                
                # 加载base.obj
                base_path = os.path.join(self.geometry_dir, model_id, "base.obj")
                base_mesh = self._load_obj(base_path)
                
                if base_mesh is None:
                    continue
                
                # 匹配base部分到whole，标记为不可动(0)
                base_matched_indices = self._match_faces(whole_mesh, base_mesh)
                labels[base_matched_indices] = 0
                
                # 查找所有moveable部分
                moveable_paths = glob.glob(os.path.join(self.geometry_dir, model_id, "moveable*.obj"))
                
                for moveable_path in moveable_paths:
                    moveable_mesh = self._load_obj(moveable_path)
                    if moveable_mesh is not None:
                        # 匹配moveable部分到whole，标记为可动(1)
                        moveable_matched_indices = self._match_faces(whole_mesh, moveable_mesh)
                        labels[moveable_matched_indices] = 1
            
            # 检查特征和标签维度是否匹配
            if len(features) != len(labels):
                print(f"警告: {model_id} 的特征维度 ({len(features)}) 与标签维度 ({len(labels)}) 不匹配，跳过")
                continue
                
            # 添加到处理好的数据
            for feat, label in zip(features, labels):
                self.processed_data.append((feat, label, model_id))
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        feature, label, model_id = self.processed_data[idx]
        return {
            'feature': torch.tensor(feature, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'model_id': model_id
        }

# 训练模型
def train_model(args):
    """训练MLP模型"""
    print("准备训练数据...")
    
    # 创建数据集
    full_dataset = MoveabilityDataset(base_dir=args.base_dir, mode="train")
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartMoveabilityMLP(
        input_dim=448, 
        hidden_dims=args.hidden_dims, 
        output_dim=2, 
        dropout=args.dropout
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"开始训练，使用设备: {device}")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 训练一个epoch
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            features = batch['feature'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        # 计算平均训练损失和准确率
        train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                features = batch['feature'].to(device)
                labels = batch['label'].to(device)
                
                # 前向传播
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # 统计
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        # 计算平均验证损失和准确率
        val_loss = val_loss / val_total
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 打印统计信息
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.output_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, model_path)
            print(f"保存最佳模型到 {model_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, f"final_model.pth")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
    }, final_model_path)
    print(f"保存最终模型到 {final_model_path}")
    
    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    plt.close()
    
    print(f"训练完成，结果保存到 {args.output_dir}")

# 推理函数
def infer_model(args):
    """使用训练好的模型进行推理"""
    # 创建数据集
    test_dataset = MoveabilityDataset(
        base_dir=args.base_dir, 
        mode="infer", 
        feature_dir=args.feature_dir,
        geometry_dir=args.geometry_dir
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartMoveabilityMLP(
        input_dim=448, 
        hidden_dims=args.hidden_dims, 
        output_dim=2, 
        dropout=0.0  # 推理时不使用dropout
    ).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 按模型ID组织预测结果
    predictions_by_model = {}
    ground_truth_by_model = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="推理"):
            features = batch['feature'].to(device)
            labels = batch['label'].to(device)
            model_ids = batch['model_id']
            
            # 前向传播
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            
            # 收集预测结果
            for i, model_id in enumerate(model_ids):
                if model_id not in predictions_by_model:
                    predictions_by_model[model_id] = []
                    ground_truth_by_model[model_id] = []
                
                predictions_by_model[model_id].append(predicted[i].item())
                ground_truth_by_model[model_id].append(labels[i].item())
    
    # 为每个模型生成可视化结果
    for model_id, predictions in predictions_by_model.items():
        # 加载原始网格
        whole_path = os.path.join(args.geometry_dir, model_id, "whole.obj")
        whole_mesh = trimesh.load(whole_path, force='mesh')
        
        # 将预测结果转换为颜色
        face_colors = np.zeros((len(predictions), 4), dtype=np.uint8)
        face_colors[np.array(predictions) == 0] = [255, 0, 0, 255]  # 红色表示不可动部分
        face_colors[np.array(predictions) == 1] = [0, 255, 0, 255]  # 绿色表示可动部分
        
        # 创建带颜色的网格
        colored_mesh = trimesh.Trimesh(
            vertices=whole_mesh.vertices,
            faces=whole_mesh.faces,
            face_colors=face_colors
        )
        
        # 导出结果
        output_path = os.path.join(args.output_dir, f"{model_id}_pred.ply")
        colored_mesh.export(output_path)
        print(f"保存预测结果到 {output_path}")
        
        # 如果存在ground truth数据，计算性能指标
        if args.eval_metrics and model_id in ground_truth_by_model:
            if len(predictions) == len(ground_truth_by_model[model_id]):
                y_pred = np.array(predictions)
                y_true = np.array(ground_truth_by_model[model_id])
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # 保存性能指标
                with open(os.path.join(args.output_dir, f"{model_id}_metrics.txt"), 'w') as f:
                    f.write(f"Accuracy: {accuracy:.4f}\n")
                    f.write(f"Precision: {precision:.4f}\n")
                    f.write(f"Recall: {recall:.4f}\n")
                    f.write(f"F1 Score: {f1:.4f}\n")
    
    print("推理完成")

# 评估函数
def evaluate_model(args):
    """评估模型性能"""
    # 创建数据集
    test_dataset = MoveabilityDataset(base_dir=args.base_dir, mode="eval")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartMoveabilityMLP(
        input_dim=448, 
        hidden_dims=args.hidden_dims, 
        output_dim=2, 
        dropout=0.0  # 评估时不使用dropout
    ).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备评估
    all_predictions = []
    all_labels = []
    model_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估"):
            features = batch['feature'].to(device)
            labels = batch['label'].to(device)
            batch_model_ids = batch['model_id']
            
            # 前向传播
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            
            # 收集结果
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            model_ids.extend(batch_model_ids)
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 计算整体性能指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # 输出性能指标
    print(f"整体评估结果:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['不可动部分', '可动部分']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # 在混淆矩阵中添加数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # 保存评估结果
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print(f"评估完成，结果保存到 {args.output_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PartField特征的可动部分分类')
    
    # 基础参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer', 'eval'],
                        help='运行模式: 训练(train), 推理(infer), 评估(eval)')
    parser.add_argument('--base_dir', type=str, default='/hy-tmp/PartField',
                        help='数据根目录路径')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录路径')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径（用于推理和评估）')
                        
    # 自定义目录结构
    parser.add_argument('--feature_dir', type=str, default=None,
                        help='特征目录路径（如/hy-tmp/PartField/yy_field_test）')
    parser.add_argument('--geometry_dir', type=str, default=None,
                        help='几何文件目录路径（如/hy-tmp/PartField/yy_finetune_data_inference）')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练周期数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128, 64],
                        help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout比例')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
                        
    # 推理参数
    parser.add_argument('--eval_metrics', action='store_true',
                        help='在推理模式下是否计算评估指标（需要ground truth数据）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed()
    
    # 根据模式执行相应的函数
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'infer':
        if args.model_path is None:
            raise ValueError("推理模式需要提供模型路径 (--model_path)")
        infer_model(args)
    elif args.mode == 'eval':
        if args.model_path is None:
            raise ValueError("评估模式需要提供模型路径 (--model_path)")
        evaluate_model(args)

if __name__ == '__main__':
    main()