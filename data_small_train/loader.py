#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention-MLP 数据加载器

功能：
1. 加载训练数据 (458维特征 + 0/1标签)
2. ID级别数据集划分 (8:1:1)
3. 批处理支持 (batch_size=1)
4. 统计信息收集

基于2_loader.md规范实现
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict, Counter
import random
import yaml
import argparse
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PartSegmentationDataset(Dataset):
    """
    3D部分分割数据集
    
    输入特征: 458维 (PartField 448维 + 置信度 1维 + 坐标 9维)
    输出标签: 0/1 (固定部分/可动部分)
    """
    
    def __init__(self, data_dir, split='train', config=None, transform=None):
        """
        Args:
            data_dir: 数据目录路径
            split: 'train', 'val', 'test'
            config: 配置字典
            transform: 数据变换函数 (可选)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config or {}
        self.transform = transform
        
        # 加载所有样本并划分数据集
        self.all_samples = self._discover_samples()
        self.train_samples, self.val_samples, self.test_samples = self._split_data()
        
        # 根据split选择对应的样本
        if split == 'train':
            self.samples = self.train_samples
        elif split == 'val':
            self.samples = self.val_samples
        elif split == 'test':
            self.samples = self.test_samples
        else:
            raise ValueError(f"不支持的split: {split}")
        
        logger.info(f"加载 {split} 数据集: {len(self.samples)} 个样本")
        
        # 收集统计信息
        if split == 'train':  # 只在训练集上收集统计信息
            self.stats = self._collect_statistics()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        返回单个样本
        
        Returns:
            features: (num_faces, 458) torch.FloatTensor
            labels: (num_faces,) torch.LongTensor
            sample_info: dict 包含样本元信息
        """
        sample_path = self.samples[idx]
        
        try:
            # 加载样本数据
            features, labels, sample_info = self._load_sample(sample_path)
            
            # 应用变换 (如果有)
            if self.transform:
                features = self.transform(features)
            
            # 转换为torch tensor
            features = torch.FloatTensor(features)
            labels = torch.LongTensor(labels)
            
            return features, labels, sample_info
            
        except Exception as e:
            logger.error(f"加载样本失败: {sample_path}, 错误: {e}")
            raise
    
    def _discover_samples(self):
        """发现所有可用的训练样本"""
        samples = []
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        for item in self.data_dir.iterdir():
            if item.is_dir():
                # 检查是否是有效的样本目录
                parsed = self._parse_filename(item.name)
                if parsed and parsed['render_type'] == 'segmentation':
                    samples.append(item)
        
        logger.info(f"发现 {len(samples)} 个样本")
        return sorted(samples)
    
    def _parse_filename(self, filename):
        """解析文件名 (来自1_NOTE.md)"""
        base_pattern = r'^([a-zA-Z]+)_(\d+)_([a-zA-Z\-]+)'
        base_match = re.match(base_pattern, filename)
        
        if not base_match:
            return None
        
        category = base_match.group(1)
        obj_id = base_match.group(2)
        render_type = base_match.group(3)
        
        remaining = filename[len(base_match.group(0)):]
        
        # 检查joint信息
        joint_pattern = r'_joint_(\d+)'
        joint_match = re.search(joint_pattern, remaining)
        joint_id = joint_match.group(1) if joint_match else None
        
        # 提取视角信息
        view = None
        if joint_match:
            view_part = remaining[:joint_match.start()]
            if view_part.startswith('_') and len(view_part) > 1:
                view = view_part[1:]
        elif remaining.startswith('_') and len(remaining) > 1:
            view = remaining[1:].split('.')[0]
        
        return {
            'category': category,
            'id': obj_id,
            'render_type': render_type,
            'joint': joint_id,
            'view': view
        }
    
    def _load_sample(self, sample_path):
        """
        加载单个样本的所有文件
        
        Returns:
            features: (num_faces, 458) numpy数组
            labels: (num_faces,) numpy数组
            sample_info: dict 样本信息
        """
        sample_name = sample_path.name
        parsed_info = self._parse_filename(sample_name)
        obj_id = parsed_info['id']
        joint_id = parsed_info['joint']
        
        # 定义文件路径
        files = {
            'features': sample_path / f"{obj_id}.npy",
            'confidence': sample_path / "pred_face_confidence.txt",
            'coordinates': sample_path / "face_vertex_mapping.txt",
            'labels': sample_path / f"labels_{joint_id}.txt"
        }
        
        # 加载PartField特征 (448维)
        partfield_features = self._load_features(files['features'])
        
        # 加载置信度 (1维)
        confidence = self._load_confidence(files['confidence'])
        
        # 加载面片坐标 (9维)
        coordinates = self._load_face_coordinates(files['coordinates'])
        
        # 加载标签
        labels = self._load_labels(files['labels'])
        
        # 验证数据维度一致性
        num_faces = partfield_features.shape[0]
        assert confidence.shape[0] == num_faces, f"置信度数量不匹配: {confidence.shape[0]} vs {num_faces}"
        assert coordinates.shape[0] == num_faces, f"坐标数量不匹配: {coordinates.shape[0]} vs {num_faces}"
        assert labels.shape[0] == num_faces, f"标签数量不匹配: {labels.shape[0]} vs {num_faces}"
        
        # 组合特征 (458维)
        features = self._combine_features(partfield_features, confidence, coordinates)
        
        # 样本信息
        sample_info = {
            'sample_name': sample_name,
            'category': parsed_info['category'],
            'obj_id': obj_id,
            'joint_id': joint_id,
            'view': parsed_info['view'],
            'num_faces': num_faces
        }
        
        return features, labels, sample_info
    
    def _load_features(self, npy_path):
        """加载PartField特征 (448维)"""
        features = np.load(npy_path)
        assert features.shape[1] == 448, f"PartField特征维度错误: {features.shape[1]}"
        return features.astype(np.float32)
    
    def _load_confidence(self, confidence_path):
        """加载面片置信度"""
        confidences = {}
        
        with open(confidence_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) == 2:
                        face_id = int(parts[0])
                        confidence = float(parts[1])
                        confidences[face_id] = confidence
        
        # 按face_id排序并转换为数组
        max_face_id = max(confidences.keys())
        confidence_array = np.zeros(max_face_id + 1, dtype=np.float32)
        
        for face_id, conf in confidences.items():
            confidence_array[face_id] = conf
        
        return confidence_array
    
    def _load_face_coordinates(self, mapping_path):
        """加载面片坐标 (9维)"""
        coordinates = {}
        
        with open(mapping_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) == 10:  # face_id + 9个坐标
                        face_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        coordinates[face_id] = coords
        
        # 按face_id排序并转换为数组
        max_face_id = max(coordinates.keys())
        coord_array = np.zeros((max_face_id + 1, 9), dtype=np.float32)
        
        for face_id, coords in coordinates.items():
            coord_array[face_id] = coords
        
        return coord_array
    
    def _load_labels(self, labels_path):
        """加载Ground Truth标签"""
        labels = []
        
        with open(labels_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    label = int(line)
                    labels.append(label)
        
        return np.array(labels, dtype=np.int64)
    
    def _combine_features(self, partfield_features, confidence, coordinates):
        """
        组合所有特征为458维向量
        
        Args:
            partfield_features: (num_faces, 448)
            confidence: (num_faces,)
            coordinates: (num_faces, 9)
            
        Returns:
            combined: (num_faces, 458)
        """
        # 扩展置信度维度
        confidence_expanded = confidence.reshape(-1, 1)
        
        # 直接拼接: [PartField(448) + confidence(1) + coordinates(9)]
        combined = np.concatenate([
            partfield_features,
            confidence_expanded,
            coordinates
        ], axis=1)
        
        assert combined.shape[1] == 458, f"组合特征维度错误: {combined.shape[1]}"
        return combined
    
    def _split_data(self):
        """
        按ID级别划分数据集 (8:1:1)
        确保同一物体ID不会同时出现在train和test中
        """
        # 获取配置
        train_ratio = self.config.get('train_ratio', 0.8)
        val_ratio = self.config.get('val_ratio', 0.1)
        test_ratio = self.config.get('test_ratio', 0.1)
        seed = self.config.get('seed', 42)
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 按物体ID分组
        id_to_samples = defaultdict(list)
        for sample_path in self.all_samples:
            parsed = self._parse_filename(sample_path.name)
            if parsed:
                obj_id = parsed['id']
                id_to_samples[obj_id].append(sample_path)
        
        # 获取所有唯一ID
        all_ids = list(id_to_samples.keys())
        random.shuffle(all_ids)
        
        # 按类别统计，尽量保持平衡
        category_ids = defaultdict(list)
        for obj_id in all_ids:
            sample_path = id_to_samples[obj_id][0]  # 取第一个样本获取类别
            parsed = self._parse_filename(sample_path.name)
            category = parsed['category']
            category_ids[category].append(obj_id)
        
        # 按类别分配ID
        train_ids, val_ids, test_ids = [], [], []
        
        for category, ids in category_ids.items():
            random.shuffle(ids)
            n_total = len(ids)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_ids.extend(ids[:n_train])
            val_ids.extend(ids[n_train:n_train + n_val])
            test_ids.extend(ids[n_train + n_val:])
        
        # 根据ID分配样本
        train_samples = []
        val_samples = []
        test_samples = []
        
        for obj_id in train_ids:
            train_samples.extend(id_to_samples[obj_id])
        
        for obj_id in val_ids:
            val_samples.extend(id_to_samples[obj_id])
        
        for obj_id in test_ids:
            test_samples.extend(id_to_samples[obj_id])
        
        logger.info(f"数据集划分完成:")
        logger.info(f"  训练集: {len(train_samples)} 样本 ({len(train_ids)} 个ID)")
        logger.info(f"  验证集: {len(val_samples)} 样本 ({len(val_ids)} 个ID)")
        logger.info(f"  测试集: {len(test_samples)} 样本 ({len(test_ids)} 个ID)")
        
        return train_samples, val_samples, test_samples
    
    def _collect_statistics(self):
        """收集数据集统计信息"""
        logger.info("收集数据集统计信息...")
        
        stats = {
            'num_samples': len(self.samples),
            'categories': Counter(),
            'face_counts': [],
            'label_distribution': Counter(),
            'confidence_stats': {'min': [], 'max': [], 'mean': []},
            'coordinate_stats': {'min': [], 'max': [], 'mean': []}
        }
        
        for sample_path in tqdm(self.samples[:10], desc="统计样本"):  # 只统计前10个样本避免太慢
            try:
                sample_name = sample_path.name
                parsed = self._parse_filename(sample_name)
                
                # 类别统计
                stats['categories'][parsed['category']] += 1
                
                # 加载数据进行统计
                features, labels, sample_info = self._load_sample(sample_path)
                
                # 面片数量
                stats['face_counts'].append(sample_info['num_faces'])
                
                # 标签分布
                unique, counts = np.unique(labels, return_counts=True)
                for label, count in zip(unique, counts):
                    stats['label_distribution'][int(label)] += int(count)
                
                # 置信度统计 (第449维)
                confidence = features[:, 448]
                stats['confidence_stats']['min'].append(confidence.min())
                stats['confidence_stats']['max'].append(confidence.max())
                stats['confidence_stats']['mean'].append(confidence.mean())
                
                # 坐标统计 (第450-458维)
                coordinates = features[:, 449:458]
                stats['coordinate_stats']['min'].append(coordinates.min())
                stats['coordinate_stats']['max'].append(coordinates.max())
                stats['coordinate_stats']['mean'].append(coordinates.mean())
                
            except Exception as e:
                logger.warning(f"统计样本 {sample_path} 时出错: {e}")
        
        # 计算汇总统计
        if stats['face_counts']:
            stats['face_count_range'] = (min(stats['face_counts']), max(stats['face_counts']))
            stats['avg_face_count'] = np.mean(stats['face_counts'])
        
        if stats['confidence_stats']['min']:
            stats['confidence_range'] = (
                min(stats['confidence_stats']['min']),
                max(stats['confidence_stats']['max'])
            )
            stats['avg_confidence'] = np.mean(stats['confidence_stats']['mean'])
        
        logger.info("统计信息收集完成")
        return stats
    
    def get_statistics(self):
        """获取数据集统计信息"""
        if hasattr(self, 'stats'):
            return self.stats
        else:
            return None
    
    def print_statistics(self):
        """打印数据集统计信息"""
        stats = self.get_statistics()
        if not stats:
            logger.info("没有可用的统计信息")
            return
        
        print("\n" + "="*50)
        print("数据集统计信息")
        print("="*50)
        
        print(f"样本总数: {stats['num_samples']}")
        
        print(f"\n类别分布:")
        for category, count in stats['categories'].items():
            print(f"  {category}: {count}")
        
        if 'face_count_range' in stats:
            print(f"\n面片数量范围: {stats['face_count_range']}")
            print(f"平均面片数量: {stats['avg_face_count']:.1f}")
        
        print(f"\n标签分布:")
        total_faces = sum(stats['label_distribution'].values())
        for label, count in stats['label_distribution'].items():
            ratio = count / total_faces * 100
            label_name = "固定部分" if label == 0 else "可动部分"
            print(f"  {label} ({label_name}): {count} ({ratio:.1f}%)")
        
        if 'confidence_range' in stats:
            print(f"\n置信度范围: {stats['confidence_range']}")
            print(f"平均置信度: {stats['avg_confidence']:.3f}")


def collate_fn(batch):
    """
    自定义collate函数 (batch_size=1)
    """
    assert len(batch) == 1, "当前设计假设batch_size=1"
    return batch[0]


def get_dataloader(dataset, config=None):
    """
    创建DataLoader
    
    Args:
        dataset: PartSegmentationDataset实例
        config: 配置字典
        
    Returns:
        DataLoader实例
    """
    config = config or {}
    
    return DataLoader(
        dataset,
        batch_size=config.get('batch_size', 1),
        shuffle=config.get('shuffle', True),
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        collate_fn=collate_fn
    )


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """测试数据加载器"""
    parser = argparse.ArgumentParser(description="测试数据加载器")
    parser.add_argument("--data_dir", type=str, 
                       default="/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small_train",
                       help="数据目录路径")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--split", type=str, default="train", 
                       choices=["train", "val", "test"], help="数据集划分")
    parser.add_argument("--test_samples", type=int, default=3, help="测试样本数量")
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    
    # 创建数据集
    print(f"创建数据集: {args.data_dir}")
    dataset = PartSegmentationDataset(
        data_dir=args.data_dir,
        split=args.split,
        config=config.get('data', {})
    )
    
    # 打印统计信息
    if args.split == 'train':
        dataset.print_statistics()
    
    # 创建数据加载器
    dataloader = get_dataloader(dataset, config.get('dataloader', {}))
    
    # 测试数据加载
    print(f"\n测试数据加载 (前{args.test_samples}个样本):")
    print("-" * 50)
    
    for i, (features, labels, sample_info) in enumerate(dataloader):
        if i >= args.test_samples:
            break
        
        print(f"\n样本 {i+1}:")
        print(f"  样本名: {sample_info['sample_name']}")
        print(f"  类别: {sample_info['category']}")
        print(f"  ID: {sample_info['obj_id']}")
        print(f"  关节: {sample_info['joint_id']}")
        print(f"  面片数: {sample_info['num_faces']}")
        print(f"  特征形状: {features.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  标签分布: 0={torch.sum(labels==0).item()}, 1={torch.sum(labels==1).item()}")
        print(f"  特征范围: [{features.min():.3f}, {features.max():.3f}]")
        
        # 验证特征维度
        assert features.shape[1] == 458, f"特征维度错误: {features.shape[1]}"
        assert features.shape[0] == labels.shape[0], f"样本数量不匹配"
        
        print("  ✓ 数据验证通过")
    
    print(f"\n数据加载器测试完成！")


if __name__ == "__main__":
    main() 