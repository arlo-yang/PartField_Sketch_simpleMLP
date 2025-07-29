#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存高效SOTA Attention-MLP模型

基于3_model.md设计规范实现
- 458维→128维大幅降维
- 分层注意力处理40,000+面片
- 四点几何位置编码
- Confidence引导机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CompactGroupedEmbedding(nn.Module):
    """紧凑分组嵌入 - 极致内存高效"""
    
    def __init__(self, partfield_dim=448, confidence_dim=1, coord_dim=9, embed_dim=64):
        super().__init__()
        
        # PartField压缩 (448→48维，极致压缩)
        self.partfield_proj = nn.Sequential(
            nn.Linear(partfield_dim, 96),   # 先降到96
            nn.GELU(),
            nn.Linear(96, 48),              # 再降到48
            nn.LayerNorm(48)
        )
        
        # Confidence扩展 (1→8维，适度增强)
        self.confidence_proj = nn.Sequential(
            nn.Linear(confidence_dim, 8),
            nn.GELU(),
            nn.LayerNorm(8)
        )
        
        # 坐标压缩 (9→8维，保持几何信息)
        self.coord_proj = nn.Sequential(
            nn.Linear(coord_dim, 8),
            nn.GELU(),
            nn.LayerNorm(8)
        )
        
        # 融合层: 48+8+8=64维
        assert 48 + 8 + 8 == embed_dim, f"维度必须匹配: {48 + 8 + 8} != {embed_dim}"
        
        # 最终融合
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
    
    def forward(self, partfield, confidence, coordinates):
        # 分别嵌入
        pf_embed = self.partfield_proj(partfield)      # (N, 48)
        conf_embed = self.confidence_proj(confidence)   # (N, 8)
        coord_embed = self.coord_proj(coordinates)      # (N, 8)
        
        # 拼接
        combined = torch.cat([pf_embed, conf_embed, coord_embed], dim=-1)  # (N, 64)
        
        # 融合
        enhanced = self.fusion(combined)
        
        return enhanced


class CompactFourPointGeometricEncoding(nn.Module):
    """极致紧凑四点几何位置编码"""
    
    def __init__(self, embed_dim=64):
        super().__init__()
        
        # 4个点的位置编码器 (每个点16维)
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 16),  # 每个点16维
            nn.GELU(),
            nn.LayerNorm(16)
        )
        
        # 几何特征融合 (4×16=64维)
        self.geometric_fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        
        # 形状特征编码器 (面积+法向量→8维)
        self.shape_encoder = nn.Sequential(
            nn.Linear(4, 8),   # 面积+法向量(3维)=4维→8维
            nn.GELU(),
            nn.LayerNorm(8)
        )
        
        # 最终融合 (64+8→64)
        self.final_fusion = nn.Sequential(
            nn.Linear(embed_dim + 8, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, face_coordinates):
        """
        Args:
            face_coordinates: (N, 9) - 3个顶点的坐标
        Returns:
            pos_encoding: (N, 64) - 极致紧凑位置编码
        """
        batch_size = face_coordinates.shape[0]
        
        # 重塑为3个顶点
        vertices = face_coordinates.view(batch_size, 3, 3)  # (N, 3, 3)
        
        # 计算重心
        centroid = vertices.mean(dim=1, keepdim=True)  # (N, 1, 3)
        
        # 四个点: 3顶点 + 重心
        four_points = torch.cat([vertices, centroid], dim=1)  # (N, 4, 3)
        
        # 每个点编码16维
        point_encodings = []
        for i in range(4):
            point_enc = self.point_encoder(four_points[:, i, :])  # (N, 16)
            point_encodings.append(point_enc)
        
        # 拼接4个点的编码
        combined_points = torch.cat(point_encodings, dim=-1)  # (N, 64)
        
        # 几何特征增强
        geo_features = self.geometric_fusion(combined_points)
        
        # 计算形状特征
        v1, v2, v3 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        
        # 面积和法向量
        edge1 = v2 - v1
        edge2 = v3 - v1
        cross_product = torch.cross(edge1, edge2, dim=-1)
        area = torch.norm(cross_product, dim=-1, keepdim=True) * 0.5
        normal = cross_product / (torch.norm(cross_product, dim=-1, keepdim=True) + 1e-8)
        
        # 形状特征
        shape_features = torch.cat([area, normal], dim=-1)  # (N, 4)
        shape_enc = self.shape_encoder(shape_features)  # (N, 8)
        
        # 最终融合
        final_encoding = torch.cat([geo_features, shape_enc], dim=-1)  # (N, 72)
        pos_encoding = self.final_fusion(final_encoding)  # (N, 64)
        
        return pos_encoding


class HierarchicalAttention(nn.Module):
    """极致内存高效分层注意力"""
    
    def __init__(self, embed_dim=64, num_heads=4, chunk_size=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size  # 更小的分块，避免内存爆炸
        
        # 局部注意力 (在chunk内)
        self.local_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # 全局注意力 (chunk间)
        self.global_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # Confidence引导
        self.confidence_gate = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Sigmoid()
        )
        
        # 归一化和前馈
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),      # 不扩展，极致节省内存
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, confidence):
        """
        Args:
            x: (N, 64) 面片特征，N可能是40,000+
            confidence: (N, 1) 置信度
        """
        N = x.shape[0]
        
        # 1. Confidence引导门控
        conf_gate = self.confidence_gate(confidence)
        x_gated = x * conf_gate
        x = self.norm1(x + x_gated)
        
        # 2. 分块局部注意力
        if N > self.chunk_size:
            # 大规模面片，分块处理
            chunks = torch.split(x, self.chunk_size, dim=0)
            local_outputs = []
            
            for chunk in chunks:
                chunk_out, _ = self.local_attention(chunk, chunk, chunk)
                local_outputs.append(chunk_out)
            
            local_out = torch.cat(local_outputs, dim=0)
        else:
            # 小规模面片，直接处理
            local_out, _ = self.local_attention(x, x, x)
        
        x = self.norm2(x + local_out)
        
        # 3. 全局上下文 (采样代表性面片，极致内存优化)
        if N > self.chunk_size:
            # 采样更少的高confidence面片作为全局代表
            conf_scores = confidence.squeeze()
            sample_size = min(self.chunk_size // 4, N)  # 进一步减少采样数量
            _, top_indices = torch.topk(conf_scores, sample_size, dim=0)
            global_representatives = x[top_indices]
            
            # 全局注意力
            global_out, _ = self.global_attention(x, global_representatives, global_representatives)
        else:
            # 小规模直接处理，但限制数量
            if N > self.chunk_size // 2:
                # 如果还是太大，进行采样
                conf_scores = confidence.squeeze()
                sample_size = self.chunk_size // 2
                _, top_indices = torch.topk(conf_scores, sample_size, dim=0)
                global_representatives = x[top_indices]
                global_out, _ = self.global_attention(x, global_representatives, global_representatives)
            else:
                global_out, _ = self.global_attention(x, x, x)
        
        x = self.norm3(x + global_out)
        
        # 4. 前馈网络
        ffn_out = self.ffn(x)
        x = x + ffn_out
        
        return x


class MemoryEfficientSOTAModel(nn.Module):
    """内存高效的SOTA模型"""
    
    def __init__(self, 
                 input_dim=458,
                 embed_dim=64,       # 极致降维
                 num_layers=3,       # 进一步减少层数
                 num_heads=4,        # 减少头数
                 chunk_size=512,     # 更小分块
                 dropout=0.1,
                 num_classes=2):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # 紧凑嵌入
        self.embedding = CompactGroupedEmbedding(
            partfield_dim=448, 
            confidence_dim=1, 
            coord_dim=9, 
            embed_dim=embed_dim
        )
        
        # 紧凑位置编码
        self.pos_encoding = CompactFourPointGeometricEncoding(embed_dim)
        
        # 分层注意力层
        self.layers = nn.ModuleList([
            HierarchicalAttention(embed_dim, num_heads, chunk_size)
            for _ in range(num_layers)
        ])
        
        # 极致轻量分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 权重初始化
        self.apply(self._init_weights)
        
        # 打印模型信息
        self._print_model_info()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n{'='*60}")
        print(f"内存高效SOTA模型信息")
        print(f"{'='*60}")
        print(f"嵌入维度: {self.embed_dim}")
        print(f"网络层数: {self.num_layers}")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        print(f"{'='*60}\n")
    
    def forward(self, features):
        """
        Args:
            features: (N, 458) 其中N可能是40,000+
        Returns:
            logits: (N, 2) 分类logits
        """
        # 分离特征
        partfield = features[:, :448]
        confidence = features[:, 448:449]
        coordinates = features[:, 449:458]
        
        # 极致紧凑嵌入 458→64
        x = self.embedding(partfield, confidence, coordinates)  # (N, 64)
        
        # 位置编码
        pos_enc = self.pos_encoding(coordinates)
        x = x + pos_enc
        
        # 分层注意力
        for layer in self.layers:
            x = layer(x, confidence)
        
        # 分类
        logits = self.classifier(x)  # (N, 2)
        
        return logits
    
    def forward_with_checkpointing(self, features):
        """使用梯度检查点的前向传播 - 节省内存"""
        from torch.utils.checkpoint import checkpoint
        
        # 分离特征
        partfield = features[:, :448]
        confidence = features[:, 448:449]
        coordinates = features[:, 449:458]
        
        # 嵌入和位置编码
        x = self.embedding(partfield, confidence, coordinates)
        pos_enc = self.pos_encoding(coordinates)
        x = x + pos_enc
        
        # 使用梯度检查点的分层注意力
        for layer in self.layers:
            x = checkpoint(layer, x, confidence, use_reentrant=False)
        
        # 分类
        logits = self.classifier(x)
        
        return logits


class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ConfidenceWeightedLoss(nn.Module):
    """Confidence引导的加权损失"""
    
    def __init__(self, base_loss='focal', conf_weight=0.3, alpha=0.25, gamma=2.0):
        super().__init__()
        
        if base_loss == 'focal':
            self.base_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='none')
        else:
            self.base_loss = nn.CrossEntropyLoss(reduction='none')
        
        self.conf_weight = conf_weight
    
    def forward(self, logits, labels, confidence):
        # 基础损失
        base_loss = self.base_loss(logits, labels)
        
        # Confidence加权 (高confidence的预测给更高权重)
        conf_weights = 1.0 + confidence.squeeze()  # [1, 2]范围
        weighted_loss = base_loss * conf_weights
        
        # Confidence一致性损失 (高confidence应该预测正确)
        probs = torch.softmax(logits, dim=-1)
        pred_confidence = probs.max(dim=-1)[0]  # 预测置信度
        
        # 一致性损失: 高输入confidence应该对应高预测confidence
        consistency_loss = F.mse_loss(pred_confidence, confidence.squeeze(), reduction='mean')
        
        # 组合损失
        total_loss = weighted_loss.mean() + self.conf_weight * consistency_loss
        
        return total_loss


def create_model(config=None):
    """创建模型的工厂函数"""
    if config is None:
        config = {
            'embed_dim': 64,
            'num_layers': 3,
            'num_heads': 4,
            'chunk_size': 512,
            'dropout': 0.1
        }
    
    model = MemoryEfficientSOTAModel(
        embed_dim=config.get('embed_dim', 64),
        num_layers=config.get('num_layers', 3),
        num_heads=config.get('num_heads', 4),
        chunk_size=config.get('chunk_size', 512),
        dropout=config.get('dropout', 0.1)
    )
    
    return model


def test_model():
    """测试模型"""
    print("测试内存高效SOTA模型...")
    
    # 创建模型
    model = create_model()
    
    # 测试不同规模的输入
    test_cases = [
        (500, "小规模面片"),
        (2000, "中规模面片"),
        (5000, "大规模面片")
    ]
    
    for num_faces, desc in test_cases:
        print(f"\n测试 {desc}: {num_faces} 个面片")
        
        # 创建测试输入
        features = torch.randn(num_faces, 458)
        
        # 前向传播
        with torch.no_grad():
            logits = model(features)
        
        print(f"  输入形状: {features.shape}")
        print(f"  输出形状: {logits.shape}")
        print(f"  内存占用: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB" if torch.cuda.is_available() else "  CPU模式")
        
        # 验证输出
        assert logits.shape == (num_faces, 2), f"输出形状错误: {logits.shape}"
        print(f"  ✓ 测试通过")
    
    print(f"\n模型测试完成！")


if __name__ == "__main__":
    test_model() 