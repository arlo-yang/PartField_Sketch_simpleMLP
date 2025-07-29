# ===========================
# file: model.py
# ===========================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# 数值稳定性工具
# ---------------------------

def _sanitize_tensor(x: torch.Tensor, clamp_val: float = 1e4) -> torch.Tensor:
    """
    将张量中的 NaN/Inf 替换为有限值并裁剪，防止数值爆炸。
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
    if clamp_val is not None:
        x = x.clamp(min=-clamp_val, max=clamp_val)
    return x


# ---------------------------
# 基础层
# ---------------------------

class RMSNorm(nn.Module):
    """RMSNorm（无 bias），数值稳定。"""
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _sanitize_tensor(x)
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (1.0 / math.sqrt(x.shape[-1]))
        x = x / (rms + self.eps)
        return self.weight * x


class MHA(nn.Module):
    """
    轻量包装 nn.MultiheadAttention，强制在 FP32 中计算注意力（避免 FP16 溢出）。
    """
    def __init__(self, d: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None):
        orig_dtype = q.dtype
        with torch.cuda.amp.autocast(enabled=False):
            out, weights = self.attn(q.float(), k.float(), v.float(), 
                                   attn_mask=attn_mask, need_weights=True)
        return out.to(orig_dtype), weights.to(orig_dtype)


# ---------------------------
# 几何感知工具
# ---------------------------

class GeometricFeatureEncoder(nn.Module):
    """几何特征编码器：提取面片的几何属性"""
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 面片自身几何特征：面积、法向量、重心
        self.face_geo_encoder = nn.Sequential(
            nn.Linear(7, 32),  # 面积(1) + 法向量(3) + 重心(3) = 7
            nn.GELU(),
            nn.LayerNorm(32)
        )
        
        # 边长特征
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, 16),  # 3条边的长度
            nn.GELU(),
            nn.LayerNorm(16)
        )
        
        # 几何特征融合
        self.geo_fusion = nn.Sequential(
            nn.Linear(32 + 16, embed_dim // 4),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 4)
        )
    
    def forward(self, coords9: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords9: (N, 9) 面片3个顶点坐标
        Returns:
            geo_features: (N, embed_dim//4) 几何特征
        """
        N = coords9.shape[0]
        vertices = coords9.view(N, 3, 3)  # (N, 3, 3)
        
        # 计算重心
        centroid = vertices.mean(dim=1)  # (N, 3)
        
        # 计算边向量和边长
        v1, v2, v3 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        edge1, edge2, edge3 = v2 - v1, v3 - v2, v1 - v3
        edge_lengths = torch.stack([
            torch.norm(edge1, dim=-1),
            torch.norm(edge2, dim=-1), 
            torch.norm(edge3, dim=-1)
        ], dim=-1)  # (N, 3)
        
        # 计算面积和法向量
        cross_product = torch.cross(edge1, edge2, dim=-1)
        area = torch.norm(cross_product, dim=-1, keepdim=True) * 0.5  # (N, 1)
        normal = cross_product / (torch.norm(cross_product, dim=-1, keepdim=True) + 1e-8)  # (N, 3)
        
        # 几何特征组合
        face_geo = torch.cat([area, normal, centroid], dim=-1)  # (N, 7)
        face_geo_feat = self.face_geo_encoder(face_geo)  # (N, 32)
        edge_feat = self.edge_encoder(edge_lengths)  # (N, 16)
        
        # 融合
        combined = torch.cat([face_geo_feat, edge_feat], dim=-1)  # (N, 48)
        geo_features = self.geo_fusion(combined)  # (N, embed_dim//4)
        
        return _sanitize_tensor(geo_features)


class ConfidenceGuidedPromptEncoder(nn.Module):
    """
    Confidence引导的Prompt编码器
    类似SAM的prompt机制，让confidence成为强引导信号
    增强对高置信度区域的关注
    """
    def __init__(self, embed_dim: int = 128, input_feature_dim: int = 96):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_feature_dim = input_feature_dim
        
        # 增强的Confidence编码器 - 提高capacity
        self.conf_encoder = nn.Sequential(
            nn.Linear(1, 64),  # 增加容量
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, embed_dim // 2),  # 增加输出维度
            nn.GELU(),
            nn.LayerNorm(embed_dim // 2)
        )
        
        # 高置信度面片提取器
        self.high_conf_proj = nn.Sequential(
            nn.Linear(input_feature_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        
        # 分层Prompt生成器 - 针对不同置信度级别
        self.high_conf_prompt = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.medium_conf_prompt = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Confidence加权融合
        self.prompt_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, features: torch.Tensor, confidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (N, input_feature_dim) 面片特征
            confidence: (N, 1) 置信度
        Returns:
            conf_features: (N, embed_dim//2) confidence编码特征 (增强后维度更大)
            prompt_signal: (embed_dim,) 全局prompt信号
        """
        # 增强的Confidence特征编码
        conf_features = self.conf_encoder(confidence)  # (N, embed_dim//2)
        
        # 分层处理不同置信度级别
        conf_scores = confidence.squeeze(-1)  # (N,)
        
        # 高置信度面片 (0.98+) - 用户提到有很多0.95+的
        high_conf_mask = conf_scores >= 0.98
        # 中等置信度面片 (0.3-0.98)
        medium_conf_mask = (conf_scores >= 0.5) & (conf_scores < 0.98)
        
        prompt_components = []
        
        # 处理高置信度区域
        if high_conf_mask.sum() > 0:
            high_conf_features = features[high_conf_mask]
            high_conf_proj = self.high_conf_proj(high_conf_features)
            high_prompt = self.high_conf_prompt(high_conf_proj.mean(dim=0))
            prompt_components.append(high_prompt)
            
            print(f"High confidence faces: {high_conf_mask.sum().item()}/{len(conf_scores)} (max conf: {conf_scores.max().item():.3f})")
        
        # 处理中等置信度区域
        if medium_conf_mask.sum() > 0:
            medium_conf_features = features[medium_conf_mask]
            medium_conf_proj = self.high_conf_proj(medium_conf_features)  # 复用projection
            medium_prompt = self.medium_conf_prompt(medium_conf_proj.mean(dim=0))
            prompt_components.append(medium_prompt)
        
        # 如果没有足够的置信度信息，使用全局特征
        if len(prompt_components) == 0:
            global_features = self.high_conf_proj(features.mean(dim=0))
            prompt_signal = self.high_conf_prompt(global_features)
        elif len(prompt_components) == 1:
            prompt_signal = prompt_components[0]
        else:
            # 融合高置信度和中等置信度的prompt
            combined = torch.cat(prompt_components, dim=-1)  # (embed_dim * 2,)
            prompt_signal = self.prompt_fusion(combined)  # (embed_dim,)
        
        return _sanitize_tensor(conf_features), _sanitize_tensor(prompt_signal)


class GeometricSpatialAttention(nn.Module):
    """
    几何感知的空间注意力机制
    考虑面片间的几何关系和空间邻接性
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 8, 
                 max_neighbors: int = 32, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_neighbors = max_neighbors
        
        # 多头注意力
        self.mha = MHA(embed_dim, num_heads, dropout)
        
        # 几何关系编码器
        self.geo_relation_encoder = nn.Sequential(
            nn.Linear(4, 32),  # 距离 + 法向量夹角 + 面积比 + 边界共享
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_heads),  # 输出每个head的bias
            nn.Tanh()
        )
        
        # 位置编码
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, embed_dim // 4),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 4)
        )
        
    def compute_geometric_relations(self, coords9: torch.Tensor) -> torch.Tensor:
        """计算面片间的几何关系矩阵"""
        N = coords9.shape[0]
        vertices = coords9.view(N, 3, 3)  # (N, 3, 3)
        
        # 计算重心和法向量
        centroids = vertices.mean(dim=1)  # (N, 3)
        v1, v2, v3 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        edge1, edge2 = v2 - v1, v3 - v1
        normals = torch.cross(edge1, edge2, dim=-1)
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
        
        # 计算面积
        areas = torch.norm(normals, dim=-1) * 0.5  # (N,)
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
        
        # 计算成对距离矩阵 - 使用分块计算避免内存爆炸
        chunk_size = min(1000, N)
        relation_matrix = torch.zeros(N, N, 4, device=coords9.device)
        
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            for j in range(0, N, chunk_size):
                end_j = min(j + chunk_size, N)
                
                # 距离
                cent_i = centroids[i:end_i].unsqueeze(1)  # (chunk_i, 1, 3)
                cent_j = centroids[j:end_j].unsqueeze(0)  # (1, chunk_j, 3)
                distances = torch.norm(cent_i - cent_j, dim=-1)  # (chunk_i, chunk_j)
                
                # 法向量夹角
                norm_i = normals[i:end_i].unsqueeze(1)  # (chunk_i, 1, 3)
                norm_j = normals[j:end_j].unsqueeze(0)  # (1, chunk_j, 3)
                cos_angles = torch.sum(norm_i * norm_j, dim=-1)  # (chunk_i, chunk_j)
                cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
                
                # 面积比
                area_i = areas[i:end_i].unsqueeze(1)  # (chunk_i, 1)
                area_j = areas[j:end_j].unsqueeze(0)  # (1, chunk_j)
                area_ratios = torch.min(area_i, area_j) / (torch.max(area_i, area_j) + 1e-8)
                
                # 简化的边界共享（基于距离）
                boundary_sharing = torch.exp(-distances / (distances.mean() + 1e-8))
                
                # 组合关系特征
                relation_matrix[i:end_i, j:end_j, 0] = distances
                relation_matrix[i:end_i, j:end_j, 1] = cos_angles
                relation_matrix[i:end_i, j:end_j, 2] = area_ratios
                relation_matrix[i:end_i, j:end_j, 3] = boundary_sharing
        
        return relation_matrix
    
    def forward(self, x: torch.Tensor, coords9: torch.Tensor, 
                prompt_signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, embed_dim) 面片特征
            coords9: (N, 9) 几何坐标
            prompt_signal: (embed_dim,) 全局prompt信号
        Returns:
            attended_x: (N, embed_dim) 注意力增强的特征
        """
        N = x.shape[0]
        
        # 如果面片数量太大，使用局部注意力
        if N > 2000:
            return self._local_attention(x, coords9, prompt_signal)
        
        # 计算几何关系
        geo_relations = self.compute_geometric_relations(coords9)  # (N, N, 4)
        
        # 编码几何关系为注意力bias
        geo_bias = self.geo_relation_encoder(geo_relations)  # (N, N, num_heads)
        
        # 添加prompt信号
        prompt_enhanced = x + prompt_signal.unsqueeze(0)  # (N, embed_dim)
        
        # 位置编码
        centroids = coords9.view(N, 3, 3).mean(dim=1)  # (N, 3)
        pos_enc = self.pos_encoder(centroids)  # (N, embed_dim//4)
        pos_enhanced = torch.cat([
            prompt_enhanced[:, :3*self.embed_dim//4], 
            pos_enc
        ], dim=-1)
        
        # 应用多头注意力
        # 创建几何感知的注意力mask
        distances = geo_relations[:, :, 0]  # (N, N)
        # 只关注最近的邻居
        if N > self.max_neighbors:
            _, topk_indices = torch.topk(distances, self.max_neighbors, dim=-1, largest=False)
            attn_mask = torch.full((N, N), float('-inf'), device=x.device)
            for i in range(N):
                attn_mask[i, topk_indices[i]] = 0.0
        else:
            attn_mask = None
        
        attended_x, attn_weights = self.mha(
            pos_enhanced.unsqueeze(0),  # (1, N, embed_dim) 
            pos_enhanced.unsqueeze(0),
            pos_enhanced.unsqueeze(0),
            attn_mask=attn_mask
        )
        
        return _sanitize_tensor(attended_x.squeeze(0))
    
    def _local_attention(self, x: torch.Tensor, coords9: torch.Tensor, 
                        prompt_signal: torch.Tensor) -> torch.Tensor:
        """处理大规模面片的局部注意力"""
        # 简化版本：只使用prompt增强 + 位置编码
        N = x.shape[0]
        centroids = coords9.view(N, 3, 3).mean(dim=1)  # (N, 3)
        pos_enc = self.pos_encoder(centroids)  # (N, embed_dim//4)
        
        prompt_enhanced = x + prompt_signal.unsqueeze(0)
        pos_enhanced = torch.cat([
            prompt_enhanced[:, :3*self.embed_dim//4], 
            pos_enc
        ], dim=-1)
        
        return _sanitize_tensor(pos_enhanced)


class HierarchicalGeometricAttentionMLP(nn.Module):
    """
    分层几何感知注意力MLP
    结合SAM的思想，实现从局部到全局的层次化理解
    """
    def __init__(self,
                 input_dim: int = 458,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 num_classes: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # 只使用 448+1+9 => 458 维（保留原始坐标信息）
        self.in_use_dim = 458
        
        # 特征分解和编码
        self.partfield_encoder = nn.Sequential(
            nn.Linear(448, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            RMSNorm(embed_dim // 2),
        )
        
        # 几何特征编码器
        self.geo_encoder = GeometricFeatureEncoder(embed_dim)
        
        # Confidence引导的Prompt编码器
        # input_feature_dim = embed_dim//2 + embed_dim//4 = 96
        self.prompt_encoder = ConfidenceGuidedPromptEncoder(embed_dim, input_feature_dim=embed_dim//2 + embed_dim//4)
        
        # 特征融合层 - 更新维度适配增强的confidence特征
        # pf_encoded (embed_dim//2) + geo_features (embed_dim//4) + conf_features (embed_dim//2) = embed_dim * 5/4
        fusion_input_dim = embed_dim // 2 + embed_dim // 4 + embed_dim // 2  # = embed_dim * 5/4
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, embed_dim),
            nn.GELU(),
            RMSNorm(embed_dim),
        )
        
        # 多层几何感知注意力
        self.attention_layers = nn.ModuleList([
            GeometricSpatialAttention(embed_dim, num_heads, max_neighbors=32, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            RMSNorm(embed_dim) for _ in range(num_layers)
        ])
        
        # 改进的分类头
        self.classifier = nn.Sequential(
            RMSNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, num_classes)
        )
        
        self._print_model_info()
    
    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size_mb = total_params * 4 / 1024 / 1024
        print(f"\n{'='*60}")
        print("HierarchicalGeometricAttentionMLP")
        print(f"Total params: {total_params:,} | Trainable: {trainable_params:,} | ~{size_mb:.2f} MB")
        print(f"Embed: {self.embed_dim} | Layers: {self.num_layers}")
        print(f"{'='*60}\n")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N, 458) [partfield(448), confidence(1), coords(9)]
        Returns:
            logits: (N, 2)
        """
        features = _sanitize_tensor(features, clamp_val=1e4)
        assert features.dim() == 2 and features.size(-1) == 458, "输入应为 (N, 458)"
        
        # 特征分解
        partfield = features[:, :448]               # (N, 448)
        confidence = features[:, 448:449]           # (N, 1)
        coords9 = features[:, 449:458]              # (N, 9)
        
        # 编码不同类型的特征
        pf_encoded = self.partfield_encoder(partfield)      # (N, embed_dim//2)
        geo_features = self.geo_encoder(coords9)            # (N, embed_dim//4)
        conf_features, prompt_signal = self.prompt_encoder(
            torch.cat([pf_encoded, geo_features], dim=-1), confidence
        )  # (N, embed_dim//4), (embed_dim,)
        
        # 特征融合
        fused_features = torch.cat([pf_encoded, geo_features, conf_features], dim=-1)
        x = self.feature_fusion(fused_features)  # (N, embed_dim)
        
        # 层次化几何感知注意力
        for i in range(self.num_layers):
            residual = x
            attended = self.attention_layers[i](x, coords9, prompt_signal)
            x = self.layer_norms[i](residual + attended)
        
        # 分类
        logits = self.classifier(x)  # (N, 2)
        return _sanitize_tensor(logits)


# ---------------------------
# 保持原有PMA实现以兼容性
# ---------------------------

class PMA(nn.Module):
    """
    Pooling by Multihead Attention（Set Transformer 思想）：
    用固定数量的可学习 "种子向量" 作为 query，对所有面片特征做一次注意力，
    产生少量全局上下文向量（eg. 4/8 个），再进行平均/拼接用于后续 MLP。
    复杂度 O(N * S)，S<<N，内存友好。
    """
    def __init__(self, dim: int, num_seeds: int = 18, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(num_seeds, dim) / math.sqrt(dim))
        self.norm_q = RMSNorm(dim)
        self.norm_kv = RMSNorm(dim)
        self.mha = MHA(dim, num_heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C) 面片特征
        Returns:
            ctx: (C,) 全局上下文（对种子输出平均）
        """
        x = _sanitize_tensor(x)
        N, C = x.shape
        q = self.seeds.unsqueeze(0)                        # (1, S, C)
        k = x.unsqueeze(0)                                 # (1, N, C)
        v = x.unsqueeze(0)                                 # (1, N, C)
        y, _ = self.mha(self.norm_q(q), self.norm_kv(k), self.norm_kv(v))  # (1, S, C)
        ctx = y.mean(dim=1).squeeze(0)                     # (C,)
        return _sanitize_tensor(ctx)


class SimpleAttentionMLP(nn.Module):
    """
    极简 Attention-MLP：
    - 输入：partfield(448) + confidence(1) + centroid(3)
    - 先线性嵌入到 dim
    - 用 PMA 从全体面片生成全局上下文（S 个种子均值）
    - 将全局上下文拼接回每个面片特征，送入 MLP 分类头
    - 无 N×N 自注意力，内存稳定，速度快
    """
    def __init__(self,
                 input_dim: int = 458,      # 原始输入（含9维坐标），内部只用 448+1+3
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_seeds: int = 8,
                 dropout: float = 0.1,
                 num_classes: int = 2):
        super().__init__()
        self.embed_dim = embed_dim

        # 只使用 448+1+3 => 452 维（partfield + confidence + centroid）
        self.in_use_dim = 452

        self.pre_embed = nn.Sequential(
            nn.Linear(self.in_use_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            RMSNorm(embed_dim),
        )

        self.pma = PMA(dim=embed_dim, num_seeds=num_seeds, num_heads=num_heads, dropout=dropout)

        # 分类头：拼接 [x, global_ctx] => 2*embed_dim
        self.head = nn.Sequential(
            RMSNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

        self._print_model_info()

    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size_mb = total_params * 4 / 1024 / 1024
        print(f"\n{'='*60}")
        print("SimpleAttentionMLP")
        print(f"Total params: {total_params:,} | Trainable: {trainable_params:,} | ~{size_mb:.2f} MB")
        print(f"Embed: {self.embed_dim}")
        print(f"{'='*60}\n")

    @staticmethod
    def _extract_used_features(features: torch.Tensor) -> torch.Tensor:
        """
        从 (N, 458) 中提取我们使用的 452 维：448 + 1 + centroid(3)
        centroid 由 9 维顶点坐标计算得到，避免改 loader。
        """
        features = _sanitize_tensor(features, clamp_val=1e4)
        assert features.dim() == 2 and features.size(-1) == 458, "输入应为 (N, 458)"

        partfield = features[:, :448]               # (N, 448)
        confidence = features[:, 448:449]           # (N, 1)
        coords9 = features[:, 449:458]              # (N, 9)
        v = coords9.view(-1, 3, 3)
        centroid = v.mean(dim=1)                    # (N, 3)
        used = torch.cat([partfield, confidence, centroid], dim=-1)  # (N, 452)
        return _sanitize_tensor(used)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N, 458) [partfield(448), confidence(1), coords(9)]
        Returns:
            logits: (N, 2)
        """
        used = self._extract_used_features(features)           # (N, 452)
        x = self.pre_embed(used)                               # (N, C)

        # 全局上下文（对全部面片做一次 PMA）
        global_ctx = self.pma(x)                               # (C,)
        global_ctx_expanded = global_ctx.unsqueeze(0).expand(x.size(0), -1)  # (N, C)

        # 拼接并分类
        fused = torch.cat([x, global_ctx_expanded], dim=-1)    # (N, 2C)
        logits = self.head(fused)                              # (N, 2)
        return _sanitize_tensor(logits)


# ---------------------------
# 损失函数（可选：带置信度权重）
# ---------------------------

class FocalLoss(nn.Module):
    """标准 Focal Loss（多类）。"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = _sanitize_tensor(logits)
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ConfidenceAwareLoss(nn.Module):
    """
    简化版：Focal + 置信度加权（可选）。
    """
    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 1.5,
                 conf_weight: float = 0.1,
                 reduction: str = 'mean'):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction='none')
        self.conf_weight = conf_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        logits = _sanitize_tensor(logits)
        base = self.focal(logits, targets)  # (N,)
        # 置信度加权：1 + w * c，c∈[0,1]
        c = torch.clamp(torch.nan_to_num(confidence.squeeze(-1), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        weights = 1.0 + self.conf_weight * c
        loss = base * weights
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def create_model(config: Optional[dict] = None) -> nn.Module:
    """
    工厂函数 - 现在默认使用改进的几何感知模型
    """
    if config is None:
        config = {}
    
    # 如果明确要求使用简单模型
    if config.get('use_simple_model', False):
        model = SimpleAttentionMLP(
            input_dim=config.get('input_dim', 458),
            embed_dim=config.get('embed_dim', 128),
            num_heads=config.get('num_heads', 4),
            num_seeds=config.get('num_seeds', 8),
            dropout=config.get('dropout', 0.1),
            num_classes=config.get('num_classes', 2),
        )
    else:
        # 默认使用新的几何感知模型
        model = HierarchicalGeometricAttentionMLP(
            input_dim=config.get('input_dim', 458),
            embed_dim=config.get('embed_dim', 128),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.1),
            num_classes=config.get('num_classes', 2),
        )
    return model


def _quick_selftest():
    print(">>> quick self-test of HierarchicalGeometricAttentionMLP")
    torch.manual_seed(0)
    N = 1000  # 减少测试规模
    x = torch.randn(N, 458) * 5
    model = create_model({
        'embed_dim': 128,
        'num_heads': 8,
        'num_layers': 3,
        'dropout': 0.1
    })
    with torch.no_grad():
        y = model(x)
    print("input:", x.shape, "output:", y.shape)
    assert y.shape == (N, 2)
    print("OK")


if __name__ == "__main__":
    _quick_selftest()
