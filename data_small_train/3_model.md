# Attention-MLP 模型设计规范 (内存高效SOTA版本)

## 1. 模型背景和目标

有了loader.py和1_NOTE.md，2_loader.md后，我们开始写model了

我先说一些背景：
1. 我们partfield 448维度的特征来自于一个partfield的模型，这个模型能对几何体整体有一个很好的特征提取
2. 我们的confidence是一个我们自己标识的1维特征，也就是我们想要highlight某一些地方，希望模型能关注到
3. 但是我们highlight的部分是不全的，没有办法做到全面，我们只是为了给我们的模型一个prompt，一个引导，
告诉模型"欸，这一块是可动的区域噢，虽然我们给了confidence，confidence大概率是准确的，但是不能cover所有
可动的区域，也不能保证high confidence的面片标识都是全对的，会有出错的可能"
4. 我们的面片3个顶点9个维度是为了模型能很好地理解我们几何体的结构

**输入**: partfield 448 + confidence 1 + 顶点9维度 = **458维**
**关键挑战**: 一个几何体有**40,000+面片**，458维×40,000 = **1,832万参数/样本**
**目标**: 降维到紧凑表示，同时保持表达能力

**输出**: 对每个面片的1/0二分类 (可动部分/固定部分)

## 2. 内存高效SOTA架构设计

### 2.1 整体架构流程

```
输入 (40,000+ faces, 458) 
    ↓
紧凑特征嵌入 (Compact Embedding) 458→128
    ↓ (40,000+ faces, 128)
四点几何位置编码 (4-Point Geometric Encoding) 
    ↓ (40,000+ faces, 128)
分层注意力机制 (Hierarchical Attention)
    ↓ (40,000+ faces, 128)
轻量分类头 (Lightweight Classification Head)
    ↓ (40,000+ faces, 2) → 二分类结果
```

### 2.2 核心设计：内存效率优先

#### 2.2.1 紧凑特征嵌入 (大幅降维)
**设计原则**: 458→128维，减少75%内存占用，同时保持关键信息

```python
class CompactGroupedEmbedding(nn.Module):
    """紧凑分组嵌入 - 内存高效"""
    def __init__(self, partfield_dim=448, confidence_dim=1, coord_dim=9, embed_dim=128):
        super().__init__()
        
        # PartField压缩 (448→96维，保留核心信息)
        self.partfield_proj = nn.Sequential(
            nn.Linear(partfield_dim, 192),  # 先降到192
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(192, 96),             # 再降到96
            nn.LayerNorm(96)
        )
        
        # Confidence扩展 (1→16维，适度增强)
        self.confidence_proj = nn.Sequential(
            nn.Linear(confidence_dim, 16),
            nn.GELU(),
            nn.LayerNorm(16)
        )
        
        # 坐标压缩 (9→16维，保持几何信息)
        self.coord_proj = nn.Sequential(
            nn.Linear(coord_dim, 16),
            nn.GELU(),
            nn.LayerNorm(16)
        )
        
        # 融合层: 96+16+16=128维
        assert 96 + 16 + 16 == embed_dim, "维度必须匹配"
        
        # 最终融合
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
    
    def forward(self, partfield, confidence, coordinates):
        # 分别嵌入
        pf_embed = self.partfield_proj(partfield)      # (N, 96)
        conf_embed = self.confidence_proj(confidence)   # (N, 16)
        coord_embed = self.coord_proj(coordinates)      # (N, 16)
        
        # 拼接
        combined = torch.cat([pf_embed, conf_embed, coord_embed], dim=-1)  # (N, 128)
        
        # 融合
        enhanced = self.fusion(combined)
        
        return enhanced
```

#### 2.2.2 内存高效的四点几何位置编码
**设计思路**: 保持几何理解能力，但控制维度在128以内

```python
class CompactFourPointGeometricEncoding(nn.Module):
    """紧凑四点几何位置编码"""
    def __init__(self, embed_dim=128):
        super().__init__()
        
        # 4个点的位置编码器 (每个点32维)
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 32),  # 每个点32维
            nn.GELU(),
            nn.LayerNorm(32)
        )
        
        # 几何特征融合 (4×32=128维)
        self.geometric_fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        
        # 形状特征编码器 (面积+法向量→16维)
        self.shape_encoder = nn.Sequential(
            nn.Linear(4, 16),  # 面积+法向量(3维)=4维→16维
            nn.GELU(),
            nn.LayerNorm(16)
        )
        
        # 最终融合 (128+16→128)
        self.final_fusion = nn.Sequential(
            nn.Linear(embed_dim + 16, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, face_coordinates):
        """
        Args:
            face_coordinates: (N, 9) - 3个顶点的坐标
        Returns:
            pos_encoding: (N, 128) - 紧凑位置编码
        """
        batch_size = face_coordinates.shape[0]
        
        # 重塑为3个顶点
        vertices = face_coordinates.view(batch_size, 3, 3)  # (N, 3, 3)
        
        # 计算重心
        centroid = vertices.mean(dim=1, keepdim=True)  # (N, 1, 3)
        
        # 四个点: 3顶点 + 重心
        four_points = torch.cat([vertices, centroid], dim=1)  # (N, 4, 3)
        
        # 每个点编码32维
        point_encodings = []
        for i in range(4):
            point_enc = self.point_encoder(four_points[:, i, :])  # (N, 32)
            point_encodings.append(point_enc)
        
        # 拼接4个点的编码
        combined_points = torch.cat(point_encodings, dim=-1)  # (N, 128)
        
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
        shape_enc = self.shape_encoder(shape_features)  # (N, 16)
        
        # 最终融合
        final_encoding = torch.cat([geo_features, shape_enc], dim=-1)  # (N, 144)
        pos_encoding = self.final_fusion(final_encoding)  # (N, 128)
        
        return pos_encoding
```

#### 2.2.3 分层注意力机制 (处理大规模面片)
**关键创新**: 处理40,000+面片的高效注意力

```python
class HierarchicalAttention(nn.Module):
    """分层注意力 - 处理大规模面片"""
    def __init__(self, embed_dim=128, num_heads=8, chunk_size=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size  # 分块处理，避免内存爆炸
        
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
            nn.Linear(embed_dim, embed_dim * 2),  # 只扩展2倍，节省内存
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(self, x, confidence):
        """
        Args:
            x: (N, 128) 面片特征，N可能是40,000+
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
        
        # 3. 全局上下文 (采样代表性面片)
        if N > self.chunk_size:
            # 采样高confidence的面片作为全局代表
            conf_scores = confidence.squeeze()
            _, top_indices = torch.topk(conf_scores, min(self.chunk_size, N), dim=0)
            global_representatives = x[top_indices]
            
            # 全局注意力
            global_out, _ = self.global_attention(x, global_representatives, global_representatives)
        else:
            global_out, _ = self.global_attention(x, x, x)
        
        x = self.norm3(x + global_out)
        
        # 4. 前馈网络
        ffn_out = self.ffn(x)
        x = x + ffn_out
        
        return x
```

### 2.3 完整内存高效模型

```python
class MemoryEfficientSOTAModel(nn.Module):
    """内存高效的SOTA模型"""
    
    def __init__(self, 
                 input_dim=458,
                 embed_dim=128,      # 大幅降维
                 num_layers=4,       # 减少层数
                 num_heads=8,        # 适中的头数
                 chunk_size=1024,    # 分块大小
                 dropout=0.1):
        super().__init__()
        
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
        
        # 轻量分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, features):
        """
        Args:
            features: (N, 458) 其中N可能是40,000+
        """
        # 分离特征
        partfield = features[:, :448]
        confidence = features[:, 448:449]
        coordinates = features[:, 449:458]
        
        # 紧凑嵌入 458→128
        x = self.embedding(partfield, confidence, coordinates)  # (N, 128)
        
        # 位置编码
        pos_enc = self.pos_encoding(coordinates)
        x = x + pos_enc
        
        # 分层注意力
        for layer in self.layers:
            x = layer(x, confidence)
        
        # 分类
        logits = self.classifier(x)  # (N, 2)
        
        return logits
```

## 3. 内存优化策略

### 3.1 关键优化点

1. **大幅降维**: 458→128维，减少75%内存
2. **分块处理**: 大于1024面片时分块计算注意力
3. **采样全局**: 用高confidence面片代表全局上下文
4. **轻量网络**: 4层×8头，平衡性能和效率
5. **梯度检查点**: 训练时使用gradient checkpointing

### 3.2 内存估算

**原始方案** (512维×8层×16头):
- 40,000面片 × 512维 = 20.48M参数/层
- 8层 = 163.84M参数/样本

**优化方案** (128维×4层×8头):
- 40,000面片 × 128维 = 5.12M参数/层  
- 4层 = 20.48M参数/样本
- **减少87.5%内存占用！**

## 4. 训练策略

```python
# 内存高效配置
EFFICIENT_CONFIG = {
    'embed_dim': 128,           # 紧凑维度
    'num_layers': 4,            # 适中层数
    'num_heads': 8,             # 平衡的头数
    'chunk_size': 1024,         # 分块大小
    'learning_rate': 1e-4,      # 标准学习率
    'gradient_checkpointing': True,  # 节省内存
    'mixed_precision': True,    # 使用FP16
}

# 使用gradient checkpointing
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, x):
    for layer in self.layers:
        x = checkpoint(layer, x, use_reentrant=False)
    return x
```

这样设计怎么样？现在是真正的**内存高效SOTA版本**了！