# Attention-MLP 数据加载器规范

## 1. 数据加载概述

我们在1_NOTE.md里面写过如何复制这些数据到我们的data_small_train。
现在我们有了这些数据，我们需要先写一个loader.py。
因为我们需要训练一个模型，但是我们目前先不用管模型的事情，一步步来。

**重要**: 由于copy_training_data.py已经做过完整的数据完整性检查，loader.py可以专注于数据加载和预处理，无需重复验证。

### 1.1 数据目录结构
```
/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small_train/
├── {类别}_{id}_{view}_joint_{joint_id}/
│   ├── pred_face_confidence.txt     # 面片置信度 (来自result目录)
│   ├── face_vertex_mapping.txt      # 面片-顶点映射 (来自urdf/{id}目录)
│   ├── {id}.npy                     # PartField特征 (来自urdf/{id}/feature/mesh目录)
│   ├── {id}.obj                     # 几何体文件 (来自urdf/{id}/yy_merged.obj，重命名)
│   └── labels_{joint_id}.txt        # Ground Truth标签 (来自urdf/{id}/yy_visualization目录)
```

### 1.2 输入输出规范

**输入特征组成**:
- **PartField特征**: 448维 (从{id}.npy)
- **置信度**: 1维 (从pred_face_confidence.txt)
- **面片坐标**: 9维 (从face_vertex_mapping.txt，3个顶点×3个坐标)

**总输入维度**: 448 + 1 + 9 = **458维** 直接concat到448后面

**输出标签**:
- 对于每一个面片判断为1或者0
- Ground Truth来自labels_{joint_id}.txt

**数据样本定义**:
- 每一个`{类别}_{id}_{view}_joint_{joint_id}/`都是我们的独立数据样本
- 需要8:1:1划分train、val、test

## 2. 数据加载器详细需求

### 2.1 数据集类设计

#### 2.1.1 基础数据集类
```python
class PartSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: 数据目录路径
            split: 'train', 'val', 'test'
            transform: 数据变换函数
        """
```

#### 2.1.2 必需方法
- `__len__()`: 返回数据集大小
- `__getitem__(idx)`: 返回单个样本
- `_load_sample(sample_path)`: 加载单个样本的所有文件
- `_split_data()`: 8:1:1划分数据集

### 2.2 数据加载和预处理

#### 2.2.1 文件读取函数
1. **读取PartField特征**
   ```python
   def load_features(npy_path):
       # 加载{id}.npy文件
       # 返回: (num_faces, 448) numpy数组
       # 注意：不对PartField特征做标准化，保持原始特征
   ```

2. **读取置信度**
   ```python
   def load_confidence(confidence_path):
       # 加载pred_face_confidence.txt
       # 格式: face_id confidence_value
       # 返回: (num_faces,) numpy数组，按face_id排序
       # 置信度已在[0,1]范围，无需额外标准化
   ```

3. **读取面片坐标**
   ```python
   def load_face_coordinates(mapping_path):
       # 加载face_vertex_mapping.txt
       # 格式: face_id v1_x v1_y v1_z v2_x v2_y v2_z v3_x v3_y v3_z
       # 返回: (num_faces, 9) numpy数组
       # 坐标编码策略：需要合理的位置编码方案
   ```

4. **读取标签**
   ```python
   def load_labels(labels_path):
       # 加载labels_{joint_id}.txt
       # 每行一个标签(0或1)
       # 返回: (num_faces,) numpy数组
   ```

#### 2.2.2 特征组合 (简化版)
```python
def combine_features(features, confidence, coordinates):
    """
    组合所有特征为458维向量
    
    Args:
        features: (num_faces, 448) - PartField特征，不做标准化
        confidence: (num_faces,) - 置信度，已在[0,1]范围
        coordinates: (num_faces, 9) - 面片坐标，需要合理编码
        
    Returns:
        combined: (num_faces, 458) - 直接concat
    """
    # 直接拼接：[PartField(448) + confidence(1) + coordinates(9)]
    confidence_expanded = confidence.reshape(-1, 1)
    combined = np.concatenate([features, confidence_expanded, coordinates], axis=1)
    return combined
```

### 2.3 坐标编码策略讨论

由于您提到"坐标主要是把position给encode进来"，我们需要考虑几种编码方案：

#### 2.3.1 选项1: 直接使用原始坐标
```python
def encode_coordinates_raw(coordinates):
    """直接使用9维原始坐标"""
    return coordinates  # (num_faces, 9)
```

#### 2.3.2 选项2: 标准化坐标
```python
def encode_coordinates_normalized(coordinates):
    """按样本标准化坐标"""
    # 选择：全局标准化 vs 按样本标准化
    mean = coordinates.mean(axis=0)
    std = coordinates.std(axis=0) + 1e-8
    return (coordinates - mean) / std
```

#### 2.3.3 选项3: 相对坐标编码
```python
def encode_coordinates_relative(coordinates):
    """使用相对于重心的坐标"""
    # 计算每个面片的重心
    face_centers = coordinates.reshape(-1, 3, 3).mean(axis=1)  # (num_faces, 3)
    # 可能的编码方案...
```

**建议**: 先从最简单的原始坐标开始，根据训练效果调整。好的 直接用原始的坐标先

### 2.4 数据集划分策略

#### 2.4.1 划分原则 (重要)
- **ID级别划分**: 确保同一个物体ID不会同时出现在train和test中
- **类别平衡**: 尽量保持各类别在各集合中的比例
- **可重现**: 使用固定随机种子

#### 2.4.2 划分实现
```python
def split_samples(sample_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    按8:1:1划分样本
    
    Strategy:
    1. 按物体ID分组 (避免数据泄漏)
    2. 按ID级别划分
    3. 保持类别分布平衡
    """
```

### 2.5 批处理和采样 (针对3D Mesh特点)

#### 2.5.1 Batch Size = 1 的设计
```python
def get_dataloader(dataset, batch_size=1, shuffle=True, num_workers=24):
    """
    创建PyTorch DataLoader
    
    注意事项:
    - batch_size=1: 3D mesh处理通常使用batch=1，与PartField模型一致
    - 不同样本的面片数量可能不同，batch=1避免了padding问题
    - num_workers=24: 利用多核并行加载
    """
```

#### 2.5.2 简化的collate函数
```python
def collate_fn(batch):
    """
    由于batch_size=1，collate函数可以简化
    直接返回单个样本的数据
    """
    assert len(batch) == 1, "当前设计假设batch_size=1"
    return batch[0]
```

## 3. 简化的质量控制

### 3.1 基础统计信息
```python
def collect_dataset_statistics(dataset):
    """
    收集必要的统计信息:
    - 样本数量分布 (train/val/test)
    - 面片数量分布范围
    - 标签分布（正负样本比例）
    """
```

### 3.2 简单验证 (可选)
```python
def quick_validation(sample):
    """
    快速验证单个样本的维度匹配
    由于copy阶段已验证完整性，这里只做维度检查
    """
    features, labels = sample
    assert features.shape[1] == 458, f"特征维度错误: {features.shape[1]}"
    assert features.shape[0] == labels.shape[0], f"样本数量不匹配"
```

## 4. 配置文件设计

### 4.1 简化的数据配置
```yaml
data:
  data_dir: "/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small_train"
  splits:
    train: 0.8
    val: 0.1
    test: 0.1
  seed: 42
  
preprocessing:
  normalize_partfield: false  # 不标准化PartField特征//既然不标准化 相关的部分可以直接不写了
  normalize_coordinates: true  # 坐标可能需要标准化
  coordinate_encoding: "raw"   # 坐标编码方式: raw/normalized/relative

dataloader:
  batch_size: 1      # 3D mesh处理标准做法
  num_workers: 24    # 多核并行
  shuffle: true
  pin_memory: true
```

## 5. 实现优先级 (简化版)

### 阶段1: 核心功能
1. 基础Dataset类
2. 4个文件读取函数
3. ID级别的8:1:1划分
4. 特征拼接 (458维)

### 阶段2: 优化
1. 坐标编码策略优化
2. 统计信息收集
3. 配置文件支持

### 阶段3: 高级功能 (可选)
1. 数据缓存优化
2. 可视化工具

## 6. 关键设计决策总结

1. **无需重复验证**: copy_training_data.py已确保数据完整性
2. **batch_size=1**: 遵循3D mesh处理惯例，避免padding复杂性
3. **保持PartField特征**: 不对448维特征做标准化
4. **ID级别划分**: 防止数据泄漏的关键策略
5. **简单拼接**: 458维特征直接concat，无复杂预处理
6. **坐标编码**: 从简单方案开始，根据效果迭代