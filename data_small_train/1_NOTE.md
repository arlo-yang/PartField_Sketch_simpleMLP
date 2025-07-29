# Attention-MLP 训练数据准备任务

## 1. 数据源概述

### 1.1 现有数据目录结构
```
/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/
├── result/ 
│   └── {类别}_{id}_{view}_joint_{joint_id}/
│       └── pred_face_confidence.txt     # 面片置信度 ⭐ 重要
├── urdf/
│   └── {id}/
│       ├── face_vertex_mapping.txt      # 面片-顶点映射 ⭐ 重要
│       ├── yy_merged.obj                # 几何体文件 ⭐ 重要
│       ├── feature/
│       │   └── mesh/
│       │       └── {id}.npy             # PartField特征 (448维) ⭐ 重要
│       └── yy_visualization/
│           └── labels_{joint_id}.txt    # Ground Truth标签 ⭐ 重要
```

### 1.2 目标数据目录
```
/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small_train/
├── {类别}_{id}_{view}_joint_{joint_id}/
│   ├── pred_face_confidence.txt     # 面片置信度 (来自result目录)
│   ├── face_vertex_mapping.txt      # 面片-顶点映射 (来自urdf/{id}目录)
│   ├── {id}.npy                     # PartField特征 (来自urdf/{id}/feature/mesh目录)
│   ├── {id}.obj                     # 几何体文件 (来自urdf/{id}/yy_merged.obj，重命名)
│   └── labels_{joint_id}.txt        # Ground Truth标签 (来自urdf/{id}/yy_visualization目录)
```

### 1.3 多关节数据示例
同一个物体ID有多个关节时，会生成多个训练样本：

```
/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small_train/
├── Storagefurniture_45677_segmentation_center_joint_0/
│   ├── pred_face_confidence.txt     # 关节0的置信度 (不同)
│   ├── face_vertex_mapping.txt      # 面片-顶点映射 (相同)
│   ├── 45677.npy                    # PartField特征 (相同)
│   ├── 45677.obj                    # 几何体文件 (相同，从yy_merged.obj重命名)
│   └── labels_0.txt                 # 关节0的标签 (不同)
│
├── Storagefurniture_45677_segmentation_center_joint_1/
│   ├── pred_face_confidence.txt     # 关节1的置信度 (不同)
│   ├── face_vertex_mapping.txt      # 面片-顶点映射 (相同)
│   ├── 45677.npy                    # PartField特征 (相同)
│   ├── 45677.obj                    # 几何体文件 (相同，从yy_merged.obj重命名)
│   └── labels_1.txt                 # 关节1的标签 (不同)
```

### 1.4 文件分类说明
**按关节变化的文件 (每个关节不同)**:
- `pred_face_confidence.txt` - 不同关节的置信度预测不同
- `labels_{joint_id}.txt` - 不同关节的ground truth标签不同

**按ID不变的文件 (同ID相同)**:
- `face_vertex_mapping.txt` - 几何体的面片-顶点映射关系
- `{id}.npy` - PartField特征提取结果
- `{id}.obj` - 几何体文件 (从yy_merged.obj重命名)

**匹配策略**: 先匹配id，再匹配joint_id

## 2. 数据组成详解

### 2.1 文件命名规则
**格式**: `{类别}_{id}_segmentation_{视角}_joint_{joint_id}`

**示例**: `Microwave_7310_segmentation_center_joint_0`
- 类别: Microwave
- ID: 7310  
- 渲染类型: segmentation (固定)
- 视角: center (当前只保留center视角)
- 关节ID: 0

### 2.2 解析函数 (请务必使用这个函数)
```python
def parse_filename(filename):
    """
    解析文件名为各个组件
    支持格式: {类别}_{id}_{渲染类型}[_{视角}][_joint_{joint_id}].png
    """
    base_pattern = r'^([a-zA-Z]+)_(\d+)_([a-zA-Z\-]+)'
    base_match = re.match(base_pattern, filename)
    
    if not base_match:
        return None
    
    category = base_match.group(1)
    obj_id = base_match.group(2)
    render_type = base_match.group(3)
    
    # 获取剩余部分（可能包含视角和关节ID）
    remaining = filename[len(base_match.group(0)):]
    
    # 检查是否包含joint信息
    joint_pattern = r'_joint_(\d+)'
    joint_match = re.search(joint_pattern, remaining)
    joint_id = joint_match.group(1) if joint_match else None
    
    # 提取视角信息（如果存在）
    view = None
    if joint_match:
        view_part = remaining[:joint_match.start()]
        if view_part.startswith('_') and len(view_part) > 1:
            view = view_part[1:]  # 去掉开头的下划线
    elif remaining.startswith('_') and len(remaining) > 1:
        view = remaining[1:].split('.')[0]  # 去掉扩展名
    
    result = {
        'category': category,
        'id': obj_id,
        'render_type': render_type,
        'joint': joint_id,
        'view': view
    }
    
    return result
```

## 3. 具体数据文件说明

### 3.1 面片置信度文件 (每个关节不同)
**源路径**: `data_small/result/{类别}_{id}_segmentation_{view}_joint_{joint_id}/pred_face_confidence.txt`
**目标路径**: `data_small_train/{类别}_{id}_segmentation_{view}_joint_{joint_id}/pred_face_confidence.txt`
**格式**: 每行包含 `face_id confidence_value`
```
0 0.856234
1 0.234567
2 0.789012
...
```

### 3.2 面片-顶点映射文件 (同ID相同)
**源路径**: `data_small/urdf/{id}/face_vertex_mapping.txt`
**目标路径**: `data_small_train/{类别}_{id}_segmentation_{view}_joint_{joint_id}/face_vertex_mapping.txt`
**格式**: `face_id v1_x v1_y v1_z v2_x v2_y v2_z v3_x v3_y v3_z`
```
# 面片-顶点坐标映射
# 格式: face_id vertex1_x vertex1_y vertex1_z vertex2_x vertex2_y vertex2_z vertex3_x vertex3_y vertex3_z
# face_id从0开始

0 0.308167 0.475299 0.421852 0.308167 0.180642 0.416997 0.308167 0.180642 0.421852
1 0.308167 0.180642 0.416997 0.308167 0.475299 0.421852 0.308167 0.475299 0.416997
...
```

### 3.3 PartField特征文件 (同ID相同)
**源路径**: `data_small/urdf/{id}/feature/mesh/{id}.npy`
**目标路径**: `data_small_train/{类别}_{id}_segmentation_{view}_joint_{joint_id}/{id}.npy`
**格式**: numpy数组 `(num_faces, 448)`
- 每个面片对应448维特征向量

### 3.4 几何体文件 (同ID相同)
**源路径**: `data_small/urdf/{id}/yy_merged.obj`
**目标路径**: `data_small_train/{类别}_{id}_segmentation_{view}_joint_{joint_id}/{id}.obj`
**操作**: 复制并重命名为 `{id}.obj`
**格式**: 标准OBJ格式
- 包含顶点坐标和面片定义
- 用于提供几何体的完整结构信息

### 3.5 Ground Truth标签文件 (每个关节不同)
**源路径**: `data_small/urdf/{id}/yy_visualization/labels_{joint_id}.txt`
**目标路径**: `data_small_train/{类别}_{id}_segmentation_{view}_joint_{joint_id}/labels_{joint_id}.txt`
**格式**: 每行一个标签 (0或1)
```
0
1
0
1
...
```
- 行号对应face_id
- 0: 固定部分 (base part)
- 1: 可动部分 (movable part)

## 4. Attention-MLP 输入数据规格

### 4.1 输入特征组成
每个面片的输入特征包括：
1. **PartField特征**: 448维向量
2. **置信度**: 1维标量
3. **三角面片坐标**: 9维向量 (3个顶点 × 3个坐标)

**总维度**: 448 + 1 + 9 = **458维**

### 4.2 数据对齐要求
- 所有数据必须按face_id对齐
- 同一个样本的所有文件必须包含相同数量的面片
- joint_id必须在各个文件中保持一致

## 5. 数据处理任务

### 5.1 数据复制策略
**复制规则**:
1. **遍历result目录**: 找到所有 `{类别}_{id}_segmentation_{view}_joint_{joint_id}` 目录
2. **验证文件存在性**: 确保对应的urdf文件存在
3. **创建训练样本**: 只有当所有必需文件都存在时才创建样本

**必需文件检查清单**:
- ✅ `result/{sample_name}/pred_face_confidence.txt`
- ✅ `urdf/{id}/face_vertex_mapping.txt`  
- ✅ `urdf/{id}/feature/mesh/{id}.npy`
- ✅ `urdf/{id}/yy_merged.obj`
- ✅ `urdf/{id}/yy_visualization/labels_{joint_id}.txt`

### 5.2 数据复制脚本功能
1. **自动发现**: 扫描result目录找到所有可用样本
2. **交叉验证**: 检查urdf目录中对应文件是否存在
3. **完整性检查**: 只复制完整的样本 (5个文件都存在)
4. **跳过策略**: 
   - 如果result有但urdf缺失任一文件 → 跳过整个样本
   - 如果urdf有但result没有对应样本 → 跳过
5. **文件重命名**: 
   - `yy_merged.obj` → `{id}.obj`
   - `labels_{joint_id}.txt` 保持原名

### 5.3 错误处理和日志
- **缺失文件**: 记录具体缺失的文件路径并跳过样本
- **重复数据**: 检测并避免重复复制
- **文件权限**: 处理文件读取/写入权限问题
- **路径错误**: 验证源路径和目标路径的有效性
- **处理日志**: 生成详细的复制报告

## 6. 注意事项

### 6.1 ID和关节匹配规则
- **同一物体多关节**: 为每个关节创建独立的训练样本
- **文件共享**: face_vertex_mapping.txt、{id}.npy 和 {id}.obj 在同ID的所有关节间共享
- **文件独立**: pred_face_confidence.txt 和 labels_{joint_id}.txt 每个关节都不同

### 6.2 数据一致性要求
- **面片数量**: 所有文件中的面片数量必须一致
- **face_id范围**: 从0开始，连续编号
- **维度匹配**: 特征维度和标签数量必须对应
- **几何一致性**: OBJ文件的面片定义必须与face_vertex_mapping.txt一致

### 6.3 实现优先级
1. **第一阶段**: 实现数据复制脚本，建立完整的训练数据集，顺便检查缺失文件
2. **第二阶段**: 实现数据验证脚本，确保数据完整性和一致性
3. **第三阶段**: 实现数据加载器，支持模型训练

### 6.4 质量控制
- **数据统计**: 记录成功/失败的样本数量
- **完整性报告**: 生成数据完整性检查报告
- **维度验证**: 确保所有样本的特征维度一致
- **几何验证**: 验证OBJ文件与面片映射的一致性