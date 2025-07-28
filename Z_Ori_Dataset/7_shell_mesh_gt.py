#!/usr/bin/env python3
"""
为shell版本的几何体生成红绿可视化

工作流程:
1. 读取原始版本的标签文件 labels_{joint_id}.txt
2. 读取shell版本的OBJ文件 (带注释的面片)
3. 根据shell版本的面片状态，调整标签
4. 生成shell版本的红绿可视化PLY文件

输入:
- /home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/urdf/{id}/yy_visualization/labels_{joint_id}.txt
- /home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/urdf_shell/{id}/yy_merged.obj

输出:
- /home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/urdf_shell/{id}/yy_visualization/moveable_{joint_id}.ply
- /home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/urdf_shell/{id}/yy_visualization/labels_{joint_id}.txt
"""

import os
import glob
import trimesh
import numpy as np
from tqdm import tqdm

# 配置路径
BASE_DIR = "/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small"
URDF_DIR = os.path.join(BASE_DIR, "urdf")
URDF_SHELL_DIR = os.path.join(BASE_DIR, "urdf_shell")

def read_shell_obj_with_face_mapping(shell_obj_path):
    """
    读取shell版本的OBJ文件，返回:
    1. trimesh对象 (只包含非注释的面片)
    2. 面片映射关系 (shell中第i个面片对应原始的第几个面片)
    """
    if not os.path.exists(shell_obj_path):
        print(f"[Error] Shell OBJ文件不存在: {shell_obj_path}")
        return None, None
    
    vertices = []
    faces = []
    face_mapping = []  # shell中第i个面片对应原始的第几个面片
    
    original_face_idx = 0  # 原始文件中的面片索引 (包括注释的)
    shell_face_idx = 0     # shell文件中的有效面片索引
    
    with open(shell_obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                # 顶点
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                # 有效面片
                parts = line.split()[1:]  # 跳过'f'
                # 解析面片索引 (可能包含纹理/法线索引)
                face_vertices = []
                for part in parts:
                    vertex_idx = int(part.split('/')[0]) - 1  # OBJ索引从1开始
                    face_vertices.append(vertex_idx)
                
                if len(face_vertices) >= 3:
                    faces.append(face_vertices[:3])  # 只取前3个顶点(三角形)
                    face_mapping.append(original_face_idx)
                    shell_face_idx += 1
                
                original_face_idx += 1
                
            elif line.startswith('# UNHIT_FACE f '):
                # 注释的面片，只增加原始索引
                original_face_idx += 1
    
    if not vertices or not faces:
        print(f"[Error] Shell OBJ文件无效: {shell_obj_path}")
        return None, None
    
    # 创建trimesh对象
    try:
        mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
        print(f"[Info] Shell mesh: {len(vertices)} 顶点, {len(faces)} 面片 (原始: {original_face_idx} 面片)")
        return mesh, face_mapping
    except Exception as e:
        print(f"[Error] 创建shell mesh失败: {e}")
        return None, None

def read_original_labels(labels_path):
    """读取原始标签文件"""
    if not os.path.exists(labels_path):
        print(f"[Error] 原始标签文件不存在: {labels_path}")
        return None
    
    labels = []
    with open(labels_path, 'r') as f:
        for line in f:
            labels.append(int(line.strip()))
    
    print(f"[Info] 读取原始标签: {len(labels)} 个面片")
    return np.array(labels)

def create_shell_labels(original_labels, face_mapping):
    """根据面片映射创建shell版本的标签"""
    shell_labels = []
    for shell_idx, original_idx in enumerate(face_mapping):
        if original_idx < len(original_labels):
            shell_labels.append(original_labels[original_idx])
        else:
            print(f"[Warning] 映射索引超出范围: {original_idx} >= {len(original_labels)}")
            shell_labels.append(0)  # 默认为固定
    
    return np.array(shell_labels)

def create_shell_visualization(shell_mesh, shell_labels, output_dir, joint_id):
    """为shell版本创建红绿可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建面片颜色
    face_colors = np.zeros((len(shell_labels), 4), dtype=np.uint8)
    face_colors[shell_labels == 0] = [255, 0, 0, 255]  # 红色表示不可动部分
    face_colors[shell_labels == 1] = [0, 255, 0, 255]  # 绿色表示可动部分
    
    # 创建带颜色的网格
    colored_mesh = trimesh.Trimesh(
        vertices=shell_mesh.vertices,
        faces=shell_mesh.faces,
        face_colors=face_colors
    )
    
    # 导出PLY文件
    ply_path = os.path.join(output_dir, f"moveable_{joint_id}.ply")
    colored_mesh.export(ply_path)
    
    # 保存shell版本的标签
    labels_path = os.path.join(output_dir, f"labels_{joint_id}.txt")
    with open(labels_path, 'w') as f:
        for label in shell_labels:
            f.write(f"{label}\n")
    
    # 统计信息
    moveable_count = np.sum(shell_labels == 1)
    fixed_count = np.sum(shell_labels == 0)
    
    print(f"[Success] Shell可视化: {ply_path}")
    print(f"[Success] Shell标签: {labels_path}")
    print(f"[Stats] {moveable_count} 个可动面片, {fixed_count} 个固定面片")
    
    return ply_path, labels_path

def process_one_model(model_id):
    """处理单个模型"""
    print(f"\n=== 处理模型 {model_id} ===")
    
    # 路径定义
    urdf_viz_dir = os.path.join(URDF_DIR, model_id, "yy_visualization")
    shell_obj_path = os.path.join(URDF_SHELL_DIR, model_id, "yy_merged.obj")
    shell_viz_dir = os.path.join(URDF_SHELL_DIR, model_id, "yy_visualization")
    
    # 检查shell OBJ文件
    if not os.path.exists(shell_obj_path):
        print(f"[Skip] Shell OBJ不存在: {shell_obj_path}")
        return False
    
    # 检查原始可视化目录
    if not os.path.exists(urdf_viz_dir):
        print(f"[Skip] 原始可视化目录不存在: {urdf_viz_dir}")
        return False
    
    # 查找所有标签文件
    label_files = glob.glob(os.path.join(urdf_viz_dir, "labels_*.txt"))
    if not label_files:
        print(f"[Skip] 未找到标签文件在: {urdf_viz_dir}")
        return False
    
    print(f"[Info] 找到 {len(label_files)} 个标签文件")
    
    # 读取shell mesh
    shell_mesh, face_mapping = read_shell_obj_with_face_mapping(shell_obj_path)
    if shell_mesh is None:
        return False
    
    success_count = 0
    
    # 处理每个关节的标签
    for label_file in label_files:
        # 提取关节ID
        filename = os.path.basename(label_file)
        joint_id = filename.replace("labels_", "").replace(".txt", "")
        
        print(f"\n--- 处理关节 {joint_id} ---")
        
        # 读取原始标签
        original_labels = read_original_labels(label_file)
        if original_labels is None:
            continue
        
        # 创建shell标签
        shell_labels = create_shell_labels(original_labels, face_mapping)
        
        # 创建shell可视化
        try:
            ply_path, labels_path = create_shell_visualization(
                shell_mesh, shell_labels, shell_viz_dir, joint_id
            )
            success_count += 1
        except Exception as e:
            print(f"[Error] 创建关节 {joint_id} 的shell可视化失败: {e}")
    
    print(f"\n=== 模型 {model_id} 完成: {success_count}/{len(label_files)} 个关节成功 ===")
    return success_count > 0

def find_models_with_visualization():
    """查找所有有可视化的模型"""
    models = []
    
    if not os.path.exists(URDF_DIR):
        print(f"[Error] URDF目录不存在: {URDF_DIR}")
        return models
    
    for item in os.listdir(URDF_DIR):
        model_path = os.path.join(URDF_DIR, item)
        if os.path.isdir(model_path):
            viz_dir = os.path.join(model_path, "yy_visualization")
            if os.path.exists(viz_dir):
                label_files = glob.glob(os.path.join(viz_dir, "labels_*.txt"))
                if label_files:
                    models.append(item)
    
    return sorted(models)

def main():
    print("="*60)
    print("Shell版本红绿可视化生成脚本")
    print("="*60)
    print(f"原始目录: {URDF_DIR}")
    print(f"Shell目录: {URDF_SHELL_DIR}")
    print()
    
    # 查找所有模型
    models = find_models_with_visualization()
    print(f"[Info] 找到 {len(models)} 个有可视化的模型")
    
    if not models:
        print("[Error] 未找到任何有可视化的模型")
        return
    
    # 显示前几个模型作为示例
    print(f"[Sample] 前5个模型: {models[:5]}")
    
    # 处理所有模型
    success_count = 0
    failed_count = 0
    
    for model_id in tqdm(models, desc="处理模型"):
        try:
            if process_one_model(model_id):
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"[Error] 处理模型 {model_id} 时出错: {e}")
            failed_count += 1
    
    print("\n" + "="*60)
    print("处理完成")
    print("="*60)
    print(f"成功: {success_count} 个模型")
    print(f"失败: {failed_count} 个模型")
    print(f"总计: {len(models)} 个模型")
    print(f"\n输出目录: {URDF_SHELL_DIR}/{{model_id}}/yy_visualization/")

if __name__ == "__main__":
    main()
