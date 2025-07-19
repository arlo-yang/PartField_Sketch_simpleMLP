"""
面片ID对应关系验证脚本
用于比较不同格式几何体的面片顺序是否一致

使用方法:
python test_face.py [--id MODEL_ID]

如果指定了MODEL_ID，则只处理该模型
否则批量处理所有模型
"""

import os
import glob
import re
import numpy as np
import trimesh
import argparse
from pathlib import Path
from tqdm import tqdm

# 全局路径配置
URDF_DIR = "/hy-tmp/PartField_Sketch_simpleMLP/data_small/urdf"
RESULT_DIR = "/hy-tmp/PartField_Sketch_simpleMLP/data_small/result"
OUTPUT_DIR = "/hy-tmp/PartField_Sketch_simpleMLP/z_face_average/greenred"

def find_model_ids():
    """查找所有可用的模型ID"""
    model_dirs = [d for d in os.listdir(URDF_DIR) if os.path.isdir(os.path.join(URDF_DIR, d))]
    return [d for d in model_dirs if d.isdigit()]

def find_joint_ids(model_id):
    """查找指定模型的所有关节ID"""
    visualization_dir = os.path.join(URDF_DIR, model_id, "yy_visualization")
    if not os.path.exists(visualization_dir):
        return []
    
    moveable_files = glob.glob(os.path.join(visualization_dir, "moveable_*.ply"))
    joint_ids = []
    
    for file_path in moveable_files:
        match = re.search(r"moveable_(\d+)\.ply", os.path.basename(file_path))
        if match:
            joint_ids.append(match.group(1))
    
    return joint_ids

def create_id_colored_mesh(mesh, name):
    """为每个面片分配基于ID的唯一颜色"""
    id_colors = np.ones((len(mesh.faces), 4), dtype=np.uint8) * 255
    
    # 使用面片ID的RGB编码作为颜色
    for i in range(len(mesh.faces)):
        # 将面片ID编码为RGB颜色
        r = (i & 0xFF0000) >> 16
        g = (i & 0x00FF00) >> 8
        b = i & 0x0000FF
        id_colors[i, :3] = [r, g, b]
    
    # 创建ID测试网格
    id_test_mesh = mesh.copy()
    id_test_mesh.visual.face_colors = id_colors
    return id_test_mesh

def process_model(model_id, joint_id=None):
    """处理单个模型"""
    print(f"\n处理模型: {model_id}, 关节: {joint_id if joint_id else 'all'}")
    
    # 构建文件路径
    obj_path = os.path.join(URDF_DIR, model_id, "yy_merged.obj")
    ply_path = os.path.join(URDF_DIR, model_id, "feature", f"{model_id}.ply")
    
    # 检查基本文件是否存在
    if not os.path.exists(obj_path):
        print(f"错误：原始OBJ文件不存在: {obj_path}")
        return False
    
    if not os.path.exists(ply_path):
        print(f"错误：特征PLY文件不存在: {ply_path}")
        return False
    
    # 处理指定关节或所有关节
    joint_ids = [joint_id] if joint_id else find_joint_ids(model_id)
    if not joint_ids:
        print(f"警告：找不到模型 {model_id} 的任何关节")
        return False
    
    success = True
    for jid in joint_ids:
        moveable_path = os.path.join(URDF_DIR, model_id, "yy_visualization", f"moveable_{jid}.ply")
        if not os.path.exists(moveable_path):
            print(f"警告：可动部件文件不存在: {moveable_path}")
            continue
        
        # 创建输出子目录
        output_subdir = f"{model_id}_joint_{jid}_faceids"
        output_path = os.path.join(OUTPUT_DIR, output_subdir)
        os.makedirs(output_path, exist_ok=True)
        
        try:
            # 加载三种网格
            print(f"加载原始OBJ文件...")
            original_mesh = trimesh.load(obj_path, force='mesh')
            print(f"- 原始网格面片数量: {len(original_mesh.faces)}")
            
            print(f"加载特征PLY文件...")
            feature_mesh = trimesh.load(ply_path, force='mesh')
            print(f"- 特征网格面片数量: {len(feature_mesh.faces)}")
            
            print(f"加载可动部件PLY文件...")
            moveable_mesh = trimesh.load(moveable_path, force='mesh')
            print(f"- 可动部件网格面片数量: {len(moveable_mesh.faces)}")
            
            # 检查面片数量是否一致
            if len(original_mesh.faces) != len(feature_mesh.faces) or len(original_mesh.faces) != len(moveable_mesh.faces):
                print(f"警告：三个网格的面片数量不一致！")
                print(f"- 原始网格: {len(original_mesh.faces)} 面片")
                print(f"- 特征网格: {len(feature_mesh.faces)} 面片")
                print(f"- 可动部件网格: {len(moveable_mesh.faces)} 面片")
            
            # 创建面片ID颜色可视化
            print("创建面片ID颜色可视化...")
            id_colored_original = create_id_colored_mesh(original_mesh, "original")
            id_colored_feature = create_id_colored_mesh(feature_mesh, "feature")
            id_colored_moveable = create_id_colored_mesh(moveable_mesh, "moveable")
            
            # 保存ID颜色可视化网格
            original_output_path = os.path.join(output_path, f"{model_id}_original_id_colored.ply")
            feature_output_path = os.path.join(output_path, f"{model_id}_feature_id_colored.ply")
            moveable_output_path = os.path.join(output_path, f"{model_id}_moveable_{jid}_id_colored.ply")
            
            id_colored_original.export(original_output_path)
            id_colored_feature.export(feature_output_path)
            id_colored_moveable.export(moveable_output_path)
            
            print(f"ID颜色可视化网格已保存至: {output_path}")
            
            # 检查面片顺序是否一致
            print("检查面片顺序...")
            
            # 比较前10个面片的顶点索引
            print("\n前10个面片的顶点索引比较:")
            print("面片ID | 原始网格 | 特征网格 | 可动部件网格")
            print("-------|----------|----------|------------")
            
            for i in range(min(10, len(original_mesh.faces), len(feature_mesh.faces), len(moveable_mesh.faces))):
                orig_face = original_mesh.faces[i].tolist()
                feat_face = feature_mesh.faces[i].tolist() if i < len(feature_mesh.faces) else ["N/A"]
                move_face = moveable_mesh.faces[i].tolist() if i < len(moveable_mesh.faces) else ["N/A"]
                print(f"{i} | {orig_face} | {feat_face} | {move_face}")
            
            # 保存比较结果
            with open(os.path.join(output_path, "comparison_summary.txt"), 'w') as f:
                f.write(f"模型ID: {model_id}, 关节ID: {jid}\n\n")
                f.write(f"面片数量比较:\n")
                f.write(f"- 原始网格: {len(original_mesh.faces)} 面片\n")
                f.write(f"- 特征网格: {len(feature_mesh.faces)} 面片\n")
                f.write(f"- 可动部件网格: {len(moveable_mesh.faces)} 面片\n\n")
                
                f.write(f"前10个面片的顶点索引比较:\n")
                f.write("面片ID | 原始网格 | 特征网格 | 可动部件网格\n")
                f.write("-------|----------|----------|------------\n")
                
                for i in range(min(10, len(original_mesh.faces), len(feature_mesh.faces), len(moveable_mesh.faces))):
                    orig_face = original_mesh.faces[i].tolist()
                    feat_face = feature_mesh.faces[i].tolist() if i < len(feature_mesh.faces) else ["N/A"]
                    move_face = moveable_mesh.faces[i].tolist() if i < len(moveable_mesh.faces) else ["N/A"]
                    f.write(f"{i} | {orig_face} | {feat_face} | {move_face}\n")
            
        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            success = False
    
    return success

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="比较不同格式几何体的面片顺序")
    parser.add_argument("--id", type=str, help="指定要处理的模型ID")
    parser.add_argument("--joint", type=str, help="指定要处理的关节ID")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if args.id:
        # 处理单个模型
        process_model(args.id, args.joint)
    else:
        # 批量处理所有模型
        model_ids = find_model_ids()
        print(f"找到 {len(model_ids)} 个模型")
        
        success_count = 0
        for i, model_id in enumerate(tqdm(model_ids, desc="处理模型")):
            print(f"\n[{i+1}/{len(model_ids)}] 处理模型: {model_id}")
            if process_model(model_id):
                success_count += 1
        
        print(f"\n批量处理完成! 成功: {success_count}/{len(model_ids)}")

if __name__ == "__main__":
    main()