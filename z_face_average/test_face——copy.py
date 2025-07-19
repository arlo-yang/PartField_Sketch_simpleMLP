"""
面片ID对应关系验证脚本
用于批量测试不同格式几何体的面片ID对应关系是否一致

使用方法:
python test_face.py [--id MODEL_ID] [--debug]

如果指定了MODEL_ID，则只处理该模型
否则批量处理所有模型

添加--debug参数可以输出更多调试信息
"""

import os
import glob
import re
import numpy as np
import trimesh
import argparse
from pathlib import Path
from tqdm import tqdm
import json

# 全局路径配置
URDF_DIR = "/hy-tmp/PartField_Sketch_simpleMLP/data_small/urdf"
RESULT_DIR = "/hy-tmp/PartField_Sketch_simpleMLP/data_small/result"
OUTPUT_DIR = "/hy-tmp/PartField_Sketch_simpleMLP/z_face_average/greenred"

# 粉红色RGB值
PINK_COLOR = np.array([255, 105, 180, 255], dtype=np.uint8)  # 粉红色

def load_face_ids(face_ids_path):
    """加载面片ID文件"""
    face_ids = []
    try:
        with open(face_ids_path, 'r') as f:
            for line in f:
                face_ids.append(int(line.strip()))
        return face_ids
    except Exception as e:
        print(f"加载面片ID文件失败: {e}")
        return []

def colorize_mesh(mesh, selected_face_ids, color=PINK_COLOR):
    """根据指定的面片ID为网格着色，保留其他面片的原始颜色"""
    
    # 检查网格是否有现有颜色
    if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
        # 复制现有颜色，保持原始外观
        face_colors = mesh.visual.face_colors.copy()
    else:
        # 如果没有现有颜色，使用默认白色
        face_colors = np.ones((len(mesh.faces), 4), dtype=np.uint8) * 255
    
    # 将指定的面片ID设为指定颜色
    mask = np.isin(np.arange(len(mesh.faces)), selected_face_ids)
    face_colors[mask] = color
    
    # 设置面片颜色
    mesh.visual.face_colors = face_colors
    return mesh

def find_result_folders(model_id=None):
    """查找所有结果文件夹或指定模型ID的结果文件夹"""
    if model_id:
        pattern = os.path.join(RESULT_DIR, f"*_{model_id}_segmentation_*_joint_*")
    else:
        pattern = os.path.join(RESULT_DIR, "*_segmentation_*_joint_*")
    
    return glob.glob(pattern)

def extract_info_from_result_folder(folder_path):
    """从结果文件夹名称中提取信息"""
    folder_name = os.path.basename(folder_path)
    match = re.match(r"([A-Za-z]+)_(\d+)_segmentation_([a-z_]+)_joint_(\d+)", folder_name)
    
    if match:
        category = match.group(1)
        model_id = match.group(2)
        view = match.group(3)
        joint_id = match.group(4)
        return {
            "category": category,
            "model_id": model_id,
            "view": view,
            "joint_id": joint_id,
            "folder_path": folder_path
        }
    return None

def debug_mesh_info(mesh, name):
    """输出网格的详细信息，用于调试"""
    info = {
        "name": name,
        "faces_count": len(mesh.faces),
        "vertices_count": len(mesh.vertices),
        "has_face_colors": hasattr(mesh.visual, 'face_colors'),
        "has_vertex_colors": hasattr(mesh.visual, 'vertex_colors'),
        "bounds": mesh.bounds.tolist() if hasattr(mesh, 'bounds') else None,
        "face_sample": mesh.faces[:5].tolist() if len(mesh.faces) > 5 else mesh.faces.tolist(),
        "vertex_sample": mesh.vertices[:5].tolist() if len(mesh.vertices) > 5 else mesh.vertices.tolist()
    }
    return info

def compare_face_geometry(mesh1, mesh2, name1, name2, face_ids=None, max_samples=10):
    """比较两个网格中指定面片的几何信息"""
    if face_ids is None or len(face_ids) == 0:
        # 如果没有指定面片ID，随机选择一些面片
        max_face_id = min(len(mesh1.faces), len(mesh2.faces)) - 1
        if max_face_id < 0:
            return {"error": "至少有一个网格没有面片"}
        face_ids = np.random.choice(max_face_id, min(max_samples, max_face_id), replace=False).tolist()
    
    comparison = []
    for face_id in face_ids[:max_samples]:  # 限制样本数量
        if face_id >= len(mesh1.faces) or face_id >= len(mesh2.faces):
            comparison.append({
                "face_id": face_id,
                "error": "面片ID超出范围"
            })
            continue
        
        # 获取两个网格中的面片顶点索引
        face1 = mesh1.faces[face_id]
        face2 = mesh2.faces[face_id]
        
        # 获取顶点坐标
        vertices1 = [mesh1.vertices[i].tolist() for i in face1]
        vertices2 = [mesh2.vertices[i].tolist() for i in face2]
        
        # 计算面片中心点
        center1 = np.mean(vertices1, axis=0).tolist()
        center2 = np.mean(vertices2, axis=0).tolist()
        
        # 计算面片法向量
        v1 = np.array(vertices1[1]) - np.array(vertices1[0])
        v2 = np.array(vertices1[2]) - np.array(vertices1[0])
        normal1 = np.cross(v1, v2).tolist()
        
        v1 = np.array(vertices2[1]) - np.array(vertices2[0])
        v2 = np.array(vertices2[2]) - np.array(vertices2[0])
        normal2 = np.cross(v1, v2).tolist()
        
        comparison.append({
            "face_id": face_id,
            f"{name1}_vertices": vertices1,
            f"{name2}_vertices": vertices2,
            f"{name1}_center": center1,
            f"{name2}_center": center2,
            f"{name1}_normal": normal1,
            f"{name2}_normal": normal2,
            "center_distance": np.linalg.norm(np.array(center1) - np.array(center2))
        })
    
    return comparison

def process_model(info, debug=False):
    """处理单个模型"""
    model_id = info["model_id"]
    joint_id = info["joint_id"]
    category = info["category"]
    view = info["view"]
    folder_path = info["folder_path"]
    
    print(f"\n处理模型: {category}_{model_id}, 关节: {joint_id}, 视角: {view}")
    
    # 构建文件路径
    obj_path = os.path.join(URDF_DIR, model_id, "yy_merged.obj")
    ply_path = os.path.join(URDF_DIR, model_id, "feature", f"{model_id}.ply")
    moveable_path = os.path.join(URDF_DIR, model_id, "yy_visualization", f"moveable_{joint_id}.ply")
    face_ids_path = os.path.join(folder_path, "pred_face_ids.txt")
    
    # 检查文件是否存在
    missing_files = []
    for path, desc in [
        (obj_path, "原始OBJ文件"),
        (ply_path, "特征PLY文件"),
        (face_ids_path, "面片ID文件")
    ]:
        if not os.path.exists(path):
            missing_files.append(f"{desc}: {path}")
    
    # 检查moveable文件是否存在（这个可选）
    has_moveable = os.path.exists(moveable_path)
    
    if missing_files:
        print(f"跳过处理，缺少以下文件:")
        for msg in missing_files:
            print(f"- {msg}")
        return False
    
    # 创建输出子目录
    output_subdir = f"{category}_{model_id}_joint_{joint_id}"
    output_path = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(output_path, exist_ok=True)
    
    # 加载面片ID
    selected_face_ids = load_face_ids(face_ids_path)
    if not selected_face_ids:
        print(f"跳过处理，面片ID文件为空或无效: {face_ids_path}")
        return False
    
    print(f"加载了 {len(selected_face_ids)} 个选定面片ID")
    
    try:
        # 处理原始OBJ文件
        print(f"正在处理原始OBJ文件...")
        original_mesh = trimesh.load(obj_path, force='mesh')
        colored_original = colorize_mesh(original_mesh.copy(), selected_face_ids)
        obj_output_path = os.path.join(output_path, f"{model_id}_original_colored.ply")
        colored_original.export(obj_output_path)
        print(f"- 原始网格面片数量: {len(original_mesh.faces)}")
        
        # 处理特征PLY文件
        print(f"正在处理特征PLY文件...")
        feature_mesh = trimesh.load(ply_path, force='mesh')
        colored_feature = colorize_mesh(feature_mesh.copy(), selected_face_ids)
        ply_output_path = os.path.join(output_path, f"{model_id}_feature_colored.ply")
        colored_feature.export(ply_output_path)
        print(f"- 特征网格面片数量: {len(feature_mesh.faces)}")
        
        # 处理moveable文件（如果存在）
        if has_moveable:
            print(f"正在处理moveable PLY文件...")
            moveable_mesh = trimesh.load(moveable_path, force='mesh')
            colored_moveable = colorize_mesh(moveable_mesh.copy(), selected_face_ids)
            moveable_output_path = os.path.join(output_path, f"{model_id}_moveable_{joint_id}_colored.ply")
            colored_moveable.export(moveable_output_path)
            print(f"- Moveable网格面片数量: {len(moveable_mesh.faces)}")
        
        # 调试信息
        if debug:
            debug_info = {
                "model_id": model_id,
                "joint_id": joint_id,
                "category": category,
                "view": view,
                "selected_face_ids": selected_face_ids[:10] if len(selected_face_ids) > 10 else selected_face_ids,  # 只显示前10个
                "original_mesh": debug_mesh_info(original_mesh, "original"),
                "feature_mesh": debug_mesh_info(feature_mesh, "feature")
            }
            
            if has_moveable:
                debug_info["moveable_mesh"] = debug_mesh_info(moveable_mesh, "moveable")
            
            # 比较不同网格中相同面片ID的几何信息
            debug_info["face_comparison_orig_feat"] = compare_face_geometry(
                original_mesh, feature_mesh, "original", "feature", selected_face_ids
            )
            
            if has_moveable:
                debug_info["face_comparison_orig_move"] = compare_face_geometry(
                    original_mesh, moveable_mesh, "original", "moveable", selected_face_ids
                )
            
            # 保存调试信息
            debug_path = os.path.join(output_path, "debug_info.json")
            with open(debug_path, 'w') as f:
                json.dump(debug_info, f, indent=2)
            print(f"调试信息已保存至: {debug_path}")
            
            # 尝试使用不同的加载方式
            print("尝试使用不同的加载方式...")
            
            # 使用pymesh加载（如果可用）
            try:
                import pymesh
                pymesh_obj = pymesh.load_mesh(obj_path)
                pymesh_ply = pymesh.load_mesh(ply_path)
                
                # 比较面片顺序
                pymesh_debug = {
                    "pymesh_obj_faces": pymesh_obj.faces[:5].tolist(),
                    "pymesh_ply_faces": pymesh_ply.faces[:5].tolist(),
                    "trimesh_obj_faces": original_mesh.faces[:5].tolist(),
                    "trimesh_ply_faces": feature_mesh.faces[:5].tolist()
                }
                
                # 保存pymesh调试信息
                pymesh_debug_path = os.path.join(output_path, "pymesh_debug.json")
                with open(pymesh_debug_path, 'w') as f:
                    json.dump(pymesh_debug, f, indent=2)
                print(f"PyMesh调试信息已保存至: {pymesh_debug_path}")
                
            except ImportError:
                print("PyMesh不可用，跳过PyMesh加载测试")
            
            # 创建面片ID测试网格
            print("创建面片ID测试网格...")
            
            # 为每个面片分配唯一颜色（基于面片ID）
            id_colors_original = np.ones((len(original_mesh.faces), 4), dtype=np.uint8) * 255
            id_colors_feature = np.ones((len(feature_mesh.faces), 4), dtype=np.uint8) * 255
            
            # 使用面片ID的RGB编码作为颜色
            for i in range(min(len(original_mesh.faces), len(feature_mesh.faces))):
                # 将面片ID编码为RGB颜色
                r = (i & 0xFF0000) >> 16
                g = (i & 0x00FF00) >> 8
                b = i & 0x0000FF
                
                if i < len(original_mesh.faces):
                    id_colors_original[i, :3] = [r, g, b]
                
                if i < len(feature_mesh.faces):
                    id_colors_feature[i, :3] = [r, g, b]
            
            # 创建ID测试网格
            id_test_original = original_mesh.copy()
            id_test_original.visual.face_colors = id_colors_original
            id_test_feature = feature_mesh.copy()
            id_test_feature.visual.face_colors = id_colors_feature
            
            # 保存ID测试网格
            id_test_original_path = os.path.join(output_path, f"{model_id}_original_id_test.ply")
            id_test_feature_path = os.path.join(output_path, f"{model_id}_feature_id_test.ply")
            id_test_original.export(id_test_original_path)
            id_test_feature.export(id_test_feature_path)
            print(f"ID测试网格已保存")
        
        print(f"处理完成! 结果保存至: {output_path}")
        return True
        
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="批量测试不同格式几何体的面片ID对应关系")
    parser.add_argument("--id", type=str, help="指定要处理的模型ID")
    parser.add_argument("--debug", action="store_true", help="输出更多调试信息")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 查找结果文件夹
    result_folders = find_result_folders(args.id)
    if not result_folders:
        print(f"找不到任何结果文件夹")
        return
    
    print(f"找到 {len(result_folders)} 个结果文件夹")
    
    # 提取信息并分组
    model_infos = []
    for folder in result_folders:
        info = extract_info_from_result_folder(folder)
        if info:
            model_infos.append(info)
    
    # 按模型ID和关节ID排序
    model_infos.sort(key=lambda x: (x["model_id"], x["joint_id"]))
    
    # 处理所有模型
    success_count = 0
    total_count = len(model_infos)
    
    for i, info in enumerate(model_infos):
        print(f"\n[{i+1}/{total_count}] 处理: {info['category']}_{info['model_id']}_joint_{info['joint_id']}")
        if process_model(info, args.debug):
            success_count += 1
    
    print(f"\n批量处理完成! 成功: {success_count}/{total_count}")

if __name__ == "__main__":
    main()