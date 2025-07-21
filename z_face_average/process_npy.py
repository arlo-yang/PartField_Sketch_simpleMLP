"""
Step 1：找到初步的候选面片集合 S
- 通过 back-projection：
    - 从 mesh 渲染出 face ID map；
    - 看哪些 face 的投影 pixel 落在 mask M 中；
    - 这些面组成集合 S：可动部分 mask 覆盖的可见面片集合。

Step 2：计算候选面的平均特征 Fₘ
公式：
    F_m = (1 / |S|) * Σ_i∈S F_i
- 拿 S 中每个面片的特征 F_i，取均值；
- 得到代表 mask 区域可动部分的"特征中心"。

Step 3：对所有 mesh 的面片做特征距离分类
- 对每个 face i，算与 Fₘ 的欧式距离：
    || F_i - F_m ||^2
- 判断规则：
    如果面片 i 的特征离 Fₘ 足够近，就归为 movable part。
- 具体逻辑：
    || F_i - F_m ||^2 ≤ max_{j ∈ S} || F_j - F_m ||^2
- 不超过这个阈值的面都归入 movable part，超出的归入 base part。

使用方法:
1. 处理单个结果:
   python process_npy.py --result_path /hy-tmp/PartField_Sketch_simpleMLP/data_small/result/类别_ID_segmentation_视角_joint_关节ID

2. 批量处理所有结果:
   python process_npy.py --batch
"""

import os
import sys
import numpy as np
import trimesh
import argparse
from pathlib import Path
import re
import shutil
import glob
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

# 全局路径配置
URDF_DIR = "/hy-tmp/PartField_Sketch_simpleMLP/data_small/urdf"
RESULT_DIR = "/hy-tmp/PartField_Sketch_simpleMLP/data_small/result"
OUTPUT_DIR = "/hy-tmp/PartField_Sketch_simpleMLP/data_small/feature_result_npy"
IMG_DIR = "/hy-tmp/PartField_Sketch_simpleMLP/data_small/img"

def parse_result_path(result_path):
    """从结果路径解析信息"""
    pattern = r"([A-Za-z]+)_(\d+)_segmentation_([a-z_]+)_joint_(\d+)"
    match = re.search(pattern, result_path)
    if match:
        category = match.group(1)
        model_id = match.group(2)
        view = match.group(3)
        joint_id = match.group(4)
        return category, model_id, view, joint_id
    else:
        raise ValueError(f"无法从路径解析信息: {result_path}")

def load_face_ids(face_ids_path):
    """加载面片ID文件"""
    face_ids = []
    with open(face_ids_path, 'r') as f:
        for line in f:
            face_ids.append(int(line.strip()))
    return face_ids

def compute_feature_center(features, face_ids):
    """Calculate feature center for the candidate face set"""
    # Ensure face_ids are within valid range
    valid_face_ids = [fid for fid in face_ids if fid < features.shape[0]]
    if not valid_face_ids:
        raise ValueError("All face IDs are out of feature range")
    
    # Extract features for these faces and calculate the mean
    selected_features = features[valid_face_ids]
    feature_center = np.mean(selected_features, axis=0)
    return feature_center

def calculate_distances(features, feature_center):
    # Use Euclidean distance to calculate feature differences
    distances = np.sum((features - feature_center) ** 2, axis=1)
    return distances

def classify_faces(distances, face_ids):
    # Calculate threshold: use max value of distances in the candidate set
    candidate_distances = distances[face_ids]
    threshold = np.max(candidate_distances)
    
    # Classify all faces
    movable_mask = distances <= threshold
    movable_face_ids = np.where(movable_mask)[0]
    
    return movable_face_ids.tolist(), threshold

def colorize_mesh(mesh, movable_face_ids):
    """将可动部件面片标为红色，非可动部件标为白色"""
    # 创建统一的颜色数组 - 先全部设为白色
    face_colors = np.ones((len(mesh.faces), 4), dtype=np.uint8) * 255
    
    # 将可动部件面片设为红色
    mask = np.isin(np.arange(len(mesh.faces)), movable_face_ids)
    face_colors[mask] = [255, 0, 0, 255]  # 红色
    
    # 设置面片颜色
    mesh.visual.face_colors = face_colors
    return mesh

def process_single_result(result_path):
    """处理单个结果目录"""
    # 解析路径信息
    category, model_id, view, joint_id = parse_result_path(result_path)
    print(f"处理: 类别={category}, ID={model_id}, 视角={view}, 关节ID={joint_id}")
    
    # 构建文件路径
    face_ids_path = os.path.join(result_path, "pred_face_ids.txt")
    npy_path = os.path.join(URDF_DIR, model_id, "feature", f"{model_id}.npy")
    obj_path = os.path.join(URDF_DIR, model_id, "yy_merged.obj")
    
    # 检查文件是否存在
    for path, desc in [(face_ids_path, "面片ID文件"), 
                       (npy_path, "NPY特征文件"),
                       (obj_path, "OBJ文件")]:
        if not os.path.exists(path):
            print(f"错误：{desc}不存在: {path}")
            return False
    
    # 加载数据
    print("加载初始面片ID集合...")
    initial_face_ids = load_face_ids(face_ids_path)
    print(f"- 加载了 {len(initial_face_ids)} 个候选面片")
    
    print("加载NPY特征数据...")
    features = np.load(npy_path)
    print(f"- 特征维度: {features.shape}")
    
    # 加载几何体
    print("加载几何体...")
    obj_mesh = trimesh.load(obj_path, force='mesh')
    print(f"- OBJ面片数: {len(obj_mesh.faces)}")
    print(f"- OBJ顶点数: {len(obj_mesh.vertices)}")
    
    # 检查面片数量是否匹配
    if features.shape[0] != len(obj_mesh.faces):
        print(f"警告：NPY特征数量 ({features.shape[0]}) 与OBJ文件面片数 ({len(obj_mesh.faces)}) 不匹配!")
    
    # 计算特征中心
    print("计算候选集特征中心...")
    feature_center = compute_feature_center(features, initial_face_ids)
    
    # 计算特征距离并分类
    print("计算特征距离并分类...")
    distances = calculate_distances(features, feature_center)
    movable_face_ids, threshold = classify_faces(distances, initial_face_ids)
    print(f"- 初始候选面片数: {len(initial_face_ids)}")
    print(f"- 最终可动部件面片数: {len(movable_face_ids)}")
    print(f"- 使用的距离阈值: {threshold:.6f} (候选集平均距离)")
    
    # 特征距离统计
    initial_distances = distances[initial_face_ids]
    print(f"候选集距离统计: 最小={np.min(initial_distances):.6f}, "
          f"平均={np.mean(initial_distances):.6f}, "
          f"最大={np.max(initial_distances):.6f}")
    
    # 对几何体着色
    print("对几何体着色...")
    colored_obj = colorize_mesh(obj_mesh.copy(), movable_face_ids)
    
    # 创建输出目录
    output_subdir = f"{category}_{model_id}_segmentation_{view}_joint_{joint_id}_npy"
    output_path = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(output_path, exist_ok=True)
    
    # 保存结果
    print("保存结果...")
    obj_output = os.path.join(output_path, f"{model_id}_mesh_colored.ply")
    face_ids_output = os.path.join(output_path, "movable_face_ids.txt")
    feature_center_output = os.path.join(output_path, "feature_center.npy")
    
    colored_obj.export(obj_output)
    
    # 保存可动面片ID
    with open(face_ids_output, 'w') as f:
        for face_id in movable_face_ids:
            f.write(f"{face_id}\n")
    
    # 保存特征中心
    np.save(feature_center_output, feature_center)
    
    # 复制原始面片ID文件以供参考
    shutil.copy2(face_ids_path, os.path.join(output_path, "initial_face_ids.txt"))
    
    print(f"处理完成! 结果保存至: {output_path}")
    return True

def find_all_result_paths():
    """查找所有结果路径"""
    # 查找所有segmentation图片
    pattern = os.path.join(IMG_DIR, "*_segmentation_*_joint_*.png")
    img_files = glob.glob(pattern)
    
    result_paths = []
    for img_file in img_files:
        # 从图片文件名解析信息
        filename = os.path.basename(img_file)
        match = re.match(r"([A-Za-z]+)_(\d+)_segmentation_([a-z_]+)_joint_(\d+)\.png", filename)
        if match:
            category = match.group(1)
            model_id = match.group(2)
            view = match.group(3)
            joint_id = match.group(4)
            
            # 构建对应的结果路径
            result_path = os.path.join(RESULT_DIR, f"{category}_{model_id}_segmentation_{view}_joint_{joint_id}")
            
            # 检查结果目录是否存在
            if os.path.isdir(result_path) and os.path.exists(os.path.join(result_path, "pred_face_ids.txt")):
                # 检查NPY文件是否存在
                npy_path = os.path.join(URDF_DIR, model_id, "feature", f"{model_id}.npy")
                if os.path.exists(npy_path):
                    result_paths.append(result_path)
    
    return result_paths

def batch_process():
    """批量处理所有结果"""
    result_paths = find_all_result_paths()
    print(f"找到 {len(result_paths)} 个结果目录待处理")
    
    success_count = 0
    failure_count = 0
    
    for i, result_path in enumerate(tqdm(result_paths, desc="处理进度")):
        print(f"\n[{i+1}/{len(result_paths)}] 处理: {result_path}")
        try:
            if process_single_result(result_path):
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            print(f"处理失败: {e}")
            failure_count += 1
    
    print(f"\n批量处理完成! 成功: {success_count}, 失败: {failure_count}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="基于特征距离的面片分类")
    parser.add_argument("--result_path", type=str, help="面片ID结果路径，例如 /hy-tmp/.../类别_ID_segmentation_视角_joint_关节ID")
    parser.add_argument("--batch", action="store_true", help="批量处理所有结果")
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if args.batch:
        batch_process()
    elif args.result_path:
        process_single_result(args.result_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
