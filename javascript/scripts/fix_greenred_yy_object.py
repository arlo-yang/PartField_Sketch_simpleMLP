#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import shutil
import numpy as np
import trimesh
from tqdm import tqdm


def process_object_dir(object_dir):
    """
    处理单个对象目录，修复绿红色可视化文件
    
    1. 找到yy_merged.obj和moveable_{joint_id}.ply
    2. 复制yy_merged.obj到visualization目录
    3. 将PLY中的颜色信息应用到复制的OBJ上
    4. 删除原始PLY和NPY文件
    5. 保存绿色面片ID到TXT文件
    """
    print(f"\n处理目录: {object_dir}")
    
    # 定位需要的文件
    yy_object_dir = os.path.join(object_dir, "yy_object")
    vis_dir = os.path.join(yy_object_dir, "yy_visualization")
    merged_obj_path = os.path.join(yy_object_dir, "yy_merged.obj")
    
    if not os.path.exists(merged_obj_path):
        print(f"错误: 未找到合并OBJ: {merged_obj_path}")
        return False
    
    if not os.path.exists(vis_dir):
        print(f"错误: 未找到可视化目录: {vis_dir}")
        return False
    
    # 查找所有moveable_{joint_id}.ply文件
    ply_files = glob.glob(os.path.join(vis_dir, "moveable_*.ply"))
    
    if not ply_files:
        print(f"错误: 未找到可视化PLY文件")
        return False
    
    # 处理每个PLY文件
    for ply_path in ply_files:
        # 提取关节ID
        joint_id = os.path.basename(ply_path).replace("moveable_", "").replace(".ply", "")
        print(f"处理关节ID: {joint_id}")
        
        # 对应的NPY文件
        npy_path = os.path.join(vis_dir, f"labels_{joint_id}.npy")
        
        try:
            # 加载原始的PLY文件以获取颜色信息
            ply_mesh = trimesh.load(ply_path)
            print(f"已加载PLY文件，包含 {len(ply_mesh.faces)} 个面片")
            
            # 检查是否有面片颜色
            if not hasattr(ply_mesh, 'visual') or not hasattr(ply_mesh.visual, 'face_colors'):
                print(f"错误: PLY文件没有面片颜色")
                continue
            
            face_colors = ply_mesh.visual.face_colors
            print(f"读取到 {len(face_colors)} 个面片颜色")
            
            # 识别绿色面片
            green_faces = []
            red_faces = []
            
            # 检测绿色和红色面片
            for i, color in enumerate(face_colors):
                if color[0] < 50 and color[1] > 200 and color[2] < 50:  # 绿色
                    green_faces.append(i)
                elif color[0] > 200 and color[1] < 50 and color[2] < 50:  # 红色
                    red_faces.append(i)
            
            print(f"发现 {len(green_faces)} 个绿色面片, {len(red_faces)} 个红色面片")
            
            # 加载原始的yy_merged.obj
            merged_mesh = trimesh.load(merged_obj_path)
            print(f"已加载原始OBJ，包含 {len(merged_mesh.faces)} 个面片")
            
            # 检查面片数量是否一致
            if len(merged_mesh.faces) != len(face_colors):
                print(f"警告: 面片数量不匹配 - OBJ: {len(merged_mesh.faces)}, PLY: {len(face_colors)}")
                
                # 如果数量不同，尝试使用质心匹配
                print("尝试使用质心匹配...")
                
                # 准备新的面片颜色数组
                new_face_colors = np.zeros((len(merged_mesh.faces), 4), dtype=np.uint8)
                new_face_colors[:] = [255, 0, 0, 255]  # 默认红色
                
                # 计算OBJ的面片质心
                obj_centroids = np.mean(merged_mesh.vertices[merged_mesh.faces], axis=1)
                
                # 计算PLY的绿色面片质心
                ply_green_centroids = np.mean(ply_mesh.vertices[ply_mesh.faces[green_faces]], axis=1)
                
                # 对每个绿色面片质心，找到OBJ中最近的面片
                green_faces_in_obj = []
                for green_centroid in ply_green_centroids:
                    distances = np.linalg.norm(obj_centroids - green_centroid, axis=1)
                    closest_face_idx = np.argmin(distances)
                    min_distance = distances[closest_face_idx]
                    
                    if min_distance < 0.05:  # 使用一个阈值
                        green_faces_in_obj.append(closest_face_idx)
                        new_face_colors[closest_face_idx] = [0, 255, 0, 255]  # 绿色
                
                print(f"质心匹配后找到 {len(green_faces_in_obj)} 个绿色面片")
                green_faces = green_faces_in_obj
                
            else:
                # 如果面片数量一致，直接复制颜色
                new_face_colors = face_colors.copy()
            
            # 创建新的带有颜色的网格，保持原始结构不变
            colored_mesh = trimesh.Trimesh(
                vertices=merged_mesh.vertices.copy(),
                faces=merged_mesh.faces.copy(),
                face_colors=new_face_colors
            )
            
            # 保存为新的PLY文件
            new_ply_path = os.path.join(vis_dir, f"moveable_{joint_id}_fixed.ply")
            colored_mesh.export(new_ply_path)
            print(f"已保存新的PLY文件: {new_ply_path}")
            
            # 保存绿色面片ID到TXT文件
            green_faces_path = os.path.join(vis_dir, f"moveable_ids_{joint_id}.txt")
            with open(green_faces_path, "w") as f:
                for face_id in sorted(green_faces):
                    f.write(f"{face_id}\n")
            print(f"已保存绿色面片ID到: {green_faces_path}")
            
            # 删除旧文件
            try:
                os.remove(ply_path)
                print(f"已删除原始PLY文件: {ply_path}")
                if os.path.exists(npy_path):
                    os.remove(npy_path)
                    print(f"已删除原始NPY文件: {npy_path}")
                
                # 重命名新文件为原来的名称
                os.rename(new_ply_path, ply_path)
                print(f"已将新PLY文件重命名为: {ply_path}")
            except Exception as e:
                print(f"删除或重命名文件时出错: {e}")
            
        except Exception as e:
            print(f"处理PLY文件时出错: {e}")
            continue
    
    return True


def main():
    """
    遍历所有对象目录并处理
    """
    # 获取根目录
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "/hy-tmp/PartField_Sketch_simpleMLP/javascript/urdf"
    
    print(f"处理根目录: {root_dir}")
    
    # 查找所有数字ID文件夹
    object_dirs = []
    for item in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, item)
        if os.path.isdir(dir_path) and item.isdigit():
            object_dirs.append(dir_path)
    
    object_dirs.sort(key=lambda x: int(os.path.basename(x)))
    print(f"找到 {len(object_dirs)} 个对象目录")
    
    # 如果有命令行参数指定ID，则只处理该ID
    if len(sys.argv) > 2:
        specified_id = sys.argv[2]
        for dir_path in object_dirs:
            if os.path.basename(dir_path) == specified_id:
                process_object_dir(dir_path)
                return
        print(f"未找到ID为 {specified_id} 的目录")
        return
    
    # 处理所有目录
    success_count = 0
    for dir_path in tqdm(object_dirs, desc="处理对象"):
        try:
            if process_object_dir(dir_path):
                success_count += 1
        except Exception as e:
            print(f"处理目录 {dir_path} 时出错: {e}")
    
    print(f"处理完成！成功处理 {success_count}/{len(object_dirs)} 个对象目录")


if __name__ == "__main__":
    main()