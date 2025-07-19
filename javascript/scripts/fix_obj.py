import os
import sys
import glob
import numpy as np
import trimesh
from pathlib import Path
import shutil

# 类似于 common.py 中的类型定义
NumpyTensor = np.ndarray

def transform(matrix, points):
    """
    应用变换矩阵到点集
    
    Args:
        matrix: 4x4 变换矩阵
        points: Nx3 点集
        
    Returns:
        变换后的点集
    """
    if points.shape[1] == 3:
        # 将点扩展为齐次坐标
        points_homogeneous = np.ones((points.shape[0], 4))
        points_homogeneous[:, :3] = points
        
        # 应用变换
        transformed_points = np.dot(points_homogeneous, matrix.T)
        
        # 返回到3D坐标
        return transformed_points[:, :3]
    else:
        return np.dot(points, matrix[:3, :3].T) + matrix[:3, 3]

def handle_pose(pose):
    """
    处理姿态矩阵，确保它是有效的4x4矩阵
    
    Args:
        pose: 输入的姿态矩阵
        
    Returns:
        有效的4x4姿态矩阵
    """
    if pose is None:
        return np.eye(4)
    if pose.shape != (4, 4):
        new_pose = np.eye(4)
        new_pose[:pose.shape[0], :pose.shape[1]] = pose
        return new_pose
    return pose

def norm_mesh(mesh, center=True, scale=True):
    """
    归一化网格
    
    Args:
        mesh: 输入网格
        center: 是否将网格中心移到原点
        scale: 是否缩放网格到单位立方体
        
    Returns:
        归一化后的网格
    """
    if center:
        # 计算中心点并移动到原点
        centroid = mesh.vertices.mean(axis=0)
        mesh.vertices -= centroid
    
    if scale:
        # 缩放到单位立方体
        max_length = np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
        if max_length > 0:
            mesh.vertices /= max_length
    
    return mesh

def optimize_mesh(mesh, process=True, remove_duplicate_vertices=True, remove_unreferenced_vertices=True):
    """
    优化网格
    
    Args:
        mesh: 输入网格
        process: 是否进行处理
        remove_duplicate_vertices: 是否移除重复顶点
        remove_unreferenced_vertices: 是否移除未引用顶点
        
    Returns:
        优化后的网格
    """
    if process:
        # 创建新的网格以应用处理
        optimized_mesh = trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),
            process=True
        )
        
        # 移除重复顶点
        if remove_duplicate_vertices:
            optimized_mesh.merge_vertices()
        
        # 移除未引用顶点
        if remove_unreferenced_vertices:
            optimized_mesh.remove_unreferenced_vertices()
        
        return optimized_mesh
    
    return mesh

def load_and_fix_obj(obj_path, normalize=True, optimize=True):
    """
    加载并修复OBJ文件
    
    Args:
        obj_path: OBJ文件路径
        normalize: 是否归一化
        optimize: 是否优化
        
    Returns:
        修复后的网格
    """
    print(f"加载OBJ文件: {obj_path}")
    
    try:
        # 加载网格
        mesh = trimesh.load(obj_path)
        
        # 打印原始网格信息
        print(f"原始网格信息:")
        print(f"  顶点数: {len(mesh.vertices)}")
        print(f"  面片数: {len(mesh.faces)}")
        print(f"  边数: {len(mesh.edges)}")
        print(f"  包围盒大小: {mesh.bounds[1] - mesh.bounds[0]}")
        
        # 优化网格
        if optimize:
            print("正在优化网格...")
            mesh = optimize_mesh(mesh)
            print(f"优化后网格信息:")
            print(f"  顶点数: {len(mesh.vertices)}")
            print(f"  面片数: {len(mesh.faces)}")
            print(f"  边数: {len(mesh.edges)}")
            print(f"  包围盒大小: {mesh.bounds[1] - mesh.bounds[0]}")
        
        # 归一化网格
        if normalize:
            print("正在归一化网格...")
            mesh = norm_mesh(mesh)
            print(f"归一化后包围盒大小: {mesh.bounds[1] - mesh.bounds[0]}")
        
        return mesh
    
    except Exception as e:
        print(f"加载OBJ文件时出错: {e}")
        return None

def process_urdf_directory(urdf_id=None):
    """
    处理URDF目录中的OBJ文件
    
    Args:
        urdf_id: 特定的URDF ID，如果为None则处理所有URDF
    """
    base_path = "/home/ipab-graphics/workplace/Sketch_Singapo/javascript/urdf"
    
    if urdf_id:
        # 处理特定ID
        urdf_dirs = [os.path.join(base_path, urdf_id)]
    else:
        # 处理所有目录
        urdf_dirs = [os.path.join(base_path, d) for d in os.listdir(base_path) 
                    if os.path.isdir(os.path.join(base_path, d))]
    
    for urdf_dir in urdf_dirs:
        obj_path = os.path.join(urdf_dir, "yy_object", "yy_merged.obj")
        
        if not os.path.exists(obj_path):
            print(f"跳过 {urdf_dir}: 未找到 yy_merged.obj")
            continue
        
        print(f"\n处理 {obj_path}")
        
        # 修改这里：备份文件名改为 yy_merged_old.obj
        backup_path = os.path.join(os.path.dirname(obj_path), "yy_merged_old.obj")
        if not os.path.exists(backup_path):
            shutil.copy2(obj_path, backup_path)
            print(f"已备份原始文件到 {backup_path}")
        
        # 加载并修复网格
        mesh = load_and_fix_obj(obj_path)
        
        if mesh is not None:
            # 保存修复后的网格
            mesh.export(obj_path)
            print(f"已保存修复后的网格到 {obj_path}")
            
            # 计算优化比例
            original_mesh = trimesh.load(backup_path)
            vertex_reduction = 1 - (len(mesh.vertices) / len(original_mesh.vertices))
            face_reduction = 1 - (len(mesh.faces) / len(original_mesh.faces))
            
            print(f"优化结果:")
            print(f"  顶点减少: {vertex_reduction:.2%}")
            print(f"  面片减少: {face_reduction:.2%}")

def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 处理特定ID
        urdf_id = sys.argv[1]
        process_urdf_directory(urdf_id)
    else:
        # 处理所有URDF
        process_urdf_directory()

if __name__ == "__main__":
    main()
