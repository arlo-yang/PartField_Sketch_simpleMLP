import os
import sys
import argparse
import math
import numpy as np
import trimesh
from pathlib import Path
import time

# 导入预设视角定义
# 定义标准视角位置
distance = 2.5
VIEW_POSITIONS = {
    # 右侧视角系列
    "bottom-right": {
        "position": [-distance * math.cos(math.pi/4) * math.sin(math.pi/4), distance * math.sin(math.pi/4), distance * math.cos(math.pi/4) * math.cos(math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "left": {
        "position": [-distance * math.cos(0) * math.sin(math.pi/4), distance * math.sin(0), distance * math.cos(0) * math.cos(math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "top-left": {
        "position": [-distance * math.cos(-math.pi/4) * math.sin(math.pi/4), distance * math.sin(-math.pi/4), distance * math.cos(-math.pi/4) * math.cos(math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    
    # 中间视角系列
    "bottom-center": {
        "position": [-distance * math.cos(math.pi/4) * math.sin(0), distance * math.sin(math.pi/4), distance * math.cos(math.pi/4) * math.cos(0)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "center": {
        "position": [0.0, 0.0, distance],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "top-center": {
        "position": [-distance * math.cos(-math.pi/4) * math.sin(0), distance * math.sin(-math.pi/4), distance * math.cos(-math.pi/4) * math.cos(0)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    
    # 左侧视角系列
    "bottom-right": {
        "position": [-distance * math.cos(math.pi/4) * math.sin(-math.pi/4), distance * math.sin(math.pi/4), distance * math.cos(math.pi/4) * math.cos(-math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "right": {
        "position": [-distance * math.cos(0) * math.sin(-math.pi/4), distance * math.sin(0), distance * math.cos(0) * math.cos(-math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "top-right": {
        "position": [-distance * math.cos(-math.pi/4) * math.sin(-math.pi/4), distance * math.sin(-math.pi/4), distance * math.cos(-math.pi/4) * math.cos(-math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    }
}

def compute_rotation_matrix(from_view, to_view):
    """
    计算从一个视角到另一个视角的旋转矩阵
    为了模拟"物体旋转"效果，我们计算相机变换的逆变换
    """
    # 从from_view的位置创建相机坐标系
    from_pos = np.array(from_view["position"], dtype=np.float32)
    from_target = np.array(from_view["target"], dtype=np.float32)
    from_up = np.array(from_view["up"], dtype=np.float32)
    
    # 计算from_view的相机坐标轴
    from_forward = from_target - from_pos
    from_forward = from_forward / np.linalg.norm(from_forward)
    
    from_right = np.cross(from_up, from_forward)
    from_right = from_right / np.linalg.norm(from_right)
    
    from_up = np.cross(from_forward, from_right)
    from_up = from_up / np.linalg.norm(from_up)
    
    # 从to_view的位置创建相机坐标系
    to_pos = np.array(to_view["position"], dtype=np.float32)
    to_target = np.array(to_view["target"], dtype=np.float32)
    to_up = np.array(to_view["up"], dtype=np.float32)
    
    # 计算to_view的相机坐标轴
    to_forward = to_target - to_pos
    to_forward = to_forward / np.linalg.norm(to_forward)
    
    to_right = np.cross(to_up, to_forward)
    to_right = to_right / np.linalg.norm(to_right)
    
    to_up = np.cross(to_forward, to_right)
    to_up = to_up / np.linalg.norm(to_up)
    
    # 构建两个相机的旋转矩阵
    from_rotation = np.identity(4)
    from_rotation[:3, 0] = from_right
    from_rotation[:3, 1] = from_up
    from_rotation[:3, 2] = from_forward
    
    to_rotation = np.identity(4)
    to_rotation[:3, 0] = to_right
    to_rotation[:3, 1] = to_up
    to_rotation[:3, 2] = to_forward
    
    # 计算从from_view到to_view的旋转矩阵
    # 为了模拟物体旋转效果，需要计算相机变换的逆变换
    # R_from^-1 * R_to 而不是 R_to * R_from^-1
    rotation = np.dot(np.linalg.inv(to_rotation), from_rotation)
    
    return rotation

def rotate_mesh(mesh, rotation_matrix):
    """
    根据旋转矩阵旋转网格
    
    Args:
        mesh: 输入的Trimesh对象
        rotation_matrix: 4x4旋转矩阵
        
    Returns:
        旋转后的Trimesh对象
    """
    # 创建网格的副本
    rotated_mesh = mesh.copy()
    
    # 应用旋转变换（只取旋转部分，忽略平移）
    rotation = rotation_matrix[:3, :3]
    rotated_mesh.vertices = np.dot(rotated_mesh.vertices, rotation.T)
    
    # 更新法线
    if rotated_mesh.faces is not None and len(rotated_mesh.faces) > 0:
        rotated_mesh.face_normals = None  # 清除现有的法线
        rotated_mesh._cache.clear()  # 清除缓存
    
    return rotated_mesh

def process_model_for_all_views(model_id, base_dir=None):
    """
    处理指定ID的模型，从所有视角进行旋转并保存
    
    Args:
        model_id: 模型ID
        base_dir: 基础目录路径，默认为当前工作目录
    """
    if base_dir is None:
        base_dir = os.getcwd()
        
    base_path = Path(base_dir)
    
    # 直接使用ID.obj文件
    input_path = base_path / "urdf" / str(model_id) / "yy_object" / f"{model_id}.obj"
    
    if not input_path.exists():
        print(f"错误: 找不到输入文件 {input_path}")
        return False
    
    # 构建输出目录
    output_dir = base_path / "urdf" / str(model_id) / "angle_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print(f"加载模型: {input_path}")
    try:
        mesh = trimesh.load(str(input_path))
        if mesh is None:
            print(f"错误: 无法加载模型 {input_path}")
            return False
    except Exception as e:
        print(f"错误: 加载模型失败 {input_path}: {str(e)}")
        return False
    
    # 假设模型默认是从中心视角(center)观察的
    default_view = VIEW_POSITIONS["center"]
    
    # 对每个视角进行处理
    for view_name, view_position in VIEW_POSITIONS.items():
        # 计算旋转矩阵
        rotation_matrix = compute_rotation_matrix(default_view, view_position)
        
        # 旋转模型
        rotated_mesh = rotate_mesh(mesh, rotation_matrix)
        if rotated_mesh is None:
            print(f"警告: 旋转后的模型为空，跳过保存 {view_name} 视角")
            continue
            
        # 构建输出文件路径
        output_filename = f"{model_id}_{view_name}.obj"
        output_path = output_dir / output_filename
        
        # 保存旋转后的模型
        try:
            rotated_mesh.export(str(output_path))
            print(f"已保存 {view_name} 视角模型: {output_path}")
        except Exception as e:
            print(f"警告: 保存 {view_name} 视角模型失败: {str(e)}")
    
    return True

def scan_urdf_directory(base_dir=None):
    """
    扫描URDF目录获取所有ID
    
    Args:
        base_dir: 基础目录路径，默认为当前工作目录
        
    Returns:
        ID列表
    """
    if base_dir is None:
        base_dir = os.getcwd()
        
    urdf_dir = os.path.join(base_dir, "urdf")
    
    if not os.path.isdir(urdf_dir):
        print(f"错误: URDF目录不存在 - {urdf_dir}")
        return []
        
    # 获取所有ID目录
    ids = []
    for item in os.listdir(urdf_dir):
        item_path = os.path.join(urdf_dir, item)
        if os.path.isdir(item_path):
            # 只检查是否存在ID.obj文件
            id_obj_path = os.path.join(item_path, "yy_object", f"{item}.obj")
            
            if os.path.isfile(id_obj_path):
                ids.append(item)
    
    return sorted(ids)

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="根据指定ID或自动扫描目录旋转3D网格模型，从所有视角生成几何体")
    
    # 添加参数
    parser.add_argument("--base-dir", type=str, default=None,
                        help="基础目录路径，默认为当前工作目录")
    parser.add_argument("--scan-all", action="store_true",
                        help="扫描URDF目录并处理所有ID")
    
    # 可选的ID参数
    id_group = parser.add_mutually_exclusive_group()
    id_group.add_argument("--id", type=str, help="要处理的单个模型ID")
    id_group.add_argument("--ids", type=str, nargs="+", help="要处理的多个模型ID列表")
    id_group.add_argument("--id-file", type=str, help="包含模型ID列表的文件，每行一个ID")
    
    args = parser.parse_args()
    
    # 确定基础目录
    base_dir = args.base_dir if args.base_dir else os.getcwd()
    
    # 确定要处理的模型ID列表
    model_ids = []
    
    if args.scan_all:
        # 自动扫描所有ID
        print("扫描URDF目录中的所有ID...")
        model_ids = scan_urdf_directory(base_dir)
        if not model_ids:
            print("未找到任何有效的ID！")
            sys.exit(1)
        print(f"找到 {len(model_ids)} 个有效ID")
    elif args.id:
        # 处理单个ID
        model_ids = [args.id]
    elif args.ids:
        # 处理多个ID
        model_ids = args.ids
    elif args.id_file:
        # 从文件读取ID列表
        try:
            with open(args.id_file, 'r') as f:
                model_ids = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"错误: 无法读取ID文件 {args.id_file}: {str(e)}")
            sys.exit(1)
    else:
        # 没有提供任何ID信息，默认扫描所有ID
        print("未指定ID，默认扫描URDF目录中的所有ID...")
        model_ids = scan_urdf_directory(base_dir)
        if not model_ids:
            print("未找到任何有效的ID！")
            sys.exit(1)
        print(f"找到 {len(model_ids)} 个有效ID")
    
    # 处理所有模型
    successful_ids = []
    failed_ids = []
    
    start_time = time.time()
    total_count = len(model_ids)
    
    for idx, model_id in enumerate(model_ids, 1):
        print(f"\n处理模型 [{idx}/{total_count}] ID: {model_id}")
        success = process_model_for_all_views(model_id, base_dir)
        if success:
            successful_ids.append(model_id)
        else:
            failed_ids.append(model_id)
            
        # 显示进度
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = avg_time * (total_count - idx)
        print(f"进度: {idx}/{total_count} ({idx/total_count*100:.1f}%) - 预计剩余时间: {int(remaining//60)}分{int(remaining%60)}秒")
    
    # 打印处理结果
    print("\n处理完成!")
    print(f"成功: {len(successful_ids)} 个模型")
    print(f"失败: {len(failed_ids)} 个模型")
    
    if failed_ids:
        print("\n失败的模型ID:")
        for failed_id in failed_ids:
            print(f"  - {failed_id}")
    
    total_time = time.time() - start_time
    print(f"\n总耗时: {int(total_time//60)}分{int(total_time%60)}秒")

if __name__ == "__main__":
    main()
