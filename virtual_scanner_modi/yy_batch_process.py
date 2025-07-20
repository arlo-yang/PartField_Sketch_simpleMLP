#!/usr/bin/env python
"""批量处理3D模型文件并使用virtualscanner直接生成PLY格式点云"""

import os
import glob
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# 参数配置
INPUT_DIR = "data/original_obj"  # 输入模型目录
OUTPUT_DIR = "data/ply"          # 输出PLY目录
MESH_DIR = "data/new_obj"        # 重建网格输出目录
VIEW_NUM = 14                    # 视角数量
NORMALIZE = True                 # 是否归一化
WORKERS = 8                      # 线程数量
SCANNER_PATH = "./build/virtualscanner"  # virtualscanner程序路径
RECONSTRUCT_MESH = True          # 是否重建网格
POISSON_DEPTH = 8                # Poisson重建深度 (降低为8)

def process_model(model_path):
    """处理单个3D模型文件"""
    # 获取文件名（不含路径和扩展名）
    file_basename = os.path.basename(model_path)
    file_name = os.path.splitext(file_basename)[0]
    
    # 构建输出路径
    ply_path = os.path.join(OUTPUT_DIR, file_name + ".ply")
    
    # 构建命令
    cmd = [SCANNER_PATH, model_path, str(VIEW_NUM), "0", "1" if NORMALIZE else "0"]
    
    # 执行命令
    start_time = time.time()
    print(f"处理: {model_path}")
    subprocess.run(cmd)
    
    # 由于virtualscanner会在原始文件旁边生成PLY文件，需要移动到OUTPUT_DIR
    # 获取生成的PLY文件路径
    generated_ply = model_path.replace(".obj", ".ply").replace(".off", ".ply")
    
    # 如果生成成功，移动文件
    if os.path.exists(generated_ply):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # 如果目标文件已存在，先删除
        if os.path.exists(ply_path):
            os.remove(ply_path)
        os.rename(generated_ply, ply_path)
        
        elapsed = time.time() - start_time
        print(f"完成: {model_path} -> {ply_path} (耗时: {elapsed:.2f}秒)")
        return True
    else:
        print(f"失败: {model_path} (未找到生成的PLY文件)")
        return False

def reconstruct_mesh_pymeshlab(ply_file):
    """使用PyMeshLab重建网格"""
    try:
        import pymeshlab
        import gc
        
        # 获取文件名（不含路径和扩展名）
        file_basename = os.path.basename(ply_file)
        file_name = os.path.splitext(file_basename)[0]
        
        # 构建输出路径
        obj_path = os.path.join(MESH_DIR, file_name + ".obj")
        
        # 创建MeshSet
        ms = pymeshlab.MeshSet()
        
        # 加载点云
        print(f"重建网格: 加载点云 {file_name}")
        ms.load_new_mesh(ply_file)
        
        # 应用Screened Poisson Surface Reconstruction
        # 使用正确的函数名称和参数
        print(f"重建网格: 应用Screened Poisson算法...")
        ms.generate_surface_reconstruction_screened_poisson(
            depth=POISSON_DEPTH,
            fulldepth=POISSON_DEPTH+2,
            cgdepth=0,
            scale=1.1,
            samplespernode=1.5,
            pointweight=4,
            iters=8,
            confidence=False,
            preclean=False
        )
        
        # 保存结果
        os.makedirs(MESH_DIR, exist_ok=True)
        ms.save_current_mesh(obj_path)
        print(f"重建网格: 保存到 {obj_path}")
        
        # 清理内存
        del ms
        gc.collect()
        
        return True
    except ImportError:
        print("PyMeshLab未安装，尝试使用meshlabserver...")
        return reconstruct_mesh_meshlabserver(ply_file)
    except Exception as e:
        print(f"PyMeshLab重建失败: {e}")
        # 如果PyMeshLab失败，尝试Open3D
        print("尝试使用Open3D进行重建...")
        return reconstruct_mesh_open3d(ply_file)

def reconstruct_mesh_meshlabserver(ply_file):
    """使用meshlabserver重建网格"""
    # 获取文件名（不含路径和扩展名）
    file_basename = os.path.basename(ply_file)
    file_name = os.path.splitext(file_basename)[0]
    
    # 构建输出路径
    obj_path = os.path.join(MESH_DIR, file_name + ".obj")
    
    # 创建临时MeshLab脚本
    script_path = os.path.join(os.path.dirname(ply_file), f"{file_name}_poisson.mlx")
    
    # 写入MeshLab脚本内容
    with open(script_path, 'w') as f:
        f.write(f'''<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Surface Reconstruction: Screened Poisson">
  <Param name="visibleLayer" value="0" type="RichMesh"/>
  <Param name="depth" value="{POISSON_DEPTH}" type="RichInt"/>
  <Param name="fulldepth" value="{POISSON_DEPTH+2}" type="RichInt"/>
  <Param name="cgdepth" value="0" type="RichInt"/>
  <Param name="scale" value="1.1" type="RichFloat"/>
  <Param name="samplespernode" value="1.5" type="RichFloat"/>
  <Param name="pointweight" value="4" type="RichFloat"/>
  <Param name="iters" value="8" type="RichInt"/>
  <Param name="confidence" value="false" type="RichBool"/>
  <Param name="preclean" value="false" type="RichBool"/>
 </filter>
</FilterScript>''')
    
    # 确保输出目录存在
    os.makedirs(MESH_DIR, exist_ok=True)
    
    # 构建meshlabserver命令
    cmd = ['meshlabserver', 
           '-i', ply_file, 
           '-o', obj_path, 
           '-s', script_path, 
           '-om', 'vn', 'fn']
    
    try:
        # 执行命令
        print(f"重建网格: 使用meshlabserver处理 {ply_file}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 检查是否成功
        if os.path.exists(obj_path):
            print(f"重建网格: 保存到 {obj_path}")
            os.remove(script_path)  # 删除临时脚本
            return True
        else:
            print(f"meshlabserver失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"meshlabserver执行失败: {e}")
        return False

def reconstruct_mesh_open3d(ply_file):
    """使用Open3D重建网格 (备选方法)"""
    try:
        import open3d as o3d
        import numpy as np
        
        # 获取文件名（不含路径和扩展名）
        file_basename = os.path.basename(ply_file)
        file_name = os.path.splitext(file_basename)[0]
        
        # 构建输出路径
        obj_path = os.path.join(MESH_DIR, file_name + ".obj")
        
        # 读取点云
        print(f"重建网格 (Open3D): 加载点云 {ply_file}")
        pcd = o3d.io.read_point_cloud(ply_file)
        
        # 确保有法向量
        if not pcd.has_normals():
            print("重建网格 (Open3D): 估计法向量...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(100)
        
        # 重建网格
        print("重建网格 (Open3D): 应用Poisson重建...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=POISSON_DEPTH, scale=1.1, linear_fit=False)
        
        # 裁剪低密度区域
        print("重建网格 (Open3D): 裁剪低密度区域...")
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # 保存结果
        os.makedirs(MESH_DIR, exist_ok=True)
        o3d.io.write_triangle_mesh(obj_path, mesh)
        print(f"重建网格 (Open3D): 保存到 {obj_path}")
        
        return True
    except ImportError:
        print("Open3D未安装，无法使用此方法重建网格")
        return False
    except Exception as e:
        print(f"Open3D重建失败: {e}")
        return False

def main():
    """批量处理模型文件"""
    # 获取所有模型文件
    obj_files = glob.glob(f"{INPUT_DIR}/*.obj")
    off_files = glob.glob(f"{INPUT_DIR}/*.off")
    model_files = obj_files + off_files
    
    print(f"找到{len(model_files)}个模型文件，开始处理...")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 使用多进程处理
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        # 添加tqdm进度条
        results = list(tqdm(executor.map(process_model, model_files), 
                           total=len(model_files), 
                           desc="处理模型", 
                           unit="个"))
    
    # 统计结果
    success_count = results.count(True)
    print(f"\n处理完成: 成功{success_count}个，失败{len(model_files)-success_count}个")
    print(f"所有PLY文件已保存到 {OUTPUT_DIR} 目录")
    
    # 如果需要重建网格
    if RECONSTRUCT_MESH and success_count > 0:
        print("\n开始重建网格...")
        os.makedirs(MESH_DIR, exist_ok=True)
        
        # 获取所有生成的PLY文件
        ply_files = glob.glob(f"{OUTPUT_DIR}/*.ply")
        
        # 首先尝试使用PyMeshLab (最推荐)
        try:
            import pymeshlab
            print("使用PyMeshLab重建网格...")
            
            # 改为串行处理，避免内存问题
            mesh_count = 0
            for ply_file in tqdm(ply_files, desc="重建网格", unit="个"):
                if reconstruct_mesh_pymeshlab(ply_file):
                    mesh_count += 1
            
            print(f"网格重建完成: 成功{mesh_count}个，失败{len(ply_files)-mesh_count}个")
            print(f"所有重建的网格文件已保存到 {MESH_DIR} 目录")
            
        except ImportError:
            # 如果PyMeshLab不可用，尝试meshlabserver
            print("PyMeshLab不可用，尝试使用meshlabserver...")
            try:
                result = subprocess.run(['meshlabserver', '-h'], capture_output=True)
                if result.returncode == 0:
                    print("使用meshlabserver重建网格...")
                    # 添加tqdm进度条
                    mesh_count = 0
                    for ply_file in tqdm(ply_files, desc="重建网格", unit="个"):
                        if reconstruct_mesh_meshlabserver(ply_file):
                            mesh_count += 1
                    
                    print(f"网格重建完成: 成功{mesh_count}个，失败{len(ply_files)-mesh_count}个")
                    print(f"所有重建的网格文件已保存到 {MESH_DIR} 目录")
                else:
                    raise Exception("meshlabserver调用失败")
            except Exception as e:
                # 如果meshlabserver也不可用，尝试Open3D
                print(f"meshlabserver不可用 ({e})，尝试使用Open3D...")
                try:
                    import open3d
                    print("使用Open3D重建网格...")
                    mesh_count = 0
                    # 添加tqdm进度条
                    for ply_file in tqdm(ply_files, desc="重建网格", unit="个"):
                        if reconstruct_mesh_open3d(ply_file):
                            mesh_count += 1
                    
                    print(f"网格重建完成: 成功{mesh_count}个，失败{len(ply_files)-mesh_count}个")
                    print(f"所有重建的网格文件已保存到 {MESH_DIR} 目录")
                except ImportError:
                    print("无法找到合适的网格重建库，请安装PyMeshLab、MeshLab或Open3D")

if __name__ == "__main__":
    main() 