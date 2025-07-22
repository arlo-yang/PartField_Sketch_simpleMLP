#!/usr/bin/env python
"""
网格简化脚本 - 使用PyMeshLab对网格进行多次迭代简化
将对/hy-tmp/virtual_scanner/data/new_obj/目录中的所有OBJ文件执行4次网格简化
"""

import os
import glob
from tqdm import tqdm
import pymeshlab
import gc

# 参数配置
INPUT_DIR = os.path.expanduser("~/workplace/User_web_dev/virtual_scanner/data/new_obj")  # 输入网格目录
OUTPUT_DIR = os.path.expanduser("~/workplace/User_web_dev/virtual_scanner/data/simplification")  # 简化后的输出目录
SIMPLIFY_STEPS = 4  # 简化次数
SIMPLIFY_PERCENTAGE = 0.5  # 每次简化的面数比例 (50%)
SIMPLIFY_QUALITY = 0.3  # 简化质量阈值

def simplify_mesh(obj_file):
    """使用PyMeshLab对网格进行多次迭代简化"""
    try:
        # 获取文件名（不含路径和扩展名）
        file_basename = os.path.basename(obj_file)
        file_name = os.path.splitext(file_basename)[0]
        
        # 创建MeshSet
        ms = pymeshlab.MeshSet()
        
        # 加载网格
        print(f"加载网格: {file_name}")
        ms.load_new_mesh(obj_file)
        
        # 获取原始面数
        original_faces = ms.current_mesh().face_number()
        print(f"原始网格面数: {original_faces}")
        
        # 构建简化输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 进行多次渐进式简化
        current_faces = original_faces
        for step in range(1, SIMPLIFY_STEPS + 1):
            # 计算目标面数
            target_faces = int(current_faces * SIMPLIFY_PERCENTAGE)
            
            print(f"简化网格步骤 {step}/{SIMPLIFY_STEPS}: 目标面数 {target_faces}")
            
            # 应用Quadric Edge Collapse Decimation简化
            # 参数设置与MeshLab界面中相同
            ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                           targetfacenum=target_faces,
                           qualitythr=SIMPLIFY_QUALITY,
                           preserveboundary=False,
                           preservenormal=True,
                           preservetopology=False,
                           optimalplacement=True,  # 最优顶点位置
                           planarquadric=False,
                           planarweight=0.00001,
                           qualityweight=False,
                           autoclean=True)  # 后处理清理
            
            # 更新当前面数
            current_faces = ms.current_mesh().face_number()
            print(f"简化后面数: {current_faces}")
            
            # 保存当前简化结果
            simplified_obj_path = os.path.join(OUTPUT_DIR, f"{file_name}_simplified_{step}.obj")
            ms.save_current_mesh(simplified_obj_path)
            print(f"保存简化网格 ({step}/{SIMPLIFY_STEPS}): {simplified_obj_path}")
            
            # 如果面数已经很少，提前结束
            if current_faces < 100:
                print(f"面数已经很少 ({current_faces})，提前结束简化")
                break
        
        # 清理内存
        del ms
        gc.collect()
        
        return True
    except Exception as e:
        print(f"简化失败: {obj_file}, 错误: {e}")
        return False

def main():
    """主函数，处理所有OBJ文件"""
    # 获取所有OBJ文件
    obj_files = glob.glob(os.path.join(INPUT_DIR, "*.obj"))
    
    if not obj_files:
        print(f"在 {INPUT_DIR} 目录中没有找到OBJ文件!")
        return
    
    print(f"找到 {len(obj_files)} 个OBJ文件，开始简化...")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 处理每个文件
    success_count = 0
    failed_count = 0
    
    for obj_file in tqdm(obj_files, desc="简化网格", unit="个"):
        if simplify_mesh(obj_file):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n处理完成: 成功简化 {success_count} 个模型，失败 {failed_count} 个")
    print(f"所有简化后的网格文件已保存到 {OUTPUT_DIR} 目录")

if __name__ == "__main__":
    main()
