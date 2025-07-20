#!/usr/bin/env python
"""
复制模型脚本 - 从Sketch2Art_data_web目录下各个ID文件夹中复制yy_merged.obj文件
到virtual_scanner/data/original_obj/目录中
"""

import os
import shutil
import glob
from tqdm import tqdm

# 源目录和目标目录
SOURCE_BASE_DIR = "/hy-tmp/Sketch2Art_data_web/javascript/urdf"
TARGET_DIR = "/hy-tmp/virtual_scanner/data/original_obj"

def copy_models():
    """复制所有模型文件"""
    # 确保目标目录存在
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # 获取所有ID文件夹
    id_folders = [f for f in os.listdir(SOURCE_BASE_DIR) if os.path.isdir(os.path.join(SOURCE_BASE_DIR, f))]
    
    print(f"找到 {len(id_folders)} 个ID文件夹，开始复制模型...")
    
    # 计数器
    success_count = 0
    skipped_count = 0
    
    # 遍历每个ID文件夹
    for folder_id in tqdm(id_folders, desc="复制模型"):
        # 构建源文件路径
        source_path = os.path.join(SOURCE_BASE_DIR, folder_id, "yy_object", "yy_merged.obj")
        
        # 构建目标文件路径（使用ID作为文件名）
        target_path = os.path.join(TARGET_DIR, f"{folder_id}.obj")
        
        # 检查源文件是否存在
        if os.path.exists(source_path):
            try:
                # 复制文件
                shutil.copy2(source_path, target_path)
                success_count += 1
            except Exception as e:
                print(f"复制 {folder_id} 失败: {e}")
        else:
            skipped_count += 1
            # 打印调试信息
            if not os.path.exists(os.path.join(SOURCE_BASE_DIR, folder_id, "yy_object")):
                print(f"跳过 {folder_id}: 未找到 yy_object 文件夹")
            else:
                print(f"跳过 {folder_id}: 未找到 yy_merged.obj 文件")
    
    print(f"\n复制完成: 成功复制 {success_count} 个文件，跳过 {skipped_count} 个")
    print(f"所有模型已保存到 {TARGET_DIR} 目录")

if __name__ == "__main__":
    copy_models()
