#!/usr/bin/env python
"""批量处理3D模型文件并使用virtualscanner直接生成PLY格式点云，支持面片ID"""

import os
import glob
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# 参数配置
INPUT_DIR = "data/original_obj"     # 输入模型目录
OUTPUT_DIR = "data/ply"             # 输出PLY目录
FACEID_DIR = "data/face_ids"        # 面片ID输出目录
VIEW_NUM = 14                       # 视角数量
NORMALIZE = True                    # 是否归一化
WORKERS = 8                         # 线程数量
SCANNER_PATH = "./build/virtualscanner"  # virtualscanner程序路径

def process_model(model_path):
    """处理单个3D模型文件"""
    # 获取文件名（不含路径和扩展名）
    file_basename = os.path.basename(model_path)
    file_name = os.path.splitext(file_basename)[0]
    
    # 构建输出路径
    ply_path = os.path.join(OUTPUT_DIR, file_name + ".ply")
    faceid_path = os.path.join(FACEID_DIR, file_name + "_face_ids.txt")
    
    # 构建命令
    cmd = [SCANNER_PATH, model_path, str(VIEW_NUM), "0", "1" if NORMALIZE else "0"]
    
    # 执行命令
    start_time = time.time()
    print(f"处理: {model_path}")
    subprocess.run(cmd)
    
    # 由于virtualscanner会在原始文件旁边生成PLY文件和面片ID文件，需要移动到指定目录
    # 获取生成的PLY文件路径和面片ID文件路径
    generated_ply = model_path.replace(".obj", ".ply").replace(".off", ".ply")
    generated_faceid = model_path.replace(".obj", "_face_ids.txt").replace(".off", "_face_ids.txt")
    
    # 如果生成成功，移动文件
    success = False
    if os.path.exists(generated_ply):
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(FACEID_DIR, exist_ok=True)
        
        # 如果目标文件已存在，先删除
        if os.path.exists(ply_path):
            os.remove(ply_path)
        os.rename(generated_ply, ply_path)
        success = True
        
        # 处理面片ID文件
        if os.path.exists(generated_faceid):
            if os.path.exists(faceid_path):
                os.remove(faceid_path)
            os.rename(generated_faceid, faceid_path)
        
        elapsed = time.time() - start_time
        print(f"完成: {model_path} -> {ply_path} (耗时: {elapsed:.2f}秒)")
        if os.path.exists(faceid_path):
            print(f"生成面片ID文件: {faceid_path}")
    else:
        print(f"失败: {model_path} (未找到生成的PLY文件)")
    
    return {
        "success": success,
        "model": model_path,
        "ply": ply_path if success else None,
        "faceid": faceid_path if success and os.path.exists(faceid_path) else None
    }

def main():
    """批量处理模型文件"""
    # 检查扫描器是否存在
    if not os.path.exists(SCANNER_PATH):
        print(f"错误: 找不到virtualscanner程序: {SCANNER_PATH}")
        print("请先运行 ./build.sh 编译程序")
        return
    
    # 获取所有模型文件
    obj_files = glob.glob(f"{INPUT_DIR}/*.obj")
    off_files = glob.glob(f"{INPUT_DIR}/*.off")
    model_files = obj_files + off_files
    
    if not model_files:
        print(f"错误: 在 {INPUT_DIR} 中找不到obj或off文件")
        return
    
    print(f"找到{len(model_files)}个模型文件，开始处理...")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FACEID_DIR, exist_ok=True)
    
    # 使用多进程处理
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        # 添加tqdm进度条
        results = list(tqdm(executor.map(process_model, model_files), 
                           total=len(model_files), 
                           desc="处理模型", 
                           unit="个"))
    
    # 统计结果
    success_count = sum(1 for r in results if r["success"])
    print(f"\n处理完成: 成功{success_count}个，失败{len(model_files)-success_count}个")
    print(f"所有PLY文件已保存到 {OUTPUT_DIR} 目录")
    print(f"所有面片ID文件已保存到 {FACEID_DIR} 目录")

if __name__ == "__main__":
    main() 