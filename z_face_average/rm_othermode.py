#!/usr/bin/env python3
"""
删除非segmentation渲染模式或非center视角的图片脚本

此脚本会查找/hy-tmp/PartField_Sketch_simpleMLP/data_small/img/目录下的所有图片，
只保留渲染模式为'segmentation'且视角为'center'的图片，删除其他所有图片。

使用方法:
python rm_othermode.py [--dry-run] [--verbose]

参数:
  --dry-run   不实际删除文件，只打印要删除的文件列表
  --verbose   显示详细信息，包括保留的文件
"""

import os
import re
import argparse
from pathlib import Path

# 图片目录路径
IMG_DIR = "/hy-tmp/PartField_Sketch_simpleMLP/data_small/img"

def parse_filename(filename):
    """解析文件名，提取类别、ID、渲染模式、视角等信息"""
    # 匹配模式: {类别}_{id}_{render_mode}_{view}[_joint_{joint_id}].png
    pattern = r"([A-Za-z]+)_(\d+)_([a-z-]+)_([a-z_]+)(?:_joint_(\d+))?\.png"
    match = re.match(pattern, filename)
    
    if match:
        category = match.group(1)
        model_id = match.group(2)
        render_mode = match.group(3)
        view = match.group(4)
        joint_id = match.group(5)  # 可能为None
        
        return {
            "category": category,
            "model_id": model_id,
            "render_mode": render_mode,
            "view": view,
            "joint_id": joint_id
        }
    return None

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="删除非segmentation渲染模式或非center视角的图片")
    parser.add_argument("--dry-run", action="store_true", help="不实际删除文件，只打印要删除的文件列表")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息，包括保留的文件")
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.isdir(IMG_DIR):
        print(f"错误: 目录不存在: {IMG_DIR}")
        return
    
    # 统计信息
    total_files = 0
    to_delete = 0
    to_keep = 0
    
    # 删除原因统计
    wrong_render_mode = 0
    wrong_view = 0
    both_wrong = 0
    
    # 遍历目录中的所有文件
    for filename in os.listdir(IMG_DIR):
        if not filename.lower().endswith(".png"):
            continue
        
        total_files += 1
        file_path = os.path.join(IMG_DIR, filename)
        
        # 解析文件名
        info = parse_filename(filename)
        if not info:
            print(f"警告: 无法解析文件名: {filename}")
            continue
        
        # 检查渲染模式和视角
        is_correct_render_mode = info["render_mode"] == "segmentation"
        is_correct_view = info["view"] == "center"
        
        # 只有同时满足两个条件才保留
        if not (is_correct_render_mode and is_correct_view):
            to_delete += 1
            
            # 统计删除原因
            if not is_correct_render_mode and not is_correct_view:
                both_wrong += 1
                reason = f"渲染模式: {info['render_mode']}, 视角: {info['view']}"
            elif not is_correct_render_mode:
                wrong_render_mode += 1
                reason = f"渲染模式: {info['render_mode']}"
            else:
                wrong_view += 1
                reason = f"视角: {info['view']}"
            
            if args.dry_run or args.verbose:
                print(f"要删除: {filename} ({reason})")
            
            if not args.dry_run:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"删除文件时出错: {file_path}")
                    print(f"错误信息: {e}")
        else:
            to_keep += 1
            if args.verbose:
                print(f"保留: {filename}")
    
    # 打印统计信息
    print("\n统计信息:")
    print(f"总文件数: {total_files}")
    print(f"要删除的文件数: {to_delete}")
    print(f"  - 仅渲染模式不符: {wrong_render_mode}")
    print(f"  - 仅视角不符: {wrong_view}")
    print(f"  - 两者都不符: {both_wrong}")
    print(f"保留的文件数: {to_keep}")
    
    if args.dry_run:
        print("\n这是一次演习运行，没有实际删除任何文件。")
        print("要实际删除文件，请不使用--dry-run参数运行脚本。")

if __name__ == "__main__":
    main()
