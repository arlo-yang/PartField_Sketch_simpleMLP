#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
URDF文件处理脚本
===============================
该脚本用于清理URDF目录结构，只保留必要的文件和文件夹。

保留的内容:
- mobility.urdf
- meta.json
- yy_merged.obj (在yy_object目录下)
- yy_visualization/ (文件夹)
- textured_objs/ (文件夹)

使用方法:
    python urdf_file_preocess.py [--dry-run] [--urdf-dir PATH]
"""

import os
import shutil
import argparse
import glob
from pathlib import Path


def process_urdf_directory(urdf_dir, dry_run=False):
    """
    处理URDF目录，只保留指定文件和文件夹
    
    参数:
        urdf_dir (str): URDF目录路径
        dry_run (bool): 如果为True，只打印要执行的操作但不实际执行
    """
    # 获取所有ID目录
    id_dirs = [d for d in os.listdir(urdf_dir) 
              if os.path.isdir(os.path.join(urdf_dir, d))]
    
    total_dirs = len(id_dirs)
    print(f"找到{total_dirs}个ID目录")
    
    # 需要保留的文件和目录
    files_to_keep = ["mobility.urdf", "meta.json"]
    dirs_to_keep = ["yy_visualization", "textured_objs"]
    
    # 特殊文件: yy_merged.obj 可能在 yy_object 目录下
    special_file = "yy_merged.obj"
    special_dir = "yy_object"
    
    processed = 0
    for id_dir in id_dirs:
        id_path = os.path.join(urdf_dir, id_dir)
        
        # 检查并处理所有文件和目录
        for item in os.listdir(id_path):
            item_path = os.path.join(id_path, item)
            
            # 处理yy_object目录 - 只保留yy_merged.obj
            if item == special_dir and os.path.isdir(item_path):
                if not dry_run:
                    # 检查是否存在yy_merged.obj
                    obj_file = os.path.join(item_path, special_file)
                    if os.path.exists(obj_file):
                        # 将yy_merged.obj移动到上一级目录
                        shutil.copy2(obj_file, os.path.join(id_path, special_file))
                        print(f"  - 移动 {obj_file} 到 {id_path}/{special_file}")
                
                # 删除整个yy_object目录
                if not dry_run:
                    shutil.rmtree(item_path)
                    print(f"  - 删除目录: {item_path}")
                else:
                    print(f"  - 将删除目录: {item_path}")
                continue
            
            # 跳过要保留的文件和目录
            if item in files_to_keep or item in dirs_to_keep:
                print(f"  - 保留: {item_path}")
                continue
                
            # 删除其他文件或目录
            if os.path.isdir(item_path):
                if not dry_run:
                    shutil.rmtree(item_path)
                    print(f"  - 删除目录: {item_path}")
                else:
                    print(f"  - 将删除目录: {item_path}")
            else:
                if not dry_run:
                    os.remove(item_path)
                    print(f"  - 删除文件: {item_path}")
                else:
                    print(f"  - 将删除文件: {item_path}")
        
        processed += 1
        if processed % 10 == 0:
            print(f"已处理 {processed}/{total_dirs} 个目录")
    
    print(f"处理完成，共处理了 {processed} 个ID目录")


def main():
    parser = argparse.ArgumentParser(description="处理URDF目录，只保留指定文件和文件夹")
    parser.add_argument("--dry-run", action="store_true", help="只打印要执行的操作但不实际执行")
    parser.add_argument("--urdf-dir", type=str, default="/hy-tmp/PartField_Sketch_simpleMLP/data/urdf",
                       help="URDF目录路径")
    
    args = parser.parse_args()
    
    print(f"开始处理URDF目录: {args.urdf_dir}")
    if args.dry_run:
        print("模拟运行模式，不会实际修改文件")
    
    process_urdf_directory(args.urdf_dir, args.dry_run)


if __name__ == "__main__":
    main()