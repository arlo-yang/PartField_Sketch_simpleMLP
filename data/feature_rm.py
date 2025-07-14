#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征文件夹清理脚本
===============================
该脚本用于删除URDF目录下所有ID目录中的feature文件夹

使用方法:
    python feature_rm.py [--dry-run] [--urdf-dir PATH]
"""

import os
import shutil
import argparse
import glob
from pathlib import Path


def remove_feature_folders(urdf_dir, dry_run=False):
    """
    删除URDF目录下所有ID目录中的feature文件夹
    
    参数:
        urdf_dir (str): URDF目录路径
        dry_run (bool): 如果为True，只打印要执行的操作但不实际执行
    """
    # 获取所有ID目录
    id_dirs = [d for d in os.listdir(urdf_dir) 
              if os.path.isdir(os.path.join(urdf_dir, d))]
    
    total_dirs = len(id_dirs)
    print(f"找到{total_dirs}个ID目录")
    
    removed = 0
    skipped = 0
    
    for id_dir in id_dirs:
        id_path = os.path.join(urdf_dir, id_dir)
        feature_dir = os.path.join(id_path, "feature")
        
        if os.path.isdir(feature_dir):
            if dry_run:
                print(f"将删除: {feature_dir}")
            else:
                try:
                    shutil.rmtree(feature_dir)
                    print(f"已删除: {feature_dir}")
                    removed += 1
                except Exception as e:
                    print(f"删除失败 {feature_dir}: {e}")
        else:
            print(f"跳过 {id_dir}: 未找到feature文件夹")
            skipped += 1
            
        # 每处理50个目录打印一次进度
        if (removed + skipped) % 50 == 0:
            print(f"进度: {removed + skipped}/{total_dirs} (已删除: {removed}, 已跳过: {skipped})")
    
    print(f"处理完成，共删除了{removed}个feature文件夹，跳过了{skipped}个目录")


def main():
    parser = argparse.ArgumentParser(description="删除URDF目录下所有ID目录中的feature文件夹")
    parser.add_argument("--dry-run", action="store_true", help="只打印要执行的操作但不实际执行")
    parser.add_argument("--urdf-dir", type=str, default="/hy-tmp/PartField_Sketch_simpleMLP/data/urdf",
                       help="URDF目录路径")
    
    args = parser.parse_args()
    
    print(f"开始清理feature文件夹，基础目录: {args.urdf_dir}")
    if args.dry_run:
        print("模拟运行模式，不会实际删除文件")
    
    remove_feature_folders(args.urdf_dir, args.dry_run)


if __name__ == "__main__":
    main()
