#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_id.py -- 清理URDF目录，只保留特定ID的文件夹

功能：扫描 urdf_09072025 目录获取有效ID列表，然后删除 urdf 目录中不在此列表中的文件夹。
"""

from pathlib import Path
import os
import sys
import shutil

def get_valid_folder_ids(source_dir):
    """
    获取来源目录下的所有文件夹ID
    
    参数:
        source_dir: 来源URDF文件夹路径
    
    返回:
        文件夹ID集合(set)
    """
    # 确保目录存在
    source_path = Path(source_dir)
    if not source_path.exists() or not source_path.is_dir():
        print(f"错误: 目录不存在 - {source_dir}")
        sys.exit(1)
    
    # 获取所有文件夹并过滤出数字ID
    folder_ids = set()
    for item in source_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            folder_ids.add(item.name)
    
    return folder_ids

def clean_target_directory(target_dir, valid_ids):
    """
    清理目标目录，只保留ID在valid_ids中的文件夹
    
    参数:
        target_dir: 要清理的目标目录
        valid_ids: 有效ID集合
    
    返回:
        (保留数量, 删除数量)
    """
    # 确保目录存在
    target_path = Path(target_dir)
    if not target_path.exists() or not target_path.is_dir():
        print(f"错误: 目录不存在 - {target_dir}")
        sys.exit(1)
    
    kept = 0
    deleted = 0
    
    # 查找并直接删除无效文件夹
    for item in target_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            if item.name in valid_ids:
                kept += 1
            else:
                try:
                    shutil.rmtree(item)
                    print(f"已删除: {item.name}")
                    deleted += 1
                except Exception as e:
                    print(f"删除 {item.name} 失败: {e}")
    
    return kept, deleted

def main():
    # 源目录和目标目录
    source_dir = "/hy-tmp/PartField_Sketch/javascript/urdf_09072025"
    target_dir = "/hy-tmp/PartField_Sketch/javascript/urdf"
    
    # 获取有效ID
    valid_ids = get_valid_folder_ids(source_dir)
    print(f"在源目录找到 {len(valid_ids)} 个有效ID")
    
    # 清理目标目录
    kept, deleted = clean_target_directory(target_dir, valid_ids)
    
    # 输出统计
    print("\n===== 清理完成 =====")
    print(f"保留文件夹: {kept}")
    print(f"删除文件夹: {deleted}")

if __name__ == "__main__":
    main()
