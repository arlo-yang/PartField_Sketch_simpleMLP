#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import sys

def main():
    # 定义路径
    urdf_dir = "/hy-tmp/PartField_Sketch/javascript/urdf"
    folders_file = "/hy-tmp/PartField_Sketch/javascript/scripts/record_old_urdf/urdf_folders.txt"
    
    # 检查目录是否存在
    if not os.path.isdir(urdf_dir):
        print(f"错误: URDF目录不存在: {urdf_dir}")
        sys.exit(1)
        
    # 检查文件是否存在
    if not os.path.isfile(folders_file):
        print(f"错误: urdf_folders.txt文件不存在: {folders_file}")
        sys.exit(1)
    
    # 读取需要保留的文件夹ID列表
    with open(folders_file, 'r') as f:
        keep_folders = set(line.strip() for line in f if line.strip())
    
    print(f"从文件中读取到 {len(keep_folders)} 个需要保留的ID")
    
    # 获取URDF目录中的所有文件夹
    try:
        all_folders = [f for f in os.listdir(urdf_dir) if os.path.isdir(os.path.join(urdf_dir, f))]
    except Exception as e:
        print(f"读取URDF目录时出错: {e}")
        sys.exit(1)
    
    print(f"URDF目录中共有 {len(all_folders)} 个文件夹")
    
    # 找出需要删除的文件夹
    folders_to_delete = [f for f in all_folders if f not in keep_folders]
    
    print(f"将要删除 {len(folders_to_delete)} 个文件夹")
    
    # 删除不在列表中的文件夹
    deleted_count = 0
    for folder in folders_to_delete:
        folder_path = os.path.join(urdf_dir, folder)
        try:
            shutil.rmtree(folder_path)
            deleted_count += 1
            print(f"已删除: {folder}")
        except Exception as e:
            print(f"删除文件夹 {folder} 时出错: {e}")
    
    print(f"操作完成. 共删除了 {deleted_count} 个文件夹。")

if __name__ == "__main__":
    main()
