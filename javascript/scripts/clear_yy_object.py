#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import sys

def clear_yy_objects():
    """
    遍历URDF目录，删除每个URDF文件夹中的yy_object目录
    """
    # 获取脚本所在目录的绝对路径
    script_dir = Path(__file__).resolve().parent
    
    # 构建urdf目录的绝对路径（在javascript目录下）
    urdf_dir = script_dir.parent / 'urdf'
    
    if not urdf_dir.exists():
        print(f"错误: URDF目录不存在: {urdf_dir}")
        return
    
    print(f"开始扫描URDF目录: {urdf_dir}")
    
    total_folders = 0
    deleted_count = 0
    
    # 遍历所有数字命名的文件夹
    for folder in sorted(os.listdir(urdf_dir)):
        folder_path = urdf_dir / folder
        
        # 确保是目录
        if not folder_path.is_dir():
            continue
            
        total_folders += 1
        
        # 检查是否存在yy_object目录
        yy_object_path = folder_path / 'yy_object'
        if yy_object_path.exists():
            try:
                # 删除yy_object目录及其所有内容
                shutil.rmtree(yy_object_path)
                print(f"已删除: {yy_object_path}")
                deleted_count += 1
            except Exception as e:
                print(f"删除 {yy_object_path} 时出错: {e}")
    
    print("\n清理完成:")
    print(f"扫描的URDF文件夹总数: {total_folders}")
    print(f"删除的yy_object目录数: {deleted_count}")

if __name__ == '__main__':
    clear_yy_objects()
