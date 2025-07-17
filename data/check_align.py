#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
from collections import defaultdict

# 定义路径
IMG_DIR = "/hy-tmp/PartField_Sketch_simpleMLP/data/img"
URDF_DIR = "/hy-tmp/PartField_Sketch_simpleMLP/data/urdf"

def extract_id_from_img_filename(filename):
    """从图片文件名中提取ID"""
    # 文件名格式: {类别}_{id}_{render mode}_{view}_joint_{joint-id}.png
    pattern = r'^[^_]+_(\d+)_.*\.png$'
    match = re.match(pattern, filename)
    if match:
        return match.group(1)
    return None

def get_img_ids():
    """获取img文件夹中所有的ID"""
    img_ids = set()
    img_id_to_files = defaultdict(list)
    
    if not os.path.exists(IMG_DIR):
        print(f"警告: 图片目录不存在 {IMG_DIR}")
        return img_ids, img_id_to_files
    
    for filename in os.listdir(IMG_DIR):
        if filename.endswith(".png"):
            img_id = extract_id_from_img_filename(filename)
            if img_id:
                img_ids.add(img_id)
                img_id_to_files[img_id].append(filename)
    
    return img_ids, img_id_to_files

def get_urdf_ids():
    """获取urdf文件夹中所有的ID"""
    urdf_ids = set()
    
    if not os.path.exists(URDF_DIR):
        print(f"警告: URDF目录不存在 {URDF_DIR}")
        return urdf_ids
    
    for item in os.listdir(URDF_DIR):
        # 假设urdf下的所有子文件夹名就是ID
        if os.path.isdir(os.path.join(URDF_DIR, item)):
            urdf_ids.add(item)
    
    return urdf_ids

def remove_extra_img_files(img_only_ids, img_id_to_files):
    """删除img目录中多余的文件（urdf中没有对应ID的文件）"""
    removed_count = 0
    
    for img_id in img_only_ids:
        for filename in img_id_to_files[img_id]:
            file_path = os.path.join(IMG_DIR, filename)
            try:
                os.remove(file_path)
                removed_count += 1
                print(f"已删除: {file_path}")
            except Exception as e:
                print(f"删除失败 {file_path}: {e}")
    
    return removed_count

def remove_extra_urdf_dirs(urdf_only_ids):
    """删除urdf目录中多余的文件夹（img中没有对应ID的文件夹）"""
    removed_count = 0
    
    for urdf_id in urdf_only_ids:
        dir_path = os.path.join(URDF_DIR, urdf_id)
        try:
            shutil.rmtree(dir_path)
            removed_count += 1
            print(f"已删除: {dir_path}")
        except Exception as e:
            print(f"删除失败 {dir_path}: {e}")
    
    return removed_count

def compare_ids():
    """比较img和urdf文件夹中的ID"""
    img_ids, img_id_to_files = get_img_ids()
    urdf_ids = get_urdf_ids()
    
    print(f"图片目录中找到的ID数量: {len(img_ids)}")
    print(f"URDF目录中找到的ID数量: {len(urdf_ids)}")
    
    # 检查img有但urdf没有的ID
    img_only = img_ids - urdf_ids
    if img_only:
        print(f"\n在图片目录中找到但在URDF目录中缺失的ID ({len(img_only)}):")
        for img_id in sorted(img_only):
            print(f"  - ID: {img_id}，对应的文件:")
            for filename in img_id_to_files[img_id][:5]:  # 只显示前5个文件
                print(f"    * {filename}")
            if len(img_id_to_files[img_id]) > 5:
                print(f"    * ... 等{len(img_id_to_files[img_id])-5}个其他文件")
    else:
        print("\n没有在图片目录中找到但在URDF目录中缺失的ID")
    
    # 检查urdf有但img没有的ID
    urdf_only = urdf_ids - img_ids
    if urdf_only:
        print(f"\n在URDF目录中找到但在图片目录中缺失的ID ({len(urdf_only)}):")
        for urdf_id in sorted(urdf_only):
            print(f"  - {urdf_id}")
    else:
        print("\n没有在URDF目录中找到但在图片目录中缺失的ID")
    
    # 检查两者都有的ID
    common_ids = img_ids.intersection(urdf_ids)
    print(f"\n两个目录共有的ID数量: {len(common_ids)}")
    
    return img_only, urdf_only, common_ids, img_id_to_files

def main():
    print("开始检查img目录和urdf目录中ID的一致性...\n")
    img_only, urdf_only, common_ids, img_id_to_files = compare_ids()
    
    print("\n摘要:")
    print(f"- 总共在图片目录中找到ID: {len(img_only) + len(common_ids)}")
    print(f"- 总共在URDF目录中找到ID: {len(urdf_only) + len(common_ids)}")
    print(f"- 只在图片目录中找到的ID: {len(img_only)}")
    print(f"- 只在URDF目录中找到的ID: {len(urdf_only)}")
    print(f"- 两个目录共有的ID: {len(common_ids)}")
    
    # 询问是否要删除不一致的部分
    if img_only or urdf_only:
        print("\n发现不一致的ID，是否要删除多余的部分？")
        print("1. 删除img目录中多余的文件（urdf中没有对应ID的文件）")
        print("2. 删除urdf目录中多余的文件夹（img中没有对应ID的文件夹）")
        print("3. 同时删除两者")
        print("4. 不删除，仅显示报告")
        
        choice = input("\n请输入选择 (1/2/3/4): ").strip()
        
        if choice == '1' or choice == '3':
            removed_img = remove_extra_img_files(img_only, img_id_to_files)
            print(f"\n已从img目录中删除 {removed_img} 个文件")
        
        if choice == '2' or choice == '3':
            removed_urdf = remove_extra_urdf_dirs(urdf_only)
            print(f"\n已从urdf目录中删除 {removed_urdf} 个文件夹")
        
        if choice == '4':
            print("\n您选择不删除任何文件。")
    else:
        print("\nimg和urdf目录中的ID完全一致，不需要删除操作。")

if __name__ == "__main__":
    main()
