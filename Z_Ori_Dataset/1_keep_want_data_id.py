#!/usr/bin/env python3
"""
根据 model_ids.txt 文件中的ID列表，在指定目录中只保留对应的文件夹，删除其他文件夹
"""

import os
import shutil
from tqdm import tqdm

# 配置路径
BASE_DIR = "/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP"
IDS_FILE = os.path.join(BASE_DIR, "Z_Ori_Dataset/model_ids.txt")
TARGET_DIR = "/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/Z_Ori_Dataset/urdf"

def load_wanted_ids(ids_file):
    """从txt文件中加载需要保留的ID列表"""
    wanted_ids = set()
    
    if not os.path.exists(ids_file):
        print(f"[Error] ID文件不存在: {ids_file}")
        return wanted_ids
    
    with open(ids_file, 'r') as f:
        for line in f:
            id_str = line.strip()
            if id_str:  # 跳过空行
                wanted_ids.add(id_str)
    
    print(f"[Loaded] 从 {ids_file} 加载了 {len(wanted_ids)} 个需要保留的ID")
    return wanted_ids

def scan_target_directory(target_dir):
    """扫描目标目录，获取所有现有的文件夹"""
    if not os.path.exists(target_dir):
        print(f"[Error] 目标目录不存在: {target_dir}")
        return []
    
    all_items = []
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        if os.path.isdir(item_path):
            all_items.append(item)
    
    print(f"[Scanned] 目标目录中找到 {len(all_items)} 个文件夹")
    return all_items

def classify_folders(all_folders, wanted_ids):
    """将文件夹分类为需要保留和需要删除的"""
    to_keep = []
    to_delete = []
    
    for folder in all_folders:
        if folder in wanted_ids:
            to_keep.append(folder)
        else:
            to_delete.append(folder)
    
    print(f"[Classify] 需要保留: {len(to_keep)} 个文件夹")
    print(f"[Classify] 需要删除: {len(to_delete)} 个文件夹")
    
    return to_keep, to_delete

def preview_operations(to_keep, to_delete, target_dir):
    """预览将要执行的操作"""
    print("\n" + "="*60)
    print("操作预览:")
    print("="*60)
    
    print(f"\n[保留] 以下 {len(to_keep)} 个文件夹将被保留:")
    if len(to_keep) <= 10:
        for folder in sorted(to_keep):
            print(f"  ✓ {folder}")
    else:
        for folder in sorted(to_keep)[:5]:
            print(f"  ✓ {folder}")
        print(f"  ... (还有 {len(to_keep)-5} 个)")
    
    print(f"\n[删除] 以下 {len(to_delete)} 个文件夹将被删除:")
    if len(to_delete) <= 10:
        for folder in sorted(to_delete):
            print(f"  ✗ {folder}")
    else:
        for folder in sorted(to_delete)[:5]:
            print(f"  ✗ {folder}")
        print(f"  ... (还有 {len(to_delete)-5} 个)")
    
    print("\n" + "="*60)

def delete_unwanted_folders(to_delete, target_dir, dry_run=True):
    """删除不需要的文件夹"""
    if dry_run:
        print(f"\n[DRY RUN] 模拟删除 {len(to_delete)} 个文件夹...")
        for folder in to_delete:
            folder_path = os.path.join(target_dir, folder)
            print(f"  [模拟] 将删除: {folder_path}")
        return
    
    print(f"\n[REAL DELETE] 开始删除 {len(to_delete)} 个文件夹...")
    deleted_count = 0
    failed_count = 0
    
    for folder in tqdm(to_delete, desc="删除文件夹"):
        folder_path = os.path.join(target_dir, folder)
        try:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                deleted_count += 1
            else:
                print(f"  [Warning] 文件夹不存在: {folder_path}")
        except Exception as e:
            print(f"  [Error] 删除失败 {folder_path}: {e}")
            failed_count += 1
    
    print(f"\n[结果] 成功删除: {deleted_count} 个, 失败: {failed_count} 个")

def main():
    print("="*60)
    print("文件夹清理脚本")
    print("="*60)
    print(f"ID文件: {IDS_FILE}")
    print(f"目标目录: {TARGET_DIR}")
    print()
    
    # 1. 加载需要保留的ID
    wanted_ids = load_wanted_ids(IDS_FILE)
    if not wanted_ids:
        print("[Error] 没有加载到任何ID，退出")
        return
    
    # 2. 扫描目标目录
    all_folders = scan_target_directory(TARGET_DIR)
    if not all_folders:
        print("[Error] 目标目录为空或不存在，退出")
        return
    
    # 3. 分类文件夹
    to_keep, to_delete = classify_folders(all_folders, wanted_ids)
    
    # 4. 预览操作
    preview_operations(to_keep, to_delete, TARGET_DIR)
    
    if not to_delete:
        print("\n[Info] 没有需要删除的文件夹，所有文件夹都在保留列表中")
        return
    
    # 5. 安全确认
    print(f"\n⚠️  警告: 即将删除 {len(to_delete)} 个文件夹!")
    print("这个操作是不可逆的!")
    
    # 先进行干运行
    print("\n[1/2] 首先进行模拟运行...")
    delete_unwanted_folders(to_delete, TARGET_DIR, dry_run=True)
    
    print("\n" + "="*60)
    choice = input("确认要执行真实删除吗? (输入 'YES' 确认, 其他任意键取消): ")
    
    if choice == 'YES':
        print("\n[2/2] 执行真实删除...")
        delete_unwanted_folders(to_delete, TARGET_DIR, dry_run=False)
        print("\n✅ 清理完成!")
    else:
        print("\n❌ 操作已取消")

if __name__ == "__main__":
    main()
