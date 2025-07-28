#!/usr/bin/env python3
"""
批量清理模型文件夹，只保留指定的文件和文件夹
- textured_objs/ (文件夹)
- meta.json (文件)
- mobility.urdf (文件)
"""

import os
import shutil
from tqdm import tqdm

# 配置路径
BASE_DIR = "/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP"
URDF_DIR = os.path.join(BASE_DIR, "Z_Ori_Dataset/urdf")

# 需要保留的文件和文件夹
KEEP_ITEMS = {
    "textured_objs",  # 文件夹
    "meta.json",      # 文件
    "mobility.urdf"   # 文件
}

def scan_model_folders():
    """扫描所有模型文件夹"""
    if not os.path.exists(URDF_DIR):
        print(f"[Error] 目录不存在: {URDF_DIR}")
        return []
    
    model_folders = []
    for item in os.listdir(URDF_DIR):
        item_path = os.path.join(URDF_DIR, item)
        if os.path.isdir(item_path):
            model_folders.append(item)
    
    print(f"[Found] 找到 {len(model_folders)} 个模型文件夹")
    return sorted(model_folders)

def analyze_folder(model_id):
    """分析单个模型文件夹的内容"""
    model_path = os.path.join(URDF_DIR, model_id)
    
    if not os.path.exists(model_path):
        return None, None, None
    
    all_items = os.listdir(model_path)
    to_keep = []
    to_delete = []
    
    for item in all_items:
        if item in KEEP_ITEMS:
            to_keep.append(item)
        else:
            to_delete.append(item)
    
    return all_items, to_keep, to_delete

def preview_cleanup(model_folders):
    """预览清理操作"""
    print("\n" + "="*60)
    print("清理操作预览")
    print("="*60)
    
    total_to_delete = 0
    missing_required = []
    
    # 分析前几个文件夹作为示例
    sample_size = min(3, len(model_folders))
    
    for i, model_id in enumerate(model_folders[:sample_size]):
        all_items, to_keep, to_delete = analyze_folder(model_id)
        
        if all_items is None:
            continue
            
        print(f"\n[示例 {i+1}] 文件夹: {model_id}")
        print(f"  总项目: {len(all_items)}")
        print(f"  保留: {len(to_keep)} 个 → {to_keep}")
        print(f"  删除: {len(to_delete)} 个 → {to_delete[:5]}{'...' if len(to_delete) > 5 else ''}")
        
        total_to_delete += len(to_delete)
        
        # 检查必需文件是否存在
        missing = KEEP_ITEMS - set(to_keep)
        if missing:
            missing_required.append((model_id, missing))
    
    # 统计所有文件夹的删除项目
    print(f"\n[统计] 预计总共删除项目数: {total_to_delete * len(model_folders) // sample_size}")
    
    if missing_required:
        print(f"\n⚠️  警告: 以下文件夹缺少必需文件:")
        for model_id, missing in missing_required:
            print(f"  {model_id}: 缺少 {missing}")

def cleanup_folder(model_id, dry_run=True):
    """清理单个文件夹"""
    model_path = os.path.join(URDF_DIR, model_id)
    all_items, to_keep, to_delete = analyze_folder(model_id)
    
    if all_items is None:
        return False, f"文件夹不存在: {model_path}"
    
    if not to_delete:
        return True, "无需删除任何内容"
    
    deleted_count = 0
    failed_items = []
    
    for item in to_delete:
        item_path = os.path.join(model_path, item)
        
        if dry_run:
            if os.path.exists(item_path):
                deleted_count += 1
        else:
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
                deleted_count += 1
            except Exception as e:
                failed_items.append(f"{item}: {e}")
    
    if failed_items:
        return False, f"删除了 {deleted_count} 个, 失败: {failed_items}"
    else:
        return True, f"删除了 {deleted_count} 个项目"

def batch_cleanup(model_folders, dry_run=True):
    """批量清理所有文件夹"""
    mode = "模拟" if dry_run else "真实"
    print(f"\n[{mode}清理] 开始处理 {len(model_folders)} 个文件夹...")
    
    success_count = 0
    failed_count = 0
    
    for model_id in tqdm(model_folders, desc=f"{mode}清理"):
        success, message = cleanup_folder(model_id, dry_run)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
            print(f"  [Error] {model_id}: {message}")
    
    print(f"\n[{mode}结果] 成功: {success_count}, 失败: {failed_count}")

def main():
    print("="*60)
    print("模型文件夹批量清理脚本")
    print("="*60)
    print(f"目标目录: {URDF_DIR}")
    print(f"保留项目: {KEEP_ITEMS}")
    print()
    
    # 1. 扫描模型文件夹
    model_folders = scan_model_folders()
    if not model_folders:
        print("[Error] 未找到任何模型文件夹")
        return
    
    # 2. 预览操作
    preview_cleanup(model_folders)
    
    # 3. 模拟运行
    print("\n" + "="*60)
    print("[1/2] 模拟运行...")
    batch_cleanup(model_folders, dry_run=True)
    
    # 4. 确认执行
    print("\n" + "="*60)
    print("⚠️  警告: 即将删除大量文件和文件夹!")
    print("只会保留: textured_objs/, meta.json, mobility.urdf")
    print("其他所有内容都将被永久删除!")
    
    choice = input("\n确认要执行真实删除吗? (输入 'YES' 确认, 其他任意键取消): ")
    
    if choice == 'YES':
        print("\n[2/2] 执行真实删除...")
        batch_cleanup(model_folders, dry_run=False)
        print("\n✅ 批量清理完成!")
    else:
        print("\n❌ 操作已取消")

if __name__ == "__main__":
    main() 