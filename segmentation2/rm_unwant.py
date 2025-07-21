#!/usr/bin/env python3
# filename: rm_unwant.py
# 用于移除数据集中不需要的图像文件（default, normal, depth等）

import os
import re
import argparse
import shutil
from collections import defaultdict
from tqdm import tqdm

# 复用loader.py中的函数
def parse_filename(filename):
    """
    解析文件名为各个组件
    支持格式:
    {类别}_{id}_{渲染类型}[_{视角}][_joint_{joint_id}].png
    
    Args:
        filename: 文件名
    Returns:
        字典包含类别、ID、渲染类型、关节ID(如果有)
    """
    # 先提取基本部分：类别、ID、渲染类型
    base_pattern = r'^([a-zA-Z]+)_(\d+)_([a-zA-Z\-]+)'
    base_match = re.match(base_pattern, filename)
    
    if not base_match:
        return None
    
    category = base_match.group(1)
    obj_id = base_match.group(2)
    render_type = base_match.group(3)
    
    # 检查是否包含joint信息
    joint_pattern = r'_joint_(\d+)'
    joint_match = re.search(joint_pattern, filename)
    joint_id = joint_match.group(1) if joint_match else None
    
    # 构建结果
    return {
        'category': category,
        'id': obj_id,
        'render_type': render_type,
        'joint': joint_id
    }


def remove_unwanted_images(data_dir, backup_dir=None, dry_run=False, verbose=True):
    """
    移除不需要的图像文件（default, normal, depth等），只保留segmentation和arrow-sketch
    
    Args:
        data_dir: 数据集目录
        backup_dir: 备份目录，如果指定，则将移除的文件移动到此处
        dry_run: 如果为True，只打印将要执行的操作，不实际删除/移动文件
        verbose: 是否打印详细信息
    
    Returns:
        元组(keep_files, removed_files)，分别为保留的文件数和移除的文件数
    """
    # 要保留的渲染类型
    KEEP_TYPES = ['segmentation', 'arrow-sketch']
    
    # 统计数据
    stats = defaultdict(int)
    keep_files = []
    remove_files = []
    
    # 扫描目录找到所有图片
    all_images = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.png'):
                all_images.append(os.path.join(root, file))
    
    # 确保备份目录存在
    if backup_dir and not dry_run:
        os.makedirs(backup_dir, exist_ok=True)
    
    # 处理进度条
    pbar = tqdm(all_images, desc="处理图像文件")
    for img_path in pbar:
        filename = os.path.basename(img_path)
        parsed = parse_filename(filename)
        
        if parsed:
            render_type = parsed['render_type']
            stats[render_type] += 1
            
            if render_type in KEEP_TYPES:
                # 保留的文件
                keep_files.append(img_path)
            else:
                # 要移除的文件
                remove_files.append(img_path)
                
                # 处理要移除的文件
                if not dry_run:
                    if backup_dir:
                        # 创建与原始目录结构相同的备份路径
                        rel_path = os.path.relpath(img_path, data_dir)
                        backup_path = os.path.join(backup_dir, rel_path)
                        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                        shutil.move(img_path, backup_path)
                    else:
                        # 直接删除
                        os.remove(img_path)
    
    # 打印统计信息
    if verbose:
        print("\n文件类型统计:")
        for render_type, count in sorted(stats.items()):
            status = "保留" if render_type in KEEP_TYPES else "移除"
            print(f"  {render_type}: {count} 文件 ({status})")
        
        print(f"\n总计:")
        print(f"  保留文件数: {len(keep_files)}")
        print(f"  移除文件数: {len(remove_files)}")
        
        if dry_run:
            print("\n这是一次演习运行，没有实际移除任何文件。")
            print("要真正移除文件，请去掉--dry-run参数。")
    
    return len(keep_files), len(remove_files)


def main():
    parser = argparse.ArgumentParser(description="移除数据集中不需要的图像文件")
    parser.add_argument("--data_dir", type=str, default="data/img",
                       help="数据集目录")
    parser.add_argument("--backup_dir", type=str, default=None,
                       help="备份目录，如果指定，则将移除的文件移动到此处；否则直接删除")
    parser.add_argument("--dry-run", action="store_true",
                       help="如果指定，只打印将要执行的操作，不实际删除/移动文件")
    
    args = parser.parse_args()
    
    print(f"数据集目录: {args.data_dir}")
    if args.backup_dir:
        print(f"备份目录: {args.backup_dir}")
    if args.dry_run:
        print("模式: 演习运行 (不会实际删除或移动文件)")
    else:
        print("模式: 实际运行 (将会" + ("移动" if args.backup_dir else "删除") + "不需要的文件)")
    
    # 确认
    if not args.dry_run:
        confirm = input("\n警告: 此操作将会" + ("移动" if args.backup_dir else "永久删除") + 
                        "除segmentation和arrow-sketch之外的图像文件。\n确定要继续吗? [y/N]: ")
        if confirm.lower() != 'y':
            print("操作已取消。")
            return
    
    print("\n开始处理...")
    keep_count, remove_count = remove_unwanted_images(
        data_dir=args.data_dir,
        backup_dir=args.backup_dir,
        dry_run=args.dry_run,
        verbose=True
    )
    
    if not args.dry_run:
        print(f"\n完成! 已保留 {keep_count} 个文件，移除 {remove_count} 个文件。")
    else:
        print(f"\n演习运行完成。如果执行实际操作，将保留 {keep_count} 个文件，移除 {remove_count} 个文件。")


if __name__ == "__main__":
    main()
