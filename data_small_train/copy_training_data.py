#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据复制脚本

功能：
1. 从data_small目录复制数据到data_small_train目录
2. 检查所有5个必需文件的完整性
3. 进行适当的重命名操作
4. 生成详细的处理报告

基于NOTE.md规范实现
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# 全局路径配置
DATA_SMALL_DIR = "/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small"
RESULT_DIR = os.path.join(DATA_SMALL_DIR, "result")
URDF_DIR = os.path.join(DATA_SMALL_DIR, "urdf")
TARGET_DIR = "/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small_train"

def parse_filename(filename):
    """
    解析文件名为各个组件 (来自NOTE.md)
    支持格式: {类别}_{id}_{渲染类型}[_{视角}][_joint_{joint_id}].png
    """
    # 先提取基本部分：类别、ID、渲染类型
    base_pattern = r'^([a-zA-Z]+)_(\d+)_([a-zA-Z\-]+)'
    base_match = re.match(base_pattern, filename)
    
    if not base_match:
        return None
    
    category = base_match.group(1)
    obj_id = base_match.group(2)
    render_type = base_match.group(3)
    
    # 获取剩余部分（可能包含视角和关节ID）
    remaining = filename[len(base_match.group(0)):]
    
    # 检查是否包含joint信息
    joint_pattern = r'_joint_(\d+)'
    joint_match = re.search(joint_pattern, remaining)
    joint_id = joint_match.group(1) if joint_match else None
    
    # 提取视角信息（如果存在）
    view = None
    if joint_match:
        view_part = remaining[:joint_match.start()]
        if view_part.startswith('_') and len(view_part) > 1:
            view = view_part[1:]  # 去掉开头的下划线
    elif remaining.startswith('_') and len(remaining) > 1:
        view = remaining[1:].split('.')[0]  # 去掉扩展名
    
    result = {
        'category': category,
        'id': obj_id,
        'render_type': render_type,
        'joint': joint_id,
        'view': view
    }
    
    return result

def discover_samples():
    """
    发现所有可用的训练样本
    返回: [(sample_name, parsed_info), ...]
    """
    print("正在扫描result目录...")
    
    samples = []
    if not os.path.exists(RESULT_DIR):
        print(f"错误：result目录不存在: {RESULT_DIR}")
        return samples
    
    for item in os.listdir(RESULT_DIR):
        item_path = os.path.join(RESULT_DIR, item)
        if os.path.isdir(item_path):
            # 解析样本名称
            parsed = parse_filename(item)
            if parsed and parsed['render_type'] == 'segmentation':
                samples.append((item, parsed))
    
    print(f"发现 {len(samples)} 个样本目录")
    return samples

def check_required_files(sample_name, parsed_info):
    """
    检查单个样本的所有必需文件是否存在
    
    Args:
        sample_name: 样本目录名
        parsed_info: 解析后的文件信息
        
    Returns:
        dict: 文件路径映射，如果文件缺失则返回None
    """
    obj_id = parsed_info['id']
    joint_id = parsed_info['joint']
    
    # 定义所有必需文件的路径
    required_files = {
        'pred_confidence': os.path.join(RESULT_DIR, sample_name, "pred_face_confidence.txt"),
        'face_mapping': os.path.join(URDF_DIR, obj_id, "face_vertex_mapping.txt"),
        'features': os.path.join(URDF_DIR, obj_id, "feature", "mesh", f"{obj_id}.npy"),
        'geometry': os.path.join(URDF_DIR, obj_id, "yy_merged.obj"),
        'labels': os.path.join(URDF_DIR, obj_id, "yy_visualization", f"labels_{joint_id}.txt")
    }
    
    # 检查每个文件是否存在
    missing_files = []
    for file_type, file_path in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append(f"{file_type}: {file_path}")
    
    if missing_files:
        return None, missing_files
    
    return required_files, []

def copy_sample_files(sample_name, parsed_info, file_paths, target_base_dir):
    """
    复制单个样本的所有文件到目标目录
    
    Args:
        sample_name: 样本目录名
        parsed_info: 解析后的文件信息
        file_paths: 源文件路径映射
        target_base_dir: 目标基础目录
        
    Returns:
        bool: 是否成功
    """
    obj_id = parsed_info['id']
    joint_id = parsed_info['joint']
    
    # 创建目标目录
    target_sample_dir = os.path.join(target_base_dir, sample_name)
    os.makedirs(target_sample_dir, exist_ok=True)
    
    try:
        # 1. 复制pred_face_confidence.txt
        shutil.copy2(
            file_paths['pred_confidence'],
            os.path.join(target_sample_dir, "pred_face_confidence.txt")
        )
        
        # 2. 复制face_vertex_mapping.txt
        shutil.copy2(
            file_paths['face_mapping'],
            os.path.join(target_sample_dir, "face_vertex_mapping.txt")
        )
        
        # 3. 复制{id}.npy
        shutil.copy2(
            file_paths['features'],
            os.path.join(target_sample_dir, f"{obj_id}.npy")
        )
        
        # 4. 复制并重命名yy_merged.obj -> {id}.obj
        shutil.copy2(
            file_paths['geometry'],
            os.path.join(target_sample_dir, f"{obj_id}.obj")
        )
        
        # 5. 复制labels_{joint_id}.txt
        shutil.copy2(
            file_paths['labels'],
            os.path.join(target_sample_dir, f"labels_{joint_id}.txt")
        )
        
        return True
        
    except Exception as e:
        print(f"错误：复制样本 {sample_name} 时发生异常: {e}")
        # 清理可能部分创建的目录
        if os.path.exists(target_sample_dir):
            shutil.rmtree(target_sample_dir)
        return False

def generate_report(success_samples, failed_samples, missing_files_log):
    """
    生成处理报告
    """
    report_path = os.path.join(TARGET_DIR, "copy_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("训练数据复制报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 统计信息
        f.write(f"总样本数: {len(success_samples) + len(failed_samples)}\n")
        f.write(f"成功复制: {len(success_samples)} 个样本\n")
        f.write(f"复制失败: {len(failed_samples)} 个样本\n\n")
        
        # 成功样本列表
        if success_samples:
            f.write("成功复制的样本:\n")
            f.write("-" * 30 + "\n")
            for sample in success_samples:
                f.write(f"  ✓ {sample}\n")
            f.write("\n")
        
        # 失败样本详情
        if failed_samples:
            f.write("复制失败的样本:\n")
            f.write("-" * 30 + "\n")
            for sample in failed_samples:
                f.write(f"  ✗ {sample}\n")
            f.write("\n")
        
        # 缺失文件详情
        if missing_files_log:
            f.write("缺失文件详情:\n")
            f.write("-" * 30 + "\n")
            for sample, missing_files in missing_files_log.items():
                f.write(f"样本: {sample}\n")
                for missing_file in missing_files:
                    f.write(f"  - {missing_file}\n")
                f.write("\n")
    
    print(f"\n处理报告已保存到: {report_path}")

def analyze_data_statistics(success_samples):
    """
    分析数据统计信息
    """
    if not success_samples:
        return
    
    # 按类别统计
    category_stats = defaultdict(int)
    # 按ID统计
    id_stats = defaultdict(int)
    # 按关节统计
    joint_stats = defaultdict(int)
    
    for sample in success_samples:
        parsed = parse_filename(sample)
        if parsed:
            category_stats[parsed['category']] += 1
            id_stats[parsed['id']] += 1
            joint_stats[parsed['joint']] += 1
    
    print(f"\n数据统计信息:")
    print(f"类别分布: {dict(category_stats)}")
    print(f"关节分布: {dict(joint_stats)}")
    print(f"唯一ID数量: {len(id_stats)}")

def main():
    parser = argparse.ArgumentParser(description="复制训练数据")
    parser.add_argument("--dry_run", action="store_true", help="只检查文件，不实际复制")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的目录")
    args = parser.parse_args()
    
    print("=" * 60)
    print("训练数据复制脚本")
    print("=" * 60)
    
    # 创建目标目录
    if not args.dry_run:
        os.makedirs(TARGET_DIR, exist_ok=True)
    
    # 发现所有样本
    samples = discover_samples()
    if not samples:
        print("没有找到任何样本，退出")
        return
    
    # 处理每个样本
    success_samples = []
    failed_samples = []
    missing_files_log = {}
    
    print(f"\n开始处理 {len(samples)} 个样本...")
    
    for sample_name, parsed_info in tqdm(samples, desc="处理样本"):
        # 检查目标目录是否已存在
        target_sample_dir = os.path.join(TARGET_DIR, sample_name)
        if os.path.exists(target_sample_dir) and not args.overwrite:
            print(f"跳过已存在的样本: {sample_name}")
            success_samples.append(sample_name)
            continue
        
        # 检查必需文件
        file_paths, missing_files = check_required_files(sample_name, parsed_info)
        
        if file_paths is None:
            failed_samples.append(sample_name)
            missing_files_log[sample_name] = missing_files
            continue
        
        # 实际复制文件（除非是dry run）
        if args.dry_run:
            print(f"[DRY RUN] 会复制样本: {sample_name}")
            success_samples.append(sample_name)
        else:
            if copy_sample_files(sample_name, parsed_info, file_paths, TARGET_DIR):
                success_samples.append(sample_name)
            else:
                failed_samples.append(sample_name)
    
    # 生成报告
    print(f"\n处理完成!")
    print(f"成功: {len(success_samples)} 个样本")
    print(f"失败: {len(failed_samples)} 个样本")
    
    if not args.dry_run:
        generate_report(success_samples, failed_samples, missing_files_log)
        analyze_data_statistics(success_samples)
    
    # 显示失败原因
    if failed_samples:
        print(f"\n失败样本数量: {len(failed_samples)}")
        print("主要失败原因:")
        for sample, missing_files in list(missing_files_log.items())[:5]:  # 只显示前5个
            print(f"  {sample}: {len(missing_files)} 个文件缺失")
        if len(missing_files_log) > 5:
            print(f"  ... 还有 {len(missing_files_log) - 5} 个样本有缺失文件")

if __name__ == "__main__":
    main() 