#!/usr/bin/env python3
import json
import os
import shutil
import sys

def delete_failed_models(json_path, urdf_base_path, dry_run=False):
    """
    删除arrow_creation_failures.json中列出的模型目录
    
    参数:
        json_path (str): JSON文件路径
        urdf_base_path (str): URDF基础路径
        dry_run (bool): 如果为True，只打印要删除的路径但不实际删除
    """
    # 读取JSON文件
    try:
        with open(json_path, 'r') as f:
            failed_models = json.load(f)
    except Exception as e:
        print(f"无法读取JSON文件: {e}")
        return
    
    # 统计
    total_models = len(failed_models)
    deleted_count = 0
    not_found_count = 0
    error_count = 0
    
    print(f"找到 {total_models} 个需要删除的模型")
    
    # 确认删除
    if not dry_run:
        confirm = input(f"确认删除这 {total_models} 个模型目录? (y/n): ")
        if confirm.lower() != 'y':
            print("操作已取消")
            return
    
    # 遍历并删除文件夹
    for model in failed_models:
        model_id = model['model_id']
        category = model.get('category', 'Unknown')
        reason = model.get('reason', 'Unknown')
        
        # 构造完整路径
        model_path = os.path.join(urdf_base_path, model_id)
        
        print(f"处理: {model_id} (类别: {category})")
        print(f"  - 失败原因: {reason}")
        print(f"  - 路径: {model_path}")
        
        if os.path.exists(model_path):
            try:
                if dry_run:
                    print(f"  - [DRY RUN] 将删除: {model_path}")
                else:
                    shutil.rmtree(model_path)
                    print(f"  - 已删除: {model_path}")
                deleted_count += 1
            except Exception as e:
                print(f"  - 删除失败: {e}")
                error_count += 1
        else:
            print(f"  - 路径不存在: {model_path}")
            not_found_count += 1
    
    # 打印总结
    print("\n删除摘要:")
    print(f"总计处理: {total_models} 个模型")
    if dry_run:
        print(f"将删除: {deleted_count} 个文件夹")
    else:
        print(f"已删除: {deleted_count} 个文件夹")
    print(f"不存在: {not_found_count} 个文件夹")
    print(f"错误: {error_count} 个文件夹")

if __name__ == "__main__":
    # 使用绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
    json_path = os.path.join(script_dir, "arrow_creation_failures.json")
    urdf_base_path = os.path.join(script_dir, "..", "urdf")
    
    # 检查参数
    if len(sys.argv) > 1 and sys.argv[1] == '--dry-run':
        print("执行测试运行（不会实际删除文件）")
        delete_failed_models(json_path, urdf_base_path, dry_run=True)
    else:
        delete_failed_models(json_path, urdf_base_path)
