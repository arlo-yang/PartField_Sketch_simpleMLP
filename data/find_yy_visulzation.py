"""
我们的/hy-tmp/PartField_Sketch_simpleMLP/data/urdf下面有一些文件夹，他们用id命名，比如7128。
去/hy-tmp/PartField_Sketch/javascript/urdf/{id}/yy_object/中找到yy_visualization，
复制到/hy-tmp/PartField_Sketch_simpleMLP/data/urdf/id/ 里面。我们用id去比对。
"""

import os
import shutil
import argparse
from pathlib import Path
import time

def copy_yy_visualization(dry_run=False):
    """
    查找并复制yy_visualization文件夹
    
    参数:
        dry_run (bool): 如果为True，只打印要执行的操作但不实际执行
    """
    # 定义源目录和目标目录的基础路径
    target_base = "/hy-tmp/PartField_Sketch_simpleMLP/data/urdf"
    source_base = "/hy-tmp/PartField_Sketch/javascript/urdf"
    
    # 确保基础路径存在
    if not os.path.exists(target_base):
        print(f"错误: 目标目录不存在 - {target_base}")
        return
    
    if not os.path.exists(source_base):
        print(f"错误: 源目录不存在 - {source_base}")
        return
        
    # 获取目标目录中所有ID文件夹
    id_dirs = [d for d in os.listdir(target_base) 
              if os.path.isdir(os.path.join(target_base, d))]
    
    total_dirs = len(id_dirs)
    print(f"在目标目录找到{total_dirs}个ID文件夹")
    
    # 开始处理
    start_time = time.time()
    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    for idx, id_dir in enumerate(id_dirs, 1):
        # 构建源路径和目标路径
        source_vis_dir = os.path.join(source_base, id_dir, "yy_object", "yy_visualization")
        target_vis_dir = os.path.join(target_base, id_dir, "yy_visualization")
        
        print(f"处理[{idx}/{total_dirs}] ID: {id_dir}")
        
        # 检查源目录是否存在
        if not os.path.exists(source_vis_dir):
            print(f"  - 源目录不存在: {source_vis_dir}")
            error_count += 1
            continue
            
        # 检查目标目录是否已存在
        if os.path.exists(target_vis_dir):
            print(f"  - 目标目录已存在，将覆盖: {target_vis_dir}")
            if not dry_run:
                # 删除已有目录
                shutil.rmtree(target_vis_dir)
        
        # 复制目录
        if not dry_run:
            try:
                shutil.copytree(source_vis_dir, target_vis_dir)
                print(f"  - 成功复制: {source_vis_dir} -> {target_vis_dir}")
                copied_count += 1
            except Exception as e:
                print(f"  - 复制失败: {e}")
                error_count += 1
        else:
            print(f"  - 将复制: {source_vis_dir} -> {target_vis_dir}")
            copied_count += 1
        
        # 显示进度
        if idx % 10 == 0 or idx == total_dirs:
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = avg_time * (total_dirs - idx)
            print(f"进度: {idx}/{total_dirs} ({idx/total_dirs*100:.1f}%) - 预计剩余时间: {int(remaining//60)}分{int(remaining%60)}秒")
    
    # 打印总结
    total_time = time.time() - start_time
    print("\n处理完成!")
    print(f"总计: {total_dirs} 个ID")
    print(f"成功复制: {copied_count}")
    print(f"跳过: {skipped_count}")
    print(f"错误: {error_count}")
    print(f"总耗时: {int(total_time//60)}分{int(total_time%60)}秒")

def main():
    parser = argparse.ArgumentParser(description="查找并复制yy_visualization文件夹")
    parser.add_argument("--dry-run", action="store_true", help="只打印要执行的操作但不实际执行")
    
    args = parser.parse_args()
    
    print("开始查找并复制yy_visualization文件夹")
    if args.dry_run:
        print("模拟运行模式，不会实际复制文件")
    
    copy_yy_visualization(args.dry_run)

if __name__ == "__main__":
    main()