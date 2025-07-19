#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
from tqdm import tqdm

def process_object_dir(object_id):
    """
    处理单个对象目录：
    1. 删除simpleMLP项目中的yy_visualization目录
    2. 从原始Sketch项目复制yy_visualization目录
    """
    # 源路径和目标路径
    target_dir = f"/hy-tmp/PartField_Sketch_simpleMLP/javascript/urdf/{object_id}/yy_object/yy_visualization"
    source_dir = f"/hy-tmp/PartField_Sketch/javascript/urdf/{object_id}/yy_object/yy_visualization"
    
    # 删除目标路径（如果存在）
    if os.path.exists(target_dir):
        print(f"删除目录: {target_dir}")
        shutil.rmtree(target_dir)
    
    # 检查源路径是否存在
    if not os.path.exists(source_dir):
        print(f"警告: 源目录不存在: {source_dir}")
        return False
    
    # 复制目录
    print(f"复制: {source_dir} -> {target_dir}")
    shutil.copytree(source_dir, target_dir)
    return True

def main():
    """
    批量处理所有对象目录
    """
    # simpleMLP项目的根目录
    root_dir = "/hy-tmp/PartField_Sketch_simpleMLP/javascript/urdf"
    
    # 查找所有数字ID文件夹
    object_ids = []
    for item in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, item)) and item.isdigit():
            object_ids.append(item)
    
    object_ids.sort(key=int)
    print(f"找到 {len(object_ids)} 个对象目录")
    
    # 处理所有目录
    success_count = 0
    for object_id in tqdm(object_ids, desc="处理对象"):
        try:
            if process_object_dir(object_id):
                success_count += 1
        except Exception as e:
            print(f"处理 ID {object_id} 时出错: {e}")
    
    print(f"处理完成! 成功处理 {success_count}/{len(object_ids)} 个对象")

if __name__ == "__main__":
    main()
