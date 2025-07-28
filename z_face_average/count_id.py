#!/usr/bin/env python3
"""
扫描 urdf 目录下的所有模型ID，生成包含所有ID的txt文件
"""

import os
import glob

# 配置路径
BASE_DIR = "/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP"
URDF_DIR = os.path.join(BASE_DIR, "data_small_copy/urdf")
OUTPUT_FILE = os.path.join(BASE_DIR, "model_ids.txt")

def collect_model_ids():
    """收集所有模型ID"""
    model_ids = []
    
    if not os.path.exists(URDF_DIR):
        print(f"[Error] 目录不存在: {URDF_DIR}")
        return model_ids
    
    # 遍历urdf目录下的所有子目录
    for item in os.listdir(URDF_DIR):
        item_path = os.path.join(URDF_DIR, item)
        if os.path.isdir(item_path):
            # 检查是否为数字ID（可选验证）
            if item.isdigit():
                model_ids.append(item)
            else:
                print(f"[Warning] 非数字ID: {item}")
                model_ids.append(item)  # 仍然添加，以防有非数字ID
    
    # 按数字排序（如果都是数字的话）
    try:
        model_ids.sort(key=int)
    except ValueError:
        # 如果有非数字ID，就按字符串排序
        model_ids.sort()
    
    return model_ids

def save_ids_to_file(model_ids, output_file):
    """将ID列表保存到txt文件"""
    with open(output_file, 'w') as f:
        for model_id in model_ids:
            f.write(f"{model_id}\n")
    
    print(f"[Success] 已保存 {len(model_ids)} 个模型ID到: {output_file}")

def main():
    print(f"[Start] 扫描目录: {URDF_DIR}")
    
    # 收集模型ID
    model_ids = collect_model_ids()
    
    if not model_ids:
        print("[Error] 未找到任何模型ID")
        return
    
    print(f"[Found] 找到 {len(model_ids)} 个模型ID")
    print(f"[Range] ID范围: {model_ids[0]} - {model_ids[-1]}")
    
    # 显示前几个和后几个ID作为示例
    print(f"[Sample] 前5个ID: {model_ids[:5]}")
    print(f"[Sample] 后5个ID: {model_ids[-5:]}")
    
    # 保存到文件
    save_ids_to_file(model_ids, OUTPUT_FILE)
    
    print(f"[Done] 输出文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
