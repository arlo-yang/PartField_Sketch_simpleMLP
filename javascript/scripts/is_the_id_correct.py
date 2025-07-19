"""
检查yy_merged.obj和moveable_{joint id}.ply之间的面片/顶点数量和顺序是否一致

使用方法:
python is_the_id_correct.py [--id MODEL_ID] [--joint JOINT_ID]

如果指定了MODEL_ID，则只处理该模型
如果同时指定了JOINT_ID，则只处理该模型的特定关节
否则批量处理所有模型的所有关节
"""

import os
import sys
import glob
import re
import numpy as np
import trimesh
import argparse
from pathlib import Path
from tqdm import tqdm
import datetime

# 全局路径配置
URDF_DIR = "/hy-tmp/PartField_Sketch_simpleMLP/javascript/urdf"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_model_ids():
    """查找所有可用的模型ID"""
    model_dirs = [d for d in os.listdir(URDF_DIR) if os.path.isdir(os.path.join(URDF_DIR, d))]
    return [d for d in model_dirs if d.isdigit()]

def find_joint_ids(model_id):
    """查找指定模型的所有关节ID"""
    visualization_dir = os.path.join(URDF_DIR, model_id, "yy_object", "yy_visualization")
    if not os.path.exists(visualization_dir):
        return []
    
    moveable_files = glob.glob(os.path.join(visualization_dir, "moveable_*.ply"))
    joint_ids = []
    
    for file_path in moveable_files:
        match = re.search(r"moveable_(\d+)\.ply", os.path.basename(file_path))
        if match:
            joint_ids.append(match.group(1))
    
    return joint_ids

def compare_face_vertex_count(mesh1, mesh2, name1, name2):
    """比较两个网格的面片和顶点数量"""
    results = []
    
    # 比较面片数量
    if len(mesh1.faces) != len(mesh2.faces):
        results.append(f"面片数量不一致: {name1}={len(mesh1.faces)}, {name2}={len(mesh2.faces)}")
    else:
        results.append(f"面片数量一致: {len(mesh1.faces)}")
    
    # 比较顶点数量
    if len(mesh1.vertices) != len(mesh2.vertices):
        results.append(f"顶点数量不一致: {name1}={len(mesh1.vertices)}, {name2}={len(mesh2.vertices)}")
    else:
        results.append(f"顶点数量一致: {len(mesh1.vertices)}")
    
    return results

def compare_face_vertex_order(mesh1, mesh2, name1, name2):
    """比较两个网格的面片和顶点顺序"""
    results = []
    
    # 检查面片顺序（比较前20个面片）
    max_check = min(20, len(mesh1.faces), len(mesh2.faces))
    face_order_consistent = True
    first_inconsistent_face = -1
    
    for i in range(max_check):
        if not np.array_equal(mesh1.faces[i], mesh2.faces[i]):
            face_order_consistent = False
            first_inconsistent_face = i
            break
    
    if face_order_consistent:
        results.append(f"面片顺序一致 (已检查前 {max_check} 个面片)")
    else:
        results.append(f"面片顺序不一致，从第 {first_inconsistent_face} 个面片开始不同")
        results.append(f"- {name1} 面片 #{first_inconsistent_face}: {mesh1.faces[first_inconsistent_face]}")
        results.append(f"- {name2} 面片 #{first_inconsistent_face}: {mesh2.faces[first_inconsistent_face]}")
    
    # 检查顶点顺序（比较前10个顶点）
    max_check = min(10, len(mesh1.vertices), len(mesh2.vertices))
    vertex_order_consistent = True
    first_inconsistent_vertex = -1
    
    for i in range(max_check):
        if not np.allclose(mesh1.vertices[i], mesh2.vertices[i], atol=1e-5):
            vertex_order_consistent = False
            first_inconsistent_vertex = i
            break
    
    if vertex_order_consistent:
        results.append(f"顶点顺序一致 (已检查前 {max_check} 个顶点)")
    else:
        results.append(f"顶点顺序不一致，从第 {first_inconsistent_vertex} 个顶点开始不同")
        results.append(f"- {name1} 顶点 #{first_inconsistent_vertex}: {mesh1.vertices[first_inconsistent_vertex]}")
        results.append(f"- {name2} 顶点 #{first_inconsistent_vertex}: {mesh2.vertices[first_inconsistent_vertex]}")
    
    return results

def check_face_properties(mesh):
    """检查网格的面片属性"""
    results = []
    
    # 检查是否有面片颜色
    has_face_colors = hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors')
    if has_face_colors:
        results.append(f"具有面片颜色，颜色数量: {len(mesh.visual.face_colors)}")
        
        # 统计绿色和红色面片
        green_count = 0
        red_count = 0
        for color in mesh.visual.face_colors:
            if color[0] < 50 and color[1] > 200 and color[2] < 50:  # 绿色
                green_count += 1
            elif color[0] > 200 and color[1] < 50 and color[2] < 50:  # 红色
                red_count += 1
        
        results.append(f"绿色面片: {green_count}, 红色面片: {red_count}")
    else:
        results.append("没有面片颜色")
    
    return results

def process_model(model_id, joint_id=None, report_file=None):
    """处理单个模型"""
    print(f"\n处理模型: {model_id}")
    
    # 构建yy_merged.obj文件路径
    obj_path = os.path.join(URDF_DIR, model_id, "yy_object", "yy_merged.obj")
    
    # 检查OBJ文件是否存在
    if not os.path.exists(obj_path):
        error_msg = f"错误：yy_merged.obj文件不存在: {obj_path}"
        print(error_msg)
        if report_file:
            report_file.write(f"模型 {model_id}: {error_msg}\n\n")
        return False
    
    try:
        # 加载OBJ文件
        print(f"加载yy_merged.obj文件...")
        obj_mesh = trimesh.load(obj_path)
        print(f"- OBJ面片数量: {len(obj_mesh.faces)}")
        print(f"- OBJ顶点数量: {len(obj_mesh.vertices)}")
    except Exception as e:
        error_msg = f"加载OBJ文件失败: {e}"
        print(error_msg)
        if report_file:
            report_file.write(f"模型 {model_id}: {error_msg}\n\n")
        return False
    
    # 处理指定关节或所有关节
    if joint_id:
        joint_ids = [joint_id]
    else:
        joint_ids = find_joint_ids(model_id)
    
    if not joint_ids:
        warning_msg = f"警告：找不到模型 {model_id} 的任何关节"
        print(warning_msg)
        if report_file:
            report_file.write(f"模型 {model_id}: {warning_msg}\n\n")
        return False
    
    success = True
    for jid in joint_ids:
        print(f"处理关节: {jid}")
        
        # 构建PLY文件路径
        ply_path = os.path.join(URDF_DIR, model_id, "yy_object", "yy_visualization", f"moveable_{jid}.ply")
        
        # 检查PLY文件是否存在
        if not os.path.exists(ply_path):
            warning_msg = f"警告：moveable_{jid}.ply文件不存在: {ply_path}"
            print(warning_msg)
            if report_file:
                report_file.write(f"模型 {model_id}, 关节 {jid}: {warning_msg}\n\n")
            continue
        
        try:
            # 加载PLY文件
            print(f"加载moveable_{jid}.ply文件...")
            ply_mesh = trimesh.load(ply_path)
            print(f"- PLY面片数量: {len(ply_mesh.faces)}")
            print(f"- PLY顶点数量: {len(ply_mesh.vertices)}")
            
            # 比较面片和顶点数量
            count_results = compare_face_vertex_count(obj_mesh, ply_mesh, "OBJ", "PLY")
            order_results = compare_face_vertex_order(obj_mesh, ply_mesh, "OBJ", "PLY")
            ply_prop_results = check_face_properties(ply_mesh)
            
            # 报告结果
            print("\n比较结果:")
            for result in count_results + order_results + ply_prop_results:
                print(f"- {result}")
            
            # 写入报告
            if report_file:
                report_file.write(f"模型 {model_id}, 关节 {jid}:\n")
                report_file.write("\n面片和顶点数量比较:\n")
                for result in count_results:
                    report_file.write(f"- {result}\n")
                
                report_file.write("\n面片和顶点顺序比较:\n")
                for result in order_results:
                    report_file.write(f"- {result}\n")
                
                report_file.write("\nPLY文件属性:\n")
                for result in ply_prop_results:
                    report_file.write(f"- {result}\n")
                
                report_file.write("\n" + "-"*50 + "\n\n")
            
            # 如果不一致，在总结中标记
            is_consistent = all("一致" in r for r in count_results + order_results)
            if not is_consistent:
                inconsistency_summary = f"模型 {model_id}, 关节 {jid}: 面片/顶点数量或顺序不一致"
                print(f"⚠️ {inconsistency_summary}")
                return False
            
        except Exception as e:
            error_msg = f"处理失败: {e}"
            print(error_msg)
            if report_file:
                report_file.write(f"模型 {model_id}, 关节 {jid}: {error_msg}\n\n")
            success = False
    
    return success

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="比较OBJ和PLY文件的面片/顶点数量和顺序")
    parser.add_argument("--id", type=str, help="指定要处理的模型ID")
    parser.add_argument("--joint", type=str, help="指定要处理的关节ID")
    args = parser.parse_args()
    
    # 创建报告文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(SCRIPT_DIR, f"mesh_comparison_report_{timestamp}.txt")
    inconsistency_path = os.path.join(SCRIPT_DIR, f"inconsistent_models_{timestamp}.txt")
    
    print(f"将保存报告到: {report_path}")
    print(f"将保存不一致模型列表到: {inconsistency_path}")
    
    inconsistent_models = []
    
    with open(report_path, 'w') as report_file:
        report_file.write(f"OBJ和PLY文件比较报告 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_file.write("="*80 + "\n\n")
        
        if args.id:
            # 处理单个模型
            print(f"处理指定模型: {args.id}")
            is_consistent = process_model(args.id, args.joint, report_file)
            if not is_consistent:
                joint_str = f", 关节 {args.joint}" if args.joint else ""
                inconsistent_models.append(f"{args.id}{joint_str}")
        else:
            # 批量处理所有模型
            model_ids = find_model_ids()
            model_ids.sort(key=int)  # 按数字顺序排序
            print(f"找到 {len(model_ids)} 个模型")
            
            consistent_count = 0
            total_processed = 0
            
            for i, model_id in enumerate(tqdm(model_ids, desc="处理模型")):
                print(f"\n[{i+1}/{len(model_ids)}] 处理模型: {model_id}")
                is_consistent = process_model(model_id, None, report_file)
                total_processed += 1
                
                if is_consistent:
                    consistent_count += 1
                else:
                    inconsistent_models.append(model_id)
            
            # 写入总结
            summary = f"\n总结: 处理了 {total_processed} 个模型，其中 {consistent_count} 个一致，{len(inconsistent_models)} 个不一致"
            print(summary)
            report_file.write("\n" + "="*80 + "\n")
            report_file.write(summary + "\n")
    
    # 保存不一致模型列表
    with open(inconsistency_path, 'w') as incon_file:
        incon_file.write("以下模型的OBJ和PLY文件不一致:\n\n")
        for model in inconsistent_models:
            incon_file.write(f"{model}\n")
    
    print(f"报告已保存到: {report_path}")
    print(f"不一致模型列表已保存到: {inconsistency_path}")

if __name__ == "__main__":
    main()