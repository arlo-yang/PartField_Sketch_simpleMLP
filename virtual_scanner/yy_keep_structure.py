#!/usr/bin/env python
"""Extract outer shell of a mesh by keeping only faces hit by Virtual‑Scanner
   point cloud.  **多进程优化版本**：内存管理 + 错误恢复 + 智能调度 + 完全保持原始文件结构。

   步骤：
   1. 加载原网格（顶点、面片索引保持不变）。
   2. Virtual‑Scanner 输出点云。
   3. 最近点查询 → 标记被至少一个点命中的面片。
   4. 在克隆网格上删除未命中的面片；不做任何其他拓扑清理。
   5. 直接从原始OBJ文件拷贝，保留所有内容（顶点、材质等）不变，只过滤掉未命中的面片行。
"""

import os
import glob
import subprocess
import numpy as np
from tqdm import tqdm
import trimesh
import trimesh.proximity
import gc
import time
import multiprocessing as mp
import psutil
import traceback
import heapq
import shutil
import re

# ---------------- 参数配置 ---------------- #
BASE_DIR       = "/home/ipab-graphics/workplace/PartField_Sketch_mlp"
INPUT_DIR      = os.path.join(BASE_DIR, "data_small/urdf")
OUTPUT_DIR     = os.path.join(BASE_DIR, "data_small/urdf_shell")
TMP_DIR        = "/tmp/virtualscanner_tmp"
SCANNER_PATH   = "./build/virtualscanner"
VIEW_NUM       = 60
NORMALIZE      = False
MAX_DISTANCE   = 0.001
BASE_BATCH_SIZE = 50_000          # 基础批处理大小，会根据系统内存动态调整
MAX_RETRIES    = 2                # 失败重试次数
NUM_PROCESSES  = max(1, min(mp.cpu_count() - 2, 8))  # 保留2个核心给系统
MAX_MEM_PERCENT = 80              # 单个进程最多使用系统内存的百分比
# ----------------------------------------- #

# 限制 OpenMP / MKL 线程数，防止单进程仍意外挂很多线程
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def format_time(seconds):
    """将秒数格式化为易读形式"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.1f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}时{minutes}分{secs:.1f}秒"


def format_bytes(size):
    """将字节数转换为易读格式"""
    power = 2**10
    n = 0
    power_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.1f} {power_labels[n]}"


def virtual_scan(src: str, ply_dst: str) -> bool:
    """调用 VirtualScanner，并把生成的 PLY 移到 ply_dst。"""
    cmd = [
        SCANNER_PATH,
        src,
        str(VIEW_NUM),
        "0",
        "1" if NORMALIZE else "0"
    ]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[VirtualScanner] 失败 code={ret.returncode} → {src}")
        return False

    gen_ply = src.replace(".obj", ".ply").replace(".off", ".ply")
    if not os.path.exists(gen_ply):
        print(f"[VirtualScanner] 未找到输出 {gen_ply}")
        return False
    os.rename(gen_ply, ply_dst)
    return True


def get_dynamic_batch_size(points_size, mesh_size):
    """根据点云大小和网格大小动态调整批处理大小"""
    # 获取当前可用内存
    available_mem = psutil.virtual_memory().available
    
    # 根据模型大小动态调整批处理大小
    # 估计每个点需要消耗的内存 (坐标和结果)
    mem_per_point = 8 * 3 + 8 * 3  # 大约每个点约50字节
    
    # 每个网格顶点和面的内存消耗
    mesh_mem = mesh_size * 100  # 粗略估计
    
    # 可用于点处理的内存 (留出20%的余量)
    usable_mem = (available_mem * MAX_MEM_PERCENT / 100) - mesh_mem
    
    # 计算可以处理的点数
    max_points = int(usable_mem / mem_per_point)
    
    # 确保批处理大小不会太小或太大
    batch_size = max(10_000, min(max_points, BASE_BATCH_SIZE * 4))
    
    return batch_size


def filter_obj_faces(input_path, output_path, hit_mask):
    """从原始OBJ直接复制内容，将未命中的面片注释掉而不是删除，保持面片ID不变

    Args:
        input_path: 原始OBJ文件路径
        output_path: 输出OBJ文件路径  
        hit_mask: 面片是否命中的布尔数组
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 读取原始OBJ文件
    with open(input_path, 'r') as fin:
        lines = fin.readlines()
    
    # 找出所有的face行
    face_lines = []
    face_indices = []
    for i, line in enumerate(lines):
        if line.strip().startswith('f '):
            face_lines.append(line)
            face_indices.append(i)
    
    # 确保面片数量与hit_mask一致
    if len(face_lines) != len(hit_mask):
        raise ValueError(f"面片数量不匹配: OBJ中有{len(face_lines)}个面片，但hit_mask中有{len(hit_mask)}个")
    
    # 创建新的OBJ文件，保留所有内容，但将未命中的面片注释掉
    with open(output_path, 'w') as fout:
        face_idx = 0
        for i, line in enumerate(lines):
            if i in face_indices:  # 这是一个面片行
                if hit_mask[face_idx]:  # 这个面片被命中
                    fout.write(line)
                else:  # 这个面片未被命中，将其注释掉
                    fout.write(f"# UNHIT_FACE {line}")  # 注释掉这一行，但保持在文件中
                face_idx += 1
            else:  # 非面片行，直接写入
                fout.write(line)


def process(job_data):
    """对单个模型执行外壳提取，成功返回 True。
    job_data: (model_id, retry_count)
    """
    model_id, retry_count = job_data
    
    # 构建输入输出路径
    in_dir = os.path.join(INPUT_DIR, model_id)
    out_dir = os.path.join(OUTPUT_DIR, model_id)
    model_path = os.path.join(in_dir, "yy_merged.obj")
    out_path = os.path.join(out_dir, "yy_merged.obj")
    ply_path = os.path.join(TMP_DIR, f"{model_id}_temp.ply")

    # 跳过已处理的模型
    if os.path.exists(out_path):
        print(f"[Skip] {model_id}: 已存在输出文件")
        return True, model_id, 0  # 添加处理时间为0
        
    # 检查输入文件是否存在
    if not os.path.exists(model_path):
        print(f"[Error] {model_id}: 输入文件不存在: {model_path}")
        return False, model_id, 0
        
    process_start_time = time.time()
    
    try:
        # 1. 加载网格，确保不处理/不变更顶点和面片
        try:
            mesh = trimesh.load(model_path, force="mesh", process=False)
        except Exception as e:
            print(f"[Error] 读取失败 {model_path}: {e}")
            return False, model_id, 0
        
        if not mesh.faces.size:
            print(f"[Error] 空网格 {model_path}")
            return False, model_id, 0
        
        mesh_size = len(mesh.vertices) + len(mesh.faces)
        
        # 2. 生成点云
        if not virtual_scan(model_path, ply_path):
            return False, model_id, 0

        # 3. 读取点云
        try:
            cloud  = trimesh.load(ply_path, force="point")
            points = np.asarray(cloud.vertices)
            if not points.size:
                print(f"[Error] 点云为空 {ply_path}")
                # 清理临时文件
                if os.path.exists(ply_path):
                    os.remove(ply_path)
                return False, model_id, 0
        except Exception as e:
            print(f"[Error] 读取点云失败 {ply_path}: {e}")
            # 清理临时文件
            if os.path.exists(ply_path):
                os.remove(ply_path)
            return False, model_id, 0

        # 4. 最近点查询
        start = time.time()
        
        # 动态确定批处理大小
        batch_size = get_dynamic_batch_size(len(points), mesh_size)
        if len(mesh.faces) > 100000 or len(points) > 500000:
            print(f"[Info] {model_id}: 大模型处理 ({len(points):,} 点, {len(mesh.faces):,} 面), "
                  f"批处理大小: {batch_size:,}")
        
        # 使用trimesh原生方法的精确距离计算
        hit_mask = np.zeros(len(mesh.faces), dtype=bool)
        for i in range(0, len(points), batch_size):
            sub = points[i:i+batch_size]
            _, dist, face_idx = trimesh.proximity.closest_point(mesh, sub)
            hit_mask[face_idx[dist <= MAX_DISTANCE]] = True
            # 主动释放临时数组，防止内存攀升
            del dist, face_idx
            gc.collect()

        hit_faces = int(hit_mask.sum())
        total     = len(mesh.faces)
        proc_time = time.time() - start
        
        # 5. 直接从原始OBJ文件拷贝，保留所有内容，只过滤掉未命中的面片
        try:
            # 创建输出目录
            os.makedirs(out_dir, exist_ok=True)
            
            # 直接处理原始OBJ文件
            filter_obj_faces(model_path, out_path, hit_mask)
            
            # 验证生成的文件
            if os.path.exists(out_path):
                # 计算输出文件大小变化
                orig_size = os.path.getsize(model_path)
                new_size = os.path.getsize(out_path)
                size_change = (new_size - orig_size) / orig_size * 100
                
                print(f"[Hit] {model_id}: {hit_faces}/{total} 面片保留 (其余面片已注释) "
                      f"({hit_faces/total*100:.2f}%), 文件大小变化: {size_change:.1f}%, "
                      f"用时={format_time(proc_time)}")
            else:
                print(f"[Error] 未能创建输出文件 {out_path}")
                return False, model_id, 0
                
        except Exception as e:
            print(f"[Error] 保存失败 {out_path}: {e}")
            traceback.print_exc()
            # 清理临时文件
            if os.path.exists(ply_path):
                os.remove(ply_path)
            return False, model_id, 0

        # 清理内存和临时文件
        del mesh, points, hit_mask
        gc.collect()
        
        # 删除临时PLY文件
        if os.path.exists(ply_path):
            os.remove(ply_path)
        
        total_process_time = time.time() - process_start_time
        return True, model_id, total_process_time
        
    except Exception as e:
        print(f"[Error] 处理 {model_id} 时发生异常: {e}")
        traceback.print_exc()
        
        # 清理临时文件
        if os.path.exists(ply_path):
            os.remove(ply_path)
            
        # 如果重试次数未用完，将任务重新加入队列
        if retry_count < MAX_RETRIES:
            print(f"[Retry] 将重试处理 {model_id} ({retry_count + 1}/{MAX_RETRIES})")
            return None, (model_id, retry_count + 1), 0
        else:
            print(f"[Failed] {model_id} 已达最大重试次数")
            return False, model_id, 0


def find_model_ids():
    """查找所有包含yy_merged.obj的文件夹ID"""
    model_ids = []
    
    # 遍历INPUT_DIR下的所有子目录
    for item in os.listdir(INPUT_DIR):
        dir_path = os.path.join(INPUT_DIR, item)
        if os.path.isdir(dir_path):
            obj_path = os.path.join(dir_path, "yy_merged.obj")
            if os.path.exists(obj_path):
                model_ids.append(item)
    
    return model_ids


def estimate_model_complexity(model_id):
    """估算模型的复杂度（面片数），用于任务排序"""
    model_path = os.path.join(INPUT_DIR, model_id, "yy_merged.obj")
    
    try:
        # 尝试只读取元数据而不加载完整网格
        mesh = trimesh.load(model_path, force="mesh", process=False)
        complexity = len(mesh.faces)
        file_size = os.path.getsize(model_path)
        return complexity, file_size
    except:
        # 如果无法读取，使用文件大小作为复杂度估计
        try:
            return 0, os.path.getsize(model_path)
        except:
            return 0, 0


def init_worker():
    """初始化工作进程"""
    # 显示工作进程信息
    worker_id = mp.current_process().name
    print(f"[Worker] {worker_id} (PID={os.getpid()}) 已启动")


def main() -> None:
    # 创建临时目录
    os.makedirs(TMP_DIR, exist_ok=True)
    
    # 查找所有模型ID
    model_ids = find_model_ids()
    
    if not model_ids:
        print(f"[Info] {INPUT_DIR} 中未找到yy_merged.obj文件")
        return

    # 显示系统信息
    mem_info = psutil.virtual_memory()
    print(f"[System] CPU: {mp.cpu_count()} 核心, "
          f"内存: {mem_info.total/1024**3:.1f}GB 总计, {mem_info.available/1024**3:.1f}GB 可用")
    print(f"[Config] 视角数量: {VIEW_NUM}, 最大距离: {MAX_DISTANCE}, "
          f"基础批处理大小: {BASE_BATCH_SIZE}, 进程数: {NUM_PROCESSES}")
    print(f"[Input] {INPUT_DIR}")
    print(f"[Output] {OUTPUT_DIR}")
    print(f"[Mode] 直接处理原始OBJ文件，保持顶点结构和面片ID完全一致 (未命中面片以注释形式保留)")
    
    # 按大小排序模型，优先处理大模型（防止最后只有一个大模型在处理）
    print(f"[Start] 排序 {len(model_ids)} 个模型...")
    model_sizes = []
    for model_id in model_ids:
        complexity, file_size = estimate_model_complexity(model_id)
        model_sizes.append((model_id, complexity, file_size))
    
    # 按复杂度（面片数）降序排序，如果复杂度相同则按文件大小排序
    model_sizes.sort(key=lambda x: (x[1], x[2]), reverse=True)
    sorted_models = [(model_id, 0) for model_id, _, _ in model_sizes]  # (model_id, retry_count)
    
    # 显示最大的几个模型
    print(f"[Info] 最大的3个模型:")
    for i, (model_id, faces, size) in enumerate(model_sizes[:min(3, len(model_sizes))], 1):
        print(f"  {i}. {model_id}: {faces:,} 面片, {format_bytes(size)}")
    
    # 使用进程池处理模型
    print(f"[Start] {len(model_ids)} 个模型，{NUM_PROCESSES} 进程并行处理 …")
    t_start = time.time()
    results = []
    processing_times = []  # 记录每个模型的处理时间
    tasks_completed = 0
    tasks_pending = sorted_models.copy()
    
    try:
        with mp.Pool(processes=NUM_PROCESSES, initializer=init_worker) as pool:
            # 初始提交任务
            pending_results = []
            for _ in range(min(NUM_PROCESSES * 2, len(tasks_pending))):
                if tasks_pending:
                    task = tasks_pending.pop(0)
                    res = pool.apply_async(process, args=(task,))
                    pending_results.append(res)
            
            # 处理完成的任务并动态提交新任务
            pbar = tqdm(total=len(model_ids), unit="个")
            while pending_results:
                # 检查已完成的任务
                still_pending = []
                for res in pending_results:
                    if res.ready():
                        try:
                            success, data, proc_time = res.get()
                            if success is None:  # 需要重试
                                tasks_pending.append(data)  # data 是 (model_id, retry_count+1)
                            else:
                                results.append(success)
                                if proc_time > 0:  # 只添加真正处理过的模型时间
                                    processing_times.append(proc_time)
                                tasks_completed += 1
                                pbar.update(1)
                        except Exception as e:
                            print(f"[Error] 任务异常: {e}")
                    else:
                        still_pending.append(res)
                
                # 提交新任务
                while len(still_pending) < NUM_PROCESSES * 2 and tasks_pending:
                    task = tasks_pending.pop(0)
                    res = pool.apply_async(process, args=(task,))
                    still_pending.append(res)
                
                # 更新pending_results并等待一小段时间
                pending_results = still_pending
                time.sleep(0.1)
            
            pbar.close()
    except KeyboardInterrupt:
        print("\n[Interrupted] 用户中断，正在停止...")
    except Exception as e:
        print(f"[Error] 多进程执行出错: {e}")
        traceback.print_exc()
    
    success_count = sum(1 for r in results if r)
    
    # 计算总墙钟时间（从开始到结束）
    wall_time = time.time() - t_start
    
    # 计算总处理时间（所有模型处理时间的总和）
    total_processing_time = sum(processing_times)
    
    # 计算平均处理时间（单个模型的平均处理时间）
    avg_processing_time = total_processing_time / len(processing_times) if processing_times else 0
    
    # 计算加速比（顺序执行时间 / 并行执行时间）
    speedup = total_processing_time / wall_time if wall_time > 0 else 0
    
    print(f"[Done] 成功 {success_count}/{len(model_ids)}")
    print(f"  - 总墙钟时间: {format_time(wall_time)}")
    print(f"  - 总累计处理时间: {format_time(total_processing_time)} (相当于单进程)")
    print(f"  - 平均每个模型: {format_time(avg_processing_time)}")
    print(f"  - 并行加速比: {speedup:.1f}x")
    print(f"  - 输出目录: {OUTPUT_DIR}")
    
    # 清理临时目录
    try:
        shutil.rmtree(TMP_DIR)
        print(f"[Cleanup] 已删除临时目录: {TMP_DIR}")
    except Exception as e:
        print(f"[Warning] 清理临时目录失败: {e}")


if __name__ == "__main__":
    try:
        # 添加依赖检查
        import psutil
    except ImportError as e:
        print(f"[警告] 缺少依赖库: {e}")
        print("       请安装: pip install psutil")
        print("       继续执行但无法获取系统资源信息...")
    main()
