#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract outer shell of a mesh by keeping only faces hit by Virtual‑Scanner point cloud.
完全功能同之前版本；新增 GPU 依赖检查 & 提示，方便定位为何降级到 CPU。
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

# ---------------- 参数配置 ---------------- #
INPUT_DIR       = "data/original_obj"
OUTPUT_DIR      = "data/shell_obj"
TMP_DIR         = "data/tmp_ply"
SCANNER_PATH    = "./build/virtualscanner"
VIEW_NUM        = 24
NORMALIZE       = False
MAX_DISTANCE    = 0.001
BATCH_SIZE_CPU  = 50_000
BATCH_SIZE_GPU  = 2_000_000
USE_GPU_WANTED  = True       # 你想用 GPU 就设 True
VERBOSE_TIMING  = True
# ----------------------------------------- #

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ---------- GPU 检测 ----------
torch_ok = False
torch_cuda_ok = False
p3d_ok = False
device_str = "cpu"

if USE_GPU_WANTED:
    try:
        import torch
        torch_ok = True
        torch_cuda_ok = torch.cuda.is_available()
        device_str = "cuda:0" if torch_cuda_ok else "cpu"
        try:
            import pytorch3d
            from pytorch3d.structures import Meshes  # noqa
            from pytorch3d.ops import point_mesh_face_distance  # noqa
            p3d_ok = True
        except Exception:
            p3d_ok = False
    except Exception:
        torch_ok = False
        torch_cuda_ok = False
        p3d_ok = False
        device_str = "cpu"

USE_GPU = USE_GPU_WANTED and torch_ok and torch_cuda_ok and p3d_ok

print("=" * 60)
print(f"PyTorch        : {'OK' if torch_ok else 'Not installed'}")
print(f" - CUDA Support: {'Yes' if torch_cuda_ok else 'No'}")
print(f"PyTorch3D      : {'OK' if p3d_ok else 'Not installed'}")
print(f"GPU will be used: {'YES' if USE_GPU else 'NO'}")
print("=" * 60)

# ---------- GPU 计算函数 ----------
def _gpu_face_hits(verts_np, faces_np, points_np, max_distance, batch_size):
    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.ops import point_mesh_face_distance

    device = torch.device("cuda:0")
    verts = torch.as_tensor(verts_np, dtype=torch.float32, device=device)
    faces = torch.as_tensor(faces_np, dtype=torch.int64, device=device)
    mesh  = Meshes(verts=[verts], faces=[faces])

    sq_thr = max_distance * max_distance
    F = faces.shape[0]
    hit_mask = torch.zeros(F, dtype=torch.bool, device=device)

    for start in range(0, len(points_np), batch_size):
        pts_np  = points_np[start:start+batch_size]
        pts_gpu = torch.as_tensor(pts_np, dtype=torch.float32, device=device).unsqueeze(0)
        d2      = point_mesh_face_distance(mesh, pts_gpu)[0]  # (n,)
        close   = d2 <= sq_thr
        if not torch.any(close):
            continue
        # 拿命中点回 CPU，获得最近面片索引
        pts_hit = pts_np[close.cpu().numpy()]
        _, _, face_idx = trimesh.proximity.closest_point(
            trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False),
            pts_hit,
        )
        hit_mask[torch.as_tensor(face_idx, dtype=torch.int64, device=device)] = True

    return hit_mask.cpu().numpy()

def _cpu_face_hits(mesh, points, max_distance, batch_size):
    hit_mask = np.zeros(len(mesh.faces), dtype=bool)
    for i in range(0, len(points), batch_size):
        sub = points[i:i+batch_size]
        _, dist, face_idx = trimesh.proximity.closest_point(mesh, sub)
        hit_mask[face_idx[dist <= max_distance]] = True
        del dist, face_idx
        gc.collect()
    return hit_mask

# ---------- VirtualScanner ----------
def virtual_scan(src, ply_dst):
    cmd = [SCANNER_PATH, src, str(VIEW_NUM), "0", "1" if NORMALIZE else "0"]
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

# ---------- 主逻辑 ----------
def process(model_path):
    name     = os.path.splitext(os.path.basename(model_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{name}.obj")
    ply_path = os.path.join(TMP_DIR,    f"{name}.ply")

    try:
        mesh = trimesh.load(model_path, force="mesh", process=False)
    except Exception as e:
        print(f"[Error] 读取失败 {model_path}: {e}")
        return False
    if not mesh.faces.size:
        print(f"[Error] 空网格 {model_path}")
        return False

    if not virtual_scan(model_path, ply_path):
        return False

    try:
        cloud  = trimesh.load(ply_path, force="point")
        points = np.asarray(cloud.vertices)
        if points.size == 0:
            print(f"[Error] 点云为空 {ply_path}")
            return False
    except Exception as e:
        print(f"[Error] 读取点云失败 {ply_path}: {e}")
        return False

    t0 = time.time()
    if USE_GPU:
        hit_mask = _gpu_face_hits(
            mesh.vertices.view(np.ndarray),
            mesh.faces.view(np.ndarray),
            points,
            MAX_DISTANCE,
            BATCH_SIZE_GPU,
        )
        mode = "GPU"
    else:
        hit_mask = _cpu_face_hits(mesh, points, MAX_DISTANCE, BATCH_SIZE_CPU)
        mode = "CPU"
    t_hit = time.time() - t0

    shell = mesh.copy()
    shell.update_faces(hit_mask)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        shell.export(out_path)
    except Exception as e:
        print(f"[Error] 保存失败 {out_path}: {e}")
        return False

    if VERBOSE_TIMING:
        kept  = int(hit_mask.sum())
        total = len(mesh.faces)
        print(f"[Hit] {name}: {kept}/{total} faces kept ({kept/total*100:.2f}%) "
              f"mode={mode} time={t_hit:.2f}s")
    return True

def main():
    models = glob.glob(os.path.join(INPUT_DIR, "*.obj")) + glob.glob(os.path.join(INPUT_DIR, "*.off"))
    if not models:
        print(f"[Info] {INPUT_DIR} 无模型")
        return

    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[Start] {len(models)} 个模型，{'GPU' if USE_GPU else 'CPU'} 模式 …")
    success = 0
    for m in tqdm(models, unit="个"):
        if process(m):
            success += 1
    print(f"[Done] 成功 {success}，失败 {len(models)-success}，输出 → {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
