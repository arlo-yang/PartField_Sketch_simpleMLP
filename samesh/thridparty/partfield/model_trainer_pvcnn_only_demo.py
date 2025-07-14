#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PartField 特征提取模型和推理系统
====================================================
该文件同时包含模型定义和推理入口，可以直接运行进行特征提取

**新增功能**
------------
- 自动批量遍历 `/hy-tmp/PartField_Sketch/segmentation/data/urdf/*/angle_geometry/*.obj`
  按 `{id}_{view}.obj` 命名规则处理全部几何体
- 结果保存到 `/hy-tmp/PartField_Sketch/segmentation/data/urdf_angle/{id}/`
  目录下（.npy / .ply），文件名与原逻辑一致
- 若仍希望使用旧的单文件 Demo 流程，可运行：
  `python model_trainer_pvcnn_only_demo.py --demo_single`

用法示例（批量模式，自动）:
    cd /hy-tmp/PartField_Sketch/samesh
    python thridparty/partfield/model_trainer_pvcnn_only_demo.py \
      -c thridparty/configs/final/demo.yaml \
      --opts \
      continue_ckpt thridparty/model/model_objaverse.ckpt \
      result_name outputs/partfield \
      dataset.data_path assets/ \
      is_pc False \
      vertex_feature False
"""

# --------------------------------------------------
# 1. 兼容性与运行环境修复
# --------------------------------------------------
import sys
from pathlib import Path

_THIS_FILE      = Path(__file__).resolve()
_THRIDPARTY_DIR = _THIS_FILE.parents[1]                 # .../samesh/thridparty
_PROJECT_ROOT   = _THIS_FILE.parents[2]                 # .../PartField_Sketch
for _p in (_THRIDPARTY_DIR, _PROJECT_ROOT):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

# --------------------------------------------------
# 2. 标准库 & 第三方库
# --------------------------------------------------
import os
import gc
import glob
import json
import random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
import trimesh
from plyfile import PlyData, PlyElement
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

# --------------------------------------------------
# 2.1  批处理相关路径常量  (***新增***)
# --------------------------------------------------
URDF_DIR   = "/hy-tmp/PartField_Sketch/segmentation/data/urdf"
OUTPUT_DIR = "/hy-tmp/PartField_Sketch/segmentation/data/urdf_angle"

# --------------------------------------------------
# 3. 项目内部模块（均从 thridparty/partfield 开始）
# --------------------------------------------------
from partfield.dataloader import (
    Demo_Dataset,
    Demo_Remesh_Dataset,
    Correspondence_Demo_Dataset,
)
from partfield.model.PVCNN.encoder_pc import (
    TriPlanePC2Encoder,
    sample_triplane_feat,
)
from partfield.model.UNet.model import ResidualUNet3D
from partfield.model.triplane import TriplaneTransformer, get_grid_coord
from partfield.model.model_utils import VanillaMLP
from partfield.config import default_argument_parser, setup

# --------------------------------------------------
# 4. LightningModule 定义
# --------------------------------------------------
class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Lightning 关闭自动优化
        self.automatic_optimization = False

        # --------- 三平面 Transformer 相关 ---------
        self.triplane_resolution      = cfg.triplane_resolution
        self.triplane_channels_low    = cfg.triplane_channels_low
        self.triplane_transformer = TriplaneTransformer(
            input_dim            = cfg.triplane_channels_low * 2,
            transformer_dim      = 1024,
            transformer_layers   = 6,
            transformer_heads    = 8,
            triplane_low_res     = 32,
            triplane_high_res    = 128,
            triplane_dim         = cfg.triplane_channels_high,
        )

        # --------- 解码/回归网络 ---------
        self.sdf_decoder = VanillaMLP(
            input_dim        = 64,
            output_dim       = 1,
            out_activation   = "tanh",
            n_neurons        = 64,
            n_hidden_layers  = 6,
        )

        self.use_pvcnn  = cfg.use_pvcnnonly
        self.use_2d_feat = cfg.use_2d_feat
        if self.use_pvcnn:
            self.pvcnn = TriPlanePC2Encoder(
                cfg.pvcnn,
                device      = "cuda",
                shape_min   = -1,
                shape_length= 2,
                use_2d_feat = self.use_2d_feat,
            )

        # 可学习 scaling
        self.logit_scale = nn.Parameter(torch.tensor([1.0], requires_grad=True))

        # 三维网格坐标
        self.grid_coord = get_grid_coord(256)

        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.l1_loss  = nn.L1Loss(reduction="none")

        # 可选 2D 特征回归
        if cfg.regress_2d_feat:
            self.feat_decoder = VanillaMLP(
                input_dim       = 64,
                output_dim      = 192,
                out_activation  = "GELU",
                n_neurons       = 64,
                n_hidden_layers = 6,
            )

    # --------------------- Dataloader ---------------------
    def predict_dataloader(self):
        if self.cfg.remesh_demo:
            dataset = Demo_Remesh_Dataset(self.cfg)
        elif self.cfg.correspondence_demo:
            dataset = Correspondence_Demo_Dataset(self.cfg)
        else:
            dataset = Demo_Dataset(self.cfg)

        return DataLoader(
            dataset,
            num_workers   = self.cfg.dataset.val_num_workers,
            batch_size    = self.cfg.dataset.val_batch_size,
            shuffle       = False,
            pin_memory    = True,
            drop_last     = False,
        )

    # --------------------- 预测流程 ---------------------
    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        # 直接使用用户指定的result_name作为输出路径
        save_dir = self.cfg.result_name
        os.makedirs(save_dir, exist_ok=True)

        uid     = batch["uid"][0]
        view_id = 0
        starttime = time.time()

        # 跳过特定模型
        if uid in {"car", "complex_car"}:
            print("Skipping this for now:", uid)
            return

        # 已处理则跳过
        already = (
            os.path.exists(f"{save_dir}/part_feat_{uid}_{view_id}.npy")
            or os.path.exists(f"{save_dir}/part_feat_{uid}_{view_id}_batch.npy")
        )
        if already:
            print("Already processed", uid)
            return

        # --------------------------------------------------
        # 3D Feature Extraction
        # --------------------------------------------------
        if self.use_2d_feat:
            print("ERROR: Dataloader not implemented with input 2d feat.")
            sys.exit(1)
        else:
            pc_feat = self.pvcnn(batch["pc"], batch["pc"])
            print(f"PVCNN 输出三平面特征维度: {pc_feat.shape}")  # [B, 3, 256, 128, 128]

        # --------------------- Transformer 增强特征 ---------------------
        planes = self.triplane_transformer(pc_feat)  # [B, 3, 512, 128, 128]
        print(f"Transformer 增强后特征维度: {planes.shape}")

        # 切分 SDF / Part 特征
        sdf_planes, part_planes = torch.split(
            planes, [64, planes.shape[2] - 64], dim=2
        )
        print(f"SDF 特征维度: {sdf_planes.shape}")
        print(f"部件特征维度: {part_planes.shape}")

        # --------------------- 点云 / 网格 处理 ---------------------
        if self.cfg.is_pc:
            self._process_pointcloud(
                batch, part_planes, uid, view_id, save_dir
            )
        else:
            self._process_mesh(
                batch, part_planes, uid, view_id, save_dir, starttime
            )
        return

    # ==========================================================
    # 内部辅助函数
    # ==========================================================
    def _process_pointcloud(self, batch, part_planes, uid, view_id, save_dir):
        tensor_vertices = (
            batch["pc"].reshape(1, -1, 3).to(torch.float16).cuda()
        )
        point_feat = sample_triplane_feat(part_planes, tensor_vertices)
        point_feat = point_feat.cpu().numpy().reshape(-1, 448)
        np.save(f"{save_dir}/part_feat_{uid}_{view_id}.npy", point_feat)
        print(f"Exported part_feat_{uid}_{view_id}.npy")

        # ---------- PCA 可视化 ----------
        self._export_pca_pointcloud(
            points=batch["pc"].squeeze().cpu().numpy(),
            features=point_feat,
            filename=f"{save_dir}/feat_pca_{uid}_{view_id}.ply",
        )

    def _process_mesh(
        self, batch, part_planes, uid, view_id, save_dir, starttime
    ):
        use_cuda_version = True
        if use_cuda_version:
            point_feat = self._mesh_cuda(
                batch, part_planes, uid, view_id
            )
        else:
            point_feat = self._mesh_cpu(batch, part_planes)

        # ---------- 保存 ----------
        np.save(f"{save_dir}/part_feat_{uid}_{view_id}_batch.npy", point_feat)
        print(f"Exported part_feat_{uid}_{view_id}.npy")
        print("Time elapsed for feature prediction:",
              time.time() - starttime)

        # ---------- PCA 可视化 ----------
        self._export_pca_mesh(
            vertices=batch["vertices"][0].cpu().numpy(),
            faces=batch["faces"][0].cpu().numpy(),
            features=point_feat,
            filename=f"{save_dir}/feat_pca_{uid}_{view_id}.ply",
            vertex_mode=self.cfg.vertex_feature,
        )

    # ---- CUDA 版本网格处理 ----
    def _mesh_cuda(self, batch, part_planes, uid, view_id):
        def sample_points(vertices, faces, n_per_face):
            n_f = faces.shape[0]
            u = torch.sqrt(torch.rand((n_f, n_per_face, 1),
                                      device=vertices.device))
            v = torch.rand((n_f, n_per_face, 1), device=vertices.device)
            w0 = 1 - u
            w1 = u * (1 - v)
            w2 = u * v
            fv0 = vertices[faces[:, 0]]
            fv1 = vertices[faces[:, 1]]
            fv2 = vertices[faces[:, 2]]
            return w0 * fv0.unsqueeze(1) + w1 * fv1.unsqueeze(1) + w2 * fv2.unsqueeze(1)

        def sample_and_average(part_planes, tensor_vertices, n_per_face):
            n_total = tensor_vertices.shape[1]
            n_each  = self.cfg.n_sample_each
            n_batches = n_total // n_each + 1
            all_feats = []
            for i in range(n_batches):
                s, e = i * n_each, min((i + 1) * n_each, n_total)
                verts = tensor_vertices[:, s:e]
                feats = sample_triplane_feat(part_planes, verts)
                feats = feats.reshape(1, -1, n_per_face, feats.shape[-1]).mean(-2)
                all_feats.append(feats)
            return torch.cat(all_feats, dim=1)

        if self.cfg.vertex_feature:
            verts = batch["vertices"][0].reshape(1, -1, 3).float()
            feats = sample_and_average(part_planes, verts, 1)
        else:
            n_per_face = self.cfg.n_point_per_face
            verts = sample_points(
                batch["vertices"][0], batch["faces"][0], n_per_face
            ).reshape(1, -1, 3).float()
            feats = sample_and_average(part_planes, verts, n_per_face)

        return feats.reshape(-1, 448).cpu().numpy()

    # ---- CPU 版本网格处理（慢）----
    def _mesh_cpu(self, batch, part_planes):
        V = batch["vertices"][0].cpu().numpy()
        F = batch["faces"][0].cpu().numpy()
        n_per_face = self.cfg.n_point_per_face

        feats = []
        for face in F:
            v0, v1, v2 = V[face]
            u = np.random.rand(n_per_face, 1)
            v = np.random.rand(n_per_face, 1)
            swap = (u + v) > 1
            u[swap] = 1 - u[swap]
            v[swap] = 1 - v[swap]
            w = 1 - u - v
            points = u * v0 + v * v1 + w * v2
            verts = torch.from_numpy(points).reshape(1, -1, 3).float().cuda()
            feat = sample_triplane_feat(part_planes, verts).mean(1).cpu().numpy()
            feats.append(feat)
        return np.array(feats).reshape(-1, 448)

    # ---------- PCA 可视化：点云 ----------
    def _export_pca_pointcloud(self, points, features, filename):
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        colors   = PCA(n_components=3).fit_transform(features)
        colors   = (colors - colors.min()) / (colors.max() - colors.min())
        colors   = (colors * 255).astype(np.uint8)

        vertex_data = np.array(
            [(*p, *c) for p, c in zip(points, colors)],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        el = PlyElement.describe(vertex_data, "vertex")
        PlyData([el], text=True).write(filename)
        print(f"Saved PLY file: {filename}")

    # ---------- PCA 可视化：网格 ----------
    def _export_pca_mesh(
        self, vertices, faces, features, filename, vertex_mode=False
    ):
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        colors   = PCA(n_components=3).fit_transform(features)
        colors   = (colors - colors.min()) / (colors.max() - colors.min())
        colors   = (colors * 255).astype(np.uint8)

        if vertex_mode:
            mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces,
                vertex_colors=colors, process=False
            )
        else:
            mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces,
                face_colors=colors, process=False
            )
        mesh.export(filename)
        print(f"Saved PLY file: {filename}")

# --------------------------------------------------
# 5. Lightning-based单文件 Demo 流程（保留原样）
# --------------------------------------------------
def predict(cfg):
    seed_everything(cfg.seed)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    ckpt_cb = ModelCheckpoint(
        monitor              = "train/current_epoch",
        dirpath              = cfg.output_dir,
        filename             = "{epoch:02d}",
        save_top_k           = 100,
        save_last            = True,
        every_n_epochs       = cfg.save_every_epoch,
        mode                 = "max",
        verbose              = True,
    )

    trainer = Trainer(
        devices              = -1,
        accelerator          = "gpu",
        precision            = "16-mixed",
        strategy             = DDPStrategy(find_unused_parameters=True),
        max_epochs           = cfg.training_epochs,
        log_every_n_steps    = 1,
        limit_train_batches  = 3500,
        callbacks            = ckpt_cb,
        default_root_dir     = cfg.result_name,  # 指定Lightning日志输出路径
    )

    if cfg.remesh_demo:
        cfg.n_point_per_face = 10

    model = Model(cfg)
    trainer.predict(model, ckpt_path=cfg.continue_ckpt)

# --------------------------------------------------
# 6. ***新增*** 批量遍历 urdf/angle_geometry 并提特征
# --------------------------------------------------
@torch.no_grad()
def batch_predict_angles(cfg):
    """
    遍历 /segmentation/data/urdf/*/angle_geometry/{id}_{view}.obj
    逐个调用 Model.predict_step()，结果存入 /urdf_angle/{id}/
    """
    # ------- 模型 -------
    model = Model(cfg)
    print("Loading checkpoint:", cfg.continue_ckpt)
    ckpt   = torch.load(cfg.continue_ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.cuda().eval()

    # ------- 遍历几何体 -------
    id_dirs = sorted(glob.glob(os.path.join(URDF_DIR, "*")))
    for id_dir in id_dirs:
        id_name = Path(id_dir).name
        angle_dir = os.path.join(id_dir, "angle_geometry")
        if not os.path.isdir(angle_dir):
            continue

        out_root = os.path.join(OUTPUT_DIR, id_name)
        os.makedirs(out_root, exist_ok=True)

        obj_paths = sorted(glob.glob(os.path.join(angle_dir, f"{id_name}_*.obj")))
        if not obj_paths:
            print("No geometry in", angle_dir)
            continue

        for obj_path in obj_paths:
            view_name = Path(obj_path).stem.split("_", 1)[1]
            uid = f"{id_name}_{view_name}"
            save_dir = out_root
            model.cfg.result_name = save_dir  # 动态修改保存路径

            # --- 构造 batch ---
            mesh = trimesh.load(obj_path, process=False)
            pc   = mesh.sample(8192)  # 8k 点用于 PVCNN
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")
            faces    = torch.tensor(mesh.faces,    dtype=torch.long,   device="cuda")

            batch = {
                "uid"     : [uid],
                "pc"      : torch.tensor(pc, dtype=torch.float32, device="cuda").unsqueeze(0),
                "vertices": [vertices],
                "faces"   : [faces],
            }

            print(f"Processing {uid} -> {save_dir}")
            model.predict_step(batch, 0)
            torch.cuda.empty_cache(); gc.collect()

# --------------------------------------------------
# 7. 主入口 (批量 vs 单文件 Demo)
# --------------------------------------------------
def main():
    parser = default_argument_parser()
    parser.add_argument(
        "--demo_single",
        action="store_true",
        help="Use original Lightning Demo (no batch angle processing)",
    )
    args   = parser.parse_args()
    cfg    = setup(args, freeze=False)

    if args.demo_single:
        predict(cfg)                    # 旧流程
    else:
        batch_predict_angles(cfg)       # 新批量流程

if __name__ == "__main__":
    main()
