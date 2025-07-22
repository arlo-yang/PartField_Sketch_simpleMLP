# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# # ---------------------------------------------------------------------------
# #                                Imports
# # ---------------------------------------------------------------------------
# import os, sys, re, glob, math, copy, shutil, json, multiprocessing as mp
# from pathlib import Path
# from collections import defaultdict, Counter

# import numpy as np
# import torch
# import torch.nn as nn
# import trimesh
# from PIL import Image
# from omegaconf import OmegaConf
# from trimesh.base import Trimesh, Scene
# from tqdm import tqdm
# from natsort import natsorted
# import matplotlib.pyplot as plt

# # project‑internal
# from samesh.data.loaders import remove_texture, read_mesh
# from samesh.data.common import NumpyTensor
# from samesh.renderer.renderer import Renderer, render_multiview, colormap_faces, colormap_norms
# from samesh.utils.cameras import *
# from samesh.utils.mesh import duplicate_verts
# # from samesh.utils.camera_views import VIEW_POSITIONS, distance  # 不再使用

# # ---------------------------------------------------------------------------
# #                               视角定义
# # ---------------------------------------------------------------------------
# distance = 2.5
# VIEW_POSITIONS = {
#     "top-left": {
#         "position": [-distance * math.cos(math.pi/4) * math.sin(math.pi/4),
#                      distance * math.sin(math.pi/4),
#                      distance * math.cos(math.pi/4) * math.cos(math.pi/4)],
#         "target": [0.0, 0.0, 0.0],
#         "up": [0.0, 1.0, 0.0]
#     },
#     "left": {
#         "position": [-distance * math.cos(0) * math.sin(math.pi/4),
#                      distance * math.sin(0),
#                      distance * math.cos(0) * math.cos(math.pi/4)],
#         "target": [0.0, 0.0, 0.0],
#         "up": [0.0, 1.0, 0.0]
#     },
#     "bottom-left": {
#         "position": [-distance * math.cos(-math.pi/4) * math.sin(math.pi/4),
#                      distance * math.sin(-math.pi/4),
#                      distance * math.cos(-math.pi/4) * math.cos(math.pi/4)],
#         "target": [0.0, 0.0, 0.0],
#         "up": [0.0, 1.0, 0.0]
#     },
#     "top-center": {
#         "position": [-distance * math.cos(math.pi/4) * math.sin(0),
#                      distance * math.sin(math.pi/4),
#                      distance * math.cos(math.pi/4) * math.cos(0)],
#         "target": [0.0, 0.0, 0.0],
#         "up": [0.0, 1.0, 0.0]
#     },
#     "center": {
#         "position": [0.0, 0.0, distance],
#         "target": [0.0, 0.0, 0.0],
#         "up": [0.0, 1.0, 0.0]
#     },
#     "bottom-center": {
#         "position": [-distance * math.cos(-math.pi/4) * math.sin(0),
#                      distance * math.sin(-math.pi/4),
#                      distance * math.cos(-math.pi/4) * math.cos(0)],
#         "target": [0.0, 0.0, 0.0],
#         "up": [0.0, 1.0, 0.0]
#     },
#     "top-right": {
#         "position": [-distance * math.cos(math.pi/4) * math.sin(-math.pi/4),
#                      distance * math.sin(math.pi/4),
#                      distance * math.cos(math.pi/4) * math.cos(-math.pi/4)],
#         "target": [0.0, 0.0, 0.0],
#         "up": [0.0, 1.0, 0.0]
#     },
#     "right": {
#         "position": [-distance * math.cos(0) * math.sin(-math.pi/4),
#                      distance * math.sin(0),
#                      distance * math.cos(0) * math.cos(-math.pi/4)],
#         "target": [0.0, 0.0, 0.0],
#         "up": [0.0, 1.0, 0.0]
#     },
#     "bottom-right": {
#         "position": [-distance * math.cos(-math.pi/4) * math.sin(-math.pi/4),
#                      distance * math.sin(-math.pi/4),
#                      distance * math.cos(-math.pi/4) * math.cos(-math.pi/4)],
#         "target": [0.0, 0.0, 0.0],
#         "up": [0.0, 1.0, 0.0]
#     },
#     "custom": {
#         "position": [0.0, 0.0, distance],
#         "target": [0.0, 0.0, 0.0],
#         "up": [0.0, 1.0, 0.0]
#     }
# }

# # ---------------------------------------------------------------------------
# #                         固定路径配置
# # ---------------------------------------------------------------------------
# PREDICT_DIR = "/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/img_pred"
# URDF_DIR    = "/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/urdf"
# RESULT_DIR  = "/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/result"
# FAILURE_DIR = os.path.join(RESULT_DIR, "failure")
# CONFIG_FILE = Path(__file__).resolve().parent / "configs" / "mesh_segmentation.yaml"

# # ---------------------------------------------------------------------------
# #                    预测文件名解析正则 + 工具函数
# # ---------------------------------------------------------------------------
# _PREDICT_RE = re.compile(
#     r"""^(?P<class>.+?)_(?P<id>\d+)_segmentation_(?P<view>.+?)_joint_(?P<joint>\d+)\.png$""",
#     re.IGNORECASE
# )

# def parse_predict_filename(fname: str):
#     """返回 (class, id, view, joint:int)"""
#     m = _PREDICT_RE.match(os.path.basename(fname))
#     if not m:
#         raise ValueError(f"Invalid predict filename: {fname}")
#     return (
#         m.group("class"),
#         m.group("id"),
#         m.group("view"),
#         int(m.group("joint"))
#     )

# # =============================================================================
# #                       =========  核心函数 =========
# # =============================================================================
# def combine_bmasks(masks: NumpyTensor['n h w'], sort=False) -> NumpyTensor['h w']:
#     mask_combined = np.zeros_like(masks[0], dtype=int)
#     if sort:
#         masks = sorted(masks, key=lambda x: x.sum(), reverse=True)
#     for i, mask in enumerate(masks):
#         mask_combined[mask] = i + 1
#     return mask_combined

# def colormap_mask(
#     mask: NumpyTensor['h w'],
#     image: NumpyTensor['h w 3'] = None,
#     background=np.array([255, 255, 255]),
#     foreground=None,
#     blend=0.25
# ) -> Image.Image:
#     palette = np.random.randint(0, 255, (np.max(mask) + 1, 3))
#     palette[0] = background
#     if foreground is not None:
#         for i in range(1, len(palette)):
#             palette[i] = foreground
#     image_mask = palette[mask.astype(int)]
#     image_blend = image_mask if image is None else image_mask * (1 - blend) + image * blend
#     image_blend = np.clip(image_blend, 0, 255).astype(np.uint8)
#     return Image.fromarray(image_blend)

# def visualize_items(items: dict, path: Path, model_id: str = None, face_labels: dict = None, input_image_path: str = None) -> None:
#     """
#     原可视化函数完全保留 —— 仅移除了与 mesh 切割无关的板块，
#     保证 projection 视图检查仍可用。
#     """
#     os.makedirs(path, exist_ok=True)
#     view_name = items.get('view_names', ['custom'])[0]
#     pred_path = input_image_path
#     faces = items['faces'][0]
#     cmask = items['cmasks'][0]
#     pose = items['poses'][0] if 'poses' in items and len(items['poses']) > 0 else None
#     view_name_formatted = view_name.replace('-', '_')

#     plt.figure(figsize=(24, 12))

#     plt.subplot(1, 3, 1)
#     face_ids_img = np.array(colormap_faces(faces))
#     plt.imshow(face_ids_img)
#     plt.title('Face IDs')
#     plt.axis('off')

#     plt.subplot(1, 3, 2)
#     segmentation_mask_img = np.array(colormap_mask(cmask))
#     plt.imshow(segmentation_mask_img)
#     plt.title('Segmentation Mask')
#     plt.axis('off')

#     plt.subplot(1, 3, 3)
#     if pred_path and os.path.exists(pred_path):
#         prediction_img = np.array(Image.open(pred_path))
#         plt.imshow(prediction_img)
#         plt.title('Input Image')
#     else:
#         plt.text(0.5, 0.5, 'Input Image Not Found',
#                  horizontalalignment='center', verticalalignment='center',
#                  transform=plt.gca().transAxes)
#         plt.title('Input Image')
#     plt.axis('off')

#     plt.tight_layout()
#     combined_path = f'{path}/projection_visualization.png'
#     plt.savefig(combined_path)
#     plt.close()
#     print(f"Projection visualization saved to: {combined_path}")

# def norms_mask(norms: NumpyTensor['h w 3'], cam2world: NumpyTensor['4 4'], threshold=0.0) -> NumpyTensor['h w 3']:
#     lookat = cam2world[:3, :3] @ np.array([0, 0, 1])
#     return np.abs(np.dot(norms, lookat)) > threshold

# def compute_face2label(
#     labels: NumpyTensor['l'],
#     faceid: NumpyTensor['h w'],
#     mask: NumpyTensor['h w'],
#     norms: NumpyTensor['h w 3'],
#     pose: NumpyTensor['4 4'],
#     label_sequence_count: int,
#     threshold_counts: int = 2,
#     threshold_percentage: float = 60.0,
#     use_percentage_threshold: bool = False,
#     view_index: int = None
# ):
#     normal_mask = norms_mask(norms, pose)
#     visible_mask = (faceid != -1)
#     visibility_mask = normal_mask & visible_mask
#     face2label = defaultdict(Counter)
#     mask_info = {}
#     face_total_pixels = {}
#     if use_percentage_threshold:
#         unique_faces, face_counts = np.unique(faceid[visibility_mask], return_counts=True)
#         for face, count in zip(unique_faces, face_counts):
#             if face != -1:
#                 face_total_pixels[int(face)] = count
#     for j, label in enumerate(labels):
#         label_sequence = label_sequence_count + j
#         faces_mask = (mask == label) & visibility_mask
#         faces, counts = np.unique(faceid[faces_mask], return_counts=True)
#         valid_indices = faces != -1
#         faces = faces[valid_indices]
#         counts = counts[valid_indices]
#         if use_percentage_threshold:
#             valid_faces = []
#             for face, count in zip(faces, counts):
#                 face_int = int(face)
#                 if face_int in face_total_pixels:
#                     total = face_total_pixels[face_int]
#                     percentage = (count / total) * 100.0
#                     if percentage >= threshold_percentage:
#                         valid_faces.append(face)
#             valid_faces = np.array(valid_faces)
#         else:
#             valid_faces = faces[counts > threshold_counts]
#         mask_id = f"view{view_index}_mask{j}" if view_index is not None else f"mask{j}"
#         mask_info[mask_id] = {
#             'faces': set(valid_faces.tolist()),
#             'label': int(label),
#             'label_sequence': int(label_sequence),
#             'pixel_count': int(np.sum(faces_mask)),
#             'face_count': len(valid_faces),
#             'is_ground_truth_foreground': j == 0
#         }
#         for face in valid_faces:
#             face2label[int(face)][label_sequence] += np.sum(faces_mask & (faceid == face))
#     return face2label, mask_info

# def custom_look_at(eye, target, up):
#     eye, target, up = map(lambda v: np.array(v, dtype=np.float32), [eye, target, up])
#     z_axis = eye - target; z_axis /= np.linalg.norm(z_axis)
#     x_axis = np.cross(up, z_axis); x_axis /= np.linalg.norm(x_axis)
#     y_axis = np.cross(z_axis, x_axis)
#     rotation = np.identity(4, dtype=np.float32); rotation[0, :3] = x_axis; rotation[1, :3] = y_axis; rotation[2, :3] = z_axis
#     translation = np.identity(4, dtype=np.float32); translation[:3, 3] = -eye
#     return np.linalg.inv(np.matmul(rotation, translation))

# # ---------------------------------------------------------------------------
# #                       SegmentationModelMesh  (移除切割函数)
# # ---------------------------------------------------------------------------
# class SegmentationModelMesh(nn.Module):
#     """
#     只保留可见面片投影到 mask 的流程。
#     """
#     def __init__(self, config: OmegaConf, device='cuda', use_segmentation=True):
#         super().__init__()
#         self.config = config
#         self.config.cache = Path(config.cache) if config.cache is not None else None
#         self.renderer = Renderer(config.renderer)
#         self.model_id = None
#         self.view_masks_to_faces = {}
#         self.mask_statistics = {}
#         self.custom_camera = None
#         self.current_predict_path: str | None = None

#     # ---------------------------- setters ----------------------------
#     def set_camera_position(self, camera_info):
#         if not isinstance(camera_info, dict):
#             raise TypeError(f"camera_info must be a dict, got {type(camera_info)}")
#         required = ['position', 'target', 'up']
#         if not all(k in camera_info for k in required):
#             raise ValueError(f"camera_info missing keys {required}, got {camera_info.keys()}")
#         self.custom_camera = camera_info

#     # ------------------------ loading/render -------------------------
#     def load(self, scene: Scene, mesh_graph=False):
#         self.renderer.set_object(scene)
#         self.renderer.set_camera()
#         if isinstance(scene, (Path, str)):
#             self.model_id = Path(scene).stem
#         elif hasattr(scene, 'path'):
#             self.model_id = Path(scene.path).stem

#     def render(self, scene: Scene, visualize_path=None, view_name=None) -> dict[str, NumpyTensor]:
#         self.scene_path = getattr(scene, 'path', None)
#         if not getattr(self, 'model_id', None):
#             self.model_id = Path(scene.path).stem if hasattr(scene, 'path') else None
#         self.view_indices = {'indexed_views': {'custom': 0}, 'current_index': 0}

#         # 相机
#         if self.custom_camera:
#             camera_position = np.array(self.custom_camera['position'], dtype=np.float32)
#             target_position = np.array(self.custom_camera['target'], dtype=np.float32)
#             up_direction    = np.array(self.custom_camera['up'], dtype=np.float32)
#         else:
#             camera_position = np.array(VIEW_POSITIONS['custom']['position'], dtype=np.float32)
#             target_position = np.array(VIEW_POSITIONS['custom']['target'], dtype=np.float32)
#             up_direction    = np.array(VIEW_POSITIONS['custom']['up'], dtype=np.float32)

#         renderer_args = {}
#         if hasattr(self.config.renderer, 'renderer_args'):
#             renderer_args = self.config.renderer.renderer_args.copy()

#         pose   = custom_look_at(camera_position, target_position, up_direction)
#         output = self.renderer.render(pose, **renderer_args)
#         output['poses']      = pose
#         output['view_name']  = "custom"
#         output['view_index'] = 0

#         renders = {
#             'faces':      [output['faces']],
#             'poses':      np.array([output['poses']]),
#             'view_names': [output['view_name']],
#             'view_indices': [output['view_index']],
#             'norms':      [output['norms']]
#         }

#         mask = self.call_segmentation(None, output['faces'] != -1, view_index="custom")
#         renders['bmasks'] = [mask]
#         renders['cmasks'] = [combine_bmasks(mask, sort=True)]
#         cmask = renders['cmasks'][0]
#         faces = renders['faces'][0]
#         cmask += 1
#         cmask[faces == -1] = 0
#         renders['cmasks'] = [cmask]
#         return renders

#     # --------------------------- lifting -----------------------------
#     def lift(self, renders: dict[str, NumpyTensor]) -> dict:
#         be, en = 0, len(renders['faces'])
#         renders = {k: [v[i] for i in range(be, en) if len(v)] for k, v in renders.items()}
#         print('Computing face2label for the current view')
#         label_sequence_count = 1
#         self.view_masks_to_faces = {}
#         self.mask_statistics     = {}
#         faceid = renders['faces'][0]
#         cmask  = renders['cmasks'][0]
#         pose   = renders['poses'][0]
#         labels = np.unique(cmask); labels = labels[labels != 0]

#         if 'norms' in renders and len(renders['norms']) > 0:
#             norms = renders['norms'][0]
#         else:
#             norms = np.ones_like(faceid, dtype=np.float32)[:, :, None].repeat(3, axis=2)

#         threshold_counts        = self.config.sam_mesh.get('face2label_threshold', 2)
#         use_percentage_threshold = self.config.sam_mesh.get('use_percentage_threshold', False)
#         threshold_percentage     = self.config.sam_mesh.get('threshold_percentage', 60.0)

#         face2label, mask_info = compute_face2label(
#             labels, faceid, cmask, norms, pose, label_sequence_count,
#             threshold_counts, threshold_percentage, use_percentage_threshold, 0
#         )
#         self.view_masks_to_faces.update(mask_info)
#         # 保留 face2label 结果即可，不再进行任何切割后处理
#         return {f: 1 if cnt else 0 for f, cnt in face2label.items()}

#     # ------------------------ segmentation --------------------------
#     def call_segmentation(self, image: Image, mask: NumpyTensor['h w'], view_index=None):
#         """
#         返回二值前景 / 背景 mask。优先读取 self.current_predict_path PNG。
#         """
#         if self.current_predict_path and os.path.exists(self.current_predict_path):
#             gt_array = np.array(Image.open(self.current_predict_path).convert("L"))
#             fg_mask = gt_array > 127
#             bg_mask = ~fg_mask
#             return np.array([fg_mask, bg_mask], dtype=bool)

#         # fallback：若无 PNG 则全部视为背景
#         h, w = mask.shape
#         return np.zeros((2, h, w), dtype=bool)

#     # --------------------------- forward ----------------------------
#     def forward(self, scene: Scene, visualize_path=None, target_labels=None, view_name=None):
#         self.load(scene)
#         renders = self.render(scene, visualize_path=None, view_name=view_name)
#         face2label_consistent = self.lift(renders)
#         if visualize_path is not None:
#             visualize_items(renders, visualize_path, self.model_id, face2label_consistent, self.current_predict_path)
#         assert self.renderer.tmesh.faces.shape[0] == len(face2label_consistent)
#         return face2label_consistent, self.renderer.tmesh

# # ---------------------------------------------------------------------------
# #                      工具函数（保留必要函数，仅投影）
# # ---------------------------------------------------------------------------
# def read_camera_position(model_id: str, render_type: str='arrow-sketch', view_name: str=None) -> dict:
#     if view_name and view_name in VIEW_POSITIONS:
#         return VIEW_POSITIONS[view_name]
#     return VIEW_POSITIONS['center']

# # ---------------------------------------------------------------------------
# #                       单模型入口 segment_mesh
# # ---------------------------------------------------------------------------
# def segment_mesh(
#     filename: Path | str,
#     config: OmegaConf,
#     predict_image_path: str,
#     visualize: bool=False,
#     extension: str='glb',
#     target_labels=None,
#     texture=False,
#     view_name: str | None=None
# ) -> Trimesh:

#     print(f"[segment_mesh] {filename}")
#     filename = Path(filename)
#     config   = copy.deepcopy(config)
#     if "cache" in config:
#         config.cache = Path(config.output) / "cache"

#     model = SegmentationModelMesh(config)
#     model.model_id = filename.stem
#     model.current_predict_path = predict_image_path

#     camera_info = read_camera_position(model.model_id, view_name=view_name)

#     tmesh = read_mesh(filename, norm=False)
#     if not texture:
#         tmesh = remove_texture(tmesh, visual_kind='vertex')

#     model.set_camera_position(camera_info)

#     output_base = Path(config.output)
#     os.makedirs(output_base, exist_ok=True)

#     visualize_path = output_base if visualize else None

#     faces2label, _ = model(tmesh, visualize_path=visualize_path, view_name=view_name)

#     # 将预测 face id 写入文本（保留）
#     with open(f"{output_base}/pred_face_ids.txt", "w") as f:
#         for fid, lbl in faces2label.items():
#             if lbl == 1:
#                 f.write(f"{fid}\n")

#     print("[segment_mesh] done")
#     return tmesh

# # ---------------------------------------------------------------------------
# #                             批量驱动函数
# # ---------------------------------------------------------------------------
# def _process_all():
#     if not Path(CONFIG_FILE).exists():
#         raise FileNotFoundError(f"Config not found: {CONFIG_FILE}")
#     base_cfg: OmegaConf = OmegaConf.load(CONFIG_FILE)

#     os.makedirs(RESULT_DIR, exist_ok=True)
#     os.makedirs(FAILURE_DIR, exist_ok=True)

#     pngs = sorted(glob.glob(os.path.join(PREDICT_DIR, "*.png")))
#     print(f"[Batch] {len(pngs)} PNGs found")

#     for idx, png in enumerate(pngs, 1):
#         cls, mid, view_raw, joint_idx = parse_predict_filename(png)
#         view_key = view_raw.replace("_", "-")

#         mesh_path = Path(URDF_DIR) / mid / "yy_merged.obj"
#         if not mesh_path.exists():
#             print(f"❌ Mesh missing: {mesh_path}")
#             continue

#         out_dir_name = f"{cls}_{mid}_segmentation_{view_raw}_joint_{joint_idx}"
#         out_dir = Path(RESULT_DIR) / out_dir_name
#         os.makedirs(out_dir, exist_ok=True)

#         cfg = copy.deepcopy(base_cfg)
#         cfg.output = str(out_dir)
#         cfg.cache  = str(out_dir / "cache")

#         print(f"[{idx}/{len(pngs)}] ID={mid} view={view_raw} joint={joint_idx}")
#         try:
#             segment_mesh(mesh_path, cfg, png,
#                          visualize=True, texture=False, view_name=view_key)
#             try:
#                 shutil.copy2(png, out_dir / Path(png).name)
#             except Exception as e:
#                 print(f"   ⚠ copy PNG failed: {e}")
#             print(f"   ✓ saved → {out_dir}")

#         except Exception as e:
#             print(f"   ✖ {e}")
#             import traceback; traceback.print_exc()
#             try:
#                 shutil.copy2(png, os.path.join(FAILURE_DIR, Path(png).name))
#             except Exception as ee:
#                 print(f"   ⚠ copy to failure failed: {ee}")
#             continue

# # ---------------------------------------------------------------------------
# #                               main 入口
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     _process_all()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ------------------------------------------------------------------------------
#                                  Imports
# ------------------------------------------------------------------------------
import os, sys, re, glob, math, copy, shutil, json, multiprocessing as mp
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import trimesh
from PIL import Image
from omegaconf import OmegaConf
from trimesh.base import Trimesh, Scene
from tqdm import tqdm
from natsort import natsorted
import matplotlib.pyplot as plt
import inspect

# project-internal
from samesh.data.loaders import remove_texture, read_mesh
from samesh.data.common import NumpyTensor
from samesh.renderer.renderer import Renderer, render_multiview, colormap_faces, colormap_norms
from samesh.utils.cameras import *
from samesh.utils.mesh import duplicate_verts
# from samesh.utils.camera_views import VIEW_POSITIONS, distance


import math

# ------------------------------------------------------------------------------
#                                视角定义
# ------------------------------------------------------------------------------
distance = 2.5
VIEW_POSITIONS = {
    "top-left": {
        "position": [-distance * math.cos(math.pi/4) * math.sin(math.pi/4),
                     distance * math.sin(math.pi/4),
                     distance * math.cos(math.pi/4) * math.cos(math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "left": {
        "position": [-distance * math.cos(0) * math.sin(math.pi/4),
                     distance * math.sin(0),
                     distance * math.cos(0) * math.cos(math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "bottom-left": {
        "position": [-distance * math.cos(-math.pi/4) * math.sin(math.pi/4),
                     distance * math.sin(-math.pi/4),
                     distance * math.cos(-math.pi/4) * math.cos(math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "top-center": {
        "position": [-distance * math.cos(math.pi/4) * math.sin(0),
                     distance * math.sin(math.pi/4),
                     distance * math.cos(math.pi/4) * math.cos(0)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "center": {
        "position": [0.0, 0.0, distance],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "bottom-center": {
        "position": [-distance * math.cos(-math.pi/4) * math.sin(0),
                     distance * math.sin(-math.pi/4),
                     distance * math.cos(-math.pi/4) * math.cos(0)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "top-right": {
        "position": [-distance * math.cos(math.pi/4) * math.sin(-math.pi/4),
                     distance * math.sin(math.pi/4),
                     distance * math.cos(math.pi/4) * math.cos(-math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "right": {
        "position": [-distance * math.cos(0) * math.sin(-math.pi/4),
                     distance * math.sin(0),
                     distance * math.cos(0) * math.cos(-math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "bottom-right": {
        "position": [-distance * math.cos(-math.pi/4) * math.sin(-math.pi/4),
                     distance * math.sin(-math.pi/4),
                     distance * math.cos(-math.pi/4) * math.cos(-math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "custom": {
        "position": [0.0, 0.0, distance],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    }
} 
# ------------------------------------------------------------------------------
#                               固定路径配置
# ------------------------------------------------------------------------------
PREDICT_DIR = "/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/img_pred"
URDF_DIR    = "/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/urdf"
RESULT_DIR  = "/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/result"
FAILURE_DIR = os.path.join(RESULT_DIR, "failure")
CONFIG_FILE = Path(__file__).resolve().parent / "configs" / "mesh_segmentation.yaml"


# ------------------------------------------------------------------------------
#                       预测文件名解析正则 + 工具函数
# ------------------------------------------------------------------------------
_PREDICT_RE = re.compile(
    r"""^(?P<class>.+?)_(?P<id>\d+)_segmentation_(?P<view>.+?)_joint_(?P<joint>\d+)\.png$""",
    re.IGNORECASE
)

def parse_predict_filename(fname: str):
    """返回 (class, id, view, joint:int)"""
    m = _PREDICT_RE.match(os.path.basename(fname))
    if not m:
        raise ValueError(f"Invalid predict filename: {fname}")
    return (
        m.group("class"),
        m.group("id"),
        m.group("view"),
        int(m.group("joint"))
    )

# ==============================================================================
#                         ========= 原始函数 =========
# 所有下列函数内容均与最初版本完全一致，没有任何删减
# ==============================================================================

def combine_bmasks(masks: NumpyTensor['n h w'], sort=False) -> NumpyTensor['h w']:
    mask_combined = np.zeros_like(masks[0], dtype=int)
    if sort:
        masks = sorted(masks, key=lambda x: x.sum(), reverse=True)
    for i, mask in enumerate(masks):
        mask_combined[mask] = i + 1
    return mask_combined

def colormap_mask(
    mask: NumpyTensor['h w'],
    image: NumpyTensor['h w 3'] = None,
    background=np.array([255, 255, 255]),
    foreground=None,
    blend=0.25
) -> Image.Image:
    palette = np.random.randint(0, 255, (np.max(mask) + 1, 3))
    palette[0] = background
    if foreground is not None:
        for i in range(1, len(palette)):
            palette[i] = foreground
    image_mask = palette[mask.astype(int)]
    image_blend = image_mask if image is None else image_mask * (1 - blend) + image * blend
    image_blend = np.clip(image_blend, 0, 255).astype(np.uint8)
    return Image.fromarray(image_blend)

def visualize_items(items: dict, path: Path, model_id: str = None, face_labels: dict = None, input_image_path: str = None) -> None:
    os.makedirs(path, exist_ok=True)
    view_name = items.get('view_names', ['custom'])[0]
    
    # 获取输入图像路径
    pred_path = input_image_path
    
    faces = items['faces'][0]
    cmask = items['cmasks'][0]
    pose = items['poses'][0] if 'poses' in items and len(items['poses']) > 0 else None
    view_name_formatted = view_name.replace('-', '_')
    
    plt.figure(figsize=(24, 16))
    plt.subplot(2, 4, 1)
    face_ids_img = np.array(colormap_faces(faces))
    plt.imshow(face_ids_img)
    title = f'Face IDs'
    plt.title(title, pad=10, y=1.05)
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    segmentation_mask_img = np.array(colormap_mask(cmask))
    plt.imshow(segmentation_mask_img)
    title = f'Segmentation Mask'
    plt.title(title, pad=10, y=1.05)
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    # 使用渲染结果中的真实法线，而不是硬编码的全1法线
    if 'norms' in items and len(items['norms']) > 0:
        norms = items['norms'][0]  # 使用渲染器提供的真实法线
        print(f"可视化时使用真实法线进行可见性计算")
    else:
        # 备用方案：生成全1法线
        norms = np.ones_like(faces, dtype=np.float32)[:, :, None].repeat(3, axis=2)
        print(f"警告：可视化时使用全1法线代替")
    
    visibility_mask = norms_mask(norms, pose, threshold=0.0) & (faces != -1)
    visible_faces_img = np.zeros((faces.shape[0], faces.shape[1], 3), dtype=np.uint8)
    visible_faces_img[visibility_mask] = [0, 0, 255]  
    visible_faces_img[~visibility_mask] = [255, 255, 255]  
    
    plt.imshow(visible_faces_img)
    title = f'Visible Faces'
    plt.title(title, pad=10, y=1.05)
    plt.axis('off')
    
    plt.subplot(2, 4, 4)
    if pred_path and os.path.exists(pred_path):
        prediction_img = np.array(Image.open(pred_path))
        plt.imshow(prediction_img)
        title = f'Input Image'
        plt.title(title, pad=10, y=1.05)
    else:
        plt.text(0.5, 0.5, 'Input Image Not Found', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
        title = f'Input Image'
        plt.title(title, pad=10, y=1.05)
    plt.axis('off')
    
    plt.subplot(2, 4, 5)
    final_selected_img = np.zeros((faces.shape[0], faces.shape[1], 3), dtype=np.uint8)
    
    if face_labels is not None:
        for i in range(faces.shape[0]):
            for j in range(faces.shape[1]):
                if faces[i, j] != -1: 
                    face_id = int(faces[i, j])
                    if face_id in face_labels and face_labels[face_id] == 1:
                        final_selected_img[i, j] = [0, 255, 0]  
                    else:
                        final_selected_img[i, j] = [255, 0, 0]  
                else:
                    final_selected_img[i, j] = [255, 255, 255]  
        
        plt.imshow(final_selected_img)
        title = f'Selected Faces'
        plt.title(title, pad=10, y=1.05)
    else:
        plt.text(0.5, 0.5, 'Face labels not available', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
        title = f'Selected Faces'
        plt.title(title, pad=10, y=1.05)
    
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    blank_img = np.ones((faces.shape[0], faces.shape[1], 3), dtype=np.uint8) * 255
    plt.imshow(blank_img)
    plt.title(f'Reserved Space 2', pad=10, y=1.05)
    plt.axis('off')
    
    plt.subplot(2, 4, 7)
    plt.imshow(blank_img)
    plt.title(f'Reserved Space 3', pad=10, y=1.05)
    plt.axis('off')
    
    plt.subplot(2, 4, 8)
    plt.imshow(blank_img)
    plt.title(f'Reserved Space 4', pad=10, y=1.05)
    plt.axis('off')

    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
    combined_path = f'{path}/combined_visualization.png'
    plt.savefig(combined_path)
    plt.close()
    
    print(f"Combined visualization saved to: {combined_path}")

def norms_mask(norms: NumpyTensor['h w 3'], cam2world: NumpyTensor['4 4'], threshold=0.0) -> NumpyTensor['h w 3']:
    lookat = cam2world[:3, :3] @ np.array([0, 0, 1])
    return np.abs(np.dot(norms, lookat)) > threshold

def compute_face2label(
    labels: NumpyTensor['l'],
    faceid: NumpyTensor['h w'],
    mask: NumpyTensor['h w'],
    norms: NumpyTensor['h w 3'],
    pose: NumpyTensor['4 4'],
    label_sequence_count: int,
    threshold_counts: int = 2,
    threshold_percentage: float = 60.0,
    use_percentage_threshold: bool = False,
    view_index: int = None
):
    normal_mask = norms_mask(norms, pose)
    visible_mask = (faceid != -1)
    visibility_mask = normal_mask & visible_mask
    face2label = defaultdict(Counter)
    mask_info = {}
    face_total_pixels = {}
    if use_percentage_threshold:
        unique_faces, face_counts = np.unique(faceid[visibility_mask], return_counts=True)
        for face, count in zip(unique_faces, face_counts):
            if face != -1:
                face_total_pixels[int(face)] = count
    for j, label in enumerate(labels):
        label_sequence = label_sequence_count + j
        faces_mask = (mask == label) & visibility_mask
        faces, counts = np.unique(faceid[faces_mask], return_counts=True)
        valid_indices = faces != -1
        faces = faces[valid_indices]
        counts = counts[valid_indices]
        if use_percentage_threshold:
            valid_faces = []
            for face, count in zip(faces, counts):
                face_int = int(face)
                if face_int in face_total_pixels:
                    total = face_total_pixels[face_int]
                    percentage = (count / total) * 100.0
                    if percentage >= threshold_percentage:
                        valid_faces.append(face)
            valid_faces = np.array(valid_faces)
        else:
            valid_faces = faces[counts > threshold_counts]
        mask_id = f"view{view_index}_mask{j}" if view_index is not None else f"mask{j}"
        mask_info[mask_id] = {
            'faces': set(valid_faces.tolist()),
            'label': int(label),
            'label_sequence': int(label_sequence),
            'pixel_count': int(np.sum(faces_mask)),
            'face_count': len(valid_faces),
            'is_ground_truth_foreground': j == 0
        }
        for face in valid_faces:
            face2label[int(face)][label_sequence] += np.sum(faces_mask & (faceid == face))
    return face2label, mask_info

def custom_look_at(eye, target, up):
    eye, target, up = map(lambda v: np.array(v, dtype=np.float32), [eye, target, up])
    z_axis = eye - target; z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.identity(4, dtype=np.float32); rotation[0, :3] = x_axis; rotation[1, :3] = y_axis; rotation[2, :3] = z_axis
    translation = np.identity(4, dtype=np.float32); translation[:3, 3] = -eye
    return np.linalg.inv(np.matmul(rotation, translation))

# ------------------------------------------------------------------------------
#                      SegmentationModelMesh  (含补丁)
# ------------------------------------------------------------------------------
class SegmentationModelMesh(nn.Module):
    def __init__(self, config: OmegaConf, device='cuda', use_segmentation=True):
        super().__init__()
        self.config = config
        self.config.cache = Path(config.cache) if config.cache is not None else None
        self.renderer = Renderer(config.renderer)
        self.model_id = None
        self.view_masks_to_faces = {}
        self.mask_statistics = {}
        self.custom_camera = None
        self.current_predict_path: str | None = None          # NEW

    # ------------------------------- setters ----------------------------------
    def set_camera_position(self, camera_info):
        if not isinstance(camera_info, dict):
            raise TypeError(f"camera_info must be a dict, got {type(camera_info)}")
        required = ['position', 'target', 'up']
        if not all(k in camera_info for k in required):
            raise ValueError(f"camera_info missing keys {required}, got {camera_info.keys()}")
        self.custom_camera = camera_info

    # ------------------------------ loading -----------------------------------
    def load(self, scene: Scene, mesh_graph=False):
        self.renderer.set_object(scene); self.renderer.set_camera()
        if isinstance(scene, (Path, str)):
            self.model_id = Path(scene).stem
        elif hasattr(scene, 'path'):
            self.model_id = Path(scene.path).stem

    # ------------------------------ rendering ---------------------------------
    def render(self, scene: Scene, visualize_path=None, view_name=None) -> dict[str, NumpyTensor]:
        self.scene_path = getattr(scene, 'path', None)
        if not getattr(self, 'model_id', None):
            self.model_id = Path(scene.path).stem if hasattr(scene, 'path') else None
        self.view_indices = {'indexed_views': {'custom': 0}, 'current_index': 0}
        if self.custom_camera:
            camera_position = np.array(self.custom_camera['position'], dtype=np.float32)
            target_position = np.array(self.custom_camera['target'], dtype=np.float32)
            up_direction = np.array(self.custom_camera['up'], dtype=np.float32)
        else:
            camera_position = np.array(VIEW_POSITIONS['custom']['position'], dtype=np.float32)
            target_position = np.array(VIEW_POSITIONS['custom']['target'], dtype=np.float32)
            up_direction = np.array(VIEW_POSITIONS['custom']['up'], dtype=np.float32)
        renderer_args = {}; 
        if hasattr(self.config.renderer, 'renderer_args'):
            renderer_args = self.config.renderer.renderer_args.copy()
        pose = custom_look_at(camera_position, target_position, up_direction)
        output = self.renderer.render(pose, **renderer_args)
        output['poses'] = pose
        output['view_name'] = "custom"
        output['view_index'] = 0
        renders = {
            'faces': [output['faces']],
            'poses': np.array([output['poses']]),
            'view_names': [output['view_name']],
            'view_indices': [output['view_index']],
            'norms': [output['norms']]  # 确保添加法线信息到渲染结果
        }
        mask = self.call_segmentation(None, output['faces'] != -1, view_index="custom")
        renders['bmasks'] = [mask]
        renders['cmasks'] = [combine_bmasks(mask, sort=True)]
        cmask = renders['cmasks'][0]
        faces = renders['faces'][0]
        cmask += 1
        cmask[faces == -1] = 0
        renders['cmasks'] = [cmask]
        return renders

    # ------------------------------ lifting -----------------------------------
    def lift(self, renders: dict[str, NumpyTensor]) -> dict:
        be, en = 0, len(renders['faces'])
        renders = {k: [v[i] for i in range(be, en) if len(v)] for k, v in renders.items()}
        print('Computing face2label for each view')
        label_sequence_count = 1
        self.view_masks_to_faces = {}; self.mask_statistics = {}
        face_gt_votes = defaultdict(int)
        total_views = len(renders['faces'])
        view_idx = 0; faceid = renders['faces'][0]; cmask = renders['cmasks'][0]; pose = renders['poses'][0]
        labels = np.unique(cmask); labels = labels[labels != 0]
        
        # 使用渲染结果中的真实法线，而不是生成全1法线
        if 'norms' in renders and len(renders['norms']) > 0:
            norms = renders['norms'][0]  # 使用渲染器提供的真实法线
            print(f"在面片标签计算中使用真实法线")
        else:
            # 备用方案：生成全1法线
            norms = np.ones_like(faceid, dtype=np.float32)[:, :, None].repeat(3, axis=2)
            print(f"警告：在面片标签计算中使用全1法线代替")
        
        threshold_counts = self.config.sam_mesh.get('face2label_threshold', 2)
        use_percentage_threshold = self.config.sam_mesh.get('use_percentage_threshold', False)
        threshold_percentage = self.config.sam_mesh.get('threshold_percentage', 60.0)
        face2label, mask_info = compute_face2label(
            labels, faceid, cmask, norms, pose, label_sequence_count,
            threshold_counts, threshold_percentage, use_percentage_threshold, view_idx
        )
        self.view_masks_to_faces.update(mask_info)
        for mask_id, info in mask_info.items():
            if "mask1" in mask_id:
                for face in info['faces']:
                    face_gt_votes[face] += 1
        total_faces = len(self.renderer.tmesh.faces)
        for mask_id, info in mask_info.items():
            self.mask_statistics[mask_id] = {
                'face_count': info['face_count'],
                'face_percentage': (info['face_count'] / total_faces) * 100,
                'pixel_count': info['pixel_count'],
                'label': info['label'],
                'label_sequence': info['label_sequence']
            }
        vote_threshold_percentage = self.config.sam_mesh.get('gt_vote_threshold_percentage', 30)
        min_votes_required = max(1, int(total_views * vote_threshold_percentage / 100))
        face2label_final = {}
        gt_faces_count = 0
        for face in range(total_faces):
            is_gt = face_gt_votes[face] >= min_votes_required
            face2label_final[face] = 1 if is_gt else 0
            if is_gt:
                gt_faces_count += 1
        print(f'After voting: {gt_faces_count} faces classified as GT '
              f'({gt_faces_count/total_faces*100:.2f}% of total faces)')
        return face2label_final

    # ------------------------------- forward ----------------------------------
    def forward(self, scene: Scene, visualize_path=None, target_labels=None, view_name=None):
        self.load(scene)
        renders = self.render(scene, visualize_path=None, view_name=view_name)
        face2label_consistent = self.lift(renders)
        if visualize_path is not None:
            visualize_items(renders, visualize_path, self.model_id, face2label_consistent, self.current_predict_path)
        assert self.renderer.tmesh.faces.shape[0] == len(face2label_consistent)
        return face2label_consistent, self.renderer.tmesh

    # ------------------ call_segmentation (patched) ---------------------------
    def call_segmentation(self, image: Image, mask: NumpyTensor['h w'], view_index=None):
        """若 self.current_predict_path 存在则直接读取 PNG"""
        if self.current_predict_path and os.path.exists(self.current_predict_path):
            gt_array = np.array(Image.open(self.current_predict_path).convert("L"))
            fg_mask = gt_array > 127
            bg_mask = ~fg_mask
            return np.array([fg_mask, bg_mask], dtype=bool)
        # fallback —— 原逻辑
        if not self.model_id:
            raise ValueError("model_id not set")
        if view_index is None:
            current_view = list(VIEW_POSITIONS.keys())[0]
        elif isinstance(view_index, int):
            current_view = list(VIEW_POSITIONS.keys())[view_index % len(VIEW_POSITIONS)]
        else:
            current_view = view_index
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        assets_dir = os.path.join(project_root, 'assets')
        gt_dir = os.path.join(assets_dir, f"{self.model_id}_gt")
        gt_path = os.path.join(gt_dir, f"{self.model_id}_pred.png")
        gt_img = Image.open(gt_path).convert('L')
        gt_array = np.array(gt_img)
        fg_mask = (gt_array > 127); bg_mask = (gt_array <= 127)
        return np.array([fg_mask, bg_mask], dtype=bool)

# ------------------------------------------------------------------------------
#                                 其他工具函数
# ------------------------------------------------------------------------------
def create_view_specific_mesh(tmesh: Trimesh, faces_set: set, output_path: str) -> Trimesh:
    faces_list = sorted([f for f in faces_set if f >= 0])
    if not faces_list:
        print(f"WARNING: no valid faces for mesh {output_path}"); return None
    new_faces = tmesh.faces[faces_list]
    used_vertices = np.unique(new_faces.flatten())
    vertex_map = {old: new for new, old in enumerate(used_vertices)}
    new_vertices = tmesh.vertices[used_vertices]
    remapped_faces = np.array([[vertex_map[v] for v in face] for face in new_faces])
    new_mesh = Trimesh(vertices=new_vertices, faces=remapped_faces)
    new_mesh.export(output_path)
    return new_mesh

def colormap_faces_mesh(mesh: Trimesh, face2label: dict[int, int], background=np.array([255,255,255])) -> Trimesh:
    mesh = duplicate_verts(mesh)
    mesh.visual.face_colors = np.zeros((len(mesh.faces),4), dtype=np.uint8)
    mesh.visual.face_colors[:, :3] = 255
    mesh.visual.face_colors[:, 3] = 255
    pred_faces = [f for f,l in face2label.items() if l == 1]
    if pred_faces: mesh.visual.face_colors[pred_faces] = [0,255,0,255]
    print(f"Processed {len(pred_faces)} predict faces, total {len(mesh.faces)}")
    return mesh

def split_mesh_by_labels(mesh: Trimesh, face2label: dict[int,int]):
    pred_faces_idx = [f for f,l in face2label.items() if l==1]
    non_faces_idx = [f for f,l in face2label.items() if l==0]
    pred_faces = mesh.faces[pred_faces_idx]
    non_faces = mesh.faces[non_faces_idx]
    pred_vertices_idx = np.unique(pred_faces.flatten())
    non_vertices_idx = np.unique(non_faces.flatten())
    pred_vertex_map = {o:n for n,o in enumerate(pred_vertices_idx)}
    non_vertex_map = {o:n for n,o in enumerate(non_vertices_idx)}
    pred_faces_remap = np.array([[pred_vertex_map[v] for v in face] for face in pred_faces])
    non_faces_remap = np.array([[non_vertex_map[v] for v in face] for face in non_faces])
    pred_mesh = Trimesh(vertices=mesh.vertices[pred_vertices_idx], faces=pred_faces_remap)
    non_mesh = Trimesh(vertices=mesh.vertices[non_vertices_idx], faces=non_faces_remap)
    pred_mesh.visual.face_colors = np.array([0,255,0,255]); non_mesh.visual.face_colors = np.array([255,255,255,255])
    return pred_mesh, non_mesh

def create_combined_separated_mesh(tmesh: Trimesh, face2label: dict[int,int]) -> Trimesh:
    original_face_count = len(tmesh.faces)
    pred_idx = [f for f,l in face2label.items() if l==1]
    non_idx = [f for f,l in face2label.items() if l==0]
    pred_faces = tmesh.faces[pred_idx]; non_faces = tmesh.faces[non_idx]
    pred_vertices_idx = np.unique(pred_faces.flatten()); non_vertices_idx = np.unique(non_faces.flatten())
    pred_vertex_map = {o:n for n,o in enumerate(pred_vertices_idx)}
    non_vertex_map = {o:n+len(pred_vertices_idx) for n,o in enumerate(non_vertices_idx)}
    pred_faces_remap = np.array([[pred_vertex_map[v] for v in face] for face in pred_faces])
    non_faces_remap = np.array([[non_vertex_map[v] for v in face] for face in non_faces])
    combined_vertices = np.vstack([tmesh.vertices[pred_vertices_idx], tmesh.vertices[non_vertices_idx]])
    combined_faces = np.vstack([pred_faces_remap, non_faces_remap])
    combined_mesh = Trimesh(vertices=combined_vertices, faces=combined_faces)
    combined_mesh = duplicate_verts(combined_mesh)
    combined_mesh.visual.face_colors = np.zeros((len(combined_mesh.faces),4), dtype=np.uint8)
    combined_mesh.visual.face_colors[:, :3] = 255; combined_mesh.visual.face_colors[:, 3] = 255
    combined_mesh.visual.face_colors[list(range(len(pred_faces_remap)))] = [0,255,0,255]
    if len(combined_mesh.faces)!=len(pred_faces)+len(non_faces):
        print("WARNING: face count mismatch after combine")
    return combined_mesh

def create_real_hole_mesh(mesh: Trimesh, face2label: dict[int,int]) -> Trimesh:
    pred_idx = [f for f,l in face2label.items() if l==1]
    non_idx = [f for f,l in face2label.items() if l==0]
    edges_to_faces = defaultdict(list)
    for fi, face in enumerate(mesh.faces):
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]])); edges_to_faces[edge].append(fi)
    boundary_edges = {e for e,fis in edges_to_faces.items()
                      if len(fis)==1 or (any(i in pred_idx for i in fis) and any(i in non_idx for i in fis))}
    boundary_non_faces = {fi for fi in non_idx
                          for i in range(3)
                          if tuple(sorted([mesh.faces[fi][i], mesh.faces[fi][(i+1)%3]])) in boundary_edges}
    final_non_idx = [i for i in non_idx if i not in boundary_non_faces]
    real_faces = mesh.faces[final_non_idx]
    real_vertices_idx = np.unique(real_faces.flatten())
    real_vertex_map = {o:n for n,o in enumerate(real_vertices_idx)}
    real_faces_remap = np.array([[real_vertex_map[v] for v in face] for face in real_faces])
    real_mesh = Trimesh(vertices=mesh.vertices[real_vertices_idx], faces=real_faces_remap)
    real_mesh.visual.face_colors = np.array([255,255,255,255])
    return real_mesh

def read_camera_position(model_id: str, render_type: str='arrow-sketch', view_name: str=None) -> dict:
    if view_name and view_name in VIEW_POSITIONS:
        print(f"Using predefined view {view_name}")
        return VIEW_POSITIONS[view_name]
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    logs_dir = os.path.join(project_root, 'Data', 'logs')
    if not os.path.exists(logs_dir): return VIEW_POSITIONS['center']
    pattern = os.path.join(logs_dir, f"*{model_id}*{render_type}*camera.txt")
    files = glob.glob(pattern)
    if not files: return VIEW_POSITIONS['center']
    cam_file = files[0]
    try:
        with open(cam_file,'r') as f: parts = f.read().strip().split(',')
        if len(parts)>=3:
            x,y,z = map(float, parts[:3])
            return {'position': [-z,y,x],'target':[0.0,0.0,0.0],'up':[0.0,1.0,0.0]}
    except Exception as e:
        print(f"Error reading camera file: {e}")
    return VIEW_POSITIONS['center']

def colormap_faces_without_duplicating(mesh: Trimesh, face2label: dict[int,int]) -> Trimesh:
    colored_mesh = mesh.copy()
    colored_mesh.visual.face_colors = np.zeros((len(mesh.faces),4), dtype=np.uint8)
    colored_mesh.visual.face_colors[:, :3] = 255; colored_mesh.visual.face_colors[:, 3] = 255
    pred_faces = [f for f,l in face2label.items() if l==1]
    if pred_faces: colored_mesh.visual.face_colors[pred_faces] = [0,255,0,255]
    print(f"colored {len(pred_faces)} faces green out of {len(mesh.faces)}")
    return colored_mesh

def extract_gt_mesh(tmesh: Trimesh, face2label: dict[int,int]) -> Trimesh:
    pred_idx = [f for f,l in face2label.items() if l==1]
    print(f"Extracting {len(pred_idx)} GT faces")
    pred_faces = tmesh.faces[pred_idx]
    pred_vertices_idx = np.unique(pred_faces.flatten())
    pred_vertex_map = {o:n for n,o in enumerate(pred_vertices_idx)}
    pred_faces_remap = np.array([[pred_vertex_map[v] for v in face] for face in pred_faces])
    mesh = Trimesh(vertices=tmesh.vertices[pred_vertices_idx], faces=pred_faces_remap)
    mesh.visual.face_colors = np.array([0,255,0,255])
    return mesh

def extract_non_gt_mesh(tmesh: Trimesh, face2label: dict[int,int]) -> Trimesh:
    non_idx = [f for f,l in face2label.items() if l==0]
    print(f"Extracting {len(non_idx)} non-GT faces")
    non_faces = tmesh.faces[non_idx]
    non_vertices_idx = np.unique(non_faces.flatten())
    non_vertex_map = {o:n for n,o in enumerate(non_vertices_idx)}
    non_faces_remap = np.array([[non_vertex_map[v] for v in face] for face in non_faces])
    mesh = Trimesh(vertices=tmesh.vertices[non_vertices_idx], faces=non_faces_remap)
    mesh.visual.face_colors = np.array([255,255,255,255])
    return mesh

def split_mesh_by_labels_with_hole(mesh: Trimesh, face2label: dict[int,int]):
    pred_mesh, non_mesh = split_mesh_by_labels(mesh, face2label)
    return pred_mesh, create_real_hole_mesh(mesh, face2label)

# ------------------------------------------------------------------------------
#                       单模型入口 segment_mesh
# ------------------------------------------------------------------------------
def segment_mesh(
    filename: Path | str,
    config: OmegaConf,
    predict_image_path: str,
    visualize: bool=False,
    extension: str='glb',
    target_labels=None,
    texture=False,
    view_name: str | None=None
) -> Trimesh:
    print(f"[segment_mesh] {filename}")
    filename = Path(filename)
    config   = copy.deepcopy(config)
    # 不再使用filename.stem作为缓存路径
    if "cache" in config:
        config.cache = Path(config.output) / "cache"
    model = SegmentationModelMesh(config)
    model.model_id = filename.stem  # 仍需保留model_id以便内部逻辑工作
    model.current_predict_path = predict_image_path

    camera_info = read_camera_position(model.model_id, view_name=view_name)

    tmesh = read_mesh(filename, norm=False)
    if not texture:
        tmesh = remove_texture(tmesh, visual_kind='vertex')

    model.set_camera_position(camera_info)
    
    # 创建输出目录
    output_base = Path(config.output)
    os.makedirs(output_base, exist_ok=True)
    
    # 设置可视化路径 - 直接使用输出目录
    visualize_path = None
    if visualize:
        visualize_path = output_base  # 使用主输出目录
        print(f"[Visualization] Will be saved to: {visualize_path}")
    
    faces2label, _ = model(tmesh, visualize_path=visualize_path, view_name=view_name)

    # 直接使用output_base作为输出路径，不创建子目录
    combined_mesh = create_combined_separated_mesh(tmesh, faces2label)
    combined_mesh.export(f"{output_base}/combined_separated{filename.suffix}")

    gt_mesh = extract_gt_mesh(tmesh, faces2label)
    gt_mesh.export(f"{output_base}/pred_area.obj")

    with open(f"{output_base}/pred_face_ids.txt", "w") as f:
        for fid, lbl in faces2label.items():
            if lbl == 1:
                f.write(f"{fid}\n")
                
    # 如果启用了可视化，确保生成可视化结果
    if visualize:
        print(f"[Visualization] Generating visualization...")
        # 重新渲染以获取渲染结果
        renders = model.render(tmesh, visualize_path=None, view_name=view_name)
        # 使用增强的visualize_items函数生成可视化
        visualize_items(renders, visualize_path, model.model_id, faces2label, predict_image_path)
        print(f"[Visualization] Complete!")

    return tmesh

# ------------------------------------------------------------------------------
#                             批量驱动函数
# ------------------------------------------------------------------------------
def _process_all():
    if not Path(CONFIG_FILE).exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_FILE}")
    base_cfg: OmegaConf = OmegaConf.load(CONFIG_FILE)

    # 确保输出目录存在
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FAILURE_DIR, exist_ok=True)

    # 修改匹配模式，查找所有PNG文件而不只是_predict.png结尾的文件
    pngs = sorted(glob.glob(os.path.join(PREDICT_DIR, "*.png")))
    print(f"[Batch] {len(pngs)} PNGs found")

    for idx, png in enumerate(pngs, 1):
        cls, mid, view_raw, joint_idx = parse_predict_filename(png)
        view_key = view_raw.replace("_", "-")
        
        # 修改几何体路径格式
        mesh_path = Path(URDF_DIR) / mid / "yy_merged.obj"
        if not mesh_path.exists():
            print(f"❌ Mesh missing: {mesh_path}")
            continue

        # 定义输出目录并创建
        out_dir_name = f"{cls}_{mid}_segmentation_{view_raw}_joint_{joint_idx}"
        out_dir = Path(RESULT_DIR) / out_dir_name
        os.makedirs(out_dir, exist_ok=True)
        
        cfg = copy.deepcopy(base_cfg)
        cfg.output = str(out_dir)  # 设置输出路径
        cfg.cache = str(out_dir / "cache")  # 使用cache子目录，而非基于模型ID
        
        print(f"[{idx}/{len(pngs)}] ID={mid} view={view_raw} joint={joint_idx}")
        try:
            # 执行分割
            segment_mesh(mesh_path, cfg, png,
                     visualize=True, texture=False, view_name=view_key)
                
            # 复制输入的预测图片到结果目录
            try:
                shutil.copy2(png, out_dir / Path(png).name)
            except Exception as e:
                print(f"   ⚠ copy PNG failed: {e}")
            print(f"   ✓ saved → {out_dir}")
            
        except Exception as e:
            print(f"   ✖ {e}")
            import traceback; traceback.print_exc()
            
            # 失败的情况只复制到failure目录
            try:
                shutil.copy2(png, os.path.join(FAILURE_DIR, Path(png).name))
            except Exception as ee:
                print(f"   ⚠ copy to failure failed: {ee}")
            continue

# ------------------------------------------------------------------------------
#                               main 入口
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    _process_all()