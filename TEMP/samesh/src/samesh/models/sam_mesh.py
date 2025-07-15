import shutil
import os
import sys
import json
import copy
import multiprocessing as mp
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import trimesh
# import igraph
from PIL import Image
from omegaconf import OmegaConf
from trimesh.base import Trimesh, Scene
from tqdm import tqdm
from natsort import natsorted

import matplotlib.pyplot as plt
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d import Axes3D, proj3d
import glob
import re
from datetime import datetime

from samesh.data.common import NumpyTensor
from samesh.data.loaders import remove_texture, read_mesh
from samesh.renderer.renderer import Renderer, render_multiview, colormap_faces, colormap_norms
from samesh.models.sam import SamModel, Sam2Model, combine_bmasks, colormap_mask
from samesh.utils.cameras import *
from samesh.utils.mesh import duplicate_verts


# 创建组合可视化，将每个视角的face id、分割结果、预测图和可见面片显示在同一张图上
def visualize_items(items: dict, path: Path, model_id: str = None, face_labels: dict = None) -> None:
    os.makedirs(path, exist_ok=True)
    subplots_dir = os.path.join(path, "subplots")
    os.makedirs(subplots_dir, exist_ok=True)
    view_name = items.get('view_names', ['custom'])[0]
    
    samesh_dir = None
    for base_path in [
        os.path.abspath(os.path.join(os.path.dirname(path), '..')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '..')
    ]:
        if os.path.exists(os.path.join(base_path, 'assets')):
            samesh_dir = base_path
            break
    
    pred_path = None
    if model_id and samesh_dir:
        gt_dir = os.path.join(samesh_dir, 'assets', f'{model_id}_gt')
        if os.path.exists(gt_dir):
            pred_file = f'{model_id}_pred.png'
            pred_path = os.path.join(gt_dir, pred_file)
    
    faces = items['faces'][0]
    cmask = items['cmasks'][0]
    pose = items['poses'][0] if 'poses' in items and len(items['poses']) > 0 else None
    view_name_formatted = view_name.replace('-', '_')
    subplot_images = []
    subplot_titles = []
    plt.figure(figsize=(24, 16))
    plt.subplot(2, 4, 1)
    face_ids_img = np.array(colormap_faces(faces))
    plt.imshow(face_ids_img)
    title = f'Face IDs - {view_name}'
    plt.title(title, pad=10, y=1.05)
    plt.axis('off')
    subplot_images.append(face_ids_img)
    subplot_titles.append("face_ids")
    
    plt.subplot(2, 4, 2)
    segmentation_mask_img = np.array(colormap_mask(cmask))
    plt.imshow(segmentation_mask_img)
    title = f'Segmentation Mask - {view_name}'
    plt.title(title, pad=10, y=1.05)
    plt.axis('off')
    subplot_images.append(segmentation_mask_img)
    subplot_titles.append("segmentation_mask")
    
    plt.subplot(2, 4, 3)
    norms = np.ones_like(faces, dtype=np.float32)[:, :, None].repeat(3, axis=2)
    visibility_mask = norms_mask(norms, pose, threshold=0.0) & (faces != -1)
    visible_faces_img = np.zeros((faces.shape[0], faces.shape[1], 3), dtype=np.uint8)
    visible_faces_img[visibility_mask] = [0, 0, 255]  
    visible_faces_img[~visibility_mask] = [255, 255, 255]  
    
    plt.imshow(visible_faces_img)
    title = f'Visible Faces - {view_name}'
    plt.title(title, pad=10, y=1.05)
    plt.axis('off')
    subplot_images.append(visible_faces_img)
    subplot_titles.append("visible_faces")
    
    plt.subplot(2, 4, 4)
    if pred_path and os.path.exists(pred_path):
        prediction_img = np.array(Image.open(pred_path))
        plt.imshow(prediction_img)
        title = f'Prediction - {view_name}'
        plt.title(title, pad=10, y=1.05)
        subplot_images.append(prediction_img)
        subplot_titles.append("prediction")
    else:
        plt.text(0.5, 0.5, 'Prediction Not Found', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
        title = f'Prediction - {view_name}'
        plt.title(title, pad=10, y=1.05)
        blank_img = np.ones((faces.shape[0], faces.shape[1], 3), dtype=np.uint8) * 255
        subplot_images.append(blank_img)
        subplot_titles.append("prediction_not_found")
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
        title = f'Actually Selected Faces - {view_name}'
        plt.title(title, pad=10, y=1.05)
        subplot_images.append(final_selected_img)
        subplot_titles.append("selected_faces")
    else:
        plt.text(0.5, 0.5, 'Face labels not available', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
        title = f'Actually Selected Faces - {view_name}'
        plt.title(title, pad=10, y=1.05)
        blank_img = np.ones((faces.shape[0], faces.shape[1], 3), dtype=np.uint8) * 255
        subplot_images.append(blank_img)
        subplot_titles.append("face_labels_not_available")
    
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    blank_img = np.ones((faces.shape[0], faces.shape[1], 3), dtype=np.uint8) * 255
    plt.imshow(blank_img)
    plt.title(f'Reserved Space 2 - {view_name}', pad=10, y=1.05)
    plt.axis('off')
    
    plt.subplot(2, 4, 7)
    plt.imshow(blank_img)
    plt.title(f'Reserved Space 3 - {view_name}', pad=10, y=1.05)
    plt.axis('off')
    
    plt.subplot(2, 4, 8)
    plt.imshow(blank_img)
    plt.title(f'Reserved Space 4 - {view_name}', pad=10, y=1.05)
    plt.axis('off')

    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
    combined_path = f'{path}/combined_{view_name_formatted}.png'
    plt.savefig(combined_path)
    plt.close()

    for i, (img, title) in enumerate(zip(subplot_images, subplot_titles)):
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f'{title.replace("_", " ").title()} - {view_name}', pad=10, y=1.05)
        plt.axis('off')
        plt.tight_layout()
        subplot_path = os.path.join(subplots_dir, f'{view_name_formatted}_{title}.png')
        plt.savefig(subplot_path)
        plt.close()
    
    print(f"Combined visualization saved to: {combined_path}")
    print(f"Individual subplots saved to: {subplots_dir}/")






# 对直接面向相机的像素进行遮罩处理，标记哪些像素直接面向相机
def norms_mask(norms: NumpyTensor['h w 3'], cam2world: NumpyTensor['4 4'], threshold=0.0) -> NumpyTensor['h w 3']:

    lookat = cam2world[:3, :3] @ np.array([0, 0, 1])
    # 计算每个像素的法向量与观察方向的点积,大于阈值的为正面
    return np.abs(np.dot(norms, lookat)) > threshold




# 计算每个面片的标签，并记录每个mask对应的面片ID，返回tuple: (face2label字典, mask_info字典)
def compute_face2label(
    labels: NumpyTensor['l'],
    faceid: NumpyTensor['h w'], 
    mask: NumpyTensor['h w'],
    norms: NumpyTensor['h w 3'],
    pose: NumpyTensor['4 4'],
    label_sequence_count: int,
    threshold_counts: int=2,
    threshold_percentage: float=60.0,  # 新增比例阈值参数，默认60%
    use_percentage_threshold: bool=False,  # 是否使用比例阈值
    view_index: int = None
):

    # 计算面向相机的面片掩码
    normal_mask = norms_mask(norms, pose)
    # 创建可见区域掩码（face id不为-1的区域）
    visible_mask = (faceid != -1)
    # 组合可见性条件：面片必须同时满足normal_mask和visible_mask
    visibility_mask = normal_mask & visible_mask
    face2label = defaultdict(Counter)
    mask_info = {}  # 存储每个mask的信息
    # 如果使用比例阈值，预先计算每个面片在当前视图中的总像素数
    face_total_pixels = {}
    if use_percentage_threshold:
        unique_faces, face_counts = np.unique(faceid[visibility_mask], return_counts=True)
        for face, count in zip(unique_faces, face_counts):
            if face != -1:  # 忽略背景
                face_total_pixels[int(face)] = count
    
    for j, label in enumerate(labels):
        label_sequence = label_sequence_count + j
        # 只处理在当前视角下完全可见的面片
        faces_mask = (mask == label) & visibility_mask
        faces, counts = np.unique(faceid[faces_mask], return_counts=True)
        # 移除-1（表示不可见）的face id
        valid_indices = faces != -1
        faces = faces[valid_indices]
        counts = counts[valid_indices]
        # 根据选择的阈值策略过滤面片
        if use_percentage_threshold:
            # 使用比例阈值
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
            # 使用绝对像素数阈值
            valid_faces = faces[counts > threshold_counts]
        
        # 记录这个mask的信息
        mask_id = f"view{view_index}_mask{j}" if view_index is not None else f"mask{j}"
        mask_info[mask_id] = {
            'faces': set(valid_faces.tolist()),  # 使用set来存储面片ID
            'label': int(label),
            'label_sequence': int(label_sequence),
            'pixel_count': int(np.sum(faces_mask)),
            'face_count': len(valid_faces),
            'is_ground_truth_foreground': j == 0  # 添加标志，j=0表示这是前景mask
        }
        
        # 更新face2label映射
        for face in valid_faces:
            face2label[int(face)][label_sequence] += np.sum(faces_mask & (faceid == face))
    
    return face2label, mask_info



# 创建一个"look at"变换矩阵，使相机从eye位置看向target位置
def custom_look_at(eye, target, up):

    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    # 计算z轴（前方向）
    z_axis = eye - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # 计算x轴（右方向）
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # 计算y轴（上方向）
    y_axis = np.cross(z_axis, x_axis)
    
    # 创建旋转部分
    rotation = np.identity(4, dtype=np.float32)
    rotation[0, :3] = x_axis
    rotation[1, :3] = y_axis
    rotation[2, :3] = z_axis
    
    # 创建平移部分
    translation = np.identity(4, dtype=np.float32)
    translation[:3, 3] = -eye
    
    # 组合旋转和平移
    transform = np.matmul(rotation, translation)
    
    # 返回转置（从相机坐标到世界坐标的变换）
    return np.linalg.inv(transform)



class SegmentationModelMesh(nn.Module):

    def __init__(self, config: OmegaConf, device='cuda', use_segmentation=True):
        super().__init__()
        self.config = config
        self.config.cache = Path(config.cache) if config.cache is not None else None
        self.renderer = Renderer(config.renderer)
        self.model_id = None
        # 添加新的属性来存储mask信息
        self.view_masks_to_faces = {}  # 存储每个视角的mask对应的面片ID
        self.mask_statistics = {}      # 存储每个mask的统计信息
        
        # 添加自定义相机位置属性
        self.custom_camera = None
        
    
    # 设置自定义相机位置，用于渲染
    def set_camera_position(self, camera_info):

        if not isinstance(camera_info, dict):
            raise TypeError(f"camera_info must be a dictionary, got {type(camera_info)}")
            
        # 验证相机信息格式
        required_keys = ['position', 'target', 'up']
        if not all(key in camera_info for key in required_keys):
            raise ValueError(f"camera_info missing required keys: {required_keys}. Got keys: {list(camera_info.keys())}")
            
        self.custom_camera = camera_info
        print(f"Set custom camera position: {camera_info['position']}")
    
    def load(self, scene: Scene, mesh_graph=True):
        self.renderer.set_object(scene)
        self.renderer.set_camera()
        
        # 从scene获取model_id
        if isinstance(scene, Path) or isinstance(scene, str):
            self.model_id = Path(scene).stem
        elif hasattr(scene, 'path'):
            self.model_id = Path(scene.path).stem

        if mesh_graph:
            self.mesh_edges = trimesh.graph.face_adjacency(mesh=self.renderer.tmesh)
            self.mesh_graph = defaultdict(set)
            for face1, face2 in self.mesh_edges:
                self.mesh_graph[face1].add(face2)
                self.mesh_graph[face2].add(face1)

    # 渲染函数，处理face id和ground truth
    def render(self, scene: Scene, visualize_path=None, view_name=None) -> dict[str, NumpyTensor]:

        # 保存场景路径和模型ID
        self.scene_path = getattr(scene, 'path', None)
        if not hasattr(self, 'model_id') or self.model_id is None:
            self.model_id = Path(scene.path).stem if hasattr(scene, 'path') else None
        
        # 初始化视角索引（简化为只有一个索引）
        self.view_indices = {'indexed_views': {'custom': 0}, 'current_index': 0}
        
        # 确定要使用的相机位置
        if hasattr(self, 'custom_camera') and self.custom_camera is not None:
            # 使用自定义相机
            camera_position = np.array(self.custom_camera['position'], dtype=np.float32)
            target_position = np.array(self.custom_camera['target'], dtype=np.float32)
            up_direction = np.array(self.custom_camera['up'], dtype=np.float32)
            print(f"使用自定义相机位置: {camera_position}")
        
        # 准备渲染参数
        renderer_args = {}
        if hasattr(self.config.renderer, 'renderer_args'):
            renderer_args = self.config.renderer.renderer_args.copy()
        
        # 渲染单一视角
        pose = custom_look_at(camera_position, target_position, up_direction)
        output = self.renderer.render(pose, **renderer_args)
        output['poses'] = pose
        output['view_name'] = "custom"
        output['view_index'] = 0
        

        renders = {
            'faces': [output['faces']],
            'poses': np.array([output['poses']]),
            'view_names': [output['view_name']],
            'view_indices': [output['view_index']]
        }
        
        # 处理分割结果
        print("\n处理自定义视角:")
        
        # 获取分割掩码
        mask = self.call_segmentation(None, output['faces'] != -1, view_index="custom")
        
        # 处理掩码和背景
        renders['bmasks'] = [mask]
        renders['cmasks'] = [combine_bmasks(mask, sort=True)]
        
        # 设置背景为0
        cmask = renders['cmasks'][0]
        faces = renders['faces'][0]
        cmask += 1  # 所有标签值加1
        cmask[faces == -1] = 0  # 背景(face id为-1)设置为0
        
        # 直接使用原始分割结果
        renders['cmasks'] = [cmask]
        
        return renders
    
    # 处理每个视角的渲染结果，使用投票机制合并为统一的二值标签（GT区域和背景）
    def lift(self, renders: dict[str, NumpyTensor]) -> dict:

        be, en = 0, len(renders['faces'])
        renders = {k: [v[i] for i in range(be, en) if len(v)] for k, v in renders.items()}

        print('Computing face2label for each view')
        label_sequence_count = 1  # background is 0
        
        # 初始化mask信息存储
        self.view_masks_to_faces = {}
        self.mask_statistics = {}
        
        # 用字典记录每个面片被识别为GT的次数
        face_gt_votes = defaultdict(int)
        total_views = len(renders['faces'])  # 总视角数
        
        # 获取视角信息（在单一视角模式下，只有一个视角）
        view_idx = 0
        faceid = renders['faces'][0]
        cmask = renders['cmasks'][0]
        pose = renders['poses'][0]
        
        # 获取标签
        labels = np.unique(cmask)
        labels = labels[labels != 0]  # remove background
        norms = np.ones_like(faceid, dtype=np.float32)[:, :, None].repeat(3, axis=2)
        
        # 获取阈值设置
        threshold_counts = self.config.sam_mesh.get('face2label_threshold', 2)
        # 从配置中读取是否使用比例阈值及比例阈值设置
        use_percentage_threshold = self.config.sam_mesh.get('use_percentage_threshold', False)
        threshold_percentage = self.config.sam_mesh.get('threshold_percentage', 60.0)
        
        print(f"Face selection method: {'Percentage threshold' if use_percentage_threshold else 'Absolute count threshold'}")
        if use_percentage_threshold:
            print(f"Using percentage threshold: {threshold_percentage}%")
        else:
            print(f"Using absolute count threshold: {threshold_counts} pixels")
        
        # 直接计算单一视角的face2label映射
        face2label, mask_info = compute_face2label(
            labels, faceid, cmask, norms, pose, 
            label_sequence_count,
            threshold_counts, 
            threshold_percentage,
            use_percentage_threshold,
            view_idx
        )
        
        # 更新mask信息
        self.view_masks_to_faces.update(mask_info)
        
        # 模拟原有投票逻辑
        for mask_id, info in mask_info.items():
            if "mask1" in mask_id:  # GT区域
                for face in info['faces']:
                    face_gt_votes[face] += 1
        
        # 计算并存储统计信息
        total_faces = len(self.renderer.tmesh.faces)
        for mask_id, info in mask_info.items():
            self.mask_statistics[mask_id] = {
                'face_count': info['face_count'],
                'face_percentage': (info['face_count'] / total_faces) * 100,
                'pixel_count': info['pixel_count'],
                'label': info['label'],
                'label_sequence': info['label_sequence']
            }
        
        # 设置投票阈值 - 保持与原始代码一致
        vote_threshold_percentage = self.config.sam_mesh.get('gt_vote_threshold_percentage', 30)
        min_votes_required = max(1, int(total_views * vote_threshold_percentage / 100))
        
        print(f'Using voting threshold: {min_votes_required}/{total_views} views '
              f'({vote_threshold_percentage}% of views required for GT)')
        
        # 创建最终的face2label映射，使用投票结果 - 与原始代码完全一致
        face2label_final = {}
        gt_faces_count = 0
        for face in range(len(self.renderer.tmesh.faces)):
            # 只有当面片在足够多的视角中被识别为GT时，才标记为GT区域
            is_gt = face_gt_votes[face] >= min_votes_required
            face2label_final[face] = 1 if is_gt else 0
            if is_gt:
                gt_faces_count += 1
        
        print(f'After voting: {gt_faces_count} faces classified as GT '
              f'({gt_faces_count/len(self.renderer.tmesh.faces)*100:.2f}% of total faces)')
        
        return face2label_final


    
    def forward(self, scene: Scene, visualize_path=None, target_labels=None, view_name=None) -> tuple[dict, Trimesh]:

        self.load(scene)
        renders = self.render(scene, visualize_path=None, view_name=view_name)  
        face2label_consistent = self.lift(renders)
        
        if visualize_path is not None:
            visualize_items(renders, visualize_path, self.model_id, face2label_consistent)
        
        assert self.renderer.tmesh.faces.shape[0] == len(face2label_consistent)
        return face2label_consistent, self.renderer.tmesh



    def label_components(self, face2label: dict) -> list[set]:
        components = []
        visited = set()

        def dfs(source: int):
            stack = [source]
            components.append({source})
            visited.add(source)
            
            while stack:
                node = stack.pop()
                for adj in self.mesh_graph[node]:
                    if adj not in visited and adj in face2label and face2label[adj] == face2label[node]:
                        stack.append(adj)
                        components[-1].add(adj)
                        visited.add(adj)

        for face in range(len(self.renderer.tmesh.faces)):
            if face not in visited and face in face2label:
                dfs(face)
        return components

    def call_segmentation(self, image: Image, mask: NumpyTensor['h w'], view_index=None):
        """Load ground truth mask"""
        if not self.model_id:
            raise ValueError("model_id not set")
        
        # Get view name
        if view_index is None:
            current_view = list(VIEW_POSITIONS.keys())[0]
        elif isinstance(view_index, int):
            view_positions_keys = list(VIEW_POSITIONS.keys())
            current_view = view_positions_keys[view_index % len(view_positions_keys)]
        else:
            current_view = view_index
        
        # 简化的路径查找逻辑
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
        assets_dir = os.path.join(project_root, 'assets')
        gt_dir = os.path.join(assets_dir, f"{self.model_id}_gt")
        
        # 处理自定义视角的情况
        if current_view == "custom":
            gt_path = os.path.join(gt_dir, f"{self.model_id}_pred.png")
            
            gt_img = Image.open(gt_path).convert('L')
            gt_array = np.array(gt_img)

            fg_mask = (gt_array > 127)
            bg_mask = (gt_array <= 127)
            
            return np.array([fg_mask, bg_mask], dtype=bool)












# ----------------------------------------------------------------------------------------
# ------------------------------ 3D Mesh Cutting Functions -------------------------------
# ----------------------------------------------------------------------------------------



def create_view_specific_mesh(tmesh: Trimesh, faces_set: set, output_path: str) -> Trimesh:
    """
    创建只包含特定面片的网格，确保只包含可见的面片
    
    Args:
        tmesh: 原始网格
        faces_set: 要保留的面片ID集合
        output_path: 输出文件路径
    
    Returns:
        Trimesh: 新的网格，只包含指定的面片
    """
    # 获取要保留的面片
    faces_list = sorted(list(faces_set))
    
    # 确保所有face id都是有效的（不为-1）
    faces_list = [f for f in faces_list if f >= 0]
    
    if not faces_list:
        print(f"警告：没有找到有效的面片用于创建mesh：{output_path}")
        return None
        
    new_faces = tmesh.faces[faces_list]
    
    # 找出使用的顶点
    used_vertices = np.unique(new_faces.flatten())
    
    # 创建顶点映射（旧索引到新索引）
    vertex_map = {old: new for new, old in enumerate(used_vertices)}
    
    # 创建新的顶点数组
    new_vertices = tmesh.vertices[used_vertices]
    
    # 更新面片索引
    remapped_faces = np.array([[vertex_map[v] for v in face] for face in new_faces])
    
    # 创建新的网格
    new_mesh = Trimesh(vertices=new_vertices, faces=remapped_faces)
    
    # 导出OBJ文件
    new_mesh.export(output_path)
    
    return new_mesh

def colormap_faces_mesh(mesh: Trimesh, face2label: dict[int, int], background=np.array([255, 255, 255])) -> Trimesh:
    """
    Color mesh faces using two fixed colors:
    - GT area: green
    - Other areas: white
    """
    mesh = duplicate_verts(mesh)  # needed to prevent face color interpolation
    
    # Initialize all faces as white (non-GT area)
    mesh.visual.face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    mesh.visual.face_colors[:, 0] = 255  # white
    mesh.visual.face_colors[:, 1] = 255  # white
    mesh.visual.face_colors[:, 2] = 255  # white
    mesh.visual.face_colors[:, 3] = 255  # opaque
    
    # Set GT area (label 1) to green
    gt_faces = [face for face, label in face2label.items() if label == 1]
    if gt_faces:
        mesh.visual.face_colors[gt_faces] = [0, 255, 0, 255]  # 纯绿色
    
    print(f"Processed {len(gt_faces)} GT area faces, total faces: {len(mesh.faces)}")
    return mesh

# 将mesh按照标签切分为两个部分：GT区域和非GT区域
def split_mesh_by_labels(mesh: Trimesh, face2label: dict[int, int]) -> tuple[Trimesh, Trimesh]:

    # 获取GT和非GT的面片索引
    gt_faces_idx = [face for face, label in face2label.items() if label == 1]
    non_gt_faces_idx = [face for face, label in face2label.items() if label == 0]
    
    # 提取GT区域的面片
    gt_faces = mesh.faces[gt_faces_idx]
    # 提取非GT区域的面片
    non_gt_faces = mesh.faces[non_gt_faces_idx]
    
    # 创建GT mesh
    gt_vertices_idx = np.unique(gt_faces.flatten())
    gt_vertices = mesh.vertices[gt_vertices_idx]
    # 创建顶点映射
    gt_vertex_map = {old: new for new, old in enumerate(gt_vertices_idx)}
    # 重新映射面片索引
    gt_faces_remapped = np.array([[gt_vertex_map[v] for v in face] for face in gt_faces])
    gt_mesh = Trimesh(vertices=gt_vertices, faces=gt_faces_remapped)
    
    # 创建非GT mesh
    non_gt_vertices_idx = np.unique(non_gt_faces.flatten())
    non_gt_vertices = mesh.vertices[non_gt_vertices_idx]
    # 创建顶点映射
    non_gt_vertex_map = {old: new for new, old in enumerate(non_gt_vertices_idx)}
    # 重新映射面片索引
    non_gt_faces_remapped = np.array([[non_gt_vertex_map[v] for v in face] for face in non_gt_faces])
    non_gt_mesh = Trimesh(vertices=non_gt_vertices, faces=non_gt_faces_remapped)
    
    # 设置颜色
    gt_mesh.visual.face_colors = np.array([0, 255, 0, 255])  # 纯绿色
    non_gt_mesh.visual.face_colors = np.array([255, 255, 255, 255])  # 白色
    
    return gt_mesh, non_gt_mesh


# 创建一个包含GT区域和非GT区域的分离网格
def create_combined_separated_mesh(tmesh: Trimesh, face2label: dict[int, int]) -> Trimesh:
    """
    创建一个包含GT区域（绿色）和非GT区域（白色）的分离网格，
    确保所有面片都被保留，不会有任何面片丢失，同时保持模型的完整结构。
    
    Args:
        tmesh: 原始网格
        face2label: 面片标签字典，1表示GT区域，0表示非GT区域
        
    Returns:
        Trimesh: 分离的网格，绿色部分为GT区域，白色部分为非GT区域
    """
    from collections import defaultdict
    
    # 记录原始面片总数，用于后续验证
    original_face_count = len(tmesh.faces)
    print(f"原始模型面片总数: {original_face_count}")
    
    # 获取GT和非GT的面片索引
    gt_faces_idx = [face for face, label in face2label.items() if label == 1]
    non_gt_faces_idx = [face for face, label in face2label.items() if label == 0]
    
    print(f"绿色(GT)面片数量: {len(gt_faces_idx)}")
    print(f"白色(非GT)面片数量: {len(non_gt_faces_idx)}")
    
    # 确认面片总数正确
    if len(gt_faces_idx) + len(non_gt_faces_idx) != original_face_count:
        print(f"警告: 面片计数不一致! GT: {len(gt_faces_idx)}, 非GT: {len(non_gt_faces_idx)}, 总计: {len(gt_faces_idx) + len(non_gt_faces_idx)}, 原始: {original_face_count}")
    
    # 分别获取GT和非GT区域的面片
    gt_faces = tmesh.faces[gt_faces_idx]
    non_gt_faces = tmesh.faces[non_gt_faces_idx]
    
    # ---- 构建独立的两个部分 ----
    # 为两个部分创建独立的顶点集
    gt_vertices = tmesh.vertices[np.unique(gt_faces.flatten())]
    non_gt_vertices = tmesh.vertices[np.unique(non_gt_faces.flatten())]
    
    # 创建顶点映射
    gt_vertex_map = {old: new for new, old in enumerate(np.unique(gt_faces.flatten()))}
    non_gt_vertex_map = {old: new + len(gt_vertices) for new, old in enumerate(np.unique(non_gt_faces.flatten()))}
    
    # 重新映射面片索引
    gt_faces_remapped = np.array([[gt_vertex_map[v] for v in face] for face in gt_faces])
    non_gt_faces_remapped = np.array([[non_gt_vertex_map[v] for v in face] for face in non_gt_faces])
    
    # 合并顶点和面片
    combined_vertices = np.vstack([gt_vertices, non_gt_vertices])
    combined_faces = np.vstack([gt_faces_remapped, non_gt_faces_remapped])
    
    # 创建合并的网格
    combined_mesh = Trimesh(vertices=combined_vertices, faces=combined_faces)
    
    # 添加关键步骤：复制顶点以防止颜色插值 - 这是解决颜色不一致的关键
    combined_mesh = duplicate_verts(combined_mesh)
    
    # 设置面片颜色 - 使用与其他函数一致的方式
    # 首先全部设为白色
    combined_mesh.visual.face_colors = np.zeros((len(combined_mesh.faces), 4), dtype=np.uint8)
    combined_mesh.visual.face_colors[:, 0] = 255  # 白色
    combined_mesh.visual.face_colors[:, 1] = 255
    combined_mesh.visual.face_colors[:, 2] = 255
    combined_mesh.visual.face_colors[:, 3] = 255  # 完全不透明
    
    # 然后设置GT部分为绿色
    gt_part_indices = list(range(len(gt_faces)))  # GT部分的面片索引
    combined_mesh.visual.face_colors[gt_part_indices] = [0, 255, 0, 255]  # 纯绿色
    
    # 验证最终面片数量
    final_face_count = len(combined_mesh.faces)
    expected_face_count = len(gt_faces) + len(non_gt_faces)
    
    print(f"合并后的面片总数: {final_face_count}")
    print(f"预期面片总数: {expected_face_count}")
    print(f"原始面片总数: {original_face_count}")
    
    if final_face_count != expected_face_count:
        print(f"警告: 合并后的面片数量与预期不符! 预期: {expected_face_count}, 实际: {final_face_count}")
    
    # 同时还保存单独的绿色部分和白色部分
    # 这已经在segment_mesh函数中通过调用新增的函数实现
    
    return combined_mesh


def create_real_hole_mesh(mesh: Trimesh, face2label: dict[int, int]) -> Trimesh:
    """
    创建一个在双面渲染条件下也能显示空洞的模型
    通过识别和移除与GT区域边界相邻的非GT面片来实现
    
    Args:
        mesh: 原始网格
        face2label: 面片标签字典，1表示GT区域，0表示非GT区域
        
    Returns:
        Trimesh: 带有真实空洞的网格
    """
    from collections import defaultdict
    
    # 1. 获取GT和非GT的面片索引
    gt_faces_idx = [face for face, label in face2label.items() if label == 1]
    non_gt_faces_idx = [face for face, label in face2label.items() if label == 0]
    
    # 2. 构建边到面片的映射
    edges_to_faces = defaultdict(list)
    for face_idx, face in enumerate(mesh.faces):
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            edges_to_faces[edge].append(face_idx)
    
    # 3. 找出边界边（连接GT和非GT区域的边）
    boundary_edges = set()
    for edge, face_indices in edges_to_faces.items():
        if len(face_indices) >= 2:  # 至少连接两个面片
            if any(idx in gt_faces_idx for idx in face_indices) and any(idx in non_gt_faces_idx for idx in face_indices):
                boundary_edges.add(edge)
    
    # 4. 找出与边界相邻的非GT面片
    boundary_non_gt_faces = set()
    for face_idx in non_gt_faces_idx:
        face = mesh.faces[face_idx]
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            if edge in boundary_edges:
                boundary_non_gt_faces.add(face_idx)
                break
    
    # 5. 移除这些边界面片，创建真正有洞的非GT mesh
    final_non_gt_faces_idx = [idx for idx in non_gt_faces_idx if idx not in boundary_non_gt_faces]
    real_hole_faces = mesh.faces[final_non_gt_faces_idx]
    
    # 6. 创建非GT mesh（与原来相同，但使用修改后的面片集合）
    real_hole_vertices_idx = np.unique(real_hole_faces.flatten())
    real_hole_vertices = mesh.vertices[real_hole_vertices_idx]
    real_hole_vertex_map = {old: new for new, old in enumerate(real_hole_vertices_idx)}
    real_hole_faces_remapped = np.array([[real_hole_vertex_map[v] for v in face] for face in real_hole_faces])
    real_hole_mesh = Trimesh(vertices=real_hole_vertices, faces=real_hole_faces_remapped)
    
    # 7. 设置颜色
    real_hole_mesh.visual.face_colors = np.array([255, 255, 255, 255])  # 白色
    
    return real_hole_mesh


def read_camera_position(model_id: str, render_type: str = 'arrow-sketch') -> dict:
    """
    读取指定模型的相机位置信息
    
    Args:
        model_id: 模型ID
        render_type: 渲染类型
        
    Returns:
        包含相机位置信息的字典
    """
    import glob
    
    print(f"Attempting to read camera position for model {model_id}")
    
    # 获取项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    print(f"Project root: {project_root}")
    
    # 构建日志目录路径
    logs_dir = os.path.join(project_root, 'Data', 'logs')
    print(f"Logs directory: {logs_dir}")
    
    if not os.path.exists(logs_dir):
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
    
    # 搜索匹配的相机位置文件
    # 文件模式：任何包含model_id和render_type的文件
    file_pattern = os.path.join(logs_dir, f"*{model_id}*{render_type}*camera.txt")
    camera_files = glob.glob(file_pattern)
    
    if not camera_files:
        raise FileNotFoundError(f"Camera position file not found for model {model_id} with pattern: {file_pattern}")
    
    # 使用找到的第一个文件
    camera_file = camera_files[0]
    print(f"Found camera position file: {camera_file}")
    
    try:
        # 读取文件内容
        with open(camera_file, 'r') as f:
            content = f.read().strip()
        
        # 解析相机位置
        # 预期格式: x,y,z 或更详细的结构
        parts = content.split(',')
        
        if len(parts) >= 3:
            x, y, z = map(float, parts[:3])
            camera_info = {
                'position': [-z, y, x],
                'target': [0.0, 0.0, 0.0],  # 默认看向原点
                'up': [0.0, 1.0, 0.0]       # Y轴朝上
            }
            print(f"Successfully read camera position: {camera_info['position']}")
            return camera_info
        else:
            raise ValueError(f"Invalid camera position format: {content}. Expected at least 3 comma-separated values.")
            
    except Exception as e:
        raise Exception(f"Error reading camera position: {e}")


def colormap_faces_without_duplicating(mesh: Trimesh, face2label: dict[int, int]) -> Trimesh:
    """
    为网格的面片上色而不复制顶点，保持原始的几何连通性
    
    Args:
        mesh: 原始网格
        face2label: 面片标签字典，1表示GT区域，0表示非GT区域
        
    Returns:
        Trimesh: 上色后的网格，保持原始顶点
    """
    # 创建网格的副本而不是修改原始网格
    colored_mesh = mesh.copy()
    
    # 初始化所有面片为白色（非GT区域）
    colored_mesh.visual.face_colors = np.zeros((len(colored_mesh.faces), 4), dtype=np.uint8)
    colored_mesh.visual.face_colors[:, 0] = 255  # 白色
    colored_mesh.visual.face_colors[:, 1] = 255  # 白色
    colored_mesh.visual.face_colors[:, 2] = 255  # 白色
    colored_mesh.visual.face_colors[:, 3] = 255  # 不透明
    
    # 设置GT区域（标签为1）为绿色
    gt_faces = [face for face, label in face2label.items() if label == 1]
    if gt_faces:
        colored_mesh.visual.face_colors[gt_faces] = [0, 255, 0, 255]  # 纯绿色
    
    print(f"处理了 {len(gt_faces)} 个GT区域面片，总面片: {len(mesh.faces)}")
    return colored_mesh


def extract_components_and_save(mesh: Trimesh, output_base: str, export_format: str = '.obj', face2label=None):
    """
    对网格进行连通组件划分并保存结果，类似extract_parts.py的功能
    
    Args:
        mesh: 要处理的网格对象
        output_base: 输出文件路径的基础部分
        export_format: 导出文件的格式，默认为.obj
        face2label: 原始面片ID到标签的字典，如果提供，将会计算每个组件包含的GT面片比例
    """
    print("\n对网格进行连通组件划分...")
    
    # 创建一个临时文件用于PyMeshLab处理
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # 导出网格到临时文件
    mesh.export(temp_path)
    
    # 创建MeshSet对象
    ms = pymeshlab.MeshSet()
    
    # 加载网格
    ms.load_new_mesh(temp_path)
    
    # 获取原始网格的面片数量
    original_mesh = ms.current_mesh()
    total_faces = original_mesh.face_number()
    
    # 保存原始面片中心点位置，用于后续匹配
    ms.set_current_mesh(0)  # 确保当前是原始网格
    original_faces = np.array(ms.current_mesh().face_matrix())
    original_vertices = np.array(ms.current_mesh().vertex_matrix())
    original_face_centers = np.zeros((len(original_faces), 3))
    
    # 计算每个面片的中心点
    for i, face in enumerate(original_faces):
        vertices = original_vertices[face]
        center = vertices.mean(axis=0)
        original_face_centers[i] = center
    
    # 分割连通部件
    ms.generate_splitting_by_connected_components()
    
    # 获取分割后的网格数量
    component_count = ms.mesh_number()
    print(f"找到 {component_count-1} 个连通部件")
    
    # 收集所有组件（从索引1开始，跳过原始网格）
    components = []
    
    # 创建组件与原始面片的映射
    component_to_faces = {}
    
    # 遍历每个组件（从索引1开始，跳过原始网格）
    for i in range(1, ms.mesh_number()):
        ms.set_current_mesh(i)
        component = ms.current_mesh()
        face_count = component.face_number()
        components.append(i)
        print(f"  组件 {i}: {face_count} 面片 ({face_count/total_faces*100:.2f}% 的总面片)")
        
        # 获取组件中的面片数据
        component_faces = np.array(component.face_matrix())
        component_vertices = np.array(component.vertex_matrix())
        
        # 计算组件面片中心点
        component_face_centers = np.zeros((len(component_faces), 3))
        for j, face in enumerate(component_faces):
            vertices = component_vertices[face]
            center = vertices.mean(axis=0)
            component_face_centers[j] = center
        
        # 查找与原始面片匹配的中心点
        matched_original_faces = []
        for j, comp_center in enumerate(component_face_centers):
            # 计算该中心点与所有原始面片中心点的距离
            distances = np.linalg.norm(original_face_centers - comp_center, axis=1)
            # 找到最小距离的索引
            closest_face_idx = np.argmin(distances)
            matched_original_faces.append(closest_face_idx)
        
        # 存储组件到原始面片的映射
        component_to_faces[i] = matched_original_faces
    
    # 创建组件输出目录
    output_dir = Path(f"{output_base}_components")
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果提供了face2label字典，分析每个组件中的GT面片
    if face2label is not None:
        print("\n分析每个组件中的GT面片比例:")
        component_gt_stats = {}
        
        for comp_idx, original_faces in component_to_faces.items():
            # 计算该组件中有多少GT面片
            gt_faces_in_component = 0
            for face_idx in original_faces:
                if face_idx in face2label and face2label[face_idx] == 1:  # 1表示GT区域
                    gt_faces_in_component += 1
            
            # 计算GT面片比例
            total_faces_in_component = len(original_faces)
            gt_ratio = gt_faces_in_component / total_faces_in_component * 100 if total_faces_in_component > 0 else 0
            
            component_gt_stats[comp_idx] = {
                'gt_faces': gt_faces_in_component,
                'total_faces': total_faces_in_component,
                'gt_ratio': gt_ratio
            }
            
            print(f"  组件 {comp_idx}: {gt_faces_in_component}/{total_faces_in_component} GT面片 ({gt_ratio:.2f}%)")
    
    # 为合并模型收集所有组件的数据
    all_vertices = []
    all_faces = []
    all_colors = []
    
    vertex_offset = 0
    
    # 保存各个组件
    for i, component_idx in enumerate(components):
        try:
            # 设置当前网格为要导出的组件
            ms.set_current_mesh(component_idx)
            
            # 获取当前组件的顶点和面片数据
            vertices = np.array(ms.current_mesh().vertex_matrix())
            faces = np.array(ms.current_mesh().face_matrix())
            
            # 创建trimesh对象 - 默认白色
            component_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # 默认白色
            component_color = [255, 255, 255, 255]  # 白色
            
            # 如果有GT信息，根据GT比例上色
            if face2label is not None and component_idx in component_gt_stats:
                gt_ratio = component_gt_stats[component_idx]['gt_ratio']
                if gt_ratio > 50:  # 如果GT面片比例超过50%，则标记为GT组件
                    component_mesh.visual.face_colors = [0, 255, 0, 255]  # 绿色
                    component_color = [0, 255, 0, 255]  # 合并模型中也使用绿色
                else:
                    component_mesh.visual.face_colors = [255, 255, 255, 255]  # 白色
            
            # 收集为合并模型的数据
            all_vertices.append(vertices)
            # 调整面片索引以适应合并
            adjusted_faces = faces + vertex_offset
            all_faces.append(adjusted_faces)
            # 为此组件的所有面片添加相同的颜色
            all_colors.extend([component_color] * len(faces))
            # 更新顶点偏移
            vertex_offset += len(vertices)
            
            # 导出组件，包含GT信息在文件名中
            gt_info = ""
            if face2label is not None and component_idx in component_gt_stats:
                gt_ratio = component_gt_stats[component_idx]['gt_ratio']
                if gt_ratio > 30:
                    gt_info = "_GT"
            
            filename = f"component_{i+1}{gt_info}{export_format}"
            output_path = str(output_dir / filename)
            component_mesh.export(output_path)
            
            # 输出信息
            face_count = ms.current_mesh().face_number()
            print(f"  导出组件 {i+1}: {filename} ({face_count} 面片)")
        except Exception as e:
            print(f"  导出组件 {i+1} 时出错: {e}")
    
    # 创建合并的模型 (只有GT为绿色，其余为白色)
    try:
        if all_vertices and all_faces:
            # 合并所有顶点和面片
            merged_vertices = np.vstack(all_vertices)
            merged_faces = np.vstack(all_faces)
            
            # 创建合并的网格
            merged_mesh = trimesh.Trimesh(vertices=merged_vertices, faces=merged_faces)
            
            # 应用收集的颜色
            merged_mesh.visual.face_colors = np.array(all_colors)
            
            # 导出合并的模型
            combined_filename = f"combined{export_format}"
            combined_output_path = str(output_dir / combined_filename)
            merged_mesh.export(combined_output_path)
            print(f"\n已创建并导出合并的模型: {combined_output_path}")
    except Exception as e:
        print(f"创建合并模型时出错: {e}")
    
    # 删除临时文件
    try:
        os.remove(temp_path)
    except:
        pass
    
    print(f"\n所有组件已保存到: {output_dir}")
    
    # 返回组件数量、组件目录路径和组件到原始面片的映射
    return len(components), output_dir, component_to_faces


# 添加提取GT区域（绿色部分）的函数
def extract_gt_mesh(tmesh: Trimesh, face2label: dict[int, int]) -> Trimesh:
    """
    提取并返回只包含GT区域（绿色部分）的网格
    
    Args:
        tmesh: 原始网格
        face2label: 面片标签字典，1表示GT区域，0表示非GT区域
        
    Returns:
        Trimesh: 只包含GT区域的网格，颜色为绿色
    """
    # 获取GT的面片索引
    gt_faces_idx = [face for face, label in face2label.items() if label == 1]
    print(f"提取绿色(GT)面片: {len(gt_faces_idx)}个")
    
    # 提取GT区域的面片
    gt_faces = tmesh.faces[gt_faces_idx]
    
    # 创建GT mesh
    gt_vertices_idx = np.unique(gt_faces.flatten())
    gt_vertices = tmesh.vertices[gt_vertices_idx]
    
    # 创建顶点映射
    gt_vertex_map = {old: new for new, old in enumerate(gt_vertices_idx)}
    
    # 重新映射面片索引
    gt_faces_remapped = np.array([[gt_vertex_map[v] for v in face] for face in gt_faces])
    
    # 创建只包含GT区域的网格
    gt_mesh = Trimesh(vertices=gt_vertices, faces=gt_faces_remapped)
    
    # 设置颜色为绿色
    gt_mesh.visual.face_colors = np.array([0, 255, 0, 255])  # 纯绿色
    
    return gt_mesh

# 添加提取非GT区域（白色部分）的函数
def extract_non_gt_mesh(tmesh: Trimesh, face2label: dict[int, int]) -> Trimesh:
    """
    提取并返回只包含非GT区域（白色部分）的网格
    
    Args:
        tmesh: 原始网格
        face2label: 面片标签字典，1表示GT区域，0表示非GT区域
        
    Returns:
        Trimesh: 只包含非GT区域的网格，颜色为白色
    """
    # 获取非GT的面片索引
    non_gt_faces_idx = [face for face, label in face2label.items() if label == 0]
    print(f"提取白色(非GT)面片: {len(non_gt_faces_idx)}个")
    
    # 提取非GT区域的面片
    non_gt_faces = tmesh.faces[non_gt_faces_idx]
    
    # 创建非GT mesh
    non_gt_vertices_idx = np.unique(non_gt_faces.flatten())
    non_gt_vertices = tmesh.vertices[non_gt_vertices_idx]
    
    # 创建顶点映射
    non_gt_vertex_map = {old: new for new, old in enumerate(non_gt_vertices_idx)}
    
    # 重新映射面片索引
    non_gt_faces_remapped = np.array([[non_gt_vertex_map[v] for v in face] for face in non_gt_faces])
    
    # 创建只包含非GT区域的网格
    non_gt_mesh = Trimesh(vertices=non_gt_vertices, faces=non_gt_faces_remapped)
    
    # 设置颜色为白色
    non_gt_mesh.visual.face_colors = np.array([255, 255, 255, 255])  # 纯白色
    
    return non_gt_mesh

# 修改segment_mesh函数，增加导出单独部分的功能
def segment_mesh(filename: Path | str, config: OmegaConf, visualize=False, extension='glb', target_labels=None, texture=False, view_name=None) -> Trimesh:
    """
    Segment 3D mesh and save results
    
    Args:
        filename: Model file path
        config: Configuration object
        visualize: Whether to generate visualization results
        extension: Export file extension
        target_labels: Target label count for post-processing (不再使用)
        texture: Whether to keep texture
        view_name: Specific single view name to use, if None use all views
        
    Returns:
        Mesh object with color labels
    """
    print(f"\n[PATH DEBUG] ===== Starting segment_mesh =====")
    print(f"[PATH DEBUG] Current working directory: {os.getcwd()}")
    print(f"[PATH DEBUG] Model file: {filename}")
    
    print('Segmenting mesh with segmentation model: ', filename)
    
    # 确保使用的是Path对象
    if isinstance(filename, str):
        filename = Path(filename)
    
    # 复制配置，避免修改原配置
    config = copy.deepcopy(config)
    
    # 设置缓存路径
    if "cache" in config:
        cache_path = Path(config.cache) / filename.stem
        config.cache = cache_path
        
    # 设置输出路径
    output_path = Path(config.output) / filename.stem
    config.output = output_path
    
    # 创建分割模型
    model = SegmentationModelMesh(config)
    
    # 读取模型
    tmesh = read_mesh(filename, norm=False)
    if not texture:
        tmesh = remove_texture(tmesh, visual_kind='vertex')
    
    # 设置模型ID
    model.model_id = filename.stem
    print(f"Model ID: {model.model_id}")
    
    # 读取相机位置信息
    model_id = filename.stem
    camera_info = read_camera_position(model_id)
    
    # 选择渲染路径
    visualize_path = None
    if visualize:
        visualize_path = f'{config.output}/{filename.stem}_visualized'
    
    # 将读取到的相机位置应用到渲染过程中
    # 注意：target_labels参数不再有效，但保留以兼容旧接口
    if hasattr(model, 'set_camera_position'):
        # 使用该方法设置相机位置
        model.set_camera_position(camera_info)
        print("Using camera position from saved file")
        faces2label, _ = model(tmesh, visualize_path=visualize_path, view_name=view_name)
    else:
        # 如果模型没有设置相机位置的方法，抛出错误
        raise AttributeError("SegmentationModelMesh does not have a set_camera_position method")
    
    # 创建输出目录
    os.makedirs(config.output, exist_ok=True)
    
    # 保存分割结果
    output_base = f'{config.output}/{filename.stem}'
    print(f"Output base: {output_base}")
    
    # 确定导出格式
    if isinstance(filename, Path):
        export_format = filename.suffix
    else:
        export_format = Path(filename).suffix
    
    # 导出面片标签的网格
    try:
        # 导出分割后的网格
        face2label_consistent = faces2label
        
        # 创建并导出组合的分离网格（包含所有面片，绿色和白色部分）
        print("\n创建并导出完整的分离模型...")
        combined_separated_mesh = create_combined_separated_mesh(tmesh, face2label_consistent)
        combined_separated_mesh.export(f'{output_base}_combined_separated.obj')
        print(f"- 已保存完整的分离模型到 {output_base}_combined_separated.obj")
        
        # 对原始几何体进行连通组件划分 - 已注释
        '''
        print("\n提取原始几何体的连通组件并分析每个组件中的GT比例...")
        components_count, components_dir, component_to_faces = extract_components_and_save(
            tmesh, 
            output_base,
            export_format='.obj',
            face2label=face2label_consistent
        )
        print(f"- 完成连通组件提取：找到 {components_count} 个组件，已保存到 {components_dir}")
        
        # 保存组件与GT面片的对应关系数据
        component_gt_info = {}
        for comp_idx, original_faces in component_to_faces.items():
            # 计算该组件中有多少GT面片
            gt_faces = [face for face in original_faces if face in face2label_consistent and face2label_consistent[face] == 1]
            gt_ratio = len(gt_faces) / len(original_faces) * 100 if original_faces else 0
            
            component_gt_info[int(comp_idx)] = {
                'gt_faces_count': len(gt_faces),
                'total_faces': len(original_faces),
                'gt_ratio': gt_ratio,
                'is_gt_component': gt_ratio > 50
            }
        
        # 将GT信息保存为JSON文件
        import json
        # 将字典的键转换为字符串，以便JSON序列化
        component_gt_info_json = {str(k): v for k, v in component_gt_info.items()}
        with open(f'{components_dir}/component_gt_info.json', 'w') as f:
            json.dump(component_gt_info_json, f, indent=2)
        
        print(f"- 保存了组件与GT面片的对应关系到 {components_dir}/component_gt_info.json")
        '''
    except Exception as e:
        print(f"Error exporting mesh: {e}")
        import traceback
        traceback.print_exc()
    
    return tmesh


def split_mesh_by_labels_with_hole(mesh: Trimesh, face2label: dict[int, int]) -> tuple[Trimesh, Trimesh]:
    """
    将mesh按照标签切分为两个部分，确保非GT部分在边界处是开放的
    """
    # 获取GT和非GT的面片索引
    gt_faces_idx = [face for face, label in face2label.items() if label == 1]
    non_gt_faces_idx = [face for face, label in face2label.items() if label == 0]
    
    # 提取GT区域的面片
    gt_faces = mesh.faces[gt_faces_idx]
    # 提取非GT区域的面片
    non_gt_faces = mesh.faces[non_gt_faces_idx]
    
    # 创建GT mesh (与原来相同)
    gt_vertices_idx = np.unique(gt_faces.flatten())
    gt_vertices = mesh.vertices[gt_vertices_idx]
    gt_vertex_map = {old: new for new, old in enumerate(gt_vertices_idx)}
    gt_faces_remapped = np.array([[gt_vertex_map[v] for v in face] for face in gt_faces])
    gt_mesh = Trimesh(vertices=gt_vertices, faces=gt_faces_remapped)
    
    # 关键修改：识别边界
    # 首先构建边到面片的映射
    edges_to_faces = defaultdict(list)
    for face_idx, face in enumerate(mesh.faces):
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            edges_to_faces[edge].append(face_idx)
    
    # 找出边界边（只属于一个面片或连接GT和非GT区域的边）
    boundary_edges = set()
    for edge, face_indices in edges_to_faces.items():
        if len(face_indices) == 1 or any(face_idx in gt_faces_idx for face_idx in face_indices) and any(face_idx in non_gt_faces_idx for face_idx in face_indices):
            boundary_edges.add(edge)
    
    # 找出与边界相邻的非GT面片
    boundary_non_gt_faces = set()
    for face_idx in non_gt_faces_idx:
        face = mesh.faces[face_idx]
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            if edge in boundary_edges:
                boundary_non_gt_faces.add(face_idx)
                break
    
    # 移除这些边界面片，创建真正有洞的非GT mesh
    final_non_gt_faces_idx = [idx for idx in non_gt_faces_idx if idx not in boundary_non_gt_faces]
    non_gt_faces = mesh.faces[final_non_gt_faces_idx]
    
    # 创建非GT mesh（与原来相同，但使用修改后的面片集合）
    non_gt_vertices_idx = np.unique(non_gt_faces.flatten())
    non_gt_vertices = mesh.vertices[non_gt_vertices_idx]
    non_gt_vertex_map = {old: new for new, old in enumerate(non_gt_vertices_idx)}
    non_gt_faces_remapped = np.array([[non_gt_vertex_map[v] for v in face] for face in non_gt_faces])
    non_gt_mesh = Trimesh(vertices=non_gt_vertices, faces=non_gt_faces_remapped)
    
    # 设置颜色
    gt_mesh.visual.face_colors = np.array([0, 255, 0, 255])  # 纯绿色
    non_gt_mesh.visual.face_colors = np.array([255, 255, 255, 255])  # 白色
    
    return gt_mesh, non_gt_mesh


if __name__ == '__main__':
    import glob
    import argparse
    from natsort import natsorted


    def read_filenames(pattern: str):
        """
        读取匹配模式的文件名列表
        """
        print(f"[PATH DEBUG] Reading filenames with pattern: {pattern}")
        filenames = glob.glob(pattern)
        filenames = map(Path, filenames)
        filenames = natsorted(list(set(filenames)))
        print(f'[PATH DEBUG] Found {len(filenames)} files matching pattern')
        for i, f in enumerate(filenames):
            print(f"  {i+1}. {f}")
        return filenames
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='3D网格分割工具')
    parser.add_argument('--config', type=str, 
                        default='configs/mesh_segmentation.yaml',
                        help='配置文件路径')
    parser.add_argument('--model', type=str, default=None,
                        help='指定要处理的单个模型文件名（位于assets目录下），如果不指定则处理所有模型')
    parser.add_argument('--view', type=str, default=None,
                        help='指定单一视角名称，例如 "center", "left", "right" 等，默认使用所有视角')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='是否生成可视化结果')
    
    args = parser.parse_args()
    print(f"[PATH DEBUG] Command line arguments: {args}")
    
    # 设置文件路径 - 使用相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[PATH DEBUG] Script directory: {script_dir}")
    
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    print(f"[PATH DEBUG] Project root: {project_root}")
    
    assets_dir = os.path.join(project_root, 'assets')
    print(f"[PATH DEBUG] Assets directory: {assets_dir}")
    print(f"[PATH DEBUG] Assets directory exists: {os.path.exists(assets_dir)}")
    
    # 打印assets目录内容
    if os.path.exists(assets_dir):
        print(f"[PATH DEBUG] Assets directory content:")
        for item in os.listdir(assets_dir):
            print(f"  - {item}")
    
    if args.model:
        # 处理单个模型
        model_path = os.path.join(assets_dir, args.model)
        print(f"[PATH DEBUG] Looking for model at: {model_path}")
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"[PATH DEBUG] Model file not found directly, trying with extensions")
            # 尝试添加扩展名
            for ext in ['.glb', '.obj', '.ply']:
                ext_path = model_path + ext
                print(f"[PATH DEBUG] Trying with extension: {ext_path}, Exists: {os.path.exists(ext_path)}")
                if os.path.exists(model_path + ext):
                    model_path = model_path + ext
                    break
        
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 '{model_path}' 不存在")
            exit(1)
            
        filenames = [Path(model_path)]
        print(f'处理单个模型: {model_path}')
    else:
        # 处理所有模型
        pattern = os.path.join(assets_dir, '*.glb')
        filenames = read_filenames(pattern)
    
    # 加载配置
    config_path = os.path.join(project_root, args.config)
    print(f"[PATH DEBUG] Looking for config at: {config_path}")
    print(f"[PATH DEBUG] Config exists: {os.path.exists(config_path)}")
    
    if not os.path.exists(config_path):
        # 尝试在当前目录查找配置文件
        config_path = args.config
        print(f"[PATH DEBUG] Looking for config in current directory: {config_path}")
        print(f"[PATH DEBUG] Config exists: {os.path.exists(config_path)}")
    
    config = OmegaConf.load(config_path)
    print(f"[PATH DEBUG] Loaded config: {config}")
    
    # 确保输出目录使用相对路径
    if 'output' in config and not os.path.isabs(config.output):
        original_output = config.output
        config.output = os.path.join(project_root, config.output)
        print(f"[PATH DEBUG] Adjusted output path from {original_output} to {config.output}")
    
    # 打印使用的视角信息
    if args.view:
        print(f"使用单一视角模式: '{args.view}'")
        # 验证视角名称有效性
        if args.view not in VIEW_POSITIONS:
            available_views = list(VIEW_POSITIONS.keys())
            print(f"错误: 视角 '{args.view}' 不存在")
            print(f"可用的视角有: {', '.join(available_views)}")
            exit(1)
    else:
        print("使用多视角模式")
    
    # 处理每个文件
    for i, filename in enumerate(filenames):
        print(f"处理 {i+1}/{len(filenames)}: {filename}")
        segment_mesh(filename, config, visualize=args.visualize, view_name=args.view)
    
    # Clean the assets directory
    def clean_assets_directory():
        """
        Remove all files from the assets directory
        """
        assets_dir = os.path.join(project_root, 'assets')
        print("Cleaning assets directory...")
        
        # Check if directory exists
        if not os.path.exists(assets_dir):
            print("Assets directory does not exist.")
            return
            
        # List all files and subdirectories
        for item in os.listdir(assets_dir):
            item_path = os.path.join(assets_dir, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"Removed file: {item}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Removed directory: {item}")
            except Exception as e:
                print(f"Error removing {item}: {e}")
                
        print("Assets directory has been cleaned.")
    
