import os
# 尝试使用 osmesa 后端
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

### START VOODOO ###
# Dark encantation for disabling anti-aliasing in pyrender (if needed)
import OpenGL.GL
antialias_active = False
old_gl_enable = OpenGL.GL.glEnable
def new_gl_enable(value):
    if not antialias_active and value == OpenGL.GL.GL_MULTISAMPLE:
        OpenGL.GL.glDisable(value)
    else:
        old_gl_enable(value)
OpenGL.GL.glEnable = new_gl_enable
import pyrender
### END VOODOO ###

import cv2
import numpy as np
import torch
from numpy.random import RandomState
from PIL import Image
from pyrender.shader_program import ShaderProgramCache as DefaultShaderCache
from trimesh import Trimesh, Scene
from omegaconf import OmegaConf
from tqdm import tqdm

from samesh.data.common import NumpyTensor
from samesh.data.loaders import scene2mesh
from samesh.utils.cameras import HomogeneousTransform, sample_view_matrices, sample_view_matrices_polyhedra
from samesh.utils.math import range_norm
from samesh.utils.mesh import duplicate_verts
from samesh.renderer.shader_programs import *


def colormap_faces(faces: NumpyTensor['h w'], background=np.array([255, 255, 255])) -> Image.Image:
    """将面片ID映射为随机颜色图像
    
    Args:
        faces: 形状为[height, width]的面片ID数组，每个值代表一个面片的唯一标识符
        background: 背景颜色，默认为白色 [255, 255, 255]
    
    Returns:
        Image.Image: 渲染后的图像，每个面片都有唯一的随机颜色
    """
    palette = RandomState(0).randint(0, 255, (np.max(faces + 2), 3))  # 为每个面片生成随机颜色
    palette[0] = background  # 设置背景色
    image = palette[faces + 1, :].astype(np.uint8)  # 将面片ID映射到对应的颜色
    return Image.fromarray(image)


def colormap_norms(norms: NumpyTensor['h w'], background=np.array([255, 255, 255])) -> Image.Image:
    """将法线向量映射为颜色图像
    
    Args:
        norms: 形状为[height, width]的法线向量数组，范围为[-1, 1]
        background: 背景颜色，默认为白色 [255, 255, 255]
    
    Returns:
        Image.Image: 渲染后的图像，法线方向被映射为RGB颜色
    """
    norms = (norms + 1) / 2  # 将[-1, 1]映射到[0, 1]
    norms = (norms * 255).astype(np.uint8)  # 转换为8位颜色值
    return Image.fromarray(norms)


DEFAULT_CAMERA_PARAMS = {'fov': 75, 'znear': 0.1, 'zfar': 15}


class Renderer:
    """3D模型渲染器，支持多种渲染模式
    
    支持的渲染模式:
    - 默认渲染：标准的PBR材质渲染
    - 法线渲染：显示表面法线方向
    - 面片ID渲染：为每个面片分配唯一颜色
    - 重心坐标渲染：显示三角形内部的插值
    """
    
    def __init__(self, config: OmegaConf):
        """初始化渲染器
        
        Args:
            config: 包含渲染设置的配置对象
                   必须包含 target_dim (目标图像尺寸)
        """
        self.config = config
        self.renderer = pyrender.OffscreenRenderer(*config.target_dim)
        # 初始化不同类型的着色器
        self.shaders = {
            'default': DefaultShaderCache(),     # 默认PBR着色器
            'normals': NormalShaderCache(),      # 法线渲染着色器
            'faceids': FaceidShaderCache(),      # 面片ID渲染着色器
            'barycnt': BarycentricShaderCache(), # 重心坐标渲染着色器
        }

    def set_object(self, source: Trimesh | Scene, smooth=False):
        """设置要渲染的3D对象
        
        Args:
            source: 输入的3D模型，可以是单个网格或场景
            smooth: 是否使用平滑着色
        """
        if isinstance(source, Scene):
            # 处理场景对象
            self.tmesh = scene2mesh(source)
            self.scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
            for name, geom in source.geometry.items():
                if name in source.graph:
                    pose, _ = source.graph[name]
                else:
                    pose = None
                self.scene.add(pyrender.Mesh.from_trimesh(geom, smooth=smooth), pose=pose)
        
        elif isinstance(source, Trimesh):
            # 处理单个网格
            self.tmesh = source
            self.scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
            self.scene.add(pyrender.Mesh.from_trimesh(source, smooth=smooth))

        else:
            raise ValueError(f'Invalid source type {type(source)}')
        
        # rearrange mesh for faceid rendering
        self.tmesh_faceid = duplicate_verts(self.tmesh)
        self.scene_faceid = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
        self.scene_faceid.add(
            pyrender.Mesh.from_trimesh(self.tmesh_faceid, smooth=smooth)
        )

    def set_camera(self, camera_params: dict = None):
        """
        """
        self.camera_params = camera_params or dict(DEFAULT_CAMERA_PARAMS)
        self.camera_params['yfov'] = self.camera_params.get('yfov', self.camera_params.pop('fov'))
        self.camera_params['yfov'] = self.camera_params['yfov'] * np.pi / 180.0
        self.camera = pyrender.PerspectiveCamera(**self.camera_params)
        
        self.camera_node        = self.scene       .add(self.camera)
        self.camera_node_faceid = self.scene_faceid.add(self.camera)
        
    def render(
        self, 
        pose: HomogeneousTransform, 
        lightdir=np.array([0.0, 0.0, 1.0]), uv_map=False, interpolate_norms=True, blur_matte=False
    ) -> dict:
        """渲染3D模型
        
        Args:
            pose: 4x4变换矩阵，定义相机位置和方向
            lightdir: 光照方向
            uv_map: 是否使用UV贴图
            interpolate_norms: 是否对法线进行插值
            blur_matte: 是否对遮罩进行模糊处理
        
        Returns:
            dict: 包含多种渲染结果的字典
                - norms: 法线图
                - depth: 深度图
                - matte: 遮罩图
                - faces: 面片ID图
        """
        self.scene       .set_pose(self.camera_node       , pose)
        self.scene_faceid.set_pose(self.camera_node_faceid, pose)

        def render(shader: str, scene):
            """使用指定的着色器渲染场景
            
            Args:
                shader: 着色器类型，可选值：'default', 'normals', 'faceids', 'barycnt'
                scene: 要渲染的场景对象
            
            Returns:
                tuple: (color_buffer, depth_buffer) 颜色缓冲和深度缓冲
            """
            self.renderer._renderer._program_cache = self.shaders[shader]
            return self.renderer.render(scene)
        
        if uv_map:
            raw_color, raw_depth = render('default', self.scene)
        raw_norms, raw_depth = render('normals', self.scene)
        raw_faces, raw_depth = render('faceids', self.scene_faceid)
        raw_bcent, raw_depth = render('barycnt', self.scene_faceid)

        def render_norms(norms: NumpyTensor['h w 3']) -> NumpyTensor['h w 3']:
            """处理法线渲染结果
            
            Args:
                norms: 原始法线渲染结果，范围[0, 255]
            
            Returns:
                处理后的法线数据，范围[-1, 1]，表示表面法线方向
            """
            return np.clip((norms / 255.0 - 0.5) * 2, -1, 1)

        def render_depth(depth: NumpyTensor['h w'], offset=2.8, alpha=0.8) -> NumpyTensor['h w']:
            """处理深度渲染结果
            
            Args:
                depth: 原始深度图
                offset: 深度偏移量，用于调整深度范围
                alpha: 深度图的透明度
            
            Returns:
                处理后的深度图，值域[0, 1]，0表示最近，1表示最远
            """
            return np.where(depth > 0, alpha * (1.0 - range_norm(depth, offset=offset)), 1)

        def render_faces(faces: NumpyTensor['h w 3']) -> NumpyTensor['h w']:
            """处理面片ID渲染结果
            
            Args:
                faces: 原始面片颜色编码，RGB三通道
            
            Returns:
                面片ID图，每个像素值代表一个唯一的面片ID，-1表示背景
            """
            faces = faces.astype(np.int32)
            # 将RGB编码转换为单一ID
            faces = faces[:, :, 0] * 65536 + faces[:, :, 1] * 256 + faces[:, :, 2]
            faces[faces == (256 ** 3 - 1)] = -1  # 设置背景为-1
            return faces

        def render_bcent(bcent: NumpyTensor['h w 3']) -> NumpyTensor['h w 3']:
            """处理重心坐标渲染结果
            
            Args:
                bcent: 原始重心坐标渲染结果
            
            Returns:
                归一化的重心坐标，范围[0, 1]，表示三角形内的插值位置
            """
            return np.clip(bcent / 255.0, 0, 1)

        def render_matte(
            norms: NumpyTensor['h w 3'],
            depth: NumpyTensor['h w'],
            faces: NumpyTensor['h w'],
            bcent: NumpyTensor['h w 3'],
            alpha=0.5, beta=0.25, 
            gaussian_kernel_width=5, 
            gaussian_sigma=1,
        ) -> NumpyTensor['h w 3']:
            """生成遮罩渲染结果
            
            Args:
                norms: 法线图
                depth: 深度图
                faces: 面片ID图
                bcent: 重心坐标图
                alpha: 漫反射强度系数
                beta: 环境光强度系数
                gaussian_kernel_width: 高斯模糊核大小
                gaussian_sigma: 高斯模糊标准差
            
            Returns:
                遮罩图，考虑了法线、深度和边缘模糊等效果
            """
            if interpolate_norms:  # 需要处理法线插值
                verts_index = self.tmesh.faces[faces.reshape(-1)]     # (n, 3)
                verts_norms = self.tmesh.vertex_normals[verts_index]  # (n, 3, 3)
                norms = np.sum(verts_norms * bcent.reshape(-1, 3, 1), axis=1)
                norms = norms.reshape(bcent.shape)

            # 计算漫反射光照
            diffuse = np.sum(norms * lightdir, axis=2)
            diffuse = np.clip(diffuse, -1, 1)
            # 生成最终遮罩
            matte = 255 * (diffuse[:, :, None] * alpha + beta)
            matte = np.where(depth[:, :, None] > 0, matte, 255)
            matte = np.clip(matte, 0, 255).astype(np.uint8)
            matte = np.repeat(matte, 3, axis=2)
            
            # 可选的边缘模糊处理
            if blur_matte:
                matte = (faces == -1)[:, :, None] * matte + \
                        (faces != -1)[:, :, None] * cv2.GaussianBlur(
                            matte, 
                            (gaussian_kernel_width, gaussian_kernel_width), 
                            gaussian_sigma
                        )
            return matte 

        norms = render_norms(raw_norms)
        depth = render_depth(raw_depth)
        faces = render_faces(raw_faces)
        bcent = render_bcent(raw_bcent)
        matte = raw_color if uv_map else render_matte(norms, raw_depth, faces, bcent) # use original depth for matte

        return {'norms': norms, 'depth': depth, 'matte': matte, 'faces': faces}


def render_multiview(
    renderer: Renderer,
    camera_generation_method='sphere',
    renderer_args: dict=None,
    sampling_args: dict=None,
    lighting_args: dict=None, 
    lookat_position=np.array([0, 0, 0]),
    verbose=True,
) -> list[Image.Image]:
    """
    """
    lookat_position_torch = torch.from_numpy(lookat_position)
    if camera_generation_method == 'sphere':
        views = sample_view_matrices(lookat_position=lookat_position_torch, **sampling_args).numpy()
    else:
        views = sample_view_matrices_polyhedra(camera_generation_method, lookat_position=lookat_position_torch, **sampling_args).numpy()
    
    def compute_lightdir(pose: HomogeneousTransform) -> NumpyTensor[3]:
        """
        """
        lightdir = pose[:3, 3] - (lookat_position)
        return lightdir / np.linalg.norm(lightdir)

    renders = []
    if verbose:
        views = tqdm(views, 'Rendering Multiviews...')
    for pose in views:
        outputs = renderer.render(pose, lightdir=compute_lightdir(pose), **renderer_args)
        outputs['matte'] = Image.fromarray(outputs['matte'])
        outputs['poses'] = pose
        renders.append(outputs)
    return {
        name: [render[name] for render in renders] for name in renders[0].keys()
    }


if __name__ == '__main__':
    from PIL import Image
    from samesh.data.loaders import read_mesh, read_scene, remove_texture, scene2mesh
    from samesh.models.shape_diameter_function import shape_diameter_function, colormap_shape_diameter_function, prep_mesh_shape_diameter_function

    # 测试不同格式的模型
    models = [
        '/hy-tmp/samesh/assets/potion.glb',      # glb格式
        '/hy-tmp/samesh/assets/yy_merged.obj',               # obj格式
        '/hy-tmp/samesh/assets/jacket.glb'       # glb格式
    ]

    pose = np.array([
        [ 1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  1,  2.5],
        [ 0,  0,  0,  1],
    ])

    renderer = Renderer(OmegaConf.create({
        'target_dim': (1024, 1024),
    }))
    
    # 测试 OBJ 模型加载和渲染
    try:
        # 使用 trimesh 加载 OBJ 文件
        import trimesh
        source = trimesh.load(models[1])  # 加载 OBJ 文件
        
        # 如果模型太大或太小，可以进行缩放
        source.vertices -= source.center_mass  # 将模型中心移到原点
        scale = 1.0 / np.max(np.abs(source.vertices))  # 计算缩放因子
        source.vertices *= scale  # 应用缩放
        
        # 准备渲染
        source = prep_mesh_shape_diameter_function(source)
        source = colormap_shape_diameter_function(source, shape_diameter_function(source))
        
        renderer.set_object(source)
        renderer.set_camera()
        renders = renderer.render(pose)
        
        # 打印模型信息
        print(f"\n模型信息:")
        print(f"顶点数量: {len(source.vertices)}")
        print(f"面片数量: {len(source.faces)}")
        print(f"渲染尺寸: {renders['matte'].shape}")
        
        # 保存渲染结果
        for k, v in renders.items():
            print(f"{k}: {v.shape}")
        
        image = Image.fromarray(renders['matte'])
        image.save('test_matte_obj.png')
        image_faceids = colormap_faces(renders['faces'])
        image_faceids.save('test_faceids_obj.png')
        image_norms = colormap_norms(renders['norms'])
        image_norms.save('test_norms_obj.png')
        
    except Exception as e:
        print(f"加载或渲染OBJ文件时出错: {e}")