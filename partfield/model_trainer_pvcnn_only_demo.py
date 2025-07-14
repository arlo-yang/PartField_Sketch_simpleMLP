import torch  # PyTorch深度学习框架，提供张量计算和自动求导
import lightning.pytorch as pl  # PyTorch Lightning框架，简化深度学习训练代码
from .dataloader import Demo_Dataset, Demo_Remesh_Dataset, Correspondence_Demo_Dataset  # 自定义数据集类，用于加载不同类型的3D数据
from torch.utils.data import DataLoader  # PyTorch数据加载器，用于批量加载数据
from partfield.model.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat  # 点云编码器和采样功能
from partfield.model.UNet.model import ResidualUNet3D  # 3D UNet模型，用于体素处理
from partfield.model.triplane import TriplaneTransformer, get_grid_coord  # 三平面Transformer模型和网格坐标生成函数
from partfield.model.model_utils import VanillaMLP  # 简单的多层感知器实现，用于特征解码
import torch.nn.functional as F  # PyTorch函数式API，提供各种激活函数和操作
import torch.nn as nn  # PyTorch神经网络模块，提供层和损失函数
import os  # 操作系统接口，用于文件和目录操作
import trimesh  # 处理三角网格的库，用于3D模型操作
import skimage  # 科学图像处理库，用于图像分析和处理
import numpy as np  # NumPy科学计算库，提供多维数组支持
import h5py  # HDF5文件格式接口，用于存储和处理大型数据集
import torch.distributed as dist  # PyTorch分布式训练支持
import json  # JSON处理库，用于配置文件和结果序列化
import gc  # 垃圾回收控制接口，用于内存管理优化
import time  # 时间相关功能，用于性能计时
from plyfile import PlyData, PlyElement  # PLY文件格式处理库，用于3D点云和网格IO操作


class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # 保存超参数，便于模型恢复和检查点加载
        self.save_hyperparameters()
        self.cfg = cfg
        # 关闭自动优化，手动控制优化过程
        self.automatic_optimization = False
        
        # 三平面特征表示的配置
        self.triplane_resolution = cfg.triplane_resolution
        self.triplane_channels_low = cfg.triplane_channels_low
        
        # 初始化三平面Transformer，用于处理和增强特征
        self.triplane_transformer = TriplaneTransformer(
            input_dim=cfg.triplane_channels_low * 2,     # 输入通道数
            transformer_dim=1024,                         # Transformer内部维度
            transformer_layers=6,                         # Transformer层数
            transformer_heads=8,                          # 注意力头数
            triplane_low_res=32,                          # 低分辨率三平面尺寸
            triplane_high_res=128,                        # 高分辨率三平面尺寸
            triplane_dim=cfg.triplane_channels_high,      # 输出通道维度
        )
        
        # SDF解码器：将特征解码为有符号距离场值
        self.sdf_decoder = VanillaMLP(input_dim=64,       # 输入特征维度
                                      output_dim=1,        # 输出维度(SDF标量)
                                      out_activation="tanh", # 输出激活函数
                                      n_neurons=64,        # 隐藏层神经元数
                                      n_hidden_layers=6)   # 隐藏层数量
        
        # 控制是否使用PVCNN点云特征提取器
        self.use_pvcnn = cfg.use_pvcnnonly
        self.use_2d_feat = cfg.use_2d_feat
        
        # 初始化PVCNN编码器，用于从点云提取三平面特征
        if self.use_pvcnn:
            self.pvcnn = TriPlanePC2Encoder(
                cfg.pvcnn,                         # PVCNN配置参数
                device="cuda",                     # 运行设备
                shape_min=-1,                      # 输入形状范围最小值
                shape_length=2,                    # 输入形状范围长度
                use_2d_feat=self.use_2d_feat)      # 是否使用2D特征
        
        # 可学习的逻辑比例因子，用于调整损失计算
        self.logit_scale = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        
        # 用于采样的网格坐标生成器
        self.grid_coord = get_grid_coord(256)
        
        # 定义损失函数
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss(reduction='none')

        # 可选的2D特征回归解码器
        if cfg.regress_2d_feat:
            self.feat_decoder = VanillaMLP(input_dim=64,
                                output_dim=192,           # 输出2D特征维度
                                out_activation="GELU",    # 输出激活函数
                                n_neurons=64,             # 隐藏层神经元数
                                n_hidden_layers=6)        # 隐藏层数量

    def predict_dataloader(self):
        """
        创建并返回用于推理的数据加载器
        根据配置选择不同类型的数据集
        """
        if self.cfg.remesh_demo:
            # 使用重新网格化处理的数据集
            dataset = Demo_Remesh_Dataset(self.cfg)        
        elif self.cfg.correspondence_demo:
            # 用于对应性分析的数据集
            dataset = Correspondence_Demo_Dataset(self.cfg)
        else:
            # 默认使用标准演示数据集
            dataset = Demo_Dataset(self.cfg)

        # 配置数据加载器参数
        dataloader = DataLoader(dataset, 
                            num_workers=self.cfg.dataset.val_num_workers,  # 数据加载的工作线程数
                            batch_size=self.cfg.dataset.val_batch_size,    # 批次大小
                            shuffle=False,                                 # 不打乱数据顺序
                            pin_memory=True,                               # 数据加载到固定内存，加速GPU传输
                            drop_last=False)                               # 保留不足一个批次的数据
        
        return dataloader           


    @torch.no_grad()  # 禁用梯度计算，节省内存
    def predict_step(self, batch, batch_idx):
        """
        执行推理步骤，从3D模型中提取特征并保存结果
        
        处理流程：
        1. 加载点云/网格数据
        2. 通过PVCNN提取初始特征
        3. 使用TriplaneTransformer增强特征
        4. 提取部分特征并进行采样
        5. 可视化特征并保存结果
        """
        # 创建结果保存目录
        save_dir = f"exp_results/{self.cfg.result_name}"
        os.makedirs(save_dir, exist_ok=True)

        # 获取模型ID和视图ID
        uid = batch['uid'][0]
        view_id = 0
        starttime = time.time()
        
        # 跳过某些特定模型
        if uid == "car" or uid == "complex_car":
            print("Skipping this for now.")
            print(uid)
            return

        # 如果已经处理过该模型，则跳过
        if os.path.exists(f'{save_dir}/part_feat_{uid}_{view_id}.npy') or os.path.exists(f'{save_dir}/part_feat_{uid}_{view_id}_batch.npy'):
            print("Already processed "+uid)
            return

        # 确认批次大小为1
        N = batch['pc'].shape[0]
        assert N == 1

        # 打印输入点云维度
        print(f"输入点云维度: {batch['pc'].shape}")  # [B, N, 3]
        
        # use pvcnn to extract 3d feat
        if self.use_2d_feat:
            print("ERROR. Dataloader not implemented with input 2d feat.")
            exit()
        else:
            # obtain 3d feat from pvcnn
            pc_feat = self.pvcnn(batch['pc'], batch['pc'])
            print(f"PVCNN输出三平面特征维度: {pc_feat.shape}")  #  [B, 3, 256, 128, 128]

        # use Triplane Transformer to enhance 3d feat
        planes = pc_feat
        planes = self.triplane_transformer(planes)
        print(f"Transformer增强后特征维度: {planes.shape}")  #  [B, 3, 512, 128, 128]
        
        # separate sdf feat and part feat
        sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)
        print(f"SDF特征维度: {sdf_planes.shape}")  #  [B, 3, 64, 128, 128]
        print(f"部件特征维度: {part_planes.shape}")  # [B, 3, 448, 128, 128]

        # 处理点云数据
        if self.cfg.is_pc:
            # 将点云转换为张量并采样特征
            tensor_vertices = batch['pc'].reshape(1, -1, 3).cuda().to(torch.float16)
            point_feat = sample_triplane_feat(part_planes, tensor_vertices) # N, M, C
            point_feat = point_feat.cpu().detach().numpy().reshape(-1, 448)

            # 保存提取的特征
            np.save(f'{save_dir}/part_feat_{uid}_{view_id}.npy', point_feat)
            print(f"Exported part_feat_{uid}_{view_id}.npy")

            ###########
            # 使用PCA降维，用于可视化
            from sklearn.decomposition import PCA
            # 归一化特征
            data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

            pca = PCA(n_components=3)

            # 将高维特征降至3维用于RGB颜色表示
            data_reduced = pca.fit_transform(data_scaled)
            data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
            colors_255 = (data_reduced * 255).astype(np.uint8)

            # 获取原始点云数据
            points = batch['pc'].squeeze().detach().cpu().numpy()

            # 为点云添加颜色
            if colors_255 is None:
                colors_255 = np.full_like(points, 255)  # 默认白色
            else:
                assert colors_255.shape == points.shape, "Colors must have the same shape as points"
            
            # 创建包含颜色的PLY格式结构
            vertex_data = np.array(
                [(*point, *color) for point, color in zip(points, colors_255)],
                dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
            )

            # 保存为PLY文件
            el = PlyElement.describe(vertex_data, "vertex")
            filename = f'{save_dir}/feat_pca_{uid}_{view_id}.ply'
            PlyData([el], text=True).write(filename)
            print(f"Saved PLY file: {filename}")
            ############
        
        else:
            # 处理网格数据
            use_cuda_version = True
            if use_cuda_version:
                # 在三角形面上采样点
                def sample_points(vertices, faces, n_point_per_face):
                    """
                    在三角网格的面上进行均匀采样
                    使用重心坐标生成每个面上的随机点
                    """
                    # 生成随机重心坐标
                    n_f = faces.shape[0]
                    u = torch.sqrt(torch.rand((n_f, n_point_per_face, 1),
                                                device=vertices.device,
                                                dtype=vertices.dtype))
                    v = torch.rand((n_f, n_point_per_face, 1),
                                    device=vertices.device,
                                    dtype=vertices.dtype)
                    w0 = 1 - u
                    w1 = u * (1 - v)
                    w2 = u * v

                    # 获取三角形的三个顶点
                    face_v_0 = torch.index_select(vertices, 0, faces[:, 0].reshape(-1))
                    face_v_1 = torch.index_select(vertices, 0, faces[:, 1].reshape(-1))
                    face_v_2 = torch.index_select(vertices, 0, faces[:, 2].reshape(-1))
                    
                    # 使用重心坐标生成点
                    points = w0 * face_v_0.unsqueeze(dim=1) + w1 * face_v_1.unsqueeze(dim=1) + w2 * face_v_2.unsqueeze(dim=1)
                    return points

                def sample_and_mean_memory_save_version(part_planes, tensor_vertices, n_point_per_face):
                    """
                    分批处理顶点以避免内存溢出
                    对每个面上采样的点进行特征提取并计算平均值
                    """
                    print(f"输入三平面特征维度: {part_planes.shape}")
                    print(f"输入顶点维度: {tensor_vertices.shape}")  # 应为 [1, n_faces * n_point_per_face, 3]
                    print(f"每面采样点数: {n_point_per_face}")
                    
                    # 每批处理的点数
                    n_sample_each = self.cfg.n_sample_each # 分批处理以避免OOM
                    n_v = tensor_vertices.shape[1]
                    n_faces_total = n_v // n_point_per_face
                    print(f"总面数: {n_faces_total}")
                    
                    n_sample = n_v // n_sample_each + 1
                    print(f"需要的批次数: {n_sample}")
                    all_sample = []
                    
                    # 分批处理所有顶点
                    for i_sample in range(n_sample):
                        batch_start = i_sample * n_sample_each
                        batch_end = min(batch_start + n_sample_each, n_v)
                        print(f"处理批次 {i_sample+1}/{n_sample}, 点索引范围: {batch_start}:{batch_end}")
                        
                        # 提取当前批次中点的特征
                        current_vertices = tensor_vertices[:, batch_start:batch_end]
                        print(f"当前批次顶点维度: {current_vertices.shape}")
                        
                        sampled_feature = sample_triplane_feat(part_planes, current_vertices)
                        print(f"采样特征原始维度: {sampled_feature.shape}")  # 应为 [1, batch_size, 448]
                        
                        assert sampled_feature.shape[1] % n_point_per_face == 0
                        n_faces_batch = sampled_feature.shape[1] // n_point_per_face
                        
                        # 重塑以计算每个面的平均特征
                        sampled_feature = sampled_feature.reshape(1, -1, n_point_per_face, sampled_feature.shape[-1])
                        print(f"重塑后特征维度: {sampled_feature.shape}")  # 应为 [1, n_faces_batch, n_point_per_face, 448]
                        
                        sampled_feature = torch.mean(sampled_feature, axis=-2)
                        print(f"平均后特征维度: {sampled_feature.shape}")  # 应为 [1, n_faces_batch, 448]
                        
                        all_sample.append(sampled_feature)
                    
                    # 合并所有批次的结果
                    result = torch.cat(all_sample, dim=1)
                    print(f"最终特征维度: {result.shape}")  # 应为 [1, n_faces_total, 448]
                    return result
                
                # 根据配置决定是提取顶点特征还是面特征
                if self.cfg.vertex_feature:
                    # 处理顶点特征
                    tensor_vertices = batch['vertices'][0].reshape(1, -1, 3).to(torch.float32)
                    point_feat = sample_and_mean_memory_save_version(part_planes, tensor_vertices, 1)
                else:
                    # 处理面特征 
                    print("!!!!!!!!!!!!!!! 处理面特征 !!!!!!!!!!!!!!!")
                    n_point_per_face = self.cfg.n_point_per_face
                    # 在每个面上采样点
                    tensor_vertices = sample_points(batch['vertices'][0], batch['faces'][0], n_point_per_face)
                    tensor_vertices = tensor_vertices.reshape(1, -1, 3).to(torch.float32)
                    # 提取面上采样点的特征并计算平均值
                    point_feat = sample_and_mean_memory_save_version(part_planes, tensor_vertices, n_point_per_face)  # N, M, C

                # 计算并输出特征提取所需时间
                print("Time elapsed for feature prediction: " + str(time.time() - starttime))
                point_feat = point_feat.reshape(-1, 448).cpu().numpy()
                # 保存提取的特征
                np.save(f'{save_dir}/part_feat_{uid}_{view_id}_batch.npy', point_feat)
                print(f"Exported part_feat_{uid}_{view_id}.npy")

                ###########
                # 使用PCA进行特征可视化
                from sklearn.decomposition import PCA
                # 归一化特征向量
                data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

                pca = PCA(n_components=3)

                # 降维到3D空间用于颜色映射
                data_reduced = pca.fit_transform(data_scaled)
                data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
                colors_255 = (data_reduced * 255).astype(np.uint8)
                
                # 获取网格顶点和面
                V = batch['vertices'][0].cpu().numpy()
                F = batch['faces'][0].cpu().numpy()
                
                # 创建着色网格：根据配置决定是顶点着色还是面着色
                if self.cfg.vertex_feature:
                    colored_mesh = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=colors_255, process=False)
                else:
                    colored_mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=colors_255, process=False)
                
                # 导出彩色网格
                colored_mesh.export(f'{save_dir}/feat_pca_{uid}_{view_id}.ply')
                ############
                # 清理GPU内存
                torch.cuda.empty_cache()

            else:
                # CPU版本的实现（较慢）
                # 获取网格顶点和面
                V = batch['vertices'][0].cpu().numpy()
                F = batch['faces'][0].cpu().numpy()

                ##### 遍历所有面进行采样 #####
                num_samples_per_face = self.cfg.n_point_per_face

                all_point_feats = []
                for face in F:
                    # 获取当前面的三个顶点
                    v0, v1, v2 = V[face]

                    # 生成随机重心坐标
                    u = np.random.rand(num_samples_per_face, 1)
                    v = np.random.rand(num_samples_per_face, 1)
                    is_prob = (u+v) >1
                    u[is_prob] = 1 - u[is_prob]
                    v[is_prob] = 1 - v[is_prob]
                    w = 1 - u - v
                    
                    # 使用重心坐标计算笛卡尔点坐标
                    points = u * v0 + v * v1 + w * v2 

                    # 对采样点进行特征提取
                    tensor_vertices = torch.from_numpy(points.copy()).reshape(1, -1, 3).cuda().to(torch.float32)
                    point_feat = sample_triplane_feat(part_planes, tensor_vertices) # N, M, C 

                    # 计算面的平均特征
                    point_feat = torch.mean(point_feat, axis=1).cpu().detach().numpy()
                    all_point_feats.append(point_feat)                  
                ##############################
                
                # 整理所有面的特征
                all_point_feats = np.array(all_point_feats).reshape(-1, 448)
                
                point_feat = all_point_feats

                # 保存特征
                np.save(f'{save_dir}/part_feat_{uid}_{view_id}.npy', point_feat)
                print(f"Exported part_feat_{uid}_{view_id}.npy")
                
                ###########
                # PCA特征可视化
                from sklearn.decomposition import PCA
                data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

                pca = PCA(n_components=3)

                data_reduced = pca.fit_transform(data_scaled)
                data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
                colors_255 = (data_reduced * 255).astype(np.uint8)

                # 创建彩色网格并导出
                colored_mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=colors_255, process=False)
                colored_mesh.export(f'{save_dir}/feat_pca_{uid}_{view_id}.ply')
                ############

        # 在reshape之后
        print(f"\n特征形状: {point_feat.shape}")  # 显示 [n_faces, 448]
        print(f"总面数: {point_feat.shape[0]}")
        print(f"特征维度: {point_feat.shape[1]}")

        # 打印一些样本特征
        print("\n第一个面的特征向量前10个值:")
        print(point_feat[0, :10])  # 打印第一个面特征的前10个值

        print("\n第二个面的特征向量前10个值:")
        print(point_feat[1, :10])  # 打印第二个面特征的前10个值

        # 计算两个面特征的余弦相似度
        from scipy.spatial.distance import cosine
        similarity = 1 - cosine(point_feat[0], point_feat[1])
        print(f"\n第一个面和第二个面的特征相似度: {similarity:.4f}")

        # 随机选择两个可能不相邻的面
        mid_index = point_feat.shape[0] // 2
        similarity_distant = 1 - cosine(point_feat[0], point_feat[mid_index])
        print(f"第一个面和中间面的特征相似度: {similarity_distant:.4f}")

        # 输出总耗时
        print("Time elapsed: " + str(time.time()-starttime))
            
        return 