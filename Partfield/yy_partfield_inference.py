'''
PartField 特征提取推理脚本 - 自定义版本
该脚本用于从3D模型中提取PartField特征，这些特征可用于后续的部件分割和分析。

使用方法:
python yy_partfield_inference.py -c configs/final/demo.yaml --opts continue_ckpt model/model_objaverse.ckpt
'''

from partfield.config import default_argument_parser, setup  # 导入配置相关函数
from lightning.pytorch import seed_everything, Trainer  # 导入PyTorch Lightning训练器
from lightning.pytorch.strategies import DDPStrategy  # 分布式数据并行策略
from lightning.pytorch.callbacks import ModelCheckpoint  # 模型检查点回调
import lightning
import torch
import glob
import os, sys
import numpy as np
import random
from partfield.dataloader import Demo_Dataset
import trimesh
from torch.utils.data import DataLoader
import time

class CustomWholeObjDataset(Demo_Dataset):
    """
    自定义数据集，继承自Demo_Dataset，专门处理URDF目录结构
    只加载每个ID文件夹中的yy_merged.obj文件
    """
    def __init__(self, cfg):
        # 不直接调用父类的__init__，因为我们需要自定义数据加载逻辑
        torch.utils.data.Dataset.__init__(self)
        
        # 确保使用正确的数据路径
        self.data_path = "/hy-tmp/PartField_Sketch_simpleMLP/data/urdf"  # 直接硬编码URDF路径
        print(f"使用数据路径: {self.data_path}")
        
        self.is_pc = False  # 我们处理的是网格数据
        self.pc_num_pts = 100000
        self.preprocess_mesh = cfg.preprocess_mesh if hasattr(cfg, 'preprocess_mesh') else False
        
        # 获取所有ID文件夹
        self.id_folders = [d for d in os.listdir(self.data_path) 
                          if os.path.isdir(os.path.join(self.data_path, d))]
        
        # 过滤掉没有yy_merged.obj文件的文件夹
        self.data_list = []
        for id_folder in self.id_folders:
            merged_obj_path = os.path.join(self.data_path, id_folder, "yy_merged.obj")
            if os.path.exists(merged_obj_path):
                self.data_list.append(id_folder)
        
        print(f"找到 {len(self.data_list)} 个有效ID文件夹，每个包含yy_merged.obj文件")
    
    def get_model(self, model_id):
        """重写get_model方法，直接使用ID而不是文件名"""
        # 构建yy_merged.obj文件路径
        obj_path = os.path.join(self.data_path, model_id, "yy_merged.obj")
        
        # 加载网格
        from partfield.utils import load_mesh_util
        mesh = load_mesh_util(obj_path)
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 标准化网格 - 已注释掉以保持模型原始大小
        # bbmin = vertices.min(0)
        # bbmax = vertices.max(0)
        # center = (bbmin + bbmax) * 0.5
        # scale = 2.0 * 0.9 / (bbmax - bbmin).max()
        # vertices = (vertices - center) * scale
        # mesh.vertices = vertices
        
        # 确保是三角形网格
        from partfield.dataloader import quad_to_triangle_mesh
        mesh.faces = quad_to_triangle_mesh(faces)
        
        # 从网格表面采样点云
        pc, _ = trimesh.sample.sample_surface(mesh, self.pc_num_pts) 
        
        # 准备返回结果
        result = {'uid': model_id}
        result['pc'] = torch.tensor(pc, dtype=torch.float32)
        result['vertices'] = mesh.vertices
        result['faces'] = mesh.faces
        
        return result
    
    def __getitem__(self, index):
        import gc
        gc.collect()
        return self.get_model(self.data_list[index])

def predict(cfg):
    """
    执行模型推理过程，提取PartField特征
    
    参数:
        cfg: 配置对象，包含所有运行参数
    """
    # 设置随机种子，确保结果可复现
    seed_everything(cfg.seed)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    # 创建模型检查点回调，用于保存模型
    checkpoint_callbacks = [ModelCheckpoint(
        monitor="train/current_epoch",
        dirpath=cfg.output_dir,
        filename="{epoch:02d}",
        save_top_k=100,
        save_last=True,
        every_n_epochs=cfg.save_every_epoch,
        mode="max",
        verbose=True
    )]

    # 初始化PyTorch Lightning训练器
    trainer = Trainer(devices=-1,  # 使用所有可用GPU
                      accelerator="gpu",  # 使用GPU加速
                      precision="16-mixed",  # 使用混合精度训练
                      strategy=DDPStrategy(find_unused_parameters=True),  # 使用分布式数据并行
                      max_epochs=cfg.training_epochs,  # 最大训练周期
                      log_every_n_steps=1,  # 每步都记录日志
                      callbacks=checkpoint_callbacks  # 使用检查点回调
                     )

    # 导入模型定义并实例化
    from partfield.model_trainer_pvcnn_only_demo import Model
    
    # 创建一个自定义模型类，重写predict_step方法
    class CustomOutputModel(Model):
        def predict_dataloader(self):
            # 使用自定义数据集
            dataset = CustomWholeObjDataset(self.cfg)
            
            # 配置数据加载器
            dataloader = DataLoader(dataset,
                                   num_workers=self.cfg.dataset.val_num_workers if hasattr(self.cfg.dataset, 'val_num_workers') else 4,
                                   batch_size=1,  # 每次处理一个模型
                                   shuffle=False,
                                   pin_memory=True,
                                   drop_last=False)
            
            return dataloader
        
        @torch.no_grad()
        def predict_step(self, batch, batch_idx):
            """重写predict_step方法，修改输出路径和文件命名"""
            # 获取模型ID
            model_id = batch['uid'][0]
            
            # 创建输出目录
            output_dir = os.path.join("/hy-tmp/PartField_Sketch_simpleMLP/data/urdf", model_id, "feature")
            os.makedirs(output_dir, exist_ok=True)
            
            # 检查是否已经处理过
            if os.path.exists(os.path.join(output_dir, f"{model_id}.npy")):
                print(f"已处理过 {model_id}，跳过")
                return
            
            # 记录开始时间
            start_time = time.time()
            

            # 调用原始predict_step的核心逻辑
            # 使用PVCNN提取点云特征
            pc_feat = self.pvcnn(batch['pc'], batch['pc'])
            
            # 使用Transformer处理特征
            planes = pc_feat
            planes = self.triplane_transformer(planes)
            
            # 分离SDF特征和部分特征
            sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)
            
            # 处理网格数据
            from partfield.model.PVCNN.encoder_pc import sample_triplane_feat
            
            # 在三角形面上采样点云
            def sample_points(vertices, faces, n_point_per_face):
                # 分批处理面，每批处理1000个面
                batch_size = 1000
                n_f = faces.shape[0]
                all_points = []
                
                for i in range(0, n_f, batch_size):
                    # 当前批次的面数
                    batch_faces = faces[i:min(i+batch_size, n_f)]
                    batch_n_f = batch_faces.shape[0]
                    
                    # 生成随机重心坐标
                    u = torch.sqrt(torch.rand((batch_n_f, n_point_per_face, 1),
                                           device=vertices.device,
                                           dtype=vertices.dtype))
                    v = torch.rand((batch_n_f, n_point_per_face, 1),
                                device=vertices.device,
                                dtype=vertices.dtype)
                    w0 = 1 - u
                    w1 = u * (1 - v)
                    w2 = u * v

                    # 获取三角形的三个顶点
                    face_v_0 = torch.index_select(vertices, 0, batch_faces[:, 0].reshape(-1))
                    face_v_1 = torch.index_select(vertices, 0, batch_faces[:, 1].reshape(-1))
                    face_v_2 = torch.index_select(vertices, 0, batch_faces[:, 2].reshape(-1))
                    
                    # 使用重心坐标生成点
                    batch_points = w0 * face_v_0.unsqueeze(dim=1) + w1 * face_v_1.unsqueeze(dim=1) + w2 * face_v_2.unsqueeze(dim=1)
                    all_points.append(batch_points)
                    
                    # 主动清理内存
                    del u, v, w0, w1, w2, face_v_0, face_v_1, face_v_2, batch_points
                    torch.cuda.empty_cache()
                
                # 合并所有批次的结果
                return torch.cat(all_points, dim=0)

            def sample_and_mean_memory_save_version(part_planes, tensor_vertices, n_point_per_face):
                # 每批处理的点数
                n_sample_each = self.cfg.n_sample_each if hasattr(self.cfg, 'n_sample_each') else 10000
                n_v = tensor_vertices.shape[1]
                n_sample = n_v // n_sample_each + 1
                all_sample = []
                
                # 分批处理所有顶点
                for i_sample in range(n_sample):
                    # 提取当前批次中点的特征
                    end_idx = min((i_sample + 1) * n_sample_each, n_v)
                    sampled_feature = sample_triplane_feat(part_planes, tensor_vertices[:, i_sample * n_sample_each: end_idx])
                    if sampled_feature.shape[1] % n_point_per_face == 0:
                        # 重塑以计算每个面的平均特征
                        sampled_feature = sampled_feature.reshape(1, -1, n_point_per_face, sampled_feature.shape[-1])
                        sampled_feature = torch.mean(sampled_feature, axis=-2)
                        all_sample.append(sampled_feature)
                
                # 合并所有批次的结果
                return torch.cat(all_sample, dim=1)
            
            # 处理面特征
            n_point_per_face = self.cfg.n_point_per_face if hasattr(self.cfg, 'n_point_per_face') else 10
            
            # 在每个面上采样点
            tensor_vertices = sample_points(batch['vertices'][0], batch['faces'][0], n_point_per_face)
            tensor_vertices = tensor_vertices.reshape(1, -1, 3).to(torch.float32)
            
            # 提取面上采样点的特征并计算平均值
            point_feat = sample_and_mean_memory_save_version(part_planes, tensor_vertices, n_point_per_face)
            
            # 转换为numpy数组
            point_feat = point_feat.reshape(-1, 448).cpu().numpy()
            
            # 保存特征
            np.save(os.path.join(output_dir, f"{model_id}.npy"), point_feat)
            print(f"保存特征到 {output_dir}/{model_id}.npy")
            
            # 使用PCA进行特征可视化
            from sklearn.decomposition import PCA
            data_scaled = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)
            pca = PCA(n_components=3)
            
            # 降维到3D空间用于颜色映射
            data_reduced = pca.fit_transform(data_scaled)
            data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())
            colors_255 = (data_reduced * 255).astype(np.uint8)
            
            # 获取网格顶点和面
            V = batch['vertices'][0].cpu().numpy()
            F = batch['faces'][0].cpu().numpy()
            
            # 创建着色网格
            colored_mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=colors_255, process=False)
            
            # 导出彩色网格
            colored_mesh.export(os.path.join(output_dir, f"{model_id}.ply"))
            print(f"保存可视化到 {output_dir}/{model_id}.ply")
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            
            # 输出总耗时
            print(f"处理 {model_id} 完成，耗时: {time.time() - start_time:.2f}秒")
            
            return

    # 实例化自定义模型
    model = CustomOutputModel(cfg)        

    # 执行模型推理，加载指定的检查点
    trainer.predict(model, ckpt_path=cfg.continue_ckpt)
        
def main():
    """
    主函数，解析命令行参数并调用推理函数
    """
    # 解析命令行参数
    parser = default_argument_parser()
    args = parser.parse_args()
    
    # 设置配置，不冻结以允许后续修改
    cfg = setup(args, freeze=False)
    
    # 执行推理
    predict(cfg)
    
if __name__ == '__main__':
    main()