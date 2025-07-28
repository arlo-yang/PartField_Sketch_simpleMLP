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
    恢复标准化以保证模型预测性能
    """
    def __init__(self, cfg):
        # 不直接调用父类的__init__，因为我们需要自定义数据加载逻辑
        torch.utils.data.Dataset.__init__(self)
        
        # 确保使用正确的数据路径
        self.data_path = "/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/urdf"  # 直接硬编码URDF路径
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
        """重写get_model方法，直接使用ID而不是文件名，并恢复标准化"""
        # 构建yy_merged.obj文件路径
        obj_path = os.path.join(self.data_path, model_id, "yy_merged.obj")
        
        # 加载网格
        from partfield.utils import load_mesh_util
        mesh = load_mesh_util(obj_path)
        vertices = mesh.vertices
        faces = mesh.faces
        
        # ============ 恢复标准化处理 ============
        # 保存原始的几何信息，用于后续逆变换
        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * 0.9 / (bbmax - bbmin).max()
        
        # 应用标准化变换
        vertices = (vertices - center) * scale
        mesh.vertices = vertices
        
        # 保存变换参数到mesh对象，供后续使用
        mesh.original_center = center
        mesh.original_scale = scale
        mesh.original_bbmin = bbmin
        mesh.original_bbmax = bbmax
        # ========================================
        
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
        
        # 传递变换参数 - 使用张量格式避免分布式问题
        result['transform_center'] = torch.tensor(center, dtype=torch.float32)
        result['transform_scale'] = torch.tensor(scale, dtype=torch.float32)
        result['transform_bbmin'] = torch.tensor(bbmin, dtype=torch.float32)
        result['transform_bbmax'] = torch.tensor(bbmax, dtype=torch.float32)
        
        return result
    
    def __getitem__(self, index):
        import gc
        gc.collect()
        return self.get_model(self.data_list[index])

def compare_mesh_faces(original_mesh, generated_mesh, output_path):
    """
    比较两个网格的面片顺序并保存结果
    
    参数:
        original_mesh: 原始网格对象
        generated_mesh: 生成的网格对象
        output_path: 输出结果的路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 比较面片数量
    orig_faces = len(original_mesh.faces)
    gen_faces = len(generated_mesh.faces)
    
    with open(output_path, 'w') as f:
        f.write(f"面片数量比较:\n")
        f.write(f"- 原始网格: {orig_faces} 面片\n")
        f.write(f"- 生成网格: {gen_faces} 面片\n\n")
        
        if orig_faces != gen_faces:
            f.write(f"警告: 面片数量不一致!\n\n")
        
        # 比较前10个面片的顶点索引
        f.write(f"前10个面片的顶点索引比较:\n")
        f.write("面片ID | 原始网格 | 生成网格\n")
        f.write("-------|----------|----------\n")
        
        for i in range(min(10, orig_faces, gen_faces)):
            orig_face = original_mesh.faces[i].tolist()
            gen_face = generated_mesh.faces[i].tolist() if i < gen_faces else ["N/A"]
            f.write(f"{i} | {orig_face} | {gen_face}\n")
        
        # 检查几何一致性
        f.write("\n几何一致性检查 (前10个面片):\n")
        f.write("面片ID | 原始网格面中心 | 生成网格面中心 | 距离\n")
        f.write("-------|----------------|----------------|------\n")
        
        for i in range(min(10, orig_faces, gen_faces)):
            if i < orig_faces and i < gen_faces:
                # 计算面片中心点
                orig_center = np.mean([original_mesh.vertices[v] for v in original_mesh.faces[i]], axis=0)
                gen_center = np.mean([generated_mesh.vertices[v] for v in generated_mesh.faces[i]], axis=0)
                
                # 计算中心点距离
                distance = np.linalg.norm(orig_center - gen_center)
                
                f.write(f"{i} | {orig_center.tolist()} | {gen_center.tolist()} | {distance:.6f}\n")
    
    print(f"面片比较结果已保存到: {output_path}")

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
                                   batch_size=1,
                                   shuffle=False,
                                   pin_memory=True,
                                   drop_last=False)
            
            return dataloader
        
        @torch.no_grad()
        def predict_step(self, batch, batch_idx):
            """重写predict_step方法，修改输出路径和文件命名，并添加逆变换功能"""
            # 获取模型ID
            model_id = batch['uid'][0]
            
            # 获取变换参数 - 从张量格式重建
            transform_info = None
            if 'transform_center' in batch:
                transform_info = {
                    'center': batch['transform_center'][0].cpu().numpy(),
                    'scale': batch['transform_scale'][0].cpu().numpy().item(),
                    'bbmin': batch['transform_bbmin'][0].cpu().numpy(),
                    'bbmax': batch['transform_bbmax'][0].cpu().numpy()
                }
            
            # 创建输出目录
            output_dir = os.path.join("/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/urdf", model_id, "feature", "mesh")
            os.makedirs(output_dir, exist_ok=True)
            
            # 检查是否已经处理过
            if os.path.exists(os.path.join(output_dir, f"{model_id}.npy")):
                print(f"已处理过 {model_id}，跳过")
                return
            
            # 记录开始时间
            start_time = time.time()
            
            print(f"处理模型 {model_id}")
            if transform_info:
                print(f"变换参数 - 中心: {transform_info['center']}, 缩放: {transform_info['scale']}")

            # 调用原始predict_step的核心逻辑
            # 使用PVCNN提取点云特征
            pc_feat = self.pvcnn(batch['pc'], batch['pc'])
            
            # 使用Transformer处理特征
            planes = pc_feat
            planes = self.triplane_transformer(planes)
            
            # 分离SDF特征和部分特征
            sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)
            
            # ============ 保存2D feature maps ============
            # 创建2D map保存目录
            map_2d_dir = os.path.join("/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/urdf", model_id, "feature", "2D_map")
            os.makedirs(map_2d_dir, exist_ok=True)
            
            # 检查是否已经保存过feature map
            feature_map_path = os.path.join(map_2d_dir, f"{model_id}.npy")
            if not os.path.exists(feature_map_path):
                # 转换为numpy并保存
                part_planes_np = part_planes.cpu().numpy()  # [1, 3, 448, 128, 128]
                np.save(feature_map_path, part_planes_np)
                print(f"保存2D feature maps到: {feature_map_path}")
                print(f"Feature maps维度: {part_planes_np.shape}")
                
                # 保存feature map说明文件
                info_path = os.path.join(map_2d_dir, f"{model_id}_info.txt")
                with open(info_path, 'w') as f:
                    f.write(f"模型ID: {model_id}\n")
                    f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Feature maps维度: {part_planes_np.shape}\n\n")
                    f.write(f"数据格式:\n")
                    f.write(f"- 形状: [batch, planes, features, height, width]\n")
                    f.write(f"- [1, 3, 448, 128, 128]\n")
                    f.write(f"  - batch=1: 单个模型\n")
                    f.write(f"  - planes=3: XY、YZ、XZ三个平面\n")
                    f.write(f"  - features=448: 每个像素的特征维度\n")
                    f.write(f"  - 128x128: 平面分辨率\n\n")
                    f.write(f"使用方法:\n")
                    f.write(f"feature_maps = np.load('{model_id}.npy')\n")
                    f.write(f"xy_plane = feature_maps[0, 0]  # XY平面 [448, 128, 128]\n")
                    f.write(f"yz_plane = feature_maps[0, 1]  # YZ平面 [448, 128, 128]\n")
                    f.write(f"xz_plane = feature_maps[0, 2]  # XZ平面 [448, 128, 128]\n\n")
                    f.write(f"注意: 所有坐标已标准化到[-1, 1]范围\n")
            else:
                print(f"2D feature maps已存在，跳过保存")
            # =============================================
            
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
            
            def inverse_transform_vertices(vertices, transform_info):
                """将标准化的顶点坐标逆变换回原始坐标系"""
                if transform_info is None:
                    return vertices
                
                # 逆变换: 先除以缩放因子，再加上中心偏移
                center = np.array(transform_info['center'])
                scale = transform_info['scale']
                
                # 逆变换公式: original = normalized / scale + center
                original_vertices = vertices / scale + center
                return original_vertices
            
            # 保存原始网格信息，用于后续比较
            original_mesh_path = os.path.join("/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/data_small/urdf", model_id, "yy_merged.obj")
            from partfield.utils import load_mesh_util
            original_mesh = load_mesh_util(original_mesh_path)
            # 确保是三角形网格
            from partfield.dataloader import quad_to_triangle_mesh
            original_mesh.faces = quad_to_triangle_mesh(original_mesh.faces)
            
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
            
            # ============ 使用原始坐标系的顶点 ============
            # 获取标准化后的网格顶点和面（用于特征提取）
            V_normalized = batch['vertices'][0].cpu().numpy()
            F = batch['faces'][0].cpu().numpy()
            
            # 将顶点逆变换回原始坐标系（用于可视化和保存）
            V_original = inverse_transform_vertices(V_normalized, transform_info)
            
            # 创建着色网格 - 使用原始坐标系的顶点
            colored_mesh = trimesh.Trimesh(vertices=V_original, faces=F, face_colors=colors_255, process=False)
            
            # 导出彩色网格
            ply_path = os.path.join(output_dir, f"{model_id}.ply")
            colored_mesh.export(ply_path)
            print(f"保存可视化到 {ply_path}")
            
            # 保存变换参数信息
            if transform_info:
                transform_path = os.path.join(output_dir, f"{model_id}_transform_info.json")
                import json
                with open(transform_path, 'w') as f:
                    # 将numpy数组转换为列表以便JSON序列化
                    transform_data = {
                        'center': transform_info['center'].tolist(),
                        'scale': float(transform_info['scale']),
                        'bbmin': transform_info['bbmin'].tolist(),
                        'bbmax': transform_info['bbmax'].tolist()
                    }
                    json.dump(transform_data, f, indent=2)
                print(f"保存变换参数到 {transform_path}")
            # =======================================================
            
            # 比较原始网格和生成的PLY网格的面片顺序
            generated_mesh = trimesh.load(ply_path, force='mesh', process=False)
            comparison_path = os.path.join(output_dir, f"{model_id}_face_comparison.txt")
            compare_mesh_faces(original_mesh, generated_mesh, comparison_path)
            
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