import os
import re
import cv2
import torch
import numpy as np
import json
import random
import shutil
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


def default_transforms(train=True):
    """
    默认的图像变换函数，训练时可加入数据增强。
    Args:
        train (bool): 是否为训练集，用于决定是否应用数据增强。
    Returns:
        transforms.Compose: 组合的变换函数。
    """
    if train:
        return T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            # 可根据需要添加更多数据增强操作
        ])
    else:
        return T.Compose([
            T.ToTensor(),
        ])


def parse_filename(filename):
    """
    解析文件名为各个组件
    支持格式:
    {类别}_{id}_{渲染类型}[_{视角}][_joint_{joint_id}].png
    
    Args:
        filename: 文件名
    Returns:
        字典包含类别、ID、渲染类型、视角、关节ID(如果有)
    """
    # 先提取基本部分：类别、ID、渲染类型
    base_pattern = r'^([a-zA-Z]+)_(\d+)_([a-zA-Z\-]+)'
    base_match = re.match(base_pattern, filename)
    
    if not base_match:
        return None
    
    category = base_match.group(1)
    obj_id = base_match.group(2)
    render_type = base_match.group(3)
    
    # 获取剩余部分（可能包含视角和关节ID）
    remaining = filename[len(base_match.group(0)):]
    
    # 检查是否包含joint信息
    joint_pattern = r'_joint_(\d+)'
    joint_match = re.search(joint_pattern, remaining)
    joint_id = joint_match.group(1) if joint_match else None
    
    # 提取视角信息（如果存在）
    # 视角通常在渲染类型之后，关节ID之前
    view = None
    if joint_match:
        view_part = remaining[:joint_match.start()]
        if view_part.startswith('_') and len(view_part) > 1:
            view = view_part[1:]  # 去掉开头的下划线
    elif remaining.startswith('_') and len(remaining) > 1:
        # 没有关节ID但有视角
        view = remaining[1:].split('.')[0]  # 去掉扩展名
    
    # 构建结果
    result = {
        'category': category,
        'id': obj_id,
        'render_type': render_type,
        'joint': joint_id,
        'view': view
    }
    
    return result


class SimplifiedArticulatedDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, verbose=True):
        """
        简化版的数据集类，只处理同时具有segmentation和arrow-sketch的图像，
        并确保它们的类别、ID、关节ID和视角完全匹配
        
        Args:
            root_dir (str): 数据集根目录。
            split (str): 数据集类型，'train', 'val', 'test'。
            transform (callable, optional): 可选的变换函数。
            verbose (bool): 是否打印调试信息。
        """
        super().__init__()
        self.root_dir = os.path.join(root_dir, split) if os.path.exists(os.path.join(root_dir, split)) else root_dir
        self.split = split
        self.transform = transform if transform else default_transforms(split=='train')
        self.verbose = verbose

        # 扫描目录找到所有图片
        self.all_images = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    self.all_images.append(os.path.join(root, file))
        
        if self.verbose:
            print(f"[DEBUG] Found {len(self.all_images)} PNG files in {self.root_dir}")

        # 按照类别_ID分组
        self.grouped_images = {}
        for img_path in self.all_images:
            filename = os.path.basename(img_path)
            parsed = parse_filename(filename)
            if not parsed:
                continue
                
            # 创建图像组键: category_id
            group_key = f"{parsed['category']}_{parsed['id']}"
            
            if group_key not in self.grouped_images:
                self.grouped_images[group_key] = {}
            
            # 只处理segmentation和arrow-sketch图像
            if parsed['render_type'] == 'segmentation' or parsed['render_type'] == 'arrow-sketch':
                if parsed['joint'] is not None:
                    # 创建包含渲染类型、关节ID和视角的完整键
                    view_part = f"_{parsed['view']}" if parsed['view'] else ""
                    render_joint_view_key = f"{parsed['render_type']}_joint_{parsed['joint']}{view_part}"
                    self.grouped_images[group_key][render_joint_view_key] = {
                        'path': img_path,
                        'parsed': parsed
                    }
        
        # 过滤，只保留同时具有segmentation和arrow-sketch的完整组，且确保视角匹配
        self.complete_groups = []
        for group_key, files in self.grouped_images.items():
            # 找出所有segmentation类型
            seg_entries = {k: v for k, v in files.items() if k.startswith('segmentation_joint_')}
            
            # 为每个segmentation找到对应视角的arrow-sketch
            for seg_key, seg_info in seg_entries.items():
                seg_parsed = seg_info['parsed']
                joint_id = seg_parsed['joint']
                view = seg_parsed['view']
                
                # 创建对应的arrow-sketch键
                view_part = f"_{view}" if view else ""
                arrow_key = f"arrow-sketch_joint_{joint_id}{view_part}"
                
                if arrow_key in files:  # 只有同时具有两种类型且视角匹配的才加入
                    # 分解group_key以获取类别和ID
                    parts = group_key.split('_')
                    category = parts[0]
                    obj_id = parts[1]
                    
                    sample = {
                        'category': category,
                        'id': obj_id,
                        'joint': joint_id,
                        'view': view,
                        'arrow-sketch': files[arrow_key]['path'],
                        'segmentation': files[seg_key]['path']
                    }
                    
                    self.complete_groups.append(sample)
                elif self.verbose:
                    print(f"[SKIP] Group {group_key} joint {joint_id} view '{view}': missing matching arrow-sketch")
        
        if self.verbose:
            print(f"[INFO] Created {len(self.complete_groups)} complete samples from {len(self.grouped_images)} groups")
        
        if len(self.complete_groups) == 0:
            raise RuntimeError(f"[ERROR] No complete samples found in {self.root_dir}!")

    def __len__(self):
        return len(self.complete_groups)

    def __getitem__(self, idx):
        sample = self.complete_groups[idx]
        
        # 读取各种图像
        arrow_sketch_path = sample['arrow-sketch']
        segmentation_path = sample['segmentation']
        
        # 读取图像
        segment_img = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)
        arrow_img = cv2.imread(arrow_sketch_path, cv2.IMREAD_COLOR)
        
        # 检查图像是否成功读取
        if segment_img is None:
            raise RuntimeError(f"[ERROR] Failed to read segmentation image: {segmentation_path}")
        if arrow_img is None:
            raise RuntimeError(f"[ERROR] Failed to read arrow-sketch image: {arrow_sketch_path}")
        
        # 处理segmentation图像 - 确保是单通道并二值化
        if segment_img.ndim == 3:
            segment_img = cv2.cvtColor(segment_img, cv2.COLOR_BGR2GRAY)
        segment_img = (segment_img > 127).astype(np.float32)
        
        # 归一化并转换BGR到RGB (对arrow-sketch)
        arrow_img = arrow_img.astype(np.float32) / 255.0
        # BGR->RGB
        arrow_img = arrow_img[..., ::-1]
        
        # 创建单通道分割图像
        H, W = segment_img.shape
        seg_1ch = segment_img.reshape(H, W, 1)
        
        # 输入是arrow-sketch（3通道）
        inp_ch = arrow_img
        num_channels = 3
        
        if self.transform:
            # 合并输入和标签以确保相同的变换应用于两者
            combined = np.concatenate([inp_ch, seg_1ch], axis=-1)
            transformed = self.transform(combined)
            inp_tensor = transformed[:num_channels, :, :]
            lab_tensor = transformed[num_channels:, :, :]
        else:
            inp_tensor = torch.from_numpy(inp_ch.transpose(2, 0, 1))
            lab_tensor = torch.from_numpy(seg_1ch.transpose(2, 0, 1))
        
        # 元数据
        meta = {
            'category': sample['category'],
            'id': sample['id'],
            'joint': sample['joint'],
            'view': sample['view'],
            'has_arrow': True  # 所有样本现在都有arrow-sketch
        }
        
        return inp_tensor, lab_tensor, meta


def simplified_collate_fn(batch):
    """
    简化版的 collate_fn，适用于只处理同时具有segmentation和arrow-sketch的数据集
    """
    inp_list = []
    lab_list = []
    meta_list = []
    
    # 所有样本现在都有相同的格式
    for inp, lab, meta in batch:
        inp_list.append(inp)
        lab_list.append(lab)
        meta_list.append(meta)
    
    # 创建结果张量
    inp_tensor = torch.stack(inp_list, dim=0)
    lab_tensor = torch.stack(lab_list, dim=0)
    
    return inp_tensor, lab_tensor, meta_list


def split_dataset(data_dir, output_dir=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    随机划分数据集并记录数据的分配
    
    Args:
        data_dir: 数据根目录，包含所有图片或子目录
        output_dir: 输出目录，将创建train/val/test子目录
        train_ratio, val_ratio, test_ratio: 各集比例
        seed: 随机种子
    
    Returns:
        分割记录字典，包含每个(类别, ID)对所在的分割
    """
    random.seed(seed)
    
    # 如果没有指定输出目录，则使用输入目录
    if output_dir is None:
        output_dir = data_dir
    
    # 创建分割目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 收集所有图片并按类别-ID分组
    all_images = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.png'):
                all_images.append(os.path.join(root, file))
    
    # 解析文件名并按类别-ID分组
    category_id_dict = {}
    for img_path in all_images:
        filename = os.path.basename(img_path)
        parsed = parse_filename(filename)
        if parsed:
            # 只处理segmentation和arrow-sketch图像
            if parsed['render_type'] == 'segmentation' or parsed['render_type'] == 'arrow-sketch':
                key = (parsed['category'], parsed['id'])
                if key not in category_id_dict:
                    category_id_dict[key] = []
                category_id_dict[key].append(img_path)
    
    # 获取所有唯一的(类别, ID)对
    all_category_ids = list(category_id_dict.keys())
    random.shuffle(all_category_ids)
    
    # 计算各集合的大小
    total = len(all_category_ids)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # 划分数据集
    train_ids = all_category_ids[:train_size]
    val_ids = all_category_ids[train_size:train_size+val_size]
    test_ids = all_category_ids[train_size+val_size:]
    
    # 创建划分映射
    split_mapping = {}
    for key in train_ids:
        split_mapping[key] = 'train'
    for key in val_ids:
        split_mapping[key] = 'val'
    for key in test_ids:
        split_mapping[key] = 'test'
    
    # 记录分割信息到JSON文件
    split_info = {
        'train': [{'category': c, 'id': i} for c, i in train_ids],
        'val': [{'category': c, 'id': i} for c, i in val_ids],
        'test': [{'category': c, 'id': i} for c, i in test_ids]
    }
    
    with open(os.path.join(output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"数据集划分完成：训练集 {len(train_ids)}，验证集 {len(val_ids)}，测试集 {len(test_ids)}")
    print(f"分割信息已保存到 {os.path.join(output_dir, 'split_info.json')}")
    
    # 如果需要，可以创建物理划分的文件结构
    physical_split = False  # 设为True可以将文件复制到对应的train/val/test目录
    
    if physical_split:
        # 复制文件到各自的目录
        for (category, id_str), split in split_mapping.items():
            target_dir = os.path.join(output_dir, split, category, id_str)
            os.makedirs(target_dir, exist_ok=True)
            
            # 复制该类别-ID的所有图片
            for img_path in category_id_dict[(category, id_str)]:
                filename = os.path.basename(img_path)
                shutil.copy2(img_path, os.path.join(target_dir, filename))
        
        print("已将文件复制到相应的训练/验证/测试目录")
    
    return split_mapping


def visualize_pairs(dataset, num_samples=10, save_dir="./data_pairs"):
    """
    可视化数据集中的arrow-sketch和segmentation配对
    
    Args:
        dataset: SimplifiedArticulatedDataset实例
        num_samples: 要可视化的样本数量
        save_dir: 保存可视化结果的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 确保不超出数据集大小
    num_samples = min(num_samples, len(dataset))
    
    print(f"\n[VISUALIZE] 正在生成{num_samples}个数据对的可视化...")
    
    for i in range(num_samples):
        # 获取样本
        inp, lab, meta = dataset[i]
        
        # 转换为numpy数组用于可视化
        inp_np = inp.numpy().transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
        lab_np = lab.numpy()[0]  # [1,H,W] -> [H,W]
        
        # 确保值在0-1范围内
        inp_np = np.clip(inp_np, 0, 1)
        
        # 创建2x2的子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 显示arrow-sketch输入
        axes[0].imshow(inp_np)
        axes[0].set_title("Arrow-Sketch Input")
        axes[0].axis('off')
        
        # 显示segmentation标签
        axes[1].imshow(lab_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Segmentation GT")
        axes[1].axis('off')
        
        # 叠加显示
        overlay = inp_np.copy()
        # 创建红色掩码
        mask = np.zeros_like(overlay)
        mask[:, :, 0] = lab_np  # 红色通道
        
        # 应用半透明掩码
        overlay = overlay * 0.7 + mask * 0.3
        overlay = np.clip(overlay, 0, 1)
        
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        # 添加图像元信息作为标题
        category = meta['category']
        obj_id = meta['id']
        joint_id = meta['joint']
        view = meta['view'] if meta['view'] else "no_view"
        
        fig.suptitle(f"Category: {category}, ID: {obj_id}, Joint: {joint_id}, View: {view}", fontsize=14)
        
        # 保存图像
        save_path = os.path.join(save_dir, f"pair_{i:02d}_{category}_joint{joint_id}_{view}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        print(f"  已保存: {save_path}")
    
    print(f"[VISUALIZE] 完成! 可视化结果已保存到 {save_dir}/")
    return save_dir


def test_simplified_loader():
    """测试简化版的数据集加载器"""
    # 指定数据目录
    data_dir = "data/img"
    
    # 测试数据解析
    print("\n[TEST] Testing filename parsing...")
    test_filenames = [
        "Microwave_7366_default_top_left.png",  # 将被忽略
        "Microwave_7366_segmentation_front_center_joint_1.png",  # 带视角的segmentation
        "Microwave_7366_segmentation_top_view_joint_1.png",  # 不同视角的segmentation
        "Microwave_7366_arrow-sketch_front_center_joint_1.png",  # 匹配第一个segmentation的arrow-sketch
        "Safe_102316_depth_front_high.png",  # 将被忽略
        "Table_19898_arrow-sketch_joint_4.png",  # 无视角的arrow-sketch
        "InvalidName.png"
    ]
    
    for filename in test_filenames:
        parsed = parse_filename(filename)
        if parsed:
            print(f"File: {filename}")
            print(f"  Parsed: {parsed}")
            # 检查是否是我们需要的图像类型
            if parsed['render_type'] in ['segmentation', 'arrow-sketch']:
                print(f"  This is a required image type: {parsed['render_type']}")
            else:
                print(f"  This is NOT a required image type: {parsed['render_type']}")
        else:
            print(f"File: {filename} -> Invalid format")
    
    # 测试数据集加载
    print("\n[TEST] Testing SimplifiedArticulatedDataset...")
    try:
        dataset = SimplifiedArticulatedDataset(root_dir=data_dir, split='train', transform=None, verbose=True)
        print(f"Successfully created dataset with {len(dataset)} samples")
        
        # 测试获取几个样本
        if len(dataset) > 0:
            print("\n[TEST] Testing sample access...")
            for i in range(min(3, len(dataset))):
                inp, lab, meta = dataset[i]
                print(f"\nSample {i}:")
                print(f"  Input shape: {inp.shape}")
                print(f"  Label shape: {lab.shape}")
                print(f"  Metadata: {meta}")
        
        # 测试DataLoader
        print("\n[TEST] Testing DataLoader...")
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=simplified_collate_fn
        )
        
        for i, (inp_batch, lab_batch, meta_batch) in enumerate(loader):
            print(f"\nBatch {i}:")
            print(f"  Input shape: {inp_batch.shape}")
            print(f"  Label shape: {lab_batch.shape}")
            print(f"  Metadata: {meta_batch}")
            
            # 只测试前2个批次
            if i >= 1:
                break
        
        # 可视化10个配对样本
        if len(dataset) > 0:
            vis_dir = visualize_pairs(dataset, num_samples=10, save_dir="./data_pairs")
            print(f"\n[INFO] 10个数据对的可视化结果已保存在: {vis_dir}")
            
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
    
    # 测试数据集划分
    print("\n[TEST] Testing dataset splitting...")
    try:
        # 在临时目录测试，避免修改原始数据
        test_output_dir = os.path.join(data_dir, "test_split")
        os.makedirs(test_output_dir, exist_ok=True)
        
        split_mapping = split_dataset(
            data_dir=data_dir,
            output_dir=test_output_dir,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        # 打印部分划分信息
        print("\nSplit mapping sample:")
        count = 0
        for key, split in split_mapping.items():
            if count < 5:
                print(f"  {key} -> {split}")
                count += 1
            else:
                break
        print("  ...")
        
    except Exception as e:
        print(f"[ERROR] An error occurred during splitting: {e}")


# 如果直接运行此脚本，则执行测试
if __name__ == "__main__":
    test_simplified_loader()
