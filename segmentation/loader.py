import os
import re
import cv2
import torch
import numpy as np
import json
import random
import shutil

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


def default_transforms(train=True):
    if train:
        return T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
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
        字典包含类别、ID、渲染类型、关节ID(如果有)
    """
    # 先提取基本部分：类别、ID、渲染类型
    base_pattern = r'^([a-zA-Z]+)_(\d+)_([a-zA-Z\-]+)'
    base_match = re.match(base_pattern, filename)
    
    if not base_match:
    return None
    
    category = base_match.group(1)
    obj_id = base_match.group(2)
    render_type = base_match.group(3)
    
    # 检查是否包含joint信息
    joint_pattern = r'_joint_(\d+)'
    joint_match = re.search(joint_pattern, filename)
    joint_id = joint_match.group(1) if joint_match else None
    
    # 构建结果
    return {
        'category': category,
        'id': obj_id,
        'render_type': render_type,
        'joint': joint_id
    }


class NewArticulatedDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, verbose=True):
        """
        适用于新数据格式的数据集类
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

        # 按照类别_ID分组（不再包含视图）
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
                
            # 处理带joint信息的文件
            if parsed['joint'] is not None:
                # 创建特定joint的键，例如"arrow-sketch_joint_1"或"segmentation_joint_1"
                render_joint_key = f"{parsed['render_type']}_joint_{parsed['joint']}"
                self.grouped_images[group_key][render_joint_key] = img_path
            else:
                # 基本渲染类型(无joint信息)
                self.grouped_images[group_key][parsed['render_type']] = img_path
        
        # 过滤掉不完整的组（必须有default, normal, depth和至少一个segmentation）
        self.complete_groups = []
        for group_key, files in self.grouped_images.items():
            # 检查是否包含必要的基本渲染类型
            has_default = 'default' in files
            has_normal = 'normal' in files
            has_depth = 'depth' in files
            
            # 找出所有segmentation和arrow-sketch类型（包含joint信息）
            seg_types = [k for k in files.keys() if k.startswith('segmentation_joint_')]
            arrow_types = [k for k in files.keys() if k.startswith('arrow-sketch_joint_')]
            
            # 如果具有所有必要的渲染类型和至少一个分割，则添加到完整组
            if has_default and has_normal and has_depth and len(seg_types) > 0:
                    # 分解group_key以获取类别和ID
                    parts = group_key.split('_')
                    category = parts[0]
                    obj_id = parts[1]
                    
                # 为每个joint创建一个完整的样本
                for seg_type in seg_types:
                    # 提取joint编号
                        joint_id = seg_type.split('_')[-1]
                    
                    # 找到对应的arrow-sketch（如果存在）
                    arrow_key = f"arrow-sketch_joint_{joint_id}"
                    has_arrow = arrow_key in files
                    
                    sample = {
                        'category': category,
                        'id': obj_id,
                        'joint': joint_id,
                        'default': files.get('default'),
                        'normal': files.get('normal'),
                        'depth': files.get('depth'),
                        'arrow-sketch': files.get(arrow_key) if has_arrow else None,
                        'segmentation': files.get(seg_type)
                    }
                    
                    # 确保segmentation一定存在
                    if sample['segmentation'] is None:
                        if self.verbose:
                            print(f"[WARN] 跳过 {group_key} 的 joint {joint_id}: 缺少分割图")
                        continue
                        
                    # 如果缺少arrow-sketch，记录警告但仍然继续
                    if not has_arrow:
                        if self.verbose:
                            print(f"[WARN] Group {group_key} joint {joint_id} missing arrow-sketch, but will continue")
                    
                    self.complete_groups.append(sample)
            else:
                if self.verbose:
                    missing = []
                    if not has_default: missing.append('default')
                    if not has_normal: missing.append('normal')
                    if not has_depth: missing.append('depth')
                    print(f"[DEBUG] Incomplete group {group_key}: missing {missing}")
        
        if self.verbose:
            print(f"[INFO] Created {len(self.complete_groups)} complete samples from {len(self.grouped_images)} groups")
        
        if len(self.complete_groups) == 0:
            raise RuntimeError(f"[ERROR] No complete samples found in {self.root_dir}!")

    def __len__(self):
        return len(self.complete_groups)

    def __getitem__(self, idx):
        sample = self.complete_groups[idx]
        
        # 读取各种图像
        default_path = sample['default']
        normal_path = sample['normal']
        depth_path = sample['depth']
        arrow_sketch_path = sample['arrow-sketch']
        segmentation_path = sample['segmentation']
        
        # 读取图像
        default_img = cv2.imread(default_path, cv2.IMREAD_COLOR)
        normal_img = cv2.imread(normal_path, cv2.IMREAD_COLOR)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        segment_img = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)
        
        # 处理arrow-sketch图像（必须存在）
        if arrow_sketch_path and os.path.exists(arrow_sketch_path):
            arrow_img = cv2.imread(arrow_sketch_path, cv2.IMREAD_COLOR)
        else:
            raise RuntimeError(f"[ERROR] Arrow-sketch is mandatory but missing: {arrow_sketch_path}")
        
        # 检查图像是否成功读取
        if default_img is None:
            raise RuntimeError(f"[ERROR] Failed to read default image: {default_path}")
        if normal_img is None:
            raise RuntimeError(f"[ERROR] Failed to read normal image: {normal_path}")
        if depth_img is None:
            raise RuntimeError(f"[ERROR] Failed to read depth image: {depth_path}")
        if segment_img is None:
            raise RuntimeError(f"[ERROR] Failed to read segmentation image: {segmentation_path}")
        if arrow_img is None:
            raise RuntimeError(f"[ERROR] Failed to read arrow-sketch image: {arrow_sketch_path}")
        
        # 处理depth图像 - 确保是单通道并归一化
        if depth_img.ndim == 3:
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
        
        if depth_img.max() > 0:
            depth_img = depth_img.astype(np.float32) / depth_img.max()
        else:
            depth_img = depth_img.astype(np.float32)
        
        # 处理segmentation图像 - 确保是单通道并二值化
        if segment_img.ndim == 3:
            segment_img = cv2.cvtColor(segment_img, cv2.COLOR_BGR2GRAY)
        segment_img = (segment_img > 127).astype(np.float32)
        
        # 归一化并转换BGR到RGB
        default_img = default_img.astype(np.float32) / 255.0
        normal_img = normal_img.astype(np.float32) / 255.0
        arrow_img = arrow_img.astype(np.float32) / 255.0
        
        # BGR->RGB
        default_img = default_img[..., ::-1]
        normal_img = normal_img[..., ::-1]
        arrow_img = arrow_img[..., ::-1]
        
        # 创建单通道depth和分割图像
        H, W = depth_img.shape
        depth_1ch = depth_img.reshape(H, W, 1)
        seg_1ch = segment_img.reshape(H, W, 1)
        
        # 合并为10通道：3(default) + 3(normal) + 1(depth) + 3(arrow-sketch)
        inp_ch = np.concatenate([default_img, normal_img, depth_1ch, arrow_img], axis=-1)
        num_channels = 10
        
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
            'joint': sample['joint']
        }
        
        return inp_tensor, lab_tensor, meta


def new_collate_fn(batch):
    """
    自定义 collate_fn，适用于新的数据集格式
    处理可变长度的输入张量（因为有些样本可能包含arrow-sketch而有些不包含）
    """
    inp_lists = {
        'with_arrow': [],    # 包含arrow-sketch的样本
        'without_arrow': []  # 不包含arrow-sketch的样本
    }
    lab_lists = {
        'with_arrow': [],
        'without_arrow': []
    }
    meta_list = []
    
    # 根据是否有arrow-sketch分类
    for inp, lab, meta in batch:
        if meta['has_arrow']:
            inp_lists['with_arrow'].append(inp)
            lab_lists['with_arrow'].append(lab)
        else:
            inp_lists['without_arrow'].append(inp)
            lab_lists['without_arrow'].append(lab)
        meta_list.append(meta)
    
    # 创建结果张量
    result_tensors = []
    
    # 处理包含arrow-sketch的样本
    if inp_lists['with_arrow']:
        inp_with_arrow = torch.stack(inp_lists['with_arrow'], dim=0)
        lab_with_arrow = torch.stack(lab_lists['with_arrow'], dim=0)
        result_tensors.append((inp_with_arrow, lab_with_arrow))
    
    # 处理不包含arrow-sketch的样本
    if inp_lists['without_arrow']:
        inp_without_arrow = torch.stack(inp_lists['without_arrow'], dim=0)
        lab_without_arrow = torch.stack(lab_lists['without_arrow'], dim=0)
        result_tensors.append((inp_without_arrow, lab_without_arrow))
    
    # 如果所有样本都有相同数量的通道，我们可以简化返回值
    if not inp_lists['without_arrow'] or not inp_lists['with_arrow']:
        if result_tensors:
            return result_tensors[0][0], result_tensors[0][1], meta_list
    
    # 如果有不同通道数的样本，返回分开的张量和元数据
    return result_tensors, meta_list


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


def test_new_loader():
    """测试新的数据集加载器"""
    # 指定数据目录
    data_dir = "data/img"
    
    # 测试数据解析
    print("\n[TEST] Testing filename parsing...")
    test_filenames = [
        "Microwave_7366_default_top_left.png",
        "Microwave_7366_normal_direct_right.png",
        "Microwave_7366_segmentation_bottom_center_joint_1.png",
        "Safe_102316_depth_front_high.png",
        "Table_19898_arrow-sketch_joint_4.png",  # 添加一个arrow-sketch示例
        "InvalidName.png"
    ]
    
    for filename in test_filenames:
        parsed = parse_filename(filename)
        if parsed:
            print(f"File: {filename}")
            print(f"  Parsed: {parsed}")
        else:
            print(f"File: {filename} -> Invalid format")
    
    # 测试数据集加载
    print("\n[TEST] Testing NewArticulatedDataset...")
    try:
        dataset = NewArticulatedDataset(root_dir=data_dir, split='train', transform=None, verbose=True)
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
            collate_fn=new_collate_fn
        )
        
        for i, (inp_batch, lab_batch, meta_batch) in enumerate(loader):
            print(f"\nBatch {i}:")
            print(f"  Input shape: {inp_batch.shape}")
            print(f"  Label shape: {lab_batch.shape}")
            print(f"  Metadata: {meta_batch}")
            
            # 只测试前2个批次
            if i >= 1:
                break
        
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


if __name__ == "__main__":
    test_new_loader() 