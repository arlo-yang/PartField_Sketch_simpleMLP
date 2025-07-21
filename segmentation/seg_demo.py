import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import shutil  # 导入用于文件复制的模块
import trimesh

# 导入自定义模块
from network import FlexibleUNet
from loader import parse_filename


def get_project_root():
    """获取项目根目录"""
    # 直接返回workplace目录
    return "/home/ipab-graphics/workplace/User_web_dev"


def load_and_preprocess_images(img_dir):
    """
    加载并预处理指定目录下的图像文件
    使用新的文件命名格式：{类别}_{id}_{渲染类型}.png
    """
    # 确保使用绝对路径
    img_dir = os.path.abspath(img_dir)
    
    # 搜集所有PNG文件
    all_images = list(Path(img_dir).glob("*.png"))
    print(f"在 {img_dir} 中找到 {len(all_images)} 个PNG文件")
    
    # 按照类别_ID分组
    grouped_images = {}
    for img_path in all_images:
        filename = img_path.name
        parsed = parse_filename(filename)
        if not parsed:
            print(f"警告：无法解析文件名: {filename}")
            continue
            
        # 创建图像组键: category_id
        group_key = f"{parsed['category']}_{parsed['id']}"
        
        if group_key not in grouped_images:
            grouped_images[group_key] = {}
        
        grouped_images[group_key][parsed['render_type']] = str(img_path)
    
    # 处理每个组
    processed_groups = []
    for group_key, files in grouped_images.items():
        # 检查是否包含必要的渲染类型
        has_default = 'default' in files
        has_normal = 'normal' in files
        has_depth = 'depth' in files
        has_arrow_sketch = 'arrow-sketch' in files
        
        # 如果有所有必要的渲染类型，则处理
        if has_default and has_normal and has_depth and has_arrow_sketch:
            # 分解group_key以获取类别和ID
            parts = group_key.split('_')
            category = parts[0]
            obj_id = parts[1]
            
            sample = {
                'category': category,
                'id': obj_id,
                'default': files.get('default'),
                'normal': files.get('normal'),
                'depth': files.get('depth'),
                'arrow-sketch': files.get('arrow-sketch')
            }
            
            processed_groups.append(sample)
        else:
            missing = []
            if not has_default: missing.append('default')
            if not has_normal: missing.append('normal')
            if not has_depth: missing.append('depth')
            if not has_arrow_sketch: missing.append('arrow-sketch')
            print(f"组 {group_key} 不完整: 缺少 {missing}")
    
    return processed_groups


def process_sample(sample):
    """
    处理单个样本，将其转换为10通道输入
    """
    default_path = sample['default']
    normal_path = sample['normal']
    depth_path = sample['depth']
    arrow_sketch_path = sample['arrow-sketch']
    
    # 读取图像
    default_img = cv2.imread(default_path, cv2.IMREAD_COLOR)
    normal_img = cv2.imread(normal_path, cv2.IMREAD_COLOR)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    arrow_img = cv2.imread(arrow_sketch_path, cv2.IMREAD_COLOR)
    
    # 检查图像是否成功读取
    if default_img is None:
        raise RuntimeError(f"无法读取default图像: {default_path}")
    if normal_img is None:
        raise RuntimeError(f"无法读取normal图像: {normal_path}")
    if depth_img is None:
        raise RuntimeError(f"无法读取depth图像: {depth_path}")
    if arrow_img is None:
        raise RuntimeError(f"无法读取arrow-sketch图像: {arrow_sketch_path}")
    
    # 处理depth图像 - 确保是单通道并归一化
    if depth_img.ndim == 3:
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
    
    if depth_img.max() > 0:
        depth_img = depth_img.astype(np.float32) / depth_img.max()
    else:
        depth_img = depth_img.astype(np.float32)
    
    # 归一化并转换BGR到RGB
    default_img = default_img.astype(np.float32) / 255.0
    normal_img = normal_img.astype(np.float32) / 255.0
    arrow_img = arrow_img.astype(np.float32) / 255.0
    
    # BGR->RGB
    default_img = default_img[..., ::-1]
    normal_img = normal_img[..., ::-1]
    arrow_img = arrow_img[..., ::-1]
    
    # 创建单通道depth图像
    H, W = depth_img.shape
    depth_1ch = depth_img.reshape(H, W, 1)
    
    # 合并为10通道：3(default) + 3(normal) + 1(depth) + 3(arrow-sketch)
    inp_ch = np.concatenate([default_img, normal_img, depth_1ch, arrow_img], axis=-1)
    
    # 转换为PyTorch张量 [C, H, W]
    inp_tensor = torch.from_numpy(inp_ch.transpose(2, 0, 1))
    
    return inp_tensor, sample


def convert_obj_to_glb(obj_id, base_output_dir):
    """
    将用户上传的OBJ文件转换为GLB格式并保存到指定目录
    
    Args:
        obj_id: 用户模型ID
        base_output_dir: 输出目录
        
    Returns:
        bool: 是否成功转换
    """
    project_root = get_project_root()
    
    # 用户OBJ文件路径
    obj_path = os.path.join(project_root, 'Data', 'UserOBJ', f'{obj_id}.obj')
    
    # 目标GLB文件路径
    glb_path = os.path.join(base_output_dir, f"Uploaded_{obj_id}.glb")
    
    # 检查OBJ文件是否存在
    if not os.path.exists(obj_path):
        print(f"找不到用户OBJ文件: {obj_path}")
        return False
    
    try:
        # 加载OBJ文件
        print(f"[加载OBJ] 正在读取: {obj_path}")
        mesh = trimesh.load(obj_path)
        
        # 导出为GLB格式
        mesh.export(glb_path)
        print(f"[转换为GLB] => {glb_path}")
        return True
    except Exception as e:
        print(f"[转换为GLB] 转换失败: {str(e)}")
        return False


def main():
    # 获取项目根目录
    project_root = get_project_root()
    print(f"项目根目录: {project_root}")
    
    # 直接使用Data/frontend_img
    img_dir = os.path.join(project_root, 'Data', 'frontend_img')
    base_output_dir = os.path.join(project_root, 'samesh', 'assets')
    
    print(f"图片目录: {img_dir}")
    
    # 创建必要的目录
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 加载图像
    print(f"从 {img_dir} 加载图像...")
    samples = load_and_preprocess_images(img_dir)
    
    if not samples:
        print("未找到符合要求的样本组！")
        return
    
    print(f"成功加载 {len(samples)} 个完整样本组")
    
    # 记录处理过的ID，避免重复处理
    processed_ids = set()
    
    # 处理每个样本
    for sample in samples:
        try:
            obj_id = sample['id']
            category = sample['category']
            print(f"\n处理样本: {category}_{obj_id}")
            tensor, sample_info = process_sample(sample)
            
            print(f"生成的张量形状: {tensor.shape}")
            
            # 检查是否是用户上传的OBJ模型 (Uploaded_XX格式)
            is_user_obj = False
            user_model_id = None
            
            if category.lower() == 'uploaded':
                is_user_obj = True
                user_model_id = obj_id
                print(f"检测到用户上传模型: Uploaded_{user_model_id}")
            
            # 处理GLB文件
            if obj_id not in processed_ids:
                if is_user_obj:
                    # 对于用户上传的OBJ模型，尝试转换为GLB
                    print(f"处理用户OBJ模型 ID={user_model_id}")
                    convert_success = convert_obj_to_glb(user_model_id, base_output_dir)
                    if convert_success:
                        print(f"成功转换用户OBJ模型为GLB: {user_model_id}")
                else:
                    # 尝试转换URDF模型的GLB (不推荐，URDF已不再使用)
                    print(f"尝试处理URDF模型，但已不推荐使用该方法。")
                
                processed_ids.add(obj_id)
            
            # 尝试加载模型（如果存在）
            model_path = os.path.join(os.path.dirname(__file__), "outputs", "best_model.pth")
            if os.path.exists(model_path):
                print(f"\n尝试加载模型: {model_path}")
                
                try:
                    # 创建模型
                    model = FlexibleUNet(base_features=64, out_channels=1, bilinear=True)
                    
                    # 加载权重
                    checkpoint = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(checkpoint["model_state"])
                    
                    # 设置为评估模式
                    model.eval()
                    
                    # 添加批次维度
                    input_batch = tensor.unsqueeze(0)
                    
                    # 进行推理
                    with torch.no_grad():
                        output = model(input_batch)
                        pred = torch.sigmoid(output)
                    
                    # 二值化预测结果(阈值为0.5)
                    binary_pred = (pred > 0.5).float()
                    binary_mask = binary_pred[0, 0].numpy()
                    
                    # 确定输出目录和文件名
                    if is_user_obj:
                        # 用户上传的OBJ模型
                        gt_dir = os.path.join(base_output_dir, f"Uploaded_{user_model_id}_gt")
                        output_filename = f"Uploaded_{user_model_id}_pred.png"
                    else:
                        # URDF模型
                        gt_dir = os.path.join(base_output_dir, f"{obj_id}_gt")
                        # 首字母大写的类别名
                        capitalized_category = category.capitalize()
                        output_filename = f"{capitalized_category}_{obj_id}_pred.png"
                    
                    # 创建gt目录
                    os.makedirs(gt_dir, exist_ok=True)
                    
                    # 保存分割结果
                    output_path = os.path.join(gt_dir, output_filename)
                    cv2.imwrite(output_path, (binary_mask * 255).astype(np.uint8))
                    print(f"分割结果保存至: {output_path}")
                
                except Exception as e:
                    print(f"模型推理出错: {str(e)}")
            else:
                print(f"找不到模型文件: {model_path}")
            
        except Exception as e:
            print(f"处理样本时出错: {str(e)}")

    print("\n所有操作已完成!")


if __name__ == "__main__":
    main()
