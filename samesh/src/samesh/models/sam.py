import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoModel


# 使用 SAM2 模型时导入相应模块
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from samesh.data.common import NumpyTensor


def combine_bmasks(masks: NumpyTensor['n h w'], sort=False) -> NumpyTensor['h w']:
    mask_combined = np.zeros_like(masks[0], dtype=int)
    if sort:
        masks = sorted(masks, key=lambda x: x.sum(), reverse=True)
    for i, mask in enumerate(masks):
        mask_combined[mask] = i + 1
    return mask_combined


def decompose_mask(mask: NumpyTensor['h w'], background=0) -> NumpyTensor['n h w']:
    labels = np.unique(mask)
    labels = labels[labels != background]
    return mask == labels[:, None, None]


def remove_artifacts(mask: NumpyTensor['h w'], mode: str, min_area=128) -> NumpyTensor['h w']:
    assert mode in ['holes', 'islands']
    mode_holes = (mode == 'holes')

    def remove_helper(bmask):
        bmask = (mode_holes ^ bmask).astype(np.uint8)
        nregions, regions, stats, _ = cv2.connectedComponentsWithStats(bmask, 8)
        sizes = stats[:, -1][1:]
        fill = [i + 1 for i, s in enumerate(sizes) if s < min_area] + [0]
        if not mode_holes:
            fill = [i for i in range(nregions) if i not in fill]
        return np.isin(regions, fill)

    mask_combined = np.zeros_like(mask)
    for label in np.unique(mask):
        mask_combined[remove_helper(mask == label)] = label
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


def colormap_bmask(bmask: NumpyTensor['h w']) -> Image.Image:
    return colormap_mask(bmask, background=np.array([0, 0, 0]), foreground=np.array([255, 255, 255]))


def colormap_bmasks(
    masks: NumpyTensor['n h w'],
    image: NumpyTensor['h w 3'] = None,
    background=np.array([255, 255, 255]),
    blend=0.25
) -> Image.Image:
    mask = combine_bmasks(masks)
    return colormap_mask(mask, image, background=background, blend=blend)


def point_grid_from_mask(mask: NumpyTensor['h w'], n: int) -> NumpyTensor['n 2']:
    valid = np.argwhere(mask)
    if len(valid) == 0:
        raise ValueError('No valid points in mask')
    h, w = mask.shape
    n = min(n, len(valid))
    indices = np.random.choice(len(valid), n, replace=False)
    samples = valid[indices].astype(float)
    samples[:, 0] /= h - 1
    samples[:, 1] /= w - 1
    samples = samples[:, [1, 0]]
    samples = samples[np.lexsort((samples[:, 1], samples[:, 0]))]
    return samples


# class SamModel(nn.Module):
#     """
#     封装 SAM 模型，仅使用 SAM2 模型。
#     """
#     def __init__(self, config: OmegaConf, device='cuda'):
#         super().__init__()
#         self.config = config
#         self.device = device
#         if config.sam.auto:
#             self.setup_sam(mode='auto')
#         else:
#             if config.sam.ground:
#                 self.setup_grounding_dino()
#             self.setup_sam(mode='pred')
# 
#     def setup_sam(self, mode='auto'):
#         """
#         使用 SAM2 模型加载和构建分割引擎。
#         """
#         # 使用相对于 sam2 包的配置路径
#         model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
#         print("Using model_config:", model_config, flush=True)
#         
#         self.sam_model = build_sam2(model_config, self.config.sam.checkpoint,
#                                     device=self.device, apply_postprocessing=False)
#         self.engine = {
#             'pred': SAM2ImagePredictor,
#             'auto': SAM2AutomaticMaskGenerator,
#         }[mode](self.sam_model, **self.config.sam.get('engine_config', {}))
#         self.sam_model = self.sam_model.to(self.device)
#         self.sam_model.eval()
# 
#     def setup_grounding_dino(self):
#         self.grounding_dino_processor, self.grounding_dino_model = (
#             AutoProcessor.from_pretrained(self.config.grounding_dino.checkpoint),
#             AutoModel.from_pretrained(self.config.grounding_dino.checkpoint).to(self.device)
#         )
# 
#     def process_image(self, image: Image, prompt: dict = None) -> NumpyTensor['n h w']:
#         # 确保图像是 RGB 格式
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
#         image = np.array(image)
#         if self.config.sam.auto:
#             annotations = self.engine.generate(image)
#         else:
#             self.engine.set_image(image)
#             annotations = self.engine.predict(**prompt)[0]
#             annotations = [{'segmentation': m, 'area': m.sum().item()} for m in annotations]
#         annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
#         masks = np.stack([anno['segmentation'] for anno in annotations])
#         return masks
# 
#     def process_boxes(self, image: Image, texts: list[str]) -> tuple[
#         list[NumpyTensor[4]],
#         list[NumpyTensor[2]]
#     ]:
#         texts = '. '.join(texts)
#         inputs = self.grounding_dino_processor(texts, return_tensors='pt').to(self.device)
#         with torch.no_grad():
#             outputs = self.grounding_dino_model(**inputs)
#         boxes, logits = self.grounding_dino_processor.post_process_grounded_object_detection(
#             outputs,
#             inputs.input_ids,
#             box_threshold=0.4, text_threshold=0.3, target_sizes=[image.size[::-1]]
#         )
#         return boxes, logits
# 
#     def forward(self, image: Image, texts: list[str] = None) -> NumpyTensor['n h w']:
#         if self.config.sam.auto:
#             masks = self.process_image(image)
#         else:
#             boxes, _ = self.process_boxes(image, texts)
#             masks = []
#             for box in boxes:
#                 masks.append(self.process_image(image, {'box': box}))
#             masks = np.concatenate(masks)
#         return masks
# 
# 
# class Sam2Model(SamModel):
#     """
#     Sam2Model 继承自 SamModel，仅使用 SAM2 模型。
#     """
#     def setup_sam(self, mode='auto'):
#         # 使用相对于 sam2 包的配置路径
#         model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
#         print("Using model_config:", model_config, flush=True)
#         
#         self.sam_model = build_sam2(model_config, self.config.sam.checkpoint,
#                                     device=self.device, apply_postprocessing=False)
#         self.sam_model.eval()
#         self.engine = {
#             'pred': SAM2ImagePredictor,
#             'auto': SAM2AutomaticMaskGenerator,
#         }[mode](self.sam_model, **self.config.sam.get('engine_config', {}))

# if __name__ == '__main__':
#     import time
# 
#     device = 'cuda'
#     # 加载图像并确保是 RGB 格式
#     image_path = './assets/Microwave_7128_default_right.png'
#     print(f"Loading image from: {image_path}")
#     image = Image.open(image_path)
#     print(f"Original image mode: {image.mode}")
#     if image.mode != 'RGB':
#         image = image.convert('RGB')
#         print("Converted image to RGB mode")
# 
#     # 示例：使用 SAM2 模型
#     config2 = OmegaConf.create({
#         'sam': {
#             'checkpoint': '../../../checkpoint/sam2.1_hiera_large.pt',
#             'auto': True,
#             'engine_config': {'points_per_side': 32},
#         },
#     })
# 
#     sam2 = Sam2Model(config2, device)
#     start_time = time.time()
#     masks = sam2(image)
#     print(f'Elapsed time: {time.time() - start_time:.2f} s')
#     image = colormap_bmasks(masks, np.array(image))
#     image.save('test_mask2.png')

