# filename: all_confidence.py
"""
遍历 data/img 下所有 PNG，输出：
  1. 逐像素置信度 .npy ⟶ data/npy（保持原目录结构，扩展名改 .npy）
  2. 二值预测掩码 PNG  ⟶ data/img_pred（保持原目录结构，扩展名改 .png）

- 输入固定目录:
    /home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/segmentation2/data/img
- .npy 输出目录:
    /home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/segmentation2/data/npy
- 掩码 PNG 输出目录:
    /home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/segmentation2/data/img_pred

用法:
    python all_confidence.py --resume ./outputs_sketch2seg/best_model.pth
"""

import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from network import SketchSegmentUNet


# --------------------------- 参数解析 ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Dump per‑pixel confidence & pred PNGs for all PNGs")
    parser.add_argument("--img_dir", type=str,
                        default="/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/segmentation2/data/img",
                        help="Directory containing PNGs (searched 递归).")
    parser.add_argument("--npy_dir", type=str,
                        default="/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/segmentation2/data/npy",
                        help="Directory to save .npy (mirrors img names).")
    parser.add_argument("--pred_dir", type=str,
                        default="/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/segmentation2/data/img_pred",
                        help="Directory to save predicted mask PNGs.")
    parser.add_argument("--resume",  type=str, required=True,
                        help="Path to trained checkpoint (.pth).")
    parser.add_argument("--features", type=int, default=128,
                        help="UNet base feature channels (match training).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference.")
    return parser.parse_args()


# --------------------------- 图像读取 & 预处理 ---------------------------
def preprocess_png(path):
    """BGR PNG → float32 RGB Tensor in [0,1], shape (3,H,W)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1))  # CHW


# --------------------------- 主逻辑 ---------------------------
@torch.no_grad()
def main():
    args = parse_args()

    os.makedirs(args.npy_dir,  exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    # -------- 收集 PNG 路径 --------
    png_list = []
    for root, _, files in os.walk(args.img_dir):
        for f in files:
            if f.lower().endswith(".png"):
                png_list.append(os.path.join(root, f))
    if not png_list:
        raise RuntimeError("No PNG found!")

    # -------- 模型加载 --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SketchSegmentUNet(in_channels=3,
                              out_channels=1,
                              base_features=args.features,
                              bilinear=False)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    ckpt = torch.load(args.resume, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt["model_state"])
    model.eval()

    # -------- 批量推理 --------
    batch_imgs = []
    batch_paths = []
    pbar = tqdm(png_list, desc="Infer", ncols=120)

    for path in pbar:
        batch_imgs.append(preprocess_png(path))
        batch_paths.append(path)

        flush = (len(batch_imgs) == args.batch_size) or (path == png_list[-1])
        if not flush:
            continue

        inp = torch.stack(batch_imgs, dim=0).to(device)        # (B,3,H,W)
        logits = model(inp)                                     # (B,1,H,W)
        probs  = torch.sigmoid(logits).cpu().numpy()[:, 0]      # (B,H,W)

        for img_path, prob in zip(batch_paths, probs):
            rel = os.path.relpath(img_path, args.img_dir)       # 相对路径 e.g. chair/x.png

            # ---------- 保存 .npy ----------
            npy_path = os.path.join(args.npy_dir,
                                   os.path.splitext(rel)[0] + ".npy")
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, prob.astype(np.float32))

            # ---------- 保存掩码 PNG ----------
            pred_bin = (prob > 0.5).astype(np.uint8) * 255      # 0/255
            png_out  = os.path.join(args.pred_dir, rel)
            os.makedirs(os.path.dirname(png_out), exist_ok=True)
            cv2.imwrite(png_out, pred_bin)

        batch_imgs.clear()
        batch_paths.clear()


if __name__ == "__main__":
    main()
