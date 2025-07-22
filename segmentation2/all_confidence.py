#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遍历 data/img 下 **所有 arrow‑sketch PNG**，输出：
  1. 逐像素置信度 .npy ⟶ data/npy（保持原目录结构，扩展名改 .npy）
  2. 二值预测掩码 PNG  ⟶ data/img_pred（保持原目录结构，扩展名同 .png）
  3. 三联可视化图  ⟶ data/img_vis（保持原目录结构，扩展名同 _vis.png）

固定目录：
  ─ 输入原图  : /home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/segmentation2/data/img
  ─ 输出 .npy: /home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/segmentation2/data/npy
  ─ 输出掩码 : /home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/segmentation2/data/img_pred
  ─ 输出可视化: /home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/segmentation2/data/img_vis

仅处理文件名包含 “_arrow‑sketch_” 的 PNG（与训练/测试 loader 一致）。
用法示例：
    python all_confidence.py --resume ./outputs_sketch2seg/best_model.pth
"""

# --------------------------------------------------------------------------- #
#                                   Imports                                   #
# --------------------------------------------------------------------------- #
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm

from network import SketchSegmentUNet


# --------------------------------------------------------------------------- #
#                               Argument parser                               #
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump per‑pixel confidence・pred PNG・visualization for arrow‑sketch images")
    parser.add_argument("--img_dir", type=str,
                        default="/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/segmentation2/data/img",
                        help="Root directory containing PNGs (scanned recursively).")
    parser.add_argument("--npy_dir", type=str,
                        default="/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/segmentation2/data/npy",
                        help="Directory to save .npy files (mirrors img hierarchy).")
    parser.add_argument("--pred_dir", type=str,
                        default="/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/segmentation2/data/img_pred",
                        help="Directory to save predicted mask PNGs.")
    parser.add_argument("--vis_dir", type=str,
                        default="/home/ipab-graphics/workplace/PartField_Sketch_simpleMLP/segmentation2/data/img_vis",
                        help="Directory to save visualization PNGs.")
    parser.add_argument("--resume", type=str, required=True,
                        help="Path to trained checkpoint (.pth).")
    parser.add_argument("--features", type=int, default=128,
                        help="UNet base feature channels (must match training).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#                          Image I/O & preprocessing                           #
# --------------------------------------------------------------------------- #
def preprocess_png(path):
    """
    Read BGR PNG → float32 RGB tensor [0,1], shape (3,H,W).
    Also return the HWC RGB array for later visualization.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)), rgb


# --------------------------------------------------------------------------- #
#                              Visualization util                             #
# --------------------------------------------------------------------------- #
def save_visualization(orig_img, prob_np, img_path, vis_dir, rel_root):
    """
    Save a 3‑panel PNG: [input image | binary mask | masked relative‑confidence].
    """
    pred_bin = (prob_np > 0.5).astype(np.uint8)
    mask_vals = prob_np[pred_bin == 1]
    if mask_vals.size > 0 and mask_vals.max() > 0.5:
        vis_vals = (prob_np - 0.5) / (mask_vals.max() - 0.5)
        vis_vals = np.clip(vis_vals, 0.0, 1.0)
    else:
        vis_vals = np.zeros_like(prob_np)

    cmap = cm.get_cmap('jet')
    color_rgba = cmap(vis_vals)
    color_rgba[..., 3] = pred_bin  # 透明度：非 mask 区域 0

    basename = os.path.basename(img_path)
    filename_no_ext = os.path.splitext(basename)[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(orig_img)
    axes[0].set_title("Input Image")
    axes[1].imshow(pred_bin, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Pred Mask")
    axes[2].imshow(color_rgba)
    axes[2].set_title("Masked Rel‑Confidence")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(filename_no_ext, fontsize=14)
    fig.tight_layout()

    rel_path = os.path.relpath(img_path, rel_root)
    vis_path = os.path.join(vis_dir, os.path.splitext(rel_path)[0] + "_vis.png")
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    plt.savefig(vis_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#                                 Main logic                                  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def main():
    args = parse_args()

    os.makedirs(args.npy_dir,  exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)
    os.makedirs(args.vis_dir,  exist_ok=True)

    # -------- Collect arrow‑sketch PNGs --------
    png_list = []
    for root, _, files in os.walk(args.img_dir):
        for f in files:
            if f.lower().endswith(".png") and "_arrow-sketch_" in f.lower():
                png_list.append(os.path.join(root, f))
    if not png_list:
        raise RuntimeError("No arrow‑sketch PNG found!")

    # -------- Load model --------
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

    # -------- Batched inference --------
    batch_imgs   = []
    batch_orig   = []
    batch_paths  = []

    pbar = tqdm(png_list, desc="Infer", ncols=120)
    for path in pbar:
        tensor_img, orig_img = preprocess_png(path)
        batch_imgs.append(tensor_img)
        batch_orig.append(orig_img)
        batch_paths.append(path)

        flush = (len(batch_imgs) == args.batch_size) or (path == png_list[-1])
        if not flush:
            continue

        inp    = torch.stack(batch_imgs, dim=0).to(device)   # (B,3,H,W)
        logits = model(inp)                                  # (B,1,H,W)
        probs  = torch.sigmoid(logits).cpu().numpy()[:, 0]   # (B,H,W)

        for img_path, orig_img_np, prob in zip(batch_paths, batch_orig, probs):
            rel = os.path.relpath(img_path, args.img_dir)    # 相对路径 (for mirrors)

            # ---------- save .npy ----------
            npy_path = os.path.join(args.npy_dir,
                                   os.path.splitext(rel)[0] + ".npy")
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, prob.astype(np.float32))

            # ---------- save binary mask PNG ----------
            pred_bin = (prob > 0.5).astype(np.uint8) * 255
            png_out  = os.path.join(args.pred_dir, rel)
            os.makedirs(os.path.dirname(png_out), exist_ok=True)
            cv2.imwrite(png_out, pred_bin)

            # ---------- save visualization ----------
            # save_visualization(orig_img_np, prob, img_path, args.vis_dir, args.img_dir)

        batch_imgs.clear()
        batch_orig.clear()
        batch_paths.clear()


if __name__ == "__main__":
    main()
