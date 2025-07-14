'''
PartField 特征提取推理脚本
python partfield_inference.py -c configs/final/demo.yaml --opts continue_ckpt model/model_objaverse.ckpt result_name partfield_features/output_folder dataset.data_path data/input_folder
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
                      max_epochs=cfg.training_epochs,  # 最大训练周期，✗ 无用
                      log_every_n_steps=1,  # 每步都记录日志，✗ 无用
                      limit_train_batches=3500,  # 限制训练批次数，✗ 无用
                      limit_val_batches=None,  # 不限制验证批次，✗ 无用
                      callbacks=checkpoint_callbacks  # 使用检查点回调，✗ 无用
                     )

    # 导入模型定义并实例化
    from partfield.model_trainer_pvcnn_only_demo import Model   ###!!!partfield/model_trainer_pvcnn_only_demo.py
    model = Model(cfg)        

    # 如果是重网格化演示模式，调整采样点数
    if cfg.remesh_demo:
        cfg.n_point_per_face = 10

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