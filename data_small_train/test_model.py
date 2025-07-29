#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型测试脚本

功能：
1. 加载训练好的模型
2. 在测试集上评估性能
3. 可视化预测结果
4. 生成详细的测试报告
5. 单样本预测分析
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score
)
from tqdm import tqdm
import logging
import trimesh

# 导入我们的模块
from loader import PartSegmentationDataset, get_dataloader
from model import create_model

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTester:
    """模型测试器"""
    
    def __init__(self, config, checkpoint_path, output_dir=None):
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir) if output_dir else self.checkpoint_path.parent / 'test_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = self._load_model()
        
        # 创建数据集
        self.test_dataset = PartSegmentationDataset(
            data_dir=config['data']['data_dir'],
            split='test',
            config=config['data']
        )
        
        # 确保测试时不shuffle
        test_dataloader_config = config['dataloader'].copy()
        test_dataloader_config['shuffle'] = False
        self.test_loader = get_dataloader(self.test_dataset, test_dataloader_config)
        
        logger.info(f"测试设备: {self.device}")
        logger.info(f"测试样本: {len(self.test_dataset)}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def _load_model(self):
        """加载训练好的模型"""
        logger.info(f"加载模型: {self.checkpoint_path}")
        
        # 创建模型
        model = create_model(self.config['model'])
        model.to(self.device)
        
        # 加载权重 - 适配新的检查点格式
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            # 旧格式
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            # 新格式
            model.load_state_dict(checkpoint['model'])
        else:
            # 直接是state_dict
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        logger.info("模型加载完成")
        return model
    
    @torch.no_grad()
    def evaluate_model(self):
        """评估模型性能"""
        logger.info("开始模型评估...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_confidences = []
        sample_results = []
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.test_loader, desc="测试进度")
        
        for features, labels, sample_info in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)
            confidence = features[:, 448:449]
            
            # 数值清理（与训练时保持一致）
            features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)
            confidence = torch.clamp(torch.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            features[:, 448:449] = confidence
            
            # 前向传播
            logits = self.model(features)
            
            # 检查输出是否有效
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.warning("模型输出包含NaN/Inf，跳过此batch")
                continue
                
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            # 计算损失
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            num_batches += 1
            
            # 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy().flatten())
            
            # 样本级别统计
            sample_acc = (preds == labels).float().mean().item()
            sample_results.append({
                'sample_name': sample_info['sample_name'],
                'category': sample_info['category'],
                'obj_id': sample_info['obj_id'],
                'joint_id': sample_info['joint_id'],
                'num_faces': sample_info['num_faces'],
                'accuracy': sample_acc,
                'loss': loss.item()
            })
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{total_loss/num_batches:.4f}',
                'Acc': f'{sample_acc:.4f}'
            })
        
        # 转换为numpy数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_confidences = np.array(all_confidences)
        
        # 计算指标
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        metrics['avg_loss'] = total_loss / num_batches
        
        return metrics, sample_results, all_labels, all_preds, all_probs, all_confidences
    
    def _calculate_metrics(self, labels, preds, probs):
        """计算详细指标"""
        metrics = {}
        
        # 基本指标
        metrics['accuracy'] = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # 类别级别指标
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            labels, preds, average=None
        )
        metrics['per_class'] = {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class,
            'support': support
        }
        
        # AUC (如果是二分类)
        if len(np.unique(labels)) == 2:
            metrics['auc'] = roc_auc_score(labels, probs[:, 1])
        
        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(labels, preds)
        
        return metrics
    
    def generate_report(self, metrics, sample_results):
        """生成测试报告"""
        logger.info("生成测试报告...")
        
        report_path = self.output_dir / 'test_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("模型测试报告\n")
            f.write("="*60 + "\n\n")
            
            # 基本信息
            f.write(f"模型路径: {self.checkpoint_path}\n")
            f.write(f"测试样本数: {len(sample_results)}\n")
            f.write(f"总面片数: {sum(r['num_faces'] for r in sample_results)}\n\n")
            
            # 整体指标
            f.write("整体性能指标:\n")
            f.write("-" * 30 + "\n")
            f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n")
            f.write(f"精确率 (Precision): {metrics['precision']:.4f}\n")
            f.write(f"召回率 (Recall): {metrics['recall']:.4f}\n")
            f.write(f"F1分数: {metrics['f1']:.4f}\n")
            f.write(f"平均损失: {metrics['avg_loss']:.4f}\n")
            
            if 'auc' in metrics:
                f.write(f"AUC: {metrics['auc']:.4f}\n")
            
            f.write("\n")
            
            # 类别级别指标
            f.write("类别级别指标:\n")
            f.write("-" * 30 + "\n")
            class_names = ['固定部分', '可动部分']
            for i, name in enumerate(class_names):
                if i < len(metrics['per_class']['precision']):
                    f.write(f"{name} (类别{i}):\n")
                    f.write(f"  精确率: {metrics['per_class']['precision'][i]:.4f}\n")
                    f.write(f"  召回率: {metrics['per_class']['recall'][i]:.4f}\n")
                    f.write(f"  F1分数: {metrics['per_class']['f1'][i]:.4f}\n")
                    f.write(f"  样本数: {metrics['per_class']['support'][i]}\n\n")
            
            # 混淆矩阵
            f.write("混淆矩阵:\n")
            f.write("-" * 30 + "\n")
            cm = metrics['confusion_matrix']
            f.write("预测\\实际  固定部分  可动部分\n")
            f.write(f"固定部分    {cm[0,0]:8d}  {cm[0,1]:8d}\n")
            f.write(f"可动部分    {cm[1,0]:8d}  {cm[1,1]:8d}\n\n")
            
            # 样本级别结果
            f.write("样本级别结果:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'样本名':<40} {'准确率':<8} {'损失':<8}\n")
            f.write("-" * 60 + "\n")
            
            # 按准确率排序
            sorted_results = sorted(sample_results, key=lambda x: x['accuracy'], reverse=True)
            for result in sorted_results:
                f.write(f"{result['sample_name']:<40} {result['accuracy']:<8.4f} {result['loss']:<8.4f}\n")
        
        logger.info(f"测试报告已保存: {report_path}")
    
    def plot_results(self, metrics, labels, preds, probs, confidences):
        """绘制测试结果可视化"""
        logger.info("生成可视化结果...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 混淆矩阵
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['固定', '可动'], yticklabels=['固定', '可动'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('混淆矩阵')
        axes[0, 0].set_xlabel('预测标签')
        axes[0, 0].set_ylabel('真实标签')
        
        # 2. 类别分布
        unique, counts = np.unique(labels, return_counts=True)
        axes[0, 1].bar(['固定部分', '可动部分'], counts, color=['skyblue', 'lightcoral'])
        axes[0, 1].set_title('真实标签分布')
        axes[0, 1].set_ylabel('数量')
        
        # 3. 预测置信度分布
        pred_confidences = probs.max(axis=1)
        axes[0, 2].hist(pred_confidences, bins=50, alpha=0.7, color='green')
        axes[0, 2].set_title('预测置信度分布')
        axes[0, 2].set_xlabel('置信度')
        axes[0, 2].set_ylabel('频次')
        
        # 4. 输入置信度 vs 预测准确性
        correct_mask = (preds == labels)
        axes[1, 0].scatter(confidences[correct_mask], pred_confidences[correct_mask], 
                          alpha=0.5, color='green', label='正确预测', s=1)
        axes[1, 0].scatter(confidences[~correct_mask], pred_confidences[~correct_mask], 
                          alpha=0.5, color='red', label='错误预测', s=1)
        axes[1, 0].set_xlabel('输入置信度')
        axes[1, 0].set_ylabel('预测置信度')
        axes[1, 0].set_title('置信度关系')
        axes[1, 0].legend()
        
        # 5. ROC曲线 (如果是二分类)
        if 'auc' in metrics:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            axes[1, 1].plot(fpr, tpr, label=f'ROC (AUC = {metrics["auc"]:.3f})')
            axes[1, 1].plot([0, 1], [0, 1], 'k--', label='随机')
            axes[1, 1].set_xlabel('假正率')
            axes[1, 1].set_ylabel('真正率')
            axes[1, 1].set_title('ROC曲线')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, '多分类任务\n无ROC曲线', ha='center', va='center', 
                           transform=axes[1, 1].transAxes)
        
        # 6. 性能指标雷达图
        categories = ['准确率', '精确率', '召回率', 'F1分数']
        values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合
        angles += angles[:1]
        
        axes[1, 2].plot(angles, values, 'o-', linewidth=2, label='模型性能')
        axes[1, 2].fill(angles, values, alpha=0.25)
        axes[1, 2].set_xticks(angles[:-1])
        axes[1, 2].set_xticklabels(categories)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_title('性能雷达图')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"可视化结果已保存: {self.output_dir / 'test_visualization.png'}")
    
    def test_single_sample(self, sample_idx=0):
        """测试单个样本的详细分析"""
        logger.info(f"分析单个样本: 索引 {sample_idx}")
        
        # 获取样本
        features, labels, sample_info = self.test_dataset[sample_idx]
        features = features.to(self.device)
        labels = labels.to(self.device)
        
        # 数值清理
        features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)
        confidence = torch.clamp(torch.nan_to_num(features[:, 448:449], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        features[:, 448:449] = confidence
        
        # 预测
        with torch.no_grad():
            logits = self.model(features)
            
            # 检查输出
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.error("模型输出包含NaN/Inf")
                return None
                
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
        
        # 分析结果
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        # 保存详细分析
        analysis_path = self.output_dir / f'sample_{sample_idx}_analysis.txt'
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(f"样本详细分析: {sample_info['sample_name']}\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"基本信息:\n")
            f.write(f"  类别: {sample_info['category']}\n")
            f.write(f"  物体ID: {sample_info['obj_id']}\n")
            f.write(f"  关节ID: {sample_info['joint_id']}\n")
            f.write(f"  面片数量: {sample_info['num_faces']}\n\n")
            
            f.write(f"预测统计:\n")
            f.write(f"  准确率: {(preds_np == labels_np).mean():.4f}\n")
            f.write(f"  预测为固定: {(preds_np == 0).sum()}\n")
            f.write(f"  预测为可动: {(preds_np == 1).sum()}\n")
            f.write(f"  实际固定: {(labels_np == 0).sum()}\n")
            f.write(f"  实际可动: {(labels_np == 1).sum()}\n\n")
            
            f.write(f"置信度统计:\n")
            f.write(f"  平均预测置信度: {probs_np.max(axis=1).mean():.4f}\n")
            f.write(f"  最高预测置信度: {probs_np.max():.4f}\n")
            f.write(f"  最低预测置信度: {probs_np.max(axis=1).min():.4f}\n")
        
        logger.info(f"单样本分析已保存: {analysis_path}")
        
        return {
            'sample_info': sample_info,
            'accuracy': (preds_np == labels_np).mean(),
            'predictions': preds_np,
            'probabilities': probs_np,
            'labels': labels_np
        }
    
    def generate_colored_ply(self, sample_idx=0):
        """生成带颜色的PLY文件：绿色=固定部分(0)，红色=可动部分(1)"""
        logger.info(f"生成样本 {sample_idx} 的彩色PLY文件...")
        
        # 获取样本预测结果
        result = self.test_single_sample(sample_idx)
        sample_info = result['sample_info']
        predictions = result['predictions']
        
        # 构建OBJ文件路径
        obj_path = Path(self.config['data']['data_dir']) / sample_info['sample_name'] / f"{sample_info['obj_id']}.obj"
        
        if not obj_path.exists():
            logger.error(f"OBJ文件不存在: {obj_path}")
            return None
        
        try:
            # 加载原始OBJ文件
            mesh = trimesh.load(obj_path)
            
            # 确保面片数量匹配
            if len(mesh.faces) != len(predictions):
                logger.warning(f"面片数量不匹配: OBJ={len(mesh.faces)}, 预测={len(predictions)}")
                # 取最小值以避免索引错误
                min_faces = min(len(mesh.faces), len(predictions))
                predictions = predictions[:min_faces]
                mesh.faces = mesh.faces[:min_faces]
            
            # 为每个面片分配颜色
            face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)  # RGBA
            
            for i, pred in enumerate(predictions):
                if pred == 0:  # 固定部分 - 绿色
                    face_colors[i] = [0, 255, 0, 255]  # 纯绿色
                else:  # 可动部分 - 红色
                    face_colors[i] = [255, 0, 0, 255]  # 纯红色
            
            # 设置面片颜色
            mesh.visual.face_colors = face_colors
            
            # 创建PLY输出目录
            ply_dir = self.output_dir / 'colored_ply'
            ply_dir.mkdir(exist_ok=True)
            
            # 保存为PLY文件
            ply_filename = f"{sample_info['sample_name']}_colored.ply"
            ply_path = ply_dir / ply_filename
            
            mesh.export(str(ply_path))
            
            logger.info(f"彩色PLY文件已保存: {ply_path}")
            
            # 生成颜色说明文件
            info_path = ply_dir / f"{sample_info['sample_name']}_color_info.txt"
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"彩色PLY文件说明: {ply_filename}\n")
                f.write("="*50 + "\n\n")
                f.write(f"样本信息:\n")
                f.write(f"  样本名: {sample_info['sample_name']}\n")
                f.write(f"  类别: {sample_info['category']}\n")
                f.write(f"  物体ID: {sample_info['obj_id']}\n")
                f.write(f"  关节ID: {sample_info['joint_id']}\n")
                f.write(f"  面片总数: {sample_info['num_faces']}\n\n")
                
                f.write(f"颜色编码:\n")
                f.write(f"  绿色 = 固定部分 (标签 0)\n")
                f.write(f"  红色 = 可动部分 (标签 1)\n\n")
                
                f.write(f"预测统计:\n")
                f.write(f"  预测为固定: {(predictions == 0).sum()} 个面片\n")
                f.write(f"  预测为可动: {(predictions == 1).sum()} 个面片\n")
                f.write(f"  预测准确率: {result['accuracy']:.4f}\n")
            
            return ply_path
            
        except Exception as e:
            logger.error(f"生成PLY文件时出错: {e}")
            return None
    
    def generate_all_colored_ply(self):
        """为所有测试样本生成彩色PLY文件"""
        logger.info("为所有测试样本生成彩色PLY文件...")
        
        ply_dir = self.output_dir / 'colored_ply'
        ply_dir.mkdir(exist_ok=True)
        
        successful_files = []
        failed_files = []
        
        for sample_idx in tqdm(range(len(self.test_dataset)), desc="生成PLY文件"):
            try:
                ply_path = self.generate_colored_ply(sample_idx)
                if ply_path:
                    successful_files.append(ply_path)
                else:
                    failed_files.append(sample_idx)
            except Exception as e:
                logger.error(f"处理样本 {sample_idx} 时出错: {e}")
                failed_files.append(sample_idx)
        
        # 生成总结报告
        summary_path = ply_dir / 'generation_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("彩色PLY文件生成总结\n")
            f.write("="*50 + "\n\n")
            f.write(f"成功生成: {len(successful_files)} 个文件\n")
            f.write(f"失败: {len(failed_files)} 个样本\n\n")
            
            if successful_files:
                f.write("成功生成的文件:\n")
                for ply_path in successful_files:
                    f.write(f"  {ply_path.name}\n")
                f.write("\n")
            
            if failed_files:
                f.write("失败的样本索引:\n")
                for idx in failed_files:
                    f.write(f"  样本 {idx}\n")
        
        logger.info(f"PLY文件生成完成: 成功 {len(successful_files)}, 失败 {len(failed_files)}")
        logger.info(f"PLY文件保存在: {ply_dir}")
        
        return successful_files, failed_files
    
    def run_full_test(self):
        """运行完整测试"""
        logger.info("开始完整测试...")
        
        # 1. 模型评估
        metrics, sample_results, labels, preds, probs, confidences = self.evaluate_model()
        
        # 2. 生成报告
        self.generate_report(metrics, sample_results)
        
        # 3. 可视化
        self.plot_results(metrics, labels, preds, probs, confidences)
        
        # 4. 单样本分析
        if len(self.test_dataset) > 0:
            self.test_single_sample(0)
        
        # 5. 生成彩色PLY文件
        self.generate_all_colored_ply()
        
        # 6. 保存数值结果
        results = {
            'metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in metrics.items() if k != 'confusion_matrix'},
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'sample_results': sample_results
        }
        
        import json
        with open(self.output_dir / 'test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info("完整测试完成！")
        
        # 打印总结
        print("\n" + "="*60)
        print("测试完成总结")
        print("="*60)
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        if 'auc' in metrics:
            print(f"AUC: {metrics['auc']:.4f}")
        print(f"平均损失: {metrics['avg_loss']:.4f}")
        print(f"测试样本数: {len(sample_results)}")
        print(f"结果保存在: {self.output_dir}")


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="测试训练好的SimpleAttentionMLP模型")
    parser.add_argument("--config", type=str, default="train_config.yaml", help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, default="experiments/attn_mlp_run1/best.pth", 
                       help="模型检查点路径")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--sample_idx", type=int, default=0, help="单样本分析的索引")
    parser.add_argument("--generate_ply", action="store_true", help="生成彩色PLY文件")
    parser.add_argument("--ply_only", action="store_true", help="只生成PLY文件，跳过其他测试")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建测试器
    tester = ModelTester(
        config=config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )
    
    # 根据参数运行不同的测试
    if args.ply_only:
        # 只生成PLY文件
        tester.generate_all_colored_ply()
    elif args.generate_ply:
        # 运行完整测试（包括PLY生成）
        tester.run_full_test()
    else:
        # 运行测试但不生成PLY文件
        # 临时修改run_full_test以跳过PLY生成
        logger.info("开始完整测试...")
        
        # 1. 模型评估
        metrics, sample_results, labels, preds, probs, confidences = tester.evaluate_model()
        
        # 2. 生成报告
        tester.generate_report(metrics, sample_results)
        
        # 3. 可视化
        tester.plot_results(metrics, labels, preds, probs, confidences)
        
        # 4. 单样本分析
        if len(tester.test_dataset) > 0:
            tester.test_single_sample(0)
        
        # 跳过PLY生成
        
        # 5. 保存数值结果
        results = {
            'metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in metrics.items() if k != 'confusion_matrix'},
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'sample_results': sample_results
        }
        
        import json
        with open(tester.output_dir / 'test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info("完整测试完成！")
        
        # 打印总结
        print("\n" + "="*60)
        print("测试完成总结")
        print("="*60)
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        if 'auc' in metrics:
            print(f"AUC: {metrics['auc']:.4f}")
        print(f"平均损失: {metrics['avg_loss']:.4f}")
        print(f"测试样本数: {len(sample_results)}")
        print(f"结果保存在: {tester.output_dir}")


if __name__ == "__main__":
    main() 