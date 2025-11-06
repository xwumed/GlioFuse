import argparse
import os
import torch
from config import (
    BASE_TRAINING_PARAMS, 
    PROCESSED_DATA_DIR, 
    CHECKPOINTS_DIR, 
    INPUT_SIZE_PATH, 
    INPUT_SIZE_RAD
)


# ================================================================
# 模型命名别名与工具函数（简洁命名 ↔ 内部模式名）
# ================================================================
# 规范化：
#  - 内部模式名（旧）：'rad_only','path_only','simple_fusion','multiscale_fusion','coattn','bilinear_fusion','late_fusion'
#  - 简洁显示名（新）：'RadNet','PathNet','EarlyFusionNet','MultiScaleFusionNet','CoAttentionNet','BilinearFusion','LateFusion'

_INTERNAL_TO_SIMPLIFIED = {
    'rad_only': 'RadNet',
    'path_only': 'PathNet',
    'simple_fusion': 'EarlyFusionNet',
    'multiscale_fusion': 'MultiScaleFusionNet',
    'coattn': 'CoAttentionNet',
    'bilinear_fusion': 'BilinearFusionNet',
    'late_fusion': 'LateFusionNet',
}

_SIMPLIFIED_TO_INTERNAL = {
    'radnet': 'rad_only',
    'pathnet': 'path_only',
    'earlyfusionnet': 'simple_fusion',
    'multiscalefusionnet': 'multiscale_fusion',
    'coattentionnet': 'coattn',
    'coattention': 'coattn',
    'coattnnet': 'coattn',
    'bilinearfusion': 'bilinear_fusion',
    'bilinearfusionnet': 'bilinear_fusion',
    'latefusion': 'late_fusion',
    'latefusionnet': 'late_fusion',
}

_INTERNAL_CANONICAL = set(_INTERNAL_TO_SIMPLIFIED.keys())

def normalize_mode(mode: str) -> str:
    """将任意输入（简洁名/旧名/大小写不敏感）规范为内部模式名。"""
    if not isinstance(mode, str):
        return mode
    m = mode.strip()
    if m in _INTERNAL_CANONICAL:
        return m
    ml = m.replace('-', '_').lower()
    # 直接支持旧名的大小写变体
    if ml in _INTERNAL_CANONICAL:
        return ml
    # 简洁名映射
    return _SIMPLIFIED_TO_INTERNAL.get(ml, m)

def to_simplified_mode(mode: str) -> str:
    """将内部模式名映射为简洁显示名；若已是简洁名则规范后返回。"""
    internal = normalize_mode(mode)
    return _INTERNAL_TO_SIMPLIFIED.get(internal, mode)

def get_all_simplified_modes():
    """返回简洁显示名列表（用于 CLI 默认值与展示）。"""
    return [
        _INTERNAL_TO_SIMPLIFIED['path_only'],
        _INTERNAL_TO_SIMPLIFIED['rad_only'],
        _INTERNAL_TO_SIMPLIFIED['simple_fusion'],
        _INTERNAL_TO_SIMPLIFIED['multiscale_fusion'],
        _INTERNAL_TO_SIMPLIFIED['coattn'],
        _INTERNAL_TO_SIMPLIFIED['bilinear_fusion'],
    ]

def get_base_parser():
    """
    获取一个包含所有可配置参数的 ArgumentParser 实例。
    """
    parser = argparse.ArgumentParser(
        description='Radiomic-Pathomic Fusion Framework for Survival Analysis',
        formatter_class=argparse.RawTextHelpFormatter
    )
    defaults = BASE_TRAINING_PARAMS

    # --- 1. 基本设置 ---
    parser.add_argument('--dataroot', default=PROCESSED_DATA_DIR, help="包含处理后数据的根目录")
    parser.add_argument('--checkpoints_dir', default=CHECKPOINTS_DIR, help='所有模型和日志的保存根目录')
    parser.add_argument('--exp_name', type=str, default='fusion_experiment', help='本次实验的名称 (将在此目录下创建子文件夹)')
    parser.add_argument('--gpu_ids', type=str, default=defaults['gpu_ids'], help='GPU IDs (e.g., 0,1). 使用 -1 代表 CPU')
    parser.add_argument('--model_name', type=str, default='FusionModel', help='为本次运行保存的模型指定名称')

    # --- 2. 数据与维度配置 ---
    parser.add_argument('--input_size_path', type=int, default=INPUT_SIZE_PATH, help="WSI (Pathomics) 特征原始维度")
    parser.add_argument('--input_size_rad', type=int, default=INPUT_SIZE_RAD, help="MRI (Radiomics) 特征原始维度")
    # 已采用离线降维流程，移除模型内再次降维的相关选项
    parser.add_argument('--path_dim', type=int, default=128, help='WSI特征编码后的最终维度')
    parser.add_argument('--rad_dim', type=int, default=128, help='MRI特征编码后的最终维度')
    parser.add_argument('--fusion_proj_dim', type=int, default=128, help='融合前将两模态投影到的公共维度 (用于简单/加权/注意力融合)')
    parser.add_argument('--lr_encoder_ratio', type=float, default=0.1, help='编码器相对主学习率的比例 (融合模型差分学习率)')
    parser.add_argument('--wd_encoder_ratio', type=float, default=0.1, help='编码器相对主 weight_decay 的比例 (融合模型差分权重衰减)')
    parser.add_argument('--use_fuse_mlp', type=int, default=1, help='是否在融合前对投影特征加一层轻量 MLP 残差块 (1=是, 0=否)')
    parser.add_argument('--fuse_mlp_hidden_multiplier', type=float, default=2.0, help='融合前 MLP 隐藏维度系数 (hidden=fuse_mlp_hidden_multiplier*fusion_proj_dim)')

    # --- 3. 训练超参数 ---
    parser.add_argument('--optimizer_type', type=str, default=defaults['optimizer_type'], choices=['adam', 'adamw'], help='优化器类型')
    parser.add_argument('--lr_policy', default=defaults['lr_policy'], type=str, choices=['linear', 'cosine', 'plateau', 'adaptive'], help='学习率衰减策略')
    parser.add_argument('--lr', default=defaults['lr'], type=float, help='初始学习率')
    parser.add_argument('--weight_decay', default=defaults['weight_decay'], type=float, help='L2权重衰减 (用于 Adam, AdamW, Adagrad)')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam/AdamW 的 beta1 参数')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam/AdamW 的 beta2 参数')
    parser.add_argument('--niter', type=int, default=defaults['niter'], help='保持初始学习率的epoch数')
    parser.add_argument('--niter_decay', type=int, default=defaults['niter_decay'], help='学习率衰减所需的epoch数')

    parser.add_argument('--batch_size', type=int, default=defaults['batch_size'], help='批量大小')
    parser.add_argument('--dropout_rate', default=defaults['dropout_rate'], type=float, help='Dropout比率')

    # --- 3.2 训练稳定化选项（关闭：回退到原始行为）---
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='梯度累积步数(>1时等效于更大批量)')
    parser.add_argument('--use_ema', type=int, default=0, help='是否启用参数EMA用于验证/早停 (1/0)')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA衰减系数，越接近1越平滑')

    # --- 3.1 单模态（rad_only/path_only）稳健化选项 ---
    parser.add_argument('--unimodal_norm_type', type=str, default='ln', choices=['ln', 'bn'], help='单模态编码器隐藏层归一化类型')
    parser.add_argument('--unimodal_feature_dropout_p', type=float, default=0.1, help='单模态输入特征级随机丢弃概率')
    parser.add_argument('--unimodal_se_gate', type=int, default=1, help='是否启用SE门控进行重标定(1/0)')
    parser.add_argument('--unimodal_input_noise_std', type=float, default=0.02, help='训练时对单模态输入注入高斯噪声的标准差')

    # --- 4. 损失与正则化 ---
    parser.add_argument('--lambda_cox', type=float, default=1.0, help='Cox生存损失的权重')
    parser.add_argument('--lambda_reg', type=float, default=defaults.get('lambda_reg', 1e-5), help='L1正则化损失的权重')
    parser.add_argument('--align_loss_weight', type=float, default=0.1, help='两模态投影后特征对齐损失（MSE）的权重；0 关闭')
    parser.add_argument('--modality_dropout_p', type=float, default=0.1, help='训练时随机丢弃任一模态的概率（提升鲁棒性）；0 关闭')
    parser.add_argument('--freeze_encoder_epochs', type=int, default=0, help='前若干个 epoch 冻结编码器，仅训练融合头 (0 表示不冻结)')
    parser.add_argument('--reg_type', default='rad', type=str, choices=['none', 'path', 'rad', 'fusion'], 
                        help='L1正则化的类型。\n'
                             '"rad": (默认) 仅对 MRI 分支正则化。\n'
                             '"fusion": 仅对融合层和后续分类器正则化。\n'
                             '"path": 仅对 WSI 分支正则化。\n'
                             '"none": 不使用L1正则化。')
    
    # --- 5. 模型专用参数 ---
    parser.add_argument('--act_type', type=str, default='none', help='模型最终输出的激活函数')
    parser.add_argument('--label_dim', type=int, default=1, help='输出维度 (生存分析通常为1)')


    return parser

def parse_gpuids(opt):
    """解析GPU ID字符串为列表"""
    str_ids = opt.gpu_ids.split(','); opt.gpu_ids = []
    for str_id in str_ids:
        try: 
            id = int(str_id)
            if id >= 0: opt.gpu_ids.append(id)
        except ValueError: pass
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available(): torch.cuda.set_device(opt.gpu_ids[0])
    return opt

def print_options(opt, parser):
    """
    打印并保存所有配置参数到一个文本文件中。
    """
    message = '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = f'\t[default: {default}]'
        message += f'{str(k):>25}: {str(v):<30}{comment}\n'
    message += '----------------- End -------------------\n'
    print(message)

    # 将配置保存到实验目录下的文本文件中
    expr_dir = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)
    os.makedirs(expr_dir, exist_ok=True)
    file_name = os.path.join(expr_dir, 'opt_summary.txt')
    try:
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
    except Exception as e:
        print(f"警告: 无法保存 options 到 {file_name}。错误: {e}")