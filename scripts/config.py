# config.py

import os

# =======================================================================================
# 1. 项目基础路径配置
# =======================================================================================
PROJECT_ROOT = os.environ.get(
    'PROJECT_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)  # 自动推断项目根目录，允许通过环境变量覆盖

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

CV_SPLITS_DIR = os.path.join(PROCESSED_DATA_DIR, 'cv_splits')
EXTERNAL_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'external_test')
COX_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, 'cox_baselines')
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')
PRETRAINED_MODELS_DIR = CHECKPOINTS_DIR 

# =======================================================================================
# 2. 数据集配置
# =======================================================================================
AVAILABLE_COHORTS = ['nanfang', 'huaqiao', 'tcga', 'cptac']

# =======================================================================================
# 3. 数据处理与模型共享配置
# =======================================================================================
N_SPLITS = 5
RANDOM_STATE = 42
PATIENT_ID_COLUMN = 'case_id'
TIME_COLUMN = 'os.days'
EVENT_COLUMN = 'os.status'
INPUT_SIZE_RAD = 768  # 更新为降维后的MRI特征维度 (原12216)
INPUT_SIZE_PATH = 768  # WSI特征维度保持不变
RAD_REDUCTION_DIM = 768  # MRI降维后目标维度为768（4序列×192），与INPUT_SIZE_RAD一致

# =======================================================================================
# 4. 模型训练基础参数 (已使用优化后的参数作为默认值)
# =======================================================================================
BASE_TRAINING_PARAMS = {
    'dataroot': PROCESSED_DATA_DIR,
    'checkpoints_dir': CHECKPOINTS_DIR,
    'gpu_ids': '0',
    'optimizer_type': 'adamw',  # 大部分模型的最佳优化器
    'lr_policy': 'cosine',
    'niter': 80,
    'niter_decay': 40,
    'batch_size': 32,
    'lr': 7e-05,                # 精细调整学习率
    'dropout_rate': 0.25,       # 保持适中dropout
    'weight_decay': 1.2e-04,    # 适中的权重衰减
    'lambda_reg': 1.5e-05,      # 适中的L1正则化
}

# =======================================================================================
# 5. 实验配置 (用于 run_baselines.py)
# =======================================================================================
BASELINE_EXPERIMENTS = [
    {'name': 'WSI-Only Baseline (Pathomics)', 'params': {'mode': 'path_only', 'model_name': 'Path_Only_Baseline'}},
    {'name': 'MRI-Only Baseline (Radiomics)', 'params': {'mode': 'rad_only', 'model_name': 'Rad_Only_Baseline'}},
    {'name': 'Simple Fusion (Concatenation)', 'params': {'mode': 'simple_fusion', 'model_name': 'SimpleFusion_Baseline'}},
    {'name': 'Multi-Scale Fusion (Innovative)', 'params': {'mode': 'multiscale_fusion', 'model_name': 'MultiScaleFusion_Baseline'}},

    {'name': 'Co-Attention Fusion (768D input)', 'params': {'mode': 'coattn', 'model_name': 'CoAttention_Baseline'}},
    {'name': 'Bilinear Fusion (New)', 'params': {'mode': 'bilinear_fusion', 'model_name': 'BilinearFusion_Baseline'}},
]

# =======================================================================================
# 6. OPTIMIZED HYPERPARAMETERS (由 apply_tuning_results.py 自动更新)
# =======================================================================================
OPTIMIZED_HYPERPARAMS = {
    'bilinear_fusion': {
        'align_loss_weight': 0,
        'dropout_rate': 0.35,
        'fusion_proj_dim': 128,
        'gate_path': True,
        'gate_rad': True,
        'lambda_reg': 1e-05,
        'lr': 0.0002,
        'lr_encoder_ratio': 0.1,
        'modality_dropout_p': 0.05,
        'model_name': 'BilinearFusion_optimized',
        'scale_dim1': 4,
        'scale_dim2': 4,
        'skip': True,
        'weight_decay': 3e-05,
    },
    'coattn': {
        'act_type': 'none',
        'alpha_surv': 0,
        'coattn_align_loss_weight': 0.08,
        'coattn_embed_dim': 256,
        'coattn_num_heads': 4,
        'coattn_num_layers': 2,
        'coattn_use_cls_token': True,
        'coattn_use_gating': True,
        'coattn_use_pos_encoding': True,
        'dropinput': 0.12,
        'dropout': 0.25,
        'fusion_proj_dim': 128,
        'fusion_type': 'bilinear',
        'gate_path': True,
        'gate_rad': True,
        'lambda_reg': 0.00015,
        'modality_dropout_p': 0.08,
        'n_classes': 1,
        'path_input_dim': 768,
        'rad_input_dim': 768,
        'reg_type': 'none',
        'scale_dim1': 8,
        'scale_dim2': 8,
        'size_arg': 'small',
        'skip': True,
    },
    'multiscale_fusion': {
        'align_loss_weight': 0.05,
        'dropout_rate': 0.39550224,
        'freeze_encoder_epochs': 0,
        'fusion_proj_dim': 64,
        'lambda_reg': 9.8880573e-05,
        'lr': 0.00021301733,
        'lr_encoder_ratio': 0.056284045,
        'modality_dropout_p': 0.014663924,
        'weight_decay': 1.3262575e-05,
    },
    'path_only': {
        'dropout_rate': 0.33920171,
        'lambda_reg': 9.0629253e-07,
        'lr': 0.00029336312,
        'weight_decay': 0.000467723,
    },
    'rad_only': {
        'dropout_rate': 0.78321771,
        'lambda_reg': 2.2685626e-07,
        'lr': 0.00028196263,
        'unimodal_feature_dropout_p': 0.12219004,
        'unimodal_input_noise_std': 0.023841597,
        'unimodal_norm_type': 'bn',
        'unimodal_se_gate': 0,
        'weight_decay': 0.00048550776,
    },
    'simple_fusion': {
        'align_loss_weight': 0,
        'dropout_rate': 0.52581967,
        'freeze_encoder_epochs': 5,
        'fusion_proj_dim': 64,
        'lambda_reg': 1.1958806e-05,
        'lr': 0.00049813828,
        'lr_encoder_ratio': 0.19472183,
        'modality_dropout_p': 0.071840503,
        'weight_decay': 6.0024133e-05,
    },
}

# =======================================================================================
# 7. 工具函数
# =======================================================================================
def ensure_directories_exist():
    """确保所有必要的目录存在"""
    directories = [PROCESSED_DATA_DIR, CV_SPLITS_DIR, EXTERNAL_DATA_DIR, COX_DATA_DIR, RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)