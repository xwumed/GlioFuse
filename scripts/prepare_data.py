import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
import logging
from neuroCombat.neuroCombat import neuroCombat
from sklearn.preprocessing import StandardScaler
 
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

# =======================================================================================
# 1. 导入统一配置
# =======================================================================================
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, CV_SPLITS_DIR, EXTERNAL_DATA_DIR, COX_DATA_DIR,
    N_SPLITS, RANDOM_STATE,
    PATIENT_ID_COLUMN, TIME_COLUMN, EVENT_COLUMN,
    ensure_directories_exist
)
from logger_manager import LoggerManager

logger = logging.getLogger(__name__)
COMBAT_PARAMS = {}

# =======================================================================================
# 2. 核心数据处理函数
# =======================================================================================

def mri_sequence_aware_dimensionality_reduction(features_df, target_dim=768, method='sequence_autoencoder'):
    """
    针对MRI多序列特征的无监督降维方案（精简版，仅保留每序列自编码器 DAE）
    
    Args:
        features_df: 特征数据框，包含4个序列的特征 (每个序列3072个特征)
        target_dim: 目标维度 (768)
        method: 降维方法
            - 'sequence_autoencoder': 每个序列 DAE 到 target_dim/n_sequences 后拼接
    
    Returns:
        reduced_features_df: 降维后的特征数据框
        reducer_info: 降维器信息
    """
    logger.info(f"  - MRI序列感知降维: {features_df.shape[1]} → {target_dim} 维")
    logger.info(f"  - 降维方法: {method}")
    
    # 假设特征按序列顺序排列: [T1: 0-3071, T2: 3072-6143, FLAIR: 6144-9215, T1CE: 9216-12287]
    sequence_size = 3072
    total_features = features_df.shape[1]
    n_sequences = total_features // sequence_size
    
    logger.info(f"  - 检测到 {n_sequences} 个MRI序列，每个序列 {sequence_size} 个特征")
    
    # 为每个序列分配的目标维度
    dim_per_sequence = target_dim // n_sequences  # 768 // 4 = 192
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    reducer_info = {
        'scaler': scaler,
        'method': method,
        'sequence_size': sequence_size,
        'n_sequences': n_sequences,
        'dim_per_sequence': dim_per_sequence,
        'feature_names': features_df.columns.tolist()
    }
    
    if method == 'sequence_autoencoder':
        # 方案: 每序列去噪自编码器 (DAE) 到 dim_per_sequence，最后拼接
        logger.info(f"  - 每序列 Denoising AutoEncoder 降维到 {dim_per_sequence} 维")

        # AE 训练超参（可根据需要微调）
        ae_hidden_dims = [1024, 384]
        ae_code_dim = dim_per_sequence
        ae_epochs = 80
        ae_batch_size = 64
        ae_lr = 1e-3
        ae_weight_decay = 1e-4
        ae_noise_std = 0.05
        ae_patience = 10

        class SeqAutoEncoder(nn.Module):
            def __init__(self, input_dim: int, code_dim: int, hidden_dims):
                super().__init__()
                h1, h2 = hidden_dims
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, h1), nn.ReLU(),
                    nn.Linear(h1, h2), nn.ReLU(),
                    nn.Linear(h2, code_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(code_dim, h2), nn.ReLU(),
                    nn.Linear(h2, h1), nn.ReLU(),
                    nn.Linear(h1, input_dim)
                )
            def forward(self, x):
                z = self.encoder(x)
                x_hat = self.decoder(z)
                return x_hat, z

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        sequence_encoders_state = []
        reduced_sequences = []

        for i in range(n_sequences):
            start_idx = i * sequence_size
            end_idx = start_idx + sequence_size
            X_seq = features_scaled[:, start_idx:end_idx].astype(np.float32)

            # 构建数据集
            tensor = torch.from_numpy(X_seq)
            dataset = TensorDataset(tensor)
            loader = DataLoader(dataset, batch_size=ae_batch_size, shuffle=True, num_workers=0, drop_last=False)

            # 初始化模型
            model = SeqAutoEncoder(input_dim=sequence_size, code_dim=ae_code_dim, hidden_dims=ae_hidden_dims).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=ae_lr, weight_decay=ae_weight_decay)
            best_loss = float('inf')
            epochs_no_improve = 0

            model.train()
            for epoch in range(1, ae_epochs + 1):
                running = 0.0
                for (batch_x,) in loader:
                    batch_x = batch_x.to(device)
                    # 去噪输入
                    noise = ae_noise_std * torch.randn_like(batch_x)
                    noisy_x = batch_x + noise
                    opt.zero_grad()
                    recon, _ = model(noisy_x)
                    loss = nn.MSELoss()(recon, batch_x)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                    running += loss.item() * batch_x.size(0)
                epoch_loss = running / len(dataset)
                if epoch % 10 == 0 or epoch == 1:
                    logger.info(f"    - 序列 {i+1} AE Epoch {epoch}/{ae_epochs}, Recon Loss: {epoch_loss:.6f}")
                # 简单早停
                if epoch_loss + 1e-6 < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    best_state = {k: v.cpu() for k, v in model.encoder.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= ae_patience:
                        logger.info(f"    - 序列 {i+1} 早停于 epoch {epoch} (best recon={best_loss:.6f})")
                        break

            # 保存 encoder 状态并提取编码
            sequence_encoders_state.append(best_state)
            with torch.no_grad():
                model.encoder.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                model.eval()
                codes = []
                for (batch_x,) in DataLoader(dataset, batch_size=ae_batch_size, shuffle=False):
                    batch_x = batch_x.to(device)
                    _, z = model(batch_x)
                    codes.append(z.cpu().numpy())
                seq_code = np.concatenate(codes, axis=0)
                reduced_sequences.append(seq_code)

        features_reduced = np.concatenate(reduced_sequences, axis=1)
        feature_names = []
        for i in range(n_sequences):
            for j in range(reduced_sequences[i].shape[1]):
                feature_names.append(f'Seq{i+1}_AE{j+1}')

        reducer_info['sequence_autoencoder'] = {
            'encoder_state_dicts': sequence_encoders_state,
            'hidden_dims': ae_hidden_dims,
            'code_dim': ae_code_dim,
            'input_dim': sequence_size,
            'hparams': {
                'epochs': ae_epochs,
                'batch_size': ae_batch_size,
                'lr': ae_lr,
                'weight_decay': ae_weight_decay,
                'noise_std': ae_noise_std,
                'patience': ae_patience,
            }
        }

        reduced_features_df = pd.DataFrame(features_reduced, columns=feature_names, index=features_df.index)
        logger.info(f"    - 最终降维结果: {features_df.shape[1]} → {reduced_features_df.shape[1]} 维")
        return reduced_features_df, reducer_info
    else:
        raise ValueError(f"不支持的降维方法: {method}")

def apply_mri_sequence_reduction(features_df, reducer_info):
    """
    应用已训练的MRI序列降维器到新数据
    """
    if reducer_info is None:
        return features_df
    
    # 标准化
    features_scaled = reducer_info['scaler'].transform(features_df)
    method = reducer_info['method']
    
    if method == 'sequence_pca' or method == 'sequence_ica' or method == 'hybrid_reduction':
        # 序列感知方法
        sequence_size = reducer_info['sequence_size']
        n_sequences = reducer_info['n_sequences']
        sequence_reducers = reducer_info['sequence_reducers']
        
        reduced_sequences = []
        
        for i in range(n_sequences):
            start_idx = i * sequence_size
            end_idx = start_idx + sequence_size
            
            sequence_data = features_scaled[:, start_idx:end_idx]
            
            if method == 'hybrid_reduction':
                # 先方差选择，再PCA
                variance_selector = sequence_reducers[i]['variance_selector']
                pca = sequence_reducers[i]['pca']
                
                sequence_selected = variance_selector.transform(sequence_data)
                if pca is not None:
                    sequence_reduced = pca.transform(sequence_selected)
                else:
                    sequence_reduced = sequence_selected
            else:
                # 直接应用PCA或ICA
                sequence_reduced = sequence_reducers[i].transform(sequence_data)
            
            reduced_sequences.append(sequence_reduced)
        
        features_reduced = np.concatenate(reduced_sequences, axis=1)
        
    elif method == 'global_pca':
        # 全局PCA
        features_reduced = reducer_info['global_pca'].transform(features_scaled)
        
    elif method == 'variance_pca':
        # 方差选择 + PCA
        top_indices = reducer_info['top_indices']
        features_selected = features_scaled[:, top_indices]
        features_reduced = reducer_info['pca'].transform(features_selected)
    elif method == 'sequence_autoencoder':
        # 还原并应用每序列的 encoder
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        seq_info = reducer_info['sequence_autoencoder']
        enc_states = seq_info['encoder_state_dicts']
        hidden_dims = seq_info['hidden_dims']
        code_dim = seq_info['code_dim']
        input_dim = seq_info['input_dim']

        class SeqAutoEncoder(nn.Module):
            def __init__(self, input_dim: int, code_dim: int, hidden_dims):
                super().__init__()
                h1, h2 = hidden_dims
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, h1), nn.ReLU(),
                    nn.Linear(h1, h2), nn.ReLU(),
                    nn.Linear(h2, code_dim)
                )
                self.decoder = nn.Identity()
            def forward(self, x):
                z = self.encoder(x)
                return z

        reduced_sequences = []
        for i in range(reducer_info['n_sequences']):
            start_idx = i * reducer_info['sequence_size']
            end_idx = start_idx + reducer_info['sequence_size']
            X_seq = features_scaled[:, start_idx:end_idx].astype(np.float32)
            model = SeqAutoEncoder(input_dim, code_dim, hidden_dims).to(device)
            model.encoder.load_state_dict({k: v.to(device) for k, v in enc_states[i].items()})
            model.eval()
            with torch.no_grad():
                tensor = torch.from_numpy(X_seq).to(device)
                z = model(tensor).cpu().numpy()
            reduced_sequences.append(z)
        features_reduced = np.concatenate(reduced_sequences, axis=1)
    
    # 重建特征名
    if method == 'sequence_pca':
        feature_names = []
        for i in range(reducer_info['n_sequences']):
            for j in range(reduced_sequences[i].shape[1]):
                feature_names.append(f'Seq{i+1}_PC{j+1}')
    elif method == 'sequence_ica':
        feature_names = []
        for i in range(reducer_info['n_sequences']):
            for j in range(reduced_sequences[i].shape[1]):
                feature_names.append(f'Seq{i+1}_IC{j+1}')
    elif method == 'hybrid_reduction':
        feature_names = []
        for i in range(reducer_info['n_sequences']):
            for j in range(reduced_sequences[i].shape[1]):
                feature_names.append(f'Seq{i+1}_Hybrid{j+1}')
    elif method == 'global_pca':
        feature_names = [f'Global_PC{i+1}' for i in range(features_reduced.shape[1])]
    elif method == 'variance_pca':
        feature_names = [f'VarPCA_PC{i+1}' for i in range(features_reduced.shape[1])]
    elif method == 'sequence_autoencoder':
        feature_names = []
        for i in range(reducer_info['n_sequences']):
            # 对应 apply 阶段 reduced_sequences 的列数
            # 由于不能直接访问 reduced_sequences 这里，根据 code_dim 填充名称
            for j in range(reducer_info['dim_per_sequence']):
                feature_names.append(f'Seq{i+1}_AE{j+1}')
    
    return pd.DataFrame(features_reduced, columns=feature_names, index=features_df.index)

def apply_trained_mri_reducer(test_mri_df, mri_reducer):
    """
    将训练好的MRI降维器应用到测试队列
    """
    logger.info(f"  - 将训练好的MRI降维器应用到 {test_mri_df.shape[0]} 个测试样本")
    
    # 基础预处理：添加source_cohort, 缺失值填充, 方差筛选
    meta_cols = [PATIENT_ID_COLUMN]
    if 'source_cohort' not in test_mri_df.columns:
        test_mri_df = test_mri_df.copy()
        test_mri_df['source_cohort'] = 'tcga'  # 假设测试队列是tcga
    
    meta_cols = [PATIENT_ID_COLUMN, 'source_cohort']
    meta_df = test_mri_df[meta_cols]
    features_df = test_mri_df.drop(columns=meta_cols)
    
    # 填充缺失值
    if features_df.isnull().sum().sum() > 0:
        features_df.fillna(0, inplace=True)
        logger.info(f"    - 填充了测试队列的缺失值")
    
    # 对齐训练时的特征列顺序，缺失列填0，多余列丢弃
    if mri_reducer is not None and 'feature_names' in mri_reducer:
        expected = mri_reducer['feature_names']
        features_df = features_df.reindex(columns=expected, fill_value=0)

    # 直接应用降维器
    if mri_reducer is not None:
        reduced_features_df = apply_mri_sequence_reduction(features_df, mri_reducer)
        logger.info(f"    - 测试队列MRI降维: {features_df.shape[1]} → {reduced_features_df.shape[1]} 维")
    else:
        reduced_features_df = features_df
        logger.info(f"    - 测试队列MRI保持原维度: {features_df.shape[1]} 维")
    
    # 合并meta信息
    final_df = pd.concat([meta_df.reset_index(drop=True), reduced_features_df.reset_index(drop=True)], axis=1)
    return final_df

def load_cohort_data(cohort_name):
    """根据队列名称加载 MRI, WSI, 和临床数据。"""
    logger.info(f"  - 正在加载队列: {cohort_name}...")
    try:
        mri_df = pd.read_csv(os.path.join(RAW_DATA_DIR, f"{cohort_name}_mri.csv"))
        wsi_df = pd.read_csv(os.path.join(RAW_DATA_DIR, f"{cohort_name}_wsi.csv"))
        cli_df = pd.read_csv(os.path.join(RAW_DATA_DIR, f"{cohort_name}_cli.csv"))
        
        # --- 【检查点】打印原始数据维度 ---
        logger.info(f"    - 原始 MRI 数据维度: {mri_df.shape}")
        logger.info(f"    - 原始 WSI 数据维度: {wsi_df.shape}")
        logger.info(f"    - 原始临床数据维度: {cli_df.shape}")
        
        mri_df['source_cohort'] = cohort_name
        wsi_df['source_cohort'] = cohort_name
        
        return mri_df, wsi_df, cli_df
    except FileNotFoundError as e:
        logger.error(f"  - ❌ 错误: 找不到队列 '{cohort_name}' 的数据文件。缺失文件: {e.filename}")
        exit()

def preprocess_and_select_features(df, modality_name, target_dim=None, reduction_method='sequence_pca'):
    """预处理特征数据：填充缺失值、移除低方差特征，并进行无监督降维。"""
    logger.info(f"  - 正在预处理 {modality_name} 特征...")
    
    meta_cols = [PATIENT_ID_COLUMN, 'source_cohort']
    meta_df = df[meta_cols]
    features_df = df.drop(columns=meta_cols)
    original_feature_count = features_df.shape[1]
    
    missing_vals = features_df.isnull().sum().sum()
    if missing_vals > 0:
        logger.info(f"    - 发现 {missing_vals} 个缺失值，用 0 填充。")
        features_df.fillna(0, inplace=True)

    # 保持特征全集一致以确保跨队列特征名对齐，不移除方差为0的特征
    features_df_selected = features_df.copy()
    
    # --- 【新增】MRI序列感知无监督降维 ---
    reducer_info = None
    if target_dim is not None and features_df_selected.shape[1] > target_dim:
        logger.info(f"    - 开始MRI序列感知无监督降维: {features_df_selected.shape[1]} → {target_dim}")
        
        if modality_name == "MRI":
            # 使用MRI序列感知降维
            features_df_selected, reducer_info = mri_sequence_aware_dimensionality_reduction(
                features_df_selected, 
                target_dim=target_dim, 
                method=reduction_method
            )
        else:
            # 非MRI模态不进行降维，保持原始特征
            logger.info(f"    - {modality_name} 不进行降维，保持原始特征维度 {features_df_selected.shape[1]}")
    
    # --- 【检查点】打印预处理后数据维度 ---
    logger.info(f"    - 处理后 {modality_name} 特征维度: ({features_df_selected.shape[0]}, {features_df_selected.shape[1]})")
        
    final_df = pd.concat([meta_df.reset_index(drop=True), features_df_selected.reset_index(drop=True)], axis=1)
    return final_df, reducer_info

def run_combat(df, modality_name):
    """对合并后的特征数据应用 ComBat 校正。"""
    logger.info(f"\n--- 正在对 {modality_name} 特征应用 ComBat 校正 ---")
    
    features = df.drop(columns=[PATIENT_ID_COLUMN, 'source_cohort'])
    covars = df[['source_cohort']]
    
    corrected_dict = neuroCombat(dat=features.T, covars=covars, batch_col='source_cohort')
    corrected_df = pd.DataFrame(corrected_dict['data'], index=features.T.index, columns=features.T.columns).T
    corrected_df[PATIENT_ID_COLUMN] = df[PATIENT_ID_COLUMN].values

    # 保存 ComBat 估计参数（不包含数据矩阵本身）以便复现
    try:
        params_only = copy.deepcopy(corrected_dict)
        if 'data' in params_only:
            params_only.pop('data')
        COMBAT_PARAMS[modality_name] = params_only
    except Exception as e:
        logger.warning(f"保存 ComBat 参数失败: {e}")
    
    logger.info(f"  - ✅ {modality_name} 特征 ComBat 校正完成。")
    # --- 【检查点】打印校正后数据维度和预览 ---
    logger.info(f"    - 校正后 {modality_name} 数据维度: {corrected_df.shape}")
    logger.info("    - 校正后数据预览 (前5行，前5列):\n%s", corrected_df.iloc[:5, :5])
    
    return corrected_df
    
# =======================================================================================
# 4. 可选：UMAP 可视化（集成自 0_explore_data_umap.py）
# =======================================================================================

# UMAP 配置（无需命令行）
DO_UMAP_VIS = False
# 控制是否执行 ComBat（便于对比前后差异）
DO_COMBAT = True
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42
UMAP_COLOR_PALETTE = 'viridis'
UMAP_FONT_SIZE_TITLE = 18
UMAP_FONT_SIZE_LABELS = 14
UMAP_OUTPUT_DIR = 'data_exploration_plots_umap'
os.makedirs(UMAP_OUTPUT_DIR, exist_ok=True)


def plot_umap_visualization(df: pd.DataFrame, modality_name: str, title_suffix: str):
    """对输入 DataFrame 进行 UMAP 降维并绘制散点图（按 cohort 上色）。
    需要 df 至少包含 `PATIENT_ID_COLUMN` 和 `source_cohort` 两列，其余列为特征。
    """
    logger.info(f"\n--- Generating UMAP visualization for {modality_name} {title_suffix} data ---")
    if df is None or df.empty:
        logger.warning(f"Input DataFrame is empty. Skipping UMAP for {modality_name} {title_suffix}.")
        return
    if 'source_cohort' not in df.columns:
        logger.error(f"'source_cohort' column not found for {modality_name} {title_suffix}. Skipping.")
        return

    features = df.drop(columns=[PATIENT_ID_COLUMN, 'source_cohort'])
    cohorts = df['source_cohort']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        random_state=UMAP_RANDOM_STATE
    )
    embedding = reducer.fit_transform(features_scaled)

    plot_df = pd.DataFrame(data=embedding, columns=['UMAP-1', 'UMAP-2'])
    plot_df['Cohort'] = cohorts.values

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 10))
    ax = sns.scatterplot(
        x='UMAP-1', y='UMAP-2',
        hue='Cohort',
        palette=UMAP_COLOR_PALETTE,
        data=plot_df,
        s=50,
        alpha=0.8,
        edgecolor='w',
        linewidth=0.5
    )
    plt.title(f'UMAP Visualization of {modality_name} Features ({title_suffix})', fontsize=UMAP_FONT_SIZE_TITLE)
    plt.xlabel('UMAP Component 1', fontsize=UMAP_FONT_SIZE_LABELS)
    plt.ylabel('UMAP Component 2', fontsize=UMAP_FONT_SIZE_LABELS)
    ax.legend(title='Cohort', fontsize='large', title_fontsize='x-large')

    save_path = os.path.join(UMAP_OUTPUT_DIR, f'umap_{modality_name}_{title_suffix.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  - UMAP plot saved to: {save_path}")

# =======================================================================================
# 3. 主程序
# =======================================================================================

if __name__ == '__main__':
    # 初始化日志
    from config import LOGS_DIR
    exp_name = 'prepare_data'
    log_dir = os.path.join(LOGS_DIR, exp_name)
    LoggerManager(experiment_name=exp_name, log_dir=log_dir)

    TRAIN_COHORTS = ['nanfang', 'huaqiao']
    TEST_COHORT = 'tcga'
    
    # 【配置】MRI降维参数
    MRI_TARGET_DIM = 768   # MRI降维目标维度 (4序列 × 192 = 768)
    
    # 【选择】MRI降维方法 - 推荐分序列降维
    MRI_REDUCTION_METHOD = 'sequence_autoencoder'  # 改为每序列 DAE，无监督且不受样本数限制

    logger.info("=== MRI多序列无监督降维 + ComBat 校正流程 ===")
    logger.info(f"【配置】MRI降维方法: {MRI_REDUCTION_METHOD}")
    logger.info(f"【配置】MRI目标维度: {MRI_TARGET_DIM} (4序列 × 192维/序列)")
    logger.info(f"【配置】WSI: 不进行降维，保持原始768维")
    logger.info(f"【策略】在训练队列上训练降维器，再应用到测试队列")
    ensure_directories_exist()

    logger.info("\n--- 步骤 1: 加载所有指定队列的数据 ---")
    all_cohorts = TRAIN_COHORTS + [TEST_COHORT]
    mri_dfs, wsi_dfs, cli_dfs = [], [], []
    for cohort in all_cohorts:
        mri_df, wsi_df, cli_df = load_cohort_data(cohort)
        mri_dfs.append(mri_df); wsi_dfs.append(wsi_df); cli_dfs.append(cli_df)
    
    # 分别处理训练队列和测试队列
    train_mri_dfs = [mri_dfs[i] for i, cohort in enumerate(all_cohorts) if cohort in TRAIN_COHORTS]
    train_wsi_dfs = [wsi_dfs[i] for i, cohort in enumerate(all_cohorts) if cohort in TRAIN_COHORTS]
    train_cli_dfs = [cli_dfs[i] for i, cohort in enumerate(all_cohorts) if cohort in TRAIN_COHORTS]
    
    test_mri_df = [mri_dfs[i] for i, cohort in enumerate(all_cohorts) if cohort == TEST_COHORT][0]
    test_wsi_df = [wsi_dfs[i] for i, cohort in enumerate(all_cohorts) if cohort == TEST_COHORT][0]
    test_cli_df = [cli_dfs[i] for i, cohort in enumerate(all_cohorts) if cohort == TEST_COHORT][0]
    
    # 合并训练队列数据
    train_mri_df = pd.concat(train_mri_dfs, ignore_index=True)
    train_wsi_df = pd.concat(train_wsi_dfs, ignore_index=True)
    train_cli_df = pd.concat(train_cli_dfs, ignore_index=True)

    logger.info("\n--- 步骤 2: 在训练队列上训练MRI降维器 ---")
    logger.info("【策略】: 只在训练队列上学习降维器，确保无数据泄漏")
    
    # 2.1 在训练队列上训练MRI降维器
    train_mri_processed, mri_reducer = preprocess_and_select_features(
        train_mri_df, "MRI", 
        target_dim=MRI_TARGET_DIM, 
        reduction_method=MRI_REDUCTION_METHOD
    )
    
    # 2.2 WSI不降维，只做基础预处理
    logger.info("  - WSI数据不进行降维，保持原始特征")
    train_wsi_processed, _ = preprocess_and_select_features(
        train_wsi_df, "WSI", 
        target_dim=None,  # 不降维
        reduction_method=None
    )

    logger.info("\n--- 步骤 3: 将训练好的降维器应用到测试队列 ---")
    logger.info("【优势】: 避免数据泄漏，测试队列使用训练队列学到的降维模式")
    
    # 3.1 对测试队列应用MRI降维器
    test_mri_processed = apply_trained_mri_reducer(test_mri_df, mri_reducer)
    
    # 3.2 测试队列WSI基础预处理
    test_wsi_processed, _ = preprocess_and_select_features(
        test_wsi_df, "WSI", 
        target_dim=None,  # 不降维
        reduction_method=None
    )
    
    # 3.3 合并所有数据用于ComBat
    all_mri_df_processed = pd.concat([train_mri_processed, test_mri_processed], ignore_index=True)
    all_wsi_df_processed = pd.concat([train_wsi_processed, test_wsi_processed], ignore_index=True)
    all_cli_df = pd.concat([train_cli_df, test_cli_df], ignore_index=True)

    # 可选：在 ComBat 前进行 UMAP 可视化
    if DO_UMAP_VIS:
        try:
            plot_umap_visualization(all_mri_df_processed, 'MRI', 'Before ComBat')
            plot_umap_visualization(all_wsi_df_processed, 'WSI', 'Before ComBat')
        except Exception as e:
            logger.warning(f"UMAP pre-ComBat visualization failed: {e}")

    logger.info("\n--- 步骤 4: 在降维后的特征空间进行 ComBat 校正 ---")
    logger.info("【优势】: 降维后的特征空间更稳定，ComBat校正效果更好")
    
    if DO_COMBAT and len(all_mri_df_processed['source_cohort'].unique()) > 1:
        all_mri_combat = run_combat(all_mri_df_processed, f"MRI({MRI_REDUCTION_METHOD})")
        all_wsi_combat = run_combat(all_wsi_df_processed, "WSI(原始)")
    else:
        if not DO_COMBAT:
            logger.info("\n--- 已关闭 ComBat：将直接使用降维后的特征 ---")
        else:
            logger.info("\n--- 检测到只有一个数据队列，跳过 ComBat 校正 ---")
        all_mri_combat = all_mri_df_processed.drop(columns=['source_cohort'])
        all_wsi_combat = all_wsi_df_processed.drop(columns=['source_cohort'])

    # 可选：在 ComBat 后进行 UMAP 可视化（需补回 cohort 信息）
    if DO_UMAP_VIS:
        try:
            # 从 pre-ComBat 数据映射 cohort
            mri_cohort_map = dict(zip(all_mri_df_processed[PATIENT_ID_COLUMN], all_mri_df_processed['source_cohort']))
            wsi_cohort_map = dict(zip(all_wsi_df_processed[PATIENT_ID_COLUMN], all_wsi_df_processed['source_cohort']))

            mri_combat_vis = all_mri_combat.copy()
            wsi_combat_vis = all_wsi_combat.copy()
            mri_combat_vis['source_cohort'] = mri_combat_vis[PATIENT_ID_COLUMN].map(mri_cohort_map)
            wsi_combat_vis['source_cohort'] = wsi_combat_vis[PATIENT_ID_COLUMN].map(wsi_cohort_map)

            plot_umap_visualization(mri_combat_vis, 'MRI', 'After ComBat')
            plot_umap_visualization(wsi_combat_vis, 'WSI', 'After ComBat')
        except Exception as e:
            logger.warning(f"UMAP post-ComBat visualization failed: {e}")

    logger.info("\n--- 步骤 5: 拆分校正后的数据 ---")
    train_ids = train_cli_df[PATIENT_ID_COLUMN].tolist()
    test_ids = test_cli_df[PATIENT_ID_COLUMN].tolist()
    
    train_mri_corrected = all_mri_combat[all_mri_combat[PATIENT_ID_COLUMN].isin(train_ids)]
    train_wsi_corrected = all_wsi_combat[all_wsi_combat[PATIENT_ID_COLUMN].isin(train_ids)]
    train_labels = all_cli_df[all_cli_df[PATIENT_ID_COLUMN].isin(train_ids)]
    
    test_mri_corrected = all_mri_combat[all_mri_combat[PATIENT_ID_COLUMN].isin(test_ids)]
    test_wsi_corrected = all_wsi_combat[all_wsi_combat[PATIENT_ID_COLUMN].isin(test_ids)]
    test_labels = all_cli_df[all_cli_df[PATIENT_ID_COLUMN].isin(test_ids)]
    
    train_dev_df = pd.merge(pd.merge(train_mri_corrected, train_wsi_corrected, on=PATIENT_ID_COLUMN), train_labels, on=PATIENT_ID_COLUMN)
    external_test_df = pd.merge(pd.merge(test_mri_corrected, test_wsi_corrected, on=PATIENT_ID_COLUMN), test_labels, on=PATIENT_ID_COLUMN)
    
    # --- 【检查点】打印最终拆分后数据维度 ---
    logger.info("  - ✅ 数据拆分完成。")
    logger.info(f"    - 最终训练/验证集维度: {train_dev_df.shape}")
    logger.info(f"    - 最终外部测试集维度: {external_test_df.shape}")

    mri_cols = [col for col in train_mri_corrected.columns if col != PATIENT_ID_COLUMN]
    wsi_cols = [col for col in train_wsi_corrected.columns if col != PATIENT_ID_COLUMN]
    
    # 【新增】保存降维器信息
    logger.info("\n--- 步骤 5: 保存降维器信息 ---")
    reducers_info = {
        'mri_reducer': mri_reducer,
        'mri_reduction_method': MRI_REDUCTION_METHOD,
        'mri_target_dim': MRI_TARGET_DIM,
        'mri_final_dim': len(mri_cols),
        'wsi_final_dim': len(wsi_cols),
        'train_cohorts': TRAIN_COHORTS,
        'test_cohort': TEST_COHORT,
        'train_case_ids': train_ids,
        'random_state': RANDOM_STATE,
        'umap_enabled': DO_UMAP_VIS,
        'combat_enabled': DO_COMBAT,
    }
    
    reducer_save_path = os.path.join(PROCESSED_DATA_DIR, 'dimensionality_reducers.pkl')
    with open(reducer_save_path, 'wb') as f:
        # 若开启了 ComBat，额外保存参数
        if DO_COMBAT and COMBAT_PARAMS:
            reducers_info['combat_params'] = COMBAT_PARAMS
        pickle.dump(reducers_info, f)
    logger.info(f"✅ 降维器信息已保存至: {reducer_save_path}")
    logger.info(f"    - MRI({MRI_REDUCTION_METHOD}): 4序列×3072特征 → {MRI_TARGET_DIM} → {len(mri_cols)} (最终维度)")
    logger.info(f"    - WSI: 保持原维度 {len(wsi_cols)} 维")
    
    logger.info(f"\n--- 步骤 6: 为深度学习模型创建 {N_SPLITS}-折交叉验证文件 ---")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for i, (train_idx, val_idx) in enumerate(skf.split(train_dev_df, train_dev_df[EVENT_COLUMN])):
        cv_train_df, cv_val_df = train_dev_df.iloc[train_idx], train_dev_df.iloc[val_idx]
        split_file_path = os.path.join(CV_SPLITS_DIR, f'split_{i}_data.pkl')
        data_dict = {'train': {'x_path': cv_train_df[wsi_cols].values.tolist(), 'x_rad': cv_train_df[mri_cols].values.tolist(), 'e': cv_train_df[EVENT_COLUMN].tolist(), 't': cv_train_df[TIME_COLUMN].tolist(), 'g': [0] * len(cv_train_df)}, 'test': {'x_path': cv_val_df[wsi_cols].values.tolist(), 'x_rad': cv_val_df[mri_cols].values.tolist(), 'e': cv_val_df[EVENT_COLUMN].tolist(), 't': cv_val_df[TIME_COLUMN].tolist(), 'g': [0] * len(cv_val_df)}}
        with open(split_file_path, 'wb') as f: pickle.dump(data_dict, f)
        logger.info(f"✅ {N_SPLITS} 个 .pkl 分割文件已成功生成在 '{CV_SPLITS_DIR}'")

    logger.info("\n--- 步骤 7: 保存处理后的外部测试集 ---")
    external_data_dict = {'test': {
        'x_path': external_test_df[wsi_cols].values.tolist(),
        'x_rad': external_test_df[mri_cols].values.tolist(),
        'e': external_test_df[EVENT_COLUMN].tolist(),
        't': external_test_df[TIME_COLUMN].tolist(),
        'g': [0] * len(external_test_df),
        'ids': external_test_df[PATIENT_ID_COLUMN].tolist()
    }}
    external_save_path = os.path.join(EXTERNAL_DATA_DIR, 'external_test_data.pkl')
    with open(external_save_path, 'wb') as f: pickle.dump(external_data_dict, f)
    logger.info(f"✅ 独立的外部测试集 .pkl 文件已成功保存至 '{external_save_path}'")
    
    logger.info("\n" + "="*80)
    logger.info("🎉 MRI多序列无监督降维流程全部完成！")
    logger.info("="*80)
    logger.info(f"  - 训练/验证队列: {', '.join(TRAIN_COHORTS)}")
    logger.info(f"  - 独立测试队列: {TEST_COHORT}")
    logger.info(f"  - 最终训练/验证集样本数: {len(train_dev_df)}")
    logger.info(f"  - 最终外部测试集样本数: {len(external_test_df)}")
    logger.info(f"  - MRI降维策略: {MRI_REDUCTION_METHOD}")
    logger.info(f"  - MRI特征维度: 12,288 → {len(mri_cols)}")
    logger.info(f"  - WSI特征维度: 768 → {len(wsi_cols)}")
    logger.info(f"  - 所有数据均已通过【序列感知降维 + ComBat】处理！")
    logger.info("\n现在您可以安全地运行所有模型训练脚本了。")
    logger.info("="*80)