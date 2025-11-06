import os
import pickle
import pandas as pd
import time
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
from typing import Optional

# 导入核心模块和配置
from options import get_base_parser, parse_gpuids, print_options, normalize_mode, to_simplified_mode
from config import BASELINE_EXPERIMENTS, OPTIMIZED_HYPERPARAMS, EXTERNAL_DATA_DIR, CV_SPLITS_DIR, N_SPLITS, RESULTS_DIR
from logger_manager import LoggerManager
from data_loaders import MriWsiDataset
from networks import define_net
from train_test import test
from cv_manager import CrossValidationManager

# =======================================================================================
# 核心交叉验证函数 (从 core_runner.py 整合而来)
# =======================================================================================
def run_cv_experiment(opt, parser=None, trial=None):
    """
    运行一次完整的交叉验证实验。
    parser 只是可选的，用于打印配置。
    """
    if not trial:
        print("\n" + "="*80)
        print(f"🚀 开始运行实验: {opt.exp_name} | 模型: {opt.model_name}")
        print("="*80)
        if parser: # 只有当 parser 被传入时才打印
            print_options(opt, parser)
    
    # --- 环境设置 ---
    torch.manual_seed(2019); random.seed(2019); np.random.seed(2019)
    if opt.gpu_ids and torch.cuda.is_available(): torch.cuda.manual_seed_all(2019)
    cudnn.deterministic = True
    device = torch.device(f'cuda:{opt.gpu_ids[0]}') if opt.gpu_ids and torch.cuda.is_available() else torch.device('cpu')
    
    # --- 针对融合模型的默认正则范围：优先只约束融合层/分类头 ---
    try:
        fusion_modes = ['simple_fusion', 'multiscale_fusion', 'coattn', 'bilinear_fusion']
        if getattr(opt, 'mode', None) in fusion_modes:
            current_reg = getattr(opt, 'reg_type', 'rad')
            if current_reg in ['rad', 'path']:
                opt.reg_type = 'fusion'
    except Exception:
        pass

    # --- 运行交叉验证 ---
    cv_manager = CrossValidationManager(opt, device)
    # 【核心修正】确保 trial 参数被正确传递
    results = cv_manager.run_training_cv(trial=trial)
    
    # --- 打印和保存结果 (仅在非调优模式下) ---
    if not trial:
        cv_manager.print_summary(results)
        cv_manager.save_results(results)
    
    return results

# =======================================================================================
# 辅助函数 (保持不变)
# =======================================================================================
def evaluate_folds_on_external(opt_exp, external_test_data_raw, logger):
    """在外部测试集上评估K折交叉验证训练出的所有模型。
    
    提供两种评估模式：
    1. 传统模式：每个折分别评估，然后计算均值和标准差
    2. 聚合模式：聚合所有折的预测结果，基于平均预测计算单一指标值
    """
    logger.info(f"--- Starting evaluation on external test set for model: {opt_exp.model_name} ---")
    
    # 存储每个折的预测结果用于聚合
    all_fold_predictions = []
    all_fold_survtimes = []
    all_fold_censors = []
    all_fold_ids = []
    
    # 存储每个折的单独评估结果（用于传统模式）
    cindex_results = []
    iauc_results = []
    ibrier_results = []
    timepoint_auc_results = {}
    
    per_fold_rows = []
    for k in range(N_SPLITS):
        # 找到对应折的数据以获取scaler
        split_data_path = os.path.join(CV_SPLITS_DIR, f'split_{k}_data.pkl')
        try:
            with open(split_data_path, 'rb') as f:
                fold_data = pickle.load(f)
        except FileNotFoundError:
            logger.warning(f"Could not find {split_data_path}, skipping fold {k}.")
            continue
            
        # 创建用于获取scaler的训练集
        train_dataset_for_scaler = MriWsiDataset(opt_exp, fold_data, split='train')
        scalers = train_dataset_for_scaler.get_scalers()
        
        # 应用scaler到外部测试集
        external_dataset = MriWsiDataset(opt_exp, {'test': external_test_data_raw}, 'test', scalers)
        
        # 加载模型权重
        model_weights_path = os.path.join(opt_exp.checkpoints_dir, opt_exp.exp_name, opt_exp.model_name, f'split_{k}_best_weights.pt')
        if not os.path.exists(model_weights_path):
            logger.warning(f"Could not find model weights {model_weights_path}, skipping fold {k}.")
            continue

        device = torch.device(f'cuda:{opt_exp.gpu_ids[0]}') if opt_exp.gpu_ids and torch.cuda.is_available() else torch.device('cpu')
        model = define_net(opt_exp, k)
        model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
        model.to(device)
        
        # 在外部测试集上进行测试
        _, cindex_test, _, _, iauc_test, ibrier_test, timepoint_aucs_test, raw_results = test(opt_exp, model, external_dataset, device)
        
        # 存储原始预测结果用于聚合
        risk_pred, survtime, censor = raw_results
        all_fold_predictions.append(risk_pred)
        all_fold_survtimes.append(survtime)
        all_fold_censors.append(censor)
        # IDs（如果 external_test_data_raw 提供了 'ids' 字段，则使用，否则占位）
        if 'ids' in external_test_data_raw:
            all_fold_ids.append(external_test_data_raw['ids'])
        
        # 存储单独评估结果
        cindex_results.append(cindex_test)
        iauc_results.append(iauc_test)
        ibrier_results.append(ibrier_test)
        
        # 收集四个临床时间点的AUC结果
        if k == 0:  # 初始化时间点AUC结果字典
            timepoint_auc_results = {timepoint: [] for timepoint in timepoint_aucs_test.keys()}
        for timepoint, auc_value in timepoint_aucs_test.items():
            timepoint_auc_results[timepoint].append(auc_value)
        
        logger.info(f"  - Fold {k+1} external test C-Index: {cindex_test:.4f}, I-AUC: {iauc_test:.4f}, I-Brier: {ibrier_test:.4f}")
        logger.info(f"    Time-dependent AUCs: 1-year: {timepoint_aucs_test['1-year']:.4f}, 2-year: {timepoint_aucs_test['2-year']:.4f}, 3-year: {timepoint_aucs_test['3-year']:.4f}, 5-year: {timepoint_aucs_test['5-year']:.4f}")

        per_fold_rows.append({
            'fold': k + 1,
            'test_cindex': float(cindex_test),
            'test_iauc': float(iauc_test),
            'test_ibrier': float(ibrier_test),
            'auc_1-year': float(timepoint_aucs_test.get('1-year', float('nan'))),
            'auc_2-year': float(timepoint_aucs_test.get('2-year', float('nan'))),
            'auc_3-year': float(timepoint_aucs_test.get('3-year', float('nan'))),
            'auc_5-year': float(timepoint_aucs_test.get('5-year', float('nan'))),
        })

    if not cindex_results:
        logger.warning("Failed to evaluate any folds on the external test set.")
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), {}, {}
    
    # === 传统模式：计算每个折结果的均值和标准差 ===
    mean_cindex = np.mean(cindex_results)
    std_cindex = np.std(cindex_results)
    mean_iauc = np.mean(iauc_results)
    std_iauc = np.std(iauc_results)
    mean_ibrier = np.mean(ibrier_results)
    std_ibrier = np.std(ibrier_results)
    
    # 计算四个临床时间点AUC的均值和标准差
    timepoint_auc_stats = {}
    for timepoint in timepoint_auc_results.keys():
        timepoint_auc_stats[timepoint] = {
            'mean': np.mean(timepoint_auc_results[timepoint]),
            'std': np.std(timepoint_auc_results[timepoint])
        }
    
    # === 聚合模式：基于平均预测计算单一指标值 ===
    from utils import CIndex_lifeline, integrated_brier_score, integrated_auc, clinical_timepoints_auc
    
    # 计算平均预测分数（五折的平均）
    avg_predictions = np.mean(all_fold_predictions, axis=0)
    
    # 使用第一个折的生存时间和删失状态（所有折应该相同）
    final_survtime = all_fold_survtimes[0]
    final_censor = all_fold_censors[0]
    
    # 基于平均预测计算聚合指标
    aggregated_cindex = CIndex_lifeline(avg_predictions, final_censor, final_survtime)
    aggregated_iauc = integrated_auc(avg_predictions, final_censor, final_survtime)
    aggregated_ibrier = integrated_brier_score(avg_predictions, final_censor, final_survtime)
    aggregated_timepoint_aucs = clinical_timepoints_auc(avg_predictions, final_censor, final_survtime)
    
    # 获取患者ID信息：优先使用外部测试数据中的 ids 字段
    if 'ids' in external_test_data_raw:
        patient_ids = external_test_data_raw['ids']
    else:
        patient_ids = [f'Patient_{i}' for i in range(len(avg_predictions))]
    
    # 保存聚合预测分数用于后续分析
    aggregated_predictions_data = {
        'patient_ids': patient_ids,
        'predictions': avg_predictions,
        'survtime': final_survtime,
        'censor': final_censor,
        'model_name': opt_exp.model_name,
        'exp_name': opt_exp.exp_name
    }
    
    # 创建保存目录
    predictions_save_dir = os.path.join(opt_exp.checkpoints_dir, opt_exp.exp_name, 'aggregated_predictions')
    os.makedirs(predictions_save_dir, exist_ok=True)
    
    # 保存为pickle格式（用于程序加载）
    predictions_save_path_pkl = os.path.join(predictions_save_dir, f'{opt_exp.model_name}_aggregated_predictions.pkl')
    with open(predictions_save_path_pkl, 'wb') as f:
        pickle.dump(aggregated_predictions_data, f)
    
    # 保存为CSV格式（用户友好格式）
    predictions_save_path_csv = os.path.join(predictions_save_dir, f'{opt_exp.model_name}_aggregated_predictions.csv')
    predictions_df = pd.DataFrame({
        'case_id': patient_ids,
        'aggregated_prediction': avg_predictions,
        'survival_time': final_survtime,
        'censor_status': final_censor,
        'model_name': opt_exp.model_name,
        'experiment_name': opt_exp.exp_name
    })
    predictions_df.to_csv(predictions_save_path_csv, index=False, float_format='%.6f')
    
    logger.info(f"Aggregated predictions saved to:")
    logger.info(f"  - Pickle format: {predictions_save_path_pkl}")
    logger.info(f"  - CSV format: {predictions_save_path_csv}")
    
    # 另存每折外部测试结果到模型目录
    try:
        import pandas as _pd
        model_dir = os.path.join(opt_exp.checkpoints_dir, opt_exp.exp_name, opt_exp.model_name)
        os.makedirs(model_dir, exist_ok=True)
        _pd.DataFrame(per_fold_rows).to_csv(os.path.join(model_dir, 'external_results.csv'), index=False, float_format='%.6f')
    except Exception as e:
        logger.warning(f"Failed to save per-fold external results CSV: {e}")

    # 创建聚合结果字典
    aggregated_results = {
        'cindex': aggregated_cindex,
        'iauc': aggregated_iauc,
        'ibrier': aggregated_ibrier,
        'timepoint_aucs': aggregated_timepoint_aucs,
        'predictions_path_pkl': predictions_save_path_pkl,  # pickle文件路径
        'predictions_path_csv': predictions_save_path_csv   # CSV文件路径
    }
    
    logger.info(f"--- Traditional mode (fold-wise average): C-Index {mean_cindex:.4f} ± {std_cindex:.4f}, I-AUC {mean_iauc:.4f} ± {std_iauc:.4f}, I-Brier {mean_ibrier:.4f} ± {std_ibrier:.4f} ---")
    logger.info(f"--- Aggregated mode (ensemble prediction): C-Index {aggregated_cindex:.4f}, I-AUC {aggregated_iauc:.4f}, I-Brier {aggregated_ibrier:.4f} ---")
    logger.info(f"--- Traditional time-dependent AUCs:")
    for timepoint, stats in timepoint_auc_stats.items():
        logger.info(f"    {timepoint}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    logger.info(f"--- Aggregated time-dependent AUCs:")
    for timepoint, auc_value in aggregated_timepoint_aucs.items():
        logger.info(f"    {timepoint}: {auc_value:.4f}")
    logger.info("---")
    
    return mean_cindex, std_cindex, mean_iauc, std_iauc, mean_ibrier, std_ibrier, timepoint_auc_stats, aggregated_results


# =======================================================================================
# 新增：后融合（late_fusion）评估工具
# =======================================================================================
def _build_opt_for_model(args, model_mode, base_parser):
    """根据 args 构建用于加载指定模型的 opt（不一定训练）。"""
    from options import parse_gpuids, normalize_mode, to_simplified_mode
    from config import OPTIMIZED_HYPERPARAMS
    opt = base_parser.parse_args([])
    opt.exp_name = args.exp_name if args.exp_name != 'fusion_experiment' else ('optimized_evaluation' if args.use_optimized else 'baseline_evaluation')
    internal_mode = normalize_mode(model_mode)
    opt.mode = internal_mode
    # 继承与主运行一致的基础参数
    for k, v in vars(args).items():
        setattr(opt, k, v)
    # 覆盖优化超参（若选择使用）
    if getattr(args, 'use_optimized', False) and internal_mode in OPTIMIZED_HYPERPARAMS:
        for k, v in OPTIMIZED_HYPERPARAMS[internal_mode].items():
            setattr(opt, k, v)
    # 一致的模型命名策略（不追加 _optimized 后缀，统一目录命名）
    opt.model_name = to_simplified_mode(internal_mode)
    opt = parse_gpuids(opt)
    return opt


def _weights_path_for(opt, fold_idx):
    import os
    return os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, f'split_{fold_idx}_best_weights.pt')


def _ensure_unimodal_models_trained(args, base_parser, logger):
    """若缺少后融合所需的单模态权重，则自动训练对应模型。"""
    from os.path import exists
    # 构建两个单模态配置
    opt_rad = _build_opt_for_model(args, 'rad_only', base_parser)
    opt_path = _build_opt_for_model(args, 'path_only', base_parser)
    need_train_rad = not exists(_weights_path_for(opt_rad, 0))
    need_train_path = not exists(_weights_path_for(opt_path, 0))
    if not (need_train_rad or need_train_path):
        return opt_rad, opt_path
    logger.info("Late fusion requires unimodal weights. Missing detected -> training needed.")
    # 训练缺失者
    if need_train_rad:
        logger.info("Training missing unimodal model: rad_only ...")
        run_cv_experiment(opt=opt_rad, parser=base_parser, trial=None)
    if need_train_path:
        logger.info("Training missing unimodal model: path_only ...")
        run_cv_experiment(opt=opt_path, parser=base_parser, trial=None)
    return opt_rad, opt_path


def evaluate_late_fusion_on_cv(args, base_parser, logger):
    """在K折CV上评估后融合（rad_only + path_only 风险平均）。"""
    import os
    import numpy as np
    import torch
    from config import CV_SPLITS_DIR, N_SPLITS
    from data_loaders import MriWsiDataset
    from networks import define_net
    from utils import CIndex_lifeline, integrated_auc, integrated_brier_score

    # 确保单模态权重可用
    opt_rad, opt_path = _ensure_unimodal_models_trained(args, base_parser, logger)

    device = torch.device(f'cuda:{opt_rad.gpu_ids[0]}') if opt_rad.gpu_ids and torch.cuda.is_available() else torch.device('cpu')

    train_cis, val_cis = [], []
    train_iaucs, val_iaucs = [], []
    train_ibriers, val_ibriers = [], []
    fold_results = []

    for k in range(N_SPLITS):
        # 加载当前折数据并构建scaler
        import pickle
        split_data_path = os.path.join(CV_SPLITS_DIR, f'split_{k}_data.pkl')
        try:
            with open(split_data_path, 'rb') as f:
                fold_data = pickle.load(f)
        except FileNotFoundError:
            logger.warning(f"Could not find {split_data_path}, skipping fold {k}.")
            continue

        train_dataset_for_scaler = MriWsiDataset(opt_rad, fold_data, split='train')
        scalers = train_dataset_for_scaler.get_scalers()
        val_dataset = MriWsiDataset(opt_rad, fold_data, split='test', scalers=scalers)
        tr_dataset = MriWsiDataset(opt_rad, fold_data, split='train', scalers=scalers)

        # 加载两个单模态模型
        m_rad = define_net(opt_rad, k)
        m_path = define_net(opt_path, k)
        rad_w = _weights_path_for(opt_rad, k)
        path_w = _weights_path_for(opt_path, k)
        if not (os.path.exists(rad_w) and os.path.exists(path_w)):
            logger.warning(f"Missing weights for fold {k}: rad_only -> {rad_w}, path_only -> {path_w}. Skipping fold.")
            continue
        m_rad.load_state_dict(torch.load(rad_w, map_location='cpu'))
        m_path.load_state_dict(torch.load(path_w, map_location='cpu'))
        m_rad.to(device); m_path.to(device)

        # 在验证集上评估并融合
        _, _, _, _, _, _, _, raw_rad_val = test(opt_rad, m_rad, val_dataset, device)
        _, _, _, _, _, _, _, raw_path_val = test(opt_path, m_path, val_dataset, device)
        pred_val = (np.array(raw_rad_val[0]) + np.array(raw_path_val[0])) / 2.0
        surv_val = np.array(raw_rad_val[1]); cens_val = np.array(raw_rad_val[2])
        ci_val = CIndex_lifeline(pred_val, cens_val, surv_val)
        iauc_val = integrated_auc(pred_val, cens_val, surv_val)
        ibrier_val = integrated_brier_score(pred_val, cens_val, surv_val)

        # 在训练集上评估并融合
        _, _, _, _, _, _, _, raw_rad_tr = test(opt_rad, m_rad, tr_dataset, device)
        _, _, _, _, _, _, _, raw_path_tr = test(opt_path, m_path, tr_dataset, device)
        pred_tr = (np.array(raw_rad_tr[0]) + np.array(raw_path_tr[0])) / 2.0
        surv_tr = np.array(raw_rad_tr[1]); cens_tr = np.array(raw_rad_tr[2])
        ci_tr = CIndex_lifeline(pred_tr, cens_tr, surv_tr)
        iauc_tr = integrated_auc(pred_tr, cens_tr, surv_tr)
        ibrier_tr = integrated_brier_score(pred_tr, cens_tr, surv_tr)

        train_cis.append(ci_tr); val_cis.append(ci_val)
        train_iaucs.append(iauc_tr); val_iaucs.append(iauc_val)
        train_ibriers.append(ibrier_tr); val_ibriers.append(ibrier_val)
        fold_results.append({'fold': k + 1, 'train_cindex': ci_tr, 'val_cindex': ci_val, 'train_iauc': iauc_tr, 'val_iauc': iauc_val, 'train_ibrier': ibrier_tr, 'val_ibrier': ibrier_val})

        logger.info(f"  - [LateFusion] Fold {k+1}: Train CI={ci_tr:.4f}, Val CI={ci_val:.4f}, Val I-AUC={iauc_val:.4f}, Val I-Brier={ibrier_val:.4f}")

    if not val_cis:
        logger.warning("Late fusion CV evaluation failed: no valid folds.")
        return {}

    results = {
        'mean_train_cindex': float(np.mean(train_cis)),
        'std_train_cindex': float(np.std(train_cis)),
        'mean_val_cindex': float(np.mean(val_cis)),
        'std_val_cindex': float(np.std(val_cis)),
        'mean_train_iauc': float(np.mean(train_iaucs)),
        'std_train_iauc': float(np.std(train_iaucs)),
        'mean_val_iauc': float(np.mean(val_iaucs)),
        'std_val_iauc': float(np.std(val_iaucs)),
        'mean_train_ibrier': float(np.mean(train_ibriers)),
        'std_train_ibrier': float(np.std(train_ibriers)),
        'mean_val_ibrier': float(np.mean(val_ibriers)),
        'std_val_ibrier': float(np.std(val_ibriers)),
        'fold_results': fold_results,
    }
    # 保存每折 CV 结果（训练/验证）
    try:
        import pandas as _pd
        model_dir = os.path.join(opt_rad.checkpoints_dir, opt_rad.exp_name, 'LateFusionNet')
        os.makedirs(model_dir, exist_ok=True)
        _pd.DataFrame(fold_results).to_csv(os.path.join(model_dir, 'cv_results.csv'), index=False, float_format='%.6f')
    except Exception as e:
        logger.warning(f"Failed to save LateFusionNet CV results CSV: {e}")
    return results


def evaluate_late_fusion_on_external(args, external_test_data_raw, base_parser, logger):
    """在外部测试集上评估后融合（rad_only + path_only 风险平均）。"""
    import os
    import numpy as np
    import torch
    from config import CV_SPLITS_DIR, N_SPLITS
    from data_loaders import MriWsiDataset
    from networks import define_net
    from utils import CIndex_lifeline, integrated_auc, integrated_brier_score, clinical_timepoints_auc
    import pickle

    # 确保单模态权重可用
    opt_rad, opt_path = _ensure_unimodal_models_trained(args, base_parser, logger)

    device = torch.device(f'cuda:{opt_rad.gpu_ids[0]}') if opt_rad.gpu_ids and torch.cuda.is_available() else torch.device('cpu')

    all_fold_predictions = []
    all_fold_survtimes = []
    all_fold_censors = []

    cindex_results = []
    iauc_results = []
    ibrier_results = []
    timepoint_auc_results = {}

    per_fold_rows = []
    for k in range(N_SPLITS):
        split_data_path = os.path.join(CV_SPLITS_DIR, f'split_{k}_data.pkl')
        try:
            with open(split_data_path, 'rb') as f:
                fold_data = pickle.load(f)
        except FileNotFoundError:
            logger.warning(f"Could not find {split_data_path}, skipping fold {k}.")
            continue

        # 拿到训练scaler并应用到外部集
        train_dataset_for_scaler = MriWsiDataset(opt_rad, fold_data, split='train')
        scalers = train_dataset_for_scaler.get_scalers()
        external_dataset = MriWsiDataset(opt_rad, {'test': external_test_data_raw}, 'test', scalers)

        # 加载两个模型
        m_rad = define_net(opt_rad, k)
        m_path = define_net(opt_path, k)
        rad_w = _weights_path_for(opt_rad, k)
        path_w = _weights_path_for(opt_path, k)
        if not (os.path.exists(rad_w) and os.path.exists(path_w)):
            logger.warning(f"Missing weights for fold {k}: rad_only -> {rad_w}, path_only -> {path_w}. Skipping fold.")
            continue
        m_rad.load_state_dict(torch.load(rad_w, map_location='cpu'))
        m_path.load_state_dict(torch.load(path_w, map_location='cpu'))
        m_rad.to(device); m_path.to(device)

        # 分别预测并后融合
        _, _, _, _, _, _, timepoint_aucs_rad, raw_rad = test(opt_rad, m_rad, external_dataset, device)
        _, _, _, _, _, _, timepoint_aucs_path, raw_path = test(opt_path, m_path, external_dataset, device)
        pred = (np.array(raw_rad[0]) + np.array(raw_path[0])) / 2.0
        surv = np.array(raw_rad[1]); cens = np.array(raw_rad[2])

        cindex_test = CIndex_lifeline(pred, cens, surv)
        iauc_test = integrated_auc(pred, cens, surv)
        ibrier_test = integrated_brier_score(pred, cens, surv)
        t_auc = clinical_timepoints_auc(pred, cens, surv)

        all_fold_predictions.append(pred)
        all_fold_survtimes.append(surv)
        all_fold_censors.append(cens)

        cindex_results.append(cindex_test)
        iauc_results.append(iauc_test)
        ibrier_results.append(ibrier_test)

        if not timepoint_auc_results:
            timepoint_auc_results = {tp: [] for tp in t_auc.keys()}
        for tp, auc_val in t_auc.items():
            timepoint_auc_results[tp].append(auc_val)

        logger.info(f"  - [LateFusion] Fold {k+1} external: C-Index={cindex_test:.4f}, I-AUC={iauc_test:.4f}, I-Brier={ibrier_test:.4f}")
        per_fold_rows.append({
            'fold': k + 1,
            'test_cindex': float(cindex_test),
            'test_iauc': float(iauc_test),
            'test_ibrier': float(ibrier_test),
            'auc_1-year': float(t_auc.get('1-year', float('nan'))),
            'auc_2-year': float(t_auc.get('2-year', float('nan'))),
            'auc_3-year': float(t_auc.get('3-year', float('nan'))),
            'auc_5-year': float(t_auc.get('5-year', float('nan'))),
        })

    if not cindex_results:
        logger.warning("No valid folds for late fusion external evaluation.")
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), {}, {}

    mean_cindex = float(np.mean(cindex_results)); std_cindex = float(np.std(cindex_results))
    mean_iauc = float(np.mean(iauc_results)); std_iauc = float(np.std(iauc_results))
    mean_ibrier = float(np.mean(ibrier_results)); std_ibrier = float(np.std(ibrier_results))

    timepoint_auc_stats = {tp: {'mean': float(np.nanmean(vals)), 'std': float(np.nanstd(vals))} for tp, vals in timepoint_auc_results.items()}

    # 聚合模式
    from utils import CIndex_lifeline as _C, integrated_auc as _IA, integrated_brier_score as _IB, clinical_timepoints_auc as _CTA
    avg_predictions = np.mean(all_fold_predictions, axis=0)
    final_survtime = all_fold_survtimes[0]
    final_censor = all_fold_censors[0]
    aggregated_cindex = _C(avg_predictions, final_censor, final_survtime)
    aggregated_iauc = _IA(avg_predictions, final_censor, final_survtime)
    aggregated_ibrier = _IB(avg_predictions, final_censor, final_survtime)
    aggregated_timepoint_aucs = _CTA(avg_predictions, final_censor, final_survtime)

    predictions_save_dir = os.path.join(opt_rad.checkpoints_dir, opt_rad.exp_name, 'aggregated_predictions')
    os.makedirs(predictions_save_dir, exist_ok=True)
    import pandas as pd
    predictions_save_path_pkl = os.path.join(predictions_save_dir, 'late_fusion_aggregated_predictions.pkl')
    predictions_save_path_csv = os.path.join(predictions_save_dir, 'late_fusion_aggregated_predictions.csv')
    with open(predictions_save_path_pkl, 'wb') as f:
        import pickle as _p
        _p.dump({'predictions': avg_predictions, 'survtime': final_survtime, 'censor': final_censor, 'model_name': 'late_fusion', 'exp_name': opt_rad.exp_name}, f)
    pd.DataFrame({'aggregated_prediction': avg_predictions, 'survival_time': final_survtime, 'censor_status': final_censor, 'model_name': 'late_fusion', 'experiment_name': opt_rad.exp_name}).to_csv(predictions_save_path_csv, index=False, float_format='%.6f')

    # 保存每折外部测试结果（LateFusionNet）
    try:
        import pandas as _pd
        model_dir = os.path.join(opt_rad.checkpoints_dir, opt_rad.exp_name, 'LateFusionNet')
        os.makedirs(model_dir, exist_ok=True)
        _pd.DataFrame(per_fold_rows).to_csv(os.path.join(model_dir, 'external_results.csv'), index=False, float_format='%.6f')
    except Exception as e:
        logger.warning(f"Failed to save LateFusionNet external results CSV: {e}")

    aggregated_results = {
        'cindex': aggregated_cindex,
        'iauc': aggregated_iauc,
        'ibrier': aggregated_ibrier,
        'timepoint_aucs': aggregated_timepoint_aucs,
        'predictions_path_pkl': predictions_save_path_pkl,
        'predictions_path_csv': predictions_save_path_csv,
    }

    return mean_cindex, std_cindex, mean_iauc, std_iauc, mean_ibrier, std_ibrier, timepoint_auc_stats, aggregated_results

def load_external_test_data(logger):
    """加载外部测试集数据。"""
    external_test_path = os.path.join(EXTERNAL_DATA_DIR, 'external_test_data.pkl')
    try:
        with open(external_test_path, 'rb') as f:
            data = pickle.load(f)['test']
        logger.info(f"Successfully loaded external test set: {external_test_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load external test data: {e}. Please run 1_prepare_all_datasets.py first.", exc_info=True)
        return None

# =======================================================================================
# 主函数
# =======================================================================================
def main():
    # 1. 参数解析和日志设置
    base_parser = get_base_parser()
    base_parser.add_argument('--use_optimized', action='store_true', default=True, help="Use optimized hyperparameters (default: True)")
    base_parser.add_argument('--use_baseline', action='store_true', help="Use baseline hyperparameters instead of optimized")
    all_model_modes = [exp['params']['mode'] for exp in BASELINE_EXPERIMENTS]
    simplified_defaults = [to_simplified_mode(m) for m in all_model_modes]
    extended_modes = simplified_defaults + ['LateFusionNet']
    base_parser.add_argument('--models_to_run', nargs='+', type=str, default=extended_modes, help="Specify models to run (support 'LateFusionNet')")
    base_parser.add_argument('--pretrain_exp_name', type=str, default='', help='Experiment name containing pretrained weights')
    args = base_parser.parse_args()
    
    # 如果指定了use_baseline，则不使用优化参数
    if args.use_baseline:
        args.use_optimized = False
    
    EVALUATION_NAME = args.exp_name if args.exp_name != 'fusion_experiment' else ('optimized_evaluation' if args.use_optimized else 'baseline_evaluation')
    logger_manager = LoggerManager(experiment_name=EVALUATION_NAME)
    logger = logger_manager.get_logger(__name__)
    
    logger.info("="*80 + f"\n🚀 Starting Model Evaluation Workflow ({'Optimized Params' if args.use_optimized else 'Default Baseline Params'})")
    logger.info(f"Experiment Set Name: {EVALUATION_NAME}")
    logger.info(f"Models to run: {', '.join(args.models_to_run)}" + "\n" + "="*80)
    
    external_data_raw = load_external_test_data(logger)
    if external_data_raw is None: return

    # 2. 选择参数来源 (默认使用优化参数)
    if args.use_optimized:
        # keys 为内部模式名
        experiments_source = {model: {'name': f"{to_simplified_mode(model)}", 'params': params} for model, params in OPTIMIZED_HYPERPARAMS.items()}
    else:
        experiments_source = {exp['params']['mode']: exp for exp in BASELINE_EXPERIMENTS}

    all_final_results = []
    # 3. 循环执行实验
    for model_mode_in in args.models_to_run:
        mode_internal = normalize_mode(model_mode_in)
        mode_display = to_simplified_mode(mode_internal)
        if mode_internal == 'late_fusion':
            logger.info(f"\n{'='*80}\n➡️  Running Experiment: LateFusionNet\n{'='*80}")
            # CV评估（不训练）
            cv_results = evaluate_late_fusion_on_cv(args, base_parser, logger)
            # 外部评估
            mean_ext_ci, std_ext_ci, mean_ext_iauc, std_ext_iauc, mean_ext_ibrier, std_ext_ibrier, timepoint_auc_stats, aggregated_results = evaluate_late_fusion_on_external(args, external_data_raw, base_parser, logger)
            result_dict = {
                'Model Architecture': 'LateFusionNet',
                'CV Train C-Index': cv_results.get('mean_train_cindex', float('nan')),
                'CV Val C-Index': cv_results.get('mean_val_cindex', float('nan')),
                'CV Val Std': cv_results.get('std_val_cindex', float('nan')),
                'CV Train I-AUC': cv_results.get('mean_train_iauc', float('nan')),
                'CV Val I-AUC': cv_results.get('mean_val_iauc', float('nan')),
                'CV Train I-Brier': cv_results.get('mean_train_ibrier', float('nan')),
                'CV Val I-Brier': cv_results.get('mean_val_ibrier', float('nan')),
                'External Test C-Index': mean_ext_ci,
                'External Test Std': std_ext_ci,
                'External Test I-AUC': mean_ext_iauc,
                'External Test I-AUC Std': std_ext_iauc,
                'External Test I-Brier': mean_ext_ibrier,
                'External Test I-Brier Std': std_ext_ibrier,
            }
            for timepoint, stats in timepoint_auc_stats.items():
                result_dict[f'External Test {timepoint} AUC'] = stats['mean']
                result_dict[f'External Test {timepoint} AUC Std'] = stats['std']
            result_dict['External Test C-Index (Aggregated)'] = aggregated_results['cindex']
            result_dict['External Test I-AUC (Aggregated)'] = aggregated_results['iauc']
            result_dict['External Test I-Brier (Aggregated)'] = aggregated_results['ibrier']
            for timepoint, auc_value in aggregated_results['timepoint_aucs'].items():
                result_dict[f'External Test {timepoint} AUC (Aggregated)'] = auc_value
            all_final_results.append(result_dict)
            continue

        # 常规模型路径
        if mode_internal not in experiments_source:
            logger.warning(f"Configuration for model '{model_mode_in}' not found in source, skipping.")
            continue
        exp_config = experiments_source[mode_internal]
        logger.info(f"\n{'='*80}\n➡️  Running Experiment: {to_simplified_mode(mode_internal)}\n{'='*80}")

        opt = base_parser.parse_args([]) # Create a fresh opt object
        opt.exp_name = EVALUATION_NAME
        opt.mode = mode_internal
        for key, value in vars(args).items(): setattr(opt, key, value)
        for key, value in exp_config['params'].items(): setattr(opt, key, value)
        opt.model_name = mode_display
        opt = parse_gpuids(opt)

        cv_results = run_cv_experiment(opt=opt, parser=base_parser, trial=None)
        mean_ext_ci, std_ext_ci, mean_ext_iauc, std_ext_iauc, mean_ext_ibrier, std_ext_ibrier, timepoint_auc_stats, aggregated_results = evaluate_folds_on_external(opt, external_data_raw, logger)

        result_dict = {
            'Model Architecture': exp_config['name'], 
            'CV Train C-Index': cv_results.get('mean_train_cindex', float('nan')),
            'CV Val C-Index': cv_results.get('mean_val_cindex', float('nan')), 
            'CV Val Std': cv_results.get('std_val_cindex', float('nan')),
            'CV Train I-AUC': cv_results.get('mean_train_iauc', float('nan')),
            'CV Val I-AUC': cv_results.get('mean_val_iauc', float('nan')),
            'CV Train I-Brier': cv_results.get('mean_train_ibrier', float('nan')),
            'CV Val I-Brier': cv_results.get('mean_val_ibrier', float('nan')),
            'External Test C-Index': mean_ext_ci, 
            'External Test Std': std_ext_ci,
            'External Test I-AUC': mean_ext_iauc,
            'External Test I-AUC Std': std_ext_iauc,
            'External Test I-Brier': mean_ext_ibrier,
            'External Test I-Brier Std': std_ext_ibrier
        }
        for timepoint, stats in timepoint_auc_stats.items():
            result_dict[f'External Test {timepoint} AUC'] = stats['mean']
            result_dict[f'External Test {timepoint} AUC Std'] = stats['std']
        result_dict['External Test C-Index (Aggregated)'] = aggregated_results['cindex']
        result_dict['External Test I-AUC (Aggregated)'] = aggregated_results['iauc']
        result_dict['External Test I-Brier (Aggregated)'] = aggregated_results['ibrier']
        for timepoint, auc_value in aggregated_results['timepoint_aucs'].items():
            result_dict[f'External Test {timepoint} AUC (Aggregated)'] = auc_value
        all_final_results.append(result_dict)

    # 4. 【核心修改】生成最终报告
    if all_final_results:
        df = pd.DataFrame(all_final_results).sort_values(by='External Test C-Index', ascending=False).reset_index(drop=True)
        
        logger.info("\n\n" + "="*120)
        logger.info("--- Final Performance Summary: Cross-Validation vs. External Test Set ---")
        logger.info("="*120 + "\n")
        
        # 定义要显示的列顺序
        display_cols = [
            'Model Architecture', 
            'CV Train C-Index', 
            'CV Val C-Index', 
            'CV Val Std', # 【新增】
            'CV Train I-AUC',  # 【新增I-AUC】
            'CV Val I-AUC',    # 【新增I-AUC】
            'CV Train I-Brier',  # 【新增I-Brier】
            'CV Val I-Brier',    # 【新增I-Brier】
            'External Test C-Index', 
            'External Test Std',
            'External Test C-Index (Aggregated)',  # 【新增聚合C-Index】
            'External Test I-AUC',      # 【新增外部测试I-AUC】
            'External Test I-AUC Std',   # 【新增外部测试I-AUC标准差】
            'External Test I-AUC (Aggregated)',    # 【新增聚合I-AUC】
            'External Test I-Brier',      # 【新增外部测试I-Brier】
            'External Test I-Brier Std',   # 【新增外部测试I-Brier标准差】
            'External Test I-Brier (Aggregated)',  # 【新增聚合I-Brier】
            'External Test 1-year AUC',    # 【新增1年AUC】
            'External Test 1-year AUC Std', # 【新增1年AUC标准差】
            'External Test 1-year AUC (Aggregated)', # 【新增聚合1年AUC】
            'External Test 2-year AUC',    # 【新增2年AUC】
            'External Test 2-year AUC Std', # 【新增2年AUC标准差】
            'External Test 2-year AUC (Aggregated)', # 【新增聚合2年AUC】
            'External Test 3-year AUC',    # 【新增3年AUC】
            'External Test 3-year AUC Std', # 【新增3年AUC标准差】
            'External Test 3-year AUC (Aggregated)', # 【新增聚合3年AUC】
            'External Test 5-year AUC',    # 【新增5年AUC】
            'External Test 5-year AUC Std',  # 【新增5年AUC标准差】
            'External Test 5-year AUC (Aggregated)'  # 【新增聚合5年AUC】
        ]
        
        # 确保所有需要的列都存在，以防万一
        for col in display_cols:
            if col not in df.columns:
                df[col] = float('nan')
        
        # 打印到控制台
        logger.info("\n" + df[display_cols].to_string(index=False, float_format="%.4f"))
        logger.info("\n" + "="*120)
        
        # 保存到CSV文件
        report_path = os.path.join(RESULTS_DIR, f'{EVALUATION_NAME}_complete_report.csv')
        df.to_csv(report_path, index=False, float_format="%.4f")
        logger.info(f"\nComplete report saved to: {report_path}")

if __name__ == '__main__':
    main()