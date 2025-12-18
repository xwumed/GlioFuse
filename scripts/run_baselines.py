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
        fusion_modes = ['EarlyFusionNet', 'BilinearFusionNet', 'LateFusionNet']
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
    # 使用实验名称（exp['name']）作为默认运行列表，这样可以区分同一模型的不同变体
    all_model_names = [exp['name'] for exp in BASELINE_EXPERIMENTS]
    base_parser.add_argument('--models_to_run', nargs='+', type=str, default=all_model_names, help="Specify models to run")
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
        # 使用 OPTIMIZED_HYPERPARAMS，键为模型名（含变体后缀）
        experiments_source = {model: {'name': model, 'params': params} for model, params in OPTIMIZED_HYPERPARAMS.items()}
    else:
        # 使用 BASELINE_EXPERIMENTS，键为实验名
        experiments_source = {exp['name']: exp for exp in BASELINE_EXPERIMENTS}

    all_final_results = []
    # 3. 循环执行实验
    for model_name_in in args.models_to_run:
        # 直接使用模型名查找配置
        if model_name_in not in experiments_source:
            logger.warning(f"Configuration for model '{model_name_in}' not found in source, skipping.")
            continue
        exp_config = experiments_source[model_name_in]
        logger.info(f"\n{'='*80}\n➡️  Running Experiment: {model_name_in}\n{'='*80}")

        opt = base_parser.parse_args([]) # Create a fresh opt object
        opt.exp_name = EVALUATION_NAME
        # 继承命令行参数
        for key, value in vars(args).items(): setattr(opt, key, value)
        # 应用配置中的参数
        for key, value in exp_config['params'].items(): setattr(opt, key, value)
        # 如果 mode 未设置，根据模型名推断（去掉 _avg, _weighted 等后缀）
        if not hasattr(opt, 'mode') or not opt.mode:
            base_mode = model_name_in.split('_')[0] if '_' in model_name_in else model_name_in
            # 处理特殊情况：LateFusionNet_avg -> LateFusionNet
            if model_name_in.startswith('LateFusionNet'):
                opt.mode = 'LateFusionNet'
            else:
                opt.mode = base_mode
        # 使用完整的模型名（含变体后缀）作为保存目录名
        opt.model_name = model_name_in
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