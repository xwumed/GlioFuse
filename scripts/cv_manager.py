import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from argparse import Namespace
import optuna

# 导入项目模块
from config import N_SPLITS, CV_SPLITS_DIR
from data_loaders import create_standardized_datasets
from train_test import train, test

class CrossValidationManager:
    """
    统一的交叉验证管理器。
    负责管理 K-折交叉验证的训练和评估流程，并支持 Optuna 剪枝。
    """
    
    def __init__(self, opt: Namespace, device: torch.device):
        self.opt = opt
        self.device = device
        self.n_splits = N_SPLITS
        
        # 使用清晰的变量名 'val' 和 'train'
        self.results = {
            'train_cindices': [],  # 【新增】
            'val_cindices': [],
            'val_pvalues': [],
            'train_iaucs': [],     # 【新增I-AUC】
            'val_iaucs': [],       # 【新增I-AUC】
            'train_ibriers': [],   # 【新增I-Brier】
            'val_ibriers': [],     # 【新增I-Brier】
            'fold_results': []
        }
    
    def _get_model_paths(self, fold: int) -> Tuple[str, str]:
        # ... (代码不变) ...
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.exp_name, self.opt.model_name)
        weights_path = os.path.join(expr_dir, f'split_{fold}_best_weights.pt')
        metadata_path = os.path.join(expr_dir, f'split_{fold}_metadata.pkl')
        return weights_path, metadata_path
    
    def load_fold_data(self, fold: int) -> Dict[str, Any]:
        # ... (代码不变) ...
        split_data_path = os.path.join(CV_SPLITS_DIR, f'split_{fold}_data.pkl')
        try:
            with open(split_data_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {split_data_path}. Please run the data preparation script first.")
    
    def save_model_and_metadata(self, model: nn.Module, fold: int, metric_logger: Dict) -> None:
        # ... (代码不变) ...
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.exp_name, self.opt.model_name)
        os.makedirs(expr_dir, exist_ok=True)
        
        weights_path, metadata_path = self._get_model_paths(fold)
        torch.save(model.state_dict(), weights_path)
        
        metadata = {'split': fold, 'opt': vars(self.opt), 'metrics': metric_logger}
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

    def run_training_cv(self, trial: Optional[optuna.trial.Trial] = None) -> Dict[str, Any]:
        """
        运行K-折交叉验证的训练流程，并评估训练集和验证集。
        """
        print(f"🚀 Starting {self.n_splits}-fold cross-validation training for model: {self.opt.model_name}...")
        
        for k in range(self.n_splits):
            print(f"\n📊 [Fold {k + 1}/{self.n_splits}]")
            
            fold_data_dict = self.load_fold_data(k)
            print("  - Data for fold loaded successfully.")
            
            model, _, metric_logger = train(self.opt, fold_data_dict, self.device, k)
            print("  - Model training completed.")
            
            # --- 【核心修改】在训练集和验证集上进行最终评估 ---
            train_dataset, val_dataset, _ = create_standardized_datasets(self.opt, fold_data_dict)
            
            # 在验证集上评估
            _, cindex_val, pvalue_val, _, iauc_val, ibrier_val, timepoint_aucs_val, _ = test(self.opt, model, val_dataset, self.device)
            print(f"  - Final evaluation on validation set: C-Index = {cindex_val:.4f}, I-AUC = {iauc_val:.4f}, I-Brier = {ibrier_val:.4f}")

            # 在训练集上评估
            _, cindex_train, _, _, iauc_train, ibrier_train, timepoint_aucs_train, _ = test(self.opt, model, train_dataset, self.device)
            print(f"  - Final evaluation on training set:   C-Index = {cindex_train:.4f}, I-AUC = {iauc_train:.4f}, I-Brier = {ibrier_train:.4f}")
            
            # 记录结果
            self.results['train_cindices'].append(cindex_train) # 【新增】
            self.results['val_cindices'].append(cindex_val)
            self.results['val_pvalues'].append(pvalue_val)
            self.results['train_iaucs'].append(iauc_train)      # 【新增I-AUC】
            self.results['val_iaucs'].append(iauc_val)          # 【新增I-AUC】
            self.results['train_ibriers'].append(ibrier_train)  # 【新增I-Brier】
            self.results['val_ibriers'].append(ibrier_val)      # 【新增I-Brier】
            
            fold_result = {
                'fold': k + 1, 
                'train_cindex': cindex_train, # 【新增】
                'val_cindex': cindex_val, 
                'val_pvalue': pvalue_val,
                'train_iauc': iauc_train,     # 【新增I-AUC】
                'val_iauc': iauc_val,         # 【新增I-AUC】
                'train_ibrier': ibrier_train, # 【新增I-Brier】
                'val_ibrier': ibrier_val      # 【新增I-Brier】
            }
            self.results['fold_results'].append(fold_result)
            self.save_model_and_metadata(model, k, metric_logger)
            print(f"  - Model weights and metadata for fold {k+1} have been saved.")

            if trial:
                trial.report(cindex_val, k)
                if trial.should_prune():
                    print(f"  - ✂️ Trial pruned at fold {k+1} due to poor performance.")
                    raise optuna.exceptions.TrialPruned()
        
        return self._compute_final_statistics()

    def _compute_final_statistics(self) -> Dict[str, Any]:
        """计算交叉验证的最终统计结果（均值和标准差）。"""
        train_cindices_np = np.array(self.results['train_cindices']) # 【新增】
        val_cindices_np = np.array(self.results['val_cindices'])
        train_iaucs_np = np.array(self.results['train_iaucs'])       # 【新增I-AUC】
        val_iaucs_np = np.array(self.results['val_iaucs'])           # 【新增I-AUC】
        train_ibriers_np = np.array(self.results['train_ibriers'])   # 【新增I-Brier】
        val_ibriers_np = np.array(self.results['val_ibriers'])       # 【新增I-Brier】
        
        stats = {
            'mean_train_cindex': np.mean(train_cindices_np), # 【新增】
            'std_train_cindex': np.std(train_cindices_np),   # 【新增】
            'mean_val_cindex': np.mean(val_cindices_np),
            'std_val_cindex': np.std(val_cindices_np),
            'mean_train_iauc': np.mean(train_iaucs_np),      # 【新增I-AUC】
            'std_train_iauc': np.std(train_iaucs_np),        # 【新增I-AUC】
            'mean_val_iauc': np.mean(val_iaucs_np),          # 【新增I-AUC】
            'std_val_iauc': np.std(val_iaucs_np),            # 【新增I-AUC】
            'mean_train_ibrier': np.mean(train_ibriers_np),  # 【新增I-Brier】
            'std_train_ibrier': np.std(train_ibriers_np),    # 【新增I-Brier】
            'mean_val_ibrier': np.mean(val_ibriers_np),      # 【新增I-Brier】
            'std_val_ibrier': np.std(val_ibriers_np),        # 【新增I-Brier】
            'fold_results': self.results['fold_results']
        }
        return stats
    
    # save_results 和 print_summary 函数可以保持不变，因为它们依赖于 fold_results 和 _compute_final_statistics，
    # 而这些已经更新了。不过为了清晰，我们也一并更新它们。
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """保存每折与汇总统计结果到实验目录，并返回CSV路径。"""
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.exp_name, self.opt.model_name)
        os.makedirs(expr_dir, exist_ok=True)

        # 1) 保存按折结果
        df = pd.DataFrame(results.get('fold_results', []))
        cv_csv_path = os.path.join(expr_dir, 'cv_results.csv')
        df.to_csv(cv_csv_path, index=False)

        # 2) 保存汇总统计
        summary = {
            'mean_train_cindex': float(results.get('mean_train_cindex', float('nan'))),
            'std_train_cindex': float(results.get('std_train_cindex', float('nan'))),
            'mean_val_cindex': float(results.get('mean_val_cindex', float('nan'))),
            'std_val_cindex': float(results.get('std_val_cindex', float('nan'))),
            'mean_train_iauc': float(results.get('mean_train_iauc', float('nan'))),
            'std_train_iauc': float(results.get('std_train_iauc', float('nan'))),
            'mean_val_iauc': float(results.get('mean_val_iauc', float('nan'))),
            'std_val_iauc': float(results.get('std_val_iauc', float('nan'))),
            'mean_train_ibrier': float(results.get('mean_train_ibrier', float('nan'))),
            'std_train_ibrier': float(results.get('std_train_ibrier', float('nan'))),
            'mean_val_ibrier': float(results.get('mean_val_ibrier', float('nan'))),
            'std_val_ibrier': float(results.get('std_val_ibrier', float('nan'))),
            'n_splits': int(self.n_splits),
        }
        summary_json_path = os.path.join(expr_dir, 'cv_summary.json')
        try:
            import json
            with open(summary_json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"警告: 无法保存交叉验证汇总到 {summary_json_path}: {e}")

        return cv_csv_path

    def print_summary(self, results: Dict[str, Any]):
        """在控制台打印交叉验证结果的摘要。"""
        print("\n" + "="*80)
        print(f"🎉 Cross-Validation Completed: {self.opt.model_name}")
        print("="*80)
        df = pd.DataFrame(results['fold_results'])
        print("Fold-wise Performance on Train/Validation Sets:")
        # to_string 会自动打印所有列
        print(df.to_string(index=False, float_format="%.4f")) 
        print("-" * 80)
        print(f"🎯 Average Train C-Index:      {results['mean_train_cindex']:.4f} ± {results['std_train_cindex']:.4f}")
        print(f"🎯 Average Validation C-Index: {results['mean_val_cindex']:.4f} ± {results['std_val_cindex']:.4f}")
        print(f"🎯 Average Train I-AUC:        {results['mean_train_iauc']:.4f} ± {results['std_train_iauc']:.4f}")
        print(f"🎯 Average Validation I-AUC:   {results['mean_val_iauc']:.4f} ± {results['std_val_iauc']:.4f}")
        print("="*80)