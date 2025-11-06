import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader, WeightedRandomSampler

# 导入核心模块
from data_loaders import MriWsiDataset, create_standardized_datasets
# 【修改】导入 BaseFusionModel 以便进行类型检查
from networks import define_net, define_scheduler, BaseFusionModel, MultiScaleFusionNet 
from utils import CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, calculate_l1_reg, count_parameters, integrated_auc, integrated_brier_score, clinical_timepoints_auc

# =======================================================================================
# 1. 早停机制 (Early Stopping) - 保持不变
# =======================================================================================
class EarlyStopping:
    """改进的早停机制，支持动态调整和更智能的停止策略"""
    def __init__(self, patience=12, min_delta=0.002, restore_best_weights=True, 
                 dynamic_patience=True, min_patience=8, max_patience=20):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.dynamic_patience = dynamic_patience
        self.min_patience = min_patience
        self.max_patience = max_patience
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.improvement_history = []
        self.plateau_count = 0
        print(f"-> Enhanced early stopping initialized: patience={self.patience}, min_delta={self.min_delta}, dynamic={self.dynamic_patience}")
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            if self.restore_best_weights:
                self.best_model_state = copy.deepcopy(model.state_dict())
            self.improvement_history.append(val_score)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            self.plateau_count += 1
            
            # 动态调整patience
            if self.dynamic_patience and len(self.improvement_history) > 5:
                recent_improvements = [self.improvement_history[i] - self.improvement_history[i-1] 
                                     for i in range(-5, 0) if i < len(self.improvement_history)]
                avg_improvement = sum(recent_improvements) / len(recent_improvements)
                
                if avg_improvement > 0.01:  # 如果最近改进较大，增加patience
                    self.patience = min(self.max_patience, self.patience + 2)
                elif avg_improvement < 0.005:  # 如果改进很小，减少patience
                    self.patience = max(self.min_patience, self.patience - 1)
            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            improvement = val_score - self.best_score
            self.best_score = val_score
            if self.restore_best_weights:
                self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.plateau_count = 0
            self.improvement_history.append(val_score)
            
            # 限制历史记录长度
            if len(self.improvement_history) > 20:
                self.improvement_history = self.improvement_history[-20:]


# =======================================================================================
# 2. 核心训练函数 (train) - 【核心修正部分】
# =======================================================================================
def train(opt, data, device, k):
    """主训练函数，内置对融合模型微调的差分学习率逻辑。"""
    # 环境设置
    cudnn.deterministic = True
    torch.manual_seed(2019); random.seed(2019)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(2019)
    
    # 1. 创建模型
    model = define_net(opt, k)
    
    # --- 2. 【关键修正】根据模型类型选择优化器 ---
    # 检查模型是否是 BaseFusionModel 的子类 (即多模态融合模型)
    if isinstance(model, BaseFusionModel):
        print("  - ⚙️  检测到融合模型，正在为全模型微调模式设置差分学习率...")
        
        encoder_prefixes = ('path_encoder', 'rad_encoder', 'path_encoder_fine', 'rad_encoder_fine', 'rad_reducer_fine')
        encoder_params, fusion_params = [], []
        
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if name.startswith(encoder_prefixes):
                encoder_params.append(param)
            else:
                fusion_params.append(param)
        
        encoder_weight_decay = getattr(opt, 'weight_decay', 1e-5) * float(getattr(opt, 'wd_encoder_ratio', 0.1))
        param_groups = [
            # 【重要】确保 opt 对象有 lr_encoder_ratio 属性
            {'params': encoder_params, 'lr': opt.lr * getattr(opt, 'lr_encoder_ratio', 0.1), 'weight_decay': encoder_weight_decay},
            {'params': fusion_params, 'lr': opt.lr, 'weight_decay': opt.weight_decay}
        ]
        
        optimizer = optim.AdamW(param_groups)
        
        print(f"  - 编码器学习率: {opt.lr * getattr(opt, 'lr_encoder_ratio', 0.1):.2e}")
        print(f"  - 融合层学习率: {opt.lr:.2e}")
        
    else:
        # 如果是单模态模型 (PathFeatureEncoder, RadFeatureEncoder)
        print("  - ⚙️  检测到单模态模型，正在设置标准优化器...")
        # 我们需要一个标准的优化器定义函数
        from networks import define_optimizer # 可以在这里局部导入
        optimizer = define_optimizer(opt, model)
        print(f"  - 标准学习率: {opt.lr:.2e}")

    # 3. 创建学习率调度器和数据集加载器
    # 仅当策略为 linear/cosine 时通过工厂函数创建；其它策略在此处直接构建
    if hasattr(opt, 'lr_policy') and opt.lr_policy in ('linear', 'cosine'):
        scheduler = define_scheduler(opt, optimizer)
    elif hasattr(opt, 'lr_policy') and opt.lr_policy == 'plateau':
        from torch.optim import lr_scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5,
            patience=3, threshold=0.001,
            threshold_mode='abs',
            min_lr=1e-7, cooldown=2
        )
    elif hasattr(opt, 'lr_policy') and opt.lr_policy == 'cosine_warm':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    elif hasattr(opt, 'lr_policy') and opt.lr_policy == 'adaptive':
        # 新增：自适应学习率调度器
        class AdaptiveLRScheduler:
            def __init__(self, optimizer, patience=5, factor=0.5, min_lr=1e-7):
                self.optimizer = optimizer
                self.patience = patience
                self.factor = factor
                self.min_lr = min_lr
                self.best_score = None
                self.counter = 0
                self.last_lr_change = 0
                
            def step(self, val_score=None, epoch=0):
                if val_score is not None:
                    if self.best_score is None or val_score > self.best_score:
                        self.best_score = val_score
                        self.counter = 0
                    else:
                        self.counter += 1
                        
                    # 如果性能停滞且距离上次调整足够远，则降低学习率
                    if self.counter >= self.patience and epoch - self.last_lr_change >= 3:
                        for param_group in self.optimizer.param_groups:
                            old_lr = param_group['lr']
                            new_lr = max(old_lr * self.factor, self.min_lr)
                            if new_lr < old_lr:
                                param_group['lr'] = new_lr
                                print(f"Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")
                                self.last_lr_change = epoch
                                self.counter = 0
        scheduler = AdaptiveLRScheduler(optimizer)
    else:
        # 默认回退到 cosine
        scheduler = define_scheduler(opt, optimizer)
    
    print(f"✅ 模型创建完成 | 可训练参数量: {count_parameters(model):,}")

    train_dataset, test_dataset, _ = create_standardized_datasets(opt, data)
    # 事件感知采样：确保每个batch包含事件样本
    try:
        events_np = np.array(train_dataset.e)
        num_pos = int((events_np == 1).sum())
        num_total = len(events_np)
        num_neg = num_total - num_pos
        if num_pos > 0 and num_neg > 0:
            pos_w = 0.5 / num_pos
            neg_w = 0.5 / num_neg
            sample_weights = np.where(events_np == 1, pos_w, neg_w).astype(np.float64)
            sampler = WeightedRandomSampler(sample_weights, num_samples=num_total, replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    except Exception:
        # 回退到普通随机打乱
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # ... (后续训练循环完全不变) ...
    early_stopping = EarlyStopping(patience=getattr(opt, 'patience', 12), min_delta=0.0015)  # 更精细的早停：patience调整为12，min_delta调整为0.0015

    # 回退：不使用EMA
    total_epochs = opt.niter + opt.niter_decay

    # 可选：冻结编码器若干 epoch（仅适用于融合模型）
    freeze_epochs = int(getattr(opt, 'freeze_encoder_epochs', 0))
    if isinstance(model, BaseFusionModel) and freeze_epochs > 0:
        for p in model.path_encoder.parameters():
            p.requires_grad = False
        for p in model.rad_encoder.parameters():
            p.requires_grad = False
        print(f"  - 冻结编码器参数前 {freeze_epochs} 个 epoch")

    grad_accum_steps = 1  # 回退：关闭梯度累积
    for epoch in range(getattr(opt, 'epoch_count', 1), total_epochs + 1):
        model.train()
        loss_epoch = 0.0

        for step, (x_path, x_rad, censor, survtime, _) in enumerate(tqdm(train_loader, desc=f"Split '{k}' Epoch {epoch}/{total_epochs}", leave=False)):
            x_path, x_rad, censor, survtime = x_path.to(device), x_rad.to(device), censor.to(device), survtime.to(device)

            # 模态丢弃（提升鲁棒性）
            p_drop = getattr(opt, 'modality_dropout_p', 0.0)
            if model.training and p_drop > 0.0:
                with torch.no_grad():
                    mask = torch.rand(2, device=device)
                    drop_path = (mask[0] < p_drop)
                    drop_rad = (mask[1] < p_drop)
                if drop_path:
                    x_path = torch.zeros_like(x_path)
                if drop_rad:
                    x_rad = torch.zeros_like(x_rad)
            
            
            input_data = {'x_path': x_path, 'x_rad': x_rad}
            model_output = model(**input_data)
            
            # 解析模型输出 - 支持新的CoAttentionFusionNet格式
            if isinstance(model_output, tuple) and len(model_output) == 5:
                # 新的CoAttentionFusionNet格式: (features_tuple, logits, align_loss, attention_weights, sig_outputs)
                features_tuple, logits, model_align_loss, attention_weights, sig_outputs = model_output
                fused_features, attn_entropy, path_z, rad_z = features_tuple
            elif isinstance(model_output, tuple) and len(model_output) == 2:
                # 标准格式: (features_tuple, logits)
                features_tuple, logits = model_output
                model_align_loss, attention_weights, sig_outputs = 0.0, None, None
                # 解析features_tuple
                if isinstance(features_tuple, tuple) and len(features_tuple) >= 2:
                    if len(features_tuple) == 4:
                        fused_features, attn_entropy, path_z, rad_z = features_tuple
                    elif len(features_tuple) == 3:
                        fused_features, path_z, rad_z = features_tuple
                        attn_entropy = None
                    else:
                        fused_features = features_tuple[0]
                        attn_entropy, path_z, rad_z = None, None, None
                else:
                    fused_features = features_tuple
                    attn_entropy, path_z, rad_z = None, None, None
            else:
                # 兼容旧格式
                fused_features = model_output
                logits = None
                attn_entropy, path_z, rad_z = None, None, None
                model_align_loss, attention_weights, sig_outputs = 0.0, None, None
            
            # 规整logits维度：确保为 [B]
            if isinstance(logits, torch.Tensor):
                if logits.dim() == 2 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                elif logits.dim() == 2 and logits.size(1) > 1:
                    logits = logits[:, 0]

            loss_cox = CoxLoss(survtime, censor, logits, device)
            
            # 两模态对齐损失：若模型已提供内部对齐损失，则禁用外层MSE，避免重复计权
            align_w_outer = float(getattr(opt, 'align_loss_weight', 0.0))
            loss_align_model = torch.tensor(0.0, device=device)
            has_model_align = False
            if isinstance(model_align_loss, torch.Tensor) and model_align_loss.numel() > 0:
                loss_align_model = model_align_loss
                has_model_align = True
            elif isinstance(model_align_loss, (float, int)) and model_align_loss > 0:
                loss_align_model = torch.tensor(model_align_loss, device=device)
                has_model_align = True

            loss_align_outer = torch.tensor(0.0, device=device)
            if (not has_model_align) and align_w_outer > 0.0 and (path_z is not None) and (rad_z is not None):
                loss_align_outer = torch.mean((path_z - rad_z) ** 2) * align_w_outer
            
            # 注意力熵正则（仅 WeightedFusionNet 返回）
            attn_entropy_w = float(getattr(opt, 'attn_entropy_weight', 0.0))
            loss_attn_entropy = torch.tensor(0.0, device=device)
            if attn_entropy_w > 0.0 and attn_entropy is not None:
                loss_attn_entropy = -attn_entropy  # 最大化熵 -> 最小化 -熵
            
            # Signature网络损失 (如果启用)
            loss_sig = torch.tensor(0.0, device=device)
            alpha_surv = getattr(opt, 'alpha_surv', 0.0)
            if alpha_surv > 0.0 and sig_outputs is not None:
                # 计算signature网络的生存损失
                for sig_logits in sig_outputs:
                    loss_sig += CoxLoss(survtime, censor, sig_logits, device)
                loss_sig = loss_sig * alpha_surv / len(sig_outputs)
            
            loss_reg = calculate_l1_reg(model, opt, device)
            loss = (
                getattr(opt, 'lambda_cox', 1.0) * loss_cox
                + getattr(opt, 'lambda_reg', 3e-5) * loss_reg
                + loss_align_outer
                + loss_align_model
                + attn_entropy_w * loss_attn_entropy
                + loss_sig
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 温和的梯度裁剪
            optimizer.step()
            loss_epoch += loss.item()

        # 先进行验证评估，再根据策略进行调度（避免使用未定义的 cindex_val）
        # 直接使用当前权重验证
        val_loss, cindex_val, _, _, i_auc_val, i_brier_val, timepoint_aucs_val, _ = test(opt, model, test_dataset, device)
        
        # 智能学习率调度（根据刚得到的验证指标）
        if hasattr(scheduler, 'step') and hasattr(opt, 'lr_policy'):
            if opt.lr_policy == 'plateau':
                scheduler.step(cindex_val)
            elif opt.lr_policy == 'adaptive':
                scheduler.step(cindex_val, epoch)
            else:
                scheduler.step()
        elif hasattr(scheduler, 'step'):
            scheduler.step()

        early_stopping(cindex_val, model)

        if epoch % 5 == 0 or epoch == total_epochs or early_stopping.early_stop:
            print(f"\nEpoch {epoch}/{total_epochs}: Train Loss: {loss_epoch/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val C-Index: {cindex_val:.4f} (Best: {early_stopping.best_score:.4f}) | Val I-AUC: {i_auc_val:.4f}")
        
        if early_stopping.early_stop:
            print(f"\n>> Early stopping triggered at epoch {epoch}. Best validation C-Index: {early_stopping.best_score:.4f}")
            break

        # 解冻编码器
        if isinstance(model, BaseFusionModel) and freeze_epochs > 0 and epoch == freeze_epochs:
            for p in model.path_encoder.parameters():
                p.requires_grad = True
            for p in model.rad_encoder.parameters():
                p.requires_grad = True
            # 需重建优化器以包含新解冻参数的梯度
            # 复用差分学习率设置
            encoder_prefixes = ('path_encoder', 'rad_encoder', 'path_encoder_fine', 'rad_encoder_fine', 'rad_reducer_fine')
            encoder_params, fusion_params = [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith(encoder_prefixes):
                    encoder_params.append(param)
                else:
                    fusion_params.append(param)
            encoder_wd = getattr(opt, 'weight_decay', 1e-5) * float(getattr(opt, 'wd_encoder_ratio', 0.1))
            lr_enc = opt.lr * float(getattr(opt, 'lr_encoder_ratio', 0.1))
            param_groups = [
                {'params': encoder_params, 'lr': lr_enc, 'weight_decay': encoder_wd},
                {'params': fusion_params, 'lr': opt.lr, 'weight_decay': opt.weight_decay},
            ]
            optimizer = optim.AdamW(param_groups)
            print(f"  - 解冻编码器，重建优化器：encoder lr={lr_enc:.2e}, wd={encoder_wd:.1e} | fusion lr={opt.lr:.2e}, wd={opt.weight_decay:.1e}")
    
    print("-> Loading best model weights found by early stopping...")
    model.load_state_dict(early_stopping.best_model_state)
    
    return model, optimizer, {'best_val_cindex': early_stopping.best_score}

# =======================================================================================
# 3. 核心测试/评估函数 (test) - 保持不变
# =======================================================================================
def test(opt, model, test_dataset, device):
    # ... (代码不变)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    risk_pred_all, censor_all, survtime_all = [], [], []
    loss_test = 0.0

    with torch.no_grad():
        for x_path, x_rad, censor, survtime, _ in test_loader:
            x_path, x_rad, censor, survtime = x_path.to(device), x_rad.to(device), censor.to(device), survtime.to(device)
            
            input_data = {'x_path': x_path, 'x_rad': x_rad}
            model_output = model(**input_data)
            
            # 解析模型输出 - 支持新的CoAttentionFusionNet格式
            if isinstance(model_output, tuple) and len(model_output) == 5:
                # 新的CoAttentionFusionNet格式: (features_tuple, logits, align_loss, attention_weights, sig_outputs)
                _, logits, _, _, _ = model_output
            elif isinstance(model_output, tuple) and len(model_output) == 2:
                # 标准格式: (features_tuple, logits)
                _, logits = model_output
            else:
                # 兼容旧格式 - 假设只返回logits
                logits = model_output

            # 规整logits维度：确保为 [B]
            if isinstance(logits, torch.Tensor):
                if logits.dim() == 2 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                elif logits.dim() == 2 and logits.size(1) > 1:
                    logits = logits[:, 0]

            loss_cox = CoxLoss(survtime, censor, logits, device)
            loss_test += loss_cox.item()
            
            risk_pred_all.extend(logits.detach().cpu().numpy().flatten())
            censor_all.extend(censor.detach().cpu().numpy().flatten())
            survtime_all.extend(survtime.detach().cpu().numpy().flatten())
    
    avg_loss = loss_test / len(test_loader)
    risk_pred_all, censor_all, survtime_all = np.array(risk_pred_all), np.array(censor_all), np.array(survtime_all)
    
    cindex = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
    pvalue = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    surv_acc = accuracy_cox(risk_pred_all, censor_all)
    
    # 计算积分AUC (I-AUC)
    i_auc = integrated_auc(risk_pred_all, censor_all, survtime_all)
    
    # 计算积分Brier Score (IBS)
    i_brier = integrated_brier_score(risk_pred_all, censor_all, survtime_all)
    
    # 计算四个临床时间点的时间依赖性AUC
    timepoint_aucs = clinical_timepoints_auc(risk_pred_all, censor_all, survtime_all)
    
    return avg_loss, cindex, pvalue, surv_acc, i_auc, i_brier, timepoint_aucs, [risk_pred_all, survtime_all, censor_all]