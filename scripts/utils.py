import torch
import torch.nn as nn
import numpy as np
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_auc_score
from scipy import integrate

# =======================================================================================
# 1. 生存分析损失函数和评估指标
# =======================================================================================

def CoxLoss(survtime: torch.Tensor, censor: torch.Tensor, hazard_pred: torch.Tensor, device: torch.device):
    """
    稳定高效的 Cox 部分似然负对数损失。

    采用按生存时间降序排序 + logcumsumexp 实现风险集的对数和，
    复杂度 O(B log B) + O(B)，数值更稳定，避免 O(B^2) 的风险集矩阵。
    
    Args:
        survtime: 生存时间
        censor: 事件指示器
        hazard_pred: 风险预测值
        device: 计算设备
    """
    # 展平为 1D
    t = survtime.view(-1)
    e = censor.view(-1).float()
    h = hazard_pred.view(-1)

    # 事件总数
    num_events = torch.sum(e)
    if num_events.item() == 0:
        return torch.tensor(0.0, device=device)

    # 按生存时间降序排序（较大时间在前），风险集为前缀
    order = torch.argsort(t, descending=True)
    t_sorted = t[order]
    e_sorted = e[order]
    h_sorted = h[order]

    # 风险集对数和：log(sum_j exp(h_j)) for j in risk set i
    log_cum_sum_exp = torch.logcumsumexp(h_sorted, dim=0)

    # 仅对事件样本累计损失项
    contributions = h_sorted - log_cum_sum_exp
    loss = -torch.sum(contributions * e_sorted) / num_events
    return loss


def CIndex_lifeline(hazard_pred: np.ndarray, censor: np.ndarray, survtime: np.ndarray) -> float:
    """
    使用 lifelines 包计算 C-Index (一致性指数)。
    注意：lifelines 的 C-Index 期望的 event_indicator 是布尔或0/1，与我们的定义一致。
    对于 hazard_pred，更高的值应对应更差的生存结果，因此我们需要对其取反。
    """
    return concordance_index(survtime, -hazard_pred, censor)


def cox_log_rank(hazard_pred: np.ndarray, censor: np.ndarray, survtime: np.ndarray) -> float:
    """
    使用对数秩检验 (log-rank test) 计算 p-value。
    通过预测的风险值中位数，将样本分为高风险组和低风险组。
    """
    # 根据风险预测值的中位数将患者分为两组
    median_hazard = np.median(hazard_pred)
    is_high_risk = (hazard_pred >= median_hazard)
    
    # 为高风险组和低风险组准备数据
    T_high, E_high = survtime[is_high_risk], censor[is_high_risk]
    T_low, E_low = survtime[~is_high_risk], censor[~is_high_risk]
    
    # 执行对数秩检验
    results = logrank_test(T_high, T_low, event_observed_A=E_high, event_observed_B=E_low)
    return results.p_value


def accuracy_cox(hazard_pred: np.ndarray, censor: np.ndarray) -> float:
    """
    一个简化的"准确率"计算，衡量风险预测方向与事件是否发生的一致性。
    注意：这个指标在生存分析中不常用，因为审查样本的存在使其意义模糊。
    仅用于粗略参考。
    """
    # 假设风险值 > 0.5 (如果经过sigmoid) 预测为事件发生 (1)
    # 这依赖于模型输出的尺度，可能需要调整阈值
    pred = (hazard_pred >= np.median(hazard_pred)).astype(int)
    # 计算预测与实际事件发生的一致性
    acc = (pred == censor).sum() / len(censor)
    return acc


def time_dependent_auc(hazard_pred: np.ndarray, censor: np.ndarray, survtime: np.ndarray, time_point: float) -> float:
    """
    计算特定时间点的时间依赖性AUC。
    
    Args:
        hazard_pred: 风险预测值
        censor: 事件指示器 (1=事件发生, 0=审查)
        survtime: 生存时间
        time_point: 评估的时间点
    
    Returns:
        该时间点的AUC值
    """
    # 在时间点t之前发生事件的患者标记为正例
    # 在时间点t之后仍存活（包括审查）的患者标记为负例
    
    # 筛选有效样本：在时间点t之前发生事件 或 在时间点t之后被观察到
    valid_mask = ((censor == 1) & (survtime <= time_point)) | (survtime > time_point)
    
    if not np.any(valid_mask):
        return np.nan
    
    valid_hazard = hazard_pred[valid_mask]
    valid_censor = censor[valid_mask]
    valid_survtime = survtime[valid_mask]
    
    # 定义标签：在时间点t之前发生事件为正例(1)，其他为负例(0)
    labels = ((valid_censor == 1) & (valid_survtime <= time_point)).astype(int)
    
    # 检查是否有正例和负例
    if len(np.unique(labels)) < 2:
        return np.nan
    
    try:
        auc = roc_auc_score(labels, valid_hazard)
        return auc
    except ValueError:
        return np.nan


def integrated_auc(hazard_pred: np.ndarray, censor: np.ndarray, survtime: np.ndarray, 
                  time_points: np.ndarray = None, max_time: float = None) -> float:
    """
    计算积分AUC (I-AUC)，通过在多个时间点计算时间依赖性AUC并积分得到。
    
    Args:
        hazard_pred: 风险预测值
        censor: 事件指示器 (1=事件发生, 0=审查)
        survtime: 生存时间
        time_points: 评估的时间点数组，如果为None则自动生成
        max_time: 最大评估时间，如果为None则使用观察到的最大事件时间

    
    Returns:
        积分AUC值
    """
    # 基本数据验证
    if len(hazard_pred) == 0 or len(censor) == 0 or len(survtime) == 0:
        return np.nan
    
    # 确定评估的时间范围
    if max_time is None:
        # 使用观察到的最大事件时间（排除审查样本）
        event_times = survtime[censor == 1]
        if len(event_times) == 0:
            return np.nan
        max_time = np.max(event_times)
    
    # 生成时间点
    if time_points is None:
        # 使用临床相关的固定时间点：1年、2年、3年、5年（以天为单位）
        time_points = np.array([365, 730, 1095, 1825])  # 1年=365天, 2年=730天, 3年=1095天, 5年=1825天
    
    # 计算每个时间点的AUC
    auc_values = []
    valid_times = []
    
    for t in time_points:
        auc_t = time_dependent_auc(hazard_pred, censor, survtime, t)
        if not np.isnan(auc_t):
            auc_values.append(auc_t)
            valid_times.append(t)
    

    
    if len(auc_values) < 2:
        return np.nan
    
    # 使用梯形法则进行数值积分
    try:
        # 使用scipy.integrate.trapezoid替代已弃用的trapz
        try:
            from scipy.integrate import trapezoid
            integrated_value = trapezoid(auc_values, valid_times)
        except ImportError:
            # 如果scipy版本较老，回退到numpy.trapz
            integrated_value = np.trapz(auc_values, valid_times)
        
        time_range = valid_times[-1] - valid_times[0]
        i_auc = integrated_value / time_range if time_range > 0 else np.nan
        return i_auc
    except Exception as e:
        return np.nan


def clinical_timepoints_auc(hazard_pred: np.ndarray, censor: np.ndarray, survtime: np.ndarray) -> dict:
    """
    计算四个临床时间点（1年、2年、3年、5年）的时间依赖性AUC。
    
    Args:
        hazard_pred: 风险预测值
        censor: 事件指示器 (1=事件发生, 0=审查)
        survtime: 生存时间（以天为单位）
    
    Returns:
        包含四个时间点AUC的字典：{'1-year': auc_1y, '2-year': auc_2y, '3-year': auc_3y, '5-year': auc_5y}
    """
    # 定义四个临床时间点（以天为单位）
    timepoints = {
        '1-year': 365,
        '2-year': 730, 
        '3-year': 1095,
        '5-year': 1825
    }
    
    results = {}
    for label, timepoint in timepoints.items():
        auc = time_dependent_auc(hazard_pred, censor, survtime, timepoint)
        results[label] = auc
    
    return results


def brier_score_at_time(hazard_pred: np.ndarray, censor: np.ndarray, survtime: np.ndarray, time_point: float) -> float:
    """
    计算在特定时间点的 Brier Score。
    
    Brier Score 衡量预测概率与实际结果之间的差异。
    对于生存分析，在时间 t，Brier Score = (1/n) * Σ[w_i * (S_i(t) - Ŝ_i(t))^2]
    其中 S_i(t) 是真实生存状态，Ŝ_i(t) 是预测生存概率，w_i 是 IPCW 权重。
    
    Args:
        hazard_pred: 风险预测值 (越高风险越大)
        censor: 事件指示器 (1=事件发生, 0=审查)
        survtime: 生存时间
        time_point: 计算 Brier Score 的时间点
    
    Returns:
        Brier Score 值 (越小越好)
    """
    if len(hazard_pred) == 0 or time_point <= 0:
        return np.nan
    
    # 转换为生存概率（简化版本：使用风险预测的逆sigmoid变换）
    # 这里假设 hazard_pred 已经是对数风险比
    survival_prob = 1 / (1 + np.exp(hazard_pred))  # sigmoid的逆变换
    
    # 真实生存状态：在 time_point 时刻是否仍然生存
    true_survival = (survtime > time_point).astype(float)
    
    # 简化的权重计算（实际应该使用 IPCW，这里使用简化版本）
    # 对于观察到事件且事件时间 <= time_point 的样本，权重为 1
    # 对于审查且审查时间 > time_point 的样本，权重为 1
    # 其他情况权重为 0（审查且审查时间 <= time_point）
    weights = np.ones(len(hazard_pred))
    
    # 审查且审查时间 <= time_point 的样本权重设为 0（信息不完整）
    censored_before_time = (censor == 0) & (survtime <= time_point)
    weights[censored_before_time] = 0
    
    # 计算 Brier Score
    if np.sum(weights) == 0:
        return np.nan
    
    squared_errors = (true_survival - survival_prob) ** 2
    weighted_errors = weights * squared_errors
    
    brier_score = np.sum(weighted_errors) / np.sum(weights)
    return brier_score


def integrated_brier_score(hazard_pred: np.ndarray, censor: np.ndarray, survtime: np.ndarray, 
                           time_points: np.ndarray = None, max_time: float = None) -> float:
    """
    计算积分 Brier Score (IBS)，在多个时间点上计算 Brier Score 并积分。
    
    Args:
        hazard_pred: 风险预测值 (越高风险越大)
        censor: 事件指示器 (1=事件发生, 0=审查)
        survtime: 生存时间
        time_points: 计算 Brier Score 的时间点数组，如果为 None 则自动生成
        max_time: 最大时间点，如果为 None 则使用观察到的最大事件时间的75%
        use_clinical_timepoints: 是否使用临床相关的固定时间点（1年、2年、3年、5年）
    
    Returns:
        积分 Brier Score 值 (越小越好)
    """
    # 数据验证
    if len(hazard_pred) == 0:
        return np.nan
    
    # 检查是否有事件发生
    events = censor == 1
    if not np.any(events):
        return np.nan
    
    # 获取事件时间
    event_times = survtime[events]
    
    # 设置最大时间（使用事件时间的75%分位数，避免稀疏数据区域）
    if max_time is None:
        max_time = np.percentile(event_times, 75)
    
    if max_time <= 0 or not np.isfinite(max_time):
        return np.nan
    
    # 生成时间点
    if time_points is None:
        # 使用临床相关的固定时间点：1年、2年、3年、5年（以天为单位）
        time_points = np.array([365, 730, 1095, 1825])  # 1年=365天, 2年=730天, 3年=1095天, 5年=1825天
    else:
        time_points = time_points[time_points <= max_time]
    
    if len(time_points) < 2:
        return np.nan
    
    # 计算每个时间点的 Brier Score
    brier_values = []
    valid_time_points = []
    
    for t in time_points:
        brier = brier_score_at_time(hazard_pred, censor, survtime, t)
        if not np.isnan(brier):
            brier_values.append(brier)
            valid_time_points.append(t)
    
    if len(brier_values) < 2:
        return np.nan
    
    # 使用梯形法则进行积分
    try:
        # 尝试使用 scipy.integrate.trapezoid (新版本)
        if hasattr(integrate, 'trapezoid'):
            integrated_brier_value = integrate.trapezoid(brier_values, valid_time_points)
        else:
            # 回退到 numpy.trapz
            integrated_brier_value = np.trapz(brier_values, valid_time_points)
        
        # 归一化（除以时间范围）
        time_range = valid_time_points[-1] - valid_time_points[0]
        if time_range > 0:
            return integrated_brier_value / time_range
        else:
            return np.nan
    except Exception as e:
        return np.nan


# =======================================================================================
# 2. 模型工具函数
# =======================================================================================

def init_max_weights(module: nn.Module):
    """
    递归地使用 Kaiming (He) Normal 初始化线性层的权重。
    """
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Bilinear)):
            # Kaiming 初始化对于 ReLU/GELU 等激活函数效果很好
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def dfs_freeze(model: nn.Module):
    """
    递归地冻结一个模型的所有参数 (param.requires_grad = False)。
    """
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def count_parameters(model: nn.Module) -> int:
    """
    计算一个模型中可训练参数的数量。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_l1_reg(model: nn.Module, opt, device: torch.device) -> torch.Tensor:
    """
    根据 opt.reg_type 计算 L1 正则化损失。
    """
    reg_type = getattr(opt, 'reg_type', 'rad').lower()
    if reg_type == 'none':
        return torch.tensor(0.0, device=device)
        
    l1_reg = torch.tensor(0.0, device=device)
    
    # 处理 DataParallel 包装的模型
    model_to_reg = model.module if hasattr(model, 'module') else model
    
    # 定义需要正则化的模块前缀
    target_prefixes = []
    if reg_type == 'fusion':
        # 正则化除编码器外的所有层
        for name, _ in model_to_reg.named_parameters():
             if not name.startswith('path_encoder.') and not name.startswith('rad_encoder.'):
                param = dict(model_to_reg.named_parameters())[name]
                if param.requires_grad:
                    l1_reg = l1_reg + torch.norm(param, 1)
        return l1_reg # 直接返回，因为逻辑特殊
    
    elif reg_type == 'path':
        target_prefixes.append('path_encoder.')
        if getattr(opt, 'mode', None) == 'path_only':
            # 如果是单模态模型，所有参数都属于这个模态
            target_prefixes = [''] 
    elif reg_type == 'rad':
        target_prefixes.append('rad_encoder.')
        if getattr(opt, 'mode', None) == 'rad_only':
            target_prefixes = ['']
            
    # 对目标模块施加正则化
    for name, param in model_to_reg.named_parameters():
        if param.requires_grad:
            if any(name.startswith(p) for p in target_prefixes):
                l1_reg = l1_reg + torch.norm(param, 1)
                
    return l1_reg