import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
from argparse import Namespace
from utils import init_max_weights, dfs_freeze, count_parameters 
from options import normalize_mode




# =======================================================================================
# Part 1: 工厂函数 (Network Utils)
# =======================================================================================
def define_act_layer(act_type='Tanh'):
    act_type = act_type.lower()
    if act_type == 'sigmoid': return nn.Sigmoid()
    if act_type == 'relu': return nn.ReLU()
    if act_type == 'tanh': return nn.Tanh()
    if act_type == "none": return None
    raise NotImplementedError(f'Activation layer [{act_type}] is not implemented')

def define_optimizer(opt, model):
    beta1, beta2 = getattr(opt, 'beta1', 0.9), getattr(opt, 'beta2', 0.999)
    optimizer_type = opt.optimizer_type.lower()
    if optimizer_type == 'adam': return optim.Adam(model.parameters(), lr=opt.lr, betas=(beta1, beta2), weight_decay=opt.weight_decay)
    elif optimizer_type == 'adamw': return optim.AdamW(model.parameters(), lr=opt.lr, betas=(beta1, beta2), weight_decay=opt.weight_decay)
    raise NotImplementedError(f'Optimizer [{opt.optimizer_type}] is not implemented')

def define_scheduler(opt, optimizer):
    lr_policy = opt.lr_policy.lower()
    if lr_policy == 'linear':
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter + opt.niter_decay, eta_min=opt.lr * 0.01)
    raise NotImplementedError(f'Learning rate policy [{opt.lr_policy}] is not implemented')

def define_net(opt, k):
    device = torch.device(f'cuda:{opt.gpu_ids[0]}') if opt.gpu_ids and torch.cuda.is_available() else torch.device('cpu')
    net = None
    mode = normalize_mode(getattr(opt, 'mode', ''))
    if mode == "PathNet": net = PathNet(opt, input_dim=opt.input_size_path, output_dim=opt.path_dim)
    elif mode == "RadNet": net = RadNet(opt, input_dim=opt.input_size_rad, output_dim=opt.rad_dim)
    elif mode == "EarlyFusionNet": net = SimpleFusionNet(opt, k=k)
    elif mode == "BilinearFusionNet": net = BilinearFusionNet(opt, k=k)
    elif mode == "LateFusionNet" or mode.startswith("LateFusionNet_"): net = LateFusionNet(opt, k=k)
    else: raise NotImplementedError(f'Mode [{getattr(opt, "mode", "undefined")}] is not defined')
    return net.to(device)

# =======================================================================================
# Part 2: 单模态编码器
# =======================================================================================
class PathNet(nn.Module):
    """PathNet: 病理特征编码器（单模态完整模型）"""
    def __init__(self, opt: Namespace, input_dim, output_dim):
        super(PathNet, self).__init__()
        hidden_dim = max(output_dim, int(1.2 * output_dim))
        in_drop = float(getattr(opt, 'path_input_dropout', 0.1))
        drop = float(getattr(opt, 'dropout_rate', 0.25))

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(in_drop)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(drop)

        self.residual = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(output_dim, output_dim)
        )

        self.activation = nn.GELU()
        self.predictor = nn.Linear(output_dim, getattr(opt, 'label_dim', 1))
        self.act = define_act_layer(getattr(opt, 'act_type', 'Tanh'))
        init_max_weights(self)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x = self.input_norm(x_path)
        x = self.input_dropout(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.activation(x)
        x = self.dropout(x)
        features = self.fc2(x)
        features = features + self.residual(features)

        hazard = self.predictor(features)
        if self.act:
            hazard = self.act(hazard)
        return features, hazard

class RadNet(nn.Module):
    """RadNet: MRI 影像特征编码器（单模态完整模型）"""
    def __init__(self, opt: Namespace, input_dim, output_dim):
        super(RadNet, self).__init__()
        hidden_dim = max(output_dim, int(1.2 * output_dim))
        in_drop = float(getattr(opt, 'rad_input_dropout', 0.1))
        drop = float(getattr(opt, 'dropout_rate', 0.25))

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(in_drop)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(drop)

        self.residual = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(output_dim, output_dim)
        )

        self.activation = nn.GELU()
        self.predictor = nn.Linear(output_dim, getattr(opt, 'label_dim', 1))
        self.act = define_act_layer(getattr(opt, 'act_type', 'Tanh'))
        init_max_weights(self)

    def forward(self, **kwargs):
        x_rad = kwargs['x_rad']
        x = self.input_norm(x_rad)
        x = self.input_dropout(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.activation(x)
        x = self.dropout(x)
        features = self.fc2(x)
        features = features + self.residual(features)

        hazard = self.predictor(features)
        if self.act:
            hazard = self.act(hazard)
        return features, hazard

# =======================================================================================
# Part 3: 多模态融合模型
# =======================================================================================
class SimpleFusionNet(nn.Module):
    """
    Early Fusion (特征级融合)
    
    策略：直接拼接两个模态的特征，然后通过 MLP 进行预测。
    """
    def __init__(self, opt: Namespace, k: int = None):
        super().__init__()
        # 使用与单模态相同的编码器
        self.path_encoder = PathNet(opt, opt.input_size_path, opt.path_dim)
        self.rad_encoder = RadNet(opt, opt.input_size_rad, opt.rad_dim)
        self.act = define_act_layer(getattr(opt, 'act_type', 'Tanh'))
        
        # 融合分类器
        d = opt.path_dim
        self.post_fuse = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Linear(2 * d, 4 * d), nn.GELU(), nn.Dropout(opt.dropout_rate), nn.Linear(4 * d, 2 * d)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(opt.dropout_rate * 1.5),
            nn.Linear(2 * d, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(opt.dropout_rate),
            nn.Linear(64, getattr(opt, 'label_dim', 1))
        )
    
    def forward(self, **kwargs):
        # 编码
        path_features, _ = self.path_encoder(**kwargs)
        rad_features, _ = self.rad_encoder(**kwargs)
        
        # 拼接融合
        fused_features = torch.cat([path_features, rad_features], dim=1)
        fused_features = fused_features + self.post_fuse(fused_features)
        
        # 预测
        hazard = self.classifier(fused_features)
        if self.act: 
            hazard = self.act(hazard)
        return (fused_features, None, path_features, rad_features), hazard



class BilinearFusionNet(nn.Module):
    """
    Bilinear Fusion (双线性融合)
    
    基于 MCAT 论文的双线性融合机制：
    1. Gated Multimodal Units: 门控单元让每个模态根据另一模态调节自身
    2. Bilinear Interaction: 通过外积捕获两模态间的二阶交互
    3. Skip Connection: 保留原始特征信息
    """
    def __init__(self, opt: Namespace, k: int = None):
        super().__init__()
        # 使用与单模态相同的编码器
        self.path_encoder = PathNet(opt, opt.input_size_path, opt.path_dim)
        self.rad_encoder = RadNet(opt, opt.input_size_rad, opt.rad_dim)
        self.act = define_act_layer(getattr(opt, 'act_type', 'Tanh'))
        
        # 配置参数
        embed_dim = opt.path_dim
        dropout = opt.dropout_rate
        self.skip = bool(getattr(opt, 'skip', True))
        self.gate_path = bool(getattr(opt, 'gate_path', True))
        self.gate_rad = bool(getattr(opt, 'gate_rad', True))
        
        # 缩放后的维度
        scale_dim1 = int(getattr(opt, 'scale_dim1', 8))
        scale_dim2 = int(getattr(opt, 'scale_dim2', 8))
        dim1 = embed_dim // scale_dim1
        dim2 = embed_dim // scale_dim2
        skip_dim = 2 * embed_dim if self.skip else 0
        
        # ===== Gated Multimodal Units for Path =====
        self.linear_h1 = nn.Sequential(nn.Linear(embed_dim, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(embed_dim, embed_dim, dim1)
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout))
        
        # ===== Gated Multimodal Units for Rad =====
        self.linear_h2 = nn.Sequential(nn.Linear(embed_dim, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(embed_dim, embed_dim, dim2)
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout))
        
        # ===== Bilinear Fusion Layers =====
        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1), embed_dim), nn.ReLU(), nn.Dropout(p=dropout)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(embed_dim + skip_dim, embed_dim), nn.ReLU(), nn.Dropout(p=dropout)
        )
        
        # ===== Classifier =====
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, getattr(opt, 'label_dim', 1))
        )

    def forward(self, **kwargs):
        # 编码
        path_z, _ = self.path_encoder(**kwargs)
        rad_z, _ = self.rad_encoder(**kwargs)
        
        # 双线性融合
        path_orig, rad_orig = path_z, rad_z
        
        # Gated Multimodal Unit for Path
        if self.gate_path:
            h1 = self.linear_h1(path_z)
            z1 = self.linear_z1(path_z, rad_z)
            o1 = self.linear_o1(torch.sigmoid(z1) * h1)
        else:
            o1 = self.linear_o1(path_z)
        
        # Gated Multimodal Unit for Rad
        if self.gate_rad:
            h2 = self.linear_h2(rad_z)
            z2 = self.linear_z2(path_z, rad_z)
            o2 = self.linear_o2(torch.sigmoid(z2) * h2)
        else:
            o2 = self.linear_o2(rad_z)
        
        # Bilinear Fusion: 外积交互
        device = o1.device
        o1 = torch.cat([o1, torch.ones(o1.shape[0], 1, device=device)], dim=1)
        o2 = torch.cat([o2, torch.ones(o2.shape[0], 1, device=device)], dim=1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        
        fused = self.post_fusion_dropout(o12)
        fused = self.encoder1(fused)
        
        # Skip Connection
        if self.skip:
            fused = torch.cat([fused, path_orig, rad_orig], dim=1)
        fused = self.encoder2(fused)
        
        # 预测
        logits = self.classifier(fused)
        if self.act:
            logits = self.act(logits)
        return (fused, None, path_z, rad_z), logits

class LateFusionNet(nn.Module):
    """
    Late Fusion (决策级融合) - 多模态评估的"黄金基准"
    
    设计理念：两个模态分别通过独立的分支预测风险分数，然后在决策层融合。
    与特征级融合不同，模态在特征空间不交互，只在预测结果层面融合。
    
    融合策略：
    - 'avg': 平均融合 - 综合两个模态的意见
    - 'max': 最大融合 - 只要一个模态显示高风险就判定高风险
    - 'weighted': 可学习权重融合 - 自适应学习最优权重
    """
    
    def __init__(self, opt: Namespace, k: int = None):
        super().__init__()
        # 使用与单模态相同的编码器
        self.path_encoder = PathNet(opt, opt.input_size_path, opt.path_dim)
        self.rad_encoder = RadNet(opt, opt.input_size_rad, opt.rad_dim)
        self.act = define_act_layer(getattr(opt, 'act_type', 'Tanh'))
        
        d = opt.path_dim
        dropout = opt.dropout_rate
        self.fusion_mode = getattr(opt, 'late_fusion_mode', 'avg')
        
        # ===== 独立的模态预测头 =====
        self.path_classifier = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, getattr(opt, 'label_dim', 1))
        )
        
        self.rad_classifier = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, getattr(opt, 'label_dim', 1))
        )
        
        # ===== 可学习融合权重（仅 weighted 模式使用）=====
        if self.fusion_mode == 'weighted':
            self.weight_net = nn.Sequential(
                nn.Linear(2 * d, d // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(d // 2, 2),
                nn.Softmax(dim=1)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化分类器权重"""
        for module in [self.path_classifier, self.rad_classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, **kwargs):
        # 编码
        path_z, _ = self.path_encoder(**kwargs)
        rad_z, _ = self.rad_encoder(**kwargs)
        
        # 独立预测各模态的风险分数
        path_hazard = self.path_classifier(path_z)
        rad_hazard = self.rad_classifier(rad_z)
        
        # 决策级融合
        if self.fusion_mode == 'max':
            hazard_fused = torch.maximum(path_hazard, rad_hazard)
            fusion_weights = None
        elif self.fusion_mode == 'weighted':
            joint_repr = torch.cat([path_z, rad_z], dim=1)
            weights = self.weight_net(joint_repr)
            hazard_fused = weights[:, 0:1] * path_hazard + weights[:, 1:2] * rad_hazard
            fusion_weights = weights
        else:  # 'avg'
            hazard_fused = (path_hazard + rad_hazard) / 2.0
            fusion_weights = None
        
        # 激活
        if self.act:
            hazard_fused = self.act(hazard_fused)
        
        # 返回格式保持一致
        fused_features_placeholder = torch.cat([path_z, rad_z], dim=1)
        fusion_info = {
            'path_hazard': path_hazard,
            'rad_hazard': rad_hazard,
            'fusion_weights': fusion_weights,
            'fusion_mode': self.fusion_mode
        }
        
        return (fused_features_placeholder, fusion_info, path_z, rad_z), hazard_fused
