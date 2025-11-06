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
# Part 0: 官方MCAT核心组件
# =======================================================================================
class Attn_Net_Gated(nn.Module):
    """官方MCAT中的门控注意力网络，用于全局表示汇聚"""
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

# SNN_Block类已移除，因为用户的模型只包含rad模态，不需要多signature处理

class BilinearFusion(nn.Module):
    """官方MCAT中的双线性融合模块"""
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=96, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        
        # 保存原始维度用于skip连接
        self.dim1_og = dim1
        self.dim2_og = dim2
        
        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og + dim2_og if skip else 0  # 修复：使用原始维度计算skip_dim
        
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))
        
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))
        
        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        
    def forward(self, vec1, vec2):
        # 保存原始输入用于skip连接
        vec1_orig, vec2_orig = vec1, vec2
        
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            o1 = self.linear_o1(vec1)
            
        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            o2 = self.linear_o2(vec2)
            
        ### Fusion
        # 使用设备兼容的tensor创建方式
        device = o1.device
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=device)), 1)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1, device=device)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: 
            out = torch.cat((out, vec1_orig, vec2_orig), 1)
        out = self.encoder2(out)
        return out

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
    if mode == "rad_only": net = RadNet(opt, input_dim=opt.input_size_rad, output_dim=opt.rad_dim, is_pretrained_phase=True)
    elif mode == "path_only": net = PathNet(opt, input_dim=opt.input_size_path, output_dim=opt.path_dim, is_pretrained_phase=True)
    elif mode == "simple_fusion": net = SimpleFusionNet(opt, k=k)
    elif mode == "multiscale_fusion": net = MultiScaleFusionNet(opt, k=k)
    elif mode == "coattn": net = CoAttentionFusionNet(opt, k=k)
    elif mode == "bilinear_fusion": net = BilinearFusionNet(opt, k=k)
    else: raise NotImplementedError(f'Mode [{getattr(opt, "mode", "undefined")}] is not defined')
    return net.to(device)

# =======================================================================================
# Part 2: 单模态编码器
# =======================================================================================
class PathNet(nn.Module):
    def __init__(self, opt: Namespace, input_dim, output_dim, is_pretrained_phase=False):
        super(PathNet, self).__init__()
        self.opt = opt
        self.is_pretrained_phase = is_pretrained_phase
        hidden_dim = max(output_dim, int(1.2 * output_dim))
        in_drop = float(getattr(opt, 'path_input_dropout', 0.1))
        drop = float(getattr(opt, 'dropout_rate', 0.25))
        feat_drop = float(getattr(opt, 'unimodal_feature_dropout_p', 0.1)) if getattr(opt, 'mode', None) == 'path_only' else 0.0
        noise_std = float(getattr(opt, 'unimodal_input_noise_std', 0.0)) if getattr(opt, 'mode', None) == 'path_only' else 0.0
        norm_type = getattr(opt, 'unimodal_norm_type', 'ln') if getattr(opt, 'mode', None) == 'path_only' else 'ln'

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(in_drop)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim) if norm_type == 'bn' else nn.Identity()
        self.ln1 = nn.LayerNorm(hidden_dim) if norm_type == 'ln' else nn.Identity()
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
        # 轻量 SE 门控
        use_se = bool(getattr(opt, 'unimodal_se_gate', 0)) if getattr(opt, 'mode', None) == 'path_only' else False
        self.se = nn.Sequential(
            nn.Linear(output_dim, max(4, output_dim // 4)), nn.ReLU(), nn.Linear(max(4, output_dim // 4), output_dim), nn.Sigmoid()
        ) if use_se else None
        self.feature_dropout_p = feat_drop
        self.input_noise_std = noise_std

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x = self.input_norm(x_path)
        x = self.input_dropout(x)
        if self.training and self.feature_dropout_p > 0.0:
            mask = (torch.rand_like(x, device=x.device) > self.feature_dropout_p).float()
            x = x * mask
        if self.training and self.input_noise_std > 0.0:
            x = x + self.input_noise_std * torch.randn_like(x)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.ln1(x)
        x = self.activation(x)
        x = self.dropout(x)
        features = self.fc2(x)
        features = features + self.residual(features)
        if self.se is not None:
            gate = self.se(features)
            features = features * gate

        hazard = None
        if self.is_pretrained_phase or getattr(self.opt, 'mode', None) == 'path_only':
            hazard = self.predictor(features)
            if self.act:
                hazard = self.act(hazard)
        return features, hazard

class PathEncoderV1(nn.Module):
    """稳定版 Path 编码器（与当前融合模型行为保持一致）"""
    def __init__(self, opt: Namespace, input_dim, output_dim, is_pretrained_phase=False):
        super(PathEncoderV1, self).__init__()
        self.opt = opt
        self.is_pretrained_phase = is_pretrained_phase
        self.feature_backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(opt.dropout_rate * 0.8),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        self.predictor = nn.Linear(output_dim, getattr(opt, 'label_dim', 1))
        self.act = define_act_layer(getattr(opt, 'act_type', 'Tanh'))
        init_max_weights(self)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        features = self.feature_backbone(x_path)
        hazard = None
        if self.is_pretrained_phase or getattr(self.opt, 'mode', None) == 'path_only':
            hazard = self.predictor(features)
            if self.act:
                hazard = self.act(hazard)
        return features, hazard

class RadNet(nn.Module):
    """RadNet V2：更强的正则与容量控制以缓解过拟合"""
    def __init__(self, opt: Namespace, input_dim, output_dim, is_pretrained_phase=False):
        super(RadNet, self).__init__()
        self.opt = opt
        self.is_pretrained_phase = is_pretrained_phase
        hidden_dim = max(output_dim, int(1.2 * output_dim))
        in_drop = float(getattr(opt, 'rad_input_dropout', 0.1))
        drop = float(getattr(opt, 'dropout_rate', 0.25))
        feat_drop = float(getattr(opt, 'unimodal_feature_dropout_p', 0.1)) if getattr(opt, 'mode', None) == 'rad_only' else 0.0
        noise_std = float(getattr(opt, 'unimodal_input_noise_std', 0.0)) if getattr(opt, 'mode', None) == 'rad_only' else 0.0
        norm_type = getattr(opt, 'unimodal_norm_type', 'ln') if getattr(opt, 'mode', None) == 'rad_only' else 'ln'

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(in_drop)

        # 两层 MLP + 残差块
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim) if norm_type == 'bn' else nn.Identity()
        self.ln1 = nn.LayerNorm(hidden_dim) if norm_type == 'ln' else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(drop)

        # 残差分支（输出维度）
        self.residual = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(output_dim, output_dim)
        )
        # 轻量 SE 门控
        use_se = bool(getattr(opt, 'unimodal_se_gate', 0)) if getattr(opt, 'mode', None) == 'rad_only' else False
        self.se = nn.Sequential(
            nn.Linear(output_dim, max(4, output_dim // 4)), nn.ReLU(), nn.Linear(max(4, output_dim // 4), output_dim), nn.Sigmoid()
        ) if use_se else None
        self.feature_dropout_p = feat_drop
        self.input_noise_std = noise_std

        self.activation = nn.GELU()
        self.predictor = nn.Linear(output_dim, getattr(opt, 'label_dim', 1))
        self.act = define_act_layer(getattr(opt, 'act_type', 'Tanh'))
        init_max_weights(self)

    def forward(self, **kwargs):
        x_rad = kwargs['x_rad']
        x = self.input_norm(x_rad)
        x = self.input_dropout(x)
        # 可选：特征级 Dropout 与噪声注入（训练阶段）
        if self.training and self.feature_dropout_p > 0.0:
            mask = (torch.rand_like(x, device=x.device) > self.feature_dropout_p).float()
            x = x * mask
        if self.training and self.input_noise_std > 0.0:
            x = x + self.input_noise_std * torch.randn_like(x)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.ln1(x)
        x = self.activation(x)
        x = self.dropout(x)
        features = self.fc2(x)
        # 残差细化
        features = features + self.residual(features)
        if self.se is not None:
            gate = self.se(features)
            features = features * gate

        hazard = None
        if self.is_pretrained_phase or getattr(self.opt, 'mode', None) == 'rad_only':
            hazard = self.predictor(features)
            if self.act:
                hazard = self.act(hazard)
        return features, hazard

class RadEncoderV1(nn.Module):
    """稳定版 Rad 编码器（与当前融合模型行为保持一致）"""
    def __init__(self, opt: Namespace, input_dim, output_dim, is_pretrained_phase=False):
        super(RadEncoderV1, self).__init__()
        self.opt = opt
        self.is_pretrained_phase = is_pretrained_phase
        self.feature_backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(getattr(opt, 'dropout_rate', 0.25) * 0.5)
        )
        self.predictor = nn.Linear(output_dim, getattr(opt, 'label_dim', 1))
        self.act = define_act_layer(getattr(opt, 'act_type', 'Tanh'))
        init_max_weights(self)

    def forward(self, **kwargs):
        x_rad = kwargs['x_rad']
        features = self.feature_backbone(x_rad)
        hazard = None
        if self.is_pretrained_phase or getattr(self.opt, 'mode', None) == 'rad_only':
            hazard = self.predictor(features)
            if self.act:
                hazard = self.act(hazard)
        return features, hazard

# =======================================================================================
# Part 3: 多模态融合模型
# =======================================================================================
class BaseFusionModel(nn.Module):
    def __init__(self, opt: Namespace, k: int = None):
        super(BaseFusionModel, self).__init__()
        self.opt = opt
        # 不使用预训练权重，直接训练编码器
        self.path_encoder = PathEncoderV1(opt, opt.input_size_path, opt.path_dim, is_pretrained_phase=False)
        # 使用稳定版 Rad 编码器，确保融合模型效果不受后续 RadNet 修改影响
        self.rad_encoder = RadEncoderV1(opt, opt.input_size_rad, opt.rad_dim, is_pretrained_phase=False)
        self.act = define_act_layer(getattr(opt, 'act_type', 'Tanh'))

        # 统一投影到公共维度，便于稳定融合
        proj_dim = getattr(opt, 'fusion_proj_dim', 128)
        self.pre_fuse_proj_path = nn.Sequential(
            nn.LayerNorm(opt.path_dim), nn.Linear(opt.path_dim, proj_dim)
        )
        self.pre_fuse_proj_rad = nn.Sequential(
            nn.LayerNorm(opt.rad_dim), nn.Linear(opt.rad_dim, proj_dim)
        )
        self.fusion_proj_dim = proj_dim

        # 轻量 MLP 残差块（可选）
        self.use_fuse_mlp = bool(getattr(opt, 'use_fuse_mlp', 1))
        hidden_mult = float(getattr(opt, 'fuse_mlp_hidden_multiplier', 2.0))
        hidden_dim = max(4, int(hidden_mult * proj_dim))
        if self.use_fuse_mlp:
            self.pre_mlp_path = nn.Sequential(
                nn.LayerNorm(proj_dim), nn.Linear(proj_dim, hidden_dim), nn.GELU(), nn.Dropout(opt.dropout_rate), nn.Linear(hidden_dim, proj_dim)
            )
            self.pre_mlp_rad = nn.Sequential(
                nn.LayerNorm(proj_dim), nn.Linear(proj_dim, hidden_dim), nn.GELU(), nn.Dropout(opt.dropout_rate), nn.Linear(hidden_dim, proj_dim)
            )

    def encode_features(self, **kwargs):
        path_features, _ = self.path_encoder(**kwargs)
        rad_features, _ = self.rad_encoder(**kwargs)
        # 投影到统一维度
        path_z = self.pre_fuse_proj_path(path_features)
        rad_z = self.pre_fuse_proj_rad(rad_features)
        if self.use_fuse_mlp:
            path_z = path_z + self.pre_mlp_path(path_z)
            rad_z = rad_z + self.pre_mlp_rad(rad_z)
        return path_z, rad_z
    
    def forward(self, **kwargs): 
        raise NotImplementedError("子类必须实现 forward 方法")

class SimpleFusionNet(BaseFusionModel):
    """融合策略1: 简化的特征拼接，减少过拟合"""
    def __init__(self, opt: Namespace, k: int = None):
        super().__init__(opt, k=k)
        
        # 简化分类器，增强正则化
        d = self.fusion_proj_dim
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
        # 1. 使用基类的编码器
        path_features, rad_features = self.encode_features(**kwargs)
        
        # 2. 本模型的核心逻辑：拼接
        fused_features = torch.cat([path_features, rad_features], dim=1)
        # post-fuse residual refinement
        fused_features = fused_features + self.post_fuse(fused_features)
        
        # 3. 预测
        hazard = self.classifier(fused_features)
        if self.act: hazard = self.act(hazard)
        return (fused_features, None, path_features, rad_features), hazard



class MultiScaleFusionNet(BaseFusionModel):
    """融合策略3: 简化的多尺度注意力融合，防止过拟合"""
    def __init__(self, opt: Namespace, k: int = None):
        super().__init__(opt, k=k)
        
        # 简化多尺度编码器
        # WSI简单编码器（细粒度）
        self.path_encoder_fine = nn.Sequential(
            nn.Linear(opt.input_size_path, opt.path_dim), 
            nn.ReLU(),
            nn.Dropout(opt.dropout_rate * 0.5)
        )
        
        # MRI多尺度编码器（细粒度）— 已离线降维，直接使用恒等
        self.rad_reducer_fine = nn.Identity()
        reducer_out_dim = opt.input_size_rad
        self.rad_encoder_fine = nn.Sequential(
            nn.Linear(reducer_out_dim, opt.rad_dim), 
            nn.ReLU(),
            nn.Dropout(opt.dropout_rate * 0.5)
        )
        
        # 简化注意力网络
        self.scale_attention = nn.Sequential(
            nn.Linear((opt.path_dim + opt.rad_dim) * 2, 64), 
            nn.ReLU(), 
            nn.Dropout(opt.dropout_rate),
            nn.Linear(64, 4), 
            nn.Softmax(dim=1)
        )
        
        # 简化分类器
        self.post_fuse = nn.Sequential(
            nn.LayerNorm(opt.path_dim), nn.Linear(opt.path_dim, 2 * opt.path_dim), nn.GELU(), nn.Dropout(opt.dropout_rate), nn.Linear(2 * opt.path_dim, opt.path_dim)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(opt.dropout_rate),
            nn.Linear(opt.path_dim, 32), 
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(opt.dropout_rate * 0.5),
            nn.Linear(32, getattr(opt, 'label_dim', 1))
        )

    def forward(self, **kwargs):
        x_path, x_rad = kwargs['x_path'], kwargs['x_rad']

        # 1. 使用原始编码器输出作为"粗粒度"特征以保持维度一致
        path_coarse, _ = self.path_encoder(**kwargs)
        rad_coarse, _ = self.rad_encoder(**kwargs)
        
        # 2. 使用本模型的编码器作为"细粒度"特征
        path_fine = self.path_encoder_fine(x_path)
        rad_fine = self.rad_encoder_fine(self.rad_reducer_fine(x_rad))
            
        # 3. 注意力融合
        all_features = torch.cat([path_fine, path_coarse, rad_fine, rad_coarse], dim=1)
        weights = self.scale_attention(all_features)
        
        fused_features = (weights[:, 0].unsqueeze(1) * path_fine + 
                         weights[:, 1].unsqueeze(1) * path_coarse +
                         weights[:, 2].unsqueeze(1) * rad_fine +
                         weights[:, 3].unsqueeze(1) * rad_coarse)
        fused_features = fused_features + self.post_fuse(fused_features)
        
        hazard = self.classifier(fused_features)
        if self.act: hazard = self.act(hazard)
        return (fused_features, None, path_coarse, rad_coarse), hazard


class BilinearFusionNet(BaseFusionModel):
    """纯双线性融合分支：将两模态统一到 embed_dim 后用 BilinearFusion 融合"""
    def __init__(self, opt: Namespace, k: int = None):
        super().__init__(opt, k=k)
        embed_dim = getattr(opt, 'fusion_proj_dim', 128)
        self.mm = BilinearFusion(
            skip=bool(getattr(opt, 'skip', True)),
            use_bilinear=True,
            gate1=bool(getattr(opt, 'gate_path', True)),
            gate2=bool(getattr(opt, 'gate_rad', True)),
            dim1=embed_dim,
            dim2=embed_dim,
            scale_dim1=int(getattr(opt, 'scale_dim1', 8)),
            scale_dim2=int(getattr(opt, 'scale_dim2', 8)),
            mmhid=embed_dim,
            dropout_rate=getattr(opt, 'dropout_rate', 0.25)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, getattr(opt, 'label_dim', 1))
        )
        self.act = define_act_layer(getattr(opt, 'act_type', 'Tanh'))

    def forward(self, **kwargs):
        path_z, rad_z = self.encode_features(**kwargs)
        fused = self.mm(path_z, rad_z)
        logits = self.classifier(fused)
        if self.act:
            logits = self.act(logits)
        return (fused, None, path_z, rad_z), logits

 


class CoAttentionFusionNet(nn.Module):
    """官方MCAT共注意力融合网络 - 完全对齐官方实现"""
    def __init__(self, opt: Namespace, k: int = None):
        super(CoAttentionFusionNet, self).__init__()
        self.opt = opt
        self.act = define_act_layer(getattr(opt, 'act_type', 'Tanh'))
        
        # 获取配置参数
        self.embed_dim = getattr(opt, 'coattn_embed_dim', 256)
        self.num_heads = getattr(opt, 'coattn_num_heads', 1)
        self.num_layers = getattr(opt, 'coattn_num_layers', 2)
        self.fusion_type = getattr(opt, 'fusion_type', 'concat')
        self.n_classes = getattr(opt, 'n_classes', 4)
        self.dropout = getattr(opt, 'dropout', 0.25)
        self.gate_path = getattr(opt, 'gate_path', True)
        self.gate_rad = getattr(opt, 'gate_rad', True)
        self.scale_dim1 = getattr(opt, 'scale_dim1', 8)
        self.scale_dim2 = getattr(opt, 'scale_dim2', 8)
        self.skip = getattr(opt, 'skip', True)
        self.dropinput = getattr(opt, 'dropinput', 0.1)
        self.path_input_dim = getattr(opt, 'path_input_dim', 768)
        self.rad_input_dim = getattr(opt, 'rad_input_dim', 768)
        self.fusion_proj_dim = getattr(opt, 'fusion_proj_dim', None)  # 融合后投影维度
        self.fallback_no_coattn_when_singleton = True
        
        # 输入dropout
        self.path_input_dropout = nn.Dropout(self.dropinput)
        self.rad_input_dropout = nn.Dropout(self.dropinput)
        
        # Path模态处理 - 修复维度不匹配问题
        # Attn_Net_Gated的D参数应该与输入维度匹配，而不是embed_dim
        self.path_attention_head = Attn_Net_Gated(L=self.path_input_dim, D=min(256, self.path_input_dim), dropout=self.dropout > 0, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(self.path_input_dim, self.embed_dim), nn.ReLU(), nn.Dropout(self.dropout)])
        
        # Rad模态处理 - 修复维度不匹配问题
        # Attn_Net_Gated的D参数应该与输入维度匹配，而不是embed_dim
        self.rad_attention_head = Attn_Net_Gated(L=self.rad_input_dim, D=min(256, self.rad_input_dim), dropout=self.dropout > 0, n_classes=1)
        self.rad_rho = nn.Sequential(*[nn.Linear(self.rad_input_dim, self.embed_dim), nn.ReLU(), nn.Dropout(self.dropout)])
        
        # 分支残差MLP，加强非线性表达
        # 回滚：分支MLP恢复为4x宽度
        self.path_branch_mlp = nn.Sequential(
            nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, 4 * self.embed_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(4 * self.embed_dim, self.embed_dim)
        )
        self.rad_branch_mlp = nn.Sequential(
            nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, 4 * self.embed_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(4 * self.embed_dim, self.embed_dim)
        )

        # 可选门控
        self.path_gate_layer = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.Sigmoid()) if self.gate_path else nn.Identity()
        self.rad_gate_layer = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.Sigmoid()) if self.gate_rad else nn.Identity()

        # 多层共注意力机制
        self.coattn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.15, batch_first=True)
            for _ in range(max(1, self.num_layers))
        ])
        self.coattn_norms_1 = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(max(1, self.num_layers))])
        self.coattn_norms_2 = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(max(1, self.num_layers))])
        
        # 融合方式
        if self.fusion_type == 'bilinear':
            self.mm = BilinearFusion(
                skip=self.skip, use_bilinear=True, gate1=self.gate_path, gate2=self.gate_rad,
                dim1=self.embed_dim, dim2=self.embed_dim, scale_dim1=self.scale_dim1, scale_dim2=self.scale_dim2,
                mmhid=self.embed_dim, dropout_rate=self.dropout
            )
            classifier_input_dim = self.embed_dim
        else:  # concat
            classifier_input_dim = 2 * self.embed_dim
        
        # 融合投影层（如果指定）
        if self.fusion_proj_dim is not None:
            self.fusion_proj = nn.Sequential(
                nn.Linear(classifier_input_dim, self.fusion_proj_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            final_input_dim = self.fusion_proj_dim
        else:
            self.fusion_proj = None
            final_input_dim = classifier_input_dim
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(final_input_dim, self.n_classes),
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Parameter):
                nn.init.normal_(m, 0, 0.02)
    
    def forward(self, **kwargs):
        """前向传播 - 完全对齐官方MCAT实现"""
        x_path = kwargs.get('x_path')  # [B, N_path, path_input_dim]
        x_rad = kwargs.get('x_rad') if kwargs.get('x_rad') is not None else kwargs.get('x_omic')   # [B, rad_input_dim] - 兼容x_omic和x_rad参数名
        
        # 输入dropout
        x_path = self.path_input_dropout(x_path)
        x_rad = self.rad_input_dropout(x_rad)
        
        batch_size = x_path.size(0)
        
        # 统一的注意力池化函数（若为单元素序列则退化为线性映射）
        def attention_pooling_with_transform(features, attention_head, transform_layer, is_sequence=True):
            if is_sequence:
                if len(features.shape) == 2:
                    features = features.unsqueeze(1)  # [B, 1, D]
                B, N, feature_dim = features.shape
                if N == 1 and self.fallback_no_coattn_when_singleton:
                    h_pooled = features.squeeze(1)
                    h_transformed = transform_layer(h_pooled)
                    A = torch.ones(B, 1, device=features.device)
                    return h_transformed, A
                features_flat = features.view(-1, feature_dim)
                A_flat, h_flat = attention_head(features_flat)
                A = A_flat.view(B, N, -1).squeeze(-1)
                A = F.softmax(A, dim=1)
                h_reshaped = h_flat.view(B, N, -1)
                h_pooled = torch.bmm(A.unsqueeze(1), h_reshaped).squeeze(1)
                h_transformed = transform_layer(h_pooled)
                return h_transformed, A
            else:
                # 单向量输入：直接线性变换
                h_transformed = transform_layer(features)
                A = torch.ones(features.size(0), 1, device=features.device)
                return h_transformed, A
        
        # Path和Rad模态处理
        batch_size = x_path.size(0)
        h_path, A_path = attention_pooling_with_transform(x_path, self.path_attention_head, self.path_rho, is_sequence=True)
        h_rad, A_rad = attention_pooling_with_transform(x_rad, self.rad_attention_head, self.rad_rho, is_sequence=False)
        
        # 分支残差 MLP 与门控
        h_path = h_path + self.path_branch_mlp(h_path)
        h_rad = h_rad + self.rad_branch_mlp(h_rad)
        if self.gate_path:
            h_path = h_path * self.path_gate_layer(h_path)
        if self.gate_rad:
            h_rad = h_rad * self.rad_gate_layer(h_rad)

        # 多层共注意力（若为单元素序列已在上方退化，仍可进行表示对齐）
        coattn_weights = None
        for layer, n1, n2 in zip(self.coattn_layers, self.coattn_norms_1, self.coattn_norms_2):
            h1_norm = n1(h_path)
            h2_norm = n2(h_rad)
            stacked = torch.stack([h1_norm, h2_norm], dim=1)  # [B, 2, D]
            attn_output, attn_w = layer(stacked, stacked, stacked)
            h_path = h_path + attn_output[:, 0, :]
            h_rad = h_rad + attn_output[:, 1, :]
            coattn_weights = attn_w
        h_path_coattn, h_rad_coattn = h_path, h_rad
        
        # 融合
        if self.fusion_type == 'bilinear':
            h = self.mm(h_path_coattn, h_rad_coattn)  # [B, embed_dim]
        else:  # concat
            h = torch.cat([h_path_coattn, h_rad_coattn], dim=1)  # [B, 2*embed_dim]
        
        # 融合投影（如果启用）
        if self.fusion_proj is not None:
            h = self.fusion_proj(h)  # [B, fusion_proj_dim]
        
        # 分类预测
        logits = self.classifier(h)  # [B, n_classes]
        
        # 返回格式对齐训练代码期望
        features_tuple = (
            h,  # fused_features
            None,  # attn_entropy (暂时为None)
            h_path_coattn,  # path_z
            h_rad_coattn   # rad_z
        )
        
        # 计算对齐损失 (如果需要)
        align_loss = 0.0
        align_loss_weight = getattr(self.opt, 'coattn_align_loss_weight', 0.1)
        if align_loss_weight > 0:
            # 简单的L2对齐损失
            align_loss = F.mse_loss(h_path_coattn, h_rad_coattn) * align_loss_weight
        
        # 返回注意力权重用于可解释性
        attention_weights = {
            'A_path': A_path,
            'A_rad': A_rad,
            'A_coattn': coattn_weights
        }
        
        if self.act:
            logits = self.act(logits)
            
        return features_tuple, logits, align_loss, attention_weights, None
