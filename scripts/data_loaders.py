# data_loaders.py
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from config import N_SPLITS, RANDOM_STATE

class MriWsiDataset(Dataset):
    """
    简化的多模态数据集类，参考PathomicFusion的结构
    支持MRI和WSI特征的加载和预处理（优化版本，支持数据预加载和缓存）
    """
    def __init__(self, opt, data_dict, split='train', scalers=None, enable_cache=True):
        self.opt = opt
        self.split = split
        self.enable_cache = enable_cache
        self.cache = {} if enable_cache else None
        
        # 提取数据
        self.x_path = np.array(data_dict[split]['x_path'])
        self.x_rad = np.array(data_dict[split]['x_rad'])
        self.e = np.array(data_dict[split]['e'])  # 事件状态
        self.t = np.array(data_dict[split]['t'])  # 生存时间
        self.g = np.array(data_dict[split]['g'])  # 分组信息
        
        # 数据标准化
        if scalers is None:
            self.scalers = self._fit_scalers()
        else:
            self.scalers = scalers
            self._transform_data()
        
        # 预加载数据到内存（如果数据集不太大）
        if enable_cache and len(self.x_path) < 1000:  # 小于1000样本时预加载
            self._preload_data()
    
    def _fit_scalers(self):
        """拟合标准化器"""
        scalers = {}
        
        # WSI特征标准化
        scalers['path'] = StandardScaler()
        self.x_path = scalers['path'].fit_transform(self.x_path)
        
        # MRI特征标准化
        scalers['rad'] = StandardScaler()
        self.x_rad = scalers['rad'].fit_transform(self.x_rad)
        
        return scalers
    
    def _transform_data(self):
        """使用预训练的标准化器转换数据"""
        self.x_path = self.scalers['path'].transform(self.x_path)
        self.x_rad = self.scalers['rad'].transform(self.x_rad)
    
    def get_scalers(self):
        """返回标准化器，用于外部测试集"""
        return self.scalers
    
    def __len__(self):
        return len(self.x_path)
    
    def _preload_data(self):
        """预加载所有数据到内存缓存"""
        print(f"预加载 {len(self.x_path)} 个样本到内存缓存...")
        for idx in range(len(self.x_path)):
            if idx not in self.cache:
                x_path = torch.FloatTensor(self.x_path[idx])
                x_rad = torch.FloatTensor(self.x_rad[idx])
                e = torch.FloatTensor([self.e[idx]])
                t = torch.FloatTensor([self.t[idx]])
                g = torch.FloatTensor([self.g[idx]])
                self.cache[idx] = (x_path, x_rad, e, t, g)
        print("数据预加载完成")
    
    def __getitem__(self, idx):
        # 优先从缓存获取数据
        if self.enable_cache and idx in self.cache:
            return self.cache[idx]
        
        # 实时加载数据
        x_path = torch.FloatTensor(self.x_path[idx])
        x_rad = torch.FloatTensor(self.x_rad[idx])
        e = torch.FloatTensor([self.e[idx]])
        t = torch.FloatTensor([self.t[idx]])
        g = torch.FloatTensor([self.g[idx]])
        
        # 如果启用缓存但未预加载，则缓存当前数据
        if self.enable_cache and idx not in self.cache:
            self.cache[idx] = (x_path, x_rad, e, t, g)
        
        return x_path, x_rad, e, t, g

def create_standardized_datasets(opt, data_dict):
    """
    创建标准化的训练和测试数据集
    参考PathomicFusion的create_datasets函数
    """
    # 创建训练数据集（用于拟合标准化器）
    train_dataset = MriWsiDataset(opt, data_dict, split='train')
    
    # 使用训练集的标准化器创建测试数据集
    test_dataset = MriWsiDataset(opt, data_dict, split='test', scalers=train_dataset.get_scalers())
    
    return train_dataset, test_dataset, train_dataset.get_scalers()

def prepare_cross_validation_splits(data_paths, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    """
    准备交叉验证分割
    参考PathomicFusion的数据分割方法
    """
    # 加载数据
    mri_df = pd.read_csv(data_paths['mri'])
    wsi_df = pd.read_csv(data_paths['wsi'])
    labels_df = pd.read_csv(data_paths['labels'])
    
    # 合并数据
    df = pd.merge(pd.merge(mri_df, wsi_df, on='case_id'), labels_df, on='case_id')
    
    # 准备特征列
    mri_cols = [col for col in mri_df.columns if col != 'case_id']
    wsi_cols = [col for col in wsi_df.columns if col != 'case_id']
    
    # 创建交叉验证分割
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for train_idx, test_idx in skf.split(df, df['os.status']):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        split_data = {
            'train': {
                'x_path': [row for row in train_df[wsi_cols].to_numpy()],
                'x_rad': [row for row in train_df[mri_cols].to_numpy()],
                'e': train_df['os.status'].tolist(),
                't': train_df['os.days'].tolist(),
                'g': [0] * len(train_df)
            },
            'test': {
                'x_path': [row for row in test_df[wsi_cols].to_numpy()],
                'x_rad': [row for row in test_df[mri_cols].to_numpy()],
                'e': test_df['os.status'].tolist(),
                't': test_df['os.days'].tolist(),
                'g': [0] * len(test_df)
            }
        }
        splits.append(split_data)
    
    return splits

def save_splits_to_files(splits, output_dir):
    """保存交叉验证分割到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, split_data in enumerate(splits):
        split_path = os.path.join(output_dir, f'split_{i}_data.pkl')
        with open(split_path, 'wb') as f:
            pickle.dump(split_data, f)
        print(f"保存分割 {i} 到 {split_path}")

def load_split_from_file(split_path):
    """从文件加载交叉验证分割"""
    with open(split_path, 'rb') as f:
        return pickle.load(f)