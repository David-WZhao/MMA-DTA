import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Dataset
import os
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.utils.data import Subset
import torch_scatter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, X_ll, X_pp, y):
        self.X_ll = torch.tensor(X_ll, dtype=torch.float32)
        self.X_pp = torch.tensor(X_pp, dtype=torch.float32)
        # 将标签转换为二维形状 (N, 1)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        assert self.X_ll.ndim == 4, f"LL数据维度错误: {self.X_ll.shape}"
        assert self.X_pp.ndim == 4, f"PP数据维度错误: {self.X_pp.shape}"
        # 新增标签维度验证
        assert self.y.ndim == 2, f"标签维度错误: {self.y.shape}"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_ll[idx].to(device),  # 提前移动数据到设备
            self.X_pp[idx].to(device),
            self.y[idx].to(device)
        )

def load_single_npz(file_path, target_shape):
    """加载单个npz文件并转换为目标形状"""
    with np.load(file_path) as data:
        sparse_mat = sp.coo_matrix(
            (data['data'], (data['row'], data['col'])),
            shape=data['shape']
        )
        dense = sparse_mat.toarray()

    # 调整尺寸到目标形状
    if dense.shape != target_shape:
        padded = np.zeros(target_shape, dtype=np.float32)
        min_h = min(dense.shape[0], target_shape[0])
        min_w = min(dense.shape[1], target_shape[1])
        padded[:min_h, :min_w] = dense[:min_h, :min_w]
    return padded if dense.shape != target_shape else dense

def load_sample_features(folder_path, prefix):
    """加载单个样本的3个特征文件"""
    suffixes = ['torsion', 'angle', 'bond']
    features = []

    for suffix in suffixes:
        file_path = os.path.join(folder_path, f"{prefix}_{suffix}.npz")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"缺少特征文件: {file_path}")

        # 修改点2：明确目标形状
        target_shape = (64, 64) if "ll" in prefix else (512, 512)
        dense = load_single_npz(file_path, target_shape)
        features.append(dense)

    # 合并为多通道 (H, W, C)
    return np.stack(features, axis=-1)

def pad_3d_tensor(tensor, target_shape, pad_value=0):
    """
    三维张量填充函数（适用于通道在前格式：C, H, W）
    tensor: 输入张量 (C, H, W)
    target_shape: 目标形状 (C, H, W)
    pad_value: 填充值
    """
    # 确保输入是numpy数组
    if not isinstance(tensor, np.ndarray):
        tensor = np.array(tensor)

    # 创建目标形状的零矩阵
    padded = np.full(target_shape, pad_value, dtype=np.float32)

    # 计算各维度填充范围
    c = min(tensor.shape[0], target_shape[0])
    h = min(tensor.shape[1], target_shape[1])
    w = min(tensor.shape[2], target_shape[2])

    # 填充数据
    padded[:c, :h, :w] = tensor[:c, :h, :w]
    return padded

def load_data_from_directory(base_dir):
    X_ll, X_pp, y = [], [], []

    # 定义标准形状
    LL_TARGET_SHAPE = (3, 64, 64)  # C, H, W
    PP_TARGET_SHAPE = (3, 512, 512)

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        try:
            # 加载特征
            ll_data = load_sample_features(folder_path, f"{folder}_ll")  # (C,H,W)
            pp_data = load_sample_features(folder_path, f"{folder}_pp")

            # 统一填充维度
            ll_data = pad_3d_tensor(ll_data, LL_TARGET_SHAPE)
            pp_data = pad_3d_tensor(pp_data, PP_TARGET_SHAPE)

            # 加载标签
            with open(os.path.join(folder_path, f"{folder}_info.txt")) as f:
                affinity = float(f.readlines()[1].split()[-1])

            X_ll.append(ll_data)
            X_pp.append(pp_data)
            y.append(affinity)

        except Exception as e:
            print(f"跳过样本 {folder}: {str(e)}")
            continue

    # 转换为numpy数组时添加维度验证
    def safe_array_convert(data_list, target_shape):
        arr = np.stack([pad_3d_tensor(x, target_shape) for x in data_list])
        assert arr.shape[1:] == target_shape, f"维度不匹配: {arr.shape} vs {target_shape}"
        return arr

    X_ll = safe_array_convert(X_ll, LL_TARGET_SHAPE)
    X_pp = safe_array_convert(X_pp, PP_TARGET_SHAPE)
    y = np.array(y, dtype=np.float32)

    return X_ll, X_pp, y

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, values, keys, query):
        batch_size = query.shape[0]
        Q = self.query(query)
        K = self.key(keys)
        V = self.value(values)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim**0.5
        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, V).transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.num_heads * self.head_dim)
        return self.fc_out(out)

class HighwayConv2D(nn.Module):
    """二维高速卷积网络层"""

    def __init__(self, in_channels, kernel_size=3, padding=1):
        super().__init__()
        # 常规变换分支
        self.H = nn.Conv2d(in_channels, in_channels, kernel_size,
                           padding=padding, bias=False)
        # 门控分支
        self.T = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        H = torch.relu(self.H(x))
        T = self.T(x)
        return H * T + x * (1 - T)

class EnhancedLLModel(nn.Module):
    def __init__(self, input_dim=(3, 64, 64)):
        super().__init__()
        # 高速网络部分
        self.highway = nn.Sequential(
            HighwayConv2D(3),
            HighwayConv2D(3),
            nn.BatchNorm2d(3)
        )

        # 原有CNN结构
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 注意力机制部分保持不变
        self.attn = MultiHeadAttention(embed_dim=64, num_heads=4)
        with torch.no_grad():
            x = torch.randn(1, *input_dim)  # 输入形状应为(1,3,64,64)
            x = self.highway(x)
            x = self.cnn(x)

            # 模拟注意力处理流程
            batch, c, h, w = x.shape
            x_flat = x.view(batch, c, h * w).transpose(1, 2)  # (1, h*w, c)
            x_attn = self.attn(x_flat, x_flat, x_flat)  # (1, h*w, c)
            x_attn = x_attn.mean(dim=1)  # (1, c)

            # 获取实际特征维度
            self.feature_dim = x_attn.shape[1]

        self.fc = nn.Linear(self.feature_dim, 128)

    def forward(self, x):
        x = self.highway(x)
        x = self.cnn(x)
        batch, c, h, w = x.shape
        x = x.view(batch, c, -1).transpose(1, 2)
        x = self.attn(x, x, x).mean(dim=1)
        return self.fc(x)

class EnhancedPPModel(nn.Module):
    def __init__(self, input_dim=(512, 512)):
        super().__init__()
        # 增强版高速网络
        self.highway = nn.Sequential(
            HighwayConv2D(3, kernel_size=5, padding=2),
            HighwayConv2D(3),
            nn.BatchNorm2d(3),
            nn.Dropout2d(0.1)
        )

        # 深层CNN结构（修正变量名）
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attn = MultiHeadAttention(embed_dim=256, num_heads=8)

        # 动态计算合并后的特征维度（修正层名称）
        with torch.no_grad():
            test_input = torch.randn(1, 3, *input_dim)

            # 完整前向过程
            x = self.highway(test_input)  # 先通过高速网络
            conv_out = self.cnn(x)        # 修正为self.cnn

            # 全局池化特征
            pooled = self.global_pool(conv_out)
            pooled_feature = pooled.view(1, -1)

            # 注意力特征
            batch, c, h, w = conv_out.shape
            flat = conv_out.view(batch, c, h*w).transpose(1, 2)
            attn_out = self.attn(flat, flat, flat)
            attn_feature = attn_out.mean(dim=1)

            # 合并特征
            combined = torch.cat([pooled_feature, attn_feature], dim=1)
            self.feature_dim = combined.shape[1]

        self.fc1 = nn.Linear(self.feature_dim, 128)

    def forward(self, x):
        x = self.highway(x)
        x = self.cnn(x)  # 修正为self.cnn

        # 全局池化路径
        x_pooled = self.global_pool(x)
        pooled_feature = x_pooled.view(x.size(0), -1)

        # 注意力路径
        batch, c, h, w = x.shape
        x_flat = x.view(batch, c, h*w).transpose(1, 2)
        x_attn = self.attn(x_flat, x_flat, x_flat).mean(dim=1)

        # 特征合并
        x_combined = torch.cat([pooled_feature, x_attn], dim=1)
        return self.fc1(x_combined)

class CombinedModelWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.ll_model = EnhancedLLModel()
        self.pp_model = EnhancedPPModel()
        # 确保输出维度为 (batch_size, 1)
        self.fc_final = nn.Linear(256, 1)

    def forward(self, ll_x, pp_x):
        ll_out = self.ll_model(ll_x)  # shape: (batch, 128)
        pp_out = self.pp_model(pp_x)  # shape: (batch, 128)
        combined = torch.cat([ll_out, pp_out], dim=1)  # (batch, 256)
        # 直接输出二维张量 (batch, 1)
        return self.fc_final(combined)


def pad_tensor(tensor, target_shape, pad_value=0):
    """
    将张量填充到目标形状，不足部分用pad_value填充
    tensor: 输入张量 (任意维度)
    target_shape: 目标形状元组
    pad_value: 填充值
    """
    pads = []
    for i in range(len(tensor.shape)):
        if i >= len(target_shape):
            break
        diff = target_shape[i] - tensor.shape[i]
        if diff > 0:
            pads.extend([0, diff])  # 在右侧填充
        else:
            pads.extend([0, 0])
    pads = pads[::-1]  # 逆序，因为F.pad的参数是从最后一维开始

    if sum(pads) == 0:
        return tensor
    else:
        return F.pad(tensor, pads, value=pad_value)


def load_info_file(file_path):
    """
    从 info.txt 文件中加载蛋白质节点信息、配体节点信息以及目标值 Affinity。
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        protein_onehot = []
        ligand_onehot = []
        affinity = None

        for line in lines:
            line = line.strip()
            if line.startswith("Protein_OneHot"):
                protein_onehot = eval(line.split(":")[1].strip())
            elif line.startswith("Ligand_OneHot"):
                ligand_onehot = eval(line.split(":")[1].strip())
            elif line.startswith("Affinity"):
                affinity = float(line.split(":")[1].strip())

        protein_onehot_tensor = torch.tensor(protein_onehot, dtype=torch.float32)
        ligand_onehot_tensor = torch.tensor(ligand_onehot, dtype=torch.float32)
        affinity_tensor = torch.tensor([affinity], dtype=torch.float32)

        # 蛋白质节点数不足512时填充
        protein_onehot_tensor = pad_tensor(protein_onehot_tensor, (512, 5))
        # 配体节点数不足64时填充（假设配体One-hot是64x10）
        ligand_onehot_tensor = pad_tensor(ligand_onehot_tensor, (64, 10))

        return protein_onehot_tensor, ligand_onehot_tensor, affinity_tensor

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None


def load_molecular_matrices(subdir_path):
    """
    加载分子力学矩阵（如 _pl_vdw, _pl_electrostatic, _pl_hbond 文件）并根据文件名分类返回它们的张量表示。
    """
    matrices = {'vdw': {}, 'hbond': {}, 'electrostatic': {}}
    try:
        for file_name in os.listdir(subdir_path):
            # 只处理文件名以 _pl_electrostatic, _pl_hbond, _pl_vdw 开头的矩阵文件
            if file_name.endswith(".npz") and (
                    'pl_vdw' in file_name or
                    'pl_hbond' in file_name or
                    'pl_electrostatic' in file_name):
                matrix_path = os.path.join(subdir_path, file_name)  # 这里修正路径
                matrix_data = np.load(matrix_path)
                # 读取稀疏矩阵的行、列、数据
                rows = matrix_data['row'].astype(int)
                cols = matrix_data['col'].astype(int)
                data = matrix_data['data'].astype(float)

                # 将稀疏矩阵转换为密集矩阵，假设是512x64的矩阵
                dense_matrix = sp.coo_matrix((data, (rows, cols)), shape=(512, 64))
                dense_matrix = dense_matrix.toarray()  # 转换为密集矩阵

                # 根据文件名中的类型分类
                if 'pl_vdw' in file_name:
                    matrices['vdw'][file_name] = torch.tensor(dense_matrix, dtype=torch.float32)
                elif 'pl_hbond' in file_name:
                    matrices['hbond'][file_name] = torch.tensor(dense_matrix, dtype=torch.float32)
                elif 'pl_electrostatic' in file_name:
                    matrices['electrostatic'][file_name] = torch.tensor(dense_matrix, dtype=torch.float32)

        for category in ['vdw', 'hbond', 'electrostatic']:
            for file_name in matrices[category]:
                matrix_tensor = matrices[category][file_name]
                padded_matrix = pad_tensor(matrix_tensor, (512, 64))
                matrices[category][file_name] = padded_matrix
        return matrices
    except Exception as e:
        print(f"Error loading molecular matrices from {subdir_path}: {e}")
        return {'vdw': {}, 'hbond': {}, 'electrostatic': {}}


def load_spatial_data(subdir_path):
    """
    加载空间数据（例如 _spatial_data.npz 文件）并返回张量表示。
    """
    try:
        spatial_data_path = os.path.join(subdir_path, f"{subdir_path[-4:]}_spatial_data.npz")  # 这里修正路径

        spatial_data = np.load(spatial_data_path)
        # 通过检查文件中的所有数组来动态读取数据
        spatial_data_dict = {}
        for arr_name in spatial_data.files:
            spatial_data_dict[arr_name] = torch.tensor(spatial_data[arr_name], dtype=torch.float32)

        if 'protein_coords' in spatial_data_dict:
            spatial_data_dict['protein_coords'] = pad_tensor(
                spatial_data_dict['protein_coords'], (512, 3)
            )
        if 'ligand_coords' in spatial_data_dict:
            spatial_data_dict['ligand_coords'] = pad_tensor(
                spatial_data_dict['ligand_coords'], (64, 3)
            )
        return spatial_data_dict
    except Exception as e:
        print(f"Error loading spatial data from {subdir_path}: {e}")
        return None


def process_all_info_files(base_directory):
    """
    批量处理指定目录下的所有文件夹中的_info.txt文件，并加载相应的分子力学矩阵。
    每个文件夹包含一个info.txt文件，代表一个个体。
    """
    all_protein_onehot = []
    all_ligand_onehot = []
    all_affinities = []
    all_matrices = []
    all_spatial_data = []

    # 遍历当前目录下的所有文件夹
    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)

        # 只处理文件夹
        if os.path.isdir(subdir_path):
            info_file_path = os.path.join(subdir_path, f"{subdir}_info.txt")

            if os.path.exists(info_file_path):  # 确保info.txt文件存在
                protein_onehot, ligand_onehot, affinity = load_info_file(info_file_path)

                # 如果成功加载数据，则添加到列表
                if protein_onehot is not None:
                    all_protein_onehot.append(protein_onehot)
                    all_ligand_onehot.append(ligand_onehot)
                    all_affinities.append(affinity)

                    # 加载分子力学矩阵
                    matrices = load_molecular_matrices(subdir_path)
                    all_matrices.append(matrices)

                    # 加载空间数据
                    spatial_data = load_spatial_data(subdir_path)
                    if spatial_data is not None:
                        all_spatial_data.append(spatial_data)

    # 如果列表不为空，则进行堆叠
    if all_protein_onehot:
        all_protein_onehot_tensor = torch.stack(all_protein_onehot)
        all_ligand_onehot_tensor = torch.stack(all_ligand_onehot)
        all_affinities_tensor = torch.stack(all_affinities)
        return all_protein_onehot_tensor, all_ligand_onehot_tensor, all_affinities_tensor, all_matrices, all_spatial_data
    else:
        print("No valid info.txt files found.")
        return None, None, None, None, None


def save_molecular_matrices(subdir, matrices, spatial_data):
    """
    保存分子力学矩阵和空间数据到相应的 `.npz` 文件。
    """
    try:
        # 保存矩阵数据（按分类存储：vdw, hbond, electrostatic）
        for category in ['vdw', 'hbond', 'electrostatic']:
            category_matrices = matrices[category]
            for matrix_name, matrix_tensor in category_matrices.items():
                matrix_path = os.path.join(subdir, f"{category}_{matrix_name}")
                np.savez(matrix_path, arr_0=matrix_tensor.numpy())

        # 保存空间数据
        for spatial_name, spatial_tensor in spatial_data.items():
            spatial_data_path = os.path.join(subdir, f"{subdir}_{spatial_name}.npz")
            np.savez(spatial_data_path, arr_0=spatial_tensor.numpy())


    except Exception as e:
        print(f"Error saving matrices for {subdir}: {e}")


class SignedExponentialScaler:
    def __init__(self, epsilon=1e-8, scale_factor=1.0):
        self.epsilon = epsilon
        self.scale_factor = scale_factor

    def transform(self, tensor):
        sign = torch.sign(tensor)
        abs_values = torch.abs(tensor)
        scaled = torch.log1p(abs_values * self.scale_factor)
        return sign * scaled


class MolecularDataset(Dataset):
    def __init__(self, base_dir):
        super().__init__()
        (self.protein_data,
         self.ligand_data,
         self.affinities,
         self.matrices,
         self.spatial_data) = process_all_info_files(base_dir)

    def len(self):
        return len(self.protein_data)

    def get(self, idx):
        data = create_pyg_data(
            protein_feat=self.protein_data[idx],
            ligand_feat=self.ligand_data[idx],
            matrices=self.matrices[idx],
            spatial_data=self.spatial_data[idx],
            affinity=self.affinities[idx]  # 正确传递亲和力
        ).to(device)
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.edge_attr = data.edge_attr.to(device)
        return data


def create_pyg_data(protein_feat, ligand_feat, matrices, spatial_data, affinity):
    protein_feat = protein_feat.to(device)
    ligand_feat = ligand_feat.to(device)
    # 合并坐标
    protein_coords = spatial_data['protein_coords'].to(device)
    ligand_coords = spatial_data['ligand_coords'].to(device)

    # 节点特征矩阵
    num_nodes = 512 + 64
    x = torch.zeros((num_nodes, 15), device=device)
    x[:512, :5] = protein_feat.to(device)
    x[512:, 5:15] = ligand_feat.to(device)

    # 预处理分子相互作用矩阵
    def process_matrix(matrix_dict, matrix_type):
        scaler_config = {
            'vdw': 1e7,
            'electrostatic': 1e9,
            'hbond': 1e6
        }
        raw = next(iter(matrix_dict[matrix_type].values())).to(device)
        scaler = SignedExponentialScaler(scale_factor=scaler_config[matrix_type])
        return scaler.transform(raw).to(device)

    vdw = process_matrix(matrices, 'vdw').to(device)
    hbond = process_matrix(matrices, 'hbond').to(device)
    electro = process_matrix(matrices, 'electrostatic').to(device)

    # 预处理空间距离特征
    distance_matrix = torch.cdist(protein_coords, ligand_coords)

    # 生成边属性
    edge_mask = distance_matrix < 20.0
    rows, ligand_cols = torch.where(edge_mask)  # ligand_cols是0-63的局部索引
    cols = ligand_cols + 512  # 偏移到合并后的节点索引范围(512-575)
    print("边数目", len(rows))

    # 使用预处理后的距离矩阵
    edge_dist = distance_matrix[rows, ligand_cols].unsqueeze(1)  # 使用原始局部索引获取距离
    coord_diff = protein_coords[rows] - ligand_coords[ligand_cols]
    spatial_attr = torch.cat([edge_dist, coord_diff], dim=1)

    inter_attr = torch.stack([
        vdw[rows, ligand_cols],
        hbond[rows, ligand_cols],
        electro[rows, ligand_cols]
    ], dim=1)
    return Data(
        x=x.to(device),
        edge_index=torch.stack([rows, cols], dim=0).to(device),  # 使用修正后的全局索引
        edge_attr=torch.cat([inter_attr, spatial_attr], dim=1),
        y=affinity.to(device),
        num_protein=512,
        num_ligand=64,
        vdw_matrix=vdw,
        hbond_matrix=hbond,
        electro_matrix=electro
    )

class EnhancedGraphTransformerLayer(MessagePassing):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__(aggr='mean')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_per_head = hidden_dim // num_heads
        self.scale_factor = torch.sqrt(torch.tensor(self.dim_per_head, dtype=torch.float32))

        # 调整输入维度以包含分子力学特征 (hidden_dim + 3个分子特征)
        self.q_proj = nn.Sequential(
            nn.Linear((self.hidden_dim * 2) + 3, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.k_proj = nn.Sequential(
            nn.Linear((self.hidden_dim * 2) + 3, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.v_proj = nn.Sequential(
            nn.Linear((self.hidden_dim * 2) + 3, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 空间编码器
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_heads)
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        # 用于特征增强的残差连接
        self.res_connect = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        original_x = x  # 保存原始节点特征
        residual = x
        # 消息传递
        x = self.norm(x)
        out = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            size = (x.size(0), x.size(0))
        )
        # 增强型残差连接（融合原始特征）
        x = residual + out
        fused = torch.cat([x, original_x], dim=-1)
        x = x + self.res_connect(fused)
        # 前馈网络
        x = x + self.ffn(self.norm(x))
        return x

    def message(self, edge_index, x, edge_attr):
        src, dst = edge_index
        src_features = x[src]  # 新增：获取源节点特征
        dst_features = x[dst]  # 新增：获取目标节点特征
        # 合并节点特征与分子特征 [E, hidden_dim+3]
        molecular_feats = edge_attr[:, :3]
        combined_feats = torch.cat([
            src_features,  # 源节点特征
            dst_features,  # 目标节点特征
            molecular_feats  # 分子作用力特征
        ], dim=1)  # 新维度: [E, (hidden_dim*2)+3]
        spatial_feats = edge_attr[:, 3:7]  # [E, 4]
        # 线性投影
        Q = self.q_proj(combined_feats).view(-1, self.num_heads, self.dim_per_head).contiguous()
        K = self.k_proj(combined_feats).view(-1, self.num_heads, self.dim_per_head).contiguous()
        V = self.v_proj(combined_feats).view(-1, self.num_heads, self.dim_per_head).contiguous()

        # 稳定的注意力计算
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale_factor  # [E, H, D] x [E, D, H] => [E, H, H]

        # 修正空间编码维度
        spatial_bias = self.spatial_encoder(spatial_feats).unsqueeze(2)  # [E, H] => [E, H, 1]
        attn = F.softmax(attn_scores + spatial_bias, dim=1)  # [E, H, H]
        weighted_v = torch.matmul(attn, V)  # [E, H, H] x [E, H, D] => [E, H, D]
        return weighted_v.reshape(-1, self.hidden_dim)  # 正确reshape为[E, H*D]

class AffinityGNN(nn.Module):
    def __init__(self, hidden_dim=256, dropout_rate=0.3):
        super().__init__()

        self.protein_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        self.ligand_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        self.feature_unifier = nn.Linear(hidden_dim // 2, hidden_dim)

        self.gt_layers = nn.ModuleList([
            EnhancedGraphTransformerLayer(hidden_dim, num_heads=8)
            for _ in range(6)
        ])

        # 贡献权重分支
        self.contribution_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, 1),
            nn.ELU()
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        ptr = getattr(data, 'ptr', torch.tensor([0, data.num_nodes]))
        x_list = []
        for i in range(len(ptr) - 1):
            start = ptr[i]
            end = ptr[i + 1]
            sample_x = data.x[start:end]

            protein_feats = self.protein_encoder(sample_x[:data.num_protein[i], :5])
            ligand_feats = self.ligand_encoder(sample_x[data.num_protein[i]:, 5:15])
            unified = torch.cat([
                self.feature_unifier(protein_feats),
                self.feature_unifier(ligand_feats)
            ], dim=0)
            x_list.append(unified)

        x = torch.cat(x_list, dim=0)
        edge_index = data.edge_index
        identity = x
        for layer in self.gt_layers:
            x = layer(x, edge_index, data.edge_attr)
            x = 0.7 * x + 0.3 * identity
        batch = getattr(data, 'batch', torch.zeros(x.size(0), device=x.device))
        pooled = global_mean_pool(x, batch)

        contrib_weights = self.contribution_head(pooled)  # (batch_size, 3)
        # affinity_pred = self.fc(pooled)  # 训练时用

        return pooled, contrib_weights


# 2. FeatureExtractorGNN 返回特征和贡献权重
class FeatureExtractorGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = AffinityGNN(hidden_dim=256)

    def forward(self, data):
        gnn_feat, contrib_weights = self.gnn(data)
        return gnn_feat, contrib_weights


# 3. FeatureExtractorLLPP保持不变
class FeatureExtractorLLPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.ll_model = EnhancedLLModel()
        self.pp_model = EnhancedPPModel()

    def forward(self, ll_x, pp_x):
        ll_feat = self.ll_model(ll_x)  # (batch, 128)
        pp_feat = self.pp_model(pp_x)  # (batch, 128)
        return torch.cat([ll_feat, pp_feat], dim=1)  # (batch, 256)


# 4. GateFusion保持不变
class GateFusion(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        self.gnn_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim)
        )
        self.llpp_proj = nn.Linear(input_dim, input_dim)

        with torch.no_grad():
            self.gate[2].bias.data.fill_(0.5)

    def forward(self, gnn_feat, llpp_feat):
        gnn_feat = self.gnn_proj(gnn_feat)
        llpp_feat = self.llpp_proj(llpp_feat)

        combined = torch.cat([gnn_feat, llpp_feat], dim=1)
        gate_weight = self.gate(combined) * 1.2
        gate_weight = torch.clamp(gate_weight, 0, 1)

        fused = gate_weight * gnn_feat + (1 - gate_weight) * llpp_feat
        return fused


# 5. EnhancedFusionModel返回预测和贡献权重
class EnhancedFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = FeatureExtractorLLPP().to(device)
        self.model2 = FeatureExtractorGNN().to(device)
        self.fusion = GateFusion(input_dim=256).to(device)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(512),
            nn.Linear(512, 1)
        ).to(device)

    def forward(self, ll_input, pp_input, gnn_data):
        llpp_feat = self.model1(ll_input, pp_input)  # (batch, 256)
        gnn_feat, contrib_weights = self.model2(gnn_data)  # (batch,256), (batch,3)
        fused = self.fusion(gnn_feat, llpp_feat)
        output = self.classifier(fused)
        return output, contrib_weights

def plot_combined_heatmap(vdw, hbond, electro, weights, true_affinity, pred_affinity, sample_id, save_path):
    vdw_norm = (vdw - np.min(vdw)) / (np.max(vdw) - np.min(vdw) + 1e-8)
    hbond_norm = (hbond - np.min(hbond)) / (np.max(hbond) - np.min(hbond) + 1e-8)
    electro_norm = (electro - np.min(electro)) / (np.max(electro) - np.min(electro) + 1e-8)

    combined = (weights['vdw'] * vdw_norm +
                weights['hbond'] * hbond_norm +
                weights['electrostatic'] * electro_norm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(combined, cmap='inferno')

    plt.title(
              f"True Affinity: {true_affinity:.2f}  Predicted: {pred_affinity:.2f}\n")
    plt.xlabel("Ligand Nodes (64)")
    plt.ylabel("Protein Nodes (512)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()


# 7. 修改 test_fusion_model 函数，支持贡献权重和pearson相关系数
def test_fusion_model(model_path, base_dir, batch_size=2):
    from torch_geometric.loader import DataLoader

    # 加载数据（假设load_data_from_directory和MolecularDataset代码保持不变）
    X_ll, X_pp, y = load_data_from_directory(base_dir)
    ll_dataset = CustomDataset(X_ll, X_pp, y)
    graph_dataset = MolecularDataset(base_dir)

    class FusionDataset(torch.utils.data.Dataset):
        def __init__(self, ll_dataset, graph_dataset):
            assert len(ll_dataset) == len(graph_dataset)
            self.ll_dataset = ll_dataset
            self.graph_dataset = graph_dataset

        def __len__(self):
            return len(self.ll_dataset)

        def __getitem__(self, idx):
            ll_x, pp_x, y = self.ll_dataset[idx]
            graph_data = self.graph_dataset[idx]
            return (ll_x, pp_x, graph_data, idx), y

    def fusion_collate(batch):
        inputs = [item[0] for item in batch]
        targets = torch.stack([item[1] for item in batch]).to(device)

        ll_batch = torch.stack([x[0] for x in inputs])
        pp_batch = torch.stack([x[1] for x in inputs])
        graph_batch = Batch.from_data_list([x[2] for x in inputs])
        indices = torch.tensor([x[3] for x in inputs])

        return (ll_batch, pp_batch, graph_batch, indices), targets

    full_dataset = FusionDataset(ll_dataset, graph_dataset)
    data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=fusion_collate)

    model = EnhancedFusionModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    result_dir = "molecular_heatmaps"
    os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            (ll_in, pp_in, graph_data, indices), y = inputs, targets

            pred, contrib_weights = model(ll_in, pp_in, graph_data)  # pred:(B,1), contrib_weights:(B,3)

            for i, sample_idx in enumerate(indices):
                matrices = graph_dataset.matrices[sample_idx.item()]
                vdw_matrix = next(iter(matrices['vdw'].values())).cpu().numpy().flatten()
                hbond_matrix = next(iter(matrices['hbond'].values())).cpu().numpy().flatten()
                electro_matrix = next(iter(matrices['electrostatic'].values())).cpu().numpy().flatten()

                # 归一化模型贡献权重
                cw = contrib_weights[i].cpu().numpy()
                cw = cw / (cw.sum() + 1e-8)

                weights = {
                    'vdw': cw[0],
                    'hbond': cw[1],
                    'electrostatic': cw[2]
                }

                save_path = os.path.join(result_dir, f"sample_{sample_idx.item()}_combined_heatmap.png")
                plot_combined_heatmap(
                    vdw_matrix.reshape(512, 64),
                    hbond_matrix.reshape(512, 64),
                    electro_matrix.reshape(512, 64),
                    weights,
                    true_affinity=y[i].item(),
                    pred_affinity=pred[i].item(),
                    sample_id=sample_idx.item(),
                    save_path=save_path
                )

            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * batch_size} samples...")

    print(f"All heatmaps saved to {result_dir} directory")


if __name__ == "__main__":
    model_path = "./best_fusion_model_5_21.pth"
    base_dir = "./features_matrices_5_21"
    test_fusion_model(model_path, base_dir)
