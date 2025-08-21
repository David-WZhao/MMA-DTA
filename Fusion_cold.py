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
from tqdm.auto import tqdm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
        return create_pyg_data(
            protein_feat=self.protein_data[idx],
            ligand_feat=self.ligand_data[idx],
            matrices=self.matrices[idx],
            spatial_data=self.spatial_data[idx],
            affinity=self.affinities[idx]  # 正确传递亲和力
        )


def create_pyg_data(protein_feat, ligand_feat, matrices, spatial_data, affinity):

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
        raw = next(iter(matrix_dict[matrix_type].values()))
        scaler = SignedExponentialScaler(scale_factor=scaler_config[matrix_type])
        return scaler.transform(raw)

    vdw = process_matrix(matrices, 'vdw').to(device)
    hbond = process_matrix(matrices, 'hbond').to(device)
    electro = process_matrix(matrices, 'electrostatic').to(device)

    # 预处理空间距离特征
    distance_matrix = torch.cdist(protein_coords, ligand_coords)
    distance_matrix = torch.tanh(distance_matrix / 10.0)  # 压缩到[-1, 1]

    # 生成边属性
    edge_mask = distance_matrix < 20.0
    rows, cols = torch.where(edge_mask)

    # 使用预处理后的距离矩阵
    edge_dist = distance_matrix[rows, cols].unsqueeze(1)
    coord_diff = protein_coords[rows] - ligand_coords[cols]
    spatial_attr = torch.cat([edge_dist, coord_diff], dim=1)

    inter_attr = torch.stack([vdw[rows, cols],
                              hbond[rows, cols],
                              electro[rows, cols]], dim=1)

    return Data(
        x=x,
        edge_index=torch.stack([rows, 512 + cols], dim=1).t().contiguous(),
        edge_attr=torch.cat([inter_attr, spatial_attr], dim=1),
        y=affinity.to(device)
    )


class EnhancedGraphTransformerLayer(MessagePassing):
    def __init__(self, hidden_dim, num_heads=8):  # 增加注意力头数
        super().__init__(aggr='mean')  # 聚合方式改为mean
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_per_head = hidden_dim // num_heads
        # 新增缺失的注意力融合层
        self.fc_attn = nn.Linear(3 * self.dim_per_head, 1)
        # 新增scale_factor的定义
        self.scale_factor = torch.sqrt(torch.tensor(self.dim_per_head, dtype=torch.float32))

        # 增强的QKV投影
        self.q_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.k_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.v_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 增强的空间编码器
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_heads * self.dim_per_head)
        )

        # 新增层归一化
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # 残差连接
        residual = x

        # 层归一化
        x = self.norm(x)

        # 消息传递
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # 前馈网络
        x = residual + x
        x = x + self.ffn(self.norm(x))

        return x

    def message(self, x, edge_index, edge_attr):
        # 生成Q1,K1,V1
        Q1 = self.q_proj(x).view(-1, self.num_heads, self.dim_per_head)
        K1 = self.k_proj(x).view(-1, self.num_heads, self.dim_per_head)
        V1 = self.v_proj(x).view(-1, self.num_heads, self.dim_per_head)

        src, dst = edge_index
        E = src.size(0)  # 边的数量

        # 矩阵乘法改进 (保持维度对齐)
        Q2 = torch.einsum('ehd, ehd -> eh', Q1[src], K1[dst]).unsqueeze(-1)  # [E, H, 1]
        K2 = torch.einsum('ehd, ehd -> eh', K1[dst], Q1[src]).unsqueeze(-1)  # [E, H, 1]

        # 缩放处理
        Q3 = Q2 / self.scale_factor
        K3 = K2 / self.scale_factor

        # 空间特征处理改进
        spatial_feat = self.spatial_encoder(edge_attr[:, 3:7])  # [E, H*d]
        spatial_feat = spatial_feat.view(E, self.num_heads, self.dim_per_head)  # [E, H, d]

        # 维度对齐拼接
        attn_feat = torch.cat([
            Q3.expand(-1, -1, self.dim_per_head),
            K3.expand(-1, -1, self.dim_per_head),
            spatial_feat
        ], dim=-1)  # [E, H, 3*d]

        # 注意力权重计算
        attn_weight = torch.sigmoid(self.fc_attn(attn_feat).squeeze(-1))  # [E, H]
        attn_weight = F.softmax(attn_weight, dim=0)

        # 信息聚合
        return (attn_weight.unsqueeze(-1) * V1[dst]).view(-1, self.hidden_dim)

    def aggregate(self, inputs, index, dim_size=None):
        # 使用PyG内置聚合，自动处理维度
        return torch_scatter.scatter_add(
            inputs,
            index,
            dim=0,
            dim_size=dim_size  # 确保输出形状匹配输入节点数
        )


class AffinityGNN(nn.Module):
    def __init__(self, hidden_dim=256,dropout_rate=0.3):  # 增大隐藏层维度
        super().__init__()

        # 保持原有编码器结构，仅调整输出维度
        self.protein_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.ligand_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 增加Transformer层数（2层→4层）
        self.gt_layers = nn.ModuleList([
            EnhancedGraphTransformerLayer(hidden_dim, num_heads=8)
            for _ in range(4)
        ])

        # 增强预测头
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 1)
        )
        self.dropout = nn.Dropout(dropout_rate)  # 添加dropout层
    def forward(self, data):
        # 编码特征（保持原有逻辑）
        x = torch.cat([
            self.protein_encoder(data.x[:512, :5]),
            self.ligand_encoder(data.x[512:, 5:15])
        ], dim=0)

        # 多层Transformer
        for layer in self.gt_layers:
            x = layer(x, data.edge_index, data.edge_attr)

        # 全局池化
        x = global_mean_pool(x, data.batch)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


def collate_fn(batch):
    return Batch.from_data_list(batch)


class FeatureExtractorLLPP(nn.Module):
    """修改后的特征提取模型（替代原CombinedModelWithAttention）"""

    def __init__(self):
        super().__init__()
        self.ll_model = EnhancedLLModel()
        self.pp_model = EnhancedPPModel()

        # 冻结原模型参数（可选）
        # for param in self.ll_model.parameters():
        #     param.requires_grad = False
        # for param in self.pp_model.parameters():
        #     param.requires_grad = False

    def forward(self, ll_x, pp_x):
        ll_feat = self.ll_model(ll_x)  # (batch, 128)
        pp_feat = self.pp_model(pp_x)  # (batch, 128)
        return torch.cat([ll_feat, pp_feat], dim=1)  # (batch, 256)


class FeatureExtractorGNN(nn.Module):
    """修改后的GNN特征提取器"""

    def __init__(self):
        super().__init__()
        self.gnn = AffinityGNN(hidden_dim=256)
        # 移除最后的预测层
        self.gnn.fc = nn.Identity()

        # # 冻结原模型参数（可选）
        # for param in self.gnn.parameters():
        #     param.requires_grad = False

    def forward(self, data):
        return self.gnn(data)  # (batch, 256)


# 新建融合模型
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = FeatureExtractorLLPP().to(device)  # 显式指定设备
        self.model2 = FeatureExtractorGNN().to(device)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        ).to(device)

    def forward(self, ll_input, pp_input, gnn_data):
        feat1 = self.model1(ll_input, pp_input)  # (batch, 256)
        feat2 = self.model2(gnn_data)  # (batch, 256)
        fused = torch.cat([feat1, feat2], dim=1)  # (batch, 512)
        return self.classifier(fused)


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def pearsonr(preds, targets):
    preds = preds - preds.mean()
    targets = targets - targets.mean()
    covariance = (preds * targets).mean()
    pred_std = torch.sqrt((preds ** 2).mean())
    target_std = torch.sqrt((targets ** 2).mean())
    return (covariance / (pred_std * target_std + 1e-8)).item()


def train_fusion_model():
    # 创建组合数据集
    class FusionDataset(Dataset):
        def __init__(self, ll_dataset, graph_dataset):
            assert len(ll_dataset) == len(graph_dataset)
            self.ll_dataset = ll_dataset
            self.graph_dataset = graph_dataset

        def __len__(self):
            return len(self.ll_dataset)

        def __getitem__(self, idx):
            ll_x, pp_x, y = self.ll_dataset[idx]
            graph_data = self.graph_dataset[idx].to(device)  # 确保图数据设备一致
            return (ll_x, pp_x, graph_data), y


    # 数据加载器配置
    def fusion_collate(batch):
        # 去除冗余的设备转换
        inputs = [item[0] for item in batch]  # (ll, pp, graph) 已在正确设备
        targets = torch.stack([item[1] for item in batch])

        # 重组批次数据
        ll_batch = torch.stack([x[0] for x in inputs])
        pp_batch = torch.stack([x[1] for x in inputs])
        graph_batch = Batch.from_data_list([x[2] for x in inputs])

        return (ll_batch, pp_batch, graph_batch), targets

    train_base_dir = './split_features/train_10%/train_10%'  # 修改为实际路径
    test_base_dir = './split_features/test/test'  # 修改为实际路径

    # 加载传统特征数据
    print("Loading training features...")
    X_ll_train, X_pp_train, y_train = load_data_from_directory(train_base_dir)
    print("Loading testing features...")
    X_ll_test, X_pp_test, y_test = load_data_from_directory(test_base_dir)

    # 创建传统特征数据集
    ll_train_dataset = CustomDataset(X_ll_train, X_pp_train, y_train)
    ll_test_dataset = CustomDataset(X_ll_test, X_pp_test, y_test)

    # 加载图结构数据
    print("Loading training graphs...")
    graph_train_dataset = MolecularDataset(train_base_dir)
    print("Loading testing graphs...")
    graph_test_dataset = MolecularDataset(test_base_dir)

    # 创建融合数据集
    train_dataset = FusionDataset(ll_train_dataset, graph_train_dataset)
    test_dataset = FusionDataset(ll_test_dataset, graph_test_dataset)

    # 数据加载器配置
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=fusion_collate
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        num_workers=0,
        persistent_workers=False,
        collate_fn=fusion_collate
    )

    # 模型初始化 --------------------------------------------------------
    model = FusionModel().to(device)

    # 参数分组优化策略
    optimizer_groups = [
        # 主干网络参数（较低学习率）
        {'params': model.model1.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
        {'params': model.model2.parameters(), 'lr': 1e-4, 'weight_decay': 0.01},
        # 分类器参数（较高学习率）
        {'params': model.classifier.parameters(), 'lr': 3e-4, 'weight_decay': 0.005}
    ]
    optimizer = torch.optim.AdamW(optimizer_groups)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=100
    )

    # 损失函数与混合精度
    criterion = nn.SmoothL1Loss()
    scaler = torch.cuda.amp.GradScaler()

    early_stopper = EarlyStopper(patience=50, min_delta=0.001)

    # 训练循环 --------------------------------------------------------
    best_metrics = {'loss': float('inf'), 'mae': float('inf')}

    for epoch in range(100):
        model.train()
        train_metrics = {'loss': 0.0, 'mae': 0.0}

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            ll_input, pp_input, graph_data = inputs

            # 暂时禁用混合精度进行测试
            # with torch.cuda.amp.autocast():
            outputs = model(ll_input, pp_input, graph_data)
            outputs = outputs.view(-1)  # 转换为1D张量 [batch_size]
            targets = targets.view(-1)  # 确保目标也是1D张量
            loss = criterion(outputs, targets)
            mae = F.l1_loss(outputs, targets)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 参数更新
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            # 记录指标
            train_metrics['loss'] += loss.item()
            train_metrics['mae'] += mae.item()

        # 验证阶段
        model.eval()
        val_metrics = {'loss': 0.0, 'mae': 0.0}
        all_preds = []  # 新增：收集所有预测值
        all_targets = []  # 新增：收集所有真实值

        with torch.no_grad():
            for inputs, targets in test_loader:
                ll_input, pp_input, graph_data = inputs

                outputs = model(ll_input, pp_input, graph_data).squeeze()
                outputs = outputs.view(-1)  # 转换为1D张量 [batch_size]
                targets = targets.view(-1)  # 确保目标也是1D张量
                # 收集数据
                all_preds.append(outputs.detach().cpu().view(-1))  # 确保1D张量
                all_targets.append(targets.detach().cpu().view(-1))

                loss = criterion(outputs, targets)
                mae = F.l1_loss(outputs, targets)

                val_metrics['loss'] += loss.item()
                val_metrics['mae'] += mae.item()

        # 合并所有数据
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # 计算统计指标
        sd = torch.std(all_preds - all_targets).item()  # 残差标准差
        pearson = pearsonr(all_preds, all_targets)  # 皮尔逊系数

        # 计算平均指标
        avg_train_loss = train_metrics['loss'] / len(train_loader)
        avg_train_mae = train_metrics['mae'] / len(train_loader)
        avg_val_loss = val_metrics['loss'] / len(test_loader)
        avg_val_mae = val_metrics['mae'] / len(test_loader)

        # 修改打印信息
        print(f"\nEpoch {epoch + 1:03d}")
        print(f"Train Loss: {avg_train_loss:.4f} | MAE: {avg_train_mae:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | MAE: {avg_val_mae:.4f}")
        print(f"Pearson: {pearson:.4f} | SD: {sd:.4f}")  # 新增指标输出

        # 修改最佳模型保存判断
        if avg_val_loss < best_metrics['loss']:
            best_metrics = {
                'loss': avg_val_loss,
                'mae': avg_val_mae,
                'pearson': pearson,  # 新增
                'sd': sd,  # 新增
                'epoch': epoch + 1
            }
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metrics': best_metrics
            }, 'best_fusion_model.pth')

        # 早停判断
        if early_stopper(avg_val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 最终报告
    print(f"\nTraining completed. Best epoch {best_metrics['epoch']}:")
    print(f"Loss: {best_metrics['loss']:.4f} | MAE: {best_metrics['mae']:.4f}")

if __name__ == "__main__":
    train_fusion_model()