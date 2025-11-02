import random
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser, NeighborSearch, Selection
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
import logging
from scipy.sparse import lil_matrix, coo_matrix, save_npz, load_npz
from Bio.PDB import Atom
import pandas


# 设置日志记录
logging.basicConfig(
    filename='feature_capture.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# 忽略PDB解析中的警告
warnings.simplefilter('ignore', PDBConstructionWarning)

# 定义元素列表
PROTEIN_ELEMENTS = ['C', 'N', 'O', 'S', 'H']
LIGAND_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Unknown']

def load_affinity_data(directory):
    """
    从extracted_data.txt加载亲和力数据
    原代码中这部分逻辑存在于save_features_to_files_sequential()中
    """
    affinity_file = os.path.join(directory, "extracted_data.txt")
    affinity_dict = {}

    if not os.path.exists(affinity_file):
        logging.error(f"Affinity file not found: {affinity_file}")
        return affinity_dict

    try:
        with open(affinity_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 原代码中的解析逻辑
                parts = line.split("\t")
                if len(parts) != 2:
                    logging.warning(f"Invalid line format: {line}")
                    continue
                
                pdb_id, affinity = parts
                try:
                    affinity_dict[pdb_id] = float(affinity)
                except ValueError:
                    logging.warning(f"Invalid affinity value for {pdb_id}: {affinity}")
    
    except Exception as e:
        logging.error(f"Failed to load affinity data: {str(e)}")
    
    return affinity_dict

# 定义键长和键能（示例值，需根据具体需求调整）
BOND_LENGTHS = {
    ('C', 'N'): (1.33, 300),
    ('C', 'C'): (1.54, 280),
    ('C', 'O'): (1.43, 320),
    ('N', 'N'): (1.45, 275),
    ('N', 'O'): (1.40, 350),
    ('O', 'O'): (1.48, 210),
    ('C', 'H'): (1.09, 340),
    ('N', 'H'): (1.01, 350),
    ('O', 'H'): (0.96, 450),
    ('S', 'S'): (2.05, 200),
    ('C', 'S'): (1.82, 250),
    ('N', 'S'): (1.72, 275),
    ('O', 'S'): (1.70, 280),
    ('P', 'O'): (1.61, 320),
    ('P', 'N'): (1.77, 290),
    ('P', 'S'): (2.10, 260),
    ('C', 'P'): (1.87, 260),
    ('N', 'P'): (1.77, 275),
}

# 定义角度和扭转参数（示例值，需根据具体需求调整）
ANGLE_PARAMS = {
    'k_theta': 50,  # 力常数
    'theta0': np.radians(109.5)  # 理想角度（弧度）
}
TORSION_PARAMS = {
    'Vn': [4, 2, 3],  # 力常数列表
    'gamma': [0, np.pi, np.pi / 2],  # 相位列表
    'n': [1, 2, 3]  # 周期数列表
}

# 最大原子数量
MAX_PROTEIN_ATOMS = 512  # 根据需要调整
MAX_LIGAND_ATOMS = 64


def one_hot_encode(elements, element_list):
    """
    对元素列表进行One-Hot编码。

    参数:
    - elements (list): 元素符号列表。
    - element_list (list): 可能的元素符号列表。

    返回:
    - numpy.ndarray: One-Hot编码矩阵。
    """
    encoding = np.zeros((len(elements), len(element_list)), dtype=int)
    element_to_index = {element: idx for idx, element in enumerate(element_list)}
    for i, element in enumerate(elements):
        index = element_to_index.get(element, -1)
        if index != -1:
            encoding[i, index] = 1
        else:
            # 如果元素不在预定义列表中，编码为'Unknown'或全0
            # 这里假设'Unknown'在列表的最后一位
            if 'Unknown' in element_list:
                encoding[i, -1] = 1
            else:
                # 如果没有'Unknown'，则保持全0
                pass
    return encoding


def save_spatial_data(protein_coords, ligand_coords, pdb_output_dir, pdb_id):
    """
    保存蛋白质和配体的空间坐标到一个.npz文件中。

    参数:
    - protein_coords (numpy.ndarray): 蛋白质原子的坐标矩阵 (num_protein_atoms, 3)。
    - ligand_coords (numpy.ndarray): 配体原子的坐标矩阵 (num_ligand_atoms, 3)。
    - pdb_output_dir (str): 输出目录路径。
    - pdb_id (str): 当前处理的PDB ID。
    """
    spatial_data = {
        'protein_coords': protein_coords,
        'ligand_coords': ligand_coords
    }
    file_path = os.path.join(pdb_output_dir, f"{pdb_id}_spatial_data.npz")
    np.savez_compressed(file_path, **spatial_data)


def calculate_angle(v1, v2):
    """计算两个向量之间的夹角，返回弧度值"""
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    return np.arccos(np.clip(dot_product, -1.0, 1.0))


def initialize_sparse_matrix(rows, cols):
    """初始化稀疏矩阵"""
    return lil_matrix((rows, cols), dtype=np.float32)


def calculate_bond_energy_matrix(atoms1, atoms2, bond_lengths, max_atoms1, max_atoms2):
    """
    计算两组原子之间的键能矩阵（稀疏矩阵）
    """
    bond_matrix = initialize_sparse_matrix(max_atoms1, max_atoms2)
    num_atoms1 = min(len(atoms1), max_atoms1)
    num_atoms2 = min(len(atoms2), max_atoms2)
    for i in range(num_atoms1):
        atom1 = atoms1[i]
        # 获取元素符号和坐标
        if isinstance(atom1, Chem.Atom):
            element1 = atom1.GetSymbol()
            pos1 = np.array([
                atom1.GetOwningMol().GetConformer().GetAtomPosition(atom1.GetIdx()).x,
                atom1.GetOwningMol().GetConformer().GetAtomPosition(atom1.GetIdx()).y,
                atom1.GetOwningMol().GetConformer().GetAtomPosition(atom1.GetIdx()).z
            ])
        else:
            element1 = atom1.element
            pos1 = np.array(atom1.coord)

        for j in range(num_atoms2):
            atom2 = atoms2[j]
            # 获取元素符号和坐标
            if isinstance(atom2, Chem.Atom):
                element2 = atom2.GetSymbol()
                pos2 = np.array([
                    atom2.GetOwningMol().GetConformer().GetAtomPosition(atom2.GetIdx()).x,
                    atom2.GetOwningMol().GetConformer().GetAtomPosition(atom2.GetIdx()).y,
                    atom2.GetOwningMol().GetConformer().GetAtomPosition(atom2.GetIdx()).z
                ])
            else:
                element2 = atom2.element
                pos2 = np.array(atom2.coord)

            if element1 is None or element2 is None:
                continue
            atom_pair = (element1, element2)
            # 考虑元素对的顺序
            if atom_pair not in bond_lengths:
                atom_pair = (element2, element1)
            if atom_pair in bond_lengths:
                r0, k = bond_lengths[atom_pair]
                distance = np.linalg.norm(pos1 - pos2)
                energy = 0.5 * k * (distance - r0) ** 2
                bond_matrix[i, j] = energy
    return bond_matrix


def calculate_angle_bending_matrix(atoms1, atoms2, angle_indices, k_theta, theta0, max_atoms1, max_atoms2):
    """
    计算两组原子之间的角度能量矩阵（稀疏矩阵）
    """
    angle_matrix = initialize_sparse_matrix(max_atoms1, max_atoms2)
    num_atoms1 = min(len(atoms1), max_atoms1)
    num_atoms2 = min(len(atoms2), max_atoms2)
    for angle in angle_indices:
        i, j, k = angle
        if i >= num_atoms1 or j >= num_atoms1 or k >= num_atoms1:
            continue
        atom_i = atoms1[i]
        atom_j = atoms1[j]
        atom_k = atoms1[k]
        # 获取位置
        if isinstance(atom_i, Chem.Atom):
            pos_i = np.array([
                atom_i.GetOwningMol().GetConformer().GetAtomPosition(atom_i.GetIdx()).x,
                atom_i.GetOwningMol().GetConformer().GetAtomPosition(atom_i.GetIdx()).y,
                atom_i.GetOwningMol().GetConformer().GetAtomPosition(atom_i.GetIdx()).z
            ])
        else:
            pos_i = np.array(atom_i.coord)

        if isinstance(atom_j, Chem.Atom):
            pos_j = np.array([
                atom_j.GetOwningMol().GetConformer().GetAtomPosition(atom_j.GetIdx()).x,
                atom_j.GetOwningMol().GetConformer().GetAtomPosition(atom_j.GetIdx()).y,
                atom_j.GetOwningMol().GetConformer().GetAtomPosition(atom_j.GetIdx()).z
            ])
        else:
            pos_j = np.array(atom_j.coord)

        if isinstance(atom_k, Chem.Atom):
            pos_k = np.array([
                atom_k.GetOwningMol().GetConformer().GetAtomPosition(atom_k.GetIdx()).x,
                atom_k.GetOwningMol().GetConformer().GetAtomPosition(atom_k.GetIdx()).y,
                atom_k.GetOwningMol().GetConformer().GetAtomPosition(atom_k.GetIdx()).z
            ])
        else:
            pos_k = np.array(atom_k.coord)

        ba = pos_i - pos_j
        bc = pos_k - pos_j
        theta = calculate_angle(ba, bc)
        energy = 0.5 * k_theta * (theta - theta0) ** 2
        angle_matrix[i, k] = energy  # 使用i和k作为交互对
    return angle_matrix


def calculate_torsion_rotation_matrix(atoms1, atoms2, torsion_indices, Vn, gamma, max_atoms1, max_atoms2):
    """
    计算两组原子之间的扭转能量矩阵（稀疏矩阵）
    """
    torsion_matrix = initialize_sparse_matrix(max_atoms1, max_atoms2)
    num_atoms1 = min(len(atoms1), max_atoms1)
    num_atoms2 = min(len(atoms2), max_atoms2)
    for torsion in torsion_indices:
        i, j, k, l = torsion
        if i >= num_atoms1 or j >= num_atoms1 or k >= num_atoms1 or l >= num_atoms1:
            continue
        atom_i = atoms1[i]
        atom_j = atoms1[j]
        atom_k = atoms1[k]
        atom_l = atoms1[l]
        # 获取位置
        if isinstance(atom_i, Chem.Atom):
            pos_i = np.array([
                atom_i.GetOwningMol().GetConformer().GetAtomPosition(atom_i.GetIdx()).x,
                atom_i.GetOwningMol().GetConformer().GetAtomPosition(atom_i.GetIdx()).y,
                atom_i.GetOwningMol().GetConformer().GetAtomPosition(atom_i.GetIdx()).z
            ])
        else:
            pos_i = np.array(atom_i.coord)

        if isinstance(atom_j, Chem.Atom):
            pos_j = np.array([
                atom_j.GetOwningMol().GetConformer().GetAtomPosition(atom_j.GetIdx()).x,
                atom_j.GetOwningMol().GetConformer().GetAtomPosition(atom_j.GetIdx()).y,
                atom_j.GetOwningMol().GetConformer().GetAtomPosition(atom_j.GetIdx()).z
            ])
        else:
            pos_j = np.array(atom_j.coord)

        if isinstance(atom_k, Chem.Atom):
            pos_k = np.array([
                atom_k.GetOwningMol().GetConformer().GetAtomPosition(atom_k.GetIdx()).x,
                atom_k.GetOwningMol().GetConformer().GetAtomPosition(atom_k.GetIdx()).y,
                atom_k.GetOwningMol().GetConformer().GetAtomPosition(atom_k.GetIdx()).z
            ])
        else:
            pos_k = np.array(atom_k.coord)

        if isinstance(atom_l, Chem.Atom):
            pos_l = np.array([
                atom_l.GetOwningMol().GetConformer().GetAtomPosition(atom_l.GetIdx()).x,
                atom_l.GetOwningMol().GetConformer().GetAtomPosition(atom_l.GetIdx()).y,
                atom_l.GetOwningMol().GetConformer().GetAtomPosition(atom_l.GetIdx()).z
            ])
        else:
            pos_l = np.array(atom_l.coord)

        # 计算扭转角
        v1 = pos_j - pos_i
        v2 = pos_k - pos_j
        v3 = pos_l - pos_k
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        if n1_norm == 0 or n2_norm == 0:
            continue  # 避免除以零
        n1_unit = n1 / n1_norm
        n2_unit = n2 / n2_norm
        m1 = np.cross(n1_unit, v2 / np.linalg.norm(v2))
        x = np.dot(n1_unit, n2_unit)
        y = np.dot(m1, n2_unit)
        phi = np.arctan2(y, x)

        # 计算扭转能量
        torsion_energy = 0
        for vn, g in zip(Vn, gamma):
            torsion_energy += vn * (1 + np.cos(phi - g))
        torsion_matrix[i, l] = torsion_energy  # 使用i和l作为交互对
    return torsion_matrix


def calculate_hydrogen_bonds_matrix(atoms_protein, mol, distance_cutoff=4.0, angle_cutoff=np.radians(90),
                                     energy_scale=5.0, max_atoms_protein=5000, max_atoms_ligand=200):
    """
    计算蛋白质与配体之间的氢键能量矩阵（稀疏矩阵）
    仅考虑配体中的氢键受体
    """
    hbond_matrix = initialize_sparse_matrix(max_atoms_protein, max_atoms_ligand)
    conformer = mol.GetConformer()

    # 收集配体中的氢键受体（O、N、S原子）
    acceptors = []
    ligand_atoms = list(mol.GetAtoms())  # 转换为列表
    num_ligand_atoms = min(len(ligand_atoms), max_atoms_ligand)
    for atom in ligand_atoms[:num_ligand_atoms]:
        if atom.GetSymbol() in ['O', 'N', 'S']:
            acceptors.append(atom)

    # 如果没有找到氢键受体，输出调试信息
    if len(acceptors) == 0:
        print("No hydrogen acceptors found!")

    # 输出受体信息
    print(f"Found {len(acceptors)} hydrogen acceptors")

    # 遍历所有配体中的氢键受体，计算与蛋白质的相互作用
    for acceptor in acceptors:
        acceptor_idx = acceptor.GetIdx()
        if acceptor_idx >= max_atoms_ligand:
            continue
        pos_acceptor = np.array([
            conformer.GetAtomPosition(acceptor_idx).x,
            conformer.GetAtomPosition(acceptor_idx).y,
            conformer.GetAtomPosition(acceptor_idx).z
        ])

        for i, atom in enumerate(atoms_protein[:max_atoms_protein]):
            if isinstance(atom, Chem.Atom):
                try:
                    pos_protein = np.array([
                        atom.GetOwningMol().GetConformer().GetAtomPosition(atom.GetIdx()).x,
                        atom.GetOwningMol().GetConformer().GetAtomPosition(atom.GetIdx()).y,
                        atom.GetOwningMol().GetConformer().GetAtomPosition(atom.GetIdx()).z
                    ])
                except Exception as e:
                    logging.error(f"Failed to get position for protein atom {i}: {e}")
                    continue
            else:
                pos_protein = np.array(atom.coord)

            # 计算受体和蛋白质原子之间的距离
            distance = np.linalg.norm(pos_protein - pos_acceptor)
            if distance > distance_cutoff:
                continue  # 距离超出阈值，跳过

            # 计算蛋白质原子和受体之间的角度
            vector_PA = pos_acceptor - pos_protein
            norm_PA = vector_PA / np.linalg.norm(vector_PA)

            # 计算能量
            energy = energy_scale / distance
            hbond_matrix[i, acceptor_idx] = energy

    return hbond_matrix



def calculate_van_der_waals_matrix(atoms_protein, mol, epsilon=0.1, sigma=3.5, max_atoms_protein=5000,
                                   max_atoms_ligand=200):
    """
    计算蛋白质与配体之间的范德华力能量矩阵（稀疏矩阵）

    参数:
    atoms_protein (list): 蛋白质的原子列表
    mol (rdkit.Chem.Mol): 配体分子对象
    epsilon (float): 范德华力参数
    sigma (float): 范德华力参数
    max_atoms_protein (int): 蛋白质的最大原子数量
    max_atoms_ligand (int): 配体的最大原子数量

    返回:
    scipy.sparse.lil_matrix: 蛋白质-配体范德华力能量矩阵
    """
    pl_vdw = initialize_sparse_matrix(max_atoms_protein, max_atoms_ligand)
    conformer = mol.GetConformer()

    num_protein_atoms = min(len(atoms_protein), max_atoms_protein)
    ligand_atoms = list(mol.GetAtoms())[:max_atoms_ligand]  # 转换为列表并切片
    selected_protein_atoms = atoms_protein[:num_protein_atoms]

    for i, atom1 in enumerate(selected_protein_atoms):
        if isinstance(atom1, Chem.Atom):
            try:
                pos1 = np.array([
                    atom1.GetOwningMol().GetConformer().GetAtomPosition(atom1.GetIdx()).x,
                    atom1.GetOwningMol().GetConformer().GetAtomPosition(atom1.GetIdx()).y,
                    atom1.GetOwningMol().GetConformer().GetAtomPosition(atom1.GetIdx()).z
                ])
            except Exception as e:
                logging.error(f"Failed to get position for protein atom {i}: {e}")
                continue
        else:
            pos1 = np.array(atom1.coord)
        for j, atom2 in enumerate(ligand_atoms):
            pos2 = np.array([
                conformer.GetAtomPosition(atom2.GetIdx()).x,
                conformer.GetAtomPosition(atom2.GetIdx()).y,
                conformer.GetAtomPosition(atom2.GetIdx()).z
            ])
            distance = np.linalg.norm(pos1 - pos2)
            if distance == 0:
                continue
            r_over_sigma = sigma / distance
            energy = 4 * epsilon * ((r_over_sigma ** 12) - (r_over_sigma ** 6))
            pl_vdw[i, j] = energy
    return pl_vdw



def randomize_charge(atom):
    """
    为每个原子随机分配电荷。假设电荷可以是正或负值。
    例如：碳、氢、氧、氮等原子的电荷。
    """
    try:
        # 尝试使用 RDKit Atom 获取原子符号
        atom_type = atom.GetSymbol()  # 获取原子符号，如 H, C, N, O (RDKit Atom)
    except AttributeError:
        # 如果是 Biopython 的 Atom 对象，使用 get_element 方法
        if isinstance(atom, Atom.Atom):
            atom_type = atom.element # 获取原子符号，如 H, C, N, O (Biopython Atom)
        else:
            raise ValueError("Unknown atom type, cannot get element symbol.")
    if atom_type == 'H':
        # 氢原子通常为 0 或 +1
        return random.choice([0.0, 1.0])
    elif atom_type == 'C':
        return random.choice([0.1, -0.1])
    elif atom_type == 'N':
        # 氮原子通常为 -1 或 +1
        return random.choice([-0.5, 0.5])
    elif atom_type == 'O':
        # 氧原子通常为 -1 或 0
        return random.choice([-0.5, 0.0])
    else:
        # 其他类型的原子默认为中性电荷
        return 0.0

def calculate_electrostatic_interactions_matrix(atoms_protein, mol, epsilon=1, max_atoms_protein=5000,
                                                max_atoms_ligand=200):
    """
    计算蛋白质与配体之间的静电相互作用能量矩阵（稀疏矩阵）
    """
    electrostatic_matrix = initialize_sparse_matrix(max_atoms_protein, max_atoms_ligand)
    K = 8.9875517873681764e9  # 库仑常数，N m²/C²
    conformer = mol.GetConformer()

    num_protein_atoms = min(len(atoms_protein), max_atoms_protein)
    ligand_atoms = list(mol.GetAtoms())
    num_ligand_atoms = min(len(ligand_atoms), max_atoms_ligand)
    selected_protein_atoms = atoms_protein[:num_protein_atoms]
    selected_ligand_atoms = ligand_atoms[:num_ligand_atoms]

    for i, atom1 in enumerate(selected_protein_atoms):
        if isinstance(atom1, Chem.Atom):
            try:
                pos1 = np.array([
                    atom1.GetOwningMol().GetConformer().GetAtomPosition(atom1.GetIdx()).x,
                    atom1.GetOwningMol().GetConformer().GetAtomPosition(atom1.GetIdx()).y,
                    atom1.GetOwningMol().GetConformer().GetAtomPosition(atom1.GetIdx()).z
                ])
                charge1 = randomize_charge(atom1)  # 使用随机化电荷
            except Exception as e:
                logging.error(f"Failed to get position or charge for protein atom {i}: {e}")
                charge1 = 0.0
        else:
            pos1 = np.array(atom1.coord)
            charge1 = randomize_charge(atom1)  # 使用随机化电荷
        if charge1 is None:
            charge1 = 0.0  # 默认电荷为0

        for j, atom2 in enumerate(selected_ligand_atoms):
            pos2 = np.array([
                conformer.GetAtomPosition(atom2.GetIdx()).x,
                conformer.GetAtomPosition(atom2.GetIdx()).y,
                conformer.GetAtomPosition(atom2.GetIdx()).z
            ])
            charge2 = randomize_charge(atom2)  # 使用随机化电荷
            distance = np.linalg.norm(pos1 - pos2)
            if distance == 0:
                continue
            energy = (K * charge1 * charge2) / (epsilon * distance)
            electrostatic_matrix[i, j] = energy
    return electrostatic_matrix


def calculate_features_matrices(atoms_protein, mol, bond_lengths, angle_indices_pp, torsion_indices_pp,
                                angle_indices_ll, torsion_indices_ll, angle_params, torsion_params):
    """
    计算蛋白质与配体的分子力学特征矩阵
    返回包含PP, PL, LL各三个特征矩阵的字典（稀疏矩阵）
    """
    # 蛋白质-蛋白质特征
    pp_bond = calculate_bond_energy_matrix(atoms_protein, atoms_protein, bond_lengths, MAX_PROTEIN_ATOMS,
                                           MAX_PROTEIN_ATOMS)
    pp_angle = calculate_angle_bending_matrix(atoms_protein, atoms_protein, angle_indices_pp, angle_params['k_theta'],
                                              angle_params['theta0'], MAX_PROTEIN_ATOMS, MAX_PROTEIN_ATOMS)
    pp_torsion = calculate_torsion_rotation_matrix(atoms_protein, atoms_protein, torsion_indices_pp,
                                                   torsion_params['Vn'], torsion_params['gamma'], MAX_PROTEIN_ATOMS,
                                                   MAX_PROTEIN_ATOMS)

    # 配体-配体特征
    ligand_atoms = list(mol.GetAtoms())  # 转换为列表
    ll_bond = calculate_bond_energy_matrix(ligand_atoms, ligand_atoms, bond_lengths, MAX_LIGAND_ATOMS, MAX_LIGAND_ATOMS)
    ll_angle = calculate_angle_bending_matrix(ligand_atoms, ligand_atoms, angle_indices_ll, angle_params['k_theta'],
                                              angle_params['theta0'], MAX_LIGAND_ATOMS, MAX_LIGAND_ATOMS)
    ll_torsion = calculate_torsion_rotation_matrix(ligand_atoms, ligand_atoms, torsion_indices_ll, torsion_params['Vn'],
                                                   torsion_params['gamma'], MAX_LIGAND_ATOMS, MAX_LIGAND_ATOMS)

    # 蛋白质-配体特征
    pl_bond = calculate_bond_energy_matrix(atoms_protein, ligand_atoms, bond_lengths, MAX_PROTEIN_ATOMS,
                                           MAX_LIGAND_ATOMS)
    # 由于 PL 角度和扭转索引未生成，设置为零稀疏矩阵
    pl_angle = initialize_sparse_matrix(MAX_PROTEIN_ATOMS, MAX_LIGAND_ATOMS)
    pl_torsion = initialize_sparse_matrix(MAX_PROTEIN_ATOMS, MAX_LIGAND_ATOMS)

    # 蛋白质-配体氢键特征
    pl_hbond = calculate_hydrogen_bonds_matrix(atoms_protein, mol, max_atoms_protein=MAX_PROTEIN_ATOMS,
                                               max_atoms_ligand=MAX_LIGAND_ATOMS)

    # 蛋白质-配体范德华特征
    pl_vdw = calculate_van_der_waals_matrix(atoms_protein, mol, epsilon=0.1, sigma=3.5,
                                            max_atoms_protein=MAX_PROTEIN_ATOMS, max_atoms_ligand=MAX_LIGAND_ATOMS)

    # 蛋白质-配体静电相互作用特征
    pl_electrostatic = calculate_electrostatic_interactions_matrix(atoms_protein, mol, epsilon=1,
                                                                   max_atoms_protein=MAX_PROTEIN_ATOMS,
                                                                   max_atoms_ligand=MAX_LIGAND_ATOMS)

    # 构建特征矩阵字典
    feature_matrices = {
        'pp_bond': pp_bond,
        'pp_angle': pp_angle,
        'pp_torsion': pp_torsion,
        'll_bond': ll_bond,
        'll_angle': ll_angle,
        'll_torsion': ll_torsion,
        'pl_bond': pl_bond,
        'pl_angle': pl_angle,
        'pl_torsion': pl_torsion,
        'pl_vdw': pl_vdw,
        'pl_electrostatic': pl_electrostatic,
        'pl_hbond': pl_hbond
    }
    return feature_matrices


def generate_angle_torsion_indices_rdkit(mol):
    """
    生成 RDKit Mol 对象的角度和扭转索引。

    参数:
    - mol: RDKit Mol 对象

    返回:
    - angle_indices: 列表 of (i, j, k)
    - torsion_indices: 列表 of (i, j, k, l)
    """
    angle_indices = []
    torsion_indices = []

    ligand_atoms = list(mol.GetAtoms())  # 转换为列表
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        # 获取 a1 和 a2 的邻居，排除对方
        neighbors_a1 = [b.GetOtherAtomIdx(a1) for b in mol.GetAtomWithIdx(a1).GetBonds() if b.GetOtherAtomIdx(a1) != a2]
        neighbors_a2 = [b.GetOtherAtomIdx(a2) for b in mol.GetAtomWithIdx(a2).GetBonds() if b.GetOtherAtomIdx(a2) != a1]
        # 生成角度索引
        for n1 in neighbors_a1:
            for n2 in neighbors_a2:
                angle_indices.append((n1, a1, a2))
        # 生成扭转索引
        for n1 in neighbors_a1:
            for n2 in neighbors_a2:
                torsion_indices.append((n1, a1, a2, n2))

    return angle_indices, torsion_indices


def generate_angle_torsion_indices_biopdb(model, distance_threshold=1.6):
    """
    生成 Bio.PDB Model 对象的角度和扭转索引。

    参数:
    - model: Bio.PDB Model 对象
    - distance_threshold: 确定原子之间是否存在键连接的距离阈值（默认为1.6 Å）

    返回:
    - angle_indices: 列表 of (i, j, k)
    - torsion_indices: 列表 of (i, j, k, l)
    """
    angle_indices = []
    torsion_indices = []
    atoms = list(model.get_atoms())
    atom_to_index = {atom: idx for idx, atom in enumerate(atoms)}

    # 初始化 NeighborSearch
    ns = NeighborSearch(atoms)

    # 构建键连接字典
    bonds = {}
    for atom in atoms:
        idx = atom_to_index[atom]
        neighbors = ns.search(atom.get_coord(), distance_threshold, level='A')  # level='A' for Atom
        for neighbor in neighbors:
            neighbor_idx = atom_to_index[neighbor]
            if neighbor_idx != idx:
                # 由于是双向的，只添加一次
                if neighbor_idx not in bonds.get(idx, set()):
                    bonds.setdefault(idx, set()).add(neighbor_idx)
                    bonds.setdefault(neighbor_idx, set()).add(idx)

    # 生成角度索引
    for j in range(len(atoms)):
        neighbors_j = bonds.get(j, set())
        for i in neighbors_j:
            for k in neighbors_j:
                if k != i:
                    angle_indices.append((i, j, k))

    # 生成扭转索引
    for j in range(len(atoms)):
        neighbors_j = bonds.get(j, set())
        for i in neighbors_j:
            neighbors_i = bonds.get(i, set())
            for k in neighbors_j:
                if k != i:
                    neighbors_k = bonds.get(k, set())
                    for l in neighbors_k:
                        if l != j:
                            torsion_indices.append((i, j, k, l))

    return angle_indices, torsion_indices


def calculate_features_matrices(atoms_protein, mol, bond_lengths, angle_indices_pp, torsion_indices_pp,
                                angle_indices_ll, torsion_indices_ll, angle_params, torsion_params):
    """
    计算蛋白质与配体的分子力学特征矩阵
    返回包含PP, PL, LL各三个特征矩阵的字典（稀疏矩阵）
    """
    # 蛋白质-蛋白质特征
    pp_bond = calculate_bond_energy_matrix(atoms_protein, atoms_protein, bond_lengths, MAX_PROTEIN_ATOMS,
                                           MAX_PROTEIN_ATOMS)
    pp_angle = calculate_angle_bending_matrix(atoms_protein, atoms_protein, angle_indices_pp, angle_params['k_theta'],
                                              angle_params['theta0'], MAX_PROTEIN_ATOMS, MAX_PROTEIN_ATOMS)
    pp_torsion = calculate_torsion_rotation_matrix(atoms_protein, atoms_protein, torsion_indices_pp,
                                                   torsion_params['Vn'], torsion_params['gamma'], MAX_PROTEIN_ATOMS,
                                                   MAX_PROTEIN_ATOMS)

    # 配体-配体特征
    ligand_atoms = list(mol.GetAtoms())  # 转换为列表
    ll_bond = calculate_bond_energy_matrix(ligand_atoms, ligand_atoms, bond_lengths, MAX_LIGAND_ATOMS, MAX_LIGAND_ATOMS)
    ll_angle = calculate_angle_bending_matrix(ligand_atoms, ligand_atoms, angle_indices_ll, angle_params['k_theta'],
                                              angle_params['theta0'], MAX_LIGAND_ATOMS, MAX_LIGAND_ATOMS)
    ll_torsion = calculate_torsion_rotation_matrix(ligand_atoms, ligand_atoms, torsion_indices_ll, torsion_params['Vn'],
                                                   torsion_params['gamma'], MAX_LIGAND_ATOMS, MAX_LIGAND_ATOMS)

    # 蛋白质-配体特征
    pl_bond = calculate_bond_energy_matrix(atoms_protein, ligand_atoms, bond_lengths, MAX_PROTEIN_ATOMS,
                                           MAX_LIGAND_ATOMS)
    # 由于 PL 角度和扭转索引未生成，设置为零稀疏矩阵
    pl_angle = initialize_sparse_matrix(MAX_PROTEIN_ATOMS, MAX_LIGAND_ATOMS)
    pl_torsion = initialize_sparse_matrix(MAX_PROTEIN_ATOMS, MAX_LIGAND_ATOMS)

    # 蛋白质-配体氢键特征
    pl_hbond = calculate_hydrogen_bonds_matrix(atoms_protein, mol, max_atoms_protein=MAX_PROTEIN_ATOMS,
                                               max_atoms_ligand=MAX_LIGAND_ATOMS)

    # 蛋白质-配体范德华特征
    pl_vdw = calculate_van_der_waals_matrix(atoms_protein, mol, epsilon=0.1, sigma=3.5,
                                            max_atoms_protein=MAX_PROTEIN_ATOMS, max_atoms_ligand=MAX_LIGAND_ATOMS)

    # 蛋白质-配体静电相互作用特征
    pl_electrostatic = calculate_electrostatic_interactions_matrix(atoms_protein, mol, epsilon=1,
                                                                   max_atoms_protein=MAX_PROTEIN_ATOMS,
                                                                   max_atoms_ligand=MAX_LIGAND_ATOMS)

    # 构建特征矩阵字典
    feature_matrices = {
        'pp_bond': pp_bond,
        'pp_angle': pp_angle,
        'pp_torsion': pp_torsion,
        'll_bond': ll_bond,
        'll_angle': ll_angle,
        'll_torsion': ll_torsion,
        'pl_bond': pl_bond,
        'pl_angle': pl_angle,
        'pl_torsion': pl_torsion,
        'pl_vdw': pl_vdw,
        'pl_electrostatic': pl_electrostatic,
        'pl_hbond': pl_hbond
    }
    return feature_matrices


def process_pdb_id(directory, pdb_id, bond_lengths, angle_params, torsion_params, affinity_dict, output_directory, split_type):
    """
    处理单个PDB ID，计算并保存特征矩阵（稀疏矩阵），
    以及保存原子的One-Hot编码和空间数据。
    """
    protein_file = os.path.join(directory, pdb_id, f"{pdb_id}_protein.pdb")
    ligand_file = os.path.join(directory, pdb_id, f"{pdb_id}_ligand.mol2")

    if not os.path.exists(protein_file) or not os.path.exists(ligand_file):
        logging.warning(f"Missing files for {pdb_id}, skipping.")
        return

    # 解析蛋白质文件
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('Protein', protein_file)
    model = structure[0]

    # 获取配体文件
    mol = Chem.MolFromMol2File(ligand_file, sanitize=False)
    if mol is None:
        logging.error(f"Failed to load molecule from {ligand_file}")
        return

    # 尝试Sanitize分子
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Chem.rdchem.KekulizeException as e:
        logging.error(f"Sanitization failed for {ligand_file}: {e}")
        return
    except Exception as e:
        logging.error(f"Sanitization failed for {ligand_file}: {e}")
        return

    # 添加氢原子
    mol = Chem.AddHs(mol)
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.AssignAtomChiralTagsFromStructure(mol)
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol)

        # 检查是否生成了有效的构象
        if mol.GetNumConformers() == 0:
            logging.error(f"Failed to generate a valid conformer for {ligand_file}, skipping.")
            return
    except Exception as e:
        logging.error(f"Failed to optimize molecule from {ligand_file}: {e}")
        return

    # 获取蛋白质的原子
    atoms_protein = list(model.get_atoms())

    # 生成角度和扭转索引
    angle_indices_pp, torsion_indices_pp = generate_angle_torsion_indices_biopdb(model)
    angle_indices_ll, torsion_indices_ll = generate_angle_torsion_indices_rdkit(mol)
    angle_indices_pl, torsion_indices_pl = [], []

    # 计算特征矩阵
    feature_matrices = calculate_features_matrices(
        atoms_protein, mol, bond_lengths,
        angle_indices_pp, torsion_indices_pp,
        angle_indices_ll, torsion_indices_ll,
        angle_params, torsion_params
    )

    # 获取亲和力标签
    affinity_label = affinity_dict.get(pdb_id, None)

    # 获取蛋白质原子数量和配体 SMILES
    atoms_protein_count = min(len(atoms_protein), MAX_PROTEIN_ATOMS)
    ligand_smiles = Chem.MolToSmiles(mol)

    # 进行One-Hot编码
    protein_elements = [atom.element for atom in atoms_protein[:MAX_PROTEIN_ATOMS]]
    ligand_atoms = list(mol.GetAtoms())[:MAX_LIGAND_ATOMS]
    ligand_elements = [atom.GetSymbol() for atom in ligand_atoms]

    # 进行One-Hot编码
    protein_one_hot = one_hot_encode(protein_elements, PROTEIN_ELEMENTS)
    ligand_one_hot = one_hot_encode(ligand_elements, LIGAND_ELEMENTS)

    # 提取蛋白质和配体的坐标
    protein_coords = []
    for atom in atoms_protein[:MAX_PROTEIN_ATOMS]:
        if hasattr(atom, 'coord'):
            pos = np.array(atom.coord)
        else:
            pos = np.array([
                atom.GetOwningMol().GetConformer().GetAtomPosition(atom.GetIdx()).x,
                atom.GetOwningMol().GetConformer().GetAtomPosition(atom.GetIdx()).y,
                atom.GetOwningMol().GetConformer().GetAtomPosition(atom.GetIdx()).z
            ])
        protein_coords.append(pos)
    protein_coords = np.array(protein_coords)

    ligand_coords = []
    conformer = mol.GetConformer()
    for atom in ligand_atoms:
        pos = np.array([
            conformer.GetAtomPosition(atom.GetIdx()).x,
            conformer.GetAtomPosition(atom.GetIdx()).y,
            conformer.GetAtomPosition(atom.GetIdx()).z
        ])
        ligand_coords.append(pos)
    ligand_coords = np.array(ligand_coords)

    # 创建 PDB ID 的输出目录
    pdb_output_dir = os.path.join(output_directory, split_type, pdb_id)  # 新增split_type层级
    os.makedirs(pdb_output_dir, exist_ok=True)

    # 保存特征矩阵到文件（使用稀疏矩阵的 `.npz` 格式）
    for feature_name, matrix in feature_matrices.items():
        file_path = os.path.join(pdb_output_dir, f"{pdb_id}_{feature_name}.npz")
        save_npz(file_path, matrix.tocoo())

    # 保存附加信息到一个单独的文件，包括One-Hot编码
    info_str = (
        f"PDB_ID: {pdb_id}\n"
        f"Affinity: {affinity_label}\n"
        f"Atoms_Protein: {atoms_protein_count} atoms\n"
        f"Ligand: {ligand_smiles}\n\n"
        f"Protein_OneHot: {protein_one_hot.tolist()}\n"
        f"Ligand_OneHot: {ligand_one_hot.tolist()}\n"
    )
    info_file = os.path.join(pdb_output_dir, f"{pdb_id}_info.txt")
    with open(info_file, 'w') as f:
        f.write(info_str)

    # 保存空间数据
    save_spatial_data(protein_coords, ligand_coords, pdb_output_dir, pdb_id)

    logging.info(f"Processed and saved features for {pdb_id}")


def save_features_to_files_sequential(directory, output_directory):
    """
    顺序处理所有PDB ID，计算并保存特征矩阵（稀疏矩阵）
    """
    affinity_file = os.path.join(directory, "extracted_data.txt")

    if not os.path.exists(affinity_file):
        logging.error(f"Error: {affinity_file} does not exist.")
        return

    # 读取亲和力标签
    affinity_dict = {}
    with open(affinity_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) != 2:
                logging.warning(f"Invalid line format: {line.strip()}")
                continue
            pdb_id, affinity = parts
            try:
                affinity_dict[pdb_id] = float(affinity)
            except ValueError:
                logging.warning(f"Invalid affinity value for {pdb_id}: {affinity}")

    # 获取所有 pdb_id
    pdb_ids = [folder for folder in os.listdir(directory)
               if folder != "extracted_data.txt" and os.path.isdir(os.path.join(directory, folder))]

    # 依次处理每个PDB ID
    for pdb_id in pdb_ids:
        process_pdb_id(
            directory=directory,
            pdb_id=pdb_id,
            bond_lengths=BOND_LENGTHS,
            angle_params=ANGLE_PARAMS,
            torsion_params=TORSION_PARAMS,
            affinity_dict=affinity_dict,
            output_directory=output_directory
        )

    logging.info(f"All features saved to {output_directory}")


def save_features_with_splits(directory, output_root, split_config_dir):
    """
    新的主处理函数，根据划分配置处理数据
    """
    # 加载所有划分配置
    split_files = {
        f.split('_split')[0]: os.path.join(split_config_dir, f)
        for f in os.listdir(split_config_dir) if f.endswith(".csv")
    }

    # 创建全局输出目录
    os.makedirs(output_root, exist_ok=True)

    # 读取亲和力数据
    affinity_dict = load_affinity_data(directory)

    # 处理每个划分方案
    for split_name, split_file in split_files.items():
        print(f"Processing split: {split_name}")

        # 读取划分数据
        df = pd.read_csv(split_file)
        pdb_ids = df['pdb_id'].tolist()

        # 创建对应输出目录
        split_output_dir = os.path.join(output_root, split_name)
        os.makedirs(split_output_dir, exist_ok=True)

        # 处理每个PDB ID
        for pdb_id in pdb_ids:
            process_pdb_id(
                directory=directory,
                pdb_id=pdb_id,
                bond_lengths=BOND_LENGTHS,
                angle_params=ANGLE_PARAMS,
                torsion_params=TORSION_PARAMS,
                affinity_dict=affinity_dict,
                output_directory=split_output_dir,
                split_type=split_name  # 传递划分类型
            )


if __name__ == "__main__":
    # 新配置参数
    data_directory = './refined-set'
    split_config_dir = './proportional_splits'  # 划分文件目录
    output_root = './split_features'  # 新的输出根目录

    # 创建输出目录结构
    os.makedirs(output_root, exist_ok=True)

    # 运行新的处理流程
    save_features_with_splits(
        directory=data_directory,
        output_root=output_root,
        split_config_dir=split_config_dir
    )
    print("All features saved with split configuration")