"""
统一数据集模块 - 支持单粒度和多粒度模式
"""

import torch
import dgl
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
import os
from torch.utils.data import Dataset


class PDBToDGLConverter:
    """将PDB文件转换为DGL图对象"""

    def __init__(self, distance_threshold=8.0):
        """
        Args:
            distance_threshold: 原子间距离阈值(Å)，小于此值的原子对会连边
        """
        self.distance_threshold = distance_threshold
        self.parser = PDBParser(QUIET=True)

        # 原子类型编码
        self.atom_types = {
            'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4,
            'CG': 5, 'CD': 6, 'CE': 7, 'CZ': 8, 'SD': 9,
            'NH1': 10, 'NH2': 11, 'OG': 12, 'OD1': 13, 'OD2': 14,
            'NE': 15, 'NE1': 16, 'NE2': 17, 'ND1': 18, 'ND2': 19,
            'OE1': 20, 'OE2': 21, 'SG': 22, 'CD1': 23, 'CD2': 24,
            'CE1': 25, 'CE2': 26, 'CE3': 27, 'CG1': 28, 'CG2': 29,
            'OH': 30, 'NZ': 31, 'OXT': 32
        }

        # 氨基酸类型编码
        self.residue_types = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
        }

    def pdb_to_graph(self, pdb_file):
        """
        将PDB文件转换为DGL图

        Args:
            pdb_file: PDB文件路径

        Returns:
            g: DGL图对象
                - g.ndata['feat']: 节点特征 [num_atoms, 59]
                - g.ndata['pos']: 原子坐标 [num_atoms, 3]
                - g.edata['dist']: 边特征(距离) [num_edges, 1]
        """
        structure = self.parser.get_structure('peptide', pdb_file)

        atoms_info = []
        coords = []

        # 提取原子信息
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_name = residue.get_resname()
                    res_id = residue.get_id()[1]

                    for atom in residue:
                        atom_name = atom.get_name()
                        coord = atom.get_coord()
                        bfactor = atom.get_bfactor()

                        atoms_info.append({
                            'atom_name': atom_name,
                            'residue_name': res_name,
                            'residue_id': res_id,
                            'bfactor': bfactor,
                            'coord': coord
                        })
                        coords.append(coord)

        coords = np.array(coords)
        num_atoms = len(coords)

        # 构建边列表
        src_nodes = []
        dst_nodes = []
        edge_distances = []

        dist_matrix = cdist(coords, coords)

        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j and dist_matrix[i, j] < self.distance_threshold:
                    src_nodes.append(i)
                    dst_nodes.append(j)
                    edge_distances.append(dist_matrix[i, j])

        # 创建DGL图
        if len(src_nodes) == 0:
            # 如果没有边，添加自环
            src_nodes = list(range(num_atoms))
            dst_nodes = list(range(num_atoms))
            edge_distances = [0.0] * num_atoms

        g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_atoms)

        # 添加节点特征
        node_features = self._encode_atoms(atoms_info, coords)
        g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
        g.ndata['pos'] = torch.tensor(coords, dtype=torch.float32)

        # 添加边特征
        if len(edge_distances) > 0:
            g.edata['dist'] = torch.tensor(edge_distances, dtype=torch.float32).unsqueeze(1)

        return g

    def _encode_atoms(self, atoms_info, coords):
        """
        编码原子特征

        Returns:
            features: numpy array of shape [num_atoms, 60]
        """
        features = []

        for i, atom_info in enumerate(atoms_info):
            feat = []

            # 1. 原子类型 one-hot (34维)
            atom_type = self.atom_types.get(atom_info['atom_name'], 33)
            atom_onehot = np.zeros(34)
            atom_onehot[atom_type] = 1.0
            feat.extend(atom_onehot)

            # 2. 残基类型 one-hot (20维)
            res_type = self.residue_types.get(atom_info['residue_name'], 19)
            res_onehot = np.zeros(20)
            res_onehot[res_type] = 1.0
            feat.extend(res_onehot)

            # 3. 残基序号归一化 (1维)
            feat.append(atom_info['residue_id'] / 50.0)

            # 4. B-factor归一化 (1维)
            feat.append(atom_info['bfactor'] / 100.0)

            # 5. 坐标 (3维)
            feat.extend(atom_info['coord'])

            # 6. 主干/侧链标识 (1维)
            is_backbone = 1.0 if atom_info['atom_name'] in ['N', 'CA', 'C', 'O'] else 0.0
            feat.append(is_backbone)

            features.append(feat)

        return np.array(features)  # [num_atoms, 60]


class PeptideMICDataset(Dataset):
    """
    统一数据集类 - 支持单粒度和多粒度模式

    模式:
    - 单粒度基础: 仅PDB图
    - 单粒度增强: PDB图 + numpy特征
    - 多粒度: 序列 + PDB图 + numpy特征(可选)
    """

    def __init__(
        self,
        csv_file,
        pdb_dir,
        distance_threshold=8.0,
        # 单粒度参数
        feature_dir=None,
        feature_files=None,
        # 多粒度参数
        use_multigrain=False,
    ):
        """
        Args:
            csv_file: CSV文件路径
            pdb_dir: PDB文件目录
            distance_threshold: 距离阈值
            feature_dir: numpy特征目录 (可选)
            feature_files: 要加载的特征文件列表
            use_multigrain: 是否使用多粒度模式
        """
        self.df = pd.read_csv(csv_file)
        self.pdb_dir = pdb_dir
        self.feature_dir = feature_dir
        self.use_multigrain = use_multigrain
        self.converter = PDBToDGLConverter(distance_threshold=distance_threshold)

        # 检查必需列
        required_cols = ['ID', 'Activity']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"CSV必须包含'{col}'列")

        # 多粒度模式需要序列
        if use_multigrain:
            if 'Sequence' not in self.df.columns:
                print("  警告: 多粒度模式需要'Sequence'列，将从PDB提取")
                self.has_sequence = False
            else:
                self.has_sequence = True

        # 加载numpy特征（如果有）
        self.features_dict = {}
        if feature_dir is not None:
            self._load_features(feature_files)

    def _load_features(self, feature_files):
        """加载numpy特征（只加载二维 (num_samples, 30) 形状的特征）"""
        if feature_files is None:
            feature_files = [
                'DSSP.npy',
                'fa_atr.npy',
                'fa_elec.npy',
                'fa_rep.npy',
                'fa_sol.npy',
                'fa_dun.npy',
                'fa_intra_rep.npy',
                'fa_intra_sol.npy',
                'hbond_bb_sc.npy',
                'hbond_lr_bb.npy',
                'hbond_sc.npy',
                'hbond_sr_bb.npy',
                'omega.npy',
                'p_aa_pp.npy',
                'rama.npy',
                'ref.npy',
            ]

        print(f"  加载特征从: {self.feature_dir}")
        print(f"  [过滤规则] 只加载形状为 (num_samples, 30) 的二维特征")

        for fname in feature_files:
            fpath = os.path.join(self.feature_dir, fname)
            if os.path.exists(fpath):
                feat_array = np.load(fpath)

                # 检查形状：必须是二维且第二维是30
                if feat_array.ndim == 2 and feat_array.shape[1] == 30:
                    self.features_dict[fname] = feat_array
                    print(f"    ✓ {fname} - shape {feat_array.shape}")
                else:
                    print(f"    ⊗ {fname} - shape {feat_array.shape} (跳过，不符合规则)")
            else:
                print(f"    ✗ {fname} - 文件不存在")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pdb_id = row['ID']
        activity = row['Activity']

        # 获取PDB图
        pdb_file = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")
        graph = self.converter.pdb_to_graph(pdb_file)

        # 获取额外特征（如果有）
        extra_features = None
        if self.features_dict:
            feat_list = []
            for _, feat_array in self.features_dict.items():
                feat = feat_array[idx]
                if feat.ndim > 1:
                    feat = feat.flatten()
                feat_list.append(feat)
            extra_features = np.concatenate(feat_list)
            extra_features = torch.tensor(extra_features, dtype=torch.float32)

        # 多粒度模式: 返回字典
        if self.use_multigrain:
            # 获取序列
            if self.has_sequence:
                sequence = row['Sequence']
            else:
                sequence = self._extract_sequence_from_pdb(pdb_id)

            seq_encoded = self._encode_sequence(sequence)

            return {
                'pdb_id': pdb_id,
                'sequence': sequence,
                'seq_encoded': seq_encoded,
                'graph': graph,
                'extra_features': extra_features,
                'activity': activity
            }

        # 单粒度模式: 返回图和标签
        else:
            graph.graph_label = torch.tensor([activity], dtype=torch.float32)
            if extra_features is not None:
                graph.extra_features = extra_features

            return graph, activity

    def _encode_sequence(self, sequence):
        """将氨基酸序列编码为数字 (1-20)"""
        aa_to_id = {
            'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
            'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
            'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
            'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20,
            'X': 0, 'U': 0, 'B': 0, 'Z': 0, 'O': 0
        }

        encoded = []
        for aa in sequence.upper():
            encoded.append(aa_to_id.get(aa, 0))

        return torch.tensor(encoded, dtype=torch.long)

    def _extract_sequence_from_pdb(self, pdb_id):
        """从PDB文件提取序列"""
        pdb_file = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_file)

        three_to_one = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }

        sequence = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_name = residue.get_resname()
                    if res_name in three_to_one:
                        sequence.append(three_to_one[res_name])

        return ''.join(sequence)


# ============= Collate Functions =============

def collate_fn(batch):
    """单粒度基础collate"""
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return batched_graph, labels


def collate_hybrid_fn(batch):
    """单粒度混合collate (带额外特征)"""
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)

    # 处理额外特征
    if hasattr(graphs[0], 'extra_features') and graphs[0].extra_features is not None:
        extra_features_list = [g.extra_features for g in graphs]
        batched_graph.extra_features = torch.stack(extra_features_list, dim=0)
    else:
        batched_graph.extra_features = None

    return batched_graph, labels


def collate_multigrain_fn(batch):
    """多粒度collate"""
    pdb_ids = [item['pdb_id'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    seq_encoded_list = [item['seq_encoded'] for item in batch]
    graphs = [item['graph'] for item in batch]
    activities = torch.tensor([item['activity'] for item in batch], dtype=torch.float32)

    # 序列padding
    seq_lengths = torch.tensor([len(seq) for seq in seq_encoded_list])
    max_len = seq_lengths.max().item()

    seq_encoded_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(seq_encoded_list):
        seq_encoded_padded[i, :len(seq)] = seq

    # 图batch
    batched_graph = dgl.batch(graphs)

    # 额外特征 - 附加到graph上以便模型访问
    extra_features = None
    if batch[0]['extra_features'] is not None:
        extra_features = torch.stack([item['extra_features'] for item in batch])
        batched_graph.extra_features = extra_features

    return {
        'pdb_ids': pdb_ids,
        'sequences': sequences,
        'seq_encoded': seq_encoded_padded,
        'seq_lengths': seq_lengths,
        'graph': batched_graph,
        'extra_features': extra_features,
        'labels': activities
    }


def get_collate_fn(use_multigrain=False, use_hybrid=False):
    """
    获取合适的collate函数

    Args:
        use_multigrain: 是否多粒度模式
        use_hybrid: 是否混合模式 (单粒度时有效)

    Returns:
        collate_fn
    """
    if use_multigrain:
        return collate_multigrain_fn
    elif use_hybrid:
        return collate_hybrid_fn
    else:
        return collate_fn
