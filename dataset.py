"""
Unified dataset module - supports both single-granularity and multi-granularity modes
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
    """Convert PDB files into DGL graph objects"""

    def __init__(self, distance_threshold=8.0):
        """
        Args:
            distance_threshold: Distance threshold between atoms (Å); atom pairs closer than this value will be connected by edges
        """
        self.distance_threshold = distance_threshold
        self.parser = PDBParser(QUIET=True)

        # Atom type encoding
        self.atom_types = {
            'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4,
            'CG': 5, 'CD': 6, 'CE': 7, 'CZ': 8, 'SD': 9,
            'NH1': 10, 'NH2': 11, 'OG': 12, 'OD1': 13, 'OD2': 14,
            'NE': 15, 'NE1': 16, 'NE2': 17, 'ND1': 18, 'ND2': 19,
            'OE1': 20, 'OE2': 21, 'SG': 22, 'CD1': 23, 'CD2': 24,
            'CE1': 25, 'CE2': 26, 'CE3': 27, 'CG1': 28, 'CG2': 29,
            'OH': 30, 'NZ': 31, 'OXT': 32
        }

        # Amino acid type encoding
        self.residue_types = {
            'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
            'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
            'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
            'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
        }

    def pdb_to_graph(self, pdb_file):
        """
        Convert a PDB file into a DGL graph

        Args:
            pdb_file: Path to the PDB file

        Returns:
            g: DGL graph object
                - g.ndata['feat']: Node features [num_atoms, 60]
                - g.ndata['pos']: Atom coordinates [num_atoms, 3]
                - g.edata['dist']: Edge features (distance) [num_edges, 1]
        """
        structure = self.parser.get_structure('peptide', pdb_file)

        atoms_info = []
        coords = []

        # Extract atom information
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

        # Build edge list
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

        # Create DGL graph
        if len(src_nodes) == 0:
            # If there are no edges, add self-loops
            src_nodes = list(range(num_atoms))
            dst_nodes = list(range(num_atoms))
            edge_distances = [0.0] * num_atoms

        g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_atoms)

        # Add node features
        node_features = self._encode_atoms(atoms_info, coords)
        g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
        g.ndata['pos'] = torch.tensor(coords, dtype=torch.float32)

        # Add edge features
        if len(edge_distances) > 0:
            g.edata['dist'] = torch.tensor(edge_distances, dtype=torch.float32).unsqueeze(1)

        return g

    def _encode_atoms(self, atoms_info, coords):
        """
        Encode atom features

        Returns:
            features: numpy array of shape [num_atoms, 60]
        """
        features = []

        for i, atom_info in enumerate(atoms_info):
            feat = []

            # 1. Atom type one-hot (34 dims)
            atom_type = self.atom_types.get(atom_info['atom_name'], 33)
            atom_onehot = np.zeros(34)
            atom_onehot[atom_type] = 1.0
            feat.extend(atom_onehot)

            # 2. Residue type one-hot (20 dims)
            res_type = self.residue_types.get(atom_info['residue_name'], 19)
            res_onehot = np.zeros(20)
            res_onehot[res_type] = 1.0
            feat.extend(res_onehot)

            # 3. Normalized residue index (1 dim)
            feat.append(atom_info['residue_id'] / 50.0)

            # 4. Normalized B-factor (1 dim)
            feat.append(atom_info['bfactor'] / 100.0)

            # 5. Coordinates (3 dims)
            feat.extend(atom_info['coord'])

            # 6. Backbone/side-chain indicator (1 dim)
            is_backbone = 1.0 if atom_info['atom_name'] in ['N', 'CA', 'C', 'O'] else 0.0
            feat.append(is_backbone)

            features.append(feat)

        return np.array(features)  # [num_atoms, 60]


class PeptideMICDataset(Dataset):
    """
    Unified dataset class - supports both single-granularity and multi-granularity modes

    Modes:
    - Single-granularity basic: PDB graph only
    - Single-granularity enhanced: PDB graph + numpy features
    - Multi-granularity: sequence + PDB graph + numpy features (optional)
    """

    def __init__(
        self,
        csv_file,
        pdb_dir,
        distance_threshold=8.0,
        # Single-granularity parameters
        feature_dir=None,
        feature_files=None,
        # Multi-granularity parameters
        use_multigrain=False,
    ):
        """
        Args:
            csv_file: Path to the CSV file
            pdb_dir: Directory containing PDB files
            distance_threshold: Distance threshold
            feature_dir: Directory of numpy features (optional)
            feature_files: List of feature files to load
            use_multigrain: Whether to use multi-granularity mode
        """
        self.df = pd.read_csv(csv_file)
        self.pdb_dir = pdb_dir
        self.feature_dir = feature_dir
        self.use_multigrain = use_multigrain
        self.converter = PDBToDGLConverter(distance_threshold=distance_threshold)

        # Check required columns
        required_cols = ['ID', 'Activity']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"CSV must contain column '{col}'")

        # Multi-granularity mode requires sequences
        if use_multigrain:
            if 'Sequence' not in self.df.columns:
                print("  Warning: Multi-granularity mode requires a 'Sequence' column; it will be extracted from PDB")
                self.has_sequence = False
            else:
                self.has_sequence = True

        # Load numpy features (if any)
        self.features_dict = {}
        if feature_dir is not None:
            self._load_features(feature_files)

    def _load_features(self, feature_files):
        """Load numpy features (only features with 2D shape (num_samples, 30))"""
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

        print(f"  Loading features from: {self.feature_dir}")
        print(f"  [Filter rule] Only loading 2D features with shape (num_samples, 30)")

        for fname in feature_files:
            fpath = os.path.join(self.feature_dir, fname)
            if os.path.exists(fpath):
                feat_array = np.load(fpath)

                # Check shape: must be 2D and second dimension must be 30
                if feat_array.ndim == 2 and feat_array.shape[1] == 30:
                    self.features_dict[fname] = feat_array
                    print(f"    ✓ {fname} - shape {feat_array.shape}")
                else:
                    print(f"    ⊗ {fname} - shape {feat_array.shape} (skipped, does not meet the rule)")
            else:
                print(f"    ✗ {fname} - file does not exist")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pdb_id = row['ID']
        activity = row['Activity']

        # Get PDB graph
        pdb_file = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")
        graph = self.converter.pdb_to_graph(pdb_file)

        # Get extra features (if any)
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

        # Multi-granularity mode: return a dictionary
        if self.use_multigrain:
            # Get sequence
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

        # Single-granularity mode: return graph and label
        else:
            graph.graph_label = torch.tensor([activity], dtype=torch.float32)
            if extra_features is not None:
                graph.extra_features = extra_features

            return graph, activity

    def _encode_sequence(self, sequence):
        """Encode the amino acid sequence as numbers (1-20)"""
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
        """Extract sequence from a PDB file"""
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
    """Single-granularity basic collate"""
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return batched_graph, labels


def collate_hybrid_fn(batch):
    """Single-granularity hybrid collate (with extra features)"""
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Handle extra features
    if hasattr(graphs[0], 'extra_features') and graphs[0].extra_features is not None:
        extra_features_list = [g.extra_features for g in graphs]
        batched_graph.extra_features = torch.stack(extra_features_list, dim=0)
    else:
        batched_graph.extra_features = None

    return batched_graph, labels


def collate_multigrain_fn(batch):
    """Multi-granularity collate"""
    pdb_ids = [item['pdb_id'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    seq_encoded_list = [item['seq_encoded'] for item in batch]
    graphs = [item['graph'] for item in batch]
    activities = torch.tensor([item['activity'] for item in batch], dtype=torch.float32)

    # Sequence padding
    seq_lengths = torch.tensor([len(seq) for seq in seq_encoded_list])
    max_len = seq_lengths.max().item()

    seq_encoded_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(seq_encoded_list):
        seq_encoded_padded[i, :len(seq)] = seq

    # Graph batch
    batched_graph = dgl.batch(graphs)

    # Extra features - attach to graph for model access
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
    Get the appropriate collate function

    Args:
        use_multigrain: Whether to use multi-granularity mode
        use_hybrid: Whether to use hybrid mode (only effective in single-granularity mode)

    Returns:
        collate_fn
    """
    if use_multigrain:
        return collate_multigrain_fn
    elif use_hybrid:
        return collate_hybrid_fn
    else:
        return collate_fn
