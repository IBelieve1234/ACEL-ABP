"""
从PDB文件提取序列并添加到CSV
"""

import pandas as pd
from Bio.PDB import PDBParser
import os


def extract_sequence_from_pdb(pdb_file):
    """从PDB文件提取序列"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('peptide', pdb_file)

    # 三字母到单字母转换
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


def add_sequences_to_csv(csv_file, pdb_dir, output_file=None):
    """
    从PDB文件提取序列并添加到CSV

    Args:
        csv_file: 输入CSV文件路径
        pdb_dir: PDB文件目录
        output_file: 输出CSV文件路径(None则覆盖原文件)
    """
    print(f"读取CSV: {csv_file}")
    df = pd.read_csv(csv_file)

    print(f"总样本数: {len(df)}")

    # 检查是否已有Sequence列
    if 'Sequence' in df.columns:
        print("警告: CSV已包含Sequence列，将被覆盖")

    # 提取序列
    sequences = []
    failed_ids = []

    for idx, row in df.iterrows():
        pdb_id = row['ID']
        pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")

        if not os.path.exists(pdb_file):
            print(f"  警告: PDB文件不存在 - {pdb_id}")
            sequences.append('')
            failed_ids.append(pdb_id)
            continue

        try:
            seq = extract_sequence_from_pdb(pdb_file)
            sequences.append(seq)

            if (idx + 1) % 100 == 0:
                print(f"  已处理: {idx + 1}/{len(df)}")

        except Exception as e:
            print(f"  错误: 提取序列失败 - {pdb_id}, {e}")
            sequences.append('')
            failed_ids.append(pdb_id)

    # 添加Sequence列
    df['Sequence'] = sequences

    # 移除失败的样本
    if failed_ids:
        print(f"\n警告: {len(failed_ids)} 个样本失败，将被移除:")
        print(failed_ids[:10])  # 只打印前10个
        df = df[df['Sequence'] != '']

    # 调整列顺序: ID, Sequence, value, Activity
    if 'value' in df.columns:
        df = df[['ID', 'Sequence', 'value', 'Activity']]
    else:
        df = df[['ID', 'Sequence', 'Activity']]

    # 保存
    if output_file is None:
        output_file = csv_file

    df.to_csv(output_file, index=False)
    print(f"\n✓ 已保存到: {output_file}")
    print(f"✓ 成功处理: {len(df)} 个样本")

    # 显示示例
    print(f"\n示例 (前5行):")
    print(df.head())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='从PDB提取序列并添加到CSV')
    parser.add_argument('--csv', type=str, required=True, help='输入CSV文件')
    parser.add_argument('--pdb_dir', type=str, default='pdb', help='PDB目录')
    parser.add_argument('--output', type=str, default=None, help='输出CSV文件(默认覆盖原文件)')

    args = parser.parse_args()

    add_sequences_to_csv(args.csv, args.pdb_dir, args.output)
