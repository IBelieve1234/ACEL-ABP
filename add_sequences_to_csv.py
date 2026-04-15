"""
Extract sequences from PDB files and add them to a CSV
"""

import pandas as pd
from Bio.PDB import PDBParser
import os


def extract_sequence_from_pdb(pdb_file):
    """Extract sequence from a PDB file"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('peptide', pdb_file)

    # Convert three-letter residue codes to one-letter codes
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
    Extract sequences from PDB files and add them to a CSV

    Args:
        csv_file: Path to the input CSV file
        pdb_dir: Directory containing PDB files
        output_file: Path to the output CSV file (None means overwrite the original file)
    """
    print(f"Reading CSV: {csv_file}")
    df = pd.read_csv(csv_file)

    print(f"Total number of samples: {len(df)}")

    # Check whether the Sequence column already exists
    if 'Sequence' in df.columns:
        print("Warning: The CSV already contains a Sequence column, which will be overwritten")

    # Extract sequences
    sequences = []
    failed_ids = []

    for idx, row in df.iterrows():
        pdb_id = row['ID']
        pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")

        if not os.path.exists(pdb_file):
            print(f" Warning: PDB file does not exist - {pdb_id}")
            sequences.append('')
            failed_ids.append(pdb_id)
            continue

        try:
            seq = extract_sequence_from_pdb(pdb_file)
            sequences.append(seq)

            if (idx + 1) % 100 == 0:
                print(f"  Processed:  {idx + 1}/{len(df)}")

        except Exception as e:
            print(f"  Error: Failed to extract sequence - {pdb_id}, {e}")
            sequences.append('')
            failed_ids.append(pdb_id)

    # Add Sequence column
    df['Sequence'] = sequences

    # Remove failed samples
    if failed_ids:
        print(f"\nWarning:  {len(failed_ids)} samples failed and will be removed:")
        print(failed_ids[:10]) # Print only the first 10
        df = df[df['Sequence'] != '']

    # Reorder columns: ID, Sequence, value, Activity
    if 'value' in df.columns:
        df = df[['ID', 'Sequence', 'value', 'Activity']]
    else:
        df = df[['ID', 'Sequence', 'Activity']]

    # Save
    if output_file is None:
        output_file = csv_file

    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to:  {output_file}")
    print(f"✓ Successfully processed: {len(df)} samples")

    # Show examples
    print(f"\nExample (first 5 rows):")
    print(df.head())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract sequences from PDB files and add them to a CSV')
    parser.add_argument('--csv', type=str, required=True, help='Input CSV file')
    parser.add_argument('--pdb_dir', type=str, default='pdb', help='PDB directory')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (default: overwrite original file)')

    args = parser.parse_args()

    add_sequences_to_csv(args.csv, args.pdb_dir, args.output)
