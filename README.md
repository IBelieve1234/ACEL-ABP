# ACEL-ABP (ACEL-ABP: activity cliff-aware evidential deep learning for predicting antibacterial peptide activity)

## 1. Project Overview

This repository provides code for "ACEL-ABP: activity cliff-aware evidential deep learning for predicting antibacterial peptide activity"
The current main training path uses evidential regression to make predictions while estimating uncertainty to enhance prediction quality.

<img width="2981" height="1970" alt="Fig1" src="https://github.com/user-attachments/assets/f2a1dbe4-32e7-444c-8d9b-34bea1971f8e" />

Core scripts:

- `train.py`: training entrypoint
- `eval.py`: checkpoint evaluation entrypoint
- `dataset.py`: CSV/PDB to DGL data pipeline
- `models.py`: model definitions (fusion + evidential heads)
- `utils.py`: metrics, checkpointing, plotting, reproducibility helpers
- `add_sequences_to_csv.py`: fill `Sequence` from PDB files

---

## 2. Repository Structure


```text
.
├─ train.py
├─ eval.py
├─ dataset.py
├─ models.py
├─ utils.py
├─ add_sequences_to_csv.py
├─ checkpoint/
│  └─ best_model_10.pt
├─ data/
│  ├─ data_generation.ipynb
│  ├─ grampa_s_aureus_7_25_with_GRAMPA.csv
│  └─ 0_93__10_structure/
│     ├─ train.csv
│     ├─ test.csv
│     ├─ grampa_s_aureus_7_25_with_GRAMPA.csv
│     └─ add_activity.py
├─ llm/
│  └─ ...
└─ pdb/
   └─ *.pdb
```

## 3. Environment & Dependencies

## 3.1 Suggested Environment

- Python 3.8+ (project notebook metadata shows 3.8.18)
- CUDA GPU recommended for training

## 3.2 Python Packages

Install at least:

```bash
pip install torch dgl transformers esm biopython scipy pandas numpy matplotlib tqdm tmtools
```

---

## 4. Data Format and Available Split

## 4.1 CSV Schema

Minimum required columns used by `PeptideMICDataset`:

- `ID` (must match `pdb/<ID>.pdb`)
- `Activity` 
- `Sequence` (if missing, sequence can be extracted from PDB)
- `value` (regression target)

## 4.2 Public Split in This Repo

- `data/0_93__10_structure/train.csv`: **2390** samples
- `data/0_93__10_structure/test.csv`: **367** samples
- `pdb/*.pdb`: **2757** structures

---

## 5. Training and Evaluation

## 5.1 Optional: Fill Sequence Column from PDB

```bash
python add_sequences_to_csv.py --csv data/0_93__10_structure/train.csv --pdb_dir pdb --output data/0_93__10_structure/train.csv
python add_sequences_to_csv.py --csv data/0_93__10_structure/test.csv --pdb_dir pdb --output data/0_93__10_structure/test.csv
```

## 5.2 Train

Use explicit paths (recommended), because script defaults currently point to `0_93__10_structure`.

```bash
python train.py \
  --train_csv data/0_93__10_structure/train.csv \
  --test_csv data/0_93__10_structure/test.csv \
  --pdb_dir pdb \
  --save_dir experiments_structure \
  --exp_name transformer_evidential_0_93_10 \
  --num_epochs 200 \
  --batch_size 32 \
  --lr 0.0005 \
  --model_type cross_attention \
  --fusion_strategy structure_enhanced \
  --lm_model prot_t5_xl_uniref50 \
  --loss_type evidential \
  --evidential_coeff 0.01
```

## 5.3 Evaluate a Checkpoint

```bash
python eval.py \
  --checkpoint checkpoint/best_model_10.pt \
  --test_csv data/0_93__10_structure/test.csv \
  --pdb_dir pdb \
  --output_dir eval_results
```

`eval.py` runs multiple evaluation passes with different batch sizes and writes summary JSON files.

---

## 6. Output Artifacts

### 6.1 Training Outputs (`train.py`)

Under `--save_dir/<exp_name_with_timestamp>/`:

- `checkpoints/best_model.pt`
- `logs/train.log`
- `config.json`
- `code_snapshot/` (copied source snapshot)

### 6.2 Evaluation Outputs (`eval.py`)

Under `--output_dir`:

- `all_runs_metrics.json`
- `metrics_summary.json`

---

## 7. Notes and Known Caveats

1. `train.py` and `eval.py` hardcode the PLM base path as:
   `./llm/`
   The PLMs are from https://huggingface.co/Rostlab and https://github.com/facebookresearch/esm
   Update this path in code if your local model files are elsewhere.
2. The grampa_s_aureus_7_25_with_GRAMPA.csv are from [AMPCliff-generation/data/grampa_s_aureus_7_25.csv at main · Kewei2023/AMPCliff-generation](https://github.com/Kewei2023/AMPCliff-generation/blob/main/data/grampa_s_aureus_7_25.csv)
3. The weight best_model_10.pt have been uploaded to https://huggingface.co/datasets/shi7455/ACEL-ABP
