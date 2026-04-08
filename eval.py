"""
评估脚本 - 加载训练好的权重在测试集上评估
"""

import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
import argparse
import os
import json
import numpy as np

from dataset import PeptideMICDataset, get_collate_fn
from models import get_model
from utils import (
    set_seed,
    get_device,
    calculate_metrics,
    format_metrics
)


def load_language_model(lm_model, device):
    print(f"\nload llm: {lm_model}")

    from transformers import (
        BertModel, BertTokenizer,
        T5Tokenizer, T5EncoderModel,
        XLNetModel, XLNetTokenizer,
        EsmTokenizer, EsmForSequenceClassification
    )

    lm_base_path = "./llm/"

    if lm_model == "prot_bert_bfd":
        tokenizer = BertTokenizer.from_pretrained(
            f"{lm_base_path}/Rostlab/prot_bert_bfd",
            do_lower_case=False, legacy=False
        )
        pretrained_lm = BertModel.from_pretrained(
            f"{lm_base_path}/Rostlab/prot_bert_bfd"
        )
    elif lm_model == "prot_bert":
        tokenizer = BertTokenizer.from_pretrained(
            f"{lm_base_path}/Rostlab/prot_bert",
            do_lower_case=False, legacy=False
        )
        pretrained_lm = BertModel.from_pretrained(
            f"{lm_base_path}/Rostlab/prot_bert"
        )
    elif lm_model == "prot_t5_xl_bfd":
        tokenizer = T5Tokenizer.from_pretrained(
            f"{lm_base_path}/Rostlab/prot_t5_xl_bfd",
            do_lower_case=False, legacy=False
        )
        pretrained_lm = T5EncoderModel.from_pretrained(
            f"{lm_base_path}/Rostlab/prot_t5_xl_bfd"
        )
    elif lm_model == "prot_t5_xl_uniref50":
        tokenizer = T5Tokenizer.from_pretrained(
            f"{lm_base_path}/Rostlab/prot_t5_xl_uniref50",
            do_lower_case=False, legacy=False
        )
        pretrained_lm = T5EncoderModel.from_pretrained(
            f"{lm_base_path}/Rostlab/prot_t5_xl_uniref50"
        )
    elif lm_model == "prot_xlnet":
        tokenizer = XLNetTokenizer.from_pretrained(
            f"{lm_base_path}/Rostlab/prot_xlnet",
            do_lower_case=False, legacy=False
        )
        pretrained_lm = XLNetModel.from_pretrained(
            f"{lm_base_path}/Rostlab/prot_xlnet",
            mem_len=1024
        )
    elif lm_model == "ProstT5":
        tokenizer = T5Tokenizer.from_pretrained(
            f"{lm_base_path}/Rostlab/ProstT5",
            do_lower_case=False, legacy=False
        )
        pretrained_lm = T5EncoderModel.from_pretrained(
            f"{lm_base_path}/Rostlab/ProstT5"
        )
    elif lm_model == "esm2_t6_8M_UR50D":
        pretrained_lm = EsmForSequenceClassification.from_pretrained(
            f"{lm_base_path}/facebook/esm2_t6_8M_UR50D",
            num_labels=320
        )
        tokenizer = EsmTokenizer.from_pretrained(
            f"{lm_base_path}/facebook/esm2_t6_8M_UR50D"
        )
    elif lm_model == "esm2_t33_650M_UR50D":
        pretrained_lm = EsmForSequenceClassification.from_pretrained(
            f"{lm_base_path}/facebook/esm2_t33_650M_UR50D",
            num_labels=1280
        )
        tokenizer = EsmTokenizer.from_pretrained(
            f"{lm_base_path}/facebook/esm2_t33_650M_UR50D"
        )
    else:
        raise ValueError(f"Unknown language model: {lm_model}")

    pretrained_lm = pretrained_lm.to(device)
    pretrained_lm.eval()

    for param in pretrained_lm.parameters():
        param.requires_grad = False
    return tokenizer, pretrained_lm


def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.no_grad()
def evaluate_single_grain(model, loader, device, return_predictions=False, mc_dropout=False):
    """
    Args:
        mc_dropout
    """
    model.eval()
    if mc_dropout:
        enable_dropout(model) 

    all_preds = []
    all_targets = []

    for batched_graph, labels in loader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)

        if hasattr(batched_graph, 'extra_features') and batched_graph.extra_features is not None:
            batched_graph.extra_features = batched_graph.extra_features.to(device)

        pred = model(batched_graph)

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_preds, all_targets)

    if return_predictions:
        return metrics, all_preds, all_targets
    else:
        return metrics


@torch.no_grad()
def evaluate_multi_grain(model, loader, device, return_predictions=False):
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        batch['seq_encoded'] = batch['seq_encoded'].to(device)
        batch['graph'] = batch['graph'].to(device)
        batch['labels'] = batch['labels'].to(device)

        if hasattr(batch['graph'], 'extra_features') and batch['graph'].extra_features is not None:
            batch['graph'].extra_features = batch['graph'].extra_features.to(device)

        pred = model(batch)

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(batch['labels'].cpu().numpy())

    metrics = calculate_metrics(all_preds, all_targets)

    if return_predictions:
        return metrics, all_preds, all_targets
    else:
        return metrics


def load_checkpoint_with_config(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')

    model_config = checkpoint.get('model_config', {})
    model_class = checkpoint.get('model_class', 'Unknown')
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})

    print(f"\nCheckpoint Information:")
    print(f"  Model Class: {model_class}")
    print(f"  Training Epochs: {epoch}")
    print(f"  Training Metrics: {metrics}")


    return checkpoint, model_config


def evaluate(args):

    set_seed(args.seed)
    device = get_device(args.gpu_id)

    print("=" * 60)
    print("model evaluate")
    print("=" * 60)

    print(f"\n加载检查点: {args.checkpoint}")
    checkpoint, saved_config = load_checkpoint_with_config(args.checkpoint)

    exp_dir = os.path.dirname(os.path.dirname(args.checkpoint))
    config_path = os.path.join(exp_dir, 'config.json')

    if os.path.exists(config_path):
        print(f"Loading training config: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            train_config = json.load(f)
    else:
        train_config = {}
        print("config.json not found, using configuration from checkpoint")



    use_multigrain = saved_config.get('use_multigrain', train_config.get('use_multigrain', args.use_multigrain))
    use_extra_features = False
    model_type = saved_config.get('model_type', train_config.get('model_type', args.model_type))
    tech = train_config.get('tech', args.tech)
    lm_model = train_config.get('lm_model', args.lm_model)

    print(f"\nconfig:")
    print(f"  use_multigrain: {use_multigrain}")
    print(f"  model_type: {model_type}")
    print(f"  tech: {tech}")
    print(f"  lm_model: {lm_model}")

    tokenizer, pretrained_lm = None, None
    if use_multigrain:
        tokenizer, pretrained_lm = load_language_model(lm_model, device)

    print("\n" + "=" * 60)
    print("load dataset")
    print("=" * 60)

    test_dataset = PeptideMICDataset(
        csv_file=args.test_csv,
        pdb_dir=args.pdb_dir,
        distance_threshold=args.distance_threshold,
        feature_dir=None,
        use_multigrain=use_multigrain
    )

    print(f"test set: {len(test_dataset)} samples")

    collate = get_collate_fn(
        use_multigrain=use_multigrain,
        use_hybrid=False
    )

    test_loader = GraphDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0 if use_multigrain else args.num_workers
    )

    print("\n" + "=" * 60)
    print("create model")
    print("=" * 60)

    model_kwargs = {}

    exclude_keys = ['use_multigrain', 'model_type', 'use_extra_features', 'use_transformer']
    for key, value in saved_config.items():
        if key not in exclude_keys and value is not None:
            model_kwargs[key] = value

    if not model_kwargs:
        if use_multigrain:
            lm_hidden_dims = {
                'prot_bert_bfd': 1024, 'prot_bert': 1024,
                'prot_t5_xl_bfd': 1024, 'prot_t5_xl_uniref50': 1024,
                'prot_xlnet': 1024, 'ProstT5': 1024,
                'esm2_t6_8M_UR50D': 320, 'esm2_t33_650M_UR50D': 1280,
            }
            lm_hidden_dim = lm_hidden_dims.get(lm_model, 1024)

            model_kwargs = {
                'graph_input_dim': args.input_dim,
                'graph_hidden_dim': args.hidden_dim,
                'num_gnn_layers': args.num_layers,
                'lm_hidden_dim': lm_hidden_dim,
                'use_lm': True,
                'interaction_hidden_dim': args.interaction_dim,
                'pooling': args.pooling,
                'dropout': args.dropout,
                'conv_type': args.conv_type,
            }

            #  Evidential  (when tech=transformer_evidential)
            if tech == 'transformer_evidential':
                model_kwargs['use_evidential'] = args.use_evidential

        else:
            model_kwargs = {
                'input_dim': args.input_dim,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
            }

    print(f"model Parameters: {model_kwargs}")

    model = get_model(
        use_multigrain=use_multigrain,
        model_type=model_type,
        use_extra_features=False,
        ablation_tech=tech,
        **model_kwargs
    ).to(device)

    if use_multigrain:
        model.set_language_model(tokenizer, pretrained_lm)

    # load
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型权重已加载")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 评估 ==========
    # 定义不同的batch_size列表
    batch_sizes = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
    num_runs = len(batch_sizes)

    print("\n" + "=" * 60)
    print(f"开始评估 (共 {num_runs} 次, 不同batch_size)")
    print("=" * 60)

    if use_multigrain:
        eval_fn = evaluate_multi_grain
    else:
        eval_fn = evaluate_single_grain

    # 多次评估
    all_metrics = []
    all_preds_list = []

    for run_idx, batch_size in enumerate(batch_sizes):
        print(f"\n--- 第 {run_idx + 1}/{num_runs} 次评估 (batch_size={batch_size}) ---")

        run_loader = GraphDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0 if use_multigrain else args.num_workers
        )

        metrics, preds, targets = eval_fn(model, run_loader, device, return_predictions=True)
        all_metrics.append(metrics)
        all_preds_list.append(preds)

        print(f"  MSE: {metrics['mse']:.4f} | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R2: {metrics['r2']:.4f} | Pearson: {metrics['pearson']:.4f} | Spearman: {metrics['spearman']:.4f} | Recall@50: {metrics['recall_at_50']:.4f}")

    # 计算平均值和标准差
    avg_metrics = {}
    std_metrics = {}
    metric_keys = all_metrics[0].keys()

    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)

    print("\n" + "=" * 60)
    print(f"评估结果汇总 ({num_runs} 次, batch_sizes: {batch_sizes})")
    print("=" * 60)
    print(f"\n{'指标':<12} {'平均值':<12} {'标准差':<12}")
    print("-" * 36)
    for key in ['rmse', 'r2', 'pearson', 'spearman', 'recall_at_50', 'mse', 'mae']:
        if key in avg_metrics:
            print(f"{key.upper():<12} {avg_metrics[key]:<12.4f} {std_metrics[key]:<12.4f}")

    # ========== 保存结果 ==========
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # 保存每次评估的详细结果
        all_runs_path = os.path.join(args.output_dir, 'all_runs_metrics.json')
        all_runs_data = {
            'num_runs': num_runs,
            'batch_sizes': batch_sizes,
            'runs': [
                {'batch_size': bs, **{k: float(v) for k, v in m.items()}}
                for bs, m in zip(batch_sizes, all_metrics)
            ],
            'average': {k: float(v) for k, v in avg_metrics.items()},
            'std': {k: float(v) for k, v in std_metrics.items()}
        }
        with open(all_runs_path, 'w', encoding='utf-8') as f:
            json.dump(all_runs_data, f, indent=2)
        print(f"\n所有评估结果已保存: {all_runs_path}")

        # 保存最后一次的预测结果
        pred_path = os.path.join(args.output_dir, 'predictions.txt')
        with open(pred_path, 'w', encoding='utf-8') as f:
            f.write("Target\tPrediction\n")
            for target, pred in zip(targets, preds):
                f.write(f"{target:.4f}\t{pred:.4f}\n")
        print(f"预测结果已保存: {pred_path}")

        # 保存汇总指标
        metrics_path = os.path.join(args.output_dir, 'metrics_summary.json')
        summary = {
            'num_runs': num_runs,
            'batch_sizes': batch_sizes,
            'average': {k: float(v) for k, v in avg_metrics.items()},
            'std': {k: float(v) for k, v in std_metrics.items()}
        }
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"汇总指标已保存: {metrics_path}")

        # 绘制预测图 (使用最后一次的预测)
        try:
            from utils import plot_predictions
            plot_path = os.path.join(args.output_dir, 'predictions.png')
            plot_predictions(preds, targets, plot_path)
            print(f"预测图已保存: {plot_path}")
        except Exception as e:
            print(f"绘图失败: {e}")

    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)

    return avg_metrics, std_metrics


def main():
    parser = argparse.ArgumentParser(description='evaluation')

    # 必需参数
    parser.add_argument('--checkpoint', type=str,default='./checkpoint/best_model_10.pt',
                        help='检查点文件路径 (best_model.pt)')

    # 数据参数
    parser.add_argument('--test_csv', type=str, default='data/0_93__10_structure/test.csv',#data/0_9__5/test.csv
                        help='test set CSV')
    parser.add_argument('--pdb_dir', type=str, default='pdb',
                        help='PDB')
    parser.add_argument('--distance_threshold', type=float, default=8.0)

    # 模型参数 (如果checkpoint中没有保存配置，则使用这些默认值)
    parser.add_argument('--use_multigrain', action='store_true', default=True)
    parser.add_argument('--model_type', type=str, default='cross_attention')
    parser.add_argument('--tech', type=str, default='_evidential')
    parser.add_argument('--lm_model', type=str, default='prot_t5_xl_uniref50')
    parser.add_argument('--input_dim', type=int, default=60)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--pooling', type=str, default='mean_sum')
    parser.add_argument('--interaction_dim', type=int, default=256)
    parser.add_argument('--fusion_strategy', type=str, default='structure_enhanced')

    # Transformer/Evidential参数 (与train.py保持一致)
    parser.add_argument('--use_evidential', action='store_true', default=False,
                        help='使用Evidential不确定性量化 (仅 tech=_evidential 时有效)')
    parser.add_argument('--conv_type', type=str, default='graphconv',
                        choices=['graphconv'],
                        help='图卷积类型')

    # 评估参数
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_runs', type=int, default=10,
                        help='评估次数 (默认10次，计算平均值和标准差)')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (保存预测结果和指标)')

    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()

    # 如果没有指定output_dir，使用checkpoint所在目录
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.checkpoint)

    evaluate(args)


if __name__ == '__main__':
    main()
