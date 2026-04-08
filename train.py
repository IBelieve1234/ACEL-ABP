import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
import argparse
import os
import json
from datetime import datetime

from dataset import PeptideMICDataset, get_collate_fn
from models import get_model, EvidentialRegressionLoss, FocalRegressionLoss
from utils import (
    set_seed,
    worker_init_fn,
    get_device,
    save_checkpoint,
    load_checkpoint,
    calculate_metrics,
    EarlyStopping,
    AverageMeter,
    format_metrics,
    create_exp_dir,
    Logger,
    get_lr,
    plot_training_curves,
    plot_predictions,
    save_code_snapshot
)


def load_language_model(lm_model, device):
    print(f"\nload llm: {lm_model}")

    # transformers
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

    print(f"  [OK]  {device}")
    print(f"  [OK] model fixed")

    return tokenizer, pretrained_lm


def train_epoch_single_grain(model, loader, optimizer, device, epoch, criterion=None):

    model.train()

    if hasattr(model, 'pretrained_lm') and model.pretrained_lm is not None:
        model.pretrained_lm.eval()
    loss_meter = AverageMeter()

    # evidential
    use_evidential = hasattr(model, 'use_evidential') and model.use_evidential
    if criterion is None:
        criterion = F.mse_loss

    for batch_idx, (batched_graph, labels) in enumerate(loader):
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)

        if hasattr(batched_graph, 'extra_features') and batched_graph.extra_features is not None:
            batched_graph.extra_features = batched_graph.extra_features.to(device)

        optimizer.zero_grad()

        if use_evidential and isinstance(criterion, EvidentialRegressionLoss):
            # Evidential
            _, _, _, (gamma, nu, alpha, beta) = model(batched_graph, return_uncertainty=True)
            loss = criterion(gamma, nu, alpha, beta, labels)
        else:
            pred = model(batched_graph)
            if isinstance(criterion, EvidentialRegressionLoss):
                loss = F.mse_loss(pred, labels)
            elif isinstance(criterion, FocalRegressionLoss):
                loss = criterion(pred, labels)
            else:
                loss = criterion(pred, labels)

        loss.backward()

        if use_evidential:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        loss_meter.update(loss.item(), len(labels))

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}] | Loss: {loss.item():.4f}")

    return loss_meter.avg


def train_epoch_multi_grain(model, loader, optimizer, device, epoch, criterion=None):
    model.train()
    if hasattr(model, 'pretrained_lm') and model.pretrained_lm is not None:
        model.pretrained_lm.eval()
    loss_meter = AverageMeter()

    # evidential
    use_evidential = hasattr(model, 'use_evidential') and model.use_evidential
    if criterion is None:
        criterion = F.mse_loss

    for batch_idx, batch in enumerate(loader):
        batch['seq_encoded'] = batch['seq_encoded'].to(device)
        batch['graph'] = batch['graph'].to(device)
        batch['labels'] = batch['labels'].to(device)

        if hasattr(batch['graph'], 'extra_features') and batch['graph'].extra_features is not None:
            batch['graph'].extra_features = batch['graph'].extra_features.to(device)

        optimizer.zero_grad()

        if use_evidential and isinstance(criterion, EvidentialRegressionLoss):
            # Evidential
            _, _, _, (gamma, nu, alpha, beta) = model(batch, return_uncertainty=True)
            loss = criterion(gamma, nu, alpha, beta, batch['labels'])
        else:
            pred = model(batch)
            if isinstance(criterion, EvidentialRegressionLoss):
                loss = F.mse_loss(pred, batch['labels'])
            elif isinstance(criterion, FocalRegressionLoss):
                loss = criterion(pred, batch['labels'])
            else:
                loss = criterion(pred, batch['labels'])

        loss.backward()

        if use_evidential:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        loss_meter.update(loss.item(), len(batch['labels']))

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}] | Loss: {loss.item():.4f}")

    return loss_meter.avg


@torch.no_grad()
def evaluate_single_grain(model, loader, device, return_predictions=False):
    model.eval()
    all_preds = []
    all_targets = []
    loss_meter = AverageMeter()

    for batched_graph, labels in loader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)

        if hasattr(batched_graph, 'extra_features') and batched_graph.extra_features is not None:
            batched_graph.extra_features = batched_graph.extra_features.to(device)

        pred = model(batched_graph)
        loss = F.mse_loss(pred, labels)

        loss_meter.update(loss.item(), len(labels))
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_preds, all_targets)
    metrics['loss'] = loss_meter.avg

    if return_predictions:
        return metrics, all_preds, all_targets
    else:
        return metrics


@torch.no_grad()
def evaluate_multi_grain(model, loader, device, return_predictions=False):
    model.eval()
    all_preds = []
    all_targets = []
    loss_meter = AverageMeter()

    for batch in loader:
        batch['seq_encoded'] = batch['seq_encoded'].to(device)
        batch['graph'] = batch['graph'].to(device)
        batch['labels'] = batch['labels'].to(device)
        if hasattr(batch['graph'], 'extra_features') and batch['graph'].extra_features is not None:
            batch['graph'].extra_features = batch['graph'].extra_features.to(device)

        pred = model(batch)
        loss = F.mse_loss(pred, batch['labels'])

        loss_meter.update(loss.item(), len(batch['labels']))
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(batch['labels'].cpu().numpy())

    metrics = calculate_metrics(all_preds, all_targets)
    metrics['loss'] = loss_meter.avg

    if return_predictions:
        return metrics, all_preds, all_targets
    else:
        return metrics


def train(args):

    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "multigrain" if args.use_multigrain else "single"
    exp_name = f"{args.exp_name}_{mode_suffix}_{timestamp}" if args.exp_name else f"{mode_suffix}_{timestamp}"
    dirs = create_exp_dir(exp_name, args.save_dir)


    logger = Logger(os.path.join(dirs['logs_dir'], 'train.log'))
    logger.log(f"\n{'='*60}")
    logger.log(f"Training Mode: {'Multi-Granularity' if args.use_multigrain else 'Single-Granularity'}")
    logger.log(f"Experiment Name: {exp_name}")
    logger.log(f"{'='*60}\n")

    saved_files = save_code_snapshot(dirs['exp_dir'])
    logger.log(f"Code snapshot saved: {saved_files}")


    device = get_device(args.gpu_id)

    tokenizer, pretrained_lm = None, None
    if args.use_multigrain and args.use_lm:
        logger.log("\n" + "="*60)
        logger.log("Loading Pretrained Language Model")
        logger.log("="*60)
        tokenizer, pretrained_lm = load_language_model(args.lm_model, device)
    elif args.use_multigrain and not args.use_lm:
        logger.log("\n" + "="*60)
        logger.log("[Ablation Study] Language Model Disabled (--use_lm False)")
        logger.log("="*60)

    # ========== Load Dataset ==========
    logger.log("\n" + "="*60)
    logger.log("Loading Dataset")
    logger.log("="*60)


    train_dataset = PeptideMICDataset(
        csv_file=args.train_csv,
        pdb_dir=args.pdb_dir,
        distance_threshold=args.distance_threshold,
        feature_dir=None,
        use_multigrain=args.use_multigrain
    )

    test_dataset = PeptideMICDataset(
        csv_file=args.test_csv,
        pdb_dir=args.pdb_dir,
        distance_threshold=args.distance_threshold,
        feature_dir=None,
        use_multigrain=args.use_multigrain
    )

    logger.log(f"train set: {len(train_dataset)} samples")
    logger.log(f"test set: {len(test_dataset)} samples")

    # DataLoader
    collate = get_collate_fn(
        use_multigrain=args.use_multigrain,
        use_hybrid=False
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0 if args.use_multigrain else args.num_workers,
        worker_init_fn=worker_init_fn,
        generator=g
    )

    test_loader = GraphDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0 if args.use_multigrain else args.num_workers,
        worker_init_fn=worker_init_fn,
        generator=g
    )

    logger.log("\n" + "="*60)
    logger.log("create model")
    logger.log("="*60)

    model_kwargs = {
        'dropout': args.dropout,
    }


    if args.model_type in ['base', 'hybrid'] or args.use_multigrain:
        model_kwargs['conv_type'] = args.conv_type
        print(f"[DEBUG] Added conv_type={args.conv_type} to model_kwargs")
    else:
        print(f"[DEBUG] Current configuration does not require conv_type")


    if args.use_multigrain:
        lm_hidden_dims = {
            'prot_bert_bfd': 1024,
            'prot_bert': 1024,
            'prot_t5_xl_bfd': 1024,
            'prot_t5_xl_uniref50': 1024,
            'prot_xlnet': 1024,
            'ProstT5': 1024,
            'esm2_t6_8M_UR50D': 320,
            'esm2_t33_650M_UR50D': 1280,
        }
        lm_hidden_dim = lm_hidden_dims.get(args.lm_model, 1024)
        logger.log(f"Language Model Hidden Dimension: {lm_hidden_dim}")
        model_kwargs.update({
            'graph_input_dim': args.input_dim,
            'graph_hidden_dim': args.hidden_dim,
            'num_gnn_layers': args.num_layers,
            'lm_hidden_dim': lm_hidden_dim,
            'use_lm': args.use_lm,
            'interaction_hidden_dim': args.interaction_dim,
            'pooling': args.pooling
        })

        if args.conv_type == 'gat':
            model_kwargs['num_heads'] = 4
            logger.log(f"GAT Convolution Parameter: num_heads=4")

        model_kwargs['fusion_strategy'] = args.fusion_strategy
        model_kwargs['use_evidential'] = args.use_evidential
        logger.log(f"Fusion Strategy: {args.fusion_strategy}")
        logger.log(f"Use Language Model (LM): {args.use_lm}")
        logger.log(f"Use Evidential: {args.use_evidential}")
    else:
        extra_dim = 0

        if args.model_type == 'hybrid':
            model_kwargs.update({
                'graph_input_dim': args.input_dim,
                'graph_hidden_dim': args.hidden_dim,
                'num_gnn_layers': args.num_layers,
                'extra_feature_dim': extra_dim,
                'pooling': args.pooling,
                'conv_type': args.conv_type  
            })
            if args.conv_type == 'gat':
                model_kwargs['num_heads'] = 4
                logger.log(f"HybridGNN use GAT: num_heads=4")
        else:
            model_kwargs.update({
                'input_dim': args.input_dim,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
            })
            if args.model_type in ['base', 'schnet']:
                model_kwargs['pooling'] = args.pooling
            if args.model_type in ['gat', 'simple_gat']:
                model_kwargs['num_heads'] = 4
                logger.log(f"GAT: num_heads=4")

    tech_value = None if args.tech == 'none' else args.tech

    print(f"[DEBUG] model_kwargs = {model_kwargs}")
    print(f"[DEBUG] tech = {tech_value}")
    model = get_model(
        use_multigrain=args.use_multigrain,
        model_type=args.model_type,
        use_extra_features=False,
        ablation_tech=tech_value,
        **model_kwargs
    ).to(device)


    if args.use_multigrain and args.use_lm:
        model.set_language_model(tokenizer, pretrained_lm)

    logger.log(f"Model Type: {args.model_type}")
    logger.log(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.log(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    config_dict = vars(args).copy()
    config_dict['actual_model_class'] = model.__class__.__name__

    config_path = os.path.join(dirs['exp_dir'], 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    logger.log(f"\nConfiguration saved: {config_path}")
    logger.log(f"  Actual Model Used: {model.__class__.__name__}")

    model_config = {
        'use_multigrain': args.use_multigrain,
        'model_type': args.model_type,
        **model_kwargs
    }

    # ========== Loss Function ==========
    logger.log("\n" + "="*60)
    logger.log("Configuring Loss Function")
    logger.log("="*60)

    criterion = None

    should_use_evidential_loss = False
    if args.loss_type == 'evidential' and args.use_evidential:
        should_use_evidential_loss = True

    if should_use_evidential_loss:
        criterion = EvidentialRegressionLoss(coeff=args.evidential_coeff)
        logger.log(f"loss: Evidential (coeff={args.evidential_coeff})")
    elif args.loss_type == 'focal':
        criterion = FocalRegressionLoss(gamma=args.focal_gamma, loss_type=args.focal_loss_base)
        logger.log(f"loss: Focal (gamma={args.focal_gamma}, base={args.focal_loss_base})")
    elif args.loss_type == 'mae':
        criterion = F.l1_loss
        logger.log("loss: MAE")
    else:
        criterion = F.mse_loss
        logger.log("loss: MSE")

    # ========== Optimizer and Scheduler ==========
    # A smaller learning rate is recommended for Evidential models
    effective_lr = args.lr
    if should_use_evidential_loss:
        logger.log(f"Evidential mode detected, adjusting learning rate: {args.lr} -> {effective_lr}")


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=effective_lr,
        weight_decay=args.weight_decay
    )

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            verbose=True
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma
        )
    else:
        scheduler = None

    early_stopping = EarlyStopping(
        patience=args.early_stop_patience,
        mode='min'
    ) if args.early_stop else None

    # ==========train ==========
    logger.log("\n" + "="*60)
    logger.log("Starting Training")
    logger.log("="*60 + "\n")

    best_rmse = float('inf')
    train_losses = []
    val_losses = []

    if args.use_multigrain:
        train_fn = train_epoch_multi_grain
        eval_fn = evaluate_multi_grain
    else:
        train_fn = train_epoch_single_grain
        eval_fn = evaluate_single_grain

    for epoch in range(1, args.num_epochs + 1):
        logger.log(f"\nEpoch [{epoch}/{args.num_epochs}]")
        logger.log("-" * 60)

        train_loss = train_fn(model, train_loader, optimizer, device, epoch,
                             criterion=criterion)
        train_losses.append(train_loss)
        logger.log(f"  Train Loss: {train_loss:.4f} | LR: {get_lr(optimizer):.6f}")

        should_eval = (epoch % args.eval_interval == 0) or (epoch == args.num_epochs)

        if should_eval:
            test_metrics = eval_fn(model, test_loader, device)
            val_losses.append(test_metrics['loss'])

            logger.log(f"  Test  {format_metrics(test_metrics, prefix='')}")

            if scheduler is not None:
                if args.scheduler == 'plateau':
                    scheduler.step(test_metrics['rmse'])
                else:
                    scheduler.step()

            is_best = test_metrics['rmse'] < best_rmse
            if is_best:
                best_rmse = test_metrics['rmse']
                best_checkpoint_path = os.path.join(dirs['checkpoints_dir'], 'best_model.pt')
                save_checkpoint(model, optimizer, epoch, test_metrics, best_checkpoint_path, model_config)
                logger.log(f"  [BEST] new best RMSE: {best_rmse:.4f}")

            
            if early_stopping is not None:
                early_stopping(test_metrics['rmse'])
                if early_stopping.early_stop:
                    logger.log(f"\nEarly stopping triggered! Best RMSE: {best_rmse:.4f}")
                    break

            # RMSE threshold check: if RMSE is still unsatisfactory after the specified epoch, terminate early
            if args.rmse_threshold_epoch > 0 and epoch >= args.rmse_threshold_epoch:
                if best_rmse > args.rmse_threshold:
                    logger.log(f"\nRMSE threshold check: after {args.rmse_threshold_epoch} epochs, best RMSE={best_rmse:.4f} > threshold {args.rmse_threshold}")
                    logger.log(f"Training performance is unsatisfactory, terminating early!")
                    break


        if args.save_interval > 0 and epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(dirs['checkpoints_dir'], f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(model, optimizer, epoch, {'epoch': epoch}, checkpoint_path, model_config)

    logger.log("\n" + "="*60)
    logger.log("Final Evaluation")
    logger.log("="*60)

    best_checkpoint_path = os.path.join(dirs['checkpoints_dir'], 'best_model.pt')
    if os.path.exists(best_checkpoint_path):
        load_checkpoint(best_checkpoint_path, model)

        test_metrics, test_preds, test_targets = eval_fn(
            model, test_loader, device, return_predictions=True
        )

        logger.log(f"\nFinal Test Set Performance:")
        logger.log(format_metrics(test_metrics, prefix=''))

        results_path = os.path.join(dirs['exp_dir'], 'predictions.txt')
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("Target\tPrediction\n")
            for target, pred in zip(test_targets, test_preds):
                f.write(f"{target:.4f}\t{pred:.4f}\n")
        logger.log(f"\nPrediction results saved: {results_path}")

        if len(val_losses) > 0:
            plot_training_curves(
                train_losses[::args.eval_interval],
                val_losses,
                os.path.join(dirs['exp_dir'], 'training_curves.png')
            )

        plot_predictions(
            test_preds,
            test_targets,
            os.path.join(dirs['exp_dir'], 'predictions.png')
        )

    logger.log("\n" + "="*60)
    logger.log("Training completed!")
    logger.log(f"Experiment directory: {dirs['exp_dir']}")
    logger.log("="*60 + "\n")



def main():
    """
    ========== loss ==========

    --loss_type evidential     # Evidential loss


    --loss_type focal          
                                # --focal_gamma 2.0

    --loss_type mse           
    
    --loss_type mae            

    """
    parser = argparse.ArgumentParser(description='training script for peptide MIC prediction')

    # ========== Arguments ==========
    parser.add_argument('--use_multigrain', action='store_true', default=True,
                        help='Enabled by default')
    parser.add_argument('--tech', type=str, default='transformer_evidential',  # actually no transformer layer
                        choices=['transformer_evidential', 'node_level_fusion', 'none'],
                        help='Model technique selection:\n'
                            '  transformer_evidential: use HybridMultiGrainGNN_Evidential (ACEL-ABP)\n')

    # ========== Data Arguments ==========
    parser.add_argument('--train_csv', type=str, default='data/0_93__10_structure/train.csv')

    parser.add_argument('--test_csv', type=str, default='data/0_93__10_structure/test.csv')

    parser.add_argument('--pdb_dir', type=str, default='pdb')
    parser.add_argument('--distance_threshold', type=float, default=8.0)


    # ========== Model Parameters ==========
    parser.add_argument('--model_type', type=str, default='cross_attention', #
                        help='Model type:\n'
                            '  Multi-granularity: cross_attention, concat')
    parser.add_argument('--input_dim', type=int, default=60)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--pooling', type=str, default='mean_sum',
                        choices=['mean', 'sum', 'mean_sum', 'max'])
    parser.add_argument('--conv_type', type=str, default='graphconv')
    parser.add_argument('--interaction_dim', type=int, default=256,
                        help='Interaction layer dimension (used in multi-granularity mode)')
    parser.add_argument('--fusion_strategy', type=str, default='structure_enhanced',
                        choices=['late', 'structure_enhanced', 'early', 'parallel'])

    # ========== Transformer/Evidential Parameters ==========
    parser.add_argument('--use_evidential', action='store_true', default=True,
                        help='Use evidential uncertainty quantification')
    parser.add_argument('--use_lm', type=lambda x: str(x).lower() in ('true', '1', 'yes'), default=True,
                        help='Whether to use a pretrained language model (default: True; set to False for ablation)')
    # ========== Language Model Parameters (multi-granularity only) ==========
    parser.add_argument('--lm_model', type=str, default='prot_t5_xl_uniref50',
                        choices=['prot_bert_bfd', 'prot_bert', 'prot_t5_xl_bfd',
                                'prot_t5_xl_uniref50', 'prot_xlnet', 'ProstT5',
                                'esm2_t6_8M_UR50D', 'esm2_t33_650M_UR50D'])
    parser.add_argument('--loss_type', type=str, default='evidential',
                        choices=['mse', 'mae', 'focal', 'evidential'],
                        help='Loss function type:\n'
                            '  mse: standard mean squared error\n'
                            '  focal: focal regression loss (focuses on hard samples)\n'
                            '  evidential: evidential loss (uncertainty quantification)')
    parser.add_argument('--evidential_coeff', type=float, default=0.01,  # can be fixed
                        help='Regularization coefficient for evidential loss (0.001-0.1)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,  # can be fixed
                        help='Gamma parameter for focal loss (1.0-5.0)')

    parser.add_argument('--focal_loss_base', type=str, default='mse',  # can be fixed
                        choices=['mse', 'mae', 'huber'],
                        help='Base loss type for focal loss')


    # ========== Training Parameters ==========
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32,  # 32
                        help='Batch size (16 recommended for multi-granularity)')
    parser.add_argument('--lr', type=float, default=0.0005)  # 0.001   0.0005  # can be fixed
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Evaluation interval')

    # ========== Scheduler ==========
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'step', 'none'])
    parser.add_argument('--scheduler_patience', type=int, default=10)
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    parser.add_argument('--scheduler_step_size', type=int, default=30)
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)

    # ========== Early Stopping ==========
    parser.add_argument('--early_stop', action='store_true', default=False)
    parser.add_argument('--early_stop_patience', type=int, default=20)

    # ========== RMSE Threshold Check (for automatic hyperparameter tuning) ==========
    parser.add_argument('--rmse_threshold', type=float, default=0.64,
                        help='RMSE threshold; values above this are considered unsatisfactory')
    parser.add_argument('--rmse_threshold_epoch', type=int, default=0,
                        help='Check RMSE threshold after this epoch; 0 means disabled (default: 0)')

    # ========== Save Parameters ==========
    parser.add_argument('--save_dir', type=str, default='experiments_structure')  # experiments  experiments_structure
    parser.add_argument('--exp_name', type=str, default='transformer_evidential')  # default no
    parser.add_argument('--save_interval', type=int, default=0)

    # ========== Other ==========
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()


    # Argument check
    if args.use_multigrain:
        if args.model_type not in ['cross_attention', 'bilinear', 'concat']:
            print(f"[WARN] Warning: in multi-granularity mode, model_type should be cross_attention/bilinear/concat")
            print(f"   Current: {args.model_type}, automatically switching to cross_attention")
            args.model_type = 'cross_attention'
    else:
        if args.model_type not in ['base', 'hybrid']:
            print(f"[WARN] Warning: in single-granularity mode, model_type should be base/hybrid")
            print(f"   Current: {args.model_type}, automatically switching to base")
            args.model_type = 'base'

    # If Evidential is disabled, automatically switch the loss function to MSE
    if not args.use_evidential and args.loss_type == 'evidential':
        args.loss_type = 'mse'
        print(f"[INFO] use_evidential=False, automatically switching loss_type to mse")

    # Start training
    train(args)



if __name__ == '__main__':
    main()
