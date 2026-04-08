"""
工具函数模块
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import shutil
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import dgl
from dgl import shortest_dist
import time
import random
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
from dgl.nn import GATConv
from typing import Optional
import numpy as np
import gc
from transformers import BertModel, BertTokenizer,\
                         T5Tokenizer, T5EncoderModel,\
                         AlbertModel, AlbertTokenizer,\
                         XLNetModel, XLNetTokenizer,\
                         EsmTokenizer
import esm


def set_seed(seed=42):
    """设置随机种子以保证可复现性"""
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 确保 cuBLAS 确定性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 强制使用确定性算法（PyTorch 1.8+）
    try:
        torch.use_deterministic_algorithms(True)
    except:
        pass
    # DGL 随机种子
    try:
        import dgl
        dgl.seed(seed)
        dgl.random.seed(seed)
    except:
        pass


def worker_init_fn(worker_id):
    """DataLoader worker 初始化函数，确保多进程数据加载的可复现性"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(gpu_id=0):
    """获取计算设备"""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"使用GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable
    }


def save_checkpoint(model, optimizer, epoch, metrics, filepath, model_config=None):
    """保存模型检查点（包含模型配置信息）"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'model_class': model.__class__.__name__,  # 模型类名
        'model_config': model_config  # 模型配置参数
    }
    torch.save(checkpoint, filepath)
    print(f"  保存检查点: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """加载模型检查点"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"  加载检查点: {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Metrics: {checkpoint['metrics']}")

    return checkpoint['epoch'], checkpoint['metrics']


def calculate_metrics(preds, targets):
    """
    计算回归指标

    Args:
        preds: 预测值 numpy array
        targets: 真实值 numpy array

    Returns:
        metrics: dict包含各种指标
    """
    preds = np.array(preds)
    targets = np.array(targets)

    # MSE
    mse = np.mean((preds - targets) ** 2)

    # RMSE
    rmse = np.sqrt(mse)

    # MAE
    mae = np.mean(np.abs(preds - targets))

    # R²
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Pearson相关系数 (PCC)
    if len(preds) > 1:
        pearson = np.corrcoef(preds, targets)[0, 1]
    else:
        pearson = 0.0

    # Spearman秩相关系数 (SCC)
    from scipy.stats import spearmanr
    if len(preds) > 1:
        spearman, _ = spearmanr(preds, targets)
    else:
        spearman = 0.0

    # Recall@K：取活性最强的top-K真实值与预测值的交集比例，K=50
    K = min(50, len(targets))
    top_k_true = set(np.argsort(targets)[::-1][:K])
    top_k_pred = set(np.argsort(preds)[::-1][:K])
    recall_at_k = len(top_k_true & top_k_pred) / K

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson,
        'spearman': spearman,
        'recall_at_50': recall_at_k,
    }


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=20, min_delta=0.0, mode='min'):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改进量
            mode: 'min' 或 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        Args:
            score: 当前指标值

        Returns:
            is_best: 是否是最佳
        """
        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_metrics(metrics, prefix=''):
    """格式化指标输出"""
    metric_str = []
    for key, value in metrics.items():
        metric_str.append(f"{prefix}{key.upper()}: {value:.4f}")
    return " | ".join(metric_str)


def print_model_info(model):
    """打印模型信息"""
    print("\n" + "=" * 60)
    print("模型信息")
    print("=" * 60)
    print(f"模型类型: {model.__class__.__name__}")

    params = count_parameters(model)
    print(f"总参数量: {params['total']:,}")
    print(f"可训练参数: {params['trainable']:,}")

    print("\n模型结构:")
    print(model)
    print("=" * 60 + "\n")


def create_exp_dir(exp_name, base_dir='experiments'):
    """创建实验目录"""
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 创建子目录
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    logs_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    return {
        'exp_dir': exp_dir,
        'checkpoints_dir': checkpoints_dir,
        'logs_dir': logs_dir
    }


class Logger:
    """简单的日志记录器"""

    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, message, print_msg=True):
        """记录日志"""
        if print_msg:
            print(message)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')


def normalize_features(features, mean=None, std=None):
    """
    标准化特征

    Args:
        features: numpy array or torch tensor
        mean: 均值 (None则计算)
        std: 标准差 (None则计算)

    Returns:
        normalized_features, mean, std
    """
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)

    # 避免除以0
    std = np.where(std == 0, 1.0, std)

    normalized = (features - mean) / std

    return normalized, mean, std


def get_lr(optimizer):
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def plot_training_curves(train_losses, val_losses, save_path):
    """
    绘制训练曲线

    Args:
        train_losses: list of training losses
        val_losses: list of validation losses
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Curves', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  训练曲线已保存: {save_path}")
    except ImportError:
        print("  警告: matplotlib未安装，无法绘制训练曲线")


def plot_predictions(preds, targets, save_path):
    """
    绘制预测vs真实值散点图

    Args:
        preds: 预测值
        targets: 真实值
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.scatter(targets, preds, alpha=0.6, s=50)

        # 绘制y=x线
        min_val = min(min(targets), min(preds))
        max_val = max(max(targets), max(preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Predictions', fontsize=12)
        plt.title('Predictions vs True Values', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  预测散点图已保存: {save_path}")
    except ImportError:
        print("  警告: matplotlib未安装，无法绘制散点图")


seq_re_dir = {
    1: 'A', 2: 'R', 3: 'N', 4: 'D', 5: 'C',
    6: 'Q', 7: 'E', 8: 'G', 9: 'H', 10: 'I',
    11: 'L', 12: 'K', 13: 'M', 14: 'F', 15: 'P',
    16: 'S', 17: 'T', 18: 'W', 19: 'Y', 20: 'V'
}

def get_lm_embedding_(seqs_al: torch.Tensor, tokenizer, pretrained_lm, use_lm: bool):
    # 如果使用语言模型
    if use_lm:
        freeze_layers = True  # 冻结语言模型的参数
        if freeze_layers:
            # T5 模型冻结操作
            if isinstance(tokenizer, T5Tokenizer) and isinstance(pretrained_lm, T5EncoderModel):
                for layer in pretrained_lm.encoder.block[:29]:
                    for param in layer.parameters():
                        param.requires_grad = False
            # BERT 模型冻结操作
            elif isinstance(tokenizer, BertTokenizer) and isinstance(pretrained_lm, BertModel):
                modules = [pretrained_lm.embeddings, *pretrained_lm.encoder.layer[:29]]
                for module in modules:
                    for param in module.parameters():
                        param.requires_grad = False
            # XLNet 模型冻结操作
            elif isinstance(tokenizer, XLNetTokenizer) and isinstance(pretrained_lm, XLNetModel):
                modules = [pretrained_lm.word_embedding, *pretrained_lm.layer[:29]]
                for module in modules:
                    for param in module.parameters():
                        param.requires_grad = False

        #"""
    # 如果使用的是 ESM 模型 对每条序列编码
    #print(f"Tokenizer type: {type(tokenizer)}")
    if isinstance(tokenizer, EsmTokenizer):
        seqs = seqs_al
        sequences = seqs
        seq_strings = []
        for seq in sequences:
            seq_string = ''.join([seq_re_dir[int(num.item())] for num in seq if int(num.item()) in seq_re_dir])
            seq_strings.append(seq_string)
        sequences = [''.join(list(seq)) for seq in seq_strings]
        # 使用tokenizer对输入的序列进行标记化处理
        ids = tokenizer(sequences, padding='max_length', truncation=True, max_length=30, return_tensors='pt')
        ids = {key: value.to(seqs.device) for key, value in ids.items()}

        input_ids = ids['input_ids'].to(pretrained_lm.device)  # 将输入序列转换为input_ids
        attention_mask = ids['attention_mask'].to(pretrained_lm.device)  # 生成attention_mask
        
        # 通过BERT提取特征
        with torch.no_grad():
            bert_output = pretrained_lm(input_ids=input_ids, attention_mask=attention_mask)

        # 从BERT的输出中获取logits (直接使用模型输出维度,不强制转换)
        output_feature = bert_output["logits"].to(seqs_al.device)
        output_feature = output_feature.unsqueeze(1).repeat(1, seqs_al.size(1), 1)
        features_tensor = output_feature.to(seqs_al.device)
    #"""
    
    else:
        seq_strings = []
        for seq in seqs_al:
            seq_string = ''.join([seq_re_dir[int(num.item())] for num in seq if int(num.item()) in seq_re_dir])
            seq_strings.append(seq_string)
        sequence = [' '.join(list(seq)) for seq in seq_strings]
        #print(sequence)
        #sequence = re.sub(r"[UZOB]", "X", sequence)
        ids = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True,max_length=30)
        #encoded_input = tokenizer(sequence, return_tensors='pt', max_length=255)
        # 将 encoded_input 转移到 seqs_al 的设备上
        ids = {key: value.to(seqs_al.device) for key, value in ids.items()}
        input_ids = ids['input_ids']
        attention_mask = ids['attention_mask']
        #print(input_ids.shape)
        #print(attention_mask.shape)
        output= []
        lm_embedding= []
        if use_lm is True:
            with torch.no_grad():
                output = pretrained_lm(input_ids)
                #print(output.last_hidden_state.shape)
                lm_embedding = output.last_hidden_state
        else:
            #print("Hey! I am hot")
            seqs_al = seqs_al.long()
            with torch.no_grad():
                lm_embedding = F.one_hot(seqs_al, num_classes=1024)
        features = [] 
        max_len = 30 # sequence length
        for seq_num in range(len(lm_embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = lm_embedding[seq_num][:seq_len]
            features.append(seq_emd)
            max_len = max(max_len, seq_emd.size(0))

        # 对所有特征进行 padding 处理并堆叠成张量
        padded_features = [F.pad(feature, (0, 0, 0, max_len - feature.size(0))) for feature in features]
        features_tensor = torch.stack(padded_features)
    #one-hot 

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return features_tensor


def save_code_snapshot(exp_dir, code_files=None):

    if code_files is None:
        code_files = ['models.py', 'dataset.py', 'train.py', 'utils.py']

    code_dir = os.path.join(exp_dir, 'code_snapshot')
    os.makedirs(code_dir, exist_ok=True)

    saved_files = []
    for filename in code_files:
        if os.path.exists(filename):
            dest_path = os.path.join(code_dir, filename)
            shutil.copy2(filename, dest_path)
            saved_files.append(filename)


    snapshot_info = os.path.join(code_dir, 'snapshot_info.txt')
    with open(snapshot_info, 'w', encoding='utf-8') as f:
        f.write(f"Code Snapshot\n")
        f.write(f"={'='*50}\n")
        f.write(f"Saved files:\n")
        for filename in saved_files:
            f.write(f"  - {filename}\n")

    return saved_files
