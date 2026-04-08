import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

# ============= GCN =============

class DeterministicGraphConv(nn.Module):
    """
    h' = D^{-1} * A * h * W + b
    """

    def __init__(self, in_channels, out_channels, allow_zero_in_degree=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.allow_zero_in_degree = allow_zero_in_degree

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, g, node_feat):
        """
        Args:
            g: DGL 
            node_feat:  [num_nodes, in_channels]
        Returns:
             [num_nodes, out_channels]
        """
        with g.local_scope():
            num_nodes = g.num_nodes()
            device = node_feat.device

            
            src, dst = g.edges()

            
            adj = torch.zeros(num_nodes, num_nodes, device=device)
            adj[dst, src] = 1.0  # dst <- src 

            
            adj = adj + torch.eye(num_nodes, device=device)

            #  D^{-1/2} * A * D^{-1/2} (与 DGL GraphConv 一致)
            deg = adj.sum(dim=1).clamp(min=1)
            deg_inv_sqrt = deg.pow(-0.5)
            adj_norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

            #  h' = D^{-1/2} * A * D^{-1/2} * h
            h = torch.matmul(adj_norm, node_feat)

            #  h' * W + b
            out = torch.matmul(h, self.weight) + self.bias

            return out


def create_conv_layer(conv_type, in_channels, out_channels, num_filters=128, num_heads=4):
    """

    Args:
        conv_type: 'graphconv'
        in_channels
        out_channels
        num_filters
        num_heads
    """

    if conv_type == 'graphconv':
        return DeterministicGraphConv(in_channels, out_channels, allow_zero_in_degree=True)

    else:
        raise ValueError(f"Unknown conv_type: {conv_type}. Choose 'graphconv'.")


# ============= base model =============


class BaseGNN(nn.Module):
    """base GNN"""
    """

    def __init__(
        self,
        input_dim=59,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2,
        pooling='mean_sum',
        conv_type='schnet'  # 新增: 'schnet' or 'graphconv'
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        self.conv_type = conv_type

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(create_conv_layer(conv_type, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        if pooling == 'mean_sum':
            graph_repr_dim = hidden_dim * 2
        else:
            graph_repr_dim = hidden_dim

        self.fc1 = nn.Linear(graph_repr_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc_out = nn.Linear(hidden_dim // 4, 1)

    def forward(self, g):
        x = g.ndata['feat']
        edge_feat = g.edata.get('dist', None)

        x = F.relu(self.input_proj(x))

        for i in range(self.num_layers):
            if self.conv_type == 'schnet':
                x_new = self.convs[i](g, x, edge_feat)
            else:  # graphconv
                x_new = self.convs[i](g, x)
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # 残差

        # pooling
        x_graph = self._graph_pooling(g, x)

        x_graph = F.dropout(x_graph, p=self.dropout, training=self.training)
        x_graph = F.relu(self.fc1(x_graph))
        x_graph = F.dropout(x_graph, p=self.dropout, training=self.training)
        x_graph = F.relu(self.fc2(x_graph))

        out = self.fc_out(x_graph)
        return out.squeeze(-1)

    def _graph_pooling(self, g, node_feat):
        with g.local_scope():
            g.ndata['h'] = node_feat

            if self.pooling == 'mean':
                return dgl.mean_nodes(g, 'h')
            elif self.pooling == 'sum':
                return dgl.sum_nodes(g, 'h')
            elif self.pooling == 'max':
                return dgl.max_nodes(g, 'h')
            elif self.pooling == 'mean_sum':
                x_mean = dgl.mean_nodes(g, 'h')
                x_sum = dgl.sum_nodes(g, 'h')
                return torch.cat([x_mean, x_sum], dim=1)


# ============= cross attention =============

class CrossAttention(nn.Module):

    def __init__(self, seq_dim, struct_dim, hidden_dim=128):
        super().__init__()

        self.query_seq = nn.Linear(seq_dim, hidden_dim)
        self.key_struct = nn.Linear(struct_dim, hidden_dim)
        self.value_struct = nn.Linear(struct_dim, hidden_dim)

        self.query_struct = nn.Linear(struct_dim, hidden_dim)
        self.key_seq = nn.Linear(seq_dim, hidden_dim)
        self.value_seq = nn.Linear(seq_dim, hidden_dim)

        self.scale = hidden_dim ** -0.5

    def forward(self, seq_feat, struct_feat):
        # seq attend to struc
        Q_seq = self.query_seq(seq_feat)
        K_struct = self.key_struct(struct_feat)
        V_struct = self.value_struct(struct_feat)

        attn_seq = torch.softmax(Q_seq * K_struct * self.scale, dim=-1)
        seq_attended = attn_seq * V_struct

        # struc attend to seq
        Q_struct = self.query_struct(struct_feat)
        K_seq = self.key_seq(seq_feat)
        V_seq = self.value_seq(seq_feat)

        attn_struct = torch.softmax(Q_struct * K_seq * self.scale, dim=-1)
        struct_attended = attn_struct * V_seq

        return seq_attended, struct_attended


class BilinearFusion(nn.Module):

    def __init__(self, dim1, dim2, hidden_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(dim1, dim2, hidden_dim)
        self.linear1 = nn.Linear(dim1, hidden_dim)
        self.linear2 = nn.Linear(dim2, hidden_dim)

    def forward(self, x1, x2):
        bilinear_out = self.bilinear(x1, x2)
        linear1_out = self.linear1(x1)
        linear2_out = self.linear2(x2)

        fused = bilinear_out + linear1_out + linear2_out
        return F.relu(fused)


# ============= Evidential =============

class EvidentialLayer(nn.Module):
    """
    Evidential Deep Learning
    (gamma, nu, alpha, beta) -> Normal-Inverse-Gamma
        - value : gamma
        - epistemic: beta / (nu * (alpha - 1))
    """

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.gamma_head = nn.Linear(hidden_dim, 1) 
        self.nu_head = nn.Linear(hidden_dim, 1)     
        self.alpha_head = nn.Linear(hidden_dim, 1)  
        self.beta_head = nn.Linear(hidden_dim, 1)  

    def forward(self, x):
        """
        Args:
            x: [batch, input_dim]

        Returns:
            gamma: [batch, 1] 预测值
            nu: [batch, 1] 虚拟样本数 (>0)
            alpha: [batch, 1] 形状参数 (>1)
            beta: [batch, 1] 尺度参数 (>0)
        """
        h = self.hidden(x)

        gamma = self.gamma_head(h)
        nu = F.softplus(self.nu_head(h)) + 0.01
        alpha = F.softplus(self.alpha_head(h)) + 1.01  # alpha > 1
        beta = F.softplus(self.beta_head(h)) + 0.01

        return gamma, nu, alpha, beta


class EvidentialRegressionLoss(nn.Module):
    """
    Evidential回归损失函数

    Loss = NLL + λ * Regularization
    """

    def __init__(self, coeff=1.0):
        super().__init__()
        self.coeff = coeff

    def forward(self, gamma, nu, alpha, beta, target):
        """
        Args:
            gamma, nu, alpha, beta: evidential参数 [batch, 1]
            target: 真实值 [batch]

        Returns:
            loss: 标量损失
        """
        target = target.unsqueeze(-1)  # [batch, 1]

        # NLL (negative log likelihood)
        omega = 2 * beta * (1 + nu)
        nll = 0.5 * torch.log(torch.pi / nu) \
              - alpha * torch.log(omega) \
              + (alpha + 0.5) * torch.log(nu * (target - gamma)**2 + omega) \
              + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

        # 正则化项 (鼓励高alpha，降低不确定性)
        error = torch.abs(target - gamma)
        reg = error * (2 * nu + alpha)

        loss = (nll + self.coeff * reg).mean()

        return loss


class FocalRegressionLoss(nn.Module):
    """
    Focal Loss for Regression - 聚焦难样本

    基于: 多个2024会议论文
    FL(x) = |y - y_pred|^gamma * loss(x)

    gamma > 0: 难样本权重更高
    """

    def __init__(self, gamma=2.0, loss_type='mse'):
        super().__init__()
        self.gamma = gamma
        self.loss_type = loss_type

    def forward(self, pred, target):
        """
        Args:
            pred: [batch]
            target: [batch]
        """
        error = torch.abs(pred - target)

        # Focal权重: 误差越大，权重越高
        focal_weight = torch.pow(error, self.gamma)

        if self.loss_type == 'mse':
            base_loss = (pred - target) ** 2
        elif self.loss_type == 'mae':
            base_loss = error
        elif self.loss_type == 'huber':
            delta = 1.0
            base_loss = torch.where(
                error < delta,
                0.5 * error ** 2,
                delta * (error - 0.5 * delta)
            )
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        focal_loss = focal_weight * base_loss

        return focal_loss.mean()


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer层 with enhanced positional encoding

    基于:
    - NeurIPS 2024: "Enhancing Graph Transformers with Hierarchical Distance Structural Encoding"
    - IJCAI 2024: "Gradformer: Graph Transformer with Exponential Decay"
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x, pe=None):
        """
        Args:
            g: DGL图 (保留用于未来扩展，如边特征)
            x: 节点特征 [num_nodes, hidden_dim]
            pe: 位置编码 [num_nodes, hidden_dim] (可选)

        Returns:
            x: 更新后的节点特征
        """
        # 加入位置编码
        if pe is not None:
            x = x + pe

        # Multi-head Self-Attention
        residual = x
        x = self.norm1(x)

        B, D = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Attention计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).reshape(B, D)
        x = self.proj(x)
        x = self.dropout(x)
        x = residual + x

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


# ============= 工厂函数 =============

def get_model(
    use_multigrain=False,
    model_type='base',
    use_extra_features=False,
    ablation_tech=None,
    **kwargs
):
    """
    统一模型创建函数

    Args:
        use_multigrain: 是否多粒度模式
        model_type:
            - 单粒度: 'base', 'hybrid', 'simple', 'schnet', 'gat'
            - 多粒度: 'cross_attention', 'bilinear', 'concat'
        use_extra_features: 是否使用手工特征（仅多粒度模式有效）
        ablation_tech: 消融实验技术
            - 'transformer_evidential': Transformer + Evidential (唯一支持的消融模型)
            - None: 不使用消融模型
        **kwargs: 模型参数

    Returns:
        model

    Examples:
        # 消融实验: Transformer + Evidential
        >>> model = get_model(use_multigrain=True, use_extra_features=True,
        ...                   ablation_tech='transformer_evidential', model_type='cross_attention')

        # 多粒度 + 手工特征
        >>> model = get_model(use_multigrain=True, use_extra_features=True,
        ...                   model_type='cross_attention')
    """

    # 消融实验模型
    if ablation_tech is not None:
        if not use_multigrain:
            raise ValueError("ablation_tech requires use_multigrain=True")

        # 消融模型通用参数
        ablation_base_params = [
            'graph_input_dim', 'graph_hidden_dim', 'num_gnn_layers', 'lm_hidden_dim',
            'use_lm', 'interaction_hidden_dim', 'extra_feature_dim', 'fusion_strategy',
            'dropout', 'pooling', 'conv_type'
        ]
        ablation_kwargs = {k: v for k, v in kwargs.items() if k in ablation_base_params}

        # 如果没有使用额外特征，确保 extra_feature_dim = 0
        if not use_extra_features:
            ablation_kwargs['extra_feature_dim'] = 0

        if ablation_tech == 'transformer_evidential':
            # Transformer + Evidential (支持独立控制)
            if 'num_transformer_layers' in kwargs:
                ablation_kwargs['num_transformer_layers'] = kwargs['num_transformer_layers']
            if 'num_heads' in kwargs:
                ablation_kwargs['num_heads'] = kwargs['num_heads']
            if 'use_evidential' in kwargs:
                ablation_kwargs['use_evidential'] = kwargs['use_evidential']

            use_evid = ablation_kwargs.get('use_evidential', True)

            print(f"\n{'='*60}")
            print(f"[模型创建] 使用模型: HybridMultiGrainGNN_Evidential")
            print(f"  - 交互类型: {model_type}")
            print(f"  - 使用 Evidential: {use_evid}")
            print(f"{'='*60}\n")
            return HybridMultiGrainGNN_Evidential(
                interaction_type=model_type,
                **ablation_kwargs
            )

        else:
            raise ValueError(f"Unknown ablation_tech: {ablation_tech}. "
                           f"Supported: 'transformer_evidential', 'node_level_fusion'.")

    else:
        # 单粒度模型
        if model_type == 'base':
            print(f"\n{'='*60}")
            print(f" BaseGNN")
            print(f"{'='*60}\n")
            return BaseGNN(**kwargs)
        else:
            raise ValueError(f"Unknown single-grain model type: {model_type}. ")

# ============= ACEL-ABP =============
class HybridMultiGrainGNN_Evidential(nn.Module):
    """
        - use_transformer=False, use_evidential=True →  (Transformer:No , evidential loss:Yes)
        transformer层已关闭，仅使用Evidential
    """

    def __init__(
        self,
        graph_input_dim=60,
        graph_hidden_dim=128,
        num_gnn_layers=3,
        lm_hidden_dim=1024,
        use_lm=True,
        interaction_type='cross_attention',
        interaction_hidden_dim=256,
        extra_feature_dim=0,
        fusion_strategy='late',
        dropout=0.2,
        pooling='mean_sum',
        conv_type='schnet',
        num_transformer_layers=2,
        num_heads=4,
        use_transformer=False,  # close Transformer
        use_evidential=True     # use Evidential
    ):
        super().__init__()

        self.use_lm = use_lm
        self.interaction_type = interaction_type
        self.pooling = pooling
        self.extra_feature_dim = extra_feature_dim
        self.has_extra_features = extra_feature_dim > 0
        self.fusion_strategy = fusion_strategy
        self.dropout = dropout
        self.num_gnn_layers = num_gnn_layers
        self.conv_type = conv_type
        self.num_transformer_layers = num_transformer_layers
        self.use_transformer = use_transformer
        self.use_evidential = use_evidential

        # 打印消融配置
        print(f"\n[实验模型] 配置:")
        print(f"  - 使用语言模型(LM): {use_lm}")
        print(f"  - 使用 Evidential: {use_evidential}")
        print(f"  - 融合策略: {fusion_strategy}")

        # 细粒度分支
        self.input_proj = nn.Linear(graph_input_dim, graph_hidden_dim)
        self.gnn_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_convs.append(create_conv_layer(conv_type, graph_hidden_dim, graph_hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(graph_hidden_dim))

        # Graph Transformer层 (可选)
        if use_transformer:
            self.transformer_layers = nn.ModuleList([
                GraphTransformerLayer(graph_hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_transformer_layers)
            ])
        else:
            self.transformer_layers = None

        if pooling == 'mean_sum':
            self.graph_repr_dim = 2 * graph_hidden_dim
        else:
            self.graph_repr_dim = graph_hidden_dim

        self.seq_repr_dim = lm_hidden_dim

        if self.has_extra_features:
            feature_hidden = min(256, extra_feature_dim * 2)
            self.feature_mlp = nn.Sequential(
                nn.Linear(extra_feature_dim, feature_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(feature_hidden, interaction_hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )

        self._build_fusion_modules(interaction_type, interaction_hidden_dim, dropout)
        self.tokenizer = None
        self.pretrained_lm = None

    def _build_fusion_modules(self, interaction_type, interaction_hidden_dim, dropout):
        # ========== LM 消融模式：不使用序列表示 ==========
        if not self.use_lm:
            # 只使用结构表示 + 额外特征
            self.struct_proj = nn.Sequential(
                nn.Linear(self.graph_repr_dim, interaction_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            if self.has_extra_features:
                # 额外特征投影（已在__init__中定义feature_mlp，这里不需要重复）
                # 根据 interaction_type 选择融合方式
                if interaction_type == 'cross_attention':
                    self.struct_feat_interaction = CrossAttention(
                        seq_dim=interaction_hidden_dim,
                        struct_dim=interaction_hidden_dim,
                        hidden_dim=interaction_hidden_dim
                    )
                    fusion_input_dim = interaction_hidden_dim * 2
                elif interaction_type == 'bilinear':
                    self.struct_feat_interaction = BilinearFusion(
                        dim1=interaction_hidden_dim,
                        dim2=interaction_hidden_dim,
                        hidden_dim=interaction_hidden_dim
                    )
                    fusion_input_dim = interaction_hidden_dim
                else:  # concat - 直接拼接
                    self.struct_feat_interaction = None
                    fusion_input_dim = interaction_hidden_dim * 2
            else:
                fusion_input_dim = interaction_hidden_dim

            # 根据 use_evidential 选择预测头
            if self.use_evidential:
                self.predictor = EvidentialLayer(fusion_input_dim, hidden_dim=512)
            else:
                self.predictor = nn.Sequential(
                    nn.Linear(fusion_input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, 1)
                )
            return  # 提前返回，不构建序列相关模块

        # ========== 正常模式：使用序列 + 结构 融合 ==========
        if self.fusion_strategy == 'late':
            self.seq_proj = nn.Sequential(nn.Linear(self.seq_repr_dim, interaction_hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            self.graph_proj = nn.Sequential(nn.Linear(self.graph_repr_dim, interaction_hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            if interaction_type == 'cross_attention':
                self.interaction = CrossAttention(seq_dim=interaction_hidden_dim, struct_dim=interaction_hidden_dim, hidden_dim=interaction_hidden_dim)
                multigrain_fusion_dim = interaction_hidden_dim * 2
            elif interaction_type == 'bilinear':
                self.interaction = BilinearFusion(dim1=interaction_hidden_dim, dim2=interaction_hidden_dim, hidden_dim=interaction_hidden_dim)
                multigrain_fusion_dim = interaction_hidden_dim
            else:
                self.interaction = None
                multigrain_fusion_dim = interaction_hidden_dim * 2
            fusion_input_dim = multigrain_fusion_dim + (interaction_hidden_dim if self.has_extra_features else 0)
        elif self.fusion_strategy == 'structure_enhanced':
            self.seq_proj = nn.Sequential(nn.Linear(self.seq_repr_dim, interaction_hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            if self.has_extra_features:
                self.struct_base_proj = nn.Linear(self.graph_repr_dim, interaction_hidden_dim)
                self.gate = nn.Sequential(nn.Linear(interaction_hidden_dim * 2, interaction_hidden_dim), nn.Sigmoid())
                self.enhanced_proj = nn.Sequential(nn.Linear(interaction_hidden_dim * 2, interaction_hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            else:
                self.struct_base_proj = nn.Sequential(nn.Linear(self.graph_repr_dim, interaction_hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            if interaction_type == 'cross_attention':
                self.interaction = CrossAttention(seq_dim=interaction_hidden_dim, struct_dim=interaction_hidden_dim, hidden_dim=interaction_hidden_dim)
                fusion_input_dim = interaction_hidden_dim * 2
            elif interaction_type == 'bilinear':
                self.interaction = BilinearFusion(dim1=interaction_hidden_dim, dim2=interaction_hidden_dim, hidden_dim=interaction_hidden_dim)
                fusion_input_dim = interaction_hidden_dim
            else:
                self.interaction = None
                fusion_input_dim = interaction_hidden_dim * 2
        elif self.fusion_strategy == 'early':
            if self.has_extra_features:
                self.seq_feat_fusion = nn.Sequential(nn.Linear(self.seq_repr_dim + interaction_hidden_dim, interaction_hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            else:
                self.seq_feat_fusion = nn.Sequential(nn.Linear(self.seq_repr_dim, interaction_hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            self.graph_proj = nn.Sequential(nn.Linear(self.graph_repr_dim, interaction_hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            if interaction_type == 'cross_attention':
                self.interaction = CrossAttention(seq_dim=interaction_hidden_dim, struct_dim=interaction_hidden_dim, hidden_dim=interaction_hidden_dim)
                fusion_input_dim = interaction_hidden_dim * 2
            elif interaction_type == 'bilinear':
                self.interaction = BilinearFusion(dim1=interaction_hidden_dim, dim2=interaction_hidden_dim, hidden_dim=interaction_hidden_dim)
                fusion_input_dim = interaction_hidden_dim
            else:
                self.interaction = None
                fusion_input_dim = interaction_hidden_dim * 2
        elif self.fusion_strategy == 'parallel':
            self.seq_proj = nn.Sequential(nn.Linear(self.seq_repr_dim, interaction_hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            self.graph_proj = nn.Sequential(nn.Linear(self.graph_repr_dim, interaction_hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            if interaction_type == 'cross_attention':
                self.interaction1 = CrossAttention(seq_dim=interaction_hidden_dim, struct_dim=interaction_hidden_dim, hidden_dim=interaction_hidden_dim)
                path1_dim = interaction_hidden_dim * 2
            elif interaction_type == 'bilinear':
                self.interaction1 = BilinearFusion(dim1=interaction_hidden_dim, dim2=interaction_hidden_dim, hidden_dim=interaction_hidden_dim)
                path1_dim = interaction_hidden_dim
            else:
                self.interaction1 = None
                path1_dim = interaction_hidden_dim * 2
            if self.has_extra_features:
                self.interaction2 = nn.Sequential(nn.Linear(interaction_hidden_dim * 2, interaction_hidden_dim), nn.ReLU(), nn.Dropout(dropout))
                fusion_input_dim = path1_dim + interaction_hidden_dim
            else:
                fusion_input_dim = path1_dim
        else:
            raise ValueError(f"Unknown fusion_strategy: {self.fusion_strategy}")

        # 根据 use_evidential 选择预测头
        if self.use_evidential:
            # Evidential预测头 (输出不确定性)
            self.predictor = EvidentialLayer(fusion_input_dim, hidden_dim=512)
        else:
            # 普通MLP预测头 (用于MSE损失)
            self.predictor = nn.Sequential(
                nn.Linear(fusion_input_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            )

    def set_language_model(self, tokenizer, pretrained_lm):
        self.tokenizer = tokenizer
        self.pretrained_lm = pretrained_lm

    def _graph_pooling(self, g, x):
        with g.local_scope():
            g.ndata['h'] = x
            if self.pooling == 'mean':
                return dgl.mean_nodes(g, 'h')
            elif self.pooling == 'sum':
                return dgl.sum_nodes(g, 'h')
            elif self.pooling == 'max':
                return dgl.max_nodes(g, 'h')
            else:
                return torch.cat([dgl.mean_nodes(g, 'h'), dgl.sum_nodes(g, 'h')], dim=1)

    def _forward_late(self, seq_repr, struct_repr, feature_repr):
        seq_feat = self.seq_proj(seq_repr)
        struct_feat = self.graph_proj(struct_repr)
        if self.interaction_type == 'cross_attention':
            seq_attended, struct_attended = self.interaction(seq_feat, struct_feat)
            multigrain_fused = torch.cat([seq_attended, struct_attended], dim=1)
        elif self.interaction_type == 'bilinear':
            multigrain_fused = self.interaction(seq_feat, struct_feat)
        else:
            multigrain_fused = torch.cat([seq_feat, struct_feat], dim=1)
        if feature_repr is not None:
            return torch.cat([multigrain_fused, feature_repr], dim=1)
        return multigrain_fused

    def _forward_structure_enhanced(self, seq_repr, struct_repr, feature_repr):
        seq_feat = self.seq_proj(seq_repr)
        if feature_repr is not None:
            struct_base = self.struct_base_proj(struct_repr)
            combined = torch.cat([struct_base, feature_repr], dim=1)
            gate = torch.sigmoid(self.gate(combined))  # 输出门控权重 [0,1]
            enhanced_struct = self.enhanced_proj(combined)
            struct_feat = gate * enhanced_struct + (1 - gate) * struct_base
        else:
            struct_feat = self.struct_base_proj(struct_repr)
        if self.interaction_type == 'cross_attention':
            seq_attended, struct_attended = self.interaction(seq_feat, struct_feat)
            return torch.cat([seq_attended, struct_attended], dim=1)
        elif self.interaction_type == 'bilinear':
            return self.interaction(seq_feat, struct_feat)
        else:
            return torch.cat([seq_feat, struct_feat], dim=1)

    def _forward_early(self, seq_repr, struct_repr, feature_repr):
        if feature_repr is not None:
            seq_feat = self.seq_feat_fusion(torch.cat([seq_repr, feature_repr], dim=1))
        else:
            seq_feat = self.seq_feat_fusion(seq_repr)
        struct_feat = self.graph_proj(struct_repr)
        if self.interaction_type == 'cross_attention':
            seq_attended, struct_attended = self.interaction(seq_feat, struct_feat)
            return torch.cat([seq_attended, struct_attended], dim=1)
        elif self.interaction_type == 'bilinear':
            return self.interaction(seq_feat, struct_feat)
        else:
            return torch.cat([seq_feat, struct_feat], dim=1)

    def _forward_parallel(self, seq_repr, struct_repr, feature_repr):
        seq_feat = self.seq_proj(seq_repr)
        struct_feat = self.graph_proj(struct_repr)
        if self.interaction_type == 'cross_attention':
            seq_attended, struct_attended = self.interaction1(seq_feat, struct_feat)
            path1_feat = torch.cat([seq_attended, struct_attended], dim=1)
        elif self.interaction_type == 'bilinear':
            path1_feat = self.interaction1(seq_feat, struct_feat)
        else:
            path1_feat = torch.cat([seq_feat, struct_feat], dim=1)
        if feature_repr is not None:
            path2_feat = self.interaction2(torch.cat([struct_feat, feature_repr], dim=1))
            return torch.cat([path1_feat, path2_feat], dim=1)
        return path1_feat

    def forward(self, batch, return_uncertainty=False):
        seq_encoded = batch['seq_encoded']
        graph = batch['graph']
        x = graph.ndata['feat']
        edge_feat = graph.edata.get('dist', None)
        x = F.relu(self.input_proj(x))

        # GNN层
        for i in range(self.num_gnn_layers):
            if self.conv_type == 'schnet':
                x_new = self.gnn_convs[i](graph, x, edge_feat)
            else:
                x_new = self.gnn_convs[i](graph, x)
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new

        # Transformer层 (可选)
        if self.use_transformer and self.transformer_layers is not None:
            for transformer_layer in self.transformer_layers:
                x = transformer_layer(graph, x)

        struct_repr = self._graph_pooling(graph, x)

        # ========== LM 消融模式：只使用结构表示 ==========
        if not self.use_lm:
            # 结构表示投影
            struct_feat = self.struct_proj(struct_repr)

            # 处理额外特征
            if self.has_extra_features and hasattr(graph, 'extra_features') and graph.extra_features is not None:
                feature_repr = self.feature_mlp(graph.extra_features)
                # 根据 interaction_type 融合结构 + 额外特征
                if hasattr(self, 'struct_feat_interaction') and self.struct_feat_interaction is not None:
                    if self.interaction_type == 'cross_attention':
                        struct_attended, feat_attended = self.struct_feat_interaction(struct_feat, feature_repr)
                        final_fused = torch.cat([struct_attended, feat_attended], dim=1)
                    elif self.interaction_type == 'bilinear':
                        final_fused = self.struct_feat_interaction(struct_feat, feature_repr)
                else:  # concat - 直接拼接
                    final_fused = torch.cat([struct_feat, feature_repr], dim=1)
            else:
                final_fused = struct_feat

            # 预测
            if self.use_evidential:
                gamma, nu, alpha, beta = self.predictor(final_fused)
                pred = gamma.squeeze(-1)
                if return_uncertainty:
                    aleatoric = (beta / (alpha - 1)).squeeze(-1)
                    epistemic = (beta / (nu * (alpha - 1))).squeeze(-1)
                    return pred, aleatoric, epistemic, (gamma, nu, alpha, beta)
                return pred
            else:
                pred = self.predictor(final_fused)
                return pred.squeeze(-1)

        # ========== 正常模式：使用序列 + 结构 融合 ==========
        if self.tokenizer is not None and self.pretrained_lm is not None:
            from utils import get_lm_embedding_
            seq_repr = get_lm_embedding_(seqs_al=seq_encoded, tokenizer=self.tokenizer, pretrained_lm=self.pretrained_lm, use_lm=True)
            seq_repr = seq_repr.mean(dim=1)
        else:
            # LM未设置但use_lm=True，使用zero placeholder (不应该发生)
            batch_size = struct_repr.shape[0]
            seq_repr = torch.zeros(batch_size, self.seq_repr_dim, device=struct_repr.device)

        if self.has_extra_features and hasattr(graph, 'extra_features') and graph.extra_features is not None:
            feature_repr = self.feature_mlp(graph.extra_features)
        else:
            feature_repr = None

        if self.fusion_strategy == 'late':
            final_fused = self._forward_late(seq_repr, struct_repr, feature_repr)
        elif self.fusion_strategy == 'structure_enhanced':
            final_fused = self._forward_structure_enhanced(seq_repr, struct_repr, feature_repr)
        elif self.fusion_strategy == 'early':
            final_fused = self._forward_early(seq_repr, struct_repr, feature_repr)
        elif self.fusion_strategy == 'parallel':
            final_fused = self._forward_parallel(seq_repr, struct_repr, feature_repr)

        # 根据 use_evidential 选择预测方式
        if self.use_evidential:
            # Evidential预测
            gamma, nu, alpha, beta = self.predictor(final_fused)
            pred = gamma.squeeze(-1)

            if return_uncertainty:
                aleatoric = (beta / (alpha - 1)).squeeze(-1)
                epistemic = (beta / (nu * (alpha - 1))).squeeze(-1)
                return pred, aleatoric, epistemic, (gamma, nu, alpha, beta)
            return pred
        else:
            # 普通MLP预测
            pred = self.predictor(final_fused)
            return pred.squeeze(-1)


