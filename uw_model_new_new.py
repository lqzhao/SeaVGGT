import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

class PrototypeSet(nn.Module):
    def __init__(self, token_dim=2048, num_prototypes=16):
        super().__init__()
        self.token_dim = token_dim
        self.num_prototypes = num_prototypes

        self.global_prototype = nn.Parameter(torch.ones(token_dim))
        self.local_prototypes = nn.Parameter(torch.randn(num_prototypes, token_dim))

        self.prototype_interaction = nn.Sequential(
            nn.Linear(token_dim, token_dim // 4),
            nn.GELU(),
            nn.Linear(token_dim // 4, token_dim)
        )

        self.depthwise_conv = nn.Conv1d(
            in_channels=token_dim,
            out_channels=token_dim,
            kernel_size=1,
            groups=token_dim,
            bias=False
        )

        self.beta_mlp = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.scale_mlp = ScaleMLP(token_dim)

        self.spatial_adaptor = nn.Sequential(
            nn.Conv2d(token_dim, token_dim // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(token_dim // 16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, tokens, cond=None):
        """
        tokens: [1, H, N, C]
        cond: [C] ← refined_prototype from GAT
        """
        H, N, C = tokens.shape[1:]
        assert C == self.token_dim

        # 融合 GAT refined prototype 与 global_prototype
        if cond is not None:
            gate = torch.sigmoid(torch.dot(self.global_prototype, cond) / C)
            global_proto = gate * self.global_prototype + (1 - gate) * cond
        else:
            global_proto = self.global_prototype

        global_scaled = tokens * global_proto.view(1, 1, 1, -1)

        tokens_reshape = global_scaled.view(-1, C, 1)
        keys = self.depthwise_conv(tokens_reshape).squeeze(-1)

        interacted_prototypes = self.local_prototypes + self.prototype_interaction(self.local_prototypes)

        attn_scores = torch.matmul(keys, interacted_prototypes.t()) / (C ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        local_bias = torch.matmul(attn_weights, interacted_prototypes).view(1, H, N, C)

        spatial_weights = self.spatial_adaptor(tokens.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        modulated_tokens = spatial_weights * global_scaled + (1 - spatial_weights) * local_bias

        beta = self.beta_mlp(global_proto).view(1, 1, 1, 3)
        scale = self.scale_mlp(global_proto).view(1, 1, 1, 2)

        return modulated_tokens, beta, scale


class TokenPrototypeModulator(nn.Module):
    def __init__(self, token_dim=2048, num_prototypes=16):
        super().__init__()
        self.token_dim = token_dim
        self.num_prototypes = num_prototypes

        self.prototype_sets = nn.ModuleList([
            PrototypeSet(token_dim, num_prototypes)
            for _ in range(num_prototypes)
        ])

        # Learnable prototype centers with soft assignment
        A_vals = get_fixed_As(num_prototypes)
        self.A_values = nn.Parameter(A_vals)  # Learnable prototype centers
        
        # 修复错误：简化熵值计算，输入维度改为1
        self.entropy_mlp = nn.Sequential(
            nn.Linear(1, 16),  # 输入维度改为1
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出归一化到[0,1]
        )
        
        # Prototype usage tracker
        self.register_buffer('usage_count', torch.zeros(num_prototypes))

    def forward(self, aggregated_tokens_list, A_pred):
        """
        A_pred: [1, 3] (batch size=1)
        aggregated_tokens_list: list of [1, H, N, C] tensors
        """
        num_layers = len(aggregated_tokens_list)
        
        # 修复错误：正确计算熵值
        # 使用torch.clamp防止log(0)
        # A_prob = F.softmax(A_pred, dim=1)  # [B,3]


        clamped_A = torch.clamp(A_pred, min=1e-8)
        entropy = -torch.sum(clamped_A * torch.log(clamped_A), dim=1)  # [1]
        
        # 修复错误：正确计算动态k值
        entropy_factor = self.entropy_mlp(entropy.view(1, 1)).squeeze()  # 标量
        # entropy_factor = self.entropy_mlp(torch.ones_like(entropy.view(1, 1))).squeeze()  # 标量
        k = max(1, min(self.num_prototypes, 
                     int(round(entropy_factor.item() * (self.num_prototypes - 1) + 1))))

        # k = max(4, min(self.num_prototypes, 
        #                int(round(entropy_factor.item() * (self.num_prototypes - 1) + 1))))
        
        # Calculate similarity (single sample)
        sigmoid_A = torch.sigmoid(self.A_values)  # Constrain to [0,1]
        similarities = torch.exp(-10 * torch.norm(A_pred - sigmoid_A, dim=1))  # [P]

        
        # Select top-k prototypes
        topk_vals, topk_indices = torch.topk(similarities, k)
        weights = topk_vals / topk_vals.sum()
        
        # Update usage count
        self.usage_count[topk_indices] += 1
        
        # Only activate selected prototypes
        modulated_tokens_list = [torch.zeros_like(tokens) for tokens in aggregated_tokens_list]
        beta_stack = torch.zeros(1, 1, 1, 3).to(A_pred.device)
        scale_stack = torch.zeros(1, 1, 1, 2).to(A_pred.device)
        
        for l in range(num_layers):
            layer_tokens = aggregated_tokens_list[l]  # [1, H, N, C]
            
            for i, idx in enumerate(topk_indices):
                ps = self.prototype_sets[idx]
                w = weights[i].item()
                
                # Process tokens with this prototype set
                mod_tokens, beta, scale = ps(layer_tokens)
                
                # Weighted sum
                modulated_tokens_list[l] += w * mod_tokens
                
                if l == 0:
                    beta_stack += w * beta
                    scale_stack += w * scale
        
        # Get A_value for the most relevant prototype
        # A_val = sigmoid_A[topk_indices[0]].view(1, 3)
        A_val = (weights.unsqueeze(1) * sigmoid_A[topk_indices]).sum(0, keepdim=True)  # [1,3]

        # pdb.set_trace()
        # visualize_entropy_mapping(self.entropy_mlp)
        # visualize_As(sigmoid_A, save_path='A_values_vis.png', title='Learned A Prototypes')

        return modulated_tokens_list, A_val, beta_stack, scale_stack

    def get_most_used_prototypes(self, top_n=5):
        """获取最常使用的原型索引"""
        _, indices = torch.topk(self.usage_count, top_n)
        return indices.cpu().numpy()

# MLP to map global prototype to scale parameters
class ScaleMLP(nn.Module):
    def __init__(self, token_dim):
        super().__init__()
        self.fc1 = nn.Linear(token_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x1 = F.softplus(x[0])  # Ensure positive
        x2 = torch.sigmoid(x[1])  # Constrain to [0,1]
        return torch.stack([x1, x2])

def get_fixed_As(num_prototypes):
    # assert num_prototypes % 6 == 0, "建议数量为8的倍数，便于均分类型"

    num_types = 6  # blue / green / yellow / cyan / red / dark
    num_each = num_prototypes // num_types

    def lin_range(low, high, n):
        return torch.linspace(low, high, n)

    # 清澈蓝水
    A_R_blue = lin_range(0.3, 0.6, num_each)
    A_G_blue = lin_range(0.5, 0.8, num_each)
    A_B_blue = lin_range(0.8, 1.0, num_each)

    # 绿水
    A_R_green = lin_range(0.3, 0.5, num_each)
    A_G_green = lin_range(0.7, 1.0, num_each)
    A_B_green = lin_range(0.4, 0.7, num_each)

    # 黄绿浑水
    A_R_yellow = lin_range(0.5, 0.9, num_each)
    A_G_yellow = lin_range(0.6, 0.9, num_each)
    A_B_yellow = lin_range(0.3, 0.6, num_each)

    # 青绿色水体
    A_R_cyan = lin_range(0.4, 0.7, num_each)
    A_G_cyan = lin_range(0.6, 0.9, num_each)
    A_B_cyan = lin_range(0.5, 0.8, num_each)

    # 日落红水 / 红色水体
    A_R_red = lin_range(0.7, 1.0, num_each)
    A_G_red = lin_range(0.2, 0.5, num_each)
    A_B_red = lin_range(0.2, 0.4, num_each)

    # 黑水 / 深海 / 浑浊
    A_R_dark = lin_range(0.05, 0.3, num_each)
    A_G_dark = lin_range(0.05, 0.3, num_each)
    A_B_dark = lin_range(0.05, 0.3, num_each)

    # 拼接
    A_R = torch.cat([A_R_blue, A_R_green, A_R_yellow, A_R_cyan, A_R_red, A_R_dark], dim=0)
    A_G = torch.cat([A_G_blue, A_G_green, A_G_yellow, A_G_cyan, A_G_red, A_G_dark], dim=0)
    A_B = torch.cat([A_B_blue, A_B_green, A_B_yellow, A_B_cyan, A_B_red, A_B_dark], dim=0)

    A_vals = torch.stack([A_R, A_G, A_B], dim=1)  # [num_prototypes, 3]
    return A_vals


def get_random_As(num_prototypes, low=0.0, high=1.0):
    """
    随机生成 A 值（颜色原型）
    输出形状: [num_prototypes, 3]
    每个通道在 [low, high] 内随机生成
    """
    return torch.rand(num_prototypes, 3) * (high - low) + low


def visualize_As(A_values, save_path, highlight_idx=None, title=None, h=50, w=100):
    """
    以列优先顺序可视化 A_values（N×3）为 RGB 色块，
    每列 4 个，共最多支持 6 列（最多 24 个 prototype）

    参数:
        A_values: Tensor (N, 3) in [0,1]
        save_path: 图片保存路径
        highlight_idx: 要高亮的 prototype 索引（可选）
        title: 图片上方标题（可选）
    """
    num_prototypes = A_values.shape[0]
    assert num_prototypes <= 24, "最多支持 24 个 prototype 的可视化"

    rows, cols = 4, 6  # 固定为 4 行 6 列
    # h, w = 50, 100  # 每个格子的大小

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.8, rows * 0.2))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for idx in range(rows * cols):
        row = idx % rows
        col = idx // rows
        ax = axes[row][col]

        if idx < num_prototypes:
            A_val = A_values[idx].detach().cpu().numpy()
            color_img = np.ones((h, w, 3), dtype=np.float32) * A_val
            color_img_uint8 = (color_img * 255).astype(np.uint8)

            ax.imshow(color_img_uint8)
            # ax.set_title(f'P{idx}', fontsize=8)
            ax.axis('off')

            if highlight_idx is not None and idx == highlight_idx:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
        else:
            ax.axis('off')  # 隐藏空格子

    if title:
        fig.suptitle(title, fontsize=14)

    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved A visualization to {save_path}")

def visualize_As_with_number(A_values, save_path, highlight_idx=None, title=None, h=50, w=100):
    """
    以列优先顺序可视化 A_values（N×3）为 RGB 色块，
    每列 4 个，共最多支持 6 列（最多 24 个 prototype）
    并在每个块上显示编号 0-23。

    参数:
        A_values: Tensor (N, 3) in [0,1]
        save_path: 图片保存路径
        highlight_idx: 要高亮的 prototype 索引（可选）
        title: 图片上方标题（可选）
    """
    num_prototypes = A_values.shape[0]
    assert num_prototypes <= 24, "最多支持 24 个 prototype 的可视化"

    rows, cols = 4, 6  # 固定为 4 行 6 列

    # 调整 figsize，让图更扁
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.8, rows * 0.2))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for idx in range(rows * cols):
        row = idx % rows
        col = idx // rows
        ax = axes[row][col]

        if idx < num_prototypes:
            A_val = A_values[idx].detach().cpu().numpy()
            color_img = np.ones((h, w, 3), dtype=np.float32) * A_val
            color_img_uint8 = (color_img * 255).astype(np.uint8)

            ax.imshow(color_img_uint8)
            ax.axis('off')

            # 在每个块上加编号
            ax.text(w // 2, h // 2, str(idx), color='white', fontsize=8,
                    ha='center', va='center')

            # 高亮边框
            if highlight_idx is not None and idx == highlight_idx:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
        else:
            ax.axis('off')  # 隐藏空格子

    if title:
        fig.suptitle(title, fontsize=14)

    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved A visualization to {save_path}")


class LightAEstimator(nn.Module):
    def __init__(self):
        super(LightAEstimator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),  # [B, 16, H/2, W/2]
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # [B, 32, H/4, W/4]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [B, 64, H/8, W/8]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))  # [B, 64, 1, 1]
        )

        self.fc = nn.Linear(64, 3)  # 输出 [B, 3]

    def forward(self, x):
        x = self.features(x)      # [B, 64, 1, 1]
        x = x.view(x.size(0), -1) # [B, 64]
        x = self.fc(x)            # [B, 3]
        x = torch.sigmoid(x)      # 限制到 0~1
        return x


def visualize_entropy_mapping(entropy_mlp, device='cuda', save_path='entropy_mapping.jpg'):
    """
    可视化 entropy 到 entropy_factor 的映射曲线。

    参数：
        modulator: 你的 TokenPrototypeModulator 实例
        device: 使用的设备，'cuda' 或 'cpu'
        save_path: 如果给定路径，就保存图片到该路径，否则直接显示
    """

    # 构造 0~1 的 entropy 值
    entropy_values = torch.linspace(0, 1, 100).unsqueeze(1).to(device)

    with torch.no_grad():
        # 计算对应 entropy_factor
        factors = entropy_mlp(entropy_values).squeeze().cpu().numpy()

    # 转 numpy
    entropy_values_np = entropy_values.squeeze().cpu().numpy()

    # 绘图
    plt.figure(figsize=(6, 4))
    plt.plot(entropy_values_np, factors, linewidth=2)
    plt.xlabel("Entropy", fontsize=12)
    plt.ylabel("Entropy Factor", fontsize=12)
    plt.title("Entropy → Entropy Factor Mapping", fontsize=14)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved entropy mapping curve to {save_path}")

    plt.close()

def kmeans_cluster(colors, K=4, iters=10):
    """
    纯 PyTorch 的 K-Means，用于 RGB 颜色聚类。
    colors: [N, 3] in [0,1]
    K: 聚类中心数量
    iters: K-Means 迭代次数
    返回:
        centers: [K, 3]
        labels:  [N]
    """

    N = colors.shape[0]

    # 随机初始化 K 个中心
    indices = torch.randperm(N)[:K]
    centers = colors[indices].clone()

    for _ in range(iters):
        # Step 1: 分配簇标签
        dists = torch.cdist(colors, centers)   # [N, K]
        labels = dists.argmin(dim=1)           # [N]

        # Step 2: 更新中心
        new_centers = []
        for k in range(K):
            cluster_points = colors[labels == k]
            if len(cluster_points) > 0:
                new_centers.append(cluster_points.mean(dim=0))
            else:
                # 避免空簇，随机挑一个点
                new_centers.append(colors[torch.randint(0, N, (1,))])

        centers = torch.stack(new_centers, dim=0)

    return centers, labels


def pull_towards_kmeans(colors, K=2, strength=0.4, iters=10):
    """
    颜色向 K-Means 聚类中心靠拢
    colors: [N,3]
    K: 聚类中心数
    strength: 0~1，越大越靠近聚类中心
    """
    centers, labels = kmeans_cluster(colors, K=K, iters=iters)
    cluster_centers = centers[labels]   # [N,3]

    # 按比例向聚类中心靠拢
    new_colors = colors * (1 - strength) + cluster_centers * strength
    return new_colors

class TokenPrototypeModulatorGAT(nn.Module):
    def __init__(self, token_dim=2048, num_prototypes=16):
        super().__init__()
        self.token_dim = token_dim
        self.num_prototypes = num_prototypes

        self.prototype_sets = nn.ModuleList([
            PrototypeSet(token_dim, num_prototypes)
            for _ in range(num_prototypes)
        ])

        # A_vals = get_fixed_As(num_prototypes)
        A_vals = get_random_As(num_prototypes); self.A_vals_random = A_vals
        self.A_values = nn.Parameter(A_vals)


        # visualize_As_with_number(self.A_values, save_path='init_values_vis.png', title='', h=30, w=120)

        # pdb.set_trace()

        self.gat_new = SimpleGATLayer(6, 128)
        # self.gat = SimpleGATLayer(3, 128)

        self.proj_mlp = nn.Sequential(
            nn.Linear(128, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim)
        )
        # pdb.set_trace()

    def forward(self, aggregated_tokens_list, A_pred):
        """
        A_pred: [1, 3]
        """
        num_layers = len(aggregated_tokens_list)
        P = self.num_prototypes

        dist_matrix = torch.cdist(self.A_values, self.A_values, p=2)
        adj = (dist_matrix < 0.3).float().to(A_pred.device)

        gat_input = torch.cat([self.A_values, A_pred.expand(P, -1)], dim=1)  # [P, 6]
        proto_feat, attn = self.gat_new(gat_input, adj)

        # proto_feat, attn = self.gat(self.A_values, adj)
        refined_prototypes = self.proj_mlp(proto_feat)  # [P, token_dim]


        plot_adj_matrix(adj, save_path="adj_matrix_heatmap.png")
        plot_graph_with_rgb_node_colors(adj, torch.sigmoid(self.A_values), save_path="graph1.png")        

        pdb.set_trace()
        # self.A_values1 = torch.sigmoid(self.A_values)
        # dist_matrix = torch.cdist(self.A_values1, self.A_values1, p=2)
        # adj = (dist_matrix < 0.3).float()
        # new_colors = pull_towards_kmeans(self.A_values1, K=5, strength=0.45)
        # plot_graph_with_rgb_node_colors(adj, torch.sigmoid(new_colors), save_path="graph1.png")     

        similarities = torch.exp(-10 * torch.norm(A_pred - torch.sigmoid(self.A_values), dim=1))  # [P]
        topk_vals, topk_indices = torch.topk(similarities, 4)
        weights = topk_vals / topk_vals.sum()

        modulated_tokens_list = [torch.zeros_like(tokens) for tokens in aggregated_tokens_list]
        beta_stack = torch.zeros(1, 1, 1, 3).to(A_pred.device)
        scale_stack = torch.zeros(1, 1, 1, 2).to(A_pred.device)

        for l in range(num_layers):
            layer_tokens = aggregated_tokens_list[l]
            for i, idx in enumerate(topk_indices):
                ps = self.prototype_sets[idx]
                w = weights[i].item()
                cond = refined_prototypes[idx]  # ← 引入 GAT 输出

                mod_tokens, beta, scale = ps(layer_tokens, cond=cond)

                modulated_tokens_list[l] += w * mod_tokens
                if l == 0:
                    beta_stack += w * beta
                    scale_stack += w * scale

        A_val = (weights.unsqueeze(1) * torch.sigmoid(self.A_values[topk_indices])).sum(0, keepdim=True)

        return modulated_tokens_list, A_val, beta_stack, scale_stack


class SimpleGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        """
        h: [P, D] 原始 prototype 特征
        adj: [P, P] 邻接矩阵 (0/1 或相似度)
        """
        Wh = self.fc(h)  # [P, D']
        P = h.shape[0]

        a_input = torch.cat([Wh.repeat(1, P).view(P * P, -1), Wh.repeat(P, 1)], dim=1)  # [P*P, 2D']
        e = self.leaky_relu(self.attn_fc(a_input)).view(P, P)  # [P, P]

        # Mask 非邻接项
        e = e.masked_fill(adj == 0, -9e15)
        attention = torch.softmax(e, dim=1)  # [P, P]

        h_prime = torch.matmul(attention, Wh)  # [P, D']

        return h_prime, attention


def plot_adj_matrix(adj, title="Adjacency Matrix Heatmap", save_path=None):
    """
    绘制邻接矩阵热力图

    参数：
    - adj: torch.Tensor, 形状为 [N, N] 的邻接矩阵，支持 CUDA 张量
    - title: 图像标题（可选）
    - save_path: 如果指定，将保存图片到该路径（可选）
    """
    if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"邻接矩阵应为方阵，当前 shape 为 {adj.shape}")

    adj_cpu = adj.detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    sns.heatmap(adj_cpu, cmap='viridis', square=True, cbar=True)
    plt.title(title)
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"邻接矩阵已保存到 {save_path}")
    else:
        plt.show()

def plot_graph_with_rgb_node_colors(adj, node_features, save_path=None):
    """
    根据邻接矩阵绘制图结构，每个节点以 node_features 的 RGB 映射着色。
    
    参数：
    - adj: torch.Tensor，形状为 [N, N] 的邻接矩阵（可为 CUDA 张量）
    - node_features: torch.Tensor，形状为 [N, 3]，值应为 [0, 1] 范围内，作为 RGB 颜色
    - save_path: 可选，若给定路径，则保存图片
    """
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("adj 应为方阵")
    if adj.shape[0] != node_features.shape[0]:
        raise ValueError("节点数与特征数不一致")
    if node_features.shape[1] != 3:
        raise ValueError("node_features 应为每个节点一个 RGB 值，即 [N, 3]")

    # 转为 numpy
    adj_np = adj.detach().cpu().numpy()
    node_colors = node_features.detach().cpu().numpy()  # [N, 3]，值应在 [0, 1]
    node_colors = amplify_dominant_diff_channel(node_colors, group_size=4)

    # 构图
    G = nx.Graph()
    num_nodes = adj_np.shape[0]
    G.add_nodes_from(range(num_nodes))

    # 添加边
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_np[i, j] != 0:
                G.add_edge(i, j)

    # 使用 spring layout 可视化
    pos = nx.spring_layout(G, seed=42)

    # 绘图
    plt.figure(figsize=(12, 4))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=10)

    # plt.title("Graph Visualization with RGB Node Colors")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图已保存至 {save_path}")
    else:
        plt.show()


def amplify_dominant_diff_channel(colors, group_size=4, amplify_factors=[0.94, 0.98, 1.03, 1.08], clamp=True):
    """
    在每组 prototype 中，找到颜色差异最大的通道（R/G/B），并分别放大该通道。
    
    参数:
        colors: np.ndarray of shape [N, 3]
        group_size: 每组个数
        amplify_factors: 每组内部4个颜色的放大倍数
        clamp: 是否将结果裁剪到 [0,1]
        
    返回:
        enhanced_colors: np.ndarray
    """
    N = colors.shape[0]
    enhanced = colors.copy()
    num_groups = (N + group_size - 1) // group_size

    for g in range(num_groups):
        start = g * group_size
        end = min((g + 1) * group_size, N)
        group = enhanced[start:end]

        # 找到组内差异最大的通道
        std_per_channel = group.std(axis=0)  # shape: [3]
        dominant_channel = np.argmax(std_per_channel)

        # 为组内每个颜色分别放大 dominant_channel
        for i in range(end - start):
            factor = amplify_factors[i] if i < len(amplify_factors) else 1.0  # fallback
            group[i, dominant_channel] *= factor

        # 更新回 enhanced
        enhanced[start:end] = group

    if clamp:
        enhanced = np.clip(enhanced, 0, 1)

    return enhanced