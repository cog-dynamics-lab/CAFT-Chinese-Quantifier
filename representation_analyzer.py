import torch
import torch.nn as nn
import numpy as np


class RepresentationAnalyzer:
    def __init__(self, model: nn.Module, target_layer_name: str = None):
        """
        初始化表征分析器。

        Args:
            model: 要分析的 PyTorch 模型。
            target_layer_name: 目标层的名字。如果为 None，将自动寻找模型中的最后一个 nn.Linear 层
                               （通常是分类头），并挂载 hook 到它的输入端（即倒数第二层表征）。
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.extracted_features = []
        self.hook_handle = None

        # 定位目标层
        if target_layer_name is not None:
            self.target_layer = dict(self.model.named_modules())[target_layer_name]
        else:
            # 自动定位最后一个 nn.Linear 层
            linear_layers = [(name, module) for name, module in self.model.named_modules() if
                             isinstance(module, nn.Linear)]
            if not linear_layers:
                raise ValueError("Could not find any nn.Linear layer in the model.")
            self.target_layer_name, self.target_layer = linear_layers[-1]
            print(f"[Analyzer] Auto-detected target layer for hook: '{self.target_layer_name}'")

    def _hook_fn(self, module, input, output):
        # input[0] 是该线性层的输入，即我们需要的倒数第二层表征 (Batch, Hidden_Dim)
        # 将其克隆并分离出计算图，转到 CPU 上保存以释放显存
        features = input[0].detach().cpu()
        # 处理可能存在的 3D 序列张量 (Batch, Seq_Len, Hidden_Dim) -> (Batch, Hidden_Dim)
        # 通常分类器输入已经是 pooling 后的 2D 张量。如果是 3D，我们取 CLS 位置 (dim 0) 或 mean pooling
        if len(features.shape) == 3:
            features = features.mean(dim=1)  # Mean-pooling over sequence length
        self.extracted_features.append(features)

    def register(self):
        """注册 Forward Hook"""
        self.extracted_features = []
        self.hook_handle = self.target_layer.register_forward_hook(self._hook_fn)

    def remove(self):
        """移除 Hook，防止影响正常的推理或训练速度"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def collect_representations(self, data_loader_or_tensor, device=None):
        """
        前向传播模型，收集所有数据的表征。
        """
        self.register()
        self.model.eval()
        dev = device if device is not None else self.device

        with torch.no_grad():
            if isinstance(data_loader_or_tensor, torch.Tensor):
                # 如果输入是 Tensor
                _ = self.model(data_loader_or_tensor.to(dev))
            else:
                # 如果输入是 DataLoader
                for batch in data_loader_or_tensor:
                    # 假设 batch 的第一个元素是输入 X
                    x = batch[0].to(dev) if isinstance(batch, (list, tuple)) else batch.to(dev)
                    _ = self.model(x)

        self.remove()

        # 拼接所有 Batch 的特征向量
        if not self.extracted_features:
            raise RuntimeError("No features were captured. Check if the model forward pass was executed.")
        all_features = torch.cat(self.extracted_features, dim=0)
        return all_features

    @staticmethod
    def calculate_metrics(H: torch.Tensor):
        """
        根据表征矩阵 H 计算 Effective Rank 和 Average Cosine Similarity。

        Args:
            H: 张量，形状为 (N, D)，其中 N 是样本数，D 是隐藏层维度。
        """
        N, D = H.shape
        if N < 2:
            return 1.0, 1.0  # 样本数太少无法计算相似度

        # 1. 计算各向异性度（余弦相似度均值）
        # 对每一行进行 L2 归一化
        H_norm = H / (torch.norm(H, p=2, dim=1, keepdim=True) + 1e-9)
        # 计算余弦相似度矩阵 (N, N)
        cos_sim_matrix = torch.mm(H_norm, H_norm.t())
        # 提取非对角线的上三角部分（避免自相关）
        triu_indices = torch.triu_indices(N, N, offset=1)
        avg_cos_sim = cos_sim_matrix[triu_indices[0], triu_indices[1]].mean().item()

        # 2. 计算有效秩 (Effective Rank)
        # 对 H 进行奇异值分解 (SVD)
        # 使用 torch.linalg.svd，设置 full_matrices=False
        try:
            _, S, _ = torch.linalg.svd(H, full_matrices=False)
            # 归一化奇异值，使其成为概率分布
            sum_S = torch.sum(S)
            if sum_S > 0:
                p = S / sum_S
                # 计算奇异值分布的熵
                entropy = -torch.sum(p * torch.log(p + 1e-9))
                # 有效秩为熵的指数形式
                erank = torch.exp(entropy).item()
            else:
                erank = 1.0
        except Exception as e:
            # 捕获可能的 SVD 不收敛异常
            print(f"[Warning] SVD failed to converge: {e}. Returning erank=1.0")
            erank = 1.0

        return erank, avg_cos_sim
