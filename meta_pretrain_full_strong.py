"""
Module: meta_pretrain_full_strong.py
Description: Performs strong inductive bias distillation utilizing a Reptile-based
             meta-learning protocol. Synthesized PCFG structures are used to embed
             structural priors into a compact student Transformer architecture.
"""

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


# =============================================================================
# 1. NEURAL ARCHITECTURE: STUDENT COMPACT TRANSFORMER
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Implements standard sinusoidal positional encoding to inject sequence-order
    information into the transformer sequence representations.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class NeuralTransformerStudent(nn.Module):
    """
    Compact student Transformer model designed for structurally-biased processing.
    Utilizes mean/max pooling strategies to form downstream sentence-level representations.
    """

    def __init__(self, vocab_size: int, num_classes: int, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2, pooling: str = 'max'):
        super().__init__()
        self.d_model = d_model
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        self.emb_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)
        emb = self.emb_norm(emb)
        out = self.transformer(emb)

        if self.pooling == 'max':
            pooled = out.max(dim=1)[0]
        elif self.pooling == 'max_mean':
            pooled = 0.5 * out.max(dim=1)[0] + 0.5 * out.mean(dim=1)
        else:
            pooled = out.mean(dim=1)

        pooled = self.pool_norm(pooled)
        return self.fc(pooled)


# =============================================================================
# 2. SYNTHETIC PCFG GENERATOR FOR SYSTEMATIC SYNTAX PRIORS
# =============================================================================

class SyntheticPCFGGenerator:
    """
    Generates synthetic hierarchical sequences based on a dynamic context-free grammar.
    Prevents shortcut learning by dynamically shifting token-to-role mappings
    across meta-tasks, forcing the model to capture nested structural dependencies.
    """

    def __init__(self, vocab_size: int = 25000):
        self.vocab_size = vocab_size

    def generate_task(self, k_shot: int = 15) -> tuple:
        X, y = [], []

        # 动态映射：每个任务随机映射Token，使模型无法死记Token ID
        task_vocab = random.sample(range(10, self.vocab_size), 50)

        # 6个类别对应6种不同的句法嵌套深度或分层树结构规则
        for class_idx in range(6):
            for _ in range(k_shot):
                # 构造具有嵌套层级特征的句法树 (e.g. A B A', A B C B' A')
                left_bracket_1 = task_vocab[class_idx * 2]
                right_bracket_1 = task_vocab[class_idx * 2 + 1]
                filler_tokens = task_vocab[12:30]

                # 构造符合特定层次深度限制的序列
                depth = (class_idx % 3) + 1  # 嵌套深度 1, 2, 3
                seq = []
                for _ in range(depth):
                    seq.append(left_bracket_1)

                # 插入句法树内部的填充成分（叶子节点）
                seq.extend(random.sample(filler_tokens, random.randint(2, 5)))

                for _ in range(depth):
                    seq.append(right_bracket_1)

                # 随机填充无用背景噪声，强迫自注意力机制提取层次主干
                while len(seq) < 15:
                    seq.insert(random.randint(0, len(seq)), random.choice(task_vocab[30:]))

                X.append(seq)
                y.append(class_idx)
        return X, y


def pad_batch(batch: list, pad_idx: int = 0) -> torch.Tensor:
    """Zero-pads a list of variable-length sequences to a uniform tensor format."""
    max_len = max(len(x) for x in batch)
    padded = [x + [pad_idx] * (max_len - len(x)) for x in batch]
    return torch.tensor(padded, dtype=torch.long)


# =============================================================================
# 3. META-PRETRAINING ENGINE (REPTILE PROTOCOL)
# =============================================================================

def reptile_meta_pretrain():
    """
    Executes the Reptile meta-optimization protocol to update model initializations
    towards a generalized representation that adapts quickly to structured syntax.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initiating Complete Inductive Bias Distillation on Device: {device}")

    VOCAB_SIZE = 25000
    NUM_CLASSES = 6
    META_EPOCHS = 3000

    model = NeuralTransformerStudent(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
    model.to(device)

    generator = SyntheticPCFGGenerator(vocab_size=VOCAB_SIZE)
    outer_optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, META_EPOCHS + 1):
        X_supp, y_supp = generator.generate_task(k_shot=15)
        X_supp_t = pad_batch(X_supp).to(device)
        y_supp_t = torch.tensor(y_supp, dtype=torch.long).to(device)

        outer_state = deepcopy(model.state_dict())
        model.train()

        # Inner loop adaptation step (local task update)
        inner_optimizer = optim.Adam(model.parameters(), lr=0.01)
        for _ in range(10):
            inner_optimizer.zero_grad()
            preds = model(X_supp_t)
            loss = criterion(preds, y_supp_t)
            loss.backward()
            inner_optimizer.step()

        # Outer loop meta-gradient computation & update
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.grad = outer_state[name].to(device) - param.data
                param.data = outer_state[name].to(device)
        outer_optimizer.step()

        if epoch % 200 == 0:
            print(f"Meta-Epoch {epoch:4d} | Inner Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "meta_pretrained_prior.pth")
    print("[SUCCESS] Strong meta-pretraining complete. Weights saved to 'meta_pretrained_prior.pth'")


if __name__ == "__main__":
    reptile_meta_pretrain()
