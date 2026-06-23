"""
Module: meta_pretrain_no_noise.py
Description: Performs unstructured (no-noise / structure-free) meta-pretraining
             ablation. The student model architecture is kept strictly identical
             to the main model, but the task generator outputs flat, unstructured
             noun tokens to isolate the effect of hierarchical syntactic priors.
"""

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


# =============================================================================
# 1. NEURAL ARCHITECTURE: IDENTICAL STUDENT TRANSFORMER
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
    Compact student Transformer architecture. Keeps complete structural alignment
    with the main pipeline (including normalization steps) for fair ablation comparison.
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
# 2. ABLATION TASK GENERATOR (STRUCTURE-FREE/FLAT PATTERNS)
# =============================================================================

class SyntheticPCFGGenerator:
    """
    Generates flat, unstructured token tasks. Bypasses hierarchical syntactic layouts
    to provide the baseline control group for ablation studies.
    """

    def __init__(self, vocab_size: int = 25000):
        self.vocab_size = vocab_size
        self.word_pool = list(range(10, 15000))

    def generate_task(self, k_shot: int = 15) -> tuple:
        # Avoid seed collision across epochs while maintaining local random variations
        random.seed(random.randint(0, 100000))
        nouns = random.sample(self.word_pool, 3000)

        noun_to_label = {}
        for label_idx in range(6):
            for n in nouns[label_idx * 500: (label_idx + 1) * 500]:
                noun_to_label[n] = label_idx

        def make_sentence():
            head_noun = random.choice(nouns)
            label = noun_to_label[head_noun]
            # Structure-free ablation: contains only the target noun token
            seq = [head_noun]
            return seq, label

        def build_dataset(size):
            X, y = [], []
            for _ in range(size):
                seq, label = make_sentence()
                X.append(seq)
                y.append(label)
            return X, y

        X_supp, y_supp = build_dataset(6 * k_shot)
        return X_supp, y_supp


def pad_batch(X: list) -> torch.Tensor:
    """Zero-pads sequence inputs to generate a uniform tensor block."""
    max_len = max((len(seq) for seq in X), default=1)
    padded = [seq + [0] * (max_len - len(seq)) for seq in X]
    return torch.tensor(padded, dtype=torch.long)


# =============================================================================
# 3. META-PRETRAINING CONTROL EXECUTION
# =============================================================================

def reptile_meta_pretrain():
    """
    Performs Reptile meta-learning using structure-free inputs.
    Outputs weights to 'meta_no_noise.pth' for ablation benchmarking.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initiating No-Noise Ablation Meta-Pretraining on Device: {device}")

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

        # Adaptation training step on the local task
        inner_optimizer = optim.Adam(model.parameters(), lr=0.01)
        for _ in range(10):
            inner_optimizer.zero_grad()
            preds = model(X_supp_t)
            loss = criterion(preds, y_supp_t)
            loss.backward()
            inner_optimizer.step()

        # Update meta-weights using first-order Reptile gradients
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.grad = outer_state[name].to(device) - param.data
                param.data = outer_state[name].to(device)

        outer_optimizer.step()

        if epoch % 200 == 0:
            print(f"Meta-Epoch {epoch:4d} | Inner Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "meta_no_noise.pth")
    print("[SUCCESS] No-noise meta-pretraining complete. Weights saved to 'meta_no_noise.pth'")


if __name__ == "__main__":
    reptile_meta_pretrain()
