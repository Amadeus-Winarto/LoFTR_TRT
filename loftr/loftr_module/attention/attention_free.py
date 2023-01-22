"""
Attention Free Transformer proposed in: https://arxiv.org/pdf/2105.14103.pdf
Modified from: https://github.com/rish-16/aft-pytorch
"""

import torch
import torch.nn as nn


class AFTSimple(nn.Module):
    def __init__(self, nheads, dim, hidden_dim=64):
        super().__init__()
        """Mechanism proposed in "Attention Free Transformer"
        Args: 
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
        Returns: 
            queried_values: (N, L, H, D) 
        """
        self.H = nheads
        self.D = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)

    def forward(self, queries, keys, values):
        B = queries.size(0)

        Q = self.to_q(queries).view(B, -1, self.hidden_dim)
        K = self.to_k(keys).view(B, -1, self.hidden_dim)
        V = self.to_v(values).view(B, -1, self.hidden_dim)

        # """
        # From the paper
        # """
        weights = torch.mul(torch.softmax(K, 1), V).sum(dim=1, keepdim=True)
        Q_sig = torch.sigmoid(Q)
        Yt = torch.mul(Q_sig, weights)

        Yt = Yt.view(B, -1, self.hidden_dim)
        Yt = self.project(Yt)

        return Yt


if __name__ == "__main__":
    N = 1
    L = 256
    H = 128
    D = 64

    S = 256

    aft = AFTSimple(dim=D)  # DIM = d_model //n_head, n_head = 8
    queries = torch.randn(N, L, H, D)
    keys = torch.randn(N, S, H, D)
    values = torch.randn(N, S, H, D)

    aft(queries, keys, values)
