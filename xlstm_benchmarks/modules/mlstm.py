import torch
import torch.nn as nn
from torch import Tensor


class MLSTMCell(nn.Module):
    def __init__(self, input_size:int, head_size:int, num_heads:int):
        super(MLSTMCell, self).__init__()
        self.head_size = head_size
        hidden_size = head_size * num_heads

        self.W_i = nn.Parameter(torch.empty(input_size, hidden_size * 3))
        self.b_i = nn.Parameter(torch.empty(hidden_size * 3))

        self.W_qkv = nn.ParameterList([
            nn.Parameter(torch.empty(head_size, head_size * 3))
            for _ in range(num_heads)
        ])
        self.b_qkv = nn.Parameter(torch.empty(hidden_size * 3))

        self.init_params()

    def init_params(self):
        nn.init.xavier_normal_(self.W_i)
        for block in self.W_h:
            nn.init.xavier_normal_(block)
        nn.init.zeros_(self.b_h), nn.init.zeros_(self.b_i)

    def forward(self, x:Tensor, h:Tensor, C:Tensor, n:Tensor)->tuple[Tensor, Tensor, Tensor]:
        i_tilde, f_tilde, o_tilde = (x @ self.W_i + self.b_i).chunk(3, dim=-1)
        q, k, v = (x @ torch.block_diag(*self.W_qkv) + self.b_qkv).chunk(3, dim=-1)
        k /= torch.sqrt(self.head_size) # not directly the same as in the paper (bias not divided by sqrt(d_k))

        i, f, o = torch.exp(i_tilde), torch.exp(f_tilde), torch.sigmoid(o_tilde)

        n = f @ C + i @ k
        h = o * (C @ q) / torch.maximum(torch.abs(n.T @ q), 1)
        C = f @ C + i @ (v @ k.T)

        return h, C, n
