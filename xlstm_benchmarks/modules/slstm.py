import torch
import torch.nn as nn
from torch import Tensor as T


class SLSTMCell(nn.Module):
    def __init__(self, input_size:int, head_size:int, num_heads:int):
        super(SLSTMCell, self).__init__()

        self.W_i = nn.Parameter(torch.empty(input_size, head_size * num_heads * 4)) # we can perform all the operations on the input in one go
        self.b_i = nn.Parameter(torch.empty(head_size * num_heads * 4))

        # later we create a block diagonal matrix of these parameters
        # * 4 to perform all the operations in one go
        self.W_h = nn.ParameterList([
            nn.Parameter(torch.empty(head_size, head_size * 4))
            for _ in range(num_heads)
        ])
        self.b_h = nn.Parameter(torch.empty(head_size * num_heads * 4))

        self.init_params()

    def init_params(self):
        nn.init.xavier_normal_(self.W_i)
        for block in self.W_h:
            nn.init.xavier_normal_(block)
        nn.init.zeros_(self.b_h), nn.init.zeros_(self.b_i)

    def forward(self, x:T, h:T, c:T, m:T, n:T)->tuple[T, T, T, T]:
        # input gate, forget gate, input node / input modulation, output gate
        i_tilde, f_tilde, z_tilde, o_tilde = (x @ self.W_i + self.b_i + h @ torch.block_diag(*self.W_h) + self.b_h).chunk(4, dim=-1)

        # stabilizer gate, when activations are exp, the log cancels out
        m_next = torch.maximum(f_tilde + m, i_tilde)

        i = torch.exp(i_tilde - m)
        f = torch.exp(f_tilde + m_next - m)
        z = torch.tanh(z_tilde)
        o = torch.sigmoid(o_tilde)

        # cell state, forget old information (forget_gate * cell_state), add new information (input_gate * input_modulation)
        c_next = f * c + i * z
        # normalizer state, 
        n_next = f * n + i
        # hidden state, select from the cell state
        h_next = o * (c_next / n_next)
    
        return h_next, c_next, m_next, n_next






