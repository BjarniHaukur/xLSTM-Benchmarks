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
        # input gate, forget gate, input node / input modulation, output gate - (pre-activations)
        i_tilde, f_tilde, z_tilde, o_tilde = torch.chunk(
            x @ self.W_i + self.b_i + h @ torch.block_diag(*self.W_h) + self.b_h, 
            chunks=4, dim=-1
        )

        # stabilizer gate, when activations are exp, the log cancels out
        m_next = torch.maximum(f_tilde + m, i_tilde)

        i = torch.exp(i_tilde - m_next)
        f = torch.exp(f_tilde - m_next + m)
        z = torch.tanh(z_tilde)
        o = torch.sigmoid(o_tilde)

        # cell state, forget old information (forget_gate * cell_state), add new information (input_gate * input_modulation)
        c_next = f * c + i * z
        # normalizer state, 
        n_next = f * n + i
        # hidden state, select from the cell state
        h_next = o * (c_next / n_next)
    
        return h_next, c_next, m_next, n_next


class SLSTM(nn.Module):
    def __init__(self, input_size, head_size, num_heads, n_layers):
        super(SLSTM, self).__init__()
        self.input_size, self.head_size, self.num_heads, self.n_layers = input_size, head_size, num_heads, n_layers

        self.slstm_cells = nn.ModuleList([SLSTMCell(input_size if i==0 else head_size*num_heads, head_size, num_heads) for i in range(n_layers)])

    def forward(self, x:T, h_0:T=None, c_0:T=None, m_0:T=None, n_0:T=None)->tuple[T, tuple[T]]:
        B, L, _ = x.shape

        if h_0 is None: h_0 = torch.zeros(self.n_layers, B, self.head_size * self.num_heads, device=x.device)
        if c_0 is None: c_0 = torch.ones(self.n_layers, B, self.head_size * self.num_heads, device=x.device)
        if m_0 is None: m_0 = torch.ones(self.n_layers, B, self.head_size * self.num_heads, device=x.device)
        if n_0 is None: n_0 = torch.ones(self.n_layers, B, self.head_size * self.num_heads, device=x.device)

        h_t = [h for h in h_0] # avoid in-place operations
        c_t = [c for c in c_0]
        m_t = [m for m in m_0]
        n_t = [n for n in n_0]

        output = []
        for t in range(L):
            for i, cell in enumerate(self.slstm_cells):
                h_t[i], c_t[i], m_t[i], n_t[i] = cell(x[:,t] if i==0 else h_t[i-1], h_t[i], c_t[i], m_t[i], n_t[i])

            output.append(h_t[-1])

        return torch.stack(output, dim=1), (torch.stack(h_t, dim=0), torch.stack(c_t, dim=0), torch.stack(m_t, dim=0), torch.stack(n_t, dim=0))
                                            





