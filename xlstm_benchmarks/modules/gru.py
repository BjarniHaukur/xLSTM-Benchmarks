import torch
import torch.nn as nn
from torch import Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        super(GRUCell, self).__init__()

        self.W_i = nn.Parameter(torch.empty(input_size, hidden_size * 3)) # we can perform all the operations on the input in one go
        self.b_i = nn.Parameter(torch.empty(hidden_size * 3))
        self.W_h = nn.Parameter(torch.empty(hidden_size, hidden_size * 3)) # same for the hidden state
        self.b_h = nn.Parameter(torch.empty(hidden_size * 3))

        self.init_params()

    def init_params(self):
        nn.init.xavier_normal_(self.W_i), nn.init.zeros_(self.b_i)
        nn.init.xavier_normal_(self.W_h), nn.init.zeros_(self.b_h)

    def forward(self, x:Tensor, h:Tensor)->Tensor:
        i_r, i_z, i_n = torch.chunk(x @ self.W_i + self.b_i, 3, dim=-1)
        h_r, h_z, h_n = torch.chunk(h @ self.W_h + self.b_h, 3, dim=-1)
        # reset gate, update gate
        r, z = torch.sigmoid(i_r + h_r), torch.sigmoid(i_z + h_z)
        # candidate hidden state
        h_tilde = torch.tanh(i_n + r * h_n)
        # update hidden state
        return z * h + (1 - z) * h_tilde
    
class GRU(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, n_layers:int=1):
        super(GRU, self).__init__()
        self.input_size, self.hidden_size, self.n_layers = input_size, hidden_size, n_layers

        self.gru_cells = nn.ModuleList([GRUCell(input_size if i==0 else hidden_size, hidden_size) for i in range(n_layers)])

    def forward(self, x:Tensor, h_0:Tensor=None)->tuple[Tensor, Tensor]:
        B, L, _ = x.shape

        if h_0 is None: h_0 = torch.zeros(self.n_layers, B, self.hidden_size, device=x.device)

        h_t = [h.clone() for h in h_0]

        output = []
        for t in range(L):
            for i, cell in enumerate(self.gru_cells):
                h_t[i] = cell(x[:, t] if i==0 else h_t[i - 1], h_t[i]) # stacked layers receive the output of the previous layer
                
            output.append(h_t[-1].clone())

        return torch.stack(output, dim=1), torch.stack(h_t, dim=0)

if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--forward_tolerance", type=float, default=1e-6)
    parser.add_argument("--backward_tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    gru = GRU(args.input_size, args.hidden_size, args.n_layers)
    gru_torch = nn.GRU(args.input_size, args.hidden_size, args.n_layers, batch_first=True)

    for i, cell in enumerate(gru.gru_cells):
        getattr(gru_torch, 'weight_ih_l' + str(i)).data.copy_(cell.W_i.T)
        getattr(gru_torch, 'weight_hh_l' + str(i)).data.copy_(cell.W_h.T)
        getattr(gru_torch, 'bias_ih_l' + str(i)).data.copy_(cell.b_i)
        getattr(gru_torch, 'bias_hh_l' + str(i)).data.copy_(cell.b_h)

    x = torch.randn(args.batch_size, args.seq_len, args.input_size)
    out, h = gru(x)
    out_torch, h_torch = gru_torch(x)
   
    np.testing.assert_allclose(out.detach().numpy(), out_torch.detach().numpy(), atol=args.forward_tolerance)
    np.testing.assert_allclose(h.detach().numpy(), h_torch.detach().numpy(), atol=args.forward_tolerance)

    print(f"Forwards pass match with {args.forward_tolerance=}")

    out.sum().backward()
    out_torch.sum().backward()

    for i, cell in enumerate(gru.gru_cells):
        np.testing.assert_allclose(cell.W_i.grad.detach().numpy().T, getattr(gru_torch, "weight_ih_l" + str(i)).grad.detach().numpy(), atol=args.backward_tolerance)
        np.testing.assert_allclose(cell.W_h.grad.detach().numpy().T, getattr(gru_torch, "weight_hh_l" + str(i)).grad.detach().numpy(), atol=args.backward_tolerance)
        np.testing.assert_allclose(cell.b_i.grad.detach().numpy().T, getattr(gru_torch, "bias_ih_l" + str(i)).grad.detach().numpy(), atol=args.backward_tolerance)
        np.testing.assert_allclose(cell.b_h.grad.detach().numpy().T, getattr(gru_torch, "bias_hh_l" + str(i)).grad.detach().numpy(), atol=args.backward_tolerance)

    print(f"Backwards pass match with {args.backward_tolerance=}")

    
        
        


        
