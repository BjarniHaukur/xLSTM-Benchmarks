import torch
import torch.nn as nn
from torch import Tensor as T


class LSTMCell(nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        super(LSTMCell, self).__init__()

        self.W_i = nn.Parameter(torch.empty(input_size, hidden_size * 4)) # we can perform all the operations on the input in one go
        self.b_i = nn.Parameter(torch.empty(hidden_size * 4))
        self.W_h = nn.Parameter(torch.empty(hidden_size, hidden_size * 4)) # same for the hidden state
        self.b_h = nn.Parameter(torch.empty(hidden_size * 4))

        self.init_params()
    
    def init_params(self):
        nn.init.xavier_normal_(self.W_i), nn.init.zeros_(self.b_i)
        nn.init.xavier_normal_(self.W_h), nn.init.zeros_(self.b_h)

    def forward(self, x:T, h:T, c:T)->tuple[T, T]:
        i_tilde, f_tilde, z_tilde, o_tilde = (x @ self.W_i + self.b_i + h @ self.W_h + self.b_h).chunk(4, dim=-1)
        # input gate, forget gate, input node / input modulation, output gate
        i, f, z, o = torch.sigmoid(i_tilde), torch.sigmoid(f_tilde), torch.tanh(z_tilde), torch.sigmoid(o_tilde)
        # forget old information (forget_gate * cell_state), add new information (input_gate * input_modulation)
        c = f * c + i * z 
        # update hidden state
        h = o * torch.tanh(c)
        return h, c
        
class LSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, n_layers:int=1):
        super(LSTM, self).__init__()
        self.input_size, self.hidden_size, self.n_layers = input_size, hidden_size, n_layers

        self.lstm_cells = nn.ModuleList([LSTMCell(input_size if i==0 else hidden_size, hidden_size) for i in range(n_layers)])

    def forward(self, x:T, h_0:T=None, c_0:T=None)->tuple[T, tuple[T]]:
        B, L, _ = x.shape

        if h_0 is None: h_0 = torch.zeros(self.n_layers, B, self.hidden_size, device=x.device)
        if c_0 is None: c_0 = torch.zeros(self.n_layers, B, self.hidden_size, device=x.device)

        h_t = [h for h in h_0] # avoid in-place operations
        c_t = [c for c in c_0]

        output = []
        for t in range(L):
            for i, cell in enumerate(self.lstm_cells):
                h_t[i], c_t[i] = cell(x[:, t] if i==0 else h_t[i - 1], h_t[i], c_t[i]) # stacked layers receive the output of the previous layer
                
            output.append(h_t[-1])

        return torch.stack(output, dim=1), (torch.stack(h_t, dim=0), torch.stack(c_t, dim=0))

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
    parser.add_argument("--backward_tolerance", type=float, default=2e-4)
    args = parser.parse_args()

    lstm = LSTM(args.input_size, args.hidden_size, args.n_layers)
    lstm_torch = nn.LSTM(args.input_size, args.hidden_size, args.n_layers, batch_first=True)

    for i, cell in enumerate(lstm.lstm_cells):
        getattr(lstm_torch, f"weight_ih_l{i}").data.copy_(cell.W_i.T)
        getattr(lstm_torch, f"weight_hh_l{i}").data.copy_(cell.W_h.T)
        getattr(lstm_torch, f"bias_ih_l{i}").data.copy_(cell.b_i)
        getattr(lstm_torch, f"bias_hh_l{i}").data.copy_(cell.b_h)

    x = torch.randn(args.batch_size, args.seq_len, args.input_size)
    out, (h, c) = lstm(x)
    out_torch, (h_torch, c_torch) = lstm_torch(x)

    np.testing.assert_allclose(out.detach().numpy(), out_torch.detach().numpy(), atol=args.forward_tolerance)
    np.testing.assert_allclose(h.detach().numpy(), h_torch.detach().numpy(), atol=args.forward_tolerance)
    np.testing.assert_allclose(c.detach().numpy(), c_torch.detach().numpy(), atol=args.forward_tolerance)

    print(f"Forwards pass match with {args.forward_tolerance=}")

    out.sum().backward()
    out_torch.sum().backward()

    for i, cell in enumerate(lstm.lstm_cells):
        np.testing.assert_allclose(cell.W_i.grad.detach().numpy().T, getattr(lstm_torch, f"weight_ih_l{i}").grad.detach().numpy(), atol=args.backward_tolerance)
        np.testing.assert_allclose(cell.W_h.grad.detach().numpy().T, getattr(lstm_torch, f"weight_hh_l{i}").grad.detach().numpy(), atol=args.backward_tolerance)
        np.testing.assert_allclose(cell.b_i.grad.detach().numpy().T, getattr(lstm_torch, f"bias_ih_l{i}").grad.detach().numpy(), atol=args.backward_tolerance)
        np.testing.assert_allclose(cell.b_h.grad.detach().numpy().T, getattr(lstm_torch, f"bias_hh_l{i}").grad.detach().numpy(), atol=args.backward_tolerance)
    
    print(f"Backwards pass match with {args.backward_tolerance=}")
        
