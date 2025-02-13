import torch
import torch.nn as nn
from torch import Tensor as T

class RNNCell(nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        super(RNNCell, self).__init__()

        self.W_i = nn.Parameter(torch.empty((input_size, hidden_size)))
        self.b_i = nn.Parameter(torch.empty(hidden_size))
        self.W_h = nn.Parameter(torch.empty((hidden_size, hidden_size)))
        self.b_h = nn.Parameter(torch.empty(hidden_size))

        self.init_params()

    def init_params(self):
        nn.init.xavier_normal_(self.W_i), nn.init.zeros_(self.b_i)
        nn.init.xavier_normal_(self.W_h), nn.init.zeros_(self.b_h)

    def forward(self, x:T, h:T)->T:
        return torch.tanh(x @ self.W_i + self.b_i + h @ self.W_h + self.b_h)


class RNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, n_layers:int=1):
        super(RNN, self).__init__()
        self.input_size, self.hidden_size, self.n_layers = input_size, hidden_size, n_layers

        self.rnn_cells = nn.ModuleList([RNNCell(input_size if i==0 else hidden_size, hidden_size) for i in range(n_layers)])

    def forward(self, x:T, h_0:T=None)->tuple[T, T]:
        B, L, _ = x.shape # batch, length, dimensionality

        if h_0 is None: h_0 = torch.zeros(self.n_layers, B, self.hidden_size, device=x.device)

        h_t = [h for h in h_0] # avoid in-place operations

        output = [] 
        for t in range(L):
            for i, cell in enumerate(self.rnn_cells):
                h_t[i] = cell(x[:, t] if i==0 else h_t[i - 1], h_t[i]) # stacked layers receive the output of the previous layer
            
            output.append(h_t[-1])

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
    parser.add_argument("--forward_tolerance", type=float, default=1e-16) # perfect match
    parser.add_argument("--backward_tolerance", type=float, default=1e-4) # lossy match
    args = parser.parse_args()
    
    rnn = RNN(args.input_size, args.hidden_size, args.n_layers)
    rnn_torch = nn.RNN(args.input_size,  args.hidden_size, args.n_layers, nonlinearity="tanh", batch_first=True)

    for i, cell in enumerate(rnn.rnn_cells):
        getattr(rnn_torch, "weight_ih_l" + str(i)).data.copy_(cell.W_i.data.T)
        getattr(rnn_torch, "weight_hh_l" + str(i)).data.copy_(cell.W_h.data.T)
        getattr(rnn_torch, "bias_ih_l" + str(i)).data.copy_(cell.b_i.data)
        getattr(rnn_torch, "bias_hh_l" + str(i)).data.copy_(cell.b_h.data)

    x = torch.randn(args.batch_size, args.seq_len, args.input_size)  # Batch size of 5, sequence length of 3, feature size of 10
    out, h = rnn(x)
    out_torch, h_torch = rnn_torch(x)

    np.testing.assert_allclose(out.detach().numpy(), out_torch.detach().numpy(), atol=args.forward_tolerance)
    np.testing.assert_allclose(h.detach().numpy(), h_torch.detach().numpy(), atol=args.forward_tolerance)

    print(f"Forwards pass match with {args.forward_tolerance=}")

    out.sum().backward()
    out_torch.sum().backward()

    for i, cell in enumerate(rnn.rnn_cells):
        np.testing.assert_allclose(cell.W_i.grad.detach().numpy().T, getattr(rnn_torch, "weight_ih_l" + str(i)).grad.detach().numpy(), atol=args.backward_tolerance)
        np.testing.assert_allclose(cell.W_h.grad.detach().numpy().T, getattr(rnn_torch, "weight_hh_l" + str(i)).grad.detach().numpy(), atol=args.backward_tolerance)
        np.testing.assert_allclose(cell.b_i.grad.detach().numpy().T, getattr(rnn_torch, "bias_ih_l" + str(i)).grad.detach().numpy(), atol=args.backward_tolerance)
        np.testing.assert_allclose(cell.b_h.grad.detach().numpy().T, getattr(rnn_torch, "bias_hh_l" + str(i)).grad.detach().numpy(), atol=args.backward_tolerance)

    print(f"Backwards pass match with {args.backward_tolerance=}")
    
    
