import torch
import torch.nn as nn
import math

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, bidirectional=False):
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.weight_ih = nn.ParameterList([
            nn.Parameter(torch.Tensor(3 * hidden_size, input_size if layer == 0 else hidden_size * num_directions))
            for layer in range(num_layers * num_directions)
        ])
        
        self.weight_hh = nn.ParameterList([
            nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
            for layer in range(num_layers * num_directions)
        ])
        
        if bias:
            self.bias_ih = nn.ParameterList([
                nn.Parameter(torch.Tensor(3 * hidden_size))
                for layer in range(num_layers * num_directions)
            ])
            
            self.bias_hh = nn.ParameterList([
                nn.Parameter(torch.Tensor(3 * hidden_size))
                for layer in range(num_layers * num_directions)
            ])
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weight_ih:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        for weight in self.weight_hh:
            nn.init.orthogonal_(weight)
        if self.bias:
            for bias in self.bias_ih:
                nn.init.zeros_(bias)
            for bias in self.bias_hh:
                nn.init.zeros_(bias)

    def forward(self, input, hx=None):
        if self.batch_first:
            input = input.transpose(0, 1)  
        
        seq_len, batch_size, _ = input.size()
        num_directions = 2 if self.bidirectional else 1

        if hx is None:
            h_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=input.device, dtype=input.dtype)
        else:
            h_0 = hx

        h_n = []

        output = []
        for t in range(seq_len):
            x = input[t]
            for layer in range(self.num_layers):
                for direction in range(num_directions):
                    idx = layer * num_directions + direction
                    h_prev = h_0[idx]

                    gates = (torch.matmul(x, self.weight_ih[idx].t()) + 
                             torch.matmul(h_prev, self.weight_hh[idx].t()) + 
                             (self.bias_ih[idx] if self.bias else 0) + 
                             (self.bias_hh[idx] if self.bias else 0))

                    r, z, n = gates.chunk(3, 1)

                    r = torch.sigmoid(r)
                    z = torch.sigmoid(z)
                    n = torch.tanh(n + r * torch.matmul(h_prev, self.weight_hh[idx][2 * self.hidden_size:].t()))

                    h = (1 - z) * n + z * h_prev

                    x = h
                    h_0[idx] = h

                    if self.bidirectional and direction == 0:
                        x = torch.cat((x, h), dim=1)  

            output.append(h)

        output = torch.stack(output, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)  

        h_n = h_0

        return output, h_n
