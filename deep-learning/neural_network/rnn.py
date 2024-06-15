import torch
import torch.nn as nn
import torch.nn.functional as F 
import math


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, 
                 num_layers=1, bias=True, batch_first=False):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias 
        self.batch_first = batch_first

        self.weight_ih = nn.ParameterList([
            nn.Parameter(torch.Tensor(hidden_size, input_size if layer == 0 else hidden_size)) 
            for layer in range(num_layers)
        ])

        self.weight_hh = nn.ParameterList([
            nn.Parameter(torch.Tensor(hidden_size, hidden_size)) 
            for layer in range(num_layers)
        ])

        if bias:
            self.bias_ih = nn.ParameterList([
                nn.Parameter(torch.Tensor(hidden_size)) 
                for layer in range(num_layers)
            ])

            self.bias_hh = nn.ParameterList([
                nn.Parameter(torch.Tensor(hidden_size)) 
                for layer in range(num_layers)
            ])
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        
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
    
    def forward(self, input: torch.Tensor, hx=None):
        if self.batch_first:
            input = input.transpose(0, 1)
        
        seq_len, batch_size, _ = input.size()

        if hx is None:
            hx = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                             device=input.device, dtype=input.dtype)

        output = []
        for t in range(seq_len):
            x = input[t]
            for layer in range(self.num_layers):
                h_prev = hx[layer]
                h = torch.matmul(x, self.weight_ih[layer].t()) + (self.bias_ih[layer]) if self.bias else 0
                h += torch.matmul(h_prev, self.weight_hh[layer].t()) + (self.bias_hh[layer]) if self.bias else 0
                h = torch.tanh(h)
                x = h 
                hx[layer] = h 
            output.append(h)
        
        output = torch.stack(output, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return output, hx 
    