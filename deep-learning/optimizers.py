import torch
from torch.optim import Optimizer
import math 


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, 
                 weight_decay=0, nesterov=False):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, 
                        weight_decay=weight_decay, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-lr, d_p)
        
        return loss 
    
class Adagrad(Optimizer):
    def __init__(self, params, lr=0.01, lr_decay=0, weight_decay=0, eps=1e-10):
        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, eps=eps)
        super(Adagrad, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None 
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            lr_decay = group['lr_decay']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['sum'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                state['sum'].addcmul_(grad, grad, value=1)

                if weight_decay != 0:
                    grad = grad.add(weight_decay, p.data)
                
                clr = lr / (1 + (state['step'] - 1) * lr_decay)
                std = state['sum'].sqrt().add_(eps)
                p.data.addcdiv_(-clr, grad, std)

        return loss 
    
class RMSProp(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, 
                 momentum=0, centered=False):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                        momentum=momentum, centered=centered)
        super(RMSProp, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None 
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            centered = group['centered']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data 
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if momentum > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if centered:
                        state['grad_avg'] = torch.zeros_like(p.data)
            
                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                if centered:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul_(-1, grad_avg, grad_avg).sqrt().add_(eps)
                else:
                    avg = square_avg.sqrt().add_(eps)
                
                if weight_decay != 0:
                    grad = grad.add(weight_decay, p.data)
                
                if momentum > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).addcdiv_(grad, avg)
                    p.data.add_(-lr, buf)
                else:
                    p.data.addcdiv_(-lr, grad, avg)
        
        return loss 