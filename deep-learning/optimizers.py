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

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, amsgrad=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None 
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups():
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")
                
                amsgrad = group['amsgrad']
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
        
        return loss 