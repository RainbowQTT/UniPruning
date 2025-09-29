import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional
import numpy as np 
import os

import atexit
import signal

FIRST_LAYER_PRINTED = False
EVENT_DIR = None

class AdamWSPP(Optimizer):
    def __init__(self, params, lr=1e-3, kappa=1, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-1, amsgrad=False, *, maximize: bool = False,
                 foreach: Optional[bool] = None,
                 capturable: bool = False,
                 namelist=None, event_dir: str = None, lambda_param: float = 0.01,lambda2_param: float = 0.01,
                    use_wanda: bool = True,           
                 wanda_normalize: bool = True,
                 sapmle_num: int = 128):
   
        global EVENT_DIR, EVENT_LOGGER
       
        EVENT_DIR = event_dir
        os.makedirs(EVENT_DIR, exist_ok=True)
    
        self.lambda_param = lambda_param
        self.lambda2_param = lambda2_param
        self._param_idx_counter = 0
        self.use_wanda = use_wanda
        self.wanda_normalize = wanda_normalize
        self.sapmle_num = sapmle_num

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, kappa=kappa, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,sp_role=None,
                        foreach=foreach, maximize=maximize, capturable=capturable,
                        lambda_param=lambda_param)

        super(AdamWSPP, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
      
        for s in state_values:
            if isinstance(s, dict):
                if 'step' not in s:
                    s['step'] = torch.tensor(0.)
                elif not torch.is_tensor(s['step']):
                    s['step'] = torch.tensor(float(s['step']))
      
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        sparsity_attn=0
        sparsity_mlp=0

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            gamma_buffers = []
            z_buffers = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            kappa= group['kappa']
            role = group['sp_role']
            role_is_attn = (role == "attn")
            role_is_mlp = (role == "mlp")
            names_list = group.get('names', None)

            for idx_in_group, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]
                            
                # State initialization
                if "step" not in state:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state["is_attn"]  = role_is_attn
                    state["is_mlp"]   = role_is_mlp
                    state["is_layer"] = role_is_attn or role_is_mlp

                    state["param_idx"] = self._param_idx_counter
                    self._param_idx_counter += 1
        
                    if names_list is not None and idx_in_group < len(names_list):
                        state["name"] = names_list[idx_in_group]
                    else:
                        print("ERROR: names_list is None")

                    state["shape"] = tuple(p.shape)
                    state["role"] = "attn" if role_is_attn else ("mlp" if role_is_mlp else "other")

                    # record gamma_buffer
                    safe_name = state["name"]
                    layer_dir = os.path.join(EVENT_DIR, safe_name)
                    os.makedirs(layer_dir, exist_ok=True)
                
                    # intial -1
                    state["first_step_mat"] = torch.full(state["shape"], -1, dtype=torch.int32, device=p.device)
            
                    # writing to param_map.csv
                    map_path = os.path.join(EVENT_DIR, "param_map.csv")
                    with open(map_path, "a") as mf:
                        mf.write(f"{state['param_idx']},{state['name']},{state['shape']},{state['role']}\n")

                    if state["is_layer"]:
                        state["gamma_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["z_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["step_record"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    else:
                        state["gamma_buffer"] = torch.zeros([1])
                        state["z_buffer"] = torch.zeros([1])
                        state["step_record"] = torch.zeros([1])

                if amsgrad and "max_exp_avg_sq" not in state:
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                gamma_buffers.append(state['gamma_buffer'])
                z_buffers.append(state['z_buffer'])
                
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                state_steps.append(state['step'])

            states = [self.state[p] for p in params_with_grad]

            adamwprox(params_with_grad,
                  grads,
                  exp_avgs,
                  exp_avg_sqs,
                  max_exp_avg_sqs,
                  gamma_buffers,
                  z_buffers,
                  sparsity_attn,
                  sparsity_mlp,
                  state_steps,
                  amsgrad=amsgrad,
                  kappa=kappa,
                  beta1=beta1,
                  beta2=beta2,
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  eps=group['eps'],
                  maximize=group['maximize'],
                  foreach=group['foreach'],
                  capturable=group['capturable'],
                  states=states,
                  lambda_param=self.lambda_param,
                  lambda2_param=self.lambda2_param,
                  sapmle_num=self.sapmle_num
                  )

        return loss

    def _save_first_activation(self, state):

        layer_name = state.get("name", "unknown_layer")
        layer_dir = os.path.join(EVENT_DIR, layer_name)
        os.makedirs(layer_dir, exist_ok=True)
        
        first_step_file = os.path.join(layer_dir, "first_step.pt")
        
        torch.save(state["first_step_mat"], first_step_file)

    def _save_gamma_buffer(self, state):

        layer_name = state.get("name", "unknown_layer")
        layer_dir = os.path.join(EVENT_DIR, layer_name)
        os.makedirs(layer_dir, exist_ok=True)
        
        gamma_file = os.path.join(layer_dir, "gamma_buffer.pt")
        torch.save(state["gamma_buffer"], gamma_file)

    def _signal_handler(self, signum, frame):
        self._save_all_on_exit()
        exit(0)
    
    def _save_all_on_exit(self):
        for param in self.param_groups[0]['params']:
            if param in self.state:
                state = self.state[param]
                if "first_step_mat" in state:
                    self._save_first_activation(state)
                if "gamma_buffer" in state and state.get("is_layer", False):
                    self._save_gamma_buffer(state)

def adamwprox(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          max_exp_avg_sqs: List[Tensor],
          gamma_buffers: List[Tensor],
          z_buffers: List[Tensor],
          sparsity_attn: float,
          sparsity_mlp: float,
          state_steps: List[Tensor],
          # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
          # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
          foreach: bool = None,
          capturable: bool = False,
          *,
          amsgrad: bool,
          kappa: float,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float,
          maximize: bool,
          states:List[dict],
          lambda_param: float,
          lambda2_param: float,
          sapmle_num: int
          ):

    if not all([isinstance(t, torch.Tensor) for t in state_steps]):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamwprox
    else:
        func = _single_tensor_adamwprox

    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         gamma_buffers,
         z_buffers,
         sparsity_attn,
         sparsity_mlp,
         state_steps,
         amsgrad=amsgrad,
         kappa=kappa,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
         capturable=capturable,
         states=states,
         lambda_param=lambda_param,
         lambda2_param=lambda2_param,
         sapmle_num=sapmle_num
         )

def reshape_weights(weight_matrix):
    m, n = weight_matrix.shape
    weight_matrix = weight_matrix.view(-1, 4)
    return weight_matrix, m, n

def reshape_weights_back(weight_matrix, m, n):
    weight_matrix = weight_matrix.view(m, n)
    return weight_matrix

def process_weights(weight_matrix): 

    #  abs and signed
    abs_weights = torch.abs(weight_matrix)
    #sign_mask = torch.sign(weight_matrix)  
    
    # Reorder descendingly
    sorted_weights, original_indices = torch.sort(abs_weights, descending=True, dim=1)
    
    return sorted_weights,  original_indices

def reorder_to_original(sorted_weights, original_indices):
    # Reorder back 
    k = sorted_weights.shape[0]
    row_indices = torch.arange(k).unsqueeze(1).expand_as(original_indices)
    reordered_weights = torch.zeros_like(sorted_weights)     # scatter    
    reordered_weights[row_indices, original_indices] = sorted_weights
    
    return reordered_weights

def sparsify_top2_abs_weights(weight_matrix):
    abs_weights = torch.abs(weight_matrix)             # (n, 4)
    top2_values, top2_indices = torch.topk(abs_weights, k=2, dim=1)

    sparse_weights = torch.zeros_like(weight_matrix)
    sparse_weights.scatter_(dim=1, index=top2_indices, src=top2_values)

    return sparse_weights


def soft_threshold_nonneg(x, tau):
    return torch.maximum(x - tau, torch.tensor(0.0, device=x.device))

def solve_prox_sorted_alternating(z, lamb, iter_num=100):
    w = torch.zeros_like(z)
    wprev = torch.zeros_like(z)
    for _ in range(iter_num):
        wprev.copy_(w)
        
        inter_term = lamb * (w[:, 0] * w[:, 1] + w[:, 1] * w[:, 2] + w[:, 2] * w[:, 0])
        w[:, 3] = soft_threshold_nonneg(z[:, 3], inter_term)
        
        inter_term = lamb * (w[:, 0] * w[:, 1] + w[:, 1] * w[:, 3] + w[:, 3] * w[:, 0])
        w[:, 2] = soft_threshold_nonneg(z[:, 2], inter_term)
        
        inter_term = lamb * (w[:, 0] * w[:, 2] + w[:, 2] * w[:, 3] + w[:, 3] * w[:, 0])
        w[:, 1] = soft_threshold_nonneg(z[:, 1], inter_term)
        
        inter_term = lamb * (w[:, 1] * w[:, 2] + w[:, 2] * w[:, 3] + w[:, 3] * w[:, 1])
        w[:, 0] = soft_threshold_nonneg(z[:, 0], inter_term)
        
        # Check for convergence
        if torch.sum(torch.abs(w - wprev)) < 1e-8:
            break

    return w

def solve_prox_sorted_alternating_top3(z, lamb, iter_num=100):
    w = torch.zeros_like(z)
    wprev = torch.zeros_like(z)

    for _ in range(iter_num):
        wprev.copy_(w)
        
        inter_term = lamb * (w[:, 0] * w[:, 1])
        w[:, 2] = soft_threshold_nonneg(z[:, 2], inter_term)
        
        inter_term = lamb * (w[:, 0] * w[:, 2])
        w[:, 1] = soft_threshold_nonneg(z[:, 1], inter_term)
        
        inter_term = lamb * (w[:, 1] * w[:, 2])
        w[:, 0] = soft_threshold_nonneg(z[:, 0], inter_term)
        
        # Check for convergence
        if torch.sum(torch.abs(w - wprev)) < 1e-8:
            break

    return w

def reg(w):
    w1 = w[:, 0]
    w2 = w[:, 1]
    w3 = w[:, 2]
    w4 = w[:, 3]
    return (torch.abs(w1 * w2 * w3) + torch.abs(w2 * w3 * w4) + torch.abs(w3 * w4 * w1) + torch.abs(w4 * w1 * w2))

def obj(w, z, lamb):
    return 0.5 * torch.norm(w - z, p=2, dim=1)**2 + lamb * reg(w)   

def prox_op(weight_matrix, lambda_ = 0.1): 

    weight_matrix, m, n = reshape_weights(weight_matrix) # check
    sorted_weights, original_indices = process_weights(weight_matrix)
    
    del weight_matrix
    
    w2sparse = sorted_weights.clone()
    w2sparse[:, 2] = 0
    w2sparse[:, 3] = 0
    
    w3sparse = solve_prox_sorted_alternating_top3(sorted_weights, lambda_)
    w4sparse = solve_prox_sorted_alternating(sorted_weights, lambda_)

    obj_2sparse = obj(w2sparse, sorted_weights, lambda_)
    obj_3sparse = obj(w3sparse, sorted_weights, lambda_)
    obj_4sparse = obj(w4sparse, sorted_weights, lambda_)


    min_obj_values = torch.min(torch.stack([obj_2sparse, obj_3sparse, obj_4sparse]), dim=0)
    best_weights = torch.where(min_obj_values.indices.unsqueeze(1) == 0, w2sparse, 
                                torch.where(min_obj_values.indices.unsqueeze(1) == 1, w3sparse, w4sparse))
    signed_weights = reorder_to_original(best_weights, original_indices)

    sorted_weights = reshape_weights_back(signed_weights, m, n)
    del w2sparse, w3sparse, w4sparse
    return sorted_weights

def _single_tensor_adamwprox(params: List[Tensor],
                         grads: List[Tensor],
                         exp_avgs: List[Tensor],
                         exp_avg_sqs: List[Tensor],
                         max_exp_avg_sqs: List[Tensor],
                         gamma_buffers: List[Tensor],
                         z_buffers: List[Tensor],
                         sparsity_attn: float,
                         sparsity_mlp: float,
                         state_steps: List[Tensor],
                         *,
                         amsgrad: bool,
                         kappa: float,
                         beta1: float,
                         beta2: float,
                         lr: float,
                         weight_decay: float,
                         eps: float,
                         maximize: bool,
                         capturable: bool,
                         states:List[dict],
                         lambda_param: float,
                         lambda2_param: float,
                         sapmle_num: int
                         ):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        gamma_buffer = gamma_buffers[i]
        z_buffer = z_buffers[i]
        st = states[i]
        is_layer= st["is_layer"]
        is_attn= st["is_attn"]
        is_mlp= st["is_mlp"]

        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        ext_metric = st.get("w_metric", None)
        scaler_row = st.get("scaler_row",None)

        if capturable:
            assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."

        # update step
        step_t += 1
        #print(step_t)

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        #print("capturable",capturable)
        if capturable:
            step = step_t

            # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
            # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
            bias_correction1 = 1 - torch.pow(beta1, step)
            bias_correction2 = 1 - torch.pow(beta2, step)

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            #print("cap\n")
            step = step_t.item()
            #step = step_t

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = math.sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            
            param.addcdiv_(exp_avg, denom, value=-step_size)



            if is_layer:
                kappa=0.01
       
                W_metric = ext_metric
                #z_buffer.add_(kappa * (param - gamma_buffer))
                z_buffer.add_(kappa * (W_metric-gamma_buffer))
                
                # if W_metric is not None:
                #     z_buffer.add_(kappa * W_metric * direction)
                # else:
                #     z_buffer.add_(kappa * (param - gamma_buffer))

            # use_24_panelty=False
            # if use_24_panelty:
            #     g_sign=prox_op(z_buffer,lamda)
        if is_layer:
            lamda=lambda_param

            if lamda > 0:
                z_buffer = z_buffer.float()
      
                z_flat = z_buffer.abs().flatten()
                if z_flat.numel() > 1000000:  
                    sample_size = min(100000, z_flat.numel())
                    indices = torch.randperm(z_flat.numel(), device=z_flat.device)[:sample_size]
                    z_sample = z_flat[indices]
                    layer_quantile = torch.quantile(z_sample, lamda)
                else:
                    layer_quantile = torch.quantile(z_flat, lamda)
                # soft-thresholding
                g_sign = torch.sign(z_buffer) * torch.clamp(z_buffer.abs() - layer_quantile, min=0.0)

            else:
                g_sign = z_buffer
            # g_sign=param.data

            # del w2sparse 
            #param.mul_(1+lr*0.1*g_sign/W_metric)
            new_mask = (gamma_buffer == 0) & (g_sign != 0)
            step_val = int(step if isinstance(step, (int, float)) else step_t.item())
            if new_mask.any():
                # idx = torch.nonzero(new_mask, as_tuple=False)
                # vals = g_sign[new_mask].to(torch.float)  # value
                

                if "first_step_mat" not in st:
                    st["first_step_mat"] = torch.full(st["shape"], -1, dtype=torch.int32,device=param.device)
                
                mask_to_record = (st["first_step_mat"] == -1) & new_mask
                if mask_to_record.any():
                    st["first_step_mat"][mask_to_record] = step_val
                
            if EVENT_DIR is not None and (step_val % sapmle_num == 0):
                layer_name = st.get("name", "unknown_layer")
                layer_dir = os.path.join(EVENT_DIR, layer_name)
                os.makedirs(layer_dir, exist_ok=True)
                
                first_step_file = os.path.join(layer_dir, "first_step.pt")
                torch.save(st["first_step_mat"], first_step_file)
                
                gamma_file = os.path.join(layer_dir, "gamma_buffer.pt")
                torch.save(gamma_buffer, gamma_file)

            gamma_buffer.copy_(g_sign)
            #gamma_buffer.copy_(param.data)
        
        if is_layer:
            step_val = step_t.item()
            layer_name = st.get("name", "unknown_layer")
            
            if layer_name == "model.layers.0.self_attn.k_proj.weight":
                print(f"[AdamWSPP] {layer_name} step {step_val}: z_buffer.mean={z_buffer.mean().item():.6f} gamma_buffer.mean={gamma_buffer.mean().item():.6f}  param.mean={param.mean().item():.12f} grad.mean={grad.mean().item():.6f}")
        #         print(f"  [DEBUG] step_size: {step_size:.6f}, param_change_after_adam: {(param_after_adam - param_before_adam).norm().item():.9f}")
        #         print(f"  [DEBUG] exp_avg.norm: {exp_avg.norm().item():.6f}, denom.norm: {denom.norm().item():.6f}")
        #         print(f"  [DEBUG] param_before_adam.norm: {param_before_adam.norm().item():.6f}, param_after_adam.norm: {param_after_adam.norm().item():.6f}")
        #         print(f"  [DEBUG] param.norm at end: {param.norm().item():.6f}")


def _multi_tensor_adamwprox(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        gamma_buffers: List[Tensor],
                        z_buffers: List[Tensor],
                        state_steps: List[Tensor],
                        *,
                        amsgrad: bool,
                        kappa: float,
                        beta1: float,
                        beta2: float,
                        lr: float,
                        weight_decay: float,
                        eps: float,
                        maximize: bool,
                        capturable: bool,
                        states: List[dict],
                        lambda_param: float):
    if len(params) == 0:
        return

    if capturable:
        assert all(p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)), \
            "If capturable=True, params and state_steps must be CUDA tensors."

    if maximize:
        grads = torch._foreach_neg(tuple(grads))

    # update steps
    torch._foreach_add_(state_steps, 1)

    # Perform stepweight decay
    torch._foreach_mul_(params, 1 - lr * weight_decay)

    # Decay the first and second moment running average coefficient
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)

    if capturable:
        # TODO: use foreach_pow if/when foreach_pow is added
        bias_correction1 = [torch.pow(beta1, step) for step in state_steps]
        bias_correction2 = [torch.pow(beta2, step) for step in state_steps]
        # foreach_sub doesn't allow a scalar as the first arg
        torch._foreach_sub_(bias_correction1, 1)
        torch._foreach_sub_(bias_correction2, 1)
        torch._foreach_neg_(bias_correction1)
        torch._foreach_neg_(bias_correction2)

        # foreach_div doesn't allow a scalar as the first arg
        step_size = torch._foreach_div(bias_correction1, lr)
        torch._foreach_reciprocal_(step_size)
        torch._foreach_neg_(step_size)

        bias_correction2_sqrt = torch._foreach_sqrt(bias_correction2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            max_exp_avg_sqs = torch._foreach_maximum(max_exp_avg_sqs, exp_avg_sqs)  # type: ignore[assignment]

            # Use the max. for normalizing running avg. of gradient
            max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
            # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
            # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
            torch._foreach_div_(max_exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
            eps_over_step_size = torch._foreach_div(step_size, eps)
            torch._foreach_reciprocal_(eps_over_step_size)
            denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps_over_step_size)
        else:
            exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
            eps_over_step_size = torch._foreach_div(step_size, eps)
            torch._foreach_reciprocal_(eps_over_step_size)
            denom = torch._foreach_add(exp_avg_sq_sqrt, eps_over_step_size)

        torch._foreach_addcdiv_(params, exp_avgs, denom)
    else:
        bias_correction1 = [1 - beta1 ** step.item() for step in state_steps]
        bias_correction2 = [1 - beta2 ** step.item() for step in state_steps]

        step_size = [(lr / bc) * -1 for bc in bias_correction1]

        bias_correction2_sqrt = [math.sqrt(bc) for bc in bias_correction2]

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            max_exp_avg_sqs = torch._foreach_maximum(max_exp_avg_sqs, exp_avg_sqs)  # type: ignore[assignment]

            # Use the max. for normalizing running avg. of gradient
            max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
            torch._foreach_div_(max_exp_avg_sq_sqrt, bias_correction2_sqrt)
            denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps)
        else:
            exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            denom = torch._foreach_add(exp_avg_sq_sqrt, eps)

        torch._foreach_addcdiv_(params, exp_avgs, denom, step_size)