import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM
import re

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


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
    sign_mask = torch.sign(weight_matrix)  
    
    # Reorder descendingly
    sorted_weights, original_indices = torch.sort(abs_weights, descending=True, dim=1)
    
    return sorted_weights, sign_mask, original_indices

def apply_sign_mask(weights, sign_mask):
    return weights * sign_mask

def reorder_to_original(sorted_weights, original_indices):
    # Reorder back 
    k = sorted_weights.shape[0]
    row_indices = torch.arange(k).unsqueeze(1).expand_as(original_indices)
    reordered_weights = torch.zeros_like(sorted_weights)     # scatter    
    reordered_weights[row_indices, original_indices] = sorted_weights
    
    return reordered_weights

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

def prox_op(weight_matrix, lambda_ = 0.1): # in this implementation, lambda could either be a value or tensor
    weight_matrix, m, n = reshape_weights(weight_matrix) # check
    sorted_weights, sign_mask, original_indices = process_weights(weight_matrix)
    
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
    reordered_weights = reorder_to_original(best_weights, original_indices)

    signed_weights = apply_sign_mask(reordered_weights, sign_mask)

    sorted_weights = reshape_weights_back(signed_weights, m, n)
    del w2sparse, w3sparse, w4sparse
    return sorted_weights
def prox_op_row(weight_param, module, lambda_=0.1, frac=0.5, threshold="median"):

    if not hasattr(module, "w_metric"):
        raise ValueError(f"Module {module} 没有 w_metric buffer，请先调用 recompute_wanda_metrics_for_step_param")

    W_metric = module.w_metric.to(weight_param.device).to(weight_param.dtype) 

    # Step 1. 
    row_sums = W_metric.abs().sum(dim=1)

    if threshold == "median":
        th = row_sums.median()
    elif threshold == "mean":
        th = row_sums.mean()
    else:
        th = row_sums.new_tensor(float(threshold))  # 保持 dtype

    # Step 2. mask & penalty
    mask = (row_sums < th).to(weight_param.dtype).unsqueeze(1)  # [out,1]
    penalty = 1.0 - mask * (1.0 - frac)
    penalty = penalty.to(weight_param.device).to(weight_param.dtype)

    # Step 3. apply penalty
    new_weight = weight_param * penalty
    return new_weight

def ria_prox_op(weight_matrix, lambda_ = 0.1,frac=0.5): # in this implementation, lambda could either be a value or tensor
    #W = weight_matrix
    W_ref = weight_matrix  # state["ref"]
    # col_sum_ref = torch.mean(torch.abs(W_ref), dim=0, keepdim=True)
    # RI_col = torch.abs(W_ref) / (col_sum_ref  + 1e-12)

    # shape: [out_features, 1]
    row_sum_ref = torch.mean(torch.abs(W_ref), dim=1, keepdim=True)
    RI_row = torch.abs(W_ref) / (row_sum_ref  + 1e-12)

    #RI = 0.5 * (RI_col + RI_row)
    RI =  RI_row
    weight_matrix, m, n = reshape_weights(RI) # check
    sorted_weights, sign_mask, original_indices = process_weights(weight_matrix)
    
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
    reordered_weights = reorder_to_original(best_weights, original_indices)

    signed_weights = apply_sign_mask(reordered_weights, sign_mask)

    sorted_weights = reshape_weights_back(signed_weights, m, n)
    del w2sparse, w3sparse, w4sparse
    return sorted_weights

def ours_prox_op(weight_matrix, lambda_ = 0.1, frac=0.5): # in this implementation, lambda could either be a value or tensor
    #print("prox op step")
    weight_matrix, m, n = reshape_weights(weight_matrix) # check
    sorted_weights, sign_mask, original_indices = process_weights(weight_matrix)
    
    del weight_matrix
    
    w2sparse = sorted_weights.clone()
    # w2sparse[:, 2] = 0
    # w2sparse[:, 3] = 0
    
    for col in [2, 3]:
        w = w2sparse[:, col]
        w2sparse[:, col] = torch.sign(w) * torch.clamp(torch.abs(w) - lambda_*frac, min=0.0)
        #w2sparse[:, col] = np.sign(w) * np.maximum(np.abs(w) - lambda_, 0)
    # w3sparse = solve_prox_sorted_alternating_top3(sorted_weights, lambda_)
    # w4sparse = solve_prox_sorted_alternating(sorted_weights, lambda_)

    # obj_2sparse = obj(w2sparse, sorted_weights, lambda_)
    # obj_3sparse = obj(w3sparse, sorted_weights, lambda_)
    # obj_4sparse = obj(w4sparse, sorted_weights, lambda_)


    # min_obj_values = torch.min(torch.stack([obj_2sparse, obj_3sparse, obj_4sparse]), dim=0)
    # best_weights = torch.where(min_obj_values.indices.unsqueeze(1) == 0, w2sparse, 
    #                             torch.where(min_obj_values.indices.unsqueeze(1) == 1, w3sparse, w4sparse))
    best_weights = w2sparse
    reordered_weights = reorder_to_original(best_weights, original_indices)

    signed_weights = apply_sign_mask(reordered_weights, sign_mask)

    sorted_weights = reshape_weights_back(signed_weights, m, n)
    del w2sparse
    return sorted_weights

def compute_mask_loss(model, loss, lambda2_ = 0.01):
    loss_mask = torch.tensor(0, dtype = loss.dtype, device = loss.device)
    for n, m in model.named_modules():
         if "k_proj" in n or "q_proj" in n or "v_proj" in n or "o_proj" in n or "up_proj" in n or "down_proj" in n or "gate_proj" in n:
            custom_loss = m.compute_proximal_loss() 
            loss_mask += custom_loss.to(loss.device)
            torch.cuda.empty_cache()
            # print("here")
    return lambda2_ * loss_mask

def explicit_sparsify_magnitude(weight, n, m): 
    dim = weight.shape
    weight = weight.view(-1, m) 
    w_mask = torch.zeros_like(weight)
    w_mask.scatter_(1, torch.topk(torch.abs(weight), n, dim = 1, largest = True)[1], True)
    weight = w_mask * weight
    weight = weight.view(dim[0], dim[1])
    return weight

def replace_weight(model, lam=0.1, frac=0.5, method='prox' ,global_op=True, print_shapes=False):
    """
    lam: 
    global_op: 
    """
    #Llama 2 7b 32 Qwen 2.5 14b 48
    num_layers = 32
    
    with torch.no_grad():
        for name, p in model.named_parameters():

            if "bias" not in name and ("k_proj" in name  or "q_proj" in name  or "v_proj" in name  or "o_proj" in name or "up_proj" in name  or "down_proj" in name  or "gate_proj" in name  or "out_proj" in name  or "fc1" in name  or "fc2" in name ): 
                p.copy_(prox_op(p.data, lam).detach())
                #p.data = prox_op(p.data, lam)
                torch.cuda.empty_cache()
         
            # else:
            #     print("noprox name",name,p.data.shape)

def replace_weight_prox(model, lambda_ = 0.1):
    for n, m in model.named_parameters():
        if "bias" not in n and ("k_proj" in n or "q_proj" in n or "v_proj" in n or "o_proj" in n or "up_proj" in n or "down_proj" in n or "gate_proj" in n or "out_proj" in n or "fc1" in n or "fc2" in n):
            m.data = prox_op(m.data, lambda_) 
            torch.cuda.empty_cache()

def replace_weight_projected(model):
    for n, m in model.named_parameters():
        if "k_proj" in n or "q_proj" in n or "v_proj" in n or "o_proj" in n or "up_proj" in n or "down_proj" in n or "gate_proj" in n:
            # print(n)
            m.data = explicit_sparsify_magnitude(m.data, 2, 4) 
            torch.cuda.empty_cache()

def replace_weight_projected_lambda2(model):
    for n, m in model.named_modules():
         if "k_proj" in n or "q_proj" in n or "v_proj" in n or "o_proj" in n or "up_proj" in n or "down_proj" in n or "gate_proj" in n:
            # print(n)
            m.project_mask()
            torch.cuda.empty_cache()
        
if __name__ == '__main__':
    seed_everything(42)
    n, m = 4, 8 
    weight_matrix = torch.randn(n, m)
    print("Original weights:\n", weight_matrix)
