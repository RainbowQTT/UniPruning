import os
import time
import math
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

TARGET_KEYS = [
    "k_proj", "q_proj", "v_proj", "o_proj",
    "up_proj", "down_proj", "gate_proj",
    "out_proj", "fc1", "fc2"
]

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def is_target(name: str) -> bool:
    return name.endswith("weight") and any(k in name for k in TARGET_KEYS) and ("bias" not in name)

def resolve_module_and_param(model: nn.Module, full_name: str):
    parts = full_name.split(".")
    mod = model
    for p in parts[:-1]:
        mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
    return mod, parts[-1]

@torch.no_grad()
def apply_mask_to_linear_weight(linear: nn.Linear, mask: torch.Tensor):
    w = linear.weight
    assert w.shape == mask.shape
    w.mul_(mask.to(w.device, w.dtype))

def count_parameters(model: nn.Module) -> Tuple[int, int, int]:
    total_params = target_params = other_params = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        total_params += n
        if is_target(name):
            target_params += n
        else:
            other_params += n
    return total_params, target_params, other_params

def count_nonzero_parameters(model: nn.Module) -> Tuple[int, int, int]:
    total_nnz = target_nnz = other_nnz = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nnz = torch.count_nonzero(p).detach().cpu().item()
        total_nnz += nnz
        if is_target(name):
            target_nnz += nnz
        else:
            other_nnz += nnz
    return total_nnz, target_nnz, other_nnz

# =========================
# Mask builders
# =========================
@torch.no_grad()
def mask_group_topk_from_gamma(weight: torch.Tensor, gamma: torch.Tensor, keep: int, group: int, verbose=False):
    assert weight.dim() == 2 and gamma.shape == weight.shape
    assert weight.shape[1] % group == 0
    gamma = torch.abs(gamma)
    rows, cols = weight.shape
    gb = gamma.view(rows, -1, group)
    _, keep_idx = torch.topk(gb, keep, dim=2)
    mask = torch.zeros_like(gb, dtype=weight.dtype)
    mask.scatter_(2, keep_idx, 1.0)
    if verbose:
        nz = (gamma != 0).sum().item()
        print(f"[Gamma Sparsity] nonzero={nz}/{gamma.numel()} ({1 - nz/gamma.numel():.4f} sparsity)")
    return mask.view(rows, cols)

@torch.no_grad()
def mask_group_topk_from_first_step(
    weight: torch.Tensor,
    first_step: torch.Tensor,
    keep: int,
    group: int,
    gamma: torch.Tensor = None,             
    gamma_weight: float = 1.0               
):
    assert weight.dim() == 2 and first_step.shape == weight.shape
    assert weight.shape[1] % group == 0

    rows, _ = weight.shape
    sb = first_step.view(rows, -1, group).to(dtype=torch.float32, device=first_step.device)       

    act_mask = (sb != -1)
    has_active = act_mask.any(dim=2) 
   
    total_groups = has_active.numel()
    active_groups = has_active.sum().item()
    inactive_groups = total_groups - active_groups
    
    # Initial mask
    mask = torch.zeros_like(sb, dtype=weight.dtype)
    
    if gamma is not None:
        g = torch.abs(gamma).view(rows, -1, group)
        # min-max normalize
        g_min = torch.amin(g, dim=(0,1,2), keepdim=True)
        g_max = torch.amax(g, dim=(0,1,2), keepdim=True)
        g_norm = (g - g_min) / (g_max - g_min + 1e-12)

    if has_active.any():

        masked = sb.clone()
        masked[sb == -1] = 1000.0
        

        if gamma is not None:
            masked = masked + (-gamma_weight * g_norm)
    
        keep_idx = (-masked).topk(k=keep, dim=2).indices
        temp_mask = torch.zeros_like(sb, dtype=weight.dtype)
        temp_mask.scatter_(2, keep_idx, 1.0)
        
        temp_mask = temp_mask * act_mask.to(temp_mask.dtype)
        
        mask = torch.where(has_active.unsqueeze(-1), temp_mask, mask)
    
    if (~has_active).any() and gamma is not None:
      
        gamma_scores = g_norm 
        keep_idx_gamma = gamma_scores.topk(k=keep, dim=2).indices
        
        gamma_mask = torch.zeros_like(sb, dtype=weight.dtype)
        gamma_mask.scatter_(2, keep_idx_gamma, 1.0)
        
        mask = torch.where((~has_active).unsqueeze(-1), gamma_mask, mask)
    elif (~has_active).any() and gamma is None:
        pass

    return mask.view_as(weight)


@torch.no_grad()
def mask_unstructured_from_gamma(weight: torch.Tensor, gamma: torch.Tensor, prune_ratio: float):
    assert weight.shape == gamma.shape
    numel = weight.numel()
    keep_k = int(round(numel * (1.0 - prune_ratio)))
    if keep_k <= 0:
        return torch.zeros_like(weight, dtype=weight.dtype)
    flat = gamma.view(-1)
    # keep the largest |gamma| → prune the smallest |gamma|
    _, idx = torch.topk(flat, k=keep_k, largest=True)
    mask_flat = torch.zeros_like(flat, dtype=weight.dtype)
    mask_flat[idx] = 1.0
    return mask_flat.view_as(weight)

@torch.no_grad()
def mask_unstructured_from_gamma_by_row(weight: torch.Tensor, gamma: torch.Tensor, prune_ratio: float):
    assert weight.shape == gamma.shape
    rows, cols = weight.shape
    mask = torch.zeros_like(weight, dtype=weight.dtype)
    
    for i in range(rows):
        row_gamma = gamma[i, :]  
        row_numel = cols
        keep_k = int(round(row_numel * (1.0 - prune_ratio))) 
    
        _, idx = torch.topk(torch.abs(row_gamma), k=keep_k, largest=True)
        mask[i, idx] = 1.0
    
    return mask

# =========================
# Pruner Abstractions
# =========================
class BasePruner:
    def __init__(self, model: nn.Module):
        self.model = model

    def _target_layers(self) -> List[Tuple[str, torch.Tensor]]:
        layers = []
        for name, p in self.model.named_parameters():
            if p.requires_grad and is_target(name):
                layers.append((name, p))
        return layers

    def prune(self):
        raise NotImplementedError

class SemiStructuredPruner(BasePruner):
    """2:4 (or keep/group) pruning. Support gamma or first_step."""
    def __init__(self, model: nn.Module, record_path: str, keep: int, group: int, 
                use_gamma: bool, verbose: bool = False,
                combine_gamma_step: bool = False, gamma_weight: float = 1.0):
               
        super().__init__(model)
        self.record_path = record_path
        self.keep = keep
        self.group = group
        self.use_gamma = use_gamma
        self.verbose = verbose
        self.combine_gamma_step = combine_gamma_step
        self.gamma_weight = gamma_weight

    def _load_side_info(self, name: str, device: torch.device) -> torch.Tensor:
        if self.use_gamma:
            path = os.path.join(self.record_path, "gamma_record", name, "gamma_buffer.pt")
        else:
            path = os.path.join(self.record_path, "gamma_record", name, "first_step.pt")
        if not os.path.exists(path):
            print(f"  [Skip] side-info not found: {path}")
            return None
        return torch.load(path, map_location=device)

    @torch.no_grad()
    def prune(self):
        layers = self._target_layers()
        total = len(layers)
        processed = skipped = 0
        accum_sparsity = 0.0
        t0 = time.time()

        print(f"[Semi-Structured] keep={self.keep}, group={self.group}, use_gamma={self.use_gamma}")
        print(f"Found {total} target layers.")

        for idx, (name, p) in enumerate(layers, 1):
            print(f"\n[{idx}/{total}] {name} shape={tuple(p.shape)}")
            if p.shape[1] % self.group != 0:
                print(f"  [Skip] incompatible shape for group={self.group}")
                skipped += 1
                continue

            side_step = None
            side_gamma = None
            # step
            if not self.use_gamma or self.combine_gamma_step:
                side_step = self._load_first_step_only(name, p.device)
                side_step = side_step.to(p.device, non_blocking=True)
            # gamma
            if self.use_gamma or self.combine_gamma_step:
                side_gamma = self._load_gamma_only(name, p.device)
                side_gamma = side_gamma.to(p.device, non_blocking=True)

            if self.combine_gamma_step:
                if side_step is None:
                    print(f"  [Skip] missing step file for {name}")
                    skipped += 1
                    continue
                if side_gamma is None:
                    print(f"  [Skip] missing gamma file for {name}")
                    skipped += 1
                    continue

                mask = mask_group_topk_from_first_step(
                    p, side_step, self.keep, self.group,
                    gamma=side_gamma, gamma_weight=self.gamma_weight
                )

            elif self.use_gamma:
                if side_gamma is None:
                    print(f"  [Skip] missing gamma file for {name}")
                    skipped += 1
                    continue

                mask = mask_group_topk_from_gamma(p, side_gamma, self.keep, self.group, verbose=self.verbose)

            else:  # step-only
                if side_step is None:
                    print(f"  [Skip] missing step file for {name}")
                    skipped += 1
                    continue
                mask = mask_group_topk_from_first_step(p, side_step, self.keep, self.group)

            mod, last = resolve_module_and_param(self.model, name)
            if last == "weight" and isinstance(mod, nn.Linear):
                apply_mask_to_linear_weight(mod, mask)
                k = mask.sum().item()
                sparsity = 1.0 - k / mask.numel()
                accum_sparsity += sparsity
                processed += 1
                print(f"  Applied mask. layer sparsity={sparsity:.2%}")
            else:
                print(f"  [Warn] not Linear weight")
                skipped += 1

            # progress ETA
            elapsed = time.time() - t0
            if processed > 0:
                avg = elapsed / processed
                remain = total - processed - skipped
                eta_min = (remain * avg) / 60
                print(f"  Progress {processed}/{total}, ETA {eta_min:.1f} min")

        dt = time.time() - t0
        print(f"\n[Semi-Structured] Done in {dt/60:.1f} min. processed={processed}, skipped={skipped}")
        if processed > 0:
            print(f"Average layer sparsity: {accum_sparsity/processed:.2%}")
    
    def _load_gamma_only(self, name: str, device: torch.device):
        path = os.path.join(self.record_path, "gamma_record", name, "gamma_buffer.pt")
        if not os.path.exists(path):
            print(f"  [Skip] gamma not found: {path}")
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def _load_first_step_only(self, name: str, device: torch.device):
        path = os.path.join(self.record_path, "gamma_record",name, "first_step.pt")
        if not os.path.exists(path):
            print(f"  [Skip] first_step not found: {path}")
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

class UnstructuredPruner(BasePruner):
    """Gamma-based magnitude pruning by ratio."""
    def __init__(self, model: nn.Module, record_path: str, prune_ratio: float, by_row: bool = False):
        super().__init__(model)
        self.record_path = record_path
        self.prune_ratio = prune_ratio
        self.by_row = by_row 

    def _load_gamma(self, name: str, device: torch.device):
        path = os.path.join(self.record_path, "gamma_record", name, "gamma_buffer.pt")
        if not os.path.exists(path):
            print(f"  [Skip] gamma not found: {path}")
            return None
        return torch.load(path, map_location=device)

    @torch.no_grad()
    def prune(self):
        layers = self._target_layers()
        total = len(layers)
        processed = skipped = 0
        t0 = time.time()
        
        prune_mode = "by-row" if self.by_row else "by-layer"
        print(f"[Unstructured] prune_ratio={self.prune_ratio:.2%}, by |gamma|, mode={prune_mode}")

        for idx, (name, p) in enumerate(layers, 1):
            print(f"\n[{idx}/{total}] {name} shape={tuple(p.shape)}")
            gamma = self._load_gamma(name, p.device)
            if gamma is None:
                skipped += 1
                continue
            if self.by_row:
                mask = mask_unstructured_from_gamma_by_row(p, torch.abs(gamma), self.prune_ratio)
            else:
                mask = mask_unstructured_from_gamma(p, torch.abs(gamma), self.prune_ratio)
            
            mod, last = resolve_module_and_param(self.model, name)
            if last == "weight" and isinstance(mod, nn.Linear):
                apply_mask_to_linear_weight(mod, mask)
                kept = mask.sum().item()
                sparsity = 1.0 - kept / mask.numel()
                print(f"  Applied mask ({prune_mode}). layer sparsity={sparsity:.2%}")
                processed += 1
            else:
                print(f"  [Warn] not Linear weight")
                skipped += 1

        dt = time.time() - t0
        print(f"\n[Unstructured] Done in {dt/60:.1f} min. processed={processed}, skipped={skipped}")

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Structured / Unstructured pruning + PPL eval")
    # model & io
    parser.add_argument('--model_path', type=str, required=True, help='HF model path or local directory')
    parser.add_argument('--record_path', type=str, required=True, help='Directory containing gamma/first_step records')
    parser.add_argument('--output_path', type=str, required=True, help='Where to save pruned model')
    # mode
    parser.add_argument('--mode', type=str, default='semi', choices=['semi', 'unstructured'],
                        help='Pruning mode: semi (2:4 etc.) or unstructured')
    # semi-structured args
    parser.add_argument('--keep', type=int, default=2, help='#kept per group (semi-structured)')
    parser.add_argument('--group', type=int, default=4, help='group size (semi-structured)')
    parser.add_argument('--use_gamma', type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=True,
                        help='Use gamma (True) or first_step (False) for semi-structured')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging for gamma stats')

    parser.add_argument('--combine_gamma_step', action='store_true',
                    help='In semi-structured mode: combine first_step with normalized -|gamma|')
    parser.add_argument('--gamma_weight', type=float, default=1.0,
                    help='Weight for normalized gamma when combining with step')                
    # unstructured args
    parser.add_argument('--unstructured_ratio', type=float, default=0.6, help='global prune ratio for unstructured')
    parser.add_argument('--unstructured_by_row', action='store_true', 
                        help='For unstructured mode: prune by row instead of by layer')
    # eval
    parser.add_argument('--ctx_len', type=int, default=4096, help='context length for perplexity eval')
    # misc
    parser.add_argument('--seed', type=int, default=42, help='seed')

    args = parser.parse_args()
    seed_everything(args.seed)

    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("\n=== Before Pruning: Parameter Stats ===")
    total_b, target_b, other_b = count_parameters(model)
    print(f"Total params: {total_b:,}")
    print(f"Target-layer params: {target_b:,}")
    print(f"Other-layer params:  {other_b:,}")

    nnz_total_b, nnz_target_b, nnz_other_b = count_nonzero_parameters(model)
    print(f"Nonzeros (total/target/other): {nnz_total_b:,} / {nnz_target_b:,} / {nnz_other_b:,}")

    # ---- Prune ----
    if args.mode == 'semi':
        pruner = SemiStructuredPruner(
            model=model,
            record_path=args.record_path,
            keep=args.keep,
            group=args.group,
            use_gamma=args.use_gamma,
            verbose=args.verbose,
            combine_gamma_step=args.combine_gamma_step,
            gamma_weight=args.gamma_weight
        )
    else:
        pruner = UnstructuredPruner(
            model=model,
            record_path=args.record_path,
            prune_ratio=args.unstructured_ratio,
            by_row=args.unstructured_by_row 
        )

    print(f"\n== Start Pruning: mode={args.mode} ==")
    pruner.prune()

    print("\n=== After Pruning: Parameter Stats ===")
    total_a, target_a, other_a = count_parameters(model)
    print(f"Total params: {total_a:,}")
    print(f"Target-layer params: {target_a:,}")
    print(f"Other-layer params:  {other_a:,}")

    nnz_total_a, nnz_target_a, nnz_other_a = count_nonzero_parameters(model)
    print(f"Nonzeros (total/target/other): {nnz_total_a:,} / {nnz_target_a:,} / {nnz_other_a:,}")

    # pruning ratios (by nonzeros)
    total_pruned   = nnz_total_b  - nnz_total_a
    target_pruned  = nnz_target_b - nnz_target_a
    total_rate  = (total_pruned  / max(1, nnz_total_b))  * 100
    target_rate = (target_pruned / max(1, nnz_target_b)) * 100
    comp_ratio  = (nnz_total_b / max(1, nnz_total_a)) if nnz_total_a > 0 else float('inf')

    print("\n=== Pruning Summary (by nonzeros) ===")
    print(f"Total pruned:           {total_pruned:,}")
    print(f"Target-layer pruned:    {target_pruned:,}")
    print(f"Overall pruning rate:   {total_rate:.2f}%")
    print(f"Target-layer rate:      {target_rate:.2f}%")
    print(f"Compression ratio:      {comp_ratio:.2f}x")

    # ---- Save ----
    if args.mode == "semi":
        if args.combine_gamma_step:
            mode_tag = "semi_mixed"
        elif args.use_gamma:
            mode_tag = "semi_gamma_only"
        else:
            mode_tag = "semi_step_only"
    else:  # unstructured
        if args.unstructured_by_row:
            mode_tag = f"unstructured_by_row_{args.unstructured_ratio}"
        else:
            mode_tag = f"unstructured_by_layer_{args.unstructured_ratio}"

    saving_path = os.path.join(args.output_path, f"pruned_model_{mode_tag}")
    print(f"\nSaving pruned model to: {saving_path}")
    os.makedirs(saving_path, exist_ok=True)
    model.save_pretrained(saving_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
