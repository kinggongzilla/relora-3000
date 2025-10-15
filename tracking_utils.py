import torch
import pickle
import wandb
from pathlib import Path
from collections import defaultdict

def compute_effective_rank(singular_values):
    """
    Compute Effective Rank (ER) using entropy of normalized singular values.
    ER(θ) = exp(-∑(σ_k/||σ||_1 * log(σ_k/||σ||_1)))
    
    Args:
        singular_values: torch.Tensor of singular values in descending order
    
    Returns:
        float: Effective rank
    """
    # Normalize singular values by their L1 norm
    sigma_normalized = singular_values / (torch.sum(singular_values) + 1e-10)
    
    # Compute entropy: -∑(p * log(p))
    entropy = -torch.sum(sigma_normalized * torch.log(sigma_normalized + 1e-10))
    
    # Effective rank is exp(entropy)
    effective_rank = torch.exp(entropy)
    
    return effective_rank.item()

def compute_per(singular_values, d_inter):
    """
    Compute Proportional Effective Rank (PER).
    PER(θ) = ER(θ) / d_inter
    
    Args:
        singular_values: torch.Tensor of singular values
        d_inter: dimension of intermediate representations
    
    Returns:
        float: Proportional effective rank (between 0 and 1)
    """
    er = compute_effective_rank(singular_values)
    per = er / d_inter
    return per

def extract_layer_number(layer_name):
    """
    Extract layer number from LLaMA layer name.
    Examples:
      - 'model.layers.0.self_attn.q_proj.weight' -> 0
      - 'model.layers.15.mlp.gate_proj.weight' -> 15
      - 'model.embed_tokens.weight' -> None (not a layer)
      - 'lm_head.weight' -> None (not a layer)
    """
    parts = layer_name.split('.')
    
    # LLaMA structure: model.layers.{N}.{component}.weight
    if 'layers' in parts:
        layers_idx = parts.index('layers')
        if layers_idx + 1 < len(parts) and parts[layers_idx + 1].isdigit():
            return int(parts[layers_idx + 1])
    
    return None

def extract_weight_type(layer_name):
    """
    Extract weight type from LLaMA layer name.
    
    LLaMA weight types:
      - self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj
      - mlp.gate_proj, mlp.up_proj, mlp.down_proj
      - embed_tokens, lm_head
    
    Examples:
      - 'model.layers.0.self_attn.q_proj.weight' -> 'self_attn.q_proj'
      - 'model.layers.0.mlp.gate_proj.weight' -> 'mlp.gate_proj'
      - 'model.embed_tokens.weight' -> 'embed_tokens'
      - 'lm_head.weight' -> 'lm_head'
    """
    parts = layer_name.split('.')
    
    # LLaMA attention projections: self_attn.{q,k,v,o}_proj
    if 'self_attn' in parts:
        attn_idx = parts.index('self_attn')
        if attn_idx + 1 < len(parts) and parts[attn_idx + 1] != 'weight':
            return f"self_attn.{parts[attn_idx + 1]}"
        return 'self_attn'
    
    # LLaMA MLP projections: mlp.{gate,up,down}_proj
    if 'mlp' in parts:
        mlp_idx = parts.index('mlp')
        if mlp_idx + 1 < len(parts) and parts[mlp_idx + 1] != 'weight':
            return f"mlp.{parts[mlp_idx + 1]}"
        return 'mlp'
    
    # Embedding and output head
    if 'embed_tokens' in parts:
        return 'embed_tokens'
    if 'lm_head' in parts:
        return 'lm_head'
    
    # Fallback
    if 'weight' in parts:
        idx = parts.index('weight')
        if idx >= 2:
            return f"{parts[idx-2]}.{parts[idx-1]}"
        elif idx >= 1:
            return parts[idx-1]
    
    return 'unknown'

class WeightUpdateTracker:
    def __init__(self, model, track_every_n_steps=100, save_dir="weight_metrics"):
        self.step = 0
        self.track_every_n_steps = track_every_n_steps
        self.tracked_this_step = False
        
        # Store all metrics locally for later analysis
        self.all_metrics = []
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Create mapping from parameter id to layer name
        self.param_to_name = {}
        for name, param in model.named_parameters():
            self.param_to_name[id(param)] = name
    
    def track(self, optimizer, step=None):
        self.tracked_this_step = False

        if self.step % self.track_every_n_steps != 0:
            self.step += 1
            return
        
        with torch.no_grad():
            step_metrics = {}
            wandb_logs = {}
            
            # For aggregating metrics
            metrics_by_layer = defaultdict(lambda: {'per': [], 'condition_number': []})
            metrics_by_type = defaultdict(lambda: {'per': [], 'condition_number': []})
            
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None or p.ndim < 2:
                        continue
                    
                    state = optimizer.state.get(p, {})
                    if 'exp_avg' not in state:
                        continue
                    
                    # Reconstruct update from optimizer state
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    step_count = state['step']
                    
                    beta1, beta2 = group['betas']
                    bias_correction1 = 1 - beta1 ** step_count
                    bias_correction2 = 1 - beta2 ** step_count
                    
                    update = exp_avg / bias_correction1 / \
                            (torch.sqrt(exp_avg_sq / bias_correction2) + group['eps'])
                    
                    # Convert to float32 for SVD (BFloat16 not supported)
                    update_float32 = update.float()
                    
                    # Compute singular values
                    s = torch.linalg.svdvals(update_float32)
                    
                    # d_inter is the smaller dimension of the weight matrix
                    d_inter = min(update.shape)
                    
                    # Compute all metrics
                    per = compute_per(s, d_inter)
                    condition_number = (s[0] / s[-1]).item()
                    
                    # Get layer name, number, and weight type
                    param_id = id(p)
                    layer_name = self.param_to_name.get(param_id, f"param_{param_id}")
                    layer_number = extract_layer_number(layer_name)
                    weight_type = extract_weight_type(layer_name)
                    
                    # Store ALL singular values locally
                    step_metrics[layer_name] = {
                        'per': per,
                        'd_inter': d_inter,
                        'condition_number': condition_number,
                        'singular_values': s.cpu().numpy().tolist(),
                        'layer_number': layer_number,
                        'weight_type': weight_type,
                    }
                    
                    # Accumulate for aggregation
                    if layer_number is not None:
                        metrics_by_layer[layer_number]['per'].append(per)
                        metrics_by_layer[layer_number]['condition_number'].append(condition_number)
                    
                    metrics_by_type[weight_type]['per'].append(per)
                    metrics_by_type[weight_type]['condition_number'].append(condition_number)
                    
                    del update, update_float32, s
            
            # Log aggregated metrics by layer (averaged over all weight types in that layer)
            for layer_num, metrics in metrics_by_layer.items():
                mean_per = sum(metrics['per']) / len(metrics['per'])
                mean_cond = sum(metrics['condition_number']) / len(metrics['condition_number'])
                
                wandb_logs[f'weight_updates/by_layer/layer_{layer_num}/mean_per'] = mean_per
                wandb_logs[f'weight_updates/by_layer/layer_{layer_num}/mean_condition_number'] = mean_cond
            
            # Log aggregated metrics by weight type (averaged over all layers for that type)
            for weight_type, metrics in metrics_by_type.items():
                mean_per = sum(metrics['per']) / len(metrics['per'])
                mean_cond = sum(metrics['condition_number']) / len(metrics['condition_number'])
                
                wandb_weight_type = weight_type.replace('.', '/')
                wandb_logs[f'weight_updates/by_type/{wandb_weight_type}/mean_per'] = mean_per
                wandb_logs[f'weight_updates/by_type/{wandb_weight_type}/mean_condition_number'] = mean_cond
            

            #overall averages
            all_pers = []
            all_condition_numbers = []
            for metrics in metrics_by_type.values():
                all_pers.extend(metrics['per'])
                all_condition_numbers.extend(metrics['condition_number'])

            if all_pers:
                wandb_logs['weight_updates/overall/mean_per'] = sum(all_pers) / len(all_pers)
                wandb_logs['weight_updates/overall/mean_condition_number'] = sum(all_condition_numbers) / len(all_condition_numbers)

            # Store metrics locally
            self.all_metrics.append({
                'step': self.step,
                'layers': step_metrics
            })
            
            # Log to wandb
            if wandb_logs:
                wandb.log(wandb_logs, step=step)
            
            self.tracked_this_step = True
        
        self.step += 1

    def save_metrics(self, filename=None):
        """Save all stored metrics to disk"""
        if filename is None:
            filename = self.save_dir / f"metrics_step_{self.step}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(self.all_metrics, f)
        
        print(f"Saved {len(self.all_metrics)} metric checkpoints to {filename}")
    
    def get_storage_estimate(self):
        """Estimate storage size"""
        if not self.all_metrics:
            return "No metrics stored yet"
        
        total_singular_values = 0
        for metric_step in self.all_metrics:
            for layer_metrics in metric_step['layers'].values():
                total_singular_values += len(layer_metrics['singular_values'])
        
        bytes_estimate = total_singular_values * 8
        
        return {
            'num_checkpoints': len(self.all_metrics),
            'total_singular_values': total_singular_values,
            'estimated_mb': bytes_estimate / (1024**2),
        }