import math
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import Namespace as NS
from typing import Any, List
from composer.core.time import Time

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class Mask(nn.Module):
    def __init__(self, 
                 name: str,
                 mask_shape: List, 
                 num_params_per_mask: int, 
                 mask_output_shape: List, 
                 target_sparsity: float,
                 target_mask_size: int,
                 device: str,
                 eval_target_model: bool=True) -> None:
        super().__init__()
        self.name = name
        self.num_params_per_mask = num_params_per_mask
        self.mask_output_shape = mask_output_shape
        self.target_sparsity = target_sparsity
        self.droprate_init = 0.5
        self.temperature = 2./3.
        self.magical_number = 0.8
        self.beta = 0.83  # Hard Concrete temperature parameter
        self.device = device
        
        self.target_mask_size = target_mask_size
        self.eval_target_model = eval_target_model
        
        self.z_loga = self.initialize_mask(mask_shape) 
        self.mask_size = self.z_loga.shape[-1]
        
    def param_init_fn(self, module):
        mean = 0
        if isinstance(module, nn.Parameter):
            module.data.normal_(mean, 1e-2)
        else:
            for tensor in module.parameters():
                tensor.data.normal_(mean, 1e-2)
    
    def initialize_mask(self, mask_shape):
        z_loga = nn.Parameter(torch.zeros(mask_shape, device=self.device))
        
        # Target-aware initialization to bias toward target architecture
        if self.target_mask_size is not None and mask_shape[-1] > 0:
            target_keep_prob = self.target_mask_size / mask_shape[-1]
            target_keep_prob = max(0.01, min(0.99, target_keep_prob))  # Clamp to valid range
            alpha_init = math.log(target_keep_prob / (1 - target_keep_prob))
            z_loga.data.fill_(alpha_init)
        else:
            self.param_init_fn(z_loga)
        
        return z_loga
    
    def cdf_qz(self, x=None):
        """
        CDF of Hard Concrete distribution - probability that output is 0
        """
        if x is None:
            x = self.z_loga
        # Threshold where stretched sigmoid becomes 0
        threshold = -limit_a / (limit_b - limit_a)  # 0.1 / 1.2 ≈ 0.083
        # Inverse sigmoid to find x value that gives this threshold
        threshold_logit = math.log(threshold / (1 - threshold))
        # CDF: P(output = 0) = P(x < threshold_logit * beta)
        return torch.sigmoid((threshold_logit * self.beta - x) / self.beta)
    
    def sample_z(self):
        # Hard Concrete distribution sampling
        eps = torch.rand_like(self.z_loga)
        eps = torch.clamp(eps, epsilon, 1 - epsilon)  # Avoid log(0)
        
        # Step 1: Sample from logistic distribution
        s = torch.sigmoid((torch.log(eps / (1 - eps)) + self.z_loga) / self.beta)
        
        # Step 2: Stretch to [γ, ζ] range (limit_a, limit_b)
        s_stretched = s * (limit_b - limit_a) + limit_a
        
        # Step 3: Hard clamp to [0, 1] - this creates the "hard" concrete distribution
        z = torch.clamp(s_stretched, 0, 1)
        
        return z
    
    def _deterministic_z(self, z_loga):
        # CDF-based expected sparsity calculation
        if self.target_mask_size is not None and self.eval_target_model:
            # Use exact target for final pruning
            expected_num_zeros = z_loga.shape[-1] - self.target_mask_size
        else:
            # Use CDF to compute expected sparsity from learned distribution
            expected_score = 1 - self.cdf_qz(z_loga)  # Probability of keeping
            expected_num_nonzeros = expected_score.sum()
            expected_num_zeros = z_loga.nelement() - expected_num_nonzeros.item()
        
        try:
            num_zeros = round(expected_num_zeros)
        except:
            print("num of zeros is nan....")
            num_zeros = 0
            
        num_zeros = max(0, min(num_zeros, z_loga.shape[-1]))
        
        soft_mask = torch.sigmoid(z_loga / self.temperature * self.magical_number)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(z_loga.device)
            else:
                _, indices = torch.topk(z_loga, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
        return soft_mask
    
    def deterministic_z(self):
        if self.z_loga.ndim == 1:
            z = self._deterministic_z(self.z_loga).reshape(*self.mask_output_shape)
        else:
            z_loga = self.z_loga.reshape(-1, self.z_loga.shape[-1])
            z = []
            for i in range(z_loga.shape[0]):
                z_ = self._deterministic_z(z_loga[i])
                z.append(z_)
            z = torch.stack(z).reshape(*self.mask_output_shape)
        return z
    
    def forward(self):
        func = self.sample_z if self.training else self.deterministic_z
        z = func().reshape(self.mask_output_shape)
        return z
    
    def constrain_parameters(self):
        self.z_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def calculate_expected_score_sparsity(self):
        # CDF-based expected score calculation
        # Use same method as _deterministic_z for consistency
        expected_score = 1 - self.cdf_qz(self.z_loga)  # Probability of keeping
        soft_mask = expected_score
        sparsity = 1 - soft_mask.mean(dim=-1)
        return soft_mask, sparsity

class L0ModuleEmbedding(nn.Module):
    def __init__(self, cfg, device, pretrained_model=None):
        super(L0ModuleEmbedding, self).__init__()
        
        # Extract model info from pretrained model if available
        if pretrained_model:
            self.base_model_info = self.extract_model_info(pretrained_model)
        else:
            self.base_model_info = self.set_model_info(cfg)
            
        l0_module_cfg = cfg.l0_module
        self.target_model_info = None
        target_model_cfg = getattr(l0_module_cfg, "target_model", None)
        if target_model_cfg is not None:
            # Set target model info if it has any pruning-related fields
            if (hasattr(target_model_cfg, 'n_layers') or hasattr(target_model_cfg, 'n_heads') or 
                hasattr(target_model_cfg, 'intermediate_size') or hasattr(target_model_cfg, 'd_model') or 
                hasattr(target_model_cfg, 'hidden_size')):
                self.target_model_info = self.set_model_info(target_model_cfg)
        
        # Focus on head/layer/intermediate pruning only
        self.pruning_modules = [m for m in l0_module_cfg.pruning_modules if m in ['head', 'layer', 'intermediate']]
        self.start_sparsity = l0_module_cfg.start_sparsity
        self.lagrangian_warmup_steps = Time.from_timestring(l0_module_cfg.lagrangian_warmup_steps).value
        self.device = device
        self.eval_target_model = l0_module_cfg.get("eval_target_model", True)
        
        # LLM-Shearing style sparsity warmup (20% of total training for warmup)
        self.sparsity_warmup_steps = int(0.2 * Time.from_timestring("12000ba").value)  # 20% of 12000 steps
        self.current_step = 0
        
        self.masks = nn.ModuleDict()
        self.lambdas = nn.ParameterDict()
        
        for module_name in self.pruning_modules:
            self.initialize_one_module(module_name)
    
    def update_training_step(self, step: int):
        """Update current training step for sparsity warmup"""
        self.current_step = step
    
    def get_warmup_target_sparsity(self, final_target_sparsity: float) -> float:
        """LLM-Shearing style sparsity warmup: 0 → target over warmup period"""
        if self.current_step >= self.sparsity_warmup_steps:
            return final_target_sparsity
        
        # Linear warmup from 0 to final_target_sparsity
        progress = self.current_step / self.sparsity_warmup_steps
        return progress * final_target_sparsity
    
    def extract_model_info(self, model):
        """Extract model configuration from pretrained model"""
        info = NS()
        config = model.config
        info.hidden_size = config.hidden_size
        info.intermediate_size = config.intermediate_size
        info.num_attention_heads = config.num_attention_heads
        info.num_layers = config.num_hidden_layers  # Map to num_layers
        info.head_dim = config.hidden_size // config.num_attention_heads
        info.vocab_size = config.vocab_size
        info.max_position_embeddings = config.max_position_embeddings
        
        # Calculate parameters for BGE-M3
        info.params_per_head = info.hidden_size * info.head_dim + info.head_dim * info.hidden_size
        info.params_per_intermediate_dim = info.hidden_size + info.hidden_size
        info.params_per_mlp_layer = info.hidden_size * info.intermediate_size + info.intermediate_size * info.hidden_size
        return info
    

    
    def set_model_info(self, cfg):
        info = NS()
        # Handle both old and new config field names
        info.hidden_size = getattr(cfg, 'd_model', getattr(cfg, 'hidden_size', 1024))
        info.intermediate_size = getattr(cfg, 'intermediate_size', 4096)
        info.num_attention_heads = getattr(cfg, 'n_heads', getattr(cfg, 'num_attention_heads', 16))
        info.num_layers = getattr(cfg, 'n_layers', getattr(cfg, 'num_hidden_layers', 24))
        info.vocab_size = getattr(cfg, 'vocab_size', 250002)
        
        # BGE-M3 specific parameters for XLM-RoBERTa
        info.head_dim = info.hidden_size // info.num_attention_heads
        info.params_per_head = info.hidden_size * info.head_dim + info.head_dim * info.hidden_size
        info.params_per_intermediate_dim = info.hidden_size + info.hidden_size
        info.params_per_mlp_layer = info.hidden_size * info.intermediate_size + info.intermediate_size * info.hidden_size
        
        return info
    
    def compute_num_params(self, model_info):
        # Embedding layer
        embedding_params = model_info.vocab_size * model_info.hidden_size
        
        # Per layer parameters
        per_layer_params = (
            # Self-attention: Q, K, V projections + output projection
            4 * model_info.hidden_size * model_info.hidden_size +
            # MLP: intermediate + output projections  
            2 * model_info.hidden_size * model_info.intermediate_size +
            # Layer norms (2 per layer)
            2 * model_info.hidden_size
        )
        
        total_params = embedding_params + model_info.num_layers * per_layer_params
        
        # Add output heads for BGE-M3 (dense, sparse, multi-vector)
        output_head_params = 3 * model_info.hidden_size * model_info.hidden_size
        total_params += output_head_params
        
        return total_params
    
    def compute_prunable_params(self):
        prunable_model_size = 0
        
        # Attention heads pruning
        if "head" in self.pruning_modules:
            prunable_head_size = (
                self.base_model_info.num_layers * 
                self.base_model_info.num_attention_heads * 
                self.base_model_info.params_per_head
            )
            prunable_model_size += prunable_head_size
        
        # MLP intermediate dimension pruning
        if "intermediate" in self.pruning_modules:
            prunable_mlp_size = (
                self.base_model_info.num_layers * 
                self.base_model_info.intermediate_size * 
                self.base_model_info.params_per_intermediate_dim
            )
            prunable_model_size += prunable_mlp_size
        
        # Layer pruning
        if "layer" in self.pruning_modules:
            layer_params = (
                4 * self.base_model_info.hidden_size * self.base_model_info.hidden_size +
                2 * self.base_model_info.hidden_size * self.base_model_info.intermediate_size +
                2 * self.base_model_info.hidden_size
            )
            prunable_model_size += self.base_model_info.num_layers * layer_params
        
        # Skip hidden dimension pruning (not included in production version)
        
        return prunable_model_size
    
    def initialize_one_module(self, module_name: str):
        func_name = f"initialize_{module_name}"
        method = getattr(self, func_name)
        method()
    
    # Hidden dimension pruning removed for production version
    # Focus on structured pruning: head, layer, intermediate only
    
    def initialize_head(self):
        mask_shape = [self.base_model_info.num_layers, self.base_model_info.num_attention_heads]
        num_params_per_mask = self.base_model_info.params_per_head
        # Simplified mask shape for proper broadcasting: [num_layers, num_heads]
        mask_output_shape = [self.base_model_info.num_layers, self.base_model_info.num_attention_heads]
        
        target_head_sparsity = None
        target_mask_size = None
        if self.target_model_info is not None and hasattr(self.target_model_info, 'num_attention_heads'):
            target_head_sparsity = 1 - self.target_model_info.num_attention_heads / self.base_model_info.num_attention_heads
            target_mask_size = self.target_model_info.num_attention_heads
            self.lambdas.update({
                "lambda_1_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                "lambda_2_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device))
            })
        
        head_mask = Mask(
            name="head",
            mask_shape=mask_shape,
            num_params_per_mask=num_params_per_mask,
            mask_output_shape=mask_output_shape,
            target_sparsity=target_head_sparsity,
            target_mask_size=target_mask_size,
            device=self.device,
            eval_target_model=self.eval_target_model
        )
        self.masks["head"] = head_mask
    
    def initialize_layer(self):
        mask_shape = [self.base_model_info.num_layers]
        num_params_per_mask = (
            4 * self.base_model_info.hidden_size * self.base_model_info.hidden_size +
            2 * self.base_model_info.hidden_size * self.base_model_info.intermediate_size +
            2 * self.base_model_info.hidden_size
        )
        mask_output_shape = [self.base_model_info.num_layers]
        
        target_layer_sparsity = None
        target_mask_size = None
        if self.target_model_info is not None and hasattr(self.target_model_info, 'num_layers'):
            target_layer_sparsity = 1 - self.target_model_info.num_layers / self.base_model_info.num_layers
            target_mask_size = self.target_model_info.num_layers
            self.lambdas.update({
                "lambda_1_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                "lambda_2_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device))
            })
        
        layer_mask = Mask(
            name="layer",
            mask_shape=mask_shape,
            num_params_per_mask=num_params_per_mask,
            mask_output_shape=mask_output_shape,
            target_sparsity=target_layer_sparsity,
            target_mask_size=target_mask_size,
            device=self.device,
            eval_target_model=self.eval_target_model
        )
        self.masks["layer"] = layer_mask
    
    def initialize_intermediate(self):
        mask_shape = [self.base_model_info.num_layers, self.base_model_info.intermediate_size]
        num_params_per_mask = self.base_model_info.params_per_intermediate_dim
        # Simplified mask shape: [num_layers, intermediate_size]
        mask_output_shape = [self.base_model_info.num_layers, self.base_model_info.intermediate_size]
        
        target_int_sparsity = None
        target_mask_size = None
        if self.target_model_info is not None and hasattr(self.target_model_info, 'intermediate_size'):
            target_int_sparsity = 1 - self.target_model_info.intermediate_size / self.base_model_info.intermediate_size
            target_mask_size = self.target_model_info.intermediate_size
            self.lambdas.update({
                "lambda_1_intermediate": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                "lambda_2_intermediate": torch.nn.Parameter(torch.tensor(0.0, device=self.device))
            })
        
        int_mask = Mask(
            name="intermediate",
            mask_shape=mask_shape,
            num_params_per_mask=num_params_per_mask,
            mask_output_shape=mask_output_shape,
            target_sparsity=target_int_sparsity,
            target_mask_size=target_mask_size,
            device=self.device,
            eval_target_model=self.eval_target_model
        )
        self.masks["intermediate"] = int_mask
    
    def get_sparsity_loss(self):
        sparsity_loss = 0
        expected_sparsity = {}
        expected_score = {}
        
        for mask_name, mask in self.masks.items():
            score, sparsity = mask.calculate_expected_score_sparsity()
            expected_sparsity[mask_name] = sparsity
            expected_score[mask_name] = score
            
            if mask_name in ["head", "layer", "intermediate"]:
                if mask.target_sparsity is not None:
                    sparsity_loss += torch.abs(sparsity.mean() - mask.target_sparsity)
        
        return sparsity_loss, expected_sparsity, expected_score
    
    def forward(self):
        zs = {}
        for mask_name, mask in self.masks.items():
            zs[f"{mask_name}_z"] = mask()
        return zs
    
    def constrain_parameters(self):
        for mask in self.masks.values():
            mask.constrain_parameters()
        
        # Constrain lambda parameters to be non-negative
        for param in self.lambdas.values():
            param.data.clamp_(min=0.0)
    

