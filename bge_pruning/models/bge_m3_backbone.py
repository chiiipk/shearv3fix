import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel, XLMRobertaConfig
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from typing import Optional, Dict, Any

class MaskedXLMRobertaAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads
        )

        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)

        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        head_z: Optional[torch.Tensor] = None,
    ) -> tuple:

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if head_z is not None:
            # Reshape for broadcasting: [num_heads] → [1, num_heads, 1, 1]
            head_z = head_z.view(1, -1, 1, 1)
            query_layer = query_layer * head_z
            key_layer = key_layer * head_z
            value_layer = value_layer * head_z

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class MaskedXLMRobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states: torch.Tensor, 
                intermediate_z: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        
        if intermediate_z is not None:
            hidden_states = hidden_states * intermediate_z
            
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class MaskedXLMRobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class MaskedXLMRobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MaskedXLMRobertaAttention(config)
        self.intermediate = MaskedXLMRobertaIntermediate(config)
        self.output = MaskedXLMRobertaOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        head_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
    ) -> tuple:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            head_z=head_z,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]

        layer_output = self.feed_forward_chunk(attention_output, intermediate_z)
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output, intermediate_z=None):
        intermediate_output = self.intermediate(attention_output, intermediate_z)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class MaskedXLMRobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([MaskedXLMRobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        layer_z: Optional[torch.Tensor] = None,
        head_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
    ) -> tuple:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            # Check if layer is pruned
            if layer_z is not None and layer_z[i].item() == 0:
                continue

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Get layer-specific masks
            layer_head_z = head_z[i] if head_z is not None else None
            layer_intermediate_z = intermediate_z[i] if intermediate_z is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                output_attentions,
                layer_head_z,
                layer_intermediate_z,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

class MaskedBGEM3Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = XLMRobertaModel(config).embeddings
        self.encoder = MaskedXLMRobertaEncoder(config)
        self.pooler = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        layer_z: Optional[torch.Tensor] = None,
        head_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
    ) -> tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            layer_z=layer_z,
            head_z=head_z,
            intermediate_z=intermediate_z,
        )
        sequence_output = encoder_outputs[0]

        return {
            "last_hidden_state": sequence_output,
            "hidden_states": encoder_outputs[1] if output_hidden_states else None,
            "attentions": encoder_outputs[2] if output_attentions else None,
        }

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, input_shape: tuple) -> torch.Tensor:
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})")

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        return extended_attention_mask

    def get_head_mask(self, head_mask: Optional[torch.Tensor], num_hidden_layers: int) -> Optional[torch.Tensor]:
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
    
    def save_pretrained(self, save_path: str):
        """Save model in HuggingFace format"""
        import os
        from pathlib import Path
        
        # Ensure save directory exists
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save model state dict in PyTorch format
        model_path = os.path.join(save_path, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save config with actual embedding dimensions
        actual_vocab_size = self.embeddings.word_embeddings.weight.shape[0]
        actual_max_pos = self.embeddings.position_embeddings.weight.shape[0]
        actual_type_vocab = self.embeddings.token_type_embeddings.weight.shape[0]
        
        config_dict = {
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "intermediate_size": self.config.intermediate_size,
            "vocab_size": actual_vocab_size,
            "max_position_embeddings": actual_max_pos,
            "type_vocab_size": actual_type_vocab,
            "model_type": "xlm-roberta",
            "architectures": ["XLMRobertaModel"],
        }
        
        import json
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def prune_params(self, zs: Dict[str, torch.Tensor]):
        """Prune parameters based on masks"""
        if "layer_z" in zs:
            # Remove pruned layers and make indices contiguous
            layer_mask = zs["layer_z"]
            remaining_layers = torch.where(layer_mask > 0)[0]
            print(f"Pruning layers: keeping {len(remaining_layers)}/{len(layer_mask)} layers {remaining_layers.tolist()}")
            
            # Create new ModuleList with contiguous indices
            self.encoder.layer = nn.ModuleList([self.encoder.layer[i] for i in remaining_layers])
            self.config.num_hidden_layers = len(remaining_layers)
            
            print(f"New layer count: {self.config.num_hidden_layers} (indices now 0-{self.config.num_hidden_layers-1})")

        if "head_z" in zs:
            # Prune attention heads
            head_mask = zs["head_z"]
            total_remaining_heads = 0
            for i, layer in enumerate(self.encoder.layer):
                if i < head_mask.shape[0]:
                    layer_head_mask = head_mask[i].squeeze()
                    pruned_heads = [j for j, mask in enumerate(layer_head_mask) if mask == 0]
                    layer.attention.prune_heads(pruned_heads)
                    total_remaining_heads += layer.attention.num_attention_heads
            
            # Update config ensuring mathematical consistency
            if len(self.encoder.layer) > 0:
                avg_heads = total_remaining_heads // len(self.encoder.layer)
                # Ensure hidden_size is divisible by num_attention_heads
                if self.config.hidden_size % avg_heads != 0:
                    # Adjust to nearest valid head count that divides hidden_size
                    for h in range(avg_heads, 0, -1):
                        if self.config.hidden_size % h == 0:
                            avg_heads = h
                            break
                self.config.num_attention_heads = avg_heads



        if "intermediate_z" in zs:
            # Prune intermediate dimensions in MLP layers
            intermediate_mask = zs["intermediate_z"]
            total_remaining_intermediate = 0
            for i, layer in enumerate(self.encoder.layer):
                if i < intermediate_mask.shape[0]:
                    layer_int_mask = intermediate_mask[i]
                    remaining_idx = torch.where(layer_int_mask > 0)[0]
                    total_remaining_intermediate += len(remaining_idx)
                    
                    # Prune intermediate layers
                    if hasattr(layer, 'intermediate') and hasattr(layer.intermediate, 'dense'):
                        old_weight = layer.intermediate.dense.weight.data
                        old_bias = layer.intermediate.dense.bias.data
                        layer.intermediate.dense = nn.Linear(old_weight.size(1), len(remaining_idx))
                        layer.intermediate.dense.weight.data = old_weight[remaining_idx]
                        layer.intermediate.dense.bias.data = old_bias[remaining_idx]
                    
                    if hasattr(layer, 'output') and hasattr(layer.output, 'dense'):
                        old_weight = layer.output.dense.weight.data
                        layer.output.dense = nn.Linear(len(remaining_idx), old_weight.size(0))
                        layer.output.dense.weight.data = old_weight[:, remaining_idx]
            
            # Update global config with average intermediate size per layer
            if len(self.encoder.layer) > 0:
                self.config.intermediate_size = total_remaining_intermediate // len(self.encoder.layer)
