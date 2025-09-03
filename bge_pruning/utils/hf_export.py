"""Minimal HuggingFace export utilities for BGE-M3 pruned models (with vocab remap)"""

import torch
import json
import os
from pathlib import Path
from transformers import AutoModel, AutoConfig, AutoTokenizer


def get_layer_count_from_state_dict(state_dict):
    """Extract number of layers from state dict keys"""
    max_layer = -1
    for key in state_dict.keys():
        if "encoder.layer." in key:
            try:
                layer_num = int(key.split("encoder.layer.")[1].split(".")[0])
                max_layer = max(max_layer, layer_num)
            except (ValueError, IndexError):
                continue
    return max_layer + 1 if max_layer >= 0 else 0


def create_hf_config_from_backbone(backbone):
    """Create HuggingFace config from pruned backbone"""
    config = backbone.config
    
    # Get actual dimensions from (pruned) model
    actual_layers = len(backbone.encoder.layer)
    actual_vocab_size = backbone.embeddings.word_embeddings.weight.shape[0]
    actual_max_pos = backbone.embeddings.position_embeddings.weight.shape[0]
    actual_type_vocab = backbone.embeddings.token_type_embeddings.weight.shape[0]
    
    # Use first layer to get actual head count and intermediate size
    if actual_layers > 0:
        first_layer = backbone.encoder.layer[0]
        actual_heads = first_layer.attention.num_attention_heads
        actual_intermediate = first_layer.intermediate.dense.out_features
    else:
        actual_heads = config.num_attention_heads
        actual_intermediate = config.intermediate_size
    
    return {
        "architectures": ["XLMRobertaModel"],
        "attention_probs_dropout_prob": getattr(config, "attention_probs_dropout_prob", 0.1),
        "bos_token_id": 0,
        "classifier_dropout": None,
        "eos_token_id": 2,
        "hidden_act": "gelu",
        "hidden_dropout_prob": getattr(config, "hidden_dropout_prob", 0.1),
        "hidden_size": config.hidden_size,
        "initializer_range": getattr(config, "initializer_range", 0.02),
        "intermediate_size": actual_intermediate,
        "layer_norm_eps": getattr(config, "layer_norm_eps", 1e-05),
        "max_position_embeddings": actual_max_pos,
        "model_type": "xlm-roberta",
        "num_attention_heads": actual_heads,
        "num_hidden_layers": actual_layers,
        "output_past": True,
        "pad_token_id": 1,
        "position_embedding_type": "absolute",
        "type_vocab_size": actual_type_vocab,
        "use_cache": True,
        # sẽ được override về 512 nếu bạn bật remap
        "vocab_size": actual_vocab_size
    }


def _strip_prefixes(state_dict, prefixes=("backbone.", "model.", "module.")):
    """Remove common prefixes from keys so they match HF naming."""
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def convert_backbone_to_hf_state_dict(backbone_state_dict):
    """Convert backbone state dict keys to HuggingFace format"""
    hf_state_dict = {}
    for key, value in backbone_state_dict.items():
        # Convert attention keys: .attention.query -> .attention.self.query (etc.)
        if ".attention.query." in key:
            new_key = key.replace(".attention.query.", ".attention.self.query.")
        elif ".attention.key." in key:
            new_key = key.replace(".attention.key.", ".attention.self.key.")
        elif ".attention.value." in key:
            new_key = key.replace(".attention.value.", ".attention.self.value.")
        elif ".attention.out_proj." in key:
            new_key = key.replace(".attention.out_proj.", ".attention.output.dense.")
        else:
            new_key = key
        hf_state_dict[new_key] = value
    return hf_state_dict


# -------------------- NEW: vocab remap helpers --------------------
def _build_pruned_mapping(pruned_repo, pruned_subfolder=None, base_repo="BAAI/bge-m3"):
    """
    Build index mapping from pruned tokenizer (new, small) -> base tokenizer (old, large)
    by token *text*. Returns (idx_tensor, new_vocab_size).
    """
    new_tok = AutoTokenizer.from_pretrained(pruned_repo, subfolder=pruned_subfolder)
    old_tok = AutoTokenizer.from_pretrained(base_repo)

    new_tokens = new_tok.convert_ids_to_tokens(list(range(new_tok.vocab_size)))
    old_tokens = old_tok.convert_ids_to_tokens(list(range(old_tok.vocab_size)))
    old_to_id = {t: i for i, t in enumerate(old_tokens)}

    unk = old_tok.unk_token_id if old_tok.unk_token_id is not None else 0
    idx = []
    missing = 0
    for t in new_tokens:
        j = old_to_id.get(t, unk)
        if j == unk and t not in old_to_id:
            missing += 1
        idx.append(j)

    if missing:
        print(f"[WARN] {missing} pruned tokens not found in base tokenizer; mapped to <unk>={unk}")
    return torch.tensor(idx, dtype=torch.long), new_tok.vocab_size


def _shrink_vocab_state_(state, idx, hidden_size_hint=None):
    """
    Cut & reorder any token-indexed 2D tensor by selecting rows with idx.
    Targets: word embeddings / embed_tokens / lm_head (if tied).
    """
    changed = 0
    for k in list(state.keys()):
        v = state[k]
        if isinstance(v, torch.Tensor) and v.ndim == 2:
            # Name-based filter for token matrices:
            if (k.endswith("embeddings.word_embeddings.weight")
                or k.endswith("embed_tokens.weight")
                or k.endswith("lm_head.weight")):
                # Optional: check hidden-size matches hint
                if hidden_size_hint is None or v.shape[1] == hidden_size_hint:
                    state[k] = v.index_select(0, idx)
                    changed += 1
                    if changed <= 8:
                        print(f"[DEBUG] remap {k}: {tuple(v.shape)} -> ({len(idx)}, {v.shape[1]})")
    print(f"[INFO] vocab shrink: remapped {changed} tensors to new size={len(idx)}")
    return state


def save_backbone_as_hf_model(
    backbone,
    save_path,
    base_model_name="BAAI/bge-m3",
    pruned_vocab_repo= None,          # e.g. "minhchuxuan/bge_pruned_298"
    pruned_vocab_subfolder= None, # e.g. "exp_high_hf"
    tokenizer_to_save=None,
):
    """
    Save (possibly pruned) backbone as a HuggingFace XLM-R model.
    If pruned_vocab_repo is provided, we:
      1) build mapping new(pruned)->old(base) by token text,
      2) cut/reorder token-indexed tensors (embeddings, lm_head),
      3) set config.vocab_size = len(pruned tokenizer),
      4) save pruned tokenizer alongside model.
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # 1) Build config from backbone, then override vocab_size if we will prune vocab.
    hf_config_dict = create_hf_config_from_backbone(backbone)

    # Create config from base model then patch in real dims
    config = AutoConfig.from_pretrained(base_model_name)
    for key, value in hf_config_dict.items():
        setattr(config, key, value)

    # 2) Prepare state dict (strip prefixes first)
    backbone_state = backbone.state_dict()
    backbone_state = _strip_prefixes(backbone_state)

    # 3) Optional: shrink vocab to pruned tokenizer
    idx = None
    if pruned_vocab_repo is not None:
        idx, new_vs = _build_pruned_mapping(
            pruned_repo=pruned_vocab_repo,
            pruned_subfolder=pruned_vocab_subfolder,
            base_repo=base_model_name
        )
        # Try to guess hidden size for extra safety (not mandatory)
        hidden = backbone.embeddings.word_embeddings.weight.shape[1]
        backbone_state = _shrink_vocab_state_(backbone_state, idx, hidden_size_hint=hidden)
        # make sure config matches pruned vocab
        config.vocab_size = new_vs
        print(f"[INFO] config.vocab_size set to pruned size = {new_vs}")

    # 4) Create fresh HF model and load converted weights
    hf_model = AutoModel.from_config(config)

    # Attention-key fix & any other minor renames
    hf_state_dict = convert_backbone_to_hf_state_dict(backbone_state)

    # 5) Load weights
    missing_keys, unexpected_keys = hf_model.load_state_dict(hf_state_dict, strict=False)
    if len(missing_keys) > 0:
        print(f"[WARN] {len(missing_keys)} missing keys (may be normal for pruned model)")
    if len(unexpected_keys) > 0:
        print(f"[WARN] {len(unexpected_keys)} unexpected keys (may be normal for custom backbone)")

    # 6) Save model and the right tokenizer
    hf_model.save_pretrained(save_path)
    if tokenizer_to_save is not None:
        # ưu tiên lưu đúng tokenizer hiện tại (đã sync vocab)
        tokenizer_to_save.save_pretrained(save_path)
    else:
        if pruned_vocab_repo is not None:
            tok = AutoTokenizer.from_pretrained(pruned_vocab_repo, subfolder=pruned_vocab_subfolder)
        else:
            tok = AutoTokenizer.from_pretrained(base_model_name)
        tok.save_pretrained(save_path)


    return save_path
