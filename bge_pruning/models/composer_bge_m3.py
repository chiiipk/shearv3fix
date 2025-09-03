import os
import json
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models.base import ComposerModel
from composer.utils import get_device
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer

from .l0_module_embedding import L0ModuleEmbedding
from .embedding_heads import BGEEmbeddingHeads
from .bge_m3_backbone import MaskedBGEM3Backbone


# [VOCAB_SYNC] helper
def _safe_get_tokenizer_vocab_size(tok) -> int:
    try:
        return int(len(tok))
    except Exception:
        pass
    v = getattr(tok, "vocab_size", None)
    if isinstance(v, int) and v > 0:
        return v
    get_vocab = getattr(tok, "get_vocab", None)
    if callable(get_vocab):
        return len(get_vocab())
    raise ValueError("Cannot determine tokenizer vocab size.")


class ComposerBGEM3(ComposerModel):
    def __init__(self, cfg):
        super().__init__()

        model_name = getattr(cfg, 'base_model', 'BAAI/bge-m3')
        self.base_model_name = model_name
        base_model = AutoModel.from_pretrained(model_name)
        self.config = base_model.config

        # --- optional overrides ---
        if hasattr(cfg, 'd_model'):
            self.config.hidden_size = cfg.d_model
        if hasattr(cfg, 'n_layers'):
            self.config.num_hidden_layers = cfg.n_layers
        if hasattr(cfg, 'n_heads'):
            self.config.num_attention_heads = cfg.n_heads

        # [VOCAB_SYNC] luôn đồng bộ vocab từ tokenizer TRƯỚC khi tạo backbone
        tok_name = getattr(cfg, 'tokenizer_name', model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name)
        self.config.vocab_size = _safe_get_tokenizer_vocab_size(self.tokenizer)

        # --- build backbone với vocab đã sync ---
        self.backbone = MaskedBGEM3Backbone(self.config)

        # --- load state_dict, bỏ qua các key sai shape (đặc biệt embeddings) ---
        base_sd = base_model.state_dict()
        own_sd = self.backbone.state_dict()
        for k in list(base_sd.keys()):
            if k in own_sd and base_sd[k].shape != own_sd[k].shape:
                # ví dụ: embeddings.word_embeddings.weight [8194, 1024] vs [512, 1024]
                del base_sd[k]
        self.backbone.load_state_dict(base_sd, strict=False)

        # --- heads biết vocab hiện tại ---
        self.embedding_heads = BGEEmbeddingHeads(self.config)
        if hasattr(self.embedding_heads, "set_vocab_size"):
            self.embedding_heads.set_vocab_size(self.config.vocab_size)
        else:
            self.embedding_heads.vocab_size = self.config.vocab_size

        # device normalize
        device_name = get_device(None).name
        if device_name == "gpu":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"

        self._validate_config()
        self.l0_module = L0ModuleEmbedding(cfg, device_name, self.backbone)

        self.use_sts_loss = getattr(cfg, 'use_sts_loss', True)
        self.use_contrastive_loss = getattr(cfg, 'use_contrastive_loss', True)
        self.temperature = getattr(cfg, 'temperature', 0.02)

        self.train_metrics: Dict[str, Any] = {}
        self.eval_metrics: Dict[str, Any] = {}
        self.ref_model = None

    # [VOCAB_SYNC] dùng khi đổi tokenizer giữa chừng (không resize embedding vật lý)
    def sync_vocab_everywhere(self, vocab_size: Optional[int] = None):
        if vocab_size is None:
            vocab_size = _safe_get_tokenizer_vocab_size(self.tokenizer)
        vocab_size = int(vocab_size)
        self.config.vocab_size = vocab_size
        self.backbone.config.vocab_size = vocab_size
        if hasattr(self.embedding_heads, "set_vocab_size"):
            self.embedding_heads.set_vocab_size(vocab_size)
        else:
            self.embedding_heads.vocab_size = vocab_size

    # -------- forward/loss giữ nguyên như cũ --------
    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)

        l0_output = self.l0_module()
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_z=l0_output.get('layer_z'),
            head_z=l0_output.get('head_z'),
            intermediate_z=l0_output.get('intermediate_z'),
        )
        embedding_outputs = self.embedding_heads(
            hidden_states=backbone_outputs["last_hidden_state"],
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_sparse=False,
            return_multi_vector=False,
        )
        return {
            'embeddings': embedding_outputs,
            'backbone_outputs': backbone_outputs,
            'l0_output': l0_output,
            'batch_size': input_ids.size(0),
        }

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs, batch):
        embeddings = outputs['embeddings']
        batch_size = batch['input_ids'].size(0) // 2

        total = 0.0
        if 'similarity_scores' in batch:
            d = embeddings['dense_embedding']
            s = batch['similarity_scores']
            sim = F.cosine_similarity(d[0::2], d[1::2], dim=-1)
            sim = (sim + 1) * 2.5
            total += F.mse_loss(sim, s)
        else:
            q = embeddings['dense_embedding'][0::2]
            p = embeddings['dense_embedding'][1::2]
            logits = (q @ p.t()) / self.temperature
            labels = torch.arange(batch_size, device=q.device)
            total += F.cross_entropy(logits, labels)

        if hasattr(self.l0_module, 'get_sparsity_loss'):
            s_loss, exp_s, _ = self.l0_module.get_sparsity_loss()
            c_loss = self.compute_constraint_loss(exp_s)
            total += 20.0 * (s_loss + c_loss)
        return total

    def compute_constraint_loss(self, expected_sparsity: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = 0.0
        for name, sp in expected_sparsity.items():
            if name in self.l0_module.masks:
                m = self.l0_module.masks[name]
                if getattr(m, 'target_sparsity', None) is not None:
                    tgt = self.l0_module.get_warmup_target_sparsity(m.target_sparsity)
                    diff = sp.mean() - tgt
                    out += torch.abs(diff) + 5.0 * (diff ** 2)
        return out

    def get_metrics(self, is_train: bool = False) -> Dict[str, Any]:
        return self.train_metrics if is_train else self.eval_metrics

    def prune_params(self, zs: Optional[Dict[str, torch.Tensor]] = None):
        if zs is None:
            zs = self.l0_module()
        self.backbone.prune_params(zs)

    def get_model_info(self):
        return {
            'base_model_info': self.l0_module.base_model_info,
            'target_model_info': self.l0_module.target_model_info,
            'pruning_modules': self.l0_module.pruning_modules,
        }

    def _validate_config(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({self.config.hidden_size}) must be divisible by "
                f"number of attention heads ({self.config.num_attention_heads})."
            )
