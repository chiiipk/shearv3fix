import os
import json
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models.base import ComposerModel
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from transformers import AutoModel, AutoConfig, AutoTokenizer

from .l0_module_embedding import L0ModuleEmbedding
from .embedding_heads import BGEEmbeddingHeads
from .bge_m3_backbone import MaskedBGEM3Backbone


# =========================
# [VOCAB_SYNC] Utilities
# =========================
def _safe_get_tokenizer_vocab_size(tokenizer) -> int:
    """Ưu tiên len(tokenizer) (bao gồm added tokens), fallback tokenizer.vocab_size."""
    size_by_len = None
    try:
        size_by_len = len(tokenizer)
    except Exception:
        pass
    size_by_attr = getattr(tokenizer, "vocab_size", None)
    candidates = [s for s in [size_by_len, size_by_attr] if isinstance(s, int) and s > 0]
    if not candidates:
        # fallback cuối cùng
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if callable(get_vocab):
            return len(get_vocab())
        raise ValueError("Cannot determine tokenizer vocab size.")
    return max(candidates)


class ComposerBGEM3(ComposerModel):
    """BGE-M3 model with L0 pruning and Composer interface"""

    def __init__(self, cfg):
        super().__init__()

        # === Base / config ===
        model_name = getattr(cfg, 'base_model', 'BAAI/bge-m3')
        self.base_model_name = model_name  # keep for HF export
        base_model = AutoModel.from_pretrained(model_name)
        self.config = base_model.config

        # === Optional overrides from cfg (keep consistent with math constraints) ===
        if hasattr(cfg, 'd_model'):
            self.config.hidden_size = cfg.d_model
        if hasattr(cfg, 'n_layers'):
            self.config.num_hidden_layers = cfg.n_layers
        if hasattr(cfg, 'n_heads'):
            self.config.num_attention_heads = cfg.n_heads

        # === Tokenizer (allow custom/pruned tokenizer) ===
        # [VOCAB_SYNC] Luôn lấy tokenizer trước khi khởi tạo backbone để đồng bộ vocab
        tok_name = getattr(cfg, 'tokenizer_name', model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name)
        target_vocab_size = _safe_get_tokenizer_vocab_size(self.tokenizer)

        # [VOCAB_SYNC] Đồng bộ vào config trước khi build backbone/heads
        self.config.vocab_size = int(target_vocab_size)

        # === Device name normalization ===
        device_name = get_device(None).name
        if device_name == "gpu":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"

        # === Backbone ===
        self.backbone = MaskedBGEM3Backbone(self.config)

        # === Nạp state_dict base_model (lọc các key lệch shape an toàn)
        base_sd = base_model.state_dict()
        own_sd = self.backbone.state_dict()
        # [VOCAB_SYNC] Bỏ qua các embedding không cùng shape (vì vocab đã đồng bộ theo tokenizer)
        skip_prefixes = {
            "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight",
        }
        for k in list(base_sd.keys()):
            if k in skip_prefixes:
                if k in own_sd and base_sd[k].shape != own_sd[k].shape:
                    # skip embedding weight không khớp vocab
                    del base_sd[k]
            elif k in own_sd and base_sd[k].shape != own_sd[k].shape:
                # Bất kỳ layer nào khác lệch shape (hiếm) -> skip
                # (tránh crash do khác cấu hình n_heads/intermediate đã override)
                del base_sd[k]

        self.backbone.load_state_dict(base_sd, strict=False)

        # === Heads ===
        self.embedding_heads = BGEEmbeddingHeads(self.config)
        # [VOCAB_SYNC] Đảm bảo heads biết vocab hiện tại
        if hasattr(self.embedding_heads, "set_vocab_size"):
            self.embedding_heads.set_vocab_size(self.config.vocab_size)
        else:
            # fallback nếu class cũ chưa có setter
            self.embedding_heads.vocab_size = self.config.vocab_size

        # === Loss/L0 flags ===
        self.l0_module = L0ModuleEmbedding(cfg, device_name, self.backbone)
        self.use_sts_loss = getattr(cfg, 'use_sts_loss', True)
        self.use_contrastive_loss = getattr(cfg, 'use_contrastive_loss', True)
        self.temperature = getattr(cfg, 'temperature', 0.02)

        # === Metrics ===
        self.train_metrics: Dict[str, Any] = {}
        self.eval_metrics: Dict[str, Any] = {}
        self.ref_model = None

        # === HF export options ===
        self.hf_export_opts = {}
        if hasattr(cfg, "hf_export"):
            self.hf_export_opts["pruned_vocab_repo"] = getattr(cfg.hf_export, "pruned_vocab_repo", None)
            self.hf_export_opts["pruned_vocab_subfolder"] = getattr(cfg.hf_export, "pruned_vocab_subfolder", None)

        # === Final config check ===
        self._validate_config()

    # -------------------- [VOCAB_SYNC] Helpers --------------------

    def _resize_backbone_token_embeddings(self, new_size: int):
        """
        Resize word embeddings in-place to `new_size` (copy phần chung, init phần dư = mean vector).
        Không phụ thuộc vào backbone triển khai riêng.
        """
        we: nn.Embedding = self.backbone.embeddings.word_embeddings
        old_size, dim = we.weight.size()
        if new_size == old_size:
            return
        device, dtype = we.weight.device, we.weight.dtype
        padding_idx = getattr(we, "padding_idx", None)

        new_we = nn.Embedding(new_size, dim, padding_idx=padding_idx, device=device, dtype=dtype)
        n = min(old_size, new_size)
        with torch.no_grad():
            new_we.weight[:n].copy_(we.weight[:n])
            if new_size > n:
                fill = we.weight.mean(dim=0, keepdim=True).to(dtype)
                new_we.weight[n:].copy_(fill)

        self.backbone.embeddings.word_embeddings = new_we
        # đồng bộ config
        self.backbone.config.vocab_size = int(new_size)
        self.config.vocab_size = int(new_size)

    def sync_vocab_everywhere(
        self,
        tokenizer: Optional[Any] = None,
        vocab_size: Optional[int] = None,
        resize_backbone: bool = True,
    ):
        """
        [VOCAB_SYNC] One-shot đồng bộ vocab cho config/backbone/heads.
        - Nếu `vocab_size` None -> lấy từ `tokenizer` (hoặc self.tokenizer).
        - Nếu `resize_backbone` True -> resize word embeddings để khớp.
        """
        if vocab_size is None:
            if tokenizer is None:
                tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None:
                raise ValueError("sync_vocab_everywhere: need tokenizer or vocab_size.")
            vocab_size = _safe_get_tokenizer_vocab_size(tokenizer)

        vocab_size = int(vocab_size)
        if resize_backbone:
            self._resize_backbone_token_embeddings(vocab_size)
        else:
            self.config.vocab_size = vocab_size
            self.backbone.config.vocab_size = vocab_size

        if hasattr(self.embedding_heads, "set_vocab_size"):
            self.embedding_heads.set_vocab_size(vocab_size)
        else:
            self.embedding_heads.vocab_size = vocab_size

    # -------------------- Forward / Loss --------------------

    def forward(self, batch):
        """Forward pass through the model"""
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)

        actual_batch_size = input_ids.size(0)

        # L0 masks
        l0_output = self.l0_module()

        # Backbone forward (masked)
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_z=l0_output.get('layer_z'),
            head_z=l0_output.get('head_z'),
            intermediate_z=l0_output.get('intermediate_z'),
        )

        # Heads: dense only for training efficiency
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
            'batch_size': actual_batch_size,
        }

    def eval_forward(self, batch, outputs=None):
        """Evaluation forward pass"""
        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs, batch):
        """Production loss computation with proper tensor handling"""
        embeddings = outputs['embeddings']
        l0_output = outputs['l0_output']

        # Interleaved pairs → batch_size
        batch_size = batch['input_ids'].size(0) // 2

        total_loss = 0.0

        if 'similarity_scores' in batch:
            # STS
            sts_loss = self.compute_sts_loss(embeddings, batch, batch_size)
            total_loss += sts_loss
        else:
            # Retrieval
            contrastive_loss = self.compute_contrastive_loss(embeddings, batch_size)
            total_loss += contrastive_loss

        # L0 sparsity + constraint (scaled)
        if hasattr(self.l0_module, 'get_sparsity_loss'):
            sparsity_loss, expected_sparsity, expected_score = self.l0_module.get_sparsity_loss()
            constraint_loss = self.compute_constraint_loss(expected_sparsity)
            pruning_weight = 20.0
            total_loss += pruning_weight * (sparsity_loss + constraint_loss)

        return total_loss

    def compute_sts_loss(self, embeddings: Dict[str, torch.Tensor], batch: Dict[str, Any], batch_size: int) -> torch.Tensor:
        dense_emb = embeddings['dense_embedding']  # [2B, D]
        similarity_scores = batch['similarity_scores']  # [B]

        # pairwise split
        sent1_emb = dense_emb[0::2]
        sent2_emb = dense_emb[1::2]

        # cosine -> [0,5]
        predicted_sim = F.cosine_similarity(sent1_emb, sent2_emb, dim=-1)
        predicted_sim = (predicted_sim + 1) * 2.5

        sts_loss = F.mse_loss(predicted_sim, similarity_scores)
        return sts_loss

    def compute_contrastive_loss(self, embeddings: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        dense_emb = embeddings['dense_embedding']  # [2B, D]

        query_emb = dense_emb[0::2]
        passage_emb = dense_emb[1::2]

        sim = torch.matmul(query_emb, passage_emb.t()) / self.temperature
        labels = torch.arange(batch_size, device=query_emb.device)
        return F.cross_entropy(sim, labels)

    def compute_constraint_loss(self, expected_sparsity: Dict[str, torch.Tensor]) -> torch.Tensor:
        """LLM-Shearing style constraint loss with warmup and quadratic penalty"""
        constraint_loss = 0.0
        for mask_name, sparsity in expected_sparsity.items():
            if mask_name in self.l0_module.masks:
                mask = self.l0_module.masks[mask_name]
                if hasattr(mask, 'target_sparsity') and mask.target_sparsity is not None:
                    current_target = self.l0_module.get_warmup_target_sparsity(mask.target_sparsity)
                    diff = sparsity.mean() - current_target
                    constraint_loss += torch.abs(diff) + 5.0 * (diff ** 2)
        return constraint_loss

    # -------------------- Pruning / info --------------------

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

    def compute_spearman_correlation(self, predicted_scores: torch.Tensor,
                                     ground_truth_scores: torch.Tensor) -> float:
        try:
            from scipy.stats import spearmanr
            pred_np = predicted_scores.detach().cpu().numpy()
            gt_np = ground_truth_scores.detach().cpu().numpy()
            correlation, _ = spearmanr(pred_np, gt_np)
            return float(correlation)
        except ImportError:
            pred_centered = predicted_scores - predicted_scores.mean()
            gt_centered = ground_truth_scores - ground_truth_scores.mean()
            correlation = (pred_centered * gt_centered).sum() / (
                torch.sqrt((pred_centered ** 2).sum() * (gt_centered ** 2).sum()) + 1e-8
            )
            return float(correlation)

    def extract_pruned_model(self) -> 'ComposerBGEM3':
        """Skeleton for making a physically smaller model; depends on your pruning impl."""
        zs = self.l0_module()
        pruned_config = self._create_pruned_config(zs)
        pruned_model = ComposerBGEM3(pruned_config)
        self._copy_pruned_weights(pruned_model, zs)
        return pruned_model

    def _create_pruned_config(self, zs: Dict[str, torch.Tensor]) -> DictConfig:
        # TODO: implement if you actually shrink hidden/intermediate/head dims
        pass

    def _copy_pruned_weights(self, target_model: 'ComposerBGEM3', zs: Dict[str, torch.Tensor]):
        # TODO: implement copying only surviving slices if you physically shrink tensors
        pass

    # -------------------- HF Export (UPDATED to support vocab remap) --------------------

    def save_pruned_hf_model(
        self,
        save_path: str,
        tokenizer_name: Optional[str] = None,
        pruned_vocab_repo: Optional[str] = None,
        pruned_vocab_subfolder: Optional[str] = None,
    ):
        """
        Apply current L0 masks, physically prune backbone params (via .prune_params),
        then export to 🤗 format. Supports pruned tokenizer remap.
        """
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Use the new hf_export that supports vocab remap
        from utils.hf_export import save_backbone_as_hf_model

        was_training = self.training
        self.eval()

        # 1) Gather deterministic masks
        zs = self.l0_module()
        print("\n🎯 Applying pruning masks...")
        for mask_name, mask_tensor in zs.items():
            sparsity = (mask_tensor == 0).float().mean().item()
            print(f"  {mask_name}: {sparsity:.1%} sparsity")

        # 2) Physically prune weights
        self.prune_params(zs)
        self._validate_pruned_model()

        # [VOCAB_SYNC] đảm bảo vocab hiện tại khớp tokenizer trước khi export
        tok_name = tokenizer_name or getattr(self, 'base_model_name', 'BAAI/bge-m3')
        tok = AutoTokenizer.from_pretrained(tok_name)
        self.sync_vocab_everywhere(tokenizer=tok, resize_backbone=False)

        # 3) Export → HF
        print(f"\n💾 Saving pruned model to {save_path}")
        base_model_name = tokenizer_name or getattr(self, 'base_model_name', 'BAAI/bge-m3')

        # Wire default pruned vocab from self.hf_export_opts if args not given
        pruned_repo = pruned_vocab_repo or self.hf_export_opts.get("pruned_vocab_repo")
        pruned_sub = pruned_vocab_subfolder or self.hf_export_opts.get("pruned_vocab_subfolder")

        save_backbone_as_hf_model(
            backbone=self.backbone,
            save_path=save_path,
            base_model_name=base_model_name,
            pruned_vocab_repo=pruned_repo,
            pruned_vocab_subfolder=pruned_sub,
        )

        # 4) Save pruning summary
        pruning_info = {
            'pruning_results': {name: float((mask == 0).float().mean()) for name, mask in zs.items()},
            'base_model': base_model_name,
            'final_config': {
                'num_hidden_layers': len(self.backbone.encoder.layer),
                'num_attention_heads': (
                    self.backbone.encoder.layer[0].attention.num_attention_heads
                    if len(self.backbone.encoder.layer) > 0 else 0
                ),
                'intermediate_size': (
                    self.backbone.encoder.layer[0].intermediate.dense.out_features
                    if len(self.backbone.encoder.layer) > 0 else 0
                ),
                'hidden_size': self.config.hidden_size,
                'vocab_size': self.config.vocab_size,
            },
            'tokenizer_remap': {
                'repo': pruned_repo,
                'subfolder': pruned_sub,
            }
        }
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'pruning_info.json'), 'w') as f:
            json.dump(pruning_info, f, indent=2)

        print("✅ Pruned model saved in HuggingFace format!")
        print(f"📁 Location: {save_path}")
        print(f"🔧 Usage: model = AutoModel.from_pretrained('{save_path}')")

        if was_training:
            self.train()
        return save_path

    def export_without_pruning(
        self,
        save_path: str,
        tokenizer_name: Optional[str] = None,
        pruned_vocab_repo: Optional[str] = None,
        pruned_vocab_subfolder: Optional[str] = None,
    ):
        """
        NEW: Export current (unpruned) backbone to 🤗 format, optionally with vocab remap.
        Useful for quick A/B eval before permanently pruning.
        """
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from utils.hf_export import save_backbone_as_hf_model

        was_training = self.training
        self.eval()

        base_model_name = tokenizer_name or getattr(self, 'base_model_name', 'BAAI/bge-m3')
        pruned_repo = pruned_vocab_repo or self.hf_export_opts.get("pruned_vocab_repo")
        pruned_sub = pruned_vocab_subfolder or self.hf_export_opts.get("pruned_vocab_subfolder")

        # [VOCAB_SYNC] bảo đảm vocab đồng bộ trước khi export
        tok = AutoTokenizer.from_pretrained(base_model_name)
        self.sync_vocab_everywhere(tokenizer=tok, resize_backbone=False)

        print(f"\n💾 Exporting (no pruning) to {save_path}")
        save_backbone_as_hf_model(
            backbone=self.backbone,
            save_path=save_path,
            base_model_name=base_model_name,
            pruned_vocab_repo=pruned_repo,
            pruned_vocab_subfolder=pruned_sub,
        )
        print("✅ Export (no pruning) completed.")

        if was_training:
            self.train()
        return save_path

    # -------------------- Validation helpers --------------------

    def _validate_config(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({self.config.hidden_size}) must be divisible by "
                f"number of attention heads ({self.config.num_attention_heads})."
            )

    def _validate_pruned_model(self):
        backbone_config = self.backbone.config
        actual_layers = len(self.backbone.encoder.layer)
        config_layers = backbone_config.num_hidden_layers
        if actual_layers != config_layers:
            raise ValueError(f"Layer count mismatch: actual={actual_layers}, config={config_layers}")

        if backbone_config.hidden_size % backbone_config.num_attention_heads != 0:
            raise ValueError(
                "Invalid attention configuration after pruning: "
                f"hidden_size={backbone_config.hidden_size}, "
                f"num_attention_heads={backbone_config.num_attention_heads}"
            )

        print(
            f"✅ Model validation passed: {actual_layers} layers, "
            f"{backbone_config.num_attention_heads} heads, "
            f"{backbone_config.intermediate_size} intermediate size, "
            f"{backbone_config.vocab_size} vocab"
        )
