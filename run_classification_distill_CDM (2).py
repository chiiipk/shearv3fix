"""
run_classification_distill_CDM.py
================================

This script is modelled after the ``run_clm_distill_CDM.py`` script from the
Contextual Dynamic Mapping (CDM) repository.  It retains a similar command
line interface and argument structure but is adapted for **classification
tasks** instead of causal language modelling.  The goal is to enable
cross‑tokenizer knowledge distillation from a decoder‑based teacher (e.g.
LLM2Vec‑Mistral‑7B‑SFT) to an encoder‑based student (e.g. BERT‑base) on
supervised classification datasets such as IMDb, Banking77 or STS.

The core ideas remain the same: we compute per‑token hidden states for both
teacher and student, align them using a weighted Dynamic Time Warping (DTW)
algorithm and minimise the mean squared error between aligned hidden states in
addition to the usual classification loss.  Much of the heavy engineering
(DeepSpeed, accelerate, distributed launch) has been removed for clarity, but
the overall code structure mirrors the original repo to ease migration of
hyperparameters and datasets.

Example usage::

    python run_classification_distill_CDM.py \
        --dataset_name imdb \
        --teacher_path McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised \
        --model_name_or_path bert-base-uncased \
        --num_train_epochs 3 \
        --output_dir ./checkpoints/llm2vec_to_bert_imdb

This will fine‑tune a BERT‑base model on the IMDb sentiment dataset, using
LLM2Vec‑Mistral as a teacher for cross‑tokenizer distillation.  The dataset
is loaded via ``datasets.load_dataset`` and any necessary label mappings
are inferred automatically.

Note: This script does not implement the ``--teacher_to_student_id_mapping``
argument used in the original CDM repository, because the weighted DTW
alignment obviates the need for a static vocabulary map.  If you wish to
experiment with exact or approximate vocabulary mappings, you can extend the
``dtw`` function accordingly.

"""

import dataclasses
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

logger = logging.getLogger(__name__)


def calculate_weight(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy‑based weights for each token.

    See the original CDM implementation for details.  The weights are
    integers in the range [1,3] derived from the entropy of the teacher's
    predicted distribution for each token.  Higher entropy implies a
    larger weight, allowing more flexibility in the DTW alignment.
    """
    probs = torch.softmax(logits, dim=-1) + 1e-10
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    entropy_min = entropy.min()
    entropy_max = entropy.max()
    normalised = (entropy - entropy_min) / (entropy_max - entropy_min + 1e-9)
    factor = torch.sigmoid(normalised * 4 - 2)
    weights = (factor * 3 + 1).round().clamp(1, 3)
    return weights.to(torch.int64)


def dtw(
    series1: List[torch.Tensor],
    series2: List[torch.Tensor],
    weights1: List[int],
    weights2: List[int],
) -> Tuple[List[List[int]], List[List[int]]]:
    """Weighted dynamic time warping.

    Returns mappings from indices in ``series1`` (student) to indices in
    ``series2`` (teacher) and vice versa.  This simplified implementation
    mirrors the function used in the original CDM repo【374232827969353†L845-L903】.
    """
    n = len(series1)
    m = len(series2)
    matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    matrix[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(series1[i - 1] - series2[j - 1]) * weights1[i - 1] * weights2[j - 1]
            matrix[i, j] = cost + min(matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1])
    i, j = n, m
    map1: List[List[int]] = [[] for _ in range(n)]
    map2: List[List[int]] = [[] for _ in range(m)]
    while i > 0 or j > 0:
        map1[i - 1].append(j - 1)
        map2[j - 1].append(i - 1)
        diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        up = matrix[i - 1, j] if i > 0 else np.inf
        left = matrix[i, j - 1] if j > 0 else np.inf
        if diag <= up and diag <= left:
            i -= 1
            j -= 1
        elif up < left:
            i -= 1
        else:
            j -= 1
    for lst in map1:
        lst.reverse()
    for lst in map2:
        lst.reverse()
    return map1, map2


@dataclass
class ModelArguments:
    """Arguments for loading models.

    ``model_name_or_path``: path or identifier of the student model (BERT‑base).
    ``teacher_path``: path or identifier of the teacher model (LLM2Vec).
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pre-trained student model or model identifier."}
    )
    teacher_path: str = field(
        metadata={"help": "Path to pre-trained teacher model."}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use a fast tokenizer."}
    )


@dataclass
class DataTrainingArguments:
    """Arguments for the dataset and preprocessing."""

    dataset_name: str = field(
        metadata={"help": "Name of the HuggingFace dataset to load."}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum input sequence length after tokenization."}
    )


@dataclass
class DistillationArguments:
    """Arguments specific to distillation."""

    kd_alpha: float = field(
        default=0.5,
        metadata={"help": "Weight for the distillation loss (0 => no distillation)."}
    )
    # Additional flags (topk, temperature) could be added here for parity with the
    # original CDM script.


class ClassificationDistillationModel(nn.Module):
    """Wrapper around a student model with classification head and distillation logic."""

    def __init__(self, student_model: AutoModel, num_labels: int, kd_alpha: float):
        super().__init__()
        self.student = student_model
        hidden_size = student_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.kd_alpha = kd_alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_distill_loss(
        self,
        student_hidden: torch.Tensor,
        student_mask: torch.Tensor,
        teacher_hidden_list: List[List[torch.Tensor]],
        teacher_weights_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute mean squared error between student and teacher hidden states using DTW mapping."""
        batch_size = student_hidden.size(0)
        distill_losses = []
        for b in range(batch_size):
            seq_len = student_mask[b].sum().item()
            stu_vecs = student_hidden[b, :seq_len].detach()
            tea_vecs = teacher_hidden_list[b]
            tea_weights = teacher_weights_list[b]
            stu_weights = torch.ones(seq_len, dtype=torch.int64)
            map_stu, map_tea = dtw(
                [v.cpu().numpy() for v in stu_vecs],
                [v.cpu().numpy() for v in tea_vecs],
                stu_weights.tolist(),
                tea_weights.cpu().tolist(),
            )
            loss_sum = 0.0
            count = 0
            for i, js in enumerate(map_stu):
                v_s = stu_vecs[i]
                for j in js:
                    v_t = tea_vecs[j]
                    loss_sum += torch.mean((v_s - v_t) ** 2)
                    count += 1
            distill_losses.append(loss_sum / max(count, 1))
        return torch.mean(torch.stack(distill_losses))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        teacher_hidden_list: List[List[torch.Tensor]],
        teacher_weights_list: List[torch.Tensor],
    ) -> Dict[str, Any]:
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state  # (b, s, h)
        # pool by mean
        mask_exp = attention_mask.unsqueeze(-1)
        pooled = torch.sum(hidden_states * mask_exp, dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1e-9)
        logits = self.classifier(pooled)
        loss_cls = self.ce_loss(logits, labels)
        loss_distill = self.compute_distill_loss(hidden_states, attention_mask, teacher_hidden_list, teacher_weights_list)
        loss = (1 - self.kd_alpha) * loss_cls + self.kd_alpha * loss_distill
        return {"loss": loss, "logits": logits, "loss_cls": loss_cls.detach(), "loss_distill": loss_distill.detach()}


def prepare_teacher_features(
    sentences: List[str],
    teacher_tokenizer: AutoTokenizer,
    teacher_model: AutoModel,
    device: torch.device,
) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor]]:
    """Prepare teacher hidden states and weights for a batch of sentences."""
    enc = teacher_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = teacher_model(**enc, output_hidden_states=True, return_dict=True)
    last_hidden = outputs.last_hidden_state
    logits = getattr(outputs, "logits", None)
    teacher_hidden_list: List[List[torch.Tensor]] = []
    teacher_weights_list: List[torch.Tensor] = []
    for b in range(last_hidden.size(0)):
        seq_len = enc["attention_mask"][b].sum().item()
        vecs = last_hidden[b, :seq_len].cpu().split(1, dim=0)
        teacher_hidden_list.append([v.squeeze(0) for v in vecs])
        if logits is not None:
            token_logits = logits[b, :seq_len]
            weights = calculate_weight(token_logits)
        else:
            weights = torch.ones(seq_len, dtype=torch.int64)
        teacher_weights_list.append(weights)
    return teacher_hidden_list, teacher_weights_list


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, DistillationArguments))
    model_args, data_args, training_args, kd_args = parser.parse_args_into_dataclasses()
    # Load dataset
    raw_dataset = load_dataset(data_args.dataset_name)
    # Determine label field and number of classes
    if "label" in raw_dataset["train"].features:
        num_labels = raw_dataset["train"].features["label"].num_classes
        label_field = "label"
    elif "labels" in raw_dataset["train"].features:
        num_labels = raw_dataset["train"].features["labels"].num_classes
        label_field = "labels"
    else:
        raise ValueError("Cannot determine label field in dataset")
    # Load tokenizers and models
    student_tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer)
    student_config = AutoConfig.from_pretrained(model_args.model_name_or_path, output_hidden_states=True)
    student_model = AutoModel.from_pretrained(model_args.model_name_or_path, config=student_config)
    # Teacher
    teacher_tokenizer = AutoTokenizer.from_pretrained(model_args.teacher_path, trust_remote_code=True)
    teacher_config = AutoConfig.from_pretrained(model_args.teacher_path, trust_remote_code=True)
    # When training with large teachers (e.g. 7B parameters) on multiple GPUs, we can
    # automatically shard the model across available devices via ``device_map='auto'``.
    # We also request half precision to reduce memory.  If only a single GPU is
    # available, the model will reside entirely on that device.
    device_map = None
    if torch.cuda.device_count() > 1:
        # Let transformers infer an automatic device map across GPUs
        device_map = "auto"
    teacher_model = AutoModel.from_pretrained(
        model_args.teacher_path,
        config=teacher_config,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    # Do not call `.to(device)` when using device_map; the weights are already
    # placed on the correct devices.  Set to evaluation mode.
    teacher_model.eval()
    # Determine a device to host the teacher inputs.  When using model
    # parallelism (device_map='auto'), inputs are typically placed on the
    # first CUDA device.  If CUDA is unavailable, fall back to CPU.
    if torch.cuda.is_available():
        teacher_device = torch.device("cuda:0")
    else:
        teacher_device = torch.device("cpu")
    # Preprocess dataset
    def preprocess_examples(examples):
        # Use "text" if present, otherwise try other common fields
        if "text" in examples:
            texts = examples["text"]
        elif "sentence" in examples:
            texts = examples["sentence"]
        elif "content" in examples:
            texts = examples["content"]
        else:
            raise ValueError("Cannot find a text field in the dataset")
        enc = student_tokenizer(texts, truncation=True, max_length=data_args.max_length)
        labels = examples[label_field]
        return {**enc, "labels": labels, "texts": texts}
    tokenized_datasets = raw_dataset.map(preprocess_examples, batched=True)
    # Create student model with classification/distillation wrapper
    distill_model = ClassificationDistillationModel(student_model, num_labels, kd_args.kd_alpha)
    distill_model.to(training_args.device)
    # Custom data collator to include raw text for teacher features
    def collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = default_data_collator(features)
        batch["texts"] = [f["texts"] for f in features]
        return batch
    # Define compute_loss via Trainer subclass
    class CDMTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            texts = inputs.pop("texts")
            # Prepare teacher features on the fly
            teacher_hiddens, teacher_weights = prepare_teacher_features(texts, teacher_tokenizer, teacher_model, teacher_device)
            outputs = model(
                input_ids=inputs["input_ids"].to(training_args.device),
                attention_mask=inputs["attention_mask"].to(training_args.device),
                labels=labels.to(training_args.device),
                teacher_hidden_list=teacher_hiddens,
                teacher_weights_list=teacher_weights,
            )
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss
    trainer = CDMTrainer(
        model=distill_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("test", tokenized_datasets.get("validation")),
        tokenizer=student_tokenizer,
        data_collator=collator,
    )
    # Train and evaluate
    train_result = trainer.train()
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()