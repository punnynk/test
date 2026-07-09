# =============================================================================
# RoBERTa MLM Pre-training + GAL Fine-tuning Pipeline
# Phase 1: Continue pre-training บน unlabeled data (Masked Language Modeling)
# Phase 2: Fine-tune ด้วย GAL mechanism บน labeled data
#
# วิธีรัน:
#   python newGAL_model_conf_data.py --num_gpus 3 --end_line 53000 --phase 1
#   python newGAL_model_conf_data.py --num_gpus 3 --phase 2
# =============================================================================

# ===== IMPORTS =====
import os
import math
import json
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

@dataclass
class Phase1Config:
    """ตั้งค่าสำหรับ Phase 1: MLM Pre-training บน unlabeled data"""

    data_file:   str = "/share/ksl_pub/resource/Dataset/FumanDataset/FumanDataset/table-posts.csv"
    text_column: str = "text"
    start_line:  int = 3000             # จำนวนบรรทัดที่ข้าม
    end_line:    Optional[int] = None   # บรรทัดสุดท้ายทีใช้ประมวลผล

    pretrained_model_name: str   = "rinna/japanese-roberta-base"
    use_fast_tokenizer:    bool  = True
    max_seq_len:           int   = 256
    mlm_probability:       float = 0.15

    batch_size:    int   = 39
    num_epochs:    int   = 3
    learning_rate: float = 3e-4
    weight_decay:  float = 0.01
    warmup_ratio:  float = 0.1
    max_grad_norm: float = 1.0
    random_seed:   int   = 18

    output_dir:     str = "newGAL_V1c_P1_output"
    checkpoint_dir: str = "newGAL_V1c_P1_checkpoints"
    log_dir:        str = "runs"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    # num_gpus: None = ใช้ทุกการ์ดที่มี | N = ใช้ N การ์ดแรก | 1 = single GPU
    num_gpus: Optional[int] = None

    def __post_init__(self):
        os.makedirs(self.output_dir,     exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir,        exist_ok=True)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in self.__dict__.items()}, f, indent=2, default=str)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f: data = json.load(f)
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k): setattr(cfg, k, v)
        return cfg


@dataclass
class Phase2Config:
    """ตั้งค่าสำหรับ Phase 2: GAL Fine-tuning บน labeled data"""

    data_file:    str = "./sep_kosen_data.csv"
    text_column:  str = "fulltext"
    label_column: str = "label"
    # confident_level_column: กรองเฉพาะแถวที่ค่าเป็น 1 หรือ 2 (ตัด 0 ทิ้ง)
    confident_level_column: str = "confident_level"
    label_map: Dict[str, int] = field(
        default_factory=lambda: {
                                "emotional attack": 0, 
                                "emotional persuasion": 1, 
                                "rational persuasion": 2,
                                "sarcasm / irony": 3,
                                "indirect expression" : 4,
                                "no anger" : 5
                                }
    )
    start_line: int = 0

    # ── Model ─────────────────────────────────────────────────────────────────
    pretrained_dir:    str  = "newGAL_V1_P1_output"   # backbone จาก Phase 1
    use_fast_tokenizer: bool = True
    max_seq_len:        int  = 256
    num_classes:        int  = 6

    # ── GAL Hyperparameters ───────────────────────────────────────────────────
    # hidden_dim ต้องตรงกับ hidden size ของ backbone
    hidden_dim:          int   = 768
    num_attention_heads: int   = 8
    num_gal_layers:      int   = 2     # จำนวน layer ใน GAL Fusion
    lcf_mode:            str   = "cdm" # "cdm" / "cdw" / "both"
    dropout_rate:        float = 0.1

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:    int   = 33
    num_epochs:    int   = 8
    backbone_lr:   float = 1e-5   # fine-tune backbone
    gal_lr:        float = 2e-4   # GAL components + classifier (train from scratch)
    weight_decay:  float = 0.01
    warmup_ratio:  float = 0.1
    max_grad_norm: float = 1.0
    val_split:     float = 0.1
    test_split:    float = 0.1
    random_seed:   int   = 18

    output_dir:     str = "newGAL_V1c_P2_output_2"
    checkpoint_dir: str = "newGAL_V1c_P2_checkpoints_2"
    log_dir:        str = "runs"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    # num_gpus: None = ใช้ทุกการ์ดที่มี | N = ใช้ N การ์ดแรก | 1 = single GPU
    num_gpus: Optional[int] = None

    def __post_init__(self):
        os.makedirs(self.output_dir,     exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir,        exist_ok=True)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in self.__dict__.items()}, f, indent=2, default=str)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f: data = json.load(f)
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k): setattr(cfg, k, v)
        return cfg


# =============================================================================
# SECTION 2: DATASET
# =============================================================================

def _read_csv_from_line(filepath: str, start_line: int) -> pd.DataFrame:
    """อ่าน CSV โดยข้าม start_line แถวแรก (ไม่นับ header)"""
    if start_line > 0:
        return pd.read_csv(filepath, skiprows=range(1, start_line + 1))
    return pd.read_csv(filepath)


class UnlabeledDataset(Dataset):
    """Dataset Phase 1 — text เท่านั้น, ไม่มี label"""

    def __init__(self, file: str, tokenizer, config: Phase1Config):
        df = _read_csv_from_line(file, config.start_line)
        if config.text_column not in df.columns:
            raise ValueError(f"ไม่พบ column '{config.text_column}' | มี: {list(df.columns)}")

        # ── จำกัดช่วงข้อมูลด้วย end_line (ถ้าตั้งไว้) ──────────────────────────
        # df ตอนนี้เริ่มนับจากแถวที่ (start_line + 1) ของไฟล์ต้นฉบับแล้ว
        # จำนวนแถวที่ต้องการ = end_line - start_line
        if config.end_line is not None:
            row_limit = config.end_line - config.start_line
            if row_limit <= 0:
                raise ValueError(
                    f"end_line ({config.end_line}) ต้องมากกว่า start_line ({config.start_line})"
                )
            df = df.iloc[:row_limit]

        self.texts   = df[config.text_column].dropna().tolist()
        self.tokenizer = tokenizer
        self.max_len = config.max_seq_len
        print(f"  Loaded {len(self.texts):,} rows  "
              f"(start_line={config.start_line}, end_line={config.end_line})")

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


class LabeledDataset(Dataset):
    """
    Dataset Phase 2 — text + label
    สร้าง GAL masks จาก attention_mask
    (เนื่องจากไม่มี aspect term แยก ใช้ full text เป็น context ทั้งหมด)
    """

    def __init__(self, data: pd.DataFrame, tokenizer, config: Phase2Config):
        self.data      = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config    = config

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row  = self.data.iloc[idx]
        text = str(row[self.config.text_column])

        raw = str(row[self.config.label_column]).strip().lower()
        if raw in self.config.label_map:
            label = self.config.label_map[raw]
        elif raw.lstrip("-").isdigit():
            label = int(raw)
        else:
            raise KeyError(
                f"label '{raw}' ไม่อยู่ใน label_map {self.config.label_map} | "
                "แก้ไข: ปรับ label_map ให้ตรงกับค่าในไฟล์"
            )

        enc = self.tokenizer(
            text, max_length=self.config.max_seq_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # ── GAL masks: ใช้ attention_mask ทั้งหมด (full text = aspect) ──────
        aspect_mask    = attention_mask.float()
        lcf_cdm_mask   = attention_mask.float()
        lcf_cdw_weight = attention_mask.float()

        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "aspect_mask":     aspect_mask,
            "lcf_cdm_mask":    lcf_cdm_mask,
            "lcf_cdw_weight":  lcf_cdw_weight,
            "label":           torch.tensor(label, dtype=torch.long),
        }


def load_labeled_data(config: Phase2Config):
    df = _read_csv_from_line(config.data_file, config.start_line)
    for col in [config.text_column, config.label_column, config.confident_level_column]:
        if col not in df.columns:
            raise ValueError(f"ไม่พบ column '{col}' | มี: {list(df.columns)}")

    # กรองเฉพาะแถวที่ confident_level เป็น 1 หรือ 2 (ตัด 0 ทิ้ง)
    before = len(df)
    df = df[df[config.confident_level_column].isin([2])].reset_index(drop=True)
    print(f"  Filtered by confident_level [1,2]: {len(df):,} / {before:,} rows")

    total_val = config.val_split + config.test_split
    train_df, temp_df = train_test_split(df, test_size=total_val, random_state=config.random_seed)
    val_df, test_df   = train_test_split(temp_df, test_size=config.test_split / total_val,
                                         random_state=config.random_seed)
    return train_df, val_df, test_df


# =============================================================================
# SECTION 3: GAL MODEL COMPONENTS
# ── ใช้ร่วมกับ backbone จาก Phase 1
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention + residual + LayerNorm"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.q_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2).bool(), float("-inf"))
        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        out  = torch.matmul(self.dropout(attn), V)
        out  = self.out_proj(out.transpose(1, 2).contiguous().view(B, L, D))
        return self.norm(out + x)


class GlobalContextModule(nn.Module):
    """สกัด Global Context จาก full text"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)

    def forward(self, h, attention_mask):
        return self.mhsa(h, mask=attention_mask)


class AspectContextModule(nn.Module):
    """สกัด Aspect Context (ไม่มี aspect term → ใช้ full text)"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)

    def forward(self, h, aspect_mask):
        return self.mhsa(h, mask=aspect_mask)


class LocalContextFocusModule(nn.Module):
    """Local Context Focus — CDM / CDW / both"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1, mode="cdm"):
        super().__init__()
        assert mode in ("cdm", "cdw", "both")
        self.mode = mode
        self.mhsa = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)

    def forward(self, h, lcf_cdm_mask, lcf_cdw_weight, attention_mask):
        local_h = h
        if self.mode in ("cdm", "both"):
            local_h = local_h * lcf_cdm_mask.unsqueeze(-1)
        if self.mode in ("cdw", "both"):
            local_h = local_h * lcf_cdw_weight.unsqueeze(-1)
        combined = attention_mask * lcf_cdm_mask if self.mode in ("cdm", "both") else attention_mask
        return self.mhsa(local_h, mask=combined)


class GALFusionLayer(nn.Module):
    """รวม Global + Aspect + Local → Multi-Layer Attention"""
    def __init__(self, hidden_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim),
        )
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, global_out, aspect_out, local_out, attention_mask):
        fused = self.fusion_proj(torch.cat([global_out, aspect_out, local_out], dim=-1))
        for layer in self.attention_layers:
            fused = layer(fused, mask=attention_mask)
        return fused


class GALClassifier(nn.Module):
    """
    RoBERTa backbone (Phase 1) + GAL mechanism + Classification Head

    ┌──────────────────────────────────────────────────────────┐
    │  RoBERTa backbone (Phase 1)  ← fine-tune, backbone_lr   │
    │              ↓ hidden states                             │
    │   ┌──────────┼───────────┐                              │
    │   ↓          ↓           ↓                              │
    │  Global   Aspect      Local                             │
    │  Context  Context     Context (LCF)                     │
    │   └──────────┴───────────┘                              │
    │       ↓ GAL Fusion + Multi-Layer Attention              │
    │       ↓ [CLS] pooling                                   │
    │  Linear → num_classes  ← train from scratch, gal_lr    │
    └──────────────────────────────────────────────────────────┘
    """

    def __init__(self, backbone: nn.Module, config: Phase2Config):
        super().__init__()

        # ── Backbone จาก Phase 1 ──────────────────────────────────────────────
        self.backbone = backbone

        # ── GAL Components (train from scratch) ───────────────────────────────
        self.global_module = GlobalContextModule(
            config.hidden_dim, config.num_attention_heads, config.dropout_rate
        )
        self.aspect_module = AspectContextModule(
            config.hidden_dim, config.num_attention_heads, config.dropout_rate
        )
        self.local_module = LocalContextFocusModule(
            config.hidden_dim, config.num_attention_heads,
            config.dropout_rate, config.lcf_mode
        )
        self.gal_fusion = GALFusionLayer(
            config.hidden_dim, config.num_attention_heads,
            config.num_gal_layers, config.dropout_rate
        )

        # ── Classification Head ───────────────────────────────────────────────
        self.dropout    = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, input_ids, attention_mask,
                aspect_mask, lcf_cdm_mask, lcf_cdw_weight):
        # Step 1: Backbone encoding
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # [B, L, D]

        # Step 2: GAL three-branch extraction
        global_out = self.global_module(hidden, attention_mask)
        aspect_out = self.aspect_module(hidden, aspect_mask)
        local_out  = self.local_module(hidden, lcf_cdm_mask, lcf_cdw_weight, attention_mask)

        # Step 3: Fusion + Multi-layer attention
        fused = self.gal_fusion(global_out, aspect_out, local_out, attention_mask)

        # Step 4: Classification
        return self.classifier(self.dropout(fused[:, 0, :]))


# =============================================================================
# SECTION 4: TRAINING UTILITIES
# =============================================================================

def get_base_model(model):
    """คืน model จริง (unwrap จาก DataParallel ถ้าห่อไว้)"""
    return model.module if isinstance(model, nn.DataParallel) else model


def setup_multi_gpu(model: nn.Module, config, device: str) -> nn.Module:
    """
    ห่อโมเดลด้วย DataParallel ถ้ามีมากกว่า 1 GPU และ config อนุญาต
    config.num_gpus: None = ใช้ทุกการ์ดที่มี | N = ใช้ N การ์ดแรก | 1 = single GPU
    """
    if device != "cuda":
        print(f"  Using device: {device}")
        return model

    available = torch.cuda.device_count()
    use_n     = available if config.num_gpus is None else min(config.num_gpus, available)

    if use_n > 1:
        gpu_ids = list(range(use_n))
        model   = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"  Using {use_n} GPU(s): {gpu_ids}")
    else:
        print(f"  Using 1 GPU")

    return model


def build_phase1_optimizer(model, config: Phase1Config, num_steps: int):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(num_steps * config.warmup_ratio), num_steps
    )
    return optimizer, scheduler


def build_phase2_optimizer(model: GALClassifier, config: Phase2Config, num_steps: int):
    """
    LR แยก 2 กลุ่ม:
    - backbone   → backbone_lr (fine-tune)
    - GAL + head → gal_lr     (train from scratch)
    รองรับทั้งโมเดลปกติและที่ห่อด้วย DataParallel
    """
    base = get_base_model(model)
    backbone_ids = {id(p) for p in base.backbone.parameters()}
    gal_params   = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW([
        {"params": list(base.backbone.parameters()),
         "lr": config.backbone_lr, "weight_decay": config.weight_decay},
        {"params": gal_params,
         "lr": config.gal_lr,      "weight_decay": config.weight_decay},
    ])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(num_steps * config.warmup_ratio), num_steps
    )
    return optimizer, scheduler


def evaluate(model: GALClassifier, loader: DataLoader, device: str) -> dict:
    model.eval()
    criterion                         = nn.CrossEntropyLoss()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            amask = batch["aspect_mask"].to(device)
            cdm   = batch["lcf_cdm_mask"].to(device)
            cdw   = batch["lcf_cdw_weight"].to(device)
            lbl   = batch["label"].to(device)

            logits     = model(ids, mask, amask, cdm, cdw)
            total_loss += criterion(logits, lbl).item()
            all_preds.extend(torch.argmax(logits, -1).cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    return {
        "loss":       total_loss / len(loader),
        "accuracy":   accuracy_score(all_labels, all_preds),
        "f1_macro":   f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "all_preds":  all_preds,
        "all_labels": all_labels,
    }


def print_analysis(all_labels, all_preds, label_map, writer=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
    cm  = confusion_matrix(all_labels, all_preds, labels=list(range(len(label_names))))
    col_w = max(len(n) for n in label_names) + 2
    print(f"\n  Confusion Matrix  (row=actual, col=predicted)")
    print("  " + " " * (col_w + 2) + "  ".join(f"{n:>{col_w}}" for n in label_names))
    for i, row in enumerate(cm):
        print(f"  {label_names[i]:>{col_w}}  " + "  ".join(f"{v:>{col_w}}" for v in row))
    print(f"\n  Classification Report")
    for line in classification_report(all_labels, all_preds,
                                      labels=list(range(len(label_names))),
                                      target_names=label_names, zero_division=0).splitlines():
        print(f"  {line}")
 
    if writer:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues"); plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(label_names))); ax.set_xticklabels(label_names, rotation=30, ha="right")
        ax.set_yticks(range(len(label_names))); ax.set_yticklabels(label_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=12)
        plt.tight_layout(); writer.add_figure("Test/confusion_matrix", fig); plt.close(fig)
        print("  ✓ Confusion matrix → TensorBoard > IMAGES tab")


# =============================================================================
# SECTION 5: PHASE 1 — MLM PRE-TRAINING
# =============================================================================

def run_phase1(config: Phase1Config):
    torch.manual_seed(config.random_seed); np.random.seed(config.random_seed)
    device   = config.device
    run_name = f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer   = SummaryWriter(os.path.join(config.log_dir, run_name))

    print(f"\n{'='*60}")
    print(f"  Phase 1: MLM Pre-training  |  {run_name}")
    print(f"  Model : {config.pretrained_model_name}  |  Device: {device}")
    print(f"  Start line: {config.start_line}  |  MLM prob: {config.mlm_probability}")
    print(f"{'='*60}\n")

    print("[1/3] Loading data...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name, use_fast=config.use_fast_tokenizer
    )
    dataset  = UnlabeledDataset(config.data_file, tokenizer, config)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config.mlm_probability
    )
    loader   = DataLoader(dataset, batch_size=config.batch_size,
                          shuffle=True, collate_fn=collator, num_workers=0)

    print("[2/3] Loading model...")
    model   = AutoModelForMaskedLM.from_pretrained(config.pretrained_model_name).to(device)
    # ป้องกัน "index out of bounds" — บังคับให้ embedding matrix ตรงกับ vocab ของ tokenizer
    model.resize_token_embeddings(len(tokenizer))
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model = setup_multi_gpu(model, config, device)

    print("[3/3] Pre-training...\n")
    num_steps            = len(loader) * config.num_epochs
    optimizer, scheduler = build_phase1_optimizer(model, config, num_steps)
    global_step          = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train(); epoch_loss = 0.0
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl  = batch["labels"].to(device)
            optimizer.zero_grad()
            # .mean() จำเป็นเมื่อใช้ DataParallel เพราะ loss จะถูกคืนมาเป็น vector ตามจำนวน GPU
            loss = model(input_ids=ids, attention_mask=mask, labels=lbl).loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step(); scheduler.step()
            epoch_loss += loss.item(); global_step += 1
            if global_step % 50 == 0:
                writer.add_scalar("Phase1/loss_step", loss.item(), global_step)

        avg = epoch_loss / len(loader)
        ppl = math.exp(min(avg, 20))
        print(f"Epoch {epoch:>2}/{config.num_epochs}  Loss={avg:.4f}  Perplexity={ppl:.2f}")
        writer.add_scalar("Phase1/loss_epoch", avg, epoch)
        writer.add_scalar("Phase1/perplexity", ppl, epoch)

    print(f"\n  Saving backbone → {config.output_dir}")
    get_base_model(model).base_model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    config.save(os.path.join(config.output_dir, "phase1_config.json"))
    writer.close()
    print(f"  TensorBoard → tensorboard --logdir={config.log_dir}")
    print(f"\n  ✅  Phase 1 เสร็จ — ใช้ '--pretrained_dir {config.output_dir}' ใน Phase 2\n")


# =============================================================================
# SECTION 6: PHASE 2 — GAL FINE-TUNING
# =============================================================================

def run_phase2(config: Phase2Config):
    torch.manual_seed(config.random_seed); np.random.seed(config.random_seed)
    device   = config.device
    run_name = f"phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer   = SummaryWriter(os.path.join(config.log_dir, run_name))

    print(f"\n{'='*60}")
    print(f"  Phase 2: GAL Fine-tuning  |  {run_name}")
    print(f"  Backbone: {config.pretrained_dir}  |  Device: {device}")
    print(f"  GAL: heads={config.num_attention_heads}  layers={config.num_gal_layers}"
          f"  lcf={config.lcf_mode}")
    print(f"{'='*60}\n")

    print("[1/4] Loading data...")
    tokenizer             = AutoTokenizer.from_pretrained(
        config.pretrained_dir, use_fast=config.use_fast_tokenizer
    )
    train_df, val_df, test_df = load_labeled_data(config)
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}"
          f"  (start_line={config.start_line})")

    train_loader = DataLoader(LabeledDataset(train_df, tokenizer, config),
                              batch_size=config.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(LabeledDataset(val_df,   tokenizer, config),
                              batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(LabeledDataset(test_df,  tokenizer, config),
                              batch_size=config.batch_size, shuffle=False, num_workers=0)

    print("[2/4] Loading backbone + building GAL model...")
    backbone = AutoModel.from_pretrained(config.pretrained_dir)
    backbone.resize_token_embeddings(len(tokenizer))
    model    = GALClassifier(backbone, config).to(device)
    total_p  = sum(p.numel() for p in model.parameters())
    backbone_p = sum(p.numel() for p in model.backbone.parameters())
    print(f"  Total: {total_p:,}  |  Backbone: {backbone_p:,}"
          f"  |  GAL+Head: {total_p - backbone_p:,}")
    model = setup_multi_gpu(model, config, device)

    print("[3/4] Setting up optimizer...")
    num_steps            = len(train_loader) * config.num_epochs
    optimizer, scheduler = build_phase2_optimizer(model, config, num_steps)
    criterion            = nn.CrossEntropyLoss()

    print("[4/4] Fine-tuning...\n")
    best_val_acc = 0.0; global_step = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train(); epoch_loss = 0.0
        for batch in train_loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            amask = batch["aspect_mask"].to(device)
            cdm   = batch["lcf_cdm_mask"].to(device)
            cdw   = batch["lcf_cdw_weight"].to(device)
            lbl   = batch["label"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(ids, mask, amask, cdm, cdw), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step(); scheduler.step()

            epoch_loss += loss.item(); global_step += 1
            if global_step % 10 == 0:
                writer.add_scalar("Phase2/loss_step",    loss.item(),                     global_step)
                writer.add_scalar("Phase2/lr_backbone",  optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("Phase2/lr_gal",       optimizer.param_groups[1]["lr"], global_step)

        avg   = epoch_loss / len(train_loader)
        val_m = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:>2}/{config.num_epochs}"
              f"  TrainLoss={avg:.4f}  ValLoss={val_m['loss']:.4f}"
              f"  ValAcc={val_m['accuracy']:.4f}  ValF1={val_m['f1_macro']:.4f}")

        writer.add_scalars("Loss",        {"train": avg, "val": val_m["loss"]}, epoch)
        writer.add_scalar("Val/accuracy", val_m["accuracy"], epoch)
        writer.add_scalar("Val/f1_macro", val_m["f1_macro"], epoch)

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            torch.save(
                {"epoch": epoch, "model_state_dict": get_base_model(model).state_dict(),
                 "metrics": val_m},
                os.path.join(config.checkpoint_dir, "best.pt")
            )
            print(f"  ✓ Saved best (ValAcc={best_val_acc:.4f})")

    print(f"\n{'='*60}")
    print("  Final Test Evaluation")
    ckpt = torch.load(os.path.join(config.checkpoint_dir, "best.pt"), map_location=device)
    get_base_model(model).load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate(model, test_loader, device)
    print(f"  Test Accuracy : {test_m['accuracy']:.4f}")
    print(f"  Test F1 Macro : {test_m['f1_macro']:.4f}")
    print_analysis(test_m["all_labels"], test_m["all_preds"], config.label_map, writer)
    print(f"{'='*60}")

    writer.add_hparams(
        {"backbone_lr": config.backbone_lr, "gal_lr": config.gal_lr,
         "num_gal_layers": config.num_gal_layers, "lcf_mode": config.lcf_mode,
         "batch_size": config.batch_size},
        {"hparam/test_accuracy": test_m["accuracy"], "hparam/test_f1": test_m["f1_macro"]},
    )
    writer.close()
    config.save(os.path.join(config.output_dir, f"config_{run_name}.json"))
    print(f"\n  TensorBoard → tensorboard --logdir={config.log_dir}\n")
    return test_m


# =============================================================================
# SECTION 7: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoBERTa MLM Pre-training + GAL Fine-tuning")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])

    # shared
    parser.add_argument("--data_file",         default=None)
    parser.add_argument("--text_column",       default=None)
    parser.add_argument("--start_line",        type=int, default=None,
                        help="ข้ามกี่แถวจากต้น (ไม่นับ header)")
    parser.add_argument("--max_seq_len",       type=int, default=None)
    parser.add_argument("--batch_size",        type=int, default=None)
    parser.add_argument("--epochs",            type=int, default=None)
    parser.add_argument("--seed",              type=int, default=None)
    parser.add_argument("--num_gpus",          type=int, default=None,
                        help="จำนวน GPU ที่ใช้ — ไม่ใส่ = ใช้ทุกการ์ดที่มี, 1 = single GPU")
    parser.add_argument("--no_fast_tokenizer", action="store_true")

    # phase 1
    parser.add_argument("--pretrained",      default=None)
    parser.add_argument("--end_line",        type=int, default=None,
                        help="หยุดอ่านที่แถวไหน (ไม่นับ header) — เฉพาะ Phase 1")
    parser.add_argument("--mlm_probability", type=float, default=None)
    parser.add_argument("--learning_rate",   type=float, default=None)
    parser.add_argument("--output_dir",      default=None)
    parser.add_argument("--checkpoint_dir",  default=None)

    # phase 2
    parser.add_argument("--label_column",             default=None)
    parser.add_argument("--confident_level_column",   default=None)
    parser.add_argument("--pretrained_dir",    default=None)
    parser.add_argument("--num_classes",       type=int,   default=None)
    parser.add_argument("--hidden_dim",        type=int,   default=None)
    parser.add_argument("--num_heads",         type=int,   default=None)
    parser.add_argument("--num_gal_layers",    type=int,   default=None)
    parser.add_argument("--lcf_mode",          default=None, choices=["cdm", "cdw", "both"])
    parser.add_argument("--dropout",           type=float, default=None)
    parser.add_argument("--backbone_lr",       type=float, default=None)
    parser.add_argument("--gal_lr",            type=float, default=None)
    parser.add_argument("--p2_output_dir",     default=None)
    parser.add_argument("--p2_checkpoint_dir", default=None)

    args = parser.parse_args()

    if args.phase == 1:
        # mapping: CLI arg name → Phase1Config field name
        arg_to_field = {
            "data_file":      "data_file",
            "text_column":    "text_column",
            "start_line":     "start_line",
            "end_line":       "end_line",
            "pretrained":     "pretrained_model_name",
            "max_seq_len":    "max_seq_len",
            "mlm_probability":"mlm_probability",
            "batch_size":     "batch_size",
            "epochs":         "num_epochs",
            "learning_rate":  "learning_rate",
            "output_dir":     "output_dir",
            "checkpoint_dir": "checkpoint_dir",
            "seed":           "random_seed",
            "num_gpus":       "num_gpus",
        }
        overrides = {
            field_name: getattr(args, arg_name)
            for arg_name, field_name in arg_to_field.items()
            if getattr(args, arg_name) is not None
        }
        overrides["use_fast_tokenizer"] = not args.no_fast_tokenizer

        cfg = Phase1Config(**overrides)
        run_phase1(cfg)

    else:
        # mapping: CLI arg name → Phase2Config field name
        arg_to_field = {
            "data_file":      "data_file",
            "text_column":    "text_column",
            "label_column":           "label_column",
            "confident_level_column": "confident_level_column",
            "start_line":     "start_line",
            "pretrained_dir": "pretrained_dir",
            "max_seq_len":    "max_seq_len",
            "num_classes":    "num_classes",
            "hidden_dim":     "hidden_dim",
            "num_heads":      "num_attention_heads",
            "num_gal_layers": "num_gal_layers",
            "lcf_mode":       "lcf_mode",
            "dropout":        "dropout_rate",
            "batch_size":     "batch_size",
            "epochs":         "num_epochs",
            "backbone_lr":    "backbone_lr",
            "gal_lr":         "gal_lr",
            "p2_output_dir":     "output_dir",
            "p2_checkpoint_dir": "checkpoint_dir",
            "seed":           "random_seed",
            "num_gpus":       "num_gpus",
        }
        overrides = {
            field_name: getattr(args, arg_name)
            for arg_name, field_name in arg_to_field.items()
            if getattr(args, arg_name) is not None
        }
        overrides["use_fast_tokenizer"] = not args.no_fast_tokenizer

        cfg = Phase2Config(**overrides)
        run_phase2(cfg)