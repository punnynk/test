# =============================================================================
# GAL: Global-Aspect-Local Mechanism for Aspect-Based Sentiment Classification
# Paper: "GAL: A global aspect local extraction mechanism for aspect-based
#         sentiment classification" — Information Sciences (2025)
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
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

FINAL_OUTPUT_NAME = "gal_best.pt"

# =============================================================================
# SECTION 1: CONFIGURATION
# ── รวม hyperparameter และการตั้งค่าทั้งหมดไว้ที่เดียว
# =============================================================================

@dataclass
class GALConfig:
    """
    คลาสตั้งค่าหลักสำหรับโมเดล GAL
    แก้ไขค่าในนี้เพื่อปรับ hyperparameter หรือ path ข้อมูล
    """

    # ── ที่อยู่ไฟล์ข้อมูล ────────────────────────────────────────────────────
    data_file: str = "data/dataset.csv"   # ไฟล์ CSV เดียวที่มีทั้ง text และ labels

    # ── ชื่อ column (ปรับตามไฟล์ของคุณ) ─────────────────────────────────────
    text_column:  str = "text"            # column ของ input text
    label_column: str = "labels"          # column ของ sentiment label

    # ── label mapping ─────────────────────────────────────────────────────────
    # ปรับ key ให้ตรงกับค่าใน CSV, value คือ index class
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

    # ── Pretrained Model ──────────────────────────────────────────────────────
    # ใส่ชื่อโมเดลจาก HuggingFace Hub หรือ path ไปยังโมเดล local
    pretrained_model_name: str = "rinna/japanese-roberta-base"
    # use_fast_tokenizer: False สำหรับโมเดลที่ใช้ SentencePiece (เช่น rinna, mBART, XLM-R)
    use_fast_tokenizer:    bool = True
    max_seq_len:           int = 512

    # ── GAL Architecture ──────────────────────────────────────────────────────
    hidden_dim:          int   = 768   # ต้องตรงกับ hidden size ของ pretrained model
    num_attention_heads: int   = 8     # จำนวน heads ใน MHSA
    num_gal_layers:      int   = 2     # จำนวน attention layers ใน GAL Fusion
    dropout_rate:        float = 0.1
    # lcf_mode: "cdm" = hard mask | "cdw" = soft weight | "both" = ใช้ทั้งคู่
    lcf_mode:            str   = "cdw"
    num_classes:         int   = 6     # จำนวน class (positive/negative/neutral)

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:    int   = 16
    num_epochs:    int   = 10
    # LR แยกกัน: pretrained_lr ต่ำ (fine-tune) / gal_lr สูงกว่า (train from scratch)
    pretrained_lr: float = 2e-5
    gal_lr:        float = 1e-4
    weight_decay:  float = 0.01
    warmup_ratio:  float = 0.1      # สัดส่วน warm-up steps ต่อ total steps
    max_grad_norm: float = 1.0      # gradient clipping

    val_split:     float = 0.1
    test_split:    float = 0.1
    random_seed:   int   = 42

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir:     str  = "outputs"
    checkpoint_dir: str  = "checkpoints"
    log_dir:        str  = "runs"
    save_best_only: bool = True

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    def __post_init__(self):
        os.makedirs(self.output_dir,     exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir,        exist_ok=True)

    def save(self, path: str):
        """บันทึก config เป็น JSON"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in self.__dict__.items()}, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "GALConfig":
        """โหลด config จาก JSON"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# =============================================================================
# SECTION 2: DATASET & DATA LOADER
# ── โหลด CSV ไฟล์เดียว และเตรียม tokenized inputs
# =============================================================================

class SentimentDataset(Dataset):
    """
    Dataset สำหรับ Sentiment Classification

    รับ DataFrame (text + labels) และแปลงเป็น tensor:
    - input_ids, attention_mask, token_type_ids : สำหรับ pretrained model
    - aspect_mask    : เท่ากับ attention_mask (ใช้ text ทั้งหมดเป็น context)
    - lcf_cdm_mask   : เท่ากับ attention_mask (ไม่มี SRD filtering)
    - lcf_cdw_weight : เท่ากับ attention_mask (น้ำหนักสม่ำเสมอ)
    - label          : class index

    หมายเหตุ: เนื่องจากไม่มี aspect term แยก GAL mechanism จะใช้ full text
    เป็น "aspect" ทำให้ Global / Aspect / Local branches ทำงานบน sequence เดียวกัน
    แต่ด้วย attention pattern ที่ต่างกัน (ผ่าน MHSA weights)
    """

    def __init__(self, data: pd.DataFrame, tokenizer, config: GALConfig):
        self.data      = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config    = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row   = self.data.iloc[idx]
        text  = str(row[self.config.text_column])
        raw_label = str(row[self.config.label_column]).strip().lower()
        # รองรับทั้ง string (positive) และ numeric (0, 1, 2)
        if raw_label in self.config.label_map:
            label = self.config.label_map[raw_label]
        elif raw_label.lstrip("-").isdigit():
            label = int(raw_label)
        else:
            raise KeyError(
                f"label '{raw_label}' ไม่อยู่ใน label_map {self.config.label_map} | "
                "แก้ไข: ปรับ label_map ให้ตรงกับค่าในไฟล์ หรือใช้ตัวเลข 0,1,2 โดยตรง"
            )

        # ── Tokenize text ──────────────────────────────────────────────────────
        enc = self.tokenizer(
            text,
            max_length=self.config.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        token_type_ids = enc.get("token_type_ids",
                                  torch.zeros_like(input_ids)).squeeze(0)

        # ── GAL Masks: ใช้ attention_mask ทั้งหมด (full text = aspect) ─────────
        # aspect_mask   = ทุก valid token เป็น "aspect"
        # lcf_cdm_mask  = ไม่ mask ใดๆ (SRD = 0 ทุก position)
        # lcf_cdw_weight = น้ำหนักสม่ำเสมอ = 1.0 ทุก valid token
        aspect_mask    = attention_mask.float()
        lcf_cdm_mask   = attention_mask.float()
        lcf_cdw_weight = attention_mask.float()

        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "token_type_ids":  token_type_ids,
            "aspect_mask":     aspect_mask,
            "lcf_cdm_mask":    lcf_cdm_mask,
            "lcf_cdw_weight":  lcf_cdw_weight,
            "label":           torch.tensor(label, dtype=torch.long),
        }


def load_and_split_data(config: GALConfig):
    """
    โหลด CSV ไฟล์เดียว แล้ว split เป็น train / val / test

    Returns:
        train_df, val_df, test_df (pandas DataFrame)
    """
    df = pd.read_csv(config.data_file)

    # ตรวจสอบ column ที่จำเป็น
    for col in [config.text_column, config.label_column]:
        if col not in df.columns:
            raise ValueError(
                f"ไม่พบ column '{col}' | columns ที่มี: {list(df.columns)}\n"
                "กรุณาแก้ไข config.text_column หรือ config.label_column"
            )

    total_val = config.val_split + config.test_split
    train_df, temp_df = train_test_split(
        df, test_size=total_val, random_state=config.random_seed, shuffle=True
    )
    relative_test = config.test_split / total_val
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, random_state=config.random_seed
    )
    return train_df, val_df, test_df


# =============================================================================
# SECTION 3: MODEL ARCHITECTURE — GAL Mechanism Components
# =============================================================================

# ── 3.1  Multi-Head Self-Attention (MHSA) ────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention พร้อม residual + LayerNorm
    ใช้ใน Global, Aspect, Local branches และ GAL Fusion layers
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) ต้องหารด้วย num_heads ({num_heads}) ลงตัว"

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
        """
        x    : [B, L, D]
        mask : [B, L] — 1 คือ valid, 0 คือ masked
        return: [B, L, D]
        """
        B, L, D = x.shape
        residual = x

        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, h, L, L]

        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2).bool()
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)   # กัน NaN กรณี mask ทั้งแถว
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return self.norm(out + residual)


# ── 3.2  Global Context Module ───────────────────────────────────────────────
class GlobalContextModule(nn.Module):
    """
    สกัด Global Context จาก text ทั้งหมด
    ใช้ MHSA บน hidden states ทุก token
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        return self.mhsa(hidden_states, mask=attention_mask)


# ── 3.3  Aspect Context Module ───────────────────────────────────────────────
class AspectContextModule(nn.Module):
    """
    สกัด Aspect Context โดยจำกัด attention เฉพาะ aspect tokens
    (เมื่อไม่มี aspect แยก จะใช้ full text เป็น aspect)
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)

    def forward(self, hidden_states: torch.Tensor,
                aspect_mask: torch.Tensor) -> torch.Tensor:
        return self.mhsa(hidden_states, mask=aspect_mask)


# ── 3.4  Local Context Focus (LCF) Module ────────────────────────────────────
class LocalContextFocusModule(nn.Module):
    """
    Local Context Focus (LCF) mechanism
    เน้น attention เฉพาะ token ที่อยู่ใกล้ aspect ตาม SRD threshold

    mode="cdm"  : Context Dynamic Mask — hard mask token ไกล
    mode="cdw"  : Context Dynamic Weight — soft weight ลดตามระยะห่าง
    mode="both" : CDM ก่อน แล้วตามด้วย CDW
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 dropout: float = 0.1, mode: str = "cdm"):
        super().__init__()
        assert mode in ("cdm", "cdw", "both")
        self.mode = mode
        self.mhsa = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)

    def forward(self, hidden_states: torch.Tensor,
                lcf_cdm_mask: torch.Tensor,
                lcf_cdw_weight: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        local_h = hidden_states

        if self.mode in ("cdm", "both"):
            local_h = local_h * lcf_cdm_mask.unsqueeze(-1)

        if self.mode in ("cdw", "both"):
            local_h = local_h * lcf_cdw_weight.unsqueeze(-1)

        combined_mask = attention_mask * lcf_cdm_mask if self.mode in ("cdm", "both") \
                        else attention_mask
        return self.mhsa(local_h, mask=combined_mask)


# ── 3.5  GAL Fusion + Multi-Layer Attention ──────────────────────────────────
class GALFusionLayer(nn.Module):
    """
    รวม Global + Aspect + Local features เข้าด้วยกัน
    แล้วผ่าน multi-layer attention เพื่อสกัด feature สุดท้าย

    Fusion: concat [G; A; L] → Linear (3D→D) → GELU → LayerNorm
    Multi-layer: stack MHSA layers ตาม num_layers
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, global_out: torch.Tensor, aspect_out: torch.Tensor,
                local_out: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([global_out, aspect_out, local_out], dim=-1)
        fused = self.fusion_proj(fused)
        out = fused
        for layer in self.attention_layers:
            out = layer(out, mask=attention_mask)
        return out


# ── 3.6  Complete GAL Model ───────────────────────────────────────────────────
class GALModel(nn.Module):
    """
    โมเดล GAL สมบูรณ์สำหรับ Sentiment Classification

    Architecture:
    ┌────────────────────────────────────────────────────────────────┐
    │         Input: [CLS] text [SEP]                                │
    │                    ↓                                           │
    │         ┌──────────────────┐                                   │
    │         │  Pretrained LM   │  ← ← ใส่โมเดลที่นี่                    │
    │         └────────┬─────────┘                                   │
    │            hidden states [B, L, D]                             │
    │         ┌────────┴──────────┐──────────────┐                   │
    │         ↓                   ↓              ↓                   │
    │    Global Context     Aspect Context   Local Context           │
    │    (MHSA ทั้ง text)   (MHSA full text)   (LCF: CDM/CDW)          │
    │         ↓                   ↓              ↓                   │
    │         └───────────────────┴──────────────┘                   │
    │                      ↓ concat [G;A;L]                          │
    │               GAL Fusion Projection                            │
    │                      ↓                                         │
    │            Multi-Layer Attention                               │
    │                      ↓                                         │
    │              [CLS] token pooling                               │
    │                      ↓                                         │
    │             Linear → num_classes                               │
    └────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, pretrained_model: nn.Module, config: GALConfig):
        """
        Args:
            pretrained_model : HuggingFace AutoModel ใด ๆ
                               ต้อง output .last_hidden_state [B, L, hidden_dim]
            config           : GALConfig
        """
        super().__init__()

        # ── [PLACEHOLDER] Pretrained Language Model ───────────────────────────
        # ใส่โมเดล pretrained ที่ต้องการผ่าน argument pretrained_model
        # ตัวอย่าง:
        #   AutoModel.from_pretrained("bert-base-uncased")
        #   AutoModel.from_pretrained("roberta-base")
        #   AutoModel.from_pretrained("./my_local_model")
        self.pretrained = pretrained_model

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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor, aspect_mask: torch.Tensor,
                lcf_cdm_mask: torch.Tensor, lcf_cdw_weight: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, num_classes]"""

        # ── Step 1: Encode ด้วย pretrained model ────────────────────────────
        lm_out = self.pretrained(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = lm_out.last_hidden_state   # [B, L, D]

        # ── Step 2: GAL three-branch extraction ─────────────────────────────
        global_out = self.global_module(hidden, attention_mask)
        aspect_out = self.aspect_module(hidden, aspect_mask)
        local_out  = self.local_module(hidden, lcf_cdm_mask, lcf_cdw_weight, attention_mask)

        # ── Step 3: Fusion + Multi-layer attention ──────────────────────────
        fused = self.gal_fusion(global_out, aspect_out, local_out, attention_mask)

        # ── Step 4: Classification ───────────────────────────────────────────
        cls = self.dropout(fused[:, 0, :])   # ใช้ [CLS] token
        return self.classifier(cls)          # [B, num_classes]


# =============================================================================
# SECTION 4: TRAINING UTILITIES
# ── Optimizer, Metrics, Checkpoint
# =============================================================================

def build_optimizer_and_scheduler(model: GALModel, config: GALConfig,
                                   num_training_steps: int):
    """
    AdamW optimizer ที่มี LR แยกกัน:
    - pretrained model → config.pretrained_lr (fine-tune)
    - GAL components   → config.gal_lr        (train from scratch)
    พร้อม linear scheduler ที่มี warmup
    """
    pretrained_ids = {id(p) for p in model.pretrained.parameters()}
    gal_params     = [p for p in model.parameters() if id(p) not in pretrained_ids]

    optimizer = torch.optim.AdamW([
        {"params": list(model.pretrained.parameters()),
         "lr": config.pretrained_lr, "weight_decay": config.weight_decay},
        {"params": gal_params,
         "lr": config.gal_lr,        "weight_decay": config.weight_decay},
    ])

    num_warmup = int(num_training_steps * config.warmup_ratio)
    scheduler  = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def evaluate_model(model: GALModel, loader: DataLoader,
                   device: str, return_preds: bool = False) -> Dict:
    """ประเมินโมเดล — returns loss, accuracy, f1_macro
    return_preds=True: เพิ่ม all_preds, all_labels ใน dict (ใช้สำหรับ confusion matrix)
    """
    model.eval()
    criterion                    = nn.CrossEntropyLoss()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            ttype = batch["token_type_ids"].to(device)
            amask = batch["aspect_mask"].to(device)
            cdm   = batch["lcf_cdm_mask"].to(device)
            cdw   = batch["lcf_cdw_weight"].to(device)
            lbl   = batch["label"].to(device)

            logits = model(ids, mask, ttype, amask, cdm, cdw)
            total_loss += criterion(logits, lbl).item()
            all_preds.extend(torch.argmax(logits, -1).cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    result = {
        "loss":     total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    if return_preds:
        result["all_preds"]  = all_preds
        result["all_labels"] = all_labels
    return result


def print_analysis(all_labels, all_preds, config: GALConfig,
                   writer: Optional[SummaryWriter] = None):
    """
    แสดง Confusion Matrix และ Per-class Report หลัง test evaluation
    - พิมพ์ confusion matrix แบบ text ใน terminal
    - พิมพ์ per-class precision / recall / F1
    - log confusion matrix เป็น image ใน TensorBoard (ถ้าส่ง writer มา)
    """
    import matplotlib
    matplotlib.use("Agg")   # ไม่ pop-up window
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # สร้าง label names จาก label_map (เรียงตาม index)
    label_names = [k for k, _ in sorted(config.label_map.items(), key=lambda x: x[1])]
    if not label_names:
        label_names = [str(i) for i in range(config.num_classes)]

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(config.num_classes)))

    # ── พิมพ์ Confusion Matrix ─────────────────────────────────────────────────
    col_w = max(len(n) for n in label_names) + 2
    header = " " * (col_w + 2) + "  ".join(f"{n:>{col_w}}" for n in label_names)
    print(f"\n  Confusion Matrix (row=actual, col=predicted)")
    print(f"  {header}")
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>{col_w}}" for v in row)
        print(f"  {label_names[i]:>{col_w}}  {row_str}")

    # ── พิมพ์ Per-class Report ────────────────────────────────────────────────
    print(f"\n  Classification Report")
    report = classification_report(
        all_labels, all_preds,
        target_names=label_names, zero_division=0
    )
    for line in report.splitlines():
        print(f"  {line}")

    # ── Log ใน TensorBoard เป็นรูป ─────────────────────────────────────────
    if writer is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(label_names)))
        ax.set_yticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=30, ha="right")
        ax.set_yticklabels(label_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix (Test Set)")

        # ใส่ตัวเลขในแต่ละช่อง
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", color=color, fontsize=12)

        plt.tight_layout()
        writer.add_figure("Test/confusion_matrix", fig)
        plt.close(fig)
        print("\n  ✓ Confusion matrix logged → TensorBoard > IMAGES tab")


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, tag="best"):
    """บันทึก checkpoint"""
    path = os.path.join(config.checkpoint_dir, f"gal_{tag}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics":              metrics,
    }, path)
    print(f"  ✓ Saved checkpoint → {path}")
    return path


def load_checkpoint(model, optimizer, scheduler, path: str, device: str):
    """โหลด checkpoint กลับเข้าโมเดล"""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch   = ckpt.get("epoch", 0)
    metrics = ckpt.get("metrics", {})
    print(f"  ✓ Loaded checkpoint (epoch {epoch}) ← {path}")
    return epoch, metrics


# =============================================================================
# SECTION 5: TRAINING LOOP
# ── Training + Validation + TensorBoard Logging
# =============================================================================

def train(config: GALConfig):
    """
    ฟังก์ชัน training หลัก:
    1. โหลดข้อมูลจาก CSV ไฟล์เดียว
    2. สร้างโมเดล (pretrained + GAL)
    3. วน train epoch
    4. บันทึก best checkpoint
    5. ประเมิน test set ด้วย labels
    6. log ลง TensorBoard
    """
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    device   = config.device
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer   = SummaryWriter(log_dir=os.path.join(config.log_dir, run_name))

    print(f"\n{'='*60}")
    print(f"  GAL Training  |  Session: {run_name}")
    print(f"  Device: {device}  |  Pretrained: {config.pretrained_model_name}")
    print(f"{'='*60}\n")

    # ── [1] โหลดข้อมูล ─────────────────────────────────────────────────────
    print("[1/4] Loading data...")
    tokenizer             = AutoTokenizer.from_pretrained(
        config.pretrained_model_name, use_fast=config.use_fast_tokenizer
    )
    train_df, val_df, test_df = load_and_split_data(config)

    train_loader = DataLoader(SentimentDataset(train_df, tokenizer, config),
                              batch_size=config.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(SentimentDataset(val_df,   tokenizer, config),
                              batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(SentimentDataset(test_df,  tokenizer, config),
                              batch_size=config.batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

    # ── [2] สร้างโมเดล ─────────────────────────────────────────────────────
    print("[2/4] Building model...")
    # ─────────────────────────────────────────────────────────────────────────
    # PLUG IN YOUR PRE-TRAINED MODEL HERE
    # ตัวอย่าง:
    #   pretrained = AutoModel.from_pretrained("roberta-base")
    #   pretrained = AutoModel.from_pretrained("./my_local_model")
    # ─────────────────────────────────────────────────────────────────────────
    pretrained = AutoModel.from_pretrained(config.pretrained_model_name)
    model      = GALModel(pretrained, config).to(device)

    total_p   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters — Total: {total_p:,}  |  Trainable: {trainable:,}")

    # ── [3] Optimizer + Scheduler ──────────────────────────────────────────
    print("[3/4] Setting up optimizer...")
    num_steps            = len(train_loader) * config.num_epochs
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, num_steps)
    criterion            = nn.CrossEntropyLoss()

    # ── [4] Training Loop ──────────────────────────────────────────────────
    print("[4/4] Training...\n")
    best_val_acc = 0.0
    global_step  = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            ttype = batch["token_type_ids"].to(device)
            amask = batch["aspect_mask"].to(device)
            cdm   = batch["lcf_cdm_mask"].to(device)
            cdw   = batch["lcf_cdw_weight"].to(device)
            lbl   = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask, ttype, amask, cdm, cdw)
            loss   = criterion(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                writer.add_scalar("Train/loss_step",     loss.item(),                      global_step)
                writer.add_scalar("Train/lr_pretrained", optimizer.param_groups[0]["lr"],  global_step)
                writer.add_scalar("Train/lr_gal",        optimizer.param_groups[1]["lr"],  global_step)

        # ── Validation ─────────────────────────────────────────────────────
        avg_loss = epoch_loss / len(train_loader)
        val_m    = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch:>3}/{config.num_epochs}"
              f"  TrainLoss={avg_loss:.4f}"
              f"  ValLoss={val_m['loss']:.4f}"
              f"  ValAcc={val_m['accuracy']:.4f}"
              f"  ValF1={val_m['f1_macro']:.4f}")

        writer.add_scalars("Loss",        {"train": avg_loss, "val": val_m["loss"]}, epoch)
        writer.add_scalar("Val/accuracy", val_m["accuracy"], epoch)
        writer.add_scalar("Val/f1_macro", val_m["f1_macro"], epoch)

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            save_checkpoint(model, optimizer, scheduler, epoch, val_m, config, "best")

        if not config.save_best_only:
            save_checkpoint(model, optimizer, scheduler, epoch, val_m, config, f"epoch_{epoch}")

    # ── Final Test Evaluation (ใช้ labels validate ผลสุดท้าย) ─────────────
    print(f"\n{'='*60}")
    print("  Final Test Evaluation — validate ด้วย labels")
    best_path = os.path.join(config.checkpoint_dir, FINAL_OUTPUT_NAME)
    load_checkpoint(model, None, None, best_path, device)
    test_m = evaluate_model(model, test_loader, device, return_preds=True)
    print(f"  Test Accuracy : {test_m['accuracy']:.4f}")
    print(f"  Test F1 Macro : {test_m['f1_macro']:.4f}")
    print_analysis(test_m["all_labels"], test_m["all_preds"], config, writer)
    print(f"{'='*60}")

    writer.add_hparams(
        hparam_dict={
            "pretrained_lr":  config.pretrained_lr,
            "gal_lr":         config.gal_lr,
            "batch_size":     config.batch_size,
            "num_gal_layers": config.num_gal_layers,
            "lcf_mode":       config.lcf_mode,
        },
        metric_dict={
            "hparam/test_accuracy": test_m["accuracy"],
            "hparam/test_f1":       test_m["f1_macro"],
        },
    )
    writer.close()

    cfg_path = os.path.join(config.output_dir, f"config_{run_name}.json")
    config.save(cfg_path)
    print(f"\n  Config saved → {cfg_path}")
    print(f"  TensorBoard  → tensorboard --logdir={config.log_dir}\n")

    return test_m


# =============================================================================
# SECTION 6: ENTRY POINT — Command-Line Interface
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GAL model for Sentiment Classification"
    )

    # Dataset
    parser.add_argument("--data_file",     default="data/dataset.csv",
                        help="path ไปยัง CSV ที่มี text และ labels")
    parser.add_argument("--text_column",   default="text")
    parser.add_argument("--label_column",  default="labels")

    # Model
    parser.add_argument("--pretrained",     default="bert-base-uncased",
                        help="ชื่อโมเดลจาก HuggingFace หรือ path local")
    parser.add_argument("--no_fast_tokenizer", action="store_true",
                        help="ใช้ use_fast=False สำหรับโมเดล SentencePiece เช่น rinna")
    parser.add_argument("--hidden_dim",     type=int,   default=768)
    parser.add_argument("--num_heads",      type=int,   default=8)
    parser.add_argument("--num_gal_layers", type=int,   default=2)
    parser.add_argument("--lcf_mode",       default="both", choices=["cdm", "cdw", "both"])
    parser.add_argument("--max_seq_len",    type=int,   default=128)

    # Training
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--pretrained_lr", type=float, default=2e-5)
    parser.add_argument("--gal_lr",        type=float, default=1e-4)
    parser.add_argument("--dropout",       type=float, default=0.1)
    parser.add_argument("--seed",          type=int,   default=42)

    args = parser.parse_args()

    cfg = GALConfig(
        data_file             = args.data_file,
        text_column           = args.text_column,
        label_column          = args.label_column,
        pretrained_model_name = args.pretrained,
        use_fast_tokenizer    = not args.no_fast_tokenizer,
        hidden_dim            = args.hidden_dim,
        num_attention_heads   = args.num_heads,
        num_gal_layers        = args.num_gal_layers,
        lcf_mode              = args.lcf_mode,
        max_seq_len           = args.max_seq_len,
        batch_size            = args.batch_size,
        num_epochs            = args.epochs,
        pretrained_lr         = args.pretrained_lr,
        gal_lr                = args.gal_lr,
        dropout_rate          = args.dropout,
        random_seed           = args.seed,
    )

    train(cfg)