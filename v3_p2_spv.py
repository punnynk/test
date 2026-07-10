# =============================================================================
# train_supervised.py — Stage 2: Supervised Fine-tuning
# เทรนด้วย soft label + sample_weight จาก vote distribution
#
# รัน: python train_supervised.py
# ผลลัพธ์:
#   - model v1 → SUPERVISED_OUTPUT_DIR/best.pt
#   - gold eval set → SUPERVISED_OUTPUT_DIR/gold_eval.csv
# =============================================================================

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from typing import Optional

load_dotenv()


# =============================================================================
# SECTION 1: CONFIG — อ่านจาก .env ทั้งหมด
# =============================================================================

CFG = {
    "labeled_file":        os.getenv("LABELED_FILE"),
    "text_column":         os.getenv("TEXT_COLUMN"),
    "label_column":        os.getenv("LABEL_COLUMN"),
    "vote_prefix":         os.getenv("VOTE_COLUMN_PREFIX", "class_"),
    "supervised_out":      os.getenv("SUPERVISED_OUTPUT_DIR"),
    "dapt_output_dir":     os.getenv("DAPT_OUTPUT_DIR"),
    "log_dir":             os.getenv("LOG_DIR"),
    "model_name":          os.getenv("MODEL_NAME"),
    "use_fast":            os.getenv("USE_FAST_TOKENIZER", "false").lower() == "true",
    "max_seq_len":         int(os.getenv("MAX_SEQ_LEN")),
    "hidden_dim":          int(os.getenv("HIDDEN_DIM")),
    "num_heads":           int(os.getenv("NUM_ATTENTION_HEADS")),
    "num_gal_layers":      int(os.getenv("NUM_GAL_LAYERS")),
    "lcf_mode":            os.getenv("LCF_MODE", "cdm"),
    "num_classes":         int(os.getenv("NUM_CLASSES")),
    "dropout":             float(os.getenv("DROPOUT")),
    "batch_size":          int(os.getenv("SUP_BATCH_SIZE")),
    "num_epochs":          int(os.getenv("SUP_EPOCHS")),
    "backbone_lr":         float(os.getenv("BACKBONE_LR")),
    "gal_lr":              float(os.getenv("GAL_LR")),
    "weight_decay":        float(os.getenv("SUP_WEIGHT_DECAY")),
    "warmup_ratio":        float(os.getenv("SUP_WARMUP_RATIO")),
    "max_grad_norm":       float(os.getenv("SUP_MAX_GRAD_NORM")),
    "val_split":           float(os.getenv("SUP_VAL_SPLIT")),
    "sample_weight_min":   float(os.getenv("SAMPLE_WEIGHT_MIN")),
    "gold_split":          float(os.getenv("GOLD_SPLIT")),
    "num_gpus":            int(os.getenv("NUM_GPUS", "0")),
    "seed":                int(os.getenv("RANDOM_SEED", "42")),
}

os.makedirs(CFG["supervised_out"], exist_ok=True)
os.makedirs(CFG["log_dir"],        exist_ok=True)

# ── Class name mapping ────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "emotional attack",
    1: "emotional persuasion",
    2: "rational persuasion",
    3: "sarcasm / irony",
    4: "indirect expression",
    5: "no anger",
}


# =============================================================================
# SECTION 2: DATA PREP — Soft Label + Sample Weight
# =============================================================================

def compute_soft_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    แปลง vote columns → soft label + sample_weight

    vote_cols: class_0 .. class_5
    soft_label[i] = class_i / total_votes  (fillna 0 ถ้าไม่มี vote)
    entropy_norm  = -Σ p·log(p) / log(6)   (normalize เป็น [0,1])
    sample_weight = clip(1 - entropy_norm, min, 1.0)
    """
    n = CFG["num_classes"]
    vote_cols = [f"{CFG['vote_prefix']}{i}" for i in range(n)]

    for col in vote_cols:
        if col not in df.columns:
            raise ValueError(f"ไม่พบ vote column '{col}' | มี: {list(df.columns)}")

    votes = df[vote_cols].fillna(0).values.astype(float)
    total = votes.sum(axis=1, keepdims=True)
    total = np.where(total == 0, 1, total)   # กัน div-by-zero
    soft  = votes / total

    eps     = 1e-9
    entropy = -(soft * np.log(soft + eps)).sum(axis=1)
    ent_max = math.log(n)
    ent_norm = entropy / ent_max

    weight = np.clip(1.0 - ent_norm, CFG["sample_weight_min"], 1.0)

    soft_df = pd.DataFrame(soft, columns=[f"soft_{i}" for i in range(n)])
    df = pd.concat([df.reset_index(drop=True), soft_df], axis=1)
    df["entropy_norm"]  = ent_norm
    df["sample_weight"] = weight
    return df


def split_gold_and_train(df: pd.DataFrame):
    """
    แยก gold eval set แบบ stratified — รับประกันทุก class มีตัวแทน
    ส่วนที่เหลือ split เป็น train/val แบบ stratified เช่นกัน
    """
    # แยก gold set ด้วย stratify (สัดส่วนตาม GOLD_SPLIT)
    rest_df, gold_df = train_test_split(
        df, test_size=CFG["gold_split"], random_state=CFG["seed"],
        stratify=df[CFG["label_column"]]
    )
    rest_df = rest_df.reset_index(drop=True)
    gold_df = gold_df.reset_index(drop=True)

    print(f"  Gold eval set : {len(gold_df):,} rows (stratified {CFG['gold_split']*100:.0f}%)")
    print(f"  Train pool    : {len(rest_df):,} rows")
    print(f"  Gold class dist:\n{gold_df[CFG['label_column']].value_counts().sort_index().to_string()}")

    train_df, val_df = train_test_split(
        rest_df, test_size=CFG["val_split"], random_state=CFG["seed"],
        stratify=rest_df[CFG["label_column"]]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), gold_df


# =============================================================================
# SECTION 3: DATASET
# =============================================================================

class LabeledDataset(Dataset):
    """Dataset สำหรับ supervised training — รองรับ soft label + sample_weight"""

    def __init__(self, data: pd.DataFrame, tokenizer):
        self.data      = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.n         = CFG["num_classes"]

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row  = self.data.iloc[idx]
        text = str(row[CFG["text_column"]])

        enc = self.tokenizer(
            text, max_length=CFG["max_seq_len"],
            padding="max_length", truncation=True, return_tensors="pt",
        )
        soft_label    = torch.tensor(
            [row[f"soft_{i}"] for i in range(self.n)], dtype=torch.float
        )
        sample_weight = torch.tensor(row["sample_weight"], dtype=torch.float)
        hard_label    = torch.tensor(int(row[CFG["label_column"]]), dtype=torch.long)

        attention_mask = enc["attention_mask"].squeeze(0)
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": attention_mask,
            "aspect_mask":    attention_mask.float(),
            "lcf_cdm_mask":   attention_mask.float(),
            "lcf_cdw_weight": attention_mask.float(),
            "soft_label":     soft_label,
            "sample_weight":  sample_weight,
            "hard_label":     hard_label,
        }


# =============================================================================
# SECTION 4: GAL MODEL ARCHITECTURE
# (importable โดย train_selftraining.py)
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.q_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.norm      = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
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
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)

    def forward(self, h, mask): return self.mhsa(h, mask=mask)


class AspectContextModule(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)

    def forward(self, h, mask): return self.mhsa(h, mask=mask)


class LocalContextFocusModule(nn.Module):
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
        combined = attention_mask * lcf_cdm_mask if self.mode in ("cdm", "both") \
                   else attention_mask
        return self.mhsa(local_h, mask=combined)


class GALFusionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList([
            MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, g, a, l, mask):
        fused = self.fusion_proj(torch.cat([g, a, l], dim=-1))
        for layer in self.layers:
            fused = layer(fused, mask=mask)
        return fused


class GALClassifier(nn.Module):
    """
    RoBERTa backbone + GAL mechanism + Linear head

    input  : tokenized text
    output : logits [B, num_classes]  (ใช้กับ soft-CE loss)
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        h = CFG["hidden_dim"]
        n = CFG["num_heads"]
        d = CFG["dropout"]

        self.backbone      = backbone
        self.global_module = GlobalContextModule(h, n, d)
        self.aspect_module = AspectContextModule(h, n, d)
        self.local_module  = LocalContextFocusModule(h, n, d, CFG["lcf_mode"])
        self.gal_fusion    = GALFusionLayer(h, n, CFG["num_gal_layers"], d)
        self.dropout       = nn.Dropout(d)
        self.classifier    = nn.Linear(h, CFG["num_classes"])

    def forward(self, input_ids, attention_mask, aspect_mask, lcf_cdm_mask, lcf_cdw_weight):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        g = self.global_module(hidden, attention_mask)
        a = self.aspect_module(hidden, aspect_mask)
        l = self.local_module(hidden, lcf_cdm_mask, lcf_cdw_weight, attention_mask)

        fused = self.gal_fusion(g, a, l, attention_mask)
        return self.classifier(self.dropout(fused[:, 0, :]))


# =============================================================================
# SECTION 5: LOSS FUNCTION
# =============================================================================

def weighted_soft_ce(logits, soft_labels, sample_weights, class_weights=None):
    """
    Weighted Soft Cross-Entropy พร้อม optional class weight

    logits        : [B, C]
    soft_labels   : [B, C]  (sum=1)
    sample_weights: [B]     — น้ำหนักต่อ sample (จาก entropy)
    class_weights : [C]     — น้ำหนักต่อ class (จาก inverse frequency) หรือ None
    """
    log_probs = F.log_softmax(logits, dim=-1)

    if class_weights is not None:
        # ถ่วงน้ำหนักแต่ละ class ก่อน sum → penalize class ที่โมเดล over-predict
        per_sample = -(soft_labels * log_probs * class_weights).sum(dim=-1)
    else:
        per_sample = -(soft_labels * log_probs).sum(dim=-1)

    total_w = sample_weights.sum().clamp(min=1e-9)
    return (per_sample * sample_weights).sum() / total_w


# =============================================================================
# SECTION 6: MULTI-GPU HELPER
# =============================================================================

def setup_gpu(model):
    if not torch.cuda.is_available():
        print("  Using CPU")
        return model, "cpu"
    available = torch.cuda.device_count()
    use_n     = available if CFG["num_gpus"] == 0 else min(CFG["num_gpus"], available)
    device    = "cuda"
    model     = model.to(device)
    if use_n > 1:
        model = nn.DataParallel(model, device_ids=list(range(use_n)))
        print(f"  Using {use_n} GPU(s): {list(range(use_n))}")
    else:
        print("  Using 1 GPU")
    return model, device


def get_base(model):
    return model.module if isinstance(model, nn.DataParallel) else model


# =============================================================================
# SECTION 7: EVALUATION
# =============================================================================

def evaluate(model, loader, device):
    """ประเมินด้วย hard label — คืน accuracy, f1_macro, all_preds, all_labels"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["aspect_mask"].to(device),
                batch["lcf_cdm_mask"].to(device),
                batch["lcf_cdw_weight"].to(device),
            )
            all_preds.extend(torch.argmax(logits, -1).cpu().numpy())
            all_labels.extend(batch["hard_label"].numpy())

    return {
        "accuracy":   accuracy_score(all_labels, all_preds),
        "f1_macro":   f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "all_preds":  all_preds,
        "all_labels": all_labels,
    }


def print_analysis(all_labels, all_preds, writer=None, tag="Test"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label_names = [CLASS_NAMES[i] for i in range(CFG["num_classes"])]
    cm = confusion_matrix(all_labels, all_preds,
                          labels=list(range(CFG["num_classes"])))

    col_w = max(len(n) for n in label_names) + 2
    print(f"\n  Confusion Matrix  (row=actual, col=predicted)")
    print("  " + " " * (col_w + 2) + "  ".join(f"{n:>{col_w}}" for n in label_names))
    for i, row in enumerate(cm):
        print(f"  {label_names[i]:>{col_w}}  " +
              "  ".join(f"{v:>{col_w}}" for v in row))

    print(f"\n  Classification Report")
    for line in classification_report(
        all_labels, all_preds,
        labels=list(range(CFG["num_classes"])),
        target_names=label_names, zero_division=0
    ).splitlines():
        print(f"  {line}")

    if writer:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap="Blues"); plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=30, ha="right")
        ax.set_yticks(range(len(label_names)))
        ax.set_yticklabels(label_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {tag}")
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=10)
        plt.tight_layout()
        writer.add_figure(f"{tag}/confusion_matrix", fig)
        plt.close(fig)
        print(f"  ✓ Confusion matrix → TensorBoard > IMAGES tab")


# =============================================================================
# SECTION 8: TRAINING LOOP
# =============================================================================

def run_supervised():
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
    run_name = f"supervised_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer   = SummaryWriter(os.path.join(CFG["log_dir"], run_name))

    print(f"\n{'='*60}")
    print(f"  Supervised Fine-tuning  |  {run_name}")
    print(f"  Backbone: {CFG['dapt_output_dir']}")
    print(f"{'='*60}\n")

    # ── [1] Data ───────────────────────────────────────────────────────────
    print("[1/4] Loading & preparing data...")
    tokenizer = AutoTokenizer.from_pretrained(CFG["dapt_output_dir"],
                                               use_fast=CFG["use_fast"])
    df = pd.read_csv(CFG["labeled_file"])
    df = compute_soft_labels(df)
    train_df, val_df, gold_df = split_gold_and_train(df)

    # บันทึก gold eval set สำหรับ train_selftraining.py
    gold_path = os.path.join(CFG["supervised_out"], "gold_eval.csv")
    gold_df.to_csv(gold_path, index=False)
    print(f"  Gold eval set saved → {gold_path}")
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}")

    # คำนวณ class weight จาก train set (inverse frequency)
    cw_values = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(CFG["num_classes"]),
        y=train_df[CFG["label_column"]].values
    )
    cw_values = np.sqrt(cw_values)   # ลด intensity ของ class weight
    class_weights = torch.tensor(cw_values, dtype=torch.float)
    print(f"  Class weights: { {i: f'{v:.3f}' for i, v in enumerate(cw_values)} }")

    train_loader = DataLoader(LabeledDataset(train_df, tokenizer),
                              batch_size=CFG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(LabeledDataset(val_df,   tokenizer),
                              batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    # ── [2] Model ──────────────────────────────────────────────────────────
    print("[2/4] Building model...")
    backbone = AutoModel.from_pretrained(CFG["dapt_output_dir"])
    backbone.resize_token_embeddings(len(tokenizer))
    model = GALClassifier(backbone)
    total_p    = sum(p.numel() for p in model.parameters())
    backbone_p = sum(p.numel() for p in model.backbone.parameters())
    print(f"  Total: {total_p:,}  |  Backbone: {backbone_p:,}"
          f"  |  GAL+Head: {total_p - backbone_p:,}")
    model, device = setup_gpu(model)

    # ── [3] Optimizer ──────────────────────────────────────────────────────
    print("[3/4] Setting up optimizer...")
    num_steps    = len(train_loader) * CFG["num_epochs"]
    base         = get_base(model)
    backbone_ids = {id(p) for p in base.backbone.parameters()}
    gal_params   = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW([
        {"params": list(base.backbone.parameters()),
         "lr": CFG["backbone_lr"], "weight_decay": CFG["weight_decay"]},
        {"params": gal_params,
         "lr": CFG["gal_lr"],      "weight_decay": CFG["weight_decay"]},
    ])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(num_steps * CFG["warmup_ratio"]), num_steps
    )

    # ── [4] Training Loop ──────────────────────────────────────────────────
    print("[4/4] Training...\n")
    best_val_f1  = 0.0
    global_step  = 0
    cw_device    = class_weights.to(device)   # ย้าย class_weights ไป device เดียวกับโมเดล

    for epoch in range(1, CFG["num_epochs"] + 1):
        model.train(); epoch_loss = 0.0

        for batch in train_loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            amask = batch["aspect_mask"].to(device)
            cdm   = batch["lcf_cdm_mask"].to(device)
            cdw   = batch["lcf_cdw_weight"].to(device)
            soft  = batch["soft_label"].to(device)
            sw    = batch["sample_weight"].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask, amask, cdm, cdw)
            loss   = weighted_soft_ce(logits, soft, sw, class_weights=cw_device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
            optimizer.step(); scheduler.step()

            epoch_loss += loss.item(); global_step += 1
            if global_step % 10 == 0:
                writer.add_scalar("Supervised/loss_step",    loss.item(),                     global_step)
                writer.add_scalar("Supervised/lr_backbone",  optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("Supervised/lr_gal",       optimizer.param_groups[1]["lr"], global_step)

        avg   = epoch_loss / len(train_loader)
        val_m = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:>2}/{CFG['num_epochs']}"
              f"  TrainLoss={avg:.4f}"
              f"  ValAcc={val_m['accuracy']:.4f}"
              f"  ValF1={val_m['f1_macro']:.4f}")

        writer.add_scalar("Supervised/loss_epoch", avg,                  epoch)
        writer.add_scalar("Supervised/val_acc",    val_m["accuracy"],    epoch)
        writer.add_scalar("Supervised/val_f1",     val_m["f1_macro"],    epoch)

        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            torch.save(
                {"epoch": epoch,
                 "model_state_dict": get_base(model).state_dict(),
                 "metrics": val_m},
                os.path.join(CFG["supervised_out"], "best.pt")
            )
            print(f"  ✓ Saved best (ValF1={best_val_f1:.4f})")

    # ── Final Eval บน Gold Set ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Final Eval — Gold Set")
    ckpt = torch.load(os.path.join(CFG["supervised_out"], "best.pt"),
                      map_location=device)
    get_base(model).load_state_dict(ckpt["model_state_dict"])
    gold_loader = DataLoader(LabeledDataset(gold_df, tokenizer),
                             batch_size=CFG["batch_size"], shuffle=False, num_workers=0)
    gold_m = evaluate(model, gold_loader, device)
    print(f"  Gold Accuracy : {gold_m['accuracy']:.4f}")
    print(f"  Gold F1 Macro : {gold_m['f1_macro']:.4f}")
    print_analysis(gold_m["all_labels"], gold_m["all_preds"], writer, tag="Gold_v1")
    print(f"{'='*60}")

    writer.add_hparams(
        {"backbone_lr": CFG["backbone_lr"], "gal_lr": CFG["gal_lr"],
         "num_gal_layers": CFG["num_gal_layers"], "lcf_mode": CFG["lcf_mode"]},
        {"hparam/gold_f1": gold_m["f1_macro"], "hparam/gold_acc": gold_m["accuracy"]},
    )
    writer.close()
    print(f"\n  TensorBoard → tensorboard --logdir={CFG['log_dir']}")
    print(f"  Model v1 saved → {CFG['supervised_out']}/best.pt")
    print(f"  ✅  Supervised เสร็จ — รัน train_selftraining.py ต่อ\n")


if __name__ == "__main__":
    run_supervised()