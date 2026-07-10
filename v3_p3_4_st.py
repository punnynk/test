# =============================================================================
# train_selftraining.py — Stage 3+4: Self-Training Loop + Validation
# วนรอบ: predict unlabeled → filter → mix → fine-tune → validate บน gold set
#
# รัน: python train_selftraining.py
# ผลลัพธ์: model vN (ดีที่สุด) → SELFTRAINING_OUTPUT_DIR/best.pt
#
# ต้องรัน train_supervised.py ก่อน เพื่อให้มี:
#   - SUPERVISED_OUTPUT_DIR/best.pt      (model v1)
#   - SUPERVISED_OUTPUT_DIR/gold_eval.csv (gold eval set)
# =============================================================================

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from typing import Optional

# import model architecture และ helper จาก train_supervised.py
sys.path.insert(0, os.path.dirname(__file__))
from v3_p2_spv import (
    GALClassifier, LabeledDataset,
    compute_soft_labels, weighted_soft_ce,
    print_analysis, CLASS_NAMES,
)

load_dotenv()


def _int_or_none(val: str):
    return None if val.strip() == "" else int(val)


# =============================================================================
# SECTION 1: CONFIG — อ่านจาก .env ทั้งหมด
# =============================================================================

CFG = {
    "labeled_file":    os.getenv("LABELED_FILE"),
    "unlabeled_file":  os.getenv("UNLABELED_FILE"),
    "supervised_out":  os.getenv("SUPERVISED_OUTPUT_DIR"),
    "st_out":          os.getenv("SELFTRAINING_OUTPUT_DIR"),
    "log_dir":         os.getenv("LOG_DIR"),
    "text_column":     os.getenv("TEXT_COLUMN"),
    "label_column":    os.getenv("LABEL_COLUMN"),
    "vote_prefix":     os.getenv("VOTE_COLUMN_PREFIX", "class_"),
    "dapt_output_dir": os.getenv("DAPT_OUTPUT_DIR"),
    "use_fast":        os.getenv("USE_FAST_TOKENIZER", "false").lower() == "true",
    "max_seq_len":     int(os.getenv("MAX_SEQ_LEN")),
    "num_classes":     int(os.getenv("NUM_CLASSES")),
    "batch_size":      int(os.getenv("ST_BATCH_SIZE")),
    "confidence_thr":  float(os.getenv("ST_CONFIDENCE_THRESHOLD")),
    "ratio_cap":       int(os.getenv("ST_RATIO_CAP")),
    "max_rounds":      int(os.getenv("ST_MAX_ROUNDS")),
    "patience":        int(os.getenv("ST_PATIENCE")),
    "backbone_lr":     float(os.getenv("ST_BACKBONE_LR")),
    "gal_lr":          float(os.getenv("ST_GAL_LR")),
    "weight_decay":    float(os.getenv("ST_WEIGHT_DECAY")),
    "warmup_ratio":    float(os.getenv("ST_WARMUP_RATIO")),
    "max_grad_norm":   float(os.getenv("ST_MAX_GRAD_NORM")),
    "epochs_per_round":int(os.getenv("ST_EPOCHS_PER_ROUND")),
    "num_gpus":        int(os.getenv("NUM_GPUS", "0")),
    "seed":            int(os.getenv("RANDOM_SEED", "42")),
    "unlabeled_start": int(os.getenv("ST_UNLABELED_START_LINE", "0")),
    "unlabeled_end":   _int_or_none(os.getenv("ST_UNLABELED_END_LINE", "")),
}

os.makedirs(CFG["st_out"], exist_ok=True)
os.makedirs(CFG["log_dir"], exist_ok=True)


# =============================================================================
# SECTION 2: PSEUDO-LABEL DATASET
# =============================================================================

class PseudoLabelDataset(Dataset):
    """
    Dataset สำหรับ pseudo-labeled samples
    soft_label = softmax output ของโมเดล (แทน vote distribution)
    sample_weight = confidence ที่ clip ตาม entropy เหมือน labeled
    """

    def __init__(self, texts, soft_labels, sample_weights, tokenizer):
        self.texts          = texts
        self.soft_labels    = soft_labels     # np.array [N, C]
        self.sample_weights = sample_weights  # np.array [N]
        self.tokenizer      = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=CFG["max_seq_len"], padding="max_length",
            truncation=True, return_tensors="pt",
        )
        attention_mask = enc["attention_mask"].squeeze(0)
        soft_label     = torch.tensor(self.soft_labels[idx],    dtype=torch.float)
        sample_weight  = torch.tensor(self.sample_weights[idx], dtype=torch.float)
        hard_label     = torch.tensor(int(soft_label.argmax()), dtype=torch.long)

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


class UnlabeledTextDataset(Dataset):
    """Dataset ข้อความเดียว (ไม่มี label) สำหรับ inference"""

    def __init__(self, texts, tokenizer):
        self.texts     = texts
        self.tokenizer = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=CFG["max_seq_len"], padding="max_length",
            truncation=True, return_tensors="pt",
        )
        attention_mask = enc["attention_mask"].squeeze(0)
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": attention_mask,
            "aspect_mask":    attention_mask.float(),
            "lcf_cdm_mask":   attention_mask.float(),
            "lcf_cdw_weight": attention_mask.float(),
        }


# =============================================================================
# SECTION 3: MULTI-GPU HELPER
# =============================================================================

def setup_gpu(model):
    if not torch.cuda.is_available():
        print("  Using CPU"); return model, "cpu"
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
# SECTION 4: INFERENCE — สร้าง Pseudo Labels
# =============================================================================

def generate_pseudo_labels(model, texts, tokenizer, device):
    """
    ส่งข้อความทั้งหมดเข้าโมเดล → ได้ softmax distribution + confidence

    Returns:
        soft_probs : np.array [N, C]  — softmax output
        confidence : np.array [N]     — max prob ต่อ sample
    """
    model.eval()
    dataset = UnlabeledTextDataset(texts, tokenizer)
    loader  = DataLoader(dataset, batch_size=CFG["batch_size"],
                         shuffle=False, num_workers=0)
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["aspect_mask"].to(device),
                batch["lcf_cdm_mask"].to(device),
                batch["lcf_cdw_weight"].to(device),
            )
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    soft_probs = np.vstack(all_probs)
    confidence = soft_probs.max(axis=1)
    return soft_probs, confidence


def filter_pseudo_labels(texts, soft_probs, confidence, n_labeled):
    """
    กรองเฉพาะ samples ที่ confidence > threshold
    และจำกัดจำนวนไม่เกิน ratio_cap × n_labeled
    """
    mask          = confidence >= CFG["confidence_thr"]
    filtered_text = [t for t, m in zip(texts, mask) if m]
    filtered_prob = soft_probs[mask]
    filtered_conf = confidence[mask]

    # ratio cap
    max_pseudo = CFG["ratio_cap"] * n_labeled
    if len(filtered_text) > max_pseudo:
        # เรียงตาม confidence สูงสุดก่อน เลือก top-N
        top_idx       = np.argsort(filtered_conf)[::-1][:max_pseudo]
        filtered_text = [filtered_text[i] for i in top_idx]
        filtered_prob = filtered_prob[top_idx]
        filtered_conf = filtered_conf[top_idx]

    # sample_weight สำหรับ pseudo-label = confidence (ยิ่งมั่นใจ ยิ่งมีน้ำหนัก)
    sample_weights = np.clip(filtered_conf, 0.1, 1.0)

    print(f"  Pseudo-labels: {mask.sum():,} pass threshold → "
          f"keep {len(filtered_text):,} "
          f"(ratio_cap={CFG['ratio_cap']}×{n_labeled}={max_pseudo})")

    return filtered_text, filtered_prob, sample_weights


# =============================================================================
# SECTION 5: EVALUATION ON GOLD SET
# =============================================================================

def evaluate_gold(model, loader, device):
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


# =============================================================================
# SECTION 6: FINE-TUNE ONE ROUND
# =============================================================================

def fine_tune_one_round(model, combined_loader, optimizer, scheduler, device):
    """Fine-tune โมเดล 1 epoch ด้วย combined dataset"""
    model.train()
    total_loss = 0.0

    for batch in combined_loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        amask = batch["aspect_mask"].to(device)
        cdm   = batch["lcf_cdm_mask"].to(device)
        cdw   = batch["lcf_cdw_weight"].to(device)
        soft  = batch["soft_label"].to(device)
        sw    = batch["sample_weight"].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask, amask, cdm, cdw)
        loss   = weighted_soft_ce(logits, soft, sw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(combined_loader)


# =============================================================================
# SECTION 7: SELF-TRAINING LOOP
# =============================================================================

def run_selftraining():
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
    run_name = f"selftraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer   = SummaryWriter(os.path.join(CFG["log_dir"], run_name))

    print(f"\n{'='*60}")
    print(f"  Self-Training Loop  |  {run_name}")
    print(f"  Max rounds: {CFG['max_rounds']}  |  Patience: {CFG['patience']}")
    print(f"  Confidence threshold: {CFG['confidence_thr']}")
    print(f"  Ratio cap: {CFG['ratio_cap']}:1")
    print(f"{'='*60}\n")

    # ── [1] โหลด tokenizer + gold eval set + labeled train data ────────────
    print("[1/3] Loading data...")
    tokenizer = AutoTokenizer.from_pretrained(CFG["dapt_output_dir"],
                                               use_fast=CFG["use_fast"])

    gold_path = os.path.join(CFG["supervised_out"], "gold_eval.csv")
    if not os.path.exists(gold_path):
        raise FileNotFoundError(
            f"ไม่พบ {gold_path} — กรุณารัน train_supervised.py ก่อน"
        )
    gold_df   = pd.read_csv(gold_path)
    gold_loader = DataLoader(LabeledDataset(gold_df, tokenizer),
                             batch_size=CFG["batch_size"], shuffle=False, num_workers=0)
    print(f"  Gold eval set: {len(gold_df):,} rows")

    labeled_df = pd.read_csv(CFG["labeled_file"])
    labeled_df = compute_soft_labels(labeled_df)
    n_labeled  = len(labeled_df)
    print(f"  Labeled data : {n_labeled:,} rows")

    if CFG["unlabeled_start"] > 0:
        unlabeled_df = pd.read_csv(CFG["unlabeled_file"],
                                   skiprows=range(1, CFG["unlabeled_start"] + 1))
    else:
        unlabeled_df = pd.read_csv(CFG["unlabeled_file"])

    if CFG["unlabeled_end"] is not None:
        limit = CFG["unlabeled_end"] - CFG["unlabeled_start"]
        if limit <= 0:
            raise ValueError("ST_UNLABELED_END_LINE ต้องมากกว่า ST_UNLABELED_START_LINE")
        unlabeled_df = unlabeled_df.iloc[:limit]

    unlabeled_texts = unlabeled_df[CFG["text_column"]].dropna().tolist()
    print(f"  Unlabeled    : {len(unlabeled_texts):,} rows "
          f"(start={CFG['unlabeled_start']}, end={CFG['unlabeled_end']})")

    # ── [2] โหลด model v1 จาก train_supervised.py ──────────────────────────
    print("[2/3] Loading model v1...")
    backbone = AutoModel.from_pretrained(CFG["dapt_output_dir"])
    backbone.resize_token_embeddings(len(tokenizer))
    model = GALClassifier(backbone)

    v1_path = os.path.join(CFG["supervised_out"], "best.pt")
    if not os.path.exists(v1_path):
        raise FileNotFoundError(
            f"ไม่พบ {v1_path} — กรุณารัน train_supervised.py ก่อน"
        )
    ckpt = torch.load(v1_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded model v1 ← {v1_path}")
    model, device = setup_gpu(model)

    # ── [3] Self-training Loop ─────────────────────────────────────────────
    print("[3/3] Self-training...\n")
    best_f1     = 0.0
    no_improve  = 0
    labeled_ds  = LabeledDataset(labeled_df, tokenizer)

    for rnd in range(1, CFG["max_rounds"] + 1):
        print(f"── Round {rnd}/{CFG['max_rounds']} {'─'*40}")

        # Step A: generate pseudo-labels
        soft_probs, confidence = generate_pseudo_labels(
            model, unlabeled_texts, tokenizer, device
        )

        # Step B: filter + ratio cap
        pseudo_texts, pseudo_probs, pseudo_weights = filter_pseudo_labels(
            unlabeled_texts, soft_probs, confidence, n_labeled
        )

        if len(pseudo_texts) == 0:
            print("  ไม่มี pseudo-label ผ่าน threshold — หยุด self-training")
            break

        # Step C: สร้าง combined dataset (labeled + pseudo-labeled)
        pseudo_ds    = PseudoLabelDataset(pseudo_texts, pseudo_probs,
                                          pseudo_weights, tokenizer)
        combined_ds  = ConcatDataset([labeled_ds, pseudo_ds])
        combined_loader = DataLoader(combined_ds, batch_size=CFG["batch_size"],
                                     shuffle=True, num_workers=0)
        print(f"  Combined dataset: {len(combined_ds):,} "
              f"(labeled={n_labeled:,} + pseudo={len(pseudo_ds):,})")

        # Step D: optimizer + scheduler สำหรับ round นี้
        base         = get_base(model)
        backbone_ids = {id(p) for p in base.backbone.parameters()}
        gal_params   = [p for p in model.parameters() if id(p) not in backbone_ids]
        num_steps    = len(combined_loader) * CFG["epochs_per_round"]

        optimizer = torch.optim.AdamW([
            {"params": list(base.backbone.parameters()),
             "lr": CFG["backbone_lr"], "weight_decay": CFG["weight_decay"]},
            {"params": gal_params,
             "lr": CFG["gal_lr"],      "weight_decay": CFG["weight_decay"]},
        ])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(num_steps * CFG["warmup_ratio"]), num_steps
        )

        # Step E: fine-tune
        for ep in range(CFG["epochs_per_round"]):
            avg_loss = fine_tune_one_round(
                model, combined_loader, optimizer, scheduler, device
            )
            print(f"  Epoch {ep+1}/{CFG['epochs_per_round']}  Loss={avg_loss:.4f}")

        # Step F: validate บน gold set
        gold_m = evaluate_gold(model, gold_loader, device)
        print(f"  Gold Acc={gold_m['accuracy']:.4f}  Gold F1={gold_m['f1_macro']:.4f}")

        writer.add_scalar("SelfTraining/gold_acc", gold_m["accuracy"], rnd)
        writer.add_scalar("SelfTraining/gold_f1",  gold_m["f1_macro"], rnd)
        writer.add_scalar("SelfTraining/n_pseudo",  len(pseudo_ds),    rnd)

        # Step G: save best + early stopping
        if gold_m["f1_macro"] > best_f1:
            best_f1    = gold_m["f1_macro"]
            no_improve = 0
            torch.save(
                {"round": rnd,
                 "model_state_dict": get_base(model).state_dict(),
                 "metrics": gold_m},
                os.path.join(CFG["st_out"], "best.pt")
            )
            print(f"  ✓ Saved best (GoldF1={best_f1:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{CFG['patience']})")
            if no_improve >= CFG["patience"]:
                print(f"\n  Early stopping — val F1 ไม่ขึ้น {CFG['patience']} รอบติด")
                break

        print()

    # ── Final Analysis ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Final Evaluation — Best Model (Gold Set)")
    best_ckpt = torch.load(os.path.join(CFG["st_out"], "best.pt"), map_location=device)
    get_base(model).load_state_dict(best_ckpt["model_state_dict"])
    final_m = evaluate_gold(model, gold_loader, device)
    print(f"  Gold Accuracy : {final_m['accuracy']:.4f}")
    print(f"  Gold F1 Macro : {final_m['f1_macro']:.4f}")
    print_analysis(final_m["all_labels"], final_m["all_preds"],
                   writer, tag="Gold_final")
    print(f"{'='*60}")

    writer.add_hparams(
        {"confidence_thr": CFG["confidence_thr"],
         "ratio_cap":      CFG["ratio_cap"],
         "backbone_lr":    CFG["backbone_lr"],
         "gal_lr":         CFG["gal_lr"]},
        {"hparam/final_gold_f1":  final_m["f1_macro"],
         "hparam/final_gold_acc": final_m["accuracy"]},
    )
    writer.close()
    print(f"\n  TensorBoard → tensorboard --logdir={CFG['log_dir']}")
    print(f"  Best model   → {CFG['st_out']}/best.pt")
    print(f"\n  Class names  : {CLASS_NAMES}")
    print(f"\n  ✅  Self-training เสร็จ\n")


if __name__ == "__main__":
    run_selftraining()