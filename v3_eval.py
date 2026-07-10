# =============================================================================
# evaluate.py — ประเมินโมเดลท้ายสุด (ไม่ retrain)
# วัด 2 มิติ:
#   1. Hard label: Accuracy / F1 (argmax vs majority label)
#   2. Distribution agreement: JS Divergence (model dist vs vote dist)
#
# รัน: python evaluate.py
# ต้องรัน train_supervised.py และ train_selftraining.py ก่อน
# =============================================================================

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, os.path.dirname(__file__))
from v3_p2_spv import (
    GALClassifier, LabeledDataset,
    compute_soft_labels, CLASS_NAMES,
)
from transformers import AutoTokenizer, AutoModel

load_dotenv()


# =============================================================================
# SECTION 1: CONFIG — อ่านจาก .env
# =============================================================================

CFG = {
    "gold_eval_file":  os.path.join(os.getenv("SUPERVISED_OUTPUT_DIR"), "gold_eval.csv"),
    "model_path":      os.path.join(os.getenv("SELFTRAINING_OUTPUT_DIR"), "best.pt"),
    "dapt_output_dir": os.getenv("DAPT_OUTPUT_DIR"),
    "use_fast":        os.getenv("USE_FAST_TOKENIZER", "false").lower() == "true",
    "max_seq_len":     int(os.getenv("MAX_SEQ_LEN")),
    "num_classes":     int(os.getenv("NUM_CLASSES")),
    "batch_size":      int(os.getenv("SUP_BATCH_SIZE")),
    "output_dir":      os.getenv("SELFTRAINING_OUTPUT_DIR"),
}

os.makedirs(CFG["output_dir"], exist_ok=True)


# =============================================================================
# SECTION 2: EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_model(model, loader, device):
    """
    Inference บน loader — คืน model probabilities และ soft labels

    Returns:
        probs       : np.array [N, C] — softmax output ของโมเดล
        soft_labels : np.array [N, C] — vote distribution จาก data
        hard_labels : np.array [N]    — argmax ของ soft_label (majority vote)
    """
    model.eval()
    all_probs, all_soft, all_hard = [], [], []

    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["aspect_mask"].to(device),
            batch["lcf_cdm_mask"].to(device),
            batch["lcf_cdw_weight"].to(device),
        )
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_soft.append(batch["soft_label"].numpy())
        all_hard.append(batch["hard_label"].numpy())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_soft),
        np.concatenate(all_hard),
    )


def compute_metrics(probs, soft_labels, hard_labels):
    """
    คำนวณ 2 มิติ:
    1. Hard: accuracy + F1 (argmax model vs hard label)
    2. Soft: JS divergence (model distribution vs vote distribution)
    """
    pred_class = probs.argmax(axis=1)

    # ── Hard label metrics ─────────────────────────────────────────────────
    acc = accuracy_score(hard_labels, pred_class)
    f1  = f1_score(hard_labels, pred_class, average="macro", zero_division=0)

    # ── JS Divergence (0 = identical, 1 = completely different) ───────────
    eps = 1e-9
    probs_safe  = np.clip(probs,        eps, 1)
    soft_safe   = np.clip(soft_labels,  eps, 1)
    probs_safe  /= probs_safe.sum(axis=1, keepdims=True)
    soft_safe   /= soft_safe.sum(axis=1, keepdims=True)

    js_per_row = np.array([
        jensenshannon(p, s) for p, s in zip(probs_safe, soft_safe)
    ])
    js_mean = js_per_row.mean()

    return {
        "accuracy":    acc,
        "f1_macro":    f1,
        "js_mean":     js_mean,
        "js_per_row":  js_per_row,
        "pred_class":  pred_class,
    }


def print_results(metrics, hard_labels):
    label_names = [CLASS_NAMES[i] for i in range(CFG["num_classes"])]
    pred_class  = metrics["pred_class"]

    print(f"\n{'='*60}")
    print("  Evaluation Results")
    print(f"{'='*60}")
    print(f"\n  [Hard Label]")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  F1 Macro   : {metrics['f1_macro']:.4f}")
    print(f"\n  [Distribution Agreement]")
    print(f"  JS Divergence (mean) : {metrics['js_mean']:.4f}  "
          f"({'ดี' if metrics['js_mean'] < 0.3 else 'พอใช้' if metrics['js_mean'] < 0.5 else 'แย่'})")
    print(f"  (0 = identical, 1 = completely different)")

    # JS Divergence ต่อ class
    print(f"\n  JS Divergence per class:")
    js_per_class = {}
    for i, name in enumerate(label_names):
        mask = hard_labels == i
        if mask.sum() > 0:
            js_c = metrics["js_per_row"][mask].mean()
            js_per_class[name] = js_c
            print(f"    {name:<25} {js_c:.4f}")

    # Confusion Matrix (text)
    cm = confusion_matrix(hard_labels, pred_class,
                          labels=list(range(CFG["num_classes"])))
    col_w = max(len(n) for n in label_names) + 2
    print(f"\n  Confusion Matrix  (row=actual, col=predicted)")
    print("  " + " " * (col_w + 2) + "  ".join(f"{n:>{col_w}}" for n in label_names))
    for i, row in enumerate(cm):
        print(f"  {label_names[i]:>{col_w}}  " +
              "  ".join(f"{v:>{col_w}}" for v in row))

    # Classification Report
    print(f"\n  Classification Report")
    for line in classification_report(
        hard_labels, pred_class,
        labels=list(range(CFG["num_classes"])),
        target_names=label_names, zero_division=0
    ).splitlines():
        print(f"  {line}")

    print(f"{'='*60}\n")
    return cm, js_per_class


def save_plots(metrics, hard_labels, cm, js_per_class):
    """บันทึกรูป confusion matrix และ JS divergence distribution"""
    label_names = [CLASS_NAMES[i] for i in range(CFG["num_classes"])]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Confusion Matrix ──────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(label_names)))
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix\nAcc={metrics['accuracy']:.3f}  F1={metrics['f1_macro']:.3f}")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)

    # ── JS Divergence per class ────────────────────────────────────────────
    ax = axes[1]
    classes = list(js_per_class.keys())
    values  = list(js_per_class.values())
    bars    = ax.barh(classes, values, color="steelblue")
    ax.axvline(x=metrics["js_mean"], color="red", linestyle="--",
               label=f"mean={metrics['js_mean']:.3f}")
    ax.set_xlabel("JS Divergence (↓ better)")
    ax.set_title("JS Divergence per Class\n(model dist vs vote dist)")
    ax.legend()
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(CFG["output_dir"], "evaluation_result.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {out_path}")


# =============================================================================
# SECTION 3: MAIN
# =============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    # ── โหลด tokenizer + gold eval set ─────────────────────────────────────
    print("  Loading data...")
    if not os.path.exists(CFG["gold_eval_file"]):
        raise FileNotFoundError(
            f"ไม่พบ {CFG['gold_eval_file']} — กรุณารัน train_supervised.py ก่อน"
        )
    gold_df = pd.read_csv(CFG["gold_eval_file"])
    # soft columns ถูกบันทึกมาใน CSV แล้วจากไฟล์ 2 — ไม่ต้องคำนวณซ้ำ
    if "soft_0" not in gold_df.columns:
        gold_df = compute_soft_labels(gold_df)
    print(f"  Gold eval set: {len(gold_df):,} rows")

    tokenizer = AutoTokenizer.from_pretrained(CFG["dapt_output_dir"],
                                               use_fast=CFG["use_fast"])
    loader = DataLoader(LabeledDataset(gold_df, tokenizer),
                        batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    # ── โหลดโมเดล ───────────────────────────────────────────────────────────
    print("  Loading model...")
    if not os.path.exists(CFG["model_path"]):
        raise FileNotFoundError(
            f"ไม่พบ {CFG['model_path']} — กรุณารัน train_selftraining.py ก่อน"
        )
    backbone = AutoModel.from_pretrained(CFG["dapt_output_dir"])
    backbone.resize_token_embeddings(len(tokenizer))
    model = GALClassifier(backbone).to(device)

    ckpt = torch.load(CFG["model_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded ← {CFG['model_path']}")

    # ── Inference ────────────────────────────────────────────────────────────
    print("  Running inference...")
    probs, soft_labels, hard_labels = evaluate_model(model, loader, device)

    # ── Compute + Print ──────────────────────────────────────────────────────
    metrics           = compute_metrics(probs, soft_labels, hard_labels)
    cm, js_per_class  = print_results(metrics, hard_labels)

    # ── Save plots ───────────────────────────────────────────────────────────
    save_plots(metrics, hard_labels, cm, js_per_class)