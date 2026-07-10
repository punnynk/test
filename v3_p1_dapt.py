# =============================================================================
# train_dapt.py — Stage 1: Domain-Adaptive Pre-Training (DAPT)
# MLM pretrain บน unlabeled data เพื่อให้ backbone เข้าใจ domain
#
# รัน: python train_dapt.py
# ผลลัพธ์: backbone weights ที่ DAPT_OUTPUT_DIR (ใช้ต่อใน train_supervised.py)
# =============================================================================

import os
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

load_dotenv()


# =============================================================================
# SECTION 1: CONFIG — อ่านจาก .env ทั้งหมด
# =============================================================================

def _int_or_none(val: str):
    return None if val.strip() == "" else int(val)

CFG = {
    "unlabeled_file":  os.getenv("UNLABELED_FILE"),
    "text_column":     os.getenv("TEXT_COLUMN"),
    "dapt_output_dir": os.getenv("DAPT_OUTPUT_DIR"),
    "log_dir":         os.getenv("LOG_DIR"),
    "model_name":      os.getenv("MODEL_NAME"),
    "use_fast":        os.getenv("USE_FAST_TOKENIZER", "false").lower() == "true",
    "max_seq_len":     int(os.getenv("MAX_SEQ_LEN")),
    "batch_size":      int(os.getenv("DAPT_BATCH_SIZE")),
    "num_epochs":      int(os.getenv("DAPT_EPOCHS")),
    "lr":              float(os.getenv("DAPT_LR")),
    "weight_decay":    float(os.getenv("DAPT_WEIGHT_DECAY")),
    "warmup_ratio":    float(os.getenv("DAPT_WARMUP_RATIO")),
    "max_grad_norm":   float(os.getenv("DAPT_MAX_GRAD_NORM")),
    "mlm_prob":        float(os.getenv("DAPT_MLM_PROB")),
    "start_line":      int(os.getenv("DAPT_START_LINE", "0")),
    "end_line":        _int_or_none(os.getenv("DAPT_END_LINE", "")),
    "num_gpus":        int(os.getenv("NUM_GPUS", "0")),
    "seed":            int(os.getenv("RANDOM_SEED", "42")),
}

os.makedirs(CFG["dapt_output_dir"], exist_ok=True)
os.makedirs(CFG["log_dir"],         exist_ok=True)


# =============================================================================
# SECTION 2: DATASET
# =============================================================================

class UnlabeledDataset(Dataset):
    """โหลด text จาก CSV — ไม่มี label, รองรับ start_line/end_line"""

    def __init__(self, tokenizer):
        if CFG["start_line"] > 0:
            df = pd.read_csv(CFG["unlabeled_file"],
                             skiprows=range(1, CFG["start_line"] + 1))
        else:
            df = pd.read_csv(CFG["unlabeled_file"])

        if CFG["end_line"] is not None:
            limit = CFG["end_line"] - CFG["start_line"]
            if limit <= 0:
                raise ValueError("DAPT_END_LINE ต้องมากกว่า DAPT_START_LINE")
            df = df.iloc[:limit]

        col = CFG["text_column"]
        if col not in df.columns:
            raise ValueError(f"ไม่พบ column '{col}' | มี: {list(df.columns)}")

        self.texts     = df[col].dropna().tolist()
        self.tokenizer = tokenizer
        print(f"  Loaded {len(self.texts):,} rows "
              f"(start={CFG['start_line']}, end={CFG['end_line']})")

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=CFG["max_seq_len"], padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# =============================================================================
# SECTION 3: MULTI-GPU HELPER
# =============================================================================

def setup_gpu(model):
    """ห่อ DataParallel ตามค่า NUM_GPUS"""
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
# SECTION 4: TRAINING
# =============================================================================

def run_dapt():
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
    run_name = f"dapt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer   = SummaryWriter(os.path.join(CFG["log_dir"], run_name))

    print(f"\n{'='*60}")
    print(f"  DAPT  |  {run_name}")
    print(f"  Model : {CFG['model_name']}  |  MLM prob: {CFG['mlm_prob']}")
    print(f"{'='*60}\n")

    # ── [1] Data ───────────────────────────────────────────────────────────
    print("[1/3] Loading data...")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"],
                                               use_fast=CFG["use_fast"])
    dataset   = UnlabeledDataset(tokenizer)
    collator  = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=CFG["mlm_prob"]
    )
    loader    = DataLoader(dataset, batch_size=CFG["batch_size"],
                           shuffle=True, collate_fn=collator, num_workers=0)

    # ── [2] Model ──────────────────────────────────────────────────────────
    print("[2/3] Loading model...")
    model = AutoModelForMaskedLM.from_pretrained(CFG["model_name"])
    model.resize_token_embeddings(len(tokenizer))
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model, device = setup_gpu(model)

    # ── [3] Training ───────────────────────────────────────────────────────
    print("[3/3] Pre-training...\n")
    num_steps = len(loader) * CFG["num_epochs"]
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(num_steps * CFG["warmup_ratio"]), num_steps
    )
    global_step = 0

    for epoch in range(1, CFG["num_epochs"] + 1):
        model.train(); epoch_loss = 0.0

        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl  = batch["labels"].to(device)

            optimizer.zero_grad()
            loss = model(input_ids=ids, attention_mask=mask, labels=lbl).loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
            optimizer.step(); scheduler.step()

            epoch_loss += loss.item(); global_step += 1
            if global_step % 50 == 0:
                writer.add_scalar("DAPT/loss_step", loss.item(), global_step)

        avg = epoch_loss / len(loader)
        ppl = math.exp(min(avg, 20))
        print(f"Epoch {epoch:>2}/{CFG['num_epochs']}  "
              f"Loss={avg:.4f}  Perplexity={ppl:.2f}")
        writer.add_scalar("DAPT/loss_epoch",  avg, epoch)
        writer.add_scalar("DAPT/perplexity",  ppl, epoch)

    # ── บันทึก backbone ────────────────────────────────────────────────────
    print(f"\n  Saving backbone → {CFG['dapt_output_dir']}")
    get_base(model).base_model.save_pretrained(CFG["dapt_output_dir"])
    tokenizer.save_pretrained(CFG["dapt_output_dir"])
    writer.close()

    print(f"  TensorBoard → tensorboard --logdir={CFG['log_dir']}")
    print(f"\n  ✅  DAPT เสร็จ — โหลด backbone จาก '{CFG['dapt_output_dir']}' ใน train_supervised.py\n")


if __name__ == "__main__":
    run_dapt()