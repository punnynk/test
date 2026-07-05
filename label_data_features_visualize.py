# =============================================================================
# RoBERTa Feature Extraction + UMAP Visualization
# สกัด vector จาก RoBERTa แล้ว visualize ด้วย UMAP (2D)
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import umap

# =============================================================================
# SECTION 1: CONFIG
# =============================================================================
DATA_FILE    = "/share/ksl_pub/resource/Dataset/FumanDataset/Dataset/human_data_with_kosen.csv"
TEXT_COLUMN  = "text"
LABEL_COLUMN = "labels"

MODEL_NAME   = "rinna/japanese-roberta-base"   # เปลี่ยนโมเดลได้ที่นี่
USE_FAST     = False                            # False สำหรับ SentencePiece (rinna)
MAX_SEQ_LEN  = 510
BATCH_SIZE   = 16
POOLING_MODE = "mean"   # "cls" หรือ "mean"

# UMAP
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1

# GPU: None = ใช้ทุกการ์ดที่มี | N = ใช้ N การ์ดแรก | 1 = single GPU
NUM_GPUS = 3


# =============================================================================
# SECTION 2: DATASET
# =============================================================================
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts     = texts
        self.tokenizer = tokenizer
 
    def __len__(self): return len(self.texts)
 
    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=MAX_SEQ_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
 
 
# =============================================================================
# SECTION 3: FEATURE EXTRACTION (multi-GPU)
# =============================================================================
def extract_features(texts, tokenizer, model, device):
    dataset = TextDataset(texts, tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_vecs = []
 
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
 
            if POOLING_MODE == "cls":
                vec = hidden[:, 0, :]
            else:  # mean
                vec = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
 
            all_vecs.append(vec.cpu().float())
 
    return torch.cat(all_vecs, dim=0).numpy()
 
 
# =============================================================================
# SECTION 4: MAIN
# =============================================================================
if __name__ == "__main__":
 
    # ── โหลดข้อมูล ─────────────────────────────────────────────────────────
    df     = pd.read_csv(DATA_FILE)[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
    texts  = df[TEXT_COLUMN].tolist()
    labels = df[LABEL_COLUMN].tolist()
    print(f"Loaded {len(texts):,} rows | classes: {sorted(set(labels))}")
 
    # ── โหลด tokenizer + model ────────────────────────────────────────────
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=USE_FAST)
    model     = AutoModel.from_pretrained(MODEL_NAME).eval()
 
    # ── Multi-GPU setup ───────────────────────────────────────────────────
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    available = torch.cuda.device_count()
    use_n     = available if NUM_GPUS is None else min(NUM_GPUS, available)
 
    if device == "cuda" and use_n > 1:
        model = nn.DataParallel(model, device_ids=list(range(use_n)))
        print(f"Using {use_n} GPU(s): {list(range(use_n))}")
    elif device == "cuda":
        print("Using 1 GPU")
    else:
        print("Using CPU")
    model = model.to(device)
 
    # ── Feature Extraction ────────────────────────────────────────────────
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    print(f"Extracting features (pooling={POOLING_MODE})...")
    vectors = extract_features(texts, tokenizer, model, device)
    print(f"Vectors shape: {vectors.shape}")
 
    # ── UMAP ─────────────────────────────────────────────────────────────
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=UMAP_N_NEIGHBORS,
                        min_dist=UMAP_MIN_DIST, random_state=42)
    coords  = reducer.fit_transform(vectors)
 
    # ── Plot ──────────────────────────────────────────────────────────────
    label_names = sorted(set(labels))
    colors      = cm.tab10(np.linspace(0, 1, len(label_names)))
    label_to_color = {name: colors[i] for i, name in enumerate(label_names)}
 
    fig, ax = plt.subplots(figsize=(10, 8))
    for name in label_names:
        idx = [i for i, l in enumerate(labels) if l == name]
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=[label_to_color[name]], label=name, alpha=0.6, s=20)
 
    ax.set_title(f"UMAP — {MODEL_NAME} ({POOLING_MODE} pooling)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig("umap_result_mean.png", dpi=150, bbox_inches="tight")
    print("Saved → umap_result.png")