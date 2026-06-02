#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CityVG — Adaptive BGE Contrastive Fine-tuning
---------------------------------------------
基于 caption 相似度自动构造 Triplet 样本，微调 BGE，使其更适合 City-Scale 3DVG。
"""

import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


# ================================================================
# 1. 加载带 caption 的 scenegraph
# ================================================================
def load_captions(scenegraph_path):
    with open(scenegraph_path, "r", encoding="utf-8") as f:
        sg = json.load(f)

    caps = []
    for e in sg["edges"]:
        cap = e.get("caption", "").strip()
        if cap:
            caps.append(cap)

    print(f"📦 Loaded {len(caps)} captions.")
    return caps


# ================================================================
# 2. BGE embedding
# ================================================================
def embed(model, sentences):
    print("🔢 Computing BGE embeddings...")
    with torch.inference_mode():
        emb = model.encode(
            sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
    return emb


# ================================================================
# 3. 构建自适应 Triplet 样本
# ================================================================
def build_triplets(captions, emb_matrix, max_samples=60000):

    n = len(captions)
    print("🧱 Building triplets...")
    sim_matrix = torch.matmul(emb_matrix, emb_matrix.T).cpu().numpy()

    triplets = []

    for i in tqdm(range(n), desc="Triplet build"):

        sims = sim_matrix[i]

        # 自适应阈值
        pos_th = np.quantile(sims, 0.60)
        neg_th = np.quantile(sims, 0.10)

        pos_ids = [k for k in np.where(sims > pos_th)[0] if k != i]
        neg_ids = [k for k in np.where(sims < neg_th)[0] if k != i]

        if not pos_ids or not neg_ids:
            continue

        pos = random.choice(pos_ids)
        neg = random.choice(neg_ids)

        triplets.append(
            InputExample(texts=[captions[i], captions[pos], captions[neg]])
        )

        if len(triplets) >= max_samples:
            break

    print(f"📦 Triplets built: {len(triplets)}")
    return triplets


# ================================================================
# 4. 主训练流程（使用 model.fit → 不报错）
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenegraph_desc", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./bge_adaptive")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load captions
    captions = load_captions(args.scenegraph_desc)

    # Load base BGE
    print("🚀 Loading BGE-large-en-v1.5 ...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    # Compute embeddings
    emb = embed(model, captions)

    # Build triplets
    triplets = build_triplets(captions, emb)
    if not triplets:
        print("❌ No triplets constructed.")
        return

    # DataLoader
    train_loader = DataLoader(
        triplets,
        batch_size=args.batch_size,
        shuffle=True
    )

    train_loss = losses.TripletLoss(model)
    warmup = max(50, int(0.05 * len(train_loader) * args.epochs))

    print("\n🚀 Start training...\n")

    # --- 关键：使用 SentenceTransformer 内置训练 ---
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        output_path=args.outdir,
        show_progress_bar=True  # 一个进度条
    )

    print(f"\n💾 Saved → {args.outdir}")
    print("🎉 Done.")


if __name__ == "__main__":
    main()
