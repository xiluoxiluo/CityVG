#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CityVG — Qwen3-8B Spatial Scorer (adapted to graph_captions.json)
-----------------------------------------------------------------
使用 scenegraph_desc（如 birmingham_block_9_graph_captions.json）中的 caption：
- 先用 BGE 构造 (anchor, pos, neg) 伪 triplet
- 再构造 (anchor, candidate, label in {Yes/No}) pair
- 用这些 pair 微调 Qwen3-8B，作为 “空间一致性打分器”

训练后：
- 给定 (query_caption, candidate_caption)
- Qwen 输出 P(Yes)，作为空间语义一致性 score
"""

import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)

from sentence_transformers import SentenceTransformer


# ================================================================
# 1. 加载带 caption 的 scenegraph_desc
#    适配两种格式：
#    (a) dict + "edges"（老版本）
#    (b) list，每个元素含 "caption"（graph_captions 格式）
# ================================================================
def load_captions(scenegraph_path, min_words: int = 5):
    """
    从 scenegraph_desc 中抽取 caption 文本列表。

    - 对于你现在的 graph_captions.json：
      顶层是 list，每个元素有一个 caption(list[str])
      我们会把 list 里的多句 join 成一个字符串。
    - 过滤掉空 caption，以及词数太少的 caption（可通过 min_words 控制）。
    """
    with open(scenegraph_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    captions = []

    # Case 1: 旧格式：{"nodes": ..., "edges": [...]}，每个 edge 里有 "caption"
    if isinstance(data, dict) and "edges" in data:
        for e in data["edges"]:
            cap = e.get("caption", "")
            # 兼容 caption 为 string 或 list 的情况
            if isinstance(cap, list):
                cap_text = " ".join([c.strip() for c in cap if isinstance(c, str)])
            else:
                cap_text = str(cap).strip()

            if not cap_text:
                continue
            if len(cap_text.split()) < min_words:
                continue
            captions.append(cap_text)

    # Case 2: 新格式：顶层就是 list（如 birmingham_block_9_graph_captions.json）
    elif isinstance(data, list):
        for item in data:
            cap = item.get("caption", "")
            # 你的文件中 caption 是 list[str]
            if isinstance(cap, list):
                cap_text = " ".join([c.strip() for c in cap if isinstance(c, str)])
            else:
                cap_text = str(cap).strip()

            if not cap_text:
                continue
            if len(cap_text.split()) < min_words:
                # 太短的就不拿来训练，对 Qwen 学习没什么帮助
                continue
            captions.append(cap_text)
    else:
        raise ValueError(
            f"Unsupported scenegraph_desc format: type={type(data)}; "
            f"expect dict(with 'edges') or list of items with 'caption'."
        )

    # 去重一下（可选）
    captions = list(dict.fromkeys(captions))

    print(f"📦 Loaded {len(captions)} captions from {scenegraph_path}")
    return captions


# ================================================================
# 2. 用 BGE 计算向量（只用于构造伪 triplet，不参与最终推理）
# ================================================================
def embed_bge(sentences):
    print("🔢 Computing BGE embeddings for pseudo labels...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    with torch.inference_mode():
        emb = model.encode(
            sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
    return emb


def build_triplets(captions, emb_matrix, max_samples=30000):
    """
    和你原来的 BGE 脚本类似：
    - 对每个 caption i，根据相似度矩阵选高相似作为 positive，低相似作为 negative
    - 构造 (anchor, pos, neg)
    """
    n = len(captions)
    print("🧱 Building pseudo triplets (anchor, pos, neg)...")
    sim_matrix = torch.matmul(emb_matrix, emb_matrix.T).cpu().numpy()

    triplets = []

    for i in tqdm(range(n), desc="Triplet build"):
        sims = sim_matrix[i]

        pos_th = np.quantile(sims, 0.60)
        neg_th = np.quantile(sims, 0.10)

        pos_ids = [k for k in np.where(sims > pos_th)[0] if k != i]
        neg_ids = [k for k in np.where(sims < neg_th)[0] if k != i]

        if not pos_ids or not neg_ids:
            continue

        pos = random.choice(pos_ids)
        neg = random.choice(neg_ids)

        triplets.append((captions[i], captions[pos], captions[neg]))

        if len(triplets) >= max_samples:
            break

    print(f"📦 Triplets built: {len(triplets)}")
    return triplets


# ================================================================
# 3. Triplet → pair：(anchor, candidate, label in {1,0})
# ================================================================
def build_pair_dataset(triplets):
    """
    每个 triplet (a, p, n) → 两个 pair：
      (a, p, 1) 正样本
      (a, n, 0) 负样本
    """
    pairs = []
    for anchor, pos, neg in triplets:
        pairs.append((anchor, pos, 1))
        pairs.append((anchor, neg, 0))
    random.shuffle(pairs)
    print(f"📦 Pair samples: {len(pairs)}")
    return pairs


# ================================================================
# 4. Dataset & collate_fn（使用 Qwen chat_template + Yes/No）
# ================================================================
class QwenPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor, cand, label = self.pairs[idx]
        return {
            "anchor": anchor,
            "candidate": cand,
            "label": int(label),
        }


def make_collate_fn(tokenizer, max_length=512):
    """
    把 (anchor, candidate, label) 组装成 Qwen chat：

    system: 你是城市级 3DVG 的空间推理模型……
    user:   Reference description: ...
            Candidate relation/object: ...
            Question: ...
    assistant: Yes / No

    然后整体做 LM loss（简化实现；后续可以改成只在 answer 段上打 loss）。
    """
    yes_token = "Yes"
    no_token = "No"

    def collate(batch):
        texts = []
        for item in batch:
            anchor = item["anchor"]
            cand = item["candidate"]
            label = item["label"]

            answer = yes_token if label == 1 else no_token

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a spatial reasoning model for city-scale 3D visual grounding. "
                        "Given a reference description and a candidate relation/object description, "
                        "you must judge whether they refer to the same spatial configuration. "
                        "Answer with a single word: Yes or No."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Reference description:\n{anchor}\n\n"
                        f"Candidate relation/object:\n{cand}\n\n"
                        "Question: Does the candidate describe the same object and spatial relation as the reference?"
                    )
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # 简化版：对整个序列做 LM loss
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate


# ================================================================
# 5. 主训练入口
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenegraph_desc", type=str, required=True,
                        help="如 birmingham_block_9_graph_captions.json")
    parser.add_argument("--outdir", type=str, default="./qwen_spatial_scorer")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_samples", type=int, default=30000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--min_words", type=int, default=5,
                        help="过滤掉词数少于 min_words 的 caption")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Using device: {device}")

    # 1. Load captions（已经适配 graph_captions.json 格式）
    captions = load_captions(args.scenegraph_desc, min_words=args.min_words)
    if len(captions) < 10:
        print("❌ Captions too few, check your scenegraph_desc.")
        return

    # 2. 用 BGE 构造伪 triplets
    emb = embed_bge(captions)
    triplets = build_triplets(captions, emb, max_samples=args.max_samples)
    if not triplets:
        print("❌ No triplets constructed.")
        return

    pairs = build_pair_dataset(triplets)
    dataset = QwenPairDataset(pairs)

    # 3. Load Qwen3-8B
    print("🚀 Loading Qwen/Qwen3-8B ...")
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.train()

    # 减显存（可选）
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    collate_fn = make_collate_fn(tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(100, int(0.05 * num_training_steps)),
        num_training_steps=num_training_steps,
    )

    print("\n🚀 Start Qwen spatial scorer training...\n")

    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch+1} avg loss: {epoch_loss / len(train_loader):.4f}")

        # 每个 epoch 存一次
        save_dir = os.path.join(args.outdir, f"epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"💾 Model saved to {save_dir}")

    print("\n🎉 Training finished.")


if __name__ == "__main__":
    main()
