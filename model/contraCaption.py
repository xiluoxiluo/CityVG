#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
City-Scale 3DVG (Unsupervised) — ULCL Pipeline
ULCL = Unsupervised Landmark-guided Contrastive Learning

功能:
1) 读取/合并 JSON: /home/zjj/Code/CityVG/data/birmingham_block_4_merged.json
2) 生成文本嵌入 (BGE)
3) 无监督聚类 -> 构造对比学习训练对 (簇内正样本)
4) 微调 SentenceTransformer 文本编码器
5) Qwen3-8B 语义摘要增强 + Top-K 候选检索
6) 输出准确率 (Top1/5/10)
7) 可选输出候选实例 JSON (用于 VLM)

依赖:
pip install -U sentence-transformers transformers scikit-learn torch tqdm
"""

import os
import re
import json
import argparse
import warnings
from typing import List, Dict, Tuple
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans


# -----------------------------
# 默认路径 & 参数
# -----------------------------
DEFAULT_JSON = "/home/zjj/Code/CityVG/results/birmingham_block_4_graph.json"
DEFAULT_OUT  = "/home/zjj/Code/CityVG/checkpoints/bge_ulcl"
DEFAULT_TOPVIEW_DIR = "/home/zjj/Code/CityVG/data/cityrefer_preprocessed/birmingham_block_4"


# -----------------------------
# 工具函数
# -----------------------------
def ensure_merged(json_path: str) -> List[Dict]:
    """合并/读取 JSON"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise ValueError("JSON 为空")

    if isinstance(data[0].get("caption", None), list):
        return data

    merged = {}
    for item in data:
        tid = str(item["target_id"])
        if tid not in merged:
            merged[tid] = {
                "scan_id": item["scan_id"],
                "target_id": tid,
                "obj_name": item["obj_name"],
                "caption": []
            }
        merged[tid]["caption"].append(item["caption"])
    return list(merged.values())


def extract_landmarks(text: str) -> List[str]:
    """抽取地标（大写短语）"""
    return re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b", text)


def build_corpus(merged_data: List[Dict]) -> Tuple[List[str], List[str], List[List[str]]]:
    """构建语料文本和ID映射"""
    corpus_texts, id_map, obj_lm_list = [], [], []
    for item in merged_data:
        merged_caption = " ".join(item["caption"])
        corpus_texts.append(merged_caption)
        id_map.append(str(item["target_id"]))
        lm_set = set()
        for c in item["caption"]:
            for lm in extract_landmarks(c):
                lm_set.add(lm)
        obj_lm_list.append(sorted(list(lm_set)))
    return corpus_texts, id_map, obj_lm_list


def ulcl_contrastive_finetune(
    captions_all: List[str],
    base_model_name: str = "BAAI/bge-large-en-v1.5",
    out_dir: str = DEFAULT_OUT,
    clusters: int = 50,
    batch_size: int = 32,
    epochs: int = 2,
    seed: int = 42
) -> SentenceTransformer:
    """无监督 ULCL 微调"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(base_model_name, device=device)

    with torch.inference_mode():
        Z = model.encode(captions_all, convert_to_tensor=False, normalize_embeddings=True, show_progress_bar=True)

    km = KMeans(n_clusters=clusters, random_state=seed, n_init="auto")
    labels = km.fit_predict(Z)

    train_pairs: List[InputExample] = []
    cluster_to_indices: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        cluster_to_indices.setdefault(lab, []).append(idx)

    for lab, idxs in cluster_to_indices.items():
        if len(idxs) < 2:
            continue
        for i in range(len(idxs) - 1):
            a = captions_all[idxs[i]]
            b = captions_all[idxs[i + 1]]
            train_pairs.append(InputExample(texts=[a, b]))

    if len(train_pairs) == 0:
        warnings.warn("⚠️ 聚类未形成有效正样本对，跳过微调。")
        return model

    train_loader = DataLoader(train_pairs, shuffle=True, batch_size=batch_size, drop_last=False)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        warmup_steps=max(10, int(0.05 * len(train_loader))),
        show_progress_bar=True,
        output_path=out_dir
    )

    tuned = SentenceTransformer(out_dir, device=device)
    return tuned


def rephrase_with_qwen3(text: str, model_name: str = "Qwen/Qwen3-8B", max_new_tokens: int = 96) -> str:
    """用 Qwen3-8B 摘要增强描述"""
    tok = AutoTokenizer.from_pretrained(model_name)
    qwen = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto").eval()
    prompt = (
        "Rephrase the following description into a concise, unambiguous spatial scene summary. "
        "Keep proper-noun landmarks and clear relations like near/behind/on corner of.\n"
        f"Input: {text}\nSummary:"
    )
    inputs = tok(prompt, return_tensors="pt").to(qwen.device)
    with torch.no_grad():
        out = qwen.generate(**inputs, max_new_tokens=max_new_tokens)
    summary = tok.decode(out[0], skip_special_tokens=True)
    parts = summary.split("Summary:")
    summary = parts[-1].strip() if len(parts) >= 2 else summary.strip()
    return summary


def landmark_weight(score: float, query_text: str, object_captions: List[str], w_match: float = 1.2) -> float:
    """地标一致性加权"""
    q_lms = set(extract_landmarks(query_text))
    if not q_lms:
        return score
    obj_lms = set()
    for c in object_captions:
        obj_lms.update(extract_landmarks(c))
    if q_lms & obj_lms:
        return score * w_match
    return score


# -----------------------------
# 主流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="City-Scale 3DVG ULCL Pipeline with Accuracy Evaluation")
    parser.add_argument("--json", type=str, default=DEFAULT_JSON)
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUT)
    parser.add_argument("--clusters", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--lm_weight", action="store_true")
    parser.add_argument("--skip_ulcl", action="store_true")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3-8B")
    args = parser.parse_args()

    print("📥 读取/合并 JSON ...")
    merged = ensure_merged(args.json)
    print(f"✅ 对象数: {len(merged)}")

    print("🧱 构建语料 ...")
    corpus_texts, id_map, _ = build_corpus(merged)
    captions_all = [c for item in merged for c in item["caption"]]
    print(f"📝 Caption 条数(ULCL): {len(captions_all)}")

    # 1️⃣ ULCL 微调
    base_model_name = "BAAI/bge-large-en-v1.5"
    if args.skip_ulcl:
        print("⏭ 跳过 ULCL 微调，使用原始 BGE")
        text_model = SentenceTransformer(base_model_name)
    else:
        print("🚀 开始 ULCL 无监督对比学习微调 ...")
        text_model = ulcl_contrastive_finetune(
            captions_all, base_model_name, args.outdir,
            args.clusters, args.batch, args.epochs
        )
        print(f"💾 微调模型已加载: {args.outdir}")

    # 2️⃣ 编码语料
    print("🧮 编码对象语料 ...")
    print(corpus_texts)
    with torch.inference_mode():
        corpus_embeddings = text_model.encode(
            corpus_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
        )

    # 3️⃣ 读取验证集
    cityrefer_json = "/home/zjj/Code/CityVG/data/CityRefer_val_ND_birmingham_block_4.json"
    print(f"\n📖 读取 CityRefer JSON: {cityrefer_json}")
    with open(cityrefer_json, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    test_samples = [
        {"caption": item["caption"], "GTID": str(item["target_id"]), "scan_id": item["scan_id"], "obj_name": item["obj_name"]}
        for item in val_data if "caption" in item and "target_id" in item and "scan_id" in item and "obj_name" in item
    ]
    print(f"✅ 样本数: {len(test_samples)}\n")

    # 4️⃣ 检索评估
    results, top1, top5, top10 = [], 0, 0, 0
    total = len(test_samples)
    print("🧩 开始检索评估...\n")

    for sample in tqdm(test_samples, desc="Evaluating", ncols=100):
        caption, target_id, scan_id, obj_name = sample["caption"], sample["GTID"], sample["scan_id"], sample["obj_name"]
        summary = rephrase_with_qwen3(caption, model_name=args.qwen_model)

        with torch.inference_mode():
            query_emb = text_model.encode(summary, convert_to_tensor=True, normalize_embeddings=True)
            raw_scores = util.cos_sim(query_emb, corpus_embeddings)[0].cpu()

        if args.lm_weight:
            weighted = [
                landmark_weight(float(s), summary, item["caption"], 1.2)
                for s, item in zip(raw_scores, merged)
            ]
            scores = torch.tensor(weighted)
        else:
            scores = raw_scores

        K = min(args.topk, len(id_map))
        topk_vals, topk_idx = torch.topk(scores, k=K)
        top_candidates = [id_map[int(i)] for i in topk_idx]

        results.append({
            "scan_id":scan_id,
            "caption": caption,
            "summary": summary,
            "GTID": target_id,
            "obj_name": obj_name,
            "candidates": top_candidates
        })

        if target_id == top_candidates[0]:
            top1 += 1
        if target_id in top_candidates[:5]:
            top5 += 1
        if target_id in top_candidates[:10]:
            top10 += 1

    # 5️⃣ 输出准确率
    print("\n🎯 Evaluation Results:")
    print(f"Top-1 Accuracy : {top1 / total * 100:.2f}%")
    print(f"Top-5 Accuracy : {top5 / total * 100:.2f}%")
    print(f"Top-10 Accuracy: {top10 / total * 100:.2f}%")

    # 6️⃣ 保存结果
    out_json = "/home/zjj/Code/CityVG/results/birmingham_block_4_candidates.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 候选结果已保存 → {out_json}")
    print("✅ 可直接用于后续 VLM grounding 阶段\n")


if __name__ == "__main__":
    main()
