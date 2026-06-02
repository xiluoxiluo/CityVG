#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
City-Scale 3DVG (Keyword + BGE) Pipeline — Parallel Qwen3-max Version
---------------------------------------------------------------------

并行调用 DashScope Qwen3-max 来抽取 caption 主题词（car/building）
替换你原来的 Qwen3-8B（本地模型），速度提升巨大。
"""

import os
import json
import torch
from tqdm import tqdm
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from sentence_transformers import SentenceTransformer, util
from openai import OpenAI


# ======================================================
# Qwen3-max Client（DashScope OpenAI-Compatible）
# ======================================================
client = OpenAI(
    api_key="sk-0f59497ddded4858b33d2xxxxxxxxxxxxxxxxxxxxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_MODEL = "qwen3-max"


# ======================================================
# 单条 caption 的主题词抽取
# ======================================================
def qwen_extract_topic(text: str) -> List[str]:
    """
    调用 Qwen3-max 来抽取主题词（car / building）
    返回 Python list
    """

    prompt = (
        "Extract ONLY the object type from the text. "
        "Valid outputs are strictly: ['car'] or ['building'] or ['ground'] or ['parking'].\n"
        "Return ONLY the Python list. Do NOT explain. Do NOT add any extra text.\n\n"
        f"Text: {text}\n"
        "Object type:"
    )

    try:
        completion = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False   # ❗检索任务不能使用流式
        )

        raw = completion.choices[0].message.content.strip()

        # 提取 Python list
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            data = eval(raw[start:end+1])
            if isinstance(data, list):
                return data

    except Exception as e:
        print("⚠ Qwen extract error:", e)

    return []


# ======================================================
# 并行主题词抽取（核心）
# ======================================================
def batch_extract_topics(captions: List[str], max_workers=8) -> List[List[str]]:
    results = [None] * len(captions)

    def worker(idx, txt):
        return idx, qwen_extract_topic(txt)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, i, cap): i for i, cap in enumerate(captions)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Qwen Topic Extract"):
            idx, res = future.result()
            results[idx] = res

    return results


# ======================================================
# 主题词过滤逻辑（原样保留）
# ======================================================
def topic_match(query_entities: List[str], obj_name: str) -> bool:
    q = set([x.lower() for x in query_entities])
    obj = obj_name.lower()

    base_types = {"car", "building","ground","parking"}

    if q & base_types:
        return obj in q

    return True


# ======================================================
# JSON 加载
# ======================================================
def load_graph(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================================================
# 主流程
# ======================================================
def main():

    block_name = "birmingham_block_9"
    out_json = os.path.join("/home/zjj/Code/CityVG/data/cityrefer_candidate", f"{block_name}_candidates.json")

    GRAPH_JSON = os.path.join("/home/zjj/Code/CityVG/data/cityrefer_graph", f"{block_name}_graph_captions.json") 
    VAL_JSON = os.path.join("/home/zjj/Code/CityVG/data/cityrefer_block", f"{block_name}.json") 

    print("📥 Loading graph ...")
    graph = load_graph(GRAPH_JSON)

    print("📥 Loading validation set ...")
    with open(VAL_JSON, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    captions = [item["description"] for item in val_data]

    # ============================================================
    # 1) 并行 Qwen3-max 抽主题词
    # ============================================================
    print("\n🚀 Parallel Qwen3-max topic extraction...\n")
    topic_list = batch_extract_topics(captions, max_workers=8)

    # ============================================================
    # 2) 加载 BGE
    # ============================================================
    print("🔢 Loading BGE-large-en-v1.5 ...")
    bge = SentenceTransformer("BAAI/bge-large-en-v1.5")

    corpus_texts = [" ".join(item["caption"]) for item in graph]
    ids = [str(item["object_id"]) for item in graph]

    print("🧮 Encoding corpus with BGE...")
    with torch.inference_mode():
        corpus_emb = bge.encode(
            corpus_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    # ============================================================
    # 3) 检索 Evaluation
    # ============================================================
    print("\n🚀 Start Evaluation...\n")

    top1 = top5 = top10 = 0
    results = []

    for idx, item in enumerate(tqdm(val_data, desc="Evaluating")):

        caption = item["description"]
        gtid = str(item["object_id"])
        q_topics = topic_list[idx]      # ← 并行 Qwen 结果

        # 过滤 Car / Building
        filtered_idx = []
        for i, tid in enumerate(ids):
            obj_name = graph[i]["object_name"]
            if topic_match(q_topics, obj_name):
                filtered_idx.append(i)

        if len(filtered_idx) == 0:  # fallback
            filtered_idx = list(range(len(ids)))

        # Query embedding
        with torch.inference_mode():
            q_emb = bge.encode(caption, convert_to_tensor=True, normalize_embeddings=True)

        # 子集检索
        sub_emb = corpus_emb[filtered_idx]
        scores = util.cos_sim(q_emb, sub_emb)[0]

        topk_vals, topk_idx = torch.topk(scores, k=min(10, len(filtered_idx)))
        top_candidates = [ids[filtered_idx[int(i)]] for i in topk_idx]

        # Accuracy
        if gtid == top_candidates[0]:
            top1 += 1
        if gtid in top_candidates[:5]:
            top5 += 1
        if gtid in top_candidates[:10]:
            top10 += 1

        results.append({
            "caption": caption,
            "topics": q_topics,
            "GTID": gtid,
            "candidates": top_candidates
        })

    # 输出 accuracy
    total = len(val_data)
    print("\n🎯 Accuracy Results")
    print(f"Top-1 Accuracy : {top1 / total * 100:.2f}%")
    print(f"Top-5 Accuracy : {top5 / total * 100:.2f}%")
    print(f"Top-10 Accuracy: {top10 / total * 100:.2f}%")

    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved → {out_json}")


if __name__ == "__main__":
    main()
