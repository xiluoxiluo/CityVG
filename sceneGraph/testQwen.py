#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CityVG — Candidate Generation using Fine-Tuned Qwen Spatial Scorer
==================================================================
整合功能：
1) 并行 Qwen3-max 抽取 caption 主题词（car/building/...）
2) 使用 微调后的 Qwen_scorer 计算 P(Yes)
3) 以 P(Yes) 排序生成候选集（替代 BGE cosine）

无需任何外部文件依赖。
"""

import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI


# ======================================================
# 0. Qwen3-max client（用于并行抽主题词）
# ======================================================
client = OpenAI(
    api_key="sk-0f59497ddded48xxxxxxxxxxxxxxxxxxxxxxxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_MAX_MODEL = "qwen3-max"


# ======================================================
# 1. 单条 caption 的主题词抽取
# ======================================================
def qwen_extract_topic(text: str) -> List[str]:
    """
    调用 Qwen3-max 预测 caption 的对象类别。
    返回 Python list，如 ['car']、['building'] 等。
    """

    prompt = (
        "Extract ONLY the object type from the text. "
        "Valid outputs: ['car'], ['building'], ['ground'], ['parking'].\n"
        "Return ONLY the Python list. No explanation.\n\n"
        f"Text: {text}\nObject type:"
    )

    try:
        completion = client.chat.completions.create(
            model=QWEN_MAX_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        raw = completion.choices[0].message.content.strip()

        # 提取 Python list 的内容
        s, e = raw.find("["), raw.rfind("]")
        if s != -1 and e != -1:
            res = eval(raw[s:e+1])
            if isinstance(res, list):
                return res
    except Exception as e:
        print("⚠ Qwen topic extract error:", e)

    return []


# ======================================================
# 2. 并行主题词抽取
# ======================================================
def batch_extract_topics(captions: List[str], max_workers=8) -> List[List[str]]:
    results = [None] * len(captions)

    def worker(idx, txt):
        return idx, qwen_extract_topic(txt)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, i, cap): i for i, cap in enumerate(captions)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Qwen3-max Topic Extract"):
            idx, res = future.result()
            results[idx] = res

    return results


# ======================================================
# 3. Topic match（用于过滤 car/building 等）
# ======================================================
def topic_match(query_entities: List[str], obj_name: str) -> bool:
    q = set([x.lower() for x in query_entities])
    obj = obj_name.lower()
    base = {"car", "building", "ground", "parking"}

    if q & base:
        return obj in q
    return True


# ======================================================
# 4. 加载微调后的 Qwen 空间 scorer
# ======================================================
def load_qwen_scorer(model_dir):
    print(f"🔥 Loading Qwen scorer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    return tokenizer, model


# ======================================================
# 5. 构造与训练一致的 prompt
# ======================================================
def build_prompt(tokenizer, anchor, cand):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a spatial reasoning model for city-scale 3D visual grounding. "
                "Judge whether the candidate describes the same object and spatial relation as the reference. "
                "Answer only Yes or No."
            )
        },
        {
            "role": "user",
            "content": (
                f"Reference description:\n{anchor}\n\n"
                f"Candidate relation/object:\n{cand}\n\n"
                "Question: Does the candidate refer to the same object and spatial relation?"
            )
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # 让模型续写 Yes/No
    )
    return tokenizer(text, return_tensors="pt")


# ======================================================
# 6. Qwen scorer 输出 P(Yes)
# ======================================================
def score_pair(tokenizer, model, anchor, cand, device="cuda"):
    with torch.inference_mode():
        enc = build_prompt(tokenizer, anchor, cand)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, -1, :]  # 最后一个 token 的概率分布

        yes_id = tokenizer(" Yes", add_special_tokens=False).input_ids[0]
        no_id  = tokenizer(" No", add_special_tokens=False).input_ids[0]

        probs = F.softmax(logits, dim=-1)
        return probs[0, yes_id].item()  # 只取 Yes 的概率


# ======================================================
# 7. 主流程
# ======================================================
def main():

    block = "birmingham_block_9"

    GRAPH_JSON = f"/home/zjj/Code/CityVG/data/cityrefer_graph/{block}_graph_captions.json"
    VAL_JSON   = f"/home/zjj/Code/CityVG/data/cityrefer_block/{block}.json"
    OUT_JSON   = f"/home/zjj/Code/CityVG/data/cityrefer_candidate/{block}_candidates_qwen.json"

    SCORER_DIR = "/home/zjj/Code/CityVG/qwen_spatial_scorer/epoch_1"

    # ----------------------------
    # Load data
    # ----------------------------
    print("📥 Loading graph captions...")
    graph = json.load(open(GRAPH_JSON, "r", encoding="utf-8"))

    print("📥 Loading val set...")
    val_data = json.load(open(VAL_JSON, "r", encoding="utf-8"))

    captions = [x["description"] for x in val_data]
    ids = [str(item["object_id"]) for item in graph]
    graph_caps = [" ".join(item["caption"]) for item in graph]

    # ----------------------------
    # Load fine-tuned Qwen scorer
    # ----------------------------
    tokenizer, scorer = load_qwen_scorer(SCORER_DIR)

    # ----------------------------
    # Parallel Qwen3-max topic extract
    # ----------------------------
    print("\n🚀 Extracting query topics using Qwen3-max ...\n")
    topic_list = batch_extract_topics(captions, max_workers=8)

    # ----------------------------
    # Candidate generation using Qwen scorer
    # ----------------------------
    print("\n🚀 Start evaluation with Qwen Spatial Scorer...\n")

    top1 = top5 = top10 = 0
    results = []

    for idx, item in enumerate(tqdm(val_data, desc="Evaluating")):

        query = item["description"]
        gtid = str(item["object_id"])
        q_topics = topic_list[idx]

        # ① topic filtering
        filtered_idx = []
        for i, obj_id in enumerate(ids):
            obj_name = graph[i]["object_name"]
            if topic_match(q_topics, obj_name):
                filtered_idx.append(i)
        if len(filtered_idx) == 0:
            filtered_idx = list(range(len(ids)))

        # ② Qwen scorer：计算 P(Yes)
        scores = []
        for gi in filtered_idx:
            cand = graph_caps[gi]
            p_yes = score_pair(tokenizer, scorer, query, cand)
            scores.append((p_yes, ids[gi]))

        # ③ 排序取 top-k
        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
        top_candidates = [x[1] for x in sorted_scores[:10]]

        # accuracy 统计
        if gtid == top_candidates[0]:
            top1 += 1
        if gtid in top_candidates[:5]:
            top5 += 1
        if gtid in top_candidates[:10]:
            top10 += 1

        results.append({
            "caption": query,
            "topics": q_topics,
            "GTID": gtid,
            "candidates": top_candidates,
            "scores": [float(x[0]) for x in sorted_scores[:10]]
        })

    # ----------------------------
    # Results
    # ----------------------------
    total = len(val_data)
    print("\n🎯 Accuracy Results")
    print(f"Top-1 Accuracy : {top1/total*100:.2f}%")
    print(f"Top-5 Accuracy : {top5/total*100:.2f}%")
    print(f"Top-10 Accuracy: {top10/total*100:.2f}%")

    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n💾 Saved → {OUT_JSON}")


if __name__ == "__main__":
    main()
