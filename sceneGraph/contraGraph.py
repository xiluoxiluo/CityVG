#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import time
import logging
import argparse
from tqdm import tqdm
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ["SENTENCE_TRANSFORMERS_SHOW_PROGRESS_BAR"] = "false"
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import logging
logging.disable(logging.CRITICAL)


client = OpenAI(
    api_key="sk-0f59497xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_MODEL = "qwen3-max"

def qwen_extract_topic(text: str) -> List[str]:

    prompt = (
        "Extract ONLY the object type from the text. "
        "Valid outputs: ['car'], ['building'], ['ground'] or ['parking'].\n"
        "Return ONLY the Python list.\n\n"
        f"Text: {text}\nObject type:"
    )

    try:
        completion = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        raw = completion.choices[0].message.content.strip()

        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            data = eval(raw[start:end+1])
            if isinstance(data, list):
                return data

    except Exception as e:
        print("⚠ Qwen extract error:", e)

    return []


def batch_extract_topics(captions: List[str], max_workers=8) -> List[List[str]]:
    results = [None] * len(captions)

    def worker(idx, txt):
        return idx, qwen_extract_topic(txt)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, i, cap): i for i, cap in enumerate(captions)}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Qwen Topic Extract",
            leave=True
        ):
            idx, res = future.result()
            results[idx] = res

    return results

def topic_match(query_entities: List[str], obj_name: str) -> bool:
    q = set([x.lower() for x in query_entities])
    obj = obj_name.lower()
    base_types = {"car", "building", "ground", "parking"}

    if q & base_types:
        return obj in q
    return True

def load_graph(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--SCENE_ID", type=str, required=True)
    args = parser.parse_args()
    SCENE_ID = args.SCENE_ID

    GRAPH_JSON = f"/home/zjj/Code/CityVG/data/cityrefer_graph/{SCENE_ID}_graph_captions.json"
    VAL_JSON   = f"/home/zjj/Code/CityVG/data/cityrefer_block/{SCENE_ID}.json"
    OUT_JSON   = f"/home/zjj/Code/CityVG/data/cityrefer_candidate/{SCENE_ID}_candidates_qwen.json"
    FINETUNE_MODEL_DIR = f"/home/zjj/Code/CityVG/checkpoints/{SCENE_ID}_Fine-tuned_BGE"

    logging.basicConfig(level=logging.INFO)
    start = time.perf_counter()

    print("📥 Loading graph ...")
    graph = load_graph(GRAPH_JSON)

    print("📥 Loading validation set ...")
    with open(VAL_JSON, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    captions = [item["description"] for item in val_data]

    print("\n🚀 Parallel Qwen3-max topic extraction...\n")
    topic_list = batch_extract_topics(captions, max_workers=8)

    print("🔢 Loading fine-tuned BGE model ...")
    bge = SentenceTransformer(FINETUNE_MODEL_DIR) 

    corpus_texts = [" ".join(item["caption"]) for item in graph]
    ids = [str(item["target_id"]) for item in graph]

    print("🧮 Encoding corpus embeddings with fine-tuned BGE ...")
    with torch.inference_mode():
        corpus_emb = bge.encode(
            corpus_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    print("\n🚀 Start Evaluation...\n")

    top1 = top5 = top10 = 0
    results = []

    for idx, item in enumerate(tqdm(val_data, desc="Evaluating")):

        caption = item["description"]
        gtid = item["object_id"]
        q_topics = topic_list[idx]

        # ---------- Topic Filter ----------
        filtered_idx = []
        for i, tid in enumerate(ids):
            obj_name = graph[i]["object_name"]
            if topic_match(q_topics, obj_name):
                filtered_idx.append(i)

        if not filtered_idx:
            filtered_idx = list(range(len(ids)))

        with torch.inference_mode():
            q_emb = bge.encode(
                caption,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

        sub_emb = corpus_emb[filtered_idx]
        scores = util.cos_sim(q_emb, sub_emb)[0]

        topk_vals, topk_idx = torch.topk(scores, k=min(10, len(filtered_idx)))
        top_candidates = [ids[filtered_idx[int(i)]] for i in topk_idx]

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

    end = time.perf_counter()
    logging.info("Inference time: %.4f s", end - start)

    total = len(val_data)
    print("\n🎯 Accuracy Results")
    print(f"Top-1 Accuracy : {top1 / total * 100:.2f}%")
    print(f"Top-5 Accuracy : {top5 / total * 100:.2f}%")
    print(f"Top-10 Accuracy: {top10 / total * 100:.2f}%")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved → {OUT_JSON}")

    print("****************************************")
    print("STEP 8/9: Learn graph-aligned embeddings via contrastive optimization")
    print("****************************************")



if __name__ == "__main__":
    main()
