#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from typing import List
from tqdm import tqdm
from openai import OpenAI

# =========================
# Qwen API
# =========================
client = OpenAI(
    api_key="sk-0f59497ddded4858bxxxxxxxxxxxxxxxxxxxxxxxxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_MODEL = "qwen3-max"

# =========================
# Extract object category from caption
# =========================
def qwen_extract_category(text: str) -> List[str]:
    prompt = (
        "Extract ONLY the object category mentioned in the text.\n"
        "Valid outputs: ['car'], ['building'], ['ground'], ['parking'].\n"
        "If no clear category is mentioned, return an empty list.\n"
        "Return ONLY the Python list.\n\n"
        f"Text: {text}\nCategory:"
    )

    try:
        completion = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        raw = completion.choices[0].message.content.strip()
        start, end = raw.find("["), raw.rfind("]")
        if start != -1 and end != -1:
            return eval(raw[start:end + 1])
    except Exception as e:
        print("⚠ Qwen error:", e)

    return []

# =========================
# Category match
# =========================
def category_match(query_cats: List[str], obj_name: str) -> bool:
    if not query_cats:
        return True
    return obj_name.lower() in query_cats

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--SCENE_ID", type=str, required=True)
    args = parser.parse_args()
    SCENE_ID = args.SCENE_ID

    VAL_JSON = f"/home/zjj/Code/CityVG/data/cityrefer_block/{SCENE_ID}.json"
    INS_JSON = f"/home/zjj/Code/CityVG/data/cityrefer_graph/{SCENE_ID}_graph_captions.json"
    OUT_JSON = f"/home/zjj/Code/CityVG/data/cityrefer_candidate/{SCENE_ID}_candidates_category.json"

    print("📥 Loading validation data...")
    with open(VAL_JSON, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    print("📥 Loading instance list...")
    with open(INS_JSON, "r", encoding="utf-8") as f:
        instances = json.load(f)

    instance_ids = [str(x["target_id"]) for x in instances]
    instance_names = [x["object_name"].lower() for x in instances]

    top1 = top5 = top10 = 0
    results = []

    print("\n🚀 Start category-based filtering evaluation...\n")

    for item in tqdm(val_data, desc="Evaluating"):
        caption = item["description"]
        gtid = str(item["object_id"])

        query_cats = qwen_extract_category(caption)

        filtered = [
            iid for iid, name in zip(instance_ids, instance_names)
            if category_match(query_cats, name)
        ]

        if not filtered:
            filtered = instance_ids.copy()

        candidates = filtered[:10]

        # ---------- Accuracy ----------
        if gtid == candidates[0]:
            top1 += 1
        if gtid in candidates[:5]:
            top5 += 1
        if gtid in candidates:
            top10 += 1

        results.append({
            "caption": caption,
            "category": query_cats,
            "GTID": gtid,
            "candidates": candidates
        })

    total = len(val_data)

    print("\n🎯 Accuracy Results (Category-based Filtering)")
    print(f"Top-1 Accuracy : {top1 / total * 100:.2f}%")
    print(f"Top-5 Accuracy : {top5 / total * 100:.2f}%")
    print(f"Top-10 Accuracy: {top10 / total * 100:.2f}%")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved → {OUT_JSON}")
    print("****************************************")
    print("STEP X/9: Category-based instance filtering and evaluation")
    print("****************************************")


if __name__ == "__main__":
    main()
