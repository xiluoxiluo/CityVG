#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import base64
import time
import logging
import argparse
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="5314b178-d06c-454f-afxxxxxxxxxxxxxxxxxx",
)

MODEL_NAME = "doubao-seed-1-6-vision-250815"

def encode_image_to_base64(image_path: str):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def vlm_grounding_api(caption: str, candidate_ids: list, img_root: str):

    if not candidate_ids:
        raise ValueError("⚠ NO candidate_ids")

    id_list_str = ", ".join([str(x) for x in candidate_ids])

    user_prompt = f"""
You are a visual grounding assistant.

Each candidate consists of five vertically stacked images:
1) raw top-down RGB image (no red box),
2) the same RGB image with a red bounding box marking the target object,
3) medium-scale top-down RGB,
4) large-scale top-down RGB,
5) semantic segmentation map.

COLOR RULES:
- The object inside the red bounding box is the ONLY target.
- Determine object color using BOTH the raw image (1) and red-box image (2).
- PRIORITIZE the raw image for color.
- ONLY use pixels fully enclosed inside the red box.
- Do NOT use occluded pixels or background pixels outside the box.
- Do NOT use semantic map or larger-scale RGB for color.

SPATIAL RULES:
- After color validation, use all five images to infer spatial relations,
  such as relative location, orientation, nearby roads or buildings.

Your task:
Choose the candidate whose red-boxed object best matches the described object.

You MUST return ONLY valid JSON:
{{
  "best_id": "<one of [{id_list_str}]>",
  "reason": "<short explanation>"
}}

Caption: "{caption}"
""".strip()

    user_content = []
    user_content.append({"type": "text", "text": user_prompt})

    real_candidates = []

    for cid in candidate_ids:
        img_path = os.path.join(img_root, str(cid), "canvas.jpg")

        if not os.path.exists(img_path):
            print("⚠ Missing:", img_path)
            continue

        b64 = encode_image_to_base64(img_path)
        real_candidates.append(str(cid))

        user_content.append({"type": "text", "text": f"Candidate ID = {cid}"})

        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
        })

    messages = [
        {"role": "user", "content": user_content}
    ]
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        raw = completion.choices[0].message.content
    except Exception as e:
        print("❌ API error:", e)
        return {"best_id": real_candidates[0], "reason": "API fallback"}

    try:
        parsed = json.loads(raw)
        best_id = str(parsed["best_id"])
        reason = parsed.get("reason", "")
    except Exception:
        print("⚠ JSON parse failed:", raw)
        return {"best_id": real_candidates[0], "reason": "parse fallback"}

    if best_id not in real_candidates:
        print("⚠ Model returned unknown id:", best_id)
        best_id = real_candidates[0]

    return {"caption": caption, "best_id": best_id, "reason": reason}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--SCENE_ID", type=str, required=True)
    args = parser.parse_args()
    SCENE_ID = args.SCENE_ID

    CANDIDATES_JSON = os.path.join("/home/zjj/Code/CityVG/data/cityrefer_candidate", f"{SCENE_ID}_candidates_qwen.json")
    IMG_ROOT = os.path.join("/home/zjj/Code/CityVG/data/cityrefer_preprocessed/", SCENE_ID)
    SAVE_PATH = os.path.join("/home/zjj/Code/CityVG/data/cityrefer_inference", f"{SCENE_ID}_inference.json") 

    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"📦 Loaded {len(items)} caption groups.\n")

    results = []
    hit1, hitk = 0, 0
    K = 5

    logging.basicConfig(level=logging.INFO)
    start = time.perf_counter()

    for item in tqdm(items, desc="🔎 Doubao Vision Grounding", ncols=100):
        cap = item["caption"]
        cands = item["candidates"]
        gtid = item.get("GTID")

        print("\n" + "=" * 80)
        print("📝 Caption:", cap)
        print("📍 Candidates:", cands)
        print("🎯 GT:", gtid)

        res = vlm_grounding_api(cap, cands, IMG_ROOT)
        res["GTID"] = gtid
        res["candidates"] = cands
        results.append(res)

        print("🤖 Pred:", res["best_id"])
        print("🧠 Reason:", res["reason"])

        if gtid:
            if res["best_id"] == gtid:
                hit1 += 1
                print("✅ Hit@1 correct!")
            else:
                print("❌ Hit@1 wrong.")
            if gtid in cands[:K]:
                hitk += 1

    end = time.perf_counter()
    logging.info("Inference time: %.4f s", end - start)
    
    total = len(items)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n💾 Saved →", SAVE_PATH)
    print(f"🎯 Hit@1 = {hit1/total:.3f}, Hit@{K} = {hitk/total:.3f} ({hit1}/{total})\n")

    print("****************************************")
    print("STEP 9/9: Perform graph-based visual grounding inference with VLM")
    print("****************************************")

