#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
City-Scale 3DVG Visual Grounding via DashScope Qwen3-VL-Plus API
---------------------------------------------------------------
输入:
  /home/zjj/Code/CityVG/data/birmingham_block_4_candidates.json
输出:
  results/qwen3vl_plus_grounding_birmingham.json

功能:
  - 一次性输入全部候选图像
  - 模型直接输出 best_id
  - 输出理由 + Hit@1 / Hit@5 统计
"""

import os
import json
import base64
from tqdm import tqdm
from openai import OpenAI


# ======================================================
# 1️⃣ 初始化 DashScope API
# ======================================================
client = OpenAI(
    api_key="sk-2dacb4566bxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL_NAME = "qwen3-vl-plus"



# ======================================================
# 2️⃣ 工具函数
# ======================================================
def encode_image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")



# ======================================================
# 3️⃣ 一次性输入全部候选图像（核心函数）
# ======================================================
def vlm_grounding_api(caption: str, candidate_ids: list, img_root: str):
    """
    输入 caption + 所有 candidate_ids
    输出:
        {
            "best_id": <candidate_id>,
            "reason": "..."
        }
    """

    if not candidate_ids:
        raise ValueError("candidate_ids 为空！")

    # ⛳ 强制模型必须从这些 ID 里选
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

You MUST output ONLY JSON:
{{
  "best_id": "<one of [{id_list_str}]>",
  "reason": "<short explanation>"
}}

Caption: "{caption}"
""".strip()

    # ====== 构造 multimodal content ======
    user_content = []
    user_content.append({"type": "text", "text": user_prompt})

    real_candidates = []

    for cid in candidate_ids:
        img_path = os.path.join(img_root, str(cid), "canvas.jpg")
        if not os.path.exists(img_path):
            print(f"⚠ Missing: {img_path}")
            continue

        b64img = encode_image_to_base64(img_path)
        real_candidates.append(str(cid))

        # 明确告诉模型这张图属于哪个 ID
        user_content.append({
            "type": "text",
            "text": f"Candidate ID = {cid}"
        })
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64img}",
                "detail": "high"
            }
        })

    # ======= 调用 DashScope =======
    messages = [
        {"role": "system", "content": "You are a helpful multimodal reasoning assistant."},
        {"role": "user", "content": user_content}
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            extra_body={"enable_thinking": True, "thinking_budget": 8192}
        )
        raw = completion.choices[0].message.content
    except Exception as e:
        print("❌ API Error:", e)
        return {"best_id": real_candidates[0], "reason": "API fallback"}

    # ======= JSON 输出解析 =======
    try:
        parsed = json.loads(raw)
        best_id = str(parsed["best_id"])
        reason = parsed.get("reason", "")
    except Exception:
        print("⚠ JSON Parse Failed:", raw)
        return {"best_id": real_candidates[0], "reason": "parse fallback"}

    # double check
    if best_id not in real_candidates:
        print(f"⚠ Model returned unknown id {best_id}, fallback to {real_candidates[0]}")
        best_id = real_candidates[0]

    return {"best_id": best_id, "reason": reason}



# ======================================================
# 4️⃣ 主流程（保持你的结构，完全不动）
# ======================================================
if __name__ == "__main__":

    CAND_JSON = "/home/zjj/Code/CityVG/data/birmingham_block_4_candidates.json"
    IMG_ROOT = "/home/zjj/Code/CityVG/data/cityrefer_preprocessed/birmingham_block_4"
    SAVE_PATH = "results/qwen3vl_plus_grounding_birmingham.json"

    os.makedirs("results", exist_ok=True)

    with open(CAND_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"📦 Loaded {len(items)} caption groups.\n")

    results = []
    hit1 = hit5 = 0
    K = 5

    for item in tqdm(items, desc="🔎 DashScope Grounding", ncols=100):

        caption = item["caption"]
        cands = item["candidates"]
        gtid = str(item.get("GTID"))

        print("\n" + "=" * 80)
        print("📝 Caption:", caption)
        print("Candidates:", cands)
        print("GTID:", gtid)

        res = vlm_grounding_api(caption, cands, IMG_ROOT)
        best_id = res["best_id"]

        print("🤖 Pred:", best_id)
        print("🧠 Reason:", res["reason"])

        # Accuracy
        if gtid:
            if best_id == gtid:
                hit1 += 1
                print("✅ Hit@1 correct")
            else:
                print("❌ Hit@1 wrong")

            if gtid in cands[:K]:
                hit5 += 1

        res.update({
            "caption": caption,
            "candidates": cands,
            "GTID": gtid,
        })
        results.append(res)

    # 保存
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(items)
    print("\n🎯 Final Accuracy:")
    print(f"Hit@1 = {hit1/total:.3f}")
    print(f"Hit@5 = {hit5/total:.3f}")
    print("Saved →", SAVE_PATH)
