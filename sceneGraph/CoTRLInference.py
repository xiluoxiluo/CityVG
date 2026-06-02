#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
City-Scale 3DVG — 3-Stage CoT/RL Visual Grounding (Doubao Vision)
=================================================================

三阶段推理:
  Stage 1: Category Matching      (类别)
  Stage 2: Color & Appearance     (颜色+外观)
  Stage 3: Spatial Relations      (空间关系)

奖励函数:
  Reward = (CatScore + ColorScore + SpatialScore) / 3

RL-Refine:
  若 reward < 0.5 → 触发“Reflect-and-Refine”重推理

输出:
  results/cot_rl_doubao.json
"""

import os
import json
import base64
from tqdm import tqdm
from openai import OpenAI



# ==========================================================
# 1. 初始化 API
# ==========================================================
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="5314b178-d06c-454f-af7axxxxxxxxxxxxxxxxxxx",   # ← 换成你自己的
)
MODEL_NAME = "doubao-seed-1-6-vision-250815"



# ==========================================================
# 2. 工具函数：编码图片
# ==========================================================
def encode_image_to_base64(image_path: str):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")



# ==========================================================
# 3. 三阶段 CoT/RL Prompt
# ==========================================================
def build_cot_rl_prompt(caption: str, candidate_ids: list):

    id_list_str = ", ".join([str(x) for x in candidate_ids])

    return f"""
You are a 3-Stage CoT/RL Visual Grounding Model.

Your reasoning MUST strictly follow this ordered chain:

===========================
STAGE 1 — CATEGORY MATCHING
===========================
- Identify the object category inside the red bounding box 
  (e.g., car type such as sedan/SUV/van, building type, construction equipment, rooftop type).
- Use ONLY the red-boxed object (do not infer category from background or surrounding context).
- Reject candidates whose category is inconsistent with the caption description.

===========================
STAGE 2 — COLOR & APPEARANCE MATCHING
===========================
You must strictly follow the COLOR RULES:

COLOR RULES:
- The object inside the red bounding box is the ONLY target.
- Determine object color using BOTH:
    (1) raw RGB image (no red box)  
    (2) RGB image with red box  
- PRIORITIZE the raw image for color estimation.
- ONLY use pixels fully enclosed inside the red box.
- Do NOT use any occluded pixels or background pixels outside the box.
- Do NOT use pixels from semantic maps or medium/large-scale RGB images for color.
- Determine dominant color(s), then secondary colors.
- Reject candidates whose color/appearance do NOT match the caption.

===========================
STAGE 3 — SPATIAL RELATION MATCHING
===========================
You must strictly follow the SPATIAL RULES:

SPATIAL RULES:
- After color validation, use all five images (1 raw, 2 red-box, 3 medium, 4 large, 5 segmentation map)
  to infer spatial relations.
- Match relative geometry and spatial relationships, such as:
  • left / right  
  • in front of / behind  
  • between / next to  
  • row / column position within a parking lot  
  • relative layout of buildings, roads, and bridges  
  • placement within global scene context
- Choose the candidate whose spatial configuration best aligns with the caption.

===========================
SELF-REWARD CHECKER
===========================
CatScore    = 1 if category matches caption, else 0
ColorScore  = 1 if color matches caption, else 0
SpatialScore= 1 if spatial relations match caption, else 0

Reward = (CatScore + ColorScore + SpatialScore) / 3

If Reward < 0.5:
    → Perform REFLECT-AND-REFINE (重新推理)

===========================
OUTPUT FORMAT (STRICT JSON)
===========================
{{
  "best_id": "<one of [{id_list_str}]>",
  "reason": "<explain Stage1/Stage2/Stage3 reasoning>",
  "reward": <float 0~1>
}}

Caption: "{caption}"
""".strip()



# ==========================================================
# 4. API 调用 + RL 重推理逻辑
# ==========================================================
def cot_rl_query_once(caption: str, candidate_ids: list, img_root: str):
    prompt = build_cot_rl_prompt(caption, candidate_ids)

    content = [{"type": "text", "text": prompt}]
    real_ids = []

    for cid in candidate_ids:
        img_path = os.path.join(img_root, str(cid), "canvas.jpg")
        if not os.path.exists(img_path):
            continue
        real_ids.append(str(cid))
        content.append({"type": "text", "text": f"Candidate {cid}"})
        content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + encode_image_to_base64(img_path)}
        })

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content}],
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
        return parsed
    except Exception as e:
        print("⚠ API/JSON Error:", e)
        return None



def vlm_grounding_cotr_rl(caption: str, candidate_ids: list, img_root: str):

    result = cot_rl_query_once(caption, candidate_ids, img_root)

    # 若失败或 reward 太低 → 反思再推理
    if (result is None) or (float(result.get("reward", 0)) < 0.5):
        print("🔁 Reward < 0.5 → REFLECT-AND-REFINE")
        result = cot_rl_query_once(caption, candidate_ids, img_root)

    # 最终 fallback
    if result is None:
        return {"best_id": str(candidate_ids[0]), "reason": "fallback", "reward": 0}

    # 确保 best_id 合法
    best_id = str(result.get("best_id", ""))
    real_ids = [str(x) for x in candidate_ids]
    if best_id not in real_ids:
        result["best_id"] = real_ids[0]

    return result



# ==========================================================
# 5. 主流程：遍历 caption groups 进行评估
# ==========================================================
if __name__ == "__main__":

    CAND_JSON = "/home/zjj/Code/CityVG/results/birmingham_block_4_candidates_keyword_parallel.json"
    IMG_ROOT = "/home/zjj/Code/CityVG/data/cityrefer_preprocessed/birmingham_block_4"
    SAVE_PATH = "results/cot_rl_doubao.json"

    os.makedirs("results", exist_ok=True)

    items = json.load(open(CAND_JSON, "r", encoding="utf-8"))
    print(f"📦 Loaded {len(items)} caption groups.\n")

    results = []
    hit1 = hit5 = 0

    for item in tqdm(items, desc="🔎 CoT-RL Grounding", ncols=100):

        caption = item["caption"]
        cands = item["candidates"]
        gtid = item.get("GTID")

        print("\n===============================================")
        print("📝 Caption:", caption)
        print("📍 Candidates:", cands)
        print("🎯 GT:", gtid)

        res = vlm_grounding_cotr_rl(caption, cands, IMG_ROOT)

        res["caption"] = caption
        res["candidates"] = cands
        res["GTID"] = gtid

        print("🧠 Reason:", res.get("reason"))
        print("🎚 Reward:", res.get("reward"))
        print("🤖 Pred:", res["best_id"])

        # Hit@1
        if gtid and res["best_id"] == gtid:
            hit1 += 1
        
        # Hit@5
        if gtid and gtid in cands[:5]:
            hit5 += 1
        
        results.append(res)

    # 保存结果
    json.dump(results, open(SAVE_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    total = len(items)
    print("\n=========== FINAL METRICS ===========")
    print(f"🎯 Hit@1 = {hit1/total:.3f} ({hit1}/{total})")
    print(f"🎯 Hit@5 = {hit5/total:.3f} ({hit5}/{total})")
    print("=====================================\n")
