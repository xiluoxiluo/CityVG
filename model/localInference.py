#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
City-Scale 3DVG Grounding via Qwen3-VL-8B-Instruct (Official Compatible)
特点：
  - 基于官方 API 接口结构 (Qwen3VLForConditionalGeneration + AutoProcessor)
  - 显存优化 (dtype="auto" + flash_attention_2)
  - 完全兼容多模态输入格式
"""

import os
import json
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# =========================
# 1️⃣ 模型初始化
# =========================
model_name = "Qwen/Qwen3-VL-8B-Instruct"
print(f"🚀 Loading {model_name} ...")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# ✅ 官方推荐配置：自动 dtype + 多 GPU 支持 + FlashAttention
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    #attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)
print("✅ Model loaded successfully.\n")

# =========================
# 2️⃣ 数据路径
# =========================
CANDIDATES_JSON = "/home/zjj/Code/CityVG/data/birmingham_block_4_candidates.json"
IMG_ROOT = "/home/zjj/Code/CityVG/data/cityrefer_preprocessed/birmingham_block_4"

with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
    items = json.load(f)
print(f"📦 Loaded {len(items)} caption groups.\n")


# =========================
# 3️⃣ 工具函数
# =========================
def load_image(path, max_size=512):
    """加载图片并缩放以节省显存"""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size))
    return img


def qwen3_infer(messages, max_new_tokens=128):
    """基于官方 API 的统一推理接口"""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
        output_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    torch.cuda.empty_cache()
    return output_text


def describe_candidates(caption, candidate_ids):
    """逐个候选生成视觉描述"""
    descs = {}
    for tid in candidate_ids:
        img_path = os.path.join(IMG_ROOT, str(tid), "canvas.jpg")
        if not os.path.exists(img_path):
            continue

        img = load_image(img_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": (
                        f"Describe the object inside the red bounding box in one concise English sentence. "
                        f"Focus on color, shape, and surroundings. Caption context: '{caption}'."
                    )},
                ],
            }
        ]
        desc = qwen3_infer(messages, max_new_tokens=64)
        descs[tid] = desc
    return descs


def vlm_grounding_with_descriptions(caption, candidate_ids):
    """结合描述进行匹配推理"""
    descs = describe_candidates(caption, candidate_ids)
    desc_text = "\n".join([f"- Candidate {tid}: {desc}" for tid, desc in descs.items()])

    imgs = []
    for tid in candidate_ids:
        img_path = os.path.join(IMG_ROOT, str(tid), "canvas.jpg")
        if os.path.exists(img_path):
            imgs.append(load_image(img_path))

    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "text", "text": (
                    "You are a visual grounding assistant.\n\n"
                    "Each candidate image shows the same region with a red box highlighting one object.\n"
                    f"Descriptions:\n{desc_text}\n\n"
                    "Task: Given the caption, choose which candidate best matches the described object.\n"
                    'Return ONLY JSON: {"best_image_index": <int>, "reason": "<brief explanation>"}\n'
                    f'Caption: "{caption}"'
                )}] +
                [{"type": "image", "image": img} for img in imgs]
            ),
        }
    ]

    out_text = qwen3_infer(messages, max_new_tokens=256)

    try:
        match = re.search(r"\{.*\}", out_text, re.S)
        parsed = json.loads(match.group(0)) if match else {}
        idx = int(parsed.get("best_image_index", 0))
        reason = parsed.get("reason", "")
    except Exception as e:
        print("⚠️ JSON parse failed:", e)
        idx, reason = 0, "fallback"

    best_tid = candidate_ids[idx] if 0 <= idx < len(candidate_ids) else candidate_ids[0]
    return {"caption": caption, "best_id": best_tid, "reason": reason, "descriptions": descs}


# =========================
# 4️⃣ 主推理循环
# =========================
results, hit1, hitk = [], 0, 0
K = 5

for item in tqdm(items, desc="🔎 Grounding", ncols=100):
    cap, cands, gtid = item["caption"], item["candidates"], item.get("GTID")

    print("\n" + "=" * 80)
    print(f"📝 Caption: {cap}")
    print(f"📍 Candidates: {cands}")
    if gtid:
        print(f"🎯 Ground Truth: {gtid}")

    res = vlm_grounding_with_descriptions(cap, cands)
    print("\n🖼️ Candidate Descriptions:")
    for tid, desc in res["descriptions"].items():
        print(f"   - {tid}: {desc}")

    print(f"\n🤖 Predicted: {res['best_id']}")
    print(f"🧠 Reason: {res['reason']}")

    res["GTID"] = gtid
    results.append(res)

    if gtid and res["best_id"] == gtid:
        hit1 += 1
        print("✅ Hit@1")
    if gtid and gtid in cands[:K]:
        hitk += 1

    torch.cuda.empty_cache()


# =========================
# 5️⃣ 保存结果 + 精度
# =========================
os.makedirs("results", exist_ok=True)
out_path = "results/qwen3vl_8b_grounding.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

total = len(items)
print(f"\n💾 Saved → {out_path}")
print(f"🎯 Hit@1 = {hit1/total:.3f}, Hit@{K} = {hitk/total:.3f} ({hit1}/{total})\n")
