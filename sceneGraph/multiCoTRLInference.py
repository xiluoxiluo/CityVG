#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
City-Scale 3DVG — Multi-Trajectory Best-of-M CoT/RL Visual Grounding (Doubao Vision)
====================================================================================

改进点：
1. Multi-Trajectory Best-of-M：
   - 对同一组候选生成 M 条 CoT 轨迹
   - 每条轨迹有自己的 (best_id, reward, reason)
   - 选 reward 最大的一条作为最终结果 (Best-of-M)

2. 视觉缓存框架：
   - VISION_CACHE_B64: 缓存每个候选 cid 对应的 base64 图像，避免重复读盘
   - VISION_CACHE_EMB: 预留“共享视觉 embedding”接口（需要你按 Doubao 文档填充）

   当前版本默认仍然用 base64 图片传入 chat.completions，
   真正要“共享视觉 embedding、降低图像 token 消耗”，
   需要你调用 Doubao 的视觉 embedding / cache token 接口，并在下面 TODO 部分实现。

输出:
  results/cot_rl_doubao_bestofM_shared.json
"""

import os
import re
import json
import base64
import copy
from typing import List, Dict, Any, Optional

from tqdm import tqdm
from openai import OpenAI


# ==========================================================
# 0. 全局配置
# ==========================================================
# Multi-Trajectory 条数 (建议 3~5，太大会很慢)
N_TRAJ = 3

# Reward 低于该阈值时，标记为低置信度
REWARD_WARN_THRES = 0.5

# 是否启用“共享视觉 embedding”模式（需要你自己实现 get_vision_embedding_via_api）
USE_SHARED_EMBEDDING = False


# ==========================================================
# 1. 初始化 API
# ==========================================================
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="5314b178-d06c-454f-af7a-xxxxxxxxxxxxxxxxxxxxxx4",   # ← 换成你自己的 key
)
MODEL_NAME = "doubao-seed-1-6-vision-250815"


# ==========================================================
# 2. 视觉缓存结构
# ==========================================================
# 缓存每个 cid 对应的 base64 图片，避免重复读盘
VISION_CACHE_B64: Dict[int, str] = {}

# 预留：缓存每个 cid 对应的视觉 embedding（你接 Doubao 的 embedding API 时用）
VISION_CACHE_EMB: Dict[int, Any] = {}


def encode_image_to_base64(image_path: str) -> str:
    """本地图片 → base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_b64_from_cache(cid: int, img_root: str) -> Optional[str]:
    """
    读取/缓存候选 cid 对应图片的 base64。
    当前用于 chat.completions 的 image_url 传参。
    """
    if cid in VISION_CACHE_B64:
        return VISION_CACHE_B64[cid]

    img_path = os.path.join(img_root, str(cid), "canvas.jpg")
    if not os.path.exists(img_path):
        return None

    b64 = encode_image_to_base64(img_path)
    VISION_CACHE_B64[cid] = b64
    return b64


# ==========================================================
# 2.1 预留：视觉 embedding 接口（需要你根据 Doubao 文档实现）
# ==========================================================
def get_vision_embedding_via_api(image_path: str) -> Any:
    """
    TODO: 这里是“真正降低图像 token 消耗”的关键：
      - 如果 Doubao 提供视觉 embedding / cache token 的 API，
        你可以在这里调用 embedding 接口，将图片编码成 embedding 或 image_id。

      例如（伪代码，示意用，不能直接跑）：
        resp = client.embeddings.create(
            model="doubao-vision-embedding",
            input={"image": encode_image_to_base64(image_path)}
        )
        return resp.data[0].embedding

      或者：
        resp = client.files.create(...)
        return resp.id   # 后续在 chat.completions 中引用这个 id

      目前我不敢编造具体字段，你需要用你账号的官方文档来填。
    """
    raise NotImplementedError(
        "请根据 Doubao Vision 的 embedding/cache 文档，实现 get_vision_embedding_via_api()"
    )


def get_vision_embedding_from_cache(cid: int, img_root: str) -> Optional[Any]:
    """
    若开启 USE_SHARED_EMBEDDING，则在此获取/缓存视觉 embedding。
    """
    if cid in VISION_CACHE_EMB:
        return VISION_CACHE_EMB[cid]

    img_path = os.path.join(img_root, str(cid), "canvas.jpg")
    if not os.path.exists(img_path):
        return None

    emb = get_vision_embedding_via_api(img_path)
    VISION_CACHE_EMB[cid] = emb
    return emb


# ==========================================================
# 3. 安全 JSON 解析
# ==========================================================
def safe_json_loads(raw: str) -> Optional[Dict[str, Any]]:
    """
    尝试从模型输出中提取 JSON:
      - 先直接 json.loads
      - 若失败，截取第一个 {...} 再解析
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


# ==========================================================
# 4. 三阶段 CoT/RL Prompt
# ==========================================================
def build_cot_rl_prompt(caption: str, candidate_ids: List[int]) -> str:
    """
    构造 3-Stage CoT + Self-Reward + JSON 输出的 Prompt。
    每次调用对应一条 CoT 轨迹。
    """
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
CatScore     = 1 if category matches caption, else 0
ColorScore   = 1 if color matches caption, else 0
SpatialScore = 1 if spatial relations match caption, else 0

Reward = (CatScore + ColorScore + SpatialScore) / 3

You MUST compute Reward consistently with the above rules.

===========================
OUTPUT FORMAT (STRICT JSON)
===========================
Return ONLY a single JSON object, no extra text:

{{
  "best_id": "<one of [{id_list_str}]>",
  "reason": "<explain Stage1/Stage2/Stage3 reasoning>",
  "reward": <float between 0 and 1>
}}

Caption: "{caption}"
""".strip()


# ==========================================================
# 5. 单条 CoT 轨迹：一次 API 调用
# ==========================================================
def cot_rl_single_trajectory(
    caption: str,
    candidate_ids: List[int],
    img_root: str,
) -> Optional[Dict[str, Any]]:
    """
    为给定 caption + 候选集合，生成一条 CoT 轨迹 (Single Trajectory)。
    模型输出一个 JSON: {"best_id": "...", "reason": "...", "reward": 0.x}
    """

    prompt = build_cot_rl_prompt(caption, candidate_ids)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

    # 收集真实存在图片的 ids
    real_ids: List[str] = []

    for cid in candidate_ids:
        # 1) 若以后你实现了 embedding 缓存，可在这里调用 get_vision_embedding_from_cache
        if USE_SHARED_EMBEDDING:
            emb = get_vision_embedding_from_cache(cid, img_root)
            if emb is None:
                continue
            # ⚠ TODO：这里需要根据 Doubao 文档，填“如何在 chat.completions 里传入 embedding / cache_token”
            # 伪代码示例（不能直接跑）：
            # content.append({"type": "vision_embedding", "embedding": emb})
            # 暂时先 raise，提醒你这里没实现：
            raise NotImplementedError("请在 cot_rl_single_trajectory 内实现 embedding 传参逻辑")

        # 2) 默认：仍然用 base64 图片（只不过从缓存里取，避免重复读盘）
        else:
            b64 = get_image_b64_from_cache(cid, img_root)
            if b64 is None:
                continue
            real_ids.append(str(cid))
            content.append({"type": "text", "text": f"Candidate {cid}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + b64
                    },
                }
            )

    if not real_ids:
        return None

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content}],
        )
        raw = resp.choices[0].message.content
        parsed = safe_json_loads(raw)
        if not parsed:
            return None
        return parsed
    except Exception as e:
        print("⚠ API/JSON Error in single trajectory:", e)
        return None


# ==========================================================
# 6. Multi-Trajectory Best-of-M 推理
# ==========================================================
def vlm_grounding_cotr_bestofM(
    caption: str,
    candidate_ids: List[int],
    img_root: str,
    n_traj: int = N_TRAJ,
) -> Dict[str, Any]:
    """
    Multi-Trajectory Best-of-M:
      - 生成 n_traj 条 CoT 轨迹
      - 对每条轨迹解析出 best_id & reward
      - 选 reward 最大的作为最终结果
    """
    real_ids = [str(x) for x in candidate_ids]

    best_res: Optional[Dict[str, Any]] = None
    best_reward: float = -1.0
    all_trajs: List[Dict[str, Any]] = []

    for m in range(n_traj):
        print(f"   ↪ Trajectory {m+1}/{n_traj} ...")
        traj_res = cot_rl_single_trajectory(caption, candidate_ids, img_root)

        if traj_res is None:
            print("     ⚠ Trajectory result is None, skip.")
            continue

        # 解析 reward
        try:
            reward = float(traj_res.get("reward", 0.0))
        except Exception:
            reward = 0.0

        # 把当前轨迹记录下来（注意使用“轻量字典”，避免循环引用）
        all_trajs.append(
            {
                "traj_id": m,
                "best_id": traj_res.get("best_id"),
                "reward": reward,
                "reason": traj_res.get("reason", ""),
            }
        )

        # 记录当前最优轨迹
        if reward > best_reward:
            best_reward = reward
            best_res = traj_res

    # 若所有轨迹都失败，fallback
    if best_res is None:
        print("⚠ All trajectories failed. Fallback to first candidate.")
        return {
            "best_id": str(candidate_ids[0]),
            "reason": "fallback_all_failed",
            "reward": 0.0,
            "all_trajs": all_trajs,
        }

    # 确保 best_id 合法
    best_id = str(best_res.get("best_id", ""))
    if best_id not in real_ids:
        print(f"⚠ best_id={best_id} not in candidates, fallback to {real_ids[0]}")
        best_res["best_id"] = real_ids[0]

    # 加上 Multi-Trajectory 信息（用 copy 确保可序列化）
    best_res = copy.deepcopy(best_res)
    best_res["all_trajs"] = all_trajs
    best_res["R_final"] = best_reward
    best_res["low_confidence"] = (best_reward < REWARD_WARN_THRES)

    return best_res


# ==========================================================
# 7. 主流程：遍历 caption groups 进行评估
# ==========================================================
if __name__ == "__main__":

    CAND_JSON = "/home/zjj/Code/CityVG/results/birmingham_block_4_candidates_keyword_parallel.json"
    IMG_ROOT = "/home/zjj/Code/CityVG/data/cityrefer_preprocessed/birmingham_block_4"
    SAVE_PATH = "results/cot_rl_doubao_bestofM_shared.json"

    os.makedirs("results", exist_ok=True)

    items = json.load(open(CAND_JSON, "r", encoding="utf-8"))
    print(f"📦 Loaded {len(items)} caption groups.\n")
    print(f"🔁 Using Multi-Trajectory Best-of-{N_TRAJ} inference.")
    print(f"🧠 USE_SHARED_EMBEDDING = {USE_SHARED_EMBEDDING}\n")

    results: List[Dict[str, Any]] = []
    hit1 = hit5 = 0

    for item in tqdm(items, desc="🔎 CoT-RL Best-of-M Grounding", ncols=100):

        caption = item["caption"]
        cands = item["candidates"]
        gtid = item.get("GTID")

        print("\n===============================================")
        print("📝 Caption:", caption)
        print("📍 Candidates:", cands)
        print("🎯 GT:", gtid)

        res = vlm_grounding_cotr_bestofM(caption, cands, IMG_ROOT, n_traj=N_TRAJ)

        res["caption"] = caption
        res["candidates"] = cands
        res["GTID"] = gtid

        print("🧠 Best Reason:", res.get("reason"))
        print("🎚 Best Reward (R_final):", res.get("R_final", res.get("reward")))
        print("🤖 Pred:", res["best_id"])
        if res.get("low_confidence"):
            print("⚠ Low-confidence prediction (Reward < {:.2f})".format(REWARD_WARN_THRES))

        # Hit@1
        if gtid and res["best_id"] == gtid:
            hit1 += 1

        # Hit@5 依然按候选 Top-5 是否含 GT 统计
        if gtid and gtid in cands[:5]:
            hit5 += 1

        results.append(res)

    # 保存结果
    json.dump(
        results,
        open(SAVE_PATH, "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    total = len(items)
    print("\n=========== FINAL METRICS ===========")
    print(f"🎯 Hit@1 = {hit1/total:.3f} ({hit1}/{total})")
    print(f"🎯 Hit@5 = {hit5/total:.3f} ({hit5}/{total})")
    print("=====================================\n")
