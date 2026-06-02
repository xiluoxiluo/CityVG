#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SceneGraph Caption Generator (Qwen3-VL-Plus, CityRefer-style)
-------------------------------------------------------------
最终版本：输入 block_9_graph.json，输出 block_4_graph.json 格式文件
- prompt 原封不动
- 多线程 caption 生成
- 不生成中间文件，只输出最终 birmingham_block_4_graph.json
"""

import os
import json
import base64
import argparse
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_PATH = "/home/zjj/Code/CityVG/data/cityrefer_preprocessed"

parser = argparse.ArgumentParser()
parser.add_argument("--SCENE_ID", type=str, required=True)
args = parser.parse_args()
SCENE_ID = args.SCENE_ID
SCENEGRAPH = f"/home/zjj/Code/CityVG/data/cityrefer_graph/{SCENE_ID}_graph.json"
OUTPUT_FINAL = f"/home/zjj/Code/CityVG/data/cityrefer_graph/{SCENE_ID}_graph_captions.json"

IMG_ROOT = f"{BASE_PATH}/{SCENE_ID}"


# ============================================================
# DashScope Qwen3-VL-Plus
# ============================================================
client = OpenAI(
    api_key="sk-0f59497ddded4858b33dxxxxxxxxxxxxxxxxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL_NAME = "qwen3-vl-plus"


def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_qwen(messages) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        extra_body={"enable_thinking": True, "thinking_budget": 4096},
    )
    return completion.choices[0].message.content.strip()


with open(SCENEGRAPH, "r", encoding="utf-8") as f:
    sg = json.load(f)

nodes_dict = {n["id"]: n for n in sg["nodes"]}
edges = sg["edges"]

print(f"📦 Loaded {len(nodes_dict)} nodes, {len(edges)} edges.\n")

def process_edge(e):
    obj_id = e["from"]
    lm_id  = e["to"]

    node_obj = nodes_dict[obj_id]
    node_lm  = nodes_dict[lm_id]

    object_name   = node_obj.get("object_name", "")
    landmark_name = node_lm.get("landmark_name", "")
    is_landmark   = node_obj.get("is_landmark", False)

    obj_raw   = f"{IMG_ROOT}/{obj_id}/canvas_raw_5.jpg"
    obj_stack = f"{IMG_ROOT}/{obj_id}/canvas.jpg"
    lm_raw    = f"{IMG_ROOT}/{lm_id}/canvas_raw_5.jpg"
    lm_stack  = f"{IMG_ROOT}/{lm_id}/canvas.jpg"

    if not (os.path.exists(obj_raw) and os.path.exists(obj_stack)):
        return None

    if is_landmark:

        raw_b64   = encode_image_base64(obj_raw)
        stack_b64 = encode_image_base64(obj_stack)

        prompt = f"""
You are generating a short caption for a landmark in a city-scale 3D visual grounding dataset.

The landmark is called "{landmark_name}".

CAPTION STYLE (CityRefer-like):
- Start the description naturally, e.g. "The grey building ..." or "The brown industrial building ...".
- Mention:
  • the landmark's approximate color (roof or facade),
  • its basic shape or structure (e.g., rectangular building, L-shaped building, large warehouse),
  • its context if visible (e.g., next to a parking lot, at a street corner, along a main road).

RULES:
- Describe ONLY the landmark itself and its immediate surroundings.
- Do NOT mention bounding boxes, markers, panels, or multi-view imagery.
- Do NOT hallucinate content that is not clearly visible.

Output ONE short paragraph (1–2 sentences) in natural CityRefer style.
""".strip()

        messages = [
            {"role": "system", "content": "You are a helpful multimodal caption assistant for city-scale visual grounding."},
            {"role": "user", "content": [
                {"type": "image_url","image_url":{"url":f"data:image/jpeg;base64,{raw_b64}"}} ,
                {"type": "image_url","image_url":{"url":f"data:image/jpeg;base64,{stack_b64}"}} ,
                {"type": "text", "text": prompt},
            ]},
        ]

        caption = call_qwen(messages)
        caption = caption.replace("the landmark", landmark_name)
        caption = caption.replace("The landmark", landmark_name)

        e["caption"] = caption
        return e

    if not (os.path.exists(lm_raw) and os.path.exists(lm_stack)):
        return None

    obj_raw_b64   = encode_image_base64(obj_raw)
    obj_stack_b64 = encode_image_base64(obj_stack)
    lm_raw_b64    = encode_image_base64(lm_raw)
    lm_stack_b64  = encode_image_base64(lm_stack)

    prompt = f"""
You are generating a CityRefer-style scene grounding caption for an object named "{object_name}"
and its spatial relation to the landmark "{landmark_name}".

IMAGE INPUTS:
- Two multi-scale BEV image composites: one for the landmark and one for the object.
- The object and the landmark are clearly visible in their highest-resolution views.

CAPTION STYLE (CityRefer-like):
Generate a natural descriptive paragraph following this structural pattern as closely as possible:

"The {{color}} {object_name} {{local_position_phrase}}, {{local_relation_phrase}},
in or near the parking or road area of/behind/next to {landmark_name}{{optional_aerial_phrase}}."

Where:
- {{color}}: the main color of the object, inferred ONLY from the object itself.
- {{local_position_phrase}}: describe the local placement of the object,
  e.g. "at the end of the row", "in the far right corner", "second from the left",
  "in the top row of cars", etc.
- {{local_relation_phrase}}: relation to nearby visible objects,
  e.g. "between a white car and a black car",
  "next to another red car",
  "behind a white van", etc.
- {{optional_aerial_phrase}} (optional): a short aerial reference phrase,
  e.g. "when viewed from above with Birchfield Road on the right",
  "near the edge of the map", or similar.

CONTENT RULES:
1. Sentence 1: describe ONLY the object:
   - object type (car or building, following "{object_name}"),
   - main color,
   - local position in the parking or road layout,
   - nearby vehicles or structures immediately around it.
2. Sentence 2: describe its relation to "{landmark_name}"
   with a simple phrase, such as:
   "in the parking lot of {landmark_name}",
   "behind {landmark_name}",
   "next to {landmark_name}",
   or "near {landmark_name}".
3. Optionally, you may add a very short third phrase about aerial orientation.

STRICT RULES:
- Do NOT mention bounding boxes, red boxes, markers, regions, or multi-view panels.
- Do NOT refer to "the highlighted area" or similar.
- Do NOT talk about semantic maps or model views.
- Do NOT hallucinate objects that are not clearly visible.

Output ONE short paragraph (2–3 sentences) in natural CityRefer style.
""".strip()

    messages = [
        {"role": "system","content":"You are a helpful multimodal caption assistant for city-scale visual grounding."},
        {"role": "user","content":[
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{lm_raw_b64}"}} ,
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{lm_stack_b64}"}} ,
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{obj_raw_b64}"}} ,
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{obj_stack_b64}"}} ,
            {"type":"text","text":prompt},
        ]},
    ]

    caption = call_qwen(messages)
    caption = caption.replace("the landmark", landmark_name)
    caption = caption.replace("The landmark", landmark_name)

    e["caption"] = caption
    return e

new_edges = []

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(process_edge, e): e for e in edges}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Generating captions", ncols=90):
        r = future.result()
        if r is not None:
            new_edges.append(r)

sg_cap_json = {
    "nodes": sg["nodes"],
    "edges": new_edges,
}

print("\n✨ Captions generated and stored in RAM.\n")

print("========== Caption Replacement Stage ==========\n")

edge_caps = {}
for e in sg_cap_json["edges"]:
    from_id = str(e["from"])
    cap = e.get("caption", "")
    if cap:
        edge_caps[from_id] = [cap] if isinstance(cap, str) else cap

print(f"✔ Loaded {len(edge_caps)} captions.\n")

items = []
for e in sg["edges"]:
    obj_id = e["from"]
    lm_id = e["to"]
    node_obj = nodes_dict[obj_id]
    node_lm = nodes_dict[lm_id]

    items.append({
        "target_id": str(obj_id),
        "object_id": str(obj_id),
        "landmark_id": str(lm_id),
        "object_name": node_obj.get("object_name", ""),
        "landmark_name": node_lm.get("landmark_name", ""),
        "caption": [],
        "bbox": e.get("bbox", None)
    })

updated = []
replace_count = 0

for item in items:
    tid = str(item["target_id"])

    if tid in edge_caps:
        item["caption"] = edge_caps[tid]
        replace_count += 1

    updated.append(item)

with open(OUTPUT_FINAL, "w", encoding="utf-8") as f:
    json.dump(updated, f, indent=2, ensure_ascii=False)

print(f"🎉 Final file saved → {OUTPUT_FINAL}")

with open(OUTPUT_FINAL, "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned = []
for item in data:
    cap = item.get("caption", None)

    if cap is None:
        continue
    if isinstance(cap, list) and len(cap) == 0:
        continue
    if isinstance(cap, str) and cap.strip() == "":
        continue

    cleaned.append(item)
    
with open(OUTPUT_FINAL, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

print("****************************************")
print("STEP 6/9: Generate graph-aware textual descriptions for instances")
print("****************************************")
