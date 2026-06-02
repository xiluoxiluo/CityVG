#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract FIRST color phrase in caption using Qwen3-VL-Max
--------------------------------------------------------
输入 : scenegraph_with_descriptions.json
输出 : scenegraph_with_descriptions_color.json
新增字段:  "color": "<first color phrase>"
"""

import os
import json
from openai import OpenAI
from tqdm import tqdm


# ============================================================
# 1. DashScope Qwen3-VL-Max
# ============================================================
client = OpenAI(
    api_key="sk-2dacb4566b084a1xxxxxxxxxxxxxxxxxxx",  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL = "qwen3-vl-plus"


# ============================================================
# 2. 抽取 caption 中的第一个颜色表达
# ============================================================
def extract_color_qwen(caption: str) -> str:
    """
    使用 Qwen3-VL-Max 抽取 caption 中第一次出现的“颜色词或颜色短语”
    返回一个 Python 字符串，例如:
        "bright blue"
        "dark silver metallic"
        "red"
        "none"
    """

    prompt = f"""
Extract the FIRST color word or color phrase mentioned in the text.

Definition:
- A color can be any natural-language color expression such as:
  "blue", "bright red", "dark metallic silver", "light grey", etc.
- If multiple colors appear, return ONLY the first one in reading order.
- If no color appears, return "none".

Return Rules:
1. Return ONLY a Python string, for example:
       "bright blue"
       "red"
       "none"
2. Do NOT add explanations.
3. Do NOT add any extra characters.

Text: {caption}

Color:
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )

        raw = resp.choices[0].message.content.strip()

        # delete surrounding quotes if present
        if raw.startswith('"') and raw.endswith('"'):
            color = raw.strip('"')
        else:
            color = raw

        return color if color else "none"

    except Exception as e:
        print("⚠ Qwen color extract error:", e)
        return "none"


# ============================================================
# 3. 主流程
# ============================================================
def main():

    IN_JSON  = "./scenegraph_with_descriptions.json"
    OUT_JSON = "./scenegraph_with_descriptions_color.json"

    print(f"📥 Loading {IN_JSON} ...")
    with open(IN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    edges = data["edges"]

    print(f"🎨 Extracting color phrases using Qwen3-VL-Max ...\n")

    for e in tqdm(edges, desc="Processing edges"):
        caption = e.get("caption", "")
        color_phrase = extract_color_qwen(caption)
        e["color"] = color_phrase

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved with 'color' field → {OUT_JSON}")


if __name__ == "__main__":
    main()
