import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import json
import os

# ========== 1️⃣ 加载模型 ==========
model_name = "Qwen/Qwen3-8B"
print("🚀 Loading Qwen3-8B model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
).eval()
print("✅ Qwen3-8B model loaded successfully.\n")

# === 加载文本相似度模型（BGE） ===
print("🚀 Loading SentenceTransformer for text similarity...")
text_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
print("✅ BGE model loaded.\n")

# ========== 2️⃣ 读取 JSON ==========
json_path = "/home/zjj/Code/CityVG/data/birmingham_block_4_merged.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ========== 3️⃣ 准备数据库向量 ==========
corpus = []
id_map = []

for item in data:
    merged_caption = " ".join(item["caption"])
    corpus.append(merged_caption)
    id_map.append(item["target_id"])

print(f"📦 Loaded {len(corpus)} merged objects from {json_path}\n")

# 计算语义向量
corpus_embeddings = text_model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)

# ========== 4️⃣ 评估全部样本 ==========
top1, top5, top10 = 0, 0, 0
total = len(data)

print("🚀 Starting evaluation...\n")

for item in tqdm(data, desc="Evaluating", ncols=100):
    gt_id = item["target_id"]
    caption = " ".join(item["caption"])

    # 用 Qwen 生成摘要增强
    prompt = f"Rephrase the following description to a concise spatial scene summary:\n{caption}\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Summary:")[-1].strip()

    # 向量化
    query_emb = text_model.encode(summary, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(query_emb, corpus_embeddings)[0]

    # top-k索引
    topk = torch.topk(scores, k=10)
    topk_ids = [id_map[i] for i in topk.indices.tolist()]

    # 统计正确率
    if gt_id == topk_ids[0]:
        top1 += 1
    if gt_id in topk_ids[:5]:
        top5 += 1
    if gt_id in topk_ids[:10]:
        top10 += 1

# ========== 5️⃣ 输出整体准确率 ==========
print("\n🎯 Evaluation Results:")
print(f"Top-1 Accuracy:  {top1 / total * 100:.2f}%")
print(f"Top-5 Accuracy:  {top5 / total * 100:.2f}%")
print(f"Top-10 Accuracy: {top10 / total * 100:.2f}%")
