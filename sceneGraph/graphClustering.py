#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

import re
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

def load_captions(scenegraph_path):
    with open(scenegraph_path, "r", encoding="utf-8") as f:
        sg = json.load(f)

    if not isinstance(sg, list):
        raise ValueError("❌ Scenegraph must be a list of objects!")

    caps = []
    for e in sg:
        cap_list = e.get("caption", None)

        if isinstance(cap_list, list) and len(cap_list) > 0:
            cap = cap_list[0].strip()
            if cap:
                caps.append(cap)

    print(f"📦 Loaded {len(caps)} captions (from list-style scenegraph).")
    return caps

def embed(model, sentences):
    print("🔢 Computing BGE embeddings...")
    with torch.inference_mode():
        emb = model.encode(
            sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
    return emb

def load_llm_qwen3():
    print("🚀 Loading Qwen/Qwen3-8B ...")
    model_name = "Qwen/Qwen3-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    return tokenizer, llm

def llm_batch_score_qwen3(tokenizer, llm, batch_pairs):

    prompts = []
    for cap, lm in batch_pairs:

        prompt = (
            "<system>\n"
            "You are a strict semantic similarity evaluator.\n"
            "You must output ONLY a JSON object and NO explanations.\n"
            "The required format is: {\"score\": X} where X is an integer from 0 to 10.\n"
            "\n"
            "Your evaluation must consider THREE factors:\n"
            "1) CATEGORY MATCH: Is the object's category consistent with what usually belongs to the landmark?\n"
            "2) LANDMARK SEMANTICS: Is the object logically part of, associated with, or located at this landmark?\n"
            "3) COLOR / APPEARANCE MATCH: Does the object's color or appearance fit the landmark-related objects?\n"
            "You must integrate all three factors into ONE final similarity score.\n"
            "</system>\n\n"

            "<input>\n"
            f"Landmark Name: {lm}\n"
            f"Object Caption: {cap}\n"
            "</input>\n\n"

            "<task>\n"
            "Evaluate how semantically relevant this Object is to the Landmark.\n"
            "The score must reflect category consistency, landmark relevance, and color/appearance similarity.\n"
            "Return an integer from 0 to 10.\n"
            "STRICTLY output only: {\"score\": X}\n"
            "</task>\n"
        )

        prompts.append(prompt)

    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(llm.device)

    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    scores = []
    for text in decoded:
        m = re.search(r'\"score\"\s*:\s*(\d+)', text)
        if m:
            s = int(m.group(1))
            scores.append(min(10, max(0, s)))
        else:
            scores.append(0)
    return scores

def build_triplets_LDEC(scenegraph_path, captions, emb_matrix, max_samples=60000):

    print("🏙️ LDEC: Building Triplets for your scenegraph format...")

    with open(scenegraph_path, "r", encoding="utf-8") as f:
        sg = json.load(f)

    # FIXED: sg is list
    if not isinstance(sg, list):
        raise ValueError("❌ Expected scenegraph to be a list!")

    lm2obj = {}
    obj_caps = {}

    for e in sg:
        lm = e.get("landmark_name", "").strip()
        oid = e.get("object_id", "").strip()
        cap_list = e.get("caption", None)

        if not lm or not oid:
            continue

        # FIXED: caption list
        if isinstance(cap_list, list) and len(cap_list) > 0:
            cap = cap_list[0].strip()
        else:
            continue

        if not cap:
            continue

        lm2obj.setdefault(lm, []).append(oid)
        obj_caps[oid] = cap

    print(f"📍 Landmarks detected: {len(lm2obj)}")
    print(f"📦 Valid objects with captions: {len(obj_caps)}")

    tokenizer, llm = load_llm_qwen3()

    landmark_clusters = {}

    print("🧠 Running Qwen3 semantic scoring...")

    for lm, obj_ids in lm2obj.items():
        batch_pairs = [(obj_caps[oid], lm) for oid in obj_ids]
        scores = llm_batch_score_qwen3(tokenizer, llm, batch_pairs)

        C_L = []
        for oid, s in zip(obj_ids, scores):
            if s >= 6:
                C_L.append(oid)

        landmark_clusters[lm] = C_L

    print("🧱 Building Triplets ...")

    obj_all = list(obj_caps.keys())
    triplets = []

    for lm, C_L in landmark_clusters.items():
        if len(C_L) < 2:
            continue

        neg_pool = [oid for oid in obj_all if oid not in C_L]

        for oid in C_L:
            anchor = obj_caps[oid]

            pos_oid = random.choice([x for x in C_L if x != oid])
            positive = obj_caps[pos_oid]

            neg_oid = random.choice(neg_pool)
            negative = obj_caps[neg_oid]

            triplets.append(InputExample(texts=[anchor, positive, negative]))

            if len(triplets) >= max_samples:
                break

    print(f"🎯 Triplets built: {len(triplets)}")
    return triplets
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--SCENE_ID", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    SCENE_ID = args.SCENE_ID
    scenegraph_desc = f"/home/zjj/Code/CityVG/data/cityrefer_graph/{SCENE_ID}_graph_captions.json"
    outdir = f"/home/zjj/Code/CityVG/checkpoints/{SCENE_ID}_Fine-tuned_BGE"

    os.makedirs(outdir, exist_ok=True)

    captions = load_captions(scenegraph_desc)

    print("🚀 Loading BGE-large-en-v1.5 ...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    emb = embed(model, captions)

    triplets = build_triplets_LDEC(scenegraph_desc, captions, emb)
    if not triplets:
        print("❌ No triplets constructed.")
        return

    train_loader = DataLoader(triplets, batch_size=args.batch_size, shuffle=True)

    train_loss = losses.TripletLoss(model)
    warmup = max(50, int(0.05 * len(train_loader) * args.epochs))

    print("\n🚀 Start training...\n")

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        output_path=outdir,
        show_progress_bar=True
    )

    print(f"\n💾 Saved → {outdir}")
    print("****************************************")
    print("STEP 7/9: Perform graph-aware clustering for contrastive supervision")
    print("****************************************")

if __name__ == "__main__":
    main()
