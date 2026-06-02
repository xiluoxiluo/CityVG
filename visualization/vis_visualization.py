#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌆 CityVG Interactive Visualization (Final Stream Edition)
---------------------------------------------------------
✅ Caption → Candidate → Grounding → 3D Render (viser)
✅ 📜 Grounding Log 实时流式输出
✅ 自动以预测 bbox 中心为原点
"""

import os
import sys
import torch
import numpy as np
# ==============================
# ⚠️ FIX: psutil race condition for viser
# ==============================
import psutil

_original_process_iter = psutil.process_iter

def safe_process_iter(*args, **kwargs):
    try:
        return list(_original_process_iter(*args, **kwargs))
    except (FileNotFoundError, ProcessLookupError):
        # Process disappeared during iteration (Linux /proc race)
        return []

psutil.process_iter = safe_process_iter

import viser
import gradio as gr
import threading
import time
import socket
import json

# === 导入子模块 ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vis_genCandidates import generate_vis_candidates
from vis_mainEvaluate import run_vis_grounding

# ==============================
# 1️⃣ 数据路径
# ==============================
DATA_ROOT = "/home/zjj/Code/CityVG/data/data_cityrefer/sensaturban/pointgroup_data/balance_split/random-50_crop-250"


# ==============================
# 2️⃣ 工具函数
# ==============================
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def load_scene(scene_name):
    """加载点云"""
    pth_path = os.path.join(DATA_ROOT, f"{scene_name}.pth")
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"⚠️ Scene file not found: {pth_path}")

    data = torch.load(pth_path, map_location="cpu", weights_only=False)
    coords = np.array(data[0])
    colors = np.array(data[1])
    instance_bboxes = data[6]
    pts = np.concatenate([coords, colors], axis=1)
    bboxes = list(instance_bboxes.values()) if isinstance(instance_bboxes, dict) else instance_bboxes
    return pts, bboxes


def center_point_cloud(pts):
    center = np.mean(pts[:, :3], axis=0)
    centered_pts = pts[:, :3] - center
    centered_pts = np.hstack([centered_pts, pts[:, 3:6]])
    return centered_pts, center


# ==============================
# 3️⃣ viser 封装
# ==============================
class CityVGViserServer:
    def __init__(self, port=8080):
        self.port = port
        self.server = viser.ViserServer(port=port)
        print(f"🌐 viser server started at http://localhost:{port}")

    def clear_scene(self):
        self.server.scene.reset()

    def show_point_cloud(self, pts):
        points = pts[:, :3].astype(np.float32)
        colors = pts[:, 3:6].astype(np.uint8)
        self.server.scene.add_point_cloud(
            name="/scene_points",
            points=points,
            colors=colors,
            point_size=0.02,
            point_shape="circle",
        )

    def show_bbox(self, bbox, names, color=(255, 0, 0)):
        x, y, z, dx, dy, dz = bbox[:6]
        self.server.scene.add_box(
            name=f"/bbox_{names}",
            position=(x, y, z),
            dimensions=(dx, dy, dz),
            color=color,
            opacity=0.25,
            flat_shading=True,
            side="double",
            material="standard",
            visible=True,
        )


# ==============================
# 4️⃣ 全局状态
# ==============================
server_instance = None
current_scene = None
current_pts = None
current_bboxes = None
local_ip = get_local_ip()


# ==============================
# 5️⃣ 可视化加载函数
# ==============================
def visualize_scene(scene_name):
    global server_instance, current_scene, current_pts, current_bboxes

    if server_instance is None:
        server_instance = CityVGViserServer(port=8080)

    pts, bboxes = load_scene(scene_name)
    centered_pts, center = center_point_cloud(pts)
    current_scene, current_pts, current_bboxes = scene_name, pts, bboxes

    server_instance.clear_scene()
    server_instance.show_point_cloud(centered_pts)

    log_html = f"""
    <div style="border:2px solid #ccc;border-radius:10px;padding:10px;background:#fff;">
        <h4>📜 Scene Information</h4>
        <div style="height:100px;overflow:auto;padding:6px;line-height:1.5;">
            <p>🌇 Scene: <b>{scene_name}</b></p>
            <p>✅ Loaded {len(pts)} points and {len(bboxes)} bounding boxes.</p>
            <p>📍 Center: {center.round(2).tolist()}</p>
        </div>
    </div>
    """

    iframe_html = f"""
    <div style='border:2px solid #ccc;border-radius:10px;padding:10px;background:#fafafa;margin-top:10px;'>
        <h4 style='margin:0;'>🛰️ 3D Viewer (viser)</h4>
        <iframe src="http://{local_ip}:8080" width="100%" height="600"
                frameborder="0" style="border-radius:8px;margin-top:8px;"></iframe>
    </div>
    """
    return log_html, iframe_html


# ==============================
# 6️⃣ Grounding 流式执行
# ==============================
def run_grounding(scene_name, caption):
    """执行完整 grounding 流程（流式输出 + 📜Grounding Log + Pred/GT BBox 对齐居中）"""
    global server_instance, current_pts, current_bboxes

    if server_instance is None or current_pts is None:
        yield "<p>⚠️ Please visualize a scene first!</p>", ""
        return

    merged_json = f"/home/zjj/Code/CityVG/visualization/{scene_name}_merged.json"
    cand_json = f"/home/zjj/Code/CityVG/visualization/vis_{scene_name}_candidates.json"
    result_json = f"/home/zjj/Code/CityVG/visualization/vis_{scene_name}_results.json"
    img_root = f"/home/zjj/Code/CityVG/data/cityrefer_preprocessed/{scene_name}"
    os.makedirs(os.path.dirname(result_json), exist_ok=True)

    # === 初始化 Grounding Log ===
    log_html = """
    <div style="border:2px solid #ccc;border-radius:10px;padding:10px;background:#fff;">
      <h4>📜 Grounding Log</h4>
      <div id="log_box" style="height:100px;overflow:auto;padding:6px;line-height:1.5;">
    """

    def add_log(msg):
        nonlocal log_html
        log_html += f"<p>{msg}</p>"
        return log_html + "</div></div>"

    def iframe_view():
        return f"""
        <div style='border:2px solid #ccc;border-radius:10px;padding:10px;background:#fafafa;margin-top:10px;'>
          <h4 style='margin:0;'>🛰️ 3D Viewer (viser)</h4>
          <span style='color:gray;font-size:14px;'>🖱️ Rotate: LMB | Pan: RMB | Zoom: Scroll</span>
          <iframe src="http://{local_ip}:8080" width="100%" height="600"
                  frameborder="0" style="border-radius:8px;margin-top:8px;"></iframe>
        </div>
        """

    try:
        # === Step 1️⃣ Caption → Candidate ===
        add_log(f"🚀 Starting for scene <b>{scene_name}</b> with caption: “{caption}”")
        add_log("📦 Step 1/3 — Generating candidates ...")
        yield add_log("⏳ Processing ..."), ""
        generate_vis_candidates(scene_name, caption, merged_json, cand_json, topk=10)
        add_log("✅ Step 1 Completed.")
        yield add_log("✅ Candidate JSON generated."), ""

        # === Step 2️⃣ Candidate → Grounding ===
        add_log("🧠 Step 2/3 — Running Qwen3-VL grounding inference ...")
        yield add_log("⏳ Inference running ..."), ""
        run_vis_grounding(scene_name, cand_json, img_root, result_json)
        add_log("✅ Step 2 Completed.")
        yield add_log("✅ Grounding finished."), ""

        # === Step 3️⃣ Rendering ===
        add_log("🎯 Step 3/3 — Rendering predicted & GT bounding boxes ...")
        yield add_log("⏳ Rendering visualization ..."), ""

        with open(result_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        pred_id = str(data[0].get("predicted_id", "None"))
        # gt_id = str(data[0].get("GTID", "None"))
        gt_id = "1"
        reason = data[0].get("reason", "No reason provided")

        target_bbox = None
        gt_bbox = None
        for bbox in current_bboxes:
            if str(int(bbox[-1])) == pred_id:
                target_bbox = bbox
            if str(int(bbox[-1])) == gt_id:
                gt_bbox = bbox

        if target_bbox is not None:
            bx, by, bz, dx, dy, dz = target_bbox[:6]
            center_shift = np.array([bx, by, bz])

            # === 点云中心化 ===
            pts_shifted = current_pts.copy()
            pts_shifted[:, :3] -= center_shift

            # === viser绘制 ===
            server_instance.clear_scene()
            server_instance.show_point_cloud(pts_shifted)

            # 🔴 Pred bbox（原点）
            server_instance.show_bbox(np.array([0, 0, 0, dx, dy, dz]), "Pred_Grounding", color=(255, 0, 0))

            # 🟢 GT bbox（相对Pred偏移）
            if gt_bbox is not None:
                gx, gy, gz, gdx, gdy, gdz = gt_bbox[:6]
                gt_center_shifted = np.array([gx, gy, gz]) - center_shift
                server_instance.show_bbox(np.array([*gt_center_shifted, gdx, gdy, gdz]), "GT_Grounding", color=(0, 255, 0))

            add_log(f"✅ Completed! Predicted ID = <b>{pred_id}</b> | GT ID = <b>{gt_id}</b>")
            add_log(f"🧩 Reason: {reason}")
        else:
            add_log(f"⚠️ Predicted bbox ID {pred_id} not found in current scene.")

        add_log("🎉 All steps finished successfully.")
        yield log_html + "</div></div>", iframe_view()

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        add_log(f"❌ Error occurred: <b>{e}</b>")
        add_log(f"<pre>{tb}</pre>")
        yield log_html + "</div></div>", ""

# ==============================
# 7️⃣ 启动 Gradio UI
# ==============================
def launch_ui():
    with gr.Blocks() as demo:
        gr.Markdown("""
       # 🌆 CityVG: Contrastive Fine-Tuning and Reward-Based Chain-of-Thought Reasoning for Zero-Shot City-Scale 3D Visual Grounding
        ### 🧭 Usage Guide:
        1️⃣ **Select a point cloud scene** — choose from the dropdown list below.  
        2️⃣ **Click “🚀 Visualize Scene”** — the system will load the point cloud and render it in the 3D viewer.  
        3️⃣ **Enter a grounding prompt** — type a natural language query (e.g., *"the white car near the bridge"*).  
        4️⃣ **Click “🎯 Run Grounding”** — the system will highlight one candidate bounding box to grounding.
        """)

        with gr.Row():
            scene_id = gr.Dropdown(
                choices=["birmingham_block_4", "birmingham_block_5", "cambridge_block_10"],
                value="birmingham_block_4",
                label="Select Scene"
            )
            caption = gr.Textbox(label="Input Caption (e.g. 'The outer section of a large building on Birchfield Road's corner of the map has a white and blue roof.')")

        with gr.Row():
            vis_btn = gr.Button("🚀 Visualize Scene")
            run_btn = gr.Button("🎯 Run Grounding")

        log_box = gr.HTML(value="<p style='color:gray;'>🪶 Waiting for scene...</p>", label="Log Output")
        viewer_box = gr.HTML(value="<p style='color:gray;'>🕓 Viewer not loaded yet...</p>", label="3D Viewer")

        vis_btn.click(fn=visualize_scene, inputs=scene_id, outputs=[log_box, viewer_box])
        run_btn.click(fn=run_grounding, inputs=[scene_id, caption], outputs=[log_box, viewer_box], show_progress=True)

    return demo


# ==============================
# 8️⃣ 主程序入口
# ==============================
if __name__ == "__main__":
    demo = launch_ui()

    def start_viser():
        global server_instance
        if server_instance is None:
            server_instance = CityVGViserServer(port=8080)
        while True:
            time.sleep(1)

    threading.Thread(target=start_viser, daemon=True).start()
    local_ip = get_local_ip()
    print("🚀 CityVG Final Stream UI launched:")
    print(f" - Gradio: http://{local_ip}:7860")
    print(f" - viser:  http://{local_ip}:8080")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
