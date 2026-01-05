#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joschek’s Captioner
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import os
import shutil
import threading
import signal
import glob
import base64
import json
from pathlib import Path
from PIL import Image, ImageTk

# ---------------- CONFIG ----------------
CONFIG_FILE = Path.home() / ".config" / "joschek_captioner.json"
DEFAULT_PORT = "11434"
DEFAULT_CTX = "16384"
DEFAULT_BATCH = "512"
DEFAULT_GPU = "33"
DEFAULT_TOKENS = "1024"
API_URL = f"http://localhost:{DEFAULT_PORT}/v1"
DEFAULT_PROMPT = "Describe this image in detail for an AI training dataset. Focus on clothing, background, textures, and lighting."
TARGETS = [768, 1024, 1536, 2048]

# ---------------- MODERN PALETTE ----------------
# Primary colors - sleek dark theme with modern accents
BG = "#1a1b26"  # Darker background for better contrast
CARD = "#24283b"  # Modern card color
INPUT = "#2a2d3e"  # Input field background
TEXT = "#f8f8f2"  # Brighter text for better readability
DIM = "#6272a4"  # Subtle dim text
BORDER = "#44475a"  # Border color
ACCENT = "#6272a4"  # Primary accent color
BLUE = "#5294e2"  # Modern blue
GREEN = "#50fa7b"  # Vibrant green
RED = "#ff5555"  # Modern red
PURPLE = "#bd93f9"  # Purple accent
CYAN = "#8be9fd"  # Cyan accent
ORANGE = "#ff79c6"  # Orange accent
HOVER = "#3d424e"  # Hover state color


# ---------------- UTILS ----------------
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.after_id = None
        widget.bind("<Enter>", self.on_enter)
        widget.bind("<Leave>", self.on_leave)

    def on_enter(self, event=None):
        self.after_id = self.widget.after(500, self.show_tip)

    def on_leave(self, event=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        self.hide_tip()

    def show_tip(self):
        if not self.text:
            return
        # Get mouse coordinates and add offset to avoid overlap/flicker
        x = self.widget.winfo_pointerx() + 15
        y = self.widget.winfo_pointery() + 15
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("Sans", "8", "normal"),
            padx=4,
            pady=2,
        )
        label.pack(ipadx=1)

    def hide_tip(self):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


class Config:
    def __init__(self):
        self.config_dir = CONFIG_FILE.parent
        self.data = self.load()

    def load(self):
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            if CONFIG_FILE.exists():
                return json.loads(CONFIG_FILE.read_text())
        except Exception as e:
            print("Config load error:", e)
        return {
            "server_binary": "./build/bin/llama-server",
            "model_file": "",
            "projector_file": "",
            "port": DEFAULT_PORT,
            "context": DEFAULT_CTX,
            "gpu_layers": DEFAULT_GPU,
            "max_tokens": DEFAULT_TOKENS,
            "last_prompt": DEFAULT_PROMPT,
            "last_dir": str(Path.home()),
        }

    def save(self):
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            CONFIG_FILE.write_text(json.dumps(self.data, indent=2))
        except Exception as e:
            print("Config save error:", e)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save()


# ---------------- WIDGETS ----------------
class QueueItem(tk.Frame):
    def __init__(
        self, parent, path: Path, remove_cb, status_cb, config, overwrite_default=False
    ):
        super().__init__(parent, bg=CARD)
        self.folder_path = path
        self.status = "draft"
        self.remove_cb = remove_cb
        self.status_cb = status_cb
        self.config = config
        self.overwrite_var = tk.BooleanVar(value=overwrite_default)

        main = tk.Frame(self, bg=CARD)
        main.pack(fill="both", expand=True, padx=14, pady=10)

        header = tk.Frame(main, bg=CARD)
        header.pack(fill="x", pady=(0, 6))

        close = tk.Label(
            header, text="×", bg=CARD, fg=DIM, font=("Sans", 14), cursor="hand2"
        )
        close.pack(side="right")
        close.bind("<Button-1>", lambda e: remove_cb(self))
        close.bind("<Enter>", lambda e: close.config(fg=RED))
        close.bind("<Leave>", lambda e: close.config(fg=DIM))

        self.status_lbl = tk.Label(
            header, text="Draft", bg=CARD, fg=ORANGE, font=("Sans", 8, "bold")
        )
        self.status_lbl.pack(side="right", padx=8)

        self.btn_queue = tk.Button(
            header,
            text="Add to Queue",
            bg=BLUE,
            fg="white",
            bd=0,
            relief="flat",
            font=("Sans", 8, "bold"),
            cursor="hand2",
            command=self.add_to_queue,
            activebackground=HOVER,
            activeforeground=TEXT,
            highlightthickness=0,
        )
        self.btn_queue.pack(side="right", padx=(0, 8), ipadx=8, ipady=2)

        self.cb_overwrite = tk.Checkbutton(
            header,
            text="Overwrite",
            variable=self.overwrite_var,
            bg=CARD,
            fg=DIM,
            selectcolor=INPUT,
            activebackground=CARD,
            font=("Sans", 8),
            highlightthickness=0,
        )
        self.cb_overwrite.pack(side="right", padx=(0, 8))

        tk.Label(
            header,
            text=self.folder_path.name,
            bg=CARD,
            fg=TEXT,
            font=("Sans", 9, "bold"),
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        tk.Label(
            main,
            text=str(self.folder_path),
            bg=CARD,
            fg=DIM,
            font=("Sans", 7),
            anchor="w",
        ).pack(fill="x", pady=(0, 8))

        self.prompt = tk.Text(
            main,
            height=3,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            font=("Sans", 8),
            insertbackground=BLUE,
            wrap="word",
        )
        self.prompt.insert("1.0", config.get("last_prompt", DEFAULT_PROMPT))
        self.prompt.bind(
            "<KeyRelease>", lambda e: config.set("last_prompt", self.get_prompt())
        )
        self.prompt.pack(fill="x", pady=(0, 8))

    def set_status(self, state, msg=""):
        self.status = state
        color = {
            "processing": BLUE,
            "done": GREEN,
            "error": RED,
            "pending": CYAN,
            "draft": ORANGE,
        }.get(state, DIM)
        display_text = msg if msg else state.title()
        self.status_lbl.config(text=display_text, fg=color)

        if state == "draft":
            self.btn_queue.pack(side="right", padx=(0, 8), ipadx=8, ipady=2)
            self.cb_overwrite.pack(side="right", padx=(0, 8))
        else:
            self.btn_queue.pack_forget()
            self.cb_overwrite.pack_forget()
            if state == "pending":
                self.status_lbl.config(text="In Queue")

        if self.status_cb:
            self.status_cb()

    def add_to_queue(self):
        self.set_status("pending")

    def get_prompt(self):
        return self.prompt.get("1.0", "end-1c").strip()


class ScrollFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(
            self,
            orient="vertical",
            command=canvas.yview,
            width=30,
            bg=CARD,
            troughcolor=BG,
            bd=0,
            highlightthickness=0,
            activebackground=HOVER,
            elementborderwidth=0,
        )
        self.content = tk.Frame(canvas, bg=BG)
        self.content.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        self.window_id = canvas.create_window((0, 0), window=self.content, anchor="nw")
        canvas.bind(
            "<Configure>", lambda e: canvas.itemconfig(self.window_id, width=e.width)
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"),
        )


# ---------------- CROP WORKER (VLM) ----------------
class VLMCropWorker:
    def __init__(
        self,
        input_path,
        output_path,
        target_object,
        ensure_include,
        mogrify_enabled,
        force_square,
        one_per_image,
        conform_strategy,
        min_snap_side,
        safety_margin,
        conform_policy,
        pad_color,
        config,
        update_progress,
        update_log,
        finished_cb,
        list_cb,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.target_object = target_object
        self.ensure_include = ensure_include
        self.mogrify_enabled = mogrify_enabled
        self.force_square = force_square
        self.one_per_image = one_per_image
        self.conform_strategy = conform_strategy
        self.min_snap_side = min_snap_side
        self.safety_margin = safety_margin
        self.conform_policy = conform_policy
        self.pad_color = pad_color
        self.config = config
        self.update_progress = update_progress
        self.update_log = update_log
        self.finished_cb = finished_cb
        self.list_cb = list_cb
        self.running = True

    def _calc_iou(self, boxA, boxB):
        # box: [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if boxAArea + boxBArea - interArea == 0:
            return 0
        return interArea / float(boxAArea + boxBArea - interArea)

    def run(self):
        import cv2
        import numpy as np
        import re
        from openai import OpenAI

        try:
            self.update_log("Initializing VLM Crop Worker...")
            os.makedirs(self.output_path, exist_ok=True)
            extensions = [
                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.webp",
                "*.JPG",
                "*.JPEG",
                "*.PNG",
                "*.WEBP",
            ]
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(self.input_path, ext)))

            if not files:
                self.update_log("No images found in folder.")
                self.finished_cb()
                return

            port = self.config.get("port", "11434")
            base_url = f"http://127.0.0.1:{port}/v1"
            self.update_log(f"Connecting to server at {base_url}...")
            client = OpenAI(base_url=base_url, api_key="sk-no-key")

            model_id = ""
            try:
                models = client.models.list()
                if models.data:
                    model_id = models.data[0].id
                    self.update_log(f"Connected. Using model: {model_id}")
                else:
                    model_id = Path(self.config.get("model_file", "default")).stem
                    self.update_log(f"Connected (No models listed). Trying: {model_id}")
            except Exception as e:
                self.update_log(f"Connection Failed: {e}")
                self.finished_cb()
                return

            self.update_log(f"Found {len(files)} images. Starting processing...")

            for i, f in enumerate(files):
                if not self.running:
                    break
                filename = os.path.basename(f)
                name_no_ext = os.path.splitext(filename)[0]

                try:
                    # Pre-processing
                    if self.mogrify_enabled:
                        try:
                            subprocess.run(
                                ["mogrify", "-trim", "-fuzz", "10%", "+repage", f],
                                check=True,
                            )
                        except Exception as e:
                            self.update_log(f"Mogrify error on {filename}: {e}")

                    image = cv2.imread(f)
                    if image is None:
                        continue
                    img_h, img_w = image.shape[:2]

                    # Prompt
                    prompt = (
                        f"Task: Precise Object Detection.\n"
                        f"Object: {self.target_object}\n"
                        f"Coordinates: Normalized 0-1000 (X=Left to Right, Y=Top to Bottom).\n"
                        f"Respond for each instance with exactly: DETECTION: BOX=[x1, y1, x2, y2]"
                    )

                    with open(f, "rb") as img_file:
                        b64 = base64.b64encode(img_file.read()).decode()
                    resp = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{b64}"
                                        },
                                    },
                                ],
                            }
                        ],
                        max_tokens=512,
                        temperature=0.1,
                        timeout=120,
                    )
                    content = resp.choices[0].message.content

                    # Parse boxes
                    raw_boxes = []
                    instances = re.findall(
                        r"BOX=\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]",
                        content,
                        re.IGNORECASE,
                    )
                    if not instances:
                        raw_ints = [int(x) for x in re.findall(r"\d+", content)]
                        for k in range(0, len(raw_ints) - 3, 4):
                            v = raw_ints[k : k + 4]
                            if all(0 <= x <= 1000 for x in v):
                                raw_boxes.append(v)
                    else:
                        raw_boxes = [list(map(int, b)) for b in instances]

                    if not raw_boxes:
                        self.update_log(f"No valid detections for {filename}")
                        self.list_cb(f"[NO DETECT] {filename}", ORANGE)
                        continue

                    # Denormalize and Convert to [x1, y1, x2, y2]
                    real_boxes = []
                    for b in raw_boxes:
                        x1, y1, x2, y2 = b
                        bx1, by1 = (x1 / 1000.0) * img_w, (y1 / 1000.0) * img_h
                        bx2, by2 = (x2 / 1000.0) * img_w, (y2 / 1000.0) * img_h
                        real_boxes.append(
                            [min(bx1, bx2), min(by1, by2), max(bx1, bx2), max(by1, by2)]
                        )

                    # Deduplicate using IoU
                    unique_boxes = []
                    for rb in real_boxes:
                        is_dup = False
                        for ub in unique_boxes:
                            if self._calc_iou(rb, ub) > 0.5:  # 0.5 IoU Threshold
                                is_dup = True
                                break
                        if not is_dup:
                            unique_boxes.append(rb)

                    # Filter small boxes
                    unique_boxes = [
                        b
                        for b in unique_boxes
                        if (b[2] - b[0]) > 30 and (b[3] - b[1]) > 30
                    ]

                    # One per image logic (Largest area)
                    if self.one_per_image and len(unique_boxes) > 1:
                        unique_boxes.sort(
                            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True
                        )
                        unique_boxes = [unique_boxes[0]]

                    saved_count = 0
                    for box in unique_boxes:
                        if not self.running:
                            break
                        bx1, by1, bx2, by2 = box
                        bw, bh = bx2 - bx1, by2 - by1

                        # Apply Margin
                        try:
                            m_pct = float(self.safety_margin) / 100.0
                        except Exception:
                            m_pct = 0.25

                        # Box with margin
                        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                        mw, mh = bw * (1 + 2 * m_pct), bh * (1 + 2 * m_pct)

                        # Calculate Required Crop Dimensions
                        req_w, req_h = mw, mh

                        # Conform Logic
                        longest = max(req_w, req_h)
                        target_long = longest

                        should_conform = False
                        if self.conform_strategy == "Full conform":
                            should_conform = True
                        elif self.conform_strategy in ["10%", "20%"]:
                            lower_m = (int(longest) // 256) * 256
                            lower_m = max(256, lower_m)
                            upper_m = lower_m + 256
                            # Find nearest
                            nearest = (
                                lower_m
                                if abs(longest - lower_m) < abs(longest - upper_m)
                                else upper_m
                            )
                            limit = 0.1 if self.conform_strategy == "10%" else 0.2
                            if abs(longest - nearest) / 256.0 <= limit:
                                should_conform = True

                        if should_conform:
                            # Default target: Snap UP to next multiple
                            try:
                                min_side = int(self.min_snap_side)
                            except Exception:
                                min_side = 256

                            target_long = (int(longest) // 256) * 256
                            if target_long < longest:
                                target_long += 256
                            if target_long < min_side:
                                target_long = min_side

                            # Apply Policy for Boundaries
                            # Check if target size fits in image
                            # If force square, check both dims. Else check longest dim fit (we scale other dim)

                            # First, establish desired dimensions based on target_long
                            if self.force_square:
                                final_w = final_h = target_long
                            else:
                                # Scale proportionally
                                scale = target_long / longest
                                final_w, final_h = req_w * scale, req_h * scale

                            # Now check fit
                            fits = True
                            if final_w > img_w or final_h > img_h:
                                fits = False

                            if not fits:
                                if self.conform_policy == "Snap Down":
                                    # Reduce target_long by 256 until it fits or hits min
                                    while True:
                                        # Recalculate dims
                                        if self.force_square:
                                            t_w, t_h = target_long, target_long
                                        else:
                                            s = target_long / longest
                                            t_w, t_h = req_w * s, req_h * s

                                        if t_w <= img_w and t_h <= img_h:
                                            final_w, final_h = t_w, t_h
                                            break

                                        target_long -= 256
                                        if target_long < 256:
                                            # Can't fit even 256.
                                            target_long = 256
                                            if self.force_square:
                                                final_w, final_h = 256, 256
                                            else:
                                                s = 256 / longest
                                                final_w, final_h = req_w * s, req_h * s
                                            break
                                else:  # Pad to Fit
                                    # Keep final_w/final_h as calculated (large)
                                    pass
                        else:
                            # No conform, just use margin box
                            final_w, final_h = req_w, req_h
                            if self.force_square:
                                s = max(final_w, final_h)
                                final_w = final_h = s

                        # Do Crop
                        # Center final box on object center (cx, cy)
                        # Ensure target_long is EXACT multiple of 256 for Full conform
                        if should_conform and self.conform_strategy == "Full conform":
                            target_long = (round(target_long) // 256) * 256
                            if self.force_square:
                                final_w = final_h = target_long
                            else:
                                # Re-scale to ensure longest side is exactly target_long
                                s = target_long / longest
                                final_w, final_h = req_w * s, req_h * s

                        fw, fh = int(round(final_w)), int(round(final_h))
                        x_start = int(round(cx - fw / 2.0))
                        y_start = int(round(cy - fh / 2.0))
                        x_end = x_start + fw
                        y_end = y_start + fh

                        # Shift if within image (to avoid padding if possible)
                        if fw <= img_w:
                            if x_start < 0:
                                x_end -= x_start
                                x_start = 0
                            if x_end > img_w:
                                x_start -= x_end - img_w
                                x_end = img_w
                        if fh <= img_h:
                            if y_start < 0:
                                y_end -= y_start
                                y_start = 0
                            if y_end > img_h:
                                y_start -= y_end - img_h
                                y_end = img_h

                        # Force padding if Full conform is selected to guarantee output size
                        always_pad = (
                            should_conform and self.conform_strategy == "Full conform"
                        )
                        pad_needed = (
                            x_start < 0 or y_start < 0 or x_end > img_w or y_end > img_h
                        )

                        if always_pad or (
                            pad_needed and self.conform_policy == "Pad to Fit"
                        ):
                            # Create canvas
                            c_val = 0
                            if self.pad_color == "White":
                                c_val = 255
                            elif self.pad_color == "Grey":
                                c_val = 128
                            canvas = np.full((fh, fw, 3), c_val, dtype=np.uint8)

                            # Calculate overlap coordinates
                            ix1, iy1 = max(0, x_start), max(0, y_start)
                            ix2, iy2 = min(img_w, x_end), min(img_h, y_end)

                            cx1, cy1 = ix1 - x_start, iy1 - y_start
                            cx2, cy2 = cx1 + (ix2 - ix1), cy1 + (iy2 - iy1)

                            if ix2 > ix1 and iy2 > iy1:
                                canvas[cy1:cy2, cx1:cx2] = image[iy1:iy2, ix1:ix2]

                            final_crop = canvas
                        else:
                            # Standard Crop (Clip)
                            ix1, iy1 = max(0, x_start), max(0, y_start)
                            ix2, iy2 = min(img_w, x_end), min(img_h, y_end)
                            final_crop = image[iy1:iy2, ix1:ix2]

                        if final_crop.shape[0] < 10 or final_crop.shape[1] < 10:
                            continue

                        safe_target = (
                            "".join(c for c in self.target_object if c.isalnum())
                            or "object"
                        )
                        save_path = os.path.join(
                            self.output_path,
                            f"{name_no_ext}_{safe_target}_{saved_count}_{max(final_crop.shape)}px.jpg",
                        )
                        cv2.imwrite(save_path, final_crop)
                        saved_count += 1

                    if saved_count > 0:
                        self.list_cb(f"[SAVED {saved_count}] {filename}", GREEN)
                    else:
                        self.list_cb(f"[SKIPPED] {filename}", DIM)

                except Exception as e:
                    self.update_log(f"Error processing {filename}: {e}")
                    self.list_cb(f"[ERROR] {filename}", RED)
                    continue

                self.update_progress(i + 1, len(files))
            self.update_log("Done.")
        except Exception as e:
            self.update_log(f"Worker Critical Error: {e}")
        finally:
            self.finished_cb()


# ---------------- MAIN APP ----------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Joschek's Captioner V 1.0")
        root.geometry("1100x720")
        root.configure(bg=BG)
        root.option_add("*Font", ("Sans", 10))
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "jc32x32.png")
            if os.path.exists(icon_path):
                self.icon_img = tk.PhotoImage(file=icon_path)
                root.iconphoto(True, self.icon_img)
        except Exception as e:
            print(f"Error loading icon: {e}")
        self.config = Config()
        self.setup_styles()
        self.server_proc = None
        self.batch_running = False
        self.queue = []
        self.client = None
        self.current_editor_folder = None
        self.editor_items = []
        self.thumb_size = 200
        self.thumb_cache = {}
        self.is_loading_more = False
        self.all_filtered_paths = []
        self.loaded_count = 0
        self.tooltips = {}
        try:
            tt_file = Path(__file__).parent / "tooltips.json"
            if tt_file.exists():
                self.tooltips = json.loads(tt_file.read_text())
        except Exception as e:
            print("Error loading tooltips:", e)
        self.create_modern_tab_system()
        self.build_server()
        self.build_batch()
        self.build_editor()
        self.build_filter()
        self.build_crop()
        self.root.after(1000, self.update_vram_info)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(
            "TProgressbar",
            background=BLUE,
            troughcolor=BG,
            borderwidth=1,
            bordercolor=CARD,
            thickness=8,
            lightcolor=CARD,
            darkcolor=CARD,
        )
        s.configure(
            "Horizontal.TProgressbar",
            background=BLUE,
            troughcolor=BG,
            borderwidth=1,
            bordercolor=CARD,
            thickness=8,
            lightcolor=CARD,
            darkcolor=CARD,
        )
        s.configure(
            "Vertical.TScrollbar",
            background=BG,
            troughcolor=BG,
            borderwidth=0,
            arrowsize=0,
            width=30,
        )
        s.map("Vertical.TScrollbar", background=[("active", HOVER), ("!active", CARD)])
        s.configure(
            "Horizontal.TScrollbar",
            background=BG,
            troughcolor=BG,
            borderwidth=0,
            arrowsize=0,
            width=30,
        )
        s.map(
            "Horizontal.TScrollbar", background=[("active", HOVER), ("!active", CARD)]
        )
        s.configure(
            "TCombobox",
            background=INPUT,
            foreground=TEXT,
            borderwidth=0,
            fieldbackground=INPUT,
            selectbackground=BLUE,
            selectforeground=TEXT,
            arrowcolor=TEXT,
        )
        s.map(
            "TCombobox",
            fieldbackground=[("readonly", INPUT)],
            background=[("readonly", INPUT)],
            foreground=[("readonly", TEXT)],
        )

    def create_modern_tab_system(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)
        self.tab_bar = tk.Frame(main, bg=BG, height=50)
        self.tab_bar.pack(side="top", fill="x", pady=(10, 0), padx=20)
        self.content_area = tk.Frame(main, bg=BG)
        self.content_area.pack(side="bottom", fill="both", expand=True)
        self.tab_srv = tk.Frame(self.content_area, bg=BG)
        self.tab_batch = tk.Frame(self.content_area, bg=BG)
        self.tab_editor = tk.Frame(self.content_area, bg=BG)
        self.tab_filter = tk.Frame(self.content_area, bg=BG)
        self.tab_crop = tk.Frame(self.content_area, bg=BG)
        self.tabs = {}
        self.tab_btns = {}
        self.current_tab = None
        self.add_tab("Server", self.tab_srv)
        self.add_tab("Batch Captioning", self.tab_batch)
        self.add_tab("Manual Edit", self.tab_editor)
        self.add_tab("Filter & Move", self.tab_filter)
        self.add_tab("Automatic Cropping", self.tab_crop)
        tk.Frame(main, bg=INPUT, height=1).pack(side="top", fill="x", pady=(0, 0))
        self.switch_tab("Server")

    def add_tab(self, name, frame):
        self.tabs[name] = frame
        btn = tk.Label(
            self.tab_bar,
            text=name,
            bg=BG,
            fg=DIM,
            font=("Sans", 10, "bold"),
            cursor="hand2",
            padx=15,
            pady=8,
        )
        btn.pack(side="left")
        btn.bind("<Button-1>", lambda e: self.switch_tab(name))
        btn.bind("<Enter>", lambda e: self._hover_tab(name, True))
        btn.bind("<Leave>", lambda e: self._hover_tab(name, False))
        self.tab_btns[name] = btn

    def switch_tab(self, name):
        if self.current_tab == name:
            return
        if self.current_tab:
            self.tabs[self.current_tab].pack_forget()
            self.tab_btns[self.current_tab].config(fg=DIM)
        self.current_tab = name
        self.tabs[name].pack(fill="both", expand=True)
        self.tab_btns[name].config(fg=BLUE)

    def _hover_tab(self, name, entering):
        if name == self.current_tab:
            return
        self.tab_btns[name].config(fg=TEXT if entering else DIM)

    def build_server(self):
        f = tk.Frame(self.tab_srv, bg=BG)
        f.pack(fill="both", expand=True, padx=25, pady=20)
        default_bin = "./build/bin/llama-server"
        if os.name == "nt":
            default_bin = "llama-server.exe"
        self.bin = tk.StringVar(value=self.config.get("server_binary", default_bin))
        self.model = tk.StringVar(value=self.config.get("model_file", ""))
        self.proj = tk.StringVar(value=self.config.get("projector_file", ""))
        self.port = tk.StringVar(value=self.config.get("port", DEFAULT_PORT))
        self.ctx = tk.StringVar(value=self.config.get("context", DEFAULT_CTX))
        self.gpu = tk.StringVar(value=self.config.get("gpu_layers", DEFAULT_GPU))
        self.max_tokens = tk.StringVar(
            value=self.config.get("max_tokens", DEFAULT_TOKENS)
        )
        for var, key in [
            (self.bin, "server_binary"),
            (self.model, "model_file"),
            (self.proj, "projector_file"),
            (self.port, "port"),
            (self.ctx, "context"),
            (self.gpu, "gpu_layers"),
            (self.max_tokens, "max_tokens"),
        ]:
            var.trace_add("write", lambda *_, v=var, k=key: self.config.set(k, v.get()))
        self.detect_binary()
        f1 = self.field(f, "Server Binary", self.bin, True, kind="file")
        ToolTip(f1, self.tooltips.get("server_binary"))
        f2 = self.field(f, "Model (.gguf)", self.model, True, kind="file")
        ToolTip(f2, self.tooltips.get("model_file"))
        f3 = self.field(f, "Projector (.gguf)", self.proj, True, kind="file")
        ToolTip(f3, self.tooltips.get("projector_file"))
        tk.Frame(f, height=12, bg=BG).pack()
        params = tk.Frame(f, bg=BG)
        params.pack(fill="x")
        for lbl, v, tt_key in [
            ("Port", self.port, "port"),
            ("Context", self.ctx, "context"),
            ("GPU Layers", self.gpu, "gpu_layers"),
            ("Max Tokens", self.max_tokens, "max_tokens"),
        ]:
            col = tk.Frame(params, bg=BG)
            col.pack(side="left", fill="x", expand=True, padx=3)
            tk.Label(col, text=lbl, bg=BG, fg=DIM, font=("Sans", 7)).pack(
                anchor="w", pady=(0, 2)
            )
            ent = tk.Entry(
                col,
                textvariable=v,
                bg=INPUT,
                fg=TEXT,
                bd=0,
                relief="flat",
                highlightthickness=0,
                font=("Sans", 8),
                insertbackground=BLUE,
                justify="center",
            )
            ent.pack(fill="x", ipady=5)
            ToolTip(ent, self.tooltips.get(tt_key))
        tk.Frame(f, height=8, bg=BG).pack()
        vram_frame = tk.Frame(f, bg=BG)
        vram_frame.pack(fill="x")
        self.vram_label = tk.Label(
            vram_frame, text="Checking VRAM...", bg=BG, fg=DIM, font=("Sans", 7)
        )
        self.vram_label.pack(side="left", fill="x", expand=True)
        self.btn_kill_gpu = self.btn(
            vram_frame, "Kill GPU Processes", BLUE, self.kill_gpu_processes
        )
        self.btn_kill_gpu.pack(side="right")
        ToolTip(self.btn_kill_gpu, self.tooltips.get("kill_gpu"))
        tk.Frame(f, height=4, bg=BG).pack()
        tip = tk.Frame(f, bg=CARD)
        tip.pack(fill="x", padx=1, pady=1)
        tk.Label(
            tip,
            text="VRAM Tip: If 'out of memory', lower GPU Layers (e.g. 33) or Context.",
            bg=CARD,
            fg=DIM,
            font=("Sans", 7),
        ).pack(pady=5)
        tk.Frame(f, height=12, bg=BG).pack()
        btns = tk.Frame(f, bg=BG)
        btns.pack(fill="x")
        self.btn_start = self.btn(btns, "Start Server", BLUE, self.start_server)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ToolTip(self.btn_start, self.tooltips.get("start_server"))
        self.btn_stop = self.btn(btns, "Stop Server", BLUE, self.stop_server)
        self.btn_stop.pack(side="left", fill="x", expand=True)
        self.btn_stop.config(state="disabled", bg=CARD, disabledforeground="white")
        tk.Frame(f, height=12, bg=BG).pack()
        log_frame = tk.Frame(f, bg=BG)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(
            log_frame,
            height=11,
            bg="#1a1d23",
            fg=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            font=("Monospace", 7),
            wrap="word",
        )
        self.log.pack(side="left", fill="both", expand=True)
        s_log = tk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.log.yview,
            width=30,
            bg=CARD,
            troughcolor=BG,
            bd=0,
            highlightthickness=0,
            activebackground=HOVER,
            elementborderwidth=0,
        )
        s_log.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=s_log.set)

    def build_batch(self):
        main = tk.Frame(self.tab_batch, bg=BG)
        main.pack(fill="both", expand=True, padx=25, pady=15)
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        tool = tk.Frame(left, bg=BG)
        tool.pack(fill="x", pady=(0, 10))
        btn_add = self.btn(tool, "Add Folder", BLUE, self.add_folder)
        btn_add.pack(side="left", padx=(0, 8))
        ToolTip(btn_add, self.tooltips.get("add_folder"))
        self.btn_proc = self.btn(tool, "Start Processing", BLUE, self.toggle_batch)
        self.btn_proc.pack(side="left")
        ToolTip(self.btn_proc, self.tooltips.get("start_batch"))
        self.queue_scroll = ScrollFrame(left)
        self.queue_scroll.pack(fill="both", expand=True)
        prog = tk.Frame(left, bg=BG)
        prog.pack(fill="x", side="bottom", pady=(10, 0))
        self.progress = ttk.Progressbar(prog, mode="determinate")
        self.progress.pack(fill="x")
        self.prog_lbl = tk.Label(prog, text="Idle", bg=BG, fg=DIM, font=("Sans", 8))
        self.prog_lbl.pack(pady=(4, 0))
        right = tk.Frame(main, bg=BG, width=250)
        right.pack(side="right", fill="both", expand=False, padx=(15, 0))
        right.pack_propagate(False)
        tk.Label(
            right, text="Up Next (Queue)", bg=BG, fg=TEXT, font=("Sans", 9, "bold")
        ).pack(anchor="w", pady=(0, 5))
        queue_frame = tk.Frame(right, bg=BG, height=150)
        queue_frame.pack(fill="x", pady=(0, 10))
        queue_frame.pack_propagate(False)
        self.queue_summary = tk.Listbox(
            queue_frame,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            highlightthickness=0,
            font=("Sans", 8),
            selectbackground=INPUT,
            activestyle="none",
        )
        self.queue_summary.pack(side="left", fill="both", expand=True)
        tk.Label(
            right, text="Processing Log", bg=BG, fg=TEXT, font=("Sans", 9, "bold")
        ).pack(anchor="w", pady=(0, 5))
        status_frame = tk.Frame(right, bg=BG)
        status_frame.pack(fill="both", expand=True)
        self.status_log = tk.Text(
            status_frame,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            font=("Monospace", 7),
            wrap="word",
            state="disabled",
        )
        self.status_log.pack(side="left", fill="both", expand=True)
        s_batch = tk.Scrollbar(
            status_frame,
            orient="vertical",
            command=self.status_log.yview,
            width=30,
            bg=CARD,
            troughcolor=BG,
            bd=0,
            highlightthickness=0,
            activebackground=HOVER,
            elementborderwidth=0,
        )
        s_batch.pack(side="right", fill="y", padx=(5, 0))
        self.status_log.configure(yscrollcommand=s_batch.set)

    def build_editor(self):
        tool = tk.Frame(self.tab_editor, bg=BG)
        tool.pack(fill="x", padx=25, pady=15)
        btn_load = self.btn(tool, "Load Folder", BLUE, self.load_editor_folder)
        btn_load.pack(side="left")
        ToolTip(btn_load, self.tooltips.get("load_editor"))
        btn_q = self.btn(
            tool, "Add to Batch Queue", BLUE, self.add_current_folder_to_batch
        )
        btn_q.pack(side="left", padx=(10, 0))
        ToolTip(btn_q, "Add currently loaded folder to Batch Queue")
        self.editor_folder_label = tk.Label(
            tool, text="No folder loaded", bg=BG, fg=DIM, font=("Sans", 8)
        )
        self.editor_folder_label.pack(side="left", padx=15)
        filter_frame = tk.Frame(tool, bg=BG)
        filter_frame.pack(side="right")
        tk.Label(
            filter_frame, text="Filter (in captions):", bg=BG, fg=DIM, font=("Sans", 9)
        ).pack(side="left", padx=(10, 2))
        self.editor_filter_var = tk.StringVar()
        filter_entry = tk.Entry(
            filter_frame,
            textvariable=self.editor_filter_var,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            font=("Sans", 10),
            insertbackground=BLUE,
            width=25,
        )
        filter_entry.pack(side="left", ipady=6)
        ToolTip(filter_entry, self.tooltips.get("filter_editor"))
        filter_entry.bind("<Enter>", lambda e: filter_entry.config(bg=HOVER))
        filter_entry.bind("<Leave>", lambda e: filter_entry.config(bg=INPUT))
        filter_entry.bind("<Return>", lambda e: self.apply_editor_filter())
        clear_btn = tk.Button(
            filter_frame,
            text="Clear",
            bg=CARD,
            fg=TEXT,
            bd=0,
            relief="flat",
            font=("Sans", 9, "bold"),
            cursor="hand2",
            command=self.clear_editor_filter,
            activebackground=HOVER,
            activeforeground=TEXT,
            highlightthickness=0,
        )
        ToolTip(clear_btn, self.tooltips.get("clear_filter"))
        clear_btn.pack(side="left", padx=(4, 0), ipady=5, ipadx=10)
        clear_btn.bind("<Enter>", lambda e: clear_btn.config(bg=HOVER))
        clear_btn.bind("<Leave>", lambda e: clear_btn.config(bg=CARD))
        content = tk.Frame(self.tab_editor, bg=BG)
        content.pack(fill="both", expand=True, padx=25, pady=(0, 15))
        self.img_canvas = tk.Canvas(content, bg=BG, highlightthickness=0, bd=0)
        img_scroll = tk.Scrollbar(
            content,
            orient="vertical",
            command=self.img_canvas.yview,
            width=30,
            bg=CARD,
            troughcolor=BG,
            bd=0,
            highlightthickness=0,
            activebackground=HOVER,
            elementborderwidth=0,
        )
        self.img_list_frame = tk.Frame(self.img_canvas, bg=BG)
        self.canvas_window = self.img_canvas.create_window(
            (0, 0), window=self.img_list_frame, anchor="nw"
        )
        self.img_canvas.bind(
            "<Configure>",
            lambda e: self.img_canvas.itemconfig(self.canvas_window, width=e.width),
        )
        self.img_list_frame.bind(
            "<Configure>",
            lambda e: self.img_canvas.configure(
                scrollregion=self.img_canvas.bbox("all")
            ),
        )

        def _on_scroll(*args):
            img_scroll.set(*args)
            if float(args[1]) > 0.9:
                self.load_more_items()

        self.img_canvas.configure(yscrollcommand=_on_scroll)
        self.img_canvas.pack(side="left", fill="both", expand=True)
        img_scroll.pack(side="right", fill="y")

        def _on_mousewheel(e):
            if e.num == 4 or e.delta > 0:
                self.img_canvas.yview_scroll(-2, "units")
            elif e.num == 5 or e.delta < 0:
                self.img_canvas.yview_scroll(2, "units")

        self.img_canvas.bind(
            "<Enter>",
            lambda e: (
                self.img_canvas.bind_all("<MouseWheel>", _on_mousewheel),
                self.img_canvas.bind_all("<Button-4>", _on_mousewheel),
                self.img_canvas.bind_all("<Button-5>", _on_mousewheel),
            ),
        )
        self.img_canvas.bind(
            "<Leave>",
            lambda e: (
                self.img_canvas.unbind_all("<MouseWheel>"),
                self.img_canvas.unbind_all("<Button-4>"),
                self.img_canvas.unbind_all("<Button-5>"),
            ),
        )
        last_folder = self.config.get("last_editor_folder")
        if last_folder and Path(last_folder).exists():
            self.root.after(100, lambda: self.load_editor_folder(last_folder))

    def build_crop(self):
        f = tk.Frame(self.tab_crop, bg=BG)
        f.pack(fill="both", expand=True, padx=25, pady=20)
        self.crop_in, self.crop_out, self.crop_target, self.crop_include = (
            tk.StringVar(),
            tk.StringVar(),
            tk.StringVar(value="human"),
            tk.StringVar(value="head, face"),
        )
        self.crop_mogrify = tk.BooleanVar(value=False)
        self.crop_square = tk.BooleanVar(value=False)
        self.crop_one_per = tk.BooleanVar(value=False)
        self.crop_conform = tk.StringVar(value="No conform")
        self.crop_conform_policy = tk.StringVar(value="Snap Down")
        self.crop_pad_color = tk.StringVar(value="Black")
        self.crop_min_side = tk.StringVar(value="256")
        self.crop_margin = tk.StringVar(value="25")
        self.crop_model = None
        self.crop_worker = None
        tk.Label(f, text="Input Folder:", bg=BG, fg=DIM, font=("Sans", 9)).pack(
            anchor="w"
        )
        row_in = tk.Frame(f, bg=BG)
        row_in.pack(fill="x", pady=(0, 10))
        ent_in = tk.Entry(
            row_in,
            textvariable=self.crop_in,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            font=("Sans", 9),
            insertbackground=BLUE,
        )
        ent_in.pack(side="left", fill="x", expand=True, ipady=6)
        ToolTip(ent_in, self.tooltips.get("crop_in"))
        tk.Button(
            row_in,
            text="…",
            bg=CARD,
            fg=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            width=4,
            command=self.crop_select_in,
        ).pack(side="right", padx=(4, 0))
        tk.Label(f, text="Output Folder:", bg=BG, fg=DIM, font=("Sans", 9)).pack(
            anchor="w"
        )
        row_out = tk.Frame(f, bg=BG)
        row_out.pack(fill="x", pady=(0, 10))
        ent_out = tk.Entry(
            row_out,
            textvariable=self.crop_out,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            font=("Sans", 9),
            insertbackground=BLUE,
        )
        ent_out.pack(side="left", fill="x", expand=True, ipady=6)
        ToolTip(ent_out, self.tooltips.get("crop_out"))
        tk.Button(
            row_out,
            text="…",
            bg=CARD,
            fg=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            width=4,
            command=lambda: self.crop_out.set(
                self._folder_picker("Select Output Folder")
            ),
        ).pack(side="right", padx=(4, 0))
        tk.Label(
            f, text="Target Object (Text Prompt):", bg=BG, fg=DIM, font=("Sans", 9)
        ).pack(anchor="w", pady=(0, 2))
        ent_target = tk.Entry(
            f,
            textvariable=self.crop_target,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            font=("Sans", 9),
            insertbackground=BLUE,
        )
        ent_target.pack(fill="x", ipady=6, pady=(0, 4))
        ToolTip(
            ent_target,
            "Enter what to detect and crop (e.g., 'human', 'face', 'red car')",
        )
        tk.Label(
            f, text="Ensure Include (Optional):", bg=BG, fg=DIM, font=("Sans", 9)
        ).pack(anchor="w", pady=(0, 2))
        ent_include = tk.Entry(
            f,
            textvariable=self.crop_include,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            font=("Sans", 9),
            insertbackground=BLUE,
        )
        ent_include.pack(fill="x", ipady=6, pady=(0, 6))
        ToolTip(
            ent_include,
            "Specify parts that MUST be included (e.g., 'head, face'). Leave empty to disable.",
        )
        prep_row = tk.Frame(f, bg=BG)
        prep_row.pack(fill="x", pady=(0, 10))
        tk.Checkbutton(
            prep_row,
            text="Pre-process (mogrify)",
            variable=self.crop_mogrify,
            bg=BG,
            fg=TEXT,
            selectcolor=INPUT,
            activebackground=BG,
            font=("Sans", 8),
            highlightthickness=0,
        ).pack(side="left", padx=(0, 15))
        cb_sq = tk.Checkbutton(
            prep_row,
            text="Force Square (1:1)",
            variable=self.crop_square,
            bg=BG,
            fg=TEXT,
            selectcolor=INPUT,
            activebackground=BG,
            font=("Sans", 8),
            highlightthickness=0,
        )
        cb_sq.pack(side="left", padx=(0, 15))
        ToolTip(cb_sq, self.tooltips.get("crop_square"))
        cb_one = tk.Checkbutton(
            prep_row,
            text="One Crop Per Image",
            variable=self.crop_one_per,
            bg=BG,
            fg=TEXT,
            selectcolor=INPUT,
            activebackground=BG,
            font=("Sans", 8),
            highlightthickness=0,
        )
        cb_one.pack(side="left")
        ToolTip(cb_one, self.tooltips.get("crop_one_per"))

        conform_row = tk.Frame(f, bg=BG)
        conform_row.pack(fill="x", pady=(0, 10))

        c_col1 = tk.Frame(conform_row, bg=BG)
        c_col1.pack(side="left", fill="x", expand=True, padx=(0, 3))
        tk.Label(c_col1, text="Snapping Mode:", bg=BG, fg=DIM, font=("Sans", 9)).pack(
            anchor="w", pady=(0, 2)
        )
        conf_opts = ["No conform", "Full conform", "10%", "20%"]
        self.conf_mb = tk.Menubutton(
            c_col1,
            textvariable=self.crop_conform,
            bg=INPUT,
            fg=TEXT,
            activebackground=HOVER,
            activeforeground=TEXT,
            bd=0,
            highlightthickness=0,
            font=("Sans", 9),
            relief="flat",
            anchor="w",
            padx=10,
        )
        self.conf_menu = tk.Menu(
            self.conf_mb,
            tearoff=0,
            bg=CARD,
            fg=TEXT,
            activebackground=BLUE,
            activeforeground=TEXT,
            bd=0,
        )
        self.conf_mb["menu"] = self.conf_menu
        for opt in conf_opts:
            self.conf_menu.add_command(
                label=opt, command=lambda o=opt: self.crop_conform.set(o)
            )
        self.conf_mb.pack(fill="x", ipady=8)
        ToolTip(self.conf_mb, self.tooltips.get("crop_conform"))

        c_col1a = tk.Frame(conform_row, bg=BG)
        c_col1a.pack(side="left", fill="x", expand=True, padx=3)
        tk.Label(c_col1a, text="If too large:", bg=BG, fg=DIM, font=("Sans", 9)).pack(
            anchor="w", pady=(0, 2)
        )
        pol_opts = ["Snap Down", "Pad to Fit"]
        self.pol_mb = tk.Menubutton(
            c_col1a,
            textvariable=self.crop_conform_policy,
            bg=INPUT,
            fg=TEXT,
            activebackground=HOVER,
            activeforeground=TEXT,
            bd=0,
            highlightthickness=0,
            font=("Sans", 9),
            relief="flat",
            anchor="w",
            padx=10,
        )
        self.pol_menu = tk.Menu(
            self.pol_mb,
            tearoff=0,
            bg=CARD,
            fg=TEXT,
            activebackground=BLUE,
            activeforeground=TEXT,
            bd=0,
        )
        self.pol_mb["menu"] = self.pol_menu
        for opt in pol_opts:
            self.pol_menu.add_command(
                label=opt, command=lambda o=opt: self.crop_conform_policy.set(o)
            )
        self.pol_mb.pack(fill="x", ipady=8)
        ToolTip(self.pol_mb, "How to handle if expanded crop exceeds image boundaries")

        c_col1c = tk.Frame(conform_row, bg=BG)
        c_col1c.pack(side="left", fill="x", expand=True, padx=3)
        tk.Label(c_col1c, text="Pad Color:", bg=BG, fg=DIM, font=("Sans", 9)).pack(
            anchor="w", pady=(0, 2)
        )
        clr_opts = ["Black", "White", "Grey"]
        self.clr_mb = tk.Menubutton(
            c_col1c,
            textvariable=self.crop_pad_color,
            bg=INPUT,
            fg=TEXT,
            activebackground=HOVER,
            activeforeground=TEXT,
            bd=0,
            highlightthickness=0,
            font=("Sans", 9),
            relief="flat",
            anchor="w",
            padx=10,
        )
        self.clr_menu = tk.Menu(
            self.clr_mb,
            tearoff=0,
            bg=CARD,
            fg=TEXT,
            activebackground=BLUE,
            activeforeground=TEXT,
            bd=0,
        )
        self.clr_mb["menu"] = self.clr_menu
        for opt in clr_opts:
            self.clr_menu.add_command(
                label=opt, command=lambda o=opt: self.crop_pad_color.set(o)
            )
        self.clr_mb.pack(fill="x", ipady=8)
        ToolTip(self.clr_mb, "Color to use for padding")

        c_col1b = tk.Frame(conform_row, bg=BG)
        c_col1b.pack(side="left", fill="x", expand=True, padx=3)
        tk.Label(c_col1b, text="Min Side (px):", bg=BG, fg=DIM, font=("Sans", 9)).pack(
            anchor="w", pady=(0, 2)
        )
        min_opts = ["256", "512", "768", "1024", "1280", "1536", "1792", "2048"]
        self.min_mb = tk.Menubutton(
            c_col1b,
            textvariable=self.crop_min_side,
            bg=INPUT,
            fg=TEXT,
            activebackground=HOVER,
            activeforeground=TEXT,
            bd=0,
            highlightthickness=0,
            font=("Sans", 9),
            relief="flat",
            anchor="w",
            padx=10,
        )
        self.min_menu = tk.Menu(
            self.min_mb,
            tearoff=0,
            bg=CARD,
            fg=TEXT,
            activebackground=BLUE,
            activeforeground=TEXT,
            bd=0,
        )
        self.min_mb["menu"] = self.min_menu
        for opt in min_opts:
            self.min_menu.add_command(
                label=opt, command=lambda o=opt: self.crop_min_side.set(o)
            )
        self.min_mb.pack(fill="x", ipady=8)
        ToolTip(self.min_mb, self.tooltips.get("crop_min_side"))

        c_col2 = tk.Frame(conform_row, bg=BG)
        c_col2.pack(side="left", fill="x", expand=True, padx=(3, 0))
        tk.Label(
            c_col2, text="Safety Margin (%):", bg=BG, fg=DIM, font=("Sans", 9)
        ).pack(anchor="w", pady=(0, 2))
        ent_margin = tk.Entry(
            c_col2,
            textvariable=self.crop_margin,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            font=("Sans", 9),
            insertbackground=BLUE,
            justify="center",
        )
        ent_margin.pack(fill="x", ipady=8)
        ToolTip(ent_margin, self.tooltips.get("crop_margin"))

        tk.Frame(f, height=15, bg=BG).pack()
        self.btn_start_crop = self.btn(
            f, "START CROP PROCESSING", BLUE, self.toggle_crop
        )
        ToolTip(self.btn_start_crop, self.tooltips.get("start_crop"))
        tk.Frame(f, height=10, bg=BG).pack()
        self.crop_progress = ttk.Progressbar(f, mode="determinate")
        self.crop_progress.pack(fill="x")
        self.crop_log_lbl = tk.Label(f, text="Ready.", bg=BG, fg=DIM, font=("Sans", 8))
        self.crop_log_lbl.pack(pady=(4, 0))
        tk.Label(
            f, text="Processed Files:", bg=BG, fg=DIM, font=("Sans", 9, "bold")
        ).pack(anchor="w", pady=(10, 5))
        self.crop_list_frame = tk.Frame(f, bg=BG, height=150)
        self.crop_list_frame.pack(fill="x")
        self.crop_list_frame.pack_propagate(False)
        self.crop_list = tk.Listbox(
            self.crop_list_frame,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            highlightthickness=0,
            font=("Sans", 8),
            selectbackground=INPUT,
            activestyle="none",
        )
        self.crop_list.pack(side="left", fill="both", expand=True)
        s_crop = tk.Scrollbar(
            self.crop_list_frame,
            orient="vertical",
            command=self.crop_list.yview,
            width=30,
            bg=CARD,
            troughcolor=BG,
            bd=0,
            highlightthickness=0,
            activebackground=HOVER,
            elementborderwidth=0,
        )
        s_crop.pack(side="right", fill="y")
        self.crop_list.configure(yscrollcommand=s_crop.set)

    def toggle_crop(self):
        if self.crop_worker and self.crop_worker.running:
            self.crop_worker.running = False
            self.btn_start_crop.config(text="Stopping...", bg=DIM)
        else:
            self.start_crop()

    def crop_select_in(self):
        path = self._folder_picker("Select Input Folder")
        if path:
            self.crop_in.set(path)
            (
                not self.crop_out.get()
                and self.crop_out.set(str(Path(path) / "cropped_humans"))
            )

    def start_crop(self):
        input_path = self.crop_in.get()
        if not os.path.isdir(input_path):
            messagebox.showwarning("Error", "Input folder does not exist.")
            return
        self.crop_log_lbl.config(text="Starting VLM Processing...")
        self.root.update_idletasks()
        self.btn_start_crop.config(text="STOP PROCESSING", bg=RED)
        self.crop_list.delete(0, "end")
        try:
            self.crop_worker = VLMCropWorker(
                input_path,
                self.crop_out.get(),
                self.crop_target.get() or "human",
                self.crop_include.get(),
                self.crop_mogrify.get(),
                self.crop_square.get(),
                self.crop_one_per.get(),
                self.crop_conform.get(),
                self.crop_min_side.get(),
                self.crop_margin.get(),
                self.crop_conform_policy.get(),
                self.crop_pad_color.get(),
                self.config,
                self.update_crop_progress,
                self.update_crop_log,
                self.crop_finished,
                self.add_crop_list_item,
            )
            threading.Thread(target=self.crop_worker.run, daemon=True).start()
        except Exception as e:
            self.update_crop_log(f"Start Error: {e}")
            self._crop_reset_ui()

    def update_crop_progress(self, current, total):
        self.root.after(
            0, lambda: self.crop_progress.configure(maximum=total, value=current)
        )

    def update_crop_log(self, msg):
        self.root.after(0, lambda: self.crop_log_lbl.config(text=msg))

    def add_crop_list_item(self, msg, color=TEXT):
        def _add():
            self.crop_list.insert("end", msg)
            self.crop_list.itemconfig("end", fg=color)
            self.crop_list.see("end")

        self.root.after(0, _add)

    def crop_finished(self):
        self.root.after(0, self._crop_reset_ui)

    def _crop_reset_ui(self):
        self.btn_start_crop.config(
            state="normal", bg=BLUE, text="START CROP PROCESSING"
        )
        self.crop_log_lbl.config(text="Finished.")
        if self.crop_worker:
            self.crop_worker.running = False

    def build_filter(self):
        f = tk.Frame(self.tab_filter, bg=BG)
        f.pack(fill="both", expand=True, padx=25, pady=20)
        tk.Label(f, text="Image-Caption folder:", bg=BG, fg=DIM, font=("Sans", 9)).pack(
            anchor="w"
        )
        self.filter_src_var = tk.StringVar()
        f_src = self.field(f, "", self.filter_src_var, True, kind="folder")
        ToolTip(f_src, self.tooltips.get("filter_src"))
        tk.Frame(f, height=8, bg=BG).pack()
        tk.Label(
            f, text="Keyword (case-insensitive):", bg=BG, fg=DIM, font=("Sans", 9)
        ).pack(anchor="w")
        self.filter_kw_var = tk.StringVar()
        kw_entry = tk.Entry(
            f,
            textvariable=self.filter_kw_var,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            font=("Sans", 10),
            insertbackground=BLUE,
        )
        kw_entry.pack(fill="x", ipady=6)
        ToolTip(kw_entry, self.tooltips.get("filter_kw"))
        tk.Frame(f, height=8, bg=BG).pack()
        tk.Label(f, text="Target folder:", bg=BG, fg=DIM, font=("Sans", 9)).pack(
            anchor="w"
        )
        self.filter_tgt_var = tk.StringVar()
        f_tgt = self.field(f, "", self.filter_tgt_var, True, kind="folder")
        ToolTip(f_tgt, self.tooltips.get("filter_tgt"))
        tk.Frame(f, height=15, bg=BG).pack()
        btn_m = self.btn(f, "Move matched pairs", BLUE, self.move_keyword_pairs)
        btn_m.pack(anchor="e")
        ToolTip(btn_m, self.tooltips.get("filter_move"))
        tk.Frame(f, height=15, bg=BG).pack()
        log_frame = tk.Frame(f, bg=BG)
        log_frame.pack(fill="both", expand=True)
        self.filter_log = tk.Text(
            log_frame,
            height=10,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            font=("Monospace", 8),
            wrap="word",
            state="disabled",
        )
        self.filter_log.pack(side="left", fill="both", expand=True)
        s_filter = tk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.filter_log.yview,
            width=30,
            bg=CARD,
            troughcolor=BG,
            bd=0,
            highlightthickness=0,
            activebackground=HOVER,
            elementborderwidth=0,
        )
        s_filter.pack(side="right", fill="y", padx=(5, 0))
        self.filter_log.configure(yscrollcommand=s_filter.set)

    def move_keyword_pairs(self):
        src, kw, tgt = (
            Path(self.filter_src_var.get()),
            self.filter_kw_var.get().strip().lower(),
            Path(self.filter_tgt_var.get()),
        )
        if not (src.is_dir() and tgt.is_dir() and kw):
            messagebox.showwarning("Input needed", "Please fill / validate all fields.")
            return
        self.filter_log.config(state="normal")
        self.filter_log.delete("1.0", "end")
        self.filter_log.insert("end", f"Searching for keyword: {kw}\n")
        self.filter_log.config(state="disabled")
        threading.Thread(
            target=self._move_worker, args=(src, kw, tgt), daemon=True
        ).start()

    def _move_worker(self, src: Path, kw: str, tgt: Path):
        matched = 0
        try:
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
                for img in sorted(src.glob(ext)):
                    txt = img.with_suffix(".txt")
                    if not txt.is_file():
                        continue
                    try:
                        if kw in txt.read_text(encoding="utf-8").lower():
                            shutil.move(str(img), str(tgt / img.name))
                            shutil.move(str(txt), str(tgt / txt.name))
                            matched += 1
                            self.root.after(
                                0, lambda m=img.name: self._log_filter(f"moved: {m}")
                            )
                    except Exception as e:
                        self.root.after(
                            0,
                            lambda m=img.name, err=str(e): self._log_filter(
                                f"error on {m}: {err}"
                            ),
                        )
        finally:
            self.root.after(
                0, lambda: self._log_filter(f"Done. Moved {matched} pairs.")
            )

    def _log_filter(self, msg):
        self.filter_log.config(state="normal")
        self.filter_log.insert("end", msg + "\n")
        self.filter_log.see("end")
        self.filter_log.config(state="disabled")

    def load_editor_folder(self, auto_path=None):
        path = Path(auto_path) if auto_path else Path(self._folder_picker())
        if path and path.is_dir():
            self.current_editor_folder = path
            self.config.set("last_editor_folder", str(path))
            self.editor_folder_label.config(text=path.name)
            self.load_editor_images()

    def create_editor_item(self, img_path: Path, thumb_img=None):
        base_height = self.thumb_size + 30
        item_frame = tk.Frame(self.img_list_frame, bg=CARD, height=base_height)
        item_frame.pack(fill="x", pady=5, padx=2)
        item_frame.pack_propagate(False)
        img_col_width = self.thumb_size + 20
        img_container = tk.Frame(item_frame, bg=BG, width=img_col_width)
        img_container.pack(side="left", fill="y")
        img_container.pack_propagate(False)
        img_label = tk.Label(img_container, bg=BG, fg=DIM)
        if thumb_img:
            photo = ImageTk.PhotoImage(thumb_img)
            img_label.configure(image=photo)
            img_label.image = photo
        else:
            img_label.configure(text="Loading...")
        img_label.pack(expand=True)
        img_label.bind("<Double-Button-1>", lambda e: self.show_zoom(img_path))
        ToolTip(img_label, "Double-click to zoom")
        right_container = tk.Frame(item_frame, bg=CARD)
        right_container.pack(side="left", fill="both", expand=True, padx=12, pady=8)
        header = tk.Frame(right_container, bg=CARD)
        header.pack(fill="x", pady=(0, 5))
        tk.Label(
            header,
            text=img_path.name,
            bg=CARD,
            fg=DIM,
            font=("Sans", 9, "bold"),
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        btn_expand = tk.Button(
            header,
            text="▼ Expand",
            bg=BLUE,
            fg="white",
            bd=0,
            relief="flat",
            font=("Sans", 8, "bold"),
            activebackground=HOVER,
            activeforeground="white",
            cursor="hand2",
            highlightthickness=0,
        )
        btn_expand.pack(side="right")
        ToolTip(btn_expand, "Toggle large view")
        txt_path = img_path.with_suffix(".txt")
        initial_content = ""
        if txt_path.exists():
            try:
                initial_content = txt_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
        text_area = tk.Text(
            right_container,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            font=("Sans", 10),
            wrap="word",
            insertbackground=BLUE,
        )
        text_area.insert("1.0", initial_content)
        text_area.pack(fill="both", expand=True)
        item_frame.expanded = False

        def _toggle_expand():
            is_exp = not item_frame.expanded
            item_frame.expanded = is_exp
            if is_exp:
                btn_expand.config(text="▲ Collapse", bg=BG)
                item_frame.pack_propagate(True)
                text_area.config(height=25)
                try:
                    img = Image.open(img_path)
                    target_size = 500
                    img.thumbnail((target_size, target_size), Image.LANCZOS)
                    ph = ImageTk.PhotoImage(img)
                    img_label.configure(image=ph)
                    img_label.image = ph
                    img_container.config(width=target_size + 20)
                except Exception:
                    pass
            else:
                btn_expand.config(text="▼ Expand", bg=BLUE)
                item_frame.pack_propagate(False)
                item_frame.config(height=base_height)
                text_area.config(height=1)
                try:
                    thumb = self.thumb_cache.get(str(img_path))
                    if thumb:
                        ph = ImageTk.PhotoImage(thumb)
                        img_label.configure(image=ph)
                        img_label.image = ph
                    img_container.config(width=img_col_width)
                except Exception:
                    pass

        btn_expand.config(command=_toggle_expand)

        def _save(event=None):
            try:
                txt_path.write_text(text_area.get("1.0", "end-1c"), encoding="utf-8")
            except Exception:
                pass

        text_area.bind("<KeyRelease>", _save)
        self.editor_items.append((img_path, item_frame))
        return img_label

    def load_editor_images(self, filter_text=None):
        for widget in self.img_list_frame.winfo_children():
            widget.destroy()
        self.editor_items = []
        self.all_filtered_paths = []
        self.loaded_count = 0
        self.is_loading_more = False
        if not self.current_editor_folder:
            return
        threading.Thread(
            target=self._find_paths_worker, args=(filter_text,), daemon=True
        ).start()

    def _find_paths_worker(self, filter_text):
        paths = []
        for ext in (
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.webp",
            "*.bmp",
            "*.PNG",
            "*.JPG",
            "*.JPEG",
            "*.WEBP",
            "*.BMP",
        ):
            paths.extend(self.current_editor_folder.glob(ext))
        paths = sorted(list(set(paths)))
        if filter_text:
            filtered = []
            for p in paths:
                txt = p.with_suffix(".txt")
                if not txt.exists():
                    txt = Path(str(p) + ".txt")
                if txt.exists():
                    try:
                        if (
                            filter_text.lower()
                            in txt.read_text(encoding="utf-8", errors="ignore").lower()
                        ):
                            filtered.append(p)
                    except Exception:
                        pass
            paths = filtered
        self.root.after(
            0,
            lambda: (
                setattr(self, "all_filtered_paths", paths),
                self.load_more_items(),
            ),
        )

    def load_more_items(self):
        if self.is_loading_more or self.loaded_count >= len(self.all_filtered_paths):
            return
        self.is_loading_more = True
        batch_size = 10
        end = min(self.loaded_count + batch_size, len(self.all_filtered_paths))
        batch_paths = self.all_filtered_paths[self.loaded_count : end]
        img_labels = []
        for p in batch_paths:
            lbl = self.create_editor_item(p, None)
            img_labels.append((p, lbl))
        self.loaded_count = end
        threading.Thread(
            target=self._load_thumbnails_batch, args=(img_labels,), daemon=True
        ).start()

    def _load_thumbnails_batch(self, items):
        for path, lbl in items:
            thumb = self.thumb_cache.get(str(path))
            if not thumb:
                try:
                    img = Image.open(path)
                    img.thumbnail((self.thumb_size, self.thumb_size), Image.LANCZOS)
                    thumb = img
                    self.thumb_cache[str(path)] = thumb
                except Exception:
                    pass
            if thumb:
                self.root.after(
                    0, lambda label=lbl, t=thumb: self._update_thumb(label, t)
                )
        self.is_loading_more = False

    def _update_thumb(self, label, thumb_img):
        try:
            photo = ImageTk.PhotoImage(thumb_img)
            label.configure(image=photo, text="")
            label.image = photo
        except Exception:
            pass

    def apply_editor_filter(self):
        self.load_editor_images(self.editor_filter_var.get().strip())

    def clear_editor_filter(self):
        self.editor_filter_var.set("")
        self.load_editor_images()

    def show_zoom(self, img_path: Path):
        if hasattr(self, "zoom_tl"):
            self.zoom_tl.destroy()
        tl = tk.Toplevel(self.root)
        tl.title(f"Zoom – {img_path.name}")
        tl.configure(bg=BG)
        tl.transient(self.root)
        x = self.root.winfo_x() - 550
        y = self.root.winfo_y() + 100
        tl.geometry(f"500x500+{x}+{y}")
        tl.focus()
        tl.bind("<Escape>", lambda e: tl.destroy())
        img = Image.open(img_path)
        img.thumbnail((480, 480), Image.LANCZOS)
        ph = ImageTk.PhotoImage(img)
        lbl = tk.Label(tl, image=ph, bg=BG)
        lbl.image = ph
        lbl.pack(expand=True)
        close = tk.Label(tl, text="✖  Close", fg=DIM, bg=BG, cursor="hand2")
        close.pack(pady=4)
        close.bind("<Button-1>", lambda e: tl.destroy())
        self.zoom_tl = tl

    def detect_binary(self):
        paths = [
            "./build/bin/llama-server",
            "./llama-server",
            "../llama.cpp/build/bin/llama-server",
        ]
        if os.name == "nt":
            paths = [
                "llama-server.exe",
                "build/bin/Release/llama-server.exe",
                "./llama-server.exe",
            ]
        for p in paths:
            if Path(p).exists():
                self.bin.set(p)
                break

    def update_vram_info(self):
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            output = (
                subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            )
            if not output:
                raise ValueError("Empty output")
            used, total = map(int, output.split(","))
            free, percent = total - used, (used / total) * 100
            color = GREEN if percent < 50 else (BLUE if percent < 80 else RED)
            self.vram_label.config(
                text=f"VRAM: {used}MB used / {free}MB free / {total}MB total ({percent:.0f}%)",
                fg=color,
            )
        except Exception:
            self.vram_label.config(text="VRAM info unavailable", fg=DIM)
        self.root.after(5000, self.update_vram_info)

    def kill_gpu_processes(self):
        if not messagebox.askyesno(
            "Kill GPU Processes", "Terminate ALL GPU processes?"
        ):
            return
        try:
            pids = map(
                int,
                subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
                .split(),
            )
            killed = 0
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                    killed += 1
                except Exception:
                    pass
            threading.Timer(1.0, self.update_vram_info).start()
            messagebox.showinfo("GPU Processes", f"Killed {killed} process(es).")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to kill GPU processes:\n{e}")

    def _folder_picker(self, title="Select folder", is_server=False):
        initial = self.config.get("last_dir", str(Path.home()))
        if os.name == "nt":
            path = filedialog.askdirectory(title=title, initialdir=initial)
            if path and not is_server:
                self.config.set("last_dir", str(Path(path).parent))
            return path
        try:
            subprocess.run(["zenity", "--version"], capture_output=True, check=True)
            result = subprocess.run(
                [
                    "zenity",
                    "--file-selection",
                    "--directory",
                    "--title",
                    title,
                    f"--filename={initial}/",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                path = result.stdout.strip()
                if path:
                    if not is_server:
                        self.config.set("last_dir", str(Path(path).parent))
                    return path
            return ""
        except (FileNotFoundError, subprocess.CalledProcessError, Exception):
            pass
        return self._folder_picker_tk(title, is_server)

    def _folder_picker_tk(self, title="Select folder", is_server=False):
        out = []
        top = tk.Toplevel(self.root)
        top.title(title)
        top.geometry("600x400")
        top.configure(bg=BG)
        top.transient(self.root)
        top.grab_set()
        top.focus()
        initial = self.config.get("last_dir", str(Path.home()))
        addr = tk.StringVar(value=initial)
        bar = tk.Frame(top, bg=BG)
        bar.pack(fill="x", padx=6, pady=6)
        tk.Entry(
            bar,
            textvariable=addr,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            highlightthickness=0,
            font=("Sans", 9),
        ).pack(side="left", fill="x", expand=True, ipady=5)
        frame = tk.Frame(top, bg=BG)
        frame.pack(fill="both", expand=True)
        lb_frame = tk.Frame(frame, bg=BG)
        lb_frame.pack(fill="both", expand=True, padx=6, pady=6)
        lb = tk.Listbox(
            lb_frame,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            selectbackground=BLUE,
            selectforeground="white",
            font=("Sans", 9),
        )
        lb.pack(side="left", fill="both", expand=True)
        sc = ttk.Scrollbar(lb_frame, orient="vertical", command=lb.yview)
        sc.pack(side="right", fill="y")
        lb.configure(yscrollcommand=sc.set)
        btn_bar = tk.Frame(top, bg=BG)
        btn_bar.pack(fill="x", padx=6, pady=6)
        tk.Button(
            btn_bar,
            text="Select",
            bg=GREEN,
            fg="white",
            bd=0,
            relief="flat",
            command=lambda: _select(),
        ).pack(side="right", padx=(6, 0))
        tk.Button(
            btn_bar,
            text="Cancel",
            bg=CARD,
            fg=TEXT,
            bd=0,
            relief="flat",
            command=lambda: _cancel(),
        ).pack(side="right")

        def _populate():
            lb.delete(0, tk.END)
            p = Path(addr.get())
            if p.parent:
                lb.insert(0, "..")
            for d in sorted(
                p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
            ):
                lb.insert(tk.END, d.name + ("/" if d.is_dir() else ""))

        def _select():
            sel = lb.curselection()
            if not sel:
                path = addr.get()
                (not is_server and self.config.set("last_dir", str(Path(path).parent)))
                out.append(path)
                top.destroy()
                return
            name = lb.get(sel[0])
            if name == "..":
                addr.set(str(Path(addr.get()).parent))
            else:
                new = Path(addr.get()) / name.rstrip("/")
                if new.is_dir():
                    addr.set(str(new))
                else:
                    out.append(str(new))
                    top.destroy()
                    return
            _populate()

        def _cancel():
            top.destroy()

        lb.bind("<Double-Button-1>", lambda e: _select())
        _populate()
        self.root.wait_window(top)
        return out[0] if out else ""

    def _file_picker(self, title="Select file"):
        if os.name == "nt":
            return filedialog.askopenfilename(title=title)
        try:
            subprocess.run(["zenity", "--version"], capture_output=True, check=True)
            result = subprocess.run(
                ["zenity", "--file-selection", "--title", title],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                path = result.stdout.strip()
                if path:
                    return path
            return ""
        except (FileNotFoundError, subprocess.CalledProcessError, Exception):
            pass
        return filedialog.askopenfilename(title=title)

    def start_server(self):
        ctx_val = self.ctx.get().strip()
        (
            not (ctx_val and ctx_val.isdigit())
            and (ctx_val := DEFAULT_CTX)
            and self.ctx.set(DEFAULT_CTX)
        )
        cmd = [
            self.bin.get(),
            "-m",
            self.model.get(),
            "--port",
            self.port.get(),
            "-c",
            ctx_val,
            "-ngl",
            self.gpu.get(),
            "-b",
            DEFAULT_BATCH,
        ]
        if self.proj.get():
            cmd.extend(["--mmproj", self.proj.get()])
        cmd_str = " ".join(cmd)
        self.log.insert("end", f"Starting server with command:\n{cmd_str}\n\n")
        kwargs = (
            {"preexec_fn": os.setsid}
            if os.name != "nt"
            else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        )
        try:
            self.server_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **kwargs,
            )
            self.btn_start.config(state="disabled", bg=CARD, disabledforeground="white")
            self.btn_stop.config(state="normal", bg=RED)
            threading.Thread(target=self.watch_server, daemon=True).start()
        except Exception as e:
            self.log.insert("end", f"Error: {e}\n")

    def stop_server(self):
        if self.server_proc:
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(self.server_proc.pid), signal.SIGTERM)
                else:
                    self.server_proc.terminate()
                threading.Thread(target=self.server_proc.wait, daemon=True).start()
            except Exception as e:
                self.log.insert("end", f"Stop error: {e}\n")
        self.root.after(100, self.reset_ui)

    def watch_server(self):
        try:
            for line in iter(self.server_proc.stdout.readline, ""):
                if line:
                    if "out of memory" in line.lower() or "cudaMalloc failed" in line:
                        line += "\n>>> ERROR: GPU OUT OF MEMORY\n>>> TRY: Lower 'GPU Layers' to 33 or 20, then try again.\n"
                    self.log.insert("end", line)
                    self.log.see("end")
        except Exception:
            pass
        self.root.after(0, self.reset_ui)

    def reset_ui(self):
        self.btn_start.config(state="normal", bg=BLUE)
        self.btn_stop.config(state="disabled", bg=CARD, disabledforeground="white")
        self.log.insert("end", "Server stopped\n")

    def add_folder(self):
        path = Path(self._folder_picker("Select folder"))
        (path and path.is_dir() and self._add_path_to_queue(path))

    def add_current_folder_to_batch(self):
        if self.current_editor_folder and self.current_editor_folder.is_dir():
            self._add_path_to_queue(self.current_editor_folder)
            messagebox.showinfo(
                "Batch Queue",
                f"Added '{self.current_editor_folder.name}' to Batch Queue.",
            )
        else:
            messagebox.showwarning(
                "Batch Queue", "No folder currently loaded in editor."
            )

    def _add_path_to_queue(self, path: Path):
        item = QueueItem(
            self.queue_scroll.content,
            path,
            self.remove_item,
            self.update_queue_summary,
            self.config,
        )
        item.pack(fill="x", pady=(0, 6), padx=4)
        self.queue.append(item)

    def start_batch_if_needed(self):
        (not self.batch_running and self.toggle_batch())

    def update_queue_summary(self):
        self.queue_summary.delete(0, "end")
        for item in self.queue:
            if item.status == "processing":
                self.queue_summary.insert("end", f"▶ {item.folder_path.name}")
            elif item.status == "pending":
                self.queue_summary.insert("end", f"• {item.folder_path.name}")

    def remove_item(self, item):
        if item.status != "processing":
            item.destroy()
            (item in self.queue and self.queue.remove(item))
            self.update_queue_summary()

    def log_status(self, msg):
        self.status_log.config(state="normal")
        self.status_log.insert("end", f"{msg}\n")
        self.status_log.see("end")
        self.status_log.config(state="disabled")

    def toggle_batch(self):
        if self.batch_running:
            self.batch_running = False
            self.btn_proc.config(text="Start Processing", bg=BLUE)
            self.prog_lbl.config(text="Stopping...")
        else:
            try:
                from openai import OpenAI

                self.client = OpenAI(base_url=API_URL, api_key="sk-no-key")
                self.client.models.list()
            except Exception as e:
                messagebox.showerror(
                    "Connection Error", f"Cannot connect to server.\n{e}"
                )
                return
            self.status_log.config(state="normal")
            self.status_log.delete("1.0", "end")
            self.status_log.config(state="disabled")
            self.batch_running = True
            self.btn_proc.config(text="Stop Processing", bg=RED)
            threading.Thread(target=self.run_batch, daemon=True).start()

    def run_batch(self):
        self.root.after(0, lambda: self.prog_lbl.config(text="Processing..."))
        while self.batch_running:
            item = next((i for i in self.queue if i.status == "pending"), None)
            if not item:
                break
            item.status = "processing"
            self.root.after(0, lambda i=item: i.set_status("processing", "Scanning..."))
            try:
                imgs = sorted(
                    [
                        p
                        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
                        for p in Path(item.folder_path).glob(ext)
                    ]
                )
            except Exception:
                self.root.after(0, lambda i=item: i.set_status("error", "Access Error"))
                continue
            total_imgs, done = len(imgs), 0
            for img in imgs:
                if not self.batch_running:
                    break
                txt = img.with_suffix(".txt")
                if txt.exists() and not item.overwrite_var.get():
                    done += 1
                    continue
                try:
                    prompt = item.get_prompt() or DEFAULT_PROMPT
                    b64 = base64.b64encode(img.read_bytes()).decode()
                    try:
                        m_tokens = int(self.max_tokens.get())
                    except Exception:
                        m_tokens = 1024
                    resp = self.client.chat.completions.create(
                        model=Path(self.model.get()).stem,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{b64}"
                                        },
                                    },
                                ],
                            }
                        ],
                        max_tokens=m_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        frequency_penalty=0.2,
                    )
                    txt.write_text(
                        resp.choices[0].message.content.strip(), encoding="utf-8"
                    )
                    done += 1
                    self.root.after(
                        0, lambda name=img.name: self.log_status(f"✓ {name}")
                    )
                except Exception as e:
                    err_msg = str(e)
                    if "context size" in err_msg.lower():
                        err_msg = "CONTEXT ERROR: Try increasing 'Context'."
                    self.root.after(
                        0,
                        lambda name=img.name, err=err_msg: self.log_status(
                            f"✗ {name}: {err}"
                        ),
                    )
                current_queue = list(self.queue)
                q_len = len(current_queue)
                if q_len > 0:
                    processed_count = sum(
                        1
                        for q_item in current_queue
                        if q_item.status in ["done", "error"]
                        and current_queue.index(q_item) < current_queue.index(item)
                    )
                    item_pct = (done / total_imgs) if total_imgs > 0 else 0
                    total_pct = ((processed_count + item_pct) / q_len) * 100
                    self.root.after(
                        0, lambda p=total_pct: self.progress.configure(value=p)
                    )
                self.root.after(
                    0,
                    lambda d=done, t=total_imgs, i=item: i.set_status(
                        "processing", f"{d}/{t}"
                    ),
                )
            status = "done" if self.batch_running else "error"
            msg = "Complete" if self.batch_running else "Stopped"
            self.root.after(0, lambda i=item, s=status, m=msg: i.set_status(s, m))
            if not self.batch_running:
                break
        self.batch_running = False
        self.root.after(
            0, lambda: self.btn_proc.config(text="Start Processing", bg=BLUE)
        )
        self.root.after(0, lambda: self.prog_lbl.config(text="Idle"))

    def field(self, parent, label, var, browse, kind="folder"):
        if label:
            tk.Label(parent, text=label, bg=BG, fg=DIM, font=("Sans", 9, "bold")).pack(
                anchor="w", pady=(0, 4)
            )
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=(0, 12))
        entry = tk.Entry(
            row,
            textvariable=var,
            bg=INPUT,
            fg=TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            font=("Sans", 10),
            insertbackground=CYAN,
        )
        entry.pack(side="left", fill="x", expand=True, ipady=8)
        entry.bind("<Enter>", lambda e: entry.config(bg=HOVER))
        entry.bind("<Leave>", lambda e: entry.config(bg=INPUT))

        def _set_file():
            var.set(self._file_picker(f"Select {label if label else 'file'}"))

        def _set_folder():
            is_server = any(x in str(var) for x in ["model", "projector", "binary"])
            var.set(
                self._folder_picker(
                    f"Select {label if label else 'folder'}", is_server=is_server
                )
            )

        if browse:
            cmd = _set_file if kind == "file" else _set_folder
            browse_btn = tk.Button(
                row,
                text="…",
                bg=CARD,
                fg=TEXT,
                bd=0,
                relief="flat",
                highlightthickness=0,
                width=4,
                font=("Sans", 10, "bold"),
                command=cmd,
            )
            browse_btn.pack(side="right", padx=(4, 0))
            browse_btn.bind("<Enter>", lambda e: browse_btn.config(bg=HOVER))
            browse_btn.bind("<Leave>", lambda e: browse_btn.config(bg=CARD))
        return entry

    def btn(self, parent, text, color, cmd):
        btn = tk.Button(
            parent,
            text=text,
            bg=color,
            fg="white",
            bd=0,
            relief="flat",
            highlightthickness=0,
            font=("Sans", 9, "bold"),
            cursor="hand2",
            command=cmd,
            activebackground=HOVER,
            activeforeground=TEXT,
            disabledforeground="white",
        )
        btn.pack(fill="x", ipady=8)
        btn.bind("<Enter>", lambda e: btn.config(bg=HOVER))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        return btn

    def on_close(self):
        # Stop all workers first
        if hasattr(self, "batch_running"):
            self.batch_running = False

        if hasattr(self, "crop_worker") and self.crop_worker:
            self.crop_worker.running = False

        # Give threads a moment to clean up
        self.root.update()

        # Terminate server process properly
        if self.server_proc:
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(self.server_proc.pid), signal.SIGKILL)
                else:
                    self.server_proc.kill()
                self.server_proc.wait(timeout=2)
            except Exception as e:
                print(f"Server cleanup error: {e}")
            finally:
                self.server_proc = None

        # Destroy the window
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
