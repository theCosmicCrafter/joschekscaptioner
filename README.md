# Joschek's Captioner

A modern, local AI captioning tool for image datasets. Optimized for Qwen-VL and other multimodal models via llama.cpp.

## Features

*   **Batch Captioning:** Process entire folders of images with customizable prompts.
*   **Manual Editor:** Review and edit captions with a clean, grid-based UI.
*   **Auto-Expand Editing:** Edit long captions comfortably with inline expansion.
*   **Smart Cropping:** Automatically crop images to human subjects using YOLO.
*   **Filtering:** Move image/caption pairs based on keyword matches.
*   **Hardware Control:** Manage context size, GPU layers, and VRAM usage directly.

## Installation (Zipped Release)

1.  **Extract the Zip:**
    Unzip the downloaded archive to a folder of your choice.

2.  **Install Python:**
    Ensure you have Python 3.10 or newer installed.

3.  **Setup Environment (Recommended):**
    Open a terminal in the extracted folder and run:

    ```bash
    # Create virtual environment
    python3 -m venv venv

    # Activate it (Linux/Mac)
    source venv/bin/activate

    # Activate it (Windows)
    venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For GPU acceleration with YOLO (auto-crop), ensure you install the CUDA-enabled version of PyTorch if on Windows/NVIDIA.*

5.  **Get `llama-server`:**
    Download the latest `llama-server` binary from the [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) and place it in the folder (or point to it in the app settings).

## Usage

Run the application:
```bash
python joschekscaptions.py
```

### Quick Start
1.  **Server Tab:** Select your `llama-server` binary and your model/projector `.gguf` files. Set Context to **16384+** for images. Click **Start Server**.
2.  **Batch Captioning:** Add a folder of images, type a prompt (or use default), and click **Start Processing**.
3.  **Manual Edit:** Load your folder to review captions. Use the **▼ Expand** button to edit long text.

## Requirements
*   Python 3.10+
*   `llama-server` (from llama.cpp)
*   A multimodal model (e.g., Qwen-VL-Chat) in GGUF format.

## License
MIT
