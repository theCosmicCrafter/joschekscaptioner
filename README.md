# Joschek's Captioner

![Batch Captioning](batch2.jpg)

 vibe coded image dataset caption tool, that does exactly what i personally need. If it works for you, that's a happy accident.

## Features
- **Model Selector**: Because we all know your file organization is a mess. Point it to wherever you hid your vision models this time.
- **Batch Captioning**: Queue up folders to caption while you go contemplate what you are doing.
- **Cropping**: Uses YOLOv8 to find people and crop them. It sometimes works.
- **Caption Editor**: A groundbreaking text box to fix the AI's hallucinations. With filter function.
- **Problem Bin**: One-click functionality to yeet problematic pairs into a separate folder so you can deal with them "later" (never).

## Installation

You need Python 3.10+ and an NVIDIA GPU.

### From Zipped Release (Recommended for most)

1.  **Extract:** Unzip the downloaded archive to a folder.
2.  **Install Python:** Ensure you have Python 3.10+ installed.
3.  **Setup Environment:**
    Open a terminal in the folder and run:

    **Linux/Mac:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

    **Windows:**
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

4.  **Dependencies:** Ensure `python-tk` (Linux) or `zenity` (optional but recommended for better file dialogs on Linux) are installed.
5.  **Imagemagick (Optional):** Necessary if you want to use the "Pre-process with Imagemagick" option in Automatic Cropping.
    - **Linux**: `sudo apt install imagemagick`
    - **macOS**: `brew install imagemagick`
    - **Windows**: Download from [imagemagick.org](https://imagemagick.org/script/download.php)

### Setup llama-server
You need `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp)
1. **Get the binary**: Put `llama-server` (or `.exe`) in root or `./build/bin/`.
2. **Get a model**:
   - Recommended: [Qwen3-VL-8B-Abliterated-Caption-it](https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it). If you are planning on captioning naughty stuff.
   - Alternative: Qwen VL 3 - standart. better sometimes.
   - Don't forget the mmproj file.

## Usage

**Linux:**
```bash
source venv/bin/activate
python joschekscaptions.py
```

**Windows:**
```powershell
.\venv\Scripts\activate
python joschekscaptions.py
```

1. **Server Tab**: Pick your binary and model. Hit start. There's a "Kill GPU processes" button for when things go south.
2. **Batch Tab**: Point it at images. Wait.
3. **Editor Tab**: Fix the captions.
4. **Crop Humans**: Automagical cropping.

## License
GPL
