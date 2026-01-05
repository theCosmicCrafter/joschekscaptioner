# Joschek's Captioner

![Batch Captioning](batch2.jpg)

 vibe coded image dataset caption tool, that does exactly what i personally need. If it works for you, that's a happy accident. now includes advanced cropping features.

## Features
- **Server**: Because we all know your file organization is a mess. Point it to wherever you hid your vision models this time.
- **Batch Captioning**: Queue up folders to caption while you go contemplate what you are doing.
- **Cropping**: Advanced auto cropping with the same vision model used for captioning. Can be prompted to crop to any image content and includes advanced options for sizing. Tested only with qwen 3 vl. 
- **Caption Editor**: A groundbreaking text box to fix the AI's hallucinations. With filter function.
- **Problem Bin**: One-click functionality to yeet problematic pairs into a separate folder so you can deal with them "later" (never).

## Installation & Usage

1. **Prepare Binaries**: Download [llama-server](https://github.com/ggerganov/llama.cpp/releases) and a Vision Model: 
The following models are recommended: [Qwen3-VL-8B-Abliterated-Caption-it](https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it) (for naughty stuff) or [Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) (standart for autocropping). *Don't forget the corresponding mmproj file.*

2. **Setup & Run (Select your system):**

### Arch Linux
```bash
sudo pacman -S uv python-tk imagemagick zenity && uv run joschekscaptions.py
```

### Debian / Ubuntu
```bash
sudo apt update && sudo apt install -y python3-tk curl imagemagick zenity && curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env && uv run joschekscaptions.py
```

### macOS
```bash
brew install uv python-tk imagemagick && uv run joschekscaptions.py
```

### Windows (PowerShell)
```powershell
# Optional: Install ImageMagick via winget if needed for cropping preprocessing
# winget install ImageMagick.ImageMagick

powershell -c "irm https://astral.sh/uv/install.ps1 | iex"; $env:Path += ";$env:USERPROFILE\.local\bin"; uv run joschekscaptions.py
```

## License
GPL
