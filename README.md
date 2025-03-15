# ComfyUI-CSM-Nodes

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) implementing the `CSM` model for text-to-speech generation.

![comfy-csm](https://github.com/thezveroboy/ComfyUI-CSM-Nodes/raw/main/picture.jpg)

## Features
- Node `Load CSM Checkpoint`: Loads a model checkpoint from `ComfyUI/models/sesame/`.
- Node `Load CSM Tokenizer`: Loads a tokenizer from `ComfyUI/models/sesame_tokenizer/`.
- Node `CSM Text-to-Speech`: Generates audio from text using the CSM-1B model.

## Requirements
- Installed ComfyUI.
- Python 3.10+.
- CUDA-compatible GPU (recommended).
- Model weights (`ckpt.pt`) from [sesame/csm-1b](https://huggingface.co/sesame/csm-1b).
- Tokenizer files (e.g., from Llama-3.2-1B) in `ComfyUI/models/sesame_tokenizer/<tokenizer_dir>/`.
- Dependencies listed in `requirements.txt`.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/ComfyUI-CSM-Nodes.git
   cd ComfyUI-CSM-Nodes
