# ComfyUI-CSM-Nodes

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) implementing the `csm` model for text-to-speech generation.

## Features
- Node `CSM Text-to-Speech`: Generates an audio file from text.
- Parameters: text, model path, output directory, sample rate, speaker ID.

## Requirements
- Installed ComfyUI.
- Python 3.8+.
- Model weights for `csm-1b` (`ckpt.pt`) from [sesame/csm-1b](https://huggingface.co/sesame/csm-1b).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/ComfyUI-CSM-Nodes.git
   cd ComfyUI-CSM-Nodes
