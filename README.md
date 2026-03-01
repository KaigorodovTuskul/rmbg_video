# RMBG TRT Video Pipeline

This repo is focused on:
- Downloading segmentation/matting models from Hugging Face (`.safetensors`).
- Exporting ONNX in FP32 opset17.
- Building TensorRT mixed-precision engines according to `TRT_MIXED_PRECISION_PLAYBOOK.md`.
- Running video background pipeline via `birefnet_trt_launcher.bat`.

## 1) Setup

Run:

```bat
setup.bat
```

`setup.bat` installs:
- Embedded Python 3.12.10 (optional reinstall)
- PyTorch CUDA (`cu130`)
- TensorRT Python packages (`cu13`)
- Packages from `requirements.txt`

## 2) Environment

Edit `.env`:

```env
FFMPEG_PATH=...
FFPROBE_PATH=...
HUGGINGFACE_TOKEN=
```

`HUGGINGFACE_TOKEN` is optional for public models, required for private repos or when your account/rate limits require auth.

## 3) Download + Convert HF model to ONNX/TRT

Interactive launcher:

```bat
download_convert_hf_to_trt.bat
```

What it does:
1. Downloads selected HF model (safetensors + config/code files) with accelerated transfer (`hf_transfer`).
2. Exports ONNX FP32 opset17.
3. Builds mixed TensorRT engine (FP16 global + sensitive layers forced FP32) using `convert_birefnet_dynamic_1024_trt_mixed.py`.

Artifacts:
- Downloaded model: `models/hf/<repo_slug>/...`
- ONNX: `models/<repo_slug>_<W>x<H>_b<B>_opset17_fp32.onnx`
- TRT engine: `models/<repo_slug>_<W>x<H>_b<B>_opset17_mixed.engine`

## 4) Run Video Processing

Put source files into:

`workfolder/`

Run:

```bat
birefnet_trt_launcher.bat
```

Output files are written to:

`output/`

## Notes

- Current runtime path is full-frame only (split-halves and ROI detector were removed).
- If TensorRT import fails (`No module named tensorrt`), run `setup.bat` again.
