# RMBG Video Pipeline (Torch / ONNX / TRT)

This repository supports three inference backends:
- Torch
- ONNX
- TensorRT (optional)

It also includes a fast Hugging Face model downloader + converter pipeline based on `TRT_MIXED_PRECISION_PLAYBOOK.md`.

## 1) Base Setup

Run:

```bat
setup.bat
```

Base setup installs:
- Embedded Python 3.12.10
- PyTorch CUDA (`cu130`)
- Core Python dependencies from `requirements.txt`

TensorRT is intentionally moved to optional installer:

```bat
install_tensorrt_optional.bat
```

FFmpeg optional installer:

```bat
install_ffmpeg_optional.bat
```

## 2) Environment

Edit `.env`:

```env
FFMPEG_PATH=...
FFPROBE_PATH=...
HUGGINGFACE_TOKEN=
```

Notes:
- `HUGGINGFACE_TOKEN` is optional for public models.
- `HUGGINGFACE_TOKEN` is required for private HF repos.
- `FFMPEG_PATH/FFPROBE_PATH` are needed for TRT video pipeline.

## 3) Download HF model and convert to ONNX/TRT

Interactive launcher:

```bat
download_convert_hf_to_trt.bat
```

This pipeline:
1. Downloads `.safetensors` + config files from Hugging Face (accelerated via `hf_transfer`).
2. Exports ONNX FP32 opset17.
3. Builds TRT mixed engine with FP32 safeguards for sensitive layers.

Artifacts:
- Local HF snapshot: `models/hf/<repo_slug>/...`
- ONNX: `models/<repo_slug>_<W>x<H>_b<B>_opset17_fp32.onnx`
- TRT: `models/<repo_slug>_<W>x<H>_b<B>_opset17_mixed.engine`

## 4) Launchers

Put input files into `workfolder/`, results are saved into `output/`.

Torch backend:

```bat
torch_launcher.bat
```

ONNX backend:

```bat
onnx_launcher.bat
```

TensorRT backend:

```bat
birefnet_trt_launcher.bat
```

## 5) Backend Selection

- If user does not have TensorRT/system deps: use `torch_launcher.bat` or `onnx_launcher.bat`.
- If user needs max speed and has TRT deps: use `birefnet_trt_launcher.bat`.

## 6) Optional Installers Summary

- TensorRT: `install_tensorrt_optional.bat`
- FFmpeg: `install_ffmpeg_optional.bat`

These are optional because some users only need Torch/ONNX path.
