import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

import torch
from transformers import AutoModelForImageSegmentation

from deform_conv2d_onnx_exporter import register_deform_conv2d_onnx_op, set_export_batch_size


def slugify_repo(repo_id: str) -> str:
    slug = repo_id.split("/")[-1].strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "_", slug)
    return slug or "hf_model"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download HF safetensors model, export ONNX FP32 opset17, build TRT mixed engine."
    )
    p.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. ZhengPeng7/BiRefNet")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--workspace-gb", type=int, default=6)
    p.add_argument("--download-workers", type=int, default=16)
    p.add_argument("--force-redownload", action="store_true")
    p.add_argument("--models-dir", default="models")
    return p.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    repo_id = args.repo_id
    models_dir = Path(args.models_dir).resolve()
    hf_root = models_dir / "hf"
    alias = slugify_repo(repo_id)
    local_model_dir = hf_root / alias
    local_model_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("HUGGINGFACE_TOKEN", "").strip() or None

    # Fast path for HF downloads.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    print(f"[1/3] Downloading model from Hugging Face: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_model_dir),
        local_dir_use_symlinks=False,
        token=token,
        max_workers=max(1, args.download_workers),
        resume_download=True,
        force_download=args.force_redownload,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.py",
            "*.txt",
            "*.md",
        ],
    )

    safetensors_files = list(local_model_dir.rglob("*.safetensors"))
    if not safetensors_files:
        print(f"ERROR: no .safetensors files found in {local_model_dir}")
        return 1

    print(f"[2/3] Exporting ONNX FP32 opset17 for {repo_id}")
    register_deform_conv2d_onnx_op(use_gathernd=False, enable_openvino_patch=False)
    set_export_batch_size(args.batch)

    model = AutoModelForImageSegmentation.from_pretrained(
        str(local_model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval().float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy = torch.randn(args.batch, 3, args.height, args.width, dtype=torch.float32, device=device)

    onnx_path = models_dir / f"{alias}_{args.width}x{args.height}_b{args.batch}_opset17_fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )
    print(f"ONNX saved: {onnx_path}")

    print(f"[3/3] Building TensorRT mixed engine")
    engine_path = models_dir / f"{alias}_{args.width}x{args.height}_b{args.batch}_opset17_mixed.engine"
    convert_script = Path(__file__).resolve().parent / "convert_birefnet_dynamic_1024_trt_mixed.py"
    cmd = [
        sys.executable,
        str(convert_script),
        "--onnx",
        str(onnx_path),
        "--output",
        str(engine_path),
        "--workspace-gb",
        str(args.workspace_gb),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    print("Done.")
    print(f"Model dir: {local_model_dir}")
    print(f"ONNX: {onnx_path}")
    print(f"Engine: {engine_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

