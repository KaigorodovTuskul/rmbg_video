import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModelForImageSegmentation


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Torch model launcher for local workfolder inputs.")
    p.add_argument("--input-source", required=True, help="Input file name inside workfolder/")
    p.add_argument("--model-dir", required=True, help="Local HF model dir with safetensors/config.")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--mask-threshold", type=float, default=0.65)
    p.add_argument("--soft-mask", action="store_true")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def pick_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_pred_tensor(model_out):
    if torch.is_tensor(model_out):
        return model_out
    if isinstance(model_out, (list, tuple)):
        for item in reversed(model_out):
            if torch.is_tensor(item):
                return item
    if isinstance(model_out, dict):
        for v in model_out.values():
            if torch.is_tensor(v):
                return v
    if hasattr(model_out, "logits") and torch.is_tensor(model_out.logits):
        return model_out.logits
    raise RuntimeError("Unsupported model output type for mask extraction.")


def build_mask(frame_bgr: np.ndarray, model, device: str, width: int, height: int, threshold: float, soft_mask: bool) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    inp = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    inp = inp / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    inp = (inp - mean) / std

    with torch.inference_mode():
        out = model(inp)
        pred = get_pred_tensor(out)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        pred = torch.sigmoid(pred)
        pred = torch.nn.functional.interpolate(pred, size=(frame_bgr.shape[0], frame_bgr.shape[1]), mode="bilinear", align_corners=False)
        if soft_mask:
            mask = ((1.0 - pred[0, 0]).clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
        else:
            mask = (pred[0, 0] <= threshold).to(torch.uint8).cpu().numpy() * 255
    return mask.astype(np.uint8)


def apply_mask(frame_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    alpha = (mask_u8.astype(np.float32) / 255.0)[..., None]
    out = (frame_bgr.astype(np.float32) * alpha).astype(np.uint8)
    return out


def process_video(input_path: Path, output_path: Path, model, device: str, width: int, height: int, threshold: float, soft_mask: bool) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask = build_mask(frame, model, device, width, height, threshold, soft_mask)
        out = apply_mask(frame, mask)
        writer.write(out)

    writer.release()
    cap.release()


def process_image(input_path: Path, output_path: Path, model, device: str, width: int, height: int, threshold: float, soft_mask: bool) -> None:
    frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {input_path}")
    mask = build_mask(frame, model, device, width, height, threshold, soft_mask)
    out = apply_mask(frame, mask)
    cv2.imwrite(str(output_path), out)


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)
    print(f"Using device: {device}")

    model = AutoModelForImageSegmentation.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval().to(device)

    input_path = Path("workfolder") / args.input_source
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}")
        return 1

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    suffix = input_path.suffix.lower()

    if suffix in VIDEO_EXTS:
        output_path = out_dir / f"{stem}_torch.mp4"
        process_video(input_path, output_path, model, device, args.width, args.height, args.mask_threshold, args.soft_mask)
    elif suffix in IMAGE_EXTS:
        output_path = out_dir / f"{stem}_torch.png"
        process_image(input_path, output_path, model, device, args.width, args.height, args.mask_threshold, args.soft_mask)
    else:
        print(f"ERROR: unsupported input extension: {suffix}")
        return 1

    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

