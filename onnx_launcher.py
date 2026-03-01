import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX model launcher for local workfolder inputs.")
    p.add_argument("--input-source", required=True, help="Input file name inside workfolder/")
    p.add_argument("--onnx-path", required=True, help="Path to ONNX model file.")
    p.add_argument("--mask-threshold", type=float, default=0.65)
    p.add_argument("--soft-mask", action="store_true")
    p.add_argument("--providers", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def build_session(onnx_path: str, providers_mode: str) -> ort.InferenceSession:
    available = ort.get_available_providers()
    if providers_mode == "cpu":
        providers = ["CPUExecutionProvider"]
    elif providers_mode == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, providers=providers)


def build_mask(frame_bgr: np.ndarray, sess: ort.InferenceSession, threshold: float, soft_mask: bool) -> np.ndarray:
    input_meta = sess.get_inputs()[0]
    shape = input_meta.shape
    h = int(shape[2]) if isinstance(shape[2], int) else 1024
    w = int(shape[3]) if isinstance(shape[3], int) else 1024

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    x = (resized - mean) / std
    x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)

    y = sess.run(None, {input_meta.name: x})[0]
    if y.ndim == 3:
        y = y[:, None, :, :]
    y = 1.0 / (1.0 + np.exp(-y))
    y = y[0, 0]
    y = cv2.resize(y, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    if soft_mask:
        mask = np.clip((1.0 - y) * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        mask = ((y <= threshold).astype(np.uint8) * 255).astype(np.uint8)
    return mask


def apply_mask(frame_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    alpha = (mask_u8.astype(np.float32) / 255.0)[..., None]
    out = (frame_bgr.astype(np.float32) * alpha).astype(np.uint8)
    return out


def process_video(input_path: Path, output_path: Path, sess: ort.InferenceSession, threshold: float, soft_mask: bool) -> None:
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
        mask = build_mask(frame, sess, threshold, soft_mask)
        out = apply_mask(frame, mask)
        writer.write(out)

    writer.release()
    cap.release()


def process_image(input_path: Path, output_path: Path, sess: ort.InferenceSession, threshold: float, soft_mask: bool) -> None:
    frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {input_path}")
    mask = build_mask(frame, sess, threshold, soft_mask)
    out = apply_mask(frame, mask)
    cv2.imwrite(str(output_path), out)


def main() -> int:
    args = parse_args()
    sess = build_session(args.onnx_path, args.providers)
    print(f"ONNX providers: {sess.get_providers()}")

    input_path = Path("workfolder") / args.input_source
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}")
        return 1

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    suffix = input_path.suffix.lower()

    if suffix in VIDEO_EXTS:
        output_path = out_dir / f"{stem}_onnx.mp4"
        process_video(input_path, output_path, sess, args.mask_threshold, args.soft_mask)
    elif suffix in IMAGE_EXTS:
        output_path = out_dir / f"{stem}_onnx.png"
        process_image(input_path, output_path, sess, args.mask_threshold, args.soft_mask)
    else:
        print(f"ERROR: unsupported input extension: {suffix}")
        return 1

    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

