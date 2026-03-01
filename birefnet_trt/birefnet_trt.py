import torch
import tensorrt as trt
import threading
import cv2
import numpy as np
import os
import sys
import time
import subprocess
import json
import concurrent.futures
import shutil
from PIL import Image
import asyncio
import argparse
import re
from pymediainfo import MediaInfo
from dotenv import load_dotenv
import pynvml

# Load environment variables
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--input-source', type=str, default=None)
parser.add_argument('--num_max_workers', type=int, default=16)
parser.add_argument('--base-edge', type=int, default=960)
parser.add_argument('--its-time', type=str, default=None)
parser.add_argument('--quality-mode', action='store_true')
parser.add_argument('--auto-workers', action='store_true', help='Automatically adjust workers based on free VRAM (GPU only)')
parser.add_argument('--gpu-limit', type=int, default=2, help='Initial or fixed number of concurrent batches')
parser.add_argument('--cpu', action='store_true', help='Force CPU processing')
parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'bf16'], default=None, help='Weights precision')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size for inference (higher = faster but more VRAM)')
parser.add_argument('--engine-path', type=str, default=os.path.join('models', 'trt', 'BiRefNet.engine'), help='TensorRT engine path')
parser.add_argument('--mask-threshold', type=float, default=0.65, help='Binary alpha threshold for background mask in [0,1]')
parser.add_argument('--soft-mask', action='store_true', help='Use soft alpha mask instead of binary threshold')
parser.add_argument('--pre-gamma', type=float, default=1.0, help='Input gamma correction (1.0 disables)')
parser.add_argument('--pre-grayworld', action='store_true', help='Apply gray-world white balance before normalization')
parser.add_argument('--pre-highlight-compress', type=float, default=0.0, help='Highlight compression strength in [0,1]')
parser.add_argument('--median-ksize', type=int, default=0, help='Median blur kernel size for mask postprocess (0 disables)')
parser.add_argument('--morph-open', type=int, default=0, help='Morphological OPEN iterations for mask cleanup')
parser.add_argument('--morph-close', type=int, default=0, help='Morphological CLOSE iterations for mask cleanup')
parser.add_argument('--min-island-area', type=int, default=0, help='Remove connected mask islands smaller than this area (px)')
parser.add_argument('--fill-holes', action='store_true', help='Fill holes inside mask regions')
parser.add_argument('--edge-aware-d', type=int, default=0, help='Bilateral filter diameter for edge-aware mask smoothing (0 disables)')
parser.add_argument('--edge-aware-sigma-color', type=float, default=50.0, help='Bilateral sigmaColor')
parser.add_argument('--edge-aware-sigma-space', type=float, default=50.0, help='Bilateral sigmaSpace')
parser.add_argument('--run-tag', type=str, default='', help='Extra tag appended to output filename for debug runs')
parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile optimization')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

# Determine target dtype
if args.precision:
    if args.precision == 'fp16': target_dtype = torch.float16
    elif args.precision == 'bf16': target_dtype = torch.bfloat16
    else: target_dtype = torch.float32
else:
    target_dtype = torch.float16 if device == "cuda" else torch.float32

if device == "cuda" and target_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
    print("Warning: BFloat16 is not natively supported on this GPU, performance may be poor.")

print(f"Using device: {device} | Precision: {target_dtype}")

if not args.input_source:
    sys.exit()
else:
    input_source = args.input_source

print(f"--num_max_workers {args.num_max_workers}, --base-edge {args.base_edge}, --its-time {args.its_time}, --auto-workers {args.auto_workers}, --batch-size {args.batch_size}")

num_max_workers = args.num_max_workers
if args.auto_workers:
    num_max_workers = 4

base_edge = args.base_edge

# Paths from .env
ffmpeg_path = os.getenv('FFMPEG_PATH')
ffprobe_path = os.getenv('FFPROBE_PATH')

if not ffmpeg_path or not ffprobe_path:
    print("Error: FFMPEG_PATH or FFPROBE_PATH not found in .env file")
    sys.exit(1)

sys_folders = ["downscaled_frames", "temp"]

for f in sys_folders:
    if os.path.exists(f):
        shutil.rmtree(f)
    os.makedirs(f, exist_ok=True)

downscaled_frames_pattern = os.path.join("downscaled_frames", "frame_%06d.jpg")
# No more processed_frames_pattern needed for pipes

# ── TensorRT requires CUDA ────────────────────────────────────────
if device != "cuda":
    print("ERROR: TensorRT backend requires CUDA."); sys.exit(1)

ENGINE_PATH = args.engine_path
if not os.path.exists(ENGINE_PATH):
    print(f"ERROR: engine not found: {ENGINE_PATH}")
    print("Build one first (example): python convert_to_trt.py --onnx <path> --output <path> --static-batch 1")
    sys.exit(1)

class BiRefNetTRT:
    """TensorRT wrapper — drop-in replacement for the PyTorch model.
    Engine has static batch (deform_conv2d bakes batch into constants).
    Batch size is read from the engine."""

    def __init__(self, engine_path):
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine  = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(
                "Failed to deserialize TensorRT engine. "
                f"Runtime TensorRT version: {getattr(trt, '__version__', 'unknown')}. "
                "This usually means the .engine was built with a different TensorRT/CUDA stack. "
                "Rebuild the engine on this machine with the current runtime."
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(
                "TensorRT engine loaded but execution context creation failed. "
                "Rebuild the engine and verify TensorRT/CUDA compatibility."
            )
        self.input_name  = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self._lock   = threading.Lock()
        self._stream = torch.cuda.Stream()                            # dedicated stream avoids TRT sync warnings
        self._trt_to_torch = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
        }
        in_trt_dtype = self.engine.get_tensor_dtype(self.input_name)
        out_trt_dtype = self.engine.get_tensor_dtype(self.output_name)
        if in_trt_dtype not in self._trt_to_torch or out_trt_dtype not in self._trt_to_torch:
            raise RuntimeError(f"Unsupported TRT tensor dtype(s): in={in_trt_dtype}, out={out_trt_dtype}")
        self.input_torch_dtype = self._trt_to_torch[in_trt_dtype]
        self.output_torch_dtype = self._trt_to_torch[out_trt_dtype]

        # Read batch size and dimensions from engine
        input_shape = self.engine.get_tensor_shape(self.input_name)
        self.BATCH = input_shape[0]
        self.H = input_shape[2]
        self.W = input_shape[3]
        
        print(f"Engine configuration: batch={self.BATCH}, resolution={self.H}x{self.W}")

        self.context.set_input_shape(self.input_name, [self.BATCH, 3, self.H, self.W])

        print(f"  Input:  {self.input_name}  [{self.BATCH}, 3, {self.H}, {self.W}] dtype={self.input_torch_dtype}")
        print(f"  Output: {self.output_name} {list(self.engine.get_tensor_shape(self.output_name))} dtype={self.output_torch_dtype}")

    # ── internal: exactly BATCH frames ──────────────────────────────
    def _run(self, input_tensor):
        """Execute one TRT pass for exactly BATCH frames."""
        output = torch.empty(self.BATCH, 1, self.H, self.W, dtype=self.output_torch_dtype, device="cuda")
        with self._lock:
            self.context.set_tensor_address(self.input_name,  input_tensor.data_ptr())
            self.context.set_tensor_address(self.output_name, output.data_ptr())
            self._stream.wait_stream(torch.cuda.current_stream())     # input must be ready
            if not self.context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            self._stream.synchronize()                                # block CPU until output is written
        return output

    # ── public: any batch size ──────────────────────────────────────
    def infer(self, input_tensor):
        """input_tensor: CUDA [N,3,H,W] contiguous -> CUDA [N,1,H,W]
        N < BATCH  ->  zero-padded to 4, result sliced back
        N > BATCH  ->  chunked in groups of 4"""
        if input_tensor.dtype != self.input_torch_dtype:
            input_tensor = input_tensor.to(dtype=self.input_torch_dtype)
        N = input_tensor.shape[0]
        if N == self.BATCH:
            return self._run(input_tensor)
        if N < self.BATCH:
            pad = torch.zeros(self.BATCH - N, 3, self.H, self.W, dtype=self.input_torch_dtype, device="cuda")
            return self._run(torch.cat([input_tensor, pad], dim=0))[:N]
        # N > BATCH: recurse in chunks of BATCH
        parts = []
        for i in range(0, N, self.BATCH):
            parts.append(self.infer(input_tensor[i:i + self.BATCH]))
        return torch.cat(parts, dim=0)

print(f"Loading TensorRT engine: {ENGINE_PATH}")
try:
    birefnet = BiRefNetTRT(ENGINE_PATH)
except Exception as e:
    print(f"ERROR: {e}")
    print("Hint: regenerate this .engine using one of the local convert_*_trt.py scripts.")
    sys.exit(1)
target_dtype = birefnet.input_torch_dtype
ENGINE_H = birefnet.H
ENGINE_W = birefnet.W

if args.batch_size != birefnet.BATCH:
    print(f"Note: engine is static batch={birefnet.BATCH}; runtime will pad/chunk from requested --batch-size {args.batch_size}.")

if base_edge != ENGINE_H:
    print(f"Note: --base-edge {base_edge} ignored. Using engine resolution {ENGINE_W}x{ENGINE_H}.")
PROCESS_H = ENGINE_H
PROCESS_W = ENGINE_W

print("Engine loaded.")

_engine_stem = os.path.splitext(os.path.basename(ENGINE_PATH))[0]
MODEL_NAME = re.sub(r'_(?:\d+x\d+|\d+)_b\d+$', '', _engine_stem, flags=re.IGNORECASE).lower()
if not MODEL_NAME:
    MODEL_NAME = 'birefnet_dynamic'


import warnings
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

# Pre-calculate normalization constants on GPU for speed
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=target_dtype).view(1, 3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=target_dtype).view(1, 3, 1, 1)

cuda_graph = None  # disabled — no speedup on this model

_PREPROC_ENABLED = (
    args.pre_grayworld
    or abs(args.pre_gamma - 1.0) > 1e-6
    or args.pre_highlight_compress > 1e-6
)
_POSTPROC_ENABLED = (
    args.median_ksize > 0
    or args.morph_open > 0
    or args.morph_close > 0
    or args.min_island_area > 0
    or args.fill_holes
    or args.edge_aware_d > 0
)

def _apply_input_preprocess(rgb_u8):
    if not _PREPROC_ENABLED:
        return rgb_u8
    x = rgb_u8.astype(np.float32) / 255.0

    if args.pre_grayworld:
        means = x.reshape(-1, 3).mean(axis=0) + 1e-6
        gray = float(means.mean())
        x *= (gray / means).reshape(1, 1, 3)

    if args.pre_highlight_compress > 1e-6:
        strength = float(np.clip(args.pre_highlight_compress, 0.0, 1.0))
        t = 0.75
        hi = np.maximum(x - t, 0.0)
        x = x - strength * (hi * hi) / max(1e-6, (1.0 - t))

    if abs(args.pre_gamma - 1.0) > 1e-6:
        g = max(0.1, float(args.pre_gamma))
        x = np.power(np.clip(x, 0.0, 1.0), g)

    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)

def _fill_holes_binary(mask_u8):
    """Fill holes inside foreground while preserving outside background.

    mask_u8 here is background-mask (255=background, 0=foreground).
    """
    fg = (mask_u8 == 0).astype(np.uint8)
    if fg.max() == 0:
        return mask_u8

    n, labels, _, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return mask_u8

    # Keep largest foreground component as the "main component".
    areas = [int((labels == i).sum()) for i in range(1, n)]
    main_id = 1 + int(np.argmax(areas))
    main = (labels == main_id).astype(np.uint8)

    # Find holes inside main foreground component:
    # background pixels not connected to image border.
    bg = (main == 0).astype(np.uint8)
    nb, bl, _, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)
    touch = set(np.unique(np.concatenate([bl[0, :], bl[-1, :], bl[:, 0], bl[:, -1]])))
    holes = np.zeros_like(main, dtype=np.uint8)
    for i in range(1, nb):
        if i not in touch:
            holes[bl == i] = 1

    main_filled = np.clip(main + holes, 0, 1)

    # Rebuild final background mask: foreground=0, background=255.
    out = np.where(main_filled == 1, 0, 255).astype(np.uint8)
    return out

def _remove_small_islands(mask_u8, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask_u8
    out = np.zeros_like(mask_u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

def _postprocess_mask(mask_u8, hard_binary):
    out = mask_u8

    if args.edge_aware_d > 0:
        out = cv2.bilateralFilter(out, d=args.edge_aware_d, sigmaColor=args.edge_aware_sigma_color, sigmaSpace=args.edge_aware_sigma_space)

    if args.median_ksize > 1:
        k = int(args.median_ksize)
        if k % 2 == 0:
            k += 1
        out = cv2.medianBlur(out, k)

    if hard_binary:
        out = ((out >= 128).astype(np.uint8) * 255)
        kernel = np.ones((3, 3), dtype=np.uint8)
        if args.morph_open > 0:
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=args.morph_open)
        if args.morph_close > 0:
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=args.morph_close)
        if args.min_island_area > 0:
            out = _remove_small_islands(out, args.min_island_area)
        if args.fill_holes:
            out = _fill_holes_binary(out)

    return out.astype(np.uint8)

def _infer_preds(batch_tensor):
    return birefnet.infer(batch_tensor).sigmoid()

input_filename = os.path.join("workfolder", input_source)

def get_video_info(input_source):
    command = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream",
        "-of", "json",
        input_source
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise Exception(f"ffprobe failed with error: {result.stderr}")

    probe = json.loads(result.stdout)
    stream = probe['streams'][0]
    
    width = stream['width']
    height = stream['height']
    time_base = stream['time_base']
    start_pts = stream['start_pts']
    start_time = stream['start_time']
    fps = stream['r_frame_rate']
    duration = stream['duration']
    codec = stream['codec_name']
    bitrate = str(int(int(stream.get('bit_rate', 0)) / 1_000_000)) if stream.get('bit_rate') else "0"

    print(f"{input_source} duration: {duration}, codec: {codec}, resolution: {width}x{height}, fps: {fps}, bitrate: {bitrate}, start_pts: {start_pts}, timebase: {time_base}, start_time: {start_time}")

    return width, height, fps, bitrate, start_pts, time_base, duration, codec, start_time

input_width, input_height, input_fps, input_bitrate, start_pts, time_base, duration, input_codec, start_time = get_video_info(input_filename)
_, video_extension = os.path.splitext(input_source)

# Calculate total frames for the pipeline, respecting --its-time
fps_num, fps_den = map(int, input_fps.split('/'))
actual_fps = fps_num / fps_den

if args.its_time:
    # Convert HH:MM:SS.ms to seconds
    try:
        h, m, s = args.its_time.split(':')
        total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
        total_frames = int(total_seconds * actual_fps)
    except:
        # Fallback to float if its just seconds
        total_frames = int(float(args.its_time) * actual_fps)
else:
    total_frames = int(float(duration) * actual_fps)

print(f"Total frames to process: {total_frames}")

def get_media_info(new_filename):
    media_info = MediaInfo.parse(new_filename)
    for track in media_info.tracks:
        if track.track_type == 'Video':
            frame_rate_mode = track.to_data().get('frame_rate_mode')
            source_duration_first_frame = track.to_data().get('source_duration_firstframe')
            if source_duration_first_frame:
                if int(source_duration_first_frame) < 0:
                    print(f"source_duration_first_frame: {source_duration_first_frame} is below zero.")
                    return int(source_duration_first_frame), frame_rate_mode
                else:
                    print(f"source_duration_first_frame is {source_duration_first_frame}")
                    return None, frame_rate_mode
            else:
                print("source_duration_first_frame does not exist")
                return None, frame_rate_mode
    return None, None

def get_trim(start_time=None):
    if start_time:
        print(f"start_time: {start_time}")
        milliseconds = int(float(start_time) * 1000)
        hours = milliseconds // (1000 * 60 * 60)
        milliseconds %= (1000 * 60 * 60)
        minutes = milliseconds // (1000 * 60)
        milliseconds %= (1000 * 60)
        seconds = milliseconds // 1000
        millis = milliseconds % 1000
        trim = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(millis):03}"
        print(f"trim: {trim}")
        return trim
    return None

trim = None
if int(start_pts) != 0:
    trim = get_trim(start_time=start_time)

source_duration_first_frame, frame_rate_mode = get_media_info(input_filename)

# Use only basename to avoid path issues
input_basename = os.path.basename(input_source)
new_filename = os.path.join("temp", f"{os.path.splitext(input_basename)[0]}{video_extension}")
audio_filename = os.path.join("temp", f"{os.path.splitext(input_basename)[0]}.aac")

print(f"frame_rate_mode: {frame_rate_mode}")
if frame_rate_mode == 'VFR':
    print(f"frame_rate_mode is VFR, generating new raw files and demuxing it")
    raw_video_file = os.path.join("temp", f"raw_video_file.{'h264' if '264' in input_codec else 'h265'}")
    raw_audio_file = os.path.join("temp", "raw_audio_file.aac")

    command = [
        ffmpeg_path, '-y', "-hide_banner", '-loglevel', 'error',
        "-r", input_fps,
        "-i", input_filename,
        "-map", "0:v",
        "-vcodec", "copy",
        "-bsf:v", f'{"h264_mp4toannexb" if "264" in input_codec else "hevc_mp4toannexb"}',
        "-an",
        raw_video_file
    ]

    if trim:
        command.insert(command.index('-i'), "-ss")
        command.insert(command.index('-i'), trim)

    if args.its_time:
        command.insert(command.index('-i'), "-to")
        command.insert(command.index('-i'), args.its_time)

    subprocess.run(command, check=True)

    command = [
        ffmpeg_path, '-y', "-hide_banner", '-loglevel', 'error',
        "-r", input_fps,
        "-fflags", "+genpts",
        "-i", raw_video_file,
        "-vcodec", "copy",
        "-time_base", time_base,
        "-an",
        new_filename
    ]
    subprocess.run(command, check=True)

    command = [
        ffmpeg_path, '-y', "-hide_banner", '-loglevel', 'error',
        "-r", input_fps,
        "-i", input_filename,
        "-map", "0:a",
        "-acodec", "copy",
        "-f", "adts",
        "-vn",
        raw_audio_file
    ]

    if trim:
        command.insert(command.index('-i'), "-ss")
        command.insert(command.index('-i'), trim)

    if args.its_time:
        command.insert(command.index('-i'), "-to")
        command.insert(command.index('-i'), args.its_time)

    subprocess.run(command, check=True)

    command = [
        ffmpeg_path, '-y', "-hide_banner", '-loglevel', 'error',
        "-r", input_fps,
        "-fflags", "+genpts",
        "-i", raw_audio_file,
        "-acodec", "copy",
        "-time_base", time_base,
        "-vn",
        audio_filename
    ]
    subprocess.run(command, check=True)
    input_filename = new_filename
else:
    print(f"frame_rate_mode is CFR, just get video and audio tracks")

    command = [
        ffmpeg_path, '-y', "-hide_banner", '-loglevel', 'error',
        "-i", input_filename,
        "-vcodec", "copy",
        "-an",
        new_filename
    ]
    if trim:
        command.insert(command.index('-i'), "-ss")
        command.insert(command.index('-i'), trim)

    if args.its_time:
        command.insert(command.index('-i'), "-to")
        command.insert(command.index('-i'), args.its_time)

    subprocess.run(command, check=True)

    command = [
        ffmpeg_path, '-y', "-hide_banner", '-loglevel', 'error',
        "-i", input_filename,
        "-acodec", "copy",
        "-vn",
        audio_filename
    ]
    if trim:
        command.insert(command.index('-i'), "-ss")
        command.insert(command.index('-i'), trim)

    if args.its_time:
        command.insert(command.index('-i'), "-to")
        command.insert(command.index('-i'), args.its_time)

    subprocess.run(command, check=True)
    input_filename = new_filename

# Refresh video info after possible processing
input_width, input_height, input_fps, input_bitrate, start_pts, time_base, duration, input_codec, start_time = get_video_info(input_filename)
source_duration_first_frame, frame_rate_mode = get_media_info(input_filename)

if device == "cuda":
    pynvml.nvmlInit()
def get_gpu_memory():
    # Returns free memory and total memory in MiB (binary)
    if device != "cuda":
        return 0, 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_mib = info.free / 1024**2
    total_mib = info.total / 1024**2
    return free_mib, total_mib

def start_input_pipe():
    """Starts FFmpeg to pipe raw frames at base_edge resolution directly to Python."""
    if device == "cuda":
        if 'hevc' in input_codec or '265' in input_codec:
            decodec = 'hevc_cuvid'
        elif '264' in input_codec:
            decodec = 'h264_cuvid'
        elif 'av1' in input_codec:
            decodec = 'av1_cuvid'
        else:
            decodec = None

        cmd = [
            ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-c:v", decodec, "-resize", f"{PROCESS_W}x{PROCESS_H}",
            "-i", input_filename,
            "-vf", "scale_cuda=format=nv12,hwdownload,format=nv12,format=rgb24",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
        ]
    else:
        cmd = [
            ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
            "-i", input_filename,
            "-vf", f"scale={PROCESS_W}:{PROCESS_H}:flags=lanczos,format=rgb24",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
        ]

    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=PROCESS_W * PROCESS_H * 3 * 10)

def process_frame_to_bytes(frame_bytes):
    """Processes raw RGB bytes and returns raw BGRA bytes at 4K resolution."""
    if not frame_bytes:
        return None

    # Load raw bytes directly to Torch via Numpy
    arr = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((PROCESS_H, PROCESS_W, 3))
    if _PREPROC_ENABLED:
        arr = _apply_input_preprocess(arr)
    # Non-blocking transfer to GPU
    input_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True).to(target_dtype).contiguous()

    # Fast fused normalization
    input_tensor = (input_tensor * 0.00392156862745098) - NORM_MEAN
    input_tensor /= NORM_STD

    target_h, target_w = input_height // 2, input_width // 2

    with torch.inference_mode():
        preds = _infer_preds(input_tensor)
        # GPU Resize to output size
        preds_resized = torch.nn.functional.interpolate(preds, size=(target_h, target_w), mode='bilinear', align_corners=False)

        if args.soft_mask:
            # Soft background mask from background probability.
            mask = ((1.0 - preds_resized[0, 0]).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        else:
            # Binary mask path.
            mask = (preds_resized[0, 0] <= args.mask_threshold).to(torch.uint8) * 255

        if _POSTPROC_ENABLED:
            mask_cpu = _postprocess_mask(mask.cpu().numpy(), hard_binary=(not args.soft_mask))
            bgra_cpu = np.zeros((target_h, target_w, 4), dtype=np.uint8)
            bgra_cpu[:, :, 1] = mask_cpu
            bgra_cpu[:, :, 3] = mask_cpu
            final_bytes = bgra_cpu.tobytes()
        else:
            # Create a 4-channel BGRA tensor on GPU
            bgra_gpu = torch.zeros((target_h, target_w, 4), dtype=torch.uint8, device=device)
            bgra_gpu[:, :, 1] = mask
            bgra_gpu[:, :, 3] = mask

            # Final download to CPU
            final_bytes = bgra_gpu.cpu().numpy().tobytes()

    return final_bytes

def _numpy_to_gpu_normalized(batch_frames_bytes):
    """CPU->GPU transfer + normalization, shared by graph and eager paths."""
    batch_size = len(batch_frames_bytes)
    batch_np = np.empty((batch_size, PROCESS_H, PROCESS_W, 3), dtype=np.uint8)
    for i, frame_bytes in enumerate(batch_frames_bytes):
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((PROCESS_H, PROCESS_W, 3))
        if _PREPROC_ENABLED:
            frame = _apply_input_preprocess(frame)
        batch_np[i] = frame
    # [N, H, W, C] -> [N, C, H, W] contiguous for GPU; single .to() fuses transfer+cast
    batch_tensor = torch.from_numpy(batch_np.transpose(0, 3, 1, 2).copy()).to(device=device, dtype=target_dtype, non_blocking=True)
    # In-place normalization — avoids 3 temporary tensor allocations
    batch_tensor.mul_(0.00392156862745098).sub_(NORM_MEAN).div_(NORM_STD)
    return batch_tensor

def _preds_to_bgra(preds, batch_size):
    """Convert raw model output [N,1,H,W] -> list of BGRA bytes."""
    target_h, target_w = input_height // 2, input_width // 2
    preds_resized = torch.nn.functional.interpolate(
        preds, size=(target_h, target_w), mode='bilinear', align_corners=False
    )

    if args.soft_mask:
        masks = ((1.0 - preds_resized[:, 0]).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    else:
        masks = (preds_resized[:, 0] <= args.mask_threshold).to(torch.uint8) * 255

    if _POSTPROC_ENABLED:
        masks_cpu = masks.cpu().numpy()
        out = []
        for i in range(batch_size):
            m = _postprocess_mask(masks_cpu[i], hard_binary=(not args.soft_mask))
            bgra = np.zeros((target_h, target_w, 4), dtype=np.uint8)
            bgra[:, :, 1] = m
            bgra[:, :, 3] = m
            out.append(bgra.tobytes())
        return out

    bgra_batch = torch.zeros((batch_size, target_h, target_w, 4), dtype=torch.uint8, device=device)
    bgra_batch[:, :, :, 1] = masks
    bgra_batch[:, :, :, 3] = masks

    bgra_cpu = bgra_batch.cpu().numpy()
    return [bgra_cpu[i].tobytes() for i in range(batch_size)]

def process_batch_to_bytes(batch_frames_bytes):
    """Processes a batch of raw RGB bytes and returns (list of BGRA bytes, infer_ms)."""
    if not batch_frames_bytes:
        return [], 0.0

    t0 = time.time()
    batch_size = len(batch_frames_bytes)

    try:
        batch_tensor = _numpy_to_gpu_normalized(batch_frames_bytes)

        with torch.inference_mode():
            preds = _infer_preds(batch_tensor)
            results = _preds_to_bgra(preds, batch_size)

        infer_ms = (time.time() - t0) * 1000
        return results, infer_ms

    except Exception as e:
        print(f"Batch processing error: {e}, falling back to single-frame mode")
        results = []
        for frame_bytes in batch_frames_bytes:
            try:
                results.append(process_frame_to_bytes(frame_bytes))
            except:
                results.append(None)
        return results, (time.time() - t0) * 1000

async def process_and_pipe(executor):
    """Process frames in batches and pipe into FFmpeg in sequence."""
    final_name = os.path.join("output", f"{os.path.splitext(os.path.basename(input_source))[0]}{video_extension}")
    target_h, target_w = input_height // 2, input_width // 2

    # Batch size from args
    batch_size = args.batch_size

    # Auto-optimization state
    gpu_limit = args.gpu_limit  # Now means concurrent batches

    # CUDA Graph serializes replay — no point in >1 concurrent batch
    if cuda_graph is not None:
        gpu_limit = 1
        window_batches = 3  # keep prefetch ahead but execute 1 at a time

    # --- Two-phase autopilot state machine ---
    # Phases: batch_ramp -> gpu_baseline -> gpu_ramp -> stable
    MAX_BATCH_SIZE = 32
    SETTLE_SAMPLES = 10
    MIN_FREE_VRAM_MIB = 2500

    autopilot_phase = "batch_ramp"          # current phase
    best_ms_per_frame = float('inf')        # best (lowest) ms/frame seen
    best_batch_size = batch_size            # batch_size that produced best_ms_per_frame
    baseline_fps = 0.0                      # FPS recorded at gpu_limit=1 (gpu_baseline phase)
    autopilot_last_check_frame = 0          # frame counter for 30-frame cadence
    autopilot_phase_entry_time = time.time()   # wall-clock at phase entry (for baseline/ramp FPS)
    autopilot_phase_entry_frame = 0            # next_frame_to_write at phase entry

    if args.auto_workers:
        print(f"Auto-pilot: batch_size={batch_size}, {gpu_limit} concurrent batches {'(CUDA Graph)' if cuda_graph else ''} | phases: batch_ramp -> gpu_baseline -> gpu_ramp -> stable")
    else:
        print(f"Fixed: batch_size={batch_size}, {gpu_limit} concurrent batches {'(CUDA Graph)' if cuda_graph else ''}")

    active_gpu_tasks = 0
    gpu_condition = asyncio.Condition()
    loop = asyncio.get_event_loop()

    # Prefetch queue for reading frames ahead
    prefetch_queue = asyncio.Queue(maxsize=gpu_limit + 2)
    read_complete = asyncio.Event()

    async def throttled_batch_process(batch_frames):
        nonlocal active_gpu_tasks
        async with gpu_condition:
            while active_gpu_tasks >= gpu_limit:
                await gpu_condition.wait()
            active_gpu_tasks += 1

        try:
            return await loop.run_in_executor(executor, process_batch_to_bytes, batch_frames)
        finally:
            async with gpu_condition:
                active_gpu_tasks -= 1
                gpu_condition.notify_all()

    if device == "cuda":
        command = [
            ffmpeg_path, '-y',
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-i", input_filename,
            "-f", "rawvideo", "-pixel_format", "bgra", "-video_size", f"{target_w}x{target_h}",
            "-framerate", input_fps, "-i", "-",
            "-i", audio_filename,
            "-filter_complex",
            "[0:v]scale_cuda=format=yuv420p[bg];"
            f"[1:v]scale={input_width}:{input_height}:flags=bilinear,format=yuva420p,hwupload_cuda[alpha];"
            "[bg][alpha]overlay_cuda,hwdownload[out];",
            "-map", "[out]", "-map", "2:0", "-c:2", "copy",
            "-c:v", "hevc_nvenc", "-b:v", f"{input_bitrate}M",
            "-time_base", time_base, "-r", input_fps,
            final_name
        ]
    else:
        # CPU Assembly Command
        command = [
            ffmpeg_path, '-y',
            "-i", input_filename,
            "-f", "rawvideo", "-pixel_format", "bgra", "-video_size", f"{target_w}x{target_h}",
            "-framerate", input_fps, "-i", "-",
            "-i", audio_filename,
            "-filter_complex",
            "[0:v]format=yuv420p[bg];"
            f"[1:v]scale={input_width}:{input_height}:flags=bilinear,format=yuva420p[alpha];"
            "[bg][alpha]overlay=shortest=1[out];",
            "-map", "[out]", "-map", "2:0", "-c:2", "copy",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "18",
            "-time_base", time_base, "-r", input_fps,
            final_name
        ]

    ffmpeg_proc = subprocess.Popen(command, stdin=subprocess.PIPE)

    print(f"--- Starting Batched Pipe Processing (batch_size={batch_size}) ---")

    input_proc = start_input_pipe()
    frame_size = PROCESS_W * PROCESS_H * 3

    # Async frame reader - reads batches ahead while GPU processes
    async def frame_reader():
        """Read frames in background and put batches into prefetch queue."""
        frame_idx = 1
        while frame_idx <= total_frames:
            batch_frames = []
            batch_start = frame_idx

            for _ in range(batch_size):
                if frame_idx > total_frames:
                    break
                # Read in executor to not block event loop
                frame_bytes = await loop.run_in_executor(None, input_proc.stdout.read, frame_size)
                if not frame_bytes or len(frame_bytes) < frame_size:
                    break
                batch_frames.append(frame_bytes)
                frame_idx += 1

            if batch_frames:
                await prefetch_queue.put((batch_start, batch_frames))
            else:
                break

        read_complete.set()

    # Start reader task
    reader_task = asyncio.create_task(frame_reader())

    # Results buffer: stores processed frame bytes by frame index
    results_buffer = {}
    pending_batches = {}  # batch_start_idx -> task
    next_frame_to_write = 1
    if cuda_graph is None:
        window_batches = max(2, gpu_limit + 2)  # How many batches to keep in flight

    # Timing diagnostics
    _t_read = []      # time spent reading from input pipe
    _t_infer = []     # list of (infer_ms, batch_len) tuples — batch_len needed for ms/frame filtering
    _t_write = []     # time spent writing to output pipe
    _t_write_start = None

    while next_frame_to_write <= total_frames:
        # 1. Two-phase autopilot (every 30 frames written)
        if args.auto_workers and cuda_graph is None and autopilot_phase != "stable" and next_frame_to_write % 30 == 0 and next_frame_to_write > autopilot_last_check_frame:
            autopilot_last_check_frame = next_frame_to_write

            if autopilot_phase == "batch_ramp":
                # Filter recent samples that match the current batch_size
                matching = [(ms, bl) for ms, bl in _t_infer if bl == batch_size]
                if len(matching) >= SETTLE_SAMPLES:
                    # Use the most recent 15 matching samples for the average
                    recent_match = matching[-15:]
                    avg_ms_frame = sum(ms / bl for ms, bl in recent_match) / len(recent_match)

                    if avg_ms_frame < best_ms_per_frame * 0.97:
                        # 3% improvement — record and try doubling
                        best_ms_per_frame = avg_ms_frame
                        best_batch_size = batch_size
                        next_bs = batch_size * 2
                        free_mib, _ = get_gpu_memory()
                        if next_bs <= MAX_BATCH_SIZE and free_mib > MIN_FREE_VRAM_MIB:
                            batch_size = next_bs
                            print(f"[{next_frame_to_write}] batch_ramp: {best_batch_size} -> {batch_size} (best {best_ms_per_frame:.2f} ms/frame, VRAM free {free_mib:.0f} MiB)")
                        else:
                            # Can't go higher — lock and move on
                            print(f"[{next_frame_to_write}] batch_ramp -> gpu_baseline: locked batch_size={best_batch_size} (VRAM {free_mib:.0f} MiB or MAX_BATCH_SIZE reached)")
                            batch_size = best_batch_size
                            if device == "cuda":
                                torch.cuda.empty_cache()
                            autopilot_phase = "gpu_baseline"
                            autopilot_phase_entry_time = time.time()
                            autopilot_phase_entry_frame = next_frame_to_write
                            autopilot_last_check_frame = next_frame_to_write
                    else:
                        # No improvement — revert to best and move on
                        print(f"[{next_frame_to_write}] batch_ramp -> gpu_baseline: no gain at batch_size={batch_size}, reverting to {best_batch_size} ({best_ms_per_frame:.2f} ms/frame)")
                        batch_size = best_batch_size
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        autopilot_phase = "gpu_baseline"
                        autopilot_phase_entry_time = time.time()
                        autopilot_phase_entry_frame = next_frame_to_write
                        autopilot_last_check_frame = next_frame_to_write

            elif autopilot_phase == "gpu_baseline":
                # One 30-frame window at locked batch_size, gpu_limit=1
                # FPS measured via wall-clock from phase entry (not sum of inference times)
                gpu_limit = 1
                window_batches = max(2, gpu_limit + 2)
                elapsed = time.time() - autopilot_phase_entry_time
                frames_in_window = next_frame_to_write - autopilot_phase_entry_frame
                if elapsed > 0 and frames_in_window > 0:
                    baseline_fps = frames_in_window / elapsed
                print(f"[{next_frame_to_write}] gpu_baseline: baseline_fps={baseline_fps:.2f} at gpu_limit=1 | batch_size={batch_size} ({frames_in_window} frames in {elapsed:.2f}s)")
                autopilot_phase = "gpu_ramp"
                gpu_limit = 2
                window_batches = max(2, gpu_limit + 2)
                autopilot_phase_entry_time = time.time()
                autopilot_phase_entry_frame = next_frame_to_write
                autopilot_last_check_frame = next_frame_to_write

            elif autopilot_phase == "gpu_ramp":
                # One 30-frame window at gpu_limit=2 — compare FPS to baseline via wall-clock
                elapsed = time.time() - autopilot_phase_entry_time
                frames_in_window = next_frame_to_write - autopilot_phase_entry_frame
                if elapsed > 0 and frames_in_window > 0:
                    ramp_fps = frames_in_window / elapsed
                else:
                    ramp_fps = baseline_fps

                if ramp_fps > baseline_fps * 1.05:
                    # 5% gain — keep gpu_limit=2
                    print(f"[{next_frame_to_write}] gpu_ramp -> stable: kept gpu_limit=2 (ramp {ramp_fps:.2f} vs baseline {baseline_fps:.2f} FPS, {frames_in_window} frames in {elapsed:.2f}s)")
                    gpu_limit = 2
                else:
                    # No gain — revert to gpu_limit=1
                    print(f"[{next_frame_to_write}] gpu_ramp -> stable: reverted gpu_limit=1 (ramp {ramp_fps:.2f} vs baseline {baseline_fps:.2f} FPS, {frames_in_window} frames in {elapsed:.2f}s)")
                    gpu_limit = 1
                window_batches = max(2, gpu_limit + 2)
                autopilot_phase = "stable"
                print(f"[{next_frame_to_write}] Autopilot locked: batch_size={batch_size}, gpu_limit={gpu_limit}")

        # 2. Submit new batches from prefetch queue
        while len(pending_batches) < window_batches:
            try:
                batch_start, batch_frames = prefetch_queue.get_nowait()
                task = asyncio.create_task(throttled_batch_process(batch_frames))
                pending_batches[batch_start] = (task, len(batch_frames))
            except asyncio.QueueEmpty:
                if read_complete.is_set() and prefetch_queue.empty():
                    break
                # Wait a tiny bit for more data
                await asyncio.sleep(0.001)
                break

        # 3. Check for completed batches and store results
        completed_starts = []
        for batch_start, (task, batch_len) in pending_batches.items():
            if task.done():
                completed_starts.append(batch_start)
                try:
                    batch_results, infer_ms = task.result()
                    _t_infer.append((infer_ms, len(batch_results)))
                    for j, result_bytes in enumerate(batch_results):
                        results_buffer[batch_start + j] = result_bytes
                except Exception as e:
                    print(f"Batch {batch_start} failed: {e}")
                    for j in range(batch_len):
                        results_buffer[batch_start + j] = None

        for bs in completed_starts:
            del pending_batches[bs]

        # 4. Write frames in order from buffer
        frames_written_this_loop = 0
        while next_frame_to_write in results_buffer:
            rgba_bytes = results_buffer.pop(next_frame_to_write)
            if rgba_bytes:
                _tw0 = time.time()
                ffmpeg_proc.stdin.write(rgba_bytes)
                _t_write.append((time.time() - _tw0) * 1000)
                frames_written_this_loop += 1
            next_frame_to_write += 1

        # Flush periodically
        if frames_written_this_loop > 0 and next_frame_to_write % 30 < batch_size:
            ffmpeg_proc.stdin.flush()

        # 5. Progress reporting
        if next_frame_to_write % 50 == 1 and frames_written_this_loop > 0:
            if device == "cuda":
                free_mib, total_mib = get_gpu_memory()
                pct = (free_mib / total_mib) * 100 if total_mib > 0 else 0
                # Timing report every 200 frames
                timing_str = ""
                if len(_t_infer) >= 2 and next_frame_to_write % 200 < 50:
                    recent = [(ms, bl) for ms, bl in _t_infer if bl == batch_size][-20:]
                    avg_ms_per_frame = sum(ms / bl for ms, bl in recent) / len(recent) if recent else 0
                    avg_write = sum(_t_write[-50:]) / min(50, len(_t_write[-50:])) if _t_write else 0
                    timing_str = f" | batch={batch_size} Infer: {avg_ms_per_frame:.1f}ms/frame Write: {avg_write:.2f}ms/frame"
                print(f"Piped {next_frame_to_write-1}/{total_frames} frames. (Batches: {len(pending_batches)} in flight / Free VRAM: {free_mib:.0f} MiB - {pct:.1f}%{timing_str})")
            else:
                print(f"Piped {next_frame_to_write-1}/{total_frames} frames. (Batches: {len(pending_batches)} in flight)")

        # 6. If nothing to write yet, wait for a batch to complete
        if frames_written_this_loop == 0 and pending_batches:
            done, _ = await asyncio.wait(
                [task for task, _ in pending_batches.values()],
                return_when=asyncio.FIRST_COMPLETED
            )
        elif frames_written_this_loop == 0 and not pending_batches:
            if read_complete.is_set() and prefetch_queue.empty():
                break
            await asyncio.sleep(0.001)

    # Cleanup
    await reader_task

    ffmpeg_proc.stdin.close()
    input_proc.terminate()
    ffmpeg_proc.wait()
    return final_name

async def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_max_workers) as executor:
        # Start piping and assembly simultaneously
        final_video_path = await process_and_pipe(executor)
        return final_video_path

if __name__ == "__main__":
    s_time = time.time()
    final_video_path = asyncio.run(main())
    final_time = time.time() - s_time
    
    sec_per_min = (final_time / total_frames * 3600) if total_frames else 0.0
    print(f"Elapsed time: {round(final_time, 2)} s. ~ {round(sec_per_min, 2)} s. per minute.")

    if args.soft_mask:
        mask_tag = "soft"
    else:
        mask_tag = f"bin_thr{args.mask_threshold:.2f}"
    if args.run_tag:
        safe_tag = re.sub(r'[^A-Za-z0-9_-]+', '-', args.run_tag).strip('-')
        if safe_tag:
            mask_tag = f"{mask_tag}_{safe_tag}"
    mask_tag = mask_tag.replace('.', 'p')
    edge_tag = f"be{ENGINE_W}x{ENGINE_H}"
    spm_tag = f"spm{sec_per_min:.2f}s".replace('.', 'p')
    ext = os.path.splitext(final_video_path)[1]
    src_stem = os.path.splitext(os.path.basename(input_source))[0]
    renamed_video_path = os.path.join(
        "output",
        f"{src_stem}_{MODEL_NAME}_{edge_tag}_{mask_tag}_{spm_tag}{ext}",
    )
    if os.path.exists(final_video_path):
        os.replace(final_video_path, renamed_video_path)
        final_video_path = renamed_video_path
    print(f"Output file: {final_video_path}")
    
    for p in sys_folders:
        if os.path.exists(p):
            shutil.rmtree(p)
    
    if device == "cuda":
        pynvml.nvmlShutdown()
