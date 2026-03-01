# BiRefNet TRT Quality Incident - Playbook

## Context
- Project: `G:\RMBG`
- Model family: `BiRefNet_dynamic`
- Target: keep TRT quality close to Torch/ONNX and preserve speed.

## Symptoms
- `Torch` and `ONNX` outputs looked almost identical.
- `TensorRT` output looked visibly wrong (different mask structure/quality).
- Simple threshold/mask-mode tweaks did not fix the core mismatch.

## Root Cause
The problem was not the model itself, but precision path during conversion:

1. ONNX exported in a way that led TRT FP16 path to unstable numerics for normalization-heavy parts.
2. TRT logs warned about LayerNorm/FP16 overflow risk.
3. Old TRT engine output was numerically far from ONNX (`corr ~ 0.67` on diagnostic frame).

So the issue was effectively:
- bad precision route (`fp16 ONNX -> fp16 TRT`) for this graph/stack.

## What Worked
Best stable path was:

1. Export ONNX in **FP32**.
2. Use **opset 17** ONNX (so normalization nodes are represented better for TRT).
3. Build TRT in **mixed precision**:
- global FP16 enabled for performance,
- critical normalization/reduction layers forced to FP32.
4. Ensure runtime code uses actual engine tensor dtypes (not hardcoded fp16).

Result:
- ONNX vs TRT correlation became near-perfect (`corr ~ 0.999998`).
- Quality returned.
- Speed improved vs full-FP32 TRT.

## Final Working Artifacts
- Export script: `export_birefnet_dynamic_opset17.py`
- Mixed TRT builder: `convert_birefnet_dynamic_1024_trt_mixed.py`
- Updated TRT runtime with dtype autodetect: `birefnet_trt/birefnet_trt.py`
- Working engine: `models/birefnet_dynamic_1024x1024_b1_opset17_mixed.engine`
- Launcher: `birefnet_dynamic_trt_1024_b1_mixed_opset17.bat`

## Repro Commands

### 1) Export ONNX FP32 opset17
```powershell
.\python_embeded\python.exe .\export_birefnet_dynamic_opset17.py --width 1024 --height 1024 --batch 1 --precision fp32 --opset 17
```

### 2) Build mixed TRT from that ONNX
```powershell
.\python_embeded\python.exe .\convert_birefnet_dynamic_1024_trt_mixed.py --onnx "models\birefnet_dynamic_1024x1024_b1_opset17_fp32.onnx" --output "models\birefnet_dynamic_1024x1024_b1_opset17_mixed.engine" --workspace-gb 6
```

### 3) Run video pipeline with mixed engine
```powershell
.\birefnet_dynamic_trt_1024_b1_mixed_opset17.bat
```

## Diagnostic Rule (Do This Every Time)
Before trusting a new TRT engine:

1. Compare ONNX vs TRT numerically on the same frame.
2. Test candidate mappings:
- `raw`
- `sigmoid(raw)`
- `1-raw`
- `1-sigmoid(raw)`
3. If best correlation is far from 1.0 (for this task), engine path is wrong.

## Recommended Conversion Policy for Similar Models
For segmentation/matting-like models with normalization-heavy blocks:

1. Keep Torch inference as needed (fp16/mixed is fine).
2. Export ONNX as FP32.
3. Prefer opset 17 where possible.
4. Build TRT with mixed precision constraints:
- performance layers in FP16,
- sensitive normalization/reduction logic in FP32.
5. Validate ONNX vs TRT numerically before deployment.

## Common Pitfalls
- Assuming `fp16 in Torch` equals `fp16 in TRT` behavior.
- Hardcoding runtime dtype to fp16 when engine I/O is fp32.
- Judging only by speed without numerical parity checks.
- Trying to fix severe engine mismatch only via threshold/postprocess tweaks.

## Quick Decision Guide
- Quality priority: Torch/ONNX or TRT mixed with parity check.
- Speed + quality: TRT mixed from FP32 ONNX (opset17) is the first attempt.
- If mixed still fails: fallback to FP32 TRT or Torch, then iterate conversion settings.
