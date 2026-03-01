import argparse
import os
import sys

sys.path.insert(0, r"G:\RMBG")
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from deform_conv2d_onnx_exporter import register_deform_conv2d_onnx_op, set_export_batch_size

register_deform_conv2d_onnx_op(use_gathernd=False, enable_openvino_patch=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    models_dir = r"G:\RMBG\models"
    model_dir = os.path.join(models_dir, "BiRefNet_dynamic")
    sys.path.insert(0, models_dir)
    sys.path.insert(0, model_dir)

    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    if args.precision == "fp16":
        model.half()
    else:
        model.float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    dummy = torch.randn(args.batch, 3, args.height, args.width, dtype=dtype, device=device)

    out_path = os.path.join(models_dir, f"birefnet_dynamic_{args.width}x{args.height}_b{args.batch}_opset{args.opset}_{args.precision}.onnx")
    set_export_batch_size(args.batch)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            out_path,
            opset_version=args.opset,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )

    print(f"Done: {out_path} ({os.path.getsize(out_path)/1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
