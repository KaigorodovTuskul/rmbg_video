import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Build BiRefNet dynamic TRT engine with mixed precision safeguards")
    parser.add_argument("--onnx", default=os.path.join("models", "birefnet_dynamic_1024_b1.onnx"))
    parser.add_argument("--output", default=os.path.join("models", "birefnet_dynamic_1024_b1_mixed.engine"))
    parser.add_argument("--workspace-gb", type=int, default=6)
    args = parser.parse_args()

    try:
        import tensorrt as trt
    except ImportError:
        print("ERROR: tensorrt is not installed")
        sys.exit(1)

    if not os.path.exists(args.onnx):
        print(f"ERROR: ONNX not found: {args.onnx}")
        sys.exit(1)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser_onnx = trt.OnnxParser(network, logger)

    print(f"TensorRT {getattr(trt, '__version__', 'unknown')}")
    print(f"Parsing ONNX: {args.onnx}")
    if not parser_onnx.parse_from_file(args.onnx):
        for i in range(parser_onnx.num_errors):
            print(f"  Parse error [{i}]: {parser_onnx.get_error(i).desc()}")
        sys.exit(1)

    inp = network.get_input(0)
    print(f"Input: {inp.name} shape={inp.shape} dtype={inp.dtype}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb) * (1 << 30))
    config.set_flag(trt.BuilderFlag.FP16)

    # Ask TRT to honor per-layer precision constraints.
    if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    elif hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    forced = []
    reduced = 0
    pows = 0
    norms = 0

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        ltype = layer.type
        force_fp32 = False

        if ltype == trt.LayerType.REDUCE:
            force_fp32 = True
            reduced += 1
        elif ltype == trt.LayerType.ELEMENTWISE:
            op = getattr(layer, "op", None)
            if op is not None and hasattr(trt, "ElementWiseOperation") and op == trt.ElementWiseOperation.POW:
                force_fp32 = True
                pows += 1
        elif hasattr(trt.LayerType, "NORMALIZATION") and ltype == trt.LayerType.NORMALIZATION:
            force_fp32 = True
            norms += 1

        # Fallback by layer name for decomposed LayerNorm graphs.
        lname = (layer.name or "").lower()
        if ("layernorm" in lname or "layer_norm" in lname) and not force_fp32:
            force_fp32 = True
            norms += 1

        if force_fp32:
            try:
                layer.precision = trt.float32
            except Exception:
                pass
            try:
                for out_idx in range(layer.num_outputs):
                    layer.set_output_type(out_idx, trt.float32)
            except Exception:
                pass
            forced.append((i, str(ltype), layer.name))

    print(f"Forced FP32 layers: {len(forced)} (Reduce={reduced}, Pow={pows}, Norm={norms})")
    if forced:
        preview = forced[:20]
        for idx, ltype, lname in preview:
            print(f"  [{idx}] {ltype} {lname}")
        if len(forced) > len(preview):
            print(f"  ... and {len(forced)-len(preview)} more")

    print("Building mixed-precision engine...")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print("ERROR: engine build failed")
        sys.exit(1)

    with open(args.output, "wb") as f:
        f.write(bytes(plan))
    print(f"Done > {args.output} ({os.path.getsize(args.output)/1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
