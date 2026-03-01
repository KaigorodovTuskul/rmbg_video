"""ONNX symbolic for torchvision::deform_conv2d (decomposed via standard ONNX ops).

Source: https://github.com/lbq779660843/BiRefNet-Tensorrt
License: MIT
"""

import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx import symbolic_helper as sym_help

try:
    from torch.onnx._type_utils import JitScalarType
except ImportError:
    JitScalarType = None

__all__ = ["register_deform_conv2d_onnx_op", "set_export_batch_size"]

onnx_opset_version = 12

# PyTorch 2.9+ marks the batch dim as symbolic in varyingSizes() even during
# concrete tracing, so _get_tensor_dim_size returns None for dim 0.
# Callers must set this before torch.onnx.export when batch > 1.
_EXPORT_BATCH_SIZE: int | None = None

def set_export_batch_size(n: int):
    global _EXPORT_BATCH_SIZE
    _EXPORT_BATCH_SIZE = n


# ── tiny graph helpers ────────────────────────────────────────────────────────

def add(g, lhs, rhs):
    return g.op("Add", lhs, rhs)

def sub(g, lhs, rhs):
    return g.op("Sub", lhs, rhs)

def mul(g, lhs, rhs):
    return g.op("Mul", lhs, rhs)

def reshape(g, x, shape):
    if isinstance(shape, list):
        shape = tensor(g, shape, dtype=torch.int64)
    return g.op("Reshape", x, shape)

def slice_(g, x, axes, starts, ends, *, steps=None):
    axes   = tensor(g, axes,   dtype=torch.int64)
    starts = tensor(g, starts, dtype=torch.int64)
    ends   = tensor(g, ends,   dtype=torch.int64)
    if steps is not None:
        steps = tensor(g, steps, dtype=torch.int64)
        return g.op("Slice", x, starts, ends, axes, steps)
    return g.op("Slice", x, starts, ends, axes)

def unsqueeze(g, inp, dims):
    return sym_help._unsqueeze_helper(g, inp, axes_i=dims)

def tensor(g, value, dtype):
    return g.op("Constant", value_t=torch.tensor(value, dtype=dtype))


def get_tensor_dim_size(t, dim):
    size = sym_help._get_tensor_dim_size(t, dim)
    if size is None:
        import typing
        from torch import _C
        x_type = typing.cast(_C.TensorType, t.type())
        # varyingSizes() returns None for symbolic dims (e.g. batch in PyTorch 2.9+)
        vsizes = x_type.varyingSizes()
        if vsizes is not None and dim < len(vsizes) and vsizes[dim] is not None:
            size = vsizes[dim]
        else:
            # strides fallback for spatial dims (works for dims 2,3 of 4-D tensors)
            strides = x_type.strides()
            if strides is not None:
                if dim == 2:
                    size = strides[1] // strides[2] if strides[2] != 0 else None
                elif dim == 3:
                    size = strides[2]
    # dim 0 (batch) is often symbolic — use the value set by set_export_batch_size()
    if size is None and dim == 0 and _EXPORT_BATCH_SIZE is not None:
        size = _EXPORT_BATCH_SIZE
    return size


# ── core math (equation 1 from the DCNv2 paper) ─────────────────────────────

def calculate_p_0(dcn_params):
    h, w       = dcn_params["out_h"],  dcn_params["out_w"]
    sh, sw     = dcn_params["stride_h"], dcn_params["stride_w"]
    K          = dcn_params["kernel_area_size"]
    aph, apw   = dcn_params["additional_pad_h"], dcn_params["additional_pad_w"]

    p0y, p0x = torch.meshgrid(
        torch.arange(0, h * sh, sh),
        torch.arange(0, w * sw, sw),
        indexing="ij",
    )
    p0y = p0y.view(1,1,1,1,h,w).repeat(1,1,K,1,1,1) + aph
    p0x = p0x.view(1,1,1,1,h,w).repeat(1,1,K,1,1,1) + apw
    return torch.cat([p0y, p0x], dim=3)

def calculate_p_k(dcn_params):
    kh, kw     = dcn_params["kernel_h"],   dcn_params["kernel_w"]
    dh, dw     = dcn_params["dilation_h"], dcn_params["dilation_w"]
    K          = dcn_params["kernel_area_size"]

    pky, pkx = torch.meshgrid(
        torch.arange(0, kh * dh, step=dh),
        torch.arange(0, kw * dw, step=dw),
        indexing="ij",
    )
    return torch.cat([
        pky.reshape(1,1,K,1,1,1),
        pkx.reshape(1,1,K,1,1,1),
    ], dim=3)


def calculate_p(g, dcn_params, offset):
    b, K   = dcn_params["batch"], dcn_params["kernel_area_size"]
    h, w   = dcn_params["out_h"],  dcn_params["out_w"]
    group  = dcn_params["n_offset_grps"]
    odtype = dcn_params["offset_dtype_pytorch"]

    offset = reshape(g, offset, [b, group, K, 2, h, w])
    p      = calculate_p_0(dcn_params) + calculate_p_k(dcn_params)
    return add(g, tensor(g, p.tolist(), dtype=odtype), offset)


def calculate_p_floor(g, _dcn_params, p):
    return g.op("Floor", p)


def calculate_p_tlbr(g, dcn_params, p_floor):
    ih, iw   = dcn_params["in_h"], dcn_params["in_w"]
    idtype_o = dcn_params["index_dtype_onnx"]
    idtype_p = dcn_params["index_dtype_pytorch"]

    p_floor = g.op("Cast", p_floor, to_i=idtype_o)
    one     = tensor(g, 1, dtype=idtype_p)

    p_t = slice_(g, p_floor, [3], [0], [1])
    p_l = slice_(g, p_floor, [3], [1], [2])
    p_b = add(g, p_t, one)
    p_r = add(g, p_l, one)

    z   = tensor(g, 0,    dtype=idtype_p)
    h_1 = tensor(g, ih-1, dtype=idtype_p)
    w_1 = tensor(g, iw-1, dtype=idtype_p)

    p_t = g.op("Clip", p_t, z, h_1)
    p_l = g.op("Clip", p_l, z, w_1)
    p_b = g.op("Clip", p_b, z, h_1)
    p_r = g.op("Clip", p_r, z, w_1)
    return {"t": p_t, "l": p_l, "b": p_b, "r": p_r}


def calculate_weight(g, dcn_params, p, p_floor):
    b, group = dcn_params["batch"], dcn_params["n_offset_grps"]
    h, w, K  = dcn_params["out_h"],  dcn_params["out_w"], dcn_params["kernel_area_size"]
    odtype   = dcn_params["offset_dtype_pytorch"]
    one      = tensor(g, 1, dtype=odtype)

    diff   = sub(g, p, p_floor)
    dy     = slice_(g, diff, [3], [0], [1])
    dx     = slice_(g, diff, [3], [1], [2])
    dy_inv = sub(g, one, dy)
    dx_inv = sub(g, one, dx)

    weights = {
        "tl": mul(g, dx_inv, dy_inv),
        "br": mul(g, dx,     dy),
        "bl": mul(g, dx_inv, dy),
        "tr": mul(g, dx,     dy_inv),
    }
    return {k: reshape(g, v, [b, group, 1, K, h, w]) for k, v in weights.items()}


# ── gather ────────────────────────────────────────────────────────────────────

def reshape_input_for_gather(g, dcn_params, inp):
    b      = dcn_params["batch"]
    group  = dcn_params["n_offset_grps"]
    ch     = dcn_params["in_ch_per_group"]
    ih, iw = dcn_params["in_h"], dcn_params["in_w"]
    ph, pw = dcn_params["padding_h"], dcn_params["padding_w"]
    aph, apw = dcn_params["additional_pad_h"], dcn_params["additional_pad_w"]

    pad_size = [0, 0,
                (ph + aph), (pw + apw),
                0, 0,
                (ph + aph), (pw + apw)]
    inp = g.op("Pad", inp, tensor(g, pad_size, dtype=torch.int64), mode_s="constant")
    return reshape(g, inp, [b, group, ch, ih, iw])


def gather_elements(g, dcn_params, inp, p_y, p_x):
    b, group = dcn_params["batch"], dcn_params["n_offset_grps"]
    ch       = dcn_params["in_ch_per_group"]
    ih, iw   = dcn_params["in_h"], dcn_params["in_w"]
    oh, ow   = dcn_params["out_h"], dcn_params["out_w"]
    K        = dcn_params["kernel_area_size"]
    idtype_p = dcn_params["index_dtype_pytorch"]

    p_y  = reshape(g, p_y, [b, group, 1, K * oh * ow])
    p_x  = reshape(g, p_x, [b, group, 1, K * oh * ow])
    idx  = add(g, g.op("Mul", p_y, tensor(g, iw, dtype=idtype_p)), p_x)
    idx  = g.op("Expand", idx, tensor(g, [b, group, ch, K * oh * ow], dtype=torch.int64))
    inp  = reshape(g, inp, [b, group, ch, ih * iw])
    v    = g.op("GatherElements", inp, idx, axis_i=3)
    return reshape(g, v, [b, group, ch, K, oh, ow])


def gather_elements_tlbr(g, dcn_params, inp, p_tlbr):
    keys = ["tl", "br", "bl", "tr"]
    return {k: gather_elements(g, dcn_params, inp,
                               p_tlbr[k[0]], p_tlbr[k[1]]) for k in keys}


def calculate_weighted_sum(g, _dcn_params, v_tlbr, w_tlbr):
    return g.op("Sum", *[mul(g, w_tlbr[k], v_tlbr[k]) for k in v_tlbr])


def apply_mask(g, dcn_params, v, mask):
    b, group = dcn_params["batch"], dcn_params["n_offset_grps"]
    oh, ow, K = dcn_params["out_h"], dcn_params["out_w"], dcn_params["kernel_area_size"]
    return mul(g, v, reshape(g, mask, [b, group, 1, K, oh, ow]))


def reshape_v_for_conv(g, dcn_params, v):
    b      = dcn_params["batch"]
    oh, ow = dcn_params["out_h"],  dcn_params["out_w"]
    ch     = dcn_params["in_ch"]
    kh, kw = dcn_params["kernel_h"], dcn_params["kernel_w"]

    v = reshape(g, v, [b, ch, kh, kw, oh, ow])
    v = g.op("Transpose", v, perm_i=[0, 1, 4, 2, 5, 3])
    return reshape(g, v, [b, ch, oh * kh, ow * kw])


def apply_conv(g, dcn_params, v, weight):
    kh, kw = dcn_params["kernel_h"], dcn_params["kernel_w"]
    return g.op("Conv", v, weight,
                group_i=dcn_params["n_weight_grps"],
                kernel_shape_i=[kh, kw],
                strides_i=[kh, kw])


def apply_bias(g, _dcn_params, v, bias):
    return add(g, v, unsqueeze(g, bias, [0, 2, 3]))


# ── param builder ─────────────────────────────────────────────────────────────

def create_dcn_params(inp, weight, offset, mask, bias,
                      stride_h, stride_w, pad_h, pad_w,
                      dilation_h, dilation_w,
                      n_weight_grps, n_offset_grps, use_mask, option):
    aph = 1 if pad_h == 0 else 0
    apw = 1 if pad_w == 0 else 0

    batch        = get_tensor_dim_size(inp,    0)
    in_ch        = get_tensor_dim_size(inp,    1)
    in_h_raw     = get_tensor_dim_size(inp,    2)
    in_w_raw     = get_tensor_dim_size(inp,    3)
    in_h         = in_h_raw + 2 * (pad_h + aph)
    in_w         = in_w_raw + 2 * (pad_w + apw)
    in_ch_per_group = in_ch // n_offset_grps

    kernel_h     = get_tensor_dim_size(weight, 2)
    kernel_w     = get_tensor_dim_size(weight, 3)
    out_h        = get_tensor_dim_size(offset, 2)
    out_w        = get_tensor_dim_size(offset, 3)

    # dtype helpers — works for PyTorch 2.x (JitScalarType is an enum with .onnx_type()/.dtype())
    od = sym_help._try_get_scalar_type(offset)
    if hasattr(od, "onnx_type"):
        # PyTorch 2.x: od is already a JitScalarType instance
        od_onnx = od.onnx_type()
        od_pt   = od.dtype()
    else:
        # Fallback (pre-2.x string API)
        od_onnx = sym_help.cast_pytorch_to_onnx[od]
        od_pt   = sym_help.scalar_type_to_pytorch_type[
            sym_help.scalar_type_to_onnx.index(od_onnx)]

    if JitScalarType is not None and hasattr(JitScalarType, "from_dtype"):
        sc_i   = JitScalarType.from_dtype(torch.int64)
        id_onnx = sc_i.onnx_type()
        id_pt   = sc_i.dtype()
    else:
        id_onnx = sym_help.cast_pytorch_to_onnx["Long"]
        id_pt   = sym_help.scalar_type_to_pytorch_type[
            sym_help.scalar_type_to_onnx.index(id_onnx)]

    return {
        "batch": batch, "kernel_h": kernel_h, "kernel_w": kernel_w,
        "kernel_area_size": kernel_h * kernel_w,
        "in_ch": in_ch, "in_ch_per_group": in_ch_per_group,
        "in_h": in_h, "in_w": in_w,
        "out_ch": get_tensor_dim_size(weight, 0),
        "out_h": out_h, "out_w": out_w,
        "stride_h": stride_h, "stride_w": stride_w,
        "dilation_h": dilation_h, "dilation_w": dilation_w,
        "n_offset_grps": n_offset_grps, "n_weight_grps": n_weight_grps,
        "offset_dtype_onnx": od_onnx, "offset_dtype_pytorch": od_pt,
        "index_dtype_onnx": id_onnx, "index_dtype_pytorch": id_pt,
        "padding_h": pad_h, "padding_w": pad_w,
        "additional_pad_h": aph, "additional_pad_w": apw,
        "option": option,
    }


# ── symbolic registration ─────────────────────────────────────────────────────

def deform_conv2d_func(use_gathernd, enable_openvino_patch):
    @sym_help.parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i",
                         "i", "i", "b")
    def deform_conv2d(g, inp, weight, offset, mask, bias,
                      stride_h, stride_w, pad_h, pad_w,
                      dilation_h, dilation_w, n_weight_grps, n_offset_grps, use_mask):
        option = {"use_gathernd": use_gathernd,
                  "enable_openvino_patch": enable_openvino_patch}
        dcn_params = create_dcn_params(
            inp, weight, offset, mask, bias,
            stride_h, stride_w, pad_h, pad_w,
            dilation_h, dilation_w, n_weight_grps, n_offset_grps, use_mask, option)

        p         = calculate_p(g, dcn_params, offset)
        p_floor   = calculate_p_floor(g, dcn_params, p)
        p_tlbr    = calculate_p_tlbr(g, dcn_params, p_floor)
        w_tlbr    = calculate_weight(g, dcn_params, p, p_floor)

        inp       = reshape_input_for_gather(g, dcn_params, inp)
        v_tlbr    = gather_elements_tlbr(g, dcn_params, inp, p_tlbr)
        v         = calculate_weighted_sum(g, dcn_params, v_tlbr, w_tlbr)

        if use_mask:
            v     = apply_mask(g, dcn_params, v, mask)

        v = reshape_v_for_conv(g, dcn_params, v)
        v = apply_conv(g, dcn_params, v, weight)
        v = apply_bias(g, dcn_params, v, bias)
        return v

    return deform_conv2d


def register_deform_conv2d_onnx_op(use_gathernd=False, enable_openvino_patch=False):
    """Register ONNX symbolic for ``torchvision::deform_conv2d``.

    Args:
        use_gathernd: Use GatherND instead of GatherElements.
                      GatherElements is better supported by TensorRT.
        enable_openvino_patch: Workaround for an OpenVINO GatherND shape bug.
    """
    register_custom_op_symbolic(
        "torchvision::deform_conv2d",
        deform_conv2d_func(use_gathernd, enable_openvino_patch),
        onnx_opset_version,
    )
