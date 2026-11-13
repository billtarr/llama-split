#!/usr/bin/env python
"""Quantize a GGUF file (f16/bf16/f32) to q8_0 / q4_0 / q4_k_m-free simple types.

Pure-python (numpy) block quantizer, equivalent to llama-quantize for the
simple (non-k-quant) types. Written because no x86_64 llama-quantize binary
matching this llama.cpp tree exists on this machine.

Usage: python quantize_gguf.py in.gguf out.gguf q8_0|q4_0|q4_1|q5_0|q5_1
"""
import sys
import numpy as np
import gguf
from gguf import GGMLQuantizationType as QT
from gguf.quants import quantize

QMAP = {
    "q8_0": QT.Q8_0,
    "q4_0": QT.Q4_0,
    "q4_1": QT.Q4_1,
    "q5_0": QT.Q5_0,
    "q5_1": QT.Q5_1,
}
BLOCK = 32


def pyval(v):
    """gguf reader returns numpy arrays/scalars for array fields; struct.pack needs python types
    (numpy bool_ has no __index__, which makes ARRAY[BOOL] fields blow up)."""
    if isinstance(v, list):
        return [pyval(x) for x in v]
    if isinstance(v, tuple):
        return tuple(pyval(x) for x in v)
    if hasattr(v, "tolist"):
        v = v.tolist()
        return [pyval(x) for x in v] if isinstance(v, list) else v
    if hasattr(v, "item"):
        return v.item()
    return v


def main() -> int:
    src, dst, qname = sys.argv[1], sys.argv[2], sys.argv[3].lower()
    qtype = QMAP[qname]

    r = gguf.GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    print(f"arch={arch} tensors={len(r.tensors)} -> {qname}", flush=True)

    w = gguf.GGUFWriter(dst, arch, use_temp_file=True)

    # copy every metadata field verbatim (except the special GGUF.* header keys)
    for name, field in r.fields.items():
        if name.startswith("GGUF."):
            continue
        # field.types is [ARRAY, <elem type>] for arrays; types[-1] is the *element* type
        vt = gguf.GGUFValueType.ARRAY if len(field.types) > 1 else field.types[0]
        sub = field.types[-1] if len(field.types) > 1 else None
        w.add_key_value(name, pyval(field.contents()), vt, sub)

    # quantize the same set llama-quantize would: ndim >= 2 and ne0 % 32 == 0
    n_quant = n_keep = 0
    for t in r.tensors:
        ne0 = t.shape[0]
        data = t.data
        if len(t.shape) >= 2 and ne0 % BLOCK == 0:
            rows = data.reshape(-1, ne0).astype(np.float32, copy=False)
            q = quantize(rows, qtype)
            # raw_shape stays the byte-shaped numpy array: the writer converts it to the
            # element shape from the block size (passing t.shape would be read as bytes/row)
            w.add_tensor(t.name, q, raw_dtype=qtype)
            n_quant += 1
        else:
            arr = data.astype(np.float32, copy=False)
            w.add_tensor(t.name, np.ascontiguousarray(arr), raw_dtype=QT.F32)
            n_keep += 1
        if (n_quant + n_keep) % 50 == 0:
            print(f"  {n_quant + n_keep}/{len(r.tensors)}", flush=True)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"done: quantized={n_quant} kept_f32={n_keep} -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
