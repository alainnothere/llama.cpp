#!/usr/bin/env python3
"""splice_head.py -- copy a GGUF, replacing its NextN/MTP head tensors (blk.<n_layer>.*) with the same-named tensors of
another GGUF, raw bytes, keeping each tensor's own type. 2026-11-07, head-precision experiment: Q4_u trunk + Q8_0 (or f16)
head. --mtp-head cannot do this: llama_model_load_mtp_head requantizes the file's head INTO the model's type (a Q8 head
on a Q4 file comes back out Q4_K). usage: splice_head.py BASE.gguf HEAD_SRC.gguf OUT.gguf [--blk 64]"""
import sys
sys.path.insert(0, "/home/xlayn/theOtherDocs/gitSrc/llama.cpp/gguf-py")
import gguf
base, hsrc, out = sys.argv[1:4]
blk = "blk." + (sys.argv[sys.argv.index("--blk") + 1] if "--blk" in sys.argv else "64") + "."
rb, rh = gguf.GGUFReader(base), gguf.GGUFReader(hsrc)
heads = {t.name: t for t in rh.tensors if t.name.startswith(blk)}
assert heads, f"no {blk}* in {hsrc}"
# --to-f16: a BF16 source head is converted to F16 on the way in (bf16 = top 16 bits of f32; f16 keeps 10 mantissa bits
# vs bf16's 7, range shrinks to 65504 which head weights never approach). Used 2026-11-07 because the ggml-vulkan bf16
# weight path is the least exercised one on this rig; f16 mul_mat/mmv is the production path for the token embedding.
to_f16 = "--to-f16" in sys.argv
import numpy as np
class Conv:
    def __init__(self, t):
        u16 = np.frombuffer(t.data.tobytes(), dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        self.data = f32.astype(np.float16).reshape([int(x) for x in reversed(t.shape.tolist())])
        self.name, self.shape, self.tensor_type = t.name, t.shape, gguf.GGMLQuantizationType.F16
if to_f16:
    heads = {n: (Conv(t) if t.tensor_type == gguf.GGMLQuantizationType.BF16 else t) for n, t in heads.items()}
arch = rb.fields["general.architecture"].contents()
w = gguf.GGUFWriter(out, arch=arch)
for f in rb.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=st)
w.add_key_value("general.mtp_head_source", hsrc.split("/")[-1], gguf.GGUFValueType.STRING)
order = []
for t in rb.tensors:
    src = heads.get(t.name, t)
    assert list(src.shape) == list(t.shape), (t.name, src.shape, t.shape)
    if t.name in heads:
        print(f"swap {t.name:45s} {t.tensor_type.name:6s} -> {src.tensor_type.name}", flush=True)
    w.add_tensor_info(src.name, src.data.shape, src.data.dtype, src.data.nbytes, src.tensor_type)
    order.append(src)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for src in order:
    w.write_tensor_data(src.data)
w.close()
print("wrote", out, flush=True)
