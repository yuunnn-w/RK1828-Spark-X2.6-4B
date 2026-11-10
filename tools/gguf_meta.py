#!/usr/bin/env python3
"""Dump GGUF metadata KV pairs (hand-rolled parser, no deps).

Type numbering follows rknn3-model-zoo/tokenizer/thirdparty/llama_vocab/ggml/include/gguf.h:
UINT8=0 INT8=1 UINT16=2 INT16=3 UINT32=4 INT32=5 FLOAT32=6 BOOL=7 STRING=8 ARRAY=9 UINT64=10 INT64=11 FLOAT64=12

用法: python3 gguf_meta.py <file.gguf> [filter_substring ...]
"""
import struct
import sys

FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}


def rd(f, fmt):
    n = struct.calcsize(fmt)
    b = f.read(n)
    if len(b) != n:
        raise EOFError("truncated")
    return struct.unpack("<" + fmt, b)[0]


def rd_str(f):
    ln = rd(f, "Q")
    if ln > 1 << 24:
        raise ValueError("implausible string length %d" % ln)
    return f.read(ln).decode("utf-8", "replace")


def read_val(f, t, key):
    if t == 8:
        return rd_str(f)
    if t == 9:
        et = rd(f, "I")
        cnt = rd(f, "Q")
        items = []
        for _ in range(cnt):
            if et == 8:
                items.append(rd_str(f))
            elif et in FMT:
                items.append(rd(f, FMT[et]))
            else:
                raise ValueError("unknown elem type %d for %s" % (et, key))
        return ("array", et, cnt, items)
    if t in FMT:
        return rd(f, FMT[t])
    raise ValueError("unknown type %d for %s" % (t, key))


def dump(path, flt):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            print("not a GGUF")
            return 1
        ver = rd(f, "I")
        n_tensors = rd(f, "Q")
        n_kv = rd(f, "Q")
        print("version=%d n_tensors=%d n_kv=%d" % (ver, n_tensors, n_kv))
        for _ in range(n_kv):
            key = rd_str(f)
            t = rd(f, "I")
            val = read_val(f, t, key)
            if isinstance(val, tuple) and val[0] == "array":
                _, et, cnt, items = val
                if key.endswith(".tokens"):
                    val = "tokens n=%d head=%r tail=%r" % (cnt, items[:8], items[-6:])
                else:
                    show = items[:40]
                    val = "array<%d> n=%d %s" % (et, cnt, show)
            if not flt or any(s in key for s in flt):
                s = repr(val) if isinstance(val, str) else str(val)
                if len(s) > 6000:
                    s = s[:6000] + " ...<trunc %d>" % len(s)
                print("%-44s t=%-3d %s" % (key, t, s))
    return 0


if __name__ == "__main__":
    sys.exit(dump(sys.argv[1], sys.argv[2:]))
