#!/usr/bin/env python3
"""零依赖 MATLAB v5 (.mat) 读取器 —— 只处理本项目用到的整数数组。

为什么要手写：系统 Python 是 3.14，scipy 只有 cp313 的 wheel，装它会退化成
源码编译/长时间下载。官方两个 .mat 文件是 zlib 压缩的 uint16 数组，解析不到 60 行。
"""
from __future__ import annotations

import struct
import zlib

_SIZE = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 8, 7: 4, 9: 1}
_FMT = {1: "b", 2: "b", 3: "H", 4: "H", 5: "i", 6: "d", 7: "f", 9: "B"}


def _tag(buf: bytes, off: int):
    v = int.from_bytes(buf[off : off + 4], "little")
    if v >> 16:  # small data element: size<<16 | type, 数据紧随其后 4 字节
        return v & 0xFFFF, v >> 16, off + 4, off + 8
    size = int.from_bytes(buf[off + 4 : off + 8], "little")
    # MAT v5 的普通元素按 8 字节边界对齐填充，size 是真实长度而非占位长度
    return v, size, off + 8, off + 8 + ((size + 7) // 8) * 8


def _matrix(buf: bytes, off: int, size: int):
    end = off + size
    p = off

    _, s, po, p = _tag(buf, p)  # array flags
    cls = int.from_bytes(buf[po : po + 4], "little") & 0xFF

    _, s, po, p = _tag(buf, p)  # dimensions
    dims = [int.from_bytes(buf[po + 4 * i : po + 4 * i + 4], "little") for i in range(s // 4)]

    _, s, po, p = _tag(buf, p)  # name
    name = buf[po : po + s].decode("latin-1").rstrip("\x00")

    typ, s, po, _ = _tag(buf, p)  # real part
    if typ not in _FMT:
        raise ValueError(f"{name}: 不支持的数组类型 {typ}")
    n = s // _SIZE[typ]
    vals = list(struct.unpack(f"<{n}{_FMT[typ]}", buf[po : po + s]))
    if p > end:
        raise ValueError(f"{name}: 越界")
    return name, dims, cls, vals


def _elem(buf: bytes, off: int):
    typ, size, poff, nxt = _tag(buf, off)
    if typ == 15:  # miCOMPRESSED —— 压缩块紧排，不做 8 字节对齐填充
        inner = zlib.decompress(buf[poff : poff + size])
        return _elem(inner, 0)[0], poff + size
    if typ == 14:  # miMATRIX
        return _matrix(buf, poff, size), nxt
    return (None, None, typ, buf[poff : poff + size]), nxt


def load_mat(path) -> dict[str, list[int]]:
    """返回 {变量名: 值列表}。只支持本项目用到的整数/浮点数组。"""
    buf = open(path, "rb").read()
    if b"MATLAB 5.0" not in buf[:128]:
        raise ValueError(f"{path}: 不是 MAT v5 文件")
    out: dict[str, list[int]] = {}
    off = 128
    while off + 8 <= len(buf):
        (name, _dims, _cls, vals), off = _elem(buf, off)
        if name:
            out[name] = vals
    return out


if __name__ == "__main__":
    import sys

    for f in sys.argv[1:]:
        d = load_mat(f)
        for k, v in d.items():
            print(f"{f}: {k} = {len(v)} 项, 范围 {min(v)}..{max(v)}")
