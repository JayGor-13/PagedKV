"""Bit-exact serialization and entropy coding (paper Section 3.3).

Design goals:
  * The decoder reconstructs from the byte stream ALONE (plus the shared,
    amortized calibration artifact). No hidden float state.
  * Every byte is counted. `Payload.breakdown()` itemizes codes, per-block
    metadata, protected raw tokens and headers, so a reviewer can audit the
    compression ratio.

Layout note: `layout` controls the ORDER of quantized symbols in the stream.
The paper (Section 3.3) says only that symbols are "packed into a single byte
array". Both orders are implemented and are bit-exact inverses; which one the
paper used is an open question that materially changes the achievable ratio.
"""
import io
import json
import struct
import zlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import QUANT_BITS

MAGIC = b"KVTC"
VERSION = 1


def _compress(raw: bytes, codec: str, level: int) -> bytes:
    if codec == "identity":
        return raw
    if codec == "deflate":
        return zlib.compress(raw, level)
    if codec == "zstd":
        import zstandard as zstd
        return zstd.ZstdCompressor(level=level).compress(raw)
    raise ValueError(f"unknown codec {codec}")


def _decompress(buf: bytes, codec: str) -> bytes:
    if codec == "identity":
        return buf
    if codec == "deflate":
        return zlib.decompress(buf)
    if codec == "zstd":
        import zstandard as zstd
        return zstd.ZstdDecompressor().decompress(buf)
    raise ValueError(f"unknown codec {codec}")


# ------------------------------------------------------------------ bit packing
def pack_codes(codes: np.ndarray, widths: np.ndarray, layout: str) -> bytes:
    """codes: (n_tok, R) uint8. widths: (R,) int, bits per column."""
    n, R = codes.shape
    if R == 0:
        return b""
    cols = []
    for j in range(R):
        w = int(widths[j])
        u = np.unpackbits(codes[:, j][:, None], axis=1)[:, 8 - w:]
        cols.append(u)
    if layout == "token_major":
        flat = np.concatenate(cols, axis=1).reshape(-1)
    elif layout == "component_major":
        flat = np.concatenate([c.reshape(-1) for c in cols])
    else:
        raise ValueError(f"unknown layout {layout}")
    return np.packbits(flat).tobytes()


def unpack_codes(buf: bytes, n: int, widths: np.ndarray, layout: str) -> np.ndarray:
    R = len(widths)
    if R == 0:
        return np.zeros((n, 0), dtype=np.uint8)
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8))
    total = int(widths.sum()) * n
    bits = bits[:total]
    out = np.zeros((n, R), dtype=np.uint8)
    if layout == "token_major":
        rows = bits.reshape(n, int(widths.sum()))
        off = 0
        for j in range(R):
            w = int(widths[j])
            seg = rows[:, off:off + w]
            pad = np.zeros((n, 8 - w), dtype=np.uint8)
            out[:, j] = np.packbits(np.concatenate([pad, seg], axis=1), axis=1)[:, 0]
            off += w
    else:
        off = 0
        for j in range(R):
            w = int(widths[j])
            seg = bits[off:off + w * n].reshape(n, w)
            pad = np.zeros((n, 8 - w), dtype=np.uint8)
            out[:, j] = np.packbits(np.concatenate([pad, seg], axis=1), axis=1)[:, 0]
            off += w * n
    return out


# ------------------------------------------------------------------ container
@dataclass
class Payload:
    blob: bytes
    parts: Dict[str, int]          # byte counts before entropy coding
    stored: Dict[str, int]         # byte counts AFTER entropy coding

    def nbytes(self) -> int:
        return len(self.blob)

    def breakdown(self) -> str:
        t = self.nbytes()
        rows = [f"  {k:22s} {v:>12,} B  ({100*v/t:5.1f}%)"
                for k, v in sorted(self.stored.items(), key=lambda x: -x[1])]
        return f"total {t:,} B\n" + "\n".join(rows)


def serialize(codes: np.ndarray, widths: np.ndarray,
              scales: np.ndarray, shifts: np.ndarray,
              protected: np.ndarray, header: dict,
              codec: str = "deflate", level: int = 6,
              layout: str = "token_major") -> Payload:
    """Build the stored blob.

    codes:     (n_tok, R) uint8  quantized symbols for coded coordinates
    widths:    (R,) uint8        bits per coded coordinate
    scales:    (n_blocks, n_tok) float16
    shifts:    (n_blocks, n_tok) float16
    protected: (n_prot, p) float32 -> stored bf16, raw sink + window tokens
    header:    JSON-serializable metadata (block table, shapes, config)
    """
    code_raw = pack_codes(codes, widths, layout)
    meta_raw = (np.ascontiguousarray(scales, dtype=np.float16).tobytes()
                + np.ascontiguousarray(shifts, dtype=np.float16).tobytes())

    # Protected tokens (sinks + window) are stored uncompressed at 16 bits.
    # fp16 is preferred over bf16: same width, 10 mantissa bits vs 7, and KV
    # magnitudes sit far inside fp16 range. Fall back to bf16 only on overflow.
    amax = float(np.abs(protected).max()) if protected.size else 0.0
    if amax < 65504.0:
        header = dict(header, prot_dtype="fp16")
        prot_raw = protected.astype(np.float16).tobytes()
    else:
        header = dict(header, prot_dtype="bf16")
        u = protected.astype(np.float32).view(np.uint32)
        prot_raw = (u >> 16).astype(np.uint16).tobytes()
    hdr = json.dumps(header, separators=(",", ":")).encode()

    c_code = _compress(code_raw, codec, level)
    c_meta = _compress(meta_raw, codec, level)
    c_prot = _compress(prot_raw, codec, level)

    buf = io.BytesIO()
    buf.write(MAGIC)
    buf.write(struct.pack("<B", VERSION))
    for chunk in (hdr, c_code, c_meta, c_prot):
        buf.write(struct.pack("<Q", len(chunk)))
        buf.write(chunk)
    blob = buf.getvalue()

    parts = dict(header=len(hdr), codes=len(code_raw),
                 block_meta=len(meta_raw), protected=len(prot_raw))
    stored = dict(header=len(hdr), codes=len(c_code),
                  block_meta=len(c_meta), protected=len(c_prot),
                  container=len(blob) - len(hdr) - len(c_code) - len(c_meta) - len(c_prot))
    return Payload(blob=blob, parts=parts, stored=stored)


def deserialize(blob: bytes, codec: str = "deflate"):
    assert blob[:4] == MAGIC, "bad magic -- not a KVTC blob"
    ver = struct.unpack("<B", blob[4:5])[0]
    assert ver == VERSION, f"version {ver} != {VERSION}"
    off = 5
    chunks = []
    for _ in range(4):
        (ln,) = struct.unpack("<Q", blob[off:off + 8])
        off += 8
        chunks.append(blob[off:off + ln])
        off += ln
    hdr = json.loads(chunks[0].decode())
    code_raw = _decompress(chunks[1], codec)
    meta_raw = _decompress(chunks[2], codec)
    prot_raw = _decompress(chunks[3], codec)

    n_tok = hdr["n_compressed"]
    widths = np.array(hdr["widths"], dtype=np.uint8)
    codes = unpack_codes(code_raw, n_tok, widths, hdr["layout"])

    nb = hdr["n_blocks"]
    half = nb * n_tok
    m = np.frombuffer(meta_raw, dtype=np.float16)
    scales = m[:half].reshape(nb, n_tok)
    shifts = m[half:half + half].reshape(nb, n_tok)

    p = hdr["p"]
    n_prot = hdr["n_protected"]
    if hdr.get("prot_dtype", "bf16") == "fp16":
        protected = np.frombuffer(prot_raw, dtype=np.float16).astype(np.float32)
    else:
        pr = (np.frombuffer(prot_raw, dtype=np.uint16).astype(np.uint32) << 16)
        protected = pr.view(np.float32)
    protected = protected.reshape(n_prot, p)
    return hdr, codes, widths, scales, shifts, protected


def widths_for(blocks: List[Tuple[int, int, str]]) -> np.ndarray:
    """Per-coded-coordinate bit widths, in block order ('none' contributes none)."""
    w = []
    for s, e, t in blocks:
        if t == "none":
            continue
        w.extend([QUANT_BITS[t]] * (e - s))
    return np.array(w, dtype=np.uint8)
