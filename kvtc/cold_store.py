"""Indexed, independently decodable token pages using the existing KVTC codec.

The shared calibration artifact is external, just as for monolithic KVTC.
This CPU storage prototype does not select pages or integrate model attention.
"""
from dataclasses import dataclass, replace
import json
import operator
import struct
import zlib

import torch

from .codec import KVTCCodec
from .serialize import Payload
from . import serialize as ser

_PREFIX = struct.Struct('<4sBQQ')  # magic, version, stored metadata length, page count
_ENTRY = struct.Struct('<QQQQQ')  # start, end, offset, key bytes, value bytes
_OFFSET = struct.Struct('<Q')
_PAGE = struct.Struct('<B6I')  # protected dtypes, six compressed stream lengths


def _protection(length, start, end, sink_tokens, window_tokens):
    if length <= sink_tokens + window_tokens:
        return end - start, 0
    return (max(0, min(end, sink_tokens) - start),
            max(0, end - max(start, length - window_tokens)))


def _split_payload(blob):
    """Read an existing KVTC container without entropy decoding its streams."""
    off, chunks = 5, []
    for _ in range(4):
        size, = struct.unpack_from('<Q', blob, off)
        off += 8
        chunks.append(blob[off:off + size])
        off += size
    return json.loads(chunks[0]), chunks[1:]


def _legacy_payload(header, streams):
    """Transient adapter to reuse the original decoder; never stored in v2."""
    hdr = json.dumps(header, separators=(',', ':')).encode()
    chunks = [hdr, *streams]
    blob = b''.join([ser.MAGIC, bytes([ser.VERSION]),
                     *(struct.pack('<Q', len(c)) + c for c in chunks)])
    return Payload(blob, {}, {})


@dataclass(frozen=True)
class DecodedPages:
    positions: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    page_ids: tuple
    payload_bytes_read: int
    index_bytes: int  # loaded once when opening an archive


class ColdStore:
    """An immutable byte archive plus a shared calibrated decoder.

    Page IDs are supplied by the caller. Unselected pages stay compressed.
    Returned positions refer to the original document, not the packed subset.
    """

    def __init__(self, blob: bytes, codec: KVTCCodec):
        if codec.art is None:
            raise ValueError('calibrate the codec first')
        if len(blob) < _PREFIX.size:
            raise ValueError('truncated cold archive')
        magic, version, meta_len, count = _PREFIX.unpack_from(blob)
        if magic != b'KVPG' or version not in (1, 2):
            raise ValueError('unsupported cold archive')
        self.format_version = version
        table_bytes = count * _ENTRY.size if version == 1 else (count + 1) * _OFFSET.size
        self.index_bytes = _PREFIX.size + meta_len + table_bytes
        if self.index_bytes > len(blob):
            raise ValueError('truncated cold archive index')
        meta = blob[_PREFIX.size:_PREFIX.size + meta_len]
        self.metadata = json.loads(meta if version == 1 else zlib.decompress(meta))
        if self.metadata['p'] != codec.art.p:
            raise ValueError('calibration feature dimension does not match')
        if version == 1:
            self._entries = tuple(_ENTRY.unpack_from(blob, _PREFIX.size + meta_len + i * _ENTRY.size)
                                  for i in range(count))
        else:
            length, page_tokens = self.metadata['seq_len'], self.metadata['page_tokens']
            if page_tokens <= 0 or length < 0 or count != (length + page_tokens - 1) // page_tokens:
                raise ValueError('invalid page count')
            offsets = tuple(_OFFSET.unpack_from(blob, _PREFIX.size + meta_len + i * _OFFSET.size)[0]
                            for i in range(count + 1))
            if offsets[0] != self.index_bytes or offsets[-1] != len(blob):
                raise ValueError('invalid cold archive offsets')
            self._entries = tuple((i * page_tokens, min(length, (i + 1) * page_tokens),
                                   offsets[i], offsets[i + 1] - offsets[i], 0)
                                  for i in range(count))
        next_pos, next_offset = 0, self.index_bytes
        for start, end, offset, nk, nv in self._entries:
            if (start != next_pos or end <= start or offset != next_offset or nk <= 0
                    or (version == 1 and nv <= 0) or (version == 2 and nk < _PAGE.size)):
                raise ValueError('invalid cold archive index')
            next_pos, next_offset = end, offset + nk + nv
        if next_pos != self.metadata['seq_len'] or next_offset != len(blob):
            raise ValueError('cold archive length mismatch')
        self.blob = blob
        # Snapshot decoding configuration so later codec.reassign() cannot
        # silently change entropy decoding. Basis tensors remain shared.
        self._codec = KVTCCodec(replace(codec.cfg, entropy_codec=self.metadata['entropy_codec']),
                                device=codec.device)
        self._codec.art = replace(codec.art)
        # Zero-length streams are implicit in v2. Generate their valid empty
        # representation once for the unchanged legacy deserializer.
        self._empty_stream = (ser._compress(b'', self.metadata['entropy_codec'], 6)
                              if version == 2 else b'')

    @classmethod
    def encode(cls, codec, keys, values, page_tokens=128, format_version=2):
        """Encode using compact v2 by default; v1 is retained for comparisons.

        v2 shares metadata, derives token ranges and omits empty streams. The
        underlying nonempty entropy streams and numerical quantizers are unchanged.
        """
        page_tokens = operator.index(page_tokens)
        if page_tokens <= 0:
            raise ValueError('page_tokens must be positive')
        if format_version not in (1, 2):
            raise ValueError('unsupported cold archive format_version')
        if codec.art is None:
            raise ValueError('calibrate the codec first')
        if keys.ndim != 2 or keys.shape != values.shape or keys.shape[1] != codec.art.p:
            raise ValueError('keys and values must have matching (tokens, features) shapes')
        cfg = codec.cfg
        if cfg.sink_tokens < 0 or cfg.window_tokens < 0:
            raise ValueError('protection sizes must be nonnegative')
        length, p = keys.shape
        pages = []
        page_codec = KVTCCodec(device=codec.device)
        page_codec.art = codec.art
        for start in range(0, length, page_tokens):
            end = min(length, start + page_tokens)
            # Intersect the GLOBAL protection regions with this page. Applying
            # the original sink/window lengths to every page would protect too much.
            sink, window = _protection(length, start, end, cfg.sink_tokens, cfg.window_tokens)
            page_codec.cfg = replace(cfg, sink_tokens=sink, window_tokens=window)
            payload = page_codec.compress(keys[start:end], values[start:end])
            pages.append((start, end, payload['key'].blob, payload['value'].blob))
        if format_version == 2:
            return cls(cls._pack_compact(codec, length, page_tokens, pages), codec)
        metadata = json.dumps(dict(seq_len=length, p=p, page_tokens=page_tokens,
                                   entropy_codec=cfg.entropy_codec), separators=(',', ':')).encode()
        offset = _PREFIX.size + len(metadata) + len(pages) * _ENTRY.size
        index, bodies = [], []
        for start, end, key, value in pages:
            index.append(_ENTRY.pack(start, end, offset, len(key), len(value)))
            bodies.extend((key, value))
            offset += len(key) + len(value)
        blob = b''.join([_PREFIX.pack(b'KVPG', 1, len(metadata), len(pages)),
                         metadata, *index, *bodies])
        return cls(blob, codec)

    @staticmethod
    def _pack_compact(codec, length, page_tokens, pages):
        cfg, art = codec.cfg, codec.art
        shared = {}
        for name in ('key', 'value'):
            assignment = art.assignments[name]
            shared[name] = dict(rank=getattr(art, name).V.shape[1], blocks=assignment.blocks,
                                bits_per_token=assignment.bits_per_token)
        meta = dict(seq_len=length, p=art.p, page_tokens=page_tokens,
                    entropy_codec=cfg.entropy_codec, layout=cfg.layout,
                    sink=cfg.sink_tokens, window=cfg.window_tokens,
                    target_cr=cfg.target_cr, shared=shared)
        # This one shared metadata stream is decoded when opening the index.
        metadata = zlib.compress(json.dumps(meta, separators=(',', ':')).encode(), 6)
        bodies = []
        for _, _, key, value in pages:
            flags, streams = 0, []
            for bit, blob in enumerate((key, value)):
                hdr, chunks = _split_payload(blob)
                flags |= int(hdr['prot_dtype'] == 'bf16') << bit
                # Preserve every nonempty entropy stream byte for byte.
                if not hdr['n_compressed'] or not hdr['widths']:
                    chunks[0] = b''
                if not hdr['n_compressed'] or not hdr['n_blocks']:
                    chunks[1] = b''
                if not hdr['n_protected']:
                    chunks[2] = b''
                streams.extend(chunks)
            if any(len(c) >= 2**32 for c in streams):
                raise ValueError('a page stream exceeds 4 GiB; reduce page_tokens')
            bodies.append(_PAGE.pack(flags, *(len(c) for c in streams)) + b''.join(streams))
        offset = _PREFIX.size + len(metadata) + (len(pages) + 1) * _OFFSET.size
        offsets = [offset]
        for body in bodies:
            offset += len(body)
            offsets.append(offset)
        return b''.join([_PREFIX.pack(b'KVPG', 2, len(metadata), len(pages)), metadata,
                         *(_OFFSET.pack(o) for o in offsets), *bodies])

    def _page_payloads(self, page_id, which=None):
        if which not in (None, 'key', 'value'):
            raise ValueError('which must be key, value, or None')
        start, end, offset, nk, nv = self._entries[page_id]
        if self.format_version == 1:
            payloads = {}
            if which in (None, 'key'):
                payloads['key'] = Payload(self.blob[offset:offset + nk], {}, {})
            if which in (None, 'value'):
                payloads['value'] = Payload(self.blob[offset + nk:offset + nk + nv], {}, {})
            return payloads
        flags, *sizes = _PAGE.unpack_from(self.blob, offset)
        if flags & ~3 or _PAGE.size + sum(sizes) != nk:
            raise ValueError('invalid page record')
        cursor, streams = offset + _PAGE.size, []
        for stream_id, size in enumerate(sizes):
            needed = which is None or (stream_id < 3) == (which == 'key')
            streams.append((self.blob[cursor:cursor + size] if size else self._empty_stream)
                           if needed else b'')
            cursor += size
        meta = self.metadata
        sink, window = _protection(meta['seq_len'], start, end, meta['sink'], meta['window'])
        n = end - start - sink - window
        payloads = {}
        for bit, name in enumerate(('key', 'value')):
            if which is not None and name != which:
                continue
            shared = meta['shared'][name]
            hdr = dict(shared, which=name, p=meta['p'], seq_len=end-start,
                       n_compressed=n, n_protected=sink+window,
                       n_blocks=sum(t != 'none' for _, _, t in shared['blocks']),
                       widths=ser.widths_for(shared['blocks']).tolist() if n else [],
                       sink=sink, window=window, layout=meta['layout'],
                       target_cr=meta['target_cr'], prot_dtype='bf16' if flags & (1 << bit) else 'fp16')
            payloads[name] = _legacy_payload(hdr, streams[bit*3:bit*3+3])
        return payloads

    def storage_breakdown(self):
        """Stored byte counts; parsing this diagnostic does not decompress pages."""
        if self.format_version == 1:
            out = dict(index=self.index_bytes, page_headers=0, page_framing=0,
                       codes=0, scales_shifts=0, protected=0)
            for i in range(self.page_count):
                for payload in self._page_payloads(i).values():
                    _, chunks = _split_payload(payload.blob)
                    header_size, = struct.unpack_from('<Q', payload.blob, 5)
                    out['page_headers'] += header_size
                    out['page_framing'] += 37
                    for name, chunk in zip(('codes', 'scales_shifts', 'protected'), chunks):
                        out[name] += len(chunk)
        else:
            out = dict(index=self.index_bytes, page_headers=0, page_framing=self.page_count * _PAGE.size,
                       codes=0, scales_shifts=0, protected=0)
            for _, _, offset, _, _ in self._entries:
                _, *sizes = _PAGE.unpack_from(self.blob, offset)
                for name, size in zip(('codes', 'scales_shifts', 'protected') * 2, sizes):
                    out[name] += size
        assert sum(out.values()) == self.nbytes()
        return out

    @property
    def page_count(self):
        return len(self._entries)

    def nbytes(self):
        """Serialized archive bytes, including page headers and the index.

        Excludes shared calibration tensors, Python object overhead, transient
        buffers, and any additional hot cache; these need separate accounting.
        """
        return len(self.blob)

    def decode_pages(self, page_ids):
        ids = tuple(sorted(set(operator.index(i) for i in page_ids)))
        if any(i < 0 or i >= self.page_count for i in ids):
            raise ValueError('page ID outside the archive')
        keys, values, positions = [], [], []
        read = 0
        for i in ids:
            start, end, offset, nk, nv = self._entries[i]
            # Only selected payload slices reach the original decoder.
            payloads = self._page_payloads(i)
            key, value = self._codec.decompress(payloads)
            keys.append(key)
            values.append(value)
            positions.append(torch.arange(start, end))
            read += nk + nv
        empty = lambda: torch.empty((0, self.metadata['p']), dtype=torch.float32)
        return DecodedPages(torch.cat(positions) if positions else torch.empty(0, dtype=torch.long),
                            torch.cat(keys) if keys else empty(),
                            torch.cat(values) if values else empty(), ids, read, self.index_bytes)

    def decode_all(self):
        return self.decode_pages(range(self.page_count))
