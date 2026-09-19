"""Indexed, independently decodable token pages using the existing KVTC codec.

The shared calibration artifact is external, just as for monolithic KVTC.
This CPU storage prototype does not select pages or integrate model attention.
"""
from dataclasses import dataclass, replace
import json
import operator
import struct
import zlib

import numpy as np
import torch

from .codec import KVTCCodec
from .serialize import Payload
from . import serialize as ser

_PREFIX = struct.Struct('<4sBQQ')  # magic, version, stored metadata length, page count
_ENTRY = struct.Struct('<QQQQQ')  # start, end, offset, key bytes, value bytes
_OFFSET = struct.Struct('<Q')
_PAGE = struct.Struct('<B6I')  # v2: protected dtypes, K/V codes, metadata, protected
_PAGE_SPLIT = struct.Struct('<B7I')  # v3: K head/tail codes, K metadata/protected, V streams


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
        if magic != b'KVPG' or version not in (1, 2, 3):
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
                    or (version == 1 and nv <= 0)
                    or (version == 2 and nk < _PAGE.size)
                    or (version == 3 and nk < _PAGE_SPLIT.size)):
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
        # Zero-length streams are implicit in compact formats. Generate their valid empty
        # representation once for the unchanged legacy deserializer.
        self._empty_stream = (ser._compress(b'', self.metadata['entropy_codec'], 6)
                              if version in (2, 3) else b'')

    @classmethod
    def encode(cls, codec, keys, values, page_tokens=128, format_version=3, key_head_rank=256):
        """Encode using split-key compact v3 by default.

        v2 shares metadata, derives token ranges and omits empty streams. v3
        additionally puts the leading key coefficients in an independently
        readable entropy stream; selected-page decode still reconstructs the
        identical complete code matrix. v1/v2 remain readable for comparisons.
        """
        page_tokens = operator.index(page_tokens)
        if page_tokens <= 0:
            raise ValueError('page_tokens must be positive')
        if format_version not in (1, 2, 3):
            raise ValueError('unsupported cold archive format_version')
        key_head_rank = operator.index(key_head_rank)
        if key_head_rank <= 0:
            raise ValueError('key_head_rank must be positive')
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
        if format_version in (2, 3):
            return cls(cls._pack_compact(codec, length, page_tokens, pages,
                                         format_version, key_head_rank), codec)
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
    def _pack_compact(codec, length, page_tokens, pages, format_version=2, key_head_rank=256):
        cfg, art = codec.cfg, codec.art
        shared = {}
        for name in ('key', 'value'):
            assignment = art.assignments[name]
            shared[name] = dict(rank=getattr(art, name).V.shape[1], blocks=assignment.blocks,
                                bits_per_token=assignment.bits_per_token)
        meta = dict(seq_len=length, p=art.p, page_tokens=page_tokens,
                    entropy_codec=cfg.entropy_codec, layout=cfg.layout,
                    deflate_level=cfg.deflate_level,
                    sink=cfg.sink_tokens, window=cfg.window_tokens,
                    target_cr=cfg.target_cr, shared=shared)
        if format_version == 3:
            head_rank = min(key_head_rank, art.key.V.shape[1])
            head_columns = sum(max(0, min(end, head_rank) - start)
                               for start, end, quant in art.assignments['key'].blocks
                               if quant != 'none' and start < head_rank)
            meta.update(key_head_rank=head_rank, key_head_columns=head_columns)
        # This one shared metadata stream is decoded when opening the index.
        metadata = zlib.compress(json.dumps(meta, separators=(',', ':')).encode(), 6)
        bodies = []
        for _, _, key, value in pages:
            flags, streams = 0, []
            if format_version == 3:
                key_hdr, key_chunks = _split_payload(key)
                widths = np.array(key_hdr['widths'], dtype=np.uint8)
                raw = ser._decompress(key_chunks[0], cfg.entropy_codec)
                codes = ser.unpack_codes(raw, key_hdr['n_compressed'], widths, key_hdr['layout'])
                split = meta['key_head_columns']
                head_raw = ser.pack_codes(codes[:, :split], widths[:split], key_hdr['layout'])
                tail_raw = ser.pack_codes(codes[:, split:], widths[split:], key_hdr['layout'])
                head = ser._compress(head_raw, cfg.entropy_codec, cfg.deflate_level) if head_raw else b''
                tail = ser._compress(tail_raw, cfg.entropy_codec, cfg.deflate_level) if tail_raw else b''
                flags |= int(key_hdr['prot_dtype'] == 'bf16')
                if not key_hdr['n_compressed'] or not key_hdr['n_blocks']:
                    key_chunks[1] = b''
                if not key_hdr['n_protected']:
                    key_chunks[2] = b''
                streams.extend((head, tail, key_chunks[1], key_chunks[2]))
                blobs = ((1, value),)
            else:
                blobs = enumerate((key, value))
            for bit, blob in blobs:
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
            record = _PAGE_SPLIT if format_version == 3 else _PAGE
            bodies.append(record.pack(flags, *(len(c) for c in streams)) + b''.join(streams))
        offset = _PREFIX.size + len(metadata) + (len(pages) + 1) * _OFFSET.size
        offsets = [offset]
        for body in bodies:
            offset += len(body)
            offsets.append(offset)
        return b''.join([_PREFIX.pack(b'KVPG', format_version, len(metadata), len(pages)), metadata,
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
        record = _PAGE_SPLIT if self.format_version == 3 else _PAGE
        flags, *sizes = record.unpack_from(self.blob, offset)
        if flags & ~3 or record.size + sum(sizes) != nk:
            raise ValueError('invalid page record')
        cursor, streams = offset + record.size, []
        for stream_id, size in enumerate(sizes):
            key_streams = 4 if self.format_version == 3 else 3
            needed = which is None or (stream_id < key_streams) == (which == 'key')
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
            widths = ser.widths_for(shared['blocks']) if n else np.zeros(0, dtype=np.uint8)
            hdr = self._payload_header(name, shared, end-start, n, sink, window,
                                       widths, flags, bit)
            if self.format_version == 3 and name == 'key':
                split = meta['key_head_columns']
                head_raw = ser._decompress(streams[0], meta['entropy_codec'])
                tail_raw = ser._decompress(streams[1], meta['entropy_codec'])
                head = ser.unpack_codes(head_raw, n, widths[:split], meta['layout'])
                tail = ser.unpack_codes(tail_raw, n, widths[split:], meta['layout'])
                codes = np.concatenate((head, tail), axis=1)
                raw = ser.pack_codes(codes, widths, meta['layout'])
                code_stream = ser._compress(raw, meta['entropy_codec'], meta.get('deflate_level', 6))
                chunks = (code_stream, streams[2], streams[3])
            else:
                base = 4 if self.format_version == 3 else bit * 3
                chunks = streams[base:base+3]
            payloads[name] = _legacy_payload(hdr, chunks)
        return payloads

    def _payload_header(self, name, shared, seq_len, n, sink, window, widths, flags, bit):
        return dict(shared, which=name, p=self.metadata['p'], seq_len=seq_len,
                    n_compressed=n, n_protected=sink+window,
                    n_blocks=sum(t != 'none' for _, _, t in shared['blocks']),
                    widths=widths.tolist(), sink=sink, window=window,
                    layout=self.metadata['layout'], target_cr=self.metadata['target_cr'],
                    prot_dtype='bf16' if flags & (1 << bit) else 'fp16')

    def key_head_symbols(self, page_id, topk):
        """Deserialize only the independently stored key head when available."""
        topk = operator.index(topk)
        if topk < 1:
            raise ValueError('topk must be positive')
        start, end, offset, nk, _ = self._entries[page_id]
        if self.format_version != 3 or topk > self.metadata['key_head_rank']:
            payload = self._page_payloads(page_id, 'key')['key']
            decoded = ser.deserialize(payload.blob, codec=self.metadata['entropy_codec'])
            if self.format_version == 1:
                read = nk
            else:
                record = _PAGE_SPLIT if self.format_version == 3 else _PAGE
                _, *sizes = record.unpack_from(self.blob, offset)
                key_count = 4 if self.format_version == 3 else 3
                read = record.size + sum(sizes[:key_count])
            return (*decoded, read, 0)
        flags, *sizes = _PAGE_SPLIT.unpack_from(self.blob, offset)
        if flags & ~3 or _PAGE_SPLIT.size + sum(sizes) != nk:
            raise ValueError('invalid page record')
        stream_offsets, cursor = [], offset + _PAGE_SPLIT.size
        for size in sizes:
            stream_offsets.append(cursor)
            cursor += size
        # Deliberately do not slice the tail or value streams here. On a file-backed
        # implementation these three reads map directly to independent range reads.
        streams = tuple(self.blob[stream_offsets[i]:stream_offsets[i]+sizes[i]]
                        if sizes[i] else self._empty_stream for i in (0, 2, 3))
        meta = self.metadata
        sink, window = _protection(meta['seq_len'], start, end, meta['sink'], meta['window'])
        n = end - start - sink - window
        shared = meta['shared']['key']
        all_widths = ser.widths_for(shared['blocks']) if n else np.zeros(0, dtype=np.uint8)
        head_columns = meta['key_head_columns']
        widths = all_widths[:head_columns]
        hdr = self._payload_header('key', shared, end-start, n, sink, window,
                                   widths, flags, 0)
        decoded = ser.deserialize(_legacy_payload(hdr, streams).blob,
                                  codec=meta['entropy_codec'])
        read = _PAGE_SPLIT.size + sizes[0] + sizes[2] + sizes[3]
        return (*decoded, read, sizes[1])

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
            record = _PAGE_SPLIT if self.format_version == 3 else _PAGE
            out = dict(index=self.index_bytes, page_headers=0, page_framing=self.page_count * record.size,
                       codes=0, scales_shifts=0, protected=0)
            for _, _, offset, _, _ in self._entries:
                _, *sizes = record.unpack_from(self.blob, offset)
                names = (('codes', 'codes', 'scales_shifts', 'protected',
                          'codes', 'scales_shifts', 'protected')
                         if self.format_version == 3 else
                         ('codes', 'scales_shifts', 'protected') * 2)
                for name, size in zip(names, sizes):
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
