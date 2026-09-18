# Selective-decompression storage ablation

Measured locally on CPU on 2026-09-14. These are synthetic storage and numerical-correctness results, not LLM accuracy or speed results.

## Same workload, same 128-token pages

| Format | Stored bytes | Overhead versus original monolithic KVTC |
| --- | ---: | ---: |
| Original monolithic | 16,458 | 0% |
| Original paged format (v1) | 22,553 | 37.03% |
| Compact paged format (v2) | 17,748 | 7.84% |

The new format removes 4,805 bytes: 78.8% of the original paging overhead, or 21.3% of the old paged archive size.

The v2 format stores a single zlib-compressed metadata record for the archive, an array of 64-bit page offsets, and a 25-byte record per page containing protected-value dtype flags and six stream lengths. Page token ranges are derived from page size. Empty streams are omitted. Nonempty entropy streams remain byte-for-byte unchanged.

This is a container change. PCA calibration, quantizer choices, precision, protection regions and page size are unchanged. The decoder rebuilds transient legacy headers for selected pages to reuse the original reconstruction path. That adds allocation work; latency and peak memory have not been measured. Encoding currently creates legacy payloads before compacting them, so its transient memory use is not optimized.

## Stored-byte breakdown

| Part | Original paged v1 | Compact paged v2 |
| --- | ---: | ---: |
| Shared metadata and index | 408 | 255 |
| Repeated page JSON | 4,164 | 0 |
| Page framing | 592 | 200 |
| Quantized codes | 8,208 | 8,208 |
| Scales and shifts | 7,761 | 7,761 |
| Protected entries | 1,420 | 1,324 |

## Page-size and data sensitivity

Calibration uses 128 synthetic rows, target CR 4, at most 16 PCA coordinates, token-major serialization, four sink tokens and a 16-token recent window. These small feature dimensions are not representative of the full model-wide KVTC basis. There are two independent Gaussian seeds and one 64-feature correlated workload; no representative real-cache distribution has been evaluated.

Requested token locations are fixed at 129 and 641 for every page size. All pages containing those two locations are decoded. Thus larger pages also recover more unrelated tokens.

| Features / tokens / seed / data | Page tokens | v1 overhead | v2 overhead | v2 overhead vs compact one-page control | Tokens decoded |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 / 1024 / 31 / Gaussian | 32 | 140.17% | 21.78% | 24.65% | 64 |
| 16 / 1024 / 31 / Gaussian | 64 | 73.02% | 14.21% | 16.91% | 128 |
| 16 / 1024 / 31 / Gaussian | 128 | 37.03% | 7.84% | 10.39% | 256 |
| 16 / 1024 / 31 / Gaussian | 256 | 17.86% | 3.66% | 6.11% | 512 |
| 16 / 1024 / 31 / Gaussian | 512 | 6.63% | -0.07% | 2.29% | 1024 |
| 16 / 1024 / 37 / Gaussian | 32 | 140.11% | 21.75% | 24.62% | 64 |
| 16 / 1024 / 37 / Gaussian | 64 | 72.97% | 14.17% | 16.87% | 128 |
| 16 / 1024 / 37 / Gaussian | 128 | 36.95% | 7.76% | 10.30% | 256 |
| 16 / 1024 / 37 / Gaussian | 256 | 17.79% | 3.60% | 6.04% | 512 |
| 16 / 1024 / 37 / Gaussian | 512 | 6.69% | -0.01% | 2.35% | 1024 |
| 64 / 4096 / 31 / correlated | 32 | 72.09% | 18.83% | 19.14% | 64 |
| 64 / 4096 / 31 / correlated | 64 | 38.05% | 11.46% | 11.76% | 128 |
| 64 / 4096 / 31 / correlated | 128 | 19.55% | 6.23% | 6.51% | 256 |
| 64 / 4096 / 31 / correlated | 256 | 10.03% | 3.41% | 3.68% | 512 |
| 64 / 4096 / 31 / correlated | 512 | 5.24% | 1.98% | 2.25% | 1024 |

The compact one-page control applies the same metadata savings to full-cache storage. It prevents attributing all improvement to paging. Negative overhead relative to the original monolithic format can arise from cheaper metadata; it does not establish a fundamental compression advantage.

Every sweep row checks exact v1/v2 reconstructed K/V equality and exact selected/full-page equality within v2. Unit tests additionally cover both layouts, identity and DEFLATE coding, empty/short/partial pages, all-dropped coordinates, FP16/BF16 protected entries, invalid IDs, and selective recovery despite corruption of an unselected page.

All sizes include stored archive metadata and page indices. Shared calibration tensors, Python object overhead, transient buffers and a future hot cache are excluded. All archives use the same external calibration artifact. GPU performance, query selection and model attention remain unimplemented.

## Next measurements

Keep 128-token pages as the current development setting; measure 64/128/256 on actual model caches before selecting a deployment size. If entropy-stream fragmentation remains material, evaluate shared calibration-derived dictionaries with their storage cost counted, or independently decompressible subpages within larger indexed groups. Larger groups must be charged for all bytes and tokens decoded. These alternatives have not been implemented.

Reproduce with `python -m scripts.storage_overhead`. Raw measurements are written to `outputs/storage_overhead.json`.
