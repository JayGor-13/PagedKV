# Random-sample page controls — 2026-09-15

All six cases use the actual first-4/last-128 protection policy, 64 features, rank 16, target CR 4, and independently generated calibration/K/V tensors. This is a CPU numerical/storage audit, not an LLM-quality or GPU-performance experiment.

## Shared PCA

The actual encoder on every page was intercepted and checked to use the original calibration artifact. Recalibration was disabled during paging. All pages share one key basis and one value basis; K and V are calibrated separately. No basis tensor is serialized per page or into the archive. The common external calibration artifact is required for decoding and must be counted separately.

## Global protected tokens

Every case stores exactly 132 protected token positions per cache type, with no duplicates. Original positions 0–3 and the final 128 positions are checked explicitly; all other positions are compressed. A middle page shorter than 132 tokens is not automatically protected.

| Page | Token range (end excluded) | Protected | Compressed |
| ---: | --- | ---: | ---: |
| 0 | [0, 128) | 4 | 124 |
| 1 | [128, 256) | 0 | 128 |
| 2 | [256, 384) | 0 | 128 |
| 3 | [384, 512) | 0 | 128 |
| 4 | [512, 640) | 0 | 128 |
| 5 | [640, 768) | 0 | 128 |
| 6 | [768, 896) | 0 | 128 |
| 7 | [896, 1024) | 128 | 0 |

## Storage and exactness

| Seed / tokens | Page tokens | Monolithic bytes | Compact paged bytes | Overhead | Isolated/full exact K,V | Separate monolithic encoding exact K,V |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 20260915 / 1024 | 64 | 65031 | 67878 | 4.38% | [True, True] | [False, False] |
| 20260915 / 1024 | 128 | 65031 | 66010 | 1.51% | [True, True] | [False, False] |
| 20260915 / 1024 | 256 | 65031 | 65317 | 0.44% | [True, True] | [False, False] |
| 20260916 / 1019 | 64 | 64922 | 67858 | 4.52% | [True, True] | [True, True] |
| 20260916 / 1019 | 128 | 64922 | 66025 | 1.70% | [True, True] | [False, False] |
| 20260916 / 1019 | 256 | 64922 | 65203 | 0.43% | [True, True] | [False, False] |

The isolated/full comparison requires exact tensor equality (zero tolerance) for EVERY page, for both K and V. The independently encoded monolithic comparison separately records exact equality and maximum absolute differences; it requires numerical agreement at rtol=1e-5, atol=1e-6 because matrix multiplication batch shapes can change floating-point rounding. Finite samples do not prove bitwise equality across devices or all inputs.

Quantized codes, widths, FP16 scales and shifts match the independently encoded monolithic cache exactly in all six cases: True. The largest reconstructed-value discrepancy is 3.57627869e-07. Identical stored quantization data localizes these differences to floating-point reconstruction rather than changed quantization decisions.

For the 1,024-token / 128-token-page sample, old-format JSON headers range from 268 to 299 bytes per K or V page, totaling 4710 bytes. Header size depends on rank and block assignments; 4.5 KB is not a fixed per-page header size. The compact format stores shared compressed metadata once plus 25 bytes per page and 8-byte offsets with a final sentinel.

In that sample, splitting entropy streams adds 1187 bytes versus monolithic entropy streams; container differences add -208 bytes. The signed components sum to the total overhead. This measures fragmentation separately from metadata.

The earlier 37.03% example used a 16-token recent window, not 128. Its protected-token policy was still global: the measured overhead was dominated by repeated headers, not repeated raw windows. The current random audit explicitly covers a 128-token window and partial final pages.

All reported archive sizes include headers and indices, but exclude shared calibration, Python objects, temporary buffers and any future hot cache. No codec changes were needed for these controls.

Run `python -m scripts.audit_page_controls`; raw results are in `outputs/page_controls.json`.
