# Source and scope

The nine Python modules under `kvtc/` were copied from the user-supplied folder:

`C:\Users\jaygo\Desktop\DESKTOP\Research Papers\ICLR paper\kvtc`

The original folder is unchanged. Its source describes a third-party
reimplementation of *KV Cache Transform Coding for Compact Storage in LLM
Inference*, ICLR 2026, rather than an official author release. No new license is
asserted by this project; check source rights before public redistribution.

This working copy introduces targeted correctness fixes and tests. It does not
certify equivalence with all settings of the paper. CPU zlib, covariance-based PCA
and reproduction-specific quantization metadata remain explicit implementation
choices. There is no supplied hot/cold selector or associated experiment log.

Local corrections: preserve positive FP16 quantization scales; report actual
projected reconstruction SSE instead of the DP's baseline-relative objective;
default to unrestricted DP boundaries; reject partial stride tails; and prevent
skip-only transitions from introducing unconfigured quantized block sizes.
The SSE excludes any PCA truncation error outside the supplied projected matrix.

Validation on 2026-09-13: 23 CPU tests passed. The synthetic smoke run completed
calibration, byte serialization and decoding. This does not validate model
quality, GPU performance, or the user-reported 40-document experiment.

On 2026-09-14 the cold archive gained a compact v2 container with shared metadata,
smaller page indexing and implicit empty streams. v1 decoding and encoding remain
available for comparisons. Tests increased to 61 passing CPU cases; a 15-setting
synthetic storage sweep verified exact reconstructed K/V equality between v1 and
v2. See reports/storage_overhead.md for measurements and excluded memory costs.

On 2026-09-16, reviewed functions from the newly supplied ICLR-paper project were
adapted into `experiments/reference_task.py` and `reference_heuristics.py`. Source
file hashes and selected dataset definitions are in `experiments/reference_sources.json`.
Four local text sources were copied into `datasets/reference_texts/` for protocol
portability. Historical token IDs were not supplied; our runner freezes new token
manifests and does not claim historical-document identity.

The model adapter and stored-coefficient selector use the primary indexed codec;
the reference's full-cache masking and unquantized-key ranking were not adopted.
Local verification: 73 primary tests passed, 32 original-reference tests passed,
and all seven experiment arms completed with a tiny random-weight Qwen2. No
pretrained quality, equal-memory comparison, or H200 timing result is claimed.

On 2026-09-17 the primary runner gained reconstructed-cache refinement, local
grouped low-bit controls, telemetry, resume and calibration checkpoints. The
grouped quantization design follows the reviewed reference `exp_kivi_fair_long.py`
but handles partial groups without padding and explicitly separates estimated
packed size from actual reconstructed tensor residency. It is not official KIVI.

On 2026-09-19 compact v3 split each page's key-code symbols into independently
readable head and tail streams. The selector reads only the head, shared key
quantization metadata and protected keys; selected-page recovery reads the tail
and value streams. No coefficient or scale/shift metadata is duplicated. The
default DP stride was raised from 1 to 16 after the supplied p=7168 measurement
reported the same assignment with substantially lower calibration time.
English QA F1 follows the normalization/overlap convention documented in
https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py ; the implementation
is independent and uses only standard-library operations. Dataset preparation reads
the official LongBench archive and records its hash; benchmark data are not bundled.
Document-first prompts are deliberately identified as a different protocol from
official per-task/model prompting. Local verification: 82 tests and all 14 tiny-model
arms passed, including a resume/calibration-reuse check. No pretrained H200 run occurred.
