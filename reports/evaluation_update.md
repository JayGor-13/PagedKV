# Primary evaluation update — 17 September 2026

## Implemented

- Expanded the real indexed-page runner with local grouped K4/V2, K2/V2 and K4/V4 quality controls, approximate H2O/SnapKV and recency controls.
- Added optional query refinement using reconstructed active entries, with cumulative page-read and query-pass costs.
- Added full-row manifests for six LongBench v1 English QA tasks, pinned dataset/tokenizer identity, explicit truncation records and rejection of silent row filtering.
- Added proposed larger-Qwen preflight, reusable calibration checkpoints, atomic reports and identity-checked document-level resume.
- Added quality, first-token KL, storage/retrieval and timing telemetry, paired-document bootstrap summaries, and per-example/per-document CSV exports.
- Added explicit comparison and 16-variant one-factor ablation plans for execution after cloning on the H200.
- Added the local -> GitHub -> H200 workflow in [EVALUATION_RUNBOOK.md](../EVALUATION_RUNBOOK.md).

## Validation actually performed

- Latest full local pytest run: **82 passed in 16.65 seconds**.
- Offline random-weight tiny Qwen2: **14 arms completed**.
- Reopened the same run with `--resume`, reused its calibration artifact and confirmed identical answer rows without duplication.
- Exported smoke metrics to JSON, per-example CSV and per-document CSV.
- Checked command-line entry points, documentation file links and code-fence balance.

No pretrained model weights or benchmark archive were downloaded during this update. No H200 inference, full benchmark evaluation or speed measurement was performed. No GitHub remote was configured or repository published.

## Deliberately unclaimed

- Larger Qwen2.5 7B/14B executions are prepared, not tested on GPU here.
- Six English QA tasks are not the entire LongBench suite. Document-first prompting is not official leaderboard prompting.
- Local quantization controls are not official KIVI/MiKV or packed GPU kernels. Estimated packed bytes are distinct from actual reconstructed tensor memory.
- One-factor plans are not a full factorial ablation, and complete statistical results require successful dataset runs.
- Quality-harness telemetry is not an isolated repeated end-to-end latency/throughput benchmark. Its diagnostic caches coexist.
- Exact equal-total-memory comparisons, GPU entropy decoding, continuation KL/perplexity, and unsupported task/model families remain pending.

The code additions support a larger reproducible quality campaign; they do not yet complete all systems and baseline work needed for publication claims.
