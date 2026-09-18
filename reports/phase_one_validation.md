# Phase-one validation — 2026-09-18

**133 project tests passed across the two required environments.** No H200 run
or benchmark scores were produced locally.

| Check | Observed result |
|---|---|
| Main project suite, `pytest tests -q -p no:cacheprovider` | 115 passed, 18 version-gated adapter tests skipped; 48.92 s |
| Isolated Transformers 4.45.2 adapter suite | The 18 skipped tests ran separately: 18 passed; 13.90 s |
| Python compile check, `compileall experiments kvtc scripts` | Passed |
| Three phase-one CLI entry points, `--help` | Passed |
| Bash syntax of setup/run scripts | Passed |
| FreeKV, KVCache-Factory, RocketKV locked commits and tracked source integrity | Verified |
| Hardware available in this workspace | Windows, Torch 2.13.0+cpu; CUDA unavailable; zero CUDA devices |

The tests cover atomic checkpoints and identity rejection; GPU allocation without
overlap; continuation after a failed job; suppression of incomplete/smoke scores;
refusal to combine a stale judge result; FreeKV prompt truncation, answer parsing,
LongGenBench block/check semantics; stable sampling; cache reconstruction and
sparse absolute positions; selective query capture matching the original scorer;
and tiny-model generation/reset through each adapted baseline.

The baseline tests replace FlashAttention with a CPU SDPA implementation. The
full-cache wrapper matches unmodified HF prefill and decode logits on tiny Llama,
Llama-3 RoPE and Qwen models. Chunked H2O chooses the same cache as the dense score
calculation in its regression test. These checks validate the adapter logic,
**not** CUDA extension builds, native kernel numerics, real checkpoint outputs,
BF16 H200 memory fit, driver compatibility or physical multi-GPU transfers.

Issues addressed during validation: FreeKV's non-quantized full cache was not
returned; speculative state needed deterministic reset and device-local counters;
H2O's full score matrix would be impractical at 120K; RocketKV needed a Qwen
class/rotary API bridge and device-local rotary tables. Query capture now retains
only the positions the existing hot-cache scorer actually reads, preserving its
scores while reducing host RAM demand.

Remaining experimental qualifications: ours compresses the prompt archive and
retains generated KV, so its LongGenBench path is not a fixed-budget generation
algorithm; H2O is Factory's prefill variant. All methods are labeled accordingly.
Qwen follows FreeKV's explicit 120K unscaled context protocol. The benchmark
setup has not yet been installed or executed on H200.

The next hardware verification is:

```bash
bash scripts/setup_phase_one.sh
bash scripts/test_phase_one.sh
bash scripts/run_phase_one.sh --gpus 5 --smoke --out outputs/phase-one-smoke
```

See `PHASE_ONE_RUNBOOK.md` for authentication, full execution, automatic resume,
LongGenBench judging, and report locations. The setup/test helper requires Linux;
it is not intended to execute GPU kernels from this Windows workspace.
