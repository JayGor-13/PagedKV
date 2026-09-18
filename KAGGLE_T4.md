# Small-model experiments on Kaggle T4

Upload [notebooks/kaggle_t4.ipynb](notebooks/kaggle_t4.ipynb) to Kaggle, or copy
the cells below. Enable **Internet** and choose **GPU T4 x2** in Notebook Settings.
The runner uses **one T4**, sequentially, leaving the second GPU free. Do not run
the H200 setup script for this workflow.

Models: Qwen2.5 **0.5B**, **1.5B**, and **3B** Instruct. Start with 0.5B smoke,
then 1.5B. Methods: full cache, H2O, SnapKV, Quest, ArkVale, RocketKV, FreeKV, ours.
The seven baseline adapters use the pinned upstream math with a local PyTorch
SDPA attention replacement. Ours uses its existing SDPA adapter. Weights and KV
use **FP16**, without quantization. No FlashAttention or FlashInfer is installed.

This is a **shortened compatibility experiment**, not the H200 benchmark:

- First-N examples of LongBench v2 / FreeKV LongGenBench, with shorter generation.
- Both benchmarks can have their prompts truncated (head + tail) before chat.
- Sink/recent budgets are 64 each, selection budget 256, page size 32.
- Ours uses a rank cap of 64 and 512 WikiText training tokens for calibration.
- Ours retains generated KV. H2O remains the Factory prefill variant.
- LongGenBench reports completion only; no 32B judge is installed or run.
- No H200 speed comparison or official benchmark score should use these results.

Local CPU adapter tests validate the portable attention path, including FP16 and
GQA. Actual Kaggle installation, T4 kernels, memory fit and pretrained model
outputs still require the GPU smoke test. Every failed job stays visible.

## 1. Clone (notebook Python cell)

```python
%cd /kaggle/working
!git clone https://github.com/JayGor-13/PagedKV.git
%cd /kaggle/working/PagedKV
!git rev-parse HEAD
!python --version
!nvidia-smi
```

Use Python 3.11 or 3.12. Setup uses CUDA 12.6 PyTorch wheels and checks CUDA
availability; the notebook's driver must support them. Public Qwen models need
no gated-model approval. Internet access is required for packages, upstream
repositories, model weights and datasets. LongBench v2 also downloads a large
dataset even when evaluating a small subset.

## 2. Install (separate cell)

```bash
%%bash
set -euo pipefail
cd /kaggle/working/PagedKV
bash scripts/setup_kaggle_t4.sh
```

Two isolated environments are created: `.envs/t4-baselines` (Transformers 4.45.2)
and `.envs/t4-ours` (Transformers 5.16.1). The notebook kernel is not replaced.
Package inventories are saved under `outputs/t4-environment/`. A CUDA error here
must be resolved before running the next steps.

## 3. Regression tests

```bash
%%bash
set -euo pipefail
cd /kaggle/working/PagedKV
bash scripts/test_kaggle_t4.sh
```

These use tiny randomly initialized models, including the real portable attention
bridge, rather than downloading pretrained model weights. Version-specific tests
skip in the main environment and run in the baseline environment.

## 4. Native GPU smoke test: 0.5B, all eight methods

```bash
%%bash
set -euo pipefail
cd /kaggle/working/PagedKV
export CUDA_VISIBLE_DEVICES=0
.envs/t4-baselines/bin/python -u -m experiments.kaggle_t4 run \
  --models Qwen/Qwen2.5-0.5B-Instruct --smoke \
  --out outputs/t4-smoke
```

This runs **16 jobs**: one model x eight methods x two benchmarks. Each job uses
one example, a 1,024-token raw prompt cap, and up to eight generated tokens.
Smoke accuracy is omitted. Check `outputs/t4-smoke/comparison.md`: require all
16 jobs to be completed. Inspect each `result.log` if any job fails. The launcher
continues through failures and exits nonzero at the end if any failed.

## 5. Short experiment: 1.5B, all eight methods

```bash
%%bash
set -euo pipefail
cd /kaggle/working/PagedKV
export CUDA_VISIBLE_DEVICES=0
.envs/t4-baselines/bin/python -u -m experiments.kaggle_t4 run \
  --models Qwen/Qwen2.5-1.5B-Instruct \
  --samples 4 --prompt-cap 2048 --max-new-tokens 128 \
  --out outputs/t4-small
```

To test all three small models, replace the model selection with:

```text
--models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct
```

Use a **new output directory** when changing models, methods, benchmarks, sample
count, token limits or smoke mode. Optional subsets: `--methods full ours` and
`--benchmarks longbenchv2`. Maximum allowed prompt cap is 4,096; maximum output
length is 1,024. Start small; these upper limits are not memory-fit guarantees.

## 6. Results, progress and resume

The launcher prints the active job and its log path. Each example is atomically
saved to `outputs/t4-small/results/<model>/<benchmark>/<method>/result.json`.
The three comparison files (`.md`, `.csv`, `.json`) update after each worker exits.
Read the per-job log/JSON while a worker is running for immediate sample progress.

Run **exactly the same command again** to resume. Finished examples are skipped;
completed jobs verify identity and skip loading the model. An interrupted example
restarts. Source, package versions, configuration, upstream commits and GPU model
must match. Changing them requires a new output directory. Never reinstall or pull
new code midway through an experiment you intend to resume.

Kaggle sessions are temporary. Before ending a session, save/download a results
archive; do not assume the working directory will survive. This Python cell saves
outputs and the source commit, excluding environments and model caches:

```python
import subprocess
import zipfile
from pathlib import Path
from IPython.display import FileLink, display

root = Path('/kaggle/working/PagedKV')
archive = Path('/kaggle/working/pagedkv-t4-results.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('CODE_COMMIT.txt', subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True))
    for path in (root / 'outputs').rglob('*'):
        if path.is_file() and path.suffix != '.tmp' and path.name != '.run.lock':
            z.write(path, path.relative_to(root))
display(FileLink(str(archive)))
```

Prefer archiving when the launcher has stopped. To resume in a later session,
clone/check out the recorded commit, reinstall the same versions (use the saved
package inventories if resolution changed), and restore the archive's `outputs/`
under `/kaggle/working/PagedKV/outputs/`. Then repeat the original command with
the same GPU type. A mismatched environment will refuse resume instead of mixing
results. Downloaded model caches can be rebuilt; only checkpoints/calibration and
frozen inputs are needed for progress preservation.

To regenerate the table **after the launcher stops**:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/PagedKV
.envs/t4-baselines/bin/python -m experiments.kaggle_t4 report --out outputs/t4-small
```

The small-run accuracy describes the truncated first-N subset only. Empty judged
accuracy is intentional. Timings exclude model loading and are not an isolated
serving benchmark. Setup/downloads can dominate the first run; use recorded
sample durations for an estimate rather than extrapolating the eight-token smoke.

## Repair older RocketKV FP16 failures

Commit `c53cd86` could produce non-finite RocketKV retrieval scores during
LongGenBench sampling on a T4. The corrected path retains FP16 weights, caches and
layer outputs but accumulates RocketKV retrieval dot products, softmax and selected
values in FP32. It also records this distinction in each result.

If an existing run has that failure, first download its output archive, then pull
the latest code and run:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/PagedKV
git pull --ff-only
export CUDA_VISIBLE_DEVICES=0
.envs/t4-baselines/bin/python -u -m experiments.kaggle_t4 repair-rocket \
  --out outputs/t4-small
```

This archives the previous RocketKV result and log for **both benchmarks** under
`outputs/t4-small/repairs/pre-fp32-rocket/`, then recomputes only those two jobs.
All other completed methods remain untouched. `repairs/rocket-fp32.json` records
the reason, archived identities and repair outcomes. The action itself is
resumable: repeat the same command if the notebook disconnects.

Sources: [Qwen 0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
[Qwen 1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
[Qwen 3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct),
[FlashAttention 2.6.3 GPU support](https://github.com/Dao-AILab/flash-attention/tree/v2.6.3#installation-and-features).
