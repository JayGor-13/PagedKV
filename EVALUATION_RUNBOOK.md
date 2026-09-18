# Local development -> GitHub -> H200 evaluation

Updated 17 September 2026. The code is prepared locally; no pretrained H200 results are claimed.

## Implemented evaluation scope

- Qwen2 full-attention, batch-one adapter. Qwen2.5-1.5B is the control model; Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct are proposed larger-model runs using the same architecture. Their GPU execution and calibration capacity still need verification.
- Complete test splits of six LongBench v1 English QA tasks: narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa and musique. This is **not the entire 21-task LongBench suite**, LongBench v2, RULER, or an official leaderboard reproduction.
- Document-first plain-text prompting keeps the question out of cache compression. English QA F1 follows the benchmark's normalization/overlap convention; prompts differ from official per-task/model templates.
- All source rows are included. Oversized contexts fail by default. `--truncate-middle` explicitly shortens contexts, records every affected row, and retains the full split. Full split does not mean every original context fits without truncation.
- Main controls: vanilla, monolithic full KVTC, paged full KVTC, hot-only, approximate H2O/SnapKV/recency, and local full-retention grouped K4/V2, K2/V2 and K4/V4 controls.
- Recall arms: random, stored-key scan, optional reconstructed-cache refinement; oracle only for the planted-fact task with annotated locations.

Official dataset/metric references: [LongBench v1 instructions](https://github.com/THUDM/LongBench/blob/main/LongBench/README.md), [English QA metrics](https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py), [dataset archive](https://huggingface.co/datasets/zai-org/LongBench/tree/main).

## 1. Local checks before pushing

From a configured project root:

```powershell
python -m pytest -q
python -m scripts.smoke_model_integration
python -m experiments.summarize_results outputs/model_integration_smoke.json --out-dir outputs/smoke_summary
```

The smoke script creates a tiny random Qwen2 locally, checks all 14 arms with one refinement round, then verifies resume without duplicated answers. Its accuracy is meaningless. Tests cover numerical baseline controls, metadata-size accounting, dataset integrity, metrics, calibration identity and sweep construction in addition to the existing paging/model controls.

The GitHub repository should include code, tests, requirements, documentation and `datasets/reference_texts/`. Keep `.venv`, generated manifests, downloaded benchmark archives, model weights, calibration tensors and run outputs outside Git tracking. `.gitignore` contains these exclusions. No remote repository is configured by this update; supply the intended repository URL when ready to publish.

## 2. Clone and verify on the H200

The remaining commands are Linux shell commands, executed from the cloned repository root. Use the H200 environment's CUDA-enabled PyTorch; do not install `requirements-cpu.txt` or copy a Windows virtual environment.

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_DIRECTORY
python -m pip install -r requirements-model.txt
mkdir -p outputs
python -m pip freeze > outputs/environment.txt
nvidia-smi > outputs/nvidia-smi.txt
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m pytest -q
python -m scripts.smoke_model_integration
```

Run commands from the project root so source hashes and paths are consistent. Record the Git commit used for results. No jobs upload generated documents or predictions to an external service.

## 3. Pin revisions and check capacity

Resolve revisions once on the GPU machine, record them, and reuse them for preparation and execution:

```bash
export MODEL='Qwen/Qwen2.5-7B-Instruct'
export MODEL_REVISION=$(python -c "import os; from huggingface_hub import model_info; print(model_info(os.environ['MODEL']).sha)")
export DATASET_REVISION=$(python -c "from huggingface_hub import dataset_info; print(dataset_info('zai-org/LongBench').sha)")
printf '%s\n' "$MODEL" "$MODEL_REVISION" "$DATASET_REVISION" > outputs/revisions.txt

python -m experiments.preflight \
  --model "$MODEL" --revision "$MODEL_REVISION" \
  --context 32768 --out outputs/preflight_7b.json
```

Preflight loads configuration, not weights. It checks architecture/context compatibility and reports important memory components. These are not a peak-memory estimate: eigensolver workspaces, model weights, activations, duplicate quality-control caches and allocator overhead are additional.

Cross-layer calibration can be much more expensive on larger models. Inspect preflight before 14B. If capacity fails, measure and document a chosen rank/method change; do not silently substitute per-layer PCA or claim that an H200 guarantees a fit.

## 4. Pass a short pretrained control before a full benchmark

```bash
python -u -m experiments.run_rare_facts \
  --model "$MODEL" --revision "$MODEL_REVISION" --device cuda \
  --protocol short --ctx 1800 --docs 4 --seed 3 \
  --page 128 --hot-tokens 256 --recall-k 2 --halo 1 \
  --selection oracle,random,scan --refine 1 \
  --calibration-dir outputs/calibration \
  --manifest outputs/7b_short_control_manifest.json \
  --out outputs/7b_short_control_results.json
```

Inspect vanilla/full-paged/oracle answers and full-retention low-bit controls. If full-retention K2/V2 performs badly, report the control and investigate it; it cannot be treated as a validated baseline that our method has defeated. The local quantizers are quality simulations, not official KIVI implementations or packed GPU storage.

## 5. Prepare complete QA splits

```bash
python -m experiments.prepare_longbench \
  --model "$MODEL" --revision "$MODEL_REVISION" \
  --dataset-revision "$DATASET_REVISION" \
  --max-context 32768 --truncate-middle \
  --out-dir outputs/manifests/7b
```

Omit `--truncate-middle` if the experiment requires full original contexts; any oversized row then stops preparation instead of disappearing. The capacity budget includes question tokens and 128 answer tokens. An existing official ZIP can be supplied with `--data-zip PATH` for offline data access. ZIP contents are read without extracting files or executing a remote dataset script.

Each model needs tokenizer-matched manifests. Reuse one model's exact manifest across its ablations. Comparison across models uses the same source rows, but token counts and truncation can differ; both are recorded. Local calibration text remains the existing reproduction protocol and is not a reproduction of the paper's exact calibration corpus.

## 6. Run comparisons on all selected splits

```bash
python -m experiments.ablation_suite \
  --manifests outputs/manifests/7b/*.json \
  --profile comparison --out-dir outputs/7b_comparison \
  --plan outputs/7b_comparison_plan.json

python -m experiments.ablation_suite \
  --plan outputs/7b_comparison_plan.json --execute
```

Plan creation does not run GPU work. Execution uses sequential subprocesses and stops on failure. The comparison profile creates one run per supplied task/model manifest. Repeat preparation and comparison for 1.5B and 14B, using separate output directories and pinned model revisions. Larger-model configurations are prepared, not already validated.

## 7. Run ablations

```bash
python -m experiments.ablation_suite \
  --manifests outputs/manifests/7b/*.json \
  --profile ablation --out-dir outputs/7b_ablation \
  --plan outputs/7b_ablation_plan.json

python -m experiments.ablation_suite \
  --plan outputs/7b_ablation_plan.json --execute
```

The profile creates 16 variants per manifest: one base plus page size, coefficient count, hot budget, recall budget, halo, refinement, cold ratio and layer-band changes. Six tasks produce 96 jobs per model/seed. These are one-factor ablations, not all combinations of every parameter.

| Variable | Values represented |
| --- | --- |
| Page tokens | 16, 64, 128, 256 |
| Ranking coordinates | 64, 128, 256, 512 |
| Requested hot tokens | 512, 1024, 2048 |
| Selected pages at the base page size | 4, 8, 16 |
| Halo | 0, 1 |
| Extra refinement rounds | 0, 1 |
| Cold target ratio | 8, 16, 32 |
| Layer band | 14–20 or all layers |

Page-size variants keep the pre-halo token budget at 1,024 by adjusting page count. Halo expansion and hot overlap still change actual decoded/active tokens, which are recorded. Do not present them as exact equal-memory runs.

Use validation data for choosing settings; freeze a configuration before a final held-out comparison. The suite tool executes the supplied manifests and does not enforce a train/validation/test partition for you. `--seeds 3,7,11` varies random retrieval and calibration sampling on the same frozen documents, not the dataset sample itself. Use separately generated manifests to study planted-fact dataset seeds.

## 8. Resume safely and reuse calibration

- A report is written atomically after each question/document. A document is complete only after all requested questions/arms finish.
- `--resume` removes partial-document rows and reruns that document; fully finished documents are skipped. Config, source, manifest and model identity must match. It is not mid-token resume.
- Existing output paths fail unless `--resume` or `--overwrite` is supplied. Do not use overwrite for archival runs.
- `--calibration-dir` stores tensor-only artifacts identified by calibration inputs, model identity, tokenizer, codec config and core source hashes. It avoids repeating PCA/DP across compatible page/retrieval variants. Changed cold ratios currently use separate artifacts and may refit PCA; multi-ratio basis reuse remains an optimization.
- Each job still loads the model and validates calibration. The launcher currently does not keep one model resident across all jobs.

## 9. Recorded metrics and their scope

| Category | Recorded data |
| --- | --- |
| Quality | Raw generated token IDs/text and references, case-sensitive substring success, normalized exact match, maximum-reference English QA token F1. |
| Model divergence | First-answer-token KL relative to vanilla and top-1 agreement. This is not continuation KL or perplexity. |
| Retrieval | Page IDs/count/fraction, original active positions, attended-token fraction, annotated fact-span coverage when available. No invented oracle coverage for unannotated QA tasks. |
| Storage | Actual archive/monolithic/index/hot/active bytes, serialized ratio, selected and cumulative recovery payload bytes, coefficient scan/buffer counts, shared calibration residency. |
| Low-bit controls | Estimated packed code/metadata/residual bytes and actual reconstructed cache tensor bytes reported separately. Packed kernels are not implemented. |
| Timing | Prefill, hot scoring, compression, full decode, key scan, query prepass, page scoring, recovery, question forward/first logits, each later decode step and total generation. CUDA work is synchronized for these measurements. |
| GPU memory | Per-document quality-harness peak allocated/reserved CUDA bytes. All diagnostic caches coexist, so this is not a per-method serving peak. |
| Provenance | Manifest/source hashes, model revision, code config, library/CUDA/GPU metadata, dataset archive hash and original/truncated context lengths. |
| Statistics | Wilson intervals for binary success and document-paired bootstrap F1 differences against vanilla; raw observations exported. |

These are single quality-harness timing samples, not warmup/repeated isolated latency benchmarks. Retrieval timing excludes the once-per-document scan, which is recorded separately. Do not call a sum of selected timing fields a measured deployment end-to-end latency.

```bash
python -m experiments.summarize_results \
  outputs/7b_comparison/Qwen2.5-7B-Instruct/qasper/seed3_base.json \
  --out-dir outputs/7b_qasper_summary
```

This writes `summary.json`, `per_example.csv` and `per_document.csv`. Incomplete reports are rejected unless `--allow-partial` is used; partial status remains visible. Different datasets are not silently merged into one average.

## 10. Still required before a complete paper evaluation

1. Execute and inspect the pretrained runs on the H200; complete source splits and record failures rather than selecting successful rows.
2. Integrate official/validated baselines and actual equal-total-memory policies, including a packed low-bit execution path if making resident-memory comparisons. Current grouped controls are not that implementation.
3. Build isolated repeated end-to-end timing, GPU-only or explicitly CPU/GPU storage, and measure throughput under fixed memory. Current quality telemetry cannot establish those claims.
4. Expand beyond these six English QA tasks if claiming full LongBench/RULER coverage; add the appropriate metrics and model adapters instead of applying English QA F1 to every task.
5. Audit calibration/evaluation separation and compare quality/cost at the same operating points. A complete factorial ablation is not automatically necessary; justify each chosen factor and freeze final test choices.

The GitHub handoff should preserve this distinction: code and local controls are available; pretrained results and the remaining systems/baseline work are not yet complete.
