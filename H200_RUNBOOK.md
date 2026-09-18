# H200 runs for the primary paged KVTC implementation

For the new Llama-3.1-8B/Qwen2.5-7B/14B comparison, `--gpus N`, upstream environments,
combined tables and resume, use [H200_COMPARISON_RUNBOOK.md](H200_COMPARISON_RUNBOOK.md).

**Update 2026-09-17:** the expanded workflow is in [EVALUATION_RUNBOOK.md](EVALUATION_RUNBOOK.md).
It covers full English QA split preparation, proposed Qwen2.5 7B/14B runs,
baseline controls, refinement, calibration reuse, resume, ablation plans and metrics.
The commands below remain useful short/long planted-fact controls. Default runs now
include six extra baseline arms; add `--baselines none` to reproduce the earlier seven-arm configuration.

Run from the **KVTC improvement project root**, not the reference ICLR-paper folder.
Copy the entire project, including `datasets/reference_texts/`, to the H200 machine.
The following are Linux shell commands. The local Windows CPU environment must not be copied as a GPU environment.

## Environment

Use an H200 environment with a working CUDA PyTorch installation. Check it before installing the small model dependencies:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m pip install -r requirements-model.txt
python -m pytest -q
python -m scripts.smoke_model_integration
```

Do not install `requirements-cpu.txt` there: it pins a CPU-only torch wheel. Transformers 5.16.1 is the version used in the supplied reference and tested locally with a tiny Qwen2 model. CUDA execution of the adapter remains to be tested on the H200. For archival runs, record `python -m pip freeze`, GPU driver/runtime information, and pass a pinned Hugging Face model commit with `--revision`.

## First pretrained test: four short documents

```bash
python -u -m experiments.run_rare_facts \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda \
  --protocol short --ctx 1800 --docs 4 --seed 1 \
  --page 128 --hot-tokens 256 --recall-k 2 --halo 1 \
  --selection oracle,random,scan \
  --manifest outputs/short_seed1_4docs.json \
  --out outputs/short_seed1_p128_results.json
```

Defaults preserve the reference's 12 calibration documents of 2,048 tokens and 16x cold target. This is not a tiny calibration job: Qwen's cross-layer feature dimension is 7,168. The full basis/calibration should be run on the H200, not the local CPU.

The model/tokenizer may download on first use. Inspect each generated answer and the `completed` flag, not just aggregate percentages. The oracle arm sees the known fact span intentionally; it is a diagnostic upper-bound arm. Random/scan arms do not use fact locations for selection.

## Long-context protocol: freeze once, compare pages

Prepare the exact same tokenized inputs for all variants:

```bash
python -m experiments.run_rare_facts --prepare-only \
  --protocol long --ctx 16384 --docs 40 --seed 3 \
  --manifest outputs/long16k_seed3_40docs.json
```

Then run the first long-context evaluation:

```bash
python -u -m experiments.run_rare_facts --device cuda \
  --protocol long --ctx 16384 --docs 40 --seed 3 \
  --page 128 --hot-tokens 1024 --recall-k 8 --halo 1 \
  --selection oracle,random,scan \
  --manifest outputs/long16k_seed3_40docs.json \
  --out outputs/long16k_p128_results.json
```

Reuse that **same manifest path** when changing page size, selector or codec settings. An existing manifest determines the actual dataset/calibration, irrespective of new `--docs`, `--seed`, or `--ctx` flags; use a new manifest filename to intentionally change data. Start with 16K, then create a separate 32K manifest. The historical 32K reference JSON contains 40 outcomes, but not its document token IDs, so identical historical documents cannot be guaranteed.

## What the runner reports and does not claim

- The output includes per-question generated token IDs/text, success using the reference's `target in output` rule, selected original positions, bytes decoded, active cache bytes, hot bytes, shared calibration bytes, and key-scan coefficient-buffer size.
- `vanilla`, `monolithic_kvtc_full`, `paged_kvtc_full`, `hot_only`, `oracle`, `random`, and `scan` are evaluated on the same frozen documents.
- `scan` ranks from stored quantized keys. It scans all key entropy streams, but reconstructs full K/V only for selected pages. It is not zero-cost retrieval and does not promise that only selected compressed bytes are ever read.
- The archive uses CPU zlib; matrix operations can use the H200. This is not yet an entirely GPU-resident system.
- Full diagnostic caches coexist in this quality harness. Do not use its process peak memory or runtime as per-method deployment measurements.
- Local grouped K4/V2, K2/V2 and K4/V4 quality controls are now included. They reconstruct model-precision tensors; official packed KIVI and exact equal-memory budget matching remain pending. The reference's separate corrected KIVI file is partial (4/40 documents).
- No pretrained accuracy thresholds have been met locally. Tiny random-model tests only establish wiring and numerical controls.

Stop and investigate if the all-page control differs materially from monolithic/full decoding or if oracle recovery fails badly while full KVTC succeeds. Do not optimize the selector before identifying whether compression or sparse context is responsible.
