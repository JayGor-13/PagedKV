# CPU-only test and launch rehearsal

Use this workflow to check the repository from a clean Linux clone before using
H200 time. It does not install CUDA packages and cannot validate native attention
kernels, BF16, GPU memory, model sharding, benchmark quality, or latency.

## 1. Clone and run the dependency-free launch rehearsal

Python 3.10–3.12 is supported. Python 3.12 is shown below.

```bash
git clone https://github.com/JayGor-13/PagedKV.git
cd PagedKV
git rev-parse HEAD
python3.12 --version

python3.12 -m experiments.phase_one_cpu_rehearsal \
  --out "outputs/cpu-rehearsal-$(git rev-parse --short HEAD)"
cat "outputs/cpu-rehearsal-$(git rev-parse --short HEAD)/rehearsal-summary.json"
```

Expected: 64 completed jobs, two scheduler passes, and no populated score fields.
This uses synthetic manifests and harmless CPU workers. It exercises the same
matrix planner, GPU-slot scheduler, partial/completed checkpoint resume, and
combined report writer used by the H200 launcher.

## 2. Create the main CPU environment

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-cpu.txt -r requirements-model.txt sentencepiece datasets
```

Fetch the three pinned upstream repositories needed by parity and provenance
tests. This downloads source code, not model weights.

```bash
.venv/bin/python -m scripts.fetch_baselines --methods freekv factory rocketkv
```

## 3. Run the complete main suite and executable controls

```bash
.venv/bin/python -m pytest tests -q -p no:cacheprovider
.venv/bin/python -m scripts.smoke --output outputs/cpu-codec-smoke.json
.venv/bin/python -m scripts.smoke_model_integration
.venv/bin/python -m pip check
```

The codec smoke uses synthetic tensors. The model integration uses a tiny random
Qwen2 model created locally and tests all 14 wiring arms plus resume. Neither
produces meaningful accuracy or performance numbers.

To check a **pretrained** model on one real item from each requested dataset,
authenticate with a Hugging Face read token. Enter it at the hidden terminal
prompt; do not paste it into a notebook, command argument, log, or Git file:

```bash
read -r -s -p 'Hugging Face read token: ' HF_TOKEN
echo
export HF_TOKEN

.venv/bin/python -m scripts.real_hf_cpu_smoke \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --out outputs/real-hf-cpu-smoke.json
```

This loads real Qwen weights and real LongBench v2/LongGenBench examples. It
shortens both prompts to 512 tokens before chat formatting and generates four
tokens per example, so it is a wiring check, not a quality score. The output
records pinned model/dataset revisions and never records the token. To verify
access to every requested H200 model at the same time, add the following flags
before `--out`:

```bash
--check-models meta-llama/Llama-3.1-8B-Instruct \
  Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct \
  Qwen/Qwen2.5-72B-Instruct
```

The Meta model requires approval for the account owning the token. If only a
public local check is desired, pass `--anonymous` and omit `--check-models`.

## 4. Run the pinned Transformers 4.45 baseline adapter controls

The H200 suite isolates baseline adapters from the newer Transformers version
used by our method. Reproduce that split on CPU:

```bash
python3.12 -m venv .envs/cpu-baselines
.envs/cpu-baselines/bin/python -m pip install --upgrade pip
.envs/cpu-baselines/bin/python -m pip install -r requirements-cpu.txt
.envs/cpu-baselines/bin/python -m pip install \
  transformers==4.45.2 accelerate==0.34.2 sentencepiece einops scipy

.envs/cpu-baselines/bin/python -m pytest \
  tests/test_phase_one_backend_math.py -q -p no:cacheprovider
.envs/cpu-baselines/bin/python -m pip check
```

These tests replace CUDA attention with CPU SDPA. They check cache updates,
positions, resets, Qwen/Llama bridges, and finite FP16 RocketKV math. Passing them
does not establish that FlashAttention, FlashInfer, or the native H200 kernels
work.

## 5. Move to the H200 workflow

After CPU checks pass, follow `H200_COMMANDS.md`. On the reported server, first
select CUDA toolkit 12.4 because `nvcc` currently reports 13.1. Then complete GPU
setup, obtain five visible H200s, authenticate a Hugging Face account with Llama
access, pass online preflight, and run the native smoke matrix. Do not start the
full benchmark from CPU evidence alone.
