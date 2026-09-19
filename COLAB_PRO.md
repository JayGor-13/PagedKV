# Colab Pro single-GPU run

Use [notebooks/colab_pro_single_gpu_7b_8b.ipynb](notebooks/colab_pro_single_gpu_7b_8b.ipynb).
Select a Premium GPU runtime and run the hardware check first. The unquantized
FP16 7B/8B harness requires at least 35 GiB of GPU memory; the notebook is aimed
at an A100 40 GB assignment. Smaller single GPUs are rejected rather than
silently changing precision, quantizing weights, or offloading to CPU.

The notebook pins source commit `a0921ee`, installs the same isolated environments
as the Kaggle workflow, runs all-method smoke tests, and launches the complete
503-example LongBench v2 comparison at a 16K prompt cap and 12.5% KV target.
Checkpoints are written directly to Google Drive. Resume requires the same GPU
model because hardware is part of the run identity; performance measurements
from different GPU types must not be combined.

Colab's GPU type, usage limits, and runtime duration vary. The official FAQ says
resources are not guaranteed, managed runtimes generally run for at most 12
hours, and Colab Pro+ can run continuously for up to 24 hours when sufficient
compute units remain. Plan on multiple sessions for the 503-example comparison.
