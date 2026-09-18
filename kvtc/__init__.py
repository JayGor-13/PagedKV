"""Working reproduction of
"KV Cache Transform Coding for Compact Storage in LLM Inference"
(Staniszewski & Lancucki, NVIDIA / University of Warsaw, ICLR 2026,
 arXiv:2511.01815v2).

Copied from the user-supplied reproduction; see PROVENANCE.md. Local numerical
checks do not establish equivalence to the paper's reported model results.

Run `python -m pytest -q` for the available correctness suite.
"""
from .config import KVTCConfig, QUANT_BITS, QUANT_TYPES, ALLOWED_BLOCK_SIZES, preset_for
from .codec import KVTCCodec, Basis, CalibrationArtifact
from .dp import Assignment, assign_precision, assign_precision_bruteforce
from .pca import fit_pca_gram, fit_pca_randomized, project, reconstruct
from .quant import quantize_block, dequantize_block, used_bits
from .rope import apply_rope, undo_rope, get_cos_sin, roundtrip_error
from .capture import capture_document, feature_dim, cache_layers, restore_kv_for_model
from . import serialize

__version__ = "0.1.0"
__all__ = [
    "KVTCConfig", "KVTCCodec", "Basis", "CalibrationArtifact", "Assignment",
    "assign_precision", "assign_precision_bruteforce",
    "fit_pca_gram", "fit_pca_randomized", "project", "reconstruct",
    "quantize_block", "dequantize_block", "used_bits",
    "apply_rope", "undo_rope", "get_cos_sin", "roundtrip_error",
    "capture_document", "feature_dim", "cache_layers", "restore_kv_for_model",
    "serialize", "preset_for", "QUANT_BITS", "QUANT_TYPES", "ALLOWED_BLOCK_SIZES",
]
