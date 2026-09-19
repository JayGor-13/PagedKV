"""KVTC configuration.

Working reproduction configuration. Paper references and model presets were
inherited from the supplied source; this is not an independently validated
reproduction of the paper's model results.
"""
from dataclasses import dataclass, field, asdict
from typing import Tuple
import json

# Paper Section 3.2 / Appendix B.17: the DP selects among these quantizer types.
# NOT free-form bit widths.
QUANT_TYPES: Tuple[str, ...] = ("none", "int2", "int4", "fp8")
QUANT_BITS = {"none": 0, "int2": 2, "int4": 4, "fp8": 8}

# Paper Section 3.2 / Appendix B.17
ALLOWED_BLOCK_SIZES: Tuple[int, ...] = (1, 16, 64, 256, 1024)


@dataclass
class KVTCConfig:
    # ---- compression target -------------------------------------------------
    target_cr: float = 16.0
    """Target compression ratio BEFORE entropy coding ('kvtc16x' in the paper).
    Sets a per-token DP budget. Actual serialized compression also depends on
    protected tokens, metadata, entropy coding, and shared calibration tensors."""

    # ---- PCA (Section 3.1, Appendix B.1) ------------------------------------
    pca_rank_cap: int = 8192
    """Dimensionality cut-off for the randomized SVD.
    Paper: 8K for Qwen (fewer KV heads), 10K for Llama 3.1 8B / 3.3 70B /
    Mistral NeMo 12B / MN-Minitron 8B."""

    calib_tokens: int = 200_000
    """Paper: 200K for Qwen, 160K for Llama/Mistral."""

    svd_method: str = "gram_eigh"
    """'gram_eigh'   - exact: stream Gram matrix, then symmetric eigendecomposition.
                       Gram accumulator memory is O(p^2). The codec still retains
                       input chunks and concatenates projected calibration data.
       'randomized'  - Halko et al. 2011, 8 power iterations. What the paper used.
    These are different numerical implementations; model equivalence is unverified."""

    randomized_iters: int = 8
    """Paper Appendix B.1: 8 iterations of the Halko et al. randomized algorithm."""

    # ---- protection policy (Section 3.1 'Sliding Windows and Sink Tokens') --
    sink_tokens: int = 4
    """s = 4. Oldest tokens, left uncompressed."""

    window_tokens: int = 128
    """w = 128. Most recent tokens, left uncompressed."""

    recompress_every: int = 16
    """c = 16. Paper Section 4 'Methods': compression is re-run every 16 tokens so
    the uncompressed window stays in the 112-128 range."""

    # ---- quantization / DP (Section 3.2, Appendix B.17) ---------------------
    block_sizes: Tuple[int, ...] = ALLOWED_BLOCK_SIZES
    quant_types: Tuple[str, ...] = QUANT_TYPES

    meta_bits_per_block: int = 32
    """16-bit shift + 16-bit scale per block, per token (microscaling style,
    Rouhani et al. 2023).

    AMBIGUITY: the paper says 'groups of subsequent PCA coordinates ... each
    group with shared 16-bit shift and scaling factors' and that the budget
    'equals the sum of payload bits across all coordinates plus per-group shift
    and scaling factors'. It does not state whether the scale is shared across
    tokens as well. We charge 32 bits per block PER TOKEN, which is the
    microscaling convention and makes the DP's per-token budget accounting
    self-consistent. If the scale were global across tokens, this cost would
    amortize to ~0 and achievable CR would be slightly higher."""

    dp_stride: int = 16
    """Block END positions are restricted to multiples of this.

    Default 16 restricts the search to aligned boundaries. The measured model
    assignments were unchanged while calibration time fell sharply. Rank must
    be divisible by the stride; tiny diagnostic configurations can request 1."""

    dp_calib_subsample: int = 4096
    """Token positions sampled from the calibration set to estimate per-block
    quantization error inside the DP. The DP needs error estimates, not the full
    matrix; this keeps it fast. Set 0 to use all calibration tokens."""

    # ---- entropy coding (Section 3.3, Appendix B.8) ------------------------
    entropy_codec: str = "deflate"
    """'deflate' (paper default, via nvCOMP), 'zstd', 'identity'.
    Appendix B.8 Table 14: DEFLATE 34.7-42.9 vs Identity 29.6-32.4 on
    Mistral NeMo 12B at kvtc32x."""

    deflate_level: int = 6

    layout: str = "token_major"
    """Serialization order of quantized symbols.

    *** THE OPEN QUESTION. *** The paper says only that symbols are 'packed into
    a single byte array' (Section 3.3). It does not specify the order.
      'token_major'     - all components for token 0, then token 1, ...
                          (the natural row-major dump of an (n_tok, r) tensor)
      'component_major' - all tokens for component 0, then component 1, ...
    Default is token_major as inherited from the supplied reproduction.
    Compare actual serialized sizes on the target workload."""

    # ---- keys ---------------------------------------------------------------
    pre_rope_keys: bool = True
    """Section 3.1: 'positional embeddings distort the apparent low-rank
    structure of keys and should be removed before compression'."""

    # ---- feature grouping (Section 3.1, Appendix B.10) ---------------------
    pca_layers: int = -1
    """Number of layers concatenated into one PCA feature vector.
    -1 = all layers (the paper's setting, p = n_layers * n_kv_heads * head_dim).

    CRITICAL: Appendix B.10 shows per-layer PCA (pca_layers=1) COLLAPSES at
    16x -- GSM8K 2.5 vs 52.0 for 32 layers. Reproducing that collapse is a
    positive control on the implementation (see verify_kvtc.py check 9)."""

    seed: int = 0

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        d["block_sizes"] = tuple(d["block_sizes"])
        d["quant_types"] = tuple(d["quant_types"])
        return cls(**d)


# Per-model calibration settings from Appendix B.1
MODEL_PRESETS = {
    "qwen":    dict(pca_rank_cap=8192,  calib_tokens=200_000),
    "llama":   dict(pca_rank_cap=10000, calib_tokens=160_000),
    "mistral": dict(pca_rank_cap=10000, calib_tokens=160_000),
}


def preset_for(model_name: str) -> dict:
    n = model_name.lower()
    for k, v in MODEL_PRESETS.items():
        if k in n:
            return dict(v)
    return dict(pca_rank_cap=8192, calib_tokens=200_000)
