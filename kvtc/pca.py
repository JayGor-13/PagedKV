"""PCA basis fitting for KVTC (paper Section 3.1).

Two paths:
  gram_eigh   - stream G = sum (x-mu)(x-mu)^T, then eigh. Memory is O(p^2),
                INDEPENDENT of the number of calibration tokens. This is what
                makes 200K-token calibration possible on a 16GB GPU:
                    p=7168  (Qwen2.5-1.5B)  -> 0.21 GB
                    p=32768 (Llama-3.1-8B)  -> 4.29 GB
  randomized  - Halko et al. 2011 with 8 power iterations, what the paper used.

Both return V with columns ordered by DESCENDING explained variance, which the
DP relies on.
"""
import torch


def fit_pca_gram(chunks, rank=None, device="cuda", dtype=torch.float32):
    """Exact PCA via streaming Gram matrix + symmetric eigendecomposition.

    Args:
        chunks: iterable of (n_i, p) float tensors (CPU or GPU).
        rank: keep this many components (None = all p).
    Returns:
        mu (p,), V (p, rank), evals (rank,)   -- all float32 on `device`
    """
    p, n = None, 0
    s1 = None
    for m in chunks:
        if p is None:
            p = m.shape[1]
            s1 = torch.zeros(p, dtype=torch.float64, device=device)
        s1 += m.to(device, dtype).sum(0).double()
        n += m.shape[0]
    if p is None:
        raise ValueError("no calibration data")
    mu = (s1 / n).to(dtype)

    G = torch.zeros(p, p, dtype=dtype, device=device)
    for m in chunks:
        X = m.to(device, dtype) - mu
        G += X.T @ X
    G /= n
    G = (G + G.T) * 0.5                      # symmetrize against fp drift

    evals, evecs = torch.linalg.eigh(G.double())
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order].clamp_min(0), evecs[:, order]
    if rank is not None and rank < p:
        evals, evecs = evals[:rank], evecs[:, :rank]
    return mu.to(dtype), evecs.to(dtype).contiguous(), evals.to(dtype)


def fit_pca_randomized(chunks, rank, device="cuda", n_iter=8, seed=0,
                       dtype=torch.float32):
    """Randomized SVD (Halko et al. 2011), the paper's method.

    Requires materializing the centred data matrix, so it is only usable when
    n_calib * p fits in memory. Provided for fidelity checks against gram_eigh.
    """
    X = torch.cat([m.to(device, dtype) for m in chunks], dim=0)
    mu = X.mean(0)
    X = X - mu
    n, p = X.shape
    k = min(rank, p, n)
    g = torch.Generator(device=device).manual_seed(seed)
    Omega = torch.randn(p, k, generator=g, device=device, dtype=dtype)
    Y = X @ Omega
    Q, _ = torch.linalg.qr(Y)
    for _ in range(n_iter):                  # power iterations
        Q, _ = torch.linalg.qr(X.T @ Q)
        Q, _ = torch.linalg.qr(X @ Q)
    B = Q.T @ X                              # (k, p)
    _, S, Vh = torch.linalg.svd(B, full_matrices=False)
    V = Vh.T[:, :k].contiguous()
    evals = (S[:k] ** 2) / n
    return mu, V, evals


def project(X, mu, V):
    """(X - mu) @ V"""
    return (X - mu) @ V


def reconstruct(D, mu, V):
    """D @ V.T + mu"""
    return D @ V.T + mu


def explained_variance_ratio(evals):
    t = evals.sum()
    return (evals / t) if t > 0 else evals
