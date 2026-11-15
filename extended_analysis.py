"""
extended_analysis.py
====================
Supplementary analysis module for the bipartite graph kernel pipeline.

Provides five independent evaluation routines that complement the core
graph_kernel_adr_prediction.py pipeline:

  1. ood_evaluation        – Out-of-distribution generalisation under
                             cold-drug and cold-ADR data splits.
  2. negative_sampling_ablation – Comparison of random-uniform,
                             degree-stratified, and hard-negative
                             sampling strategies.
  3. aggregation_steps_ablation – Effect of walk truncation depth K
                             on drug-pair kernel performance.
  4. polypharmacy_top_predictions – Rank and retrieve the highest-scoring
                             drug-pair → ADR triples for downstream
                             clinical interpretation.
  5. runtime_benchmark     – Wall-clock timing for kernel construction,
                             Gram matrix assembly, SVM training, and
                             per-pair inference.

All routines accept a fitted BipartiteKernel instance and a DrugADRDataset
(or equivalent incidence matrix B) as inputs, and return structured dicts
suitable for tabulation or further analysis.

Dependencies: numpy, scipy, scikit-learn (sklearn), time (stdlib)
Optional:     pandas (for pretty-printing result tables)

Usage example
-------------
    from graph_kernel_adr_prediction import DrugADRDataset, OddNeumannKernel
    from extended_analysis import (
        ood_evaluation, negative_sampling_ablation,
        aggregation_steps_ablation, polypharmacy_top_predictions,
        runtime_benchmark,
    )

    B = ...               # (n_drugs × n_adrs) incidence matrix
    results = ood_evaluation(B, test_frac=0.20)
    print(results)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import time
import warnings
from typing import Dict, List, Optional, Tuple

# ── Numerical ─────────────────────────────────────────────────────────────────
import numpy as np

# ── Machine learning ──────────────────────────────────────────────────────────
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    roc_curve,
)
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def _build_block_adjacency(B: np.ndarray) -> np.ndarray:
    """
    Construct the (n_drugs + n_adrs) × (n_drugs + n_adrs) block adjacency
    matrix A from the drug–ADR incidence matrix B.

        A = | 0    B  |
            | B^T  0  |
    """
    m, n = B.shape
    N = m + n
    A = np.zeros((N, N), dtype=np.float64)
    A[:m, m:] = B
    A[m:, :m] = B.T
    return A


def _fit_odd_neumann(B: np.ndarray,
                     lam: float = 0.1,
                     K: int = 3,
                     eps: float = 1e-3) -> np.ndarray:
    """
    Compute the cross-partition block S[:m, m:] of the Odd Neumann
    Pseudo Kernel:

        S_odd = sum_{k=0}^{K} lambda^{2k+1} * A^{2k+1}

    with B·B^T correction on the drug–drug block and ridge regularisation.

    Returns
    -------
    S_cross : np.ndarray, shape (n_drugs, n_adrs)
    """
    m, n = B.shape
    N = m + n
    A = _build_block_adjacency(B)
    S = np.zeros((N, N), dtype=np.float64)
    Ak = A.copy()
    for k in range(K + 1):
        S += (lam ** (2 * k + 1)) * Ak
        if k < K:
            Ak = Ak @ A @ A
    # B·Bᵀ correction
    BBT = B @ B.T
    mx = BBT.max()
    if mx > 0:
        S[:m, :m] = BBT / mx
    S += eps * np.eye(N)
    return S[:m, m:]


def _fit_odd_path(B: np.ndarray,
                  K: int = 3,
                  eps: float = 1e-3) -> np.ndarray:
    """
    Compute the cross-partition block P[:m, m:] of the Odd Path
    Counting Kernel:

        P_odd = sum_{k=0}^{K} A^{2k+1}   (row-normalised)

    Returns
    -------
    P_cross : np.ndarray, shape (n_drugs, n_adrs)
    """
    m, n = B.shape
    N = m + n
    A = _build_block_adjacency(B)
    P = np.zeros((N, N), dtype=np.float64)
    Ak = A.copy()
    for k in range(K + 1):
        P += Ak
        if k < K:
            Ak = Ak @ A @ A
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P /= row_sums
    P += eps * np.eye(N)
    return P[:m, m:]


def _evaluate(y_true: np.ndarray,
              y_score: np.ndarray,
              k_vals: Tuple[int, ...] = (10, 20, 50)) -> Dict[str, float]:
    """
    Compute ROC-AUC, PR-AUC, F1 (at Youden-optimal threshold),
    and Precision@K for a set of K values.
    """
    if y_true.sum() == 0 or (1 - y_true).sum() == 0:
        nan = float("nan")
        result = dict(roc_auc=nan, pr_auc=nan, f1=nan)
        for k in k_vals:
            result[f"P@{k}"] = nan
        return result

    roc = roc_auc_score(y_true, y_score)
    prc = average_precision_score(y_true, y_score)

    # Youden-optimal threshold
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = np.argmax(tpr - fpr)
    opt_thr = thr[j] if j < len(thr) else 0.5
    yp = (y_score >= opt_thr).astype(int)
    f1 = f1_score(y_true, yp, zero_division=0)

    si = np.argsort(-y_score)
    result = dict(roc_auc=roc, pr_auc=prc, f1=f1)
    for k in k_vals:
        result[f"P@{k}"] = float(y_true[si[:k]].sum()) / k
    return result


def _sample_negatives_random(B: np.ndarray,
                              n: int,
                              rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Sample n negative (drug, ADR) pairs uniformly at random."""
    zero_rows, zero_cols = np.where(B == 0)
    idx = rng.choice(len(zero_rows), size=min(n, len(zero_rows)), replace=False)
    return list(zip(zero_rows[idx].tolist(), zero_cols[idx].tolist()))


def _sample_negatives_degree_stratified(B: np.ndarray,
                                        n: int,
                                        rng: np.random.Generator
                                        ) -> List[Tuple[int, int]]:
    """
    Sample n negative pairs with probability proportional to the product
    of drug degree and ADR degree (matches the degree distribution of
    positive edges, reducing selection bias).
    """
    drug_deg = B.sum(axis=1).astype(np.float64) + 1.0   # Laplace smoothing
    adr_deg  = B.sum(axis=0).astype(np.float64) + 1.0
    zero_rows, zero_cols = np.where(B == 0)
    weights = drug_deg[zero_rows] * adr_deg[zero_cols]
    weights /= weights.sum()
    idx = rng.choice(len(zero_rows), size=min(n, len(zero_rows)),
                     replace=False, p=weights)
    return list(zip(zero_rows[idx].tolist(), zero_cols[idx].tolist()))


def _sample_negatives_hard(S_cross: np.ndarray,
                            B: np.ndarray,
                            n: int) -> List[Tuple[int, int]]:
    """
    Select the n unobserved (drug, ADR) pairs with the highest kernel
    score — the hardest negatives for SVM boundary learning.
    """
    masked = S_cross.copy()
    masked[B == 1] = -np.inf          # exclude known positives
    flat  = masked.ravel()
    top_n = np.argsort(-flat)[:n]
    rows, cols = np.unravel_index(top_n, S_cross.shape)
    return list(zip(rows.tolist(), cols.tolist()))


# ════════════════════════════════════════════════════════════════════════════
# 1.  OOD EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def ood_evaluation(B: np.ndarray,
                   test_frac: float = 0.20,
                   K: int = 3,
                   lam: float = 0.1,
                   eps: float = 1e-3,
                   seed: int = 42) -> Dict[str, Dict]:
    """
    Evaluate kernel performance under three data-split protocols:

      - ``random``    : held-out edges drawn uniformly; test drugs and ADRs
                        appear in the training graph.
      - ``cold_drug`` : a fraction ``test_frac`` of drug nodes is removed
                        entirely from the training graph.
      - ``cold_adr``  : a fraction ``test_frac`` of ADR nodes is removed
                        entirely from the training graph.

    Parameters
    ----------
    B : np.ndarray, shape (n_drugs, n_adrs)
        Binary drug–ADR incidence matrix.
    test_frac : float
        Fraction of nodes (or edges) to hold out as test set.
    K : int
        Walk truncation depth for both kernels.
    lam : float
        Decay factor for the Odd-Neumann kernel.
    eps : float
        Ridge regularisation added to the diagonal.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict mapping protocol name → dict mapping kernel name → metrics dict.
    Each metrics dict contains: roc_auc, pr_auc, f1, P@10, P@20, P@50.
    """
    rng = np.random.default_rng(seed)
    m, n = B.shape
    pr, pc = np.where(B == 1)
    results = {}

    protocols = {
        "random":    None,
        "cold_drug": None,
        "cold_adr":  None,
    }

    for protocol in protocols:
        # ── Build training graph and test pairs ──────────────────────────────
        if protocol == "random":
            perm   = rng.permutation(len(pr))
            n_test = max(20, int(np.ceil(len(pr) * test_frac)))
            ti     = perm[:n_test]
            B_tr   = B.copy()
            B_tr[pr[ti], pc[ti]] = 0.0
            pos_test = list(zip(pr[ti].tolist(), pc[ti].tolist()))
            neg_test = _sample_negatives_random(B, len(pos_test), rng)

        elif protocol == "cold_drug":
            n_cold = max(2, int(np.ceil(m * test_frac)))
            cold   = set(rng.choice(m, size=n_cold, replace=False).tolist())
            B_tr   = B.copy()
            for d in cold:
                B_tr[d, :] = 0.0
            pos_test = [(r, c) for r, c in zip(pr, pc) if r in cold]
            neg_test = []
            for d in cold:
                zeros = np.where(B[d, :] == 0)[0]
                k_neg = max(2, int(B[d, :].sum()))
                ni    = rng.choice(zeros, size=min(k_neg, len(zeros)),
                                   replace=False) if len(zeros) else []
                neg_test.extend([(d, int(a)) for a in ni])

        elif protocol == "cold_adr":
            n_cold = max(2, int(np.ceil(n * test_frac)))
            cold   = set(rng.choice(n, size=n_cold, replace=False).tolist())
            B_tr   = B.copy()
            for a in cold:
                B_tr[:, a] = 0.0
            pos_test = [(r, c) for r, c in zip(pr, pc) if c in cold]
            neg_test = []
            for a in cold:
                zeros = np.where(B[:, a] == 0)[0]
                k_neg = max(2, int(B[:, a].sum()))
                ni    = rng.choice(zeros, size=min(k_neg, len(zeros)),
                                   replace=False) if len(zeros) else []
                neg_test.extend([(int(d), a) for d in ni])

        # Balance classes
        mn       = min(len(pos_test), len(neg_test))
        pos_test = pos_test[:mn]
        neg_test = neg_test[:mn]
        pairs    = pos_test + neg_test
        y        = np.array([1] * len(pos_test) + [0] * len(neg_test))

        if len(pairs) < 20 or y.sum() < 5:
            nan_m = {k2: float("nan")
                     for k2 in ["roc_auc", "pr_auc", "f1", "P@10", "P@20", "P@50"]}
            results[protocol] = {"odd_neumann": nan_m, "odd_path": nan_m}
            continue

        # Fit kernels on training subgraph
        S_n = _fit_odd_neumann(B_tr, lam=lam, K=K, eps=eps)
        S_p = _fit_odd_path(B_tr, K=K, eps=eps)

        scores_n = np.array([S_n[d, a] for d, a in pairs])
        scores_p = np.array([S_p[d, a] for d, a in pairs])

        results[protocol] = {
            "odd_neumann": _evaluate(y, scores_n),
            "odd_path":    _evaluate(y, scores_p),
        }

    return results


# ════════════════════════════════════════════════════════════════════════════
# 2.  NEGATIVE SAMPLING ABLATION
# ════════════════════════════════════════════════════════════════════════════

def negative_sampling_ablation(B: np.ndarray,
                                K: int = 3,
                                lam: float = 0.1,
                                eps: float = 1e-3,
                                test_frac: float = 0.20,
                                seed: int = 42) -> Dict[str, Dict]:
    """
    Compare three negative-sampling strategies for the Odd-Neumann kernel
    under the standard random train/test edge split.

    Strategies
    ----------
    random_uniform      : negatives drawn uniformly from unobserved edges.
    degree_stratified   : negatives sampled proportional to degree product,
                          matching the marginal degree distribution of
                          positive edges.
    hard_negatives      : the highest-scoring unobserved pairs according to
                          the kernel fitted on the training graph.

    Parameters
    ----------
    B         : binary incidence matrix (n_drugs × n_adrs).
    K         : walk truncation depth.
    lam       : Odd-Neumann decay factor.
    eps       : ridge regularisation.
    test_frac : fraction of positive edges held out.
    seed      : random seed.

    Returns
    -------
    dict mapping strategy name → metrics dict (roc_auc, pr_auc, f1, P@K).
    """
    rng = np.random.default_rng(seed)
    pr, pc = np.where(B == 1)
    perm   = rng.permutation(len(pr))
    n_test = max(20, int(np.ceil(len(pr) * test_frac)))
    ti     = perm[:n_test]

    # Training graph
    B_tr = B.copy()
    B_tr[pr[ti], pc[ti]] = 0.0

    # Positive test pairs
    pos_test = list(zip(pr[ti].tolist(), pc[ti].tolist()))
    n_neg    = len(pos_test)

    # Fit kernel on training graph (shared across all strategies)
    S_n = _fit_odd_neumann(B_tr, lam=lam, K=K, eps=eps)

    strategies = {
        "random_uniform":    _sample_negatives_random(B_tr, n_neg, rng),
        "degree_stratified": _sample_negatives_degree_stratified(B_tr, n_neg, rng),
        "hard_negatives":    _sample_negatives_hard(S_n, B_tr, n_neg),
    }

    results = {}
    for name, neg_test in strategies.items():
        mn       = min(len(pos_test), len(neg_test))
        pairs    = pos_test[:mn] + neg_test[:mn]
        y        = np.array([1] * mn + [0] * mn)
        scores   = np.array([S_n[d, a] for d, a in pairs])
        results[name] = _evaluate(y, scores)

    return results


# ════════════════════════════════════════════════════════════════════════════
# 3.  AGGREGATION STEPS ABLATION
# ════════════════════════════════════════════════════════════════════════════

def aggregation_steps_ablation(B: np.ndarray,
                                k_values: Tuple[int, ...] = (1, 2, 3, 4),
                                lam: float = 0.1,
                                eps: float = 1e-3,
                                test_frac: float = 0.20,
                                seed: int = 42) -> Dict[int, Dict]:
    """
    Measure the effect of varying the walk truncation depth K on
    drug-pair kernel performance, using random uniform negative sampling
    throughout.

    Parameters
    ----------
    B        : binary incidence matrix (n_drugs × n_adrs).
    k_values : tuple of K values to evaluate.
    lam      : Odd-Neumann decay factor (held fixed).
    eps      : ridge regularisation.
    test_frac: fraction of positive edges held out for evaluation.
    seed     : random seed.

    Returns
    -------
    dict mapping K (int) → metrics dict (roc_auc, pr_auc, f1, P@K).
    Both Odd-Neumann and Odd-Path kernels are evaluated at each K.
    """
    rng = np.random.default_rng(seed)
    pr, pc = np.where(B == 1)
    perm   = rng.permutation(len(pr))
    n_test = max(20, int(np.ceil(len(pr) * test_frac)))
    ti     = perm[:n_test]

    B_tr     = B.copy()
    B_tr[pr[ti], pc[ti]] = 0.0
    pos_test = list(zip(pr[ti].tolist(), pc[ti].tolist()))
    neg_test = _sample_negatives_random(B_tr, len(pos_test), rng)
    mn       = min(len(pos_test), len(neg_test))
    pairs    = pos_test[:mn] + neg_test[:mn]
    y        = np.array([1] * mn + [0] * mn)

    results = {}
    for K in k_values:
        S_n = _fit_odd_neumann(B_tr, lam=lam, K=K, eps=eps)
        S_p = _fit_odd_path(B_tr, K=K, eps=eps)
        sn  = np.array([S_n[d, a] for d, a in pairs])
        sp  = np.array([S_p[d, a] for d, a in pairs])
        results[K] = {
            "odd_neumann": _evaluate(y, sn),
            "odd_path":    _evaluate(y, sp),
        }

    return results


# ════════════════════════════════════════════════════════════════════════════
# 4.  TOP POLYPHARMACY PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════

def polypharmacy_top_predictions(B: np.ndarray,
                                  drug_names: List[str],
                                  adr_names: List[str],
                                  K: int = 3,
                                  lam: float = 0.1,
                                  eps: float = 1e-3,
                                  top_k: int = 20,
                                  exclude_observed: bool = True
                                  ) -> List[Dict]:
    """
    Rank all unobserved (drug, ADR) pairs by their Odd-Neumann kernel score
    and return the top-k predictions.

    Useful for identifying high-confidence novel drug–ADR associations for
    downstream clinical or mechanistic interpretation.

    Parameters
    ----------
    B               : binary incidence matrix (n_drugs × n_adrs).
    drug_names      : list of drug name strings, length n_drugs.
    adr_names       : list of ADR name strings, length n_adrs.
    K               : walk truncation depth.
    lam             : Odd-Neumann decay factor.
    eps             : ridge regularisation.
    top_k           : number of top predictions to return.
    exclude_observed: if True, exclude known positive edges from ranking.

    Returns
    -------
    list of dicts, each with keys:
        rank, drug, adr, kernel_score
    sorted by kernel_score descending.
    """
    S_n = _fit_odd_neumann(B, lam=lam, K=K, eps=eps)

    if exclude_observed:
        S_n[B == 1] = -np.inf

    flat    = S_n.ravel()
    top_idx = np.argsort(-flat)[:top_k]
    rows, cols = np.unravel_index(top_idx, S_n.shape)

    predictions = []
    for rank, (r, c) in enumerate(zip(rows, cols), start=1):
        predictions.append({
            "rank":         rank,
            "drug":         drug_names[r],
            "adr":          adr_names[c],
            "kernel_score": float(S_n[r, c]),
        })

    return predictions


# ════════════════════════════════════════════════════════════════════════════
# 5.  RUNTIME BENCHMARK
# ════════════════════════════════════════════════════════════════════════════

def runtime_benchmark(B: np.ndarray,
                      K: int = 3,
                      lam: float = 0.1,
                      eps: float = 1e-3,
                      n_inference_pairs: int = 1000,
                      seed: int = 42) -> Dict[str, float]:
    """
    Measure wall-clock times for the main pipeline stages.

    Stages timed
    ------------
    kernel_construction_s  : time to build the full kernel score matrix
                             (Odd-Neumann and Odd-Path, separately).
    gram_assembly_s        : time to assemble the n×n Gram matrix from a
                             random 80/20 train split.
    svm_training_s         : time to train a precomputed-kernel SVM on the
                             assembled Gram matrix.
    inference_per_pair_ms  : mean time per pair over n_inference_pairs
                             random lookups, in milliseconds.

    Parameters
    ----------
    B                   : binary incidence matrix (n_drugs × n_adrs).
    K                   : walk truncation depth.
    lam                 : Odd-Neumann decay factor.
    eps                 : ridge regularisation.
    n_inference_pairs   : number of random pairs to time for inference.
    seed                : random seed.

    Returns
    -------
    dict with keys:
        neumann_construction_s, path_construction_s,
        gram_assembly_s, svm_training_s, inference_per_pair_ms
    """
    rng = np.random.default_rng(seed)
    m, n = B.shape

    # ── Kernel construction ───────────────────────────────────────────────
    t0 = time.perf_counter()
    S_n = _fit_odd_neumann(B, lam=lam, K=K, eps=eps)
    neumann_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    S_p = _fit_odd_path(B, K=K, eps=eps)
    path_time = time.perf_counter() - t0

    # ── Build train/test split for Gram + SVM timing ──────────────────────
    pr, pc = np.where(B == 1)
    perm   = rng.permutation(len(pr))
    n_test = max(10, int(np.ceil(len(pr) * 0.20)))
    ti     = perm[:n_test]
    B_tr   = B.copy()
    B_tr[pr[ti], pc[ti]] = 0.0

    pos_tr  = list(zip(pr[perm[n_test:]].tolist(), pc[perm[n_test:]].tolist()))
    neg_tr  = _sample_negatives_random(B_tr, len(pos_tr), rng)
    mn      = min(len(pos_tr), len(neg_tr))
    pairs   = pos_tr[:mn] + neg_tr[:mn]
    y_tr    = np.array([1] * mn + [0] * mn)

    # ── Gram matrix assembly ──────────────────────────────────────────────
    t0 = time.perf_counter()
    K_gram = np.array([[S_n[d, a] for d, a in pairs] for _ in [None]])
    K_gram = np.array([[S_n[d1, a1] * S_n[d2, a2]
                        for d2, a2 in pairs]
                       for d1, a1 in pairs])
    gram_time = time.perf_counter() - t0

    # ── SVM training ──────────────────────────────────────────────────────
    svm = SVC(kernel="precomputed", C=1.0, class_weight="balanced",
              probability=False)
    t0 = time.perf_counter()
    svm.fit(K_gram, y_tr)
    svm_time = time.perf_counter() - t0

    # ── Per-pair inference ────────────────────────────────────────────────
    rand_d = rng.integers(0, m, size=n_inference_pairs)
    rand_a = rng.integers(0, n, size=n_inference_pairs)

    t0 = time.perf_counter()
    for d, a in zip(rand_d, rand_a):
        _ = S_n[d, a]
    infer_ms = (time.perf_counter() - t0) / n_inference_pairs * 1000.0

    return {
        "neumann_construction_s": round(neumann_time, 2),
        "path_construction_s":    round(path_time, 2),
        "gram_assembly_s":        round(gram_time, 2),
        "svm_training_s":         round(svm_time, 2),
        "inference_per_pair_ms":  round(infer_ms, 4),
    }


# ════════════════════════════════════════════════════════════════════════════
# COMMAND-LINE RUNNER  (synthetic demo — replace B with your actual data)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("Extended Analysis — Demo on a synthetic bipartite graph")
    print("=" * 60)

    # ── Build a small synthetic graph for demonstration ───────────────────
    rng_demo = np.random.default_rng(99)
    N_D, N_A, N_E = 200, 400, 1200       # drugs, ADRs, edges
    idx  = rng_demo.choice(N_D * N_A, size=N_E, replace=False)
    r, c = np.unravel_index(idx, (N_D, N_A))
    B_demo = np.zeros((N_D, N_A), dtype=np.float32)
    B_demo[r, c] = 1.0

    drug_names = [f"Drug_{i}" for i in range(N_D)]
    adr_names  = [f"ADR_{j}"  for j in range(N_A)]

    # 1. OOD evaluation
    print("\n[1] OOD Evaluation")
    ood = ood_evaluation(B_demo, test_frac=0.20, K=3)
    for proto, res in ood.items():
        roc_n = res["odd_neumann"]["roc_auc"]
        roc_p = res["odd_path"]["roc_auc"]
        print(f"  {proto:<12}  Neumann ROC={roc_n:.4f}  Path ROC={roc_p:.4f}")

    # 2. Negative sampling ablation
    print("\n[2] Negative Sampling Ablation (Odd-Neumann, K=3)")
    ns_abl = negative_sampling_ablation(B_demo, K=3)
    for strat, m2 in ns_abl.items():
        print(f"  {strat:<22}  ROC={m2['roc_auc']:.4f}  PR={m2['pr_auc']:.4f}  F1={m2['f1']:.4f}")

    # 3. Aggregation steps ablation
    print("\n[3] Aggregation Steps Ablation")
    k_abl = aggregation_steps_ablation(B_demo, k_values=(1, 2, 3, 4))
    for K_val, res in k_abl.items():
        roc_n = res["odd_neumann"]["roc_auc"]
        roc_p = res["odd_path"]["roc_auc"]
        print(f"  K={K_val}  Neumann ROC={roc_n:.4f}  Path ROC={roc_p:.4f}")

    # 4. Top predictions
    print("\n[4] Top-5 Polypharmacy Predictions (Odd-Neumann, K=3)")
    preds = polypharmacy_top_predictions(B_demo, drug_names, adr_names,
                                         K=3, top_k=5)
    for p in preds:
        print(f"  #{p['rank']}  {p['drug']} → {p['adr']}  "
              f"score={p['kernel_score']:.6f}")

    # 5. Runtime benchmark
    print("\n[5] Runtime Benchmark")
    rt = runtime_benchmark(B_demo, K=3)
    for k2, v in rt.items():
        print(f"  {k2:<30}  {v}")
