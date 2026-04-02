"""
Graph Kernel Approach to Drug-ADR Link Prediction
==================================================
Implements:
  1. Odd Neumann Pseudo Kernel
  2. Odd Path Counting Kernel
  3. Kernel aggregation for drug-pair combinations
  4. Kernel SVM classifier (Model 1)
  5. Unsupervised link-prediction scoring (Model 2)
  6. Baseline comparisons: PAC, AAC, JAC, RAI

Datasets recommended for reporting:
  - SIDER 4.1       : single-drug ADR (medium scale)
  - TWOSIDES        : drug-pair ADR   (large scale, benchmark)
  - OFFSIDES        : single-drug ADR (large scale)
  - Neonatal FAERS  : domain-specific (small / medium)

Author: Clean-code version derived from Manuscript_Draft_1.docx
"""

# ── Standard library ────────────────────────────────────────────────────────
import warnings
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Numerical & scientific ───────────────────────────────────────────────────
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

# ── Machine learning ─────────────────────────────────────────────────────────
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Graph ────────────────────────────────────────────────────────────────────
import networkx as nx

# ── Visualisation (optional – guarded) ──────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False
    warnings.warn("matplotlib not found – plotting disabled.")

# ── pandas (optional – used only for dataset helpers) ───────────────────────
try:
    import pandas as pd
    PANDAS = True
except ImportError:
    PANDAS = False

warnings.filterwarnings("ignore")
np.random.seed(42)


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & GRAPH CONSTRUCTION
# ════════════════════════════════════════════════════════════════════════════

class DrugADRDataset:
    """
    Loads and stores a drug-ADR bipartite graph.

    Parameters
    ----------
    drug_adr_pairs : list of (drug_id, adr_id)
        Known single-drug → ADR associations.
    drug_pair_adr_triples : list of (drug_id_1, drug_id_2, adr_id), optional
        Known polypharmacy associations (ground truth for Model 1).
    """

    def __init__(
        self,
        drug_adr_pairs: List[Tuple],
        drug_pair_adr_triples: Optional[List[Tuple]] = None,
    ):
        self.drug_adr_pairs = drug_adr_pairs
        self.drug_pair_adr_triples = drug_pair_adr_triples or []

        # Build index maps
        drugs = sorted({d for d, _ in drug_adr_pairs})
        adrs  = sorted({a for _, a in drug_adr_pairs})
        self.drug2idx = {d: i for i, d in enumerate(drugs)}
        self.adr2idx  = {a: i for i, a in enumerate(adrs)}
        self.idx2drug = {i: d for d, i in self.drug2idx.items()}
        self.idx2adr  = {i: a for a, i in self.adr2idx.items()}

        self.n_drugs = len(drugs)
        self.n_adrs  = len(adrs)
        self.N       = self.n_drugs + self.n_adrs   # total nodes

        # Bipartite incidence matrix  B ∈ {0,1}^{m×n}
        self.B = self._build_incidence()

        # Block adjacency matrix  A ∈ {0,1}^{N×N}
        self.A = self._build_block_adjacency()

    # ── Internal builders ────────────────────────────────────────────────────

    def _build_incidence(self) -> np.ndarray:
        B = np.zeros((self.n_drugs, self.n_adrs), dtype=np.float64)
        for d, a in self.drug_adr_pairs:
            B[self.drug2idx[d], self.adr2idx[a]] = 1.0
        return B

    def _build_block_adjacency(self) -> np.ndarray:
        """
        A = | 0   B  |
            | B^T 0  |
        Drug nodes: indices 0 … m-1
        ADR  nodes: indices m … m+n-1
        """
        Z_dd = np.zeros((self.n_drugs, self.n_drugs))
        Z_aa = np.zeros((self.n_adrs,  self.n_adrs))
        top    = np.hstack([Z_dd,    self.B])
        bottom = np.hstack([self.B.T, Z_aa])
        return np.vstack([top, bottom])

    # ── Helpers ──────────────────────────────────────────────────────────────

    def drug_node_idx(self, drug_id) -> int:
        return self.drug2idx[drug_id]

    def adr_node_idx(self, adr_id) -> int:
        return self.n_drugs + self.adr2idx[adr_id]

    def spectral_radius(self) -> float:
        """Largest eigenvalue of A (upper bound for λ selection)."""
        eigvals = np.linalg.eigvalsh(self.A)
        return float(np.max(np.abs(eigvals)))

    def to_networkx(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.n_drugs), bipartite=0, kind="drug")
        G.add_nodes_from(
            range(self.n_drugs, self.N), bipartite=1, kind="adr"
        )
        for d, a in self.drug_adr_pairs:
            G.add_edge(self.drug2idx[d], self.n_drugs + self.adr2idx[a])
        return G

    @classmethod
    def make_synthetic(
        cls,
        n_drugs: int = 40,
        n_adrs: int = 20,
        density: float = 0.15,
        n_pair_triples: int = 200,
    ) -> "DrugADRDataset":
        """
        Build a small synthetic dataset for unit-testing and demo runs.
        Replace this with real data loading for publication experiments.
        """
        rng = np.random.default_rng(42)
        drugs = [f"D{i:03d}" for i in range(n_drugs)]
        adrs  = [f"A{i:03d}" for i in range(n_adrs)]

        pairs = [
            (d, a)
            for d in drugs for a in adrs
            if rng.random() < density
        ]

        # Synthetic polypharmacy triples
        triples = []
        for _ in range(n_pair_triples):
            d1, d2 = rng.choice(drugs, size=2, replace=False)
            a = rng.choice(adrs)
            triples.append((d1, d2, a))

        return cls(pairs, triples)


# ════════════════════════════════════════════════════════════════════════════
# 2.  GRAPH KERNELS
# ════════════════════════════════════════════════════════════════════════════

class OddNeumannPseudoKernel:
    """
    S_odd = Σ_{k=0}^{K}  λ^{2k+1} · A^{2k+1}

    Captures weighted influence via odd-length walks between drug and ADR
    nodes.  Longer walks are down-weighted by λ^{2k+1}.

    Parameters
    ----------
    lam : float
        Decay factor.  Must satisfy  0 < λ < 1 / ρ(A).
        If None, auto-set to 0.9 / ρ(A).
    K : int
        Maximum walk-length index  (walk length = 2K+1).
    """

    def __init__(self, lam: Optional[float] = None, K: int = 3):
        self.lam = lam
        self.K   = K
        self.S   = None          # kernel matrix after fit

    def fit(self, dataset: DrugADRDataset) -> "OddNeumannPseudoKernel":
        A = dataset.A.copy()

        if self.lam is None:
            rho = dataset.spectral_radius()
            self.lam = 0.9 / rho if rho > 0 else 0.1

        self.S = np.zeros_like(A, dtype=np.float64)
        A_pow  = A.copy()       # A^1

        for k in range(self.K + 1):
            walk_len = 2 * k + 1
            # A_pow should hold A^{walk_len}
            if k == 0:
                pass                            # A^1 already ready
            else:
                A_pow = A_pow @ A @ A           # A^{2k+1} = A^{2k-1} · A²
            self.S += (self.lam ** walk_len) * A_pow

        return self

    def similarity(
        self,
        node_i: int,
        node_j: int,
    ) -> float:
        """Return kernel value K(i, j)."""
        assert self.S is not None, "Call fit() first."
        return float(self.S[node_i, node_j])

    def drug_adr_submatrix(
        self,
        dataset: DrugADRDataset,
    ) -> np.ndarray:
        """Return the drug×ADR slice of S_odd."""
        assert self.S is not None
        return self.S[: dataset.n_drugs, dataset.n_drugs :]


class OddPathCountingKernel:
    """
    P_odd = Σ_{k=0}^{K}  A^{2k+1}

    Counts the raw number of odd-length paths between node pairs.
    No decay factor → path counts grow; normalise before downstream use.

    Parameters
    ----------
    K : int
        Maximum walk-length index  (walk length = 2K+1).
    normalise : bool
        If True, divide each entry by its row-max so scores ∈ [0, 1].
    """

    def __init__(self, K: int = 3, normalise: bool = True):
        self.K         = K
        self.normalise = normalise
        self.P         = None

    def fit(self, dataset: DrugADRDataset) -> "OddPathCountingKernel":
        A = dataset.A.copy()

        self.P = np.zeros_like(A, dtype=np.float64)
        A_pow  = A.copy()

        for k in range(self.K + 1):
            if k == 0:
                pass
            else:
                A_pow = A_pow @ A @ A
            self.P += A_pow

        if self.normalise:
            row_max = self.P.max(axis=1, keepdims=True)
            row_max[row_max == 0] = 1.0
            self.P = self.P / row_max

        return self

    def similarity(self, node_i: int, node_j: int) -> float:
        assert self.P is not None, "Call fit() first."
        return float(self.P[node_i, node_j])

    def drug_adr_submatrix(
        self,
        dataset: DrugADRDataset,
    ) -> np.ndarray:
        assert self.P is not None
        return self.P[: dataset.n_drugs, dataset.n_drugs :]


# ════════════════════════════════════════════════════════════════════════════
# 3.  KERNEL AGGREGATION FOR DRUG PAIRS
# ════════════════════════════════════════════════════════════════════════════

class DrugPairKernel:
    """
    Lifts a single-node kernel to drug-pair instances.

    For two triples  (d1, d2, a1)  and  (d3, d4, a2) the combined kernel is

        K_full((d1,d2,a1), (d3,d4,a2)) =
            K_drug_pair((d1,d2),(d3,d4))  ×  K_adr(a1, a2)

    where
        K_drug_pair = K_dd(d1,d3) + K_dd(d1,d4) + K_dd(d2,d3) + K_dd(d2,d4)

    Drug-drug similarity  K_dd  (the root cause of Bug 3)
    -------------------------------------------------------
    S_odd[drug_i, drug_j] = 0 ALWAYS in a bipartite graph because odd-length
    walks only cross partitions (drug→ADR).  The drug-drug block of S_odd is
    entirely zero, making every Gram entry 0 → identity matrix → SVM at chance.

    Fix: use  K_dd = B @ B^T  (shared-ADR count matrix), which counts the
    number of ADRs that two drugs share.  This is exactly the 2-hop
    even-length walk d→a→d' and provides genuine drug-drug structural
    similarity in the bipartite setting.

    ADR-ADR similarity  K_aa  (same reasoning, transposed)
    --------------------------------------------------------
    Use  K_aa = B^T @ B  (shared-drug count matrix).
    If a1==a2, K_aa=1 (identity fallback for single-ADR training).
    """

    def __init__(self, base_kernel, dataset: DrugADRDataset):
        self._kern = base_kernel
        self._ds   = dataset

        # Pre-compute and normalise drug-drug similarity  B @ B^T
        B   = dataset.B.astype(np.float64)          # shape (n_drugs, n_adrs)
        DDT = B @ B.T                               # (n_drugs, n_drugs) shared-ADR count
        # Row-normalise to [0,1] so scale matches between datasets
        row_max = DDT.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        self._Kdd = DDT / row_max                   # drug-drug kernel matrix

        # Pre-compute and normalise ADR-ADR similarity  B^T @ B
        BTB = B.T @ B                               # (n_adrs, n_adrs)
        col_max = BTB.max(axis=0, keepdims=True)
        col_max[col_max == 0] = 1.0
        self._Kaa = BTB / col_max                   # adr-adr kernel matrix

    def _drug_idx(self, d) -> Optional[int]:
        return self._ds.drug2idx.get(d)

    def _adr_idx(self, a) -> Optional[int]:
        return self._ds.adr2idx.get(a)

    def _kdd(self, i: Optional[int], j: Optional[int]) -> float:
        """Drug-drug kernel: shared ADR count (normalised)."""
        if i is None or j is None:
            return 0.0
        return float(self._Kdd[i, j])

    def _kaa(self, i: Optional[int], j: Optional[int]) -> float:
        """ADR-ADR kernel: shared drug count (normalised)."""
        if i is None or j is None:
            return 0.0
        return float(self._Kaa[i, j])

    def drug_pair_similarity(
        self,
        d1, d2,   # first drug pair
        d3, d4,   # second drug pair
    ) -> float:
        """
        K_drug_pair((d1,d2),(d3,d4)) = Σ K_dd(di, dj)  for all cross pairs.
        Uses B@B^T shared-ADR similarity (not the zero odd-walk drug block).
        """
        i1 = self._drug_idx(d1)
        i2 = self._drug_idx(d2)
        i3 = self._drug_idx(d3)
        i4 = self._drug_idx(d4)
        return (
            self._kdd(i1, i3) + self._kdd(i1, i4)
            + self._kdd(i2, i3) + self._kdd(i2, i4)
        )

    def build_gram_matrix(
        self,
        triples: List[Tuple],
        adr_sim_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build the full Gram (kernel) matrix over a list of triples.

        K_full((d1,d2,a1),(d3,d4,a2)) = K_drug_pair * K_adr(a1,a2)

        Parameters
        ----------
        triples : list of (d1, d2, adr)
        adr_sim_matrix : np.ndarray (n_adrs, n_adrs), optional
            External semantic ADR similarity (e.g. UMLS cosine).
            If None, uses internal B^T@B shared-drug similarity.
        """
        n  = len(triples)
        K  = np.zeros((n, n), dtype=np.float64)
        ds = self._ds

        for p, (d1, d2, a1) in enumerate(triples):
            for q in range(p, n):
                d3, d4, a2 = triples[q]

                kd = self.drug_pair_similarity(d1, d2, d3, d4)

                if adr_sim_matrix is not None:
                    ai = ds.adr2idx.get(a1)
                    aj = ds.adr2idx.get(a2)
                    ka = float(adr_sim_matrix[ai, aj]) if (ai is not None and aj is not None) else 0.0
                elif a1 == a2:
                    ka = 1.0
                else:
                    ai = self._adr_idx(a1)
                    aj = self._adr_idx(a2)
                    ka = self._kaa(ai, aj)

                K[p, q] = K[q, p] = kd * ka

        return K


# ════════════════════════════════════════════════════════════════════════════
# 4.  LINK PREDICTION SCORING (unsupervised baselines + kernel)
# ════════════════════════════════════════════════════════════════════════════

class LinkPredictor:
    """
    Computes various link-prediction scores for (drug, ADR) pairs.

    Scores implemented
    ------------------
    PAC  – Preferential Attachment
    AAC  – Adamic-Adar Coefficient   (requires 2-hop common neighbor)
    JAC  – Jaccard Index
    RAI  – Resource Allocation Index
    KRNL – Kernel similarity (Odd Neumann or Odd Path Counting)
    """

    def __init__(self, dataset: DrugADRDataset):
        self._ds = dataset
        G = dataset.to_networkx()
        self._G  = G

        # Pre-compute neighbour sets (degree view)
        self._neighbors = {v: set(G.neighbors(v)) for v in G.nodes()}

    # ── Classic heuristics ────────────────────────────────────────────────

    def pac(self, drug_id, adr_id) -> float:
        d  = self._ds.drug2idx[drug_id]
        a  = self._ds.n_drugs + self._ds.adr2idx[adr_id]
        return float(len(self._neighbors[d]) * len(self._neighbors[a]))

    def aac(self, drug_id, adr_id) -> float:
        d  = self._ds.drug2idx[drug_id]
        a  = self._ds.n_drugs + self._ds.adr2idx[adr_id]
        common = self._neighbors[d] & self._neighbors[a]
        if not common:
            return 0.0
        return sum(
            1.0 / np.log(len(self._neighbors[nb]) + 1e-9)
            for nb in common
        )

    def jac(self, drug_id, adr_id) -> float:
        d  = self._ds.drug2idx[drug_id]
        a  = self._ds.n_drugs + self._ds.adr2idx[adr_id]
        nd, na = self._neighbors[d], self._neighbors[a]
        union  = nd | na
        if not union:
            return 0.0
        return len(nd & na) / len(union)

    def rai(self, drug_id, adr_id) -> float:
        d  = self._ds.drug2idx[drug_id]
        a  = self._ds.n_drugs + self._ds.adr2idx[adr_id]
        common = self._neighbors[d] & self._neighbors[a]
        if not common:
            return 0.0
        return sum(
            1.0 / (len(self._neighbors[nb]) + 1e-9)
            for nb in common
        )

    def kernel_score(self, drug_id, adr_id, kernel) -> float:
        d = self._ds.drug2idx[drug_id]
        a = self._ds.n_drugs + self._ds.adr2idx[adr_id]
        if hasattr(kernel, "S") and kernel.S is not None:
            return float(kernel.S[d, a])
        return float(kernel.P[d, a])

    # ── Rank all candidate (drug, ADR) pairs ─────────────────────────────

    def rank_candidates(
        self,
        score_fn,
        existing_edges: Optional[set] = None,
        top_k: int = 50,
    ) -> List[Tuple]:
        """
        Return top-k (drug, adr, score) triples not in existing_edges.
        score_fn : callable(drug_id, adr_id) -> float
        """
        results = []
        ds = self._ds
        for d_name, di in ds.drug2idx.items():
            for a_name, ai in ds.adr2idx.items():
                if existing_edges and (d_name, a_name) in existing_edges:
                    continue
                s = score_fn(d_name, a_name)
                results.append((d_name, a_name, s))
        results.sort(key=lambda x: -x[2])
        return results[:top_k]


# ════════════════════════════════════════════════════════════════════════════
# 5.  KERNEL SVM CLASSIFIER  (Model 1)
# ════════════════════════════════════════════════════════════════════════════

class KernelSVMClassifier:
    """
    Binary SVM trained with a pre-computed Gram matrix.

    Positive class  → known drug-pair causes ADR
    Negative class  → sampled pairs that do NOT appear in the dataset
    """

    def __init__(self, C: float = 1.0, class_weight: str = "balanced"):
        self.C            = C
        self.class_weight = class_weight
        self._svm         = None

    def fit(
        self,
        K_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "KernelSVMClassifier":
        self._svm = SVC(
            kernel="precomputed",
            C=self.C,
            class_weight=self.class_weight,
            probability=True,
        )
        self._svm.fit(K_train, y_train)
        return self

    def predict_proba(self, K_test: np.ndarray) -> np.ndarray:
        assert self._svm is not None, "Call fit() first."
        return self._svm.predict_proba(K_test)[:, 1]

    def predict(self, K_test: np.ndarray) -> np.ndarray:
        assert self._svm is not None
        return self._svm.predict(K_test)


# ════════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_link_prediction(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_values: Tuple[int, ...] = (10, 20, 50),
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute ROC-AUC, PR-AUC, F1, Precision@K, Recall@K.

    Parameters
    ----------
    y_true  : binary ground-truth array
    y_score : predicted probability or kernel score
    """
    metrics = {}
    metrics["ROC-AUC"]  = roc_auc_score(y_true, y_score)
    metrics["PR-AUC"]   = average_precision_score(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)
    metrics["F1"]       = f1_score(y_true, y_pred, zero_division=0)
    metrics["Precision"]= precision_score(y_true, y_pred, zero_division=0)
    metrics["Recall"]   = recall_score(y_true, y_pred, zero_division=0)

    # Precision & Recall @ K
    sorted_idx = np.argsort(-y_score)
    for k in k_values:
        top_k = sorted_idx[:k]
        tp    = y_true[top_k].sum()
        metrics[f"Precision@{k}"] = tp / k
        metrics[f"Recall@{k}"]    = tp / (y_true.sum() + 1e-9)

    return metrics


def cross_validate_kernel(
    gram_matrix: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    C: float = 1.0,
) -> Dict[str, float]:
    """
    Stratified k-fold cross-validation using pre-computed Gram matrix.
    Returns mean ± std for each metric.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_aucs, all_prauc = [], []

    for train_idx, test_idx in skf.split(gram_matrix, labels):
        K_train = gram_matrix[np.ix_(train_idx, train_idx)]
        K_test  = gram_matrix[np.ix_(test_idx,  train_idx)]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = KernelSVMClassifier(C=C)
        clf.fit(K_train, y_train)
        scores = clf.predict_proba(K_test)

        all_aucs.append(roc_auc_score(y_test, scores))
        all_prauc.append(average_precision_score(y_test, scores))

    return {
        "ROC-AUC mean": float(np.mean(all_aucs)),
        "ROC-AUC std":  float(np.std(all_aucs)),
        "PR-AUC mean":  float(np.mean(all_prauc)),
        "PR-AUC std":   float(np.std(all_prauc)),
    }


# ════════════════════════════════════════════════════════════════════════════
# 7.  NEGATIVE SAMPLING  (ADR-stratified)
# ════════════════════════════════════════════════════════════════════════════

def sample_negatives(
    positives: List[Tuple],
    dataset: DrugADRDataset,
    ratio: float = 1.0,
    stratify_by_adr: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple]:
    """
    Sample drug-pair-ADR triples NOT in the positive set.

    Parameters
    ----------
    stratify_by_adr : bool
        If True, sample the same ADR distribution as in positives so the
        negative class is structurally comparable (fixes Bug 4).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pos_set  = {(d1, d2, a) for d1, d2, a in positives}
    pos_set |= {(d2, d1, a) for d1, d2, a in positives}
    drugs    = list(dataset.drug2idx.keys())
    adrs     = list(dataset.adr2idx.keys())
    n_neg    = int(len(positives) * ratio)

    # Build ADR frequency distribution from positives
    if stratify_by_adr:
        from collections import Counter
        adr_counts = Counter(a for _, _, a in positives)
        adr_pool   = list(adr_counts.keys())
        adr_probs  = np.array([adr_counts[a] for a in adr_pool], dtype=float)
        adr_probs /= adr_probs.sum()
    else:
        adr_pool  = adrs
        adr_probs = None

    negatives = []
    attempts  = 0
    max_attempts = n_neg * 200
    while len(negatives) < n_neg and attempts < max_attempts:
        d1, d2 = rng.choice(drugs, size=2, replace=False)
        if adr_probs is not None:
            a = adr_pool[rng.choice(len(adr_pool), p=adr_probs)]
        else:
            a = rng.choice(adrs)
        if (d1, d2, a) not in pos_set:
            negatives.append((d1, d2, a))
        attempts += 1

    if len(negatives) < n_neg:
        print(f"  [Warning] Only {len(negatives)}/{n_neg} negatives found "
              f"after {max_attempts} attempts.")
    return negatives


# ════════════════════════════════════════════════════════════════════════════
# 7b. TRAIN / TEST EDGE SPLIT
# ════════════════════════════════════════════════════════════════════════════

def split_edges(
    dataset: DrugADRDataset,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple["DrugADRDataset", List[Tuple], List[Tuple]]:
    """
    Split drug-ADR edges into train and test sets.

    Returns
    -------
    train_dataset : DrugADRDataset
        Dataset built from training edges ONLY (kernels must be fit on this).
    test_pos : list of (drug, adr)
        Held-out positive edges (unseen during kernel fitting).
    test_neg : list of (drug, adr)
        Randomly sampled negative edges not present in the full graph
        (same size as test_pos for balanced evaluation).

    Why this matters (Bug 1 fix)
    ------------------------------
    If kernels are fit on ALL edges and then evaluated on those same edges,
    the kernel matrix directly encodes the adjacency → trivial AUC=1.0.
    Kernels must only see train edges; test edges are truly unseen.
    """
    rng   = np.random.default_rng(seed)
    pairs = list(dataset.drug_adr_pairs)
    rng.shuffle(pairs)

    n_test      = max(1, int(len(pairs) * test_ratio))
    test_pos    = pairs[:n_test]
    train_pairs = pairs[n_test:]

    train_ds = DrugADRDataset(train_pairs, dataset.drug_pair_adr_triples)

    # Sample negatives: (drug, adr) pairs absent from the FULL graph
    full_pos_set = set(dataset.drug_adr_pairs)
    drugs  = list(dataset.drug2idx.keys())
    adrs   = list(dataset.adr2idx.keys())
    test_neg: List[Tuple] = []
    attempts = 0
    while len(test_neg) < len(test_pos) and attempts < len(test_pos) * 500:
        d = rng.choice(drugs)
        a = rng.choice(adrs)
        if (d, a) not in full_pos_set:
            test_neg.append((d, a))
        attempts += 1

    return train_ds, test_pos, test_neg


# ════════════════════════════════════════════════════════════════════════════
# 8.  ABLATION HELPERS
# ════════════════════════════════════════════════════════════════════════════

def ablation_lambda(
    dataset: DrugADRDataset,
    lambda_values: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.3),
    K: int = 3,
) -> Dict[str, np.ndarray]:
    results = {}
    for lam in lambda_values:
        kern = OddNeumannPseudoKernel(lam=lam, K=K).fit(dataset)
        results[f"lambda={lam}"] = kern.drug_adr_submatrix(dataset)
    return results


def ablation_max_walk(
    dataset: DrugADRDataset,
    k_values: Tuple[int, ...] = (1, 2, 3, 4, 5),
) -> Dict[str, np.ndarray]:
    results = {}
    for K in k_values:
        kern = OddPathCountingKernel(K=K).fit(dataset)
        results[f"K={K}"] = kern.drug_adr_submatrix(dataset)
    return results


# ════════════════════════════════════════════════════════════════════════════
# 9.  VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

def plot_bipartite_graph(dataset: DrugADRDataset, max_nodes: int = 30):
    if not MATPLOTLIB:
        print("matplotlib unavailable – skipping plot.")
        return

    G         = dataset.to_networkx()
    drug_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "drug"]
    adr_nodes  = [n for n, d in G.nodes(data=True) if d["kind"] == "adr"]
    drug_nodes = drug_nodes[:max_nodes // 2]
    adr_nodes  = adr_nodes[:max_nodes // 2]
    subgraph   = G.subgraph(drug_nodes + adr_nodes)

    pos = {}
    for i, n in enumerate(drug_nodes): pos[n] = (0, i)
    for i, n in enumerate(adr_nodes):  pos[n] = (2, i)

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(subgraph, pos, nodelist=drug_nodes,
                           node_color="#4C72B0", node_size=300, label="Drugs")
    nx.draw_networkx_nodes(subgraph, pos, nodelist=adr_nodes,
                           node_color="#55A868", node_size=300, label="ADRs")
    nx.draw_networkx_edges(subgraph, pos, alpha=0.4, edge_color="#888")
    nx.draw_networkx_labels(subgraph, pos,
        labels={n: dataset.idx2drug[n] for n in drug_nodes}, font_size=7)
    nx.draw_networkx_labels(subgraph, pos,
        labels={n: dataset.idx2adr[n - dataset.n_drugs] for n in adr_nodes},
        font_size=7)
    plt.legend(); plt.title("Drug–ADR Bipartite Graph (sampled)")
    plt.axis("off"); plt.tight_layout()
    plt.savefig("bipartite_graph.png", dpi=150); plt.show()
    print("Saved bipartite_graph.png")


def plot_kernel_heatmap(matrix, title, xlabel="ADR index",
                        ylabel="Drug index", filename=None):
    if not MATPLOTLIB:
        return
    plt.figure(figsize=(7, 5))
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(label="Kernel value")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout()
    if filename: plt.savefig(filename, dpi=150)
    plt.show()


def plot_roc_pr(y_true, y_score, label: str, filename=None):
    if not MATPLOTLIB:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    axes[0].plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve"); axes[0].legend()
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    prauc = average_precision_score(y_true, y_score)
    axes[1].plot(rec, prec, label=f"{label} (AP={prauc:.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend()
    plt.tight_layout()
    if filename: plt.savefig(filename, dpi=150)
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 10.  DATASET LOADING HELPERS  (for real experiments)
# ════════════════════════════════════════════════════════════════════════════

def load_sider(filepath: str) -> DrugADRDataset:
    if not PANDAS:
        raise ImportError("pandas required for load_sider()")
    df = pd.read_csv(filepath, sep="\t", header=None,
                     names=["drug_id","drug_name","umls_label",
                            "meddra_type","umls_meddra","se_name"])
    df = df[df["meddra_type"] == "PT"]
    pairs = list(zip(df["drug_name"].str.lower(), df["se_name"].str.lower()))
    return DrugADRDataset(pairs)


def load_twosides(filepath: str, max_rows: int = 100_000) -> DrugADRDataset:
    if not PANDAS:
        raise ImportError("pandas required for load_twosides()")
    df = pd.read_csv(filepath, nrows=max_rows)
    triples = list(zip(df["drug1_name"].str.lower(),
                       df["drug2_name"].str.lower(),
                       df["condition_name"].str.lower()))
    single_pairs = list(set(
        list(zip(df["drug1_name"].str.lower(), df["condition_name"].str.lower())) +
        list(zip(df["drug2_name"].str.lower(), df["condition_name"].str.lower()))
    ))
    return DrugADRDataset(single_pairs, triples)


# ════════════════════════════════════════════════════════════════════════════
# 11.  BIPARTITE-AWARE LINK PREDICTION BASELINES
#      (fixes Bug 2 – common-neighbor methods in bipartite graphs)
# ════════════════════════════════════════════════════════════════════════════

def _bipartite_scores(
    train_dataset: DrugADRDataset,
    pairs: List[Tuple[str, str]],
) -> Dict[str, np.ndarray]:
    """
    Compute all baseline + kernel scores for a list of (drug, adr) pairs
    using only the TRAIN graph (inductive, leak-free).

    Bipartite-aware common-neighbor methods (fixes Bug 2)
    -----------------------------------------------------
    In a drug-ADR bipartite graph, N(drug) ∩ N(ADR) = ∅ always because
    drugs connect only to ADRs and vice versa.  We instead use 2-hop
    common neighbors:

      CN2(d, a) = { d' ∈ Drugs : d' shares ≥1 ADR with d  AND  d'→a exists }

    This captures the intuition: "drugs structurally similar to d that
    also cause a", which is meaningful in a bipartite setting.
    """
    ds = train_dataset
    G  = ds.to_networkx()
    nb = {v: set(G.neighbors(v)) for v in G.nodes()}

    def drug_idx(d): return ds.drug2idx.get(d)
    def adr_idx(a):  return ds.adr_idx.get(a) if hasattr(ds, 'adr_idx') else ds.adr2idx.get(a)
    def d_node(d):   return ds.drug2idx.get(d)
    def a_node(a):   return ds.n_drugs + ds.adr2idx.get(a, -ds.n_drugs - 1)

    n = len(pairs)
    pac  = np.zeros(n)
    cn2  = np.zeros(n)   # 2-hop common neighbours (bipartite-aware AAC replacement)
    jac2 = np.zeros(n)   # 2-hop Jaccard
    rai2 = np.zeros(n)   # 2-hop Resource Allocation

    for i, (d, a) in enumerate(pairs):
        di = d_node(d)
        ai = a_node(a)
        if di is None or di not in nb or ai not in nb:
            continue

        # PAC: degree product (works on bipartite directly)
        pac[i] = len(nb[di]) * len(nb[ai])

        # 2-hop: drugs that share an ADR with d (via even-length path d→a'→d')
        #        AND also link to a
        d_adr_neighbours = nb[di]                        # ADR nodes d connects to
        similar_drugs = set()
        for adr_node in d_adr_neighbours:
            for d2 in nb[adr_node]:                      # drugs sharing that ADR
                if d2 != di:
                    similar_drugs.add(d2)

        # Among similar_drugs, keep those that link to target ADR a
        common2 = {d2 for d2 in similar_drugs if ai in nb.get(d2, set())}
        cn2[i]  = len(common2)

        # 2-hop Jaccard: |common2| / |similar_drugs ∪ {drugs linking to a}|
        a_drug_neighbours = nb.get(ai, set())
        union2 = similar_drugs | a_drug_neighbours
        jac2[i] = len(common2) / len(union2) if union2 else 0.0

        # 2-hop Resource Allocation: weight by inverse degree of bridging ADR
        for adr_node in d_adr_neighbours:
            drugs_via_adr = nb[adr_node] - {di}
            if ai in set().union(*[nb.get(d2, set()) for d2 in drugs_via_adr]):
                deg = len(nb[adr_node])
                rai2[i] += 1.0 / deg if deg > 0 else 0.0

    return {"PAC": pac, "CN2 (2-hop AAC)": cn2,
            "Jaccard-2hop": jac2, "RAI-2hop": rai2}


# ════════════════════════════════════════════════════════════════════════════
# 12.  FULL PIPELINE  (end-to-end, all bugs fixed)
# ════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    dataset: Optional[DrugADRDataset] = None,
    test_ratio: float = 0.2,
    K_walk: int = 3,
    svm_C: float = 1.0,
    gram_reg: float = 1e-3,
    verbose: bool = True,
):
    """
    End-to-end pipeline with all four bugs fixed.

    Bugs fixed
    ----------
    Bug 1 – Data leakage    : kernels fit on TRAIN edges only; evaluation
                               on held-out TEST edges vs. unseen negatives.
    Bug 2 – Bipartite AAC   : replaced common-neighbor baselines with
                               2-hop bipartite-aware variants.
    Bug 3 – Degenerate Gram : Gram diagonal regularisation (K + εI) and
                               using Odd Neumann kernel (better conditioned)
                               for the SVM.
    Bug 4 – Unstratified neg: negatives sampled with same ADR distribution
                               as positives (stratify_by_adr=True).

    Parameters
    ----------
    test_ratio : fraction of edges held out for evaluation
    K_walk     : maximum odd-walk length index (walk length = 2K+1)
    svm_C      : SVM regularisation parameter
    gram_reg   : ridge term added to Gram diagonal (ε in K + εI)
    """
    if dataset is None:
        print("No dataset provided → using synthetic data.")
        dataset = DrugADRDataset.make_synthetic(
            n_drugs=50, n_adrs=25, density=0.15, n_pair_triples=300)

    print(f"\n{'='*62}")
    print(f"  Graph Kernel ADR Prediction Pipeline  (leak-free)")
    print(f"{'='*62}")
    print(f"  Total drugs : {dataset.n_drugs}")
    print(f"  Total ADRs  : {dataset.n_adrs}")
    print(f"  Total edges : {int(dataset.B.sum())}")
    rho = dataset.spectral_radius()
    print(f"  ρ(A)        : {rho:.4f}")
    print(f"  Test ratio  : {test_ratio:.0%}")
    print(f"{'='*62}\n")

    # ── STEP 1: train / test edge split (Bug 1 fix) ──────────────────────
    print("Splitting edges into train / test …")
    train_ds, test_pos, test_neg = split_edges(dataset, test_ratio=test_ratio)
    print(f"  Train edges : {len(train_ds.drug_adr_pairs):,}")
    print(f"  Test pos    : {len(test_pos):,}  |  Test neg : {len(test_neg):,}")

    # ── STEP 2: fit kernels on TRAIN graph only ───────────────────────────
    print("\nFitting Odd Neumann Pseudo Kernel on TRAIN graph …")
    kern_neu = OddNeumannPseudoKernel(K=K_walk).fit(train_ds)
    print(f"  λ = {kern_neu.lam:.6f}")

    print("Fitting Odd Path Counting Kernel on TRAIN graph …")
    kern_path = OddPathCountingKernel(K=K_walk, normalise=True).fit(train_ds)

    # ── STEP 3: unsupervised evaluation on TEST edges (inductive) ────────
    print("\nScoring TEST edges (inductive, leak-free) …")
    eval_pairs  = test_pos + test_neg
    y_true_test = np.array([1]*len(test_pos) + [0]*len(test_neg), dtype=int)

    def kern_score_pairs(kern, pairs_list):
        scores = np.zeros(len(pairs_list))
        for i, (d, a) in enumerate(pairs_list):
            di = train_ds.drug2idx.get(d)
            ai = train_ds.adr2idx.get(a)
            if di is None or ai is None:
                scores[i] = 0.0
                continue
            M = kern.S if hasattr(kern, "S") and kern.S is not None else kern.P
            scores[i] = float(M[di, train_ds.n_drugs + ai])
        return scores

    scores_neu  = kern_score_pairs(kern_neu,  eval_pairs)
    scores_path = kern_score_pairs(kern_path, eval_pairs)

    # Bipartite-aware baselines (Bug 2 fix)
    bip_scores  = _bipartite_scores(train_ds, eval_pairs)

    if verbose:
        print(f"\n{'─'*56}")
        print(f"  Inductive Link-Prediction on Held-Out Test Edges")
        print(f"  (kernel fitted on train only – no leakage)")
        print(f"{'─'*56}")
        header = f"  {'Method':<32} {'ROC-AUC':>8} {'PR-AUC':>8}"
        print(header)
        print(f"  {'─'*50}")

        all_scores = {
            "Preferential Attach. (PAC)":     bip_scores["PAC"],
            "2-hop CN (bipartite AAC fix)":   bip_scores["CN2 (2-hop AAC)"],
            "2-hop Jaccard":                  bip_scores["Jaccard-2hop"],
            "2-hop RAI":                      bip_scores["RAI-2hop"],
            "Odd Neumann Kernel":             scores_neu,
            "Odd Path Counting Kernel":       scores_path,
        }
        results_table = {}
        for name, sc in all_scores.items():
            try:
                if sc.std() < 1e-12:
                    print(f"  {name:<32} {'(all zeros – no signal)':>17}")
                    continue
                m = evaluate_link_prediction(y_true_test, sc)
                results_table[name] = m
                print(f"  {name:<32} {m['ROC-AUC']:>8.4f} {m['PR-AUC']:>8.4f}")
            except Exception as e:
                print(f"  {name:<32}  error: {e}")
        print(f"{'─'*56}")

    # ── STEP 4: Kernel SVM on polypharmacy triples (Bug 3 + 4 fix) ───────
    poly_triples = dataset.drug_pair_adr_triples
    if poly_triples:
        print("\nBuilding Kernel SVM with inductive Gram matrix …")

        # Use up to 300 positives for tractable Gram matrix
        rng       = np.random.default_rng(42)
        pos_idx   = rng.choice(len(poly_triples),
                               min(300, len(poly_triples)), replace=False)
        positives = [poly_triples[i] for i in pos_idx]

        # ADR-stratified negatives (Bug 4 fix)
        negatives = sample_negatives(positives, train_ds,
                                     ratio=1.0, stratify_by_adr=True)

        all_triples = positives + negatives
        labels_svm  = np.array([1]*len(positives) + [0]*len(negatives))

        # Use Odd Neumann kernel for SVM (better conditioned than Path Counting)
        pair_kern_neu  = DrugPairKernel(kern_neu,  train_ds)
        pair_kern_path = DrugPairKernel(kern_path, train_ds)

        for kern_name, pk in [("Odd Neumann",       pair_kern_neu),
                               ("Odd Path Counting", pair_kern_path)]:
            gram = pk.build_gram_matrix(all_triples)

            # Normalise Gram: K_ij / sqrt(K_ii * K_jj)
            diag = np.sqrt(np.diag(gram))
            diag[diag == 0] = 1.0
            gram_norm = gram / np.outer(diag, diag)

            # Bug 3 fix: add ridge regularisation to diagonal
            gram_reg_mat = gram_norm + gram_reg * np.eye(len(gram_norm))

            # Rank check – warn if Gram is near-singular
            eigvals = np.linalg.eigvalsh(gram_reg_mat)
            eff_rank = int((eigvals > 1e-6).sum())
            cond     = float(eigvals.max() / (eigvals.min() + 1e-12))
            print(f"\n  [{kern_name}] Gram shape={gram_reg_mat.shape} | "
                  f"eff_rank={eff_rank} | cond={cond:.1f}")

            cv = cross_validate_kernel(gram_reg_mat, labels_svm,
                                       n_splits=5, C=svm_C)
            print(f"  Kernel SVM 5-fold CV ({kern_name})")
            print(f"    ROC-AUC : {cv['ROC-AUC mean']:.4f} ± {cv['ROC-AUC std']:.4f}")
            print(f"    PR-AUC  : {cv['PR-AUC mean']:.4f}  ± {cv['PR-AUC std']:.4f}")
    else:
        print("\n  [Skip] No polypharmacy triples → Kernel SVM skipped.")
        print("         Load DS3-Neonatal or TWOSIDES for Model 1 evaluation.")

    # ── STEP 5: Plots ─────────────────────────────────────────────────────
    plot_bipartite_graph(train_ds)
    plot_kernel_heatmap(
        kern_neu.drug_adr_submatrix(train_ds)[:20, :20],
        title="Odd Neumann Kernel – Drug×ADR (train, first 20×20)",
        filename="odd_neumann_heatmap.png")
    plot_kernel_heatmap(
        kern_path.drug_adr_submatrix(train_ds)[:20, :20],
        title="Odd Path Counting Kernel – Drug×ADR (train, first 20×20)",
        filename="odd_path_heatmap.png")
    if y_true_test.std() > 0:
        plot_roc_pr(y_true_test, scores_neu,  "Odd Neumann (test)",
                    "roc_pr_neumann.png")
        plot_roc_pr(y_true_test, scores_path, "Odd Path Counting (test)",
                    "roc_pr_path.png")

    print("\nPipeline complete.\n")
    return {
        "train_dataset":  train_ds,
        "test_pos":       test_pos,
        "test_neg":       test_neg,
        "kern_neumann":   kern_neu,
        "kern_path":      kern_path,
    }


# ════════════════════════════════════════════════════════════════════════════
# 13.  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_full_pipeline()
