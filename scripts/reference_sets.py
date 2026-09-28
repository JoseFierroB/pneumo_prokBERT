"""Shared canonical reference loaders (single implementation; import, do not copy).

The three pneumo reference sets live in datasets/reference/<set>/{positives,negatives}_81bp.fasta.
load_reference_sets returns {col: (seqs, y)} with positives first and y as an int
numpy array — the same order eval_matrix.build_columns uses (energy/ probes must
match it exactly, otherwise cross-script label asserts fail).

Left duplicated on purpose (different sources, not worth unifying): the HF
80/10/10 split recompute, HF hub/parquet reads, and 2-line CSV trio loaders.

Usage:
  from scripts.reference_sets import CANONICAL_SETS, load_reference_sets
  data = load_reference_sets("datasets/reference", ["tigr4-high"])
"""
import numpy as np

CANONICAL_SETS = {
    "d39v": ("d39v", "positives_81bp.fasta", "negatives_81bp.fasta"),
    "tigr4-high": ("tigr4-high", "positives_81bp.fasta", "negatives_81bp.fasta"),
    "d39v-tigr4": ("d39v-tigr4", "positives_81bp.fasta", "negatives_81bp.fasta"),
}


def load_reference_sets(root: str, cols=None, verbose: bool = False) -> dict:
    """Load canonical FASTA sets. Returns {col: (seqs list, y int array)}, positives first."""
    from Bio import SeqIO
    cols = list(cols) if cols is not None else list(CANONICAL_SETS)
    data = {}
    for col in cols:
        sub, pf, nf = CANONICAL_SETS[col]
        pos = list(SeqIO.parse(f"{root}/{sub}/{pf}", "fasta"))
        neg = list(SeqIO.parse(f"{root}/{sub}/{nf}", "fasta"))
        data[col] = ([str(r.seq).upper() for r in pos + neg],
                     np.array([1] * len(pos) + [0] * len(neg)))
        if verbose:
            print(col, "OK", len(data[col][0]))
    return data
