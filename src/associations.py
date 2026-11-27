"""
Associations between instruments, country and lyrical theme (Step 7).

Lift. For an instrument family A and a label B (a release country, or a lyrical theme
= a G_lyrics community), over the songs that have both kinds of data:
    lift(A, B) = P(A and B) / (P(A) P(B)) = n(A, B) N / (n(A) n(B))
Lift 1 means A is as common among B songs as among all songs; 2 means twice as common.
It is shown on a log2 scale, so 2 and 1/2 are equally far from "no association".

Chi-square. A song usually has several instrument families, so a single families x
labels table would count songs several times and break the chi-square assumptions.
Each family is therefore tested on its own: a 2 x K table (family present / absent x
label), where every song is counted once. Many cells have small expected counts, so
the chi-square p-value is checked with a permutation p-value (labels shuffled over the
songs, keeping how many songs each label has). With 14 families, p-values are also
Bonferroni-corrected (x number of families tested). Cramer's V gives the strength of
the association on a 0-1 scale.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist

from .nulls import empirical_p


def family_matrix(families):
    """Boolean DataFrame songs x families from 'fam1;fam2' strings (NaN rows dropped)."""
    return families.dropna().str.get_dummies(sep=";").astype(bool)


def counts(F, labels):
    """n(A, B): families x labels, songs with family A and label B."""
    labels = labels.reindex(F.index)
    return pd.DataFrame({b: F[labels == b].sum() for b in labels.dropna().unique()})


def lift(F, labels):
    """lift(A, B) = n(A, B) N / (n(A) n(B)), on the songs that have a label."""
    F = F[labels.reindex(F.index).notna()]
    n = counts(F, labels)
    N = len(F)
    nb = labels.reindex(F.index).value_counts()
    return n * N / np.outer(F.sum(), nb[n.columns])


def _chi2(present, codes, k):
    """Chi-square statistic of the 2 x k table (present / absent x label code)."""
    obs = np.vstack([np.bincount(codes[present], minlength=k),
                     np.bincount(codes[~present], minlength=k)]).astype(float)
    exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
    ok = exp > 0
    return float(((obs - exp)[ok] ** 2 / exp[ok]).sum()), exp


def family_tests(F, labels, n_perm, seed, min_songs=10):
    """One row per family with at least min_songs songs: chi-square, Cramer's V,
    permutation p and Bonferroni-corrected p-values."""
    labels = labels.reindex(F.index).dropna()
    F = F.loc[labels.index]
    codes, uniq = pd.factorize(labels)
    k = len(uniq)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(codes) for _ in range(n_perm)]   # same shuffles for all
    rows = []
    for fam in F.columns:
        present = F[fam].to_numpy()
        if present.sum() < min_songs or (~present).sum() < min_songs:
            continue
        stat, exp = _chi2(present, codes, k)
        null = np.array([_chi2(present, c, k)[0] for c in perms])
        rows.append({"family": fam, "songs": int(present.sum()), "chi2": stat,
                     "dof": k - 1, "p_chi2": float(chi2_dist.sf(stat, k - 1)),
                     "p_perm": empirical_p(stat, null),
                     "cramers_v": np.sqrt(stat / (len(codes) * min(1, k - 1))),
                     "share_expected_lt5": float((exp < 5).mean())})
    out = pd.DataFrame(rows)
    m = len(out)
    out["p_chi2_bonf"] = (out.p_chi2 * m).clip(upper=1)
    out["p_perm_bonf"] = (out.p_perm * m).clip(upper=1)
    return out.sort_values("chi2", ascending=False).reset_index(drop=True)


# toy checks: family "x" only in label "a" -> lift 2 there and 0 elsewhere (half the
# songs are "a"); a family spread evenly has lift 1 everywhere
_F = pd.DataFrame({"x": [1, 1, 0, 0], "y": [1, 1, 1, 1]}, index=list("pqrs")).astype(bool)
_l = pd.Series(["a", "a", "b", "b"], index=list("pqrs"))
_L = lift(_F, _l)
assert _L.loc["x", "a"] == 2 and _L.loc["x", "b"] == 0 and (_L.loc["y"] == 1).all()
_s, _ = _chi2(_F.x.to_numpy(), pd.factorize(_l)[0], 2)
assert np.isclose(_s, 4.0)   # perfect 2x2 association on 4 songs: chi2 = N
