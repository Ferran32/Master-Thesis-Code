"""Metric helpers shared by near_duplicates.py and paired_tests.py."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from scipy import stats


@dataclass
class Metrics:
    n: int
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) and p == p and r == r else float("nan")

    @property
    def f2(self) -> float:
        p, r = self.precision, self.recall
        beta2 = 4.0
        denom = beta2 * p + r
        if not denom or p != p or r != r:
            return float("nan")
        return (1 + beta2) * p * r / denom

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else float("nan")

    @property
    def ppr(self) -> float:
        """Positive prediction rate: share of samples predicted vulnerable."""
        return (self.tp + self.fp) / self.n if self.n else float("nan")

    @property
    def base_rate(self) -> float:
        return (self.tp + self.fn) / self.n if self.n else float("nan")


def confusion_from_pairs(labels: Sequence[int], preds: Sequence[int]) -> Metrics:
    tp = fp = fn = tn = 0
    for y, p in zip(labels, preds):
        if y == 1 and p == 1:
            tp += 1
        elif y == 0 and p == 1:
            fp += 1
        elif y == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return Metrics(n=len(labels), tp=tp, fp=fp, fn=fn, tn=tn)


def binomial_precision_test(m: Metrics) -> float:
    """Exact two-sided binomial test of precision against the positive base
    rate: among the (tp+fp) positive predictions, is the hit rate (tp)
    different from what you'd expect if predictions were independent of the
    true label (i.e. base_rate)?
    """
    n_pos_pred = m.tp + m.fp
    if n_pos_pred == 0:
        return float("nan")
    res = stats.binomtest(m.tp, n_pos_pred, m.base_rate, alternative="two-sided")
    return res.pvalue
