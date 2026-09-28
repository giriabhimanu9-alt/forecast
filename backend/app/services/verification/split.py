"""Chronological train/test split by observation period (no leakage).

Samples are split on ``observation_period_end``: every training observation
period ends no later than the first test period starts. Weights are learned
from ``train`` only; ``test`` observations are used exclusively for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


class LeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class Split:
    train: xr.Dataset
    test: xr.Dataset
    train_dates: list[str]
    test_dates: list[str]
    boundary: pd.Timestamp  # last training observation_period_end

    def describe(self) -> dict[str, object]:
        return {
            "train_valid_dates": [self.train_dates[0], self.train_dates[-1]] if self.train_dates else [],
            "test_valid_dates": [self.test_dates[0], self.test_dates[-1]] if self.test_dates else [],
            "n_train_days": len(self.train_dates),
            "n_test_days": len(self.test_dates),
            "n_train_samples": int(self.train.sizes["sample"]),
            "n_test_samples": int(self.test.sizes["sample"]),
            "boundary_utc": self.boundary.isoformat(),
        }


def chronological_split(aligned: xr.Dataset, train_fraction: float) -> Split:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    ends = pd.DatetimeIndex(aligned.observation_period_end.values)
    periods = np.sort(ends.unique())
    if periods.size < 2:
        raise ValueError("need at least two distinct observation periods to split")
    n_train = min(max(1, int(round(train_fraction * periods.size))), periods.size - 1)
    boundary = pd.Timestamp(periods[n_train - 1])
    is_train = ends <= boundary
    train = aligned.isel(sample=np.flatnonzero(is_train))
    test = aligned.isel(sample=np.flatnonzero(~is_train))
    assert_no_leakage(train, test)
    return Split(
        train=train,
        test=test,
        train_dates=sorted(set(train.valid_date.values.tolist())),
        test_dates=sorted(set(test.valid_date.values.tolist())),
        boundary=boundary,
    )


def assert_no_leakage(train: xr.Dataset, test: xr.Dataset) -> None:
    """Training observation periods must end before any test period starts."""
    if train.sizes["sample"] == 0 or test.sizes["sample"] == 0:
        raise LeakageError("train and test must both be non-empty")
    last_train_end = pd.Timestamp(train.observation_period_end.values.max())
    first_test_start = pd.Timestamp(test.observation_period_start.values.min())
    if last_train_end > first_test_start:
        raise LeakageError(
            f"training observations extend to {last_train_end}, past the first test period start {first_test_start}"
        )
    overlap = set(train.valid_date.values.tolist()) & set(test.valid_date.values.tolist())
    if overlap:
        raise LeakageError(f"valid dates in both train and test: {sorted(overlap)}")
