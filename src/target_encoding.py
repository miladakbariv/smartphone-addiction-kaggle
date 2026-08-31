"""Leakage-safe target encoding utilities."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.model_selection import StratifiedKFold


_MISSING_SENTINEL = -999999.123456


def add_crossfit_exact_te(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    columns: Sequence[str],
    alpha: float = 20.0,
    n_inner_splits: int = 5,
    random_state: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add leakage-safe exact-value target encodings to train and validation data.

    Training rows receive out-of-fold encodings from inner stratified folds.
    Validation rows are encoded from the full training data only. Rare values
    are smoothed toward the corresponding training-fold target mean.
    """
    X_train_new = X_train.copy()
    X_valid_new = X_valid.copy()

    inner_skf = StratifiedKFold(
        n_splits=n_inner_splits,
        shuffle=True,
        random_state=random_state,
    )

    for col in columns:
        train_encoded = pd.Series(index=X_train.index, dtype="float64")

        for inner_train_idx, inner_val_idx in inner_skf.split(X_train, y_train):
            X_inner_train = X_train.iloc[inner_train_idx]
            X_inner_val = X_train.iloc[inner_val_idx]
            y_inner_train = y_train.iloc[inner_train_idx]

            inner_global_mean = y_inner_train.mean()
            train_key = X_inner_train[col].fillna(_MISSING_SENTINEL)
            val_key = X_inner_val[col].fillna(_MISSING_SENTINEL)

            stats = (
                pd.DataFrame(
                    {
                        "key": train_key.values,
                        "target": y_inner_train.values,
                    }
                )
                .groupby("key")["target"]
                .agg(["sum", "count"])
            )

            smoothed_map = (
                stats["sum"] + alpha * inner_global_mean
            ) / (stats["count"] + alpha)

            encoded_values = val_key.map(smoothed_map).fillna(inner_global_mean)
            train_encoded.loc[X_inner_val.index] = encoded_values.values

        outer_global_mean = y_train.mean()
        full_train_key = X_train[col].fillna(_MISSING_SENTINEL)
        valid_key = X_valid[col].fillna(_MISSING_SENTINEL)

        full_stats = (
            pd.DataFrame(
                {
                    "key": full_train_key.values,
                    "target": y_train.values,
                }
            )
            .groupby("key")["target"]
            .agg(["sum", "count"])
        )

        full_map = (
            full_stats["sum"] + alpha * outer_global_mean
        ) / (full_stats["count"] + alpha)

        valid_encoded = valid_key.map(full_map).fillna(outer_global_mean)

        X_train_new[f"{col}_exact_te"] = train_encoded
        X_valid_new[f"{col}_exact_te"] = valid_encoded.values

    return X_train_new, X_valid_new
