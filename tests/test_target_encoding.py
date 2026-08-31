import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.target_encoding import add_crossfit_exact_te


def test_crossfit_encoding_uses_only_inner_training_rows():
    X_train = pd.DataFrame({"value": np.arange(12, dtype=float)})
    y_train = pd.Series([0, 1] * 6, index=X_train.index)
    X_valid = pd.DataFrame({"value": [100.0, 101.0]})

    train_new, valid_new = add_crossfit_exact_te(
        X_train,
        y_train,
        X_valid,
        columns=["value"],
        alpha=20.0,
        n_inner_splits=3,
        random_state=123,
    )

    expected_train = pd.Series(index=X_train.index, dtype="float64")
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
    for inner_train_idx, inner_val_idx in splitter.split(X_train, y_train):
        expected_train.iloc[inner_val_idx] = y_train.iloc[inner_train_idx].mean()

    np.testing.assert_allclose(
        train_new["value_exact_te"].to_numpy(),
        expected_train.to_numpy(),
    )
    np.testing.assert_allclose(
        valid_new["value_exact_te"].to_numpy(),
        np.full(len(X_valid), y_train.mean()),
    )


def test_crossfit_encoding_handles_missing_values_and_does_not_mutate_inputs():
    X_train = pd.DataFrame(
        {"value": [1.0, 1.0, np.nan, np.nan, 2.0, 2.0, 3.0, 3.0]}
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], index=X_train.index)
    X_valid = pd.DataFrame({"value": [np.nan, 1.0, 999.0]})
    X_train_before = X_train.copy(deep=True)
    X_valid_before = X_valid.copy(deep=True)

    train_new, valid_new = add_crossfit_exact_te(
        X_train,
        y_train,
        X_valid,
        columns=["value"],
        alpha=20.0,
        n_inner_splits=2,
        random_state=123,
    )

    pd.testing.assert_frame_equal(X_train, X_train_before)
    pd.testing.assert_frame_equal(X_valid, X_valid_before)
    assert "value_exact_te" in train_new.columns
    assert "value_exact_te" in valid_new.columns
    assert train_new["value_exact_te"].notna().all()
    assert valid_new["value_exact_te"].notna().all()
    assert valid_new.loc[2, "value_exact_te"] == y_train.mean()


def test_crossfit_encoding_is_deterministic_for_fixed_random_state():
    X_train = pd.DataFrame(
        {"value": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]}
    )
    y_train = pd.Series([0, 1] * 6, index=X_train.index)
    X_valid = pd.DataFrame({"value": [1, 3, 7]})

    first_train, first_valid = add_crossfit_exact_te(
        X_train,
        y_train,
        X_valid,
        ["value"],
        n_inner_splits=3,
        random_state=7,
    )
    second_train, second_valid = add_crossfit_exact_te(
        X_train,
        y_train,
        X_valid,
        ["value"],
        n_inner_splits=3,
        random_state=7,
    )

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_valid, second_valid)
