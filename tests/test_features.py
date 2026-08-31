import numpy as np
import pandas as pd

from src.features import add_structural_features


def test_structural_features_are_computed_without_mutating_input():
    df = pd.DataFrame(
        {
            "daily_screen_time_hours": [10.0, 8.0, 6.0],
            "social_media_hours": [2.0, np.nan, 1.0],
            "gaming_hours": [1.0, 2.0, np.nan],
            "work_study_hours": [4.0, 3.0, np.nan],
        }
    )
    original = df.copy(deep=True)

    result = add_structural_features(df)

    pd.testing.assert_frame_equal(df, original)
    assert result["screen_components_observed"].tolist() == [3, 2, 1]
    np.testing.assert_allclose(
        result["observed_component_sum"].to_numpy(),
        np.array([7.0, 5.0, 1.0]),
    )
    np.testing.assert_allclose(
        result["screen_budget_slack"].to_numpy(),
        np.array([3.0, 3.0, 5.0]),
    )
    assert result.loc[0, "other_screen_complete"] == 3.0
    assert np.isnan(result.loc[1, "other_screen_complete"])
    assert np.isnan(result.loc[2, "other_screen_complete"])
