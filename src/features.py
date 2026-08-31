"""Feature engineering utilities for the smartphone addiction project."""

from __future__ import annotations

import pandas as pd


SCREEN_COMPONENT_COLUMNS = [
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
]


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add screen-time consistency features without mutating the input frame."""
    result = df.copy()

    result["screen_components_observed"] = result[SCREEN_COMPONENT_COLUMNS].notna().sum(axis=1)
    result["observed_component_sum"] = result[SCREEN_COMPONENT_COLUMNS].sum(axis=1, min_count=1)
    result["screen_budget_slack"] = (
        result["daily_screen_time_hours"] - result["observed_component_sum"]
    )
    result["other_screen_complete"] = (
        result["daily_screen_time_hours"]
        - result["social_media_hours"]
        - result["gaming_hours"]
        - result["work_study_hours"]
    )

    return result
