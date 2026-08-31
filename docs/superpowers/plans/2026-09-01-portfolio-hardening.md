# Portfolio Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the smartphone-addiction Kaggle repository into a clean, tested, CI-backed portfolio project while preserving the verified independent scores and notebook results.

**Architecture:** Keep notebooks as the experiment record, expose only the two reusable transformations through a minimal `src` API, cover those transformations with lightweight unit tests, and run those tests in GitHub Actions without competition data or GPU requirements. Documentation becomes the entry point for setup, data acquisition, reproducibility boundaries, and results.

**Tech Stack:** Python 3.12, pandas, NumPy, scikit-learn, pytest, GitHub Actions, Markdown, Kaggle CLI.

**Spec:** `docs/superpowers/specs/2026-09-01-portfolio-hardening-design.md`

## Global Constraints

- Preserve the independently reported scores: 0.96774 mean CV ROC-AUC and 0.96921 Kaggle Public LB ROC-AUC.
- Do not add external prediction-file or ensemble scores as project achievements.
- Do not require Kaggle data or a GPU for unit tests or CI.
- Do not modify recorded notebook outputs or execution history.
- Do not add a software license without an explicit owner choice.
- Do not claim full end-to-end retraining was verified in every environment.

---

### Task 1: Stabilize the reusable Python API and feature tests

**Files:**
- Create: `tests/test_features.py`
- Create: `tests/test_public_api.py`
- Create: `pytest.ini`
- Create: `src/__init__.py`
- Existing: `src/features.py`
- Existing: `src/target_encoding.py`

**Interfaces:**
- Consumes: `src.features.add_structural_features(df: pd.DataFrame) -> pd.DataFrame`
- Consumes: `src.target_encoding.add_crossfit_exact_te(X_train, y_train, X_valid, columns, alpha=20.0, n_inner_splits=5, random_state=123) -> tuple[pd.DataFrame, pd.DataFrame]`
- Produces: `from src import add_structural_features, add_crossfit_exact_te`

- [ ] **Step 1: Add characterization tests for structural features**

Create `tests/test_features.py`:

```python
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
```

- [ ] **Step 2: Create pytest configuration**

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
pythonpath = .
addopts = -q
```

- [ ] **Step 3: Run existing + feature tests**

Run:

```bash
python -m pytest tests/test_target_encoding.py tests/test_features.py
```

Expected: all tests pass.

- [ ] **Step 4: Write a failing public-API test before adding exports**

Create `tests/test_public_api.py`:

```python
from src import add_crossfit_exact_te, add_structural_features


def test_public_api_exports_core_transformations():
    assert callable(add_structural_features)
    assert callable(add_crossfit_exact_te)
```

Run:

```bash
python -m pytest tests/test_public_api.py
```

Expected before `src/__init__.py`: FAIL because the names are not exported from `src`.

- [ ] **Step 5: Add minimal public API**

Create `src/__init__.py`:

```python
"""Reusable transformations for the smartphone addiction project."""

from .features import add_structural_features
from .target_encoding import add_crossfit_exact_te

__all__ = ["add_structural_features", "add_crossfit_exact_te"]
```

- [ ] **Step 6: Run the complete lightweight suite**

Run:

```bash
python -m pytest
```

Expected: all tests pass.

---

### Task 2: Add data acquisition documentation

**Files:**
- Create: `data/README.md`
- Delete: `data/.gitkeep`

**Interfaces:**
- Produces the documented local data contract: `data/train.csv`, `data/test.csv`, `data/sample_submission.csv`.

- [ ] **Step 1: Add data README**

Create `data/README.md`:

```markdown
# Competition Data

The Kaggle competition files are intentionally not committed to this repository.

Expected local files:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

## Option 1 — Kaggle website

Download the data from the Kaggle Playground Series S6E8 competition page and place the three CSV files in this directory.

## Option 2 — Kaggle CLI

After configuring your Kaggle credentials, run from the repository root:

```bash
kaggle competitions download -c playground-series-s6e8 -p data
```

Then extract the downloaded ZIP into `data/`.

The repository `.gitignore` excludes competition CSV and ZIP files so the dataset is not accidentally committed.
```

- [ ] **Step 2: Remove the obsolete placeholder**

Delete `data/.gitkeep` after `data/README.md` exists.

- [ ] **Step 3: Verify ignored data contract**

Confirm `.gitignore` still contains rules covering `data/*.csv` and `data/*.zip`.

---

### Task 3: Add lightweight continuous integration

**Files:**
- Create: `.github/workflows/tests.yml`

**Interfaces:**
- Consumes: `pytest.ini`, `tests/`, Python 3.12.
- Produces: GitHub Actions check named `tests` on pushes to `main` and pull requests.

- [ ] **Step 1: Create the CI workflow**

Create `.github/workflows/tests.yml`:

```yaml
name: tests

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install test dependencies
        run: |
          python -m pip install --upgrade pip
          python -m pip install "numpy>=1.26" "pandas>=2.2" "scikit-learn>=1.4" "pytest>=8.0"

      - name: Run unit tests
        run: python -m pytest
```

- [ ] **Step 2: Validate YAML structure**

Parse the YAML with a local YAML parser if available, or inspect the GitHub Actions workflow through the PR after push.

- [ ] **Step 3: Verify CI scope**

Confirm the workflow does not download Kaggle data, invoke notebooks, or require XGBoost/GPU dependencies.

---

### Task 4: Clean assets and static notebook-facing inconsistencies

**Files:**
- Delete: `assets/model_results.svg`
- Keep: `assets/model_results_v3.svg`
- Inspect: `notebooks/01_eda_portfolio.ipynb`
- Inspect: `notebooks/02_baseline_portfolio.ipynb`
- Inspect: `notebooks/03_xgboost_advanced_portfolio.ipynb`

**Interfaces:**
- README must reference only `assets/model_results_v3.svg`.

- [ ] **Step 1: Remove superseded results graphic**

Delete `assets/model_results.svg` and keep `assets/model_results_v3.svg` unchanged.

- [ ] **Step 2: Search notebook text for stale names and unsafe claims**

Search for at least:

```text
02_baseline.ipynb
03_xgboost_advanced.ipynb
external prediction
external ensemble
```

- [ ] **Step 3: Decide notebook edits conservatively**

If a stale filename can be changed without rewriting a large serialized notebook or touching outputs, correct it. Otherwise leave the notebook unchanged and ensure the README contains the correct filename.

- [ ] **Step 4: Preserve executed model configuration**

Do not change recorded XGBoost scores or execution outputs. Do not silently replace `device="cuda"` in an already executed cell; document CPU fallback in the README instead.

---

### Task 5: Rewrite README as the recruiter-facing entry point

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: results graphic, actual notebook names, `src/`, `tests/`, CI workflow, `data/README.md`.
- Produces: setup and navigation documentation for a fresh reader.

- [ ] **Step 1: Add CI badge and keep results headline**

At the top, add:

```markdown
[![tests](https://github.com/miladakbariv/smartphone-addiction-kaggle/actions/workflows/tests.yml/badge.svg)](https://github.com/miladakbariv/smartphone-addiction-kaggle/actions/workflows/tests.yml)
```

Do not change the 0.96774 CV or 0.96921 Public LB headline values.

- [ ] **Step 2: Add notebook navigation**

Add links for:

```markdown
- [EDA](notebooks/01_eda_portfolio.ipynb)
- [Baseline models](notebooks/02_baseline_portfolio.ipynb)
- [Advanced XGBoost](notebooks/03_xgboost_advanced_portfolio.ipynb)
```

- [ ] **Step 3: Add Quick Start**

Document:

```bash
python -m venv .venv
# activate the environment for your OS
python -m pip install -r requirements.txt
python -m pytest
```

Then point to `data/README.md` for competition files.

- [ ] **Step 4: Document reusable code and test scope**

Explain:

```text
src/features.py           structural screen-time features
src/target_encoding.py    nested cross-fitted exact-value target encoding
tests/                    lightweight, data-free unit tests
```

State that CI verifies utilities only, not full competition retraining.

- [ ] **Step 5: Update repository tree**

Show the actual structure including `.github/workflows/tests.yml`, `assets/model_results_v3.svg`, `data/README.md`, `src/__init__.py`, both utility modules, and all tests.

- [ ] **Step 6: Add GPU/CPU reproducibility note**

State that the advanced notebook was originally executed on Kaggle with GPU-enabled XGBoost. For CPU-only local runs, users may set the XGBoost `device` parameter to `"cpu"`; recorded notebook outputs correspond to the original execution environment.

- [ ] **Step 7: Verify all README paths**

Confirm every referenced repository path exists on the branch.

---

### Task 6: Final verification, review, and merge

**Files:**
- Review all changed files.

**Interfaces:**
- Produces: merged `main` with tested portfolio-hardening changes.

- [ ] **Step 1: Reconstruct a clean lightweight checkout locally**

Copy the branch versions of:

```text
src/__init__.py
src/features.py
src/target_encoding.py
tests/test_features.py
tests/test_target_encoding.py
tests/test_public_api.py
pytest.ini
```

into an empty temporary directory.

- [ ] **Step 2: Run full lightweight verification**

Run:

```bash
python -m pytest
```

Expected: zero failures.

- [ ] **Step 3: Verify documentation and assets**

Confirm:

```text
README.md -> assets/model_results_v3.svg exists
data/README.md exists
assets/model_results.svg does not exist
README notebook links match the three actual notebook filenames
```

- [ ] **Step 4: Compare branch against main**

Review changed filenames and diff. Reject unintended modifications to notebook outputs, reported scores, or competition results.

- [ ] **Step 5: Open a pull request**

Use title:

```text
Harden Kaggle project for portfolio use
```

Describe tests, CI, data docs, README improvements, and asset cleanup.

- [ ] **Step 6: Check CI status**

If GitHub Actions reports a failure, inspect and fix before merging.

- [ ] **Step 7: Merge only after verification**

Squash-merge the pull request into `main`, then re-fetch `README.md`, `src/__init__.py`, `.github/workflows/tests.yml`, and the final commit status to verify the merged state.
