# Portfolio Hardening Design

## Goal

Turn the repository from a notebook-centered Kaggle project into a clean, recruiter-facing, reproducible machine-learning portfolio repository without overstating results or adding unverified production machinery.

## Current State

The repository already contains three portfolio notebooks, reusable structural-feature engineering in `src/features.py`, leakage-safe exact-value target encoding in `src/target_encoding.py`, a small target-encoding test suite, a results graphic, and a README that reports the independently achieved scores of 0.96774 mean CV ROC-AUC and 0.96921 Kaggle Public LB ROC-AUC.

## Design Principles

1. Preserve the project’s independent-result story. Do not add external ensemble scores or imply they belong to this repository.
2. Prefer small reusable utilities and tests over packaging or training infrastructure that cannot be fully verified against the competition data in this environment.
3. Keep notebook results intact. Static notebook cleanup must not change recorded model scores or silently rewrite executed model configurations.
4. Make a fresh checkout understandable and testable with a small number of commands.
5. Add CI only for lightweight unit tests that do not require Kaggle data or a GPU.
6. Keep the competition dataset out of Git and document how to obtain it.

## Repository Changes

### Public Python API

Create `src/__init__.py` and re-export:

- `add_structural_features`
- `add_crossfit_exact_te`

This gives the repository a minimal, explicit reusable API while keeping the existing modules focused.

### Tests

Add `tests/test_features.py` to verify:

- all four structural features are created,
- observed-component counting and sums are correct with missing values,
- `screen_budget_slack` and `other_screen_complete` follow the intended formulas,
- the input DataFrame is not mutated.

Keep the existing target-encoding tests and add a small public-API import test through `src`.

Create `pytest.ini` with `testpaths = tests`, `pythonpath = .`, and quiet output so both `pytest` and `python -m pytest` work predictably from the repository root.

### Continuous Integration

Create `.github/workflows/tests.yml` for GitHub Actions. It will run on pushes to `main` and pull requests, use Python 3.12, install only the lightweight dependencies needed by the unit tests (`numpy`, `pandas`, `scikit-learn`, `pytest`), and run `python -m pytest`.

The workflow intentionally does not install LightGBM, CatBoost, XGBoost, Jupyter, or download Kaggle data because the CI scope is utility correctness, not full competition retraining.

### Data Documentation

Create `data/README.md` explaining that competition CSVs are intentionally excluded from Git. Document the expected filenames and two acquisition paths:

- download them manually from the Kaggle Playground Series S6E8 competition page,
- or use the Kaggle CLI command `kaggle competitions download -c playground-series-s6e8` followed by extraction into `data/`.

Remove `data/.gitkeep` after the README exists. Keep placeholders in `models/` and `submissions/` because those directories remain intentionally empty in Git.

### Assets

Keep `assets/model_results_v3.svg`, which is the graphic referenced by the README. Remove the superseded `assets/model_results.svg` so the repository has one canonical results graphic.

### README

Update the README to:

- add a CI badge,
- keep the existing independent CV/Public LB results unchanged,
- add a concise Quick Start section,
- explain the reusable `src/` utilities,
- explain unit tests and how to run them,
- show the actual repository structure including `assets/`, `tests/`, and `data/README.md`,
- link the three actual portfolio notebook filenames,
- clarify that the advanced notebook was originally executed on Kaggle with GPU-enabled XGBoost and that local CPU users may need to set XGBoost `device="cpu"`,
- avoid claiming full end-to-end retraining has been verified on every environment.

### Notebook Review

Statically inspect the three notebook files for portfolio-facing inconsistencies. Fix only text-level issues that are clearly safe and that do not require re-running models. In particular, correct stale notebook-name references if they can be edited safely. Do not alter model scores, outputs, or execution history.

If editing a notebook would require replacing a large serialized notebook without reliable verification, leave the notebook untouched and ensure the README uses the correct filenames instead.

## Verification

Before merging:

1. Reconstruct the changed lightweight Python files in a clean temporary directory and run `python -m pytest`.
2. Verify the workflow YAML and README references point to files that exist on the branch.
3. Verify the results graphic referenced by the README exists and the superseded graphic is absent.
4. Compare the feature branch to `main` and review the changed-file list.
5. Open a pull request, check GitHub Actions if available, and merge only after local lightweight verification succeeds and the PR diff contains no unintended notebook/result changes.

## Out of Scope

- A new `train.py` or command-line training pipeline.
- Packaging the repository as an installable PyPI project.
- Downloading or committing the Kaggle competition data.
- Re-running the full 691k-row cross-validation pipeline in CI.
- Adding a license without an explicit license choice from the repository owner.
- Changing the reported independent scores of 0.96774 CV and 0.96921 Public LB.
