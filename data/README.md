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
