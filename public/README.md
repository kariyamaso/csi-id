# Public (distribution) folder

This `public/` folder is a self-contained subset of the repo for running the
WiSense experiments without relying on the rest of the project tree.

## Structure

- `public/train.py`: main entrypoint (train/eval, writes `runs/.../metrics.json`)
- `public/scripts/aggregate_results.py`: aggregates `runs/**/metrics.json` into CSVs
- `public/wisense/`: minimal Python package (datasets, models, utilities)
- `public/configs/*.json`: example configs (model/dataset/training knobs)

## Quick start

```bash
source .venv/bin/activate
python public/scripts/prepare_datasets.py --mode symlink
python public/train.py --config public/configs/ntu_humanid_mamba.json --seed 0 --measure_efficiency 1
python public/scripts/aggregate_results.py --runs-dir runs --out-dir artifacts/aggregate --dataset NTU-Fi-HumanID
```

You can override config keys via CLI flags (see `python public/train.py -h`).

## Run all models × seeds (example)

```bash
source .venv/bin/activate

DATASET="NTU-Fi-HumanID"
MODELS="MLP LeNet ResNet18 ResNet50 ResNet101 RNN GRU LSTM BiLSTM CNN+GRU ViT Mamba"

for m in $MODELS; do
  for s in $(seq 0 9); do
    python public/train.py --config public/configs/ntu_humanid_all_models.json --dataset "$DATASET" --model "$m" --seed "$s"
  done
done

python public/scripts/aggregate_results.py --runs-dir runs --out-dir artifacts/aggregate --dataset "$DATASET"
```

## UT-HAR / Widar

- UT-HAR: `--dataset UT_HAR_data` (expects `Data/UT_HAR/...`)
- Widar: `--dataset Widar` (expects `Data/Widardata/...`)

Example:

```bash
python public/train_all_models.py --dataset UT_HAR_data --seeds 0 1 2 --config public/configs/ntu_humanid_all_models.json
python public/train_all_models.py --dataset Widar --seeds 0 1 2 --config public/configs/ntu_humanid_all_models.json
```

## Plots

- From training logs (NTU-Fi): `public/plot_ntu_fi_results.py`, `public/export_results_table.py`
- From aggregated CSVs (all datasets): `public/scripts/plot_aggregate.py`

## One-command run

```bash
python public/orchestrate_full_run.py --datasets NTU-Fi-HumanID NTU-Fi_HAR --seeds 0 1 2 --config public/configs/ntu_humanid_all_models.json
```
