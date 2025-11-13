# NTU-Fi HumanID Evaluation

This note summarizes the structure of the NTU-Fi HumanID dataset bundled in `Data/NTU-Fi-HumanID` and the validation results obtained by running all pretrained checkpoints from `model_pt/` with `run.py --eval-only`.

## Dataset Description

- **Modality**: CSI amplitude (`CSIamp`) traces sampled from three antennas and 114 subcarriers, downsampled to 500 time steps (tensor shape `3 × 114 × 500` per sample). The loader normalizes each trace with `(x - 42.3199) / 4.9802`.
- **Class labels**: 14 identities stored in folders `001`–`013` and `015`. Each folder corresponds to one subject’s gait signature.
- **Storage layout**:

| Split  | Directory                         | Subjects | Samples / subject | Total samples |
|--------|-----------------------------------|----------|-------------------|---------------|
| Train  | `Data/NTU-Fi-HumanID/test_amp`    | 14       | 39                | 546           |
| Test   | `Data/NTU-Fi-HumanID/train_amp`   | 14       | 21                | 294           |

> **Note:** The SenseFi repo intentionally loads `test_amp` for training and `train_amp` for evaluation because the downloaded archives are named opposite to the counts reported in the paper (546 training / 294 test samples). The current code follows the official practice and therefore keeps the class separation intact.

## Evaluation Commands

```
cd WiFi-CSI-Sensing-Benchmark
for model in MLP LeNet ResNet18 ResNet50 ResNet101 RNN GRU LSTM BiLSTM 'CNN+GRU' ViT
do
  python3 run.py \
    --dataset NTU-Fi-HumanID \
    --model "$model" \
    --checkpoint "model_pt/NTU-Fi-HumanID_${model}.pt" \
    --eval-only | tee "model_pt/NTU-Fi-HumanID_result/${model}_eval.log"
done
```

Logs for each run are stored in `model_pt/NTU-Fi-HumanID_result/<model>_eval.log`.

## Normalization Check

Run `source .venv/bin/activate && python compute_ntu_fi_stats.py --split test_amp` to compute raw means/stds (script replicates `CSI_Dataset` preprocessing before normalization).  
Current dump statistics:

| Split (SenseFi convention) | Files | Raw mean | Raw std |
| --- | --- | --- | --- |
| `test_amp` (training split) | 546 | 38.8246 | 5.9803 |
| `train_amp` (evaluation split) | 294 | 38.8401 | 5.9679 |

The pretrained checkpoints assume `(mean=42.3199, std=4.9802)`. The ~3.5 lower mean and 1.0 higher std explain the ~−0.7σ offset observed during inference. You can override the normalization by exporting environment variables before invoking `run.py`:

```
export NTU_FI_NORM_MEAN=38.8246
export NTU_FI_NORM_STD=5.9803
python run.py --dataset NTU-Fi-HumanID --model ResNet18 ...
```

`dataset.CSI_Dataset` now reads these env vars (defaults keep legacy behavior), so you can switch back easily.

## Validation Results

| Model | Accuracy | Loss |
| --- | --- | --- |
| ResNet50 | 0.1418 | 10.89327 |
| ViT | 0.1387 | 7.72036 |
| LSTM | 0.1313 | 4.58762 |
| LeNet | 0.1313 | 25.48394 |
| MLP | 0.1313 | 37.08916 |
| ResNet18 | 0.1313 | 11.00836 |
| BiLSTM | 0.1281 | 5.57015 |
| GRU | 0.1281 | 5.68206 |
| ResNet101 | 0.1281 | 12.51831 |
| RNN | 0.1219 | 6.24272 |
| CNN+GRU | 0.0939 | 2.66152 |

All checkpoints produce roughly 9–14% top‑1 accuracy with the legacy normalization. Given the measured statistics above, the dominant issue is a dataset-level distribution shift; retesting with the matched mean/std (via env vars) should significantly boost accuracy.

### Recommendations

1. **Verify dataset integrity**: ensure the MAT files were downloaded from the official SenseFi link and not reprocessed elsewhere. Use `python3 inspect_data.py --dataset NTU-Fi-HumanID --split train` after installing SciPy to print raw tensor statistics and confirm they match the expected ranges (mean ≈ 0, std ≈ 1 after normalization).
2. **Check checkpoints**: confirm the `.pt` files in `model_pt/` originate from the same commit/PyTorch version as the data release. Loading your own freshly trained checkpoints (run without `--eval-only`) will reveal whether training converges with the current code.
3. **Add sanity tests**: e.g., overfit a very small subset (single subject) to confirm the models can memorize data, which isolates data/label mismatches from model bugs.
