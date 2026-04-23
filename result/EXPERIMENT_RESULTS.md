# Experiment Results Summary

- Generated: 2026-02-03 17:44:12
- Runs dir: `runs`
- Aggregates: `result/artifacts/aggregate` (`summary.csv`, `pareto.csv`, `ablation.csv`)
- Figures (custom): `result_figure`

## Key Figures (EXPERIMENT_RESULTS)

### Figure A: Accuracy–Latency–Model Size trade-off (batch=1, log-latency) — 2×2 (NTU-Fi HumanID / NTU-Fi HAR / UT-HAR / Widar)
- `result_figure/EXPERIMENT_RESULTS/FigureA_tradeoff_latency_batch1_2x2.png`
![](result_figure/EXPERIMENT_RESULTS/FigureA_tradeoff_latency_batch1_2x2.png)

### Figure B: Params (log) × Accuracy
- `result_figure/EXPERIMENT_RESULTS/FigureB_params_log_vs_accuracy.png`
![](result_figure/EXPERIMENT_RESULTS/FigureB_params_log_vs_accuracy.png)

### Figure C: Peak GPU memory × Accuracy
- `result_figure/EXPERIMENT_RESULTS/FigureC_peak_gpu_mem_vs_accuracy.png`
![](result_figure/EXPERIMENT_RESULTS/FigureC_peak_gpu_mem_vs_accuracy.png)

### Mamba vs GRU (latency cost vs accuracy gain)
| dataset | GRU acc(%) | Mamba acc(%) | Δacc(%) | GRU b1(ms) | Mamba b1(ms) | b1 ratio | GRU b64(ms) | Mamba b64(ms) | b64 ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NTU-Fi-HumanID | 97.31 | 99.97 | 2.65 | 0.236 | 1.243 | 5.3× | 0.509 | 10.922 | 21.5× |
| NTU-Fi_HAR | 99.13 | 99.89 | 0.76 | 0.244 | 1.258 | 5.1× | 0.513 | 10.910 | 21.3× |
| UT_HAR_data | 80.05 | 96.12 | 16.07 | 0.138 | 1.386 | 10.1× | 0.172 | 5.726 | 33.3× |
| Widar | 62.63 | 68.88 | 6.25 | 0.083 | 1.369 | 16.5× | 0.106 | 2.263 | 21.4× |

## Coverage
| dataset | runs (metrics.json) | models | seeds |
| --- | --- | --- | --- |
| NTU-Fi-HumanID | 120 | 12 | 0..9 (n=10) |
| NTU-Fi_HAR | 120 | 12 | 0..9 (n=10) |
| UT_HAR_data | 120 | 12 | 0..9 (n=10) |
| Widar | 120 | 12 | 0..9 (n=10) |

## Ablation / Common Settings
### NTU-Fi-HumanID
| key | value(s) |
| --- | --- |
| seq_len | 500 |
| pooling | mean |
| mamba_selective | on |
| shuffle_subcarriers | False |
| shuffle_antennas | False |
| train_fraction | 1.0 |
| noise | none |
| noise_p | 0.0 |
| val_noise | none |

### NTU-Fi_HAR
| key | value(s) |
| --- | --- |
| seq_len | 500 |
| pooling | mean |
| mamba_selective | on |
| shuffle_subcarriers | False |
| shuffle_antennas | False |
| train_fraction | 1.0 |
| noise | none |
| noise_p | 0.0 |
| val_noise | none |

### UT_HAR_data
| key | value(s) |
| --- | --- |
| seq_len | 500 |
| pooling | mean |
| mamba_selective | on |
| shuffle_subcarriers | False |
| shuffle_antennas | False |
| train_fraction | 1.0 |
| noise | none |
| noise_p | 0.0 |
| val_noise | none |

### Widar
| key | value(s) |
| --- | --- |
| seq_len | 500 |
| pooling | mean |
| mamba_selective | on |
| shuffle_subcarriers | False |
| shuffle_antennas | False |
| train_fraction | 1.0 |
| noise | none |
| noise_p | 0.0 |
| val_noise | none |

## Dataset Results (from summary.csv)
Metrics: `acc_mean/std`, `macro_f1_mean/std`, `macro_recall_mean/std`, `latency_ms_batch1_mean`, `latency_ms_batch64_mean`, `params_total_mean`.

### NTU-Fi-HumanID
- Best acc: **Mamba** (`selective_on_pool_mean_seq500`) acc=0.999660, b1=1.243ms, b64=10.922ms, params=2171534
- Fastest (batch=1): **LeNet** (`seq500`) acc=0.964286, b1=0.112ms, b64=0.413ms, params=477614
- Fastest (batch=64): **RNN** (`seq500`) acc=0.871429, b1=0.166ms, b64=0.359ms, params=27022
- Mamba vs GRU latency ratio: batch1 **5.3×**, batch64 **21.5×**

| model | variant | acc (mean±std) | macro_f1 (mean±std) | macro_recall (mean±std) | params_total | lat_b1_ms | lat_b64_ms | peak_gpu_mem_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mamba | selective_on_pool_mean_seq500 | 0.999660 ± 0.001076 | 0.999660 ± 0.001076 | 0.999660 ± 0.001076 | 2171534 | 1.243 | 10.922 | 572.2 |
| GRU | seq500 | 0.973129 ± 0.008701 | 0.972845 ± 0.008949 | 0.973129 ± 0.008701 | 79246 | 0.236 | 0.509 | 301.1 |
| LeNet | seq500 | 0.964286 ± 0.009654 | 0.964028 ± 0.009639 | 0.964286 ± 0.009654 | 477614 | 0.112 | 0.413 | 92.6 |
| LSTM | seq500 | 0.921429 ± 0.028069 | 0.919150 ± 0.030531 | 0.921429 ± 0.028069 | 105358 | 0.296 | 0.600 | 339.6 |
| ResNet18 | seq500 | 0.910204 ± 0.030934 | 0.909059 ± 0.031315 | 0.910204 ± 0.030934 | 11188322 | 0.934 | 1.646 | 128.6 |
| MLP | seq500 | 0.908844 ± 0.013492 | 0.908263 ± 0.013244 | 0.908844 ± 0.013492 | 175238030 | 0.502 | 0.819 | 744.0 |
| BiLSTM | seq500 | 0.907483 ± 0.027623 | 0.904916 ± 0.029262 | 0.907483 ± 0.027623 | 209806 | 0.537 | 0.907 | 427.6 |
| RNN | seq500 | 0.871429 ± 0.014143 | 0.866835 ± 0.015477 | 0.871429 ± 0.014143 | 27022 | 0.166 | 0.359 | 221.7 |
| ResNet50 | seq500 | 0.765306 ± 0.125221 | 0.758107 ± 0.133116 | 0.765306 ± 0.125221 | 23566946 | 2.148 | 3.364 | 182.1 |
| ResNet101 | seq500 | 0.729592 ± 0.246349 | 0.720026 ± 0.265011 | 0.729592 ± 0.246349 | 42585186 | 3.982 | 6.109 | 255.2 |
| CNN+GRU | seq500 | 0.696599 ± 0.126850 | 0.635463 ± 0.151051 | 0.696599 ± 0.126850 | 58622 | 0.451 | 1.272 | 391.9 |
| ViT | seq500 | 0.666667 ± 0.046222 | 0.653191 ± 0.047559 | 0.666667 ± 0.046222 | 836339 | 0.508 | 6.143 | 2409.0 |

#### Plots (aggregate)
- `result/artifacts/aggregate/plots/NTU-Fi-HumanID/accuracy_bar.png`
  ![](result/artifacts/aggregate/plots/NTU-Fi-HumanID/accuracy_bar.png)
- `result/artifacts/aggregate/plots/NTU-Fi-HumanID/pareto_batch1.png`
  ![](result/artifacts/aggregate/plots/NTU-Fi-HumanID/pareto_batch1.png)
- `result/artifacts/aggregate/plots/NTU-Fi-HumanID/pareto_batch64.png`
  ![](result/artifacts/aggregate/plots/NTU-Fi-HumanID/pareto_batch64.png)

#### Plots (result_figure)
- `result_figure/NTU-Fi-HumanID/accuracy_bar_meanstd.png`
  ![](result_figure/NTU-Fi-HumanID/accuracy_bar_meanstd.png)
- `result_figure/NTU-Fi-HumanID/pareto_batch1.png`
  ![](result_figure/NTU-Fi-HumanID/pareto_batch1.png)
- `result_figure/NTU-Fi-HumanID/pareto_batch64.png`
  ![](result_figure/NTU-Fi-HumanID/pareto_batch64.png)
- `result_figure/NTU-Fi-HumanID/confusion/selective_on_pool_mean_seq500/confusion_Mamba.png`
  ![](result_figure/NTU-Fi-HumanID/confusion/selective_on_pool_mean_seq500/confusion_Mamba.png)
- Confusion matrices: `result_figure/NTU-Fi-HumanID/confusion` (png files: 12)

### NTU-Fi_HAR
- Best acc: **Mamba** (`selective_on_pool_mean_seq500`) acc=0.998864, b1=1.258ms, b64=10.910ms, params=2170502
- Fastest (batch=1): **LeNet** (`seq500`) acc=0.990909, b1=0.120ms, b64=0.424ms, params=476582
- Fastest (batch=64): **RNN** (`seq500`) acc=0.884848, b1=0.177ms, b64=0.380ms, params=26502
- Mamba vs GRU latency ratio: batch1 **5.1×**, batch64 **21.3×**

| model | variant | acc (mean±std) | macro_f1 (mean±std) | macro_recall (mean±std) | params_total | lat_b1_ms | lat_b64_ms | peak_gpu_mem_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mamba | selective_on_pool_mean_seq500 | 0.998864 ± 0.002557 | 0.998863 ± 0.002558 | 0.998864 ± 0.002557 | 2170502 | 1.258 | 10.910 | 572.2 |
| GRU | seq500 | 0.991288 ± 0.009950 | 0.991277 ± 0.009969 | 0.991288 ± 0.009950 | 78726 | 0.244 | 0.513 | 301.1 |
| LeNet | seq500 | 0.990909 ± 0.008025 | 0.990899 ± 0.008037 | 0.990909 ± 0.008025 | 476582 | 0.120 | 0.424 | 92.6 |
| BiLSTM | seq500 | 0.987500 ± 0.010721 | 0.987433 ± 0.010815 | 0.987500 ± 0.010721 | 209286 | 0.535 | 0.907 | 427.5 |
| LSTM | seq500 | 0.985606 ± 0.010832 | 0.985583 ± 0.010915 | 0.985606 ± 0.010832 | 104838 | 0.308 | 0.600 | 339.6 |
| MLP | seq500 | 0.975000 ± 0.016482 | 0.974974 ± 0.016512 | 0.975000 ± 0.016482 | 175236998 | 0.504 | 0.821 | 744.0 |
| ResNet18 | seq500 | 0.939015 ± 0.037856 | 0.938703 ± 0.038451 | 0.939015 ± 0.037856 | 11184218 | 0.904 | 1.647 | 128.6 |
| ResNet50 | seq500 | 0.913258 ± 0.066775 | 0.910842 ± 0.070706 | 0.913258 ± 0.066775 | 23550554 | 2.000 | 3.363 | 182.1 |
| ViT | seq500 | 0.913258 ± 0.036044 | 0.911770 ± 0.037118 | 0.913258 ± 0.036044 | 834531 | 0.546 | 6.142 | 2409.0 |
| RNN | seq500 | 0.884848 ± 0.012901 | 0.884107 ± 0.012997 | 0.884848 ± 0.012901 | 26502 | 0.177 | 0.380 | 221.7 |
| ResNet101 | seq500 | 0.882576 ± 0.079555 | 0.881813 ± 0.079499 | 0.882576 ± 0.079555 | 42568794 | 4.044 | 6.109 | 255.1 |
| CNN+GRU | seq500 | 0.803788 ± 0.130788 | 0.787695 ± 0.157519 | 0.803788 ± 0.130788 | 57590 | 0.443 | 1.272 | 391.9 |

#### Plots (aggregate)
- `result/artifacts/aggregate/plots/NTU-Fi_HAR/accuracy_bar.png`
  ![](result/artifacts/aggregate/plots/NTU-Fi_HAR/accuracy_bar.png)
- `result/artifacts/aggregate/plots/NTU-Fi_HAR/pareto_batch1.png`
  ![](result/artifacts/aggregate/plots/NTU-Fi_HAR/pareto_batch1.png)
- `result/artifacts/aggregate/plots/NTU-Fi_HAR/pareto_batch64.png`
  ![](result/artifacts/aggregate/plots/NTU-Fi_HAR/pareto_batch64.png)

#### Plots (result_figure)
- `result_figure/NTU-Fi_HAR/accuracy_bar_meanstd.png`
  ![](result_figure/NTU-Fi_HAR/accuracy_bar_meanstd.png)
- `result_figure/NTU-Fi_HAR/pareto_batch1.png`
  ![](result_figure/NTU-Fi_HAR/pareto_batch1.png)
- `result_figure/NTU-Fi_HAR/pareto_batch64.png`
  ![](result_figure/NTU-Fi_HAR/pareto_batch64.png)
- `result_figure/NTU-Fi_HAR/confusion/selective_on_pool_mean_seq500/confusion_Mamba.png`
  ![](result_figure/NTU-Fi_HAR/confusion/selective_on_pool_mean_seq500/confusion_Mamba.png)
- Confusion matrices: `result_figure/NTU-Fi_HAR/confusion` (png files: 12)

### UT_HAR_data
- Best acc: **LeNet** (`seq500`) acc=0.969578, b1=0.124ms, b64=0.274ms, params=295655
- Fastest (batch=1): **MLP** (`seq500`) acc=0.805924, b1=0.093ms, b64=0.205ms, params=23173127
- Fastest (batch=64): **RNN** (`seq500`) acc=0.503213, b1=0.116ms, b64=0.130ms, params=10439
- Mamba vs GRU latency ratio: batch1 **10.1×**, batch64 **33.3×**

| model | variant | acc (mean±std) | macro_f1 (mean±std) | macro_recall (mean±std) | params_total | lat_b1_ms | lat_b64_ms | peak_gpu_mem_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LeNet | seq500 | 0.969578 ± 0.006438 | 0.956622 ± 0.006589 | 0.956874 ± 0.006839 | 295655 | 0.124 | 0.274 | 153.6 |
| Mamba | selective_on_pool_mean_seq500 | 0.961245 ± 0.011051 | 0.942992 ± 0.016007 | 0.941506 ± 0.016912 | 2106119 | 1.386 | 5.726 | 301.9 |
| CNN+GRU | seq500 | 0.916365 ± 0.041413 | 0.898236 ± 0.041262 | 0.890832 ± 0.046812 | 1429645 | 0.373 | 0.747 | 186.8 |
| ResNet18 | seq500 | 0.878614 ± 0.157939 | 0.839260 ± 0.211701 | 0.845201 ± 0.204169 | 11182142 | 0.932 | 0.997 | 92.2 |
| MLP | seq500 | 0.805924 ± 0.075968 | 0.771143 ± 0.088500 | 0.763120 ± 0.091019 | 23173127 | 0.093 | 0.205 | 127.5 |
| GRU | seq500 | 0.800502 ± 0.054320 | 0.765305 ± 0.060436 | 0.751339 ± 0.064139 | 30407 | 0.138 | 0.172 | 124.2 |
| ResNet101 | seq500 | 0.788554 ± 0.112546 | 0.745299 ± 0.125108 | 0.748678 ± 0.119301 | 42568254 | 4.072 | 5.688 | 218.6 |
| ResNet50 | seq500 | 0.787349 ± 0.224922 | 0.733131 ± 0.277294 | 0.741422 ± 0.252443 | 23550014 | 2.158 | 2.861 | 145.6 |
| ViT | seq500 | 0.737851 ± 0.110908 | 0.706728 ± 0.118357 | 0.707093 ± 0.118220 | 10575007 | 0.565 | 1.022 | 141.7 |
| RNN | seq500 | 0.503213 ± 0.045139 | 0.420923 ± 0.076202 | 0.418215 ± 0.060267 | 10439 | 0.116 | 0.130 | 86.0 |
| BiLSTM | seq500 | 0.415361 ± 0.086288 | 0.214615 ± 0.134112 | 0.262993 ± 0.118780 | 80327 | 0.283 | 0.379 | 192.2 |
| LSTM | seq500 | 0.385542 ± 0.044361 | 0.192053 ± 0.062440 | 0.242691 ± 0.056012 | 40391 | 0.170 | 0.212 | 144.1 |

#### Plots (aggregate)
- `result/artifacts/aggregate/plots/UT_HAR_data/accuracy_bar.png`
  ![](result/artifacts/aggregate/plots/UT_HAR_data/accuracy_bar.png)
- `result/artifacts/aggregate/plots/UT_HAR_data/pareto_batch1.png`
  ![](result/artifacts/aggregate/plots/UT_HAR_data/pareto_batch1.png)
- `result/artifacts/aggregate/plots/UT_HAR_data/pareto_batch64.png`
  ![](result/artifacts/aggregate/plots/UT_HAR_data/pareto_batch64.png)

#### Plots (result_figure)
- `result_figure/UT_HAR_data/accuracy_bar_meanstd.png`
  ![](result_figure/UT_HAR_data/accuracy_bar_meanstd.png)
- `result_figure/UT_HAR_data/pareto_batch1.png`
  ![](result_figure/UT_HAR_data/pareto_batch1.png)
- `result_figure/UT_HAR_data/pareto_batch64.png`
  ![](result_figure/UT_HAR_data/pareto_batch64.png)
- `result_figure/UT_HAR_data/confusion/selective_on_pool_mean_seq500/confusion_Mamba.png`
  ![](result_figure/UT_HAR_data/confusion/selective_on_pool_mean_seq500/confusion_Mamba.png)
- Confusion matrices: `result_figure/UT_HAR_data/confusion` (png files: 12)

### Widar
- Best acc: **LeNet** (`seq500`) acc=0.696482, b1=0.140ms, b64=0.178ms, params=298838
- Fastest (batch=1): **MLP** (`seq500`) acc=0.656383, b1=0.077ms, b64=0.126ms, params=9146262
- Fastest (batch=64): **RNN** (`seq500`) acc=0.470422, b1=0.082ms, b64=0.103ms, params=31254
- Mamba vs GRU latency ratio: batch1 **16.5×**, batch64 **21.4×**

| model | variant | acc (mean±std) | macro_f1 (mean±std) | macro_recall (mean±std) | params_total | lat_b1_ms | lat_b64_ms | peak_gpu_mem_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LeNet | seq500 | 0.696482 ± 0.006293 | 0.640427 ± 0.006301 | 0.627721 ± 0.009734 | 298838 | 0.140 | 0.178 | 54.6 |
| Mamba | selective_on_pool_mean_seq500 | 0.688804 ± 0.010753 | 0.624341 ± 0.018093 | 0.605175 ± 0.023593 | 2197654 | 1.369 | 2.263 | 79.2 |
| ViT | seq500 | 0.671831 ± 0.006108 | 0.628226 ± 0.007488 | 0.617294 ± 0.008644 | 95222 | 0.526 | 0.520 | 94.3 |
| ResNet18 | seq500 | 0.667511 ± 0.025115 | 0.601615 ± 0.039548 | 0.592184 ± 0.041613 | 11192375 | 0.956 | 1.075 | 88.8 |
| MLP | seq500 | 0.656383 ± 0.009521 | 0.600244 ± 0.015169 | 0.576962 ± 0.026576 | 9146262 | 0.077 | 0.126 | 70.5 |
| LSTM | seq500 | 0.627332 ± 0.004754 | 0.513004 ± 0.013921 | 0.492485 ± 0.016385 | 120726 | 0.092 | 0.112 | 56.1 |
| GRU | seq500 | 0.626347 ± 0.006559 | 0.499376 ± 0.015413 | 0.479630 ± 0.016668 | 90902 | 0.083 | 0.106 | 54.1 |
| CNN+GRU | seq500 | 0.625487 ± 0.013604 | 0.363577 ± 0.017225 | 0.383036 ± 0.015983 | 92238 | 0.306 | 0.329 | 56.9 |
| ResNet50 | seq500 | 0.623997 ± 0.039522 | 0.544373 ± 0.050928 | 0.533896 ± 0.049467 | 23583287 | 2.147 | 2.825 | 142.7 |
| BiLSTM | seq500 | 0.620307 ± 0.008252 | 0.497557 ± 0.018206 | 0.478538 ± 0.022468 | 240022 | 0.107 | 0.120 | 69.0 |
| ResNet101 | seq500 | 0.555214 ± 0.124170 | 0.450013 ± 0.164387 | 0.447017 ± 0.150462 | 42601527 | 4.138 | 5.677 | 216.2 |
| RNN | seq500 | 0.470422 ± 0.007958 | 0.294902 ± 0.008633 | 0.291634 ± 0.007955 | 31254 | 0.082 | 0.103 | 50.2 |

#### Plots (aggregate)
- `result/artifacts/aggregate/plots/Widar/accuracy_bar.png`
  ![](result/artifacts/aggregate/plots/Widar/accuracy_bar.png)
- `result/artifacts/aggregate/plots/Widar/pareto_batch1.png`
  ![](result/artifacts/aggregate/plots/Widar/pareto_batch1.png)
- `result/artifacts/aggregate/plots/Widar/pareto_batch64.png`
  ![](result/artifacts/aggregate/plots/Widar/pareto_batch64.png)

#### Plots (result_figure)
- `result_figure/Widar/accuracy_bar_meanstd.png`
  ![](result_figure/Widar/accuracy_bar_meanstd.png)
- `result_figure/Widar/pareto_batch1.png`
  ![](result_figure/Widar/pareto_batch1.png)
- `result_figure/Widar/pareto_batch64.png`
  ![](result_figure/Widar/pareto_batch64.png)
- `result_figure/Widar/confusion/selective_on_pool_mean_seq500/confusion_Mamba.png`
  ![](result_figure/Widar/confusion/selective_on_pool_mean_seq500/confusion_Mamba.png)
- Confusion matrices: `result_figure/Widar/confusion` (png files: 12)

## Repro / Regeneration
```bash
# 1) Aggregate runs into CSVs (public pipeline)
python3 public/scripts/aggregate_results.py --runs-dir runs --out-dir result/artifacts/aggregate

# 2) Plot from aggregate CSVs (public)
python3 public/scripts/plot_aggregate.py --in-dir result/artifacts/aggregate --out-dir result/artifacts/aggregate/plots

# 3) Plot confusion averages (public)
python3 public/scripts/plot_confusion_from_runs.py --runs-dir runs --out-dir result/artifacts/figures

# 4) Plot unified figures (non-public helper)
python3 scripts/plot_result_figures.py --runs-dir runs --aggregate-dir result/artifacts/aggregate --out-dir result_figure --no-point-labels

# 5) Rebuild this Markdown report
python3 scripts/build_results_markdown.py --out result/EXPERIMENT_RESULTS.md
```
