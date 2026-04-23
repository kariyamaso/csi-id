#!/usr/bin/env python3
"""Build a single Markdown report from `runs/`, `result/`, and `result_figure/`.

This is a convenience script for consolidating experimental outputs into one
readable document.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _to_int(x: Any) -> int | None:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _fmt_float(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:.{digits}f}"


def _fmt_int(x: int | None) -> str:
    return "-" if x is None else str(int(x))


def _fmt_mean_std(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None:
        return "-"
    if std is None:
        return _fmt_float(mean, digits)
    return f"{_fmt_float(mean, digits)} ± {_fmt_float(std, digits)}"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    def esc(s: str) -> str:
        return s.replace("|", "\\|")

    lines = []
    lines.append("| " + " | ".join(esc(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        lines.append("| " + " | ".join(esc(str(x)) for x in r) + " |")
    return "\n".join(lines)


def _collect_run_coverage(runs_dir: Path) -> Dict[str, Dict[str, Any]]:
    # dataset -> {count, seeds(set), models(set)}
    out: Dict[str, Dict[str, Any]] = {}
    for path in runs_dir.rglob("metrics.json"):
        try:
            m = json.loads(path.read_text())
        except Exception:
            continue
        ds = str(m.get("dataset", "")).strip()
        model = str(m.get("model", "")).strip()
        seed = _to_int(m.get("seed"))
        if not ds:
            continue
        stats = out.setdefault(ds, {"runs": 0, "seeds": set(), "models": set()})
        stats["runs"] += 1
        if seed is not None:
            stats["seeds"].add(seed)
        if model:
            stats["models"].add(model)
    return out


def _collect_ablation_overview(ablation_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    keys = [
        "seq_len",
        "pooling",
        "mamba_selective",
        "shuffle_subcarriers",
        "shuffle_antennas",
        "train_fraction",
        "noise",
        "noise_p",
        "val_noise",
        "val_noise_p",
    ]
    by_ds: Dict[str, Dict[str, List[str]]] = {}
    for ds in sorted({str(r.get("dataset", "")).strip() for r in ablation_rows if str(r.get("dataset", "")).strip()}):
        rds = [r for r in ablation_rows if str(r.get("dataset", "")).strip() == ds]
        by_ds[ds] = {}
        for k in keys:
            vals = sorted({str(r.get(k)) for r in rds})
            by_ds[ds][k] = vals
    return by_ds


def _pick_best(rows: List[Dict[str, Any]], key: str, higher_is_better: bool = True) -> Dict[str, Any] | None:
    def v(r: Dict[str, Any]) -> float:
        x = _to_float(r.get(key))
        if x is None:
            return -math.inf if higher_is_better else math.inf
        return x

    if not rows:
        return None
    return sorted(rows, key=v, reverse=higher_is_better)[0]


def _list_images(paths: Iterable[Path]) -> List[Path]:
    out = []
    for p in paths:
        if p.is_file() and p.suffix.lower() in {".png", ".pdf"}:
            out.append(p)
    return sorted(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--result-dir", type=Path, default=Path("result"))
    p.add_argument("--result-figure-dir", type=Path, default=Path("result_figure"))
    p.add_argument(
        "--aggregate-dir",
        type=Path,
        default=Path("result/artifacts/aggregate"),
        help="Directory containing summary.csv/pareto.csv/ablation.csv.",
    )
    p.add_argument("--out", type=Path, default=Path("result/EXPERIMENT_RESULTS.md"))
    args = p.parse_args()

    summary_path = args.aggregate_dir / "summary.csv"
    ablation_path = args.aggregate_dir / "ablation.csv"
    pareto_path = args.aggregate_dir / "pareto.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not ablation_path.is_file():
        raise FileNotFoundError(f"Missing {ablation_path}")
    if not pareto_path.is_file():
        raise FileNotFoundError(f"Missing {pareto_path}")

    summary_rows = _read_csv(summary_path)
    ablation_rows = _read_csv(ablation_path)
    pareto_rows = _read_csv(pareto_path)
    datasets = sorted({str(r.get("dataset", "")).strip() for r in summary_rows if str(r.get("dataset", "")).strip()})

    coverage = _collect_run_coverage(args.runs_dir)
    ablation_overview = _collect_ablation_overview(ablation_rows)

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append(f"# Experiment Results Summary\n")
    lines.append(f"- Generated: {now}")
    lines.append(f"- Runs dir: `{args.runs_dir}`")
    lines.append(f"- Aggregates: `{args.aggregate_dir}` (`summary.csv`, `pareto.csv`, `ablation.csv`)")
    lines.append(f"- Figures (custom): `{args.result_figure_dir}`")
    lines.append("")

    # Key figures for the paper/response-to-reviewers.
    exp_fig_dir = args.result_figure_dir / "EXPERIMENT_RESULTS"
    if exp_fig_dir.is_dir():
        lines.append("## Key Figures (EXPERIMENT_RESULTS)")
        lines.append("")
        for rel, caption in [
            ("FigureA_tradeoff_latency_batch1_2x2.png", "Figure A: Accuracy–Latency–Model Size trade-off (batch=1, log-latency) — 2×2 (NTU-Fi HumanID / NTU-Fi HAR / UT-HAR / Widar)"),
            ("FigureB_params_log_vs_accuracy.png", "Figure B: Params (log) × Accuracy"),
            ("FigureC_peak_gpu_mem_vs_accuracy.png", "Figure C: Peak GPU memory × Accuracy"),
        ]:
            pth = exp_fig_dir / rel
            if pth.is_file():
                lines.append(f"### {caption}")
                lines.append(f"- `{pth}`")
                lines.append(f"![]({pth.as_posix()})")
                lines.append("")

        # Add a compact table answering: “is improvement worth added computation?”
        def _find_best(ds: str, model: str) -> Dict[str, Any] | None:
            ds_rows = [r for r in summary_rows if str(r.get('dataset', '')).strip() == ds and str(r.get('model', '')).strip() == model]
            if not ds_rows:
                return None
            return sorted(ds_rows, key=lambda r: _to_float(r.get("acc_mean")) or -1.0, reverse=True)[0]

        ratio_rows: List[List[str]] = []
        for ds in datasets:
            mamba = _find_best(ds, "Mamba")
            gru = _find_best(ds, "GRU")
            if not mamba or not gru:
                continue
            m_acc = _to_float(mamba.get("acc_mean"))
            g_acc = _to_float(gru.get("acc_mean"))
            m_b1 = _to_float(mamba.get("latency_ms_batch1_mean"))
            g_b1 = _to_float(gru.get("latency_ms_batch1_mean"))
            m_b64 = _to_float(mamba.get("latency_ms_batch64_mean"))
            g_b64 = _to_float(gru.get("latency_ms_batch64_mean"))
            d_acc = None if (m_acc is None or g_acc is None) else (m_acc - g_acc) * 100.0
            r1 = None if (m_b1 is None or g_b1 is None or g_b1 <= 0) else (m_b1 / g_b1)
            r64 = None if (m_b64 is None or g_b64 is None or g_b64 <= 0) else (m_b64 / g_b64)
            ratio_rows.append(
                [
                    ds,
                    _fmt_float((g_acc or 0.0) * 100.0, 2),
                    _fmt_float((m_acc or 0.0) * 100.0, 2),
                    _fmt_float(d_acc, 2),
                    _fmt_float(g_b1, 3),
                    _fmt_float(m_b1, 3),
                    _fmt_float(r1, 1) + "×" if r1 is not None else "-",
                    _fmt_float(g_b64, 3),
                    _fmt_float(m_b64, 3),
                    _fmt_float(r64, 1) + "×" if r64 is not None else "-",
                ]
            )
        if ratio_rows:
            lines.append("### Mamba vs GRU (latency cost vs accuracy gain)")
            lines.append(_md_table(
                ["dataset", "GRU acc(%)", "Mamba acc(%)", "Δacc(%)", "GRU b1(ms)", "Mamba b1(ms)", "b1 ratio", "GRU b64(ms)", "Mamba b64(ms)", "b64 ratio"],
                ratio_rows,
            ))
            lines.append("")

    lines.append("## Coverage")
    cov_rows: List[List[str]] = []
    for ds in datasets:
        stats = coverage.get(ds, {"runs": 0, "seeds": set(), "models": set()})
        seeds = sorted(stats.get("seeds", set()))
        seed_str = "-" if not seeds else f"{seeds[0]}..{seeds[-1]} (n={len(seeds)})"
        cov_rows.append([ds, str(stats.get("runs", 0)), str(len(stats.get("models", set()))), seed_str])
    lines.append(_md_table(["dataset", "runs (metrics.json)", "models", "seeds"], cov_rows))
    lines.append("")

    lines.append("## Ablation / Common Settings")
    for ds in datasets:
        lines.append(f"### {ds}")
        kv = ablation_overview.get(ds, {})
        rows = []
        for k in [
            "seq_len",
            "pooling",
            "mamba_selective",
            "shuffle_subcarriers",
            "shuffle_antennas",
            "train_fraction",
            "noise",
            "noise_p",
            "val_noise",
        ]:
            vals = kv.get(k, [])
            display = ", ".join(vals) if vals else "-"
            rows.append([k, display])
        lines.append(_md_table(["key", "value(s)"], rows))
        lines.append("")

    lines.append("## Dataset Results (from summary.csv)")
    lines.append("Metrics: `acc_mean/std`, `macro_f1_mean/std`, `macro_recall_mean/std`, `latency_ms_batch1_mean`, `latency_ms_batch64_mean`, `params_total_mean`.")
    lines.append("")

    for ds in datasets:
        ds_rows = [r for r in summary_rows if str(r.get("dataset", "")).strip() == ds]
        if not ds_rows:
            continue

        lines.append(f"### {ds}")
        best_acc = _pick_best(ds_rows, "acc_mean", higher_is_better=True)
        best_f1 = _pick_best(ds_rows, "macro_f1_mean", higher_is_better=True)
        fast_b1 = _pick_best(ds_rows, "latency_ms_batch1_mean", higher_is_better=False)
        fast_b64 = _pick_best(ds_rows, "latency_ms_batch64_mean", higher_is_better=False)

        def one_line(label: str, r: Dict[str, Any] | None) -> str:
            if not r:
                return f"- {label}: -"
            return (
                f"- {label}: **{r.get('model')}** (`{r.get('variant')}`) "
                f"acc={_fmt_float(_to_float(r.get('acc_mean')), 6)}, "
                f"b1={_fmt_float(_to_float(r.get('latency_ms_batch1_mean')), 3)}ms, "
                f"b64={_fmt_float(_to_float(r.get('latency_ms_batch64_mean')), 3)}ms, "
                f"params={_fmt_int(_to_int(r.get('params_total_mean')))}"
            )

        lines.append(one_line("Best acc", best_acc))
        if best_f1 and (best_acc is None or best_f1.get("model") != best_acc.get("model")):
            lines.append(one_line("Best macro-F1", best_f1))
        lines.append(one_line("Fastest (batch=1)", fast_b1))
        lines.append(one_line("Fastest (batch=64)", fast_b64))

        # Mamba vs GRU latency ratios if available
        mamba = next((r for r in ds_rows if str(r.get("model", "")).strip() == "Mamba"), None)
        gru = next((r for r in ds_rows if str(r.get("model", "")).strip() == "GRU"), None)
        if mamba and gru:
            m1 = _to_float(mamba.get("latency_ms_batch1_mean"))
            m64 = _to_float(mamba.get("latency_ms_batch64_mean"))
            g1 = _to_float(gru.get("latency_ms_batch1_mean"))
            g64 = _to_float(gru.get("latency_ms_batch64_mean"))
            if m1 and g1 and m64 and g64 and g1 > 0 and g64 > 0:
                lines.append(
                    f"- Mamba vs GRU latency ratio: batch1 **{_fmt_float(m1/g1, 1)}×**, batch64 **{_fmt_float(m64/g64, 1)}×**"
                )
        lines.append("")

        table_rows: List[List[str]] = []
        for r in sorted(ds_rows, key=lambda x: (_to_float(x.get("acc_mean")) or -1.0), reverse=True):
            table_rows.append(
                [
                    str(r.get("model", "")),
                    str(r.get("variant", "")),
                    _fmt_mean_std(_to_float(r.get("acc_mean")), _to_float(r.get("acc_std")), digits=6),
                    _fmt_mean_std(_to_float(r.get("macro_f1_mean")), _to_float(r.get("macro_f1_std")), digits=6),
                    _fmt_mean_std(_to_float(r.get("macro_recall_mean")), _to_float(r.get("macro_recall_std")), digits=6),
                    _fmt_int(_to_int(r.get("params_total_mean"))),
                    _fmt_float(_to_float(r.get("latency_ms_batch1_mean")), 3),
                    _fmt_float(_to_float(r.get("latency_ms_batch64_mean")), 3),
                    _fmt_float(_to_float(r.get("peak_gpu_mem_mb_mean")), 1),
                ]
            )
        lines.append(
            _md_table(
                [
                    "model",
                    "variant",
                    "acc (mean±std)",
                    "macro_f1 (mean±std)",
                    "macro_recall (mean±std)",
                    "params_total",
                    "lat_b1_ms",
                    "lat_b64_ms",
                    "peak_gpu_mem_mb",
                ],
                table_rows,
            )
        )
        lines.append("")

        # Figures (aggregate plots + custom plots)
        agg_plot_dir = args.aggregate_dir / "plots" / ds
        custom_ds_dir = args.result_figure_dir / ds

        if agg_plot_dir.is_dir():
            lines.append("#### Plots (aggregate)")
            for name in ["accuracy_bar.png", "pareto_batch1.png", "pareto_batch64.png"]:
                pth = agg_plot_dir / name
                if pth.is_file():
                    lines.append(f"- `{pth}`")
                    if pth.suffix.lower() == ".png":
                        lines.append(f"  ![]({pth.as_posix()})")
            lines.append("")

        if custom_ds_dir.is_dir():
            lines.append("#### Plots (result_figure)")
            for name in ["accuracy_bar_meanstd.png", "pareto_batch1.png", "pareto_batch64.png"]:
                pth = custom_ds_dir / name
                if pth.is_file():
                    lines.append(f"- `{pth}`")
                    lines.append(f"  ![]({pth.as_posix()})")
            # Confusion matrices: include Mamba if present, plus count.
            conf_root = custom_ds_dir / "confusion"
            if conf_root.is_dir():
                mamba_png = next(conf_root.rglob("confusion_Mamba.png"), None)
                if mamba_png and mamba_png.is_file():
                    lines.append(f"- `{mamba_png}`")
                    lines.append(f"  ![]({mamba_png.as_posix()})")
                conf_pngs = list(conf_root.rglob("confusion_*.png"))
                lines.append(f"- Confusion matrices: `{conf_root}` (png files: {len(conf_pngs)})")
            lines.append("")

    lines.append("## Repro / Regeneration")
    lines.append("```bash")
    lines.append("# 1) Aggregate runs into CSVs (public pipeline)")
    lines.append("python3 public/scripts/aggregate_results.py --runs-dir runs --out-dir result/artifacts/aggregate")
    lines.append("")
    lines.append("# 2) Plot from aggregate CSVs (public)")
    lines.append("python3 public/scripts/plot_aggregate.py --in-dir result/artifacts/aggregate --out-dir result/artifacts/aggregate/plots")
    lines.append("")
    lines.append("# 3) Plot confusion averages (public)")
    lines.append("python3 public/scripts/plot_confusion_from_runs.py --runs-dir runs --out-dir result/artifacts/figures")
    lines.append("")
    lines.append("# 4) Plot unified figures (non-public helper)")
    lines.append("python3 scripts/plot_result_figures.py --runs-dir runs --aggregate-dir result/artifacts/aggregate --out-dir result_figure --no-point-labels")
    lines.append("")
    lines.append("# 5) Rebuild this Markdown report")
    lines.append("python3 scripts/build_results_markdown.py --out result/EXPERIMENT_RESULTS.md")
    lines.append("```")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report -> {args.out}")


if __name__ == "__main__":
    main()
