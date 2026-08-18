#!/usr/bin/env python3
"""Compare completed Erlang-k ablation runs stored in Weights & Biases."""

import argparse
import csv
import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import wandb


DEFAULT_BUDGETS = (1, 2, 4, 8, 16)
# val/ppl is comparable across k (standard SUBS NLL on absorbed positions), so it
# is the primary low-noise separator. sample_entropy guards against a gen_ppl
# "win" that is really mode collapse.
BASE_METRICS = ("val/nll", "val/ppl", "val/sample_entropy", "val/gen_ppl")
# Metrics used for the significance verdict (lower is better for both).
PRIMARY_METRICS = ("val/ppl", "val/gen_ppl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the latest W&B runs produced by " "scripts/run_erlang_ablation.sh."
        )
    )
    parser.add_argument(
        "--project",
        default="duo",
        help="W&B project name, or entity/project (default: duo).",
    )
    parser.add_argument(
        "--entity",
        help="W&B entity. By default, use the logged-in account's entity.",
    )
    parser.add_argument(
        "--data",
        default="openwebtext-split",
        help="Dataset suffix used in the run names (default: openwebtext-split).",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Erlang shapes to compare (default: 1 2 4).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to aggregate over. Default: auto-detect every seed found.",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help="Few-step gen-ppl budgets to include (default: 1 2 4 8 16).",
    )
    parser.add_argument(
        "--include-unfinished",
        action="store_true",
        help="Allow the latest running or failed run when no finished run exists.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Also write the comparison rows to this CSV path.",
    )
    return parser.parse_args()


def _project_path(api: Any, project: str, entity: str | None) -> str:
    if "/" in project:
        if entity is not None:
            raise ValueError("Use either --entity or entity/project, not both.")
        return project
    resolved_entity = entity or api.default_entity
    if not resolved_entity:
        raise ValueError(
            "Could not infer your W&B entity; pass --entity or "
            "--project entity/project."
        )
    return f"{resolved_entity}/{project}"


def _run_names(ks: Sequence[int], data: str) -> dict[int, str]:
    _validate_positive_unique(ks, "Erlang k values")
    return {k: f"erlang-k{k}-{data}" for k in ks}


def _validate_positive_unique(values: Sequence[int], label: str) -> None:
    if any(value < 1 for value in values):
        raise ValueError(f"{label} must be positive.")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique.")


def _select_runs(
    runs: Iterable[Any],
    expected_names: Mapping[int, str],
    include_unfinished: bool,
) -> tuple[dict[int, Any], list[int]]:
    """Select the newest eligible exact-name match for every requested k."""
    by_name: dict[str, list[Any]] = {name: [] for name in expected_names.values()}
    for run in runs:
        if run.name in by_name:
            by_name[run.name].append(run)

    selected = {}
    unfinished_fallbacks = []
    for k, name in expected_names.items():
        candidates = sorted(
            by_name[name], key=lambda run: run.created_at or "", reverse=True
        )
        finished = [run for run in candidates if run.state == "finished"]
        if finished:
            selected[k] = finished[0]
        elif include_unfinished and candidates:
            selected[k] = candidates[0]
            unfinished_fallbacks.append(k)
    return selected, unfinished_fallbacks


def _summary_dict(run: Any) -> dict[str, Any]:
    summary = run.summary
    if hasattr(summary, "_json_dict"):
        return dict(summary._json_dict)
    return dict(summary)


def _finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return value


def _metric_names(budgets: Sequence[int]) -> list[str]:
    _validate_positive_unique(budgets, "Generation budgets")
    return [*BASE_METRICS, *(f"val/gen_ppl@{budget}step" for budget in budgets)]


def _comparison_rows(
    selected: Mapping[int, Any], budgets: Sequence[int]
) -> list[dict[str, Any]]:
    metrics = _metric_names(budgets)
    rows = []
    for k, run in sorted(selected.items()):
        summary = _summary_dict(run)
        row = {
            "k": k,
            "run": run.name,
            "state": run.state,
            "wandb_step": summary.get("_step"),
            "url": run.url,
        }
        row.update({metric: _finite_number(summary.get(metric)) for metric in metrics})
        rows.append(row)

    baseline = next((row for row in rows if row["k"] == 1), None)
    for row in rows:
        for metric in metrics:
            delta_name = f"delta_pct/{metric}"
            baseline_value = baseline and baseline[metric]
            value = row[metric]
            if baseline_value in (None, 0) or value is None:
                row[delta_name] = None
            else:
                row[delta_name] = 100.0 * (value / baseline_value - 1.0)
    return rows


def _format_value(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "--"
    if percent:
        return f"{value:+.2f}%"
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 1000:
        return f"{value:.1f}"
    return f"{value:.4f}"


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render(row: Sequence[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(row, widths))

    print(render(headers))
    print(render(tuple("-" * width for width in widths)))
    for row in rows:
        print(render(row))


def _print_comparison(
    rows: Sequence[Mapping[str, Any]], budgets: Sequence[int]
) -> None:
    raw_headers = ["k", "state", "W&B step", "val/nll", "val/ppl", "gen/default"]
    raw_headers.extend(f"gen/{budget}" for budget in budgets)
    raw_rows = []
    for row in rows:
        raw_rows.append(
            [
                str(row["k"]),
                str(row["state"]),
                _format_value(row["wandb_step"]),
                _format_value(row["val/nll"]),
                _format_value(row["val/ppl"]),
                _format_value(row["val/gen_ppl"]),
                *(
                    _format_value(row[f"val/gen_ppl@{budget}step"])
                    for budget in budgets
                ),
            ]
        )
    _print_table(raw_headers, raw_rows)

    if any(row["k"] == 1 for row in rows):
        print("\nRelative generative perplexity vs k=1 (negative is better):")
        delta_metrics = [
            "val/gen_ppl",
            *(f"val/gen_ppl@{budget}step" for budget in budgets),
        ]
        delta_headers = ["k", "gen/default", *(f"gen/{b}" for b in budgets)]
        delta_rows = [
            [
                str(row["k"]),
                *(
                    _format_value(row[f"delta_pct/{metric}"], percent=True)
                    for metric in delta_metrics
                ),
            ]
            for row in rows
        ]
        _print_table(delta_headers, delta_rows)

    print("\nRuns:")
    for row in rows:
        print(f"  k={row['k']}: {row['url']}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Seed-aware collection and aggregation.                                       #
# --------------------------------------------------------------------------- #

def _run_regex(data: str) -> re.Pattern[str]:
    """Matches erlang-k<k>-<data> and erlang-k<k>-<data>-s<seed> (legacy: no seed)."""
    return re.compile(rf"^erlang-k(\d+)-{re.escape(data)}(?:-s(\d+))?$")


def _collect_seed_runs(
    runs: Iterable[Any],
    ks: Sequence[int],
    data: str,
    seeds: Sequence[int] | None,
    include_unfinished: bool,
) -> tuple[dict[tuple[int, int | None], Any], list[dict[str, Any]]]:
    """Newest run per (k, seed), plus a roster of every (k, seed) and its state.

    A (k, seed) with runs but none finished is kept as None unless
    include_unfinished, and always reported in the roster, so a crashed or
    still-syncing run shows its state instead of silently blanking."""
    pattern = _run_regex(data)
    wanted_ks = set(ks)
    wanted_seeds = set(seeds) if seeds is not None else None
    buckets: dict[tuple[int, int | None], list[Any]] = {}
    for run in runs:
        match = pattern.match(run.name or "")
        if not match:
            continue
        k = int(match.group(1))
        if k not in wanted_ks:
            continue
        seed = int(match.group(2)) if match.group(2) is not None else None
        if wanted_seeds is not None and seed not in wanted_seeds:
            continue
        buckets.setdefault((k, seed), []).append(run)

    selected: dict[tuple[int, int | None], Any] = {}
    roster: list[dict[str, Any]] = []
    for (k, seed), candidates in sorted(
        buckets.items(), key=lambda item: (item[0][0], item[0][1] is None, item[0][1])
    ):
        candidates.sort(key=lambda run: run.created_at or "", reverse=True)
        finished = [run for run in candidates if run.state == "finished"]
        chosen = finished[0] if finished else (
            candidates[0] if include_unfinished else None)
        selected[(k, seed)] = chosen
        roster.append(
            {
                "k": k,
                "seed": seed,
                "state": candidates[0].state,
                "used": chosen is not None,
                "url": candidates[0].url,
            }
        )
    return selected, roster


def _aggregate_by_k(
    selected: Mapping[tuple[int, int | None], Any], metrics: Sequence[str]
) -> dict[int, dict[str, Any]]:
    """Per k: mean/std/n over seeds for every metric (finite values only)."""
    by_k: dict[int, list[dict[str, Any]]] = {}
    for (k, _seed), run in selected.items():
        if run is None:
            continue
        summary = _summary_dict(run)
        by_k.setdefault(k, []).append(
            {metric: _finite_number(summary.get(metric)) for metric in metrics}
        )
    aggregated: dict[int, dict[str, Any]] = {}
    for k, seed_rows in by_k.items():
        stats: dict[str, Any] = {"n_seeds": len(seed_rows)}
        for metric in metrics:
            values = [row[metric] for row in seed_rows if row[metric] is not None]
            if values:
                stats[metric] = {
                    "mean": statistics.fmean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
            else:
                stats[metric] = None
        aggregated[k] = stats
    return aggregated


def _verdict(aggregated: Mapping[int, dict[str, Any]], metric: str) -> str:
    """Is any k better than k=1 on `metric` by more than ~2 standard errors?"""
    if 1 not in aggregated or aggregated[1].get(metric) is None:
        return f"{metric}: no k=1 baseline; cannot assess."
    base = aggregated[1][metric]
    others = [
        (k, aggregated[k][metric])
        for k in sorted(aggregated)
        if k != 1 and aggregated[k].get(metric) is not None
    ]
    if not others:
        return f"{metric}: no k>1 runs to compare."
    if base["n"] < 2 or any(stat["n"] < 2 for _, stat in others):
        return (f"{metric}: need >=2 seeds per k for a noise estimate "
                f"(have k=1:{base['n']}). Inconclusive.")
    best_k, best = min(others, key=lambda item: item[1]["mean"])
    delta = best["mean"] - base["mean"]  # negative = better
    se = math.sqrt(base["std"] ** 2 / base["n"] + best["std"] ** 2 / best["n"])
    rel = 100.0 * delta / base["mean"] if base["mean"] else float("nan")
    if se == 0:
        return f"{metric}: zero variance; k={best_k} delta {rel:+.2f}% (suspicious)."
    z = delta / se
    if delta < 0 and abs(z) >= 2.0:
        return (f"{metric}: POSITIVE. k={best_k} beats k=1 by {rel:+.2f}% "
                f"({abs(z):.1f} sigma, > 2). Real at this budget.")
    if delta > 0 and abs(z) >= 2.0:
        return (f"{metric}: NEGATIVE. best k={best_k} is still {rel:+.2f}% worse "
                f"than k=1 ({abs(z):.1f} sigma). Erlang hurts here.")
    return (f"{metric}: NULL. best k={best_k} is {rel:+.2f}% ({abs(z):.1f} sigma "
            f"< 2), within seed noise. No separation.")


def _fmt_stat(stat: Any, *, big_ok: bool = True) -> str:
    if stat is None:
        return "--"
    mean, std, n = stat["mean"], stat["std"], stat["n"]
    fmt = "{:.1f}" if (big_ok and abs(mean) >= 1000) else "{:.4f}"
    if n > 1:
        return f"{fmt.format(mean)}±{fmt.format(std)}"
    return fmt.format(mean)


def _print_seed_roster(roster: Sequence[Mapping[str, Any]]) -> None:
    print("Runs found (k, seed, state):")
    for entry in roster:
        seed = "legacy" if entry["seed"] is None else f"s{entry['seed']}"
        used = "" if entry["used"] else "  [SKIPPED: not finished]"
        print(f"  k={entry['k']} {seed}: {entry['state']}{used}  {entry['url']}")
    print()


def main() -> int:
    args = _parse_args()
    api = wandb.Api()
    path = _project_path(api, args.project, args.entity)
    runs = api.runs(
        path,
        filters={"display_name": {"$regex": rf"^erlang-k\d+-{re.escape(args.data)}"}},
        order="-created_at",
    )
    selected, roster = _collect_seed_runs(
        runs, args.ks, args.data, args.seeds, args.include_unfinished
    )
    metrics = _metric_names(args.budgets)

    print(f"Erlang ablation: {path} ({args.data})\n")
    if not roster:
        raise SystemExit(
            f"No runs matching erlang-k*-{args.data}[-s*] in {path}. "
            "Check --data / --entity / --project."
        )
    _print_seed_roster(roster)

    aggregated = _aggregate_by_k(selected, metrics)
    if not aggregated:
        raise SystemExit(
            "No finished runs to aggregate. Pass --include-unfinished to use the "
            "latest running/failed run per (k, seed)."
        )

    headers = ["k", "seeds", "val/nll", "val/ppl", "entropy", "gen/default",
               *(f"gen/{b}" for b in args.budgets)]
    table = []
    for k in sorted(aggregated):
        stats = aggregated[k]
        table.append([
            str(k),
            str(stats["n_seeds"]),
            _fmt_stat(stats["val/nll"]),
            _fmt_stat(stats["val/ppl"]),
            _fmt_stat(stats["val/sample_entropy"]),
            _fmt_stat(stats["val/gen_ppl"]),
            *(_fmt_stat(stats[f"val/gen_ppl@{b}step"]) for b in args.budgets),
        ])
    _print_table(headers, table)

    print("\nVerdict (2-sigma over seeds; val/ppl is the comparable separator):")
    for metric in PRIMARY_METRICS:
        print(f"  {_verdict(aggregated, metric)}")

    print("\nRuns:")
    for entry in roster:
        if entry["used"]:
            seed = "legacy" if entry["seed"] is None else f"s{entry['seed']}"
            print(f"  k={entry['k']} {seed}: {entry['url']}")

    if args.csv:
        csv_rows = []
        for k in sorted(aggregated):
            stats = aggregated[k]
            record: dict[str, Any] = {"k": k, "n_seeds": stats["n_seeds"]}
            for metric in metrics:
                stat = stats[metric]
                record[f"{metric}/mean"] = stat["mean"] if stat else None
                record[f"{metric}/std"] = stat["std"] if stat else None
            csv_rows.append(record)
        _write_csv(args.csv, csv_rows)
        print(f"\nWrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
