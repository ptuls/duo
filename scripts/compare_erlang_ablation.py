#!/usr/bin/env python3
"""Compare completed Erlang-k ablation runs stored in Weights & Biases."""

import argparse
import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import wandb


DEFAULT_BUDGETS = (1, 2, 4, 8, 16)
BASE_METRICS = ("val/nll", "val/ppl", "val/gen_ppl")


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


def main() -> int:
    args = _parse_args()
    expected_names = _run_names(args.ks, args.data)
    api = wandb.Api()
    path = _project_path(api, args.project, args.entity)
    runs = api.runs(
        path,
        filters={"display_name": {"$in": list(expected_names.values())}},
        order="-created_at",
    )
    selected, unfinished_fallbacks = _select_runs(
        runs, expected_names, args.include_unfinished
    )

    missing = [k for k in args.ks if k not in selected]
    if missing:
        names = ", ".join(expected_names[k] for k in missing)
        suffix = " Pass --include-unfinished to allow incomplete runs."
        raise SystemExit(f"No finished W&B run found for: {names}.{suffix}")
    if unfinished_fallbacks:
        joined = ", ".join(str(k) for k in unfinished_fallbacks)
        print(f"Warning: using unfinished run(s) for k={joined}.\n")

    rows = _comparison_rows(selected, args.budgets)
    print(f"Erlang ablation: {path} ({args.data})\n")
    _print_comparison(rows, args.budgets)
    if args.csv:
        _write_csv(args.csv, rows)
        print(f"\nWrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
