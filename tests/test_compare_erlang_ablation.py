from types import SimpleNamespace

import pytest

from scripts import compare_erlang_ablation


def _run(name, created_at, state="finished", summary=None):
    return SimpleNamespace(
        name=name,
        id=f"{name}-{created_at}",
        state=state,
        created_at=created_at,
        summary=summary or {},
        url=f"https://wandb.invalid/{name}/{created_at}",
    )


def test_select_runs_uses_latest_finished_exact_name():
    expected = {1: "erlang-k1-data", 2: "erlang-k2-data"}
    runs = [
        _run("erlang-k1-data", "2026-08-15", state="finished"),
        _run("erlang-k1-data", "2026-08-16", state="failed"),
        _run("erlang-k2-data", "2026-08-14", state="finished"),
        _run("erlang-k4-data", "2026-08-17", state="finished"),
    ]

    selected, fallbacks = compare_erlang_ablation._select_runs(
        runs, expected, include_unfinished=False
    )

    assert selected[1].created_at == "2026-08-15"
    assert selected[2].created_at == "2026-08-14"
    assert fallbacks == []


def test_select_runs_can_fall_back_to_latest_unfinished():
    expected = {1: "erlang-k1-data"}
    runs = [
        _run("erlang-k1-data", "2026-08-15", state="failed"),
        _run("erlang-k1-data", "2026-08-16", state="running"),
    ]

    selected, fallbacks = compare_erlang_ablation._select_runs(
        runs, expected, include_unfinished=True
    )

    assert selected[1].state == "running"
    assert fallbacks == [1]


def test_comparison_rows_compute_relative_change_from_k_one():
    baseline = _run(
        "erlang-k1-data",
        "2026-08-15",
        summary={"val/gen_ppl": 20.0, "val/gen_ppl@1step": 100.0},
    )
    candidate = _run(
        "erlang-k2-data",
        "2026-08-16",
        summary={"val/gen_ppl": 18.0, "val/gen_ppl@1step": 125.0},
    )

    rows = compare_erlang_ablation._comparison_rows(
        {1: baseline, 2: candidate}, budgets=[1]
    )

    assert rows[0]["delta_pct/val/gen_ppl"] == 0.0
    assert rows[1]["delta_pct/val/gen_ppl"] == pytest.approx(-10.0)
    assert rows[1]["delta_pct/val/gen_ppl@1step"] == pytest.approx(25.0)
    assert rows[1]["val/nll"] is None


def test_project_path_uses_default_entity():
    api = SimpleNamespace(default_entity="paul")

    assert compare_erlang_ablation._project_path(api, "duo", None) == "paul/duo"
    assert (
        compare_erlang_ablation._project_path(api, "another/duo", None) == "another/duo"
    )


@pytest.mark.parametrize("ks", [[0], [1, 1]])
def test_run_names_reject_invalid_ks(ks):
    with pytest.raises(ValueError):
        compare_erlang_ablation._run_names(ks, "data")


@pytest.mark.parametrize("budgets", [[0], [-1], [1, 1]])
def test_metric_names_reject_invalid_budgets(budgets):
    with pytest.raises(ValueError):
        compare_erlang_ablation._metric_names(budgets)
