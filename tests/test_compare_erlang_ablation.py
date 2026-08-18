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


def test_collect_seed_runs_groups_by_k_and_seed():
    runs = [
        _run("erlang-k1-owt-s0", "2026-08-16", summary={"val/ppl": 35.0}),
        _run("erlang-k1-owt-s0", "2026-08-15", state="failed"),  # older, ignored
        _run("erlang-k1-owt-s1", "2026-08-16", summary={"val/ppl": 35.4}),
        _run("erlang-k2-owt-s0", "2026-08-16", summary={"val/ppl": 34.0}),
        _run("erlang-k2-owt-s1", "2026-08-16", state="running"),  # not finished
        _run("erlang-k9-owt-s0", "2026-08-16"),  # k not requested
        _run("unrelated-run", "2026-08-16"),
    ]

    selected, roster = compare_erlang_ablation._collect_seed_runs(
        runs, ks=[1, 2], data="owt", seeds=None, include_unfinished=False
    )

    assert selected[(1, 0)].created_at == "2026-08-16"  # newest finished
    assert selected[(1, 1)] is not None
    assert selected[(2, 0)] is not None
    assert selected[(2, 1)] is None  # running, not included
    # roster reports every (k, seed) including the skipped one
    skipped = [r for r in roster if not r["used"]]
    assert (2, 1) in {(r["k"], r["seed"]) for r in skipped}


def test_aggregate_and_verdict_null_and_positive():
    # k=1 seeds ~35.5, k=2 seeds ~35.4 (within noise) -> NULL for val/ppl.
    selected = {
        (1, 0): _run("erlang-k1-owt-s0", "d", summary={"val/ppl": 35.5}),
        (1, 1): _run("erlang-k1-owt-s1", "d", summary={"val/ppl": 35.6}),
        (2, 0): _run("erlang-k2-owt-s0", "d", summary={"val/ppl": 35.4}),
        (2, 1): _run("erlang-k2-owt-s1", "d", summary={"val/ppl": 35.5}),
    }
    agg = compare_erlang_ablation._aggregate_by_k(selected, ["val/ppl"])
    assert agg[1]["n_seeds"] == 2
    assert agg[1]["val/ppl"]["mean"] == pytest.approx(35.55)
    assert "NULL" in compare_erlang_ablation._verdict(agg, "val/ppl")

    # k=2 clearly and consistently lower -> POSITIVE.
    strong = {
        (1, 0): _run("erlang-k1-owt-s0", "d", summary={"val/ppl": 35.5}),
        (1, 1): _run("erlang-k1-owt-s1", "d", summary={"val/ppl": 35.6}),
        (2, 0): _run("erlang-k2-owt-s0", "d", summary={"val/ppl": 30.0}),
        (2, 1): _run("erlang-k2-owt-s1", "d", summary={"val/ppl": 30.1}),
    }
    agg2 = compare_erlang_ablation._aggregate_by_k(strong, ["val/ppl"])
    assert "POSITIVE" in compare_erlang_ablation._verdict(agg2, "val/ppl")


def test_verdict_needs_two_seeds():
    selected = {
        (1, 0): _run("erlang-k1-owt-s0", "d", summary={"val/ppl": 35.5}),
        (2, 0): _run("erlang-k2-owt-s0", "d", summary={"val/ppl": 30.0}),
    }
    agg = compare_erlang_ablation._aggregate_by_k(selected, ["val/ppl"])
    assert "Inconclusive" in compare_erlang_ablation._verdict(agg, "val/ppl")


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
