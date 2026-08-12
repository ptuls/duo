"""Functional tests for the E1 categorical-semantics stress test."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import semantic_stress_test as e1


@pytest.mark.parametrize("vocab_size", [2, 8, 32])
def test_gaussian_calibration_matches_reference_state(vocab_size):
    sigma, target, calibrated = e1.calibrate_gaussian_scale(
        vocab_size=vocab_size,
        dimensionless_gap=3.0,
    )

    assert sigma > 0
    assert calibrated == pytest.approx(target, abs=1e-9)


def test_gumbel_argmax_matches_softmax_on_soft_logits():
    scaled_logits = torch.tensor(
        [[1.7, 0.4, -0.2, -1.1], [0.25, 0.1, -0.05, -0.3]],
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(7)

    empirical = e1.sample_empirical_distribution(
        scaled_logits=scaled_logits,
        method="gumbel",
        n_samples=100_000,
        draw_chunk_size=2_000,
        generator=generator,
    )

    assert torch.allclose(
        empirical,
        scaled_logits.softmax(dim=-1).to(torch.float64),
        atol=0.006,
    )


def test_numpy_denoiser_logits_drop_mask_column_and_track_topk_mass(tmp_path):
    logits = np.array(
        [
            [3.0, 1.0, 0.0, -2.0, -1e6],
            [2.0, 1.5, 0.5, -1.0, -1e6],
            [1.0, 0.9, 0.8, 0.7, -1e6],
            [4.0, 0.0, -1.0, -2.0, -1e6],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "logits.npy"
    np.save(path, logits)
    args = e1.build_parser().parse_args(
        [
            "--real-logits-npy",
            str(path),
            "--n-vectors",
            "3",
            "--real-top-k",
            "2",
            "--temperatures",
            "1",
        ]
    )

    case = e1.load_real_logits(args)[0]

    assert case.logits.shape == (3, 2)
    assert case.metadata["original_vocab_size"] == 4
    assert case.metadata["evaluated_vocab_size"] == 2
    coverage = case.coverage_by_temperature[1.0]
    assert bool(((coverage > 0.5) & (coverage <= 1.0)).all())


def test_smoke_experiment_writes_reproducible_artifacts(tmp_path):
    output_dir = tmp_path / "e1"
    report = e1.main(
        [
            "--smoke",
            "--no-plots",
            "--output-dir",
            str(output_dir),
            "--seed",
            "17",
        ]
    )

    assert report["experiment"] == "E1 exact softmax semantics stress test"
    assert (output_dir / "raw_metrics.csv").is_file()
    assert (output_dir / "summary.csv").is_file()
    assert (output_dir / "gaussian_calibration.csv").is_file()
    report_path = output_dir / "report.json"
    assert report_path.is_file()
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert {case["regime"] for case in written["cases"]} == {
        "one_hot",
        "random",
        "interpolated",
    }
    assert max(row["absolute_residual"] for row in written["calibration"]) < 1e-8
