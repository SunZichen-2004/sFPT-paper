from pathlib import Path
import pytest
from run_case import ROOT, resolve_case, build_command


@pytest.mark.parametrize(
    "case,steps",
    [
        ("dialanine_vacuum_sfpt", 50000),
        ("dialanine_vacuum_metad", 50000),
        ("dialanine_water_sfpt", 200000),
        ("dialanine_water_metad", 200000),
        ("tetra_alanine_sfpt", 500000),
        ("tetra_alanine_metad", 10000000),
        ("rearrangement_sfpt", 83000),
        ("rearrangement_metad", 400000),
    ],
)
def test_default_is_full_paper_and_keeps_example_output_separate(case, steps):
    paper = resolve_case(case)
    example = resolve_case(case, protocol="example")
    assert paper["production_steps"] == steps
    assert example["production_steps"] == 25000
    assert paper["output"] != example["output"]
    command = build_command(paper)
    flag = "--production-steps" if case.endswith("sfpt") else "--steps"
    assert command[command.index(flag) + 1] == str(steps)


def test_seed_and_scan_settings_select_distinct_outputs_and_prepared_pool():
    cfg = resolve_case("tetra_alanine_sfpt", replicas=4096, seed=2)
    assert Path(cfg["init_npy"]).name == "init_coords_4096.npy"
    assert Path(cfg["init_npy"]).is_file()
    assert cfg["output"].endswith("tetra_alanine_sfpt_r4096/seed_2")
    cfg = resolve_case("dialanine_vacuum_metad", replicas=128)
    assert cfg["metad_height"] == 0.01
    assert "init_coords_128.npy" in cfg["init_npy"]


def test_custom_pool_is_not_replaced_by_replica_defaults(tmp_path):
    cfg = resolve_case(
        "dialanine_vacuum_sfpt",
        replicas=7,
        init_npy=tmp_path / "pool.npy",
        output=tmp_path / "out",
    )
    assert cfg["init_npy"] == str(tmp_path / "pool.npy")


@pytest.mark.parametrize(
    "kwargs", [{"hill_height": 0.01}, {"production_lambda": 0}, {"seed": -1}]
)
def test_invalid_protocol_override_is_rejected(kwargs):
    with pytest.raises(ValueError):
        resolve_case("dialanine_vacuum_sfpt", **kwargs)
