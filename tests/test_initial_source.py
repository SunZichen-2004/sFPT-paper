import pytest
from analyze import validate_initial_source


def test_relocated_initial_pool_keeps_repository_identity():
    config = {"init_npy": "experiments/tetra_alanine_sFPT/init_coords_1024.npy"}
    validate_initial_source(
        "tetra_alanine_metad",
        config,
        {"init_npy": "/old/checkout/" + config["init_npy"]},
    )
    with pytest.raises(ValueError, match="different initial ensemble"):
        validate_initial_source(
            "tetra_alanine_metad",
            config,
            {
                "init_npy": "/old/checkout/experiments/tetra_alanine_sFPT/init_coords_2048.npy"
            },
        )


def test_reaction_box_initialization_requires_original_box():
    validate_initial_source(
        "rearrangement_sfpt",
        {},
        {
            "init_npy": None,
            "xyz": "/old/checkout/experiments/rearrangement_water_fpt/mlmm_box/box.xyz",
        },
    )
    with pytest.raises(ValueError, match="initialization from the water box"):
        validate_initial_source("rearrangement_sfpt", {}, {"init_npy": "other.npy"})
    with pytest.raises(ValueError, match="different initial ensemble"):
        validate_initial_source("rearrangement_sfpt", {}, {"xyz": "other.xyz"})
