import pytest
import numpy as np
import torch
import torch_sim as ts
from fp_transformer_torch.output import TrajectoryArchive
from checkpoint import Restart


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_full_torchsim_state_and_rng_round_trip(tmp_path, device):
    from torch_sim.integrators.md import MDState

    state = MDState(
        positions=torch.ones(2, 3, dtype=torch.float64),
        masses=torch.ones(2, dtype=torch.float64),
        cell=torch.eye(3, dtype=torch.float64) * 10,
        pbc=False,
        system_idx=torch.zeros(2, dtype=torch.long),
        atomic_numbers=torch.ones(2, dtype=torch.long),
        momenta=torch.arange(6, dtype=torch.float64).reshape(2, 3),
        energy=torch.zeros(1, dtype=torch.float64),
        forces=torch.ones(2, 3, dtype=torch.float64),
    )
    state = state.to(device=torch.device(device))
    state.rng = 73

    class Kernel:
        bandwidth = torch.tensor([0.3], dtype=torch.float64)

    kernel = Kernel()
    kernel.bandwidth = kernel.bandwidth.to(device)
    recorder = TrajectoryArchive(tmp_path / "trajectory.npz")
    recorder.append(
        "production",
        1000,
        0.001,
        0.1,
        state.positions.view(1, 2, 3),
        np.zeros((1, 2)),
        kernel.bandwidth,
    )
    restart = Restart(tmp_path, recorder, kernel)
    restart.save(1000, state)
    expected_global = torch.randn(5, device=device)
    expected_state = torch.randn(5, generator=state.rng, device=device)
    torch.manual_seed(999)
    kernel.bandwidth = torch.tensor([0.8], dtype=torch.float64, device=device)
    restored = Restart(tmp_path, TrajectoryArchive(tmp_path / "trajectory.npz"), kernel)
    actual = restored.load(state)
    assert restored.offset == 1000 and restored.resumed
    torch.testing.assert_close(actual.momenta, state.momenta)
    torch.testing.assert_close(actual.forces, state.forces)
    torch.testing.assert_close(torch.randn(5, device=device), expected_global)
    torch.testing.assert_close(
        torch.randn(5, generator=actual.rng, device=device), expected_state
    )
    torch.testing.assert_close(
        kernel.bandwidth, torch.tensor([0.3], dtype=torch.float64, device=device)
    )
