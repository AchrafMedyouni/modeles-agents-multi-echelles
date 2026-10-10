"""Tests du simulateur Chiarella : reproductibilité, dimensions, comportement moyen."""

import numpy as np
import pytest
import torch

from pinn_chiarella.sde import (
    ChiarellaParams,
    InitialState,
    diffusion,
    drift,
    is_stable,
    sample_initial_state,
    simulate,
)

PARAMS = ChiarellaParams(kappa=1.0, kappa3=0.0, beta=0.5, sigma=0.5, alpha=1.0, gamma=2.0)
T_OBS = [0.0, 0.5, 1.0, 2.0, 3.0]
DT = 1e-3


def run(seed: int, n: int = 2000, params: ChiarellaParams = PARAMS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0 = sample_initial_state(n, InitialState(), rng)
    return simulate(params, x0, T_OBS, DT, rng)


def test_reproductible_avec_la_seed():
    assert np.array_equal(run(0), run(0))
    assert not np.array_equal(run(0), run(1))


def test_dimensions():
    snaps = run(0, n=300)
    assert snaps.shape == (len(T_OBS), 300, 2)
    assert np.all(np.isfinite(snaps))


def test_premier_snapshot_est_l_etat_initial():
    rng = np.random.default_rng(0)
    x0 = sample_initial_state(100, InitialState(), rng)
    x0_avant = x0.copy()
    snaps = simulate(PARAMS, x0, T_OBS, DT, rng)
    assert np.array_equal(snaps[0], x0)
    assert np.array_equal(x0, x0_avant)


def test_moyenne_de_delta_decroit_vers_zero_en_regime_stable():
    assert is_stable(PARAMS)
    mean_delta = run(0)[:, :, 0].mean(axis=1)
    assert mean_delta[0] == pytest.approx(1.0, abs=0.05)
    assert np.all(np.diff(mean_delta) < 0)
    assert abs(mean_delta[-1]) < 0.2


def test_sans_bruit_euler_deterministe():
    params = ChiarellaParams(kappa=1.0, kappa3=0.0, beta=0.0, sigma=0.0, alpha=1.0, gamma=2.0)
    x0 = np.array([[1.0, 0.0]])
    snaps = simulate(params, x0, [1.0], DT, np.random.default_rng(0))
    assert snaps[0, 0, 0] == pytest.approx(np.exp(-1.0), rel=1e-3)


def test_drift_identique_numpy_et_torch():
    rng = np.random.default_rng(0)
    delta, m = rng.normal(size=50), rng.normal(size=50)
    kw = dict(kappa=1.0, kappa3=0.5, beta=0.5, alpha=1.5, gamma=2.0)
    a_np = drift(delta, m, **kw)
    a_torch = drift(torch.from_numpy(delta), torch.from_numpy(m), **kw)
    for x, y in zip(a_np, a_torch, strict=True):
        assert np.allclose(x, y.numpy())
    assert np.allclose(a_np[0], -delta - 0.5 * delta**3 + 0.5 * np.tanh(2.0 * m))
    assert np.allclose(a_np[1], 1.5 * (a_np[0] - m))


def test_diffusion_bruit_correle():
    assert diffusion(sigma=0.5, alpha=2.0) == (0.5, 1.0)


@pytest.mark.parametrize("t_obs", [[], [1.0, 0.5], [0.5, 0.5], [-1.0], [0.0105]])
def test_instants_invalides(t_obs):
    x0 = np.zeros((3, 2))
    with pytest.raises(ValueError):
        simulate(PARAMS, x0, t_obs, DT, np.random.default_rng(0))
