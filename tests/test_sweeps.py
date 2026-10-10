"""Tests du développement des sweeps en liste de runs."""

import pytest

from pinn_chiarella.sweeps import expand_runs, set_by_path

BASE = {
    "seed": 0,
    "data": "data/synthetic/a.npz",
    "model": {"trainable": {"kappa": 0.3, "sigma": 1.0}, "fixed": {"alpha": 1.0}},
    "training": {"tau": 10.0},
}


def test_produit_cartesien_et_seeds():
    sweep = {"grid": {"training.tau": [1, 100], "data": ["d/x.npz", "d/y.npz"]}, "seeds": [0, 1]}
    runs = expand_runs(sweep, BASE)
    assert len(runs) == 8
    assert {r.name for r in runs} >= {"tau=1_data=x_seed0", "tau=100_data=y_seed1"}
    run = next(r for r in runs if r.name == "tau=100_data=y_seed1")
    assert run.case == "tau=100_data=y"
    assert (run.seed, run.config["seed"]) == (1, 1)
    assert run.config["training"]["tau"] == 100
    assert run.config["data"] == "d/y.npz"


def test_liste_explicite_remplace_le_champ_entier():
    sweep = {
        "runs": [
            {"name": "tout_inconnu"},
            {
                "name": "sigma_fixe",
                "set": {
                    "model.trainable": {"kappa": 0.3},
                    "model.fixed": {"alpha": 1.0, "sigma": 0.5},
                },
            },
        ]
    }
    libre, fixe = expand_runs(sweep, BASE)
    assert (libre.name, libre.seed, libre.config) == ("tout_inconnu", 0, BASE)
    assert fixe.config["model"] == {
        "trainable": {"kappa": 0.3},
        "fixed": {"alpha": 1.0, "sigma": 0.5},
    }


def test_la_config_de_base_n_est_pas_modifiee():
    expand_runs({"grid": {"training.tau": [1]}}, BASE)
    assert BASE["training"]["tau"] == 10.0


def test_erreurs():
    with pytest.raises(ValueError):
        expand_runs({}, BASE)
    with pytest.raises(ValueError):
        expand_runs({"runs": [{"name": "a"}, {"name": "a"}]}, BASE)
    with pytest.raises(KeyError):
        set_by_path({"model": {}}, "modele.tau", 1)
