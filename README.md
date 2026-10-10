# pinn_chiarella

Projet S7 (CentraleSupélec, P15) : identification des paramètres d'un modèle d'agents
financiers à la Chiarella (Majewski, Ciliberti, Bouchaud 2018) et apprentissage des termes
manquants de son équation comportementale, à l'aide d'un PINN sur l'équation de Fokker-Planck
(Chen, Yang, Duan, Karniadakis 2021).

## Installation

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python ≥ 3.10.

## Commandes principales

```
python scripts/make_synthetic.py --config configs/synth_chiarella.yaml   # données synthétiques
python scripts/train.py --config configs/pinn_params.yaml             # entraînement du PINN
python scripts/summarize.py pinn_params pinn_nn                          # experiments/summary.md
pytest -q                                                                # tests
ruff check . && ruff format .                                            # lint et formatage
```

Les données (`data/`) et les résultats (`experiments/`) ne sont pas versionnés : ils se
régénèrent à partir du code et des configs.

## Organisation

- `src/pinn_chiarella/` : le package (simulateur, réseaux, losses, entraînement, données, figures).
- `scripts/` : points d'entrée en ligne de commande.
- `configs/` : une config YAML par expérience.
- `tests/` : tests pytest.
- `reference/` : prototype d'origine, conservé tel quel.
- `docs/` : articles de référence.
