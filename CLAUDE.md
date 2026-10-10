# Projet S7 — PINN pour le modèle d'agents financiers (Chiarella multi-échelles)

## Contexte
Projet P15 (CentraleSupélec, encadrant D. Challet) : modèle d'agents financiers à la Chiarella / Majewski-Ciliberti-Bouchaud (2018).
Cette partie du projet : **utiliser des réseaux de neurones (PINN) pour identifier les paramètres du modèle et apprendre les termes manquants** de l'équation comportementale, à partir de données.

## Modèle de référence
Modèle de base (log-prix P, mémoire de tendance M, valeur fondamentale V) :

```
dP = κ(V − P)dt + β tanh(γM)dt + σ_N dW^N
dM = −αM dt + α dP
dV = g dt + σ_V dW^V
```

- κ : rappel des fondamentalistes. β : intensité des trend-followers. γ : sensibilité au signal. α : 1/horizon des TF.
- Extension non linéaire des fondamentalistes : f(δ) = −κδ − κ₃δ³, avec δ = P − V.
- Multi-échelles : D_chart = Σ_k β_k tanh(γ M_k), M_{k,t+1} = (1−α_k)M_{k,t} + α_k (P_t − P_{t−1}).
  - Horizons ℓ_k log-espacés sur [1, 48] mois.
  - β_k ∝ ℓ_k^(−1.15) (Borland-Bouchaud).
  - β_k < 0 pour les mean-reverters (≈ 60 % des traders du dataset).
- Instabilité (modèle simple) : α + κ − αβγ < 0. Pour le modèle cubique : α − 2κ − αβγ < 0.

Références (PDF dans `docs/`) :
- Majewski et al. 2018 (Extended Chiarella)
- **Chen, Yang, Duan, Karniadakis 2021** (PINN + Fokker-Planck, méthode principale)
- O'Leary et al. 2022 (SPINODE, alternative par moments)
- Yang et al. (LDE-Net, baseline boîte noire uniquement)

## Méthode principale : PINN Fokker-Planck (Chen et al.)
- Un réseau p̃(x, t; w) approxime la densité de l'état x. Sortie softplus. Entrées normalisées dans [−1, 1].
- Les paramètres physiques inconnus (κ, β, σ…) sont des **scalaires entraînables**, paramétrés en log pour rester positifs. Un terme inconnu peut aussi être un petit MLP g_θ ajouté au drift connu : **drift = f_connu + g_θ**, jamais un drift entièrement boîte noire.
- La loss est L = τ·L_pde + Σ_i L_data(t_i).
  - L_pde : résidu de Fokker-Planck sur des points de collocation aléatoires, dérivées par autograd.
  - L_data : KL variationnelle, −mean log p̃(X_j, t_i) + ∫ p̃(x, t_i) dx. L'intégrale est estimée par Monte-Carlo × volume du domaine.
- Bruit corrélé : pour l'état (δ, M), le même dW entre dans dδ et dM (coefficient α). Le terme de diffusion contient donc les dérivées croisées : ½σ²(p_δδ + 2α p_δM + α² p_MM).
- Prototype fonctionnel : `reference/pinn_chiarella.py`. Résultats obtenus sur données synthétiques (V constant, 2000 échantillons, 5 snapshots, 15k itérations) :
  - β est retrouvé à 0,51 (vrai 0,5) ;
  - κ sort à 0,82 (vrai 1) et σ à 0,43 (vrai 0,5) : **κ et σ se compensent** (la variance stationnaire ≈ σ²/2κ) ;
  - le MLP retrouve la courbure du terme cubique dans la zone couverte par les données, mais pas au-delà.

## Données
<!-- À COMPLÉTER : chemin, format, colonnes, fréquence -->
- Dataset Telefonica (Gutiérrez-Roig et al. 2019) : 458 traders, 1966 jours de cotation (2000–2007), activité quotidienne. On garde 246 traders significatifs, dont 147 fondamentalistes / mean-reverters et 99 suiveurs de tendance.
- Prix Telefonica quotidiens.
- Chemin des données brutes : `data/raw/` (**ne jamais modifier**). Le prétraitement écrit dans `data/processed/`.
- **Point ouvert à trancher avant de coder l'entraînement réel** : ce qui joue le rôle de « particules ».
  - La méthode a besoin d'une distribution d'états observée à plusieurs instants.
  - Une seule trajectoire de prix ne suffit pas, et V n'est pas observé.
  - Pistes : positions/signaux des traders comme particules, fenêtres glissantes, ou plusieurs actifs.
  - Demander à l'utilisateur avant de choisir.

## Structure du dépôt
Package installable en « src layout » (`pip install -e ".[dev]"`), nom : `pinn_chiarella`.
```
CLAUDE.md  README.md  pyproject.toml
docs/                 articles PDF (ne pas modifier)
reference/            prototype pinn_chiarella.py (ne pas modifier, sert de référence)
src/pinn_chiarella/
  sde.py              drift/diffusion du modèle + simulateur Euler-Maruyama (données synthétiques)
  models.py           réseaux (PDF net, MLP terme manquant), paramètres entraînables
  losses.py           résidu Fokker-Planck, loss KL données
  train.py            boucle d'entraînement, lecture d'une config
  data.py             chargement/prétraitement (synthétique, Telefonica) -> snapshots
  plotting.py         figures standard
  sweeps.py           séries de runs : variations d'une config de base, résumé
scripts/              points d'entrée en ligne de commande, fins (la logique vit dans le package)
configs/*.yaml        une config par expérience (vrais params si synthétique, τ, lr, itérations, seed)
configs/sweeps/*.yaml un sweep = une config de base + les champs à faire varier (grille ou liste), seeds
tests/                tests pytest
data/                 raw/ (ne jamais modifier), processed/, synthetic/ — non versionnés
experiments/<nom>/    sorties : config copiée, log JSON, figures, poids — non versionnées
experiments/<sweep>/<run>/   idem pour chaque run d'un sweep
```

## Règles de travail
1. **Toujours valider sur données synthétiques d'abord.** On simule `sde.py` avec des paramètres connus, et le pipeline doit les retrouver. Un test pytest rapide (peu d'itérations, tolérance large) vérifie que la loss décroît et que les paramètres bougent dans le bon sens.
2. **Ne pas modifier les équations du modèle** (sde.py) sans demander : elles viennent du rapport du groupe.
3. Chaque expérience est pilotée par une config YAML, avec une seed fixée. Les sorties vont dans `experiments/<nom>/`.
4. Le log JSON enregistre toutes les 100 itérations : L_pde, L_data, et chaque paramètre appris (avec sa vraie valeur si synthétique).
5. Figures standard pour chaque run :
   - convergence des paramètres vs vraies valeurs ;
   - terme appris vs vrai, avec la zone couverte par les données ;
   - densité PINN vs Monte-Carlo à un instant non observé.
6. Pour un terme inconnu, essayer d'abord une forme polynomiale (κδ + κ₃δ³), puis le MLP. Le polynôme est plus précis quand la forme est connue (Chen et al.).
7. Pour l'identifiabilité, signaler explicitement toute paire de paramètres qui se compensent. Proposer de fixer l'un des deux plutôt que de rapporter des valeurs non fiables.
8. Commentaires et docstrings en français. Code en PyTorch, CPU par défaut, GPU si disponible (`device` dans la config).
9. Ne pas lancer d'entraînements longs (> 10 min) sans prévenir. Faire d'abord un run court pour estimer le temps.

## Commandes
```
pip install -e ".[dev]"
python scripts/make_synthetic.py --config configs/synth_chiarella.yaml
python scripts/train.py --config configs/pinn_params.yaml     # ajouter --iterations 500 pour un run court
python scripts/summarize.py pinn_params pinn_nn             # écrit experiments/summary.md
python scripts/sweep.py --config configs/sweeps/tau.yaml --jobs 4   # série de runs (4 en parallèle max sur ce Mac)
python scripts/summarize_sweeps.py tau kappa_sigma          # écrit experiments/summary_sweeps.md
pytest -q
```

## Pistes suivantes
- Passer à l'état multi-échelles (δ, M_1..M_K). Attention : le coût du Laplacien croît avec la dimension. Rester à K ≤ 3 pour le PINN ou passer à SPINODE (moments).
- Bruit de Lévy (Chen et al. traitent le cas α-stable) si les queues épaisses l'exigent.
- Contraintes sur g_θ : régularisation L2, signe, symétrie achat/vente.

## Règles git pour toute la suite
- Messages au format Conventional Commits en français : feat:, fix:, test:, docs:, refactor:, chore:.
- Un commit = un changement cohérent. Ne jamais committer si `pytest` échoue.
- Lance `ruff check` et `ruff format` avant chaque commit.
- Ne jamais committer de données, de poids de modèle ni de résultats d'expérience.
- Remote `origin` : https://github.com/AchrafMedyouni/modeles-agents-multi-echelles (dépôt public). Ne pousser que sur demande explicite de l'utilisateur.