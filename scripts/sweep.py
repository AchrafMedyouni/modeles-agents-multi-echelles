"""Lance une ou plusieurs séries de runs (sweeps) décrites dans configs/sweeps/.

Usage :
    python scripts/sweep.py --config configs/sweeps/tau.yaml [autres.yaml ...] --jobs 6

Chaque run écrit dans experiments/<sweep>/<run>/ (<sweep> = nom du fichier YAML). Les runs
déjà terminés avec le même nombre d'itérations sont sautés, sauf avec --force.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import yaml

from pinn_chiarella.sweeps import DEFAULT_ITERATIONS, expand_runs, is_done, run_one

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, nargs="+", required=True, help="YAML de sweep")
    parser.add_argument("--iterations", type=int, help="remplace le nombre d'itérations des sweeps")
    parser.add_argument("--jobs", type=int, default=1, help="runs en parallèle (1 thread chacun)")
    parser.add_argument("--force", action="store_true", help="relance les runs déjà terminés")
    parser.add_argument("--dry-run", action="store_true", help="liste les runs sans les lancer")
    args = parser.parse_args()

    todo = []
    for path in args.config:
        sweep = yaml.safe_load(path.read_text(encoding="utf-8"))
        base = yaml.safe_load((ROOT / sweep["base"]).read_text(encoding="utf-8"))
        iterations = args.iterations or int(sweep.get("iterations", DEFAULT_ITERATIONS))
        for run in expand_runs(sweep, base):
            out_dir = ROOT / "experiments" / path.stem / run.name
            data = ROOT / run.config["data"]
            if not data.exists():
                parser.error(f"Données absentes : {data} (voir scripts/make_synthetic.py)")
            if is_done(out_dir, iterations) and not args.force:
                print(f"déjà fait  {path.stem}/{run.name}")
                continue
            todo.append((f"{path.stem}/{run.name}", run, out_dir, iterations))

    print(f"{len(todo)} run(s) à lancer, {args.jobs} en parallèle")
    if args.dry_run:
        for label, _, _, iterations in todo:
            print(f"  {label}  ({iterations} itérations)")
        return

    start = time.perf_counter()
    threads = 1 if args.jobs > 1 else None
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
        futures = {
            pool.submit(run_one, run, ROOT, out_dir, iterations, threads): label
            for label, run, out_dir, iterations in todo
        }
        for done, future in enumerate(as_completed(futures), start=1):
            log = future.result()
            values = "  ".join(f"{k} = {v:.3f}" for k, v in log["final"].items())
            print(
                f"[{done}/{len(todo)}] {futures[future]}  {log['duration_s'] / 60:.1f} min  "
                f"{values}  ({(time.perf_counter() - start) / 60:.1f} min écoulées)",
                flush=True,
            )


if __name__ == "__main__":
    main()
