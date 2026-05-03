#!/usr/bin/env python3
"""
Final battery: everything needed for the paper in one run.

1. Ablation at official params (UH-nobandit vs components)
2. All framework baselines (Filtering, Shuffling, RandomRestarts, VC-MAP-Elites)
3. Strict persona (4th persona, never tested)
4. 30 seeds, official params (300 gens, pop 200, memory 500, interval 50)

Algorithms (8):
  Baseline (VC-MAP-Elites), Filtering, Shuffling, RandomRestarts,
  UH-nobandit, UH-noevict (adaptive rate only), AdaptRate-noreset, BdEvict-nostag

Domains: TTP, LogicPuzzles
Personas: Exploratory, Cycle, Adaptive, Strict

8 * 2 * 4 * 30 = 1,920 tasks
"""

import sys
import os
import json
import time
import random
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.path.join(SCRIPT_DIR, "framework")
TESTS_DIR = os.path.join(FRAMEWORK_DIR, "Tests")
ALGO_DIR = os.path.join(SCRIPT_DIR, "algorithms")

sys.path.insert(0, FRAMEWORK_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, ALGO_DIR)
os.chdir(TESTS_DIR)

N_GENS = 300
POP_SIZE = 200
MEMORY = 500
XO_RATE = 0.5
MUT_RATE = 0.1
INTERVAL = 50


def run_one(args):
    algo_name, domain_name, persona_name, seed = args

    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)

    os.chdir(TESTS_DIR)

    for mod_name in list(sys.modules.keys()):
        if any(x in mod_name for x in ['mvp', 'ablation', 'ProblemSpaces', 'Personas',
                                         'Algorithms', 'GeneticAlgorithm',
                                         'ProblemSpace', 'Constraints']):
            del sys.modules[mod_name]

    from ProblemSpaces.TravelingThief.TTP_ProblemSpace import TTPProblemSpace
    from ProblemSpaces.LogicPuzzles.LogicPuzzleSpace import LogicPuzzleSpace
    from Personas.Exploratory import ExploratoryUser
    from Personas.TwoForwardOneBack import TwoForOneBackUser
    from Personas.Adaptive import AdaptiveUser
    from Personas.Strict import StrictUser
    from Algorithms.VCMapElites import VariableConstraintMapElites
    from Algorithms.Filtering import Filtering
    from Algorithms.RandomRestarts import RandomRestarts
    from Algorithms.Shuffling import Shuffling

    from mvp13_adaptive_rate import AdaptiveRateElites
    from mvp9_bandit_evict import BanditEvictElites
    from mvp22_ultimate_hybrid import UltimateHybridElites

    algo_map = {
        "VC-MAP-Elites": (VariableConstraintMapElites, {}),
        "Filtering": (Filtering, {}),
        "RandomRestarts": (RandomRestarts, {}),
        "Shuffling": (Shuffling, {}),
        "UH-nobandit": (UltimateHybridElites, {"use_bandit": False}),
        "UH-noevict": (UltimateHybridElites, {"use_bandit": False, "use_eviction": False}),
        "AdaptRate-noreset": (AdaptiveRateElites, {"reset_on_change": False}),
        "BdEvict-nostag": (BanditEvictElites, {"use_stagnation": False}),
    }
    domain_map = {
        "TTP": TTPProblemSpace,
        "LogicPuzzles": LogicPuzzleSpace,
    }
    persona_map = {
        "Exploratory": ExploratoryUser,
        "Cycle": TwoForOneBackUser,
        "Adaptive": AdaptiveUser,
        "Strict": StrictUser,
    }

    algo_class, kwargs = algo_map[algo_name]

    try:
        ps = domain_map[domain_name]()
        user = persona_map[persona_name](ps)
        algo = algo_class(ps, N_GENS, POP_SIZE, MEMORY, XO_RATE, MUT_RATE, user, INTERVAL, **kwargs)
        algo.run()
        m = algo.measure_history
        qd = m.qd_score[-1] if m.qd_score else 0.0
    except Exception as e:
        qd = 0.0

    return (algo_name, domain_name, persona_name, seed, qd)


def main():
    n_seeds = 30
    n_workers = min(cpu_count(), 60)

    algorithms = [
        "VC-MAP-Elites", "Filtering", "RandomRestarts", "Shuffling",
        "UH-nobandit", "UH-noevict", "AdaptRate-noreset", "BdEvict-nostag",
    ]
    domains = ["TTP", "LogicPuzzles"]
    personas = ["Exploratory", "Cycle", "Adaptive", "Strict"]

    total_tasks = len(algorithms) * len(domains) * len(personas) * n_seeds
    print(f"Final Paper Battery")
    print(f"  Parameters: gens={N_GENS} pop={POP_SIZE} mem={MEMORY} interval={INTERVAL}")
    print(f"  Algorithms: {len(algorithms)} ({', '.join(algorithms)})")
    print(f"  Domains: {len(domains)} ({', '.join(domains)})")
    print(f"  Personas: {len(personas)} ({', '.join(personas)})")
    print(f"  Seeds: {n_seeds}")
    print(f"  Total tasks: {total_tasks}")
    print(f"  Workers: {n_workers}")
    print("=" * 70)

    tasks = []
    for aname in algorithms:
        for dname in domains:
            for pname in personas:
                for seed in range(n_seeds):
                    tasks.append((aname, dname, pname, seed))

    random.shuffle(tasks)

    all_data = {}
    t0 = time.time()
    completed = 0

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(run_one, tasks):
            aname, dname, pname, seed, qd = result
            key = f"{dname}|{aname}|{pname}"
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(qd)
            completed += 1
            if completed % 50 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (total_tasks - completed) / rate
                print(f"  {completed}/{total_tasks} ({completed/total_tasks*100:.0f}%) "
                      f"| {elapsed:.0f}s | ~{eta:.0f}s remaining", flush=True)

    elapsed = time.time() - t0
    print(f"\nCompleted {total_tasks} tasks in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    outpath = os.path.join(SCRIPT_DIR, "results", "final_battery.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved to {outpath}")

    # === RESULTS ===

    # Grand totals
    print("\n" + "=" * 100)
    print("GRAND TOTAL RANKING (official params, 4 personas)")
    print("=" * 100)
    base_name = "VC-MAP-Elites"
    base_total = sum(np.mean(all_data.get(f"{d}|{base_name}|{p}", [0]))
                     for d in domains for p in personas)
    totals = []
    for a in algorithms:
        total = sum(np.mean(all_data.get(f"{d}|{a}|{p}", [0]))
                    for d in domains for p in personas)
        totals.append((total, a))
    totals.sort(reverse=True)
    for rank, (total, aname) in enumerate(totals, 1):
        pct = ((total - base_total) / abs(base_total) * 100) if base_total != 0 else 0
        print(f"  {rank:<3} {aname:<22} {total:>12.1f}  ({pct:>+7.1f}%)")

    # Per-domain per-persona
    for dname in domains:
        print(f"\n  {dname}:")
        print(f"  {'Algorithm':<22} " + " ".join(f"{p:>14}" for p in personas) + f" {'Total':>14}")
        print(f"  {'-'*22} " + " ".join(f"{'-'*14}" for _ in personas) + f" {'-'*14}")
        rows = []
        for a in algorithms:
            vals = [np.mean(all_data.get(f"{dname}|{a}|{p}", [0])) for p in personas]
            rows.append((sum(vals), a, vals))
        rows.sort(reverse=True)
        for total, aname, vals in rows:
            print(f"  {aname:<22} " +
                  " ".join(f"{v:>14.1f}" for v in vals) +
                  f" {total:>14.1f}")

    # Stat tests: UH-nobandit vs all
    print("\n" + "=" * 100)
    print("UH-nobandit vs ALL (Mann-Whitney U)")
    print("=" * 100)
    for dname in domains:
        for rival in algorithms:
            if rival == "UH-nobandit":
                continue
            for pname in personas:
                a = np.array(all_data.get(f"{dname}|UH-nobandit|{pname}", [0]))
                b = np.array(all_data.get(f"{dname}|{rival}|{pname}", [0]))
                ma, mb = np.mean(a), np.mean(b)
                try:
                    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    p = 1.0
                sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
                pooled = np.sqrt((sa**2 + sb**2) / 2)
                d = (ma - mb) / pooled if pooled > 0 else 0
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                print(f"  {dname:14} vs {rival:<22} {pname:<12} {ma:>10.1f} vs {mb:>10.1f} "
                      f"d={d:>+.3f} p={p:.4f} {sig}")

    # Ablation: UH-nobandit vs components
    print("\n" + "=" * 100)
    print("ABLATION at official params")
    print("=" * 100)
    ablations = [
        ("UH-nobandit", "UH-noevict", "Eviction pool contribution"),
        ("UH-nobandit", "AdaptRate-noreset", "Eviction pool (different impl)"),
        ("UH-nobandit", "BdEvict-nostag", "Adaptive rate contribution"),
    ]
    for full, ablated, label in ablations:
        print(f"\n  {label} ({full} vs {ablated}):")
        for dname in domains:
            for pname in personas:
                f_qds = np.array(all_data.get(f"{dname}|{full}|{pname}", [0]))
                a_qds = np.array(all_data.get(f"{dname}|{ablated}|{pname}", [0]))
                mf, ma = np.mean(f_qds), np.mean(a_qds)
                try:
                    U, p = stats.mannwhitneyu(f_qds, a_qds, alternative="two-sided")
                except ValueError:
                    p = 1.0
                diff = mf - ma
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                print(f"    {dname:14} {pname:<12}: {full}={mf:>10.1f}  {ablated}={ma:>10.1f}  "
                      f"diff={diff:>+10.1f}  p={p:.4f} {sig}")


if __name__ == "__main__":
    main()
