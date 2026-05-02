#!/usr/bin/env python3
"""
100-seed confirmation run on top 5 algorithms.
Tightens confidence intervals for the paper's main claims.

5 algorithms x 2 domains x 3 personas x 100 seeds = 3,000 tasks
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
    from Algorithms.VCMapElites import VariableConstraintMapElites

    from mvp13_adaptive_rate import AdaptiveRateElites
    from mvp18_epsilon_bandit import EpsilonBanditElites
    from mvp22_ultimate_hybrid import UltimateHybridElites
    from mvp9_bandit_evict import BanditEvictElites

    algo_map = {
        "Baseline": (VariableConstraintMapElites, {}),
        "UH-nobandit": (UltimateHybridElites, {"use_bandit": False}),
        "AdaptRate-noreset": (AdaptiveRateElites, {"reset_on_change": False}),
        "BdEvict-nostag": (BanditEvictElites, {"use_stagnation": False}),
        "Epsilon-Bandit": (EpsilonBanditElites, {}),
    }
    domain_map = {
        "TTP": TTPProblemSpace,
        "LogicPuzzles": LogicPuzzleSpace,
    }
    persona_map = {
        "Exploratory": ExploratoryUser,
        "Cycle": TwoForOneBackUser,
        "Adaptive": AdaptiveUser,
    }

    algo_class, kwargs = algo_map[algo_name]

    try:
        ps = domain_map[domain_name]()
        user = persona_map[persona_name](ps)
        algo = algo_class(ps, 100, 50, 200, 0.7, 0.3, user, 10, **kwargs)
        algo.run()
        m = algo.measure_history
        qd = m.qd_score[-1] if m.qd_score else 0.0
    except Exception:
        qd = 0.0

    return (algo_name, domain_name, persona_name, seed, qd)


def main():
    n_seeds = 100
    n_workers = min(cpu_count(), 60)

    algorithms = [
        "Baseline", "UH-nobandit", "AdaptRate-noreset",
        "BdEvict-nostag", "Epsilon-Bandit",
    ]
    domains = ["TTP", "LogicPuzzles"]
    personas = ["Exploratory", "Cycle", "Adaptive"]

    total_tasks = len(algorithms) * len(domains) * len(personas) * n_seeds
    print(f"100-Seed Confirmation Run")
    print(f"  Algorithms: {len(algorithms)}")
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
            if completed % 200 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (total_tasks - completed) / rate
                print(f"  {completed}/{total_tasks} ({completed/total_tasks*100:.0f}%) "
                      f"| {elapsed:.0f}s | ~{eta:.0f}s remaining", flush=True)

    elapsed = time.time() - t0
    print(f"\nCompleted {total_tasks} tasks in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    outpath = os.path.join(SCRIPT_DIR, "results", "100seed_confirmation.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved to {outpath}")

    # Print results with confidence intervals
    print("\n" + "=" * 100)
    print("100-SEED RESULTS (mean +/- 95% CI)")
    print("=" * 100)

    for dname in domains:
        print(f"\n  {dname}:")
        print(f"  {'Algorithm':<22} {'Persona':<12} {'Mean':>10} {'95% CI':>16} {'Median':>10}")
        print(f"  {'-'*22} {'-'*12} {'-'*10} {'-'*16} {'-'*10}")
        for aname in algorithms:
            for pname in personas:
                key = f"{dname}|{aname}|{pname}"
                qds = np.array(all_data.get(key, [0]))
                mean = np.mean(qds)
                ci = 1.96 * np.std(qds, ddof=1) / np.sqrt(len(qds))
                median = np.median(qds)
                print(f"  {aname:<22} {pname:<12} {mean:>10.1f} [{mean-ci:>7.1f}, {mean+ci:>7.1f}] {median:>10.1f}")

    # Statistical tests
    print("\n" + "=" * 100)
    print("UH-nobandit vs ALL (Mann-Whitney U, 100 seeds)")
    print("=" * 100)
    for dname in domains:
        print(f"\n  {dname}:")
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
                print(f"    vs {rival:<20} {pname:<12} {ma:>10.1f} vs {mb:>10.1f} "
                      f"d={d:>+.3f} p={p:.6f} {sig}")


if __name__ == "__main__":
    main()
