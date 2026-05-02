#!/usr/bin/env python3
"""
Full statistical ablation: all algorithms, all domains, all personas, 30 seeds.
Runs on threadripper with multiprocessing (64 cores).

Algorithms (8 base + 4 ablations = 12 conditions):
  1. Baseline (VC-MAP-Elites)
  2. Lamarckian
  3. Coevolution
  4. SCOPE-Elites
  5. EGGROLL
  6. Island Model
  7. Evict-Restart
  8. Bandit Experts
  9. Combined (EGGROLL + Evict-Restart)
 10. EGGROLL-nodir (ablation: no direction learning)
 11. EvRst-noevict (ablation: no eviction pool)
 12. EvRst-nostag (ablation: no stagnation restart)

Domains: TTP, LogicPuzzles
Personas: Exploratory, Cycle (TwoForwardOneBack), Adaptive
Seeds: 30

Total tasks: 12 * 2 * 3 * 30 = 2160

Output: results/cluster_ablation.json + formatted tables to stdout.
"""

import sys
import os
import json
import time
import random
import copy
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy import stats

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.path.join(SCRIPT_DIR, "framework")
TESTS_DIR = os.path.join(FRAMEWORK_DIR, "Tests")
ALGO_DIR = os.path.join(SCRIPT_DIR, "algorithms")

sys.path.insert(0, FRAMEWORK_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, ALGO_DIR)
os.chdir(TESTS_DIR)


def run_one(args):
    """Run a single experiment. Worker function for multiprocessing."""
    algo_name, domain_name, persona_name, seed = args

    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)

    # Re-chdir in worker (fork may reset)
    os.chdir(TESTS_DIR)

    # Clear module cache to avoid stale imports across tasks
    for mod_name in list(sys.modules.keys()):
        if any(x in mod_name for x in ['mvp', 'ProblemSpaces', 'Personas',
                                         'Algorithms', 'GeneticAlgorithm',
                                         'ProblemSpace', 'Constraints']):
            del sys.modules[mod_name]

    from ProblemSpaces.TravelingThief.TTP_ProblemSpace import TTPProblemSpace
    from ProblemSpaces.LogicPuzzles.LogicPuzzleSpace import LogicPuzzleSpace
    from Personas.Exploratory import ExploratoryUser
    from Personas.TwoForwardOneBack import TwoForOneBackUser
    from Personas.Adaptive import AdaptiveUser
    from Algorithms.VCMapElites import VariableConstraintMapElites

    from mvp1_lamarckian import LamarckianElites
    from mvp2_coevolution import CoevolutionaryElites
    from mvp3_scope_compressed import SCOPEElites
    from mvp4_eggroll_lowrank import EGGROLLElites
    from mvp5_island_model import IslandElites
    from mvp6_evict_restart import EvictRestartElites
    from mvp7_bandit_experts import BanditExpertElites
    from ablation_winners import CombinedElites

    algo_map = {
        "Baseline": (VariableConstraintMapElites, {}),
        "Lamarckian": (LamarckianElites, {}),
        "Coevolution": (CoevolutionaryElites, {}),
        "SCOPE": (SCOPEElites, {}),
        "EGGROLL": (EGGROLLElites, {}),
        "Island": (IslandElites, {}),
        "Evict-Restart": (EvictRestartElites, {}),
        "Bandit": (BanditExpertElites, {}),
        "Combined": (CombinedElites, {}),
        "EGGROLL-nodir": (EGGROLLElites, {"direction_weight": 0.0}),
        "EvRst-noevict": (EvictRestartElites, {"stagnation_patience": 3}),
        "EvRst-nostag": (EvictRestartElites, {"stagnation_patience": 9999}),
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
    except Exception as e:
        qd = 0.0

    return (algo_name, domain_name, persona_name, seed, qd)


def print_results(all_data, algorithms, domains, personas):
    """Print formatted statistical tables."""
    print("\n" + "=" * 100)
    print("RESULTS: Mean QD Score (30 seeds)")
    print("=" * 100)

    for dname in domains:
        print(f"\n{'_' * 100}")
        print(f"  {dname}")
        print(f"{'_' * 100}")
        print(f"  {'Algorithm':<20} {'Exploratory':>12} {'Cycle':>12} {'Adaptive':>12} {'Total':>12}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for aname in algorithms:
            row = []
            for pname in personas:
                key = f"{dname}|{aname}|{pname}"
                qds = all_data.get(key, [0])
                row.append(np.mean(qds))
            total = sum(row)
            print(f"  {aname:<20} {row[0]:>12.1f} {row[1]:>12.1f} {row[2]:>12.1f} {total:>12.1f}")

    # Statistical tests vs Baseline
    print("\n" + "=" * 100)
    print("STATISTICAL TESTS vs Baseline (Mann-Whitney U, two-sided)")
    print("=" * 100)

    for dname in domains:
        print(f"\n{'_' * 100}")
        print(f"  {dname}")
        print(f"{'_' * 100}")
        print(f"  {'Algorithm':<20} {'Persona':<12} {'Algo Mean':>10} {'Base Mean':>10} {'p-value':>10} {'Cohen d':>8} {'Sig':>5}")
        print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*5}")
        for aname in algorithms:
            if aname == "Baseline":
                continue
            for pname in personas:
                a_key = f"{dname}|{aname}|{pname}"
                b_key = f"{dname}|Baseline|{pname}"
                a_qds = all_data.get(a_key, [0])
                b_qds = all_data.get(b_key, [0])
                ma, mb = np.mean(a_qds), np.mean(b_qds)
                sa, sb = np.std(a_qds, ddof=1), np.std(b_qds, ddof=1)
                try:
                    U, p = stats.mannwhitneyu(a_qds, b_qds, alternative="two-sided")
                except ValueError:
                    p = 1.0
                pooled = np.sqrt((sa**2 + sb**2) / 2)
                d = (ma - mb) / pooled if pooled > 0 else 0
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                print(f"  {aname:<20} {pname:<12} {ma:>10.1f} {mb:>10.1f} {p:>10.6f} {d:>+8.3f} {sig:>5}")

    # Head-to-head: top algorithms
    print("\n" + "=" * 100)
    print("HEAD-TO-HEAD: Top algorithms pairwise")
    print("=" * 100)
    top = ["Combined", "EGGROLL", "Evict-Restart", "Bandit"]
    for dname in domains:
        print(f"\n  {dname}:")
        for i, a in enumerate(top):
            for b in top[i+1:]:
                for pname in personas:
                    a_key = f"{dname}|{a}|{pname}"
                    b_key = f"{dname}|{b}|{pname}"
                    a_qds = all_data.get(a_key, [0])
                    b_qds = all_data.get(b_key, [0])
                    try:
                        U, p = stats.mannwhitneyu(a_qds, b_qds, alternative="two-sided")
                    except ValueError:
                        p = 1.0
                    ma, mb = np.mean(a_qds), np.mean(b_qds)
                    winner = a if ma > mb else b
                    sig = "*" if p < 0.05 else ""
                    print(f"    {a} vs {b} [{pname}]: {ma:.0f} vs {mb:.0f}, p={p:.4f} {sig} -> {winner}")

    # Ablation analysis
    print("\n" + "=" * 100)
    print("ABLATION ANALYSIS")
    print("=" * 100)
    ablations = [
        ("EGGROLL", "EGGROLL-nodir", "Direction learning"),
        ("Evict-Restart", "EvRst-noevict", "Eviction pool"),
        ("Evict-Restart", "EvRst-nostag", "Stagnation restart"),
    ]
    for dname in domains:
        print(f"\n  {dname}:")
        for full, ablated, component in ablations:
            print(f"    Component: {component} ({full} vs {ablated})")
            for pname in personas:
                f_key = f"{dname}|{full}|{pname}"
                a_key = f"{dname}|{ablated}|{pname}"
                f_qds = all_data.get(f_key, [0])
                a_qds = all_data.get(a_key, [0])
                mf, ma = np.mean(f_qds), np.mean(a_qds)
                try:
                    U, p = stats.mannwhitneyu(f_qds, a_qds, alternative="two-sided")
                except ValueError:
                    p = 1.0
                diff = mf - ma
                sig = "*" if p < 0.05 else ""
                print(f"      {pname:<12}: {full}={mf:.0f}, {ablated}={ma:.0f}, diff={diff:+.0f}, p={p:.4f} {sig}")

    # Grand totals
    print("\n" + "=" * 100)
    print("GRAND TOTAL (sum of all domain/persona means)")
    print("=" * 100)
    totals = []
    for aname in algorithms:
        total = sum(np.mean(all_data.get(f"{d}|{aname}|{p}", [0]))
                    for d in domains for p in personas)
        totals.append((total, aname))
    totals.sort(reverse=True)
    for total, aname in totals:
        print(f"  {aname:<20} {total:>12.1f}")


def main():
    n_seeds = 30
    n_workers = min(cpu_count(), 60)

    algorithms = [
        "Baseline", "Lamarckian", "Coevolution", "SCOPE", "EGGROLL",
        "Island", "Evict-Restart", "Bandit", "Combined",
        "EGGROLL-nodir", "EvRst-noevict", "EvRst-nostag",
    ]
    domains = ["TTP", "LogicPuzzles"]
    personas = ["Exploratory", "Cycle", "Adaptive"]

    total_tasks = len(algorithms) * len(domains) * len(personas) * n_seeds
    print(f"QDA-VC Full Ablation Study")
    print(f"  Algorithms: {len(algorithms)}")
    print(f"  Domains: {len(domains)}")
    print(f"  Personas: {len(personas)}")
    print(f"  Seeds: {n_seeds}")
    print(f"  Total tasks: {total_tasks}")
    print(f"  Workers: {n_workers}")
    print("=" * 70)

    # Build task list
    tasks = []
    for aname in algorithms:
        for dname in domains:
            for pname in personas:
                for seed in range(n_seeds):
                    tasks.append((aname, dname, pname, seed))

    # Run
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
            if completed % 100 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (total_tasks - completed) / rate
                print(f"  {completed}/{total_tasks} ({completed/total_tasks*100:.0f}%) "
                      f"| {elapsed:.0f}s | ~{eta:.0f}s remaining", flush=True)

    elapsed = time.time() - t0
    print(f"\nCompleted {total_tasks} tasks in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save raw data
    outpath = os.path.join(SCRIPT_DIR, "results", "cluster_ablation.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved to {outpath}")

    # Print results
    print_results(all_data, algorithms, domains, personas)


if __name__ == "__main__":
    main()
