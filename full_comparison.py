#!/usr/bin/env python3
"""
Full comparison across both available domains, all personas, 30 runs.
Parallelized across local CPU cores.
"""

import sys
import os
import json
import copy
import random
import math
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from scipy import stats

framework_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "framework")
tests_dir = os.path.join(framework_dir, "Tests")
sys.path.insert(0, framework_dir)
sys.path.insert(0, tests_dir)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "algorithms"))

# Must chdir before imports that use relative paths
os.chdir(tests_dir)

from ProblemSpaces.TravelingThief.TTP_ProblemSpace import TTPProblemSpace
from ProblemSpaces.LogicPuzzles.LogicPuzzleSpace import LogicPuzzleSpace
from Personas.Exploratory import ExploratoryUser
from Personas.TwoForwardOneBack import TwoForOneBackUser
from Personas.Adaptive import AdaptiveUser
from Algorithms.VCMapElites import VariableConstraintMapElites
from mvp4_eggroll_lowrank import EGGROLLElites
from mvp6_evict_restart import EvictRestartElites


# Worker function for multiprocessing
def _run_one(args):
    """Run a single experiment. Designed for Pool.map."""
    algo_name, algo_cls_name, domain_name, persona_name, run_id, kwargs = args

    # Re-import inside worker (multiprocessing forks, need fresh state)
    os.chdir(tests_dir)

    # Resolve classes from names (can't pickle classes directly)
    algo_map = {
        "VariableConstraintMapElites": VariableConstraintMapElites,
        "EGGROLLElites": EGGROLLElites,
        "EvictRestartElites": EvictRestartElites,
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

    algo_class = algo_map[algo_cls_name]
    ps_class = domain_map[domain_name]
    persona_class = persona_map[persona_name]

    try:
        ps = ps_class()
        user = persona_class(ps)
        algo = algo_class(ps, 100, 50, 200, 0.7, 0.3, user, 10, **kwargs)
        algo.run()
        m = algo.measure_history
        qd = m.qd_score[-1] if m.qd_score else 0.0
        div = m.diversity[-1] if m.diversity else 0.0
    except Exception as e:
        qd, div = 0.0, 0.0

    return (algo_name, domain_name, persona_name, run_id, qd, div)


def main():
    n_runs = 30
    n_workers = min(cpu_count(), 8)

    domains = ["TTP", "LogicPuzzles"]
    personas = ["Exploratory", "Cycle", "Adaptive"]

    algorithms = [
        ("Baseline", "VariableConstraintMapElites", {}),
        ("EGGROLL", "EGGROLLElites", {}),
        ("Evict-Restart", "EvictRestartElites", {}),
        ("EGGROLL-nodir", "EGGROLLElites", {"direction_weight": 0.0}),
    ]

    # Build all tasks
    tasks = []
    for aname, acls, akwargs in algorithms:
        for dname in domains:
            for pname in personas:
                for run_id in range(n_runs):
                    tasks.append((aname, acls, dname, pname, run_id, akwargs))

    total = len(tasks)
    print(f"Full Comparison | {n_runs} runs | {len(domains)} domains | "
          f"{len(personas)} personas | {len(algorithms)} algorithms")
    print(f"Total tasks: {total} | Workers: {n_workers}")
    print("=" * 70)

    # Run in parallel
    all_data = {}
    completed = 0

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(_run_one, tasks):
            aname, dname, pname, run_id, qd, div = result
            key = (dname, aname, pname)
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(qd)
            completed += 1
            if completed % (n_workers * 5) == 0:
                print(f"  {completed}/{total} done ({completed/total*100:.0f}%)", flush=True)

    print(f"  {completed}/{total} done (100%)")

    # Print results
    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)

    for dname in domains:
        print(f"\n--- {dname} ---")
        print(f"{'Algorithm':<20} {'Persona':<12} {'Mean QD':>10} {'Std':>10} {'Median':>10}")
        print("-" * 65)
        for aname, _, _ in algorithms:
            for pname in personas:
                key = (dname, aname, pname)
                qds = all_data.get(key, [0])
                mean = np.mean(qds)
                std = np.std(qds, ddof=1)
                med = np.median(qds)
                print(f"{aname:<20} {pname:<12} {mean:>10.1f} {std:>10.1f} {med:>10.1f}")

    # Statistical analysis vs baseline
    print("\n" + "=" * 90)
    print("STATISTICAL ANALYSIS: Each algorithm vs Baseline (Mann-Whitney U, two-sided)")
    print("=" * 90)

    for dname in domains:
        print(f"\n--- {dname} ---")
        print(f"{'Algorithm':<20} {'Persona':<12} {'Algo':>8} {'Base':>8} {'p':>10} {'d':>8} {'Sig':>5}")
        print("-" * 75)
        for aname, _, _ in algorithms:
            if aname == "Baseline":
                continue
            for pname in personas:
                a_qds = all_data[(dname, aname, pname)]
                b_qds = all_data[(dname, "Baseline", pname)]
                ma, mb = np.mean(a_qds), np.mean(b_qds)
                sa, sb = np.std(a_qds, ddof=1), np.std(b_qds, ddof=1)
                U, p = stats.mannwhitneyu(a_qds, b_qds, alternative="two-sided")
                pooled = np.sqrt((sa**2 + sb**2) / 2)
                d = (ma - mb) / pooled if pooled > 0 else 0
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                print(f"{aname:<20} {pname:<12} {ma:>8.0f} {mb:>8.0f} {p:>10.4f} {d:>+8.3f} {sig:>5}")

    # EGGROLL vs Evict-Restart head to head
    print("\n" + "=" * 90)
    print("HEAD TO HEAD: EGGROLL vs Evict-Restart")
    print("=" * 90)
    for dname in domains:
        print(f"\n--- {dname} ---")
        for pname in personas:
            a = all_data[(dname, "EGGROLL", pname)]
            b = all_data[(dname, "Evict-Restart", pname)]
            U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            ma, mb = np.mean(a), np.mean(b)
            print(f"  {pname:<12} EGGROLL={ma:>8.0f}  EvRst={mb:>8.0f}  p={p:.4f} {'*' if p<0.05 else ''}")

    # Aggregates
    print("\n" + "=" * 90)
    print("GRAND TOTAL")
    print("-" * 50)
    for aname, _, _ in algorithms:
        total_qd = sum(np.mean(all_data.get((dn, aname, pn), [0]))
                       for dn in domains for pn in personas)
        print(f"  {aname:<20} {total_qd:>12.1f}")

    # Save
    out = {f"{d}|{a}|{p}": v for (d, a, p), v in all_data.items()}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              "full_comparison.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to full_comparison.json")


if __name__ == "__main__":
    main()
