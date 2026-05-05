#!/usr/bin/env python3
"""
Definitive paper run: 200 seeds, all conditions needed for publication.

Algorithms (7):
  1. Shuffling (framework baseline, strongest)
  2. VC-MAP-Elites (framework baseline, original)
  3. Asym-1.2/1.5 (competition entry - mvp24)
  4. Sym-1.2/1.2 (UH-nobandit - mvp22, symmetric ablation)
  5. Shuffling-only (no adaptive rate, just the base = plain Shuffling)
  6. AdaptRate-only (adaptive rate without Shuffling base, has placement bug on Strict)
  7. Fixed-rate on Shuffling base (Shuffling + fixed high mutation 0.3, ablates the adaptation)

This gives us:
  - Entry vs strongest baseline (Asym vs Shuffling)
  - Asymmetric vs symmetric ablation (Asym vs Sym)
  - Adaptive rate contribution (Asym vs Shuffling = adaptive rate adds what?)
  - Shuffling base contribution (Asym vs AdaptRate-only = Shuffling base adds what?)
  - Adaptation vs fixed high rate (Asym vs Fixed-0.3 = is adaptation better than just high mutation?)

Official params: 300 gens, pop 200, memory 500, interval 50, xo 0.5, mut 0.1
Domains: TTP, LogicPuzzles
Personas: Exploratory, Cycle, Adaptive, Strict
Seeds: 200

7 algos * 2 domains * 4 personas * 200 seeds = 11,200 tasks
"""

import sys
import os
import json
import time
import random
import copy
import math
from multiprocessing import Pool, Lock, cpu_count

import numpy as np
import numpy.random as npr
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

_ttp_lock = None


def init_worker(lock):
    global _ttp_lock
    _ttp_lock = lock


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

    # Lock around TTP problem space construction (nextProblem.txt race condition)
    problem_idx = seed % 6
    next_problem_file = os.path.join(TESTS_DIR, "..", "ProblemSpaces",
                                     "TravelingThief", "nextProblem.txt")

    global _ttp_lock
    if domain_name == "TTP":
        with _ttp_lock:
            with open(next_problem_file, "w") as f:
                f.write(str(problem_idx))
            from ProblemSpaces.TravelingThief.TTP_ProblemSpace import TTPProblemSpace
            ps = TTPProblemSpace()
    else:
        from ProblemSpaces.LogicPuzzles.LogicPuzzleSpace import LogicPuzzleSpace
        ps = LogicPuzzleSpace()

    from Personas.Exploratory import ExploratoryUser
    from Personas.TwoForwardOneBack import TwoForOneBackUser
    from Personas.Adaptive import AdaptiveUser
    from Personas.Strict import StrictUser
    from Algorithms.Shuffling import Shuffling
    from Algorithms.VCMapElites import VariableConstraintMapElites
    from mvp22_ultimate_hybrid import UltimateHybridElites
    from mvp24_asymmetric_adapt import AsymmetricAdaptElites
    from mvp13_adaptive_rate import AdaptiveRateElites

    persona_map = {
        "Exploratory": ExploratoryUser,
        "Cycle": TwoForOneBackUser,
        "Adaptive": AdaptiveUser,
        "Strict": StrictUser,
    }

    try:
        user = persona_map[persona_name](ps)

        if algo_name == "Shuffling":
            algo = Shuffling(ps, N_GENS, POP_SIZE, MEMORY, XO_RATE, MUT_RATE, user, INTERVAL)
        elif algo_name == "VC-MAP-Elites":
            algo = VariableConstraintMapElites(ps, N_GENS, POP_SIZE, MEMORY, XO_RATE, MUT_RATE, user, INTERVAL)
        elif algo_name == "Asym-1.2/1.5":
            algo = AsymmetricAdaptElites(ps, N_GENS, POP_SIZE, MEMORY, XO_RATE, MUT_RATE, user, INTERVAL,
                                         increase_factor=1.2, decrease_factor=1.5)
        elif algo_name == "Sym-1.2/1.2":
            algo = UltimateHybridElites(ps, N_GENS, POP_SIZE, MEMORY, XO_RATE, MUT_RATE, user, INTERVAL)
        elif algo_name == "AdaptRate-only":
            algo = AdaptiveRateElites(ps, N_GENS, POP_SIZE, MEMORY, XO_RATE, MUT_RATE, user, INTERVAL,
                                      reset_on_change=False)
        elif algo_name == "Fixed-0.3":
            # Shuffling base with fixed high mutation rate (no adaptation)
            algo = UltimateHybridElites(ps, N_GENS, POP_SIZE, MEMORY, XO_RATE, 0.3, user, INTERVAL,
                                         use_adaptive_rate=False)
        else:
            raise ValueError(f"Unknown algo: {algo_name}")

        algo.run()
        m = algo.measure_history
        qd = m.qd_score[-1] if m.qd_score else 0.0
    except Exception as e:
        print(f"ERROR: {algo_name}/{domain_name}/{persona_name}/seed{seed}: {e}", flush=True)
        qd = 0.0

    return (algo_name, domain_name, persona_name, seed, qd)


def main():
    n_seeds = 200
    n_workers = min(cpu_count(), 60)

    algorithms = [
        "Asym-1.2/1.5",    # Competition entry
        "Sym-1.2/1.2",     # Symmetric ablation (original UH-nobandit)
        "Shuffling",        # Strongest framework baseline
        "VC-MAP-Elites",    # Original framework baseline
        "AdaptRate-only",   # Adaptive rate without Shuffling base
        "Fixed-0.3",        # Shuffling base + fixed high mutation (no adaptation)
    ]
    domains = ["TTP", "LogicPuzzles"]
    personas = ["Exploratory", "Cycle", "Adaptive", "Strict"]

    total_tasks = len(algorithms) * len(domains) * len(personas) * n_seeds
    print(f"Paper Final Run (200 seeds, full ablation)")
    print(f"  Parameters: gens={N_GENS} pop={POP_SIZE} mem={MEMORY} interval={INTERVAL}")
    print(f"  Algorithms: {algorithms}")
    print(f"  Domains: {domains}")
    print(f"  Personas: {personas}")
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

    lock = Lock()
    with Pool(n_workers, initializer=init_worker, initargs=(lock,)) as pool:
        for result in pool.imap_unordered(run_one, tasks):
            aname, dname, pname, seed, qd = result
            key = f"{dname}|{aname}|{pname}"
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(qd)
            completed += 1
            if completed % 500 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (total_tasks - completed) / rate
                print(f"  {completed}/{total_tasks} ({completed/total_tasks*100:.0f}%) "
                      f"| {elapsed:.0f}s | ~{eta:.0f}s remaining", flush=True)

    elapsed = time.time() - t0
    print(f"\nCompleted {total_tasks} tasks in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    outpath = os.path.join(SCRIPT_DIR, "results", "paper_final.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved to {outpath}")

    # === Analysis ===
    shuf_total = sum(np.mean(all_data.get(f"{d}|Shuffling|{p}", [0]))
                     for d in domains for p in personas)

    print("\n" + "=" * 90)
    print("GRAND TOTAL RANKING")
    print("=" * 90)
    totals = []
    for a in algorithms:
        total = sum(np.mean(all_data.get(f"{d}|{a}|{p}", [0]))
                    for d in domains for p in personas)
        totals.append((total, a))
    totals.sort(reverse=True)
    for rank, (total, a) in enumerate(totals, 1):
        pct = (total - shuf_total) / abs(shuf_total) * 100 if shuf_total else 0
        print(f"  {rank}. {a:<22} {total:>10.0f}  ({pct:>+6.1f}% vs Shuffling)")

    # Per-domain per-persona
    for dname in domains:
        print(f"\n  {dname}:")
        for a in algorithms:
            vals = [np.mean(all_data.get(f"{dname}|{a}|{p}", [0])) for p in personas]
            print(f"    {a:<22} E={vals[0]:>7.0f} C={vals[1]:>7.0f} A={vals[2]:>7.0f} S={vals[3]:>7.0f}")

    # Statistical tests
    print("\n" + "=" * 90)
    print("SIGNIFICANCE vs Shuffling (Mann-Whitney U, n=200)")
    print("=" * 90)
    for a in algorithms:
        if a == "Shuffling":
            continue
        sig_count = 0
        for dname in domains:
            for pname in personas:
                av = np.array(all_data.get(f"{dname}|{a}|{pname}", [0]))
                sv = np.array(all_data.get(f"{dname}|Shuffling|{pname}", [0]))
                ma, ms = np.mean(av), np.mean(sv)
                try:
                    U, pval = stats.mannwhitneyu(av, sv, alternative="two-sided")
                except:
                    pval = 1.0
                sa, ss = np.std(av, ddof=1), np.std(sv, ddof=1)
                pooled = np.sqrt((sa**2 + ss**2) / 2)
                d_val = (ma - ms) / pooled if pooled > 0 else 0
                sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
                if pval < 0.05:
                    sig_count += 1
                print(f"  {a:<18} {dname:<12} {pname:<12} "
                      f"{ma:>7.0f} vs {ms:>7.0f} d={d_val:>+.2f} p={pval:.4f} {sig}")
        print(f"  [{a}: {sig_count}/8 significant]")
        print()

    # Ablation comparisons
    print("=" * 90)
    print("ABLATION: Asym-1.2/1.5 vs Sym-1.2/1.2 (asymmetry helps?)")
    print("=" * 90)
    for dname in domains:
        for pname in personas:
            av = np.array(all_data.get(f"{dname}|Asym-1.2/1.5|{pname}", [0]))
            sv = np.array(all_data.get(f"{dname}|Sym-1.2/1.2|{pname}", [0]))
            ma, ms = np.mean(av), np.mean(sv)
            try:
                U, pval = stats.mannwhitneyu(av, sv, alternative="two-sided")
            except:
                pval = 1.0
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
            print(f"  {dname:<12} {pname:<12} Asym={ma:>7.0f} Sym={ms:>7.0f} diff={ma-ms:>+7.0f} p={pval:.4f} {sig}")

    print("\n" + "=" * 90)
    print("ABLATION: Asym-1.2/1.5 vs Fixed-0.3 (adaptation vs fixed high rate)")
    print("=" * 90)
    for dname in domains:
        for pname in personas:
            av = np.array(all_data.get(f"{dname}|Asym-1.2/1.5|{pname}", [0]))
            sv = np.array(all_data.get(f"{dname}|Fixed-0.3|{pname}", [0]))
            ma, ms = np.mean(av), np.mean(sv)
            try:
                U, pval = stats.mannwhitneyu(av, sv, alternative="two-sided")
            except:
                pval = 1.0
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
            print(f"  {dname:<12} {pname:<12} Asym={ma:>7.0f} Fix={ms:>7.0f} diff={ma-ms:>+7.0f} p={pval:.4f} {sig}")


if __name__ == "__main__":
    main()
