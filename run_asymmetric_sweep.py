#!/usr/bin/env python3
"""
Asymmetric adaptation sweep + parameter optimization at official params.

Tests asymmetric increase/decrease factors on the Shuffling+AdaptRate base.
Also sweeps target_success and infeasible_rate.

Conditions:
  - Symmetric baselines: f=1.2/1.2 (original), f=1.1/1.1
  - Asymmetric: 1.1/1.3, 1.1/1.5, 1.1/2.0, 1.05/1.5, 1.2/1.5
  - Target success: 0.1, 0.2, 0.3 (with best asymmetric)
  - Framework Shuffling baseline

Official params: 300 gens, pop 200, memory 500, interval 50
Domains: TTP only (LogicPuzzles scores too small to differentiate)
Personas: Exploratory, Cycle, Adaptive, Strict
Seeds: 50

10 configs * 4 personas * 50 seeds = 2,000 tasks
"""

import sys
import os
import json
import time
import random
import copy
import math
from multiprocessing import Pool, cpu_count

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
    algo_name, persona_name, seed, params = args

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

    # Fix race condition: lock around nextProblem.txt access
    problem_idx = seed % 6
    next_problem_file = os.path.join(TESTS_DIR, "..", "ProblemSpaces", "TravelingThief", "nextProblem.txt")

    global _ttp_lock
    with _ttp_lock:
        with open(next_problem_file, "w") as f:
            f.write(str(problem_idx))
        from ProblemSpaces.TravelingThief.TTP_ProblemSpace import TTPProblemSpace
        ps = TTPProblemSpace()

    # ps already constructed above under lock
    from Personas.Exploratory import ExploratoryUser
    from Personas.TwoForwardOneBack import TwoForOneBackUser
    from Personas.Adaptive import AdaptiveUser
    from Personas.Strict import StrictUser
    from Algorithms.Shuffling import Shuffling
    from mvp22_ultimate_hybrid import UltimateHybridElites

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
        else:
            # Use UltimateHybridElites with custom rate adaptation
            algo = UltimateHybridElites(
                ps, N_GENS, POP_SIZE, MEMORY, XO_RATE, MUT_RATE, user, INTERVAL,
                target_success=params.get("target", 0.2),
                adapt_factor=params.get("inc", 1.2),  # used as base, we override run_one_generation
                infeasible_rate=params.get("infeasible_rate", 0.5),
            )
            # Monkey-patch for asymmetric adaptation
            inc_factor = params.get("inc", 1.2)
            dec_factor = params.get("dec", 1.2)
            target = params.get("target", 0.2)
            rate_min = 0.05
            rate_max = 0.9

            original_run = algo.run_one_generation

            def make_patched(algo_ref, inc_f, dec_f, tgt, rmin, rmax):
                def patched_run(cons_changed):
                    if cons_changed:
                        algo_ref.re_shuffle()

                    algo_ref.infeasible_pop.sort(key=lambda x: x[0], reverse=True)
                    new_infeasible = algo_ref.infeasible_pop[:algo_ref.elitism_num]

                    successes = 0
                    total = 0

                    for _ in range(algo_ref.population_size // 2):
                        child1 = algo_ref.select()
                        child2 = algo_ref.select()
                        if _random.random() < algo_ref.cross_over_rate:
                            child1, child2 = algo_ref.problem_space.cross_over(child1, child2)
                        child1 = algo_ref.problem_space.mutate(child1, algo_ref.current_rate)
                        child2 = algo_ref.problem_space.mutate(child2, algo_ref.current_rate)
                        s1 = algo_ref.place_in_bin(child1, new_infeasible)
                        s2 = algo_ref.place_in_bin(child2, new_infeasible)
                        successes += int(s1) + int(s2)
                        total += 2

                    if total > 0:
                        sr = successes / total
                        if sr > tgt:
                            algo_ref.current_rate = min(rmax, algo_ref.current_rate * inc_f)
                        else:
                            algo_ref.current_rate = max(rmin, algo_ref.current_rate / dec_f)

                    algo_ref.infeasible_pop = new_infeasible
                    return algo_ref.bins
                return patched_run

            algo.run_one_generation = make_patched(algo, inc_factor, dec_factor, target, rate_min, rate_max)

        algo.run()
        m = algo.measure_history
        qd = m.qd_score[-1] if m.qd_score else 0.0
    except Exception as e:
        print(f"ERROR: {algo_name}/{persona_name}/seed{seed}: {e}", flush=True)
        qd = 0.0

    return (algo_name, persona_name, seed, qd)


def main():
    n_seeds = 50
    n_workers = min(cpu_count(), 60)

    # Define configurations
    configs = {
        "Shuffling": {"type": "baseline"},
        "Sym-1.2": {"inc": 1.2, "dec": 1.2, "target": 0.2},
        "Sym-1.1": {"inc": 1.1, "dec": 1.1, "target": 0.2},
        "Asym-1.1/1.3": {"inc": 1.1, "dec": 1.3, "target": 0.2},
        "Asym-1.1/1.5": {"inc": 1.1, "dec": 1.5, "target": 0.2},
        "Asym-1.1/2.0": {"inc": 1.1, "dec": 2.0, "target": 0.2},
        "Asym-1.05/1.5": {"inc": 1.05, "dec": 1.5, "target": 0.2},
        "Asym-1.2/1.5": {"inc": 1.2, "dec": 1.5, "target": 0.2},
        # Target success sweep (with best asymmetric)
        "A1.1/1.5-t0.1": {"inc": 1.1, "dec": 1.5, "target": 0.1},
        "A1.1/1.5-t0.3": {"inc": 1.1, "dec": 1.5, "target": 0.3},
    }

    personas = ["Exploratory", "Cycle", "Adaptive", "Strict"]

    total_tasks = len(configs) * len(personas) * n_seeds
    print(f"Asymmetric Adaptation Sweep (official params, TTP only)")
    print(f"  Parameters: gens={N_GENS} pop={POP_SIZE} mem={MEMORY} interval={INTERVAL}")
    print(f"  Configs: {len(configs)}")
    print(f"  Personas: {personas}")
    print(f"  Seeds: {n_seeds}")
    print(f"  Total tasks: {total_tasks}")
    print(f"  Workers: {n_workers}")
    print("=" * 70)

    tasks = []
    for aname, params in configs.items():
        for pname in personas:
            for seed in range(n_seeds):
                tasks.append((aname, pname, seed, params))

    random.shuffle(tasks)

    all_data = {}
    t0 = time.time()
    completed = 0

    from multiprocessing import Lock
    lock = Lock()

    with Pool(n_workers, initializer=init_worker, initargs=(lock,)) as pool:
        for result in pool.imap_unordered(run_one, tasks):
            aname, pname, seed, qd = result
            key = f"{aname}|{pname}"
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

    outpath = os.path.join(SCRIPT_DIR, "results", "asymmetric_sweep.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved to {outpath}")

    # === Print results ===
    shuf_total = sum(np.mean(all_data.get(f"Shuffling|{p}", [0])) for p in personas)

    print("\n" + "=" * 90)
    print(f"{'Config':<20} {'Explor':>8} {'Cycle':>8} {'Adapt':>8} {'Strict':>8} {'Total':>8} {'vs Shuf':>8}")
    print("-" * 90)

    rows = []
    for aname in configs:
        vals = [np.mean(all_data.get(f"{aname}|{p}", [0])) for p in personas]
        total = sum(vals)
        rows.append((total, aname, vals))
    rows.sort(reverse=True)

    for total, aname, vals in rows:
        pct = (total - shuf_total) / abs(shuf_total) * 100 if shuf_total else 0
        print(f"  {aname:<20} {vals[0]:>8.0f} {vals[1]:>8.0f} {vals[2]:>8.0f} {vals[3]:>8.0f} {total:>8.0f} {pct:>+7.1f}%")

    # Stats for top config vs Shuffling
    print("\n" + "=" * 90)
    print("SIGNIFICANCE (top configs vs Shuffling, Mann-Whitney U)")
    print("=" * 90)
    top_configs = [name for _, name, _ in rows[:5] if name != "Shuffling"]
    for aname in top_configs:
        for pname in personas:
            a = np.array(all_data.get(f"{aname}|{pname}", [0]))
            b = np.array(all_data.get(f"Shuffling|{pname}", [0]))
            ma, mb = np.mean(a), np.mean(b)
            try:
                U, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
            except:
                pval = 1.0
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
            print(f"  {aname:<20} {pname:<12} {ma:>7.0f} vs {mb:>7.0f} p={pval:.4f} {sig}")
        print()


if __name__ == "__main__":
    main()
