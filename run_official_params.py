#!/usr/bin/env python3
"""
Run top algorithms under OFFICIAL QDA-VC parameters.

Official (from framework/Tests/RunExperiments-TTP.py):
  NUM_GENS = 300, POP_SIZE = 200, MEMORY = 500, INTERVAL = 50
  cross_over_rate = 0.5, mutation_rate = 0.1, TRIALS = 25

Our previous runs used: 100 gens, pop 50, memory 200, interval 10, xo 0.7, mut 0.3

Results may differ significantly. This validates the competition entry.
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

# Official parameters
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
    from Algorithms.VCMapElites import VariableConstraintMapElites

    from mvp13_adaptive_rate import AdaptiveRateElites
    from mvp6_evict_restart import EvictRestartElites
    from mvp7_bandit_experts import BanditExpertElites
    from mvp18_epsilon_bandit import EpsilonBanditElites
    from mvp22_ultimate_hybrid import UltimateHybridElites

    algo_map = {
        "Baseline": (VariableConstraintMapElites, {}),
        "UH-nobandit": (UltimateHybridElites, {"use_bandit": False}),
        "AdaptRate-noreset": (AdaptiveRateElites, {"reset_on_change": False}),
        "Evict-Restart": (EvictRestartElites, {}),
        "Bandit (UCB1)": (BanditExpertElites, {}),
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
        algo = algo_class(ps, N_GENS, POP_SIZE, MEMORY, XO_RATE, MUT_RATE, user, INTERVAL, **kwargs)
        algo.run()
        m = algo.measure_history
        qd = m.qd_score[-1] if m.qd_score else 0.0
    except Exception as e:
        qd = 0.0

    return (algo_name, domain_name, persona_name, seed, qd)


def main():
    n_seeds = 25  # match official TRIALS
    n_workers = min(cpu_count(), 60)

    algorithms = [
        "Baseline", "UH-nobandit", "AdaptRate-noreset",
        "Evict-Restart", "Bandit (UCB1)", "Epsilon-Bandit",
    ]
    domains = ["TTP", "LogicPuzzles"]
    personas = ["Exploratory", "Cycle", "Adaptive"]

    total_tasks = len(algorithms) * len(domains) * len(personas) * n_seeds
    print(f"Official Parameter Validation Run")
    print(f"  Parameters: gens={N_GENS} pop={POP_SIZE} mem={MEMORY} interval={INTERVAL} xo={XO_RATE} mut={MUT_RATE}")
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
            if completed % 50 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (total_tasks - completed) / rate
                print(f"  {completed}/{total_tasks} ({completed/total_tasks*100:.0f}%) "
                      f"| {elapsed:.0f}s | ~{eta:.0f}s remaining", flush=True)

    elapsed = time.time() - t0
    print(f"\nCompleted {total_tasks} tasks in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    outpath = os.path.join(SCRIPT_DIR, "results", "official_params.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved to {outpath}")

    # Print results
    print("\n" + "=" * 100)
    print(f"OFFICIAL PARAMS: gens={N_GENS} pop={POP_SIZE} mem={MEMORY} interval={INTERVAL}")
    print("=" * 100)

    for dname in domains:
        print(f"\n  {dname}:")
        print(f"  {'Algorithm':<22} " + " ".join(f"{p:>14}" for p in personas) + f" {'Total':>14}")
        print(f"  {'-'*22} " + " ".join(f"{'-'*14}" for _ in personas) + f" {'-'*14}")
        totals_list = []
        for aname in algorithms:
            vals = []
            for pname in personas:
                qds = all_data.get(f"{dname}|{aname}|{pname}", [0])
                vals.append(np.mean(qds))
            total = sum(vals)
            totals_list.append((total, aname, vals))
        totals_list.sort(reverse=True)
        for total, aname, vals in totals_list:
            print(f"  {aname:<22} " +
                  " ".join(f"{v:>14.1f}" for v in vals) +
                  f" {total:>14.1f}")

    # Stat tests
    print("\n  UH-nobandit vs Baseline:")
    for dname in domains:
        for pname in personas:
            a = np.array(all_data.get(f"{dname}|UH-nobandit|{pname}", [0]))
            b = np.array(all_data.get(f"{dname}|Baseline|{pname}", [0]))
            try:
                U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            except:
                p = 1.0
            ma, mb = np.mean(a), np.mean(b)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print(f"    {dname:14} {pname:12} {ma:>10.1f} vs {mb:>10.1f} p={p:.4f} {sig}")


if __name__ == "__main__":
    main()
