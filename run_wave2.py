#!/usr/bin/env python3
"""
Wave 2: Targeted ablations and hybrids based on wave 1 results.

New conditions:
  - UCB c sweep: c=0.25, 0.5, 2.0, 4.0 (Bandit Experts with different exploration)
  - Bandit+Evict+Memory: Triple hybrid
  - Bandit-5Expert: 5 mutation experts (adds SCOPE multi-step)
  - Bandit+Evict with c sweep: c=0.5, 2.0
  - Direction history sweep: n=3, 5, 10, 20

Total new conditions: ~12
30 seeds x 2 domains x 3 personas x 12 = 2160 tasks
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

    from mvp7_bandit_experts import BanditExpertElites
    from mvp9_bandit_evict import BanditEvictElites
    from mvp4_eggroll_lowrank import EGGROLLElites
    from mvp13_adaptive_rate import AdaptiveRateElites
    from mvp20_bandit_evict_memory import BanditEvictMemoryElites
    from mvp21_bandit_5expert import Bandit5ExpertElites
    from mvp22_ultimate_hybrid import UltimateHybridElites

    algo_map = {
        # References from wave 1
        "Baseline": (VariableConstraintMapElites, {}),
        "Bandit": (BanditExpertElites, {}),
        "Bandit+Evict": (BanditEvictElites, {}),
        "AdaptRate-noreset": (AdaptiveRateElites, {"reset_on_change": False}),

        # === ULTIMATE HYBRID (wave 1 winners combined) ===
        "Ultimate-Hybrid": (UltimateHybridElites, {}),
        "UH-noevict": (UltimateHybridElites, {"use_eviction": False}),
        "UH-nobandit": (UltimateHybridElites, {"use_bandit": False}),

        # Adaptive rate factor sweep (noreset)
        "AR-f1.1": (AdaptiveRateElites, {"reset_on_change": False, "adapt_factor": 1.1}),
        "AR-f1.5": (AdaptiveRateElites, {"reset_on_change": False, "adapt_factor": 1.5}),
        "AR-f2.0": (AdaptiveRateElites, {"reset_on_change": False, "adapt_factor": 2.0}),

        # Target success rate sweep (noreset)
        "AR-t0.1": (AdaptiveRateElites, {"reset_on_change": False, "target_success": 0.1}),
        "AR-t0.3": (AdaptiveRateElites, {"reset_on_change": False, "target_success": 0.3}),
        "AR-t0.5": (AdaptiveRateElites, {"reset_on_change": False, "target_success": 0.5}),

        # UCB c sweep on Bandit+Evict (no stagnation, since that won)
        "BdEvNS-c0.5": (BanditEvictElites, {"ucb_c": 0.5, "use_stagnation": False}),
        "BdEvNS-c2.0": (BanditEvictElites, {"ucb_c": 2.0, "use_stagnation": False}),
        "BdEvNS-c4.0": (BanditEvictElites, {"ucb_c": 4.0, "use_stagnation": False}),

        # 5-expert bandit
        "Bandit-5Expert": (Bandit5ExpertElites, {}),

        # Direction history size sweep
        "Bandit-dir3": (BanditExpertElites, {"n_direction_history": 3}),
        "Bandit-dir10": (BanditExpertElites, {"n_direction_history": 10}),
        "Bandit-dir20": (BanditExpertElites, {"n_direction_history": 20}),

        # EGGROLL direction history sweep
        "EGGROLL-dir3": (EGGROLLElites, {"n_directions": 3}),
        "EGGROLL-dir10": (EGGROLLElites, {"n_directions": 10}),
        "EGGROLL-dir20": (EGGROLLElites, {"n_directions": 20}),
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


def main():
    n_seeds = 30
    n_workers = min(cpu_count(), 60)

    algorithms = [
        "Baseline", "Bandit", "Bandit+Evict", "AdaptRate-noreset",
        "Ultimate-Hybrid", "UH-noevict", "UH-nobandit",
        "AR-f1.1", "AR-f1.5", "AR-f2.0",
        "AR-t0.1", "AR-t0.3", "AR-t0.5",
        "BdEvNS-c0.5", "BdEvNS-c2.0", "BdEvNS-c4.0",
        "Bandit-5Expert",
        "Bandit-dir3", "Bandit-dir10", "Bandit-dir20",
        "EGGROLL-dir3", "EGGROLL-dir10", "EGGROLL-dir20",
    ]
    domains = ["TTP", "LogicPuzzles"]
    personas = ["Exploratory", "Cycle", "Adaptive"]

    total_tasks = len(algorithms) * len(domains) * len(personas) * n_seeds
    print(f"QDA-VC Wave 2: Parameter Sweeps & Hybrids")
    print(f"  Algorithms: {len(algorithms)}")
    print(f"  Domains: {len(domains)}")
    print(f"  Personas: {len(personas)}")
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
            if completed % 100 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (total_tasks - completed) / rate
                print(f"  {completed}/{total_tasks} ({completed/total_tasks*100:.0f}%) "
                      f"| {elapsed:.0f}s | ~{eta:.0f}s remaining", flush=True)

    elapsed = time.time() - t0
    print(f"\nCompleted {total_tasks} tasks in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    outpath = os.path.join(SCRIPT_DIR, "results", "wave2_results.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved to {outpath}")

    # Print rankings
    print("\n" + "=" * 90)
    print("GRAND TOTAL RANKING")
    print("=" * 90)

    totals = []
    for a in algorithms:
        total = sum(np.mean(all_data.get(f"{d}|{a}|{p}", [0]))
                    for d in domains for p in personas)
        totals.append((total, a))
    totals.sort(reverse=True)
    base_total = next((t for t, n in totals if n == "Baseline"), 0)

    for rank, (total, aname) in enumerate(totals, 1):
        pct = ((total - base_total) / abs(base_total) * 100) if base_total != 0 else 0
        print(f"  {rank:<3} {aname:<24} {total:>12.1f}  ({pct:>+7.1f}%)")

    # UCB c sweep analysis
    print("\n" + "=" * 90)
    print("UCB c SWEEP (Bandit Experts)")
    print("=" * 90)
    c_names = ["Bandit-c0.25", "Bandit-c0.5", "Bandit", "Bandit-c2.0", "Bandit-c4.0"]
    c_vals = [0.25, 0.5, 1.0, 2.0, 4.0]
    for d in domains:
        print(f"\n  {d}:")
        print(f"  {'c value':<10} " + " ".join(f"{p:>14}" for p in personas) + f" {'Total':>14}")
        for cn, cv in zip(c_names, c_vals):
            vals = [np.mean(all_data.get(f"{d}|{cn}|{p}", [0])) for p in personas]
            total = sum(vals)
            print(f"  {cv:<10} " + " ".join(f"{v:>14.1f}" for v in vals) + f" {total:>14.1f}")

    # Direction history sweep
    print("\n" + "=" * 90)
    print("DIRECTION HISTORY SIZE SWEEP")
    print("=" * 90)
    dir_names = ["Bandit-dir3", "Bandit", "Bandit-dir10", "Bandit-dir20"]
    dir_vals = [3, 5, 10, 20]
    for d in domains:
        print(f"\n  {d} (Bandit):")
        for dn, dv in zip(dir_names, dir_vals):
            vals = [np.mean(all_data.get(f"{d}|{dn}|{p}", [0])) for p in personas]
            total = sum(vals)
            print(f"    n={dv:<3}  " + " ".join(f"{v:>10.1f}" for v in vals) + f" total={total:>10.1f}")

    dir_names_eg = ["EGGROLL-dir3", "EGGROLL-dir10", "EGGROLL-dir20"]
    dir_vals_eg = [3, 10, 20]
    for d in domains:
        print(f"\n  {d} (EGGROLL):")
        for dn, dv in zip(dir_names_eg, dir_vals_eg):
            vals = [np.mean(all_data.get(f"{d}|{dn}|{p}", [0])) for p in personas]
            total = sum(vals)
            print(f"    n={dv:<3}  " + " ".join(f"{v:>10.1f}" for v in vals) + f" total={total:>10.1f}")


if __name__ == "__main__":
    main()
