#!/usr/bin/env python3
"""
Test all QDA-VC MVP algorithms against baselines.
Each algorithm is in its own file under algorithms/.
"""

import sys
import os
import json
import importlib

framework_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "framework")
tests_dir = os.path.join(framework_dir, "Tests")
sys.path.insert(0, framework_dir)
sys.path.insert(0, tests_dir)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "algorithms"))
os.chdir(tests_dir)

from ProblemSpaces.TravelingThief.TTP_ProblemSpace import TTPProblemSpace
from Personas.Exploratory import ExploratoryUser
from Personas.TwoForwardOneBack import TwoForOneBackUser
from Personas.Adaptive import AdaptiveUser
from Algorithms.VCMapElites import VariableConstraintMapElites


def run_one(algo_class, n_gens=100, pop_size=50, max_memory=200, **kwargs):
    ps = TTPProblemSpace()
    personas = [
        ("Exploratory", ExploratoryUser),
        ("Cycle", TwoForOneBackUser),
        ("Adaptive", AdaptiveUser),
    ]
    results = {}
    for pname, pcls in personas:
        user = pcls(ps)
        algo = algo_class(ps, n_gens, pop_size, max_memory, 0.7, 0.3, user, 10, **kwargs)
        try:
            algo.run()
            m = algo.measure_history
            qd = m.qd_score[-1] if m.qd_score else 0
            div = m.diversity[-1] if m.diversity else 0
        except Exception as e:
            qd, div = 0, 0
        results[pname] = {"qd": qd, "div": div}
        # Need fresh problem space for each persona (rotates TTP problems)
        ps = TTPProblemSpace()
    return results


def run_n(algo_class, n_runs=5, **kwargs):
    all_runs = []
    for _ in range(n_runs):
        all_runs.append(run_one(algo_class, **kwargs))
    # Average
    personas = list(all_runs[0].keys())
    avg = {}
    for p in personas:
        avg[p] = {
            "qd": sum(r[p]["qd"] for r in all_runs) / n_runs,
            "div": sum(r[p]["div"] for r in all_runs) / n_runs,
        }
    return avg


def print_results(name, results):
    total = sum(results[p]["qd"] for p in results)
    for p in results:
        print(f"  {name:<25} {p:<12} QD={results[p]['qd']:>10.1f}  Div={results[p]['div']:.3f}")
    print(f"  {name:<25} {'TOTAL':<12} QD={total:>10.1f}")
    print()
    return total


if __name__ == "__main__":
    n_runs = 5
    print(f"QDA-VC MVP Comparison | TTP | {n_runs} runs each\n")

    # Baseline
    print("--- Baseline ---")
    base = run_n(VariableConstraintMapElites, n_runs=n_runs)
    base_total = print_results("VC-MAP-Elites", base)

    # Load all MVP algorithms from algorithms/ directory
    algo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "algorithms")
    sys.path.insert(0, algo_dir)

    mvp_results = {"VC-MAP-Elites": {"results": base, "total": base_total}}

    print("--- MVPs ---")
    for fname in sorted(os.listdir(algo_dir)):
        if fname.endswith(".py") and fname != "__init__.py":
            mod_name = fname[:-3]
            try:
                mod = importlib.import_module(mod_name)
                # Each module should have an ALGORITHM_CLASS and ALGORITHM_NAME
                algo_class = getattr(mod, "ALGORITHM_CLASS")
                algo_name = getattr(mod, "ALGORITHM_NAME")
                algo_kwargs = getattr(mod, "ALGORITHM_KWARGS", {})

                results = run_n(algo_class, n_runs=n_runs, **algo_kwargs)
                total = print_results(algo_name, results)
                mvp_results[algo_name] = {"results": results, "total": total}
            except Exception as e:
                print(f"  FAILED: {mod_name}: {e}")
                import traceback
                traceback.print_exc()
                print()

    # Summary
    print("=" * 60)
    print(f"{'Algorithm':<25} {'Total QD':>10}  {'vs Base':>10}")
    print("-" * 50)
    for name, data in sorted(mvp_results.items(), key=lambda x: -x[1]["total"]):
        diff = (data["total"] - base_total) / abs(base_total) * 100 if base_total != 0 else 0
        print(f"{name:<25} {data['total']:>10.1f}  {diff:>+9.1f}%")

    with open("mvp_comparison.json", "w") as f:
        json.dump({k: {"total": v["total"],
                       **{p: v["results"][p] for p in v["results"]}}
                   for k, v in mvp_results.items()}, f, indent=2)
    print(f"\nSaved to mvp_comparison.json")
