#!/usr/bin/env python3
"""
Quick MVP test: run PAL-Elites against the competition baselines
on one problem space + one persona, compare QD-score and diversity.
"""

import sys
import os
import json

# Add framework to path and change to Tests/ (framework expects relative paths from there)
framework_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "framework")
tests_dir = os.path.join(framework_dir, "Tests")
sys.path.insert(0, framework_dir)
sys.path.insert(0, tests_dir)
os.chdir(tests_dir)

from ProblemSpaces.TravelingThief.TTP_ProblemSpace import TTPProblemSpace
# LogicPuzzles has broken relative imports, skip for now
# from ProblemSpaces.LogicPuzzles.LogicPuzzleSpace import LogicPuzzleSpace
from Personas.Exploratory import ExploratoryUser
from Personas.TwoForwardOneBack import TwoForOneBackUser
from Personas.Strict import StrictUser
from Personas.Adaptive import AdaptiveUser

# Import baselines
from Algorithms.Filtering import Filtering
from Algorithms.RandomRestarts import RandomRestarts
from Algorithms.Shuffling import Shuffling
from Algorithms.VCMapElites import VariableConstraintMapElites

# Import our algorithm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pal_elites import PALElites

# Import Adaptive Restart MAP-Elites for comparison
ar_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adaptive-restart-elites")
sys.path.insert(0, ar_path)
try:
    from adaptive_restart_elites import PALElites as AdaptiveRestartElites
except ImportError:
    AdaptiveRestartElites = None

# Import our algorithm (add pal-elites dir to path)
sys.path.insert(0, os.path.dirname(__file__))
from pal_elites import PALElites


def run_experiment(algo_class, algo_name, problem_space, persona_class,
                   persona_name, n_gens=100, pop_size=50, max_memory=200,
                   n_runs=3, **algo_kwargs):
    """Run an algorithm multiple times and return average metrics."""
    results = []

    for run in range(n_runs):
        ps = problem_space()
        user = persona_class(ps)

        algo = algo_class(
            problem_space=ps,
            number_generations=n_gens,
            population_size=pop_size,
            max_memory=max_memory,
            cross_over_rate=0.7,
            mutation_rate=0.3,
            user=user,
            update_interval=10,
            **algo_kwargs,
        )

        try:
            algo.run()
        except Exception as e:
            import traceback
            print(f"  ERROR in {algo_name}/{persona_name} run {run}: {e}")
            traceback.print_exc()
            continue

        m = algo.measure_history
        final_qd = m.qd_score[-1] if m.qd_score else 0
        final_div = m.diversity[-1] if m.diversity else 0
        final_qual = m.quality[-1] if m.quality else 0
        mean_qd = sum(m.qd_score) / len(m.qd_score) if m.qd_score else 0

        results.append({
            "final_qd": final_qd,
            "final_div": final_div,
            "final_quality": final_qual,
            "mean_qd": mean_qd,
        })

    avg = {k: sum(r[k] for r in results) / len(results) for k in results[0]}
    return avg


def main():
    n_gens = 100
    pop_size = 50
    max_memory = 200
    n_runs = 3

    algorithms = [
        ("VC-MAP-Elites", VariableConstraintMapElites, {}),
        ("PAL+Restart", PALElites, {"use_bias": True}),
        ("Restart-only", PALElites, {"use_bias": False}),
    ]

    personas = [
        ("Exploratory", ExploratoryUser),
        ("Cycle", TwoForOneBackUser),
        ("Adaptive", AdaptiveUser),
    ]

    domains = [
        ("TravelingThief", TTPProblemSpace),
    ]

    all_results = []
    for domain_name, problem_space in domains:
        print(f"\nDomain: {domain_name} | {n_gens} gens | {n_runs} runs each")
        print(f"{'Algorithm':<20} {'Persona':<14} {'QD-Score':>10} {'Diversity':>10} {'Quality':>10}")
        print("-" * 70)

        for algo_name, algo_class, kwargs in algorithms:
            for persona_name, persona_class in personas:
                avg = run_experiment(
                    algo_class, algo_name, problem_space, persona_class,
                    persona_name, n_gens=n_gens, pop_size=pop_size,
                    max_memory=max_memory, n_runs=n_runs, **kwargs,
                )
                print(f"{algo_name:<20} {persona_name:<14} {avg['final_qd']:>10.1f} "
                      f"{avg['final_div']:>10.3f} {avg['final_quality']:>10.1f}")
                all_results.append({
                    "domain": domain_name,
                    "algorithm": algo_name,
                    "persona": persona_name,
                    **avg,
                })

    # Save results
    with open("mvp_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to mvp_results.json")


if __name__ == "__main__":
    main()
