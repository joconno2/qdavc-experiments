#!/usr/bin/env python3
"""
Instrumented runs with fixed placement code. Generates per-generation traces
for paper figures: mutation rate adaptation, QD score, diversity over time.

Runs at official params on TTP with all 4 personas. Single seed per config
(deterministic, reproducible). Outputs JSON traces + matplotlib figures.
"""

import sys
import os
import json
import random
import copy
import math

import numpy as np
import numpy.random as npr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.path.join(SCRIPT_DIR, "framework")
TESTS_DIR = os.path.join(FRAMEWORK_DIR, "Tests")
ALGO_DIR = os.path.join(SCRIPT_DIR, "algorithms")

sys.path.insert(0, FRAMEWORK_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, ALGO_DIR)
os.chdir(TESTS_DIR)

from ProblemSpaces.TravelingThief.TTP_ProblemSpace import TTPProblemSpace
from Personas.Exploratory import ExploratoryUser
from Personas.TwoForwardOneBack import TwoForOneBackUser
from Personas.Adaptive import AdaptiveUser
from Personas.Strict import StrictUser
from Algorithms.Shuffling import Shuffling
from mvp22_ultimate_hybrid import UltimateHybridElites

N_GENS = 300
POP_SIZE = 200
MEMORY = 500
XO_RATE = 0.5
MUT_RATE = 0.1
INTERVAL = 50


def instrumented_run(algo_class, kwargs, persona_class, seed, label):
    """Run one algorithm with full per-generation state logging."""
    # Set problem to n50_bounded_strongly (index 0)
    next_problem_file = os.path.join(TESTS_DIR, "..", "ProblemSpaces",
                                     "TravelingThief", "nextProblem.txt")
    with open(next_problem_file, "w") as f:
        f.write("0")

    random.seed(seed)
    np.random.seed(seed)

    ps = TTPProblemSpace()
    user = persona_class(ps)
    algo = algo_class(ps, N_GENS, POP_SIZE, MEMORY, XO_RATE, MUT_RATE, user, INTERVAL, **kwargs)

    trace = {
        "label": label,
        "qd": [],
        "diversity": [],
        "mutation_rate": [],
        "n_constraints": [],
        "constraint_changes": [],
    }

    original_run_one = algo.run_one_generation

    def patched_run_one(cons_changed):
        result = original_run_one(cons_changed)

        if isinstance(result, list):
            pop = result
        else:
            pop = result

        # QD score and diversity
        fitnesses = []
        n_occupied = 0
        for b in pop:
            if b:
                n_occupied += 1
                if isinstance(b[0], tuple):
                    fitnesses.append(b[0][0])
                else:
                    fitnesses.append(b[0])

        qd = sum(fitnesses)
        diversity = n_occupied / len(pop) if pop else 0

        trace["qd"].append(float(qd))
        trace["diversity"].append(float(diversity))
        trace["constraint_changes"].append(bool(cons_changed))
        trace["n_constraints"].append(len(algo.variable_constraints))

        # Mutation rate (if available)
        if hasattr(algo, "current_rate"):
            trace["mutation_rate"].append(float(algo.current_rate))
        else:
            trace["mutation_rate"].append(float(MUT_RATE))

        return result

    algo.run_one_generation = patched_run_one
    algo.run()
    return trace


def make_figures(all_traces, output_dir):
    """Generate publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib not available, skipping figures")
        return

    os.makedirs(output_dir, exist_ok=True)

    personas = ["Exploratory", "Cycle", "Adaptive", "Strict"]
    colors = {"UH-nobandit": "#2196F3", "Shuffling": "#9E9E9E"}

    for persona in personas:
        uh_trace = next((t for t in all_traces if t["label"] == f"UH-nobandit/{persona}"), None)
        sh_trace = next((t for t in all_traces if t["label"] == f"Shuffling/{persona}"), None)
        if not uh_trace or not sh_trace:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        gens = range(len(uh_trace["qd"]))

        # Panel 1: QD score
        ax = axes[0]
        ax.plot(gens, uh_trace["qd"], color=colors["UH-nobandit"], label="UH-nobandit", linewidth=1.5)
        ax.plot(gens, sh_trace["qd"], color=colors["Shuffling"], label="Shuffling", linewidth=1.5, linestyle="--")
        # Mark constraint changes
        for i, changed in enumerate(uh_trace["constraint_changes"]):
            if changed:
                ax.axvline(i, color="#EF5350", alpha=0.3, linewidth=0.5)
        ax.set_ylabel("QD Score")
        ax.legend(loc="upper left")
        ax.set_title(f"TTP / {persona} Persona (official params)")

        # Panel 2: Mutation rate
        ax = axes[1]
        ax.plot(gens, uh_trace["mutation_rate"], color=colors["UH-nobandit"], linewidth=1.5)
        ax.axhline(MUT_RATE, color=colors["Shuffling"], linestyle="--", alpha=0.7, label=f"Shuffling (fixed {MUT_RATE})")
        for i, changed in enumerate(uh_trace["constraint_changes"]):
            if changed:
                ax.axvline(i, color="#EF5350", alpha=0.3, linewidth=0.5)
        ax.set_ylabel("Mutation Rate")
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper left")

        # Panel 3: Constraint count
        ax = axes[2]
        ax.plot(gens, uh_trace["n_constraints"], color="#4CAF50", linewidth=1.5)
        ax.set_ylabel("# Variable Constraints")
        ax.set_xlabel("Generation")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"trace_{persona.lower()}.png"), dpi=150)
        plt.close()
        print(f"  Saved trace_{persona.lower()}.png")


def main():
    seed = 42
    personas = [
        ("Exploratory", ExploratoryUser),
        ("Cycle", TwoForOneBackUser),
        ("Adaptive", AdaptiveUser),
        ("Strict", StrictUser),
    ]

    all_traces = []

    for pname, pcls in personas:
        print(f"Running UH-nobandit / {pname}...")
        trace = instrumented_run(UltimateHybridElites, {}, pcls, seed, f"UH-nobandit/{pname}")
        all_traces.append(trace)
        print(f"  Final QD: {trace['qd'][-1]:.0f}, final rate: {trace['mutation_rate'][-1]:.3f}")

        print(f"Running Shuffling / {pname}...")
        trace = instrumented_run(Shuffling, {}, pcls, seed, f"Shuffling/{pname}")
        all_traces.append(trace)
        print(f"  Final QD: {trace['qd'][-1]:.0f}")

    # Save traces
    outpath = os.path.join(SCRIPT_DIR, "results", "traces_official_fixed.json")
    with open(outpath, "w") as f:
        json.dump(all_traces, f)
    print(f"\nTraces saved to {outpath}")

    # Generate figures
    fig_dir = os.path.join(SCRIPT_DIR, "results", "figures_fixed")
    make_figures(all_traces, fig_dir)


if __name__ == "__main__":
    main()
