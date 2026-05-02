#!/usr/bin/env python3
"""
Instrumented runs: capture per-generation internal state for visualization.

Logs per generation:
  - QD score, diversity, quality (from Measures)
  - Mutation rate (for adaptive rate algorithms)
  - Eviction pool size
  - Constraint change events
  - Bandit expert selection distribution (if applicable)
  - Number of occupied bins

Runs a small set of algorithms with full logging, then generates figures.
"""

import sys
import os
import json
import copy
import random
import numpy as np

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
from Algorithms.VCMapElites import VariableConstraintMapElites

from mvp4_eggroll_lowrank import EGGROLLElites
from mvp6_evict_restart import EvictRestartElites
from mvp7_bandit_experts import BanditExpertElites
from mvp13_adaptive_rate import AdaptiveRateElites
from mvp18_epsilon_bandit import EpsilonBanditElites
from mvp22_ultimate_hybrid import UltimateHybridElites


def instrumented_run(algo_class, kwargs, persona_class, seed, n_gens=100):
    """Run one algorithm with full per-generation state logging."""
    random.seed(seed)
    np.random.seed(seed)

    ps = TTPProblemSpace()
    user = persona_class(ps)
    algo = algo_class(ps, n_gens, 50, 200, 0.7, 0.3, user, 10, **kwargs)

    # Monkey-patch run() to capture internal state each generation
    trace = {
        "qd": [], "diversity": [], "occupied_bins": [],
        "mutation_rate": [], "eviction_pool_size": [],
        "constraint_changes": [], "n_constraints": [],
        "expert_dist": [],  # for bandit algorithms
    }

    original_run_one = algo.run_one_generation

    def patched_run_one(cons_changed):
        result = original_run_one(cons_changed)

        # Log state after generation
        if isinstance(result, list):
            pop = result
        elif isinstance(result, tuple):
            pop = result[0]
        else:
            pop = result

        fitnesses = []
        n_occupied = 0
        for b in pop:
            if b:
                n_occupied += 1
                fitnesses.append(b[0][0])  # best fitness in bin

        trace["qd"].append(float(sum(fitnesses)) if fitnesses else 0.0)
        trace["diversity"].append(n_occupied / len(pop) if pop else 0.0)
        trace["occupied_bins"].append(n_occupied)
        trace["constraint_changes"].append(bool(cons_changed))
        trace["n_constraints"].append(len(algo.variable_constraints))

        # Capture algorithm-specific state
        rate = getattr(algo, "current_rate", algo.mutation_rate)
        trace["mutation_rate"].append(float(rate))

        evpool = getattr(algo, "eviction_pool", [])
        trace["eviction_pool_size"].append(len(evpool))

        # Bandit expert distribution
        attempts = getattr(algo, "expert_attempts", None)
        if attempts is not None:
            total = sum(attempts)
            if total > 0:
                trace["expert_dist"].append([a / total for a in attempts])
            else:
                trace["expert_dist"].append([])
        else:
            trace["expert_dist"].append([])

        return result

    algo.run_one_generation = patched_run_one
    algo.run()

    return trace


def run_all(n_seeds=10):
    """Run instrumented versions of key algorithms."""
    algorithms = {
        "Baseline": (VariableConstraintMapElites, {}),
        "AdaptRate-noreset": (AdaptiveRateElites, {"reset_on_change": False}),
        "UH-nobandit": (UltimateHybridElites, {"use_bandit": False}),
        "Ultimate-Hybrid": (UltimateHybridElites, {}),
        "Evict-Restart": (EvictRestartElites, {}),
        "Bandit (UCB1)": (BanditExpertElites, {}),
        "Epsilon-Bandit": (EpsilonBanditElites, {}),
        "EGGROLL": (EGGROLLElites, {}),
    }

    personas = {
        "Exploratory": ExploratoryUser,
        "Cycle": TwoForOneBackUser,
        "Adaptive": AdaptiveUser,
    }

    all_traces = {}
    total = len(algorithms) * len(personas) * n_seeds
    done = 0

    for aname, (acls, akw) in algorithms.items():
        for pname, pcls in personas.items():
            seeds_traces = []
            for seed in range(n_seeds):
                trace = instrumented_run(acls, akw, pcls, seed)
                seeds_traces.append(trace)
                done += 1
                if done % 10 == 0:
                    print(f"  {done}/{total}", flush=True)

            # Average across seeds
            n_gens = len(seeds_traces[0]["qd"])
            avg_trace = {}
            for key in ["qd", "diversity", "occupied_bins", "mutation_rate",
                         "eviction_pool_size", "n_constraints"]:
                vals = np.array([t[key] for t in seeds_traces])
                avg_trace[key] = vals.mean(axis=0).tolist()
                avg_trace[f"{key}_std"] = vals.std(axis=0).tolist()

            # Constraint changes: fraction of seeds that saw a change at each gen
            cc = np.array([t["constraint_changes"] for t in seeds_traces])
            avg_trace["constraint_change_frac"] = cc.mean(axis=0).tolist()

            all_traces[f"{aname}|{pname}"] = avg_trace

    return all_traces


def generate_figures(traces, outdir):
    """Generate publication figures from traces."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib not available")
        return

    os.makedirs(outdir, exist_ok=True)

    personas = ["Exploratory", "Cycle", "Adaptive"]
    n_gens = 100

    # Color scheme
    colors = {
        "UH-nobandit": "#2ecc71",
        "AdaptRate-noreset": "#3498db",
        "Ultimate-Hybrid": "#9b59b6",
        "Evict-Restart": "#e67e22",
        "Bandit (UCB1)": "#e74c3c",
        "Epsilon-Bandit": "#f39c12",
        "EGGROLL": "#1abc9c",
        "Baseline": "#95a5a6",
    }

    # === Figure 1: Rate trajectory + QD recovery (the money figure) ===
    for pname in personas:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                         gridspec_kw={"height_ratios": [1, 2]})

        # Top: mutation rate
        for aname in ["UH-nobandit", "AdaptRate-noreset", "Baseline"]:
            key = f"{aname}|{pname}"
            if key not in traces:
                continue
            t = traces[key]
            x = range(len(t["mutation_rate"]))
            ax1.plot(x, t["mutation_rate"], label=aname, color=colors.get(aname, "gray"), linewidth=2)
            ax1.fill_between(x,
                             np.array(t["mutation_rate"]) - np.array(t["mutation_rate_std"]),
                             np.array(t["mutation_rate"]) + np.array(t["mutation_rate_std"]),
                             alpha=0.15, color=colors.get(aname, "gray"))

        # Shade constraint change intervals
        cc = traces.get(f"UH-nobandit|{pname}", traces.get(f"Baseline|{pname}", {}))
        if "constraint_change_frac" in cc:
            for i, frac in enumerate(cc["constraint_change_frac"]):
                if frac > 0.3:
                    ax1.axvspan(i - 0.5, i + 0.5, alpha=0.15, color="red", zorder=0)
                    ax2.axvspan(i - 0.5, i + 0.5, alpha=0.15, color="red", zorder=0)

        ax1.set_ylabel("Mutation Rate")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.set_title(f"Rate Self-Correction & QD Recovery ({pname} Persona)")

        # Bottom: QD score
        for aname in ["UH-nobandit", "AdaptRate-noreset", "Evict-Restart",
                       "Bandit (UCB1)", "Baseline"]:
            key = f"{aname}|{pname}"
            if key not in traces:
                continue
            t = traces[key]
            x = range(len(t["qd"]))
            ax2.plot(x, t["qd"], label=aname, color=colors.get(aname, "gray"), linewidth=2)
            ax2.fill_between(x,
                             np.array(t["qd"]) - np.array(t["qd_std"]),
                             np.array(t["qd"]) + np.array(t["qd_std"]),
                             alpha=0.1, color=colors.get(aname, "gray"))

        ax2.set_xlabel("Generation")
        ax2.set_ylabel("QD Score")
        ax2.legend(loc="upper left", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"rate_recovery_{pname.lower()}.png"), dpi=200)
        plt.close()
        print(f"  Saved rate_recovery_{pname.lower()}.png")

    # === Figure 2: Eviction pool dynamics ===
    for pname in personas:
        fig, ax = plt.subplots(figsize=(12, 5))
        for aname in ["UH-nobandit", "Evict-Restart", "Ultimate-Hybrid"]:
            key = f"{aname}|{pname}"
            if key not in traces:
                continue
            t = traces[key]
            if max(t["eviction_pool_size"]) == 0:
                continue
            x = range(len(t["eviction_pool_size"]))
            ax.plot(x, t["eviction_pool_size"], label=aname,
                    color=colors.get(aname, "gray"), linewidth=2)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Eviction Pool Size")
        ax.set_title(f"Eviction Pool Dynamics ({pname} Persona)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"eviction_{pname.lower()}.png"), dpi=200)
        plt.close()
        print(f"  Saved eviction_{pname.lower()}.png")

    # === Figure 3: Bandit expert selection over time ===
    for pname in personas:
        key_uh = f"Ultimate-Hybrid|{pname}"
        key_ucb = f"Bandit (UCB1)|{pname}"
        key_eps = f"Epsilon-Bandit|{pname}"

        for aname, key in [("Ultimate-Hybrid", key_uh),
                            ("Bandit-UCB1", key_ucb),
                            ("Epsilon-Bandit", key_eps)]:
            if key not in traces:
                continue
            t = traces[key]
            dists = t.get("expert_dist", [])
            # Only if we have bandit data
            valid = [d for d in dists if d and len(d) >= 4]
            if not valid:
                continue

        # Skip bandit figure if no data (UH-nobandit has no bandit)

    # === Figure 4: Grand comparison bar chart ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    algo_order = ["UH-nobandit", "AdaptRate-noreset", "Evict-Restart",
                  "Epsilon-Bandit", "Bandit (UCB1)", "EGGROLL", "Baseline"]

    for idx, pname in enumerate(personas):
        ax = axes[idx]
        means = []
        stds = []
        names = []
        for aname in algo_order:
            key = f"{aname}|{pname}"
            if key not in traces:
                continue
            t = traces[key]
            final_qd = t["qd"][-1] if t["qd"] else 0
            final_std = t["qd_std"][-1] if t["qd_std"] else 0
            means.append(final_qd)
            stds.append(final_std)
            names.append(aname)

        y_pos = range(len(names))
        barcolors = [colors.get(n, "gray") for n in names]
        ax.barh(y_pos, means, xerr=stds, color=barcolors, alpha=0.85, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names if idx == 0 else [], fontsize=9)
        ax.set_xlabel("Final QD Score")
        ax.set_title(pname)
        ax.invert_yaxis()

    plt.suptitle("Final QD Score by Algorithm and Persona (TTP, 10 seeds)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_bars.png"), dpi=200)
    plt.close()
    print("  Saved comparison_bars.png")


def main():
    n_seeds = 10
    print(f"Instrumented runs: 8 algorithms x 3 personas x {n_seeds} seeds = {8*3*n_seeds} runs")

    traces = run_all(n_seeds=n_seeds)

    outdir = os.path.join(SCRIPT_DIR, "results", "traces")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "traces.json"), "w") as f:
        json.dump(traces, f)
    print(f"Traces saved to {outdir}/traces.json")

    figdir = os.path.join(SCRIPT_DIR, "results", "figures")
    generate_figures(traces, figdir)


if __name__ == "__main__":
    main()
