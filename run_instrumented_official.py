#!/usr/bin/env python3
"""
Instrumented runs at OFFICIAL parameters for publication figures.

300 gens, pop 200, memory 500, interval 50.
Captures per-generation: mutation rate, QD score, eviction pool size, constraint changes.
5 seeds x 5 algorithms x 4 personas = 100 sequential runs.
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
from Personas.Strict import StrictUser
from Algorithms.VCMapElites import VariableConstraintMapElites

from mvp6_evict_restart import EvictRestartElites
from mvp13_adaptive_rate import AdaptiveRateElites
from mvp22_ultimate_hybrid import UltimateHybridElites

N_GENS = 300
POP_SIZE = 200
MEMORY = 500
XO_RATE = 0.5
MUT_RATE = 0.1
INTERVAL = 50


def instrumented_run(algo_class, kwargs, persona_class, seed):
    random.seed(seed)
    np.random.seed(seed)

    ps = TTPProblemSpace()
    user = persona_class(ps)
    algo = algo_class(ps, N_GENS, POP_SIZE, MEMORY, XO_RATE, MUT_RATE, user, INTERVAL, **kwargs)

    trace = {
        "qd": [], "diversity": [], "occupied_bins": [],
        "mutation_rate": [], "eviction_pool_size": [],
        "constraint_changes": [], "n_constraints": [],
    }

    original_run_one = algo.run_one_generation

    def patched_run_one(cons_changed):
        result = original_run_one(cons_changed)
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
                fitnesses.append(b[0][0])

        trace["qd"].append(float(sum(fitnesses)) if fitnesses else 0.0)
        trace["diversity"].append(n_occupied / len(pop) if pop else 0.0)
        trace["occupied_bins"].append(n_occupied)
        trace["constraint_changes"].append(bool(cons_changed))
        trace["n_constraints"].append(len(algo.variable_constraints))

        rate = getattr(algo, "current_rate", algo.mutation_rate)
        trace["mutation_rate"].append(float(rate))

        evpool = getattr(algo, "eviction_pool", [])
        trace["eviction_pool_size"].append(len(evpool))

        return result

    algo.run_one_generation = patched_run_one
    algo.run()
    return trace


def main():
    algorithms = {
        "UH-nobandit": (UltimateHybridElites, {"use_bandit": False}),
        "UH-noevict": (UltimateHybridElites, {"use_bandit": False, "use_eviction": False}),
        "AdaptRate-noreset": (AdaptiveRateElites, {"reset_on_change": False}),
        "Evict-Restart": (EvictRestartElites, {}),
        "Baseline": (VariableConstraintMapElites, {}),
    }
    personas = {
        "Exploratory": ExploratoryUser,
        "Cycle": TwoForOneBackUser,
        "Adaptive": AdaptiveUser,
        "Strict": StrictUser,
    }

    n_seeds = 5
    total = len(algorithms) * len(personas) * n_seeds
    print(f"Instrumented runs (official params): {total} runs")
    print(f"  gens={N_GENS} pop={POP_SIZE} mem={MEMORY} interval={INTERVAL}")

    all_traces = {}
    done = 0

    for aname, (acls, akw) in algorithms.items():
        for pname, pcls in personas.items():
            seeds_traces = []
            for seed in range(n_seeds):
                trace = instrumented_run(acls, akw, pcls, seed)
                seeds_traces.append(trace)
                done += 1
                print(f"  {done}/{total} ({aname} / {pname} / seed {seed})", flush=True)

            n_gens = len(seeds_traces[0]["qd"])
            avg_trace = {}
            for key in ["qd", "diversity", "occupied_bins", "mutation_rate",
                         "eviction_pool_size", "n_constraints"]:
                vals = np.array([t[key] for t in seeds_traces])
                avg_trace[key] = vals.mean(axis=0).tolist()
                avg_trace[f"{key}_std"] = vals.std(axis=0).tolist()

            cc = np.array([t["constraint_changes"] for t in seeds_traces])
            avg_trace["constraint_change_frac"] = cc.mean(axis=0).tolist()

            all_traces[f"{aname}|{pname}"] = avg_trace

    outdir = os.path.join(SCRIPT_DIR, "results", "traces_official")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "traces.json"), "w") as f:
        json.dump(all_traces, f)
    print(f"Saved to {outdir}/traces.json")

    # Generate figures
    generate_figures(all_traces, os.path.join(SCRIPT_DIR, "results", "figures_official"))


def generate_figures(traces, figdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    os.makedirs(figdir, exist_ok=True)

    personas = ["Exploratory", "Cycle", "Adaptive", "Strict"]
    colors = {
        "UH-nobandit": "#2ecc71",
        "UH-noevict": "#9b59b6",
        "AdaptRate-noreset": "#3498db",
        "Evict-Restart": "#e67e22",
        "Baseline": "#95a5a6",
    }

    for pname in personas:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                                              gridspec_kw={"height_ratios": [1, 2, 1]})

        # Top: mutation rate
        for aname in ["UH-nobandit", "UH-noevict", "AdaptRate-noreset", "Baseline"]:
            key = f"{aname}|{pname}"
            if key not in traces:
                continue
            t = traces[key]
            x = range(len(t["mutation_rate"]))
            ax1.plot(x, t["mutation_rate"], label=aname, color=colors[aname], linewidth=2)
            ax1.fill_between(x,
                             np.array(t["mutation_rate"]) - np.array(t["mutation_rate_std"]),
                             np.array(t["mutation_rate"]) + np.array(t["mutation_rate_std"]),
                             alpha=0.15, color=colors[aname])

        # Shade constraint changes
        ref = traces.get(f"UH-nobandit|{pname}", traces.get(f"Baseline|{pname}", {}))
        if "constraint_change_frac" in ref:
            for i, frac in enumerate(ref["constraint_change_frac"]):
                if frac > 0.3:
                    for ax in [ax1, ax2, ax3]:
                        ax.axvspan(i - 0.5, i + 0.5, alpha=0.12, color="red", zorder=0)

        ax1.set_ylabel("Mutation Rate")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.set_title(f"Self-Regulating QD: Rate, Recovery & Eviction ({pname}, Official Params)")

        # Middle: QD score
        for aname in ["UH-nobandit", "UH-noevict", "AdaptRate-noreset", "Evict-Restart", "Baseline"]:
            key = f"{aname}|{pname}"
            if key not in traces:
                continue
            t = traces[key]
            x = range(len(t["qd"]))
            ax2.plot(x, t["qd"], label=aname, color=colors[aname], linewidth=2)
            ax2.fill_between(x,
                             np.array(t["qd"]) - np.array(t["qd_std"]),
                             np.array(t["qd"]) + np.array(t["qd_std"]),
                             alpha=0.1, color=colors[aname])

        ax2.set_ylabel("QD Score")
        ax2.legend(loc="upper left", fontsize=9)

        # Bottom: eviction pool
        for aname in ["UH-nobandit", "UH-noevict", "Evict-Restart"]:
            key = f"{aname}|{pname}"
            if key not in traces:
                continue
            t = traces[key]
            if max(t["eviction_pool_size"]) == 0:
                continue
            x = range(len(t["eviction_pool_size"]))
            ax3.plot(x, t["eviction_pool_size"], label=aname, color=colors[aname], linewidth=2)

        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Eviction Pool")
        ax3.legend(loc="upper left", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(figdir, f"official_{pname.lower()}.png"), dpi=200)
        plt.close()
        print(f"  Saved official_{pname.lower()}.png")


if __name__ == "__main__":
    main()
