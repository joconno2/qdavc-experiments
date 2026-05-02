#!/usr/bin/env python3
"""
Analyze QDA-VC ablation results. Produces:
  1. Ranked summary table (grand totals)
  2. Per-domain/persona breakdown
  3. Statistical tests vs baseline (Mann-Whitney U + Cohen's d)
  4. Ablation effect sizes (full vs ablated)
  5. Mechanism contribution matrix (what helps where)
  6. CSV export for further analysis

Usage:
  python3 analyze_results.py results/full_ablation_v2.json
  python3 analyze_results.py results/full_ablation_v2.json --csv results/tables.csv
  python3 analyze_results.py results/full_ablation_v2.json --figures results/figures/
"""

import sys
import os
import json
import argparse

import numpy as np
from scipy import stats


def load_data(path):
    with open(path) as f:
        return json.load(f)


def extract_structure(data):
    """Extract unique domains, algorithms, personas from keys."""
    domains, algos, personas = set(), set(), set()
    for key in data:
        parts = key.split("|")
        if len(parts) == 3:
            domains.add(parts[0])
            algos.add(parts[1])
            personas.add(parts[2])
    return sorted(domains), sorted(algos), sorted(personas, key=lambda p: {"Exploratory": 0, "Cycle": 1, "Adaptive": 2}.get(p, 3))


def ranked_table(data, domains, algos, personas):
    """Grand total ranking across all domains and personas."""
    print("\n" + "=" * 90)
    print("GRAND TOTAL RANKING")
    print("=" * 90)

    totals = []
    for a in algos:
        total = 0
        for d in domains:
            for p in personas:
                qds = data.get(f"{d}|{a}|{p}", [0])
                total += np.mean(qds)
        totals.append((total, a))
    totals.sort(reverse=True)

    base_total = next((t for t, n in totals if n == "Baseline"), 0)

    print(f"  {'Rank':<5} {'Algorithm':<24} {'Total QD':>12} {'vs Base':>10} {'':>8}")
    print(f"  {'-'*5} {'-'*24} {'-'*12} {'-'*10} {'-'*8}")
    for rank, (total, aname) in enumerate(totals, 1):
        pct = ((total - base_total) / abs(base_total) * 100) if base_total != 0 else 0
        bar = "+" * max(0, min(40, int(pct / 5))) if pct > 0 else "-" * max(0, min(40, int(-pct / 5)))
        print(f"  {rank:<5} {aname:<24} {total:>12.1f} {pct:>+9.1f}% {bar}")

    return totals


def domain_persona_table(data, domains, algos, personas):
    """Detailed per-domain per-persona means."""
    print("\n" + "=" * 90)
    print("PER-DOMAIN BREAKDOWN (mean QD, 30 seeds)")
    print("=" * 90)

    for d in domains:
        print(f"\n  {d}:")
        print(f"  {'Algorithm':<24} " + " ".join(f"{p:>14}" for p in personas) + f" {'Total':>14}")
        print(f"  {'-'*24} " + " ".join(f"{'-'*14}" for _ in personas) + f" {'-'*14}")

        rows = []
        for a in algos:
            vals = []
            for p in personas:
                qds = data.get(f"{d}|{a}|{p}", [0])
                vals.append(np.mean(qds))
            rows.append((sum(vals), a, vals))
        rows.sort(reverse=True)

        for total, aname, vals in rows:
            print(f"  {aname:<24} " +
                  " ".join(f"{v:>14.1f}" for v in vals) +
                  f" {total:>14.1f}")


def stat_tests(data, domains, algos, personas):
    """Mann-Whitney U tests vs baseline, organized by significance."""
    print("\n" + "=" * 90)
    print("STATISTICAL SIGNIFICANCE vs BASELINE")
    print("=" * 90)

    significant = []
    nonsignificant = []

    for d in domains:
        for a in algos:
            if a == "Baseline":
                continue
            for p in personas:
                a_qds = np.array(data.get(f"{d}|{a}|{p}", [0]))
                b_qds = np.array(data.get(f"{d}|Baseline|{p}", [0]))
                ma, mb = np.mean(a_qds), np.mean(b_qds)
                sa, sb = np.std(a_qds, ddof=1), np.std(b_qds, ddof=1)
                try:
                    U, pval = stats.mannwhitneyu(a_qds, b_qds, alternative="two-sided")
                except ValueError:
                    pval = 1.0
                pooled = np.sqrt((sa**2 + sb**2) / 2)
                d_val = (ma - mb) / pooled if pooled > 0 else 0

                row = (d, a, p, ma, mb, ma - mb, pval, d_val)
                if pval < 0.05:
                    significant.append(row)
                else:
                    nonsignificant.append(row)

    # Sort significant by effect size
    significant.sort(key=lambda x: -x[7])

    print(f"\n  SIGNIFICANT (p < 0.05): {len(significant)} comparisons")
    print(f"  {'Domain':<14} {'Algorithm':<24} {'Persona':<12} {'Diff':>8} {'p':>10} {'d':>8}")
    print(f"  {'-'*14} {'-'*24} {'-'*12} {'-'*8} {'-'*10} {'-'*8}")
    for d, a, p, ma, mb, diff, pval, d_val in significant:
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else "*")
        print(f"  {d:<14} {a:<24} {p:<12} {diff:>+8.0f} {pval:>10.6f} {d_val:>+8.3f} {sig}")

    print(f"\n  NOT SIGNIFICANT: {len(nonsignificant)} comparisons (omitted)")


def ablation_analysis(data, domains, personas):
    """Analyze ablation pairs to quantify mechanism contributions."""
    print("\n" + "=" * 90)
    print("MECHANISM ABLATION ANALYSIS")
    print("  (positive diff = mechanism helps; * = p < 0.05)")
    print("=" * 90)

    ablations = [
        ("EGGROLL", "EGGROLL-nodir", "Direction learning (EGGROLL)"),
        ("Evict-Restart", "EvRst-noevict", "Eviction pool"),
        ("Evict-Restart", "EvRst-nostag", "Stagnation restart"),
        ("Bandit+Evict", "BdEvict-noevict", "Eviction pool (in Bandit+Evict)"),
        ("Bandit+Evict", "BdEvict-nostag", "Stagnation restart (in Bandit+Evict)"),
        ("Bandit+Evict", "BdEvict-nobandit", "Bandit selection (in Bandit+Evict)"),
        ("Constraint-Memory", "Memory-nomemory", "Constraint memory recall"),
        ("Novelty-Selection", "Novelty-none", "Novelty bias"),
        ("Adaptive-Rate", "AdaptRate-noreset", "Rate reset on change"),
        ("Sliding-Window", "SlidingK1", "Window K=3 vs K=1"),
        ("Sliding-Window", "SlidingK5", "Window K=3 vs K=5"),
        ("Age-Weighted", "Age-nopen", "Age penalty"),
    ]

    for full_name, ablated_name, mechanism in ablations:
        print(f"\n  {mechanism}:")
        for d in domains:
            for p in personas:
                f_key = f"{d}|{full_name}|{p}"
                a_key = f"{d}|{ablated_name}|{p}"
                f_qds = np.array(data.get(f_key, [0]))
                a_qds = np.array(data.get(a_key, [0]))
                if len(f_qds) <= 1 or len(a_qds) <= 1:
                    continue
                mf, ma = np.mean(f_qds), np.mean(a_qds)
                try:
                    U, pval = stats.mannwhitneyu(f_qds, a_qds, alternative="two-sided")
                except ValueError:
                    pval = 1.0
                diff = mf - ma
                sig = " *" if pval < 0.05 else ""
                print(f"    {d:<14} {p:<12}: full={mf:>8.0f}  ablated={ma:>8.0f}  "
                      f"diff={diff:>+8.0f}  p={pval:.4f}{sig}")


def mechanism_matrix(data, domains, personas):
    """Which mechanisms help on which persona? Summary matrix."""
    print("\n" + "=" * 90)
    print("MECHANISM x PERSONA CONTRIBUTION MATRIX")
    print("  (shows mean QD improvement over baseline, * = significant)")
    print("=" * 90)

    # Group algorithms by primary mechanism
    mechanism_groups = {
        "Directed mutation": ["EGGROLL"],
        "Eviction pool": ["Evict-Restart"],
        "Bandit selection": ["Bandit"],
        "Bandit+Evict": ["Bandit+Evict"],
        "Constraint memory": ["Constraint-Memory"],
        "DE mutation": ["DE-Elites"],
        "Novelty selection": ["Novelty-Selection"],
        "Adaptive rate": ["Adaptive-Rate"],
        "Sliding window": ["Sliding-Window"],
        "Thompson bandit": ["Thompson-Bandit"],
        "Crossover primary": ["Crossover-Primary"],
        "Age-weighted": ["Age-Weighted"],
        "Epsilon bandit": ["Epsilon-Bandit"],
    }

    print(f"\n  {'Mechanism':<22} " + " ".join(f"{d}:{p[:4]:>10}" for d in domains for p in personas))
    print(f"  {'-'*22} " + " ".join(f"{'-'*14}" for _ in domains for _ in personas))

    for mech_name, algo_names in mechanism_groups.items():
        cells = []
        for d in domains:
            for p in personas:
                b_qds = np.array(data.get(f"{d}|Baseline|{p}", [0]))
                mb = np.mean(b_qds)
                best_diff = 0
                best_sig = ""
                for a in algo_names:
                    a_qds = np.array(data.get(f"{d}|{a}|{p}", [0]))
                    ma = np.mean(a_qds)
                    try:
                        _, pval = stats.mannwhitneyu(a_qds, b_qds, alternative="two-sided")
                    except ValueError:
                        pval = 1.0
                    diff = ma - mb
                    if abs(diff) > abs(best_diff):
                        best_diff = diff
                        best_sig = "*" if pval < 0.05 else ""
                cells.append(f"{best_diff:>+10.0f}{best_sig:>1}")
        print(f"  {mech_name:<22} " + " ".join(f"{c:>14}" for c in cells))


def export_csv(data, domains, algos, personas, path):
    """Export full results matrix as CSV."""
    with open(path, "w") as f:
        f.write("domain,algorithm,persona,mean_qd,std_qd,n,median_qd,min_qd,max_qd\n")
        for d in domains:
            for a in algos:
                for p in personas:
                    qds = np.array(data.get(f"{d}|{a}|{p}", [0]))
                    f.write(f"{d},{a},{p},{np.mean(qds):.2f},{np.std(qds, ddof=1):.2f},"
                            f"{len(qds)},{np.median(qds):.2f},{np.min(qds):.2f},{np.max(qds):.2f}\n")
    print(f"\nCSV exported to {path}")


def try_figures(data, domains, algos, personas, fig_dir):
    """Generate matplotlib figures if available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figures")
        return

    os.makedirs(fig_dir, exist_ok=True)

    # 1. Grand total bar chart
    totals = []
    for a in algos:
        total = sum(np.mean(data.get(f"{d}|{a}|{p}", [0]))
                    for d in domains for p in personas)
        totals.append((total, a))
    totals.sort(reverse=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    names = [t[1] for t in totals]
    vals = [t[0] for t in totals]
    base_val = next(v for v, n in totals if n == "Baseline")
    colors = ["#2ecc71" if v > base_val * 1.1 else "#e74c3c" if v < base_val * 0.9 else "#95a5a6" for v in vals]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Total QD Score (sum of all domain/persona means)")
    ax.set_title("QDA-VC Algorithm Comparison")
    ax.axvline(base_val, color="black", linestyle="--", alpha=0.5, label="Baseline")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "grand_total.png"), dpi=150)
    plt.close()

    # 2. Per-persona heatmap (TTP only)
    ttp_algos = sorted(algos, key=lambda a: -sum(
        np.mean(data.get(f"TTP|{a}|{p}", [0])) for p in personas))

    fig, ax = plt.subplots(figsize=(10, max(8, len(ttp_algos) * 0.4)))
    matrix = []
    for a in ttp_algos:
        row = [np.mean(data.get(f"TTP|{a}|{p}", [0])) for p in personas]
        matrix.append(row)
    matrix = np.array(matrix)

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(personas)))
    ax.set_xticklabels(personas)
    ax.set_yticks(range(len(ttp_algos)))
    ax.set_yticklabels(ttp_algos, fontsize=9)
    ax.set_title("TTP QD Score by Algorithm x Persona")

    for i in range(len(ttp_algos)):
        for j in range(len(personas)):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ttp_heatmap.png"), dpi=150)
    plt.close()

    print(f"\nFigures saved to {fig_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Analyze QDA-VC ablation results")
    parser.add_argument("results_file", help="Path to results JSON")
    parser.add_argument("--csv", help="Export CSV to this path")
    parser.add_argument("--figures", help="Generate figures in this directory")
    args = parser.parse_args()

    data = load_data(args.results_file)
    domains, algos, personas = extract_structure(data)

    n_keys = len(data)
    n_samples = sum(len(v) for v in data.values())
    print(f"Loaded {n_keys} conditions, {n_samples} total samples")
    print(f"Domains: {domains}")
    print(f"Algorithms: {len(algos)}")
    print(f"Personas: {personas}")

    ranked_table(data, domains, algos, personas)
    domain_persona_table(data, domains, algos, personas)
    stat_tests(data, domains, algos, personas)
    ablation_analysis(data, domains, personas)
    mechanism_matrix(data, domains, personas)

    if args.csv:
        export_csv(data, domains, algos, personas, args.csv)

    if args.figures:
        try_figures(data, domains, algos, personas, args.figures)


if __name__ == "__main__":
    main()
