#!/usr/bin/env python3
"""
Statistical ablation of the two winning algorithms.

20 runs per condition, 3 personas (skip Strict, always zero).
Mann-Whitney U, Welch's t, Cohen's d.

Conditions:
1. EGGROLL-Elites (directed mutation from success history)
2. Evict-Restart (eviction pool + stagnation restart)
3. EGGROLL + Evict-Restart combined
4. Ablations:
   a. EGGROLL without direction learning (random mutation only)
   b. Evict-Restart without eviction pool (restart only)
   c. Evict-Restart without stagnation restart (eviction only)
5. Baseline: VC-MAP-Elites
"""

import sys
import os
import json
import copy
import random

import numpy as np
from scipy import stats

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

from mvp4_eggroll_lowrank import EGGROLLElites
from mvp6_evict_restart import EvictRestartElites


# ── Combined: EGGROLL + Evict-Restart ──────────────────────────────

import numpy.random as npr


def roulette_selection(population):
    small = min(c[0] for c in population)
    add = -small if small < 0 else 0
    total = sum(c[0] + add for c in population)
    if total == 0:
        probs = [1.0 / len(population)] * len(population)
    else:
        probs = [(c[0] + add) / total for c in population]
    return population[npr.choice(len(population), p=probs)]


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "framework"))
from GeneticAlgorithmInterface import VariableConstraintGA
from ProblemSpaceInterface import ProblemSpace


class CombinedElites(VariableConstraintGA):
    """EGGROLL directed mutation + Evict-Restart eviction pool + stagnation restart."""

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3,
                 stagnation_patience=3, restart_fraction=0.5,
                 n_directions=5, direction_weight=0.7):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
        self.stagnation_patience = stagnation_patience
        self.restart_fraction = restart_fraction
        self.n_directions = n_directions
        self.direction_weight = direction_weight
        super().__init__(problem_space, number_generations, population_size,
                         max_memory, cross_over_rate, mutation_rate, user,
                         update_interval)

    def set_up(self):
        n_bins = self.problem_space.get_num_bins()
        self.infeasible_pop_size = int(self.max_memory * self.infeasible_rate)
        self.elitism_num = int(self.infeasible_pop_size * self.elitism)
        feasible_memory = self.max_memory - self.infeasible_pop_size
        self.inds_per_bin = max(1, feasible_memory // n_bins)
        self.bins = [[] for _ in range(n_bins)]
        self.infeasible_pop = []
        self.eviction_pool = []
        self.eviction_max = max(n_bins, self.max_memory // 4)
        self.prev_occupied = 0
        self.stagnation_counter = 0
        self.restart_budget = int(self.population_size * self.restart_fraction)
        self.successful_parents = []

        for _ in range(self.population_size):
            ind = self.problem_space.generate_random_individual()
            self._place(ind, self.infeasible_pop)

    def _const_ok(self, ind):
        return all(c.apply(ind) for c in self.problem_space.get_constant_constraints())

    def _var_ok(self, ind):
        return all(c.apply(ind) for c in self.variable_constraints)

    def _all_ok(self, ind):
        return self._const_ok(ind) and self._var_ok(ind)

    def _place(self, ind, infeasible_pop):
        if self._const_ok(ind):
            b = self.problem_space.place_in_bin(ind)
            fit = self.problem_space.fitness(ind)
            inserted = False
            if len(self.bins[b]) < self.inds_per_bin:
                self.bins[b].append((fit, ind))
                self.bins[b].sort(key=lambda x: x[0], reverse=True)
                inserted = True
            elif fit >= self.bins[b][-1][0]:
                self.bins[b].pop(-1)
                self.bins[b].append((fit, ind))
                self.bins[b].sort(key=lambda x: x[0], reverse=True)
                inserted = True
            return inserted
        else:
            n_sat = sum(1 for c in self.problem_space.get_constant_constraints() if c.apply(ind))
            if len(infeasible_pop) < self.infeasible_pop_size:
                infeasible_pop.append((n_sat, ind))
            return False

    def _select(self):
        total_f = sum(len(b) for b in self.bins)
        total_i = len(self.infeasible_pop)
        if total_f == 0 and total_i == 0:
            return self.problem_space.generate_random_individual()
        if total_f > 0 and (total_i == 0 or random.random() < (total_f * 2) / (total_f + total_i)):
            occupied = [i for i in range(len(self.bins)) if self.bins[i]]
            if not occupied:
                return self.problem_space.generate_random_individual()
            return roulette_selection(self.bins[random.choice(occupied)])[1]
        return roulette_selection(self.infeasible_pop)[1]

    def _directed_mutate(self, parent):
        if self.successful_parents and random.random() < self.direction_weight:
            donor = random.choice(self.successful_parents)
            child, _ = self.problem_space.cross_over(parent, donor)
            child = self.problem_space.mutate(child, self.mutation_rate * 0.3)
            return child
        return self.problem_space.mutate(parent, self.mutation_rate)

    def _restart(self, reason="constraint"):
        if reason == "constraint":
            for i in range(len(self.bins)):
                surviving = []
                for fit, ind in self.bins[i]:
                    if self._var_ok(ind):
                        surviving.append((fit, ind))
                    elif len(self.eviction_pool) < self.eviction_max:
                        self.eviction_pool.append((fit, ind, i))
                self.bins[i] = surviving

            still_evicted = []
            for fit, ind, old_bin in self.eviction_pool:
                if self._var_ok(ind) and self._const_ok(ind):
                    b = self.problem_space.place_in_bin(ind)
                    nf = self.problem_space.fitness(ind)
                    if len(self.bins[b]) < self.inds_per_bin:
                        self.bins[b].append((nf, ind))
                        self.bins[b].sort(key=lambda x: x[0], reverse=True)
                    elif nf >= self.bins[b][-1][0]:
                        self.bins[b].pop(-1)
                        self.bins[b].append((nf, ind))
                        self.bins[b].sort(key=lambda x: x[0], reverse=True)
                    else:
                        still_evicted.append((fit, ind, old_bin))
                else:
                    still_evicted.append((fit, ind, old_bin))
            self.eviction_pool = still_evicted
            self.successful_parents = []

        all_elites = [(f, i) for b in self.bins for f, i in b]
        seed_pool = all_elites if all_elites else None
        n_elite = int(self.restart_budget * 0.6)
        n_rand = self.restart_budget - n_elite
        new_inf = self.infeasible_pop[:self.elitism_num]

        if seed_pool:
            for _ in range(n_elite):
                parent = roulette_selection(seed_pool)[1]
                child = self.problem_space.mutate(parent, self.mutation_rate)
                self._place(child, new_inf)
        else:
            n_rand += n_elite

        for _ in range(n_rand):
            self._place(self.problem_space.generate_random_individual(), new_inf)
        self.infeasible_pop = new_inf

    def _check_stagnation(self):
        cur = sum(1 for b in self.bins if b)
        if cur <= self.prev_occupied:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        self.prev_occupied = cur
        return self.stagnation_counter >= self.stagnation_patience

    def run_one_generation(self, cons_changed):
        if cons_changed:
            self._restart("constraint")
        elif self._check_stagnation():
            self._restart("stagnation")

        self.infeasible_pop.sort(key=lambda x: x[0], reverse=True)
        new_inf = self.infeasible_pop[:self.elitism_num]

        for _ in range(self.population_size // 2):
            p1, p2 = self._select(), self._select()
            c1 = self._directed_mutate(p1)
            c2 = self._directed_mutate(p2)

            ins1 = self._place(c1, new_inf)
            ins2 = self._place(c2, new_inf)

            if ins1:
                self.successful_parents.append(copy.deepcopy(c1))
                if len(self.successful_parents) > self.n_directions:
                    self.successful_parents.pop(0)
            if ins2:
                self.successful_parents.append(copy.deepcopy(c2))
                if len(self.successful_parents) > self.n_directions:
                    self.successful_parents.pop(0)

        self.infeasible_pop = new_inf
        return [[(f, i) for f, i in bl if self._all_ok(i)] for bl in self.bins]


# ── Test harness ───────────────────────────────────────────────────

def run_one(algo_class, persona_class, n_gens=100, pop_size=50,
            max_memory=200, **kwargs):
    ps = TTPProblemSpace()
    user = persona_class(ps)
    algo = algo_class(ps, n_gens, pop_size, max_memory, 0.7, 0.3, user, 10, **kwargs)
    try:
        algo.run()
        m = algo.measure_history
        return m.qd_score[-1] if m.qd_score else 0.0
    except Exception:
        return 0.0


def run_n(algo_class, persona_class, n_runs, **kwargs):
    return [run_one(algo_class, persona_class, **kwargs) for _ in range(n_runs)]


def compare(name_a, qds_a, name_b, qds_b):
    mean_a, mean_b = np.mean(qds_a), np.mean(qds_b)
    std_a, std_b = np.std(qds_a, ddof=1), np.std(qds_b, ddof=1)
    U, p_u = stats.mannwhitneyu(qds_a, qds_b, alternative="two-sided")
    pooled = np.sqrt((std_a**2 + std_b**2) / 2)
    d = (mean_a - mean_b) / pooled if pooled > 0 else 0
    return mean_a, std_a, mean_b, std_b, U, p_u, d


def main():
    n_runs = 20

    personas = [
        ("Exploratory", ExploratoryUser),
        ("Cycle", TwoForOneBackUser),
        ("Adaptive", AdaptiveUser),
    ]

    conditions = [
        ("Combined", CombinedElites, {}),
        ("EGGROLL", EGGROLLElites, {}),
        ("Evict-Restart", EvictRestartElites, {}),
        ("EGGROLL-nodir", EGGROLLElites, {"direction_weight": 0.0}),
        ("EvRst-noevict", EvictRestartElites, {"stagnation_patience": 3}),
        ("EvRst-nostag", EvictRestartElites, {"stagnation_patience": 9999}),
        ("Baseline", VariableConstraintMapElites, {}),
    ]

    print(f"Ablation Study | TTP | {n_runs} runs per condition")
    print("=" * 90)

    all_data = {}

    for pname, pcls in personas:
        print(f"\n--- {pname} ---")
        for cname, acls, kwargs in conditions:
            print(f"  Running {cname}...", end="", flush=True)
            qds = run_n(acls, pcls, n_runs, **kwargs)
            mean, std = np.mean(qds), np.std(qds, ddof=1)
            all_data[(cname, pname)] = qds
            print(f" mean={mean:>10.1f} +/- {std:.1f}")

    # Statistical comparisons
    print("\n" + "=" * 90)
    print("STATISTICAL COMPARISONS")
    print("=" * 90)

    ref_name = "Combined"
    for pname, _ in personas:
        print(f"\n--- {pname}: {ref_name} vs each ---")
        print(f"{'Comparison':<30} {'Mean A':>10} {'Mean B':>10} {'p (MW)':>10} {'Cohen d':>10} {'Sig':>5}")
        print("-" * 80)
        ref = all_data[(ref_name, pname)]
        for cname, _, _ in conditions:
            if cname == ref_name:
                continue
            other = all_data[(cname, pname)]
            ma, sa, mb, sb, U, p, d = compare(ref_name, ref, cname, other)
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            print(f"{ref_name} vs {cname:<20} {ma:>10.1f} {mb:>10.1f} {p:>10.4f} {d:>+10.3f} {sig:>5}")

    # Aggregate
    print("\n" + "=" * 90)
    print("AGGREGATE (sum of persona means)")
    print("-" * 50)
    for cname, _, _ in conditions:
        total = sum(np.mean(all_data[(cname, pn)]) for pn, _ in personas)
        print(f"  {cname:<25} {total:>12.1f}")

    with open("ablation_winners.json", "w") as f:
        json.dump({f"{c}|{p}": list(v) for (c, p), v in all_data.items()}, f, indent=2)
    print(f"\nSaved to ablation_winners.json")


if __name__ == "__main__":
    main()
