#!/usr/bin/env python3
"""
Comprehensive rerun: ALL algorithms against ALL baselines, ALL personas, official params.

This is the definitive run. Everything we've built, tested properly.

Algorithms (16):
  Framework baselines: VC-MAP-Elites, Filtering, RandomRestarts, Shuffling
  Wave 1 winners: EGGROLL, Evict-Restart, Bandit (UCB1), Epsilon-Bandit
  Wave 2 winners: UH-nobandit, UH-noevict, AdaptRate-noreset, BdEvict-nostag
  New hybrids: Shuffling+AdaptRate, Shuffling+Evict, Shuffling+Both, DE-Elites

Official params: 300 gens, pop 200, memory 500, interval 50, xo 0.5, mut 0.1
Domains: TTP, LogicPuzzles
Personas: Exploratory, Cycle, Adaptive, Strict
Seeds: 30

16 * 2 * 4 * 30 = 3,840 tasks
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


# === Shuffling hybrid algorithms (defined inline) ===

def roulette_selection(population):
    small = min(c[0] for c in population)
    add = -small if small < 0 else 0
    total = sum(c[0] + add for c in population)
    if total == 0:
        probs = [1.0 / len(population)] * len(population)
    else:
        probs = [(c[0] + add) / total for c in population]
    return population[npr.choice(len(population), p=probs)]


from GeneticAlgorithmInterface import VariableConstraintGA


class ShufflingAdaptRate(VariableConstraintGA):
    """Shuffling baseline + 1/5th adaptive rate (no reset)."""

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3,
                 target_success=0.2, adapt_factor=1.2,
                 rate_min=0.05, rate_max=0.9):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
        self.target_success = target_success
        self.adapt_factor = adapt_factor
        self.rate_min = rate_min
        self.rate_max = rate_max
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
        self.current_rate = self.mutation_rate

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
            if len(self.bins[b]) < self.inds_per_bin:
                self.bins[b].append((fit, ind))
                self.bins[b].sort(key=lambda x: x[0], reverse=True)
                return True
            elif fit >= self.bins[b][-1][0]:
                self.bins[b].pop(-1)
                self.bins[b].append((fit, ind))
                self.bins[b].sort(key=lambda x: x[0], reverse=True)
                return True
        else:
            n_sat = sum(1 for c in self.problem_space.get_constant_constraints()
                        if c.apply(ind))
            if len(infeasible_pop) < self.infeasible_pop_size:
                infeasible_pop.append((n_sat, ind))
        return False

    def _select(self):
        total_f = sum(len(b) for b in self.bins)
        total_i = len(self.infeasible_pop)
        if total_f == 0 and total_i == 0:
            return self.problem_space.generate_random_individual()
        if total_f > 0 and (total_i == 0 or
                random.random() < (total_f * 2) / (total_f + total_i)):
            occupied = [i for i in range(len(self.bins)) if self.bins[i]]
            if not occupied:
                return self.problem_space.generate_random_individual()
            return roulette_selection(self.bins[random.choice(occupied)])[1]
        return roulette_selection(self.infeasible_pop)[1]

    def _shuffle(self):
        """Shuffling: re-sort all solutions into bins based on current constraints."""
        all_solutions = []
        for b in self.bins:
            for fit, ind in b:
                all_solutions.append(ind)
        for n_sat, ind in self.infeasible_pop:
            all_solutions.append(ind)

        n_bins = self.problem_space.get_num_bins()
        self.bins = [[] for _ in range(n_bins)]
        self.infeasible_pop = []

        for ind in all_solutions:
            self._place(ind, self.infeasible_pop)

    def run_one_generation(self, cons_changed):
        if cons_changed:
            self._shuffle()

        self.infeasible_pop.sort(key=lambda x: x[0], reverse=True)
        new_inf = self.infeasible_pop[:self.elitism_num]

        successes = 0
        total = 0

        for _ in range(self.population_size // 2):
            p1, p2 = self._select(), self._select()
            if random.random() < self.cross_over_rate:
                c1, c2 = self.problem_space.cross_over(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
            c1 = self.problem_space.mutate(c1, self.current_rate)
            c2 = self.problem_space.mutate(c2, self.current_rate)
            s1 = self._place(c1, new_inf)
            s2 = self._place(c2, new_inf)
            successes += int(s1) + int(s2)
            total += 2

        if total > 0:
            sr = successes / total
            if sr > self.target_success:
                self.current_rate = min(self.rate_max, self.current_rate * self.adapt_factor)
            else:
                self.current_rate = max(self.rate_min, self.current_rate / self.adapt_factor)

        self.infeasible_pop = new_inf
        return [[(f, i) for f, i in bl if self._all_ok(i)] for bl in self.bins]


class ShufflingEvict(VariableConstraintGA):
    """Shuffling + eviction pool (no stagnation restart, no adaptive rate)."""

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
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
            if len(self.bins[b]) < self.inds_per_bin:
                self.bins[b].append((fit, ind))
                self.bins[b].sort(key=lambda x: x[0], reverse=True)
                return True
            elif fit >= self.bins[b][-1][0]:
                self.bins[b].pop(-1)
                self.bins[b].append((fit, ind))
                self.bins[b].sort(key=lambda x: x[0], reverse=True)
                return True
        else:
            n_sat = sum(1 for c in self.problem_space.get_constant_constraints()
                        if c.apply(ind))
            if len(infeasible_pop) < self.infeasible_pop_size:
                infeasible_pop.append((n_sat, ind))
        return False

    def _select(self):
        total_f = sum(len(b) for b in self.bins)
        total_i = len(self.infeasible_pop)
        if total_f == 0 and total_i == 0:
            return self.problem_space.generate_random_individual()
        if total_f > 0 and (total_i == 0 or
                random.random() < (total_f * 2) / (total_f + total_i)):
            occupied = [i for i in range(len(self.bins)) if self.bins[i]]
            if not occupied:
                return self.problem_space.generate_random_individual()
            return roulette_selection(self.bins[random.choice(occupied)])[1]
        return roulette_selection(self.infeasible_pop)[1]

    def _shuffle_and_evict(self):
        """Shuffle all solutions, plus eviction pool management."""
        all_solutions = []
        for b in self.bins:
            for fit, ind in b:
                if self._var_ok(ind):
                    all_solutions.append(ind)
                elif len(self.eviction_pool) < self.eviction_max:
                    self.eviction_pool.append((fit, ind))
        for n_sat, ind in self.infeasible_pop:
            all_solutions.append(ind)

        # Try reinserting evicted solutions
        still_evicted = []
        for fit, ind in self.eviction_pool:
            if self._var_ok(ind) and self._const_ok(ind):
                all_solutions.append(ind)
            else:
                still_evicted.append((fit, ind))
        self.eviction_pool = still_evicted

        n_bins = self.problem_space.get_num_bins()
        self.bins = [[] for _ in range(n_bins)]
        self.infeasible_pop = []
        for ind in all_solutions:
            self._place(ind, self.infeasible_pop)

    def run_one_generation(self, cons_changed):
        if cons_changed:
            self._shuffle_and_evict()

        self.infeasible_pop.sort(key=lambda x: x[0], reverse=True)
        new_inf = self.infeasible_pop[:self.elitism_num]

        for _ in range(self.population_size // 2):
            p1, p2 = self._select(), self._select()
            if random.random() < self.cross_over_rate:
                c1, c2 = self.problem_space.cross_over(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
            c1 = self.problem_space.mutate(c1, self.mutation_rate)
            c2 = self.problem_space.mutate(c2, self.mutation_rate)
            self._place(c1, new_inf)
            self._place(c2, new_inf)

        self.infeasible_pop = new_inf
        return [[(f, i) for f, i in bl if self._all_ok(i)] for bl in self.bins]


class ShufflingBoth(VariableConstraintGA):
    """Shuffling + eviction pool + adaptive rate. The full hybrid on shuffling base."""

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3,
                 target_success=0.2, adapt_factor=1.2,
                 rate_min=0.05, rate_max=0.9):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
        self.target_success = target_success
        self.adapt_factor = adapt_factor
        self.rate_min = rate_min
        self.rate_max = rate_max
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
        self.current_rate = self.mutation_rate

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
            if len(self.bins[b]) < self.inds_per_bin:
                self.bins[b].append((fit, ind))
                self.bins[b].sort(key=lambda x: x[0], reverse=True)
                return True
            elif fit >= self.bins[b][-1][0]:
                self.bins[b].pop(-1)
                self.bins[b].append((fit, ind))
                self.bins[b].sort(key=lambda x: x[0], reverse=True)
                return True
        else:
            n_sat = sum(1 for c in self.problem_space.get_constant_constraints()
                        if c.apply(ind))
            if len(infeasible_pop) < self.infeasible_pop_size:
                infeasible_pop.append((n_sat, ind))
        return False

    def _select(self):
        total_f = sum(len(b) for b in self.bins)
        total_i = len(self.infeasible_pop)
        if total_f == 0 and total_i == 0:
            return self.problem_space.generate_random_individual()
        if total_f > 0 and (total_i == 0 or
                random.random() < (total_f * 2) / (total_f + total_i)):
            occupied = [i for i in range(len(self.bins)) if self.bins[i]]
            if not occupied:
                return self.problem_space.generate_random_individual()
            return roulette_selection(self.bins[random.choice(occupied)])[1]
        return roulette_selection(self.infeasible_pop)[1]

    def _shuffle_and_evict(self):
        all_solutions = []
        for b in self.bins:
            for fit, ind in b:
                if self._var_ok(ind):
                    all_solutions.append(ind)
                elif len(self.eviction_pool) < self.eviction_max:
                    self.eviction_pool.append((fit, ind))
        for n_sat, ind in self.infeasible_pop:
            all_solutions.append(ind)

        still_evicted = []
        for fit, ind in self.eviction_pool:
            if self._var_ok(ind) and self._const_ok(ind):
                all_solutions.append(ind)
            else:
                still_evicted.append((fit, ind))
        self.eviction_pool = still_evicted

        n_bins = self.problem_space.get_num_bins()
        self.bins = [[] for _ in range(n_bins)]
        self.infeasible_pop = []
        for ind in all_solutions:
            self._place(ind, self.infeasible_pop)

    def run_one_generation(self, cons_changed):
        if cons_changed:
            self._shuffle_and_evict()

        self.infeasible_pop.sort(key=lambda x: x[0], reverse=True)
        new_inf = self.infeasible_pop[:self.elitism_num]

        successes = 0
        total = 0

        for _ in range(self.population_size // 2):
            p1, p2 = self._select(), self._select()
            if random.random() < self.cross_over_rate:
                c1, c2 = self.problem_space.cross_over(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
            c1 = self.problem_space.mutate(c1, self.current_rate)
            c2 = self.problem_space.mutate(c2, self.current_rate)
            s1 = self._place(c1, new_inf)
            s2 = self._place(c2, new_inf)
            successes += int(s1) + int(s2)
            total += 2

        if total > 0:
            sr = successes / total
            if sr > self.target_success:
                self.current_rate = min(self.rate_max, self.current_rate * self.adapt_factor)
            else:
                self.current_rate = max(self.rate_min, self.current_rate / self.adapt_factor)

        self.infeasible_pop = new_inf
        return [[(f, i) for f, i in bl if self._all_ok(i)] for bl in self.bins]


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
    from Personas.Strict import StrictUser
    from Algorithms.VCMapElites import VariableConstraintMapElites
    from Algorithms.Filtering import Filtering
    from Algorithms.RandomRestarts import RandomRestarts
    from Algorithms.Shuffling import Shuffling

    from mvp4_eggroll_lowrank import EGGROLLElites
    from mvp6_evict_restart import EvictRestartElites
    from mvp7_bandit_experts import BanditExpertElites
    from mvp9_bandit_evict import BanditEvictElites
    from mvp11_de_elites import DEElites
    from mvp13_adaptive_rate import AdaptiveRateElites
    from mvp18_epsilon_bandit import EpsilonBanditElites
    from mvp22_ultimate_hybrid import UltimateHybridElites

    algo_map = {
        # Framework baselines
        "VC-MAP-Elites": (VariableConstraintMapElites, {}),
        "Filtering": (Filtering, {}),
        "RandomRestarts": (RandomRestarts, {}),
        "Shuffling": (Shuffling, {}),
        # Wave 1 winners
        "EGGROLL": (EGGROLLElites, {}),
        "Evict-Restart": (EvictRestartElites, {}),
        "Bandit-UCB1": (BanditExpertElites, {}),
        "Epsilon-Bandit": (EpsilonBanditElites, {}),
        # Wave 2 winners
        "UH-nobandit": (UltimateHybridElites, {"use_bandit": False}),
        "AdaptRate-noreset": (AdaptiveRateElites, {"reset_on_change": False}),
        "BdEvict-nostag": (BanditEvictElites, {"use_stagnation": False}),
        "DE-Elites": (DEElites, {}),
        # New shuffling hybrids
        "Shuf+AdaptRate": (ShufflingAdaptRate, {}),
        "Shuf+Evict": (ShufflingEvict, {}),
        "Shuf+Both": (ShufflingBoth, {}),
    }
    domain_map = {
        "TTP": TTPProblemSpace,
        "LogicPuzzles": LogicPuzzleSpace,
    }
    persona_map = {
        "Exploratory": ExploratoryUser,
        "Cycle": TwoForOneBackUser,
        "Adaptive": AdaptiveUser,
        "Strict": StrictUser,
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
    n_seeds = 30
    n_workers = min(cpu_count(), 60)

    algorithms = [
        "VC-MAP-Elites", "Filtering", "RandomRestarts", "Shuffling",
        "EGGROLL", "Evict-Restart", "Bandit-UCB1", "Epsilon-Bandit",
        "UH-nobandit", "AdaptRate-noreset", "BdEvict-nostag", "DE-Elites",
        "Shuf+AdaptRate", "Shuf+Evict", "Shuf+Both",
    ]
    domains = ["TTP", "LogicPuzzles"]
    personas = ["Exploratory", "Cycle", "Adaptive", "Strict"]

    total_tasks = len(algorithms) * len(domains) * len(personas) * n_seeds
    print(f"Comprehensive Rerun (official params, all baselines, all personas)")
    print(f"  Parameters: gens={N_GENS} pop={POP_SIZE} mem={MEMORY} interval={INTERVAL}")
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

    outpath = os.path.join(SCRIPT_DIR, "results", "comprehensive.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved to {outpath}")

    # === Print results ===
    shuffling_total = sum(np.mean(all_data.get(f"{d}|Shuffling|{p}", [0]))
                          for d in domains for p in personas)

    print("\n" + "=" * 110)
    print("GRAND TOTAL RANKING (vs Shuffling, the strongest baseline)")
    print("=" * 110)
    totals = []
    for a in algorithms:
        total = sum(np.mean(all_data.get(f"{d}|{a}|{p}", [0]))
                    for d in domains for p in personas)
        totals.append((total, a))
    totals.sort(reverse=True)
    for rank, (total, aname) in enumerate(totals, 1):
        pct = ((total - shuffling_total) / abs(shuffling_total) * 100) if shuffling_total != 0 else 0
        print(f"  {rank:<3} {aname:<22} {total:>12.1f}  (vs Shuffling: {pct:>+7.1f}%)")

    # Per-domain per-persona
    for dname in domains:
        print(f"\n  {dname}:")
        print(f"  {'Algorithm':<22} " + " ".join(f"{p:>14}" for p in personas) + f" {'Total':>14}")
        print(f"  {'-'*22} " + " ".join(f"{'-'*14}" for _ in personas) + f" {'-'*14}")
        rows = []
        for a in algorithms:
            vals = [np.mean(all_data.get(f"{dname}|{a}|{p}", [0])) for p in personas]
            rows.append((sum(vals), a, vals))
        rows.sort(reverse=True)
        for total, aname, vals in rows:
            print(f"  {aname:<22} " +
                  " ".join(f"{v:>14.1f}" for v in vals) +
                  f" {total:>14.1f}")

    # Stats: top algorithms vs Shuffling
    print("\n" + "=" * 110)
    print("STATISTICAL TESTS vs Shuffling (Mann-Whitney U)")
    print("=" * 110)
    for dname in domains:
        for aname in algorithms:
            if aname == "Shuffling":
                continue
            for pname in personas:
                a = np.array(all_data.get(f"{dname}|{aname}|{pname}", [0]))
                b = np.array(all_data.get(f"{dname}|Shuffling|{pname}", [0]))
                ma, mb = np.mean(a), np.mean(b)
                try:
                    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    p = 1.0
                if p < 0.05:
                    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
                    pooled = np.sqrt((sa**2 + sb**2) / 2)
                    d = (ma - mb) / pooled if pooled > 0 else 0
                    sig = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                    print(f"  {dname:14} {aname:<22} {pname:<12} {ma:>10.1f} vs {mb:>10.1f} "
                          f"d={d:>+.3f} p={p:.4f} {sig}")


if __name__ == "__main__":
    main()
