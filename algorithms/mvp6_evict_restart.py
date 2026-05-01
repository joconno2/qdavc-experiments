"""
MVP 6: Evict-Restart MAP-Elites (our best method from earlier)

The known good algorithm for comparison. Eviction pool + restart on
constraint change + stagnation restart. No PAL bias. No VC-aware selection.
"""

import sys, os, math, random, copy
import numpy.random as npr
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, "framework"))
from GeneticAlgorithmInterface import VariableConstraintGA
from ProblemSpaceInterface import ProblemSpace


def roulette_selection(population):
    small = min(c[0] for c in population)
    add = -small if small < 0 else 0
    total = sum(c[0] + add for c in population)
    if total == 0:
        probs = [1.0 / len(population)] * len(population)
    else:
        probs = [(c[0] + add) / total for c in population]
    return population[npr.choice(len(population), p=probs)]


class EvictRestartElites(VariableConstraintGA):

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3,
                 stagnation_patience=3, restart_fraction=0.5):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
        self.stagnation_patience = stagnation_patience
        self.restart_fraction = restart_fraction
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
            elif fit >= self.bins[b][-1][0]:
                self.bins[b].pop(-1)
                self.bins[b].append((fit, ind))
                self.bins[b].sort(key=lambda x: x[0], reverse=True)
        else:
            n_sat = sum(1 for c in self.problem_space.get_constant_constraints() if c.apply(ind))
            if len(infeasible_pop) < self.infeasible_pop_size:
                infeasible_pop.append((n_sat, ind))

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
                if len(seed_pool) >= 2 and random.random() < self.cross_over_rate:
                    p2 = roulette_selection(seed_pool)[1]
                    c1, c2 = self.problem_space.cross_over(parent, p2)
                    self._place(c1, new_inf)
                    self._place(c2, new_inf)
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


ALGORITHM_CLASS = EvictRestartElites
ALGORITHM_NAME = "Evict-Restart"
ALGORITHM_KWARGS = {}
