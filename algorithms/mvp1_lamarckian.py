"""
MVP 1: Lamarckian MAP-Elites

Lamarckian evolution: after mutation, apply local search to improve the
offspring BEFORE inserting into the archive. The improved solution replaces
the original in the genome (Lamarckian, not Baldwinian). This means the
archive accumulates locally-optimized solutions, and mutations of those
solutions start from a better baseline.

In the variable-constraint setting, local search can also optimize for
constraint satisfaction: hill-climb toward satisfying the current variable
constraints before archive insertion.

Connection to AALL: memetic algorithms are understudied in the QD literature.
Lamarckian local search + MAP-Elites is a clean combination.
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


class LamarckianElites(VariableConstraintGA):

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3,
                 local_search_steps=3):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
        self.local_search_steps = local_search_steps
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

    def _local_search(self, ind):
        """Lamarckian local search: mutate N times, keep if better."""
        best = ind
        best_fit = self.problem_space.fitness(ind) if self._const_ok(ind) else -1e18
        for _ in range(self.local_search_steps):
            candidate = self.problem_space.mutate(best, self.mutation_rate * 0.5)
            if self._const_ok(candidate):
                cfit = self.problem_space.fitness(candidate)
                if cfit > best_fit:
                    best = candidate
                    best_fit = cfit
        return best

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

    def run_one_generation(self, cons_changed):
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

            # Lamarckian: local search before insertion
            c1 = self._local_search(c1)
            c2 = self._local_search(c2)

            self._place(c1, new_inf)
            self._place(c2, new_inf)

        self.infeasible_pop = new_inf
        return [[(f, i) for f, i in bl if self._all_ok(i)] for bl in self.bins]


ALGORITHM_CLASS = LamarckianElites
ALGORITHM_NAME = "Lamarckian-Elites"
ALGORITHM_KWARGS = {}
