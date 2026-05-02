"""
MVP 11: Differential Evolution MAP-Elites

DE-style mutation within MAP-Elites. Instead of Gaussian mutation, offspring
are created via DE/rand/1: pick 3 archive members a, b, c. Create donor by
crossover(a, mutate(b-c direction)). Then trial = crossover(parent, donor).

Since TTP uses discrete representations (permutations + binary items), we
can't do vector arithmetic directly. Instead, we approximate DE by:
  1. Select 3 archive parents (a, b, c)
  2. Crossover b and c to get a "difference" individual
  3. Crossover a with this difference
  4. Small mutation on top

This is a different exploration pattern from standard or EGGROLL mutation.
"""

import sys, os, random, copy
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


class DEElites(VariableConstraintGA):

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3,
                 de_probability=0.7, mutation_scale=0.3):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
        self.de_probability = de_probability
        self.mutation_scale = mutation_scale
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

    def _get_archive_member(self):
        occupied = [i for i in range(len(self.bins)) if self.bins[i]]
        if not occupied:
            return None
        b = random.choice(occupied)
        return self.bins[b][0][1]  # best in random occupied bin

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

    def _de_mutate(self, parent):
        """DE/rand/1-inspired mutation for discrete spaces."""
        a = self._get_archive_member()
        b = self._get_archive_member()
        c = self._get_archive_member()

        if a is None or b is None or c is None:
            return self.problem_space.mutate(parent, self.mutation_rate)

        # "Difference vector": crossover b and c
        diff, _ = self.problem_space.cross_over(b, c)
        # "Donor": crossover a with difference
        donor, _ = self.problem_space.cross_over(a, diff)
        # "Trial": crossover parent with donor
        trial, _ = self.problem_space.cross_over(parent, donor)
        # Small mutation on top
        return self.problem_space.mutate(trial, self.mutation_scale)

    def run_one_generation(self, cons_changed):
        self.infeasible_pop.sort(key=lambda x: x[0], reverse=True)
        new_inf = self.infeasible_pop[:self.elitism_num]

        for _ in range(self.population_size // 2):
            p1, p2 = self._select(), self._select()

            if random.random() < self.de_probability:
                c1 = self._de_mutate(p1)
                c2 = self._de_mutate(p2)
            else:
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


ALGORITHM_CLASS = DEElites
ALGORITHM_NAME = "DE-Elites"
ALGORITHM_KWARGS = {}
