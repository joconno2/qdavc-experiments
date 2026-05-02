"""
MVP 10: Constraint Memory Elites

Remembers solutions that worked under specific constraint configurations.
When constraints change, checks if the new config (or a similar one) has been
seen before. If so, seeds the archive from memory.

Targets the Cycle persona directly: constraints repeat in a pattern, so
recalled solutions should be immediately useful.

Memory key: frozenset of constraint hashes. Stores top-K solutions per config.
"""

import sys, os, random, copy, hashlib
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


class ConstraintMemoryElites(VariableConstraintGA):

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3,
                 memory_k=20, use_memory=True):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
        self.memory_k = memory_k
        self.use_memory = use_memory
        super().__init__(problem_space, number_generations, population_size,
                         max_memory, cross_over_rate, mutation_rate, user,
                         update_interval)

    def _constraint_key(self):
        """Hash the current variable constraint set for memory lookup."""
        reprs = sorted(repr(c) for c in self.variable_constraints)
        return hashlib.md5("|".join(reprs).encode()).hexdigest()

    def set_up(self):
        n_bins = self.problem_space.get_num_bins()
        self.infeasible_pop_size = int(self.max_memory * self.infeasible_rate)
        self.elitism_num = int(self.infeasible_pop_size * self.elitism)
        feasible_memory = self.max_memory - self.infeasible_pop_size
        self.inds_per_bin = max(1, feasible_memory // n_bins)
        self.bins = [[] for _ in range(n_bins)]
        self.infeasible_pop = []
        self.constraint_memory = {}  # key -> [(fit, ind), ...]

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

    def _save_to_memory(self):
        """Save current archive's best solutions to constraint memory."""
        key = self._constraint_key()
        elites = []
        for b in self.bins:
            for fit, ind in b:
                if self._all_ok(ind):
                    elites.append((fit, copy.deepcopy(ind)))
        elites.sort(key=lambda x: x[0], reverse=True)
        self.constraint_memory[key] = elites[:self.memory_k]

    def _recall_from_memory(self):
        """Try to recall solutions from memory for current constraints."""
        if not self.use_memory:
            return
        key = self._constraint_key()
        recalled = self.constraint_memory.get(key, [])
        reinserted = 0
        for fit, ind in recalled:
            if self._all_ok(ind):
                new_inf = []
                if self._place(ind, new_inf):
                    reinserted += 1

    def run_one_generation(self, cons_changed):
        if cons_changed:
            self._save_to_memory()
            self._recall_from_memory()

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


ALGORITHM_CLASS = ConstraintMemoryElites
ALGORITHM_NAME = "Constraint-Memory"
ALGORITHM_KWARGS = {}
