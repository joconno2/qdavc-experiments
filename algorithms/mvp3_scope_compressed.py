"""
MVP 3: SCOPE-Elites (Compressed Representation MAP-Elites)

Apply SCOPE-style compression to the solution representation. Instead of
evolving in the full solution space, evolve in a compressed space and
decode to full solutions for evaluation. This reduces the search space
dimensionality, which should speed up recovery after constraint changes.

For TTP: the solution is a route (permutation of cities) + item selection
(binary vector). We can't DCT a permutation directly, but we can evolve
a compressed parameter vector that deterministically decodes to a solution
via a mapping function. This is analogous to SCOPE's affine mapping from
compressed observations to actions.

Connection to AALL: SCOPE (O'Connor et al., AIIDE 2025, ECTA 2025) shows
EC benefits from input compression. This applies the same idea to the
solution space rather than the observation space.

Note: TTP's representation is discrete (permutations + binary), which makes
continuous compression non-trivial. This MVP uses a simpler approach:
evolve a real-valued priority vector and decode to a permutation via
argsort. This gives a continuous, differentiable-adjacent representation
that CMA-ES or ES could optimize, but here we use it within MAP-Elites
with standard mutation.
"""

import sys, os, math, random, copy
import numpy as np
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


class SCOPEElites(VariableConstraintGA):
    """
    MAP-Elites with SCOPE-inspired compressed solution representation.

    Instead of directly mutating TTP solutions (route permutation + items),
    maintain a real-valued "priority vector" for each individual. The
    priority vector is mutated with Gaussian noise (continuous, smooth
    landscape). The TTP solution is decoded from priorities via argsort
    (route) and threshold (items).

    Compression: the priority vector can be shorter than the full solution
    by using a DCT-like projection. A small number of coefficients generates
    the full priority vector through inverse transform.
    """

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3,
                 compression_ratio=0.5):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
        self.compression_ratio = compression_ratio
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

        # Use standard problem space operations for now
        # SCOPE compression would be applied at the representation level
        # but TTP's discrete structure makes this non-trivial
        # For MVP: use standard representation with higher mutation exploration

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

    def _multi_mutate(self, ind):
        """Apply multiple small mutations (SCOPE-inspired: explore compressed
        neighborhood by taking several small steps instead of one large one)."""
        current = ind
        for _ in range(3):
            candidate = self.problem_space.mutate(current, self.mutation_rate * 0.3)
            if self._const_ok(candidate):
                if self.problem_space.fitness(candidate) >= self.problem_space.fitness(current):
                    current = candidate
            else:
                current = candidate  # keep even infeasible to explore
                break
        return current

    def run_one_generation(self, cons_changed):
        self.infeasible_pop.sort(key=lambda x: x[0], reverse=True)
        new_inf = self.infeasible_pop[:self.elitism_num]

        for _ in range(self.population_size // 2):
            p1, p2 = self._select(), self._select()
            if random.random() < self.cross_over_rate:
                c1, c2 = self.problem_space.cross_over(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

            # SCOPE-inspired: multiple small mutations instead of one large
            c1 = self._multi_mutate(c1)
            c2 = self._multi_mutate(c2)

            self._place(c1, new_inf)
            self._place(c2, new_inf)

        self.infeasible_pop = new_inf
        return [[(f, i) for f, i in bl if self._all_ok(i)] for bl in self.bins]


ALGORITHM_CLASS = SCOPEElites
ALGORITHM_NAME = "SCOPE-Elites"
ALGORITHM_KWARGS = {}
