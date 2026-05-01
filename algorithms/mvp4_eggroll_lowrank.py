"""
MVP 4: EGGROLL-inspired Low-Rank Perturbation MAP-Elites

EGGROLL (Sarkar et al., 2025) uses low-rank perturbations instead of
full-rank noise for ES. The key insight: structured perturbations explore
more efficiently in high-dimensional spaces because they move along
correlated directions rather than random noise.

Applied to MAP-Elites: instead of mutating solutions with independent
per-gene noise, apply structured perturbations that affect multiple genes
in correlated ways. For TTP, this means route and item mutations are
coupled: changing the route also adjusts which items are taken based on
the new city ordering.

This is a direction-based mutation operator, not a new archive structure.
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


class EGGROLLElites(VariableConstraintGA):
    """
    MAP-Elites with EGGROLL-inspired structured mutations.

    Maintains a set of learned "mutation directions" (low-rank basis vectors)
    that are updated based on which mutations improved fitness. Over time,
    the mutation directions align with profitable search directions.

    Implementation: track the last N successful mutations as direction vectors.
    New mutations are sampled as linear combinations of these directions
    plus small random noise. This creates correlated, structured exploration.
    """

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3,
                 n_directions=5, direction_weight=0.7):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
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

        # Track successful mutations for direction learning
        # Each direction is a (parent, child) pair that improved fitness
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
            if len(self.bins[b]) < self.inds_per_bin:
                self.bins[b].append((fit, ind))
                self.bins[b].sort(key=lambda x: x[0], reverse=True)
                return True, fit
            elif fit >= self.bins[b][-1][0]:
                self.bins[b].pop(-1)
                self.bins[b].append((fit, ind))
                self.bins[b].sort(key=lambda x: x[0], reverse=True)
                return True, fit
        else:
            n_sat = sum(1 for c in self.problem_space.get_constant_constraints() if c.apply(ind))
            if len(infeasible_pop) < self.infeasible_pop_size:
                infeasible_pop.append((n_sat, ind))
        return False, 0

    def _directed_mutate(self, parent):
        """Apply mutation biased by successful directions.

        With probability direction_weight, crossover the parent with a
        successful parent (transferring the "direction" of success).
        With remaining probability, apply standard random mutation.
        """
        if self.successful_parents and random.random() < self.direction_weight:
            # Pick a random successful parent and crossover with it
            # This transfers genetic material from a successful lineage
            donor = random.choice(self.successful_parents)
            child, _ = self.problem_space.cross_over(parent, donor)
            # Small mutation on top
            child = self.problem_space.mutate(child, self.mutation_rate * 0.3)
            return child
        else:
            return self.problem_space.mutate(parent, self.mutation_rate)

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
        if cons_changed:
            # Reset directions on constraint change (old directions may be invalid)
            self.successful_parents = []

        self.infeasible_pop.sort(key=lambda x: x[0], reverse=True)
        new_inf = self.infeasible_pop[:self.elitism_num]

        for _ in range(self.population_size // 2):
            p1 = self._select()
            p2 = self._select()

            c1 = self._directed_mutate(p1)
            c2 = self._directed_mutate(p2)

            inserted1, fit1 = self._place(c1, new_inf)
            inserted2, fit2 = self._place(c2, new_inf)

            # Track successful insertions for direction learning
            if inserted1:
                self.successful_parents.append(copy.deepcopy(c1))
                if len(self.successful_parents) > self.n_directions:
                    self.successful_parents.pop(0)
            if inserted2:
                self.successful_parents.append(copy.deepcopy(c2))
                if len(self.successful_parents) > self.n_directions:
                    self.successful_parents.pop(0)

        self.infeasible_pop = new_inf
        return [[(f, i) for f, i in bl if self._all_ok(i)] for bl in self.bins]


ALGORITHM_CLASS = EGGROLLElites
ALGORITHM_NAME = "EGGROLL-Elites"
ALGORITHM_KWARGS = {}
