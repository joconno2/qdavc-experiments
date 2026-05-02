"""
MVP 12: Novelty-Biased Selection Elites

Parent selection is biased toward bins with low occupancy or recently emptied.
This drives exploration toward gaps in the archive rather than exploiting
occupied bins. Standard mutation otherwise.

The idea: in variable constraint environments, bins empty when constraints
change. Novelty-biased selection preferentially fills those gaps, improving
diversity recovery after constraint shifts.
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


class NoveltySelectionElites(VariableConstraintGA):

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3,
                 novelty_weight=0.5):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
        self.novelty_weight = novelty_weight
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
        self.bin_occupancy_history = [0] * n_bins  # how many gens each bin occupied

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

    def _novelty_select(self):
        """Select parent from bin adjacent to empty/low-occupancy bins."""
        occupied = [i for i in range(len(self.bins)) if self.bins[i]]
        if not occupied:
            return self.problem_space.generate_random_individual()

        if random.random() < self.novelty_weight:
            # Novelty-biased: prefer occupied bins near empty bins
            n_bins = len(self.bins)
            scores = []
            for i in occupied:
                # Count empty neighbors
                empty_neighbors = 0
                for offset in [-2, -1, 1, 2]:
                    j = i + offset
                    if 0 <= j < n_bins and not self.bins[j]:
                        empty_neighbors += 1
                # Lower historical occupancy = more novel
                novelty = 1.0 / (self.bin_occupancy_history[i] + 1)
                scores.append(empty_neighbors + novelty)

            total = sum(scores)
            if total == 0:
                bi = random.choice(occupied)
            else:
                probs = [s / total for s in scores]
                bi = occupied[npr.choice(len(occupied), p=probs)]
            return roulette_selection(self.bins[bi])[1]
        else:
            # Standard selection
            total_f = sum(len(b) for b in self.bins)
            total_i = len(self.infeasible_pop)
            if total_f > 0 and (total_i == 0 or
                    random.random() < (total_f * 2) / (total_f + total_i)):
                return roulette_selection(self.bins[random.choice(occupied)])[1]
            if self.infeasible_pop:
                return roulette_selection(self.infeasible_pop)[1]
            return self.problem_space.generate_random_individual()

    def run_one_generation(self, cons_changed):
        # Update occupancy history
        for i in range(len(self.bins)):
            if self.bins[i]:
                self.bin_occupancy_history[i] += 1

        self.infeasible_pop.sort(key=lambda x: x[0], reverse=True)
        new_inf = self.infeasible_pop[:self.elitism_num]

        for _ in range(self.population_size // 2):
            p1, p2 = self._novelty_select(), self._novelty_select()
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


ALGORITHM_CLASS = NoveltySelectionElites
ALGORITHM_NAME = "Novelty-Selection"
ALGORITHM_KWARGS = {}
