"""
MVP 23: Directional Adaptive Rate

Like UH-nobandit (Shuffling base + 1/5th adaptive rate), but with directional
awareness of constraint changes:

- On constraint ADDITION: rate adapts normally via 1/5th rule (bins empty,
  success rate rises, rate increases automatically)
- On constraint REMOVAL: rate decreases by adapt_factor (landscape got easier,
  less exploration needed)
- Between changes: normal 1/5th rule

This fixes the Exploratory weakness where random add/remove causes rate to
ratchet up on additions without sufficient recovery on removals.
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


class DirectionalAdaptElites(VariableConstraintGA):

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
        self.infeasible_pop_size = int(self.max_memory * self.infeasible_rate)
        self.elitism_num = int(self.infeasible_pop_size * self.elitism)
        feasible_memory = self.max_memory - self.infeasible_pop_size
        self.inds_per_bin = max(1, feasible_memory // self.problem_space.get_num_bins())
        self.bins = [[] for _ in range(self.problem_space.get_num_bins())]
        self.num_feasible = 0
        self.infeasible_pop = []
        self.current_rate = self.mutation_rate
        self.prev_n_constraints = 0  # track constraint count for directionality

        for _ in range(self.population_size):
            ind = self.problem_space.generate_random_individual()
            self.place_in_bin(ind, self.infeasible_pop)

    def place_in_bin(self, ind, infeasible_pop):
        fes = True
        constraints_sat = 0
        for con in self.problem_space.get_constant_constraints() + self.variable_constraints:
            if not con.apply(ind):
                fes = False
            else:
                constraints_sat += 1

        if fes:
            b = self.problem_space.place_in_bin(ind)
            fit = self.problem_space.fitness(ind)
            if len(self.bins[b]) < self.inds_per_bin:
                self.bins[b].append((fit, ind))
                self.bins[b].sort(key=lambda x: x[0], reverse=True)
                self.num_feasible += 1
                return True
            elif fit >= self.bins[b][-1][0]:
                self.bins[b].pop(-1)
                self.bins[b].append((fit, ind))
                self.bins[b].sort(key=lambda x: x[0], reverse=True)
                self.num_feasible += 1
                return True
        else:
            if len(infeasible_pop) < self.infeasible_pop_size:
                infeasible_pop.append((constraints_sat, ind))
        return False

    def select(self):
        total_f = self.num_feasible
        total_i = len(self.infeasible_pop)
        if total_f == 0 and total_i == 0:
            return self.problem_space.generate_random_individual()

        if total_f > 0 and (total_i == 0 or
                random.random() < (total_f * 2) / (total_f + total_i)):
            bi = random.choice(range(len(self.bins)))
            while len(self.bins[bi]) == 0:
                bi = random.choice(range(len(self.bins)))
            return roulette_selection(self.bins[bi])[1]
        else:
            return roulette_selection(self.infeasible_pop)[1]

    def re_shuffle(self):
        all_children = self.infeasible_pop[:]
        all_children += [el for li in self.bins for el in li]

        new_infeasible = []
        self.bins = [[] for _ in range(self.problem_space.get_num_bins())]
        self.num_feasible = 0
        self.infeasible_pop = []

        for _ in range(self.population_size):
            ind = self.problem_space.generate_random_individual()
            self.place_in_bin(ind, new_infeasible)

        for c in all_children:
            self.place_in_bin(c[1], new_infeasible)

        self.infeasible_pop = new_infeasible

    def run_one_generation(self, cons_changed):
        if cons_changed:
            # Detect direction of constraint change
            n_constraints = len(self.variable_constraints)
            constraints_added = n_constraints > self.prev_n_constraints
            constraints_removed = n_constraints < self.prev_n_constraints
            self.prev_n_constraints = n_constraints

            self.re_shuffle()

            # Directional rate adjustment on constraint change
            if constraints_removed:
                # Landscape got easier: reduce rate (less exploration needed)
                self.current_rate = max(self.rate_min,
                                        self.current_rate / self.adapt_factor)
            # On addition: don't adjust here, let 1/5th rule handle it naturally
            # (empty bins -> high success rate -> rate increases)

        self.infeasible_pop.sort(key=lambda x: x[0], reverse=True)
        new_infeasible = self.infeasible_pop[:self.elitism_num]

        successes = 0
        total = 0

        for _ in range(self.population_size // 2):
            child1 = self.select()
            child2 = self.select()

            if random.random() < self.cross_over_rate:
                child1, child2 = self.problem_space.cross_over(child1, child2)

            child1 = self.problem_space.mutate(child1, self.current_rate)
            child2 = self.problem_space.mutate(child2, self.current_rate)

            s1 = self.place_in_bin(child1, new_infeasible)
            s2 = self.place_in_bin(child2, new_infeasible)
            successes += int(s1) + int(s2)
            total += 2

        # Standard 1/5th rule (no reset)
        if total > 0:
            sr = successes / total
            if sr > self.target_success:
                self.current_rate = min(self.rate_max,
                                        self.current_rate * self.adapt_factor)
            else:
                self.current_rate = max(self.rate_min,
                                        self.current_rate / self.adapt_factor)

        self.infeasible_pop = new_infeasible
        return self.bins


ALGORITHM_CLASS = DirectionalAdaptElites
ALGORITHM_NAME = "Directional-Adapt"
ALGORITHM_KWARGS = {}
