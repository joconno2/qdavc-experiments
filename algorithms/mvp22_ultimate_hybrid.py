"""
MVP 22: Ultimate Hybrid (v2 - fixed placement)

Combines adaptive mutation rate (1/5th rule, no reset) with Shuffling-style
population management. On constraint change, ALL individuals (feasible +
infeasible) are reshuffled against new constraints. Between changes, standard
MAP-Elites placement with full constraint checking.

This fixes the cold-start problem on TTP instances where 0% of random
individuals satisfy constant constraints. By maintaining both feasible and
infeasible populations and selecting from both, evolution can reach the
feasible region through incremental constraint satisfaction.

Base: Shuffling (framework baseline)
Added: 1/5th adaptive mutation rate (no reset on constraint change)
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


def decide(rate):
    return random.random() < rate


class UltimateHybridElites(VariableConstraintGA):

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3,
                 target_success=0.2, adapt_factor=1.2,
                 rate_min=0.05, rate_max=0.9,
                 use_adaptive_rate=True):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
        self.target_success = target_success
        self.adapt_factor = adapt_factor
        self.rate_min = rate_min
        self.rate_max = rate_max
        self.use_adaptive_rate = use_adaptive_rate
        super().__init__(problem_space, number_generations, population_size,
                         max_memory, cross_over_rate, mutation_rate, user,
                         update_interval)

    def _sort_pop(self, pop):
        pop.sort(key=lambda i: i[0], reverse=True)

    def set_up(self):
        self.infeasible_pop_size = self.max_memory * self.infeasible_rate
        self.elitism_num = round(self.infeasible_pop_size * self.elitism)
        self.feasible_pop_size = self.max_memory - self.infeasible_pop_size
        self.inds_per_bin = math.floor(self.feasible_pop_size /
                                        self.problem_space.get_num_bins())
        self.bins = [[] for _ in range(self.problem_space.get_num_bins())]
        self.num_feasible = 0
        self.infeasible_pop = []
        self.current_rate = self.mutation_rate

        for _ in range(self.population_size):
            ind = self.problem_space.generate_random_individual()
            self.place_in_bin(ind, self.infeasible_pop)

    def _all_ok(self, ind):
        """Check both constant and variable constraints."""
        for con in self.problem_space.get_constant_constraints():
            if not con.apply(ind):
                return False
        for con in self.variable_constraints:
            if not con.apply(ind):
                return False
        return True

    def place_in_bin(self, ind, infeasible_pop):
        """Place individual. Matches Shuffling's approach: check ALL constraints
        for bin placement, keep infeasible individuals in infeasible_pop."""
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
                self._sort_pop(self.bins[b])
                self.num_feasible += 1
                return True
            elif fit >= self.bins[b][-1][0]:
                self.bins[b].pop(-1)
                self.bins[b].append((fit, ind))
                self._sort_pop(self.bins[b])
                self.num_feasible += 1
                return True
        else:
            if len(infeasible_pop) < self.infeasible_pop_size:
                infeasible_pop.append((constraints_sat, ind))
        return False

    def select(self):
        """Select parent from feasible bins or infeasible population."""
        total_f = self.num_feasible
        total_i = len(self.infeasible_pop)
        if total_f == 0 and total_i == 0:
            return self.problem_space.generate_random_individual()

        if total_f > 0 and (total_i == 0 or
                decide((total_f * 2) / (total_f + total_i))):
            bi = random.choice(range(len(self.bins)))
            while len(self.bins[bi]) == 0:
                bi = random.choice(range(len(self.bins)))
            return roulette_selection(self.bins[bi])[1]
        else:
            return roulette_selection(self.infeasible_pop)[1]

    def re_shuffle(self):
        """On constraint change: collect all individuals, reinitialize,
        re-place everyone under new constraints. Matches Shuffling."""
        all_children = self.infeasible_pop[:]
        all_children += [el for li in self.bins for el in li]

        new_infeasible = []
        self.bins = [[] for _ in range(self.problem_space.get_num_bins())]
        self.num_feasible = 0
        self.infeasible_pop = []

        # Re-generate initial random population (like Shuffling's set_up)
        for _ in range(self.population_size):
            ind = self.problem_space.generate_random_individual()
            self.place_in_bin(ind, new_infeasible)

        # Re-place all existing individuals
        for c in all_children:
            self.place_in_bin(c[1], new_infeasible)

        self.infeasible_pop = new_infeasible

    def run_one_generation(self, cons_changed):
        if cons_changed:
            self.re_shuffle()

        self._sort_pop(self.infeasible_pop)
        new_infeasible = self.infeasible_pop[:self.elitism_num]

        successes = 0
        total = 0

        for _ in range(self.population_size // 2):
            child1 = self.select()
            child2 = self.select()

            if decide(self.cross_over_rate):
                child1, child2 = self.problem_space.cross_over(child1, child2)

            rate = self.current_rate if self.use_adaptive_rate else self.mutation_rate
            child1 = self.problem_space.mutate(child1, rate)
            child2 = self.problem_space.mutate(child2, rate)

            s1 = self.place_in_bin(child1, new_infeasible)
            s2 = self.place_in_bin(child2, new_infeasible)
            successes += int(s1) + int(s2)
            total += 2

        # Adapt mutation rate (1/5th rule, NO reset on constraint change)
        if self.use_adaptive_rate and total > 0:
            sr = successes / total
            if sr > self.target_success:
                self.current_rate = min(self.rate_max,
                                        self.current_rate * self.adapt_factor)
            else:
                self.current_rate = max(self.rate_min,
                                        self.current_rate / self.adapt_factor)

        self.infeasible_pop = new_infeasible

        # Return only fully feasible individuals (bins already contain only
        # feasible individuals since place_in_bin checks all constraints)
        return self.bins


ALGORITHM_CLASS = UltimateHybridElites
ALGORITHM_NAME = "Ultimate-Hybrid"
ALGORITHM_KWARGS = {}
