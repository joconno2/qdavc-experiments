"""
MVP 22: Ultimate Hybrid

Combines the three winning mechanisms from wave 1:
1. Adaptive mutation rate with NO reset (wave 1 champion, +94.6%)
2. Eviction pool without stagnation restart (stagnation restart hurts)
3. Epsilon-greedy bandit over mutation experts (simpler beats UCB1)

Each mechanism addresses a different aspect:
- Adaptive rate: self-tunes exploration intensity via 1/5th rule
- Eviction pool: preserves solutions across constraint changes (Adaptive persona)
- Epsilon bandit: selects among directed/standard/exploratory/conservative mutations
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


class UltimateHybridElites(VariableConstraintGA):

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3,
                 # Adaptive rate params
                 target_success=0.2, adapt_factor=1.2,
                 rate_min=0.05, rate_max=0.9,
                 # Eviction params
                 use_eviction=True,
                 # Bandit params
                 n_direction_history=5,
                 epsilon_start=0.5, epsilon_floor=0.05, epsilon_decay=0.95,
                 use_bandit=True):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
        self.target_success = target_success
        self.adapt_factor = adapt_factor
        self.rate_min = rate_min
        self.rate_max = rate_max
        self.use_eviction = use_eviction
        self.n_direction_history = n_direction_history
        self.epsilon_start = epsilon_start
        self.epsilon_floor = epsilon_floor
        self.epsilon_decay = epsilon_decay
        self.use_bandit = use_bandit
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

        # Eviction pool (no stagnation restart)
        self.eviction_pool = []
        self.eviction_max = max(n_bins, self.max_memory // 4)

        # Epsilon-greedy bandit
        self.successful_parents = []
        self.n_experts = 4
        self.expert_successes = [0] * self.n_experts
        self.expert_attempts = [1] * self.n_experts
        self.epsilon = self.epsilon_start

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

    def _select_parent(self):
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

    # -- Eviction --

    def _evict_and_reinsert(self):
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

    # -- Epsilon-greedy experts --

    def _mutate_directed(self, parent):
        if self.successful_parents:
            donor = random.choice(self.successful_parents)
            child, _ = self.problem_space.cross_over(parent, donor)
            return self.problem_space.mutate(child, self.current_rate * 0.3)
        return self.problem_space.mutate(parent, self.current_rate)

    def _mutate_standard(self, parent):
        return self.problem_space.mutate(parent, self.current_rate)

    def _mutate_exploratory(self, parent):
        child = parent
        for _ in range(3):
            child = self.problem_space.mutate(child, min(self.current_rate * 3.0, 1.0))
        return child

    def _mutate_conservative(self, parent):
        return self.problem_space.mutate(parent, self.current_rate * 0.15)

    def _apply_expert(self, eidx, parent):
        if eidx == 0: return self._mutate_directed(parent)
        elif eidx == 1: return self._mutate_standard(parent)
        elif eidx == 2: return self._mutate_exploratory(parent)
        else: return self._mutate_conservative(parent)

    def _select_expert(self):
        if not self.use_bandit:
            return 1
        if random.random() < self.epsilon:
            return random.randint(0, self.n_experts - 1)
        rates = [self.expert_successes[i] / self.expert_attempts[i]
                 for i in range(self.n_experts)]
        return rates.index(max(rates))

    def _update_expert(self, eidx, success):
        self.expert_attempts[eidx] += 1
        if success:
            self.expert_successes[eidx] += 1

    def _reset_bandit(self):
        self.expert_successes = [0] * self.n_experts
        self.expert_attempts = [1] * self.n_experts
        self.epsilon = self.epsilon_start

    # -- Main --

    def run_one_generation(self, cons_changed):
        if cons_changed:
            self._reset_bandit()
            self.successful_parents = []
            if self.use_eviction:
                self._evict_and_reinsert()
            # NO rate reset. NO stagnation restart.

        # Decay epsilon
        self.epsilon = max(self.epsilon_floor, self.epsilon * self.epsilon_decay)

        self.infeasible_pop.sort(key=lambda x: x[0], reverse=True)
        new_inf = self.infeasible_pop[:self.elitism_num]

        successes = 0
        total = 0

        for _ in range(self.population_size // 2):
            p1 = self._select_parent()
            p2 = self._select_parent()
            e1 = self._select_expert()
            e2 = self._select_expert()
            c1 = self._apply_expert(e1, p1)
            c2 = self._apply_expert(e2, p2)
            if random.random() < self.cross_over_rate:
                c1, c2 = self.problem_space.cross_over(c1, c2)
            s1 = self._place(c1, new_inf)
            s2 = self._place(c2, new_inf)
            self._update_expert(e1, s1)
            self._update_expert(e2, s2)
            successes += int(s1) + int(s2)
            total += 2
            if s1:
                self.successful_parents.append(copy.deepcopy(c1))
                if len(self.successful_parents) > self.n_direction_history:
                    self.successful_parents.pop(0)
            if s2:
                self.successful_parents.append(copy.deepcopy(c2))
                if len(self.successful_parents) > self.n_direction_history:
                    self.successful_parents.pop(0)

        # Adapt mutation rate (1/5th rule, NO reset)
        if total > 0:
            success_rate = successes / total
            if success_rate > self.target_success:
                self.current_rate = min(self.rate_max,
                                        self.current_rate * self.adapt_factor)
            else:
                self.current_rate = max(self.rate_min,
                                        self.current_rate / self.adapt_factor)

        self.infeasible_pop = new_inf
        return [[(f, i) for f, i in bl if self._all_ok(i)] for bl in self.bins]


ALGORITHM_CLASS = UltimateHybridElites
ALGORITHM_NAME = "Ultimate-Hybrid"
ALGORITHM_KWARGS = {}
