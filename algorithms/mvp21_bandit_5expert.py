"""
MVP 21: Bandit with 5 Experts (adds SCOPE multi-step)

Same bandit framework as MVP 7 but adds a 5th expert: SCOPE-style
multi-step small mutations. Tests whether a richer expert pool
improves bandit adaptation.

Experts:
0. Directed (EGGROLL-style crossover with successful parent)
1. Standard (base rate mutation)
2. Exploratory (3x high-variance mutation)
3. Conservative (0.15x low-variance)
4. Multi-step (3 sequential small mutations, keep improvements)
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


class Bandit5ExpertElites(VariableConstraintGA):

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3,
                 n_direction_history=5, ucb_c=1.0, window_size=20):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
        self.n_direction_history = n_direction_history
        self.ucb_c = ucb_c
        self.window_size = window_size
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
        self.successful_parents = []

        self.n_experts = 5
        self.expert_attempts = [1] * self.n_experts
        self.expert_successes = [1] * self.n_experts
        self.total_rounds = self.n_experts

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

    def _mutate_directed(self, parent):
        if self.successful_parents:
            donor = random.choice(self.successful_parents)
            child, _ = self.problem_space.cross_over(parent, donor)
            return self.problem_space.mutate(child, self.mutation_rate * 0.3)
        return self.problem_space.mutate(parent, self.mutation_rate)

    def _mutate_standard(self, parent):
        return self.problem_space.mutate(parent, self.mutation_rate)

    def _mutate_exploratory(self, parent):
        child = parent
        for _ in range(3):
            child = self.problem_space.mutate(child, min(self.mutation_rate * 3.0, 1.0))
        return child

    def _mutate_conservative(self, parent):
        return self.problem_space.mutate(parent, self.mutation_rate * 0.15)

    def _mutate_multistep(self, parent):
        """SCOPE-style: 3 sequential small mutations, keep best improvement."""
        best = copy.deepcopy(parent)
        best_fit = self.problem_space.fitness(best) if self._const_ok(best) else -1e18
        current = parent
        for _ in range(3):
            candidate = self.problem_space.mutate(current, self.mutation_rate * 0.5)
            if self._const_ok(candidate):
                cand_fit = self.problem_space.fitness(candidate)
                if cand_fit > best_fit:
                    best = candidate
                    best_fit = cand_fit
            current = candidate
        return best

    def _apply_expert(self, eidx, parent):
        if eidx == 0: return self._mutate_directed(parent)
        elif eidx == 1: return self._mutate_standard(parent)
        elif eidx == 2: return self._mutate_exploratory(parent)
        elif eidx == 3: return self._mutate_conservative(parent)
        else: return self._mutate_multistep(parent)

    def _select_expert(self):
        best_idx, best_score = 0, -1e18
        for i in range(self.n_experts):
            if self.expert_attempts[i] == 0: return i
            sr = self.expert_successes[i] / self.expert_attempts[i]
            explore = self.ucb_c * math.sqrt(math.log(self.total_rounds + 1) / self.expert_attempts[i])
            score = sr + explore
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _update_expert(self, eidx, success):
        self.expert_attempts[eidx] += 1
        if success: self.expert_successes[eidx] += 1
        self.total_rounds += 1
        if self.total_rounds % self.window_size == 0:
            for i in range(self.n_experts):
                self.expert_attempts[i] = max(1, self.expert_attempts[i] // 2)
                self.expert_successes[i] = max(0, self.expert_successes[i] // 2)
            self.total_rounds = sum(self.expert_attempts)

    def _reset_bandit(self):
        self.expert_attempts = [1] * self.n_experts
        self.expert_successes = [1] * self.n_experts
        self.total_rounds = self.n_experts

    def run_one_generation(self, cons_changed):
        if cons_changed:
            self._reset_bandit()
            self.successful_parents = []

        self.infeasible_pop.sort(key=lambda x: x[0], reverse=True)
        new_inf = self.infeasible_pop[:self.elitism_num]

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
            if s1:
                self.successful_parents.append(copy.deepcopy(c1))
                if len(self.successful_parents) > self.n_direction_history:
                    self.successful_parents.pop(0)
            if s2:
                self.successful_parents.append(copy.deepcopy(c2))
                if len(self.successful_parents) > self.n_direction_history:
                    self.successful_parents.pop(0)

        self.infeasible_pop = new_inf
        return [[(f, i) for f, i in bl if self._all_ok(i)] for bl in self.bins]


ALGORITHM_CLASS = Bandit5ExpertElites
ALGORITHM_NAME = "Bandit-5Expert"
ALGORITHM_KWARGS = {}
