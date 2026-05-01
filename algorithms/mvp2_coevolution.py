"""
MVP 2: Coevolutionary MAP-Elites

Coevolve two populations:
1. Solutions (the main archive, standard MAP-Elites)
2. Constraint predictors (a small population of "models" that predict
   which constraints will be added/removed next)

The constraint predictors bias parent selection: solutions that satisfy
predicted future constraints get selection priority. Predictors are
evaluated by how well they predicted the actual constraint changes at
the last update interval.

Connection to AALL: Parker's PAL work coevolves model parameters alongside
controllers. This extends the idea to coevolving constraint-environment
models alongside solutions.
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


class CoevolutionaryElites(VariableConstraintGA):
    """
    Coevolve solutions + constraint predictors.
    Predictors are represented as sets of constraints sampled from the
    problem space. Each predictor "predicts" that those constraints will
    be active. Solutions that satisfy a predictor's constraints get a
    selection bonus.
    """

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3,
                 n_predictors=5, predictor_bonus=1.5):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
        self.n_predictors = n_predictors
        self.predictor_bonus = predictor_bonus
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

        # Predictor population: each is a list of constraints
        self.predictors = []
        for _ in range(self.n_predictors):
            n_cons = random.randint(1, 4)
            pred = [self.problem_space.get_rand_constraint() for _ in range(n_cons)]
            self.predictors.append({"constraints": pred, "score": 0.0})

        self.best_predictor = self.predictors[0]
        self.prev_var_constraints = []

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

    def _evaluate_predictors(self):
        """Score predictors by how well they predicted current variable constraints."""
        for pred in self.predictors:
            # Score: fraction of current variable constraints that the
            # predictor's constraint set overlaps with (by type similarity)
            # Simple heuristic: count how many predictor constraints are
            # satisfied by individuals that also satisfy variable constraints
            score = 0
            total_in_archive = 0
            for b in self.bins:
                for fit, ind in b:
                    total_in_archive += 1
                    pred_sat = sum(1 for c in pred["constraints"] if c.apply(ind))
                    var_sat = sum(1 for c in self.variable_constraints if c.apply(ind))
                    if len(pred["constraints"]) > 0 and len(self.variable_constraints) > 0:
                        # Correlation between predictor satisfaction and variable satisfaction
                        pred_frac = pred_sat / len(pred["constraints"])
                        var_frac = var_sat / len(self.variable_constraints)
                        score += pred_frac * var_frac
            pred["score"] = score / max(total_in_archive, 1)

        self.predictors.sort(key=lambda p: p["score"], reverse=True)
        self.best_predictor = self.predictors[0]

    def _evolve_predictors(self):
        """Mutate predictors: add/remove/replace random constraints."""
        # Keep best half, replace worst half
        half = max(1, self.n_predictors // 2)
        new_preds = self.predictors[:half]
        for _ in range(self.n_predictors - half):
            parent = random.choice(new_preds)
            child_cons = list(parent["constraints"])
            # Mutation: add, remove, or replace a constraint
            r = random.random()
            if r < 0.33 and len(child_cons) < 6:
                child_cons.append(self.problem_space.get_rand_constraint())
            elif r < 0.66 and len(child_cons) > 1:
                child_cons.pop(random.randint(0, len(child_cons) - 1))
            elif child_cons:
                idx = random.randint(0, len(child_cons) - 1)
                child_cons[idx] = self.problem_space.get_rand_constraint()
            new_preds.append({"constraints": child_cons, "score": 0.0})
        self.predictors = new_preds

    def _select(self):
        total_f = sum(len(b) for b in self.bins)
        total_i = len(self.infeasible_pop)
        if total_f == 0 and total_i == 0:
            return self.problem_space.generate_random_individual()

        if total_f > 0 and (total_i == 0 or random.random() < (total_f * 2) / (total_f + total_i)):
            # Build selection pool with predictor bonus
            pool = []
            for b in self.bins:
                for fit, ind in b:
                    bonus = 1.0
                    if self.best_predictor["constraints"]:
                        pred_sat = sum(1 for c in self.best_predictor["constraints"] if c.apply(ind))
                        bonus = 1.0 + (pred_sat / len(self.best_predictor["constraints"])) * (self.predictor_bonus - 1.0)
                    pool.append((fit * bonus, ind))
            if not pool:
                return self.problem_space.generate_random_individual()
            return roulette_selection(pool)[1]
        return roulette_selection(self.infeasible_pop)[1]

    def run_one_generation(self, cons_changed):
        if cons_changed:
            self._evaluate_predictors()
            self._evolve_predictors()

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


ALGORITHM_CLASS = CoevolutionaryElites
ALGORITHM_NAME = "Coevolution-Elites"
ALGORITHM_KWARGS = {}
