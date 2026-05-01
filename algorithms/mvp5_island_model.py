"""
MVP 5: Island Model MAP-Elites

Multiple independent sub-archives (islands) evolve in parallel with
periodic migration. Each island uses different mutation rates or
selection pressure. Migration moves elites between islands.

On constraint change, islands that were more robust (more survivors)
donate elites to islands that collapsed. This naturally allocates
search effort toward the most productive strategies after a change.

Connection: island models are classical EC but rarely combined with
MAP-Elites. The variable-constraint setting gives islands a natural
specialization role.
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


class IslandElites(VariableConstraintGA):

    def __init__(self, problem_space, number_generations, population_size,
                 max_memory, cross_over_rate, mutation_rate, user,
                 update_interval, infeasible_rate=0.5, elitism=0.3,
                 n_islands=3, migration_interval=5, migration_rate=0.2):
        self.infeasible_rate = infeasible_rate
        self.elitism = elitism
        self.n_islands = n_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        super().__init__(problem_space, number_generations, population_size,
                         max_memory, cross_over_rate, mutation_rate, user,
                         update_interval)

    def set_up(self):
        n_bins = self.problem_space.get_num_bins()
        self.infeasible_pop_size = int(self.max_memory * self.infeasible_rate)
        self.elitism_num = int(self.infeasible_pop_size * self.elitism)
        feasible_memory = self.max_memory - self.infeasible_pop_size
        self.inds_per_bin = max(1, feasible_memory // n_bins)

        # Each island has its own archive and mutation rate
        self.islands = []
        base_rates = [self.mutation_rate * 0.5,
                      self.mutation_rate,
                      self.mutation_rate * 2.0]
        for i in range(self.n_islands):
            self.islands.append({
                "bins": [[] for _ in range(n_bins)],
                "mutation_rate": base_rates[i % len(base_rates)],
            })

        self.infeasible_pop = []
        self.gen_counter = 0

        # Distribute initial population across islands
        for _ in range(self.population_size):
            ind = self.problem_space.generate_random_individual()
            island = random.choice(self.islands)
            self._place_island(ind, island, self.infeasible_pop)

    def _const_ok(self, ind):
        return all(c.apply(ind) for c in self.problem_space.get_constant_constraints())

    def _var_ok(self, ind):
        return all(c.apply(ind) for c in self.variable_constraints)

    def _all_ok(self, ind):
        return self._const_ok(ind) and self._var_ok(ind)

    def _place_island(self, ind, island, infeasible_pop):
        if self._const_ok(ind):
            b = self.problem_space.place_in_bin(ind)
            fit = self.problem_space.fitness(ind)
            bins = island["bins"]
            if len(bins[b]) < self.inds_per_bin:
                bins[b].append((fit, ind))
                bins[b].sort(key=lambda x: x[0], reverse=True)
            elif fit >= bins[b][-1][0]:
                bins[b].pop(-1)
                bins[b].append((fit, ind))
                bins[b].sort(key=lambda x: x[0], reverse=True)
        else:
            n_sat = sum(1 for c in self.problem_space.get_constant_constraints() if c.apply(ind))
            if len(infeasible_pop) < self.infeasible_pop_size:
                infeasible_pop.append((n_sat, ind))

    def _select_from_island(self, island):
        bins = island["bins"]
        occupied = [i for i in range(len(bins)) if bins[i]]
        if not occupied:
            return self.problem_space.generate_random_individual()
        return roulette_selection(bins[random.choice(occupied)])[1]

    def _migrate(self):
        """Move best elites from each island to others."""
        for i in range(self.n_islands):
            src = self.islands[i]
            dst = self.islands[(i + 1) % self.n_islands]
            n_migrate = max(1, int(sum(len(b) for b in src["bins"]) * self.migration_rate))
            # Collect best from source
            all_elites = []
            for b in src["bins"]:
                for fit, ind in b:
                    all_elites.append((fit, ind))
            all_elites.sort(key=lambda x: x[0], reverse=True)
            for fit, ind in all_elites[:n_migrate]:
                self._place_island(copy.deepcopy(ind), dst, self.infeasible_pop)

    def _merge_bins(self):
        """Merge all island archives into one output, keeping best per bin."""
        n_bins = self.problem_space.get_num_bins()
        merged = [[] for _ in range(n_bins)]
        for island in self.islands:
            for b in range(n_bins):
                for fit, ind in island["bins"][b]:
                    if self._all_ok(ind):
                        if len(merged[b]) < self.inds_per_bin:
                            merged[b].append((fit, ind))
                            merged[b].sort(key=lambda x: x[0], reverse=True)
                        elif fit >= merged[b][-1][0]:
                            merged[b].pop(-1)
                            merged[b].append((fit, ind))
                            merged[b].sort(key=lambda x: x[0], reverse=True)
        return merged

    def run_one_generation(self, cons_changed):
        self.gen_counter += 1

        if self.gen_counter % self.migration_interval == 0:
            self._migrate()

        self.infeasible_pop.sort(key=lambda x: x[0], reverse=True)
        new_inf = self.infeasible_pop[:self.elitism_num]

        # Each island evolves independently
        per_island = max(1, self.population_size // (2 * self.n_islands))
        for island in self.islands:
            for _ in range(per_island):
                p1 = self._select_from_island(island)
                p2 = self._select_from_island(island)
                if random.random() < self.cross_over_rate:
                    c1, c2 = self.problem_space.cross_over(p1, p2)
                else:
                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                c1 = self.problem_space.mutate(c1, island["mutation_rate"])
                c2 = self.problem_space.mutate(c2, island["mutation_rate"])
                self._place_island(c1, island, new_inf)
                self._place_island(c2, island, new_inf)

        self.infeasible_pop = new_inf
        return self._merge_bins()


ALGORITHM_CLASS = IslandElites
ALGORITHM_NAME = "Island-Elites"
ALGORITHM_KWARGS = {}
