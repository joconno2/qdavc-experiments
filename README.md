# QDA-VC Experiments

Algorithms for the [QDA-VC Competition](https://qdavc.github.io/) at AIIDE 2026.

**Task:** Quality-Diversity under Variable Constraints. Maintain archive quality as constraints shift between personas (Exploratory, Cycle, Adaptive, Strict) on two domains (TTP, LogicPuzzles).

**Competition timeline:** Submissions Aug 1 - Sep 8, 2026. Results Nov 13.

**Competition entry:** `UH-nobandit` (Shuffling base + adaptive mutation rate). See below.

---

## Competition Entry: UH-nobandit (v2)

**Implementation:** `algorithms/mvp22_ultimate_hybrid.py`

Shuffling-based population management + Rechenberg's 1/5th success rule for adaptive mutation rate.

**How it works:**

1. **Shuffling base.** On constraint change, all individuals (feasible + infeasible) are collected, a fresh random population is injected, and everything is re-placed against the new constraints. This is the framework's Shuffling baseline, which turned out to be the strongest baseline by a wide margin.

2. **Adaptive mutation rate** via the 1/5th success rule (Rechenberg, 1973). If more than 20% of offspring enter the archive, increase the mutation rate (too conservative). If fewer than 20%, decrease it (too aggressive). The rate is **never reset** on constraint change. The rule self-corrects naturally: when constraints shift and bins empty, the success rate rises (easy to improve empty bins), which pushes the rate up automatically. This implicit adaptation is the core contribution.

**What it does NOT include:**
- No eviction pool (Shuffling's reshuffle handles displaced solutions)
- No stagnation restart (hurts when combined with reshuffling)
- No bandit operator selection (counterproductive with adaptive rate)
- No explicit constraint change detection (1/5th rule provides implicit detection)

---

## Experimental History

This project involved four major experimental phases, each building on findings from the previous. Total: **17,320+ experiments** across 54+ algorithm conditions, 2 domains, 4 personas, and 30-100 seeds per condition.

### Phase 1: Wave 1 (31 algorithms, 5,580 tasks)

Tested 10 new algorithm variants against the VC-MAP-Elites baseline. Parameters: 100 gens, pop 50, memory 200, 3 personas (Exploratory, Cycle, Adaptive), 30 seeds.

**New algorithms tested:**

| # | Algorithm | Mechanism | Result |
|---|-----------|-----------|--------|
| 9 | Bandit+Evict | UCB1 bandit + eviction/restart as background | +35% |
| 10 | Constraint-Memory | Hash constraint configs, recall on repeat | -13% |
| 11 | DE-Elites | Differential evolution crossover chain | +23% |
| 12 | Novelty-Selection | Parent selection biased toward empty bins | **-71%** |
| 13 | Adaptive-Rate | 1/5th success rule, self-tuning mutation | +3% |
| 14 | Sliding-Window | Top-K per bin, revalidate on change | **-49%** |
| 15 | Thompson-Bandit | Beta-posterior expert selection | +47% |
| 16 | Crossover-Primary | Always crossover, small post-mutation | +12% |
| 17 | Age-Weighted | Old individuals easier to replace | +7% |
| 18 | Epsilon-Bandit | Decaying epsilon-greedy experts | +49% |

Plus 9 ablation variants (EGGROLL-nodir, EvRst-noevict, EvRst-nostag, BdEvict-noevict/nostag/nobandit, Memory-nomemory, Novelty-none, AdaptRate-noreset, SlidingK1/K5, Age-nopen).

**Key finding:** `AdaptRate-noreset` (+95%) was the runaway winner. NOT resetting the mutation rate on constraint change was the single biggest improvement. The 1/5th rule self-corrects naturally because empty bins = high success rate = rate increases automatically.

**Other findings:**
- Stagnation restart actively hurts when eviction is present
- Simpler bandits (epsilon +49%, Thompson +47%) beat UCB1 (+10%)
- Bandit operator selection is counterproductive with adaptive rate
- Constraint memory, novelty selection, and sliding window all hurt

### Phase 2: Wave 2 (23 algorithms, 4,140 tasks)

Built on wave 1 findings. Tested hybrids and parameter sweeps.

**Key result:** `UH-nobandit` (+106%) combined adaptive rate + eviction pool + NO bandit. The bandit was counterproductive (UH-nobandit 66,945 >> Ultimate-Hybrid with bandit 47,413). The adaptive rate already handles exploration/exploitation, making operator selection redundant.

**Parameter sweeps:**
- UCB c sweep: c=2.0 best for BdEvict-nostag
- Adapt factor sweep: f=1.1 slightly better than f=1.2
- Target success sweep: t=0.5 marginally better than t=0.2
- Direction history: n=3 slightly better than n=5 for Bandit

### Phase 3: Validation (100-seed confirmation + official params)

**100-seed confirmation (3,000 tasks):** UH-nobandit and BdEvict-nostag tied at +70%. Both significant on Adaptive (p<0.001) and Cycle (p<0.001). Algorithms without eviction pool significantly worse than baseline on Adaptive.

**Official params validation (900 tasks):** With official parameters (300 gens, pop 200, memory 500, interval 50), UH-nobandit scored 128K total QD (+85% vs baseline), clear #1 with 22K gap to #2.

### Phase 4: Comprehensive Evaluation (3,600 tasks)

Tested all algorithms against ALL framework baselines (VC-MAP-Elites, Filtering, RandomRestarts, Shuffling) on ALL personas (including Strict, never previously tested). Official parameters.

**Critical discovery:** Shuffling was the strongest baseline (145K total QD), and all our algorithms scored **zero on Strict persona** due to a placement bug. Our `_place` method gated on constant constraint satisfaction, but on TTP instances with 150 items, 0% of random individuals pass constant constraints. The framework baselines handle this by maintaining infeasible populations that evolve toward feasibility; our algorithms did not.

**The fix (v2):** Rewrote UH-nobandit to use Shuffling's population management as the base. All individuals (feasible + infeasible) participate in evolution. Fresh random individuals are injected on constraint change. Only the adaptive mutation rate is added on top. This matches the framework's approach while adding the 1/5th rule.

**Comprehensive v2 rerun in progress** (3,600 tasks, official params, all 4 personas).

---

## Key Findings

### What works

1. **Adaptive mutation rate (1/5th rule) without reset.** The single biggest improvement across all experiments. The rule provides implicit constraint change detection: when constraints shift and many bins empty, the success rate rises, automatically increasing the mutation rate. No explicit detection mechanism needed. (+95% on 3 personas at small params, dominant on Cycle at all scales)

2. **Shuffling-based population management.** Keeping all individuals across constraint changes and re-classifying against new constraints is stronger than eviction pools, random restarts, or filtering. Shuffling was the strongest framework baseline and the only algorithm to score on Strict persona.

3. **Simplicity over complexity.** Two orthogonal mechanisms (Shuffling + adaptive rate) outperform complex hybrids. Adding bandit selection, constraint memory, novelty bias, or sliding windows to the base all hurt performance.

### What doesn't work

| Mechanism | Result | Why |
|-----------|--------|-----|
| **Bandit operator selection** | Counterproductive with adaptive rate | Rate already handles exploration; bandit over-diversifies |
| **Constraint Memory** | -13% | Exact hash matching rarely finds repeated constraint configs |
| **Novelty Selection** | -71% | Biasing toward empty bins pulls resources from productive search |
| **Sliding Window** | -49% | Extra storage per bin dilutes archive without benefit |
| **Island Model** | -84% | Splits archive, prevents coordinated response to changes |
| **Stagnation Restart** | Hurts with eviction/shuffling | Wipes progress that reshuffling manages |
| **Lamarckian** | 0% improvement | Local search adds cost without improving QD |
| **Coevolution** | -30% | Constraint predictors don't generalize |

### Critical lesson: test against ALL baselines

Our initial waves compared only against VC-MAP-Elites. Shuffling (a simpler framework baseline) was 45% stronger and exposed a fundamental flaw in our placement logic. The placement bug (gating on constant constraints) caused zero scores on 3 of 6 TTP problem instances and the entire Strict persona. This would not have been caught without testing against all baselines.

---

## Algorithms

### Framework Baselines

| Algorithm | File | Description |
|-----------|------|-------------|
| VC-MAP-Elites | `framework/Algorithms/VCMapElites.py` | 2D grid (diversity x constraint satisfaction) |
| Shuffling | `framework/Algorithms/Shuffling.py` | Re-sort all individuals on constraint change |
| RandomRestarts | `framework/Algorithms/RandomRestarts.py` | Dump population on constraint change |
| Filtering | `framework/Algorithms/Filtering.py` | Ignore variable constraints, filter at output |

### Our Algorithms (MVP 1-22)

| # | File | Method |
|---|------|--------|
| 1 | `mvp1_lamarckian.py` | Local search before archive insertion |
| 2 | `mvp2_coevolution.py` | Coevolved constraint predictors |
| 3 | `mvp3_scope_compressed.py` | Multi-step small mutations (SCOPE-inspired) |
| 4 | `mvp4_eggroll_lowrank.py` | Directed mutation from successful parents |
| 5 | `mvp5_island_model.py` | Multiple sub-archives with migration |
| 6 | `mvp6_evict_restart.py` | Eviction pool + stagnation restart |
| 7 | `mvp7_bandit_experts.py` | UCB1 bandit over 4 mutation experts |
| 8 | `ablation_winners.py` | EGGROLL + Evict-Restart combined |
| 9 | `mvp9_bandit_evict.py` | Bandit experts + eviction background |
| 10 | `mvp10_constraint_memory.py` | Hash constraint configs, recall solutions |
| 11 | `mvp11_de_elites.py` | DE/rand/1 crossover chain mutation |
| 12 | `mvp12_novelty_selection.py` | Parent selection biased toward empty bins |
| 13 | `mvp13_adaptive_rate.py` | 1/5th success rule, self-tuning mutation |
| 14 | `mvp14_sliding_window.py` | Top-K per bin, revalidate on change |
| 15 | `mvp15_thompson_bandit.py` | Thompson Sampling expert selection |
| 16 | `mvp16_crossover_primary.py` | Always crossover, small post-mutation |
| 17 | `mvp17_age_weighted.py` | Age-based replacement pressure |
| 18 | `mvp18_epsilon_bandit.py` | Decaying epsilon-greedy expert selection |
| 19 | `mvp19_bandit_ucb_sweep.py` | UCB c parameter sweep wrapper |
| 20 | `mvp20_bandit_evict_memory.py` | Bandit + eviction + constraint memory |
| 21 | `mvp21_bandit_5expert.py` | 5 experts (adds SCOPE multi-step) |
| **22** | **`mvp22_ultimate_hybrid.py`** | **Shuffling base + adaptive rate (competition entry)** |

---

## Running

```bash
# Comprehensive test: all algorithms, all baselines, all personas, official params
# 15 algorithms x 2 domains x 4 personas x 30 seeds = 3,600 tasks (~3h on 64 cores)
python run_comprehensive.py

# Analysis (tables, stats, mechanism matrix)
python analyze_results.py results/comprehensive.json

# Earlier waves (3-persona, smaller params)
python run_full_ablation.py    # Wave 1: 31 conditions, 5,580 tasks
python run_wave2.py            # Wave 2: 23 conditions, 4,140 tasks
python run_100seed.py          # 100-seed confirmation on top 5
python run_official_params.py  # Official params, 6 algorithms

# Instrumented runs (per-generation traces for figures)
python run_instrumented.py           # Small params, generates figures
python run_instrumented_official.py  # Official params, generates figures
```

Results in `results/`. Figures in `results/figures/` and `results/figures_official/`.

---

## Related Work

Full positioning notes at `~/Documents/Work/qdavc_related_work_notes.md`. Key references:

- **CMA-MAE** (Fontaine & Nikolaidis, GECCO 2023): Adapts archive acceptance thresholds via learning rate. Our approach adapts mutation intensity via success feedback. CMA-MAE uses a predetermined annealing schedule; the 1/5th rule is self-correcting.
- **Multi-Emitter MAP-Elites** (Cully, GECCO 2021): UCB bandit across emitter types. Our ablations show bandits are counterproductive with adaptive rate.
- **Dynamic QD** (Gallotta et al., GECCO 2024 poster): Handles changing fitness/behavior functions via archive re-evaluation. QDA-VC is a distinct problem (changing constraint feasibility). The 1/5th rule provides implicit change detection.
- **Rechenberg's 1/5th rule** (1973): Original self-adaptive step size for ES. To our knowledge, this is the first application to MAP-Elites / QD.
- **Adaptive Operator Selection** (Fialho et al., 2008-2010): DMAB/Compass select among discrete operators. Our approach adapts a continuous parameter with one configuration knob. AOS requires tuning its own meta-parameters.

**Novelty claims:**
1. First application of the 1/5th success rule to MAP-Elites / QD
2. First study of QD under variable (dynamic) constraints
3. Empirical demonstration that adaptive rate makes bandit operator selection redundant
4. Finding that NOT resetting mutation rate on constraint change is optimal
5. Shuffling as the strongest baseline for variable-constraint QD (previously unreported)

---

## Known Issues

- **MVPs 1-18 have a placement bug:** `_place` gates on constant constraint satisfaction, causing zero scores on TTP instances where random individuals can't reach the feasible region. Only MVP 22 (v2) is fixed. These MVPs' results are valid for relative comparison at small params (where the bug is less severe) but not at official params on all problem instances.
- **LodeRunner domain is empty** in the framework. Only TTP and LogicPuzzles are available.
- **Strict persona scoring varies significantly** by TTP problem instance due to constraint accumulation dynamics.
- **LogicPuzzles scores are 100-1000x lower than TTP** across all algorithms. May need domain-specific operators.

---

## Project Timeline

| Date | Event |
|------|-------|
| Apr 29 | QDA-VC competition discovered, initial MVPs 1-8 |
| May 2 (AM) | Wave 1: 31 algorithms, 5,580 tasks. AdaptRate-noreset champion (+95%) |
| May 2 (AM) | Wave 2: 23 algorithms, 4,140 tasks. UH-nobandit champion (+106%) |
| May 2 (PM) | 100-seed confirmation (3,000 tasks). Official params validation (900 tasks) |
| May 2 (PM) | Comprehensive eval: discovered Shuffling baseline is strongest, placement bug on Strict |
| May 2 (night) | Fixed placement (v2): Shuffling base + adaptive rate. Comprehensive v2 rerun in progress |
| Aug 1-Sep 8 | Competition submission window |
| Nov 13 | Results at AIIDE 2026 |
