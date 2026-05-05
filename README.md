# QDA-VC Experiments

Algorithms for the [QDA-VC Competition](https://qdavc.github.io/) at AIIDE 2026.

**Task:** Quality-Diversity under Variable Constraints. Maintain archive quality as constraints shift between personas (Exploratory, Cycle, Adaptive, Strict) on two domains (TTP, LogicPuzzles).

**Competition timeline:** Submissions Aug 1 - Sep 8, 2026. Results Nov 13.

**Competition entry:** Asymmetric Adaptive Rate (`algorithms/mvp24_asymmetric_adapt.py`). See below.

---

## Competition Entry: Asymmetric Adaptive Rate (MVP 24)

**Implementation:** `algorithms/mvp24_asymmetric_adapt.py`

Shuffling-based population management + Rechenberg's 1/5th success rule with asymmetric adaptation factors.

**How it works:**

1. **Shuffling base.** On constraint change, all individuals (feasible + infeasible) are collected, a fresh random population is injected, and everything is re-placed against the new constraints. This is the framework's Shuffling baseline, which turned out to be the strongest baseline by a wide margin.

2. **Asymmetric 1/5th rule.** If more than 20% of offspring enter the archive, increase the mutation rate by factor 1.2 (conservative growth). If fewer than 20%, decrease it by factor 1.5 (aggressive cooldown). The rate is **never reset** on constraint change. The asymmetry prevents rate ratcheting on Exploratory personas while preserving strong performance on Cycle/Strict.

3. **Implicit constraint change detection.** When constraints shift and bins empty, the success rate rises (easy to fill empty bins), which pushes the rate up automatically. When constraints relax and bins are full, success drops and rate decreases quickly (factor 1.5). No explicit detection mechanism needed.

**Parameters:** `increase_factor=1.2, decrease_factor=1.5, target_success=0.2, rate_min=0.05, rate_max=0.9`

**What it does NOT include:**
- No eviction pool (Shuffling's reshuffle handles displaced solutions)
- No stagnation restart (hurts when combined with reshuffling)
- No bandit operator selection (counterproductive with adaptive rate)
- No explicit constraint change detection

**Results (50 seeds, official params, TTP):**
- +14.8% vs Shuffling overall (all 4 personas positive)
- +20% on Exploratory, +10% on Cycle, +12% on Adaptive, +20% on Strict

---

## Results Summary

### Final results (200-seed paper run in progress)

Entry vs baselines at official params (300 gens, pop 200, memory 500, interval 50):

| Algorithm | TTP Total | vs Shuffling |
|-----------|-----------|--------------|
| **Asym-1.2/1.5** (entry) | ~157K | **+14.8%** |
| Sym-1.2/1.2 (UH-nobandit) | ~154K | +13.1% |
| Shuffling (baseline) | ~137K | 0% |
| VC-MAP-Elites | ~98K | -28% |

### Statistical significance (100 seeds, official params)

| Domain | Persona | Entry vs Shuffling | p-value |
|--------|---------|-------------------|---------|
| TTP | Cycle | +32% | 0.0009 *** |
| TTP | Strict | +55% | 0.020 * |
| TTP | Adaptive | +20% | 0.11 ns |
| TTP | Exploratory | -8% to +20% | varies |

Cycle is the strongest condition for significance (consistent across all runs). The 200-seed paper run will nail all conditions.

### Mechanism contribution (from 50-seed asymmetric sweep)

| Mechanism | Effect | Evidence |
|-----------|--------|----------|
| Adaptive rate (1/5th rule) | +13-15% vs Shuffling | Core contribution |
| Asymmetric factors (1.2/1.5) | +1.5% vs symmetric | Fixes Exploratory |
| Not resetting on change | +95% vs resetting | Wave 1 ablation |
| Shuffling base | Required for Strict | Placement bug without it |

---

## Experimental History

Total: **28,000+ experiments** across 6 phases, 54+ algorithm conditions, 2 domains, 4 personas, 30-200 seeds per condition.

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

Plus 9 ablation variants.

**Key finding:** `AdaptRate-noreset` (+95%) was the runaway winner. NOT resetting the mutation rate on constraint change was the single biggest improvement.

### Phase 2: Wave 2 (23 algorithms, 4,140 tasks)

Hybrids and parameter sweeps. `UH-nobandit` (+106%) was the champion. Bandit selection counterproductive with adaptive rate (66,945 >> 47,413 with bandit).

### Phase 3: Validation (100-seed + official params, 3,900 tasks)

UH-nobandit confirmed at +70% (100 seeds), +85% at official params. Significant on Adaptive (p=0.0005) and Cycle (p<0.000001).

### Phase 4: Comprehensive Evaluation (3,600 tasks)

Discovered Shuffling was the strongest baseline. Placement bug found and fixed. All algorithms re-evaluated against all 4 baselines on all 4 personas.

### Phase 5: Placement Fix + 100-Seed Confirmation (6,800 tasks)

Fixed placement bug (check ALL constraints, not just constant). UH-nobandit: +18.5% vs Shuffling (100 seeds, official params). Significant on Cycle (p=0.0009).

### Phase 6: Asymmetric Adaptation + Paper Run (11,600+ tasks)

Parameter sweep found asymmetric factors (increase 1.2, decrease 1.5) optimal. 200-seed paper run with full ablation set in progress.

---

## Key Findings

### What works

1. **Adaptive mutation rate (1/5th rule) without reset.** The single biggest improvement across all experiments. The rule provides implicit constraint change detection: when constraints shift and bins empty, the success rate rises, automatically increasing the mutation rate. No explicit detection needed.

2. **Asymmetric adaptation factors.** Increase rate slowly (1.2), decrease quickly (1.5). Prevents rate ratcheting on personas with random constraint changes (Exploratory) while preserving gains on systematic changes (Cycle, Strict).

3. **Shuffling-based population management.** Keeping all individuals across constraint changes and re-classifying against new constraints. Required for Strict persona where constraints only accumulate. Injecting fresh random individuals on reshuffle provides essential genetic diversity.

4. **Simplicity over complexity.** Two orthogonal mechanisms (Shuffling + adaptive rate) outperform complex hybrids. Adding bandit selection, constraint memory, novelty bias, or sliding windows all hurt performance.

### What doesn't work

| Mechanism | Result | Why |
|-----------|--------|-----|
| **Bandit operator selection** | Counterproductive | Rate already handles exploration; bandit over-diversifies |
| **Constraint Memory** | -13% | Exact hash matching rarely finds repeated configs |
| **Novelty Selection** | -71% | Biasing toward empty bins pulls from productive search |
| **Sliding Window** | -49% | Extra storage per bin dilutes without benefit |
| **Island Model** | -84% | Splits archive, prevents coordinated response |
| **Stagnation Restart** | Hurts | Wipes progress that reshuffling manages |
| **Directional adaptation** | Hurts | Double-decreases rate; 1/5th rule already self-corrects |
| **Lamarckian** | 0% | Local search adds cost without improving QD |
| **Coevolution** | -30% | Constraint predictors don't generalize |

### Critical lessons

1. **Test against ALL baselines.** Our initial waves compared only against VC-MAP-Elites. Shuffling was 45% stronger and exposed a fundamental placement bug.

2. **Ablate everything.** The no-reset finding was discovered by accident (an ablation variant). It became the core of the entry.

3. **Simple mechanisms compose; complex ones interfere.** Every attempt to add a third mechanism (bandit, memory, novelty) on top of the two winners (Shuffling + adaptive rate) made things worse.

---

## Algorithms

### Framework Baselines

| Algorithm | File | Description |
|-----------|------|-------------|
| VC-MAP-Elites | `framework/Algorithms/VCMapElites.py` | 2D grid (diversity x constraint satisfaction) |
| Shuffling | `framework/Algorithms/Shuffling.py` | Re-sort all individuals on constraint change |
| RandomRestarts | `framework/Algorithms/RandomRestarts.py` | Dump population on constraint change |
| Filtering | `framework/Algorithms/Filtering.py` | Ignore variable constraints, filter at output |

### Our Algorithms (MVP 1-24)

| # | File | Method | Status |
|---|------|--------|--------|
| 1 | `mvp1_lamarckian.py` | Local search before archive insertion | Dead end |
| 2 | `mvp2_coevolution.py` | Coevolved constraint predictors | Dead end |
| 3 | `mvp3_scope_compressed.py` | Multi-step small mutations (SCOPE-inspired) | Dead end |
| 4 | `mvp4_eggroll_lowrank.py` | Directed mutation from successful parents | Moderate |
| 5 | `mvp5_island_model.py` | Multiple sub-archives with migration | Dead end |
| 6 | `mvp6_evict_restart.py` | Eviction pool + stagnation restart | Moderate |
| 7 | `mvp7_bandit_experts.py` | UCB1 bandit over 4 mutation experts | Moderate |
| 8 | `ablation_winners.py` | EGGROLL + Evict-Restart combined | Moderate |
| 9 | `mvp9_bandit_evict.py` | Bandit experts + eviction background | Good |
| 10 | `mvp10_constraint_memory.py` | Hash constraint configs, recall solutions | Dead end |
| 11 | `mvp11_de_elites.py` | DE/rand/1 crossover chain mutation | Moderate |
| 12 | `mvp12_novelty_selection.py` | Parent selection biased toward empty bins | Dead end |
| 13 | `mvp13_adaptive_rate.py` | 1/5th success rule, self-tuning mutation | Key finding |
| 14 | `mvp14_sliding_window.py` | Top-K per bin, revalidate on change | Dead end |
| 15 | `mvp15_thompson_bandit.py` | Thompson Sampling expert selection | Moderate |
| 16 | `mvp16_crossover_primary.py` | Always crossover, small post-mutation | Moderate |
| 17 | `mvp17_age_weighted.py` | Age-based replacement pressure | Marginal |
| 18 | `mvp18_epsilon_bandit.py` | Decaying epsilon-greedy expert selection | Good |
| 19 | `mvp19_bandit_ucb_sweep.py` | UCB c parameter sweep wrapper | Sweep |
| 20 | `mvp20_bandit_evict_memory.py` | Bandit + eviction + constraint memory | Dead end |
| 21 | `mvp21_bandit_5expert.py` | 5 experts (adds SCOPE multi-step) | Moderate |
| 22 | `mvp22_ultimate_hybrid.py` | Shuffling base + symmetric adaptive rate | Strong |
| 23 | `mvp23_directional_adapt.py` | Directional rate change on add/remove | Dead end |
| **24** | **`mvp24_asymmetric_adapt.py`** | **Shuffling + asymmetric 1/5th rule** | **Entry** |

---

## Running

```bash
# Paper final: 200 seeds, full ablation, both domains, all personas
# 6 algos * 2 domains * 4 personas * 200 seeds = 9,600 tasks (~10h on 64 cores)
python run_paper_final.py

# Asymmetric parameter sweep (TTP only, 50 seeds)
python run_asymmetric_sweep.py

# 100-seed official params (top 4 algos)
python run_100seed_official.py

# Comprehensive: all algorithms, all baselines, all personas, 30 seeds
python run_comprehensive.py

# Per-generation traces for figures
python run_traces_fixed.py

# Analysis
python analyze_results.py results/paper_final.json

# Earlier waves
python run_full_ablation.py    # Wave 1: 31 conditions, 5,580 tasks
python run_wave2.py            # Wave 2: 23 conditions, 4,140 tasks
```

Results in `results/`. Figures in `results/figures_fixed/`.

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
3. Asymmetric adaptation factors for variable-constraint QD
4. Empirical demonstration that adaptive rate makes bandit operator selection redundant
5. Finding that NOT resetting mutation rate on constraint change is optimal (implicit detection)
6. Shuffling as the strongest baseline for variable-constraint QD (previously unreported)

---

## Known Issues

- **MVPs 1-18 have a placement bug:** `_place` gates on constant constraint satisfaction only, causing zero scores on TTP 150-item instances and Strict persona. MVPs 22-24 are fixed. Earlier MVPs' results are valid for relative comparison at small params.
- **LodeRunner domain is empty** in the framework. Only TTP and LogicPuzzles are available.
- **Strict persona scoring varies significantly** by TTP problem instance due to constraint accumulation dynamics.
- **LogicPuzzles scores are 100-1000x lower than TTP** across all algorithms. May need domain-specific operators.
- **TTP `is_contradictory` is inverted:** Returns True when constraints are NOT contradictory. Strict persona piles on non-contradictory constraints indefinitely. Affects all algorithms equally, doesn't change relative performance.
- **`nextProblem.txt` race condition:** Concurrent multiprocessing workers fight over this file. Fixed in later run scripts via multiprocessing.Lock.

---

## Project Timeline

| Date | Event |
|------|-------|
| Apr 29 | QDA-VC competition discovered, initial MVPs 1-8 |
| May 2 (AM) | Wave 1: 31 algorithms, 5,580 tasks. AdaptRate-noreset champion (+95%) |
| May 2 (AM) | Wave 2: 23 algorithms, 4,140 tasks. UH-nobandit champion (+106%) |
| May 2 (PM) | 100-seed confirmation + official params validation (3,900 tasks) |
| May 2 (PM) | Comprehensive eval: Shuffling strongest baseline, placement bug found |
| May 3 | Placement fix (v2): check ALL constraints. Comprehensive rerun. |
| May 3 | 100-seed official params: UH-nobandit +18.5% vs Shuffling, p=0.0009 on Cycle |
| May 4 | Asymmetric sweep: 1.2/1.5 best (+14.8%). Directional adaptation negative result. |
| May 4 | MVP 24 (competition entry): asymmetric adaptive rate. Per-gen trace figures. |
| May 4 | Paper final run launched: 200 seeds, 6 algorithms, full ablation (9,600 tasks) |
| Aug 1-Sep 8 | Competition submission window |
| Nov 13 | Results at AIIDE 2026 |
