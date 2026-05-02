# QDA-VC Experiments

Algorithms for the [QDA-VC Competition](https://qdavc.github.io/) at AIIDE 2026.

**Task:** Quality-Diversity under Variable Constraints. Maintain archive quality as constraints shift between personas (Exploratory, Cycle, Adaptive) on two domains (TTP, LogicPuzzles).

**Competition timeline:** Submissions Aug 1 - Sep 8, 2026. Results Nov 13.

**Competition entry:** `UH-nobandit` (Adaptive Rate + Eviction Pool). See below.

---

## Results (30 seeds, Mann-Whitney U, 2 waves, 54 conditions, 9,720 tasks)

### Competition Entry: UH-nobandit

Two non-interfering mechanisms:
1. **Adaptive mutation rate** via 1/5th success rule (Rechenberg, 1973). NO reset on constraint change. The rule self-corrects: when bins empty after a constraint shift, success rate rises, rate increases automatically.
2. **Eviction pool** stores solutions displaced by constraint changes; re-inserts them when they become feasible again. NO stagnation restart (it hurts).

Implementation: `algorithms/mvp22_ultimate_hybrid.py` with `use_bandit=False`.

### TTP Domain (Mean QD Score, 30 seeds)

| Algorithm | Exploratory | Cycle | Adaptive | Total |
|-----------|----------:|------:|--------:|------:|
| **UH-nobandit** | 7,953 | **38,310** | **20,601** | **66,863** |
| AdaptRate-noreset | **12,828** | 38,243 | 6,854 | 57,924 |
| BdEvict-nostag | 11,595 | 24,380 | 20,830 | 56,805 |
| Epsilon-Bandit | 7,820 | 34,645 | 1,911 | 44,376 |
| Thompson-Bandit | 9,783 | 31,898 | 1,974 | 43,654 |
| EvRst-nostag | 10,442 | 23,380 | 9,669 | 43,491 |
| Bandit+Evict | 7,140 | 23,525 | 9,360 | 40,025 |
| EGGROLL | 11,330 | 23,294 | 2,754 | 37,378 |
| DE-Elites | 9,223 | 25,688 | 1,583 | 36,494 |
| Bandit (UCB1) | 4,587 | 26,487 | 1,727 | 32,801 |
| Evict-Restart | 6,420 | 15,263 | 10,823 | 32,506 |
| Baseline | 7,389 | 14,907 | 7,439 | 29,735 |
| Novelty-Selection | 2,165 | 6,309 | 227 | 8,700 |
| Island | 815 | 3,952 | 25 | 4,793 |

### Statistical Significance: UH-nobandit vs Baseline

| Domain | Persona | UH-nobandit | Baseline | Diff | p-value | Cohen's d |
|--------|---------|----------:|--------:|-----:|--------:|----------:|
| TTP | Exploratory | 7,953 | 8,222 | -270 | 0.855 | -0.02 |
| TTP | **Cycle** | **38,310** | 17,013 | **+21,297** | **0.001** | **+1.07** |
| TTP | **Adaptive** | **20,601** | 7,261 | **+13,340** | **0.010** | **+0.84** |
| LogicPuzzles | Exploratory | 13 | 10 | +3 | 0.682 | +0.15 |
| LogicPuzzles | **Cycle** | **47** | 35 | **+13** | **<0.001** | **+2.09** |
| LogicPuzzles | Adaptive | 22 | 24 | -2 | 0.830 | -0.16 |

Significant improvement on Cycle (both domains, p<=0.001, d>1.0) and TTP Adaptive (p=0.010, d=0.84). Neutral on Exploratory.

### UH-nobandit vs Previous Best Algorithms (TTP)

| Comparison | Persona | UH-nobandit | Rival | p-value | Sig |
|------------|---------|----------:|------:|--------:|-----|
| vs Evict-Restart | Cycle | 38,310 | 15,263 | <0.001 | *** |
| vs Evict-Restart | Adaptive | 20,601 | 10,823 | 0.040 | * |
| vs EGGROLL | Cycle | 38,310 | 23,294 | 0.032 | * |
| vs EGGROLL | Adaptive | 20,601 | 2,754 | <0.001 | *** |
| vs Bandit (UCB1) | Adaptive | 20,601 | 1,727 | <0.001 | *** |
| vs BdEvict-nostag | Cycle | 38,310 | 24,380 | 0.016 | * |

### Grand Totals (TTP + LogicPuzzles, all personas)

| Rank | Algorithm | Total QD | vs Baseline |
|------|-----------|-------:|:-----------:|
| 1 | **UH-nobandit** | **66,945** | **+106%** |
| 2 | AdaptRate-noreset | 57,984 | +95% |
| 3 | BdEvict-nostag | 56,888 | +91% |
| 4 | AR-f1.1 (wave 2) | 52,864 | +62% |
| 5 | AR-t0.5 (wave 2) | 52,545 | +61% |
| 6 | Epsilon-Bandit | 44,440 | +49% |
| 7 | Thompson-Bandit | 43,718 | +47% |
| 8 | EvRst-nostag | 43,569 | +46% |
| 9 | Bandit+Evict | 40,119 | +35% |
| 10 | EGGROLL | 37,440 | +26% |
| 11 | DE-Elites | 36,560 | +23% |
| 12 | Bandit (UCB1) | 32,869 | +10% |
| 13 | Evict-Restart | 32,596 | +9% |
| 14 | Baseline (VC-MAP-Elites) | 29,803 | -- |
| -- | Novelty-Selection | 8,755 | -71% |
| -- | Island | 4,860 | -84% |

---

## Key Findings

1. **Adaptive mutation rate without reset is the single biggest improvement.** The 1/5th success rule self-corrects after constraint changes: empty bins mean high success rate, which pushes the rate up automatically. Forcing a reset disrupts this natural adaptation. (AdaptRate-noreset +95% vs Adaptive-Rate +3%)

2. **Eviction pool is critical for Adaptive persona.** Storing displaced solutions and re-inserting when feasible handles the forward-backward constraint pattern. Without eviction, Adaptive scores drop by 6,600+ (p<0.001).

3. **Stagnation restart hurts.** When eviction is present, stagnation restart wipes progress. BdEvict-nostag >> Bandit+Evict across all conditions.

4. **Bandit mutation selection hurts when combined with adaptive rate.** The rate already handles exploration/exploitation; adding a bandit over-diversifies. UH-nobandit (66,945) >> Ultimate-Hybrid with bandit (47,413).

5. **Simpler bandits beat UCB1.** Epsilon-greedy (+49%) and Thompson Sampling (+47%) both outperform UCB1 (+10%). UCB1 over-explores in the non-stationary constraint environment.

6. **Combined mechanisms interfere unless carefully decoupled.** EGGROLL + Evict-Restart failed (-5.5%). Bandit + Evict + Adaptive-Rate also underperformed. The winning approach: two orthogonal mechanisms (rate adaptation + archive management) that don't share state.

7. **DE-style mutation works.** Differential evolution crossover chain (+23%) provides a different exploration geometry that complements standard mutation.

### What Failed

- **Constraint Memory** (-13%): Exact hash matching rarely finds repeated constraint configs.
- **Novelty Selection** (-71%): Biasing toward empty bins pulls resources from productive search.
- **Sliding Window** (-49%): Extra storage per bin dilutes the archive.
- **Island Model** (-84%): Splits the archive, prevents coordinated response to constraint changes.
- **Lamarckian** (-14%): Local search adds cost without improving QD.
- **Coevolution** (-30%): Constraint predictors don't generalize.

---

## Algorithms

### Wave 1 (original + 10 new)

| # | File | Method | Total QD |
|---|------|--------|-------:|
| 1 | `mvp1_lamarckian.py` | Local search before archive insertion | 25,751 |
| 2 | `mvp2_coevolution.py` | Coevolved constraint predictors | 20,781 |
| 3 | `mvp3_scope_compressed.py` | Multi-step small mutations (SCOPE-inspired) | 25,460 |
| 4 | `mvp4_eggroll_lowrank.py` | Directed mutation from successful parents | 37,440 |
| 5 | `mvp5_island_model.py` | Multiple sub-archives with migration | 4,860 |
| 6 | `mvp6_evict_restart.py` | Eviction pool + stagnation restart | 32,596 |
| 7 | `mvp7_bandit_experts.py` | UCB1 bandit over 4 mutation experts | 32,869 |
| 8 | `ablation_winners.py` | EGGROLL + Evict-Restart combined | 28,168 |
| 9 | `mvp9_bandit_evict.py` | Bandit experts + eviction background | 40,119 |
| 10 | `mvp10_constraint_memory.py` | Hash constraint configs, recall solutions | 26,054 |
| 11 | `mvp11_de_elites.py` | DE/rand/1 crossover chain mutation | 36,560 |
| 12 | `mvp12_novelty_selection.py` | Parent selection biased toward empty bins | 8,755 |
| 13 | `mvp13_adaptive_rate.py` | 1/5th success rule, self-tuning mutation | 30,778 |
| 14 | `mvp14_sliding_window.py` | Top-K per bin, revalidate on change | 15,288 |
| 15 | `mvp15_thompson_bandit.py` | Thompson Sampling expert selection | 43,718 |
| 16 | `mvp16_crossover_primary.py` | Always crossover, small post-mutation | 33,410 |
| 17 | `mvp17_age_weighted.py` | Age-based replacement pressure | 31,939 |
| 18 | `mvp18_epsilon_bandit.py` | Decaying epsilon-greedy expert selection | 44,440 |

### Wave 2 (hybrids + parameter sweeps)

| # | File | Method | Total QD |
|---|------|--------|-------:|
| 22 | `mvp22_ultimate_hybrid.py` | Adaptive rate + eviction + epsilon bandit | 47,413 |
| 22 | `mvp22_ultimate_hybrid.py` (use_bandit=False) | **Adaptive rate + eviction only** | **66,945** |
| 21 | `mvp21_bandit_5expert.py` | 5 experts (adds SCOPE multi-step) | 44,525 |

Plus UCB c sweeps, adapt factor sweeps, target success sweeps, direction history sweeps. See `run_wave2.py`.

---

## Running

```bash
# Wave 1: 31 algorithms, 5,580 tasks (~90 min on 64 cores)
python run_full_ablation.py

# Wave 2: 23 algorithms, 4,140 tasks (~70 min on 64 cores)
python run_wave2.py

# Analysis (tables, stats, figures)
python analyze_results.py results/full_ablation_v2.json
python analyze_results.py results/wave2_results.json --csv results/tables.csv --figures results/figures/
```

Results: `results/full_ablation_v2.json`, `results/wave2_results.json`

---

## Next Steps

### Competition Entry (Aug 1 - Sep 8)

1. **Verify on official evaluation harness.** Current results use our local framework fork. Need to confirm scores reproduce.
2. **Fine-tune adaptive rate parameters.** Wave 2 suggests factor=1.1 and target=0.5 may be slightly better than defaults (1.2, 0.2). Run focused sweep.
3. **LogicPuzzles domain.** All algorithms score 100-1000x lower than TTP. May need domain-specific operators.

### Paper (AIIDE 2026)

- Full statistical tables ready
- Narrative: self-adapting mutation rate as a lightweight mechanism for non-stationary QD. The 1/5th rule naturally handles constraint changes without explicit detection.
- Position against: Multi-Emitter MAP-Elites (Cully), Adaptive Operator Selection (Thierens), CMA-MAE
- Key claim: simpler mechanism combinations outperform complex adaptive strategies when mechanisms are orthogonal
