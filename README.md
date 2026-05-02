# QDA-VC Experiments

Algorithms for the [QDA-VC Competition](https://qdavc.github.io/) at AIIDE 2026.

**Task:** Quality-Diversity under Variable Constraints. Maintain archive quality as constraints shift between personas (Exploratory, Cycle, Adaptive) on two domains (TTP, LogicPuzzles).

**Competition timeline:** Submissions Aug 1 - Sep 8, 2026. Results Nov 13.

---

## Results (30 seeds, Mann-Whitney U)

### TTP Domain (Mean QD Score)

| Algorithm | Exploratory | Cycle | Adaptive | Total |
|-----------|----------:|------:|--------:|------:|
| **Bandit Experts** | **10,022** | **27,225** | 1,610 | **38,857** |
| **Evict-Restart** | 7,509 | 16,445 | **14,282** | **38,236** |
| **EGGROLL** | 11,685 | 23,905 | 1,700 | 37,290 |
| EvRst-nostag | 10,100 | 16,875 | 8,919 | 35,894 |
| SCOPE | 6,055 | 23,025 | 3,554 | 32,634 |
| EvRst-noevict | 5,129 | 14,261 | 12,855 | 32,245 |
| Combined | 7,157 | 14,494 | 6,494 | 28,145 |
| EGGROLL-nodir | 9,437 | 14,526 | 1,799 | 25,761 |
| Coevolution | 6,653 | 15,026 | 2,165 | 23,844 |
| Lamarckian | 4,174 | 16,629 | 1,991 | 22,795 |
| Baseline | 2,038 | 15,943 | 4,779 | 22,760 |
| Island | 61 | 2,142 | 342 | 2,545 |

### Grand Totals (TTP + LogicPuzzles, all personas)

| Rank | Algorithm | Total QD | vs Baseline |
|------|-----------|-------:|:-----------:|
| 1 | Bandit Experts | 38,925 | +70% |
| 2 | Evict-Restart | 38,326 | +68% |
| 3 | EGGROLL | 37,353 | +64% |
| 4 | EvRst-nostag (ablation) | 35,972 | +58% |
| 5 | SCOPE | 32,688 | +43% |
| 6 | EvRst-noevict (ablation) | 32,335 | +42% |
| 7 | Combined (EGGROLL+EvRst) | 28,225 | +24% |
| 8 | EGGROLL-nodir (ablation) | 25,826 | +13% |
| 9 | Coevolution | 23,906 | +5% |
| 10 | Lamarckian | 22,860 | +0.1% |
| 11 | Baseline (VC-MAP-Elites) | 22,828 | -- |
| 12 | Island | 2,613 | -89% |

### Ablation Analysis (TTP, significant results)

| Component | Full | Ablated | Diff | Sig |
|-----------|------|---------|------|-----|
| Direction learning (EGGROLL) | 23,905 (Cycle) | 14,526 | +9,380 | p=0.028 * |
| Stagnation restart (LogicPuzzles) | 48 (Cycle) | 45 | +3 | p=0.019 * |
| Direction learning (LogicPuzzles) | 48 (Cycle) | 40 | +8 | p<0.001 *** |

Eviction pool differences are positive but not statistically significant at 30 seeds.

---

## Key Findings

1. **Bandit Experts wins overall.** UCB1 selection over 4 mutation strategies (directed, standard, exploratory, conservative) adapts to each persona's constraint pattern without manual tuning.

2. **Evict-Restart dominates Adaptive persona.** Storing evicted solutions and re-inserting them when constraints change back is uniquely effective for the forward-backward constraint pattern. Other algorithms score near zero on Adaptive.

3. **EGGROLL dominates Cycle persona.** Directed mutation from successful parent history exploits the repeating constraint pattern.

4. **Combined (EGGROLL + Evict-Restart) underperforms both components.** The two mechanisms interfere: eviction pool restart wipes the directed mutation history, and directed mutation biases exploration away from eviction pool solutions.

5. **Island model is catastrophic.** Splitting the archive across islands prevents the coordinated response needed for constraint changes.

6. **Lamarckian and Coevolution don't help.** Both are statistically indistinguishable from baseline on aggregate.

7. **SCOPE's multi-step mutation helps on Cycle.** The repeated small mutations explore a local neighborhood that happens to work well when constraints cycle predictably.

---

## Algorithms

| # | File | Method |
|---|------|--------|
| 1 | `algorithms/mvp1_lamarckian.py` | Local search before archive insertion |
| 2 | `algorithms/mvp2_coevolution.py` | Coevolved constraint predictors bias selection |
| 3 | `algorithms/mvp3_scope_compressed.py` | Multi-step small mutations (SCOPE-inspired) |
| 4 | `algorithms/mvp4_eggroll_lowrank.py` | Directed mutation from successful parent history |
| 5 | `algorithms/mvp5_island_model.py` | Multiple sub-archives with migration |
| 6 | `algorithms/mvp6_evict_restart.py` | Eviction pool + stagnation restart |
| 7 | `algorithms/mvp7_bandit_experts.py` | UCB1 bandit over 4 mutation experts |
| 8 | `ablation_winners.py` (CombinedElites) | EGGROLL + Evict-Restart combined |

Ablation variants are parameterized from their base algorithms:
- `EGGROLL-nodir`: `EGGROLLElites(direction_weight=0.0)`
- `EvRst-noevict`: `EvictRestartElites(stagnation_patience=3)` (restarts but no eviction pool reuse)
- `EvRst-nostag`: `EvictRestartElites(stagnation_patience=9999)` (eviction pool but never restarts from stagnation)

---

## Running

Full 30-seed ablation on 64-core machine (~15 min):

```bash
python run_cluster_ablation.py
```

Results written to `results/cluster_ablation.json`.

---

## Next Steps

### Competition Entry (Aug 1 - Sep 8)

The competition entry will be **Bandit Experts** with tuning:

1. **Hyperparameter sweep on UCB exploration constant** (ucb_c). Current default is 1.0. Test 0.5, 1.0, 2.0, 4.0 across all personas to find optimal exploration/exploitation tradeoff.

2. **Add Evict-Restart as a 5th expert in the Bandit.** Since Evict-Restart wins Adaptive and Bandit wins overall, making eviction-restart-style mutation an expert option under the bandit gives the system access to both strategies without the interference we saw in Combined.

3. **Persona-specific expert pool.** If the bandit can detect which persona regime it's in (by tracking constraint change patterns), it can weight experts accordingly without waiting for UCB to converge.

4. **Test on competition's actual evaluation harness.** Current results use our local framework fork. Need to verify scores reproduce on the official evaluation code.

5. **LogicPuzzles domain needs work.** All algorithms score very low on LogicPuzzles. The domain may need domain-specific operators or a different archive structure. Investigate why QD scores are 100-1000x lower than TTP.

### Paper (AIIDE 2026, if writing up)

- Full statistical tables are ready (this README)
- Need: related work positioning against Multi-Emitter MAP-Elites (Cully), Adaptive Operator Selection (Thierens), and CMA-MAE
- Narrative: bandit-based expert selection as a lightweight alternative to explicit regime detection
