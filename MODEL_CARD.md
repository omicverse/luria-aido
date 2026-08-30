# Model card — luria-aido 0.1.0

Checkpoint: `joint_f3init_z0detach_rpl`, 3 seeds.
F1 = Replogle 2022 (836 K-562 conditions) · F2 = sci-Plex 3 K-562 · F3 = f3_init warm start + z0_detach.

## Metrics, on the TEST split, with both nulls

`gap` is the loss increase when the condition→target correspondence is
permuted; `gap_z` divides it by the SD of a 30-draw null. Reference: a model
**trained** on shuffled labels scores gap_z 0.68 (F1) / 2.65 (F2).

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| gap_z F1 | 8.5 | 6.8 | 7.5 |
| gap_z F2 | 4.0 | 3.6 | 4.2 |
| gap_z F3 | 59.4 | 70.5 | 67.9 |
| F1 MSE | 0.9227 | 0.9232 | 0.9207 |

F1 floors: train-mean 0.971, zero 1.0544 — the model clears both.

## Versus the specialist for each task

| metric | ours (3 seeds) | specialist | provenance of the specialist number |
|---|---|---|---|
| F1 pearson_all | 0.246 ± 0.003 | GEARS 0.564 | **run locally**, official weights, simulation split seed 1, 107 test perturbations |
| F2 R²_all | 0.041 ± 0.006 | chemCPA 0.37 | quoted from paper |
| F3 corr_dev | 0.505 ± 0.021 | GENERator 0.71 | quoted from paper |
| F3 corr_hk | 0.792 ± 0.003 | GENERator 0.80 | quoted from paper |

Context for F1: in the same local run, **GEARS scored below its own train-mean
control** (0.564 vs 0.5786). The specialist bar in this task is low.

## Limits

1. **F2 does not identify the molecule.** With negative controls, glucose
   recovers 68% of imatinib's top-200 DE genes and dasatinib recovers 35.5%.
   The ordering is inverted. Use F2 for nothing that depends on which molecule
   was given. Cause: 166 unique training drugs.
2. **F3 depends on `f3_init` warm start.** Without it, F3 correlation is
   0.15–0.40 and degrades further under joint training as F1/F2 gradients pull
   `reg_head`. Reported numbers all use the warm start.
3. **F1 is real but 2× below the specialist.** Suitable for ranking and sign,
   not for quantitative prediction.
4. **The DNA leg of the bridge is near-constant** (pairwise cosine 0.995 across
   genes). The three-leg anchor is structurally present but information is not
   evenly distributed across the legs; this is the source of the low effective
   rank in the perturbation atlas (PR 1.95).
5. **C2 (cross-scale propagation) and C3 (multi-round) are not tested.** Not
   claimed either.
6. **No wet-lab validation of any prediction.**

## Ablations that did not work

Recorded so they are not repeated: two-stage shared/residual decomposition,
frozen-feature probe floor, token-level (per-gene) expression decoder,
per-cell latent state, anchor whitening, capacity scaling. None beat the
baseline; several degraded F3. Across 18 configurations and an 810-epoch run,
**the only change that moved the metric was replacing F1's dataset.**

## Intended use

Research. Ranking and hypothesis generation on K-562 gene knockouts, and
regulatory-sequence activity prediction. Not a substitute for measurement.
