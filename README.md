# luria-aido

A cell world model for K-562: three readout families decode from **one shared
latent state**, so a perturbation written once is visible to every readout.

The unusual thing about this repository is not the model. It is that every
number ships with the control that says whether it means anything — and by that
standard **one of the three legs works, one is weak, and one does not work at
all.** Those verdicts are below, measured, not adjectives.

```python
from luria_aido import Cell

cell = Cell("K-562")
ko = cell.clone()
ko.gene_knockout("ABL1")
ko.get_expression()            # pd.Series over 6584 genes
```

## What actually works

Every family is scored against two nulls — `zero` (predict no effect) and
`shuffle` (permute the condition→target correspondence, 30 draws) — and against
the specialist model for that task. `gap_z` is how many null-SDs the model sits
below the shuffle null: **it is the only number that says the model is using the
perturbation at all.**

| family | gap_z | vs specialist | verdict |
|---|---|---|---|
| **F3** regulatory sequence → activity | 59–71 | corr_hk 0.792 vs GENERator 0.80 | **works, at parity** |
| **F1** gene KO → expression | 6.8–8.5 | pearson 0.246 vs GEARS 0.556 | real signal, 2× below specialist |
| **F2** molecule → expression | 3.6–4.2 | R² 0.041 vs chemCPA 0.37 | **does not work** — see below |

Reference point for `gap_z`: a model *trained on shuffled labels* scores 0.68
(F1) and 2.65 (F2). F1 clears that comfortably. **F2 does not clear it.**

### F2 does not read the molecule

The honest version of "input any molecule, get the perturbed expression". We ran
the published in-context design loop, took imatinib as the reference, and asked
how many of its top-200 DE genes each molecule recovers — **with negative
controls**, which is the part usually left out:

| molecule | DEG recovery | cosine to reference |
|---|---|---|
| glucose | **68.0%** | +0.93 |
| urea | 64.0% | +0.91 |
| ibuprofen | 62.0% | +0.91 |
| scrambled SMILES | 44–51% | +0.85 |
| **dasatinib** (a real ABL1 inhibitor) | **35.5%** | +0.79 |

**The ordering is inverted.** Glucose beats the kinase inhibitor. Every molecule
correlates +0.5…+0.93 with the reference because they all predict the same
shared stress axis. Any DEG-recovery number reported without these controls is
uninterpretable — including, we think, published ones in the same range.

The cause is measured, not guessed: F2 has **166 unique training drugs**. F1 was
in the same position at 137 conditions and was fixed by data alone
(Replogle 2022, 836 conditions) — no architecture change helped. There is no
equivalent dataset for the drug leg: L1000 has no K562 arm, Tahoe-100M's 102
cell lines contain no K562.

## What "shared latent" buys, and how it was tested

The claim that distinguishes a world model from three predictors is that
improving one family does not degrade the others. Swapping F1's data source
(Norman → Replogle) left F2's gap unchanged (2.0–4.7 → 3.6–4.2) and F3's
correlation unchanged. That is the one novel claim here and it holds.

Untested: cross-scale propagation (C2) and multi-round experiments (C3). They
are not claimed.

## Install

```bash
pip install -e .            # core: Cell, expression readout
pip install -e ".[encoders,structure]"   # frozen encoders + delegated folding
```

Artifacts (anchor tables, adapters, gene vocabularies) are not in the repo. Point
`LURIA_AIDO_DATA` at them; see `docs/DATA.md`.

## Layout

```
luria_aido/          the package
  config.py          every path is an env var with a default
  cell.py            Cell: clone / perturb / read
  engine/            encoders, bridge, decoders, state
examples/            the published design loop, verified to run end to end
docs/ARCHITECTURE.md what is trained vs frozen, and why
docs/EVALUATION.md   the controls, and how to re-run them
MODEL_CARD.md        per-family numbers, floors, and limits
```

## Delegation

`get_protein_structure` and `get_protein_ligand_interactions` call out to
ESMFold / Boltz-2 / Protenix. They return `source: "delegated:<who>"`. The
expression path is self-contained.

Readouts that are not implemented raise rather than approximate. An
approximation under a real method's name is a different method reporting that
method's number.
