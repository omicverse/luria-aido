# Artifacts

The package ships code, not weights. Point `LURIA_AIDO_DATA` at a directory
holding:

```
artifacts/K-562-gf/
  adapter_Aad.npy          (2176, 256)   gene anchor -> latent displacement
  adapter_Dad.npy          (384, 256)    chemistry   -> latent displacement
  anchor_vocab.txt         genes with a complete three-leg anchor
  gene_vocab.txt           the union gene vocabulary
  decoder_W.npy            (256, 5979)   latent -> standardized delta
  z_ctrl_mean.npy          (256,)        control state
```

The anchor is `DNA(GENERanno 1280) ⊕ protein(ESM2 640) ⊕ expression(Geneformer 256)`;
only genes covered by all three appear in `anchor_vocab.txt`.

## Training data

| family | dataset | scale | note |
|---|---|---|---|
| F1 | Replogle 2022 K-562 Perturb-seq | 836 train conditions | the only change that moved the metric |
| F1 (superseded) | Norman 2019 | 137 train conditions | too small; see MODEL_CARD limits |
| F2 | sci-Plex 3, K-562 | 664 combos / **166 unique drugs** | the binding constraint on F2 |
| F3 | regulatory sequence set | 402,296 train rows | the only leg at specialist parity |

## Datasets that do not fit, and why

Checked so they are not re-checked:

- **L1000 phase 1** — 678,401 rows, 17,201 compounds, 70 cell lines, **no K562**.
  Bulk, 978 landmark genes. Cross-calibration against sci-Plex K-562 on the 59
  shared drug names fails; the cell line, the modality and the control
  convention all differ. (X-Pert trains on it successfully because its model is
  cross-cell-line with `cell_id` as an input, not a single-line model.)
- **Tahoe-100M** — 100M cells, 1,100 compounds, but its 102 cell lines are solid
  tumours; **no K562**.

The drug leg has no dataset that is simultaneously K-562, single-cell, and
large. That is a data gap, not a modelling gap — and it is the reason F2 fails
while F1, which had Replogle, does not.
