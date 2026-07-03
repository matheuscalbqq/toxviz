# ToxViz — Interactive Exploration of Peptide Toxicity

**Practical assignment — INF 723 Data Visualization 2026/01 (UFV)**
Interactive visualization proposal applied to original research (peptide toxicity), following the guidelines of the IEEE VIS Bio+MedVis Challenge 2026.

🔗 **Live demo:** https://matheuscalbqq.github.io/toxviz/

---

## The problem

Peptide toxicity databases (hemolytic, cytotoxic, neurotoxic, immunotoxic) are published in a fragmented way, each with its own annotation criteria. This makes it difficult to answer simple questions such as: *how much do these categories actually overlap?* and *how many peptides might be multi-toxic without this being annotated in any single database?*

## The proposal

I consolidated 4 public databases (Hemolytik2, NTxPred2, IEDB, Tox-Prot) into **14,389 unique sequences**, with ESM-2 embeddings (480d) per sequence, and built an interactive prototype with two coordinated visualizations:

- **UMAP** — spatial density of each isolated category (normalized by that category's own peak, so as not to be dominated by the majority class), with confirmed intersections and multi-toxicity candidates highlighted.
- **UpSet plot** — intersections among the 4 categories (log scale), with candidates stacked on top of the confirmed bars.

The two views are linked via *brushing & linking*: selecting an area on the UMAP or clicking an UpSet bar reveals the corresponding points in both.

### Candidates for unannotated multi-toxicity
Beyond the confirmed data, the prototype identifies **statistical candidates**: sequences annotated with a single category, but whose neighborhood in the ESM-2 embedding space (480 dimensions — not the 2D UMAP, which does not reliably preserve distance between distant clusters) is statistically enriched for another category. The method uses:
- k-NN with **k statistically derived per target category**, which corrects a ceiling effect that made it impossible to detect candidates for the majority category with a fixed k;
- a binomial test against the population base rate (p < 0.001);
- a posterior probability via Bayesian update (Beta prior centered on the base rate).

## Repository structure

```
toxviz/
├── index.html              # final visualization (open directly or via GitHub Pages)
├── README.md
├── src/
│   ├── clean_and_consolidate.py     # consolidates the raw databases into 1 CSV
│   ├── ESM2_embeddings.py           # extracts ESM-2 embeddings (runs on cluster/GPU)
│   └── build_interactive.py         # computes UMAP + candidates, generates index.html
└── data/
    ├── toxviz_consolidated.csv      # sequence + 4 toxicity flags
    ├── toxviz_with_embeddings.parquet
    ├── toxviz_umap.csv              # UMAP projection cache
    ├── candidates.csv               # multi-toxicity candidates (final output)
    └── raw_data/                    # raw source databases
        ├── epitope_table_export_1782853329.csv         # IEDB
        ├── NTxPred2_independent_dataset.csv             # NTxPred2
        ├── NTxPred2_cross_val_dataset.csv               # NTxPred2
        ├── ToxProt.fasta                                # Tox-Prot (UniProt)
        ├── hemolytik.fasta                               # Hemolytik2
        └── hemolytic-and-cytotoxic-activities.csv        # DBAASP-like (hemo + cyto healthy cell)
```

## How to run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install altair pandas numpy scipy scikit-learn umap-learn pyarrow fair-esm torch

python3 src/clean_and_consolidate.py      # generates data/toxviz_consolidated.csv
python3 src/ESM2_embeddings.py            # generates data/toxviz_with_embeddings.parquet
python3 src/build_interactive.py          # generates index.html + data/candidates.csv
```

`index.html` is self-contained (uses Vega/Vega-Lite/Vega-Embed via CDN) — just open it in a browser with an internet connection.

## Design and limitations

- **Colorblind-safe palette** (Wong, 2011) throughout the visualization.
- **Density normalized per category**, not by absolute count — prevents the majority category (`immunotoxin`, 76% of the data) from visually dominating the others.
- **Documented limitation**: we measured that 13 of the 27 confirmed multi-toxic peptides appear visually "absorbed" into a single region on the UMAP (zero density for the second category at the point's position) — concrete evidence that 2D position should not be read as proof of a relationship between distant categories. This is why candidate detection uses the original embedding space (480d), not the UMAP.

## Author

Matheus Cavalcanti de Albuquerque, LaBio — Department of Computer Science, Universidade Federal de Viçosa
Course: INF 723 — Data Visualization (Prof. Sabrina de Azevedo Silveira)