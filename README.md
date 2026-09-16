# SAEFS

**Semantic-Anchored Evidential Fusion for Domain-Robust Whole-Slide Survival Analysis**

Reference implementation. A whole-slide image is summarised three ways — by plain
attention pooling, by pooling that a small set of *semantic anchors* steers, and
by the anchors alone — and the three views are combined as **belief masses**
rather than logits, so a view that disagrees or has nothing to say can be
discounted instead of averaged in. That is what keeps the model from degrading
when it is moved to a cohort it was not trained on.

<!-- Replace with the teaser figure once you have one:
![SAEFS overview](assets/overview.png)
-->

---

## Method in one page

Let a case be a bag of patch embeddings `x_path ∈ R^{N×512}` (CONCH, 20×, 256 px)
and `Q = 12` anchor embeddings `x_text ∈ R^{Q×512}` — CONCH text embeddings of
the answers a VQA model gave for that slide, so patches and anchors live in the
*same* space.

**Three evidence sources.** Each produces a non-negative evidence vector over the
`K = 4` discrete survival bins via a linear head and `softplus`:

| branch | slide summary | evidence |
|---|---|---|
| WSI-only | ABMIL attention pooling | `e_wsi` |
| text-guided WSI | each anchor `q` softmax-selects the patches it matches (`A = softmax(x̂_text · x̂_pathᵀ / τ)`), averaged over anchors | `e_tg` |
| text-only | mean anchor embedding | `e_text` |

**Subjective-logic mapping.** `α = e + 1` defines a Dirichlet, which maps to a
Bayesian basic belief assignment
`b_k = e_k / (Σe + K)`, `u = K / (Σe + K)`, where `u` is the mass left on
"don't know".

**Fusion.** The two pathology views are mixed at the evidence level,
`e_path = (1 − α)·e_wsi + α·e_tg`, and the text assignment is discounted by a
reliability `r` (`b ← r·b`, `u ← u + (1−r)(1−u)`) before the two are combined
with **Denœux's cautious rule**. The cautious rule is idempotent, so it does not
double-count the evidence two views of the *same* slide necessarily share —
Dempster's rule does, and is available as `--fusion ds` for comparison.

**Objective.** An evidential discrete-time survival loss: evidential
cross-entropy on the observed bin for events, `−log P(T > censoring bin)` for
censored cases, plus an annealed KL term pulling the evidence of the wrong bins
towards uniform.

The two fusion hyper-parameters are the ones to think about:

- **`alpha` ∈ [0,1]** — 0 is plain ABMIL, 1 is purely anchor-driven pooling. The
  main configuration uses **0.5**, so both views contribute evidence.
- **`r` ∈ [0,1]** — the reliability of the text source. **1 applies no
  discounting**; smaller values push mass from the text's beliefs onto "don't
  know", and `r = 0` makes it vacuous so it abstains from the fusion entirely.
  The main configuration uses **1.0**.

Note that the two extremes switch parts of the model off, by construction rather
than by training: at `alpha = 1` the WSI-only branch is multiplied by zero and
`abmil_*`/`wsi_head` receive no gradient, and at `r = 0` the text BBA is vacuous,
so under the cautious rule the `min` always selects the pathology side and
`text_head` receives no gradient either. Use the interior of the range unless you
mean to ablate a branch.

---

## Repository layout

```
saefs/
  model.py       SAEFS: three branches + discounting + fusion
  evidence.py    Dirichlet -> BBA, discounting, Dempster & cautious rules, BetP
  losses.py      evidential discrete-time survival loss, evidence -> risk
  data.py        bag-of-patches + semantic-anchor dataset
  engine.py      train / evaluate loops, C-index, IBS, INBLL
  pipeline.py    SAEFSConfig + run_experiment
  viz.py         figure palette and plotting helpers
scripts/
  train.py       command-line training / evaluation
  encode_text.py VQA answers -> CONCH anchor embeddings
notebooks/
  reproduce_kirc.ipynb   TCGA-KIRC -> CPTAC-CCRCC reproduction, executed
checkpoints/
  saefs_kirc.pt          the weights behind the reported KIRC number
docs/
  data_preparation.md    exact file contract for your own data
```

## Install

```bash
conda create -n saefs python=3.9 -y && conda activate saefs
pip install -r requirements.txt
```

`scripts/encode_text.py` additionally needs CONCH, which is gated and therefore
not a hard dependency:

```bash
pip install git+https://github.com/Mahmoodlab/CONCH.git
```

If you already have the anchor embeddings (`text_embs_conch.npy`) you never need
CONCH at training time.

## Data

**No dataset is included in this repository.** You bring three things per cohort:
patch features, split CSVs, and the semantic anchors. The exact column names,
shapes and ordering constraints are in **[`docs/data_preparation.md`](docs/data_preparation.md)**.

## Quickstart

Train on one cohort and evaluate on another, with no adaptation in between:

```bash
python scripts/train.py \
  --feat-dirs   /data/tcga_cptac_ccrcc/feats_pt \
  --split-dir   splits/survival/TCGA_KIRC_overall_survival_k=0 \
  --train-split train_tcga.csv \
  --test-split  test_cptackirc_unfold.csv \
  --anchor-csv  data/captions/KIRC/wsi_questions_wide.csv \
  --anchor-emb  splits/survival/TCGA_KIRC_overall_survival_k=0/text_embs_conch.npy \
  --target-col  dss_survival_days \
  --alpha 0.5 --r 1.0 --lr 1e-4 --max-epochs 30 \
  --cache-features --results-dir results/ucec
```

`--cache-features` reads every bag once and keeps it in memory (~10 GB for
TCGA-KIRC + CPTAC-CCRCC); after that an epoch takes seconds instead of minutes.
Drop the flag if the cohort does not fit, or add `--bag-size 4096` to cap the
number of patches per case.

Or from Python:

```python
from saefs import SAEFSConfig, run_experiment

out = run_experiment(SAEFSConfig(
    feat_dirs=["/data/tcga_cptac_ccrcc/feats_pt"],
    split_dir="splits/survival/TCGA_KIRC_overall_survival_k=0",
    test_split="test_cptackirc_unfold.csv",
    anchor_csv="data/captions/KIRC/wsi_questions_wide.csv",
    anchor_emb="splits/survival/TCGA_KIRC_overall_survival_k=0/text_embs_conch.npy",
    alpha=0.5, r=1.0, lr=1e-4, max_epochs=30, cache_features=True,
))
print(out["metrics"]["test"]["c_index"])
```

`out["dumps"]["test"]` holds the per-case risk, predicted survival curve, and the
evidential diagnostics (`u_wsi`, `u_tg`, `u_text_raw`, `u_fused`, `K_pre`,
`cos_ht`) that the notebook's figures are built from.

## Reproducing the reported result

```bash
jupyter lab notebooks/reproduce_kirc.ipynb
```

`notebooks/reproduce_kirc.ipynb` reproduces the **TCGA-KIRC → CPTAC-CCRCC**
external-validation number from the released weights, then runs on that same
model the analyses the evidential formulation makes possible: risk-group
stratification with a log-rank test, calibration of the predicted survival
curves, per-branch uncertainty and conflict, an uncertainty-retention curve, and
a leave-one-anchor-out ranking of the semantic anchors. It evaluates; it does not
train. Edit the four paths in the first code cell (or export `SAEFS_DATA_ROOT`)
and run all cells. The committed copy carries its output, so the figures are
readable without running anything.

Loading `checkpoints/saefs_kirc.pt` and evaluating on CPTAC-CCRCC reproduces
every reported metric exactly:

| | reported | this repository |
|---|---|---|
| C-index | 0.6769 | **0.6769** |
| C-index (Antolini) | 0.6054 | **0.6054** |
| IBS | 0.2823 | **0.2823** |
| INBLL | 0.7222 | **0.7222** |

Training configuration behind those weights: `alpha=0.5`, `r=1.0`, `tau=1.0`,
lr `1e-4`, 30 epochs, AdamW with weight decay `1e-5`, cosine schedule without
warmup, batch size 1, no early stopping (the final epoch's weights are the ones
evaluated), 4 discrete survival bins, disease-specific survival.

### A note on retraining

The weights are released because **retraining does not reproduce this number.**
Two runs of the original research code on this cohort, with a byte-identical
configuration and the same `seed = 1`, produced C-index 0.5523 and 0.6769: with
batch size 1 and 30 epochs, non-deterministic GPU reductions compound into
different solutions, so the seed pins nothing. The external cohort also has only
15 observed events among 180 cases, which puts a standard error of roughly ±0.08
on any C-index measured on it before training variance is considered.

If you need repeatable training on your own data, pin it explicitly —
`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`torch.backends.cudnn.deterministic = True` — and expect a throughput cost.

### The other external-validation settings

Same command, different split directory and test file. `alpha = 0.5`, `r = 1.0`
throughout; the learning rate is selected per cohort (1e-4 for KIRC):

| train → test | `--split-dir` | `--test-split` | `--anchor-csv` |
|---|---|---|---|
| TCGA-UCEC → CPTAC-UCEC | `TCGA_UCEC_overall_survival_k=0` | `test_cptacucec_unfold.csv` | `UCEC/wsi_questions_wide.csv` |
| TCGA-LUAD → CPTAC-LUAD | `TCGA_LUAD_overall_survival_k=0` | `test_cptacluad_unfold.csv` | `LUAD/wsi_questions_wide.csv` |
| TCGA-KIRC → CPTAC-CCRCC | `TCGA_KIRC_overall_survival_k=0` | `test_cptackirc_unfold.csv` | `KIRC/wsi_questions_wide.csv` |
| TCGA-LUAD → NLST | `TCGA_NLSTLUAD_overall_survival_k=0` | `test_nlstluad_unfold.csv` | `NLSTLUAD/wsi_questions_wide.csv` |

NLST slide identifiers do not encode the case, so that setting needs
`--slide-to-case-json` (or a `case_id` column in the anchor CSV).

## Notes on reproducibility

- **Report a distribution, not a single run.** Two sources of spread stack up:
  the seed moves the external C-index (on TCGA-UCEC → CPTAC-UCEC, four seeds give
  0.667 ± 0.048), and a *fixed* seed still moves it (0.55 → 0.68 on KIRC). A gap
  smaller than that between two configurations is not evidence of anything. Train
  several runs and report mean ± sd before putting a number in a table.
- The KL annealing coefficient is `min(1, step / annealing_step)`, counted in
  **training** steps only, so the schedule does not depend on how often you
  evaluate. The default `annealing_step=10500` puts the coefficient at 1 after
  roughly 22 of the 30 epochs on a 480-case cohort; in our runs varying it by a
  factor of three moved the external C-index by less than the run-to-run noise.
- Test-split survival bins are estimated on the test split itself by default
  (`share_label_bins=False`), matching the protocol the reported numbers were
  produced with. Those bins only enter the reported *loss* and the IBS/INBLL time
  grid — never the C-index, and never the model. Pass `--share-label-bins` for
  the stricter protocol.

## Citation

```bibtex
@article{saefs,
  title  = {Semantic-Anchored Evidential Fusion for Domain-Robust Whole-Slide Survival Analysis},
  author = {TODO},
  year   = {TODO},
  journal= {TODO}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Patch and text embeddings come from
[CONCH](https://github.com/mahmoodlab/CONCH); the survival-split layout and the
bag-of-features convention follow
[PANTHER / MMP](https://github.com/mahmoodlab/MMP). Please observe those
projects' licenses for the model weights and any derived features.
