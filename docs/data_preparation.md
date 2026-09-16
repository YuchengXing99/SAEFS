# Preparing your own data

SAEFS needs three things per cohort. Nothing else is assumed about your slides,
your scanner, or your VQA model.

```
<feature dir>/            one file per slide: <slide_id>.pt  or  <slide_id>.h5
<split dir>/
    train_xxx.csv         one row per slide
    test_yyy.csv
    text_embs_conch.npy   (n_unique_cases, Q, 512) semantic anchors
<anchor csv>              one row per slide: slide_id, Q1 ... QN
```

---

## 1. Patch features

One file per slide, named exactly after its `slide_id`:

* `<slide_id>.pt` — a `torch.save`-d tensor or numpy array of shape
  `(n_patches, D)`, or `(1, n_patches, D)`;
* `<slide_id>.h5` — an HDF5 file with a `features` dataset of the same shape.

`D` must match `--emb-dim` (512 for CONCH). Because the anchors are compared to
the patches by cosine similarity, **the patch encoder and the text encoder have
to share a space** — CONCH's vision tower and text tower do. Swapping in a
vision-only encoder (UNI, CTransPath, …) breaks the text-guided branch unless you
project both sides into a common space first.

Feature extraction itself is out of scope here; the layout above is what
[CLAM](https://github.com/mahmoodlab/CLAM) and
[Trident](https://github.com/mahmoodlab/trident) produce.

## 2. Split CSVs

One row per **slide**; several slides may share a `case_id`, in which case their
patches are concatenated into a single bag and the case is one training example.

| column | meaning |
|---|---|
| `case_id` | patient identifier — the unit of prediction |
| `slide_id` | must match the feature file name (without extension) |
| `<prefix>_survival_days` | observed time, in days, ≥ 0 |
| `<prefix>_censorship` | **1 = censored**, 0 = event observed |

`<prefix>` is whatever you pass to `--target-col`; `--target-col
dss_survival_days` looks for `dss_censorship`. Extra columns are ignored.

Rows with a missing time or censorship flag are dropped with a warning. A case
present in the split but absent from the anchor table is dropped too (and vice
versa).

Event times are quantile-binned into `--n-label-bins` bins, with the edges
estimated on the *uncensored* cases of that split. Pass `--share-label-bins` to
reuse the training split's edges on the test split; by default each split gets
its own, which is what the reported numbers used. Either way the bins affect only
the loss value and the IBS/INBLL time grid — the C-index is computed from the
continuous event times.

## 3. Semantic anchors

**The anchor CSV** is wide, one row per slide:

```csv
slide_id,Q1,Q2,Q3,...,Q12
TCGA-2E-A9G8-01Z-...,Moderate cellular atypia,Moderate nuclear atypia,High mitotic activity,...
```

Column `Qi` holds the answer a VQA model gave for question *i* on that slide.
Questions must be in the **same order for every slide** — anchor `i` is a
consistent semantic direction across the cohort, which is the whole point. The
questions used in the paper cover cellular and nuclear atypia, mitotic activity,
architectural pattern, tumour infiltration, vascular/lymphatic invasion,
necrosis, perineural invasion, stromal response, margin, angiogenesis and
carcinoma in situ.

`case_id` is derived from `slide_id` if the column is absent: TCGA barcodes are
truncated to 12 characters, CPTAC barcodes (`C3...`) to 9. For any other naming
scheme add a `case_id` column yourself, or pass
`--slide-to-case-json` with a `{"slide_id": "case_id"}` map.

**The anchor embeddings** are built once with CONCH:

```bash
python scripts/encode_text.py \
    --anchor-csv        data/captions/UCEC/wsi_questions_wide.csv \
    --conch-checkpoint  /path/to/conch/pytorch_model.bin \
    --out               splits/survival/TCGA_UCEC_overall_survival_k=0/text_embs_conch.npy
```

The result has shape `(n_unique_cases, Q, 512)` and its **row order is the anchor
CSV's order after de-duplicating on `case_id`, first occurrence kept**. Do not
re-sort or filter the CSV without regenerating the `.npy`; `load_anchor_table`
raises if the two disagree in length, but it cannot detect a re-ordering.

Use `--num-questions N` to keep only the first `N` questions (both when encoding
and when training) — that is how the question-count sensitivity study was run.

---

## Memory

`--cache-features` reads every bag once and holds it in RAM (or in VRAM with
`--cache-device cuda`), which turns a disk-bound epoch into a compute-bound one.
Budget roughly `n_patches × D × 4` bytes per slide: TCGA-UCEC + CPTAC-UCEC is
about 8 GB. Without the flag every epoch re-reads the bags through
`--num-workers` dataloader processes. `--bag-size K` caps each case at `K`
randomly sampled patches, which bounds both memory and compute at some cost in
accuracy.
