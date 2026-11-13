# SAEFS

**Semantic-Anchored Evidential Fusion for Domain-Robust Whole-Slide Survival Analysis**, ECCV 2026.

[Paper](https://link.springer.com/chapter/10.1007/978-3-032-37520-9_9) · [arXiv](https://arxiv.org/abs/2606.19966) · [Cite](#citation)

SAEFS combines WSI features and VQA-derived semantic anchors through evidential fusion for survival prediction across clinical centers.

## Installation

```bash
conda create -n saefs python=3.9 -y
conda activate saefs
pip install -r requirements.txt
```

## Data

Prepare CONCH patch features (512-d), survival split CSVs, and semantic-anchor CSV/embeddings. Datasets are not included. See [data preparation](docs/data_preparation.md) for formats and anchor encoding.

## Training

Run from the repository root (TCGA-KIRC → CPTAC-CCRCC):

```bash
python scripts/train.py \
  --feat-dirs /path/to/feats_pt \
  --split-dir /path/to/TCGA_KIRC_overall_survival_k=0 \
  --test-split test_cptackirc_unfold.csv \
  --anchor-csv /path/to/wsi_questions_wide.csv \
  --anchor-emb /path/to/text_embs_conch.npy \
  --lr 1e-4 --max-epochs 30 \
  --results-dir results/kirc
```

See `python scripts/train.py --help` for options.

## Reproduce

Open [the reproduction notebook](notebooks/reproduce_kirc.ipynb), set the data paths in its configuration cell, and run all cells. It evaluates the released `checkpoints/saefs_kirc.pt` on CPTAC-CCRCC (C-index: 0.6769) and includes visualization examples.

Retraining is nondeterministic, even with the same seed; report results across multiple runs. Survival bins are estimated separately per split by default; use `--share-label-bins` to reuse training bins.

## Citation

```bibtex
@inproceedings{xing2026saefs,
  title={Semantic-Anchored Evidential Fusion for Domain-Robust Whole-Slide Survival Analysis},
  author={Xing, Yucheng and Huang, Ling and Liu, Pei and Ma, Jingying and Xu, Jiaxing and He, Kai and Feng, Mengling},
  booktitle={Computer Vision -- ECCV 2026},
  year={2026}
}
```

## Acknowledgements

Built on [CONCH](https://github.com/mahmoodlab/CONCH) and [PANTHER / MMP](https://github.com/mahmoodlab/MMP). Please follow their model and data licenses.

License: [MIT](LICENSE).
