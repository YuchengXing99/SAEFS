#!/usr/bin/env python
"""Encode the per-slide VQA answers into CONCH semantic anchors.

Input : a wide CSV with one row per slide -- ``slide_id, Q1, ..., QN``.
Output: a ``.npy`` of shape ``(n_unique_cases, N, 512)`` whose row order matches
        the CSV after de-duplicating on ``case_id`` (first occurrence wins).

CONCH is *not* vendored in this repository. Install it from
https://github.com/mahmoodlab/CONCH and download ``pytorch_model.bin`` from the
gated HuggingFace repo ``MahmoodLab/CONCH``:

    pip install git+https://github.com/Mahmoodlab/CONCH.git

Example
-------
    python scripts/encode_text.py \
        --anchor-csv data/captions/UCEC/wsi_questions_wide.csv \
        --conch-checkpoint /path/to/conch/pytorch_model.bin \
        --out splits/survival/TCGA_UCEC_overall_survival_k=0/text_embs_conch.npy
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from saefs.data import default_case_id_from_slide_id


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anchor-csv", required=True, help="wide CSV: slide_id, Q1..QN")
    p.add_argument("--conch-checkpoint", required=True, help="path to CONCH pytorch_model.bin")
    p.add_argument("--out", required=True, help="destination .npy")
    p.add_argument("--num-questions", type=int, default=None, help="use only the first N questions")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

    df = pd.read_csv(args.anchor_csv, engine="python")
    if "case_id" not in df.columns:
        df["case_id"] = default_case_id_from_slide_id(df["slide_id"])
    df = df[~df["case_id"].duplicated()].reset_index(drop=True)

    question_cols = sorted(
        [c for c in df.columns if c.startswith("Q") and c[1:].isdigit()],
        key=lambda c: int(c[1:]),
    )
    if args.num_questions:
        question_cols = question_cols[: args.num_questions]
    if not question_cols:
        raise SystemExit(f"No Q<i> columns found in {args.anchor_csv}")
    print(f"{len(df)} cases x {len(question_cols)} anchors: {question_cols}")

    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=args.conch_checkpoint)
    model = model.to(args.device).eval()
    tokenizer = get_tokenizer()

    flat = [str(t).strip() for row in df[question_cols].values.tolist() for t in row]
    embs = []
    with torch.no_grad():
        for i in range(0, len(flat), args.batch_size):
            tokens = tokenize(texts=flat[i : i + args.batch_size], tokenizer=tokenizer).to(args.device)
            embs.append(model.encode_text(tokens).cpu())
    embs = torch.cat(embs).reshape(len(df), len(question_cols), -1).numpy().astype(np.float32)

    np.save(args.out, embs)
    print(f"wrote {args.out} with shape {embs.shape}")


if __name__ == "__main__":
    main()
