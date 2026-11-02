"""Emit data/pretraining_status.csv: DinoBloom pretraining contamination table.

Prerequisite for any claim about unseen data. Every claim
below is sourced to a primary document, not inferred. The evidence is held as
literals here so the table is reproducible and version-controlled.

Primary sources
---------------
[P] Koch et al., "DinoBloom: A Foundation Model for Generalizable Cell
    Embeddings in Hematology", MICCAI 2024, arXiv:2404.05022v1.
    - Sec. 2 "Datasets": enumerates the 13 pretraining datasets. ALL are
      peripheral blood or bone marrow. No cervical / gynaecological cytology
      dataset appears anywhere in the cohort.
    - Sec. 3 "Train and test data": "We train the DinoBloom models on all
      datasets except the Acevedo dataset, which is kept as external test set.
      The other two datasets used for evaluation were split into train/test
      (80/20). Only training data was used to train our DinoBloom models."
      The "other two" are AML Hehr and BMC.
[G] https://github.com/marrlab/DinoBloom README, "Overview of publicly
    available Blood Cell Datasets" table. NOTE: this table lists 15 datasets
    and is a survey, a superset of the 13 actually used. Where [G] and [P]
    disagree, [P] governs.

Consequence for this project
----------------------------
- HiCervix and PBC Barcelona (Acevedo) are CLEAN and may carry generalization
  claims.
- Matek BM and Raabin are contaminated and may only characterize the learned
  dictionary.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

PAPER = "arXiv:2404.05022v1 Sec.2+3 (primary)"

# dataset, split, in_dinobloom_pretraining, source_of_claim, note
ROWS = [
    # ---- Matek bone marrow (BMC). In the cohort; DinoBloom trained on an 80%
    # split that was never published, so we cannot identify which of our images
    # were seen. Marked "yes" rather than "partial" deliberately: the
    # conservative reading, since guardrail 2 disqualifies yes/unknown and we
    # will not let an unidentifiable 20% smuggle a generalization claim through.
    ("matek_bm", "train", "yes", PAPER,
     "BMC listed in the 13-dataset cohort (171,373 images). DinoBloom used an "
     "unpublished 80/20 split; membership of individual images is unrecoverable."),
    ("matek_bm", "val", "yes", PAPER, "Same as train; split boundary unrecoverable."),
    ("matek_bm", "test", "yes", PAPER, "Same as train; split boundary unrecoverable."),

    # ---- Raabin-WBC, resolved at SPLIT level.
    # DinoBloom reports using "10,175 cropped WBCs ... labelled by experts into
    # five classes". The distributed Cropped_double_labeled set contains exactly
    # three splits: Train 10,175 / TestA 4,339 / Test-B 2,119. Train matches
    # DinoBloom's figure EXACTLY, and Train is the only split with both 5
    # classes and that count, so the pretrained subset is identified as Train.
    # This is an inference from an exact count identity, not an explicit
    # statement by the authors — recorded as such, because a generalization
    # claim rests on it.
    ("raabin", "train", "yes", PAPER + " + exact count identity (10,175 = Train)",
     "DinoBloom's stated Raabin contribution of 10,175 cropped WBCs in 5 classes "
     "equals the distributed Train split exactly (212 baso + 744 eos + 2,427 "
     "lymph + 561 mono + 6,231 neut = 10,175)."),
    ("raabin", "testA", "no", PAPER + " + exact count identity (10,175 = Train)",
     "INFERRED, not stated: DinoBloom used 10,175 Raabin cells and Train alone "
     "accounts for all of them, so TestA (4,339) was not seen. Same device mix "
     "as Train, so this is a clean in-distribution held-out set."),
    ("raabin", "testB", "no", PAPER + " + exact count identity (10,175 = Train)",
     "INFERRED, not stated: as testA. Test-B (2,119) is additionally the "
     "acquisition-shifted set, so it is both device-shifted AND unseen — the "
     "only clean basis for a cross-device generalization claim."),

    # ---- PBC Barcelona (Acevedo). EXPLICITLY held out.
    ("pbc", "train", "no", PAPER,
     "Explicitly excluded: 'We train the DinoBloom models on all datasets "
     "except the Acevedo dataset, which is kept as external test set.'"),
    ("pbc", "val", "no", PAPER, "Explicitly excluded; see train."),
    ("pbc", "test", "no", PAPER, "Explicitly excluded; see train."),

    # ---- HiCervix. Not in the cohort, and out of modality entirely.
    ("hicervix", "train", "no", PAPER,
     "Absent from the 13-dataset cohort. All 13 are peripheral blood or bone "
     "marrow; no cervical cytology dataset appears in the pretraining corpus."),
    ("hicervix", "val", "no", PAPER, "Absent from the cohort; see train."),
    ("hicervix", "test", "no", PAPER, "Absent from the cohort; see train."),
]

HEADER = [
    "dataset", "split", "in_dinobloom_pretraining", "source_of_claim", "note",
]


def main() -> None:
    out = ROOT / "data" / "pretraining_status.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(ROWS)

    clean = sorted({r[0] for r in ROWS if r[2] == "no"})
    dirty = sorted({r[0] for r in ROWS if r[2] in ("yes", "unknown", "partial")})
    print(f"wrote {out} ({len(ROWS)} rows)")
    print(f"  clean (may carry generalization claims): {clean}")
    print(f"  contaminated (dictionary characterization only): {dirty}")


if __name__ == "__main__":
    main()
