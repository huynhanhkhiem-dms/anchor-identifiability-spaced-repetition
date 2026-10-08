# Policy-Surface Identifiability in Adaptive Memory Scheduling

Public reproducibility materials for the manuscript:

**Policy-Surface Identifiability in Adaptive Memory Scheduling: When Predictive Fit Does Not Identify Scheduling Decisions**

Author: **Huynh Anh Khiem**  
Faculty of Information Technology, Ton Duc Thang University, Ho Chi Minh City, Vietnam  
ORCID: `0009-0007-7210-174X`

## What this repository contains

This repository provides the code and derived artifacts needed to inspect and rerun the main theoretical and empirical checks reported in the manuscript:

- NumPy implementation of the anchored forgetting-curve family and FSRS-6 state recursion;
- exact-anchor Fisher-information and dynamic-invariance checks;
- the benchmark-matched field-gate analysis for 1,000 Anki collections;
- the fixed 150-collection 21-parameter nuisance-adjusted profile runner;
- a Colab notebook that downloads gated source partitions after authentication and executes the frozen pipeline;
- deterministic scheduling-sensitivity code; and
- aggregate derived result summaries used in the manuscript.

## Raw data are not redistributed

The field analysis uses `open-spaced-repetition/anki-revlogs-10k` (DOI `10.57967/hf/3435`). The upstream license asks users not to redistribute the data publicly. Accordingly, this repository does **not** contain raw review logs. Reproduction requires obtaining access from the upstream Hugging Face dataset and then running the notebook/scripts here.

Dataset: https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k

## Frozen analysis design

- SRS benchmark commit: `1053082`
- Random seed: `20260817`
- Reduced-information field analysis: 1,000 benchmark-matched collections
- Full nuisance-adjusted profile: fixed 150-collection subsample
- `fsrs-optimizer==6.5.0`

The analysis preserves the pre-specified outcome even when reduced and full-profile summaries disagree. Thresholds, seeds, and sample sizes should not be changed to obtain a more favorable result.

## Reproducibility package

Download `anchor_reproducibility_package.zip` from this repository. It contains the complete clean source tree used for the public reproducibility package: core FSRS/theory code, field-gate scripts, full-profile checkpoint runner, Colab notebook, frozen configuration, and aggregate derived results.

After extracting:

```bash
python -m pip install -r requirements.txt
python code/decision_sensitivity.py
```

For the full gated-data rerun, open `analysis/field_gate_colab.ipynb`. It authenticates to Hugging Face, downloads the selected upstream partitions, applies the benchmark-matched filters, runs the 1,000-collection reduced analysis, and resumes the fixed 150-collection full-profile analysis from checkpoints.

## Reported aggregate field results

The repository includes `results/field_gate_summary.json` and `.csv`. Headline values include:

- reduced primary median shape s.e. `0.01636`;
- full 21-parameter median shape s.e. `0.02257`;
- full-profile 10th-90th percentile `0.0085-0.0935`;
- 35/150 (`23.3%`) collections in the descriptive high-uncertainty band; and
- Spearman association between predicted uncertainty and cross-refit instability `rho = 0.467`.

These are aggregate derived results only. Collection-level raw review data remain at the upstream gated source.

## Repository scope

This public repository is intentionally clean: no manuscript drafts, editor correspondence, internal review notes, AI/chat transcripts, superseded `v1/v2/v3` files, or raw gated review logs are included.

## License

Code in this repository is released under the MIT License. The upstream Anki review-log dataset remains governed by its own license and access conditions.
