# Policy-Surface Identifiability in Adaptive Memory Scheduling

Public reproducibility materials for the manuscript:

**Policy-Surface Identifiability in Adaptive Memory Scheduling: When Predictive Fit Does Not Identify Scheduling Decisions**

Author: **Huynh Anh Khiem**  
Faculty of Information Technology, Ton Duc Thang University, Ho Chi Minh City, Vietnam  
ORCID: `0009-0007-7210-174X`

## Public artifacts

The repository exposes the reproduction files directly:

- `code/gate_design_measure.py` — frozen 1,000-collection reduced-information gate and sensitivity analysis;
- `analysis/run_full_profile_checkpoint.py` — resumable fixed 150-collection 21-parameter profile runner;
- `analysis/field_gate_colab.ipynb` — clean Colab entry point for obtaining the gated data and rerunning the frozen analysis;
- `code/srslib/memory.py` — anchored forgetting curves and FSRS-6 state recursion;
- `code/srslib/theory.py` — Fisher-information and nuisance-profile utilities;
- `code/decision_sensitivity.py` — deterministic off-anchor scheduling-sensitivity check;
- `requirements.txt` and `data/BENCHMARK_COMMIT.txt` — pinned environment and benchmark revision; and
- `results/field_gate_summary.json` / `.csv` — aggregate derived manuscript results.

## Raw data are not redistributed

The empirical analysis uses `open-spaced-repetition/anki-revlogs-10k` (DOI `10.57967/hf/3435`). The upstream dataset requires users to accept access conditions and its license asks users not to redistribute the data publicly. Raw review logs are therefore **not** copied into this repository.

Upstream dataset: https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k

The Colab notebook downloads the source data only after the researcher authenticates to Hugging Face and has accepted the dataset conditions.

## Frozen design

- SRS benchmark commit: `1053082`
- Random seed: `20260817`
- Reduced-information field analysis: 1,000 benchmark-matched collections
- Full nuisance-adjusted profile: fixed 150-collection subsample
- `fsrs-optimizer==6.5.0`

The design preserves the pre-specified outcome when reduced and full-profile summaries disagree. Seeds, sample sizes, thresholds, and the analysis population should not be changed after inspecting results to obtain a more favorable conclusion.

## Reproduction

For the deterministic scheduling check:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=code python code/decision_sensitivity.py
```

For the review-log analysis, open `analysis/field_gate_colab.ipynb`. It authenticates to Hugging Face, clones the SRS benchmark, checks out commit `1053082`, downloads the gated upstream dataset, and calls the frozen analysis with 1,000 collections, 150 full profiles, and seed `20260817`.

## Aggregate field results

The public derived summaries report:

- reduced primary median shape s.e. `0.01636`;
- reduced 10th–90th percentile `0.0050–0.0783`;
- full 21-parameter median shape s.e. `0.02257`;
- full-profile 10th–90th percentile `0.0085–0.0935`;
- 35/150 (`23.3%`) collections in the descriptive high-uncertainty band; and
- Spearman association between predicted uncertainty and cross-refit instability `rho = 0.467`.

These are aggregate derived values. Raw review-level records remain solely at the upstream gated source.

## Repository scope

This repository intentionally excludes manuscript drafts, editor correspondence, internal review notes, chat transcripts, superseded `v1/v2/v3` files, and raw gated review logs.

## License

Repository code is released under the MIT License. The upstream Anki dataset remains governed by its own license and access conditions.
