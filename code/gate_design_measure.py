"""GATE - measure the design measure of real Anki logs, and decide.

This is the pre-registered empirical gate for the paper's quantitative claim.
The theory says the shape parameter of an anchored forgetting curve is identified
only through the dispersion of the realised log-interval ratio

    u_i = log( dt_i / S_i ),

where S_i is the memory stability the model itself reconstructs. Every number in
the design section is an arithmetic consequence of a *stated* sigma_u. None is a
measurement. This script performs the measurement.

We do not hold the raw review logs, so we do not run it in the replication
package. It is written to be run by anyone who does, including a referee.

    HOW TO GET THE DATA
    -------------------
        pip install huggingface_hub pandas pyarrow numpy scipy fsrs-optimizer
        huggingface-cli login
        huggingface-cli download open-spaced-repetition/anki-revlogs-10k \
            --repo-type dataset --local-dir ./anki-revlogs-10k

        python3 code/gate_design_measure.py \
            --data ./anki-revlogs-10k --bench ./data/srs-benchmark --users 1000

    `fsrs-optimizer` is REQUIRED, not optional: two of the benchmark's filters
    live there, and without them the benchmark-matched protocol is not
    replicated. If it is missing the script refuses to issue a verdict.

WHICH POPULATION
----------------
The raw corpus has ~727M reviews; the benchmark scores 349,923,850, because it
drops same-day reviews, over-long card histories, outlying second intervals and
non-continuous rows. Those populations are not interchangeable: measured over
everything, both n and sigma_u are larger and the implied standard error shrinks
on a population that produced none of the paper's claims. Two protocols, never
mixed:

  benchmark   replicates the benchmark's filter chain, read from its source at
              the pinned commit. The verdict is taken on this one.
  deployment  every long-term review, descriptive of the field.

THREE UNCERTAINTY NUMBERS, AND WHY THE THIRD IS NOT OPTIONAL
------------------------------------------------------------
  se_approx  the paper's closed form 1/(sigma_u sqrt(n C(psi))): a small-u
             expansion in a two-parameter (psi, alpha) model.
  se_exact   the Schur complement of the joint (psi, alpha) Fisher matrix summed
             over the *observed* log-ratios: exact within that two-parameter
             model, no expansion, no distributional assumption. This is the
             SCREENING statistic.
  se_full    1/sqrt(efficient Fisher information for w[20]) in the full
             21-parameter model, computed from a positive-semidefinite
             Gauss--Newton/Fisher construction on a subsample.

Revision 7 tried to infer a fixed ordering between the reduced and full-model
information from a numerical Hessian of the realised NLL. That Hessian can be
indefinite away from an optimum, and the apparent ordering was seed-sensitive.
The gate therefore makes no a-fortiori transfer in either direction. A verdict
requires the reduced and full-model checks to agree; disagreement is
INDETERMINATE.

PRE-REGISTERED DECISION RULE
----------------------------
Thresholds hard-coded in THRESHOLDS, on the median across collections under the
benchmark protocol with each collection's own fitted parameters.

  GO     median >= 0.05, AND the full-model profile subsample agrees.
  AMBER  median in [0.02, 0.05), AND the full-model subsample agrees.
  STOP   median < 0.02, AND the full-model profile subsample agrees. Wording
         matters: this says the field design measurement does not support the
         paper's empirical weak-identification claim. The theorems are untouched.
  INDETERMINATE  the protocols or parameter vectors disagree, the full-model
         check is absent or disagrees with the reduced check, or the benchmark
         filters could not be applied. No verdict is issued.

SENSITIVITY
-----------
S_i is latent and is reconstructed from fitted parameters -- the same parameters
the paper argues are weakly identified. The gate is therefore recomputed under
every parameter vector available: each collection's own FSRS-6 fit, the published
defaults, and each refit variant the benchmark publishes. If the verdict is not
the same in every cell the script reports INDETERMINATE. A gate that survives
only one reconstruction is not a gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srslib.memory import (  # noqa: E402
    FSRS6, FSRS6_BOUNDS, FSRS6_DEFAULT_STDDEV, FSRS6_INIT_W, FSRS6_PENALTY_GAMMA,
)
from srslib.theory import (  # noqa: E402
    fisher_curvature_constant, observed_joint_information, observed_se,
    prior_share, prior_share_from_curvature, finite_difference_probability_jacobian,
    profile_fisher_from_jacobian,
)

THRESHOLDS = dict(go_median_se=0.05, amber_median_se=0.02,
                  secondary_min_abs_rho=0.10)

REQUIRED_COLUMNS = ("card_id", "rating", "elapsed_days")
PROTOCOLS = ("benchmark", "deployment")
MAX_SEQ_LEN = 64
REFIT_VARIANTS = ["FSRS-6-short", "FSRS-6-short-recency", "FSRS-6-short-secs",
                  "FSRS-6-S0-short", "FSRS-6-binary-short"]


def _optional_fsrs_optimizer():
    try:
        from fsrs_optimizer import remove_non_continuous_rows, remove_outliers
        return remove_outliers, remove_non_continuous_rows
    except Exception:
        return None, None


def benchmark_filter(df, applied):
    """Replicate the benchmark's filter chain for the FSRS family."""
    d = df.copy()
    d["review_th"] = range(1, d.shape[0] + 1)
    d = d.sort_values(by=["card_id", "review_th"])
    applied.append("sort_by_card_and_review_th")
    d = d[d["rating"].isin([1, 2, 3, 4])]
    applied.append("valid_ratings")
    d["i"] = d.groupby("card_id").cumcount() + 1
    d = d[d["i"] <= MAX_SEQ_LEN * 2]
    applied.append(f"drop_i_gt_{MAX_SEQ_LEN * 2}")
    d["delta_t"] = d["elapsed_days"].clip(lower=0)
    d = d[d["elapsed_days"] != 0]
    applied.append("drop_same_day")
    d["i"] = d["elapsed_days"].gt(0).groupby(d["card_id"]).cumsum().add(1)
    d["first_rating"] = d.groupby("card_id")["rating"].transform("first").astype(str)

    remove_outliers, remove_non_continuous_rows = _optional_fsrs_optimizer()
    if remove_outliers is not None:
        second = d[d["i"] == 2]
        if not second.empty:
            kept = (second.groupby("first_rating", as_index=False,
                                   group_keys=False)[list(d.columns)]
                    .apply(remove_outliers))
            d = d.drop(index=second.index.difference(kept.index))
        applied.append("remove_outliers")
        d = (d.groupby("card_id", as_index=False, group_keys=False)[list(d.columns)]
             .apply(remove_non_continuous_rows))
        applied.append("remove_non_continuous_rows")
    else:
        applied.append("SKIPPED:remove_outliers")
        applied.append("SKIPPED:remove_non_continuous_rows")

    d = d[d["delta_t"] > 0].sort_values(by=["card_id", "review_th"])
    applied.append("delta_t_gt_0")
    return d


def deployment_filter(df, applied):
    d = df.copy()
    d["review_th"] = range(1, d.shape[0] + 1)
    d = d.sort_values(by=["card_id", "review_th"])
    d = d[d["rating"].isin([1, 2, 3, 4])]
    d["delta_t"] = d["elapsed_days"].clip(lower=0)
    applied.append("valid_ratings")
    applied.append("no_further_filtering")
    return d


def _card_arrays(df, col):
    groups = [g for _, g in df.groupby("card_id", sort=False)
              if g.shape[0] >= 2]
    if not groups:
        return None
    max_len = max(g.shape[0] - 1 for g in groups)
    n = len(groups)
    dt = np.zeros((n, max_len))
    rating = np.ones((n, max_len), dtype=int)
    mask = np.zeros((n, max_len), dtype=bool)
    rating0 = np.empty(n, dtype=int)
    y = np.zeros((n, max_len))
    for i, g in enumerate(groups):
        r = g["rating"].to_numpy(dtype=int)
        t = g[col].to_numpy(dtype=float)
        rating0[i] = r[0]
        k = r.size - 1
        dt[i, :k] = np.maximum(t[1:], 1e-6)
        rating[i, :k] = r[1:]
        mask[i, :k] = True
        y[i, :k] = (r[1:] > 1).astype(float)
    return rating0, dt, rating, mask, y


def reconstruct_u(df, w, min_delta_t=1.0):
    col = "delta_t" if "delta_t" in df.columns else "elapsed_days"
    packed = _card_arrays(df, col)
    if packed is None:
        return np.zeros(0), 0
    rating0, dt, rating, mask, _ = packed
    m = FSRS6(w)
    s, d = m.init_state(rating0)
    us = []
    for k in range(dt.shape[1]):
        ok = mask[:, k] & (dt[:, k] >= min_delta_t) & (s > 0)
        if ok.any():
            us.append(np.log(dt[ok, k] / s[ok]))
        step_dt = np.where(mask[:, k], dt[:, k], 1.0)
        ns, nd = m.step(s, d, step_dt, rating[:, k])
        s = np.where(mask[:, k], ns, s)
        d = np.where(mask[:, k], nd, d)
    u = np.concatenate(us) if us else np.zeros(0)
    return u, int(rating0.size)


def collection_probabilities(w, packed):
    rating0, dt, rating, mask, y = packed
    m = FSRS6(np.asarray(w, dtype=float))
    s, d = m.init_state(rating0)
    chunks = []
    for k in range(dt.shape[1]):
        p = np.clip(m.retention(dt[:, k], s), 1e-9, 1.0 - 1e-9)
        if mask[:, k].any():
            chunks.append(p[mask[:, k]])
        step_dt = np.where(mask[:, k], dt[:, k], 1.0)
        ns, nd = m.step(s, d, step_dt, rating[:, k])
        s = np.where(mask[:, k], ns, s)
        d = np.where(mask[:, k], nd, d)
    return np.concatenate(chunks) if chunks else np.zeros(0)


def collection_nll(w, packed):
    rating0, dt, rating, mask, y = packed
    m = FSRS6(np.asarray(w, dtype=float))
    s, d = m.init_state(rating0)
    total = 0.0
    for k in range(dt.shape[1]):
        p = np.clip(m.retention(dt[:, k], s), 1e-9, 1.0 - 1e-9)
        term = -(y[:, k] * np.log(p) + (1.0 - y[:, k]) * np.log1p(-p))
        total += float(np.sum(np.where(mask[:, k], term, 0.0)))
        step_dt = np.where(mask[:, k], dt[:, k], 1.0)
        ns, nd = m.step(s, d, step_dt, rating[:, k])
        s = np.where(mask[:, k], ns, s)
        d = np.where(mask[:, k], nd, d)
    return total


def full_profile_se(df, w, rel_step=1e-5):
    col = "delta_t" if "delta_t" in df.columns else "elapsed_days"
    packed = _card_arrays(df, col)
    if packed is None:
        return None
    prob_fn = lambda x: collection_probabilities(x, packed)
    p0, J = finite_difference_probability_jacobian(
        prob_fn, np.asarray(w, dtype=float), bounds=FSRS6_BOUNDS, rel_step=rel_step)
    info, rank = profile_fisher_from_jacobian(p0, J, idx=20)
    return dict(information_full=float(info), nuisance_rank=int(rank),
                se_full=(float("inf") if info <= 0 else float(1.0 / math.sqrt(info))),
                prior_share_full=prior_share_from_curvature(
                    info, FSRS6_DEFAULT_STDDEV[20], FSRS6_PENALTY_GAMMA))


def summarise(u, psi_hat):
    if u.size < 2:
        return None
    sd = float(np.std(u, ddof=1))
    mu = float(np.mean(u))
    C = float(fisher_curvature_constant(psi_hat))
    se_exact = observed_se(u, psi_hat)
    se_approx = (1.0 / (sd * math.sqrt(u.size * C))
                 if sd > 0 and C > 0 else float("inf"))
    joint = observed_joint_information(u, psi_hat)
    return dict(
        n=int(u.size), mu_u=mu, sigma_u=sd,
        p_within_1pct05=float(np.mean(np.abs(u) < math.log(1.05))),
        p_within_1pct10=float(np.mean(np.abs(u) < math.log(1.10))),
        median_abs_u=float(np.median(np.abs(u))),
        psi_hat=float(psi_hat), C=C,
        se_exact=float(se_exact), se_approx=float(se_approx),
        joint_information_det=float(np.linalg.det(joint)),
        prior_share_reduced=prior_share(u.size, sd, psi_hat,
                                        FSRS6_DEFAULT_STDDEV[20],
                                        FSRS6_PENALTY_GAMMA),
    )


def verdict_for(median_se):
    if not np.isfinite(median_se):
        return "GO"
    if median_se >= THRESHOLDS["go_median_se"]:
        return "GO"
    if median_se >= THRESHOLDS["amber_median_se"]:
        return "AMBER"
    return "STOP"


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return float("nan"), int(ok.sum())
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1]), int(ok.sum())


def load_variant_params(bench, name):
    jl = Path(bench) / "result" / f"{name}.jsonl"
    if not jl.exists():
        return {}
    out = {}
    with open(jl) as fh:
        for line in fh:
            rec = json.loads(line)
            p = rec.get("parameters")
            if isinstance(p, dict) and p:
                p = next(iter(p.values()))
            if isinstance(p, list) and len(p) == 21:
                out[int(rec["user"])] = np.asarray(p, dtype=float)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to anki-revlogs-10k")
    ap.add_argument("--bench", required=True, help="path to a srs-benchmark clone")
    ap.add_argument("--users", type=int, default=1000,
                    help="collections to process (0 = all)")
    ap.add_argument("--full-profile", type=int, default=150,
                    help="collections to also run the 21-parameter profile on "
                         "(0 disables, which forces INDETERMINATE on GO/AMBER)")
    ap.add_argument("--protocols", default="benchmark,deployment")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--allow-missing-filters", action="store_true",
                    help="proceed without fsrs-optimizer; the verdict is then "
                         "forced to INDETERMINATE and this flag only silences "
                         "the early exit")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--out", default="results/gate_design_measure.json")
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required: pip install pandas pyarrow")

    ro, _ = _optional_fsrs_optimizer()
    if ro is None and not args.allow_missing_filters:
        sys.exit(
            "fsrs-optimizer is not installed, so remove_outliers and\n"
            "remove_non_continuous_rows cannot run and the benchmark-matched\n"
            "protocol is not replicated. This gate refuses to issue a verdict\n"
            "on a population it cannot match.\n\n"
            "    pip install fsrs-optimizer\n\n"
            "Pass --allow-missing-filters to run anyway; the verdict will be\n"
            "INDETERMINATE and the output will say why.")

    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    for p in protocols:
        if p not in PROTOCOLS:
            sys.exit(f"unknown protocol {p!r}; choose from {PROTOCOLS}")

    root = Path(args.data) / "revlogs"
    if not root.is_dir():
        sys.exit(f"{root} not found; see the module docstring for how to obtain it")
    parts = sorted(p for p in root.iterdir() if p.name.startswith("user_id="))
    if not parts:
        sys.exit(f"no user_id=* partitions under {root}")
    uids = [int(p.name.split("=", 1)[1]) for p in parts]

    variants = {}
    for name in REFIT_VARIANTS:
        v = load_variant_params(args.bench, name)
        if v:
            variants[name] = v
    if "FSRS-6-short" not in variants:
        sys.exit(f"{args.bench}/result/FSRS-6-short.jsonl not found or unusable")
    fitted = variants["FSRS-6-short"]
    uids = [u for u in uids if u in fitted]

    param_sets = {"fitted": fitted, "default": None}
    if not args.no_sensitivity:
        for name, v in variants.items():
            if name != "FSRS-6-short":
                param_sets[name] = v

    rng = np.random.default_rng(args.seed)
    if args.users and args.users < len(uids):
        uids = sorted(rng.choice(uids, size=args.users, replace=False).tolist())
    full_uids = set()
    if args.full_profile:
        k = min(args.full_profile, len(uids))
        full_uids = set(np.random.default_rng(args.seed + 1)
                        .choice(uids, size=k, replace=False).tolist())

    print(f"collections:            {len(uids)}")
    print(f"full-profile subsample: {len(full_uids)}")
    print(f"protocols:              {', '.join(protocols)}")
    print(f"parameter sets:         {', '.join(param_sets)}")
    print(f"fsrs_optimizer filters: {'available' if ro else 'MISSING'}")
    print()

    cells = {(p, k): [] for p in protocols for k in param_sets}
    full_rows = []
    filters_applied = {}
    skipped = 0

    for idx, uid in enumerate(uids, 1):
        try:
            raw = pd.read_parquet(root / f"user_id={uid}")
        except Exception as exc:
            skipped += 1
            print(f"  [{idx}/{len(uids)}] user {uid}: unreadable ({exc})", flush=True)
            continue
        missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            sys.exit(f"user {uid}: revlog missing columns {missing}; "
                     f"found {list(raw.columns)}")

        for proto in protocols:
            applied = []
            d = (benchmark_filter(raw, applied) if proto == "benchmark"
                 else deployment_filter(raw, applied))
            filters_applied[proto] = applied
            if d.empty:
                continue
            for key, table in param_sets.items():
                w = FSRS6_INIT_W if table is None else table.get(uid)
                if w is None:
                    continue
                u, n_cards = reconstruct_u(d, w)
                rec = summarise(u, w[20])
                if rec is None:
                    continue
                rec.update(user=int(uid), n_cards=n_cards, protocol=proto,
                           params=key)
                cells[(proto, key)].append(rec)
                if (proto == "benchmark" and key == "fitted"
                        and uid in full_uids):
                    fp = full_profile_se(d, w)
                    if fp:
                        fp.update(user=int(uid), n=rec["n"],
                                  se_exact=rec["se_exact"])
                        full_rows.append(fp)

        if idx % 25 == 0 or idx == len(uids):
            prim = cells.get(("benchmark", "fitted")) or cells[(protocols[0], "fitted")]
            if prim:
                med = float(np.median([r["se_exact"] for r in prim]))
                print(f"  [{idx}/{len(uids)}] primary n={len(prim)} "
                      f"running median se_exact {med:.4f}", flush=True)

    primary_key = ("benchmark" if "benchmark" in protocols else protocols[0], "fitted")
    if not cells[primary_key]:
        sys.exit("no usable collections in the primary cell")

    def q(a):
        a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
        if a.size == 0:
            return None
        return dict(p10=float(np.percentile(a, 10)), median=float(np.median(a)),
                    mean=float(np.mean(a)), p90=float(np.percentile(a, 90)))

    summary = {}
    for (proto, key), rows in cells.items():
        if not rows:
            continue
        se = np.array([r["se_exact"] for r in rows])
        summary[f"{proto}|{key}"] = dict(
            protocol=proto, params=key, n_collections=len(rows),
            total_reviews=int(sum(r["n"] for r in rows)),
            sigma_u=q([r["sigma_u"] for r in rows]),
            mu_u=q([r["mu_u"] for r in rows]),
            se_exact=q(se), se_approx=q([r["se_approx"] for r in rows]),
            prior_share_reduced=q([r["prior_share_reduced"] for r in rows]),
            p_within_1pct05=q([r["p_within_1pct05"] for r in rows]),
            p_within_1pct10=q([r["p_within_1pct10"] for r in rows]),
            reviews_per_collection=q([r["n"] for r in rows]),
            median_se_exact=float(np.median(se)),
            share_se_above_0_05=float(np.mean(se >= 0.05)),
            verdict=verdict_for(float(np.median(se))),
        )

    verdicts = {k: v["verdict"] for k, v in summary.items()}
    primary_verdict = summary[f"{primary_key[0]}|{primary_key[1]}"]["verdict"]
    cells_agree = len(set(verdicts.values())) == 1

    full_summary = None
    full_verdict = None
    if full_rows:
        sef = np.array([r["se_full"] for r in full_rows])
        full_verdict = verdict_for(float(np.median(sef)))
        full_summary = dict(
            n_collections=len(full_rows),
            se_full=q(sef),
            median_se_full=float(np.median(sef)),
            prior_share_full=q([r["prior_share_full"] for r in full_rows]),
            ratio_full_over_exact=q([r["se_full"] / r["se_exact"]
                                     for r in full_rows
                                     if np.isfinite(r["se_exact"])
                                     and r["se_exact"] > 0]),
            verdict=full_verdict,
        )

    reasons = []
    if ro is None:
        reasons.append("benchmark filters could not be applied "
                       "(fsrs-optimizer missing)")
    if not cells_agree:
        reasons.append(f"protocol/parameter cells disagree: "
                       f"{sorted(set(verdicts.values()))}")
    if full_verdict is None:
        reasons.append("every reduced-model verdict must be confirmed by the "
                       "21-parameter Fisher profile, which was not run "
                       "(--full-profile 0)")
    elif full_verdict != primary_verdict:
        reasons.append(f"the 21-parameter Fisher profile says {full_verdict} where "
                       f"the two-parameter statistic says {primary_verdict}")
    final = primary_verdict if not reasons else "INDETERMINATE"

    secondary = None
    prim_rows = cells[primary_key]
    w20_by_variant = {name: {u: v[u][20] for u in v} for name, v in variants.items()}
    refit_sd, se_ok, on_bound = [], [], []
    for r in prim_rows:
        vals = [w20_by_variant[n][r["user"]] for n in w20_by_variant
                if r["user"] in w20_by_variant[n]]
        if len(vals) >= 3:
            refit_sd.append(float(np.std(vals, ddof=1)))
            se_ok.append(r["se_exact"])
            on_bound.append(float(min(abs(r["psi_hat"] - 0.1),
                                      abs(r["psi_hat"] - 0.8)) < 1e-6))
    if refit_sd:
        rho_sd, n_sd = spearman(se_ok, refit_sd)
        rho_b, n_b = spearman(se_ok, on_bound)
        secondary = dict(spearman_se_vs_refit_sd=rho_sd, n_pairs_refit=n_sd,
                         spearman_se_vs_on_bound=rho_b, n_pairs_bound=n_b,
                         mechanism_supported=bool(
                             np.isfinite(rho_sd)
                             and rho_sd >= THRESHOLDS["secondary_min_abs_rho"]))

    out = dict(
        gate="design measure of real Anki logs",
        dataset="open-spaced-repetition/anki-revlogs-10k",
        seed=args.seed, thresholds=THRESHOLDS,
        primary_cell=f"{primary_key[0]}|{primary_key[1]}",
        protocols=protocols, parameter_sets=sorted(param_sets),
        filters_applied=filters_applied,
        fsrs_optimizer_available=bool(ro),
        benchmark_protocol_fully_replicated=bool(ro),
        n_skipped=skipped, cells=summary, verdicts=verdicts,
        cells_agree=cells_agree,
        full_profile=full_summary,
        stop_means=("the field design measurement does not support the paper's "
                    "empirical weak-identification claim; the theorems, which "
                    "concern an exact policy, are untouched"),
        verdict=final, primary_verdict=primary_verdict,
        indeterminate_reasons=reasons,
        secondary=secondary,
        per_collection={f"{p}|{k}": v for (p, k), v in cells.items()},
        per_collection_full_profile=full_rows,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print()
    hdr = f"{'cell':<34}{'n':>6}{'sigma_u':>10}{'se_exact':>11}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(summary):
        v = summary[k]
        print(f"{k:<34}{v['n_collections']:>6}{v['sigma_u']['median']:>10.4f}"
              f"{v['median_se_exact']:>11.4f}  {v['verdict']}")
    if full_summary:
        print(f"\n21-parameter profile on {full_summary['n_collections']} "
              f"collections: median se_full {full_summary['median_se_full']:.4f} "
              f"-> {full_summary['verdict']}")
    if secondary:
        print(f"Spearman(se_exact, cross-refit s.d. of w20) = "
              f"{secondary['spearman_se_vs_refit_sd']:+.3f} "
              f"(n={secondary['n_pairs_refit']})")
    print()
    print(f"PRE-REGISTERED VERDICT: {final}")
    for reason in reasons:
        print(f"  - {reason}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
