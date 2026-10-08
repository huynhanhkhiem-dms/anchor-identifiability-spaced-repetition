import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--bench", required=True)
ap.add_argument("--reduced-json", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--seed", type=int, default=20260817)
args = ap.parse_args()

sys.path.insert(0, str(Path(args.project) / "code"))
import gate_design_measure as g

reduced = json.loads(Path(args.reduced_json).read_text())
primary_rows = reduced["per_collection"]["benchmark|fitted"]
uids = sorted(int(r["user"]) for r in primary_rows)
k = min(args.n, len(uids))
full_uids = sorted(np.random.default_rng(args.seed + 1).choice(
    uids, size=k, replace=False).tolist())

fitted = g.load_variant_params(args.bench, "FSRS-6-short")
cp_path = Path(args.checkpoint)
if cp_path.exists():
    cp = json.loads(cp_path.read_text())
else:
    cp = {"target_uids": full_uids, "rows": [], "failures": []}

if cp.get("target_uids") != full_uids:
    raise RuntimeError("Checkpoint target_uids do not match the frozen selection; refusing to overwrite.")

done = {int(r["user"]) for r in cp["rows"]}

for i, uid in enumerate(full_uids, 1):
    if uid in done:
        print(f"[{i}/{len(full_uids)}] user {uid}: checkpointed", flush=True)
        continue
    try:
        raw = pd.read_parquet(Path(args.data) / "revlogs" / f"user_id={uid}")
        applied = []
        d = g.benchmark_filter(raw, applied)
        if any(str(x).startswith("SKIPPED:") for x in applied):
            raise RuntimeError(f"Benchmark filter missing: {applied}")
        w = fitted.get(uid)
        if w is None:
            raise RuntimeError("Missing FSRS-6-short fitted params")
        fp = g.full_profile_se(d, w)
        if fp is None:
            raise RuntimeError("full_profile_se returned None")
        fp.update(user=int(uid))
        cp["rows"].append(fp)
        cp["filters_applied"] = applied
        cp_path.write_text(json.dumps(cp, indent=2))
        done.add(uid)
        print(f"[{i}/{len(full_uids)}] user {uid}: se_full={fp['se_full']:.6f} (saved)", flush=True)
    except Exception as e:
        existing = [x for x in cp["failures"] if int(x["user"]) != uid]
        existing.append({"user": int(uid), "error": repr(e)})
        cp["failures"] = existing
        cp_path.write_text(json.dumps(cp, indent=2))
        print(f"[{i}/{len(full_uids)}] user {uid}: ERROR {e}", flush=True)

rows = cp["rows"]
print(f"Completed full profiles: {len(rows)}/{len(full_uids)}")
print(f"Failures: {len(cp['failures'])}")
if len(rows) == len(full_uids):
    se = np.array([r["se_full"] for r in rows], dtype=float)
    med = float(np.median(se))
    print("Median se_full:", med)
    print("Full verdict:", g.verdict_for(med))
else:
    print("INCOMPLETE: rerun this stage; completed users are checkpointed.")
