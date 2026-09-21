import json, sys, re, datetime
ap, mc, t, v, status = sys.argv[1:6]
s = open(status).read()
def grab(rx, default=None):
    # LAST match, not the first: STATUS is append-only and a step may be re-run.
    # Measured 2026-09-20 (v0.68.2): the first-match version recorded the FAILED
    # clean-room run 35471384263 and installer rc=1 from the attempt that died on a
    # missing receipts dir, while the green run 35495206021 and rc=0 sat later in the
    # same file. A receipt that reports the failed attempt is worse than no receipt.
    ms = re.findall(rx, s, re.M); return ms[-1] if ms else default
rec = {"spec": "APR-RELEASE-001", "section": "4", "job": "release-train", "host": "lambda-vector", "host_class": "lambda",
       "source": ap.rstrip("/").split("/")[-1] + "/STATUS", "sha": mc, "tag": t, "version": v,
       "published": bool(re.search(r"^\S+ PUBLISHED ", s, re.M)),
       "asset_run_id": grab(r"^\S+ ASSET RUN (\d+)"), "cleanroom_run_id": grab(r"^\S+ CLEANROOM RUN (\d+)"),
       "assets_on_release": bool(re.search(r"^\S+ ASSETS all present", s, re.M)),
       "preflight_pass": bool(re.search(r"^\S+ PREFLIGHT PASS", s, re.M)),
       "dogfood_attempts": len(re.findall(r"^\S+ START autopilot", s, re.M)), "attended_minutes": 0,
       "steps": {k: bool(re.search(r"^\S+ " + re.escape(k), s, re.M)) for k in
                 ["DEEP GO", "DOGFOOD GO", "TAGGED", "CLEANROOM GREEN", "ASSETS all present", "PREFLIGHT PASS", "PUBLISHED", "INSTALL cargo", "HOST gx10", "HOST yoga", "INSTALLER intel", "INSTALLER gx10"]},
       "installer_receipts": {h: grab(r"^\S+ INSTALLER " + h + r" rc=(\d+)") for h in ["intel", "gx10"]},
       "pp26_witness": (lambda m: {"host": m.get("host"), "cc": m.get("cc"), "commit": m.get("commit"), "started_utc": m.get("started_utc"), "status": m.get("status"), "sha256": m.get("sha256"), "nightly_producer": "cuda-nightly.yml gx10 NIGHTLY-RED since 2026-09-12 (#3096, 0.70); release witness taken on lambda sm_89"})(__import__("json").load(open(f"{ap}/wt/evidence/perf041/lambda/marker.json")) if __import__("os").path.exists(f"{ap}/wt/evidence/perf041/lambda/marker.json") else {}),
       "unmeasured": ["t4_wall_minutes: derive from STATUS timestamps CASCADE..PUBLISHED"],
       "written": datetime.datetime.now(datetime.timezone.utc).isoformat()}
out = f"{ap}/{mc[:9]}-lambda-vector-train.json"
open(out, "w").write(json.dumps(rec, indent=2) + "\n")
print("ledger record", out)
