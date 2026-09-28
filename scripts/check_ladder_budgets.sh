#!/usr/bin/env bash
# check_ladder_budgets.sh — a ladder cell over its declared byte/RSS/wall budget is RED (#4520 step 4).
#
# WHY. 2026-09-27 ~07:15Z: model_ladder.sh --host lambda re-read a 17 GB GGUF off the disk for every
# verb and held the operator's desktop at IO PSI 90% / load 190 for 15+ minutes. Nothing in the
# receipt said so: cost to the host was never measured, so it could never be a defect.
#
# WHAT THIS CHECKS, IN TWO LAYERS.
#   1. THE JUDGE (scripts/lib/ladder_budget.py) over a case table of meter records, against the
#      SHIPPED contract's ladder.budgets: every rule has a must-RED row and a must-GREEN row.
#   2. THE METER ON A REAL READ. The SHIPPED apr_locked (lifted from model_ladder.sh) runs a fake
#      `apr` whose `inspect` reads a planted model file off the disk: whole (must be RED on
#      header_bytes_read_max) and 4 KiB (must be GREEN). The file's page cache is dropped first, so
#      the bytes are real storage reads. If the whole-file control reads ~0 (a tmpfs TMPDIR, say),
#      the meter would be proving nothing here: exit 2, never a pass.
# --self-test plants a judge that ignores the header budget and requires this check to go RED.
#
# Exit: 0 all as expected · 1 a case landed wrong · 2 could not check.
set -uo pipefail
SELF_TEST=0
case "${1:-}" in --self-test) SELF_TEST=1 ;; "") ;; *) echo "unknown argument '$1'" >&2; exit 2 ;; esac
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "  cannot check: not in a repository" >&2; exit 2; }
cd "$ROOT" || exit 2
command -v flock >/dev/null && command -v choom >/dev/null || { echo "  cannot check: flock/choom absent" >&2; exit 2; }
python3 -c 'import yaml' 2>/dev/null || { echo "  cannot check: python3 yaml absent" >&2; exit 2; }

T=$(mktemp -d) || exit 2
trap 'rm -rf -- "${T:?}"' EXIT
cp scripts/lib/ladder_budget.py "$T/ladder_budget.py"
if [ "$SELF_TEST" = 1 ]; then   # the mutant: the header rule never fires
  sed -i 's/if verb in b\["header_verbs"\] and/if False and/' "$T/ladder_budget.py"
fi
fails=0
ok()  { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n' "$1"; fails=$((fails + 1)); }

# ── layer 1: the judge's case table ───────────────────────────────────────────
python3 - "$T" contracts/model-capability-ladder-v1.yaml > "$T/l1.out" <<'PY'
import sys, yaml
sys.path.insert(0, sys.argv[1]); import ladder_budget as L
b = L.load_budgets(yaml.safe_load(open(sys.argv[2])))
G = 1 << 30
def rec(cell, verb, br=0, rss=1 << 20, wall=1.0, fb=4 * G):
    return {"cell": cell, "verb": verb, "bytes_read": br, "peak_rss_bytes": rss, "wall_s": wall, "file_bytes": fb}
H = b["header_bytes_read_max"]; C = b["cell_bytes_read_max_factor"]; W = b["wall_s_max"]
rss_lim = lambda fb: int(b["peak_rss_max_factor"] * fb + b["peak_rss_slack_bytes"])
cases = [  # name, records, the budget that must fire (None = must be green)
    ("header-reads-whole-file", [rec("a", "inspect", br=4 * G)], "header_bytes_read_max"),
    ("header-reads-4k",         [rec("a", "inspect", br=4096)], None),
    ("header-at-limit",         [rec("a", "inspect", br=H)], None),
    ("header-one-over",         [rec("a", "inspect", br=H + 1)], "header_bytes_read_max"),
    ("load-verb-reads-file",    [rec("a", "run", br=4 * G)], None),
    ("cell-rereads-per-verb",   [rec("a", v, br=4 * G) for v in ("qa", "run", "chat", "serve")], "cell_bytes_read"),
    ("cell-reads-once",         [rec("a", "qa", br=4 * G), rec("a", "run"), rec("a", "chat")], None),
    ("cell-at-factor",          [rec("a", "qa", br=int(C * 4 * G))], None),
    ("two-cells-not-summed",    [rec("a", "qa", br=5 * G), rec("b", "qa", br=5 * G)], None),
    ("rss-over",                [rec("a", "run", rss=rss_lim(4 * G) + 1)], "peak_rss"),
    ("rss-at-limit",            [rec("a", "run", rss=rss_lim(4 * G))], None),
    ("wall-over-run",           [rec("a", "run", wall=W + 1)], "wall_s_max"),
    ("wall-over-serve-exempt",  [rec("a", "serve", wall=W + 1)], None),
    ("no-file-bytes",           [rec("a", "run", fb=None)], "file_bytes"),
    ("no-cell-not-judged",      [rec(None, "inspect", br=4 * G)], None),
]
for name, recs, want in cases:
    got = {v["budget"] for v in L.judge(recs, b)}
    good = (not got) if want is None else (want in got and len(got) == 1)
    print(("ok   " if good else "FAIL ") + f" {name}: want {want or 'green'}, got {sorted(got) or 'green'}")
for name, mut in [("budgets-absent", lambda d: d["ladder"].pop("budgets")),
                  ("budget-zero", lambda d: d["ladder"]["budgets"].__setitem__("wall_s_max", 0)),
                  ("header-verbs-empty", lambda d: d["ladder"]["budgets"].__setitem__("header_verbs", []))]:
    d = yaml.safe_load(open(sys.argv[2])); mut(d)
    try:
        L.load_budgets(d); print(f"FAIL  {name}: accepted, must decline")
    except ValueError:
        print(f"ok    {name}: declines")
PY
[ $? = 0 ] || { echo "  cannot check: layer 1 crashed" >&2; exit 2; }
while IFS= read -r l; do case "$l" in ok*) ok "${l#ok    }" ;; *) bad "${l#FAIL  }" ;; esac; done < "$T/l1.out"

# ── layer 2: the shipped apr_locked + meter on real storage reads ─────────────
cp scripts/lib/ladder_meter.py "$T/ladder_meter.py"
if [ "$SELF_TEST" = 1 ]; then   # second mutant: the meter never enforces the RSS cap
  sed -i 's/            if seen > cap:/            if False:/' "$T/ladder_meter.py"
  grep -q 'if False:' "$T/ladder_meter.py" || { echo "  cannot check: self-test meter mutation did not apply" >&2; exit 2; }
fi
line=$(sed -n '/^apr_locked() {/p' scripts/model_ladder.sh)
eval "${line//scripts\/lib\/ladder_meter.py/$T\/ladder_meter.py}"
declare -F apr_locked >/dev/null || { echo "  cannot check: apr_locked not found in model_ladder.sh" >&2; exit 2; }
GPU_LOCK="$T/gpu.lock"; LOCK_WAIT=5; LOCK_BUSY=75
MODEL="$T/model.gguf"
head -c $((128 << 20)) /dev/urandom > "$MODEL" || exit 2
cat > "$T/apr" <<'SH'
#!/usr/bin/env bash
# fake apr: `inspect <file>` reads $FAKE_READ bytes of the file (all = the whole file)
[ "$1" = inspect ] || exit 0
if [ "${FAKE_READ:-all}" = all ]; then cat -- "$2" > /dev/null; else head -c "$FAKE_READ" -- "$2" > /dev/null; fi
SH
chmod +x "$T/apr"; APR="$T/apr"
drop_cache() { python3 -c 'import os,sys; fd=os.open(sys.argv[1], os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)' "$MODEL"; }
sync -f "$MODEL" 2>/dev/null || sync
export LADDER_METER="$T/meter.jsonl" LADDER_METER_CELL=planted LADDER_METER_FILE_BYTES; LADDER_METER_FILE_BYTES=$(stat -Lc %s "$MODEL")
: > "$LADDER_METER"
drop_cache; FAKE_READ=all apr_locked inspect "$MODEL" || { echo "  cannot check: apr_locked failed" >&2; exit 2; }
if [ ! -s "$LADDER_METER" ]; then   # apr_locked no longer meters: that is the regression, not an environment gap
  bad "meter: the shipped apr_locked wrote no meter record -- every apr call would go unjudged"
  echo "ladder budgets: $fails case(s) wrong"; exit 1
fi
whole=$(python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).readline())["bytes_read"])' "$LADDER_METER")
if [ "$whole" -lt $((64 << 20)) ]; then
  echo "  cannot check: the whole-file control read $whole storage bytes of 128 MiB -- is TMPDIR tmpfs? the meter proves nothing here" >&2; exit 2
fi
python3 "$T/ladder_budget.py" contracts/model-capability-ladder-v1.yaml "$LADDER_METER" > "$T/v1"; rc=$?
if [ $rc = 1 ] && grep -q '"budget": "header_bytes_read_max"' "$T/v1"; then ok "meter: planted inspect reading the whole file ($whole B off disk) is RED on header_bytes_read_max"
else bad "meter: planted inspect reading the whole file ($whole B) was not RED on header_bytes_read_max (judge rc=$rc)"; fi
: > "$LADDER_METER"
drop_cache; FAKE_READ=4096 apr_locked inspect "$MODEL"
python3 "$T/ladder_budget.py" contracts/model-capability-ladder-v1.yaml "$LADDER_METER" > "$T/v2"; rc=$?
if [ $rc = 0 ]; then ok "meter: planted inspect reading 4 KiB is GREEN"
else bad "meter: planted inspect reading 4 KiB was judged over budget (rc=$rc: $(head -c 300 "$T/v2"))"; fi
# the meter passes the child's status through: a verb's rc is still the verb's
FAKE_READ=4096 apr_locked inspect /nonexistent-4520 2>/dev/null; rc=$?
[ $rc = 1 ] && ok "meter: the child's exit status passes through (rc=1)" || bad "meter: child rc 1 came back as $rc"

# THE RSS CAP IS ENFORCED, not only judged: a call that grows past factor x file + slack is killed
# while it runs (gx10 cpu 27B: 58.6G against a 33.1G budget ran ~57 min into the box OOM).
cat > "$T/hog" <<'SH'
#!/usr/bin/env bash
# fake apr: holds $HOG_MB of touched memory for $HOG_S seconds
python3 -c 'import sys,time; b=bytearray(int(sys.argv[1])<<20); b[::4096]=b"x"*len(b[::4096]); time.sleep(float(sys.argv[2]))' "$HOG_MB" "${HOG_S:-20}"
SH
chmod +x "$T/hog"
: > "$LADDER_METER"
# budget 1.5 x 1 MiB + 64 MiB = 65.5 MiB; the kill cap is 1.5 x that = ~98 MiB
export LADDER_METER_RSS_FACTOR=1.5 LADDER_METER_RSS_SLACK=$((64 << 20)) LADDER_METER_FILE_BYTES=$((1 << 20)) LADDER_METER_RSS_KILL=1.5
APR="$T/hog"; s=$SECONDS; HOG_MB=256 apr_locked run x 2>/dev/null; rc=$?; dt=$((SECONDS - s))
capped=$(python3 -c 'import json,sys; r=json.loads(open(sys.argv[1]).readline()); print(r.get("rss_capped", 0))' "$LADDER_METER" 2>/dev/null)
if [ "$rc" = 137 ] && [ "${capped:-0}" -gt $((65 << 20)) ] && [ "$dt" -lt 10 ]; then ok "rss cap: a 256 MiB call over a ~98 MiB kill cap was killed in ${dt}s (rc 137, rss_capped=$capped)"
else bad "rss cap: a 256 MiB call over a ~98 MiB kill cap was not stopped (rc=$rc, rss_capped=${capped:-none}, ${dt}s)"; fi
: > "$LADDER_METER"
HOG_MB=8 HOG_S=1 apr_locked run x 2>/dev/null; rc=$?
if [ "$rc" = 0 ] && [ -s "$LADDER_METER" ] && ! grep -q rss_capped "$LADDER_METER"; then ok "rss cap: an 8 MiB call under the budget is left alone (rc 0)"
else bad "rss cap: an 8 MiB call under the budget was disturbed (rc=$rc)"; fi
# over the budget but under the kill cap (gx10 cpu 8B: 15.8G vs a 15.0G budget): the call FINISHES, so its
# functional verdict survives, and the judge still calls it RED on peak_rss -- the kill is not the judge
: > "$LADDER_METER"
HOG_MB=80 HOG_S=1 apr_locked run x 2>/dev/null; rc=$?
judged=$(python3 - "$T" "$LADDER_METER" <<'PY2'
import json, sys
sys.path.insert(0, sys.argv[1]); import ladder_budget as L
r = json.loads(open(sys.argv[2]).readline()); r.setdefault("cell", "hog")
b = {"header_verbs": [], "header_bytes_read_max": 1, "peak_rss_max_factor": 1.5, "peak_rss_slack_bytes": 64 << 20,
     "wall_s_max": 1e9, "cell_bytes_read_max_factor": 1e9}
print(",".join(v["budget"] for v in L.judge([r], b)) or "none")
PY2
)
if [ "$rc" = 0 ] && ! grep -q rss_capped "$LADDER_METER" && [ "$judged" = peak_rss ]; then ok "rss cap: an 80 MiB call over the 65.5 MiB budget but under the kill cap finishes (rc 0) and is judged RED on peak_rss"
else bad "rss cap: an 80 MiB over-budget call: rc=$rc judged=${judged:-crash} (want rc 0, not capped, judged peak_rss)"; fi
unset LADDER_METER_RSS_FACTOR LADDER_METER_RSS_SLACK LADDER_METER_RSS_KILL

if [ "$SELF_TEST" = 1 ]; then
  [ "$fails" -gt 0 ] && { echo "self-test: the planted blind judge turned this RED ($fails case(s)) -- good"; exit 0; }
  echo "self-test: FAIL -- a judge that ignores the header budget passed every case"; exit 1
fi
[ "$fails" = 0 ] && { echo "ladder budgets: all cases as expected"; exit 0; }
echo "ladder budgets: $fails case(s) wrong"; exit 1
