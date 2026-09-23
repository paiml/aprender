#!/usr/bin/env bash
# candidate_watch.sh -- SHIFT-LEFT (#4045 M8): every publish-blocking gate, on the release CANDIDATE, from the freeze.
#
# Operator 2026-09-23 (via the release cop): "all of these trivial issues should be caught hours early and are "fake"
# gates". 0.69.1 met ten complexity functions, a stale census, eight bashrs findings, CB-200, claim literals, the
# ladder scope, the R4 squash and the clean-room dispatch for the FIRST time at publish. From 0.70 this runs on
# every push to the release branch and at least hourly, on its HEAD (the candidate), and the publish step only
# re-reads a verdict that has been green for hours.
#
#   bash scripts/release/candidate_watch.sh <version> [--state <dir>] [--post <issue number>]
#
# PER RUN, on this checkout's HEAD:
#   0. refuse if a user-scope skill shadows a repo skill (scripts/check_no_shadowed_repo_skill.sh): REAL.
#   1. the publish-blocking gate set, each row CLASSIFIED by scripts/release/gate_classes.yaml
#      (scripts/lib/release_gate_classes.py derives the set; an unclassified gate is treated as REAL, never skipped):
#        dogfood --phase pre-publish (every dogfood row, the declared gates included)
#        check_publish_preflight.sh --receipt-only (R2 + R5, the rules knowable before the tag)
#        bump-version.sh --check (the version agreement autopilot checks after `wait`)
#   2. a table: gate, class, verdict, FIRST-RED-AT (kept per version in <state>, so a red that sat for hours shows its
#      age), written to <state>/<version>/watch-<ts>.{json,md}; --post comments it on the release issue.
#   3. exit 1 when any REAL gate is red (andon). A BOOKKEEPING red is reported and counted, never exit 1: it must not
#      block the publish. exit 2: usage / ENV (a gate that could not run is RED, not skipped).
#
# Seams for the case table (never set in production): WATCH_DOGFOOD_CMD (writes a dogfood receipt JSON path to
# stdout), WATCH_PREFLIGHT_CMD, WATCH_BUMP_CMD, WATCH_GH, WATCH_HOME (the home the shadow guard reads),
# WATCH_AUTOFIX_CMD (the fixer; its own table is scripts/check_bookkeeping_autofix.sh -- it switches branches, so a
# test never runs it on a developer's tree).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
PROG=candidate_watch
die() { printf '%s: %s\n' "$PROG" "$1" >&2; exit 2; }
VERSION="${1:-}"; shift || true
[ -n "$VERSION" ] || die "usage: candidate_watch.sh <version> [--state <dir>] [--post <issue>]"
STATE="${HOME}/.local/state/aprender-candidate-watch"; POST=""; AUTOFIX=0
while [ $# -gt 0 ]; do
  case "$1" in
    --state) STATE="$2"; shift 2 ;;
    --post) POST="$2"; shift 2 ;;
    --autofix) AUTOFIX=1; shift ;;   # a bookkeeping red with a fixer gets a PROPOSED commit (bookkeeping_autofix.sh)
    *) die "unknown argument '$1'" ;;
  esac
done
[[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "version '$VERSION' is not X.Y.Z"
case "$STATE" in /*) ;; *) die "--state must be an absolute path" ;; esac
case "$STATE" in *..*) die "--state must not contain '..'" ;; esac
SHA=$(git rev-parse --verify HEAD) || die "not a git checkout"
D="$STATE/$VERSION"; mkdir -p "$D" || die "cannot create $D"  # bashrs disable-line=SEC010
TS=$(date -u +%Y%m%dT%H%M%SZ)  # bashrs disable-line=DET002 (names this run's watch log, never a build artifact)
RESULTS="$D/results-$TS.tsv"; : > "$RESULTS"

# 0. the shadow guard
if ! bash scripts/check_no_shadowed_repo_skill.sh ${WATCH_HOME:+--home "$WATCH_HOME"} > "$D/shadow-$TS.log" 2>&1; then
  printf 'shadow:skills\tFAIL\t%s\n' "$(grep -m1 SHADOWED "$D/shadow-$TS.log")" >> "$RESULTS"
fi

# 1. the gate set on the candidate
if [ -n "${WATCH_DOGFOOD_CMD:-}" ]; then receipt=$(bash -c "$WATCH_DOGFOOD_CMD" 2> "$D/dogfood-$TS.log")
else
  bash scripts/dogfood.sh --phase pre-publish > "$D/dogfood-$TS.log" 2>&1
  receipt=$(sed -n 's/^receipt: //p' "$D/dogfood-$TS.log" | tail -1)
fi
python3 - "$receipt" "$SHA" >> "$RESULTS" <<'PY'
import json, sys
p, sha = sys.argv[1], sys.argv[2]
try:
    r = json.load(open(p))
except (OSError, ValueError, TypeError):
    print("step:dogfood\tFAIL\tno readable dogfood receipt (%r) -- the gate set could not be read, which is RED" % p); sys.exit(0)
if r.get("commit") and not sha.startswith(r["commit"][:7]):
    print("step:dogfood\tFAIL\tthe dogfood receipt is for %s, not the candidate %s" % (r["commit"], sha[:9])); sys.exit(0)
for g in r.get("gates") or []:
    print("dogfood:%s\t%s\t%s" % (g.get("gate"), g.get("result"), str(g.get("note") or "").replace("\t", " ").replace("\n", " ")[:160]))
PY
run_rc() { # run_rc <gate id> <cmd...>: one TSV row from an exit code
  local id=$1 rc; shift
  "$@" > "$D/${id//[:\/]/_}-$TS.log" 2>&1; rc=$?
  printf '%s\t%s\t%s\n' "$id" "$([ "$rc" = 0 ] && echo PASS || echo FAIL)" "exit $rc -- $(tail -1 "$D/${id//[:\/]/_}-$TS.log" | cut -c1-140)" >> "$RESULTS"
}
# --receipt-only is R2 + R5, the preflight rules knowable before the tag; recorded as R5 (the dogfood receipt)
if [ -n "${WATCH_PREFLIGHT_CMD:-}" ]; then run_rc preflight:R5 bash -c "$WATCH_PREFLIGHT_CMD"
else run_rc preflight:R5 bash scripts/check_publish_preflight.sh --receipt-only; fi
# the version-agreement check autopilot runs right after `wait` (autopilot.sh: bump-version.sh --check)
if [ -n "${WATCH_BUMP_CMD:-}" ]; then run_rc step:wait bash -c "$WATCH_BUMP_CMD"
else run_rc step:wait bash scripts/bump-version.sh --check; fi

# 2 + 3. classify, age, table, verdict
python3 - "$RESULTS" "scripts/release/gate_classes.yaml" "$D/first-red.json" "$D/watch-$TS" "$SHA" "$VERSION" <<'PY'
import json, os, sys, time, yaml
res_p, cls_p, first_p, out, sha, version = sys.argv[1:7]
classes = (yaml.safe_load(open(cls_p)) or {}).get("gates") or {}
try:
    first = json.load(open(first_p))
except (OSError, ValueError):
    first = {}
RED = {"FAIL", "NO-GO"}
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
rows, real_red, book_red = [], [], []
for ln in open(res_p):
    if not ln.strip():
        continue
    gid, verdict, note = (ln.rstrip("\n").split("\t") + ["", ""])[:3]
    c = (classes.get(gid) or {}).get("class")
    cls = c if c in ("real", "bookkeeping") else "real"   # UNCLASSIFIED is real: never quietly waived
    if gid == "shadow:skills":
        cls = "real"
    red = verdict in RED
    if red:
        first.setdefault(gid, {"at": now, "sha": sha})
        (real_red if cls == "real" else book_red).append(gid)
    else:
        first.pop(gid, None)
    rows.append({"gate": gid, "class": cls + ("" if c or gid == "shadow:skills" else " (UNCLASSIFIED)"), "verdict": verdict,
                 "first_red_at": first.get(gid, {}).get("at") if red else None, "note": note})
json.dump(first, open(first_p, "w"), indent=1)
doc = {"schema": "apr-candidate-watch/v1", "version": version, "sha": sha, "at": now, "rows": rows,
       "real_red": real_red, "bookkeeping_red": book_red, "andon": bool(real_red)}
json.dump(doc, open(out + ".json", "w"), indent=1)
md = ["**Candidate watch** `%s` at `%s` (%s) -- %s" % (version, sha[:9], now,
      "ANDON: %d REAL gate(s) red" % len(real_red) if real_red else "no real gate red"), "",
      "| gate | class | verdict | red since |", "|---|---|---|---|"]
for r in sorted(rows, key=lambda r: (r["verdict"] not in RED, not r["class"].startswith("real"), r["gate"])):
    if r["verdict"] in RED or r["class"].startswith("real"):
        md.append("| %s | %s | %s | %s |" % (r["gate"], r["class"], r["verdict"], r["first_red_at"] or ""))
md.append("")
md.append("%d gate(s) read; %d real red; %d bookkeeping red (reported, never a publish block)." % (len(rows), len(real_red), len(book_red)))
open(out + ".md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
sys.exit(1 if real_red else 0)
PY
rc=$?
# Bookkeeping reds are never waived: with --autofix each one that has a fixer gets a proposed commit, and the proposal
# is appended to the table the release issue reads.
if [ "$AUTOFIX" = 1 ]; then
  fixers=$(python3 - "$D/watch-$TS.json" <<'PY'
import json, sys
FIX = {"dogfood:contracts": ["census", "readme"], "dogfood:declared:check_no_claim_literals": ["claims"],
       "dogfood:pmat-verify": ["complexity"]}
w = json.load(open(sys.argv[1]))
print(" ".join(sorted({f for g in w.get("bookkeeping_red") or [] for f in FIX.get(g, [])})))
PY
)
  if [ -n "$fixers" ]; then
    # shellcheck disable=SC2086
    ${WATCH_AUTOFIX_CMD:-bash scripts/release/bookkeeping_autofix.sh} $fixers > "$D/autofix-$TS.log" 2>&1
    { echo; echo "Bookkeeping auto-fixes (proposed commits, never a silent pass):"; sed 's/^/- /' "$D/autofix-$TS.log"; } >> "$D/watch-$TS.md"
    cat "$D/autofix-$TS.log"
  fi
fi
if [ -n "$POST" ]; then
  "${WATCH_GH:-gh}" issue comment "$POST" --repo paiml/aprender --body-file "$D/watch-$TS.md" > /dev/null 2>&1 \
    || echo "$PROG: could not post to #$POST" >&2
fi
exit "$rc"
