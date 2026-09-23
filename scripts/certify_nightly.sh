#!/usr/bin/env bash
# certify_nightly.sh -- the LONG certification, nightly, on one GPU host, at origin/main (#4040, #4045).
#
# Operator, 2026-09-23: "anything huge, must be nightly only" and "we change nightly for the long releases".
# The release gate stops re-running the full ladder and full CRUX. It admits a release only on a GREEN nightly,
# at most 24 h old, measured at an ANCESTOR of the cut. check_model_ladder.sh --nightly then binds those
# receipts to the cut through the #4037 carry-forward, or refuses them as STALE, and the release re-measures.
#
#   bash scripts/certify_nightly.sh --host <id> --root <dir> [--sha <commit>] [--certification <receipt>]
#        [--no-issue] [--self-test]
#
# PER NIGHT, PER HOST, into <root>/<sha>/<host>/:
#   1. a DETACHED worktree at <sha> (default: origin/main, fetched). The instrument is the commit being judged,
#      never whatever a session left checked out.
#   2. build `apr --features cuda --locked` into <root>/target (kept between nights: incremental, and never a
#      shared ~/src target), then PROVE it: `apr --version` must read "apr <version> (<sha9>)".
#   3. the full ladder: scripts/model_ladder.sh --host <id> (its own per-call GPU lock, #3712 row B).
#   4. full CRUX: scripts/crux_sweep_shards.sh --scope admitted (every admitted prompt of every certified model).
#      Its cells go through gpu-q at CRUX_GPU_PRIO, default 5 here: a nightly yields to every interactive lane.
#   5. verdict.json (apr-nightly-certification/v1): sha, host, version, the proved version line, t_start/t_end,
#      both receipts with their rc and verdict, the certification's sha256, and `green`. GREEN = the ladder
#      receipt exists at this apr_sha with executed >= 1 and red == 0, AND the CRUX receipt's verdict is PASS.
#      Anything else is RED, and a step that could not run is RED, never skipped.
#   6. RED files ONE issue per host ("nightly certification RED: <host>"), or comments on the open one.
#
# Idempotent: a verdict already present for (sha, host) exits 0 without measuring again.
# exit: 0 GREEN (or already judged green) . 1 RED . 2 usage/ENV
#
# The seams the self-test drives (never set in production): NIGHTLY_BUILD_CMD, NIGHTLY_LADDER_CMD,
# NIGHTLY_CRUX_CMD replace steps 2-4 with a command run in the worktree; NIGHTLY_GH replaces `gh`.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
REPO=$(pwd)
PROG=certify_nightly
die() { printf '%s: %s\n' "$PROG" "$1" >&2; exit 2; }

HOST=""; ROOT=""; SHA=""; CERT=""; ISSUE=1; SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --host) [ $# -ge 2 ] || die "--host needs a value"; HOST="$2"; shift 2 ;;
    --root) [ $# -ge 2 ] || die "--root needs a value"; ROOT="$2"; shift 2 ;;
    --sha) [ $# -ge 2 ] || die "--sha needs a value"; SHA="$2"; shift 2 ;;
    --certification) [ $# -ge 2 ] || die "--certification needs a value"; CERT="$2"; shift 2 ;;
    --no-issue) ISSUE=0; shift ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) die "unknown argument '$1'" ;;
  esac
done

# ---------------------------------------------------------------- self-test
if [ "$SELF_TEST" = 1 ]; then
  exec bash scripts/check_certify_nightly.sh
fi

[ -n "$HOST" ] && [ -n "$ROOT" ] || die "usage: --host <id> --root <dir> [--sha <commit>]"
GH="${NIGHTLY_GH:-gh}"
mkdir -p "$ROOT" || die "cannot create $ROOT"
ROOT=$(cd "$ROOT" && pwd)

if [ -z "$SHA" ]; then
  git fetch -q origin main 2> /dev/null || die "git fetch origin main failed -- the nightly never measures a guess"
  SHA=$(git rev-parse --verify --quiet origin/main) || die "origin/main does not resolve"
fi
SHA=$(git rev-parse --verify --quiet "$SHA^{commit}") || die "'$SHA' is not a commit"
SHA9=${SHA:0:9}
DIR="$ROOT/$SHA/$HOST"
if [ -f "$DIR/verdict.json" ]; then
  g=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("green"))' "$DIR/verdict.json" 2> /dev/null)
  if [ "$g" = True ]; then
    echo "$PROG: $HOST at $SHA9 already judged GREEN -- $DIR/verdict.json"; exit 0
  fi
  # A RED night is RE-MEASURED on the next run: a transient failure (build, lock, host) must not cost the day's
  # admission until main moves. The old verdict is kept beside it, never overwritten.
  mv -f "$DIR/verdict.json" "$DIR/verdict.$(date +%s).red.json" || die "cannot set the RED verdict aside"
  echo "$PROG: $HOST at $SHA9 was RED -- re-measuring (the old verdict is kept)"
fi
mkdir -p "$DIR" || die "cannot create $DIR"
T0=$(date +%s.%N)
LOG="$DIR/nightly.log"; : > "$LOG"

SRC="$ROOT/$SHA/src"
if [ ! -d "$SRC" ]; then
  git worktree add --detach --quiet "$SRC" "$SHA" >> "$LOG" 2>&1 || die "cannot check out $SHA9 at $SRC"
fi
VERSION=$(cd "$SRC" && cargo metadata --no-deps --offline --format-version 1 2> /dev/null | python3 -c '
import json, os, sys
m = json.load(sys.stdin)
for p in m.get("packages", []):
    if os.path.dirname(p["manifest_path"]) == m["workspace_root"]:
        print(p["version"]); break') || VERSION=""
[ -n "$VERSION" ] || VERSION=$(sed -n 's/^version = "\([0-9.]*\)"$/\1/p' "$SRC/Cargo.toml" | head -1)
[ -n "$VERSION" ] || die "the version at $SHA9 cannot be resolved"
if [ -z "$CERT" ]; then
  CERT=$(ls -1d "$SRC"/evidence/crux/*/prompt-certification.json 2> /dev/null | sort -V | tail -1)
fi

export NIGHTLY_ROOT="$ROOT" NIGHTLY_SHA="$SHA" NIGHTLY_VERSION="$VERSION" NIGHTLY_HOST="$HOST"   # for the self-test seams
step() { # step <name> <cmd...>: run in the worktree, log it, return its rc (never through a pipe)
  local name=$1 rc; shift
  printf -- '--- %s: %s\n' "$name" "$*" >> "$LOG"
  (cd "$SRC" && "$@") >> "$LOG" 2>&1; rc=$?
  printf -- '--- %s rc=%s\n' "$name" "$rc" >> "$LOG"
  return "$rc"
}

# 2. build + prove
APR="$ROOT/target/release/apr"
build_rc=0
if [ -n "${NIGHTLY_BUILD_CMD:-}" ]; then
  step build bash -c "$NIGHTLY_BUILD_CMD" || build_rc=$?
else
  CARGO_TARGET_DIR="$ROOT/target" step build cargo build --release -p apr-cli --bin apr --features cuda --locked || build_rc=$?
fi
want="apr $VERSION ($SHA9)"
got=$("$APR" --version 2> /dev/null | head -1)
proved=0; [ "$build_rc" = 0 ] && [ "$got" = "$want" ] && proved=1

# 3 + 4. measure (only a proved binary measures anything)
ladder_rc=""; crux_rc=""
if [ "$proved" = 1 ]; then
  ladder_rc=0
  if [ -n "${NIGHTLY_LADDER_CMD:-}" ]; then
    APR="$APR" OUT="$DIR/ladder" step ladder bash -c "$NIGHTLY_LADDER_CMD" || ladder_rc=$?
  else
    DOGFOOD_ALLOW_UNPINNED=1 APR="$APR" step ladder choom -n 1000 -- bash scripts/model_ladder.sh --host "$HOST" --out "$DIR/ladder" || ladder_rc=$?
  fi
  # BOTH lanes: every ladder rung claims cpu AND cuda, and the judge wants a CRUX verdict per backend. The gpu
  # lane takes gpu-q per cell; the cpu lane takes no GPU lock (crux_sweep_shards --backend cpu).
  crux_rc=0
  for lane in ${NIGHTLY_CRUX_LANES:-gpu cpu}; do
    lrc=0
    if [ -n "${NIGHTLY_CRUX_CMD:-}" ]; then
      LANE="$lane" APR="$APR" OUT="$DIR/crux" step "crux-$lane" bash -c "$NIGHTLY_CRUX_CMD" || lrc=$?
    elif [ -z "$CERT" ] || [ ! -f "$CERT" ]; then
      printf -- '--- crux-%s: no prompt-certification receipt under %s/evidence/crux/ -- not run\n' "$lane" "$SRC" >> "$LOG"; lrc=2
    else
      CRUX_GPU_PRIO="${CRUX_GPU_PRIO:-5}" step "crux-$lane" bash scripts/crux_sweep_shards.sh "$VERSION" --host "$HOST" --apr "$APR" \
        --backend "$lane" --out "$DIR/crux" --scope admitted --certification "$CERT" || lrc=$?
    fi
    [ "$lrc" = 0 ] || crux_rc=$lrc
  done
fi
T1=$(date +%s.%N)

# 5. verdict
python3 - "$DIR" "$SHA" "$HOST" "$VERSION" "$want" "$got" "$build_rc" "$proved" "${ladder_rc:-}" "${crux_rc:-}" \
    "$T0" "$T1" "${CERT:-}" "${NIGHTLY_CRUX_LANES:-gpu cpu}" > "$DIR/verdict.json.tmp" <<'PY' || die "the verdict could not be written"
import glob, hashlib, json, os, sys
d, sha, host, version, want, got, build_rc, proved, lrc, crc, t0, t1, cert, lanes = sys.argv[1:15]
lanes = lanes.split()
why = []
def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None
lad_p = os.path.join(d, "ladder", host + ".json")
lad = load(lad_p)
crux = {}
for lane in lanes:   # exactly <host>-<lane>.json per lane: the merge meta / plan / greedy files are not receipts
    p = os.path.join(d, "crux", "%s-%s.json" % (host, lane))
    crux[lane] = (p, load(p))
if proved != "1":
    why.append("the binary is not proved: build rc %s, `apr --version` read %r, want %r" % (build_rc, got, want))
else:
    if lad is None:
        why.append("no ladder receipt at %s (ladder rc %s)" % (lad_p, lrc))
    elif lad.get("apr_sha") != sha:
        why.append("the ladder receipt is at apr_sha %r, not %s" % (lad.get("apr_sha"), sha))
    elif int(lad.get("executed") or 0) < 1 or int(lad.get("red") or 0) != 0:
        why.append("the ladder is RED: executed=%s red=%s" % (lad.get("executed"), lad.get("red")))
    for lane, (p, r) in sorted(crux.items()):
        if r is None or r.get("schema") != "crux-inference-receipt/v1":
            why.append("no CRUX %s receipt at %s (crux rc %s)" % (lane, p, crc))
        elif (r.get("summary") or {}).get("verdict") != "PASS":
            s = r.get("summary") or {}
            why.append("CRUX %s is %s: %s" % (lane, s.get("verdict"), s.get("declined_because") or "RED %s" % s.get("RED")))
cert_sha = hashlib.sha256(open(cert, "rb").read()).hexdigest() if cert and os.path.isfile(cert) else None
print(json.dumps({
    "schema": "apr-nightly-certification/v1", "sha": sha, "host": host, "version": version,
    "apr_version_line": got or None, "t_start": float(t0), "t_end": float(t1),
    "ladder": {"rc": int(lrc) if lrc else None, "receipt": lad_p if lad is not None else None},
    "crux": {"rc": int(crc) if crc else None,
             "lanes": {lane: {"receipt": p if r is not None else None,
                              "verdict": (r.get("summary") or {}).get("verdict") if r else None} for lane, (p, r) in crux.items()}},
    "certification": {"path": cert or None, "sha256": cert_sha},
    "green": not why, "why": why}, indent=2))
PY
mv -f "$DIR/verdict.json.tmp" "$DIR/verdict.json" || die "cannot place $DIR/verdict.json"
green=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["green"])' "$DIR/verdict.json")
if [ "$green" = True ]; then
  echo "$PROG: GREEN $HOST at $SHA9 -- $DIR/verdict.json"
  exit 0
fi
why=$(python3 -c 'import json,sys; print("; ".join(json.load(open(sys.argv[1]))["why"]))' "$DIR/verdict.json")
echo "$PROG: RED $HOST at $SHA9 -- $why"

# 6. a RED nightly files (or comments on) ONE issue per host
if [ "$ISSUE" = 1 ]; then
  title="nightly certification RED: $HOST"
  body=$(printf 'Nightly certification (#4040) is RED on **%s** at `%s` (apr %s).\n\n%s\n\nVerdict: `%s`. Log: `%s`.\n' \
    "$HOST" "$SHA9" "$VERSION" "$why" "$DIR/verdict.json" "$LOG")
  num=$("$GH" issue list --repo paiml/aprender --state open --search "\"$title\" in:title" --json number,title \
        -q ".[] | select(.title == \"$title\") | .number" 2> /dev/null | head -1)
  if [ -n "$num" ]; then
    "$GH" issue comment "$num" --repo paiml/aprender --body "$body" > /dev/null 2>&1 \
      || echo "$PROG: could not comment on #$num" >&2
  else
    "$GH" issue create --repo paiml/aprender --title "$title" --body "$body" > /dev/null 2>&1 \
      || echo "$PROG: could not file the RED issue" >&2
  fi
fi
exit 1
