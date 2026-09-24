#!/usr/bin/env bash
# check_ladder_provenance.sh — a ladder receipt's `bytes` and `sha256` must
# describe the SAME object (#3876).
#
# WHY. scripts/model_ladder.sh collects a model's two provenance fields on one
# line, and the two tools disagree about symlinks:
#
#     sha256sum "$ipath"     FOLLOWS the link  -> identifies the model
#     stat -c %s "$ipath"    does NOT          -> identifies the link
#
# Measured on lambda's 0.69.1 receipt: Qwen3-1.7B-Q4_K_M.gguf recorded
# `bytes: 60` beside `sha256: b139949c5…`. The link text is exactly 60
# characters; the sha is the 1.1 GB model's. One row, two objects.
#
# The cost was not the field, it was the false alarm: those rows read
# `green=true qa_rc=0 required=true bytes=60`, which reads exactly like a
# REQUIRED rung passing on a 60-byte stub, and a reader blames the gate for what
# the evidence misreported. It also manufactures a cross-host difference — the
# same model is a regular file on gx10 and a symlink on lambda, byte-identical
# sha256, `bytes` off by seven orders of magnitude.
#
# WHAT THIS CHECKS, AND WHY IT IS NOT A GREP. It extracts the collection line
# from model_ladder.sh and EVALUATES IT against a planted symlink, so the
# assertion is about the behaviour of the shipped code, not about a pattern that
# happens to appear in it. A future rewrite that resolves the path before stat-ing
# (`ipath=$(readlink -f …)`) is CORRECT and still passes; one that drops the
# follow fails. Per CLAUDE.md: assert the value, not the flag.
#
# It also pins the sha in the other direction. `sha256sum` following the link is
# the RIGHT behaviour — the receipt exists to identify the model, and a symlink's
# text is not the model — so "make them agree" must not be satisfied by hashing
# the link. Both directions are asserted.
#
# Exit: 0 both fields describe the model · 1 they do not · 2 could not check.
#       --self-test: 0 when the planted mutation turns this RED, 1 otherwise.
set -euo pipefail

SCRIPT="scripts/model_ladder.sh"
SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --script) [ $# -ge 2 ] || { echo "--script needs a value" >&2; exit 2; }; SCRIPT="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "check_ladder_provenance: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

# The one line under test, lifted from the script by the variables it assigns.
# Anti-vacuity: exactly one match, or this check has silently stopped testing
# anything. "no match" must never be "nothing to complain about".
extract_line() {
  local src="$1" n
  [ -f "$src" ] || { echo "  cannot read $src" >&2; return 2; }
  # #4131: the hash now goes through ladder_sha256 (identity-keyed cache), on ONE collection line.
  n=$(grep -c 'read -r isha isha_src < "\$WORK/.sha"; ibytes=' "$src" || true)
  if [ "$n" != "1" ]; then
    echo "  expected exactly 1 collection line in $src, found $n — this check no longer knows what to evaluate" >&2
    return 2
  fi
  grep -m1 'read -r isha isha_src < "\$WORK/.sha"; ibytes=' "$src"
}

# Plant a model and a symlink to it whose TARGET PATH LENGTH differs from the
# model's size, so the two candidate answers cannot be confused.
run_case() {
  local src="$1" tmp line ipath isha ibytes
  local want_bytes want_sha linklen rc=0

  tmp=$(mktemp -d)
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" RETURN

  mkdir -p "$tmp/real"
  # A size that is not a plausible path length, and not round.
  head -c 4097 /dev/urandom > "$tmp/real/model-target.gguf"
  ln -sfn "$tmp/real/model-target.gguf" "$tmp/model.gguf"

  want_bytes=$(stat -Lc %s "$tmp/model.gguf")
  want_sha=$(sha256sum "$tmp/real/model-target.gguf" | cut -d' ' -f1)
  linklen=$(readlink "$tmp/model.gguf" | tr -d '\n' | wc -c)

  if [ "$want_bytes" = "$linklen" ]; then
    echo "  the fixture is degenerate: target size equals link text length ($linklen)" >&2
    return 2
  fi

  line=$(extract_line "$src") || return 2

  ipath="$tmp/model.gguf"
  isha=""; ibytes=""
  # The shipped line, run verbatim -- SOURCED rather than `eval`ed, and that is
  # a fidelity fix before it is a lint fix. `eval "$line"` parses the text a
  # SECOND time, so a shipped line containing anything shell-significant after
  # one expansion would run differently HERE than it does in the script this
  # check exists to certify -- i.e. `eval` is the one form that is not verbatim.
  # Sourcing a file holding the line reproduces the shipped parse exactly.
  # Assignments still land in run_case's locals, which is what the asserts read.
  #
  # It also clears the SEC001 that check_bashrs_gate.sh (the release's
  # SEC/DET/IDEM gate) fails on. That finding was invisible on the PR because
  # guard-cargo failed an EARLIER step and GitHub skipped this one -- one defect
  # standing in front of another.
  # #4131: the line calls ladder_sha256, so the SHIPPED function is lifted too, with a scratch cache.
  awk '/^declare -A LADDER_SHA_MEMO=/{print} /^ladder_sha256\(\) \{/{f=1} f{print} f && /^\}$/{exit}' "$src" > "$tmp/shipped-fn.sh"
  grep -q '^ladder_sha256() {' "$tmp/shipped-fn.sh" || { echo "  $src defines no ladder_sha256() -- this check no longer knows what to evaluate" >&2; return 2; }
  local WORK="$tmp" LADDER_SHA_CACHE="$tmp/sha-cache.tsv"
  # ONE file, sourced ONCE: run_case's RETURN trap also fires when a `.` returns, and would delete
  # $tmp between two sources.
  { cat "$tmp/shipped-fn.sh"; printf '%s\n' "$line"; } > "$tmp/shipped-line.sh"
  # shellcheck source=/dev/null
  . "$tmp/shipped-line.sh"

  if [ "$ibytes" != "$want_bytes" ]; then
    echo "  BYTES describes the wrong object: recorded $ibytes, model is $want_bytes bytes"
    if [ "$ibytes" = "$linklen" ]; then
      echo "    (it recorded the length of the symlink's target PATH — exactly the #3876 defect)"
    fi
    rc=1
  fi
  if [ "$isha" != "$want_sha" ]; then
    echo "  SHA256 describes the wrong object: recorded ${isha:0:16}…, model is ${want_sha:0:16}…"
    echo "    (the hash must follow the link: the receipt identifies the MODEL, and a symlink's text is not the model)"
    rc=1
  fi
  return "$rc"
}

if [ "$SELF_TEST" = 1 ]; then
  # A check that has only ever seen correct input is indistinguishable from
  # `exit 0`. Plant the exact regression and require it to turn this RED.
  [ -f "$SCRIPT" ] || { echo "cannot read $SCRIPT" >&2; exit 2; }
  mutant=$(mktemp); trap 'rm -f "$mutant"' EXIT
  sed 's/stat -Lc %s "\$ipath"/stat -c %s "$ipath"/' "$SCRIPT" > "$mutant"
  if cmp -s "$SCRIPT" "$mutant"; then
    # No backticks in this message: it names a shell command, and in double
    # quotes backticks RUN it rather than print it (bashrs BRS0002, and the
    # same trap that rewrote a contract comment earlier in this release).
    echo 'SELF-TEST INCONCLUSIVE: the mutation changed nothing — the stat -Lc collection is absent from' "$SCRIPT" >&2
    exit 1
  fi
  echo "self-test: the shipped script"
  if run_case "$SCRIPT"; then echo "  GREEN (expected)"; else
    echo "SELF-TEST FAILED: the shipped script is already red" >&2; exit 1; fi
  echo "self-test: mutant A (-L removed — bytes falls back to the link)"
  if run_case "$mutant"; then
    echo "SELF-TEST FAILED: mutant A passed — this check does not discriminate" >&2; exit 1
  else
    echo "  RED (expected)"
  fi

  # Mutant B is the WRONG FIX. "Make the two fields agree" is also satisfied by
  # hashing the link, which would make both fields describe the symlink and this
  # check pass if it only compared them to each other. The receipt's job is to
  # identify the MODEL, so the sha is pinned to the followed file independently.
  # (#4131: the hash is computed inside ladder_sha256, so that is where the argument changes.)
  # The mutation must leave the collection line intact, or extract_line
  # stops matching and the case goes red on the anti-vacuity guard instead of on
  # the sha assertion — red for the wrong reason, which proves nothing. (It did
  # exactly that on the first attempt.) So only the ARGUMENT changes.
  mutant_b=$(mktemp); trap 'rm -f "$mutant" "$mutant_b"' EXIT
  sed 's|sha=\$(sha256sum "\$path"|sha=$(sha256sum <(readlink "$path" \| tr -d "\\n")|' "$SCRIPT" > "$mutant_b"
  if cmp -s "$SCRIPT" "$mutant_b"; then
    echo "SELF-TEST INCONCLUSIVE: mutant B changed nothing" >&2; exit 1
  fi
  echo "self-test: mutant B (sha hashes the link text instead of the model — the wrong fix)"
  b_out=$(run_case "$mutant_b" 2>&1) && {
    echo "SELF-TEST FAILED: mutant B passed — the check would accept both fields describing the symlink" >&2
    exit 1
  }
  # And red for the RIGHT reason: the sha assertion, not the extraction guard.
  case "$b_out" in
    *"SHA256 describes the wrong object"*) echo "  RED on the sha assertion (expected)" ;;
    *) echo "SELF-TEST FAILED: mutant B was red, but not on the sha assertion — the mechanism did not engage:" >&2
       echo "$b_out" >&2; exit 1 ;;
  esac

  echo "self-test: PASS — red on the regression AND on the wrong fix"
  exit 0
fi

echo "ladder provenance: bytes and sha256 must describe the same object ($SCRIPT)"
if run_case "$SCRIPT"; then
  echo "OK: a symlinked model records the MODEL's size and the MODEL's hash"
  exit 0
else
  rc=$?
  [ "$rc" = 2 ] && exit 2
  echo "FAIL: a symlinked model's provenance fields disagree about which object they describe (#3876)"
  exit 1
fi
