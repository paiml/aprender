#!/usr/bin/env bash
# scripts/lab/track_d_nonblocking_audit.sh -- aprender#6617 (PP-066 Track D). Rule: Track D documentation
# checks are non-blocking; a RED there is a finding, never a hold. Enforced here as: no job whose id or name
# marks documentation (doc, docs, doctest, citation, cite) is in the `needs:` of the ci.yml `gate` job, so a
# doc check cannot hold a merge. A LAB check: run on demand or at night, never inside a PR, merge-queue or
# release job (C324). Promotion to a gate needs the operator's yes, 3 green nights and one gate retired.
# Usage: scripts/lab/track_d_nonblocking_audit.sh [ci.yml] | --self-test      Exit: 0 green, 1 RED, 2 not measured
set -euo pipefail
DOCRE='(^|[-_])(docs?|doctests?|citations?|cites?)([-_]|$)'
audit() { # ci.yml -> RED lines on stdout; exit 2 when there is no gate job to read
  awk -v re="$DOCRE" '
    /^jobs:/ { in_jobs = 1; next }
    in_jobs && /^  [A-Za-z0-9_-]+:[[:space:]]*(#.*)?$/ { name = $1; sub(/:$/, "", name); isgate = (name == "gate"); innd = 0; if (isgate) seen = 1; next }
    in_jobs && isgate && /^    needs:/ { s = $0; sub(/^    needs:[[:space:]]*/, "", s); gsub(/[\[\],]/, " ", s); need = need " " s; innd = (s ~ /^[[:space:]]*$/); next }
    in_jobs && isgate && innd && /^      - / { s = $0; sub(/^      - /, "", s); need = need " " s; next }
    in_jobs && isgate && /^    [A-Za-z]/ { innd = 0 }
    END {
      if (!seen) exit 2
      n = split(need, a, " ")
      for (i = 1; i <= n; i++) if (tolower(a[i]) ~ re) print "RED: gate needs documentation job \"" a[i] "\" (Track D is non-blocking)"
    }' "$1"
}
run() {
  local f=${1:-.github/workflows/ci.yml} out rc=0
  [ -s "$f" ] || { echo "NOT MEASURED: $f missing or empty"; return 2; }
  out=$(audit "$f") || rc=$?
  [ "$rc" = 2 ] && { echo "NOT MEASURED: no gate job in $f"; return 2; }
  [ -n "$out" ] && { echo "$out"; return 1; }
  echo "GREEN: no documentation job in gate needs ($f)"; return 0
}
if [ "${1:-}" = "--self-test" ]; then
  T=$(mktemp -d); trap 'rm -rf "${T:?}"' EXIT; fails=0
  want() { local rc=0; run "$2" >/dev/null || rc=$?; if [ "$rc" = "$1" ]; then echo "ok   $3"; else echo "FAIL $3 (want $1 got $rc)"; fails=$((fails+1)); fi; }
  printf 'jobs:\n  build:\n    runs-on: x\n  gate:\n    needs: [build, tests]\n' > "$T/a.yml";            want 0 "$T/a.yml" "inline needs, no doc job: GREEN"
  printf 'jobs:\n  gate:\n    needs: [build, docs]\n' > "$T/b.yml";                                         want 1 "$T/b.yml" "planted inline doc job: RED"
  printf 'jobs:\n  gate:\n    needs:\n      - build\n      - doc-citations\n' > "$T/c.yml";              want 1 "$T/c.yml" "planted block-list doc job: RED"
  printf 'jobs:\n  docs:\n    runs-on: x\n  gate:\n    needs: [build]\n' > "$T/d.yml";                    want 0 "$T/d.yml" "doc job exists but gate does not need it: GREEN"
  printf 'jobs:\n  build:\n    runs-on: x\n' > "$T/e.yml";                                              want 2 "$T/e.yml" "no gate job: NOT MEASURED, not green"
  : > "$T/f.yml";                                                                                        want 2 "$T/f.yml" "empty file: NOT MEASURED, not green"
  printf 'jobs:\n  gate:\n    needs: [build, unit-tests]\n' > "$T/g.yml";                                  want 0 "$T/g.yml" "'unit-tests' is not a doc job (no substring match)"
  echo "track_d_nonblocking_audit self-test: $fails failure(s)"; [ "$fails" = 0 ]
  exit
fi
run "${1:-}"
