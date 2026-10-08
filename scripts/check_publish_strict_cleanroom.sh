#!/usr/bin/env bash
# publish_strict.sh is the one door to crates.io, and it must refuse a clean-room run id
# that is missing, red, or tested another commit (PMAT-4687). Before #4687 it accepted any
# non-empty file. This is its falsifier: the REAL publish_strict.sh runs against a fixture
# tag checkout, a fixture state dir (RELEASE_AP) and a stub `gh` and `cargo` first on PATH.
#
# Every row stops before an upload. The green row gets past the clean-room check and then
# stops on the absent dry-run receipt (C333: the B2-gpu run id left the release path), which proves the check let it through. The stub
# `cargo` fails the table if it is ever called.
#
#   bash scripts/check_publish_strict_cleanroom.sh
set -euo pipefail

case "${1:-}" in
  -h|--help) echo "usage: $0   (runs the publish_strict clean-room case table; no arguments)"; exit 0 ;;
  '') : ;;
  *) echo "usage: $0" >&2; exit 2 ;;
esac

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PUBLISH="${PUBLISH_STRICT_UNDER_TEST:-$REPO_ROOT/scripts/release/publish_strict.sh}"
echo "=== publish_strict clean-room door (check_publish_strict_cleanroom.sh) ==="
[ -f "$PUBLISH" ] || { echo "FAIL: $PUBLISH not found"; exit 1; }

WORK=$(mktemp -d)
case "$WORK" in
  /tmp/*|/var/folders/*) : ;;
  *) echo "FAIL: mktemp -d returned an unexpected path '$WORK'; refusing to clean up"; exit 1 ;;
esac
trap 'rm -rf "${WORK:?}"' EXIT

# ── stubs: gh answers from the row's fixture dir; the `cargo` stub must never be called ──
BIN="$WORK/bin"
mkdir -p "$BIN" "$WORK/ghconfig" "$WORK/home/.cargo"
printf 'fixture, not a token\n' > "$WORK/home/.cargo/credentials.toml"
cat > "$BIN/gh" <<'STUB'
#!/usr/bin/env bash
d=${STUB_DIR:?stub gh called without STUB_DIR}
printf '%s\n' "$*" >> "$d/calls"
case "$1 ${2:-}" in
  "auth status") exit 0 ;;
  "run view") cat "$d/jobs-${3:-x}.json" 2>/dev/null || exit 1 ;;
  "api repos/paiml/infra/actions/runs/"*) cat "$d/run-${2#repos/paiml/infra/actions/runs/}.txt" 2>/dev/null || exit 1 ;;
  "api repos/paiml/infra/actions/jobs/"*/logs)
    jid=${2#repos/paiml/infra/actions/jobs/}; cat "$d/log-${jid%/logs}.txt" 2>/dev/null || exit 1 ;;
  *) echo "UNEXPECTED $*" >> "$d/calls"; exit 97 ;;
esac
STUB
cat > "$BIN/cargo" <<'STUB'
#!/usr/bin/env bash
printf 'CARGO %s\n' "$*" >> "${STUB_DIR:?}/calls"; exit 97
STUB
chmod +x "$BIN/gh" "$BIN/cargo"

# ── fixture tag checkout: the tagged commit A, and B, a commit that is not the tag ──
mk_wt() { # mk_wt DIR: a repo detached at v1.2.3
  local w=$1 c
  git init -q "$w"
  for c in base:2026-09-01T00:00:00Z tagged:2026-09-01T01:00:00Z; do
    GIT_AUTHOR_DATE=${c#*:} GIT_COMMITTER_DATE=${c#*:} git -C "$w" -c core.hooksPath=/dev/null \
      -c user.name=t -c user.email=t@t -c commit.gpgsign=false commit -q --allow-empty -m "${c%%:*}"
  done
  git -C "$w" tag v1.2.3
  git -C "$w" checkout -q --detach v1.2.3
}
OTHER=$(printf '%040d' 0 | tr 0 b)

rc=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; rc=1; }

S="" AP=""
fx_run() { # fx_run CONCL TESTED_SHA: recorded run 9001, job 501
  printf '.github/workflows/clean-room.yml\n[{"databaseId":9001,"status":"completed","conclusion":"%s","createdAt":"2026-09-02T00:00:00Z"}]\n' "$1" > "$S/run-9001.txt"
  printf '{"jobs":[{"databaseId":501,"name":"clean-room (aprender)","status":"completed","conclusion":"%s","steps":[{"name":"Assert the commit under test","status":"completed","conclusion":"success"}]}]}\n' "$1" > "$S/jobs-9001.json"
  printf '2026-09-02T00:00:02.2000000Z     tested-sha: %s\n' "$2" > "$S/log-501.txt"
}
f_missing()  { :; }
f_empty()    { : > "$AP/cleanroom-run-id"; }
f_garbage()  { printf 'see the clean-room run\n' > "$AP/cleanroom-run-id"; }
f_red()      { printf '9001\n' > "$AP/cleanroom-run-id"; fx_run failure "$(git -C "$AP/wt" rev-parse HEAD)"; }
f_other()    { printf '9001\n' > "$AP/cleanroom-run-id"; fx_run success "$OTHER"; }
f_unknown()  { printf '9002\n' > "$AP/cleanroom-run-id"; fx_run success "$(git -C "$AP/wt" rev-parse HEAD)"; }
f_green()    { printf '9001\n' > "$AP/cleanroom-run-id"; fx_run success "$(git -C "$AP/wt" rev-parse HEAD)"; }

# row NAME NEEDLE SETUP -- publish_strict.sh must exit non-zero and say NEEDLE
row() {
  local name=$1 needle=$2 setup=$3 out got=0
  S="$WORK/stub-$name"; AP="$WORK/ap-$name"; mkdir -p "$S" "$AP"; : > "$S/calls"
  mk_wt "$AP/wt"
  "$setup"
  out=$(
    export PATH="$BIN:$PATH" STUB_DIR="$S" GH_CONFIG_DIR="$WORK/ghconfig" HOME="$WORK/home" \
      CARGO_HOME="$WORK/home/.cargo" RELEASE_AP="$AP"
    unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN
    bash "$PUBLISH" 1.2.3 2>&1
  ) || got=$?
  if grep -q '^UNEXPECTED\|^CARGO' "$S/calls"; then
    fail "$name: a call the table does not allow: $(grep '^UNEXPECTED\|^CARGO' "$S/calls" | head -1)"
  elif [ "$setup" != f_green ] && [[ "$out" == *"no committed dry-run receipt"* ]]; then
    fail "$name: the clean-room door let this run through (it stopped later, on the dry-run receipt)"
  elif [ "$got" -eq 0 ]; then
    fail "$name: publish_strict.sh exited 0"; printf '      | %s\n' "$out"
  elif [[ "$out" != *"$needle"* ]]; then
    fail "$name: rc=$got but output never said '$needle'"; printf '      | %s\n' "$out" | tail -n 4
  else
    pass "$name (rc=$got)"
  fi
}

row missing_run_id_stops          "no clean-room run id recorded for v1.2.3"              f_missing
row empty_run_id_stops            "no clean-room run id recorded for v1.2.3"              f_empty
row non_numeric_run_id_stops      "is not a run id"                                       f_garbage
row red_run_stops                 "clean-room run 9001 does not prove v1.2.3"             f_red
row run_on_other_commit_stops     "tested $OTHER"                                         f_other
row unreadable_run_stops          "recorded run 9002 could not be read"                   f_unknown
# green: the door opens, and the NEXT precondition stops the row before any upload
row green_run_passes_the_door     "no committed dry-run receipt (T-4)"                    f_green
if grep -q 'CLEAN-ROOM PROCEED: run 9001 job 501' "$WORK/ap-green_run_passes_the_door/STATUS" 2>/dev/null; then
  pass "green_run_recorded_in_STATUS"
else
  fail "green_run_recorded_in_STATUS: no PROCEED line in the state dir STATUS"
fi
if grep -q '^run view 9001 --repo paiml/infra --json jobs$' "$WORK/stub-green_run_passes_the_door/calls"; then
  pass "gh_is_the_stub (the green row read run 9001 through the stub)"
else
  fail "gh_is_the_stub: the green row never reached the stub gh"
fi

if [ "$rc" -eq 0 ]; then
  echo "PASS  publish_strict clean-room door: every row held"
else
  echo "FAIL  publish_strict clean-room door: a row broke (see above)"
fi
exit "$rc"
