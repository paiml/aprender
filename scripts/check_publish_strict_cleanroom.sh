#!/usr/bin/env bash
# publish_strict.sh is the one door to crates.io, and it must refuse a clean-room run id
# that is missing, red, or tested another commit (PMAT-4687). Before #4687 it accepted any
# non-empty file. This is its falsifier: the REAL publish_strict.sh runs against a fixture
# tag checkout, a fixture state dir (RELEASE_AP) and a stub `gh` and `cargo` first on PATH.
#
# Every row stops before an upload. The green row gets past the clean-room check and then
# stops on the absent B2-gpu run id, which proves the check let it through. The stub
# `cargo` fails the table if publish_strict.sh ever calls it.
#
# It is only the ONE door if nothing else uploads, so the last rows check the other doors
# (#4687 R5b): `make publish` must refuse a real upload and still dry-run, and no other
# tracked shell script, make file or root workflow may run a real `cargo publish`.
# PUBLISH_DOORS_ROOT points those rows at another checkout (the before numbers).
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
DOORS_ROOT="${PUBLISH_DOORS_ROOT:-$REPO_ROOT}"
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
printf 'CARGO %s\n' "$*" >> "${STUB_DIR:?}/calls"
# Only the make publish dry-run row gets a zero, and only for a --dry-run call.
[ "${STUB_CARGO_DRY_OK:-}" = 1 ] && [[ " $* " == *" --dry-run "* ]] && exit 0
exit 97
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
  elif [ "$setup" != f_green ] && [[ "$out" == *"B2-gpu run id"* ]]; then
    fail "$name: the clean-room door let this run through (it stopped later, on the B2-gpu id)"
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
row green_run_passes_the_door     "no green B2-gpu run id recorded"                       f_green
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

# ── the other doors (#4687 R5b): nothing else in the tree uploads ─────────────────────────
# `make publish` was one (an upload with no clean-room, preflight or tag check), and so was
# scripts/stack_release.sh (bump, commit, tag, push main, upload). Two rows run `make publish`
# itself from a scratch copy of the Makefile, with the stub cargo first on PATH.
if ! command -v make > /dev/null 2>&1; then
  fail "make_publish rows: make is not on PATH, so they cannot run (not_measured is not a pass)"
else
  MK="$WORK/mk" MKBIN="$WORK/mkbin"
  mkdir -p "$MK" "$MKBIN"
  cp "$DOORS_ROOT/Makefile" "$MK/Makefile"
  # The recipe asks scripts/lib/cascade_universe.py for the crate's manifest; this stub gives
  # it the one row it needs. Its own dir, not $BIN: publish_strict.sh needs the real python3.
  cat > "$MKBIN/python3" <<'STUB'
#!/usr/bin/env bash
printf 'aprender\t-\t%s/Cargo.toml\t%s\n' "$PWD" "$PWD"
STUB
  chmod +x "$MKBIN/python3"
  mk_publish() { # mk_publish STUB_DIR [VAR=VAL...]: make publish CRATE=aprender in the copy
    local s=$1
    shift
    (cd "$MK" && env -u MAKEFLAGS -u MFLAGS -u MAKELEVEL -u PUBLISH_DRY_RUN -u CRATE \
      PATH="$MKBIN:$BIN:$PATH" STUB_DIR="$s" "$@" make -s publish CRATE=aprender 2>&1)
  }
  S="$WORK/stub-make-upload"; mkdir -p "$S"; : > "$S/calls"; got=0
  out=$(mk_publish "$S") || got=$?
  if grep -q '^CARGO' "$S/calls"; then
    fail "make_publish_refuses_upload: with no PUBLISH_DRY_RUN it ran $(grep -m1 '^CARGO' "$S/calls")"
  elif [ "$got" -eq 0 ] || [[ "$out" != *"REFUSED: make publish does not upload"* ]]; then
    fail "make_publish_refuses_upload: rc=$got and no refusal"; printf '      | %s\n' "$out" | tail -n 4
  else
    pass "make_publish_refuses_upload (rc=$got, cargo never called)"
  fi
  S="$WORK/stub-make-dry"; mkdir -p "$S"; : > "$S/calls"; got=0
  out=$(mk_publish "$S" PUBLISH_DRY_RUN=1 STUB_CARGO_DRY_OK=1) || got=$?
  calls=$(grep -c '^CARGO' "$S/calls" || true)
  if [ "$got" -ne 0 ] || [ "$calls" -ne 1 ] || ! grep -q '^CARGO publish -p aprender --dry-run ' "$S/calls"; then
    fail "make_publish_dry_run_still_packages: rc=$got, $calls cargo call(s): $(grep '^CARGO' "$S/calls" | head -n 2 | tr '\n' ' ')"
    printf '      | %s\n' "$out" | tail -n 4
  else
    pass "make_publish_dry_run_still_packages (one cargo publish --dry-run call)"
  fi
fi

# A line uploads when `cargo publish` stands where a command starts (line start; after
# ; & | ! $( or an unescaped backtick; after then/do/else/if/exec/time or a workflow `run:`;
# behind env and VAR=val prefixes) and the line has no --dry-run. Comment lines are skipped.
# The case table is the pattern's spec: re-run it, never re-read the pattern.
DOOR_RE='(^|[;&|!]|\$\(|[^\\]`|(^|[[:space:]])(then|do|else|if|exec|time|run:))[[:space:]]*(env([[:space:]]+(-u[[:space:]]+[A-Za-z_][A-Za-z0-9_]*|-[A-Za-z]+|[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*))*[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+)*cargo[[:space:]]+(\+[^[:space:]]+[[:space:]]+)?publish([[:space:]]|;|$)'
door_lines() { grep -nE "$DOOR_RE" -- "$@" /dev/null | grep -vE '^[^:]+:[0-9]+:[[:space:]]*#' | grep -v -e '--dry-run'; }
cases_n=0 cases_ok=0
door_case() { # door_case door|none LINE
  local got=none
  cases_n=$((cases_n + 1))
  printf '%s\n' "$2" > "$WORK/door-case.sh"
  door_lines "$WORK/door-case.sh" > /dev/null && got=door
  if [ "$got" = "$1" ]; then cases_ok=$((cases_ok + 1)); else fail "door_regex_case_table: want $1, got $got: $2"; fi
}
door_case door 'cargo publish --no-verify --allow-dirty 2>&1 | tail -5'
door_case door $'\tcargo publish $$SEL $$DRY --allow-dirty --locked; \\'
door_case door '  cargo publish "${sel[@]}" --locked > "$log" 2>&1; rc=$?'
door_case door '        run: cargo publish -p aprender'
door_case door '      - run: cargo publish'
door_case door 'cd crates/x && cargo publish'
door_case door 'if cargo publish -p y; then'
door_case door 'then cargo publish -p y'
door_case door 'out=$(cargo publish -p y 2>&1)'
door_case door 'out=`cargo publish -p y`'
door_case door 'env -u CARGO_REGISTRY_TOKEN cargo publish -p z'
door_case door 'CARGO_REGISTRY_TOKEN=$T cargo publish -p z'
door_case door 'cargo +stable publish -p z'
door_case door '  cargo   publish'
door_case door 'x=1; cargo publish'
door_case none '# cargo publish -p x'
door_case none '    # cargo publish -p x'
door_case none 'echo "FAIL: cargo publish failed"'
door_case none 'echo "   because \`cargo publish -p $$CRATE\` from here"'
door_case none $'\tcargo publish $$SEL --dry-run --no-verify --allow-dirty --locked; \\'
door_case none 'DRY=$(env -u CARGO_REGISTRY_TOKEN cargo publish --dry-run --allow-dirty 2>&1); DRC=$?'
door_case none '        /cargo publish/ { cut = 1 }'
door_case none '  live "$c" "$v" || die "$c $v: cargo publish rc=0 but the version is not on the index"'
door_case none 'echo "   3. Publish to crates.io: cargo publish"'
door_case none '    echo "DEFER (cargo publish exited 0 but printed no line)"'
door_case none 'cargo publishing notes'
door_case none 'cargo-publish x'
[ "$cases_ok" -eq "$cases_n" ] && pass "door_regex_case_table ($cases_ok/$cases_n cases)"

# The two gated doors: publish_strict.sh (the rows above) and cascade-publish.sh (its own
# clean_room_gate, scripts/check_cascade_clean_room_gate.sh). The scan must see both, or it
# is blind. One fixture is allowed by file AND text: release_ready.sh writes a stand-in
# publish_strict.sh for its own case table, so a real upload added there is still caught.
# This file is skipped too: its case table above is made of upload lines.
GATED='scripts/release/publish_strict.sh scripts/cascade-publish.sh'
FIXTURE='scripts/release/release_ready.sh:cargo publish -p x'
SELF=scripts/check_publish_strict_cleanroom.sh
mapfile -t scope < <(git -C "$DOORS_ROOT" ls-files -- '*.sh' '*.bash' 'Makefile' '*.mk' \
  '.github/workflows/*.yml' '.github/workflows/*.yaml')
raw=$( (cd "$DOORS_ROOT" && door_lines "${scope[@]}") || true)
for g in $GATED; do
  if printf '%s\n' "$raw" | grep -qF "$g:"; then
    pass "door_scan_sees $g"
  else
    fail "door_scan_sees $g: no upload line found in it (${#scope[@]} files scanned), so the scan proves nothing"
  fi
done
doors=$(printf '%s\n' "$raw" | awk -v gated=" $GATED $SELF " -v fx="$FIXTURE" '
  { f = $0; sub(/:.*/, "", f); t = $0; sub(/^[^:]*:[0-9]+:/, "", t); gsub(/^[[:space:]]+|[[:space:]]+$/, "", t) }
  f == "" || index(gated, " " f " ") || f ":" t == fx { next }
  { print }')
if [ -z "$doors" ]; then
  pass "no_other_door (${#scope[@]} tracked shell scripts, make files and root workflows scanned)"
else
  while IFS= read -r d; do fail "no_other_door: a real cargo publish outside the gated doors: $d"; done <<< "$doors"
fi

if [ "$rc" -eq 0 ]; then
  echo "PASS  publish_strict clean-room door: every row held"
else
  echo "FAIL  publish_strict clean-room door: a row broke (see above)"
fi
exit "$rc"
