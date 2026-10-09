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

# A command uploads when it runs `cargo publish`. door_cmds reads each file the way the shell
# does: a line ending in \ is joined to the next (the row keeps the line the command starts
# on), a comment line is dropped and does not continue, a trailing ` #...` comment is cut, and
# the line is split into commands at ; && || and |. So a --dry-run on one command never hides
# another on the same line, and `cargo \` + `publish` is still one command.
# A command is a door when cargo ($CARGO, "${CARGO:-cargo}", $(CARGO)), with only global flags
# after it, runs publish: at the command's start (after ( { ! ` @ + -, then/do/else/if/elif/
# while/until or a workflow run:), after $( or an unescaped backtick, or inside sh -c "...";
# behind VAR=val and a wrapper (env exec time command nohup sudo timeout nice xargs ...).
# The case table is the pattern's spec: re-run it, never re-read the pattern.
DOOR_PRE='[[:space:]]*(([(!{`@+-]|then|do|else|if|elif|while|until|run:)[[:space:]]*)*'
DOOR_MID='.*(\$\(|[^\\]`|-c[[:space:]]+["'"'"'])[[:space:]]*'
DOOR_WRAP='((env|exec|time|command|builtin|nohup|sudo|timeout|nice|ionice|stdbuf|xargs|flock|taskset|chrt|setsid)([[:space:]]+[^[:space:]]+)*[[:space:]]+)?'
DOOR_ASSIGN='([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+)*'
DOOR_CARGO='(cargo|"?\$\{?CARGO[A-Za-z0-9_]*(:?[-=+?][^}[:space:]]*)?\}?"?|\$\(CARGO\))'
DOOR_GFLAG='([[:space:]]+(\+[^[:space:]]+|-[A-Za-z]+|--[a-z][a-z-]*(=[^[:space:]]*)?|(-Z|-C|--config|--color)[[:space:]]+[^[:space:]]+))*'
DOOR_RE="^[^:]*:[0-9]+:(${DOOR_PRE}|${DOOR_MID})${DOOR_WRAP}${DOOR_ASSIGN}${DOOR_CARGO}${DOOR_GFLAG}[[:space:]]+publish([[:space:]\"'\`)};]|\$)"
door_cmds() { # door_cmds FILE...: one "file:line:command" row per command
  awk '
    function emit(f, n, s,   k, i, c) {
      k = split(s, c, /&&|\|\||;|\|/)
      for (i = 1; i <= k; i++) {
        sub(/[[:space:]]#.*$/, "", c[i])
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", c[i])
        if (c[i] != "" && c[i] !~ /^#/) print f ":" n ":" c[i]
      }
    }
    FNR == 1 && buf != "" { emit(pf, start, buf); buf = "" }
    { pf = FILENAME }
    buf == "" && /^[[:space:]]*#/ { next }
    {
      if (buf == "") start = FNR
      if ($0 ~ /\\$/) { buf = buf substr($0, 1, length($0) - 1) " "; next }
      emit(FILENAME, start, buf $0); buf = ""
    }
    END { if (buf != "") emit(pf, start, buf) }
  ' "$@"
}
door_lines() { door_cmds "$@" | grep -E "$DOOR_RE" | grep -v -e '--dry-run'; }
cases_n=0 cases_ok=0
door_case() { # door_case door|none TEXT (TEXT may span lines)
  local got=none
  cases_n=$((cases_n + 1))
  printf '%s\n' "$2" > "$WORK/door-case.sh"
  [ -n "$(door_lines "$WORK/door-case.sh")" ] && got=door
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
door_case door '`cargo publish -p y`'
door_case door 'env -u CARGO_REGISTRY_TOKEN cargo publish -p z'
door_case door 'CARGO_REGISTRY_TOKEN=$T cargo publish -p z'
door_case door 'cargo +stable publish -p z'
door_case door '  cargo   publish'
door_case door 'x=1; cargo publish'
door_case door 'timeout 600 cargo publish -p x'
door_case door 'command cargo publish -p x'
door_case door 'nohup cargo publish -p x &'
door_case door 'nice -n 5 cargo publish -p x'
door_case door 'sudo -E cargo publish -p x'
door_case door "printf '%s\\n' a b | xargs -I{} cargo publish -p {}"
door_case door 'sh -c "cargo publish -p x"'
door_case door "bash -c 'cd x && cargo publish'"
door_case door '$CARGO publish -p x'
door_case door '"$CARGO" publish -p x --locked'
door_case door '${CARGO:-cargo} publish -p x'
door_case door $'\t$(CARGO) publish -p x'
door_case door 'cargo -q publish -p x'
door_case door 'cargo --locked --config net.retry=5 publish'
door_case door '( cargo publish -p x )'
door_case door '{ cargo publish -p x; }'
door_case door $'\t@cargo publish -p x'
door_case door $'\t-cargo publish -p x'
door_case door 'cargo publish -p x --locked && echo --dry-run'
door_case door 'cargo publish -p y --dry-run; cargo publish -p y'
door_case door 'cargo publish -p x # --dry-run'
door_case door $'cargo \\\n  publish -p x'
door_case door $'\tcd x && \\\n\tcargo publish -p x'
door_case door $'# a comment does not continue \\\ncargo publish -p x'
door_case none '# cargo publish -p x'
door_case none '    # cargo publish -p x'
door_case none 'x=1 # cargo publish -p x'
door_case none 'echo "FAIL: cargo publish failed"'
door_case none 'echo "   because \`cargo publish -p $$CRATE\` from here"'
door_case none $'\tcargo publish $$SEL --dry-run --no-verify --allow-dirty --locked; \\'
door_case none 'DRY=$(env -u CARGO_REGISTRY_TOKEN cargo publish --dry-run --allow-dirty 2>&1); DRC=$?'
door_case none 'cargo publish -p x --dry-run && echo ok'
door_case none $'cargo \\\n  publish -p x --dry-run'
door_case none '        /cargo publish/ { cut = 1 }'
door_case none '  live "$c" "$v" || die "$c $v: cargo publish rc=0 but the version is not on the index"'
door_case none 'echo "   3. Publish to crates.io: cargo publish"'
door_case none 'echo "use: cargo publish"'
door_case none '    echo "DEFER (cargo publish exited 0 but printed no line)"'
door_case none 'if [ "$CARGO" = x ]; then echo publish; fi'
door_case none 'cargo publishing notes'
door_case none 'cargo-publish x'
[ "$cases_ok" -eq "$cases_n" ] && pass "door_regex_case_table ($cases_ok/$cases_n cases)"

# The two gated doors: publish_strict.sh (the rows above) and cascade-publish.sh (its own
# clean_room_gate, scripts/check_cascade_clean_room_gate.sh). The scan must see each one's
# upload line, or it is blind, and nothing else in them: a second upload added to a gated
# file is not behind the gate just because the file has one. One fixture is allowed by file
# AND text: release_ready.sh writes a stand-in publish_strict.sh for its own case table, so a
# real upload added there is still caught. This file is scanned without its case table rows.
# Make files and git hooks at any depth (`make -C crates/x publish` is a door too). Workflows only at the
# root: GitHub runs no other .github/workflows directory.
GATED='scripts/release/publish_strict.sh scripts/cascade-publish.sh'
GATED_CMD='cargo publish "${sel[@]}" --locked > "$log" 2>&1'
FIXTURE='scripts/release/release_ready.sh:cargo publish -p x'
SELF=scripts/check_publish_strict_cleanroom.sh
mapfile -t scope < <(git -C "$DOORS_ROOT" ls-files -- '*.sh' '*.bash' '*.mk' \
  'Makefile' '*/Makefile' 'makefile' '*/makefile' 'GNUmakefile' '*/GNUmakefile' \
  '.githooks/*' '*/.githooks/*' '.github/workflows/*.yml' '.github/workflows/*.yaml' ":(exclude)$SELF")
raw=$( (cd "$DOORS_ROOT" && door_lines "${scope[@]}") || true)
if [ -f "$DOORS_ROOT/$SELF" ]; then
  sed 's/^door_case .*//' "$DOORS_ROOT/$SELF" > "$WORK/self-scan.sh"
  raw+=$'\n'$( (door_lines "$WORK/self-scan.sh" || true) | sed "s#^$WORK/self-scan.sh:#$SELF:#")
fi
for g in $GATED; do
  got=$(printf '%s\n' "$raw" | awk -v f="$g:" 'index($0, f) == 1 { sub(/^[^:]*:[0-9]+:/, ""); print }')
  if [ -z "$got" ]; then
    fail "door_scan_sees $g: no upload line found in it (${#scope[@]} files scanned), so the scan proves nothing"
  elif [ "$got" != "$GATED_CMD" ]; then
    fail "door_scan_sees $g: want its one upload, $GATED_CMD, got: $(printf '%s' "$got" | tr '\n' '|')"
  else
    pass "door_scan_sees $g (its one upload line, and no other)"
  fi
done
doors=$(printf '%s\n' "$raw" | awk -v gated=" $GATED " -v fx="$FIXTURE" '
  { f = $0; sub(/:.*/, "", f); t = $0; sub(/^[^:]*:[0-9]+:/, "", t) }
  f == "" || index(gated, " " f " ") || f ":" t == fx { next }
  { print }')
if [ -z "$doors" ]; then
  pass "no_other_door (${#scope[@]} tracked shell scripts, make files, git hooks and root workflows, and this file, scanned)"
else
  while IFS= read -r d; do fail "no_other_door: a real cargo publish outside the gated doors: $d"; done <<< "$doors"
fi

if [ "$rc" -eq 0 ]; then
  echo "PASS  publish_strict clean-room door: every row held"
else
  echo "FAIL  publish_strict clean-room door: a row broke (see above)"
fi
exit "$rc"
