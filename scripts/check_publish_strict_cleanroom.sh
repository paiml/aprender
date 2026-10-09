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
# (#4687 R5b): `make publish` must refuse a real upload and still dry-run, no other tracked
# shell script, make file, justfile, git hook, root workflow, composite action, ci/ job file or
# shell-shebang file may run a real upload, and no tracked cargo config may alias one.
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
# the line is split into commands at ; && || | and a lone & (not the & of >& <& &>). So a
# --dry-run on one command never hides another on the same line, and `cargo \` + `publish` is
# still one command. --dry-run counts only as a word of its own command. The body of a $(...),
# <(...), >(...) or `...` is a command row of its own (nested ones too, to the matching close;
# an escaped \` or \$ opens none), so --dry-run or "$APR" publish on the outer command never
# lets a cargo upload inside it through.
# A command is a door when cargo (cargo by a path, quoted, a quoted path with spaces, $CARGO, "${CARGO:-cargo}",
# $(CARGO), $$CARGO, $(command -v cargo)), with only global flags after it, runs an upload:
# publish, ws/workspaces publish, release or smart-release, or a subcommand it takes from a
# variable ("$sub", $(SUB)), which could be any of them. So is cargo-release,
# cargo-smart-release or cargo-workspaces run by its own name, and any variable run as the
# command with one of those uploads after it ($C publish, $(TOOL) release), since the variable
# may hold cargo. A $'...', $(...) or `...` word run as the command counts the same way: its
# value is only known when it runs. It counts at the command's start (after ( { ! @ + -,
# `case ... in`, a case pattern's `x)`, then/do/else/if/elif/while/until or a workflow run:),
# inside sh -c "..." (or -ec, -lc, --command, --run) or
# eval, after find's -exec/-execdir; behind VAR=val and a wrapper, by name or by a path (env exec
# time command nohup sudo doas timeout nice xargs parallel rustup ... make's shell). Anywhere
# else in a command, a cargo word with an upload after it still counts (DOOR_BARE): the words
# are read as the shell reads them (door_words), and a quoted string that holds a space is a
# message, not a word, so `fail "ran cargo publish"` is not a door and `doas "cargo" publish` is.
# cargo --config 'alias.p="publish"' and CARGO_ALIAS_P=publish make p an upload (DOOR_CALIAS);
# a PUT to crates.io's /api/v1/crates/new uploads without cargo, and so does release-plz
# release; and a crates.io token handed to a CI step (an env key CARGO_REGISTRY_TOKEN or
# CARGO_REGISTRIES_<name>_TOKEN, a secret named for cargo or crates and a token, or the
# trusted-publishing crates-io-auth-action) lets any step upload (DOOR_TOKEN). No gated door
# runs in CI, so no CI step needs one.
# The apr binary is the one variable let through: "$APR" publish uploads a model (DOOR_APR),
# so setting APR or APR_BIN to cargo is a door where it is set (DOOR_APRSET). cargo carried
# into APR through another variable (APR=$X) is run-time data, which no line scan reads.
# An upload stored for later is a door where it is stored: a shell or make variable whose
# value starts with one (X="cargo publish", UP := $(CARGO) publish, cmd=(cargo publish)) or an
# alias of one.
# The shell drops quotes and backslashes inside a word, so cargo "publish", cargo pub""lish and
# cargo \publish all run publish: door_keep matches each command as written and again with the
# quotes removed ($'...' and $"..." too) and every backslash dropped (more than the shell drops,
# never fewer), and prints it as written. door_cmds finds substitutions in the text as written,
# so an escaped \` or \$ opens none.
# The case table is the pattern's spec: re-run it, never re-read the pattern.
DOOR_PRE='[[:space:]]*(([(!{@+-]|case[[:space:]].*[[:space:]]in|[^[:space:]()]+\)|then|do|else|if|elif|while|until|run:)[[:space:]]*)*'
DOOR_MID='(.*(-[A-Za-z]*c[[:space:]]+["'"'"']|--(command|run)[[:space:]=]+["'"'"']|[[:space:]]-exec(dir)?[[:space:]]+)|(.*[^A-Za-z0-9_-])?eval[[:space:]]+["'"'"']?)[[:space:]]*'
DOOR_PATH='([^[:space:]"'"'"'=;&|]*/)?'
DOOR_WRAP='('"$DOOR_PATH"'(env|exec|time|command|builtin|nohup|sudo|doas|runuser|timeout|nice|ionice|stdbuf|xargs|parallel|flock|taskset|chrt|setsid|unshare|nsenter|chroot|firejail|systemd-run|strace|ltrace|rustup|shell)([[:space:]]+[^[:space:]]+)*[[:space:]]+)?'
# A VAR=val prefix holds no unclosed $( : in `out=$(bash "$me" release` bash is the command.
DOOR_ASSIGN='([A-Za-z_][A-Za-z0-9_]*=([^[:space:]$]|\$[^(]|\$\([^)[:space:]]*\))*[[:space:]]+)*'
DOOR_VAL='(.*[^A-Za-z0-9_.-])?[A-Za-z_][A-Za-z0-9_.-]*[[:space:]]*(::|:|\?|\+|!)?=[[:space:]]*[("'"'"']*[[:space:]]*'
DOOR_CARGO='("[^"]*/cargo"|'"'"'[^'"'"']*/cargo'"'"'|["'"'"'\\]?'"$DOOR_PATH"'cargo["'"'"']?|"?\$?\$(CARGO[A-Za-z0-9_]*|\{CARGO[A-Za-z0-9_]*(:?[-=+?][^}]*)?\})"?|\$\(CARGO\)|"?(\$\(|`)(command[[:space:]]+-v|which|type[[:space:]]+-[pP])[[:space:]]+cargo(\)|`)"?)'
# A braced ${...} ends at its } and must have one: in X="${2:-stack release}" the variable is
# the whole ${...}, so no `release` follows it.
DOOR_VAR='("?\$?\$([A-Za-z_][A-Za-z0-9_]*|[0-9@*]|\{([A-Za-z_][A-Za-z0-9_]*|[0-9@*])(:?[-=+?][^}]*)?\})"?|"?\$\([^()]*\)"?|`[^`]*`|\$'"'"'[^'"'"']*'"'"')'
DOOR_GFLAG='([[:space:]]+(\+[^[:space:]]+|-[A-Za-z]+|--[a-z][a-z-]*(=[^[:space:]]*)?|(-Z|-C|--config|--color)[[:space:]]+[^[:space:]]+))*'
DOOR_SUB='(publish|(ws|workspaces)[[:space:]]+publish|release|smart-release)([[:space:]"'"'"'`)};]|$)'
# For a variable subcommand the flags are read strictly, so -Z "$z" or --config "$f" is a flag
# and its value, never a flag and then a variable subcommand.
DOOR_VFLAG='([[:space:]]+(\+[^[:space:]]+|-[qv]+|--(locked|frozen|offline|quiet|verbose)|--[a-z][a-z-]*=[^[:space:]]*|(-Z|-C|--config|--color)[[:space:]]+[^[:space:]]+))*'
DOOR_TOOL="${DOOR_PATH}"'(cargo-(release|smart-release|workspaces|ws)|release-plz[[:space:]]+release)([[:space:]"'"'"']|$)'
DOOR_UP="(${DOOR_CARGO}(${DOOR_GFLAG}[[:space:]]+${DOOR_SUB}|${DOOR_VFLAG}[[:space:]]+\"?\\\$)|${DOOR_VAR}${DOOR_GFLAG}[[:space:]]+${DOOR_SUB}|${DOOR_TOOL})"
DOOR_APRSET='(.*[^A-Za-z0-9_.-])?APR(_BIN)?[[:space:]]*(::|:|\?|\+|!)?=[[:space:]]*[("'"'"']*[[:space:]]*'"${DOOR_CARGO}"'([[:space:]"'"'"');]|$)'
DOOR_CALIAS='(.*[^A-Za-z0-9_.-])?(CARGO_ALIAS_[A-Za-z0-9_]+|alias\.[A-Za-z0-9_-]+)[[:space:]]*(::|:|\?|\+|!)?=[[:space:]]*[[("'"'"']*[[:space:]]*(-[^[:space:]]*[[:space:]]+)*["'"'"']?(publish|(ws|workspaces)[[:space:]]+publish|release|smart-release)([^A-Za-z0-9_-]|$)'
DOOR_TOKEN='((.*[^A-Za-z0-9_])?(CARGO_REGISTRY_TOKEN|CARGO_REGISTRIES_[A-Za-z0-9_]+_TOKEN)[[:space:]]*:([[:space:]]|$)|.*secrets\.[A-Za-z0-9_]*(CARGO|CRATE)[A-Za-z0-9_]*TOKEN|.*crates-io-auth-actio[n]|.*/api/v1/crates/ne[w]([^A-Za-z0-9_-]|$))'
DOOR_RE="^[^:]*:[0-9]+:(((${DOOR_PRE}|${DOOR_MID})${DOOR_WRAP}${DOOR_ASSIGN}|${DOOR_VAL})${DOOR_UP}|${DOOR_APRSET}|${DOOR_CALIAS}|${DOOR_TOKEN})"
DOOR_BARE="^[^:]*:[0-9]+:(.*[[:space:]])?${DOOR_CARGO}${DOOR_GFLAG}[[:space:]]+${DOOR_SUB}"
DOOR_DRY='(^|[[:space:]])["'"'"']?--dry-run["'"'"']?([[:space:]]|$)'
# "$APR" and "$APR_BIN" are the pinned apr binary (scripts/apr_bin.sh): its publish uploads a
# model to Hugging Face, not a crate. Only a command that starts with one is let through.
DOOR_APR='^[^:]*:[0-9]+:"?\$\{?APR(_BIN)?\}?"?[[:space:]]+publish([[:space:]]|$)'
door_cmds() { # door_cmds FILE...: one "file:line:command" row per command, and per $(...) body
  awk '
    function emit(f, n, s,   k, i, c) {
      gsub(/>&/, "\001", s); gsub(/<&/, "\002", s); gsub(/&>/, "\003", s)
      k = split(s, c, /&&|\|\||;|\||&/)
      for (i = 1; i <= k; i++) {
        gsub(/\001/, ">\\&", c[i]); gsub(/\002/, "<\\&", c[i]); gsub(/\003/, "\\&>", c[i])
        sub(/[[:space:]]#.*$/, "", c[i])
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", c[i])
        if (c[i] != "" && c[i] !~ /^#/) { print f ":" n ":" c[i]; bodies(f, n, c[i]) }
      }
    }
    function bodies(f, n, s,   L, i, j, c, d, depth) {
      L = length(s)
      for (i = 1; i <= L; i++) {
        c = substr(s, i, 1); d = substr(s, i + 1, 1)
        if (c == "\\") { i++; continue }
        if (d == "(" && (c == "$" || c == "<" || c == ">")) {
          depth = 1
          for (j = i + 2; j <= L; j++) {
            c = substr(s, j, 1)
            if (c == "\\") { j++; continue }
            if (c == "(") depth++
            if (c == ")" && --depth == 0) break
          }
          emit(f, n, substr(s, i + 2, j - i - 2)); i = j; continue
        }
        if (c == "`") {
          for (j = i + 1; j <= L && substr(s, j, 1) != "`"; j++) if (substr(s, j, 1) == "\\") j++
          emit(f, n, substr(s, i + 1, j - i - 1)); i = j
        }
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
door_unquote() { sed -e 's/[$]["'"'"']/'"'"'/g' -e 's/\\\(.\)/\1/g' -e 's/["'"'"']//g'; }
door_words() { # a row naming cargo, as the shell splits its words: a quoted string that holds a
  # space becomes Q (a message), any other loses its quotes, and a backslash before a letter goes
  awk '
    !/[Cc][Aa][Rr][Gg][Oo]/ { print; next }
    {
      s = $0; o = ""; n = length(s); i = 1
      while (i <= n) {
        c = substr(s, i, 1); d = substr(s, i + 1, 1)
        if (c == "$" && (d == "\"" || d == "'"'"'")) { i++; continue }
        if (c == "\"" || c == "'"'"'") {
          j = i + 1
          while (j <= n && substr(s, j, 1) != c) { if (c == "\"" && substr(s, j, 1) == "\\") j++; j++ }
          w = substr(s, i + 1, j - i - 1)
          if (w ~ /[[:space:]]/) o = o "Q"; else o = o w
          i = j + 1; continue
        }
        if (c == "\\" && d ~ /[A-Za-z0-9_-]/) { i++; continue }
        o = o c; i++
      }
      print o
    }'
}
door_keep() { # a row matched as written, unquoted or by its words; --dry-run and $APR judged as written
  local t n out=""
  t=$(mktemp "$WORK/keep.XXXXXX") || return 2
  cat > "$t"
  n=$( { grep -nE "$DOOR_RE" "$t"; door_unquote < "$t" | grep -nE "$DOOR_RE"; door_words < "$t" | grep -nE "$DOOR_BARE"; } | cut -d: -f1 | sort -un)
  if [ -n "$n" ]; then
    out=$(printf '%s\n' "$n" | awk 'NR == FNR { k[$1]; next } FNR in k' - "$t" | grep -vE "$DOOR_DRY" | grep -vE "$DOOR_APR")
  fi
  rm -f "${t:?}"
  [ -n "$out" ] && printf '%s\n' "$out"
}
door_lines() { door_cmds "$@" | door_keep; }
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
door_case door 'cargo ws publish --yes'
door_case door 'cargo workspaces publish --from-git'
door_case door 'cargo release --execute'
door_case door 'cargo smart-release -u'
door_case door 'cargo-release release --execute'
door_case door 'cargo-workspaces workspaces publish'
door_case door 'sub=publish; cargo "$sub" -p x'
door_case door 'cargo $VERB -p x'
door_case door 'cargo ${VERB:-publish} -p x'
door_case door 'cargo --locked "$cmd" -p x'
door_case door '"$CARGO" "$@"'
door_case door $'\t$(CARGO) $(SUB) -p x'
door_case door 'eval "cargo publish -p x"'
door_case door 'eval cargo publish -p x'
door_case door 'find crates -name Cargo.toml -execdir cargo publish \;'
door_case door 'parallel -j2 cargo publish -p {} ::: a b'
door_case door 'cargo publish -p x & echo --dry-run'
door_case door 'cargo publish -p x & wait'
door_case door 'cargo publish -p x ${DRY:+--dry-run}'
door_case door 'cargo publish -p x --dry-run-later'
door_case door 'UP = cargo publish -p x'
door_case door 'UP := $(CARGO) publish -p x'
door_case door 'UP ?= cargo publish'
door_case door 'release: UP != cargo release'
door_case door 'X="cargo publish -p x"'
door_case door "export X='cargo publish'"
door_case door 'local -a cmd=(cargo publish -p x)'
door_case door "alias up='cargo publish'"
door_case door 'C=cargo; $C publish -p x'
door_case door '"$c" release --execute'
door_case door '"${C:-cargo}" publish -p x'
door_case door '"${CARGO:-cargo +nightly}" publish -p x'
door_case door $'\t$(TOOL) publish -p x'
door_case door $'\t$$C publish -p x'
door_case door $'\t$$CARGO publish -p x'
door_case door $'\t$(shell cargo publish -p x)'
door_case door 'diff <(cargo publish -p x) y'
door_case door '"cargo" publish -p x'
door_case door '\cargo publish -p x'
door_case door '/usr/bin/cargo publish -p x'
door_case door '"$HOME/.cargo/bin/cargo" publish -p x'
door_case door '$(command -v cargo) publish -p x'
door_case door '`which cargo` publish -p x'
door_case door '~/.cargo/bin/cargo-release release --execute'
door_case door 'publish) cargo publish -p x ;;'
door_case door 'a|b) cargo publish -p x ;;'
door_case door 'case "$1" in go) cargo publish -p x ;; esac'
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
door_case none 'cargo build --release'
door_case none 'cargo test --release -p x'
door_case none 'cargo ws list'
door_case none 'cargo releases'
door_case none 'cargo +"$TC" build'
door_case none 'cargo -Z "$z" build'
door_case none 'cargo --config "$f" build'
door_case none 'cargo build $FLAGS'
door_case none 'echo "then run cargo release"'
door_case none 'medieval "cargo publish"'
door_case none 'out=$(bash "$me" release "$d" job-1)'
door_case none 'MSG_SUFFIX="${2:-stack release}"'
door_case door 'X=$(date) cargo publish -p x'
door_case none '"$APR" publish "${APR_PUBLISH_ARGS[@]}"'
door_case none $'"$APR_BIN" publish \\\n    "$DIR" "$ID"'
door_case door '"$APRX" publish -p x'
door_case door '$APR_BIN2 publish -p x'
door_case door '"$APR" publish x && cargo publish -p y'
door_case door '"$APR" "publish" x'
door_case door '"$APR" publish "$(cargo publish -p y)"'
door_case door '"$APR" publish x `cargo publish -p y` z'
door_case door '$APR_BIN publish <(cargo publish -p y)'
door_case door 'cargo publish --dry-run -p x "$(cargo publish -p y)"'
door_case door 'X=$( (cargo publish -p x) )'
door_case door '"$APR" publish "$(env X=$(date) cargo publish -p y)"'
door_case none '"$APR" publish x --repo "$(whoami)/m"'
door_case none 'cargo publish --dry-run -p "$(cargo pkgid -p x)"'
door_case none 'X=$(cargo publish --dry-run -p x 2>&1)'
door_case none 'n=$((n + 1))'
door_case door 'APR=cargo'
door_case door 'export APR_BIN="$HOME/.cargo/bin/cargo"'
door_case door 'APR=${CARGO:-cargo} bash x.sh'
door_case door 'local APR=$(command -v cargo)'
door_case door 'APR := cargo'
door_case door "APR='cargo'; \"\$APR\" publish -p x"
door_case none 'APR="$root/target/release/apr"'
door_case none 'APR="${CARGO_HOME:-$HOME/.cargo}/bin/apr"'
door_case none 'APR=$CARGO_TARGET_DIR/release/apr'
door_case none 'APR_CARGO=1'
door_case none 'APR="$(cargo metadata --format-version 1 | jq -r .target_directory)/release/apr"'
door_case none 'MY_APR=cargo'
door_case door 'cargo "publish" -p x'
door_case door "cargo 'publish' --locked"
door_case door 'cargo pub""lish -p x'
door_case door 'cargo \publish -p x'
door_case door 'cargo pub\lish -p x'
door_case door "cargo \$'publish' -p x"
door_case door "\"\$C\" \$'publish' -p x"
door_case door '"cargo" "ws" "publish"'
door_case door 'X="cargo" "publish"'
door_case none 'cargo "publish" -p x --dry-run'
door_case none "cargo 'publish' -p x '--dry-run'"
door_case none 'echo "cargo publish"'
door_case none "grep -q 'cargo publish' notes.md"
door_case none 'cargo "build" -p x'
door_case door '"/opt/my tools/cargo" publish -p x'
door_case door "'/opt/my tools/cargo' publish -p x"
door_case door 'APR="/opt/my tools/cargo"'
door_case none '"/opt/my tools/cargo" build -p x'
door_case none 'echo "see /opt/cargo" publish'
door_case none 'echo "because \`cargo publish -p x\` failed"'
door_case door 'X=1 "$C" publish -p x'
door_case none 'cargo publish -p x 2>&1 --dry-run'
door_case none 'cargo publish -p x --dry-run &> log'
door_case none 'cargo publish -p x --dry-run >&2'
door_case none 'cargo publish -p x --dry-run <&0'
door_case none 'cargo publish -p x --dry-run &'
door_case none 'cargo publish -p x "--dry-run"'
door_case none 'echo "a & b"'
door_case none 'msg="cargo build failed"'
door_case none 'PV_CARGO_RUN := cargo run --release -p x --'
door_case none 'RUN = cargo test -p x'
door_case none 'check_outcome "$name" release "" "$got"'
door_case none $'\t$(MAKE) build'
door_case none '"$GH" pr view 1'
door_case none 'gh release view "$tag"'
door_case none '"$APR" run m.gguf'
door_case none '$C build -p x'
door_case none 'x) cargo build ;;'
door_case none 'case "$x" in a) cargo build ;; esac'
door_case none 'echo "use case x in y) cargo publish"'
door_case none '~/.cargo publish'
door_case door '/usr/bin/env cargo publish -p x'
door_case door '/usr/bin/timeout 600 cargo publish -p x'
door_case door '"/usr/bin/env" cargo publish -p x'
door_case door 'doas cargo publish -p x'
door_case door '/usr/bin/env "$C" publish -p x'
door_case door 'doas "$C" publish -p x'
door_case door 'my_wrapper cargo "publish" -p x'
door_case door 'rustup run stable "$C" publish -p x'
door_case door "my_wrapper \$'cargo' publish -p x"
door_case door 'my_wrapper cargo \publish -p x'
door_case door '"$C" \publish -p x'
door_case door 'a)cargo publish -p x;;'
door_case door 'case $x in a)cargo publish -p x;; esac'
door_case door 'my_wrapper "a\"b c" cargo publish -p x'
door_case door '$(shell $(TOOL) publish -p x)'
door_case door 'rustup run stable cargo publish -p x'
door_case door 'retry 3 cargo publish -p x'
door_case door 'my_wrapper "cargo" publish -p x'
door_case door 'echo cargo publish -p x'
door_case door 'bash -ec "cargo publish -p x"'
door_case door 'nix-shell --run '"'"'cargo publish -p x'"'"
door_case door 'cargo --config '"'"'alias.p="publish"'"'"' p -p x'
door_case door 'cargo --config alias.p=publish p -p x'
door_case door 'cargo --config '"'"'alias.p=["publish", "-p", "x"]'"'"' p'
door_case door 'CARGO_ALIAS_P=publish cargo p -p x'
door_case door 'export CARGO_ALIAS_UP='"'"'publish'"'"
door_case door "\$'\\x63argo' publish -p x"
door_case door "\$'\\143argo' publish -p x"
door_case door '"$(printf cargo)" publish -p x'
door_case door '`printf cargo` publish -p x'
door_case door 'release-plz release --git-token "$T"'
door_case door 'curl -X PUT -H "Authorization: $T" https://crates.io/api/v1/crates/new --data-binary @x'
door_case door '          CARGO_REGISTRY_TOKEN: ${{ secrets.X }}'
door_case door '          CARGO_REGISTRIES_MINE_TOKEN: x'
door_case door '        env: { T: "${{ secrets.CRATES_IO_TOKEN }}" }'
door_case door '      - uses: rust-lang/crates-io-auth-action@v1'
door_case none 'fail "ran cargo publish -p x"'
door_case none 'echo "x: cargo publish failed"'
door_case none '/usr/bin/env cargo build -p x'
door_case none 'bash -ec "cargo build -p x"'
door_case none 'git config alias.x y'
door_case none 'CARGO_ALIAS_B=build cargo b'
door_case none 'cargo --config '"'"'alias.b="build"'"'"' b'
door_case none '"$(git rev-parse --show-toplevel)/x.sh" release'
door_case none 'unset CARGO_REGISTRY_TOKEN'
door_case none '[ -z "${CARGO_REGISTRY_TOKEN:-}" ] || die "token set"'
door_case none '          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}'
door_case none 'curl -s https://crates.io/api/v1/crates/x'
door_case none 'grep -c '"'"'publish'"'"' f'
[ "$cases_ok" -eq "$cases_n" ] && pass "door_regex_case_table ($cases_ok/$cases_n cases)"

# The two gated doors: publish_strict.sh (the rows above) and cascade-publish.sh (its own
# clean_room_gate, scripts/check_cascade_clean_room_gate.sh). The scan must see each one's
# upload line, or it is blind, and nothing else in them: a second upload added to a gated
# file is not behind the gate just because the file has one. This file is scanned without
# its case table rows.
# SCOPE. By name: shell scripts, make files, justfiles and git hooks at any depth (`make -C
# crates/x publish` is a door too); workflows only at the root, as GitHub runs no other
# .github/workflows directory. By content: any other tracked file whose first line is a
# sh/bash/dash/ksh/mksh/zsh/bats shebang, so an extensionless script is read too, whatever it runs.
# Not scanned: Rust and Python, which build the call as an argv (the contract names them).
# ALLOWED is matched by file AND command, so a second upload added to one is still caught,
# and each must still be seen (the positive control for the content scope):
# - release_ready.sh writes a stand-in publish_strict.sh for its own case table;
# - the v0.68.1 audit record, publish_strict.sh as that release ran it, kept as evidence.
#   Nothing runs or cites it. Run by hand it would upload on a non-empty run-id file alone.
GATED='scripts/release/publish_strict.sh scripts/cascade-publish.sh'
GATED_CMD='cargo publish "${sel[@]}" --locked > "$log" 2>&1'
DOOR_ALLOWED='scripts/release/release_ready.sh:cargo publish -p x
docs/audits/release/v0.68.1/publish_strict.as-run.sh.txt:cargo publish "${sel[@]}" --locked > "$log" 2>&1'
export DOOR_ALLOWED
SELF=scripts/check_publish_strict_cleanroom.sh
SHEBANG_RE='^#!.*[/[:space:]](sh|bash|dash|ksh|mksh|zsh|bats)([[:space:]]|$)'
is_shell_script() { local l=''; IFS= read -r l < "$1" || [ -n "$l" ]; printf '%s\n' "$l" | grep -qE "$SHEBANG_RE"; }
sb_n=0 sb_ok=0
shebang_case() { # shebang_case shell|other FIRST-LINE
  local got=other
  sb_n=$((sb_n + 1))
  printf '%s\necho\n' "$2" > "$WORK/shebang-case"
  is_shell_script "$WORK/shebang-case" && got=shell
  if [ "$got" = "$1" ]; then sb_ok=$((sb_ok + 1)); else fail "shebang_case_table: want $1, got $got: $2"; fi
}
shebang_case shell '#!/bin/sh'
shebang_case shell '#!/bin/bash'
shebang_case shell '#! /bin/bash -e'
shebang_case shell '#!/usr/bin/env bash'
shebang_case shell '#!/usr/bin/env -S bash -eu'
shebang_case shell '#!/usr/bin/env bats'
shebang_case shell '#!/bin/dash'
shebang_case shell '#!/usr/bin/zsh'
shebang_case other '#!/usr/bin/env python3'
shebang_case other '#!/usr/bin/perl -w'
shebang_case other '#!/usr/bin/env node'
shebang_case other '# runs under bash'
shebang_case other 'cargo publish -p x'
shebang_case other '#!/usr/bin/env bashful'
[ "$sb_ok" -eq "$sb_n" ] && pass "shebang_case_table ($sb_ok/$sb_n cases)"
# shebang_files ROOT: each tracked file outside NAMES whose first line is a shell shebang. git
# grep lists every file with a line that starts with #!, whatever else it holds (a `cargo
# release` script need not say publish), and is_shell_script keeps those where it is line 1.
# git grep exits 1 when nothing matches and above 1 when it fails; a failed grep returns git's
# status, so it never reads as "no shebang file".
shebang_files() {
  local r=0 f
  git -C "$1" grep -lI -e "^#!" -- ":(exclude)$SELF" "${NAMES[@]/#/:(exclude)}" \
    ":(exclude).github/workflows/*" > "$WORK/sb-files" 2> "$WORK/sb-err" || r=$?
  [ "$r" -gt 1 ] && return "$r"
  while IFS= read -r f; do is_shell_script "$1/$f" && printf "%s\n" "$f"; done < "$WORK/sb-files"
  return 0
}
NAMES=('*.sh' '*.bash' '*.mk' 'Makefile' '*/Makefile' 'makefile' '*/makefile' 'GNUmakefile' '*/GNUmakefile'
  '*.mak' '*.make' 'Makefile.*' '*/Makefile.*'
  'justfile' '*/justfile' 'Justfile' '*/Justfile' '.justfile' '*/.justfile' '*.just'
  '.githooks/*' '*/.githooks/*' '.github/workflows/*.yml' '.github/workflows/*.yaml'
  '.github/actions/*.yml' '.github/actions/*.yaml' 'ci/*.yml' 'ci/*.yaml' 'ci/*.cmd')
# A composite action under .github/actions runs its steps wherever a root workflow uses it.
# ci/ holds commands CI runs: fat_driver.py runs the job bodies in ci/sections.yml and the
# vendored ci/vendor/sovereign-ci.yml, and a section runs each ci/explicit-test-commands.d/*.cmd.
# make -f reads a make file by any name; *.mak, *.make and Makefile.* are the names one goes
# by (automake copies a Makefile.am recipe into the Makefile it writes). makefile.* is not: a
# makefile.rs is Rust.
# The name scope's spec: a throwaway repo holds one empty file per path below, and NAMES must
# select exactly the "in" ones. Most kinds hold no door today, so without this a kind dropped
# from NAMES would go unseen.
SC="$WORK/scope-case"
SCOPE_IN='x.sh a/b.bash c.mk Makefile a/Makefile makefile GNUmakefile a/b/GNUmakefile r.mak a/b.make Makefile.release
a/Makefile.inc a/Makefile.am justfile crates/x/justfile
Justfile a/Justfile .justfile a/.justfile r.just .githooks/pre-push crates/x/.githooks/pre-commit
.github/workflows/r.yml .github/workflows/r.yaml
.github/actions/x/action.yml .github/actions/y/action.yaml ci/sections.yml ci/vendor/v.yaml ci/t.d/010-x.cmd'
SCOPE_OUT='x.py x.rs README.md a/makefile.rs x.mak.txt x.sh.txt justfile.md crates/x/.github/workflows/r.yml .github/dependabot.yml
crates/x/ci/r.yml ci/m.tsv'
mkdir -p "$SC" && git -C "$SC" init -q
for p in $SCOPE_IN $SCOPE_OUT; do mkdir -p "$SC/$(dirname "$p")" && : > "$SC/$p"; done
git -C "$SC" add -A
sc_got=$(git -C "$SC" ls-files -- "${NAMES[@]}" | sort | tr '\n' ' ')
sc_want=$(printf '%s\n' $SCOPE_IN | sort | tr '\n' ' ')
if [ "$sc_got" = "$sc_want" ]; then
  pass "scope_case_table ($(wc -w <<< "$SCOPE_IN") paths in, $(wc -w <<< "$SCOPE_OUT") out)"
else
  fail "scope_case_table: want $sc_want, got $sc_got"
fi
# The shebang scope's spec: a script is read for its first line, not for what it holds.
SB="$WORK/sb-case"
mkdir -p "$SB/tools" "$SB/t" "$SB/docs" && git -C "$SB" init -q
printf '#!/bin/sh\ncargo release -x\n' > "$SB/tools/rel"
printf '#!/usr/bin/env bats\n' > "$SB/t/x.bats"
printf '#!/usr/bin/env python3\n' > "$SB/tools/py"
printf 'notes\n#!/bin/sh\n' > "$SB/docs/n.txt"
printf '#!/bin/sh\n' > "$SB/x.sh"
git -C "$SB" add -A
sbs_rc=0
shebang_files "$SB" > "$WORK/sbs-got" || sbs_rc=$?
sbs_got=$(sort "$WORK/sbs-got" | tr '\n' ' ')
sbs_want=$(printf '%s\n' tools/rel t/x.bats | sort | tr '\n' ' ')
if [ "$sbs_rc" -eq 0 ] && [ "$sbs_got" = "$sbs_want" ]; then
  pass "shebang_scope_case_table (2 files in, 3 out)"
else
  fail "shebang_scope_case_table: rc $sbs_rc, want $sbs_want, got $sbs_got"
fi
mapfile -t scope < <(git -C "$DOORS_ROOT" ls-files -- "${NAMES[@]}" ":(exclude)$SELF")
named=${#scope[@]}
sb_rc=0
shebang_files "$DOORS_ROOT" > "$WORK/sb-scope" || sb_rc=$?
if [ "$sb_rc" -ne 0 ]; then
  fail "door_scan_shebang_scope: git grep for the shebang scope failed (rc $sb_rc): $(head -n 1 "$WORK/sb-err")"
fi
while IFS= read -r f; do scope+=("$f"); done < "$WORK/sb-scope"
# Every file in scope must be read. awk skips a file it cannot open with a warning (gawk) or
# stops there (mawk), and either would read as files with no door.
read_err="" dc_rc=0
(cd "$DOORS_ROOT" && door_cmds "${scope[@]}") > "$WORK/scope-cmds" 2> "$WORK/scope-err" || dc_rc=$?
if [ "$dc_rc" -ne 0 ] || [ -s "$WORK/scope-err" ]; then
  read_err="rc $dc_rc: $(head -n 1 "$WORK/scope-err")"
fi
raw=$(door_keep < "$WORK/scope-cmds" || true)
self_n=0
if [ -f "$DOORS_ROOT/$SELF" ]; then
  self_n=1
  # Without its case rows, and without the GATED_CMD line, whose value is the upload text.
  awk -v g="GATED_CMD='$GATED_CMD'" '/^(door|shebang|alias)_case / || $0 == g { $0 = "" } { print }' \
    "$DOORS_ROOT/$SELF" > "$WORK/self-scan.sh"
  raw+=$'\n'$( (door_lines "$WORK/self-scan.sh" || true) | sed "s#^$WORK/self-scan.sh:#$SELF:#")
fi
cmds_of() { printf '%s\n' "$raw" | awk -v f="$1:" 'index($0, f) == 1 { sub(/^[^:]*:[0-9]+:/, ""); print }'; }
for g in $GATED; do
  got=$(cmds_of "$g")
  if [ -z "$got" ]; then
    fail "door_scan_sees $g: no upload line found in it (${#scope[@]} files scanned), so the scan proves nothing"
  elif [ "$got" != "$GATED_CMD" ]; then
    fail "door_scan_sees $g: want its one upload, $GATED_CMD, got: $(printf '%s' "$got" | tr '\n' '|')"
  else
    pass "door_scan_sees $g (its one upload line, and no other)"
  fi
done
while IFS= read -r a; do
  af=${a%%:*}
  if [ "$(cmds_of "$af")" = "${a#*:}" ]; then
    pass "door_scan_sees_allowed $af (its one allowed command)"
  else
    fail "door_scan_sees_allowed $af: want exactly ${a#*:}, got: $(cmds_of "$af" | tr '\n' '|') (blind, or the file changed or is gone: then update DOOR_ALLOWED)"
  fi
done <<< "$DOOR_ALLOWED"
doors=$(printf '%s\n' "$raw" | awk -v gated=" $GATED " '
  BEGIN { n = split(ENVIRON["DOOR_ALLOWED"], a, "\n"); for (i = 1; i <= n; i++) ok[a[i]] = 1 }
  { f = $0; sub(/:.*/, "", f); t = $0; sub(/^[^:]*:[0-9]+:/, "", t) }
  f == "" || index(gated, " " f " ") || ((f ":" t) in ok) { next }
  { print }')
if [ -n "$read_err" ]; then
  fail "no_other_door: reading the ${#scope[@]} files in scope failed ($read_err), so a file went unscanned"
fi
if [ -n "$doors" ]; then
  while IFS= read -r d; do fail "no_other_door: a real cargo publish outside the gated doors: $d"; done <<< "$doors"
elif [ -z "$read_err" ]; then
  self_txt=""; [ "$self_n" -eq 1 ] && self_txt=", and this file"
  pass "no_other_door ($((${#scope[@]} + self_n)) files scanned: $named by name (tracked shell scripts, make files, justfiles, git hooks, root workflows, composite actions and ci/ job files), $((${#scope[@]} - named)) by a shell shebang$self_txt)"
fi

# A cargo alias makes `cargo <name>` an upload the door scan cannot see, so no tracked cargo
# config, and no other tracked TOML file (`cargo --config <path>` reads any), may define one: an alias ([alias] table or a top-level alias.<name> key) whose first
# word, after flags, is publish, release or smart-release, or ws/workspaces with publish.
alias_rows() { # alias_rows all|doors FILE...: "file:line:text" per alias (all) or per upload alias
  awk -v all="$1" '
    FNR == 1 { s = "" }
    /^[[:space:]]*\[/ { s = $0; sub(/#.*/, "", s); gsub(/[[:space:]]/, "", s); next }
    {
      v = $0
      if (!((s == "[alias]" && v ~ /=/) || (s == "" && v ~ /^[[:space:]]*alias\.[^=]*=/))) next
      if (all == "all") { print FILENAME ":" FNR ":" $0; next }
      sub(/^[^=]*=/, "", v); gsub(/[]["'"'"',]/, " ", v)
      k = split(v, w, /[[:space:]]+/)
      for (i = 1; i <= k; i++) if (w[i] != "" && w[i] !~ /^[-+]/) break
      if (i > k) next
      if (w[i] ~ /^(publish|release|smart-release)$/ || (w[i] ~ /^(ws|workspaces)$/ && (" " v " ") ~ /[[:space:]]publish[[:space:]]/)) print FILENAME ":" FNR ":" $0
    }' "${@:2}"
}
al_n=0 al_ok=0
alias_case() { # alias_case door|none CONFIG-TEXT
  local got=none
  al_n=$((al_n + 1))
  printf '%s\n' "$2" > "$WORK/alias-case.toml"
  [ -n "$(alias_rows doors "$WORK/alias-case.toml")" ] && got=door
  if [ "$got" = "$1" ]; then al_ok=$((al_ok + 1)); else fail "alias_case_table: want $1, got $got: $2"; fi
}
alias_case door $'[alias]\npub = "publish -p x"'
alias_case door $'[alias]\np = ["publish", "-p", "x"]'
alias_case door $'alias.pub = "publish"'
alias_case door $'[ alias ] # mine\nq = "--locked publish"'
alias_case door $'[alias]\nr = "release --execute"'
alias_case door $'[alias]\nsr = "smart-release -u"'
alias_case door $'[alias]\nw = "ws publish --yes"'
alias_case door $'[build]\njobs = 4\n[alias]\nt = "test"\np = "publish"'
alias_case none $'[alias]\nb = "build --release"'
alias_case none $'[alias]\ncl = "clippy --all-targets -- -D warnings"'
alias_case none $'[alias]\nw = "ws list"'
alias_case none $'[registry]\ndefault = "publish"'
alias_case none $'# [alias]\n# pub = "publish"'
alias_case none $'[alias]\nt = "test"\n[env]\nX = "publish"'
alias_case none $'[build]\nalias.p = "publish"'
[ "$al_ok" -eq "$al_n" ] && pass "alias_case_table ($al_ok/$al_n cases)"
CFG_NAMES=('*.toml' '.cargo/config' '*/.cargo/config')
# The config scope's spec, as for NAMES: cargo reads .cargo/config(.toml) at any depth, and
# `cargo --config <path>` reads any TOML file.
AC="$WORK/alias-scope-case"
AC_IN='.cargo/config .cargo/config.toml a/.cargo/config a/b/.cargo/config.toml Cargo.toml zz/pub.toml'
AC_OUT='config a/config.txt x.yml .cargo/env'
mkdir -p "$AC" && git -C "$AC" init -q
for p in $AC_IN $AC_OUT; do mkdir -p "$AC/$(dirname "$p")" && : > "$AC/$p"; done
git -C "$AC" add -A
ac_got=$(git -C "$AC" ls-files -- "${CFG_NAMES[@]}" | sort | tr '\n' ' ')
ac_want=$(printf '%s\n' $AC_IN | sort | tr '\n' ' ')
if [ "$ac_got" = "$ac_want" ]; then
  pass "alias_scope_case_table ($(wc -w <<< "$AC_IN") paths in, $(wc -w <<< "$AC_OUT") out)"
else
  fail "alias_scope_case_table: want $ac_want, got $ac_got"
fi
mapfile -t cfgs < <(git -C "$DOORS_ROOT" ls-files -- "${CFG_NAMES[@]}")
if [ "${#cfgs[@]}" -eq 0 ]; then
  fail "no_publish_alias: no tracked cargo config or TOML file found, so the row proves nothing"
else
  al_all=$(cd "$DOORS_ROOT" && alias_rows all "${cfgs[@]}" | grep -c . || true)
  al_doors=$(cd "$DOORS_ROOT" && alias_rows doors "${cfgs[@]}" || true)
  case "$al_all" in '' | *[!0-9]*) al_all=0 ;; esac
  if [ "$al_all" -eq 0 ]; then
    fail "no_publish_alias: ${#cfgs[@]} cargo configs and TOML files read and no alias seen in any, so the row proves nothing"
  elif [ -n "$al_doors" ]; then
    while IFS= read -r d; do fail "no_publish_alias: a cargo alias that uploads: $d"; done <<< "$al_doors"
  else
    pass "no_publish_alias (${#cfgs[@]} tracked cargo configs and TOML files, $al_all aliases read)"
  fi
fi

if [ "$rc" -eq 0 ]; then
  echo "PASS  publish_strict clean-room door: every row held"
else
  echo "FAIL  publish_strict clean-room door: a row broke (see above)"
fi
exit "$rc"
