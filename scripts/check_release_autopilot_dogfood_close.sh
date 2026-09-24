#!/usr/bin/env bash
# check_release_autopilot_dogfood_close.sh -- the release autopilot never inherits a dogfood GO,
# judges R5 at T-1 with the function T-4 uses, and closes the epic before the milestone (#3708).
#
# THE DEFECT, measured 2026-09-21 on the v0.69.0 train. autopilot's dogfood step INHERITED the
# parent's T-2 GO ("bump diff = version surface only") and wrote a receipt of its own shape.
# check_publish_preflight.sh R5 at T-4 reads `.dogfood/receipt-*.json` in the tag's worktree, found
# none, and refused: "FAIL  R5 no dogfood receipt under …/wt/.dogfood", 14:40:10Z STOP, 43 min after
# tagging. The recovery ran the real dogfood then (+~35 min, tag already public). The parent's GO is
# also at the OLD version, so rows keyed on the version (check_model_ladder) were never measured at
# the release version. Separately, `close` required 0 open milestone items, while the epic -- which
# is IN the milestone -- was never closed by anything: 0.68.2's #3477 was closed by hand.
#
# WHAT THIS RUNS. The real scripts/release/autopilot.sh, through its own from/to steps
# (`dogfood dogfood`, `close close`), in a throwaway fixture: a local bare origin whose main holds a
# PARENT commit (9.9.8) and a version-surface-only bump (9.9.8 -> 9.9.9), a T-2 GO receipt for the
# parent -- exactly the shape the old code inherited on -- a real copy of check_publish_preflight.sh,
# and stubs for gh (milestone, merge commit, epic and milestone state), the cargo-metadata call and
# dogfood.sh (writes the receipt a real run writes, with a verdict/version/commit the row picks).
#
# THE ROWS
#   never-inherits     parent GO + version-only diff: dogfood.sh still RUNS, no inherited receipt,
#                      "DOGFOOD GO at <MC> (R5 holds at T-1)".
#   t4-agrees-go       that receipt, after tagging, is accepted by the FULL T-4 gate's R5 -- the
#                      same file, the same function.
#   nogo-stops         a NO-GO dogfood STOPs at T-1.
#   t1-r5-version      the dogfood exits 0 but its receipt names another version: STOPs at T-1
#                      ("T-1 R5 refused") -- and the T-4 gate's R5 refuses the same receipt.
#   t1-r5-commit       the receipt names the parent commit: STOPs at T-1.
#   t1-r5-absent       the dogfood exits 0 and writes NO receipt: STOPs at T-1.
#   close-epic-last    the epic is the milestone's only open item: the epic is closed, then the
#                      milestone.
#   close-other-open   another item is open: STOP naming it; neither the epic nor the milestone closes.
# THE MUTANTS -- one per claim (#3708 done_when 2, "A mutant for each"); each must turn its row RED
#   inherit               the pre-#3708 dogfood step, verbatim            -> never-inherits
#   drop-nogo-stop        the NO-GO `die` line deleted                    -> nogo-stops
#   drop-t1-r5 (x3)       the --receipt-only line deleted                 -> t1-r5-version, -commit, -absent
#   t1-reads-another-file check_publish_preflight.sh --receipt-only reads
#                         a receipt dir T-4 does not                      -> never-inherits
#   t1-skips-rule-r5      --receipt-only no longer calls rule_r5()        -> t1-r5-version
#   drop-epic-close       the `gh issue close "$EPIC"` line deleted       -> close-epic-last
#   drop-others-check     the "besides epic" refusal deleted              -> close-other-open
#
# Exit 0 = every row green and every mutant killed. 1 = a row RED or a mutant survived.
# 2 = ENV: a subject or a mutation anchor is missing -- the table judged nothing.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
SUBJECT="$ROOT/scripts/release/autopilot.sh"
PARAMS="$ROOT/scripts/release/lib_release_params.sh"
PREFLIGHT="$ROOT/scripts/check_publish_preflight.sh"
DOGFOOD_BLOCK_START='# 2. dogfood: the R5 receipt'
T1_ANCHOR='bash scripts/check_publish_preflight.sh --receipt-only'
CLOSE_ANCHOR='gh issue close "$EPIC"'
OTHERS_ANCHOR='[ -z "${others// /}" ] ||'
NOGO_ANCHOR='|| die "dogfood pre-publish NO-GO'
RO_R5_ANCHOR='    if ! rule_r5 "$root" "$head" "$version"; then'
RO_DIR_ANCHOR='rdir="${PUBLISH_PREFLIGHT_RECEIPT_DIR:-$root/.dogfood}"'

env_die() { printf 'ENV   %s -- the table judged nothing, not a pass\n' "$*" >&2; exit 2; }
for f in "$SUBJECT" "$PARAMS" "$PREFLIGHT"; do [ -r "$f" ] || env_die "no $f"; done
for t in git python3; do command -v "$t" > /dev/null 2>&1 || env_die "no $t"; done
for a in "$DOGFOOD_BLOCK_START" "$T1_ANCHOR" "$CLOSE_ANCHOR" "$OTHERS_ANCHOR" "$NOGO_ANCHOR"; do
    grep -qF -- "$a" "$SUBJECT" || env_die "autopilot.sh has no '$a' line -- the subject moved"
done
for a in "$RO_R5_ANCHOR" "$RO_DIR_ANCHOR"; do
    grep -qF -- "$a" "$PREFLIGHT" || env_die "check_publish_preflight.sh has no '$a' line -- the subject moved"
done

TMP=$(mktemp -d) || exit 2
# SEC011: validate before rm -rf. An empty or '/' value must never reach it.
cleanup() { case "${TMP:-}" in ''|/) return 0 ;; *) [ -d "$TMP" ] && rm -rf -- "$TMP" ;; esac; }
trap cleanup EXIT

# Hermetic git: no global hooks, signing or identity from the operator's config.
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
export GIT_AUTHOR_NAME=fixture GIT_AUTHOR_EMAIL=fixture@example.invalid
export GIT_COMMITTER_NAME=fixture GIT_COMMITTER_EMAIL=fixture@example.invalid

mkdir -p "$TMP/bin"
# gh: milestone 7 titled 9.9.9; PR 99 merged as $FX_MC; epic 9002, whose state lives in
# $FX_STATE/epic_state; the milestone's items in $FX_STATE/open_items. Every call is logged.
cat > "$TMP/bin/gh" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FX_STATE/gh.log"
open_items() { local n; for n in $(cat "$FX_STATE/open_items"); do
    [ "$n" = 9002 ] && [ "$(cat "$FX_STATE/epic_state")" != OPEN ] && continue; printf '%s\n' "$n"; done; }
case "${1:-}" in
  api)
    shift
    [ "${1:-}" = -X ] && { printf '%s\n' "$*" >> "$FX_STATE/patch.log"; printf '{}\n'; exit 0; }
    path=$1; shift; jqarg=""
    while [ $# -gt 0 ]; do [ "$1" = --jq ] && jqarg=${2:-}; shift; done
    case "$path" in
      *'milestones?state=all'*) printf '7\n' ;;
      *'issues?milestone=7'*)
        excl=$(printf '%s' "$jqarg" | sed -n 's/.*select(.number != \([0-9]*\)).*/\1/p')
        open_items | while IFS= read -r n; do [ -n "$excl" ] && [ "$n" = "$excl" ] && continue; printf '%s\n' "$n"; done ;;
      */milestones/7) open_items | grep -c . ;;
      *) exit 1 ;;
    esac ;;
  pr) printf '%s\n' "$FX_MC" ;;
  release) printf 'https://github.com/paiml/aprender/releases/tag/v9.9.9\n' ;;
  issue)
    case "${2:-}" in
      comment) : ;;
      view) cat "$FX_STATE/epic_state" ;;
      close) printf 'CLOSED\n' > "$FX_STATE/epic_state" ;;
      *) exit 1 ;;
    esac ;;
  *) exit 1 ;;
esac
STUB
# cargo: only `metadata`, answering the version in the manifest it is pointed at
mkdir -p "$TMP/cargo/bin"
cat > "$TMP/cargo/bin/cargo" <<'STUB'
#!/usr/bin/env bash
[ "${1:-}" = metadata ] || exit 0
m=""; prev=""
for a in "$@"; do [ "$prev" = --manifest-path ] && m=$a; prev=$a; done
[ -n "$m" ] || m="$PWD/Cargo.toml"
v=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$m" | head -n 1)
printf '{"packages":[{"id":"fx","name":"fx","version":"%s","manifest_path":"%s","dependencies":[],"targets":[]}],"workspace_members":["fx"],"resolve":null}\n' "$v" "$m"
STUB
chmod +x "$TMP/bin/gh" "$TMP/cargo/bin/cargo"

# fixture NAME SUBJECT [PREFLIGHT] -> $TMP/NAME: origin with parent (9.9.8) + bump (9.9.9), the main
# checkout, the state dir with the parent's T-2 GO receipt. Returns 2 if it cannot be built.
fixture() {
    local d="$TMP/$1" subject=$2 preflight=${3:-$PREFLIGHT} r
    r="$d/repo"
    mkdir -p "$r/scripts/release" "$d/ap" "$d/state" || return 2
    cp -- "$subject" "$r/scripts/release/autopilot.sh" && cp -- "$PARAMS" "$r/scripts/release/lib_release_params.sh" \
        && cp -- "$preflight" "$r/scripts/check_publish_preflight.sh" || return 2
    printf '#!/usr/bin/env bash\nexit 0\n' > "$r/scripts/bump-version.sh"
    # dogfood.sh: writes the receipt a real run writes; the row picks verdict/version/commit
    cat > "$r/scripts/dogfood.sh" <<'STUB'
#!/usr/bin/env bash
printf 'ran %s\n' "$*" >> "$FX_STATE/dogfood.log"
v=${FX_DOGFOOD_VERDICT:-GO}
c=${FX_RECEIPT_COMMIT:-$(git rev-parse HEAD)}
mkdir -p .dogfood
[ "${FX_NO_RECEIPT:-0}" = 1 ] || printf '{"crate":"fx","version":"%s","timestamp":"20260921T000000Z","commit":"%s","gates":[],"phase":"pre-publish","deferred":[],"verdict":"%s"}\n' \
    "${FX_RECEIPT_VERSION:-9.9.9}" "$c" "$v" > .dogfood/receipt-20260921T000000Z.json
printf 'VERDICT: %s\n' "$v"
[ "$v" = GO ]
STUB
    printf '.dogfood/\ntarget/\n' > "$r/.gitignore"
    printf '[package]\nname = "fx"\nversion = "9.9.8"\nedition = "2021"\n' > "$r/Cargo.toml"
    printf '# Changelog\n' > "$r/CHANGELOG.md"
    git init -q --bare -b main "$d/origin.git" && git -C "$r" init -q -b main \
        && git -C "$r" add -A && git -C "$r" commit -q -m parent \
        && sed -i 's/^version = "9.9.8"$/version = "9.9.9"/' "$r/Cargo.toml" \
        && printf '# Changelog\n\n## [9.9.9]\n' > "$r/CHANGELOG.md" \
        && git -C "$r" commit -q -am 'release: 9.9.9' \
        && git -C "$r" remote add origin "$d/origin.git" && git -C "$r" push -q origin main \
        && git -C "$r" fetch -q origin || return 2
    git -C "$r" rev-parse HEAD > "$d/mc"
    local parent; parent=$(git -C "$r" rev-parse HEAD^)
    printf 'GO %s fixture\n' "$parent" > "$d/ap/preflight-$parent.verdict"
    printf 'VERDICT: GO (the parent, at 9.9.8)\n' > "$d/ap/preflight-$parent.log"
    printf 'OPEN\n' > "$d/state/epic_state"; printf '9002\n' > "$d/state/open_items"
    : > "$d/state/gh.log"; : > "$d/state/dogfood.log"; : > "$d/state/patch.log"
}

# autopilot NAME FROM TO [ENV=VAL ...] -> rc in $TMP/NAME/rc
autopilot() {
    local d="$TMP/$1" from=$2 to=$3; shift 3
    ( export RELEASE_AP="$d/ap" RELEASE_EPIC=9002 CARGO_HOME="$TMP/cargo" PATH="$TMP/bin:$PATH" \
          FX_STATE="$d/state" FX_MC="$(cat "$d/mc")"
      for kv in "$@"; do export "${kv?}"; done
      bash "$d/repo/scripts/release/autopilot.sh" 9.9.9 99 "$from" "$to" ) > "$d/out.log" 2>&1
    printf '%s\n' "$?" > "$d/rc"
}

# t4_gate NAME -> the FULL publish gate's output on the tagged release worktree (R5 is what we read)
t4_gate() {
    local d="$TMP/$1"
    git -C "$d/ap/wt" tag v9.9.9 > /dev/null 2>&1
    ( cd "$d/ap/wt" && PATH="$TMP/cargo/bin:$PATH" bash scripts/check_publish_preflight.sh ) 2>&1
}

fails=0; rows=0
row() { # row NAME RC MESSAGE
    rows=$((rows + 1))
    if [ "$2" = 0 ]; then printf 'ok    %s\n' "$1"
    elif [ "$2" = 2 ]; then env_die "row $1 could not build its fixture"
    else printf 'FAIL  %s: %s\n' "$1" "$3" >&2; fails=$((fails + 1)); fi
}

# ---- row bodies, parameterised by (autopilot, preflight) so the mutants can reuse them -------
never_inherits() { # TAG AUTOPILOT [PREFLIGHT] -> 0 green
    local n="inherit-$1" d; d="$TMP/inherit-$1"
    fixture "$n" "$2" "${3:-}" || return 2
    autopilot "$n" dogfood dogfood
    [ "$(cat "$d/rc")" = 0 ] || { printf 'autopilot exited %s: %s\n' "$(cat "$d/rc")" "$(tail -1 "$d/ap/STATUS" 2>/dev/null)"; return 1; }
    grep -q '^ran --phase pre-publish' "$d/state/dogfood.log" || { printf 'dogfood.sh never ran: the parent GO was inherited\n'; return 1; }
    [ ! -e "$d/ap/dogfood-inherited.receipt" ] || { printf 'an inherited receipt was written\n'; return 1; }
    grep -qF "DOGFOOD GO at $(cat "$d/mc") (R5 holds at T-1)" "$d/ap/STATUS" || { printf 'no "DOGFOOD GO … (R5 holds at T-1)" line\n'; return 1; }
    return 0
}
nogo_stops() { # TAG AUTOPILOT
    local n="nogo-$1" d; d="$TMP/nogo-$1"
    fixture "$n" "$2" || return 2
    autopilot "$n" dogfood dogfood FX_DOGFOOD_VERDICT=NO-GO
    [ "$(cat "$d/rc")" != 0 ] || { printf 'autopilot passed a NO-GO dogfood\n'; return 1; }
    grep -q 'STOP dogfood pre-publish NO-GO' "$d/ap/STATUS" || { printf 'it stopped, but not on the NO-GO: %s\n' "$(tail -1 "$d/ap/STATUS")"; return 1; }
    return 0
}
t1_r5() { # TAG VARIANT(version|commit|absent) AUTOPILOT [PREFLIGHT] -> 0 when autopilot STOPs at T-1 on R5
    local n="r5$2-$1" d kv; d="$TMP/r5$2-$1"
    fixture "$n" "$3" "${4:-}" || return 2
    case "$2" in
        version) kv=FX_RECEIPT_VERSION=9.9.8 ;;
        commit)  kv="FX_RECEIPT_COMMIT=$(git -C "$d/repo" rev-parse HEAD^)" ;;
        absent)  kv=FX_NO_RECEIPT=1 ;;
    esac
    autopilot "$n" dogfood dogfood "$kv"
    [ "$(cat "$d/rc")" != 0 ] || { printf 'autopilot passed T-1 with a receipt R5 refuses (%s)\n' "$2"; return 1; }
    grep -q 'STOP T-1 R5 refused' "$d/ap/STATUS" || { printf 'it stopped, but not on T-1 R5: %s\n' "$(tail -1 "$d/ap/STATUS")"; return 1; }
    return 0
}
close_epic_last() { # TAG AUTOPILOT
    local n="close-$1" d; d="$TMP/close-$1"
    fixture "$n" "$2" || return 2
    autopilot "$n" close close
    [ "$(cat "$d/rc")" = 0 ] || { printf 'close exited %s: %s\n' "$(cat "$d/rc")" "$(tail -1 "$d/ap/STATUS")"; return 1; }
    grep -q '^issue close 9002' "$d/state/gh.log" || { printf 'the epic was never closed\n'; return 1; }
    grep -q 'milestones/7' "$d/state/patch.log" || { printf 'the milestone was never closed\n'; return 1; }
    return 0
}
close_other_open() { # TAG AUTOPILOT
    local n="other-$1" d; d="$TMP/other-$1"
    fixture "$n" "$2" || return 2
    printf '9002 9010\n' > "$d/state/open_items"
    autopilot "$n" close close
    [ "$(cat "$d/rc")" != 0 ] || { printf 'close exited 0 with #9010 still open\n'; return 1; }
    ! grep -q '^issue close' "$d/state/gh.log" || { printf 'the epic was closed while #9010 is open\n'; return 1; }
    [ ! -s "$d/state/patch.log" ] || { printf 'the milestone was PATCHed while #9010 is open\n'; return 1; }
    grep -q 'besides epic #9002: 9010' "$d/ap/STATUS" || { printf 'the STOP never named #9010: %s\n' "$(tail -1 "$d/ap/STATUS")"; return 1; }
    return 0
}

# ---- the rows ---------------------------------------------------------------------------------
msg=$(never_inherits real "$SUBJECT"); row never-inherits "$?" "$msg"

if [ -d "$TMP/inherit-real/ap/wt" ]; then
    out=$(t4_gate inherit-real)
    grep -qE "^ok    R5 dogfood receipt receipt-20260921T000000Z.json: GO for $(cut -c1-9 "$TMP/inherit-real/mc") at 9.9.9" <<< "$out"
    row t4-agrees-go "$?" "the full T-4 gate did not accept the T-1 receipt: $(grep 'R5' <<< "$out" | head -1)"
else
    row t4-agrees-go 1 "never-inherits left no release worktree"
fi

msg=$(nogo_stops real "$SUBJECT"); row nogo-stops "$?" "$msg"

msg=$(t1_r5 real version "$SUBJECT"); rc=$?
if [ "$rc" = 0 ]; then
    out=$(t4_gate r5version-real)
    grep -q '^FAIL  R5 dogfood receipt' <<< "$out" || { rc=1; msg="T-1 refused but the T-4 gate's R5 did not: $(grep 'R5' <<< "$out" | head -1)"; }
fi
row t1-r5-version "$rc" "$msg"
msg=$(t1_r5 real commit "$SUBJECT"); row t1-r5-commit "$?" "$msg"
msg=$(t1_r5 real absent "$SUBJECT"); row t1-r5-absent "$?" "$msg"

msg=$(close_epic_last real "$SUBJECT"); row close-epic-last "$?" "$msg"
msg=$(close_other_open real "$SUBJECT"); row close-other-open "$?" "$msg"

# ---- the mutants ------------------------------------------------------------------------------
# killed NAME RC MSG -- a mutant row is green when the row it targets went RED (rc 1)
killed() {
    [ "$2" = 2 ] && env_die "$1: the mutant could not build its fixture"
    [ "$2" != 0 ]; row "mutant $1 (${3:-survived})" "$?" "the mutant PASSED -- the row does not discriminate"
}
mut() { # SRC OUT -- refuse a mutant identical to its source
    cmp -s "$1" "$2" && env_die "$(basename "$2") is identical to its source"
    return 0
}

# inherit: the pre-#3708 dogfood step, verbatim (v0.69.0's autopilot.sh at 225b2a9ab)
python3 - "$SUBJECT" "$TMP/m-inherit.sh" "$DOGFOOD_BLOCK_START" <<'PY' || env_die "inherit mutant"
import sys
src, dst, start = sys.argv[1:4]
s = open(src).read()
a = s.index(start); b = s.index("\nfi\n", a) + len("\nfi\n")
old = """# 2. dogfood: the R5 receipt, pre-publish, FULL, on THIS commit
if run_step dogfood; then
  # Inherited T-2 (operator 2026-09-17): if the bump diff touches only the version surface, the parent-sha
  # preflight GO is inherited and recorded; any other path -> full T-2.
  parent=$(git rev-parse "$MC^1"); inherit=0
  if [ -f "$AP/preflight-$parent.verdict" ] && grep -q '^GO ' "$AP/preflight-$parent.verdict"; then
    other=$(git diff --name-only "$parent" "$MC" | grep -vE '^(Cargo\\.toml|Cargo\\.lock|CHANGELOG\\.md|crates/[^/]+/Cargo\\.toml|crates/facades/[^/]+/Cargo\\.toml|crates/facades/Cargo\\.lock)$' | wc -l)
    [ "$other" -eq 0 ] && inherit=1
  fi
  if [ $inherit -eq 1 ]; then
    cp "$AP/preflight-$parent.log" "$AP/dogfood-pre-publish.log"
    printf 'inherited_from: %s\\nrelease_commit: %s\\nbump_diff: version surface only\\n' "$parent" "$MC" > "$AP/dogfood-inherited.receipt"
    say "DOGFOOD GO at $MC (INHERITED from parent $parent preflight GO; bump diff = version surface only; receipt $AP/dogfood-inherited.receipt)"
  else
  bash scripts/dogfood.sh --phase pre-publish > "$AP/dogfood-pre-publish.log" 2>&1; rc=$?
  grep -E 'VERDICT' "$AP/dogfood-pre-publish.log" >> "$STATUS"
  [ $rc -eq 0 ] || die "dogfood pre-publish NO-GO rc=$rc ($AP/dogfood-pre-publish.log)"
  [ -z "$(git status --porcelain)" ] || die "tree dirty after dogfood: $(git status --porcelain | head -3 | tr '\\n' ' ')"
  say "DOGFOOD GO at $MC"
  fi
fi
"""
open(dst, "w").write(s[:a] + old + s[b:])
PY
mut "$SUBJECT" "$TMP/m-inherit.sh"
msg=$(never_inherits m-inherit "$TMP/m-inherit.sh"); killed "inherit is killed by never-inherits" "$?" "$msg"

grep -vF -- "$NOGO_ANCHOR" "$SUBJECT" > "$TMP/m-nogo.sh"; mut "$SUBJECT" "$TMP/m-nogo.sh"
msg=$(nogo_stops m-nogo "$TMP/m-nogo.sh"); killed "drop-nogo-stop is killed by nogo-stops" "$?" "$msg"

grep -vF -- "$T1_ANCHOR" "$SUBJECT" > "$TMP/m-t1.sh"; mut "$SUBJECT" "$TMP/m-t1.sh"
for v in version commit absent; do
    msg=$(t1_r5 "m-t1" "$v" "$TMP/m-t1.sh"); killed "drop-t1-r5 is killed by t1-r5-$v" "$?" "$msg"
done

# the T-1 end of R5 reading a receipt dir the T-4 end does not: "the same file" is load-bearing
python3 - "$PREFLIGHT" "$TMP/m-dir.sh" "$RO_DIR_ANCHOR" <<'PY' || env_die "t1-reads-another-file mutant"
import sys
src, dst, anchor = sys.argv[1:4]
s = open(src).read()
# only receipt_gate's reads are redirected: rule_r5 takes an optional 4th arg (the dir) in the mutant
s = s.replace(anchor, 'rdir="${4:-${PUBLISH_PREFLIGHT_RECEIPT_DIR:-$root/.dogfood}}"', 1)
s = s.replace('    if ! rule_r5 "$root" "$head" "$version"; then', '    if ! rule_r5 "$root" "$head" "$version" "$root/.dogfood-t1"; then', 1)
open(dst, "w").write(s)
PY
mut "$PREFLIGHT" "$TMP/m-dir.sh"
msg=$(never_inherits m-dir "$SUBJECT" "$TMP/m-dir.sh"); killed "t1-reads-another-file is killed by never-inherits" "$?" "$msg"

# --receipt-only without rule_r5(): "the same function" is load-bearing
python3 - "$PREFLIGHT" "$TMP/m-ro.sh" "$RO_R5_ANCHOR" <<'PY' || env_die "t1-skips-rule-r5 mutant"
import sys
src, dst, anchor = sys.argv[1:4]
s = open(src).read()
assert s.count(anchor) == 1
open(dst, "w").write(s.replace(anchor, "    if ! true; then", 1))
PY
mut "$PREFLIGHT" "$TMP/m-ro.sh"
msg=$(t1_r5 m-ro version "$SUBJECT" "$TMP/m-ro.sh"); killed "t1-skips-rule-r5 is killed by t1-r5-version" "$?" "$msg"

grep -vF -- "$CLOSE_ANCHOR" "$SUBJECT" > "$TMP/m-close.sh"; mut "$SUBJECT" "$TMP/m-close.sh"
msg=$(close_epic_last m-close "$TMP/m-close.sh"); killed "drop-epic-close is killed by close-epic-last" "$?" "$msg"

grep -vF -- "$OTHERS_ANCHOR" "$SUBJECT" > "$TMP/m-others.sh"; mut "$SUBJECT" "$TMP/m-others.sh"
msg=$(close_other_open m-others "$TMP/m-others.sh"); killed "drop-others-check is killed by close-other-open" "$?" "$msg"

# VACUITY FLOOR: a table that ran fewer rows than it declares is not a pass.
[ "$rows" -ge 17 ] || { printf 'VACUOUS %s row(s) ran, fewer than the 17 declared\n' "$rows" >&2; exit 1; }
[ "$fails" -eq 0 ] || { printf 'RED   %s of %s row(s) failed\n' "$fails" "$rows" >&2; exit 1; }
printf 'PASS  %s row(s): autopilot never inherits a dogfood GO, judges R5 at T-1 with the T-4 function, and closes the epic before the milestone (#3708)\n' "$rows"
