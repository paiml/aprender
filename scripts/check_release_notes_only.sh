#!/usr/bin/env bash
# check_release_notes_only.sh -- a notes-only commit is ready to tag on saved results (#4939).
#
#   bash scripts/check_release_notes_only.sh   # the rows below; exit 0 only if every row holds
#
# THE PLANTED TEST (T1). A release commit M was measured: its dogfood receipt, its CRUX-smoke GO and
# its readiness Pass all name M. N is M plus a CHANGELOG.md edit and nothing else. On a real fixture
# repository (git, cargo metadata, a bare origin), with NO lane rerun, N must pass every reuse gate
# the tag step applies and get its tag, in under 10 min wall time:
#   - the autopilot's worktree MOVES from M to N, keeping the saved .dogfood receipt, re-reads the
#     release notes from N's CHANGELOG, and resumes at the tag step (the extracted block);
#   - R5 (check_publish_preflight.sh --receipt-only, the real script) accepts M's receipt for N;
#   - cut_tag (extracted from autopilot.sh) cuts the tag AT N on M's CRUX-smoke GO under the standing
#     policy, and on M's readiness Pass without it.
# It does not time the publish dry run (`rc_publish_gate.sh --verify`), which compiles every tarball
# on N in the tag step: that is a build, not a saved result, and is measured by the release itself.
#
# THE ROWS
#   C  case table   notes_only on 1 must-match and 11 must-not-match shapes (lib_notes_only.sh).
#   T1 planted      N: worktree move + R5 + cut_tag, < 600 s, tag at N, no lane step run.
#   T2 refusals     S (M plus a source edit): the worktree is rebuilt, R5 refuses, cut_tag cuts no tag.
#   M  mutants      each weakening of the predicate or of a consumer turns a row red.

set -euo pipefail
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
LIB="$ROOT/scripts/release/lib_notes_only.sh"
AP_SH="$ROOT/scripts/release/autopilot.sh"
PRE="$ROOT/scripts/check_publish_preflight.sh"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/notes-only.XXXXXX")
trap 'rm -rf -- "${TMP:?}"' EXIT
fails=0
row() { if [ "$2" = 0 ]; then printf 'ok    %s\n' "$1"; else printf 'FAIL  %s\n' "$1"; fails=$((fails + 1)); fi; }
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t GIT_CONFIG_GLOBAL=/dev/null

# ---- the fixture: P (root) -> M (measured); each shape is a commit made from M (or from P) ----
R="$TMP/repo"
git init -q -b main "$R"
git init -q --bare "$TMP/origin.git"
git -C "$R" remote add origin "$TMP/origin.git"
mkdir -p "$R/src"
printf '[package]\nname = "fx"\nversion = "0.0.0"\nedition = "2021"\n\n[workspace]\n' > "$R/Cargo.toml"
printf 'pub fn f() {}\n' > "$R/src/lib.rs"
printf '# Changelog\n\n## [0.0.0]\n\n- first draft of the notes\n\n## [0.0.0-old]\n\n- older\n' > "$R/CHANGELOG.md"
printf 'readme\n' > "$R/README.md"
git -C "$R" add -A && git -C "$R" commit -q -m P
P=$(git -C "$R" rev-parse HEAD)
printf 'pub fn f() {}\npub fn g() {}\n' > "$R/src/lib.rs"
git -C "$R" commit -q -am M
M=$(git -C "$R" rev-parse HEAD)

# shape NAME FROM CMD: commit CMD's change on a branch made at FROM; prints the new commit
shape() {
    git -C "$R" checkout -q --detach "$2"
    (cd "$R" && bash -c "$3")
    git -C "$R" add -A
    git -C "$R" commit -q -m "$1"
    git -C "$R" rev-parse HEAD
}
NOTES='sed -i "s/first draft of the notes/the corrected notes/" CHANGELOG.md'
N=$(shape notes "$M" "$NOTES")
S=$(shape src "$M" "printf 'pub fn h() {}\n' >> src/lib.rs; $NOTES")
declare -A NOT
NOT[source]=$S
NOT[version]=$(shape version "$M" "sed -i 's/^version = \"0.0.0\"/version = \"0.0.1\"/' Cargo.toml; $NOTES")
NOT[second-path]=$(shape two "$M" "$NOTES; echo more >> README.md")
# bashrs SEC010: a fixture command, run inside the mktemp -d fixture repository by shape().
# bashrs disable-next-line=SEC010
NOT[rename]=$(shape rename "$M" "git mv CHANGELOG.md CHANGES.md")
NOT[exec-bit]=$(shape exec "$M" "$NOTES; chmod +x CHANGELOG.md")
NOT[symlink]=$(shape symlink "$M" "mv CHANGELOG.md notes.md; ln -s notes.md CHANGELOG.md")
NOT[deleted]=$(shape deleted "$M" "git rm -q CHANGELOG.md")
NOT[sibling]=$(shape sibling "$P" "printf 'pub fn f() {}\npub fn g() {}\n' > src/lib.rs; $NOTES")
NOT[same]=$M
NOT[unknown]=0123456789abcdef0123456789abcdef01234567
# a tree whose build reads the notes: M2 includes them, N2 edits them only
I0=$(shape inc0 "$M" "printf 'pub const N: &str = include_str!(\"../CHANGELOG.md\");\n' >> src/lib.rs")
I1=$(shape inc1 "$I0" "$NOTES")
B0=$(shape bld0 "$M" "printf 'fn main() { println!(\"cargo:rerun-if-changed=CHANGELOG.md\"); }\n' > build.rs")
B1=$(shape bld1 "$B0" "$NOTES")
git -C "$R" checkout -q --detach "$N"

# C: the case table, judged by the LIB under test
cases() {  # LIB -> 0 when every shape is judged right; prints the first wrong one
    local lib=$1 k
    ( . "$lib"; cd "$R" || exit 2
      notes_only "$M" "$N" || { echo "      C notes (M -> N) not taken as notes-only"; exit 1; }
      notes_only "${M:0:9}" "$N" || { echo "      C notes by short id not taken"; exit 1; }
      for k in "${!NOT[@]}"; do
          ! notes_only "$M" "${NOT[$k]}" || { echo "      C $k taken as notes-only"; exit 1; }
      done
      ! notes_only "$I0" "$I1" || { echo "      C include_str! of CHANGELOG taken as notes-only"; exit 1; }
      ! notes_only "$B0" "$B1" || { echo "      C build.rs naming CHANGELOG taken as notes-only"; exit 1; }
      [ "$(notes_only_base "$N" "${S:0:9}" "${M:0:9}")" = "$M" ] || { echo "      C notes_only_base did not pick M"; exit 1; }
      [ "$(notes_only_base "$N" "${N:0:9}")" = "$N" ] || { echo "      C notes_only_base refused N itself"; exit 1; }
      ! notes_only_base "$S" "${M:0:9}" > /dev/null || { echo "      C notes_only_base took M for a source edit"; exit 1; }
    )
}
row "C  case table: 1 must-match + $(( ${#NOT[@]} + 2 )) must-not-match shapes" "$(cases "$LIB" >&2; echo $?)"

# ---- the three consumers, each from its real file ----
# a scripts/ copy for a consumer: the real file, or a mutant of it, plus the libs it sources
mkscripts() {  # DIR [PRE-FILE [LIB-FILE [AUTOPILOT-FILE]]]
    mkdir -p "$1/scripts/release" "$1/scripts/lib"
    cp -- "${2:-$PRE}" "$1/scripts/check_publish_preflight.sh"
    cp -- "${3:-$LIB}" "$1/scripts/release/lib_notes_only.sh"
    cp -- "${4:-$AP_SH}" "$1/autopilot.sh"
    cp -- "$ROOT/scripts/lib/release_policy.sh" "$ROOT"/scripts/lib/release_policy_*.awk "$1/scripts/lib/"
}
# R5: --receipt-only on a worktree at HEAD with a GO receipt for COMMIT; status of the real script
r5() {  # DIR HEAD RECEIPT-COMMIT
    local wt="$1/r5wt-$2"
    git -C "$R" worktree add -q --detach "$wt" "$2" 2> /dev/null
    mkdir -p "$1/rc"
    printf '{"verdict":"GO","commit":"%s","version":"0.0.0","phase":"pre-publish"}\n' "$3" > "$1/rc/receipt-20261008T000000Z.json"
    PUBLISH_PREFLIGHT_ROOT="$wt" PUBLISH_PREFLIGHT_RECEIPT_DIR="$1/rc" bash "$1/scripts/check_publish_preflight.sh" --receipt-only > "$1/r5.out" 2>&1
}
# cut_tag, extracted, on a worktree at MC; real git, a bare origin; stubs only for the GitHub-side checks
cut() {  # DIR MC POLICY(covers|none) LOG-COMMIT -> status; transcript in DIR/cut.out
    local d=$1 mc=$2 pol=$3 lc=$4 wt
    wt="$d/tagwt-$mc-$pol"
    git -C "$R" worktree add -q --detach "$wt" "$mc" 2> /dev/null
    mkdir -p "$wt/scripts/lib" "$wt/contracts" "$d/ap" "$d/stub/scripts/release"
    cp -- "$d"/scripts/lib/* "$wt/scripts/lib/"
    {   printf 'ladder:\n'
        [ "$pol" != covers ] || printf '  release_policy:\n    name: crux-smoke\n    since: "0.0.0"\n    date: "d"\n    quote: "q"\n    hosts: [lambda, gx10]\n    thinking: ["off"]\n    larger_rows: nightly\n    red_row_needs: ticket\n    ticket_owner: "#1"\n    release_notes: known_failures\n'
        printf '  emergency_scopes:\n'
    } > "$wt/contracts/model-capability-ladder-v1.yaml"
    printf 'MODELS GO (CRUX smoke) on lambda and gx10 at %s: the judge passed both receipts\n' "${lc:0:9}" > "$d/ap/models-t1.log"
    printf 'ok    R8 #3715 ENFORCE PASS version=0.0.0 commit=%s pv=pv_x out_sha256=0\n' "$lc" > "$d/ap/readiness-t1.log"
    for s in release/tag_coverage_gate.sh release/carry_milestone_items.sh check_milestone_cut.sh; do
        printf '#!/usr/bin/env bash\nexit 0\n' > "$d/stub/scripts/$s"
    done
    {   printf 'set -uo pipefail\nREPO_ROOT=%q\nLOG=%q\nAP=%q\n' "$d/stub" "$d/log" "$d/ap"
        printf 'say() { printf "SAY %%s\\n" "$*"; }\ndie() { printf "DIE %%s\\n" "$*"; exit 1; }\n'
        printf '. %q || exit 2\n' "$d/scripts/release/lib_notes_only.sh"
        awk '/^ap_policy_applies\(\) \{/,/^\}/' "$d/autopilot.sh"
        awk '/^cut_tag\(\) \{/,/^\}/' "$d/autopilot.sh"
        printf 'cut_tag 0.0.0 v0.0.0-%s-%s %s\n' "${mc:0:7}" "$pol" "$mc"
    } > "$d/cut.sh"
    (cd "$wt" && bash "$d/cut.sh") > "$d/cut.out" 2>&1
    git -C "$TMP/origin.git" rev-parse -q --verify "refs/tags/v0.0.0-${mc:0:7}-$pol^{commit}" 2> /dev/null | grep -qx "$mc"
}
# the autopilot's worktree move, extracted (NOTES_BASE= ... the FROM advance); FROM after it to DIR/move.out
move() {  # DIR MC -> 0 when the block ran; prints FROM NOTES_BASE RECEIPT-KEPT HEAD on DIR/move.out
    local d=$1 mc=$2 wt="$1/apwt"
    git -C "$R" worktree add -q --detach "$wt" "$M" 2> /dev/null
    mkdir -p "$wt/.dogfood" "$d/ap"
    printf '{"verdict":"GO","commit":"%s"}\n' "$M" > "$wt/.dogfood/receipt-20261008T000000Z.json"
    {   printf 'set -uo pipefail\nWT=%q\nMC=%q\nV=0.0.0\nFROM=wait\nAP=%q\nLOG=%q\n' "$wt" "$mc" "$d/ap" "$d/log"
        printf 'say() { printf "SAY %%s\\n" "$*"; }\ndie() { printf "DIE %%s\\n" "$*"; exit 1; }\n'
        printf '. %q || exit 2\n' "$d/scripts/release/lib_notes_only.sh"
        awk '/^NOTES_BASE=""$/ { on = 1 } on { print } on && /^fi$/ && ++n == 2 { exit }' "$d/autopilot.sh"
        printf 'printf "FROM=%%s BASE=%%s KEPT=%%s HEAD=%%s\\n" "$FROM" "$NOTES_BASE" "$(ls .dogfood 2>/dev/null | wc -l)" "$(git rev-parse HEAD)"\n'
    } > "$d/move.sh"
    (cd "$R" && bash "$d/move.sh") > "$d/move.out" 2>&1
}

# T1 + T2 over one scripts/ copy; 0 when every consumer behaves
consumers() {  # [PRE-FILE [LIB-FILE [AUTOPILOT-FILE]]] -> 0 when T1 and T2 hold; WHY on stderr
    local d t0 secs
    d=$(mktemp -d "$TMP/c.XXXXXX"); mkscripts "$d" "$@"
    t0=$SECONDS
    move "$d" "$N"
    grep -qx "FROM=tag BASE=$M KEPT=1 HEAD=$N" "$d/move.out" || { echo "      T1 worktree move: $(tail -n 1 "$d/move.out")"; return 1; }
    grep -qF 'the corrected notes' "$d/ap/release_notes.md" || { echo "      T1 release notes not re-read from N"; return 1; }
    r5 "$d" "$N" "$M" || { echo "      T1 R5 refused M's receipt for N: $(grep -E '^(FAIL|ok)  +R5' "$d/r5.out")"; return 1; }
    grep -qF "reused for ${N:0:9} (notes-only" "$d/r5.out" || { echo "      T1 R5 passed without naming the reuse"; return 1; }
    cut "$d" "$N" covers "$M" || { echo "      T1 cut_tag (policy) cut no tag at N: $(grep -E '^DIE' "$d/cut.out")"; return 1; }
    cp -- "$d/cut.out" "$d/cut-policy.out"
    cut "$d" "$N" none "$M" || { echo "      T1 cut_tag (readiness) cut no tag at N: $(grep -E '^DIE' "$d/cut.out")"; return 1; }
    secs=$((SECONDS - t0))
    printf '      T1 measured: notes-only commit tag-ready in %s s on saved results (bound 600 s), no lane step run\n' "$secs" >&2
    [ "$secs" -lt 600 ] || { echo "      T1 took $secs s"; return 1; }
    # T2: a source edit gets none of it
    rm -rf -- "${d:?}/ap" "${d:?}/apwt"; git -C "$R" worktree prune
    move "$d" "$S"
    grep -qx "FROM=wait BASE= KEPT=0 HEAD=$S" "$d/move.out" || { echo "      T2 worktree for a source edit: $(tail -n 1 "$d/move.out")"; return 1; }
    ! r5 "$d" "$S" "$M" || { echo "      T2 R5 took M's receipt for a source edit"; return 1; }
    ! cut "$d" "$S" covers "$M" || { echo "      T2 cut_tag tagged a source edit on M's GO"; return 1; }
    ! cut "$d" "$S" none "$M" || { echo "      T2 cut_tag tagged a source edit on M's readiness Pass"; return 1; }
    ! cut "$d" "${NOT[version]}" covers "$M" || { echo "      T2 cut_tag tagged a version change on M's GO (#3708)"; return 1; }
    # last, so a consumer that tags the wrong commit is caught by T2 above, not by a missing log line
    grep -q '^SAY NOTES-ONLY the CRUX-smoke GO' "$d/cut-policy.out" || { echo "      T1 cut_tag did not name the reuse"; return 1; }
    return 0
}
row "T1+T2 worktree move, R5, cut_tag: notes-only tags on saved results, a source edit does not" "$(consumers >&2; echo $?)"

# M: each mutant must turn a row red
mut() {  # NAME FILE SED-EXPR WHICH(lib|pre|ap)
    local m="$TMP/mut-$1" r
    sed -E "$3" "$2" > "$m"
    cmp -s "$m" "$2" && { printf 'FAIL  mutant %s did not apply\n' "$1"; fails=$((fails + 1)); return; }
    case "$4" in
        lib) if cases "$m" > /dev/null 2>&1 && consumers "$PRE" "$m" > /dev/null 2>&1; then r=survived; else r=killed; fi ;;
        pre) if consumers "$m" > /dev/null 2>&1; then r=survived; else r=killed; fi ;;
        ap)  if consumers "$PRE" "$LIB" "$m" > /dev/null 2>&1; then r=survived; else r=killed; fi ;;
    esac
    if [ "$r" = killed ]; then printf 'ok    mutant %s killed\n' "$1"; else printf 'FAIL  mutant %s survived\n' "$1"; fails=$((fails + 1)); fi
}
mut any-diff      "$LIB" 's/^        \*\) return 1 ;;$/        *) ;;/' lib
mut no-ancestor   "$LIB" 's/^    git -C "\$g" merge-base --is-ancestor .*$/    :/' lib
mut any-mode      "$LIB" 's/":100644 100644 "\*/":"*/' lib
mut no-build-read "$LIB" 's/^    \[ \$\? = 1 \] \|\| return 1$/    :/' lib
mut r5-any-commit "$PRE" 's/^             && notes_only "\$rcommit" "\$head" "\$root"; then$/             ; then/' pre
mut r5-no-reuse   "$PRE" 's/^(             && notes_only )"\$rcommit"/\1"0000000"/' pre
mut tag-any-go    "$AP_SH" 's/base=\$\(notes_only_base "\$mc" /base=$(printf "%s\\n" "$mc"; : /' ap
mut move-rebuilds "$AP_SH" 's/^  if \[ -n "\$old" \] && notes_only "\$old" "\$MC"/  if false/' ap
mut move-any      "$AP_SH" 's/^  if \[ -n "\$old" \] && notes_only "\$old" "\$MC"/  if [ -n "$old" ]/' ap

[ "$fails" = 0 ] || { printf 'check_release_notes_only: %s row(s) red\n' "$fails"; exit 1; }
printf 'check_release_notes_only: all rows green\n'
