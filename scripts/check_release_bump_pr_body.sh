#!/usr/bin/env bash
# check_release_bump_pr_body.sh -- the release bump PR's body must pass §6 R-2 BY
# CONSTRUCTION, and prepare_bump.sh --ship must refuse to open one that does not (#3699).
#
# THE DEFECT, measured 2026-09-21. `prepare_bump.sh 0.69.0 --ship` opened #3698, whose
# body embeds the CHANGELOG [0.69.0] section. That section cites EPIC #3080 and context
# refs ("refs #3602", "(#3571 layer 1)"), and ci.yml's step "A PR body must close every
# issue it cites, or say why not (§6 R-2)" failed it: "non-closing ref(s) with no
# keep-open reason: #3080 #3545 #3571 #3602 #3658". 0.68.2's bump #3498 needed the same
# hand fix. Because that step reads the FROZEN event payload, editing the body re-ran
# nothing: it cost a cancel, a force-cancel and a close+reopen.
#
# WHAT THIS RUNS. The real scripts/release/prepare_bump.sh --ship, end to end, in a
# throwaway fixture: a copy of the three scripts it needs, a bump worktree cloned from a
# LOCAL bare origin, a T-2 GO receipt, and stubs for `gh` (records calls, captures the
# --body-file), `cargo` and the arm helper. Reference kinds come from the guard's own
# PR_CLOSES_REF_KIND_CMD seam, so nothing touches the network, a token or the live train.
#
#   9001 a pull request    every other number an OPEN issue    9002 plays the epic
#
# THE ROWS
#   guard-case-table  check_pr_closes_issue.sh --self-test, the table that pins --list-owed.
#                   guard_tree.sh skips that guard as wired-with-args, and ci.yml runs it only
#                   with --body, so without this row its case table runs nowhere.
#   epic-and-refs   a CHANGELOG citing the epic, a `refs #N`, a PR and a `closes #N`: the PR is
#                   opened, its body PASSES the guard, and its ONE keep-open line is exactly
#                   "keep-open: #9002 #9005 -- <#3699's reason>": not the PR (#9001), not the
#                   closed-by-keyword ref (#9006).
#   no-keep-open    that same body with the keep-open line deleted FAILS the guard: the
#                   line is what makes it pass, not something else in the body.
#   adds-no-close   the keep-open line, alone, lists every ref it names as owed -- the
#                   line adds no closing reference (a reason reading "closes #9002" would
#                   close the epic on the bump's merge, the #3400 shape).
#   prs-only        a CHANGELOG citing only PRs gets no keep-open line and still passes.
#   landmine        a CHANGELOG carrying "no-close: #9005" is REFUSED: non-zero exit, no
#                   `gh pr create`, and no branch pushed to origin.
# THE MUTANTS (each must turn its row RED, or the table is not discriminating)
#   drop-keep-open  prepare_bump.sh without the line that writes keep-open -> epic-and-refs RED
#   drop-refusal    prepare_bump.sh without the R-2 refusal              -> landmine RED
#
# Exit 0 = every row green and both mutants killed. 1 = a row RED or a mutant survived.
# 2 = ENV: a subject or a mutation anchor is missing -- the table judged nothing.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
SUBJECT="$ROOT/scripts/release/prepare_bump.sh"
GUARD="$ROOT/scripts/check_pr_closes_issue.sh"
PARAMS="$ROOT/scripts/release/lib_release_params.sh"
KEEP_OPEN_ANCHOR='keep-open: %s'
REFUSAL_ANCHOR='bash "$CLOSES_GUARD" --body'
# #3699 done_when 1's reason, verbatim. The row pins it: the line must name exactly the owed
# refs AND carry this text, nothing else.
KEEP_OPEN_REASON='cited by the CHANGELOG for context; each closes via its own PR; the release EPIC closes at T-4'

env_die() { printf 'ENV   %s -- the table judged nothing, not a pass\n' "$*" >&2; exit 2; }
for f in "$SUBJECT" "$GUARD" "$PARAMS"; do [ -r "$f" ] || env_die "no $f"; done
command -v git > /dev/null 2>&1 || env_die "no git"
grep -qF -- "$KEEP_OPEN_ANCHOR" "$SUBJECT" || env_die "prepare_bump.sh has no '$KEEP_OPEN_ANCHOR' line -- the subject moved"
grep -qF -- "$REFUSAL_ANCHOR" "$SUBJECT" || env_die "prepare_bump.sh has no '$REFUSAL_ANCHOR' line -- the subject moved"

TMP=$(mktemp -d) || exit 2
# SEC011: validate before rm -rf. An empty or '/' value must never reach it.
cleanup() { case "${TMP:-}" in ''|/) return 0 ;; *) [ -d "$TMP" ] && rm -rf -- "$TMP" ;; esac; }
trap cleanup EXIT

# Hermetic git: no global hooks, signing or identity from the operator's config.
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
export GIT_AUTHOR_NAME=fixture GIT_AUTHOR_EMAIL=fixture@example.invalid
export GIT_COMMITTER_NAME=fixture GIT_COMMITTER_EMAIL=fixture@example.invalid

mkdir -p "$TMP/bin"
cat > "$TMP/bin/kind" <<'STUB'
#!/usr/bin/env bash
case "${1:-}" in 9001) printf 'pr' ;; *) printf 'issue' ;; esac
STUB
cat > "$TMP/bin/gh" <<'STUB'
#!/usr/bin/env bash
# records every call; `pr create` copies its --body-file and prints a PR url
printf '%s\n' "$*" >> "$FIXTURE_GH_LOG"
[ "${1:-} ${2:-}" = "pr create" ] || exit 1
while [ $# -gt 0 ]; do
    [ "$1" = "--body-file" ] && { cp -- "$2" "$FIXTURE_BODY"; break; }
    shift
done
printf 'https://github.com/paiml/aprender/pull/99999\n'
STUB
chmod +x "$TMP/bin/kind" "$TMP/bin/gh"
export PR_CLOSES_REF_KIND_CMD="$TMP/bin/kind"

# run_ship NAME SUBJECT CHANGELOG_SECTION -> the fixture dir; the exit code is in $TMP/NAME/rc
run_ship() {
    local name=$1 subject=$2 section=$3 d
    d="$TMP/$name"
    mkdir -p "$d/repo/scripts/release" "$d/ap" "$d/cargo/bin" "$d/seed/scripts"
    cp -- "$subject" "$d/repo/scripts/release/prepare_bump.sh"
    cp -- "$PARAMS" "$d/repo/scripts/release/lib_release_params.sh"
    cp -- "$GUARD" "$d/repo/scripts/check_pr_closes_issue.sh"
    printf '#!/usr/bin/env bash\nexit 0\n' > "$d/repo/scripts/arm_pr_automerge.sh"
    printf '#!/usr/bin/env bash\nexit 0\n' > "$d/cargo/bin/cargo"
    chmod +x "$d/cargo/bin/cargo"
    git -C "$d/repo" init -q && git -C "$d/repo" commit -q --allow-empty -m fixture \
        && git -C "$d/repo" tag v9.9.8 || return 2
    # the bump worktree, as the non---ship invocation leaves it: cloned from a LOCAL bare
    # origin's main, on branch release-<V>, with the CHANGELOG [9.9.9] section UNCOMMITTED
    printf '# Changelog\n\n## [Unreleased]\n\n## [9.9.8] - 2025-12-01\n\n- older\n' > "$d/seed/CHANGELOG.md"
    printf '#!/usr/bin/env bash\nexit 0\n' > "$d/seed/scripts/bump-version.sh"
    git init -q --bare -b main "$d/origin.git" \
        && git -C "$d/seed" init -q -b main && git -C "$d/seed" add -A \
        && git -C "$d/seed" commit -q -m seed && git -C "$d/seed" push -q "$d/origin.git" main \
        && git clone -q -b main "$d/origin.git" "$d/ap/bump" \
        && git -C "$d/ap/bump" checkout -q -b release-9.9.9 || return 2
    printf '# Changelog\n\n## [Unreleased]\n\n## [9.9.9] - 2026-01-01\n\n%s\n\n## [9.9.8] - 2025-12-01\n\n- older\n' \
        "$section" > "$d/ap/bump/CHANGELOG.md"
    printf 'GO %s fixture\n' "$(git -C "$d/ap/bump" rev-parse origin/main)" \
        > "$d/ap/preflight-$(git -C "$d/ap/bump" rev-parse origin/main).verdict"
    : > "$d/gh.log"
    ( export RELEASE_AP="$d/ap" RELEASE_EPIC=9002 CARGO_HOME="$d/cargo" \
          PATH="$TMP/bin:$PATH" FIXTURE_GH_LOG="$d/gh.log" FIXTURE_BODY="$d/body.md"
      bash "$d/repo/scripts/release/prepare_bump.sh" 9.9.9 --ship ) > "$d/out.log" 2>&1
    printf '%s\n' "$?" > "$d/rc"
}

guard_rc() { bash "$GUARD" --body "$1" > /dev/null 2>&1; printf '%s' "$?"; }

EPIC_SECTION='Train summary: the fixture train, EPIC #9002.

### Fixed

- fix(loader): the loader is fixed (#9001)
- fix(parser): layer 1 of the parser (refs #9005)
- fix(cache): the cache no longer leaks, closes #9006'
PRS_SECTION='Train summary: the fixture train.

### Fixed

- fix(loader): the loader is fixed (#9001)'
LANDMINE_SECTION="$PRS_SECTION
no-close: #9005 stays open for the next train"

# row_epic_and_refs SUBJECT -> 0 green, 1 red (prints why)
row_epic_and_refs() {
    local d="$TMP/epic-$1" line
    run_ship "epic-$1" "$2" "$EPIC_SECTION" || return 2
    [ "$(cat "$d/rc")" = 0 ] || { printf 'prepare_bump.sh --ship exited %s: %s\n' "$(cat "$d/rc")" "$(tail -1 "$d/out.log")"; return 1; }
    grep -q '^pr create' "$d/gh.log" || { printf 'no gh pr create was issued\n'; return 1; }
    [ "$(guard_rc "$d/body.md")" = 0 ] || { printf 'the opened body FAILS the guard: %s\n' "$(bash "$GUARD" --body "$d/body.md" 2>&1)"; return 1; }
    [ "$(grep -c '^keep-open:' "$d/body.md")" = 1 ] || { printf 'expected ONE keep-open line, found %s\n' "$(grep -c '^keep-open:' "$d/body.md")"; return 1; }
    line=$(grep '^keep-open:' "$d/body.md")
    [ "$line" = "keep-open: #9002 #9005 -- $KEEP_OPEN_REASON" ] \
        || { printf 'keep-open line is not "keep-open: #9002 #9005 -- <#3699 reason>": %s\n' "$line"; return 1; }
    return 0
}

fails=0; rows=0
row() { # row NAME RESULT_RC MESSAGE
    rows=$((rows + 1))
    if [ "$2" = 0 ]; then printf 'ok    %s\n' "$1"
    elif [ "$2" = 2 ]; then env_die "row $1 could not build its fixture"
    else printf 'FAIL  %s: %s\n' "$1" "$3" >&2; fails=$((fails + 1)); fi
}

# The generator trusts the guard's classifier, so the guard's own case table (which pins
# --list-owed) must run. Nothing else runs it: guard_tree.sh skips check_pr_closes_issue.sh
# as "wired-with-args in ci.yml", and ci.yml calls it only with --body.
out=$(bash "$GUARD" --self-test 2>&1); rc=$?
row guard-case-table "$rc" "check_pr_closes_issue.sh --self-test rc=$rc: $(printf '%s' "$out" | tail -1)"

msg=$(row_epic_and_refs real "$SUBJECT"); row epic-and-refs "$?" "$msg"
BODY="$TMP/epic-real/body.md"

if [ -f "$BODY" ]; then
    grep -v '^keep-open:' "$BODY" > "$TMP/no-keep-open.md"
    rc=$(guard_rc "$TMP/no-keep-open.md")
    [ "$rc" = 1 ]; row no-keep-open "$?" "the body without its keep-open line got guard rc=$rc, wanted 1"

    grep '^keep-open:' "$BODY" > "$TMP/line-only.md"
    named=$(grep -oE '#[0-9]+' "$TMP/line-only.md" | tr '\n' ' ')
    listed=$(bash "$GUARD" --list-owed --body "$TMP/line-only.md" | tr '\n' ' ')
    [ -n "$named" ] && [ "$named" = "$listed" ]
    row adds-no-close "$?" "the keep-open line names '$named' but only '$listed' are still owed -- it closes something"
else
    row no-keep-open 1 "no body was captured from epic-and-refs"
    row adds-no-close 1 "no body was captured from epic-and-refs"
fi

run_ship prs-only "$SUBJECT" "$PRS_SECTION" || env_die "prs-only fixture"
d="$TMP/prs-only"
[ "$(cat "$d/rc")" = 0 ] && [ -f "$d/body.md" ] && ! grep -q '^keep-open:' "$d/body.md" \
    && [ "$(guard_rc "$d/body.md")" = 0 ]
row prs-only "$?" "rc=$(cat "$d/rc"); a PR-only CHANGELOG must open a PR with no keep-open line that passes the guard"

# row_landmine NAME SUBJECT -> 0 when prepare_bump refused before push and before gh pr create
row_landmine() {
    local d="$TMP/$1"
    run_ship "$1" "$2" "$LANDMINE_SECTION" || return 2
    [ "$(cat "$d/rc")" != 0 ] || { printf 'prepare_bump.sh --ship exited 0 on a landmine body\n'; return 1; }
    ! grep -q '^pr create' "$d/gh.log" || { printf 'gh pr create was issued for a body the guard fails\n'; return 1; }
    [ -z "$(git -C "$d/origin.git" branch --list 'release-9.9.9')" ] || { printf 'the branch was pushed before the refusal\n'; return 1; }
    return 0
}
msg=$(row_landmine landmine-real "$SUBJECT"); row landmine "$?" "$msg"

# --- the mutants -------------------------------------------------------------
grep -vF -- "$KEEP_OPEN_ANCHOR" "$SUBJECT" > "$TMP/mutant-drop-keep-open.sh"
cmp -s "$SUBJECT" "$TMP/mutant-drop-keep-open.sh" && env_die "drop-keep-open mutant is identical to the subject"
msg=$(row_epic_and_refs mutant "$TMP/mutant-drop-keep-open.sh"); mrc=$?
[ "$mrc" = 2 ] && env_die "drop-keep-open mutant could not build its fixture"
[ "$mrc" != 0 ]; row "mutant drop-keep-open is killed by epic-and-refs (${msg:-survived})" "$?" "the mutant PASSED epic-and-refs -- the row does not discriminate"

grep -vF -- "$REFUSAL_ANCHOR" "$SUBJECT" > "$TMP/mutant-drop-refusal.sh"
cmp -s "$SUBJECT" "$TMP/mutant-drop-refusal.sh" && env_die "drop-refusal mutant is identical to the subject"
msg=$(row_landmine landmine-mutant "$TMP/mutant-drop-refusal.sh"); mrc=$?
[ "$mrc" = 2 ] && env_die "drop-refusal mutant could not build its fixture"
[ "$mrc" != 0 ]; row "mutant drop-refusal is killed by landmine (${msg:-survived})" "$?" "the mutant PASSED landmine -- the row does not discriminate"

# VACUITY FLOOR: a table that ran fewer rows than it declares is not a pass.
[ "$rows" -ge 8 ] || { printf 'VACUOUS %s row(s) ran, fewer than the 8 declared\n' "$rows" >&2; exit 1; }
[ "$fails" -eq 0 ] || { printf 'RED   %s of %s row(s) failed\n' "$fails" "$rows" >&2; exit 1; }
printf 'PASS  %s row(s): the bump PR body passes §6 R-2 by construction, and prepare_bump.sh refuses one that does not (#3699)\n' "$rows"
