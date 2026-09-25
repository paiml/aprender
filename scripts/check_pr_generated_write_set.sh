#!/usr/bin/env bash
# check_pr_generated_write_set.sh — no pull request writes a GENERATED shared file:
# docs/roadmaps/roadmap.yaml or a README census line. Main regenerates both
# (operator doctrine P4, 2026-09-25; #4417).
#
# WHY THIS EXISTS
# ---------------
# Both files are aggregates that every PR used to rewrite: `pmat work add` wrote
# the roadmap aggregate, and each batch re-counted the README census (crates,
# contracts, CLI commands). When P4 landed, 8 of the 10 open PRs carried a write
# to one or both files, each with DIFFERENT numbers. Every merge therefore made
# the other seven DIRTY. check_row_pr_write_set.sh (G-11) already refused these
# writes, but only for row PRs on agent/<id> branches; this guard refuses them on
# EVERY branch. What a PR writes instead:
#   * roadmap: its fragment, docs/roadmaps/entries/<ID>.yaml. The aggregate is
#     rendered on main by the one writer, `pmat roadmap sync` (pmat#1370), and
#     check_roadmap_fragment_required.sh lets it LAG on a fragment-only diff.
#   * README: nothing. check_readme_claims.sh lets the counts lag, never overstate.
#
# THE RULE, over the PR's own diff (base..head):
#   RED  the diff writes docs/roadmaps/roadmap.yaml (add, edit, delete, or rename
#        source/destination; --no-renames).
#   RED  the diff adds or removes a README.md line that is a census line: it
#        carries a CONTRACT_COUNT marker, or it matches COUNT_RE, the same
#        extractor check_row_pr_write_set.sh and check_readme_claims.sh read (the
#        self-test proves the two regexes are byte-identical).
#   EXEMPT  a head branch regen/* whose write set is a SUBSET of {roadmap.yaml,
#        README.md}. That branch is the one writer's post-merge regen PR, and its
#        content is judged by check_roadmap_fragment_required.sh (== aggregate) and
#        check_readme_claims.sh. A regen/* branch that writes anything else is RED.
#   ALLOWED  a (branch, PR number) row of scripts/pr_generated_write_allowlist.tsv
#        whose expires_utc is still in the future. These are the 8 PRs in flight at
#        P4, by cop ruling. An expired row is RED. So is a different PR number on the
#        same branch, and so is an unknown PR number, because the binding cannot be
#        checked.
#
# SHAPES. pull_request: CI checks out refs/pull/N/merge. Its first parent is the
# base branch tip, so base = HEAD^1 and the diff is exactly this PR, whether it
# targets main or a batch/* branch. merge_group and push carry no head branch;
# they REPORT that the write set was judged on the pull_request run, and exit 0.
# Local runs: pass --base (for example origin/batch/0.70.0); otherwise
# resolve_base names merge-base(origin/main, HEAD).
#
#   bash scripts/check_pr_generated_write_set.sh [--base <ref>] [--head <ref>] [--branch <name>] [--event <name>] [--pr <N>] [--allowlist <tsv>]
#   bash scripts/check_pr_generated_write_set.sh --self-test
#
# Exit: 0 clean / exempt / allowed / REPORT · 1 RED · 2 cannot judge (never a pass).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=check_pr_generated_write_set
ROADMAP_FILE="docs/roadmaps/roadmap.yaml"
README_FILE="README.md"
ALLOWLIST_DEFAULT="scripts/pr_generated_write_allowlist.tsv"
ALLOWLIST_CAP="2026-09-26T12:00:00Z"   # cop ruling: no row may outlive the Y4 24h batch limit
ALLOWLIST_MAX_ROWS=8                    # cop ruling: exactly the 8 PRs in flight at P4; shrink-only
COUNT_RE='[0-9]+\*{0,2}( +[a-z]+){0,2} +(workspace crates?|contracts?|CLI commands?)\b'   # = check_readme_claims.sh's claim extractors: a line they do not read is not a claim
MARKER_RE='CONTRACT_COUNT_(START|END)'
REGEN_RE='^regen/'

usage() { printf 'usage: %s [--base <ref>] [--head <ref>] [--branch <name>] [--event <name>] [--pr <N>] [--allowlist <tsv>] | --self-test\n' "$PROG" >&2; exit 2; }

to_epoch() { # to_epoch <ISO-8601 UTC> -> seconds, or rc 1
    python3 -c 'import sys,datetime;print(int(datetime.datetime.strptime(sys.argv[1],"%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()))' "$1" 2>/dev/null
}

# allow_lookup <tsv> <branch> -> prints "<pr>\t<expires>" for the branch's row; rc 0 found, 1 none, 2 malformed table
allow_lookup() {
    local tsv=$1 want=$2 b p e r n=0 hit=""
    [ -f "$tsv" ] || return 1
    while IFS=$'\t' read -r b p e r || [ -n "$b" ]; do
        case "$b" in ''|'#'*) continue ;; esac
        n=$((n + 1))
        if ! [[ "$p" =~ ^[0-9]+$ ]] || ! [[ "$e" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] || [ -z "$r" ]; then
            printf '%s: ENV - malformed allowlist row %s in %s: branch=%q pr=%q expires=%q reason=%q\n' "$PROG" "$n" "$tsv" "$b" "$p" "$e" "$r" >&2
            return 2
        fi
        [ "$b" = "$want" ] && hit="$p"$'\t'"$e"
    done < "$tsv"
    [ -n "$hit" ] || return 1
    printf '%s\n' "$hit"
}

remedy() {
    printf '\nREMEDY (P4, #4417). A PR never writes a generated file; main regenerates both:\n'
    printf '    git checkout <base> -- %s          # keep your docs/roadmaps/entries/<ID>.yaml fragment\n' "$ROADMAP_FILE"
    printf '    revert the README census lines to <base>   # check_readme_claims.sh lets a count LAG\n'
    printf '  The aggregate is rendered on main by `pmat roadmap sync` (pmat#1370) through a regen/* PR.\n'
}

# judge <repo> <base> <head> <branch> <event> <pr> <now epoch> <allowlist> -> 0 / 1 / 2
judge() {
    local repo=$1 base=$2 head=$3 branch=$4 event=$5 pr=$6 now=$7 tsv=$8
    local changed hits roadmap_hit=0 f others row rc_l ap ae ae_s
    case "$event" in
        merge_group|push)
            printf 'REPORT %s: the %s shape carries no head branch; the write set was judged on the pull_request run (guard_tree in the required `gate`). Not a verdict.\n' "$PROG" "$event"
            return 0 ;;
    esac
    git -C "$repo" rev-parse --verify -q "$base^{commit}" >/dev/null || { printf '%s: ENV - base %s is not a commit here (never a pass)\n' "$PROG" "$base" >&2; return 2; }
    git -C "$repo" rev-parse --verify -q "$head^{commit}" >/dev/null || { printf '%s: ENV - head %s is not a commit here (never a pass)\n' "$PROG" "$head" >&2; return 2; }
    changed=$(git -C "$repo" diff --no-renames --name-only "$base" "$head" --) || { printf '%s: ENV - git diff %s %s failed\n' "$PROG" "$base" "$head" >&2; return 2; }
    grep -qxF -- "$ROADMAP_FILE" <<<"$changed" && roadmap_hit=1
    hits=$(git -C "$repo" diff "$base" "$head" -- "$README_FILE" | grep -E '^[-+][^-+]' | grep -E -- "$COUNT_RE|$MARKER_RE" || true)

    if [ "$roadmap_hit" = 0 ] && [ -z "$hits" ]; then
        printf 'PASS  %s: %s writes no generated file (%s changed path(s))\n' "$PROG" "${branch:-<none>}" "$(grep -c . <<<"$changed" || true)"
        return 0
    fi

    if [[ "$branch" =~ $REGEN_RE ]]; then
        others=$(grep -vxF -e "$ROADMAP_FILE" -e "$README_FILE" <<<"$changed" | grep . || true)
        if [ -z "$others" ]; then
            printf 'PASS  %s: %s is the one writer'"'"'s regen PR and writes only the generated files (content judged by check_roadmap_fragment_required.sh + check_readme_claims.sh)\n' "$PROG" "$branch"
            return 0
        fi
        printf 'FAIL  %s: regen branch %s writes generated files AND other paths; a regen PR carries nothing else:\n' "$PROG" "$branch"
        sed 's/^/        /' <<<"$others"
        return 1
    fi

    [ "$roadmap_hit" = 1 ] && printf 'FAIL  %s: %s writes %s — main regenerates it; a PR writes only its fragment\n' "$PROG" "${branch:-<none>}" "$ROADMAP_FILE"
    if [ -n "$hits" ]; then
        printf 'FAIL  %s: %s edits README census line(s) — main regenerates the counts:\n' "$PROG" "${branch:-<none>}"
        cut -c1-160 <<<"$hits" | sed 's/^/        /'
    fi

    rc_l=0; row=$(allow_lookup "$tsv" "$branch") || rc_l=$?
    case "$rc_l" in
        2) return 2 ;;
        1) printf '      %s is not in %s — strict from day one (P4)\n' "${branch:-<none>}" "$tsv"; remedy; return 1 ;;
    esac
    ap=${row%%$'\t'*}; ae=${row#*$'\t'}
    ae_s=$(to_epoch "$ae") || { printf '%s: ENV - cannot parse expiry %s\n' "$PROG" "$ae" >&2; return 2; }
    if [ -z "$pr" ]; then
        printf '      allowlist row for %s is bound to PR #%s, and this run names no PR number, so the binding cannot be checked — RED\n' "$branch" "$ap"; remedy; return 1
    fi
    if [ "$pr" != "$ap" ]; then
        printf '      allowlist row for %s is bound to PR #%s; this is PR #%s — a new PR on the same branch is strict\n' "$branch" "$ap" "$pr"; remedy; return 1
    fi
    if [ "$now" -ge "$ae_s" ]; then
        printf '      allowlist row for %s (PR #%s) EXPIRED at %s — drop the generated-file edits\n' "$branch" "$ap" "$ae"; remedy; return 1
    fi
    printf 'ALLOWED %s: %s (PR #%s) is an in-flight row of %s until %s — the FAILs above are waived, not fixed; drop them the next time the branch is touched\n' "$PROG" "$branch" "$pr" "${tsv#"$ROOT"/}" "$ae"
    return 0
}

self_test() {
    local TD R BASE red=0 n=0 NOW EXP_FUT EXP_PAST
    TD=$(mktemp -d "${TMPDIR:-/tmp}/prgenws-selftest.XXXXXX")
    # shellcheck disable=SC2064
    trap "rm -rf -- '${TD:?}'" EXIT
    R="$TD/repo"; mkdir -p "$R/docs/roadmaps/entries" "$R/crates/x/src"
    git -C "$R" init -q; git -C "$R" config user.email t@t; git -C "$R" config user.name t; git -C "$R" config commit.gpgsign false
    printf -- '- id: PMAT-1\n  title: one\n' > "$R/$ROADMAP_FILE"
    printf 'roadmap notes\n' > "$R/docs/roadmaps/README.md"
    printf '# X\n\nA prose line about contracts and crates.\n| Workspace crates | **80** workspace crates | x |\n| Provable contracts | **<!-- CONTRACT_COUNT_START -->1830<!-- CONTRACT_COUNT_END -->** provable contracts | y |\n| CLI commands | **111** CLI commands | z |\nThe tree carries <!-- CONTRACT_COUNT_START -->1830<!-- CONTRACT_COUNT_END --> contracts across kernels.\n' > "$R/README.md"
    printf 'fn main() {}\n' > "$R/crates/x/src/lib.rs"
    ( cd "$R" && git add -A && git commit -qm base )
    BASE=$(git -C "$R" rev-parse HEAD)
    NOW=$(to_epoch 2026-09-25T13:00:00Z); EXP_FUT=2026-09-26T12:00:00Z; EXP_PAST=2026-09-25T12:00:00Z
    printf '# b\tpr\texp\treason\nbatch/live\t101\t%s\tfixture\nbatch/stale\t102\t%s\tfixture\n' "$EXP_FUT" "$EXP_PAST" > "$TD/allow.tsv"
    printf 'batch/live\tnot-a-number\t%s\tfixture\n' "$EXP_FUT" > "$TD/bad.tsv"

    row() { # row <want rc> <must-print> <label> <branch> <event> <pr> <allowlist> <shell mutating the tree>
        local want=$1 grepfor=$2 label=$3 branch=$4 event=$5 pr=$6 tsv=$7 mut=$8 rc=0
        n=$((n + 1))
        ( cd "$R" && git checkout -q --detach "$BASE" && bash -c "$mut" && git add -A && git commit -qm "$label" --allow-empty ) >/dev/null 2>&1
        judge "$R" "$BASE" "$(git -C "$R" rev-parse HEAD)" "$branch" "$event" "$pr" "$NOW" "$tsv" > "$TD/out.$n" 2>&1 || rc=$?
        if [ "$rc" = "$want" ] && grep -qF -- "$grepfor" "$TD/out.$n"; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %-2s rc=%s (wanted %s, printing %s)  %s\n' "$n" "$rc" "$want" "$grepfor" "$label"; sed 's/^/        /' "$TD/out.$n"; red=1; fi
    }
    local A="$TD/allow.tsv"
    # must-NOT-match: paths and lines that are not generated files
    row 0 'writes no generated file' "crate code only: PASS"                                    fix/a pull_request '' "$A" 'echo "// x" >> crates/x/src/lib.rs'
    row 0 'writes no generated file' "a roadmap FRAGMENT only: PASS (the path P4 wants)"       fix/a pull_request '' "$A" 'printf -- "- id: PMAT-2\n" > docs/roadmaps/entries/PMAT-2.yaml'
    row 0 'writes no generated file' "docs/roadmaps/README.md (not the aggregate): PASS"       fix/a pull_request '' "$A" 'echo more >> docs/roadmaps/README.md'
    row 0 'writes no generated file' "README prose mentioning contracts, no count: PASS"       fix/a pull_request '' "$A" 'sed -i "s/A prose line about/A prose line on/" README.md'
    row 0 'writes no generated file' "README new prose line with a year, no census noun: PASS" fix/a pull_request '' "$A" 'echo "Shipped in 2026 with 3 new tools." >> README.md'
    # must-match: every way to write the aggregate or a census line
    row 1 "writes $ROADMAP_FILE" "roadmap.yaml edited on a plain branch: RED (the registered mutation)" fix/a pull_request '' "$A" 'echo "- id: PMAT-2" >> docs/roadmaps/roadmap.yaml'
    row 1 "writes $ROADMAP_FILE" "roadmap.yaml deleted: RED"                                  fix/a pull_request '' "$A" 'git rm -q docs/roadmaps/roadmap.yaml'
    row 1 "writes $ROADMAP_FILE" "roadmap.yaml renamed away: RED (--no-renames)"              fix/a pull_request '' "$A" 'git mv docs/roadmaps/roadmap.yaml docs/roadmaps/old.yaml'
    row 1 'README census' "README workspace-crates count bumped: RED"                          fix/a pull_request '' "$A" 'sed -i "s/\*\*80\*\* workspace crates/**81** workspace crates/" README.md'
    row 1 'README census' "README CONTRACT_COUNT marker bumped: RED"                           fix/a pull_request '' "$A" 'sed -i "0,/1830/s//1831/" README.md'
    row 1 'README census' "README CLI-commands count bumped: RED"                              fix/a pull_request '' "$A" 'sed -i "s/111\*\* CLI/112** CLI/" README.md'
    row 1 'README census' "README census line DELETED: RED (a removed claim is a write)"      fix/a pull_request '' "$A" 'sed -i "/CLI commands/d" README.md'
    row 1 'README census' "README new line carrying a count claim: RED"                        fix/a pull_request '' "$A" 'echo "Now 12 new contracts." >> README.md'
    # the one writer's regen PR
    row 0 "one writer's regen PR" "regen/ branch writing only roadmap.yaml + README: PASS"   regen/roadmap pull_request '' "$A" 'echo "- id: PMAT-2" >> docs/roadmaps/roadmap.yaml; sed -i "0,/1830/s//1831/" README.md'
    row 1 'AND other paths' "regen/ branch that also writes code: RED"                         regen/roadmap pull_request '' "$A" 'echo "- id: PMAT-2" >> docs/roadmaps/roadmap.yaml; echo "// x" >> crates/x/src/lib.rs'
    row 1 'strict from day one' "a branch merely CONTAINING regen/ is not a regen branch: RED" fix/regen/x pull_request '' "$A" 'echo "- id: PMAT-2" >> docs/roadmaps/roadmap.yaml'
    # the cop's allowlist: live row, expired row, wrong PR, no PR, new branch
    row 0 'ALLOWED' "allowlisted branch + its PR, before expiry: ALLOWED"                     batch/live pull_request 101 "$A" 'echo "- id: PMAT-2" >> docs/roadmaps/roadmap.yaml'
    row 1 'EXPIRED' "allowlisted branch past its expiry: RED"                                 batch/stale pull_request 102 "$A" 'echo "- id: PMAT-2" >> docs/roadmaps/roadmap.yaml'
    row 1 'a new PR on the same branch is strict' "allowlisted branch, DIFFERENT PR number: RED" batch/live pull_request 999 "$A" 'echo "- id: PMAT-2" >> docs/roadmaps/roadmap.yaml'
    row 1 'cannot be checked' "allowlisted branch, NO PR number: RED"                         batch/live pull_request '' "$A" 'echo "- id: PMAT-2" >> docs/roadmaps/roadmap.yaml'
    row 1 'strict from day one' "a NEW branch not in the allowlist: RED"                      batch/new pull_request 103 "$A" 'echo "- id: PMAT-2" >> docs/roadmaps/roadmap.yaml'
    row 1 'strict from day one' "an allowlisted name's PREFIX is not the branch: RED"          batch/liv pull_request 101 "$A" 'echo "- id: PMAT-2" >> docs/roadmaps/roadmap.yaml'
    row 0 'writes no generated file' "a clean diff never reads the allowlist (malformed table ignored)" fix/a pull_request '' "$TD/bad.tsv" 'echo "// x" >> crates/x/src/lib.rs'
    row 2 'malformed allowlist row' "a malformed allowlist row is ENV (exit 2), never a pass" batch/live pull_request 101 "$TD/bad.tsv" 'echo "- id: PMAT-2" >> docs/roadmaps/roadmap.yaml'
    # shapes
    row 0 'REPORT' "merge_group shape: REPORT, exit 0"                                        '' merge_group '' "$A" 'echo "- id: PMAT-2" >> docs/roadmaps/roadmap.yaml'
    row 0 'REPORT' "push shape: REPORT, exit 0"                                               '' push '' "$A" 'echo "- id: PMAT-2" >> docs/roadmaps/roadmap.yaml'
    n=$((n + 1)); local rc=0
    judge "$R" deadbeefdeadbeefdeadbeefdeadbeefdeadbeef "$BASE" fix/a pull_request '' "$NOW" "$A" > "$TD/out.$n" 2>&1 || rc=$?
    if [ "$rc" = 2 ]; then printf 'ok    row %-2s rc=2  an unresolvable base is ENV, never a pass\n' "$n"; else printf 'FAIL  row %-2s rc=%s (wanted 2)  unresolvable base\n' "$n" "$rc"; red=1; fi

    # COUNT_RE must be the SAME extractor the row guard reads (one census definition)
    n=$((n + 1))
    if [ "$(grep -m1 '^COUNT_RE=' "$ROOT/scripts/check_row_pr_write_set.sh" | cut -d"'" -f2)" = "$COUNT_RE" ]; then
        printf 'ok    row %-2s COUNT_RE is byte-identical to check_row_pr_write_set.sh'"'"'s\n' "$n"
    else printf 'FAIL  row %-2s COUNT_RE drifted from check_row_pr_write_set.sh\n' "$n"; red=1; fi
    # the REAL allowlist honours the cop ruling: parses, <= 8 rows, no row outlives the cap
    n=$((n + 1)); local cap real_rows bad=0 b p e r
    cap=$(to_epoch "$ALLOWLIST_CAP"); real_rows=0
    while IFS=$'\t' read -r b p e r || [ -n "$b" ]; do
        case "$b" in ''|'#'*) continue ;; esac
        real_rows=$((real_rows + 1))
        allow_lookup "$ROOT/$ALLOWLIST_DEFAULT" "$b" >/dev/null 2>&1 || bad=1
        [ "$(to_epoch "$e" || echo 99999999999)" -le "$cap" ] || { printf '        row %s expires %s, after the cap %s\n' "$b" "$e" "$ALLOWLIST_CAP"; bad=1; }
    done < "$ROOT/$ALLOWLIST_DEFAULT"
    if [ "$bad" = 0 ] && [ "$real_rows" -le "$ALLOWLIST_MAX_ROWS" ]; then
        printf 'ok    row %-2s the real allowlist: %s row(s) <= %s, all parse, none past %s\n' "$n" "$real_rows" "$ALLOWLIST_MAX_ROWS" "$ALLOWLIST_CAP"
    else printf 'FAIL  row %-2s the real allowlist breaks the cop ruling (%s rows, cap %s)\n' "$n" "$real_rows" "$ALLOWLIST_MAX_ROWS"; red=1; fi

    printf '%s/%s rows, %s\n' "$n" "$n" "$([ "$red" = 0 ] && echo 'all as expected' || echo 'FAILED')"
    return "$red"
}

BASE=""; HEAD_REF="HEAD"; BRANCH="${GITHUB_HEAD_REF:-}"; EVENT="${GITHUB_EVENT_NAME:-pull_request}"; PR=""; ALLOWLIST="$ROOT/$ALLOWLIST_DEFAULT"
[[ "${GITHUB_REF:-}" =~ ^refs/pull/([0-9]+)/ ]] && PR="${BASH_REMATCH[1]}"
while [ $# -gt 0 ]; do
    case "$1" in
        --self-test) self_test; exit $? ;;
        --base) BASE=$2; shift 2 ;;
        --head) HEAD_REF=$2; shift 2 ;;
        --branch) BRANCH=$2; shift 2 ;;
        --event) EVENT=$2; shift 2 ;;
        --pr) PR=$2; shift 2 ;;
        --allowlist) ALLOWLIST=$2; shift 2 ;;
        -h|--help) usage ;;
        *) usage ;;
    esac
done
[ -n "$BRANCH" ] || BRANCH=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)
case "$EVENT" in merge_group|push) judge "$ROOT" HEAD HEAD "$BRANCH" "$EVENT" "$PR" 0 "$ALLOWLIST"; exit $? ;; esac
if [ -z "$BASE" ]; then
    # pull_request: refs/pull/N/merge — first parent is the base-branch tip, so parent1..HEAD is exactly this
    # PR, whether it targets main or a batch/* branch. The CI checkout is depth 1, where rev-list shows NO
    # parents (shallow graft), so read them off the raw commit object and fetch parent 1 by sha.
    P1=""
    if [ "${GITHUB_EVENT_NAME:-}" = pull_request ]; then
        mapfile -t PARENTS < <(git -C "$ROOT" cat-file -p "$HEAD_REF" 2>/dev/null | sed -n 's/^parent //p')
        [ "${#PARENTS[@]}" = 2 ] && P1=${PARENTS[0]}
    fi
    if [ -n "$P1" ]; then
        git -C "$ROOT" cat-file -e "$P1^{commit}" 2>/dev/null || git -C "$ROOT" fetch -q --no-tags --depth=1 origin "$P1" 2>/dev/null || true
        git -C "$ROOT" cat-file -e "$P1^{commit}" 2>/dev/null || { printf '%s: ENV - the PR merge commit'"'"'s base parent %s cannot be fetched; pass --base (never a pass)\n' "$PROG" "$P1" >&2; exit 2; }
        BASE=$P1; printf '               comparand: %s (first parent of the PR merge commit = base-branch tip)\n' "${BASE:0:9}"
    else
        REPO_ROOT="$ROOT"; . "$ROOT/scripts/lib/resolve_base.sh" || exit 2
        resolve_base "$HEAD_REF" || { printf '%s: no base can be named; pass --base\n' "$PROG" >&2; exit 2; }
        BASE="$BASE_REF"; printf '               comparand: %s (%s)\n' "${BASE:0:9}" "$BASE_HOW"
    fi
fi
judge "$ROOT" "$BASE" "$HEAD_REF" "$BRANCH" "$EVENT" "$PR" "${PR_GENERATED_NOW:-$(python3 -c 'import time;print(int(time.time()))')}" "$ALLOWLIST"
