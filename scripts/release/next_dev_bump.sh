#!/usr/bin/env bash
# next_dev_bump.sh -- APR-RELEASE-TRAIN T5: the moment vX.Y.Z is tagged, open the X.(Y+1).0-dev bump PR on main, so
# feature PRs never wait on a release. Called by one_command.sh after the tag push, before publish.
#   next_dev_bump.sh <X.Y.Z>          open the PR (idempotent: a PR from the same branch, open or merged, is a SKIP)
#   next_dev_bump.sh --next <X.Y.Z>   print the next dev version only
#   next_dev_bump.sh --self-test      the case table and its mutants; stubs only, no network
# The bump runs in its OWN worktree off origin/main: the release checkout keeps the tagged tree, because publish runs
# from it next. Never --force, never --no-verify. Seams: NDB_{GH,GIT,BUMP,REPO}. exit 0 opened or skipped, 1 failed,
# 2 usage.
set -uo pipefail
SCRATCH=(); trap 'rm -rf -- "${SCRATCH[@]}"' EXIT
GH=${NDB_GH:-gh}; GIT=${NDB_GIT:-git}; BUMP=${NDB_BUMP:-bash scripts/bump-version.sh}; REPO=${NDB_REPO:-paiml/aprender}
[ -z "${NDB_GH_TOKEN:-}" ] || export GH_TOKEN="$NDB_GH_TOKEN"   # release.yml's GH_TOKEN reads paiml/infra only

next_dev() { # X.Y.Z -> X.(Y+1).0-dev
    [[ ${1:-} =~ ^([0-9]+)\.([0-9]+)\.[0-9]+$ ]] || return 2
    printf '%s.%s.0-dev\n' "${BASH_REMATCH[1]}" "$((BASH_REMATCH[2] + 1))"
}

open_pr() {
    local v=$1 n br wt have
    n=$(next_dev "$v") || { echo "STOP bad version '$v'"; return 2; }
    br="chore/bump-$n"
    have=$($GH pr list --repo "$REPO" --head "$br" --state all --json number,state \
        --jq '[.[]|select(.state=="OPEN" or .state=="MERGED")][0].number // empty') || { echo "STOP cannot read PRs for $br"; return 1; }
    [ -z "$have" ] || { echo "SKIP $n: PR #$have from $br already exists"; return 0; }
    wt=$(mktemp -d "${TMPDIR:-/tmp}/next-dev.XXXXXX") || return 1
    SCRATCH+=("$wt")
    $GIT fetch -q origin main || { echo "STOP fetch main failed"; return 1; }
    $GIT worktree add -q -B "$br" "$wt/tree" origin/main || { echo "STOP worktree for $br failed"; return 1; }
    ( cd "$wt/tree" && $BUMP "$n" >/dev/null ) || { echo "STOP bump $n failed"; $GIT worktree remove --force "$wt/tree"; return 1; }
    $GIT -C "$wt/tree" commit -q -am "chore(release): open $n on main after v$v" || { echo "STOP commit failed"; $GIT worktree remove --force "$wt/tree"; return 1; }
    $GIT -C "$wt/tree" push -q -u origin "$br" || { echo "STOP push $br failed"; $GIT worktree remove --force "$wt/tree"; return 1; }
    $GIT worktree remove --force "$wt/tree"
    $GH pr create --repo "$REPO" --base main --head "$br" --title "chore(release): $n on main after v$v" \
        --body "v$v is tagged; main moves to $n so feature PRs never wait on the release (APR-RELEASE-TRAIN T5). Opened by scripts/release/next_dev_bump.sh." \
        || { echo "STOP gh pr create for $br failed"; return 1; }
    echo "OPENED $n from $br"
}

self_test() {
    local d src=${NDB_SRC:-${BASH_SOURCE[0]}} fail=0 pass=0
    d=$(mktemp -d "${TMPDIR:-/tmp}/ndb-st.XXXXXX") || return 2
    SCRATCH+=("$d"); mkdir -p "$d/bin"
    stub() { printf '#!/usr/bin/env bash\necho "%s $*" >> "$CALLS"\n%s\n' "$1" "$2" > "$d/bin/$1"; chmod +x "$d/bin/$1"; }
    # row NAME WANT_RC WANT_OUT_REGEX WANT_CALL_REGEX FORBID_CALL_REGEX -- env knobs: HAVE (existing PR), BUMPFAIL, PUSHFAIL
    row() {
        export CALLS="$d/calls"; : > "$CALLS"
        stub gh 'case "$*" in *"pr list"*) [ -z "${HAVE:-}" ] || echo "$HAVE";; esac; exit 0'
        stub git 'case "$*" in *" push "*) [ -z "${PUSHFAIL:-}" ] || exit 1;; "worktree add "*) mkdir -p "$6";; esac; exit 0'
        stub bump 'echo "bump-cwd $PWD" >> "$CALLS"; [ -z "${BUMPFAIL:-}" ] || exit 1; exit 0'
        local o rc=0
        o=$(PATH="$d/bin:$PATH" NDB_GH=gh NDB_GIT=git NDB_BUMP=bump TMPDIR="$d" bash "$src" 0.73.0 2>&1) || rc=$?
        if [ "$rc" = "$2" ] && grep -Eq -- "$3" <<< "$o" && { [ -z "$4" ] || grep -Eq -- "$4" "$CALLS"; } \
            && { [ -z "$5" ] || ! grep -Eq -- "$5" "$CALLS"; }; then pass=$((pass + 1)); echo "ok   $1"
        else fail=$((fail + 1)); echo "FAIL $1: rc=$rc out=${o:0:160} calls=$(tr '\n' '|' < "$CALLS" | cut -c1-200)"; fi
    }
    row opens-0.74.0-dev        0 '^OPENED 0\.74\.0-dev'  '^gh pr create .*--base main --head chore/bump-0\.74\.0-dev' ''
    row bump-in-own-worktree    0 '^OPENED'               '^bump-cwd .*/next-dev\.[A-Za-z0-9]+/tree$' '^git (checkout|switch|reset)'
    row worktree-off-origin-main 0 '^OPENED'              '^git worktree add -q -B chore/bump-0\.74\.0-dev .* origin/main$' ''
    row bumps-to-the-dev-line   0 '^OPENED'               '^bump 0\.74\.0-dev$' ''
    HAVE=6900 row existing-pr-is-a-skip 0 '^SKIP 0\.74\.0-dev: PR #6900' '' '^(bump|git worktree|gh pr create)'
    BUMPFAIL=1 row bump-red-opens-nothing 1 '^STOP bump'  '' '^(gh pr create|git .*push)'
    PUSHFAIL=1 row push-red-opens-nothing 1 '^STOP push'  '' '^gh pr create'
    local o; o=$(bash "$src" 0.73 2>&1); [ $? = 2 ] && pass=$((pass + 1)) && echo "ok   bad-version-is-usage" \
        || { fail=$((fail + 1)); echo "FAIL bad-version-is-usage: $o"; }
    [ "$(bash "$src" --next 0.73.4)" = 0.74.0-dev ] && pass=$((pass + 1)) && echo "ok   next-of-a-patch-is-next-minor" \
        || { fail=$((fail + 1)); echo "FAIL next-of-a-patch-is-next-minor"; }
    [ "$(bash "$src" --next 1.9.0)" = 1.10.0-dev ] && pass=$((pass + 1)) && echo "ok   next-minor-is-numeric-not-text" \
        || { fail=$((fail + 1)); echo "FAIL next-minor-is-numeric-not-text"; }
    [ -n "${NDB_SRC:-}" ] || {  # mutants: each must turn a row red (skipped inside a mutant run)
        local m n=0 miss=0 e
        while IFS='|' read -r m e; do
            [ -n "$m" ] || continue; n=$((n + 1))
            sed -e "$e" "${BASH_SOURCE[0]}" > "$d/m.sh"
            if cmp -s "${BASH_SOURCE[0]}" "$d/m.sh"; then echo "MUTANT $m did not apply"; miss=$((miss + 1))
            elif NDB_SRC="$d/m.sh" bash "$d/m.sh" --self-test > /dev/null 2>&1; then echo "MUTANT $m SURVIVED"; miss=$((miss + 1)); fi
        done <<'MUT'
skip-ignored|s/^    \[ -z "\$have" \] || .*$/    :/
bumps-in-release-checkout|s/^    ( cd "\$wt\/tree" \&\& \$BUMP/    ( \$BUMP/
bump-rc-ignored|s/ >\/dev\/null ) || { echo "STOP bump/ >\/dev\/null; : ) || { echo "STOP bump/
push-rc-ignored|s/push -q -u origin "\$br" || /push -q -u origin "$br"; : || /
next-minor-not-bumped|s/\$((BASH_REMATCH\[2\] + 1))/${BASH_REMATCH[2]}/
MUT
        if [ "$miss" = 0 ]; then pass=$((pass + 1)); echo "ok   all $n mutants killed"; else fail=$((fail + 1)); fi
    }
    echo "next_dev_bump self-test: $pass passed, $fail failed"
    [ "$fail" = 0 ] && [ "$pass" -gt 0 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    --next) next_dev "${2:-}" || { echo "usage: $0 --next X.Y.Z" >&2; exit 2; } ;;
    -h|--help|"") sed -n '2,9p' "$0"; exit 2 ;;
    *) open_pr "$1" ;;
esac
