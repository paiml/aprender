#!/usr/bin/env bash
# roadmap_writer_nightly.sh — the ONE writer of docs/roadmaps/roadmap.yaml (T21, operator ruling RQ-8).
#
# PRs commit docs/roadmaps/entries/<ID>.yaml only. Once a night this script, run by a host systemd user timer
# (scripts/roadmap-writer/, installed by `make roadmap-writer-install`), regenerates the aggregate on origin/main
# in a DEDICATED clone and opens ONE pull request that changes roadmap.yaml and nothing else. It never pushes to
# main, never force-pushes and never merges by itself: it lands ONLY through the PR and the merge queue (operator
# rule C310.7). It arms auto-merge (`gh pr merge --squash --auto`) so the queue merges it once the same checks as
# any PR pass, where check_roadmap_one_writer.sh and check_roadmap_fragment_required.sh judge the writer shape.
# Never --admin, never a push to main, never --delete-branch: rows N10 (what gh is asked) and N11 (the writer's
# own source) go RED if it gains any of them.
#
#   origin/main already == aggregate      -> "nothing to write", exit 0, no GitHub write
#   a writer PR is open                   -> merge origin/main into ITS branch, regenerate, push (fast-forward)
#   otherwise                             -> branch roadmap-writer/<UTC date> from origin/main, commit, push, open PR
# "A writer PR" is an open PR into main from a roadmap-writer/* branch of THIS repository (not a fork) authored by
# the writer identity (ROADMAP_WRITER_LOGIN, else `gh api user`). A human or fork PR that borrows the prefix is
# ignored, so it can neither wedge the writer nor have the writer push onto it. The writer branch only ever changes
# roadmap.yaml, so the only conflict merging origin/main can raise is on that file: it is merged with origin/main's
# side (-X theirs) and then regenerated, never pushed with force.
#
# GitHub budget (fleet rule GH-1): one `rate_limit` read first; below 1000 remaining it writes nothing (exit 2,
# NOT MEASURED). Then at most: 1 user read (only without ROADMAP_WRITER_LOGIN), 1 PR list, 1 push, 1 PR create,
# 1 auto-merge arm (repeated on an update push: arming is idempotent, and it re-arms a PR someone disarmed).
#
# USAGE   ROADMAP_WRITER_CLONE=<dedicated clone> bash scripts/roadmap_writer_nightly.sh [--dry-run]
#         bash scripts/roadmap_writer_nightly.sh --selftest | --mutants | --help
#         (--selftest-child is the self-test's own fixture run: local origin only, the one mode that reads mutants)
# EXIT    0 wrote or nothing to write · 1 the aggregate refuses the fragments on main · 2 environment
set -uo pipefail
PROG=roadmap_writer_nightly.sh
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF="$SELF_DIR/$PROG"
RM=docs/roadmaps/roadmap.yaml
ENT=docs/roadmaps/entries
PREFIX=roadmap-writer/
GH="${ROADMAP_WRITER_GH:-gh}"
# A normal run honours NO mutant, whatever the environment says (row N12): MUT stays empty. Only the
# --selftest-child mode the self-test spawns reads ROADMAP_WRITER_MUTANT, and only against a clone whose origin is
# a local fixture path, never a real remote (row N13). The bypass mutants rows N6/N10 must catch are injected
# OUTSIDE writer()/arm() so row N11's source scan of the real bodies stays meaningful.
MUT=""; MERGE_X=(); PUSH_X=()
local_origin() { case "$(git -C "${ROADMAP_WRITER_CLONE:-/nonexistent}" remote get-url origin 2> /dev/null)" in /*) return 0 ;; *) return 1 ;; esac; }
mutant_env() {
    MUT="${ROADMAP_WRITER_MUTANT:-}"
    case "$MUT" in admin) MERGE_X=(--admin) ;; delbranch) MERGE_X=(--delete-branch) ;; pushmain) PUSH_X=(HEAD:main) ;; esac
}

say() { printf '%s %s\n' "$(TZ=UTC date +%H:%M:%SZ)" "$*"; }
env2() { say "ENV   $PROG: $* — NOT MEASURED, nothing written"; exit 2; }

writer() {
    local clone=$1 dry=$2 left out rc n branch open ahead login list mx
    [ -d "$clone/.git" ] || env2 "ROADMAP_WRITER_CLONE=$clone is not a git clone"
    git -C "$clone" remote get-url origin > /dev/null 2>&1 || env2 "the clone has no origin"
    if [ "$MUT" != nodirty ] && [ -n "$(git -C "$clone" status --porcelain)" ]; then
        env2 "the clone has local changes; it must be a dedicated clone this script alone writes"
    fi
    if [ "$MUT" != norate ]; then
        left=$("$GH" api rate_limit --jq .resources.core.remaining 2> /dev/null) || env2 "gh api rate_limit failed"
        case "$left" in ''|*[!0-9]*) env2 "rate_limit answered '$left'" ;; esac
        [ "$left" -ge 1000 ] || env2 "GitHub core remaining $left < 1000 (GH-1: only the release path may call)"
    fi
    git -C "$clone" fetch -q origin main || env2 "git fetch origin main failed"
    git -C "$clone" checkout -q --detach origin/main || env2 "checkout origin/main failed"

    out=$(bash "$clone/scripts/roadmap_aggregate.sh" --check --roadmap "$clone/$RM" --entries "$clone/$ENT" 2>&1); rc=$?
    [ "$MUT" = nocheck ] && [ "$rc" = 0 ] && rc=1 && printf -- '- id: MUTANT\n' >> "$clone/$RM"
    case "$rc" in
        0) say "ok    $RM on origin/main == aggregate; nothing to write"; return 0 ;;
        1) ;;
        *) printf '%s\n' "$out" | tail -n 5; env2 "roadmap_aggregate.sh could not answer (rc $rc)" ;;
    esac

    open=""
    if [ "$MUT" != noreuse ]; then
        login=${ROADMAP_WRITER_LOGIN:-}
        [ -n "$login" ] || login=$("$GH" api user --jq .login 2> /dev/null) || env2 "gh api user failed (who is the writer?)"
        case "$login" in ''|*[!A-Za-z0-9_.-]*) env2 "writer login '$login' is not a GitHub login" ;; esac
        # TSV: branch, isCrossRepository, author — filtered HERE so the self-test can feed it hostile rows
        list=$("$GH" pr list --repo "$(git -C "$clone" remote get-url origin)" --state open --base main --limit 1000 \
                 --json headRefName,isCrossRepository,author \
                 --jq '.[] | [.headRefName, (.isCrossRepository|tostring), .author.login] | @tsv' 2> /dev/null) ||
            env2 "gh pr list failed"
        [ "$MUT" = noowner ] && login='*'
        open=$(printf '%s\n' "$list" | awk -F '\t' -v p="$PREFIX" -v me="$login" \
                 'index($1, p) == 1 && $2 == "false" && ($3 == me || me == "*") { print $1 }')
        # At most one writer PR of its own is open (contract invariant, quorum r2 K). Two mean one was opened by
        # hand or by a racing run; which one carries the truth is not this script's call, so it writes nothing.
        if [ "$MUT" != nomulti ] && [ "$(printf '%s\n' "$open" | grep -c -e .)" -gt 1 ]; then
            say "FAIL  $(printf '%s\n' "$open" | grep -c -e .) open writer PRs of $login ($(printf '%s' "$open" | tr '\n' ' ')); close all but one, nothing written"
            return 1
        fi
        open=$(printf '%s\n' "$open" | head -n 1)
    fi
    if [ -n "$open" ]; then
        branch=$open
        git -C "$clone" fetch -q origin "$branch" || env2 "fetch $branch failed"
        git -C "$clone" checkout -q -B "$branch" "origin/$branch" || env2 "checkout $branch failed"
        mx="-Xtheirs"; [ "$MUT" = notheirs ] && mx="-Xours"; [ "$MUT" = noresolve ] && mx="-Xpatience"
        git -C "$clone" -c core.hooksPath=/dev/null merge -q --no-edit "$mx" origin/main > /dev/null 2>&1 ||
            { git -C "$clone" merge --abort > /dev/null 2>&1; env2 "merging origin/main into $branch conflicts outside $RM"; }
    else
        branch="$PREFIX$(TZ=UTC date +%Y-%m-%d-%H%M)"
        git -C "$clone" checkout -q -b "$branch" origin/main || env2 "branch $branch failed"
    fi
    out=$(bash "$clone/scripts/roadmap_aggregate.sh" --write --roadmap "$clone/$RM" --entries "$clone/$ENT" 2>&1); rc=$?
    if [ "$rc" != 0 ]; then printf '%s\n' "$out" | tail -n 8; say "FAIL  the fragments on main do not aggregate; no writer PR"; return 1; fi
    n=$(find "$clone/$ENT" -maxdepth 1 -type f -name '*.yaml' | wc -l | tr -d ' ')
    git -C "$clone" add -- "$RM"
    if git -C "$clone" diff --cached --quiet; then
        say "ok    $branch already carries the fresh aggregate"
        # An open writer PR whose arm failed or was disarmed is re-armed although nothing is pushed (row N14).
        if [ -n "$open" ] && [ "$dry" != 1 ] && [ "$MUT" != norearm ]; then arm "$clone" "$branch"; fi
        return 0
    fi
    git -C "$clone" -c core.hooksPath=/dev/null commit -q \
        -m "chore(roadmap): regenerate $RM from $n fragments (one writer, RQ-8)" \
        -m "Generated by scripts/roadmap_writer_nightly.sh. Only $RM changes; check_roadmap_one_writer.sh judges the writer shape." ||
        env2 "commit failed"
    ahead=$(git -C "$clone" diff --name-only origin/main HEAD | tr '\n' ' ')
    [ "$ahead" = "$RM " ] || { say "FAIL  the writer commit changes more than $RM: $ahead"; return 1; }
    if [ "$dry" = 1 ]; then say "dry-run: would push $branch and $( [ -n "$open" ] && echo 'update the open PR' || echo 'open a PR')"; return 0; fi
    git -C "$clone" push -q origin "HEAD:refs/heads/$branch" "${PUSH_X[@]}" || env2 "push $branch failed (never forced)"
    if [ -z "$open" ]; then
        "$GH" pr create --repo "$(git -C "$clone" remote get-url origin)" --base main --head "$branch" \
            --title "chore(roadmap): regenerate roadmap.yaml (one writer, RQ-8)" \
            --body "Nightly one-writer PR: regenerates docs/roadmaps/roadmap.yaml from docs/roadmaps/entries/ on main. Only the generated file changes. Generated by scripts/roadmap_writer_nightly.sh." \
            > /dev/null || env2 "gh pr create failed (branch $branch is pushed)"
        say "wrote $branch and opened its PR"
    else
        say "wrote $branch (open writer PR updated by a fast-forward push)"
    fi
    arm "$clone" "$branch"
}

# C310.7: the writer PR lands only through the merge queue. Queue-only: no --admin, no --delete-branch (row N11).
arm() {   # arm <clone> <branch>
    [ "$MUT" = noauto ] && return 0
    "$GH" pr merge "$2" --repo "$(git -C "$1" remote get-url origin)" --squash --auto "${MERGE_X[@]}" \
        > /dev/null || env2 "arming auto-merge failed (branch $2 is pushed, its PR is open but not queued)"
    say "armed $2 into the merge queue"
}

# Row N11: the writer's own bodies (writer, arm) may ask for no bypass. Prints the offending lines; rc 1 when any
# is found or the queue arm is missing. Comments are skipped; the mutant injections live outside them on purpose.
shape() {   # shape <script>
    local body bad
    body=$(sed -n -e '/^writer() {/,/^}/p' -e '/^arm() {/,/^}/p' "$1" | grep -v -E '^[[:space:]]*#')
    [ -n "$body" ] || { printf 'no writer()/arm() body in %s\n' "$1"; return 1; }
    bad=$(printf '%s\n' "$body" | grep -n -E -e '--admin|--delete-branch|--force|[[:space:]]push[[:space:]].*(:|[[:space:]])(refs/heads/)?main([[:space:]"]|$)')
    [ -z "$bad" ] || { printf 'bypass in writer(): %s\n' "$bad"; return 1; }
    printf '%s\n' "$body" | grep -q -E 'pr merge .*--auto' || { printf 'writer() never arms the merge queue (--auto)\n'; return 1; }
}

# ---------------------------------------------------------------- self-test ----
ST_PASS=0; ST_FAIL=0
st_row() {   # st_row <name> <ok?0/1> <detail>
    if [ "$2" = 0 ]; then printf 'PASS  %s\n' "$1"; ST_PASS=$((ST_PASS+1))
    else printf 'FAIL  %s — %s\n' "$1" "$3"; ST_FAIL=$((ST_FAIL+1)); fi
}
st_fixture() {   # st_fixture <dir>: bare origin + seed + dedicated clone + fake gh
    local d=$1
    git init -q --bare "$d/origin.git" && git init -q "$d/seed" || return 1
    mkdir -p "$d/seed/scripts" "$d/seed/$ENT" "$d/bin" && cp -- "$SELF_DIR/roadmap_aggregate.sh" "$d/seed/scripts/" || return 1
    printf -- "- id: A-1\n  title: 'one'\n  status: planned\n" > "$d/seed/$RM"
    git -C "$d/seed" add -A && st_git "$d/seed" commit -q -m seed && git -C "$d/seed" push -q "$d/origin.git" HEAD:refs/heads/main || return 1
    git clone -q "$d/origin.git" "$d/clone" 2> /dev/null || return 1
    git -C "$d/clone" config user.name writer && git -C "$d/clone" config user.email noreply@invalid
    # fake gh: logs every call; rate from $d/rate; open writer branch from $d/open
    cat > "$d/bin/gh" <<EOF
#!/usr/bin/env bash
printf '%s\n' "\$*" >> "$d/gh.log"
case "\$1 \$2" in
  "api rate_limit") cat "$d/rate" ;;
  "api user") echo writerbot ;;
  "pr list") cat "$d/open" 2> /dev/null || : ;;
  "pr create"|"pr merge") : ;;
esac
EOF
    chmod +x "$d/bin/gh"; printf '4000\n' > "$d/rate"
}
st_git() { local r=$1; shift; git -C "$r" -c core.hooksPath=/dev/null -c user.name=st -c user.email=noreply@invalid "$@"; }
st_land() {   # st_land <d> <id>: a fragment lands on origin/main WITHOUT regenerating (the one-writer PR shape)
    printf -- "- id: %s\n  title: 'x'\n  status: planned\n" "$2" > "$1/seed/$ENT/$2.yaml"
    git -C "$1/seed" add -A && st_git "$1/seed" commit -q -m "frag $2" && git -C "$1/seed" push -q "$1/origin.git" HEAD:refs/heads/main
}
st_run() { ROADMAP_WRITER_GH="$1/bin/gh" ROADMAP_WRITER_CLONE="$1/clone" bash "$SELF" --selftest-child > "$1/out" 2>&1; }

st_prod() {   # st_prod <dir> <script> <mutant>: a NORMAL run (no --selftest-child) with the mutant in the env
    st_fixture "$1" > /dev/null 2>&1; st_land "$1" A-2 > /dev/null 2>&1
    ROADMAP_WRITER_MUTANT=$3 ROADMAP_WRITER_GH="$1/bin/gh" ROADMAP_WRITER_CLONE="$1/clone" bash "$2" > "$1/out" 2>&1 || return 1
    [ "$(grep -c -E -e '^pr merge .*--squash --auto' "$1/gh.log")" = 1 ] && ! grep -q -E -e '--admin|--delete-branch' "$1/gh.log" &&
        [ "$(git -C "$1/origin.git" rev-parse main)" = "$(git -C "$1/seed" rev-parse HEAD)" ]
}

selftest() {
    local d rc br
    d=$(mktemp -d) || return 2
    # N1 main fresh -> nothing written, no push, no PR
    st_fixture "$d/1" > /dev/null 2>&1; st_run "$d/1"; rc=$?
    [ "$rc" = 0 ] && grep -q 'nothing to write' "$d/1/out" && [ -z "$(git -C "$d/1/origin.git" branch --list "${PREFIX}*")" ]
    st_row 'N1 origin/main already fresh -> nothing to write, no branch' $? "rc $rc: $(tail -n 2 "$d/1/out")"
    # N2 a fragment landed -> one branch, only roadmap.yaml, == aggregate, PR created once
    st_fixture "$d/2" > /dev/null 2>&1; st_land "$d/2" A-2 > /dev/null 2>&1; st_run "$d/2"; rc=$?
    br=$(git -C "$d/2/origin.git" for-each-ref --format='%(refname:short)' "refs/heads/$PREFIX")
    [ "$rc" = 0 ] && [ -n "$br" ] && [ "$(git -C "$d/2/origin.git" diff --name-only main "$br")" = "$RM" ] &&
        git -C "$d/2/origin.git" show "$br:$RM" | grep -q 'id: A-2' && [ "$(grep -c -e '^pr create' "$d/2/gh.log")" = 1 ]
    st_row 'N2 fragment landed -> one writer branch, only roadmap.yaml, carries A-2, one pr create' $? "rc $rc br '$br': $(tail -n 2 "$d/2/out")"
    # N10 C310.7: the new PR is armed into the merge queue, and gh is never asked for a bypass
    [ "$(grep -c -E -e "^pr merge $br .*--squash --auto" "$d/2/gh.log")" = 1 ] && ! grep -q -E -e '--admin|--delete-branch' "$d/2/gh.log"
    st_row 'N10 writer PR armed into the merge queue (pr merge --squash --auto), never --admin / --delete-branch' $? "$(grep -e '^pr merge' "$d/2/gh.log")"
    # N3 a writer PR is open and another fragment lands -> fast-forward its branch, NO second PR
    printf '%s\tfalse\twriterbot\n' "$br" > "$d/2/open"; : > "$d/2/gh.log"; st_land "$d/2" A-3 > /dev/null 2>&1
    git -C "$d/2/origin.git" rev-parse "$br" > "$d/2/before"; st_run "$d/2"; rc=$?
    [ "$rc" = 0 ] && git -C "$d/2/origin.git" merge-base --is-ancestor "$(cat "$d/2/before")" "$br" &&
        git -C "$d/2/origin.git" show "$br:$RM" | grep -q 'id: A-3' && ! grep -q -e '^pr create' "$d/2/gh.log" &&
        [ "$(git -C "$d/2/origin.git" diff --name-only main "$br")" = "$RM" ]
    st_row 'N3 open writer PR -> its branch fast-forwards with A-3, no new PR' $? "rc $rc: $(tail -n 2 "$d/2/out")"
    # N14 the open writer PR already carries the fresh aggregate (a failed or disarmed arm) -> re-armed, nothing pushed
    git -C "$d/2/origin.git" rev-parse "$br" > "$d/2/before"; : > "$d/2/gh.log"; st_run "$d/2"; rc=$?
    [ "$rc" = 0 ] && grep -q 'already carries the fresh aggregate' "$d/2/out" &&
        [ "$(git -C "$d/2/origin.git" rev-parse "$br")" = "$(cat "$d/2/before")" ] && ! grep -q -e '^pr create' "$d/2/gh.log" &&
        [ "$(grep -c -E -e "^pr merge $br .*--squash --auto" "$d/2/gh.log")" = 1 ]
    st_row 'N14 open writer PR already fresh -> re-armed into the queue (pr merge --auto), nothing pushed, no PR' $? "rc $rc: $(tail -n 2 "$d/2/out"); gh: $(tr '\n' '|' < "$d/2/gh.log")"
    # N7 main edited roadmap.yaml under the open writer PR (a conflict on that file) -> merged, regenerated, no force
    printf -- "- id: Z-9\n  title: 'direct'\n  status: planned\n" >> "$d/2/seed/$RM"; st_land "$d/2" A-4 > /dev/null 2>&1
    git -C "$d/2/origin.git" rev-parse "$br" > "$d/2/before"; : > "$d/2/gh.log"; st_run "$d/2"; rc=$?
    [ "$rc" = 0 ] && git -C "$d/2/origin.git" merge-base --is-ancestor "$(cat "$d/2/before")" "$br" &&
        git -C "$d/2/origin.git" show "$br:$RM" | grep -q 'id: A-4' && git -C "$d/2/origin.git" show "$br:$RM" | grep -q 'id: Z-9' &&
        [ "$(git -C "$d/2/origin.git" diff --name-only main "$br")" = "$RM" ] && ! grep -q -e '^pr create' "$d/2/gh.log"
    st_row 'N7 conflict on roadmap.yaml under the open writer PR -> merged with main side, regenerated, fast-forward' $? "rc $rc: $(tail -n 2 "$d/2/out")"
    # N8 a fork PR and a human PR borrow the prefix -> both ignored, the writer opens its own PR
    st_fixture "$d/8" > /dev/null 2>&1; st_land "$d/8" A-2 > /dev/null 2>&1
    printf 'roadmap-writer/fork\ttrue\twriterbot\nroadmap-writer/human\tfalse\tmallory\n' > "$d/8/open"; st_run "$d/8"; rc=$?
    [ "$rc" = 0 ] && [ "$(grep -c -e '^pr create' "$d/8/gh.log")" = 1 ] && ! grep -q -e 'fetch' "$d/8/out" &&
        [ -z "$(git -C "$d/8/origin.git" branch --list "${PREFIX}human" "${PREFIX}fork")" ]
    st_row 'N8 fork / other-author roadmap-writer/* PRs are ignored -> own PR opened' $? "rc $rc: $(tail -n 2 "$d/8/out")"
    # N9 two writer PRs of its own are open -> refused, nothing pushed, no PR (at most one, contract K)
    st_fixture "$d/9" > /dev/null 2>&1; st_land "$d/9" A-2 > /dev/null 2>&1
    printf 'roadmap-writer/a\tfalse\twriterbot\nroadmap-writer/b\tfalse\twriterbot\n' > "$d/9/open"; st_run "$d/9"; rc=$?
    [ "$rc" = 1 ] && grep -q '2 open writer PRs' "$d/9/out" && ! grep -q -e '^pr create' "$d/9/gh.log" &&
        [ -z "$(git -C "$d/9/origin.git" branch --list "${PREFIX}*")" ]
    st_row 'N9 two own writer PRs open -> refused, nothing pushed, no PR' $? "rc $rc: $(tail -n 2 "$d/9/out")"
    # N11 source shape: the real writer() passes; a copy that gains --admin, --delete-branch or a push to main, or
    # loses --auto, is RED. The script's own body is the decision surface the queue-only rule is about.
    local s=$d/shape.sh n11=0 c
    shape "$SELF" > "$d/n11" 2>&1 || n11=1
    for c in 's/--squash --auto/--squash --auto --admin/' 's/--squash --auto/--squash --auto --delete-branch/' \
             's|"HEAD:refs/heads/$branch"|"HEAD:refs/heads/$branch" HEAD:refs/heads/main|' 's/--squash --auto/--squash/'; do
        sed -e "$c" "$SELF" > "$s"
        cmp -s "$SELF" "$s" && { n11=1; printf 'case did not apply: %s\n' "$c" >> "$d/n11"; continue; }
        shape "$s" > /dev/null 2>&1 && { n11=1; printf 'not RED: %s\n' "$c" >> "$d/n11"; }
    done
    [ "$n11" = 0 ]
    st_row 'N11 writer() source: real passes; +--admin / +--delete-branch / +push main / -auto each RED' $? "$(tail -n 3 "$d/n11")"
    # N12 a NORMAL run ignores ROADMAP_WRITER_MUTANT: admin / pushmain / noauto change nothing. Control: a copy that
    # honours the env in a normal run (the round-4 seam) must go RED on admin, or the row could not fail.
    local n12=0 m
    for m in admin pushmain noauto; do st_prod "$d/12$m" "$SELF" "$m" || { n12=1; printf 'real script obeyed %s: %s\n' "$m" "$(grep -e '^pr merge' "$d/12$m/gh.log")" >> "$d/n12"; }; done
    sed -e 's|^writer "\$ROADMAP_WRITER_CLONE"|mutant_env; &|' "$SELF" > "$s"
    if cmp -s "$SELF" "$s"; then n12=1; echo 'seam control did not apply' >> "$d/n12"
    elif st_prod "$d/12seam" "$s" admin; then n12=1; echo 'seam control (env honoured) not RED' >> "$d/n12"
    else printf '      N12 control: env-honouring copy asked gh for: %s\n' "$(grep -e '^pr merge' "$d/12seam/gh.log")"; fi
    [ "$n12" = 0 ]
    st_row 'N12 normal run ignores ROADMAP_WRITER_MUTANT (admin/pushmain/noauto); env-honouring copy RED' $? "$(tail -n 3 "$d/n12" 2> /dev/null)"
    # N13 --selftest-child refuses a clone whose origin is a real remote (no mutant can reach GitHub). Control: the
    # copy without that refusal must go RED.
    local n13=0
    st_fixture "$d/13" > /dev/null 2>&1; git -C "$d/13/clone" remote set-url origin https://github.invalid/x/y.git
    ROADMAP_WRITER_MUTANT=admin st_run "$d/13"; rc=$?
    [ "$rc" = 2 ] && grep -q 'local fixture origin' "$d/13/out" && [ ! -s "$d/13/gh.log" ] || n13=1
    sed -e 's/local_origin || env2/true || env2/' "$SELF" > "$s"
    cmp -s "$SELF" "$s" && n13=1
    : > "$d/13/gh.log"
    ROADMAP_WRITER_MUTANT=admin ROADMAP_WRITER_GH="$d/13/bin/gh" ROADMAP_WRITER_CLONE="$d/13/clone" bash "$s" --selftest-child > "$d/13/out2" 2>&1
    grep -q 'local fixture origin' "$d/13/out2" && n13=1
    [ "$n13" = 0 ]
    st_row 'N13 --selftest-child against a real remote -> ENV rc 2 before any gh call; copy without the refusal RED' $? "rc $rc: $(tail -n 1 "$d/13/out")"
    # N4 dirty clone -> rc 2, nothing pushed
    st_fixture "$d/4" > /dev/null 2>&1; st_land "$d/4" A-2 > /dev/null 2>&1; printf 'x\n' > "$d/4/clone/stray"; st_run "$d/4"; rc=$?
    [ "$rc" = 2 ] && [ -z "$(git -C "$d/4/origin.git" branch --list "${PREFIX}*")" ]
    st_row 'N4 dirty clone -> ENV rc 2, nothing pushed' $? "rc $rc: $(tail -n 2 "$d/4/out")"
    # N5 GitHub budget below 1000 -> rc 2, nothing pushed
    st_fixture "$d/5" > /dev/null 2>&1; st_land "$d/5" A-2 > /dev/null 2>&1; printf '999\n' > "$d/5/rate"; st_run "$d/5"; rc=$?
    [ "$rc" = 2 ] && grep -q 'NOT MEASURED' "$d/5/out" && [ -z "$(git -C "$d/5/origin.git" branch --list "${PREFIX}*")" ]
    st_row 'N5 rate_limit 999 -> ENV rc 2, nothing pushed (GH-1)' $? "rc $rc: $(tail -n 2 "$d/5/out")"
    # N6 never writes main
    [ "$(git -C "$d/2/origin.git" rev-parse main)" = "$(git -C "$d/2/seed" rev-parse HEAD)" ]
    st_row 'N6 main on origin is exactly what the seed pushed (the writer never pushes main)' $? "main moved"
    rm -rf -- "${d:?}"
    printf '%s --selftest: %s PASS, %s FAIL\n' "$PROG" "$ST_PASS" "$ST_FAIL"
    [ "$ST_FAIL" = 0 ]
}

mutants() {
    local m killed=0 total=0
    for m in nodirty norate nocheck noreuse noowner notheirs noresolve nomulti noauto admin delbranch pushmain norearm; do
        total=$((total+1))
        if ROADMAP_WRITER_MUTANT=$m bash "$SELF" --selftest > /dev/null 2>&1; then printf 'SURVIVED  %s\n' "$m"
        else killed=$((killed+1)); printf 'killed    %s\n' "$m"; fi
    done
    printf '%s --mutants: %s/%s killed\n' "$PROG" "$killed" "$total"
    [ "$killed" = "$total" ]
}

case "${1:-}" in
    --selftest) selftest; exit $? ;;
    --mutants) mutants; exit $? ;;
    -h|--help) sed -n '2,20p' "$SELF"; exit 0 ;;
    --dry-run|'') : ;;
    --selftest-child) local_origin || env2 "--selftest-child runs only against a local fixture origin (a path), never a real remote"
        mutant_env ;;
    *) printf '%s: unknown argument %s\n' "$PROG" "$1" >&2; exit 2 ;;
esac
[ -n "${ROADMAP_WRITER_CLONE:-}" ] || env2 "ROADMAP_WRITER_CLONE is unset (a dedicated clone, never a working checkout)"
writer "$ROADMAP_WRITER_CLONE" "$([ "${1:-}" = --dry-run ] && echo 1 || echo 0)"
