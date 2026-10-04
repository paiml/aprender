#!/usr/bin/env bash
# roadmap_one_writer_ci.sh — the guard-tree step that makes every roadmap reader read a FRESH aggregate (T21, RQ-8).
#
# WHY. Under one writer a PR commits docs/roadmaps/entries/<ID>.yaml and NOT docs/roadmaps/roadmap.yaml. A reader
# of the committed file would then judge main's copy, not the PR. So, in the CI clone and before guard_tree.sh:
#   a. REGENERATE roadmap.yaml in the worktree (never committed, never pushed). Every worktree reader in guard_tree
#      then reads aggregate(head). Fragments that do not aggregate BLOCK: the bash aggregator refuses exactly what
#      the EXISTING generator (scripts/lib/roadmap_fragments.py) refuses (parity, fork (i)), and R4 rule 2 blocked
#      on that refusal before RQ-8 moved it here.
#   b. `pmat work validate` on the fresh copy (pinned analyser via scripts/pmat_bin.sh, else `pmat` on PATH as the
#      vendored sov roadmap-valid uses). sov roadmap-valid keeps validating the committed copy. Before RQ-8 a PR had
#      to commit roadmap.yaml, so sov validated the PR's entries; now only this fresh-copy twin does. It is that
#      EXISTING verdict moved, so it BLOCKS in every mode.
#   c. FRESH REFS: commit objects (git commit-tree; HEAD, the index and the branch are untouched) whose roadmap.yaml
#      is aggregate(base) and aggregate(head). check_roadmap_diff_additive.sh (R3) and check_roadmap_fragment_required.sh
#      (R4) run on them, so the entry changes a one-writer PR makes are judged instead of passing on an untouched file.
#      Before RQ-8 the same guards judged those entries on the committed roadmap.yaml, so these verdicts BLOCK in
#      every mode. The same two guards still run on the raw refs in guard_tree.sh.
# Every verdict here is an EXISTING one moved onto the fresh copy, so none is SHADOW and ROADMAP_ONE_WRITER_MODE
# (printed for the log) does not change the exit. Every environment gap here sits behind a BLOCKING
# verdict, so it is NOT MEASURED and exits 2 in every mode — never a pass.
#
# USAGE   bash scripts/roadmap_one_writer_ci.sh [<base> [<head>]]   (default: resolve_base HEAD)
#         --selftest · --mutants · --help
# EXIT    0 ok · 1 blocking failure · 2 environment
set -uo pipefail
PROG=roadmap_one_writer_ci.sh
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$REPO_ROOT/scripts/$PROG"
MODE="${ROADMAP_ONE_WRITER_MODE:-shadow}"
RM=docs/roadmaps/roadmap.yaml
ENT=docs/roadmaps/entries
MUT="${ROADMAP_OW_CI_MUTANT:-}"
BLOCK=0; HENV=0; FRESH=0

blockf() { printf 'FAIL  %s\n' "$1"; BLOCK=1; }   # an EXISTING verdict, moved onto the fresh copy: blocks in every mode
henv() {   # an environment gap behind a BLOCKING verdict: NOT MEASURED, exit 2 in every mode (never a pass)
    printf 'ENV   %s: %s — NOT MEASURED, and a blocking check cannot pass unmeasured\n' "$PROG" "$1"; HENV=1
    [ "$MUT" = softenv ] && HENV=0
}

# fresh_commit <ref> -> prints a commit whose tree is <ref>'s with roadmap.yaml := aggregate(<ref>), parent <ref>
fresh_commit() {
    local ref=$1 td blob tree rc
    td=$(mktemp -d) || return 2
    if ! git -C "$REPO_ROOT" archive "$ref" docs/roadmaps | tar -x -C "$td"; then rm -rf -- "${td:?}"; return 2; fi
    mkdir -p "$td/$ENT"
    bash "$REPO_ROOT/scripts/roadmap_aggregate.sh" --print --roadmap "$td/$RM" --entries "$td/$ENT" > "$td/fresh" 2> "$td/err"; rc=$?
    if [ "$rc" != 0 ]; then sed 's/^/      /' "$td/err" >&2; rm -rf -- "${td:?}"; return "$rc"; fi
    [ "$MUT" = stalefresh ] && cp -- "$td/$RM" "$td/fresh"
    blob=$(git -C "$REPO_ROOT" hash-object -w -- "$td/fresh") &&
    tree=$(GIT_INDEX_FILE="$td/index" git -C "$REPO_ROOT" read-tree "$ref" &&
           GIT_INDEX_FILE="$td/index" git -C "$REPO_ROOT" update-index --cacheinfo "100644,$blob,$RM" &&
           GIT_INDEX_FILE="$td/index" git -C "$REPO_ROOT" write-tree) &&
    GIT_AUTHOR_NAME=fresh GIT_AUTHOR_EMAIL=noreply@invalid GIT_COMMITTER_NAME=fresh GIT_COMMITTER_EMAIL=noreply@invalid \
        git -C "$REPO_ROOT" commit-tree "$tree" -p "$ref" -m "fresh aggregate of $ref (CI only, never pushed)"; rc=$?
    rm -rf -- "${td:?}"
    return "$rc"
}

step_regen() {   # a.
    local out rc
    out=$(bash "$REPO_ROOT/scripts/roadmap_aggregate.sh" --write --roadmap "$REPO_ROOT/$RM" --entries "$REPO_ROOT/$ENT" 2>&1); rc=$?
    printf '%s\n' "$out" | sed -n '1,8p'
    [ "$MUT" = regenpass ] && [ "$rc" = 1 ] && rc=0
    [ "$rc" = 0 ] && { FRESH=1; return 0; }
    [ "$rc" = 1 ] || { henv "roadmap_aggregate.sh could not answer (rc $rc), so every reader would judge a roadmap.yaml that is NOT this PR's"; return 0; }
    blockf "the fragments do not aggregate — the existing generator refuses them too (this blocked as R4 rule 2 before RQ-8)"
}

step_validate() {   # b.
    local out rc
    if [ "$FRESH" != 1 ] && [ "$MUT" != nofreshgate ]; then
        if [ "$BLOCK" = 1 ]; then printf 'skip  pmat work validate: step a already blocks, there is no fresh roadmap.yaml to validate\n'
        else henv "roadmap.yaml was not regenerated, so validating it would judge main's copy, not this PR"; fi
        return 0
    fi
    if [ -n "${ROADMAP_OW_PMAT:-}" ]; then
        [ -x "$ROADMAP_OW_PMAT" ] || { henv "ROADMAP_OW_PMAT=$ROADMAP_OW_PMAT is not executable"; return 0; }
        PMAT=$ROADMAP_OW_PMAT
    else
        # shellcheck source=scripts/pmat_bin.sh
        if ! . "$REPO_ROOT/scripts/pmat_bin.sh" > /dev/null 2>&1; then
            PMAT=$(command -v pmat) || { henv "no pmat (pinned or on PATH) for the fresh-copy \`pmat work validate\` — sov roadmap-valid calls an absent verifier a NO-GO"; return 0; }
        fi
    fi
    out=$(cd "$REPO_ROOT" && "$PMAT" work validate 2>&1); rc=$?
    [ "$MUT" = novalidate ] && rc=0
    if [ "$rc" = 0 ]; then printf 'ok    pmat work validate passes on the FRESH roadmap.yaml\n'
    else printf '%s\n' "$out" | tail -n 6 | sed 's/^/      /'; blockf "pmat work validate refuses the FRESH roadmap.yaml (rc $rc) — sov roadmap-valid judged these entries before RQ-8"; fi
}

step_fresh_refs() {   # c.
    local base=$1 head=$2 fb fh rc g log
    if [ "$(git -C "$REPO_ROOT" rev-parse "$base^{commit}")" = "$(git -C "$REPO_ROOT" rev-parse "$head^{commit}")" ]; then
        printf 'ok    base == head: no entry change to judge on fresh refs\n'; return 0; fi
    fb=$(fresh_commit "$base") || { henv "base $base does not aggregate, so fresh refs cannot be built"; return 0; }
    fh=$(fresh_commit "$head") || {
        if [ "$BLOCK" = 1 ]; then printf 'skip  fresh refs: head does not aggregate (step a blocks on it)\n'
        else henv "head does not aggregate but step a did not block"; fi
        return 0; }
    log=$(mktemp) || { henv "mktemp"; return 0; }
    for g in check_roadmap_diff_additive.sh check_roadmap_fragment_required.sh; do
        bash "$REPO_ROOT/scripts/$g" "$fb" "$fh" > "$log" 2>&1; rc=$?
        [ "$MUT" = norefs ] && rc=0
        if [ "$rc" = 0 ]; then printf 'ok    %s on fresh refs aggregate(base)..aggregate(head)\n' "$g"
        elif [ "$rc" = 2 ]; then sed -n '1,6p' "$log" | sed 's/^/      /'; henv "$g could not run on fresh refs (rc 2)"
        else sed -n '1,12p' "$log" | sed 's/^/      /'; blockf "$g refuses the entry changes this PR makes (fresh refs, rc $rc)"; fi
    done
    rm -f -- "${log:?}"
}

run() {
    local base=${1:-} head=${2:-HEAD}
    if [ -z "$base" ]; then
        # shellcheck source=scripts/lib/resolve_base.sh
        . "$REPO_ROOT/scripts/lib/resolve_base.sh" || { henv "resolve_base.sh"; finish; return; }
        resolve_base HEAD || { henv "no base for HEAD"; finish; return; }
        base=$BASE_REF
    fi
    git -C "$REPO_ROOT" rev-parse -q --verify "$base^{commit}" > /dev/null || { henv "base $base is not a commit"; finish; return; }
    git -C "$REPO_ROOT" rev-parse -q --verify "$head^{commit}" > /dev/null || { henv "head $head is not a commit"; finish; return; }
    printf '%s: base %s, head %s, mode %s\n' "$PROG" "$base" "$head" "$MODE"
    step_regen
    step_validate
    step_fresh_refs "$base" "$head"
    finish
}

finish() {
    [ "$MUT" = noblock ] && BLOCK=0
    if [ "$BLOCK" = 1 ]; then printf 'FAIL  %s: roadmap one-writer CI step refuses this PR\n' "$PROG"; RC=1; return; fi
    if [ "$HENV" = 1 ]; then printf 'ENV   %s: a BLOCKING check above was NOT MEASURED (exit 2 in every mode)\n' "$PROG"; RC=2; return; fi
    printf 'ok    %s (mode %s)\n' "$PROG" "$MODE"; RC=0
}

# ---------------------------------------------------------------- self-test ----
ST_PASS=0; ST_FAIL=0
st_row() {   # st_row <name> <want_rc> <want_regex> <got_rc> <out_file>
    if [ "$4" = "$2" ] && LC_ALL=C grep -a -q -E -e "$3" "$5"; then printf 'PASS  %s\n' "$1"; ST_PASS=$((ST_PASS+1))
    else printf 'FAIL  %s (want rc %s + /%s/, got rc %s)\n' "$1" "$2" "$3" "$4"; sed -n '1,30p' "$5" | sed 's/^/        /'; ST_FAIL=$((ST_FAIL+1)); fi
}

fixture() {   # fixture <dir>: a repo whose origin/main has base A-1,A-2 + fragment A-3, roadmap.yaml = aggregate
    local d=$1 f
    git init -q "$d" && mkdir -p "$d/scripts/lib" "$d/$ENT" || return 1
    for f in roadmap_aggregate.sh check_roadmap_diff_additive.sh check_roadmap_fragment_required.sh "$PROG"; do
        cp -- "$REPO_ROOT/scripts/$f" "$d/scripts/" || return 1; done
    cp -R -- "$REPO_ROOT/scripts/lib/." "$d/scripts/lib/" || return 1
    [ -f "$REPO_ROOT/scripts/__init__.py" ] && cp -- "$REPO_ROOT/scripts/__init__.py" "$d/scripts/"
    printf -- "- id: A-1\n  title: 'first entry'\n  status: planned\n- id: A-2\n  title: 'second entry'\n  status: planned\n" > "$d/$RM"
    printf -- "- id: A-3\n  title: 'third entry'\n  status: planned\n" > "$d/$ENT/A-3.yaml"
    bash "$d/scripts/roadmap_aggregate.sh" --write --roadmap "$d/$RM" --entries "$d/$ENT" > /dev/null || return 1
    printf '#!/usr/bin/env bash\nif grep -q REJECTME docs/roadmaps/roadmap.yaml; then echo "invalid entry"; exit 1; fi\necho valid\n' > "$d/pmat"
    chmod +x "$d/pmat"
    git -C "$d" add -A && st_commit "$d" base && git -C "$d" update-ref refs/remotes/origin/main HEAD
}
st_commit() { git -C "$1" -c core.hooksPath=/dev/null -c user.name=st -c user.email=noreply@invalid commit -q -m "$2"; }

st_case() {   # st_case <name> <want_rc> <want_regex> <mode> <mutator-fn>  — fresh fixture, branch, mutate, commit, run
    local name=$1 want=$2 re=$3 mode=$4 fn=$5 d out rc
    d=$(mktemp -d) || return 1
    out="$d.out"
    if fixture "$d/r" > "$out" 2>&1 && git -C "$d/r" checkout -q -b pr && "$fn" "$d/r" >> "$out" 2>&1 \
       && git -C "$d/r" add -A && st_commit "$d/r" pr >> "$out" 2>&1; then
        (cd "$d/r" && ROADMAP_ONE_WRITER_MODE=$mode ROADMAP_OW_PMAT="${ST_PMAT:-$d/r/pmat}" bash scripts/"$PROG" origin/main HEAD) > "$out" 2>&1; rc=$?
    else rc=99; fi
    st_row "$name" "$want" "$re" "$rc" "$out"
    rm -rf -- "${d:?}" "${out:?}"
}
m_add()      { printf -- "- id: A-4\n  title: 'fourth entry'\n  status: planned\n" > "$1/$ENT/A-4.yaml"; }
m_delfrag()  { rm -f -- "${1:?}/$ENT/A-3.yaml"; }
m_retitle()  { printf -- "- id: A-3\n  title: 'THIRD RETITLED'\n  status: planned\n" > "$1/$ENT/A-3.yaml"; }
m_dupcanon() { m_add "$1"; printf -- "- id: a-04\n  title: 'fourth again'\n  status: planned\n" > "$1/$ENT/a-04.yaml"; }
m_broken()   { printf -- "- id: A-5\n  title: '\377\376'\n" > "$1/$ENT/A-5.yaml"; }   # invalid UTF-8: both generators refuse
m_reject()   { printf -- "- id: A-4\n  title: 'REJECTME'\n  status: planned\n" > "$1/$ENT/A-4.yaml"; }
m_none()     { printf 'x\n' > "$1/unrelated.txt"; }
m_rmroad()   { rm -f -- "${1:?}/$RM"; }

# wired_in_manifest <sections.yml>: rc 0 iff the bare regen step is in guard-tree-steps and guard_tree.sh follows it there
wired_in_manifest() {
    [ "$MUT" = wiring ] && return 0
    LC_ALL=C awk -v me="run: bash scripts/$PROG" '
        /^[^ #]/ { job = "" }
        /^  [A-Za-z0-9_.-]+:/ { job = $1; sub(/:$/, "", job) }
        job == "guard-tree-steps" { line = $0; sub(/^[ \t]*-?[ \t]*/, "", line)
            if (line == me) seen = 1
            else if (seen && line == "run: bash scripts/guard_tree.sh --no-cargo") ok = 1 }
        END { exit ok ? 0 : 1 }' "$1"
}

selftest() {
    local out rc d
    command -v python3 > /dev/null 2>&1 || { printf 'ENV   %s --selftest: needs python3 (R3/R4 are python-backed)\n' "$PROG"; return 2; }
    st_case 'O1 fragment-only add -> both guards ok on fresh'                 0 'ok    check_roadmap_fragment_required.sh on fresh' shadow m_add
    st_case 'O2 unrelated change -> ok, fresh refs equal roadmap'               0 "ok    $PROG"                                       shadow m_none
    st_case 'O3 fragment file deleted -> ok (no entry change)' 0 'ok    check_roadmap_diff_additive.sh on fresh' shadow m_delfrag
    st_case 'F1 fragment retitles an entry -> R3 on fresh refs BLOCKS in shadow' 1 'FAIL  check_roadmap_diff_additive.sh refuses' shadow m_retitle
    st_case 'F1e same, enforce -> rc 1 (R3 reserialised)'                       1 'check_roadmap_diff_additive.sh refuses'       enforce m_retitle
    st_case "P1 A-4 + a-04 fragments -> ok: parity, the existing generator keeps both too" 0 "ok    $PROG" shadow m_dupcanon
    st_case 'B1 invalid UTF-8, both generators refuse -> BLOCK rc 1 in shadow'     1 'existing generator refuses them too'          shadow m_broken
    st_case 'V1 pmat refuses FRESH copy -> BLOCKS in shadow (sov judged it pre-RQ-8)' 1 'FAIL  pmat work validate refuses' shadow m_reject
    st_case 'V1e same, enforce -> rc 1'                                         1 'pmat work validate refuses'                   enforce m_reject
    ST_PMAT=/nonexistent/pmat st_case 'E1 no pmat -> rc 2 even in shadow (blocking check unmeasured)' 2 'NOT MEASURED' shadow m_add
    st_case "B1e invalid UTF-8, enforce -> rc 1 (same verdict, the mode changes nothing)" 1 "existing generator refuses them too" enforce m_broken
    st_case 'R2 roadmap.yaml unreadable -> rc 2, validate never judges an unregenerated copy' 2 'was not regenerated' shadow m_rmroad
    ST_PMAT=/nonexistent/pmat st_case 'E1e no pmat, enforce -> rc 2'            2 'NOT MEASURED'                                 enforce m_add
    # G (quorum r2): a red regen step must not hide the guards after it. It does not, because the step lives in the
    # guard-tree-steps MANIFEST, which scripts/ci_guards.sh runs past a red step (#4415), and guard_tree.sh comes after it.
    d=$(mktemp -d)
    { wired_in_manifest "$REPO_ROOT/ci/sections.yml"; printf 'wired rc %s\n' "$?"; } > "$d/w1" 2>&1
    st_row 'W1 regen step sits in guard-tree-steps before guard_tree.sh (ci_guards runs past red)' 0 'wired rc 0' 0 "$d/w1"
    printf 'jobs:\n  guard-tree:\n    steps:\n      - run: bash scripts/%s\n      - run: bash scripts/guard_tree.sh --no-cargo\n' "$PROG" > "$d/moved.yml"
    { wired_in_manifest "$d/moved.yml"; printf 'wired rc %s\n' "$?"; } > "$d/w2" 2>&1
    st_row 'W2 regen step moved to a fail-fast job -> refused' 0 'wired rc 1' 0 "$d/w2"
    rm -rf -- "${d:?}"
    # The control the whole step exists for: the raw-ref guard PASSES the F1 diff, so only fresh refs can see it.
    d=$(mktemp -d)
    if fixture "$d/r" > /dev/null 2>&1 && git -C "$d/r" checkout -q -b pr && m_retitle "$d/r" && git -C "$d/r" add -A && st_commit "$d/r" pr; then
        (cd "$d/r" && bash scripts/check_roadmap_diff_additive.sh origin/main HEAD) > "$d.out" 2>&1; rc=$?
    else rc=99; : > "$d.out"; fi
    printf 'raw refs exit %s\n' "$rc" >> "$d.out"
    st_row 'K1 control: raw-ref R3 passes the F1 diff (why fresh refs exist)' 0 'raw refs exit 0' "$rc" "$d.out"
    rm -rf -- "${d:?}" "${d:?}.out"
    printf '%s --selftest: %s passed, %s failed\n' "$PROG" "$ST_PASS" "$ST_FAIL"
    [ "$ST_FAIL" = 0 ]
}

mutants() {
    local m killed=0 total=0 out
    out=$(mktemp) || return 2
    for m in stalefresh novalidate norefs regenpass noblock nofreshgate softenv wiring; do
        total=$((total+1))
        if ROADMAP_OW_CI_MUTANT=$m bash "$SELF" --selftest > "$out" 2>&1; then printf 'SURVIVED  %s\n' "$m"
        else killed=$((killed+1)); printf 'killed    %s (%s rows)\n' "$m" "$(LC_ALL=C grep -a -c -e '^FAIL  [A-Z][0-9]' "$out")"; fi
    done
    rm -f -- "${out:?}"
    printf '%s --mutants: %s/%s killed\n' "$PROG" "$killed" "$total"
    # R4's own T21 mutants (MOVED-branch validity, stale-writer merge-base) run from here, not from a workflow line:
    # an argument-only invocation in ci/sections.yml would make guard_tree.sh skip R4's bare run as wired-with-args.
    bash "$REPO_ROOT/scripts/check_roadmap_fragment_required.sh" --mutants || killed=-1
    [ "$killed" = "$total" ]
}

main() {
    case "${1:-}" in
        --selftest) selftest; exit $? ;;
        --mutants) mutants; exit $? ;;
        -h|--help) sed -n '2,22p' "$SELF"; exit 0 ;;
        -*) printf '%s: unknown option %s\n' "$PROG" "$1" >&2; exit 2 ;;
    esac
    RC=0
    run "$@"
    exit "$RC"
}
main "$@"
