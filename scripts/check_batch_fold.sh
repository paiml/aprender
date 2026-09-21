#!/usr/bin/env bash
# check_batch_fold.sh -- the case table for scripts/batch_fold.sh, the integration-batch
# folder (#3673, APR-RELEASE-001 §13). Batching is the DEFAULT merge path now (operator
# 2026-09-21), so the folder decides what lands with a batch: it may resolve ONLY the
# generated set, and must leave out -- never silently resolve -- every real conflict.
#
# The table runs the real batch_fold.sh against throwaway git repos. Regeneration uses
# stub tools (a stub pv via BATCH_FOLD_PV, a stub Makefile and readme_sync.sh in the
# fixture), so no build and no network are needed.
#
# WHY THIS FILE AND NOT `batch_fold.sh --self-test`: a script that carries its own
# --self-test case claims to be a guard, and check_guards_are_wired.sh requires a
# workflow to run it -- measured: the handed-off draft did exactly that and turned
# the meta-guard RED ("NEW: batch_fold.sh"). A check_*.sh is dispatched by
# guard_tree.sh --no-cargo, so the table runs on every PR with no workflow edit.
#
#   check_batch_fold.sh              the case table
#   check_batch_fold.sh --self-test  the mutants: four wrong folders, each must turn a row RED
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
SUBJECT="${BATCH_FOLD_SUBJECT:-$ROOT/scripts/batch_fold.sh}"

rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }

# mkfixture DIR -- a repo with a generated set, README count blocks, code, and branches:
#   plain (no generated path)  gen1/gen2 (roadmap-only conflict)  code1/code2 (real conflict)
#   rc1/rc2 (README count-only conflict)  rp1/rp2 (README prose conflict)
#   cen (clean census edit)  plus stub tools for --regen
mkfixture() {
    local d=$1
    git init -q -b main "$d" || return 2
    (
        cd "$d" || exit 2
        export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t
        export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath GIT_CONFIG_VALUE_0=/dev/null
        mkdir -p docs/roadmaps contracts src scripts
        printf 'a\n' > docs/roadmaps/roadmap.yaml
        printf '{"n_files": 5}\n' > contracts/census.json
        printf 't0\n' > contracts/contracts.nt
        printf 's0\n' > contracts/shapes.ttl
        printf 'x\n' > src/lib.rs
        printf '# T\nprose line one\ncount: <!-- CONTRACT_COUNT_START -->5<!-- CONTRACT_COUNT_END -->\nprose line two\n' > README.md
        printf 'roadmap-aggregate:\n\t@printf %s > docs/roadmaps/roadmap.yaml\nroadmap-aggregate-check:\n\t@test "$$(cat docs/roadmaps/roadmap.yaml)" = aggregated\n' "'aggregated\\n'" > Makefile
        cat > scripts/readme_sync.sh <<'SH'
#!/usr/bin/env bash
case "$1" in
  --write) sed -E -i 's/(<!-- CONTRACT_COUNT_START -->)[0-9]+(<!-- CONTRACT_COUNT_END -->)/\17\2/g' README.md ;;
  --check) grep -q 'CONTRACT_COUNT_START -->7<!--' README.md ;;
esac
SH
        git add -A && git commit -q -m base
        # one edit function per branch (no eval: the edit is code, not a string)
        e_plain() { printf 'o\n' > src/other.rs; }
        e_gen1()  { printf 'a\nb1\n' > docs/roadmaps/roadmap.yaml; }
        e_gen2()  { printf 'a\nb2\n' > docs/roadmaps/roadmap.yaml; }
        e_code1() { printf 'y\n' > src/lib.rs; }
        e_code2() { printf 'z\n' > src/lib.rs; }
        e_rc1()   { sed -i 's/-->5<!--/-->10<!--/' README.md; }
        e_rc2()   { sed -i 's/-->5<!--/-->11<!--/; s/prose line two/prose line two, and a clean edit from rc2/' README.md; }
        e_rp1()   { sed -i 's/prose line one/prose line ONE (rp1)/' README.md; }
        e_rp2()   { sed -i 's/prose line one/prose line uno (rp2)/' README.md; }
        e_cen()   { printf '{"n_files": 6}\n' > contracts/census.json; }
        for b in plain gen1 gen2 code1 code2 rc1 rc2 rp1 rp2 cen; do
            git checkout -q -b "$b" main && "e_$b" && git add -A && git commit -q -m "$b" && git checkout -q main || exit 2
        done
    ) || return 2
    # the stub pv: census prints 7, extract writes the graph, --check compares
    cat > "$d.pv-good" <<'SH'
#!/usr/bin/env bash
case "$1" in
  census) printf '{"n_files": 7}\n' ;;
  extract) if [ "${3:-}" = --check ]; then [ "$(cat contracts/contracts.nt)" = t7 ] && [ "$(cat contracts/shapes.ttl)" = s7 ]
           else printf 't7\n' > contracts/contracts.nt; printf 's7\n' > contracts/shapes.ttl; fi ;;
esac
SH
    sed 's/\[ "$(cat contracts\/contracts.nt)" = t7 \]/false/' "$d.pv-good" > "$d.pv-badcheck"
    chmod 755 "$d.pv-good" "$d.pv-badcheck"
}

# fold DIR BRANCH-FOR-BATCH ARGS... -> runs the subject on a fresh batch branch; sets OUT, RC
fold() {
    local d=$1 subj=$2; shift 2
    git -C "$d" checkout -q -B batch main 2>/dev/null
    OUT=$(cd "$d" && bash "$subj" "$@" 2>&1); RC=$?
}

# the table, against SUBJ; prints rows, returns 0 iff all green
table() {
    local subj=$1 d n=0 bad=0 v
    d=$(mktemp -d "${TMPDIR:-/tmp}/check-batch-fold.XXXXXX") || return 2
    row() { # row LABEL TEST-EXIT
        n=$((n + 1))
        if [ "$2" = 0 ]; then printf 'ok    row %-2s %s\n' "$n" "$1"; else printf 'FAIL  row %-2s %s\n        output: %s\n' "$n" "$1" "$(printf '%s' "$OUT" | tr '\n' '|' | cut -c1-300)"; bad=1; fi
    }
    has() { local l; while IFS= read -r l; do [[ $l =~ $1 ]] && return 0; done <<< "$OUT"; return 1; }
    mkfixture "$d/r" || { rmtree "$d"; return 2; }
    export BATCH_FOLD_PV="$d/r.pv-good"

    fold "$d/r" "$subj" plain gen1 gen2 code1 code2
    has '^folded plain generated=\[\]$';                                  row "a branch touching no generated path folds, generated=[]" $?
    has '^folded gen2 generated=\[docs/roadmaps/roadmap\.yaml\]$';       row "a conflict ONLY in roadmap.yaml folds, marked for regeneration" $?
    has '^folded code1 ';                                                row "an independent code change folds" $?
    has '^SKIP code2: conflict in src/lib\.rs$';                         row "a conflict in real code is SKIPPED, never resolved by picking a side" $?
    [ "$RC" = 1 ];                                                       row "any SKIP makes the exit 1 (rc=$RC)" $?
    v=$(git -C "$d/r" status --porcelain | wc -l | tr -d ' '); [ "$v" = 0 ]; row "the aborted merge leaves no half-merged state (dirty=$v)" $?
    v=$(cat "$d/r/src/lib.rs");                          [ "$v" = y ];   row "the skipped branch's code did not leak in (lib=$v)" $?

    fold "$d/r" "$subj" rc1 rc2
    has '^folded rc2 generated=\[README\.md\]$';                         row "a README conflict ONLY in a CONTRACT_COUNT block folds, marked for regeneration" $?
    grep -q 'and a clean edit from rc2' "$d/r/README.md";                row "  ...and the other side's clean README prose edit is kept" $?
    grep -qE 'CONTRACT_COUNT_START -->[0-9]+<!-- CONTRACT_COUNT_END' "$d/r/README.md"; row "  ...and the count block stays numeric (well-formed for readme_sync)" $?
    [ "$RC" = 3 ] && has '^REGEN REQUIRED';                              row "a folded, unregenerated generated set is never silent: REGEN REQUIRED, exit 3 (rc=$RC)" $?

    fold "$d/r" "$subj" rp1 rp2
    has '^SKIP rp2: conflict in README\.md$';                            row "a README conflict OUTSIDE a count block is a real conflict: SKIP" $?
    grep -q 'prose line ONE (rp1)' "$d/r/README.md";                     row "  ...and the README keeps the side already in the batch" $?

    fold "$d/r" "$subj" cen
    has '^folded cen generated=\[contracts/census\.json\]$' && [ "$RC" = 3 ]; row "a CLEAN fold that touches the census still marks the set stale (exit 3, rc=$RC)" $?

    git -C "$d/r" checkout -q -B batch main; printf 'dirt\n' >> "$d/r/src/lib.rs"
    OUT=$(cd "$d/r" && bash "$subj" plain 2>&1); RC=$?
    git -C "$d/r" checkout -q -- src/lib.rs
    [ "$RC" = 2 ] && has 'uncommitted changes';                          row "a dirty batch worktree is refused (exit 2, rc=$RC)" $?

    fold "$d/r" "$subj" no-such-branch
    [ "$RC" = 2 ] && has '^ERROR no-such-branch';                        row "a branch that does not exist is ERROR, exit 2 (rc=$RC)" $?

    fold "$d/r" "$subj" --regen gen1 gen2 rc1 rc2
    [ "$RC" = 0 ] && has '^regenerated the generated set and committed it'; row "--regen regenerates once, asserts the fixed points, commits (rc=$RC)" $?
    v=$(git -C "$d/r" log -1 --format=%s);  [ "${v#batch: regenerate}" != "$v" ]; row "  ...the last commit is the regeneration ($v)" $?
    v=$(cat "$d/r/docs/roadmaps/roadmap.yaml"); [ "$v" = aggregated ] && grep -q -- '-->7<!--' "$d/r/README.md"; row "  ...roadmap aggregated and README counts rewritten" $?
    v=$(git -C "$d/r" status --porcelain | wc -l | tr -d ' '); [ "$v" = 0 ]; row "  ...and the tree is clean after it (dirty=$v)" $?

    BATCH_FOLD_PV="$d/r.pv-badcheck" fold "$d/r" "$subj" --regen gen1
    [ "$RC" = 2 ] && has 'fixed point FAILED -- pv extract contracts --check'; row "a failed fixed-point check is exit 2 and names the check (rc=$RC)" $?

    unset BATCH_FOLD_PV
    rmtree "$d"
    printf '%s rows, %s\n' "$n" "$([ "$bad" = 0 ] && echo PASS || echo FAIL)"
    return "$bad"
}

case "${1:-}" in -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac
[ -f "$SUBJECT" ] || { printf 'ENV   %s is missing -- cannot judge, not a pass\n' "$SUBJECT" >&2; exit 2; }

if [ "${1:-}" = "--self-test" ]; then
    echo "=== check_batch_fold.sh --self-test: each wrong folder must turn the table RED ==="
    m=$(mktemp -d "${TMPDIR:-/tmp}/check-batch-fold-mut.XXXXXX") || exit 2
    bad=0
    mutant() { # mutant LABEL SED-EXPR
        sed -E "$2" "$SUBJECT" > "$m/mut.sh"
        if cmp -s "$SUBJECT" "$m/mut.sh"; then printf 'FAIL  mutant did not apply: %s\n' "$1"; bad=1; return; fi
        if table "$m/mut.sh" > "$m/out" 2>&1; then printf 'FAIL  mutant SURVIVED: %s\n' "$1"; bad=1
        else printf 'ok    mutant RED: %-58s (%s)\n' "$1" "$(grep -m1 '^FAIL  row' "$m/out" | cut -c7-90)"; fi
    }
    mutant "README treated as wholly generated (the draft's rule)"  's#^GENERATED_WHOLE="#GENERATED_WHOLE="README.md #'
    mutant "every conflict taken, as if generated"                   's#^        if is_generated_whole "\$f"; then$#        if true; then#'
    mutant "a clean fold never marks the set stale"                  's#^        \[ -z "\$gen" \] \|\| STALE=1$#        :#'
    mutant "a failed fixed-point check ignored"                      's#^(    "\$PV" extract contracts --check >/dev/null 2>&1 +)\|\| die .*$#\1|| true#'
    rmtree "$m"
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED: 4 mutants, all RED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi

echo "=== batch_fold.sh folds only what it may, and says so (check_batch_fold.sh) ==="
table "$SUBJECT"; rc=$?
[ "$rc" = 0 ] && echo PASS || echo "FAIL (rc=$rc)" >&2
exit "$rc"
