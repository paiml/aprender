#!/usr/bin/env bash
# check_roadmap_one_writer.sh — a PR does not write docs/roadmaps/roadmap.yaml (T21, operator ruling RQ-8).
#
# WHY. roadmap.yaml is a GENERATED aggregate of docs/roadmaps/entries/<ID>.yaml (#3297). While every PR also
# commits the aggregate, two PRs that each add one fragment both rewrite the same file and the second one
# conflicts (measured 2026-10-04: roadmap.yaml was a conflicted file in 9 of 21 open PRs). Under one writer a PR
# commits only its fragment; CI regenerates the aggregate before any reader (scripts/roadmap_one_writer_ci.sh);
# the nightly writer PR on main and the tag check (check_publish_preflight.sh R9) compare committed == fresh.
#
# RULE. When merge-base..HEAD changes roadmap.yaml, the change is allowed only in the WRITER SHAPE: every changed
# path is a generated file (today: roadmap.yaml) AND the committed roadmap.yaml is byte-equal to
# aggregate(head's base + head's entries/). The writer is recognised by WHAT it changes, never by a branch name
# (a branch name is free to choose). Anything else is named: `FAIL  roadmap.yaml is written by this PR …`.
#
# MODE. A new check lands in SHADOW (operator ruling RQ-8): it prints `SHADOW FAIL …` and exits 0. It blocks
# only after three green nights of the nightly writer, flipped by its own PR that cites those runs (MODE below).
#
# USAGE   bash scripts/check_roadmap_one_writer.sh [<base> [<head>]]   (default: resolve_base HEAD)
#         bash scripts/check_roadmap_one_writer.sh --selftest | --mutants
# EXIT    0 pass (or shadow) · 1 fail in enforce mode · 2 environment (cannot answer; never a pass)
set -uo pipefail
PROG=check_roadmap_one_writer.sh
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$REPO_ROOT/scripts/$PROG"
AGG="$REPO_ROOT/scripts/roadmap_aggregate.sh"
MODE="${ROADMAP_ONE_WRITER_MODE:-shadow}"      # shadow | enforce — the admission PR changes this default
GENERATED="docs/roadmaps/roadmap.yaml"         # the one-writer set; #4526 extends it
RM=docs/roadmaps/roadmap.yaml
MUT="${ROADMAP_ONE_WRITER_MUTANT:-}"           # mutation seam: --mutants only

is_generated() { local g; for g in $GENERATED; do [ "$1" = "$g" ] && return 0; done; return 1; }

# judge <base> <head> -> prints one verdict line; 0 ok, 1 violation, 2 env
judge() {
    local base=$1 head=$2 files f other="" td rc p2 mb="" src
    files=$(git -C "$REPO_ROOT" diff --name-only "$base" "$head" --) || { printf 'ENV   %s: git diff %s..%s failed\n' "$PROG" "$base" "$head"; return 2; }
    if ! grep -q -x -F -e "$RM" <<< "$files"; then
        printf 'ok    %s: this change does not write %s (its fragments are aggregated by CI and by the nightly writer)\n' "$PROG" "$RM"; return 0
    fi
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        is_generated "$f" || other="${other:+$other }$f"
    done <<< "$files"
    [ "$MUT" = anyshape ] && other=""
    if [ -n "$other" ]; then
        printf 'FAIL  %s is written by this PR together with non-generated files (%s); commit only docs/roadmaps/entries/<ID>.yaml — the nightly writer regenerates %s\n' \
            "$RM" "$(printf '%s' "$other" | cut -c1-160)" "$RM"; return 1
    fi
    [ "$MUT" = noequal ] && { printf 'ok    %s: writer shape\n' "$PROG"; return 0; }
    td=$(mktemp -d) || return 2
    if ! { git -C "$REPO_ROOT" archive "$head" docs/roadmaps | tar -x -C "$td"; } ||
       ! mkdir -p "$td/docs/roadmaps/entries" ||
       ! git -C "$REPO_ROOT" show "$base:$RM" > "$td/base.yaml" 2> /dev/null; then
        rm -rf -- "${td:?}"; printf 'ENV   %s: cannot read %s at %s / %s at %s\n' "$PROG" docs/roadmaps "$head" "$RM" "$base"; return 2
    fi
    bash "$AGG" --check --roadmap "$td/$RM" --entries "$td/docs/roadmaps/entries" > "$td/out" 2>&1; rc=$?
    if [ "$rc" = 1 ] && [ "$MUT" != nostale ]; then
        # STALE WRITER: in a merge ref, fragments that landed on main after the writer branched are not in its copy.
        # It is still the writer shape iff aggregate(head's copy, head's fragments) == aggregate(base's copy, head's
        # fragments): every byte it wrote by hand is superseded by a fragment, so it edited no base-only entry.
        bash "$AGG" --print --roadmap "$td/$RM" --entries "$td/docs/roadmaps/entries" > "$td/a1" 2> /dev/null &&
            bash "$AGG" --print --roadmap "$td/base.yaml" --entries "$td/docs/roadmaps/entries" > "$td/a2" 2> /dev/null || rc=2
        # ...and every entry its copy holds is byte-equal to that id's head fragment (fragments it never saw left
        # out), so a hand edit of a FRAGMENT-COVERED entry is still named, not hidden by supersession.
        # A fragment main EDITED after the writer branched is held at its version at the writer's merge-base
        # (quorum r2 E): that version was main's too, so the entry is judged against it, not called a hand edit.
        mkdir -p "$td/held"
        if p2=$(git -C "$REPO_ROOT" rev-parse -q --verify "$head^2" 2> /dev/null) && [ "$MUT" != nomb ]; then
            mb=$(git -C "$REPO_ROOT" merge-base "$head^1" "$p2" 2> /dev/null) || mb=""
        fi
        while IFS= read -r f; do
            f=${f%\"}; f=${f#\"}; f=${f%\'}; f=${f#\'}
            case "$f" in ''|*/*|.*) continue ;; esac
            [ -f "$td/docs/roadmaps/entries/$f.yaml" ] || continue
            src="$td/docs/roadmaps/entries/$f.yaml"
            if [ -n "$mb" ] && git -C "$REPO_ROOT" show "$mb:docs/roadmaps/entries/$f.yaml" > "$td/mbf" 2> /dev/null && ! cmp -s -- "$src" "$td/mbf"; then
                LC_ALL=C awk -v id="$f" '/^- id:/ { v = $0; sub(/^- id:[ \t]*/, "", v); sub(/[ \t]*$/, "", v); gsub(/^["\047]|["\047]$/, "", v); on = (v == id) } on' \
                    "$td/$RM" > "$td/blk"
                cmp -s -- "$td/blk" "$src" || src="$td/mbf"
            fi
            cp -- "$src" "$td/held/$f.yaml"
        done < <(sed -n 's/^- id:[[:space:]]*//p' "$td/$RM" | sed 's/[[:space:]]*$//')
        bash "$AGG" --print --roadmap "$td/$RM" --entries "$td/held" > "$td/a0" 2> /dev/null || rc=2
        # ...and it removed no id the base copy holds: the aggregate never deletes, so a missing id is a hand edit
        # (supersession would otherwise re-add a fragment-covered id and hide the deletion).
        sed -n 's/^- id:[[:space:]]*//p' "$td/base.yaml" | LC_ALL=C sort -u > "$td/ids.base"
        sed -n 's/^- id:[[:space:]]*//p' "$td/$RM" | LC_ALL=C sort -u > "$td/ids.head"
        [ "$MUT" = nodelete ] && cp -- "$td/ids.base" "$td/ids.head"
        if [ "$rc" = 1 ] && { [ "$MUT" = anystale ] || { cmp -s -- "$td/a1" "$td/a2" && { [ "$MUT" = nofaithful ] || cmp -s -- "$td/$RM" "$td/a0"; } &&
             [ -z "$(LC_ALL=C comm -23 "$td/ids.base" "$td/ids.head")" ]; }; }; then rc=3; fi
    fi
    rm -rf -- "${td:?}"
    case "$rc" in
        0) printf 'ok    %s: writer shape — only generated files change and %s == aggregate(head)\n' "$PROG" "$RM"; return 0 ;;
        3) printf 'ok    %s: writer shape, stale — fragments landed after it; every entry it wrote is fragment-covered (aggregate(head copy) == aggregate(base copy)); the tag check (R9) still needs committed == fresh\n' "$PROG"; return 0 ;;
        1) printf 'FAIL  %s is written by this PR and is NOT aggregate(head): a hand edit of the generated file; edit the fragment instead\n' "$RM"; return 1 ;;
        *) printf 'ENV   %s: roadmap_aggregate.sh could not answer (rc %s)\n' "$PROG" "$rc"; return 2 ;;
    esac
}

verdict() {   # verdict <judge-rc> -> exit code under MODE
    case "$1" in
        0) return 0 ;;
        1) if [ "$MODE" = enforce ]; then return 1; fi
           printf 'SHADOW FAIL  %s: the line above WOULD block in enforce mode; shadow until three green writer nights (RQ-8)\n' "$PROG"; return 0 ;;
        *) if [ "$MODE" = enforce ]; then return 2; fi
           printf 'SHADOW ENV  %s: could not answer (not_measured, not a pass); shadow mode does not block on it\n' "$PROG"; return 0 ;;
    esac
}

# ---------- self-test: a scratch repo per row; --mutants proves every row can go RED ----------
selftest() {
    local T pass=0 fail=0 b w wh
    T=$(mktemp -d) || return 2
    ent() { printf -- '- id: %s\n  title: t\n  status: %s\n' "$1" "${2:-open}"; }
    mkrepo() {   # base commit: roadmap with PMAT-1, PMAT-3; one fragment PMAT-1
        rm -rf -- "${T:?}/r"; mkdir -p "$T/r/docs/roadmaps/entries"
        git -C "$T/r" init -q -b main && git -C "$T/r" config user.email t@t && git -C "$T/r" config user.name t
        { printf 'roadmap:\n'; ent PMAT-1; ent PMAT-3; } > "$T/r/$RM"; ent PMAT-1 > "$T/r/docs/roadmaps/entries/PMAT-1.yaml"
        printf 'x\n' > "$T/r/code.rs"; git -C "$T/r" add -A && git -C "$T/r" -c core.hooksPath=/dev/null commit -q -m base
    }
    commit() { git -C "$T/r" add -A && git -C "$T/r" -c core.hooksPath=/dev/null commit -q -m "$1"; }
    row() {      # row <id> <what> <want-judge-rc> <want-grep>
        local got out
        out=$(REPO_ROOT="$T/r" judge "$b" HEAD 2>&1); got=$?
        if [ "$got" = "$3" ] && grep -q -e "$4" <<< "$out"; then pass=$((pass + 1)); printf 'PASS  %s %s\n' "$1" "$2"
        else fail=$((fail + 1)); printf 'FAIL  %s %s (rc %s, want %s): %s\n' "$1" "$2" "$got" "$3" "$(head -c 200 <<< "$out")"; fi
    }
    mkrepo; b=$(git -C "$T/r" rev-parse HEAD)
    ent PMAT-2 > "$T/r/docs/roadmaps/entries/PMAT-2.yaml"; printf 'y\n' >> "$T/r/code.rs"; commit frag
    row W1 "fragment-only PR passes" 0 "does not write"
    mkrepo; b=$(git -C "$T/r" rev-parse HEAD)
    ent PMAT-2 > "$T/r/docs/roadmaps/entries/PMAT-2.yaml"; bash "$AGG" --write --roadmap "$T/r/$RM" --entries "$T/r/docs/roadmaps/entries" > /dev/null; commit legacy
    row W2 "PR committing fragment + regenerated roadmap.yaml is named" 1 "together with non-generated files (docs/roadmaps/entries/PMAT-2.yaml)"
    mkrepo; ent PMAT-2 > "$T/r/docs/roadmaps/entries/PMAT-2.yaml"; commit "frag landed"; b=$(git -C "$T/r" rev-parse HEAD)
    bash "$AGG" --write --roadmap "$T/r/$RM" --entries "$T/r/docs/roadmaps/entries" > /dev/null; commit writer
    row W3 "writer shape (only roadmap.yaml, == aggregate) passes" 0 "writer shape"
    mkrepo; b=$(git -C "$T/r" rev-parse HEAD)
    sed -i 's/status: open/status: done/' "$T/r/$RM"; commit handedit
    row W4 "hand edit of roadmap.yaml alone (not the aggregate) is named" 1 "NOT aggregate(head)"
    mkrepo; b=$(git -C "$T/r" rev-parse HEAD)
    row W5 "no diff passes" 0 "does not write"
    # W6/W7: a writer PR in a merge ref after another fragment landed on main (b = main with PMAT-4 landed)
    mkrepo; ent PMAT-2 > "$T/r/docs/roadmaps/entries/PMAT-2.yaml"; commit "frag landed"; w=$(git -C "$T/r" rev-parse HEAD)
    bash "$AGG" --write --roadmap "$T/r/$RM" --entries "$T/r/docs/roadmaps/entries" > /dev/null; commit writer; wh=$(git -C "$T/r" rev-parse HEAD)
    git -C "$T/r" checkout -q "$w"; ent PMAT-4 > "$T/r/docs/roadmaps/entries/PMAT-4.yaml"; commit "later frag"; b=$(git -C "$T/r" rev-parse HEAD)
    git -C "$T/r" -c core.hooksPath=/dev/null merge -q --no-edit "$wh" > /dev/null 2>&1
    row W6 "stale writer in a merge ref (later fragment on main) passes" 0 "writer shape, stale"
    mkrepo; ent PMAT-2 > "$T/r/docs/roadmaps/entries/PMAT-2.yaml"; commit "frag landed"; w=$(git -C "$T/r" rev-parse HEAD)
    bash "$AGG" --write --roadmap "$T/r/$RM" --entries "$T/r/docs/roadmaps/entries" > /dev/null
    sed -i '/id: PMAT-3/,$ s/status: open/status: done/' "$T/r/$RM"; commit "writer+edit"; wh=$(git -C "$T/r" rev-parse HEAD)
    git -C "$T/r" checkout -q "$w"; ent PMAT-4 > "$T/r/docs/roadmaps/entries/PMAT-4.yaml"; commit "later frag"; b=$(git -C "$T/r" rev-parse HEAD)
    git -C "$T/r" -c core.hooksPath=/dev/null merge -q --no-edit "$wh" > /dev/null 2>&1
    row W7 "stale writer that also hand-edits a base-only entry is named" 1 "NOT aggregate(head)"
    mkrepo; ent PMAT-2 > "$T/r/docs/roadmaps/entries/PMAT-2.yaml"; commit "frag landed"; w=$(git -C "$T/r" rev-parse HEAD)
    bash "$AGG" --write --roadmap "$T/r/$RM" --entries "$T/r/docs/roadmaps/entries" > /dev/null
    sed -i '/id: PMAT-2/,/status/ s/title: t/title: BY HAND/' "$T/r/$RM"; commit "writer+covered"; wh=$(git -C "$T/r" rev-parse HEAD)
    git -C "$T/r" checkout -q "$w"; ent PMAT-4 > "$T/r/docs/roadmaps/entries/PMAT-4.yaml"; commit "later frag"; b=$(git -C "$T/r" rev-parse HEAD)
    git -C "$T/r" -c core.hooksPath=/dev/null merge -q --no-edit "$wh" > /dev/null 2>&1
    row W8 "stale writer that hand-edits a FRAGMENT-COVERED entry is named" 1 "NOT aggregate(head)"
    mkrepo; ent PMAT-2 > "$T/r/docs/roadmaps/entries/PMAT-2.yaml"; commit "frag landed"; w=$(git -C "$T/r" rev-parse HEAD)
    bash "$AGG" --write --roadmap "$T/r/$RM" --entries "$T/r/docs/roadmaps/entries" > /dev/null; commit "fresh on main"; w=$(git -C "$T/r" rev-parse HEAD)
    awk '/^- id: PMAT-2/ { skip = 1; next } /^- id:/ { skip = 0 } !skip' "$T/r/$RM" > "$T/rm.tmp" && cat -- "$T/rm.tmp" > "$T/r/$RM"
    commit "writer drops PMAT-2"; wh=$(git -C "$T/r" rev-parse HEAD)
    git -C "$T/r" checkout -q "$w"; ent PMAT-4 > "$T/r/docs/roadmaps/entries/PMAT-4.yaml"; commit "later frag"; b=$(git -C "$T/r" rev-parse HEAD)
    git -C "$T/r" -c core.hooksPath=/dev/null merge -q --no-edit "$wh" > /dev/null 2>&1
    row W9 "stale writer that DELETES a fragment-covered entry is named" 1 "NOT aggregate(head)"
    mkrepo; ent PMAT-2 > "$T/r/docs/roadmaps/entries/PMAT-2.yaml"; commit "frag landed"; w=$(git -C "$T/r" rev-parse HEAD)
    bash "$AGG" --write --roadmap "$T/r/$RM" --entries "$T/r/docs/roadmaps/entries" > /dev/null; commit writer; wh=$(git -C "$T/r" rev-parse HEAD)
    git -C "$T/r" checkout -q "$w"; ent PMAT-2 done > "$T/r/docs/roadmaps/entries/PMAT-2.yaml"; commit "main edits a held frag"; b=$(git -C "$T/r" rev-parse HEAD)
    git -C "$T/r" -c core.hooksPath=/dev/null merge -q --no-edit "$wh" > /dev/null 2>&1
    row W10 "stale writer after main EDITED a fragment it holds passes (no false RED, quorum r2 E)" 0 "writer shape, stale"
    # verdict under the two modes
    out=$(MODE=shadow; verdict 1); [ "$?" = 0 ] && grep -q -e 'SHADOW FAIL' <<< "$out" && { pass=$((pass + 1)); echo 'PASS  M1 shadow mode exits 0 and prints SHADOW FAIL'; } || { fail=$((fail + 1)); echo 'FAIL  M1 shadow verdict'; }
    (MODE=enforce; verdict 1 > /dev/null); [ "$?" = 1 ] && { pass=$((pass + 1)); echo 'PASS  M2 enforce mode exits 1 on a violation'; } || { fail=$((fail + 1)); echo 'FAIL  M2 enforce verdict'; }
    (MODE=enforce; verdict 2 > /dev/null); [ "$?" = 2 ] && { pass=$((pass + 1)); echo 'PASS  M3 enforce mode: cannot-answer is 2, never 0'; } || { fail=$((fail + 1)); echo 'FAIL  M3 enforce env'; }
    rm -rf -- "${T:?}"
    printf '%s --selftest: %s PASS, %s FAIL\n' "$PROG" "$pass" "$fail"
    [ "$fail" = 0 ] && [ "$pass" -gt 0 ]
}

mutants() {
    local m alive=0
    for m in anyshape noequal nostale anystale nofaithful nodelete nomb; do
        if ROADMAP_ONE_WRITER_MUTANT=$m bash "$SELF" --selftest > /dev/null 2>&1; then alive=$((alive + 1)); printf 'SURVIVED  mutant %s\n' "$m"
        else printf 'killed    mutant %s\n' "$m"; fi
    done
    printf '%s --mutants: %s survived\n' "$PROG" "$alive"
    [ "$alive" = 0 ]
}

case "${1:-}" in
    --selftest|--self-test) selftest; exit $? ;;
    --mutants) mutants; exit $? ;;
    -h|--help) sed -n '2,20p' "$SELF" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac
HEAD_REF="${2:-HEAD}"
if [ -n "${1:-}" ]; then BASE_REF=$1; BASE_HOW=argument
else
    # shellcheck source=scripts/lib/resolve_base.sh
    . "$REPO_ROOT/scripts/lib/resolve_base.sh" || exit 2
    if ! git -C "$REPO_ROOT" rev-parse --verify -q origin/main > /dev/null; then
        printf 'ENV   %s: origin/main is not resolvable here; fetch it (git fetch origin main)\n' "$PROG"; verdict 2; exit $?
    fi
    resolve_base "$HEAD_REF" || { verdict 2; exit $?; }
fi
printf '=== one writer for %s: base=%s (%s) head=%s mode=%s ===\n' "$RM" "$BASE_REF" "$BASE_HOW" "$HEAD_REF" "$MODE"
judge "$BASE_REF" "$HEAD_REF"; verdict $?; exit $?
