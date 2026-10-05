#!/usr/bin/env bash
# check_roadmap_fragment_required.sh — a change to docs/roadmaps/roadmap.yaml
# must arrive WITH its docs/roadmaps/entries/<ID>.yaml fragment
# (PMAT-3296, #3296; five-whys #3294).
#
# WHY THIS EXISTS
# ---------------
# #3297 made roadmap.yaml a GENERATED aggregate of docs/roadmaps/entries/ so
# that two PRs touching the roadmap touch two different FILES: unique filename
# by construction => pull requests are pairwise disjoint on the roadmap, and the
# Amdahl serial fraction on the merge path stops being 1. Nothing enforced it.
# Measured on a clean worktree cut from origin/main, 2026-09-16:
#
#     $ pmat work add "..." --github-issue 3294
#     M docs/roadmaps/roadmap.yaml        <- the MONOLITH, 16 lines, no fragment
#     $ python3 scripts/lib/roadmap_fragments.py aggregate --check
#     ok  roadmap.yaml == aggregate(1 fragment(s)), idempotent   # rc=0
#
# The aggregate check PASSES that edit, and cannot do otherwise: `aggregate`
# takes roadmap.yaml as its own base, so an entry written straight into the base
# is a fixed point. `--check` answers "is the aggregate consistent?", never "did
# this change come through the fragment path?". Without the second question the
# contention #3297 removed walks straight back in, one `pmat work add` at a
# time. (That same edit was also appended to the TAIL, which
# check_roadmap_sorted.sh refuses — two guards, two different defects, and only
# this one sees the missing fragment.)
#
# THE RULE, over the diff base..head:
#
#   1. FRAGMENT REQUIRED. For every top-level entry whose BYTES in roadmap.yaml
#      differ between base and head (added, removed or changed), the same diff
#      must change docs/roadmaps/entries/<ID>.yaml. A re-serialisation that
#      changes no field still counts as changed — check_roadmap_diff_additive.sh
#      already refuses that shape, so this gate must not be the one place it
#      reads as "nothing happened".
#   2. THE AGGREGATE MUST BE REGENERATED. Whenever either side of the pair
#      changes, head's roadmap.yaml must equal aggregate(head's entries/).
#      That placement rule is NOT restated here: it is
#      `roadmap_fragments.py aggregate --check`, the same function `make
#      roadmap-aggregate` writes with, so the guard and the generator cannot
#      drift. A fragment landed without `make roadmap-aggregate` is drift and is
#      named as such.
#
# WHAT IS STILL ALLOWED: the aggregate changing AS AN AGGREGATE — every entry
# that moved has its fragment in the same diff and the file equals
# aggregate(fragments). Also a preamble-only change (the header above the first
# `- id:` is not entry content and has no fragment), and any diff that touches
# neither roadmap.yaml nor entries/.
#
# AN ID THAT CANNOT BE A FILENAME HAS NO FRAGMENT PATH AT ALL. 52 of the real
# entries are prose ids (one contains `origin/main`, a path separator). There is
# no entries/<ID>.yaml they could be accompanied by, so a change to one is
# refused with that stated — the scope boundary RMFR-OB-005 already declares,
# now enforced rather than described.
#
# `pmat work add` DOES NOT WRITE FRAGMENTS (measured above). That is a real
# conflict with this gate, so the remedy is named in the failure message rather
# than left for the next person to rediscover:
#
#     pmat work add "<title>" --github-issue <N>          # writes the monolith
#     python3 scripts/lib/roadmap_fragments.py adopt <ID> # -> entries/<ID>.yaml
#                                                         #    + regenerates
#     git add docs/roadmaps/entries/<ID>.yaml docs/roadmaps/roadmap.yaml
#
# PUSH SHAPE IS NOT JUDGED (§8 decision, 2026-09-16). resolve_base names
# HEAD^1 as the base when HEAD is on the origin/main first-parent line, because
# for most guards a commit judged against ITSELF is the vacuous pass to avoid.
# For a DIFFERENTIAL guard that is the wrong base: HEAD^1..HEAD is the PREVIOUS
# merge's diff, not this change. A guard added today would then grade commits
# that landed before it existed -- measured here: 2 of the last 8 first-parent
# commits of origin/main (#3348, 334646674) are fragment-less, so a bare run on
# main would have red-lined guard_tree from the moment this merged. It is not
# re-litigating history. In push shape the guard reports a named SKIP, prints
# the shape and the base it WOULD have used and what it therefore did not
# check, and exits 0. The change is graded on its pull_request / merge_group
# run, where the base is a real merge-base. PR shape is unchanged.
#
#   bash scripts/check_roadmap_fragment_required.sh [<base-ref> [<head-ref>]]
#   bash scripts/check_roadmap_fragment_required.sh '' <head-ref>   # resolve the
#       base for an arbitrary head (how the retro and the case table reach the
#       push-shape arm at a commit that is not the checkout's HEAD)
#   bash scripts/check_roadmap_fragment_required.sh --self-test
#
# Exit: 0 clean · 1 a violation · 2 this box cannot judge (never a silent pass).
#
# Refs: PMAT-3296, #3296, #3294, #3297 (the fragment mechanism),
#       contracts/apr-roadmap-fragments-v1.yaml, scripts/check_roadmap_sorted.sh.

set -uo pipefail

PROG=${0##*/}
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY_LIB="$REPO_ROOT/scripts/lib/roadmap_fragments.py"
AGG="$REPO_ROOT/scripts/roadmap_aggregate.sh"   # T21 checks: bash, which refuses exactly what PY_LIB refuses (fork (i))
ROADMAP_FILE="docs/roadmaps/roadmap.yaml"
ENTRIES_DIR="docs/roadmaps/entries"

# shellcheck source=scripts/lib/resolve_base.sh
. "$REPO_ROOT/scripts/lib/resolve_base.sh" || exit 1

usage() {
    printf 'usage: %s [<base-ref> [<head-ref>]]  or  %s --self-test\n' "$PROG" "$PROG" >&2
    printf '  a roadmap.yaml change must arrive with its docs/roadmaps/entries/<ID>.yaml fragment\n' >&2
    printf "  an empty <base-ref> ('') resolves the base for <head-ref>; push shape SKIPs, never grades\n" >&2
    exit 2
}

# filename_safe ID -- the same shape roadmap_fragments.py's census uses: an id
# that cannot be a filename cannot be a fragment, so it has no write path.
filename_safe() {
    case "$1" in
        ''|*/*|*' '*) return 1 ;;
    esac
    [ "${#1}" -le 111 ] && [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
}

remedy() {
    printf '\nREMEDY (PMAT-3296, #3296). roadmap.yaml is GENERATED from %s/:\n' "$ENTRIES_DIR"
    printf '  * a NEW ticket — `pmat work add` writes the monolith and knows nothing about\n'
    printf '    fragments (measured 2026-09-16), so take the two-step:\n'
    printf '        pmat work add "<title>" --github-issue <N>\n'
    printf '        python3 scripts/lib/roadmap_fragments.py adopt <ID>\n'
    printf '  * an EXISTING entry — adopt it once, then edit %s/<ID>.yaml\n' "$ENTRIES_DIR"
    printf '    (a fragment SUPERSEDES the base entry of the same id).\n'
    printf '  * then regenerate and stage BOTH sides:\n'
    printf '        make roadmap-aggregate\n'
    printf '        git add %s/<ID>.yaml %s\n' "$ENTRIES_DIR" "$ROADMAP_FILE"
    printf '  * an id that is not filename-safe (prose, or carrying a path separator) has NO\n'
    printf '    fragment path: it stays in the base and is immutable (RMFR-OB-005).\n'
}

# ---------------------------------------------------------------------------
# judge <repo> <base-ref> <head-ref> -> 0 clean · 1 violation · 2 cannot judge
# ---------------------------------------------------------------------------
judge() {
    local repo=$1 base=$2 head=$3
    local td roadmap_changed=0 frag_paths names name verdict eid line
    local violations=0 checked=0 out rc writer=0 p2 mb src

    git -C "$repo" rev-parse --verify -q "$base^{commit}" >/dev/null || {
        printf 'ENV   %s: base ref %s is not a commit here — refusing to judge (never a pass)\n' "$PROG" "$base" >&2
        return 2
    }
    git -C "$repo" rev-parse --verify -q "$head^{commit}" >/dev/null || {
        printf 'ENV   %s: head ref %s is not a commit here — refusing to judge (never a pass)\n' "$PROG" "$head" >&2
        return 2
    }

    td=$(mktemp -d "${TMPDIR:-/tmp}/rmfragreq.XXXXXX") || return 2

    if [ -n "$(git -C "$repo" diff --name-only "$base" "$head" -- "$ROADMAP_FILE")" ]; then
        roadmap_changed=1
    fi
    frag_paths=$(git -C "$repo" diff --name-only "$base" "$head" -- "$ENTRIES_DIR")

    if [ "$roadmap_changed" = 0 ] && [ -z "$frag_paths" ]; then
        rm -rf -- "${td:?}"
        printf 'PASS  %s: this diff touches neither %s nor %s/ — nothing to judge\n' \
            "$PROG" "$ROADMAP_FILE" "$ENTRIES_DIR"
        return 0
    fi

    # The ids whose FRAGMENT this diff changes (added, edited or deleted).
    : >"$td/frag_ids"
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        case "$name" in *.yaml) ;; *) continue ;; esac
        name=${name##*/}
        printf '%s\n' "${name%.yaml}" >>"$td/frag_ids"
    done < <(printf '%s\n' "$frag_paths")

    # Materialise both roadmaps and head's fragment set. A side that does not
    # exist at that ref is EMPTY, not an error: before the first fragment lands,
    # the aggregate is just the base.
    git -C "$repo" show "$base:$ROADMAP_FILE" >"$td/base.yaml" 2>/dev/null || : >"$td/base.yaml"
    git -C "$repo" show "$head:$ROADMAP_FILE" >"$td/head.yaml" 2>/dev/null || : >"$td/head.yaml"
    mkdir -p "$td/entries"
    names=$(git -C "$repo" ls-tree -r --name-only "$head" -- "$ENTRIES_DIR" 2>/dev/null) || names=""
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        case "$name" in *.yaml) ;; *) continue ;; esac
        if ! git -C "$repo" show "$head:$name" >"$td/entries/${name##*/}"; then
            rm -rf -- "${td:?}"
            printf 'ENV   %s: cannot read %s at %s\n' "$PROG" "$name" "$head" >&2
            return 2
        fi
    done < <(printf '%s\n' "$names")

    printf '=== %s: base=%s head=%s ===\n' "$PROG" "$base" "$head"

    # WRITER SHAPE (RQ-8): the diff changes roadmap.yaml and nothing else. Its entries come from fragments that
    # landed in EARLIER diffs, so rule 1 licenses them by the fragment existing at head, and rule 2 proves the
    # bytes are that fragment's (committed == aggregate, or the stale-writer test below).
    if [ "$roadmap_changed" = 1 ] && [ -z "$frag_paths" ] &&
       [ "$(git -C "$repo" diff --name-only "$base" "$head" | grep -c -v -x -F -e "$ROADMAP_FILE")" = 0 ]; then
        writer=1
    fi

    # --- RULE 1: every changed entry carries its fragment in the same diff ----
    if ! out=$(python3 "$PY_LIB" changed --base "$td/base.yaml" --head "$td/head.yaml" 2>&1); then
        rm -rf -- "${td:?}"
        printf 'ENV   %s: the entry reader could not compare the two roadmaps:\n%s\n' "$PROG" "$out" >&2
        return 2
    fi
    while IFS=$'\t' read -r verdict eid; do
        [ -n "$verdict" ] || continue
        if [ "$verdict" = PREAMBLE ]; then
            printf 'ok    the header changed (not entry content, no fragment exists for it)\n'
            continue
        fi
        checked=$((checked + 1))
        if grep -Fxq -- "$eid" "$td/frag_ids"; then
            printf 'ok    %-8s %s — %s/%s.yaml changes in the same diff\n' "$verdict" "$eid" "$ENTRIES_DIR" "$eid"
            continue
        fi
        if [ "$writer" = 1 ] && [ "$verdict" != REMOVED ] && filename_safe "$eid" && [ -f "$td/entries/$eid.yaml" ]; then
            printf 'ok    %-8s %s — WRITER: %s/%s.yaml exists at head (landed earlier); rule 2 proves the bytes\n' "$verdict" "$eid" "$ENTRIES_DIR" "$eid"
            continue
        fi
        violations=$((violations + 1))
        if filename_safe "$eid"; then
            printf 'FAIL  %-8s %s in %s with NO change to %s/%s.yaml — a roadmap edit without its fragment\n' \
                "$verdict" "$eid" "$ROADMAP_FILE" "$ENTRIES_DIR" "$eid"
        else
            printf 'FAIL  %-8s %s in %s — that id cannot be a filename, so it has NO fragment path: it lives in the base and is immutable (RMFR-OB-005)\n' \
                "$verdict" "$eid" "$ROADMAP_FILE"
        fi
    done < <(printf '%s\n' "$out")

    # --- RULE 2: the aggregate at head is REGENERATED, not drifting ----------
    # ONE WRITER (T21, operator ruling RQ-8): a diff that changes fragments but not
    # roadmap.yaml is the one-writer PR shape. Its committed == fresh comparison
    # MOVES to the writer PR (which writes roadmap.yaml, so it lands in the
    # branch below); a tag-time check is a separate follow-up. On the PR, CI
    # regenerates before any reader (scripts/roadmap_one_writer_ci.sh). A diff that
    # DOES write roadmap.yaml keeps this rule exactly as before.
    # Only the committed == fresh COMPARISON moves: the fragments must still aggregate (quorum r2 C), or a local
    # guard_tree run would pass fragments that every later writer PR refuses.
    if [ "$roadmap_changed" = 0 ]; then
        out=$(bash "$AGG" --print --roadmap "$td/head.yaml" --entries "$td/entries" 2>&1 >/dev/null)
        rc=$?
        [ "${FRAGREQ_MUTANT:-}" = movednovalid ] && rc=0
        rm -rf -- "${td:?}"
        # rc 1 is a refusal; anything else (rc 2, an aggregator that cannot run) is ENV, never a refusal and never a
        # pass (quorum r2 H).
        if [ "$rc" != 0 ] && [ "$rc" != 1 ]; then
            printf 'ENV   %s: the existing generator could not run (rc %s), so the fragments are NOT MEASURED:\n%s\n' "$PROG" "$rc" "$out" >&2
            return 2
        fi
        if [ "$rc" = 0 ]; then
            printf 'MOVED rule 2 (committed %s == aggregate): the fragments aggregate; this diff does not write %s, so the comparison runs on the writer PR (RQ-8)\n' \
                "$ROADMAP_FILE" "$ROADMAP_FILE"
        else
            violations=$((violations + 1))
            printf 'FAIL  FRAGMENTS: at head, %s/ does not aggregate (the existing generator refuses it, rc %s)\n' "$ENTRIES_DIR" "$rc"
            printf '%s\n' "$out" | sed -n '1,6p' | sed 's/^/      /'
        fi
        rc=0
    else
    out=$(python3 "$PY_LIB" aggregate --check --roadmap "$td/head.yaml" --entries "$td/entries" 2>&1)
    rc=$?
    # STALE WRITER (RQ-8): in a merge ref, fragments that landed on main after the writer branched are not in its
    # copy, so committed != aggregate(head). Its copy is still fragment-faithful iff it equals aggregate(its copy,
    # the head fragments of the ids it CONTAINS): fragments it never saw are left out, every entry it holds must be
    # its fragment's bytes (a hand edit of a fragment-covered entry differs; a base-only edit is rule 1's).
    # committed == fresh is then proved by the next writer PR.
    # A fragment main EDITED after the writer branched (planned -> completed) is held at its older version, which
    # was main's at the writer's merge-base (quorum r2 E). So an entry may be its fragment's bytes at head OR at that
    # merge-base: both were on main. The version whose bytes the copy holds is chosen; a hand edit matches neither.
    if [ "$rc" = 1 ] && [ "$writer" = 1 ]; then
        mb=""
        mkdir -p "$td/held"
        if p2=$(git -C "$repo" rev-parse -q --verify "$head^2" 2>/dev/null) && [ "${FRAGREQ_MUTANT:-}" != nomb ]; then
            mb=$(git -C "$repo" merge-base "$head^1" "$p2" 2>/dev/null) || mb=""
        fi
        while IFS= read -r eid; do
            eid=${eid%\"}; eid=${eid#\"}; eid=${eid%\'}; eid=${eid#\'}
            filename_safe "$eid" && [ -f "$td/entries/$eid.yaml" ] || continue
            src="$td/entries/$eid.yaml"
            if [ -n "$mb" ] && git -C "$repo" show "$mb:$ENTRIES_DIR/$eid.yaml" >"$td/mbf" 2>/dev/null && ! cmp -s -- "$src" "$td/mbf"; then
                LC_ALL=C awk -v id="$eid" '/^- id:/ { v = $0; sub(/^- id:[ \t]*/, "", v); sub(/[ \t]*$/, "", v); gsub(/^["\047]|["\047]$/, "", v); on = (v == id) } on' \
                    "$td/head.yaml" >"$td/blk"
                cmp -s -- "$td/blk" "$src" || src="$td/mbf"
            fi
            cp -- "$src" "$td/held/$eid.yaml"
        done < <(sed -n 's/^- id:[[:space:]]*//p' "$td/head.yaml" | sed 's/[[:space:]]*$//')
        if bash "$AGG" --print --roadmap "$td/head.yaml" --entries "$td/held" >"$td/a0" 2>/dev/null &&
           cmp -s -- "$td/head.yaml" "$td/a0"; then
            rc=0
            printf 'ok    STALE WRITER: only %s changes; every entry it holds is its fragment at head or at its merge-base (fragments landed after it belong to the next writer PR)\n' "$ROADMAP_FILE"
        fi
    fi
    rm -rf -- "${td:?}"
    fi
    if [ "$roadmap_changed" = 0 ]; then
        :
    elif [ "$rc" = 0 ]; then
        printf 'ok    %s == aggregate(%s/) at head\n' "$ROADMAP_FILE" "$ENTRIES_DIR"
    else
        violations=$((violations + 1))
        printf 'FAIL  DRIFT: at head, %s is NOT aggregate(%s/) — a fragment changed and the aggregate was not regenerated (run `make roadmap-aggregate`)\n' \
            "$ROADMAP_FILE" "$ENTRIES_DIR"
        while IFS= read -r line; do
            [ -n "$line" ] && printf '      %s\n' "$line"
        done < <(printf '%s\n' "$out")
    fi

    if [ "$violations" != 0 ]; then
        printf '%s: %s violation(s) over %s changed entry/entries\n' "$PROG" "$violations" "$checked"
        remedy
        return 1
    fi
    printf 'PASS  %s: %s changed entry/entries, each with its fragment; the aggregate is regenerated\n' "$PROG" "$checked"
    return 0
}

# ---------------------------------------------------------------------------
# --self-test: hermetic fixture repos under mktemp -d. No file in this repo is
# read as data and none is written.
# ---------------------------------------------------------------------------
BASE_ROADMAP="roadmap_version: '1.0'
github_enabled: true
roadmap:
- id: PMAT-100
  title: first
  status: planned
- id: PMAT-300
  title: third
  status: planned
- id: Push completed work to origin/main (5 commits)
  title: a real prose id, with a path separator in it
  status: planned
"

# mkrepo DIR -- a fixture repo whose HEAD is BASE_ROADMAP plus one fragment for
# PMAT-100, i.e. the post-#3297 shape.
mkrepo() {
    local d=$1
    mkdir -p "$d/$ENTRIES_DIR" || return 2
    git -C "$d" init -q --template="$TEMPLATE" || return 2
    git -C "$d" config user.email test@example.com
    git -C "$d" config user.name test
    printf '%s' "$BASE_ROADMAP" >"$d/$ROADMAP_FILE"
    printf -- '- id: PMAT-100\n  title: first\n  status: planned\n' >"$d/$ENTRIES_DIR/PMAT-100.yaml"
    printf 'placeholder\n' >"$d/docs/roadmaps/README.md"
    git -C "$d" add -A
    git -C "$d" commit -qm base
}

self_test() {
    local td n=0 red=0 TEMPLATE
    td=$(mktemp -d "${TMPDIR:-/tmp}/rmfragreq-selftest.XXXXXX") || return 2
    cleanup() {
        local victim=${td:-}
        case "$victim" in
            *rmfragreq-selftest.*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;;
            *) return 0 ;;
        esac
    }
    trap cleanup RETURN
    TEMPLATE="$td/.empty-template"
    mkdir -p "$TEMPLATE"
    export GIT_TERMINAL_PROMPT=0

    # row NAME WANT_RC MUST_MATCH BUILDER -- BUILDER edits a fresh fixture repo
    # and commits; the guard then judges HEAD~1..HEAD inside it.
    row() {
        local label=$1 want=$2 pat=$3 builder=$4 d out rc
        n=$((n + 1))
        d="$td/r$n"
        if ! mkrepo "$d" >/dev/null 2>&1; then
            printf 'FAIL  row %-2s %s: fixture repo could not be built\n' "$n" "$label"
            red=$((red + 1))
            return
        fi
        "$builder" "$d" || { printf 'FAIL  row %-2s %s: builder failed\n' "$n" "$label"; red=$((red + 1)); return; }
        out=$(judge "$d" HEAD~1 HEAD 2>&1)
        rc=$?
        if [ "$rc" != "$want" ] || ! grep -qF -- "$pat" <<<"$out"; then
            printf 'FAIL  row %-2s rc=%s (wanted %s, must contain: %s)  %s\n' "$n" "$rc" "$want" "$pat" "$label"
            printf '%s\n' "$out" | sed 's/^/        /'
            red=$((red + 1))
            return
        fi
        printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
    }

    commit_all() { git -C "$1" add -A && git -C "$1" commit -qm head; }

    # ---- builders -------------------------------------------------------
    # THE MEASURED SHAPE: `pmat work add` appends an entry to the monolith and
    # writes no fragment.
    b_monolith_only() {
        printf -- '- id: PMAT-200\n  title: added by pmat work add\n  status: planned\n' >>"$1/$ROADMAP_FILE"
        commit_all "$1"
    }
    # The correct shape: the fragment, and the aggregate regenerated from it.
    b_fragment_and_aggregate() {
        printf -- '- id: PMAT-200\n  title: added the fragment way\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"
        python3 "$PY_LIB" aggregate --write --roadmap "$1/$ROADMAP_FILE" >/dev/null 2>&1 || return 1
        commit_all "$1"
    }
    # A fragment with no `make roadmap-aggregate`.
    b_fragment_no_regen() {
        printf -- '- id: PMAT-200\n  title: fragment only\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"
        commit_all "$1"
    }
    b_unrelated() {
        printf 'a docs-only change\n' >>"$1/docs/roadmaps/README.md"
        commit_all "$1"
    }
    # Re-serialisation: the bytes of every entry change, no field does.
    b_reserialised() {
        python3 - "$1/$ROADMAP_FILE" <<'PY' || return 1
import sys
p = sys.argv[1]
t = open(p, encoding="utf-8").read()
t = t.replace("  title: third\n", "  title: 'third'\n")
open(p, "w", encoding="utf-8").write(t)
PY
        commit_all "$1"
    }
    # Lifecycle edit straight into the base, no fragment.
    b_lifecycle_in_base() {
        python3 - "$1/$ROADMAP_FILE" <<'PY' || return 1
import sys
p = sys.argv[1]
t = open(p, encoding="utf-8").read()
t = t.replace("- id: PMAT-300\n  title: third\n  status: planned\n",
              "- id: PMAT-300\n  title: third\n  status: completed\n")
open(p, "w", encoding="utf-8").write(t)
PY
        commit_all "$1"
    }
    b_deleted_entry() {
        python3 - "$1/$ROADMAP_FILE" <<'PY' || return 1
import sys
p = sys.argv[1]
t = open(p, encoding="utf-8").read()
t = t.replace("- id: PMAT-300\n  title: third\n  status: planned\n", "")
open(p, "w", encoding="utf-8").write(t)
PY
        commit_all "$1"
    }
    # A prose id that cannot be a filename.
    b_prose_id_edit() {
        python3 - "$1/$ROADMAP_FILE" <<'PY' || return 1
import sys
p = sys.argv[1]
t = open(p, encoding="utf-8").read()
open(p, "w", encoding="utf-8").write(t.replace(
    "  title: a real prose id, with a path separator in it\n",
    "  title: an edit to a prose-id entry\n"))
PY
        commit_all "$1"
    }
    # Preamble-only: the header above the first entry.
    b_preamble_only() {
        python3 - "$1/$ROADMAP_FILE" <<'PY' || return 1
import sys
p = sys.argv[1]
t = open(p, encoding="utf-8").read()
open(p, "w", encoding="utf-8").write(t.replace("github_enabled: true\n", "github_enabled: false\n"))
PY
        commit_all "$1"
    }
    # A fragment for a DIFFERENT ticket does not license this entry.
    b_wrong_fragment() {
        printf -- '- id: PMAT-100\n  title: first\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-100.yaml"
        printf -- '- id: PMAT-200\n  title: unlicensed\n  status: planned\n' >>"$1/$ROADMAP_FILE"
        commit_all "$1"
    }
    # Writes roadmap.yaml (regenerated), then edits the fragment again: the
    # committed aggregate is stale, and because this diff WRITES roadmap.yaml,
    # rule 2 still judges it on the PR (T21 moves rule 2 only for one-writer diffs).
    b_written_but_stale() {
        printf -- '- id: PMAT-200\n  title: first draft\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"
        bash "$AGG" --write --roadmap "$1/$ROADMAP_FILE" --entries "$1/$ENTRIES_DIR" >/dev/null 2>&1 || return 1
        printf -- '- id: PMAT-200\n  title: second draft\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"
        commit_all "$1"
    }
    # Supersession: edit an ADOPTED entry through its fragment, regenerate.
    b_supersede() {
        printf -- '- id: PMAT-100\n  title: first\n  status: completed\n' >"$1/$ENTRIES_DIR/PMAT-100.yaml"
        python3 "$PY_LIB" aggregate --write --roadmap "$1/$ROADMAP_FILE" >/dev/null 2>&1 || return 1
        commit_all "$1"
    }

    # STALE WRITER in a merge ref: HEAD~1 = main after a later fragment landed, HEAD = merge of the writer PR.
    b_stale_writer_base() {   # b_stale_writer_base <dir> <edit-fn> [<main-change-fn>]
        local w wh
        printf -- '- id: PMAT-200\n  title: two\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"
        commit_all "$1" && w=$(git -C "$1" rev-parse HEAD) || return 1
        bash "$AGG" --write --roadmap "$1/$ROADMAP_FILE" --entries "$1/$ENTRIES_DIR" >/dev/null 2>&1 || return 1
        "$2" "$1" || return 1
        commit_all "$1" && wh=$(git -C "$1" rev-parse HEAD) || return 1
        git -C "$1" checkout -q "$w" || return 1
        "${3:-m_later_fragment}" "$1" || return 1
        commit_all "$1" && git -C "$1" merge -q --no-edit "$wh" >/dev/null 2>&1
    }
    m_later_fragment() { printf -- '- id: PMAT-201\n  title: later\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-201.yaml"; }
    m_edit_held() { printf -- '- id: PMAT-200\n  title: two\n  status: completed\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"; }
    e_none() { :; }
    e_drop() { sed -i '/^- id: PMAT-300$/,+2d' "$1/$ROADMAP_FILE"; }
    b_stale_writer_main_edited() { b_stale_writer_base "$1" e_none m_edit_held; }
    b_stale_writer_drops() { b_stale_writer_base "$1" e_drop; }
    b_fragment_broken() {
        printf -- '- id: PMAT-200\n  title: \377\376\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"
        commit_all "$1"
    }
    e_base_only() { sed -i 's/  title: third/  title: THIRD BY HAND/' "$1/$ROADMAP_FILE"; }
    b_stale_writer() { b_stale_writer_base "$1" e_none; }
    b_stale_writer_handedit() { b_stale_writer_base "$1" e_base_only; }
    e_covered() { sed -i 's/  title: two/  title: TWO BY HAND/' "$1/$ROADMAP_FILE"; }
    b_stale_writer_covered_edit() { b_stale_writer_base "$1" e_covered; }
    b_writer_fresh() {
        printf -- '- id: PMAT-200\n  title: two\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"
        commit_all "$1" || return 1
        bash "$AGG" --write --roadmap "$1/$ROADMAP_FILE" --entries "$1/$ENTRIES_DIR" >/dev/null 2>&1 || return 1
        commit_all "$1"
    }

    # ---- the case table --------------------------------------------------
    row 'MEASURED SHAPE: entry added to roadmap.yaml only (what pmat work add writes) -> REFUSE' \
        1 'ADDED    PMAT-200 in docs/roadmaps/roadmap.yaml with NO change' b_monolith_only
    row 'fragment + regenerated aggregate, in sync -> PASS' \
        0 'ok    ADDED    PMAT-200 — docs/roadmaps/entries/PMAT-200.yaml changes in the same diff' b_fragment_and_aggregate
    row 'ONE WRITER (RQ-8): fragment only, roadmap.yaml untouched -> PASS, rule 2 MOVED to writer+tag' \
        0 'MOVED rule 2' b_fragment_no_regen
    row 'roadmap.yaml written but NOT aggregate(head) -> REFUSE, naming the drift (rule 2 still runs here)' \
        1 'FAIL  DRIFT' b_written_but_stale
    row 'STALE WRITER: writer PR in a merge ref after a later fragment landed -> PASS (RQ-8)' \
        0 'ok    STALE WRITER' b_stale_writer
    row 'stale writer that ALSO hand-edits a base-only entry -> REFUSE, named by rule 1' \
        1 'FAIL  CHANGED  PMAT-300' b_stale_writer_handedit
    row 'stale writer that hand-edits a FRAGMENT-COVERED entry -> REFUSE, DRIFT named' \
        1 'FAIL  DRIFT' b_stale_writer_covered_edit
    row 'C: fragment-only diff whose fragments do not aggregate -> REFUSE (rule 2 moved, validity did not)' \
        1 'FAIL  FRAGMENTS' b_fragment_broken
    row 'E: stale writer after main EDITED a fragment it holds (planned -> completed) -> PASS, no false DRIFT' \
        0 'ok    STALE WRITER' b_stale_writer_main_edited
    row 'K: stale writer that DROPS an entry -> REFUSE, named by rule 1 (ids(base) subset of ids(head))' \
        1 'REMOVED  PMAT-300' b_stale_writer_drops
    row 'writer PR (fragment landed earlier), fresh -> PASS via WRITER licence' \
        0 'WRITER: docs/roadmaps/entries/PMAT-200.yaml exists' b_writer_fresh
    row 'a docs-only diff touching neither side -> PASS (no false positive)' \
        0 'nothing to judge' b_unrelated
    row 'the aggregate RE-SERIALISED with no content change -> REFUSE' \
        1 'CHANGED  PMAT-300' b_reserialised
    row 'a lifecycle edit written straight into the base -> REFUSE (adopt it first)' \
        1 'CHANGED  PMAT-300' b_lifecycle_in_base
    row 'an entry DELETED from the monolith with no fragment change -> REFUSE' \
        1 'REMOVED  PMAT-300' b_deleted_entry
    row 'an id that cannot be a filename -> REFUSE, naming the absent write path' \
        1 'NO fragment path' b_prose_id_edit
    row 'a preamble-only change -> PASS (the header is not entry content)' \
        0 'ok    the header changed (not entry content, no fragment exists for it)' b_preamble_only
    row "another ticket's fragment does not license this entry -> REFUSE" \
        1 'ADDED    PMAT-200 in docs/roadmaps/roadmap.yaml with NO change' b_wrong_fragment
    row 'supersession through the fragment, regenerated -> PASS' \
        0 'ok    CHANGED  PMAT-100 — docs/roadmaps/entries/PMAT-100.yaml changes in the same diff' b_supersede

    # H: a generator that cannot load is ENV (rc 2): never read as a fragment refusal, never a pass.
    n=$((n + 1))
    local out rc
    mkrepo "$td/r$n" >/dev/null 2>&1 && b_fragment_no_regen "$td/r$n" >/dev/null 2>&1
    out=$(AGG="$td/no-such-aggregator.sh" judge "$td/r$n" HEAD~1 HEAD 2>&1)
    rc=$?
    if [ "$rc" = 2 ] && grep -qF 'ENV ' <<<"$out" && ! grep -qF 'FAIL  FRAGMENTS' <<<"$out"; then
        printf 'ok    row %-2s rc=2  H: the generator cannot load -> ENV, never a refusal or a pass\n' "$n"
    else
        printf 'FAIL  row %-2s rc=%s (wanted 2)  H: the generator cannot load -> ENV\n' "$n" "$rc"
        printf '%s\n' "$out" | sed 's/^/        /'
        red=$((red + 1))
    fi

    # An unresolvable ref is ENV (rc 2), never a pass.
    n=$((n + 1))
    mkrepo "$td/r$n" >/dev/null 2>&1
    out=$(judge "$td/r$n" deadbeefdeadbeefdeadbeefdeadbeefdeadbeef HEAD 2>&1)
    rc=$?
    if [ "$rc" = 2 ] && grep -qF 'refusing to judge' <<<"$out"; then
        printf 'ok    row %-2s rc=2  an unresolvable base ref is ENV, never a pass\n' "$n"
    else
        printf 'FAIL  row %-2s rc=%s (wanted 2)  an unresolvable base ref is ENV, never a pass\n' "$n" "$rc"
        printf '%s\n' "$out" | sed 's/^/        /'
        red=$((red + 1))
    fi

    # ---- the push-shape pair (§8). These exercise the real DISPATCH --
    # resolve_base plus the skip -- not judge(), so the fixture carries its own
    # copy of the guard and the libraries it sources: $REPO_ROOT is then the
    # fixture, and origin/main is whatever the row points it at. Same commit,
    # same fragment-less content, two verdicts: the guard refuses the change
    # when it can SEE the change, and declines to judge when it cannot.
    dispatch_row() {
        local label=$1 want=$2 pat=$3 target=$4 d out rc f
        n=$((n + 1))
        d="$td/r$n"
        if ! mkrepo "$d" >/dev/null 2>&1 || ! b_monolith_only "$d" >/dev/null 2>&1; then
            printf 'FAIL  row %-2s %s: fixture could not be built\n' "$n" "$label"
            red=$((red + 1))
            return
        fi
        mkdir -p "$d/scripts/lib"
        if ! cp "$REPO_ROOT/scripts/$PROG" "$d/scripts/$PROG"; then
            printf 'FAIL  row %-2s %s: the guard could not be copied into the fixture\n' "$n" "$label"
            red=$((red + 1))
            return
        fi
        for f in resolve_base.sh roadmap_fragments.py roadmap_diff.py roadmap_merge.py; do
            if ! cp "$REPO_ROOT/scripts/lib/$f" "$d/scripts/lib/$f"; then
                printf 'FAIL  row %-2s %s: scripts/lib/%s could not be copied into the fixture\n' "$n" "$label" "$f"
                red=$((red + 1))
                return
            fi
        done
        git -C "$d" update-ref refs/remotes/origin/main "$(git -C "$d" rev-parse "$target")"
        out=$(bash "$d/scripts/$PROG" 2>&1)
        rc=$?
        if [ "$rc" != "$want" ] || ! grep -qF -- "$pat" <<<"$out"; then
            printf 'FAIL  row %-2s rc=%s (wanted %s, must contain: %s)  %s\n' "$n" "$rc" "$want" "$pat" "$label"
            printf '%s\n' "$out" | sed 's/^/        /'
            red=$((red + 1))
            return
        fi
        printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
    }

    dispatch_row 'PUSH shape (origin/main IS this commit), last merge fragment-less -> SKIP, exit 0 (the row that would have red-lined main)' \
        0 'SKIP  push-shape' HEAD
    dispatch_row 'PR shape (origin/main is its parent), SAME fragment-less content -> REFUSE, exit 1' \
        1 'FAIL  ADDED    PMAT-200' HEAD~1

    printf '%s/%s rows, %s failed\n' "$((n - red))" "$n" "$red"
    [ "$red" = 0 ]
}

# ---------------------------------------------------------------------------
# --mutants: each mutant removes one T21 mechanism; the self-test must go RED under every one.
mutants() {
    local m killed=0 total=0
    for m in movednovalid nomb; do
        total=$((total + 1))
        if FRAGREQ_MUTANT=$m bash "${BASH_SOURCE[0]}" --self-test >/dev/null 2>&1; then printf 'SURVIVED  %s\n' "$m"
        else killed=$((killed + 1)); printf 'killed    %s\n' "$m"; fi
    done
    printf '%s --mutants: %s/%s killed\n' "$PROG" "$killed" "$total"
    [ "$killed" = "$total" ]
}

case "${1:-}" in
    --self-test|--selftest) self_test; exit $? ;;
    --mutants) mutants; exit $? ;;
    --help|-h) usage ;;
    --*) usage ;;
esac

if ! git -C "$REPO_ROOT" rev-parse --verify -q origin/main >/dev/null; then
    printf '%s: origin/main is not resolvable here (no such remote-tracking ref).\n' "$PROG" >&2
    printf '    An environment gap, not a roadmap defect. Fetch it: git -C %s fetch origin main\n' "$REPO_ROOT" >&2
    exit 2
fi

HEAD_REF="${2:-HEAD}"
if [ -n "${1:-}" ]; then
    BASE_REF="$1"; BASE_HOW="argument"
else
    if ! resolve_base "$HEAD_REF"; then exit 2; fi
fi

# PUSH SHAPE: decline to judge, out loud. `resolve_base` says so in BASE_HOW --
# the one place that decision is made, so this cannot drift from it.
case "$BASE_HOW" in
    *"push shape"*)
        printf 'SKIP  push-shape: %s (%s) is on the origin/main first-parent line, so the only nameable base is %s.\n' \
            "$HEAD_REF" "$(git -C "$REPO_ROOT" rev-parse --short "$HEAD_REF^{commit}")" "$BASE_REF"
        printf '      base would have been: %s\n' "$BASE_HOW"
        printf '      Grading it would judge the PREVIOUS merge, not this change: a differential guard added today would\n'
        printf '      refuse commits that landed before it existed (2 of the last 8 first-parent commits of origin/main are\n'
        printf '      fragment-less) and red-line main from the moment it merges.\n'
        printf '      NOT CHECKED HERE: the %s <-> %s/ pairing, and the aggregate, over %s..%s. Both are\n' \
            "$ROADMAP_FILE" "$ENTRIES_DIR" "$BASE_REF" "$HEAD_REF"
        printf '      graded on the pull_request / merge_group run of that change, where the base is a real merge-base.\n'
        exit 0
        ;;
esac

if [ "$(git -C "$REPO_ROOT" rev-parse "$BASE_REF^{commit}")" = "$(git -C "$REPO_ROOT" rev-parse "$HEAD_REF^{commit}")" ]; then
    printf 'PASS  base and head are the same commit: there is no diff to judge here\n'
    exit 0
fi

printf '=== base=%s (%s) head=%s ===\n' "$BASE_REF" "$BASE_HOW" "$HEAD_REF"
judge "$REPO_ROOT" "$BASE_REF" "$HEAD_REF"
exit $?
