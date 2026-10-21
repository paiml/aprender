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
ROADMAP_FILE="docs/roadmaps/roadmap.yaml"
ENTRIES_DIR="docs/roadmaps/entries"
BASE_FILE="docs/roadmaps/roadmap-base.yaml"

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
# GENERATED mode: roadmap.yaml is not committed at all. It is built on demand
# by `make roadmap-aggregate` from the frozen $BASE_FILE plus entries/, so two
# PRs can never contend on it and it cannot go stale after a main merge.
# Active once head carries $BASE_FILE. Three rules over base..head:
#   G1 $ROADMAP_FILE is not tracked at head (a re-added aggregate is refused).
#   G2 $BASE_FILE is frozen: unchanged from base. The migration PR (base has no
#      $BASE_FILE) is allowed only if it is byte-equal to base's roadmap.yaml.
#   G3 base + entries/ aggregates idempotently, with no duplicate ids.
# ---------------------------------------------------------------------------
judge_generated() {
    local repo=$1 base=$2 head=$3 td violations=0 name out rc
    td=$(mktemp -d "${TMPDIR:-/tmp}/rmfraggen.XXXXXX") || return 2
    printf '=== %s (generated aggregate): base=%s head=%s ===\n' "$PROG" "$base" "$head"
    if git -C "$repo" cat-file -e "$head:$ROADMAP_FILE" 2>/dev/null; then
        violations=$((violations + 1))
        printf 'FAIL  G1  %s is tracked at head — it is GENERATED (make roadmap-aggregate), never committed\n' "$ROADMAP_FILE"
    fi
    if git -C "$repo" cat-file -e "$base:$BASE_FILE" 2>/dev/null; then
        if [ -n "$(git -C "$repo" diff --name-only "$base" "$head" -- "$BASE_FILE")" ]; then
            violations=$((violations + 1))
            printf 'FAIL  G2  %s changed — the base is frozen; change a ticket through docs/roadmaps/entries/<ID>.yaml (a fragment supersedes the base entry)\n' "$BASE_FILE"
        fi
    elif ! git -C "$repo" show "$base:$ROADMAP_FILE" 2>/dev/null | cmp -s - <(git -C "$repo" show "$head:$BASE_FILE"); then
        violations=$((violations + 1))
        printf 'FAIL  G2  migration: %s is not byte-equal to base'"'"'s %s\n' "$BASE_FILE" "$ROADMAP_FILE"
    fi
    if git -C "$repo" cat-file -e "$head:$ENTRIES_DIR" 2>/dev/null; then
        git -C "$repo" archive "$head" -- "$ENTRIES_DIR" | tar -x -C "$td" || { rm -rf -- "${td:?}"; return 2; }
    fi
    # G4 entries/ is append-only: deleting a fragment drops a ticket (or un-supersedes a base entry).
    if [ -n "$(git -C "$repo" diff --no-renames --name-only --diff-filter=D "$base" "$head" -- "$ENTRIES_DIR")" ]; then
        violations=$((violations + 1))
        printf 'FAIL  G4  a fragment under %s/ is deleted — fragments are never removed; set the ticket status instead\n' "$ENTRIES_DIR"
    fi
    # G4b the generator reads only the top-level entries/*.yaml; anything else there is invisible to it.
    while IFS=$'\t' read -r meta name; do
        if ! [[ "$name" =~ ^$ENTRIES_DIR/[^/]+\.yaml$ ]] || [ "${meta%% *}" = 120000 ]; then
            violations=$((violations + 1))
            printf 'FAIL  G4  %s: only regular top-level %s/<ID>.yaml files are fragments (no subdirs, no symlinks)\n' "$name" "$ENTRIES_DIR"
        fi
    done < <(git -C "$repo" ls-tree -r "$head" -- "$ENTRIES_DIR" 2>/dev/null)
    # G3b one fragment = one YAML list holding ONE mapping whose id is its filename (loaded, not grepped,
    # so a second entry cannot hide behind an indented or quoted line).
    local f ids
    for f in "$td/$ENTRIES_DIR"/*.yaml; do
        [ -e "$f" ] || continue
        ids=$(python3 "$PY_LIB" fragment-ids "$f" 2>/dev/null) || ids="(unparseable)"
        if [ "$ids" != "$(basename "$f" .yaml)" ]; then
            violations=$((violations + 1))
            printf 'FAIL  G3  %s/%s must hold exactly one entry whose id is its filename (found: %s)\n' "$ENTRIES_DIR" "$(basename "$f")" "$ids"
        fi
    done
    # G5 every fragment added or changed carries a `Pmat-Ticket: <ID>` trailer on a commit of base..head
    # (sov.roadmap-fragment-parity demands it once the aggregate is untracked; fail here with the remedy, not there).
    # Judged exactly as sov does: git's own trailer parser (LAST paragraph only), same command. A merge_group
    # commit is the queue's squash of the PR, whose message the queue composes, so G5 is a PR-time check only.
    local trailers
    trailers=$(git -C "$repo" log --format='%(trailers:key=Pmat-Ticket,valueonly)' "$base..$head" 2>/dev/null | tr -d ' ' | grep -v '^$' | sort -u || true)
    while IFS= read -r name; do
        [ "${GITHUB_EVENT_NAME:-}" = merge_group ] && break
        [[ "$name" =~ ^$ENTRIES_DIR/([^/]+)\.yaml$ ]] || continue
        if ! printf '%s\n' "$trailers" | grep -qFx -- "${BASH_REMATCH[1]}"; then
            violations=$((violations + 1))
            printf 'FAIL  G5  %s changes with no `Pmat-Ticket: %s` trailer in the LAST paragraph of a commit in the range (git trailers, as sov reads them) — e.g. git commit --allow-empty -m "chore: ticket trailer" -m "Pmat-Ticket: %s"\n' "$name" "${BASH_REMATCH[1]}" "${BASH_REMATCH[1]}"
        fi
    done < <(git -C "$repo" diff --no-renames --name-only --diff-filter=AM "$base" "$head" -- "$ENTRIES_DIR")
    if ! git -C "$repo" show "$head:$BASE_FILE" >"$td/base.yaml" 2>/dev/null; then
        rm -rf -- "${td:?}"
        printf 'FAIL  G2  %s is missing at head — the frozen base cannot be deleted or renamed\n' "$BASE_FILE"
        printf '%s: %s violation(s)\n' "$PROG" "$((violations + 1))"
        return 1
    fi
    out=$(python3 "$PY_LIB" aggregate --check --base-file "$td/base.yaml" --roadmap "$td/absent.yaml" --entries "$td/$ENTRIES_DIR" 2>&1)
    rc=$?
    rm -rf -- "${td:?}"
    if [ "$rc" = 0 ]; then
        printf 'ok    G3  base + entries/ aggregates idempotently\n'
    else
        violations=$((violations + 1))
        printf 'FAIL  G3  base + entries/ does not aggregate cleanly:\n%s\n' "$out"
    fi
    if [ "$violations" != 0 ]; then
        printf '%s: %s violation(s)\n' "$PROG" "$violations"
        return 1
    fi
    printf 'PASS  %s: generated aggregate untracked, base frozen, fragments aggregate\n' "$PROG"
    return 0
}

# ---------------------------------------------------------------------------
# judge <repo> <base-ref> <head-ref> -> 0 clean · 1 violation · 2 cannot judge
# ---------------------------------------------------------------------------
judge() {
    local repo=$1 base=$2 head=$3
    local td roadmap_changed=0 frag_paths names name verdict eid line
    local violations=0 checked=0 out rc

    git -C "$repo" rev-parse --verify -q "$base^{commit}" >/dev/null || {
        printf 'ENV   %s: base ref %s is not a commit here — refusing to judge (never a pass)\n' "$PROG" "$base" >&2
        return 2
    }
    git -C "$repo" rev-parse --verify -q "$head^{commit}" >/dev/null || {
        printf 'ENV   %s: head ref %s is not a commit here — refusing to judge (never a pass)\n' "$PROG" "$head" >&2
        return 2
    }

    if git -C "$repo" cat-file -e "$head:$BASE_FILE" 2>/dev/null \
        || git -C "$repo" cat-file -e "$base:$BASE_FILE" 2>/dev/null; then
        judge_generated "$repo" "$base" "$head"
        return $?
    fi

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
    out=$(python3 "$PY_LIB" aggregate --check --roadmap "$td/head.yaml" --entries "$td/entries" 2>&1)
    rc=$?
    rm -rf -- "${td:?}"
    if [ "$rc" = 0 ]; then
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
    unset GITHUB_EVENT_NAME   # the rows judge PR-shape ranges; a merge_group env would skip G5
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

    commit_all() {
        git -C "$1" add -A || return
        if [ -n "${2:-}" ]; then git -C "$1" commit -qm head -m "$2"; else git -C "$1" commit -qm head; fi
    }

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
    # Supersession: edit an ADOPTED entry through its fragment, regenerate.
    b_supersede() {
        printf -- '- id: PMAT-100\n  title: first\n  status: completed\n' >"$1/$ENTRIES_DIR/PMAT-100.yaml"
        python3 "$PY_LIB" aggregate --write --roadmap "$1/$ROADMAP_FILE" >/dev/null 2>&1 || return 1
        commit_all "$1"
    }

    # ---- the case table --------------------------------------------------
    row 'MEASURED SHAPE: entry added to roadmap.yaml only (what pmat work add writes) -> REFUSE' \
        1 'ADDED    PMAT-200 in docs/roadmaps/roadmap.yaml with NO change' b_monolith_only
    row 'fragment + regenerated aggregate, in sync -> PASS' \
        0 'ok    ADDED    PMAT-200 — docs/roadmaps/entries/PMAT-200.yaml changes in the same diff' b_fragment_and_aggregate
    row 'fragment added, aggregate NOT regenerated -> REFUSE, naming the drift' \
        1 'FAIL  DRIFT' b_fragment_no_regen
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

    # ---- GENERATED mode (roadmap.yaml untracked, roadmap-base.yaml frozen) ----
    grow() { # LABEL WANT PATTERN BUILDER -- fixture: migrated tree, then BUILDER commits a change
        local label=$1 want=$2 pat=$3 builder=$4 d out rc
        n=$((n + 1)); d="$td/g$n"
        mkdir -p "$d/$ENTRIES_DIR" && git -C "$d" init -q --template="$TEMPLATE" \
            && git -C "$d" config user.email t@e.com && git -C "$d" config user.name t \
            && printf '%s' "$BASE_ROADMAP" >"$d/$BASE_FILE" \
            && printf -- '- id: PMAT-100\n  title: first\n  status: planned\n' >"$d/$ENTRIES_DIR/PMAT-100.yaml" \
            && git -C "$d" add -A && git -C "$d" commit -qm base && "$builder" "$d" \
            || { printf 'FAIL  row %-2s %s: fixture/builder failed\n' "$n" "$label"; red=$((red + 1)); return; }
        out=$(judge "$d" HEAD~1 HEAD 2>&1); rc=$?
        if [ "$rc" != "$want" ] || ! grep -qF -- "$pat" <<<"$out"; then
            printf 'FAIL  row %-2s rc=%s (wanted %s, must contain: %s)  %s\n' "$n" "$rc" "$want" "$pat" "$label"
            printf '%s\n' "$out" | sed 's/^/        /'; red=$((red + 1)); return
        fi
        printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
    }
    g_fragment_only() { printf -- '- id: PMAT-200\n  title: new\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"; commit_all "$1" "Pmat-Ticket: PMAT-200"; }
    g_fragment_no_trailer() { printf -- '- id: PMAT-200\n  title: new\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"; commit_all "$1"; }
    g_trailer_mid() { printf -- '- id: PMAT-200\n  title: new\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"; git -C "$1" add -A && git -C "$1" commit -qm head -m "Pmat-Ticket: PMAT-200" -m "Co-Authored-By: T <t@e.com>"; }
    g_trailer_wrong() { printf -- '- id: PMAT-200\n  title: new\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"; commit_all "$1" "Pmat-Ticket: PMAT-999"; }
    g_two_entries() { printf -- '- id: PMAT-200\n  title: new\n  status: planned\n- id: PMAT-201\n  title: smuggled\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"; commit_all "$1" "Pmat-Ticket: PMAT-200"; }
    g_readd_aggregate() { cp "$1/$BASE_FILE" "$1/$ROADMAP_FILE"; commit_all "$1"; }
    g_edit_base() { printf -- '- id: PMAT-400\n  title: sneaked\n  status: planned\n' >>"$1/$BASE_FILE"; commit_all "$1"; }
    g_delete_fragment() { git -C "$1" rm -q "$ENTRIES_DIR/PMAT-100.yaml"; commit_all "$1"; }
    g_rename_fragment() { git -C "$1" mv "$ENTRIES_DIR/PMAT-100.yaml" "$ENTRIES_DIR/PMAT-199.yaml"; sed -i 's/PMAT-100/PMAT-199/' "$1/$ENTRIES_DIR/PMAT-199.yaml"; commit_all "$1"; }
    g_nested_fragment() { mkdir -p "$1/$ENTRIES_DIR/sub"; printf -- '- id: PMAT-600\n  title: hidden\n  status: planned\n' >"$1/$ENTRIES_DIR/sub/PMAT-600.yaml"; commit_all "$1"; }
    g_symlink_fragment() { ln -s ../../README.md "$1/$ENTRIES_DIR/PMAT-700.yaml"; commit_all "$1" "Pmat-Ticket: PMAT-700"; }
    g_delete_base() { git -C "$1" rm -q "$BASE_FILE"; commit_all "$1"; }
    g_rename_base() { git -C "$1" mv "$BASE_FILE" "$1/docs/roadmaps/moved.yaml"; commit_all "$1"; }
    g_dup_fragment() { printf -- '- id: PMAT-300\n  title: dup\n  status: planned\n' | tee "$1/$ENTRIES_DIR/PMAT-300.yaml" >"$1/$ENTRIES_DIR/PMAT-301.yaml"; commit_all "$1"; }
    m_migrate() { git -C "$1" mv "$ROADMAP_FILE" "$BASE_FILE"; commit_all "$1"; }
    m_migrate_edit() { git -C "$1" mv "$ROADMAP_FILE" "$BASE_FILE"; printf -- '- id: PMAT-500\n  title: x\n  status: planned\n' >>"$1/$BASE_FILE"; commit_all "$1"; }
    row 'migration: roadmap.yaml renamed byte-identically to the base -> PASS' 0 'PASS  check_roadmap_fragment_required.sh: generated aggregate' m_migrate
    row 'migration: the renamed base is not byte-equal -> REFUSE G2' 1 'FAIL  G2' m_migrate_edit
    grow 'generated: a fragment alone (no aggregate to regenerate) -> PASS' 0 'PASS  check_roadmap_fragment_required.sh: generated aggregate' g_fragment_only
    grow 'generated: roadmap.yaml re-added and tracked -> REFUSE G1' 1 'FAIL  G1' g_readd_aggregate
    grow 'generated: the frozen base edited -> REFUSE G2' 1 'FAIL  G2' g_edit_base
    grow 'generated: a duplicate id across fragments -> REFUSE G3' 1 'FAIL  G3' g_dup_fragment
    grow 'generated: a fragment with no Pmat-Ticket trailer -> REFUSE G5' 1 'FAIL  G5' g_fragment_no_trailer
    grow 'generated: trailer present but NOT in the last paragraph -> REFUSE G5' 1 'FAIL  G5' g_trailer_mid
    grow 'generated: trailer names another ticket -> REFUSE G5' 1 'FAIL  G5' g_trailer_wrong
    grow 'generated: a second entry inside one fragment -> REFUSE G3' 1 'FAIL  G3' g_two_entries
    grow 'generated: a fragment deleted -> REFUSE G4' 1 'FAIL  G4' g_delete_fragment
    grow 'generated: a fragment renamed with its id edited -> REFUSE G4' 1 'FAIL  G4' g_rename_fragment
    grow 'generated: a nested fragment the generator never reads -> REFUSE G4' 1 'FAIL  G4' g_nested_fragment
    grow 'generated: a symlink fragment (target can change unseen) -> REFUSE G4' 1 'FAIL  G4' g_symlink_fragment
    grow 'generated: the frozen base deleted -> REFUSE G2 (not the legacy judge)' 1 'FAIL  G2' g_delete_base
    grow 'generated: the frozen base renamed -> REFUSE G2' 1 'FAIL  G2' g_rename_base

    # An unresolvable ref is ENV (rc 2), never a pass.
    n=$((n + 1))
    local out rc
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
case "${1:-}" in
    --self-test|--selftest) self_test; exit $? ;;
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
