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
#      EXCEPT an entries-only diff (fragments change; roadmap.yaml and roadmap-base.yaml do not;
#      no root Cargo.toml version bump): there the regeneration is deferred to the bump PR and
#      to `--full` / `make roadmap-aggregate-check`, so open PRs stop conflicting on the tail
#      of roadmap.yaml (#6286 quorum: roadmap-aggregate verdict).
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
BASE_FILE="docs/roadmaps/roadmap-base.yaml"
ENTRIES_DIR="docs/roadmaps/entries"

# shellcheck source=scripts/lib/resolve_base.sh
. "$REPO_ROOT/scripts/lib/resolve_base.sh" || exit 1

usage() {
    printf 'usage: %s [--full] [<base-ref> [<head-ref>]]  or  %s --self-test\n' "$PROG" "$PROG" >&2
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
    printf '  * an entries-only PR stages just the fragment: %s/<ID>.yaml. The aggregate is\n' "$ENTRIES_DIR"
    printf '    regenerated by the bump PR (`make roadmap-aggregate`, then stage %s).\n' "$ROADMAP_FILE"
    printf '    A PR that touches %s, or bumps the version, must be regenerated:\n' "$ROADMAP_FILE"
    printf '        make roadmap-aggregate\n'
    printf '        git add %s/<ID>.yaml %s\n' "$ENTRIES_DIR" "$ROADMAP_FILE"
    printf '  * an id that is not filename-safe (prose, or carrying a path separator) has NO\n'
    printf '    fragment path: it stays in the base and is immutable (RMFR-OB-005).\n'
}

# intro_regenerates REPO BASE TD -- the introduced base is right when it regenerates the base
# ref's aggregate from the base ref's own fragments. Exact equality with the base ref's
# roadmap.yaml would refuse a correct snapshot whenever another roadmap PR lands first.
intro_regenerates() {
    local repo=$1 base=$2 td=$3 name
    mkdir -p "$td/intro/entries" || return 1
    cp "$td/base.yaml" "$td/intro/roadmap.yaml" || return 1
    cp "$td/roadmap-base.yaml" "$td/intro/roadmap-base.yaml" || return 1
    while IFS= read -r name; do
        case "$name" in *.yaml) ;; *) continue ;; esac
        git -C "$repo" show "$base:$name" >"$td/intro/entries/${name##*/}" || return 1
    done < <(git -C "$repo" ls-tree -r --name-only "$base" -- "$ENTRIES_DIR" 2>/dev/null)
    python3 "$PY_LIB" aggregate --check --roadmap "$td/intro/roadmap.yaml" --entries "$td/intro/entries" >/dev/null 2>&1
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

    td=$(mktemp -d "${TMPDIR:-/tmp}/rmfragreq.XXXXXX") || return 2

    if [ -n "$(git -C "$repo" diff --name-only "$base" "$head" -- "$ROADMAP_FILE")" ]; then
        roadmap_changed=1
    fi
    frag_paths=$(git -C "$repo" diff --name-only "$base" "$head" -- "$ENTRIES_DIR")
    base_changed=$(git -C "$repo" diff --name-only "$base" "$head" -- "$BASE_FILE")

    # A version bump (root Cargo.toml `version =` line) is a full-regen diff even when it touches neither
    # roadmap file (bump-version.sh does not regenerate). A git failure reading that diff fails closed:
    # it counts as a bump, so freshness is checked rather than skipped.
    bumped=0
    if cargo_diff=$(git -C "$repo" diff -U0 "$base" "$head" -- Cargo.toml 2>/dev/null); then
        grep -Eq '^[+-][[:space:]]*version[[:space:]]*=' <<<"$cargo_diff" && bumped=1
    else
        bumped=1
    fi

    if [ "$roadmap_changed" = 0 ] && [ -z "$frag_paths" ] && [ -z "$base_changed" ] && [ "$bumped" = 0 ]; then
        rm -rf -- "${td:?}"
        printf 'PASS  %s: this diff touches neither %s nor %s/ — nothing to judge\n' \
            "$PROG" "$ROADMAP_FILE" "$ENTRIES_DIR"
        return 0
    fi

    # ENTRIES-ONLY (#6286 quorum, roadmap-aggregate verdict): a diff that adds or edits fragments and
    # touches neither roadmap.yaml nor the frozen base leaves the regenerated aggregate to the bump PR.
    # Every open PR regenerating the same tail of roadmap.yaml made any two queued PRs conflict
    # (#6268 was removed twice). A version bump (root Cargo.toml `version =` line), or --full, is the
    # full regen check; so is any diff that touches roadmap.yaml.
    entries_only=0
    if [ "$roadmap_changed" = 0 ] && [ -z "$base_changed" ] && [ -n "$frag_paths" ] && [ "${FULL_REGEN:-0}" != 1 ] && [ "$bumped" = 0 ]; then
        entries_only=1
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
    # The frozen base the head aggregate grows from (#6185). Absent at head (a pre-S1
    # commit), head.yaml is its own base: said out loud below, never a silent skip.
    have_base=0
    if git -C "$repo" show "$head:$BASE_FILE" >"$td/roadmap-base.yaml" 2>/dev/null; then
        have_base=1
    else
        rm -f -- "${td:?}/roadmap-base.yaml"
    fi
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

    # The BASE ref's aggregate, REGENERATED from the base ref's own fragments and frozen base (as
    # intro_regenerates does). Entries-only PRs land without regenerating roadmap.yaml, so the base
    # ref's file can lag its fragments: judged against it, the bump PR that regenerates would be refused
    # by RULE 1 for entries it never touched. Without a frozen base at the base ref, it is judged as is.
    if git -C "$repo" show "$base:$BASE_FILE" >"$td/rb-base.yaml" 2>/dev/null; then
        mkdir -p "$td/rb/entries"
        cp "$td/rb-base.yaml" "$td/rb/roadmap-base.yaml"
        cp "$td/base.yaml" "$td/rb/roadmap.yaml"
        while IFS= read -r name; do
            case "$name" in *.yaml) ;; *) continue ;; esac
            git -C "$repo" show "$base:$name" >"$td/rb/entries/${name##*/}" 2>/dev/null || continue
        done < <(git -C "$repo" ls-tree -r --name-only "$base" -- "$ENTRIES_DIR" 2>/dev/null)
        if python3 "$PY_LIB" aggregate --write --roadmap "$td/rb/roadmap.yaml" --entries "$td/rb/entries" >/dev/null 2>&1; then
            cp "$td/rb/roadmap.yaml" "$td/base.yaml"
        else
            printf 'note  could not regenerate the base ref aggregate; RULE 1 judges against its roadmap.yaml as committed\n'
        fi
    fi

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

    # --- RULE 3: the frozen base never changes (#6185) -----------------------
    # A change to a legacy entry is a fragment that supersedes its id; editing the base
    # would make it the new hotspot and a way round rule 1. The one allowed change is its
    # introduction, as a snapshot that regenerates the base ref's aggregate.
    if [ -n "$base_changed" ]; then
        if git -C "$repo" cat-file -e "$base:$BASE_FILE" 2>/dev/null; then
            violations=$((violations + 1))
            printf 'FAIL  FROZEN: %s changed. It is frozen: supersede an entry with a fragment in %s/ instead\n' \
                "$BASE_FILE" "$ENTRIES_DIR"
        elif intro_regenerates "$repo" "$base" "$td"; then
            printf 'ok    %s introduced: aggregate(it, entries/ at the base ref) == %s at the base ref\n' "$BASE_FILE" "$ROADMAP_FILE"
        else
            violations=$((violations + 1))
            printf 'FAIL  FROZEN: %s introduced but aggregate(it, entries/ at the base ref) != %s at the base ref\n' \
                "$BASE_FILE" "$ROADMAP_FILE"
        fi
    fi
    [ "$have_base" = 1 ] || printf 'note  %s absent at head: the aggregate is judged against itself (pre-S1 shape)\n' "$BASE_FILE"

    # --- RULE 2: the aggregate at head is REGENERATED, not drifting ----------
    if [ "$entries_only" = 1 ]; then
        rc=0; out=""
    else
        out=$(python3 "$PY_LIB" aggregate --check --roadmap "$td/head.yaml" --entries "$td/entries" 2>&1)
        rc=$?
    fi
    rm -rf -- "${td:?}"
    if [ "$entries_only" = 1 ]; then
        printf 'note  entries-only diff: the regenerated-aggregate check (RULE 2) is deferred to the bump PR and to --full / make roadmap-aggregate-check\n'
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
    b_base_and_fragment_stale() {
        cp "$1/$ROADMAP_FILE" "$1/$BASE_FILE"
        printf -- '- id: PMAT-200\n  title: fragment only\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"
        commit_all "$1"
    }
    # The base ref has the frozen base and a fragment the aggregate lacks (an entries-only PR landed);
    # the head bumps the version and regenerates, touching no entry itself.
    b_lag_then_regen() {
        cp "$1/$ROADMAP_FILE" "$1/$BASE_FILE"
        printf -- '- id: PMAT-200\n  title: landed entries-only\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"
        printf '[workspace.package]\nversion = "0.1.0"\n' >"$1/Cargo.toml"
        git -C "$1" add -A && git -C "$1" commit -qm lagging-base || return 1
        git -C "$1" branch -f lagbase HEAD || return 1
        printf '[workspace.package]\nversion = "0.1.1"\n' >"$1/Cargo.toml"
        python3 "$PY_LIB" aggregate --write --roadmap "$1/$ROADMAP_FILE" >/dev/null 2>&1 || return 1
        commit_all "$1"
    }
    b_bump_nospace_both() {
        printf '[workspace.package]\nversion="0.1.0"\n' >"$1/Cargo.toml"; git -C "$1" add -A && git -C "$1" commit -qm cargo || return 1
        printf '[workspace.package]\nversion="0.1.1"\n' >"$1/Cargo.toml"
        printf -- '- id: PMAT-200\n  title: fragment only\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"
        commit_all "$1"
    }
    # A Cargo.toml-only bump (bump-version.sh regenerates nothing) over a base ref whose aggregate lags its fragments.
    b_cargo_only_bump_stale() {
        printf '[workspace.package]\nversion = "0.1.0"\n' >"$1/Cargo.toml"
        printf -- '- id: PMAT-200\n  title: landed entries-only\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"
        git -C "$1" add -A && git -C "$1" commit -qm lagging || return 1
        printf '[workspace.package]\nversion = "0.1.1"\n' >"$1/Cargo.toml"
        commit_all "$1"
    }
    b_cargo_only_bump_fresh() {
        printf '[workspace.package]\nversion = "0.1.0"\n' >"$1/Cargo.toml"; git -C "$1" add -A && git -C "$1" commit -qm cargo || return 1
        printf '[workspace.package]\nversion = "0.1.1"\n' >"$1/Cargo.toml"
        commit_all "$1"
    }
    # The real-PR path: the frozen base already exists at the base ref, then an entry goes into the monolith.
    b_base_present_monolith() {
        cp "$1/$ROADMAP_FILE" "$1/$BASE_FILE" && git -C "$1" add -A && git -C "$1" commit -qm snapshot || return 1
        b_monolith_only "$1"
    }
    b_bump_nospace_no_regen() {
        printf '[workspace.package]\nversion = "0.1.0"\n' >"$1/Cargo.toml"; git -C "$1" add -A && git -C "$1" commit -qm cargo || return 1
        printf '[workspace.package]\nversion="0.1.1"\n' >"$1/Cargo.toml"
        printf -- '- id: PMAT-200\n  title: fragment only\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"
        commit_all "$1"
    }
    b_bump_fragment_no_regen() {
        printf '[workspace.package]\nversion = "0.1.0"\n' >"$1/Cargo.toml"; git -C "$1" add -A && git -C "$1" commit -qm cargo || return 1
        printf '[workspace.package]\nversion = "0.1.1"\n' >"$1/Cargo.toml"
        printf -- '- id: PMAT-200\n  title: fragment only\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"
        commit_all "$1"
    }
    # roadmap.yaml touched (preamble edit only, so rule 1 is quiet) beside a fragment the aggregate lacks.
    b_roadmap_touched_stale() {
        printf '# touched\n' | cat - "$1/$ROADMAP_FILE" >"$1/r.tmp" && mv "$1/r.tmp" "$1/$ROADMAP_FILE"
        printf -- '- id: PMAT-200\n  title: fragment only\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"
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

    # ---- frozen base (#6185) ---------------------------------------------
    snapshot_base() { cp "$1/$ROADMAP_FILE" "$1/$BASE_FILE" && commit_all "$1"; }
    # The base introduced as a snapshot of the base ref's roadmap.yaml.
    b_base_snapshot() { snapshot_base "$1"; }
    # ... or as something else: a way to smuggle an entry past rule 1.
    b_base_not_snapshot() {
        cp "$1/$ROADMAP_FILE" "$1/$BASE_FILE"
        printf -- '- id: PMAT-999\n  title: smuggled into the base\n  status: planned\n' >>"$1/$BASE_FILE"
        commit_all "$1"
    }
    # Once the base exists, any edit to it is refused.
    b_base_edited() {
        snapshot_base "$1" || return 1
        printf -- '- id: PMAT-999\n  title: edited the frozen base\n  status: planned\n' >>"$1/$BASE_FILE"
        commit_all "$1"
    }
    # LIVENESS: another roadmap PR (a fragment and its regenerated aggregate) lands after the
    # snapshot was cut. The stale-but-correct snapshot must still be accepted.
    b_base_snapshot_stale() {
        cp "$1/$ROADMAP_FILE" "$1/old.yaml"
        printf -- '- id: PMAT-200\n  title: landed first\n  status: planned\n' >"$1/$ENTRIES_DIR/PMAT-200.yaml"
        python3 "$PY_LIB" aggregate --write --roadmap "$1/$ROADMAP_FILE" >/dev/null 2>&1 || return 1
        git -C "$1" add -A && git -C "$1" commit -qm landed-first || return 1
        mv "$1/old.yaml" "$1/$BASE_FILE" && commit_all "$1"
    }
    # PARITY: the header is not entry content, so rule 1 lets it change freely. Only a head
    # aggregate grown from the frozen base (not from itself) can see a hand-edited header.
    b_tampered_header() {
        snapshot_base "$1" || return 1
        python3 - "$1/$ROADMAP_FILE" <<'PY' || return 1
import sys
p = sys.argv[1]
t = open(p, encoding="utf-8").read()
open(p, "w", encoding="utf-8").write(t.replace("github_enabled: true\n", "github_enabled: false\n"))
PY
        commit_all "$1"
    }

    # ---- the case table --------------------------------------------------
    row 'MEASURED SHAPE: entry added to roadmap.yaml only (what pmat work add writes) -> REFUSE' \
        1 'ADDED    PMAT-200 in docs/roadmaps/roadmap.yaml with NO change' b_monolith_only
    row 'fragment + regenerated aggregate, in sync -> PASS' \
        0 'ok    ADDED    PMAT-200 — docs/roadmaps/entries/PMAT-200.yaml changes in the same diff' b_fragment_and_aggregate
    row 'fragment added, aggregate NOT regenerated (entries-only diff) -> PASS, regen deferred out loud' \
        0 'entries-only diff: the regenerated-aggregate check' b_fragment_no_regen
    FULL_REGEN=1 row 'entries-only diff under --full (main / bump check) -> REFUSE, naming the drift' \
        1 'FAIL  DRIFT' b_fragment_no_regen
    row 'frozen base introduced + a fragment, aggregate stale -> REFUSE (a touched base never reaches the skip)' \
        1 'FAIL  DRIFT' b_base_and_fragment_stale
    row 'LAG: the base ref aggregate is stale vs its fragments, a bump regenerates it -> PASS (no deadlock)' \
        0 'ok    ' b_lag_then_regen
    row 'a bump written version="x" (no spaces) with an unregenerated aggregate -> REFUSE' \
        1 'FAIL  DRIFT' b_bump_nospace_no_regen
    row 'version="x" with NO spaces on both sides of the bump, aggregate stale -> REFUSE' \
        1 'FAIL  DRIFT' b_bump_nospace_both
    row 'a Cargo.toml-only bump over a lagging aggregate -> REFUSE (never "nothing to judge")' \
        1 'FAIL  DRIFT' b_cargo_only_bump_stale
    row 'a Cargo.toml-only bump over a fresh aggregate -> PASS, freshness checked' \
        0 'ok    docs/roadmaps/roadmap.yaml == aggregate' b_cargo_only_bump_fresh
    row 'frozen base present at the base ref, entry written to the monolith only -> REFUSE (rule 1, real-PR path)' \
        1 'ADDED    PMAT-200 in docs/roadmaps/roadmap.yaml with NO change' b_base_present_monolith
    row 'a bump PR (root Cargo.toml version changes) with an unregenerated aggregate -> REFUSE' \
        1 'FAIL  DRIFT' b_bump_fragment_no_regen
    row 'a diff touching roadmap.yaml and a fragment, aggregate stale -> REFUSE (entries-only never applies)' \
        1 'FAIL  DRIFT' b_roadmap_touched_stale
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
    row 'frozen base introduced as a snapshot -> PASS' \
        0 'introduced: aggregate(it' b_base_snapshot
    row 'LIVENESS: a stale snapshot, another roadmap PR landed first -> PASS' \
        0 'introduced: aggregate(it' b_base_snapshot_stale
    row 'frozen base introduced as something other than the snapshot -> REFUSE' \
        1 'FROZEN: docs/roadmaps/roadmap-base.yaml introduced but' b_base_not_snapshot
    row 'frozen base edited after it exists -> REFUSE' \
        1 'FROZEN: docs/roadmaps/roadmap-base.yaml changed' b_base_edited
    row 'PARITY: header of roadmap.yaml hand-edited away from the frozen base -> REFUSE' \
        1 'FAIL  DRIFT' b_tampered_header

    # A git failure reading the Cargo.toml diff fails CLOSED: the entries-only skip must not be taken.
    n=$((n + 1))
    local fout frc
    mkrepo "$td/r$n" >/dev/null 2>&1 && b_fragment_no_regen "$td/r$n" >/dev/null 2>&1
    judge_with_failing_cargo_diff() {
        git() {
            case " $* " in
                *" -- Cargo.toml "*) return 1 ;;
            esac
            command git "$@"
        }
        judge "$@"
    }
    fout=$(judge_with_failing_cargo_diff "$td/r$n" HEAD~1 HEAD 2>&1)
    frc=$?
    if [ "$frc" = 1 ] && grep -qF 'FAIL  DRIFT' <<<"$fout"; then
        printf 'ok    row %-2s rc=1  a git failure reading the Cargo.toml diff fails closed (full check, DRIFT)\n' "$n"
    else
        printf 'FAIL  row %-2s rc=%s (wanted 1 + DRIFT)  a git failure reading the Cargo.toml diff must fail closed\n' "$n" "$frc"
        printf '%s\n' "$fout" | sed 's/^/        /'
        red=$((red + 1))
    fi

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
        local builder=${DR_BUILDER:-b_monolith_only}
        n=$((n + 1))
        d="$td/r$n"
        if ! mkrepo "$d" >/dev/null 2>&1 || ! "$builder" "$d" >/dev/null 2>&1; then
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
        out=$(bash "$d/scripts/$PROG" ${DR_ARGS:-} 2>&1)
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

    # --full through the real dispatch: the same entries-only change passes without it and is refused with it.
    DR_BUILDER=b_fragment_no_regen dispatch_row 'entries-only change through the CLI (no --full) -> PASS, regen deferred' \
        0 'entries-only diff: the regenerated-aggregate check' HEAD~1
    DR_BUILDER=b_fragment_no_regen DR_ARGS=--full dispatch_row 'the same change through the CLI with --full -> REFUSE, naming the drift' \
        1 'FAIL  DRIFT' HEAD~1

    printf '%s/%s rows, %s failed\n' "$((n - red))" "$n" "$red"
    [ "$red" = 0 ]
}

# ---------------------------------------------------------------------------
case "${1:-}" in
    --self-test|--selftest) unset GITHUB_EVENT_NAME GITHUB_EVENT_PATH; self_test; exit $? ;;
    --full) FULL_REGEN=1; shift ;;
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
