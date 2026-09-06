#!/usr/bin/env bash
# check_roadmap_diff_additive.sh — a PR's roadmap.yaml diff is ADDITIVE
# (PMAT-980, G-6, #2874).
#
# WHY THIS EXISTS
# ---------------
# `pmat work add` (and `pmat work complete`) re-serialise ALL of
# docs/roadmaps/roadmap.yaml on every call. One 17-line ticket arrived with
# 2,531 unrelated lines rewritten — long strings re-folded onto one line,
# `phases: []` / `subtasks: []` / `estimated_effort: null` / `labels: []`
# materialised onto every entry that had never carried them. Two in-flight
# PRs editing the roadmap in the same window then conflict by construction,
# and a reviewer cannot see the one entry that actually changed inside the
# noise. Commit 6d9ba274a is the measured instance: 2,531 changed lines for
# one real 17-line ticket addition.
#
# THE RULE (the PP-066 driver's), enforced by scripts/lib/roadmap_diff.py:
#
#   a roadmap.yaml diff may only ADD new top-level entries (the id set
#   grows), or edit a ticket's own LIFECYCLE fields (status, updated, notes,
#   github_issue, labels, assigned_to, priority). Anything else — a deleted
#   entry, an edited title/spec/acceptance_criteria, or a wholesale
#   re-render that changes no field at all — is a violation.
#
# THE REMEDY is scripts/roadmap_trim.py (or `roadmap_diff.py trim`): it
# rebuilds the working file as base's bytes for every entry that is
# unchanged or only re-serialised, keeping head's rendering for entries that
# are new or genuinely edited.
#
#   bash scripts/check_roadmap_diff_additive.sh [<base-ref> [<head-ref>]]
#   bash scripts/check_roadmap_diff_additive.sh --self-test
#
# DEFAULTS: base = `git merge-base origin/main HEAD`, head = `HEAD` — the
# same base a PR will be merged against and the tip it currently sits at. A
# missing `origin/main` (no such remote-tracking ref, e.g. a shallow clone
# that never fetched it) is exit 2 — "the box cannot answer" — never exit 1,
# because that is an environment gap, not a roadmap defect.
#
# EXIT: 0 additive; 1 a rule was violated; 2 the box cannot answer (no
# origin/main, PyYAML missing, an unreadable ref/file).

set -uo pipefail

PROG=${0##*/}
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY_LIB="${REPO_ROOT}/scripts/lib/roadmap_diff.py"
PY_TRIM="${REPO_ROOT}/scripts/roadmap_trim.py"
ROADMAP_FILE="docs/roadmaps/roadmap.yaml"

# ---------------------------------------------------------------------------
# run_check <base> <head> [<file>] -> roadmap_diff.py's stdout+exit code.
# ---------------------------------------------------------------------------
run_check() {
    local base="$1" head="$2" file="${3:-$ROADMAP_FILE}"
    python3 "$PY_LIB" check --base "$base" --head "$head" --file "$file"
}

# ---------------------------------------------------------------------------
# --self-test — a synthetic 3-entry roadmap, both polarities per rule.
# ---------------------------------------------------------------------------
# resolve_base <head-ref> -> sets BASE_REF and BASE_HOW, or prints why and returns 1.
#
# The base is `git merge-base origin/main <head>`. CI checks out the PR at
# fetch-depth 1 and fetches origin/main at depth 1, so no common ancestor is
# reachable and merge-base prints nothing. The first version of this guard
# then ran with an EMPTY base, compared the head against itself
# (`base=807 head=807 added=0`) and would have passed any diff — a gate that
# cannot fail (PR #2987, run 33991535406). So an unresolvable merge-base is
# never silently HEAD: in a shallow checkout the head must be the PR's merge
# commit (refs/pull/N/merge, what actions/checkout gives a pull_request job),
# whose FIRST PARENT is the base branch tip; that parent is the base when its
# object is present, else the origin/main tip fetched by the job. A shallow
# checkout at a non-merge head is exit 2 — the box cannot answer.
# ROADMAP_DIFF_FORCE_SHALLOW=1 makes merge-base unresolvable for the case table.
resolve_base() {
    local head=$1 mb parents
    mb=""
    if [ "${ROADMAP_DIFF_FORCE_SHALLOW:-0}" != 1 ]; then
        mb=$(git -C "$REPO_ROOT" merge-base origin/main "$head" 2>/dev/null || true)
    fi
    if [ -n "$mb" ]; then BASE_REF="$mb"; BASE_HOW="merge-base(origin/main, $head)"; return 0; fi
    # read the parents off the commit OBJECT: in a depth-1 clone `rev-list --parents` shows none (shallow graft), `cat-file -p` still does
    parents=$(git -C "$REPO_ROOT" cat-file -p "$head^{commit}" 2>/dev/null | awk '/^parent /{printf "%s ", $2} /^$/{exit}')
    local p1 main_tip; p1=$(printf '%s\n' "$parents" | cut -d' ' -f1); main_tip=$(git -C "$REPO_ROOT" rev-parse origin/main)
    if [ "$(printf '%s\n' "$parents" | wc -w)" -lt 2 ]; then
        # merge_group: the queue's temporary head is a SINGLE-parent squash-shaped commit whose parent is
        # the base branch tip (run 34002682350: parent == origin/main). That parent is the base. Any other
        # single-parent head (a branch commit) has no nameable base here and is refused.
        if [ -n "$p1" ] && [ "$p1" = "$main_tip" ]; then
            BASE_REF="$p1"; BASE_HOW="single parent == origin/main tip (merge_group squash head, shallow checkout)"; return 0
        fi
        printf '%s: merge-base(origin/main, %s) is unresolvable (shallow checkout) and %s is not a merge commit nor a commit on the origin/main tip,\n' "$PROG" "$head" "$head" >&2
        printf '    so no base can be named. A pull_request job checks out refs/pull/N/merge, a merge_group job the queue head; run with an explicit <base> otherwise.\n' >&2
        return 1
    fi
    if git -C "$REPO_ROOT" cat-file -e "$p1^{commit}" 2>/dev/null; then
        BASE_REF="$p1"; BASE_HOW="first parent of the merge commit $head (shallow checkout)"
    else
        BASE_REF=$(git -C "$REPO_ROOT" rev-parse origin/main); BASE_HOW="origin/main tip (shallow checkout; the merge commit's first parent is not fetched)"
    fi
    return 0
}

SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/${BASH_SOURCE[0]##*/}"   # absolute: the case table cds into a scratch repo before sourcing it
if [ "${1:-}" = "--lib-only" ]; then return 0 2>/dev/null || exit 0; fi
if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d) || { printf 'FAIL mktemp -d\n'; exit 2; }
    trap 'rm -rf -- "${TD:?}"' EXIT
    row=0
    fails=0

    write_base() {
        cat >"$TD/base.yaml" <<'EOF'
roadmap_version: '1.0'
github_enabled: true
github_repo: paiml/aprender
roadmap:
- id: A-1
  title: 'first entry'
  status: planned
  phases: []
  subtasks: []
  estimated_effort: null
  labels: []
  notes: null
- id: A-2
  title: 'second entry'
  status: planned
  notes: null
- id: A-3
  title: 'third entry'
  status: planned
  notes: null
EOF
    }
    write_base

    # assert_row <label> <PASS|FAIL> <head-file> [<grep-pattern>]
    #
    # The verdict is PASS/FAIL from the exit code; an optional pattern also
    # asserts the diagnostic landed on the branch it claims to, the same
    # discipline check_pr_review_wiring.sh's case table uses — a guard that
    # refuses everything for the wrong reason still shows FAIL here.
    assert_row() {
        row=$((row + 1))
        local label=$1 want=$2 headf=$3 pat=${4:-} out rc got
        out=$(run_check "${BASEF:-$TD/base.yaml}" "$headf" 2>&1)
        rc=$?
        if [ "$rc" -eq 0 ]; then got=PASS; elif [ "$rc" -eq 1 ]; then got=FAIL; else got=ERROR; fi
        if [ "$got" != "$want" ]; then
            printf 'FAIL  row %-2s %s: wanted %s, got %s (rc=%s)\n' "$row" "$label" "$want" "$got" "$rc"
            printf '%s\n' "$out" | sed 's|^|             |'
            fails=1
            return
        fi
        if [ -n "$pat" ]; then
            case "$out" in
                *"$pat"*) ;;
                *) printf 'FAIL  row %-2s %s: %s, but missing expected: %s\n' "$row" "$label" "$want" "$pat"
                   printf '%s\n' "$out" | sed 's|^|             |'
                   fails=1
                   return ;;
            esac
        fi
        printf 'ok    row %-2s %s\n' "$row" "$label"
    }

    # Row 1: append one entry -> PASS.
    cp "$TD/base.yaml" "$TD/append.yaml"
    cat >>"$TD/append.yaml" <<'EOF'
- id: A-4
  title: 'new entry'
  status: planned
  notes: null
EOF
    assert_row 'append one entry' PASS "$TD/append.yaml" 'added=1'

    # Row 2: append + re-fold every existing entry's title + materialise
    # phases: []/subtasks: []/estimated_effort: null/labels: [] on ALL of
    # them -> FAIL, reserialised=3 (the measured pmat defect shape).
    python3 - "$TD/base.yaml" "$TD/reserial.yaml" <<'PY'
import re
import sys
base = open(sys.argv[1]).read()
def materialise(block):
    # Every entry gets RE-FOLDED (its title re-wrapped across two physical
    # lines, same string once parsed — the literal pmat behaviour) so even
    # an entry that already carries the materialised empty keys (A-1) still
    # differs byte-for-byte from base.
    block = re.sub(
        r"title: '([^']*) ([^']*)'",
        lambda m: f"title: '{m.group(1)}\n    {m.group(2)}'",
        block,
        count=1,
    )
    if "phases: []" in block:
        return block
    lines = block.splitlines(keepends=True)
    out = []
    inserted = False
    for ln in lines:
        if ln.startswith("  notes:") and not inserted:
            out.append("  phases: []\n  subtasks: []\n  estimated_effort: null\n  labels: []\n")
            inserted = True
        out.append(ln)
    return "".join(out)
entries = re.split(r"(?=^- id: )", base, flags=re.M)
out = [entries[0]] + [materialise(e) for e in entries[1:]]
head = "".join(out)
head += "- id: A-4\n  title: 'new entry'\n  status: planned\n  notes: null\n"
open(sys.argv[2], "w").write(head)
PY
    assert_row 're-fold titles + materialise empty keys on all, plus one append' FAIL "$TD/reserial.yaml" 'reserialised=3'

    # Row 3: an existing entry's status+updated change -> PASS (lifecycle).
    python3 - "$TD/base.yaml" "$TD/lifecycle.yaml" <<'PY'
import sys
base = open(sys.argv[1]).read()
head = base.replace(
    "- id: A-2\n  title: 'second entry'\n  status: planned\n  notes: null\n",
    "- id: A-2\n  title: 'second entry'\n  status: completed\n  updated: '2026-09-05T00:00:00Z'\n  notes: 'done'\n",
)
open(sys.argv[2], "w").write(head)
PY
    assert_row 'status + updated change on one entry' PASS "$TD/lifecycle.yaml" 'lifecycle=1'

    # Row 4: an existing entry's title changed -> FAIL.
    python3 - "$TD/base.yaml" "$TD/titlechg.yaml" <<'PY'
import sys
base = open(sys.argv[1]).read()
head = base.replace("title: 'second entry'", "title: 'SECOND ENTRY CHANGED'")
open(sys.argv[2], "w").write(head)
PY
    assert_row 'title changed on an existing entry' FAIL "$TD/titlechg.yaml" 'reserialised'

    # Row 5: an existing entry deleted -> FAIL.
    python3 - "$TD/base.yaml" "$TD/deleted.yaml" <<'PY'
import re, sys
base = open(sys.argv[1]).read()
head = re.sub(r"- id: A-2\n(?:  .*\n)+?(?=- id: A-3)", "", base)
open(sys.argv[2], "w").write(head)
PY
    assert_row 'an existing entry deleted' FAIL "$TD/deleted.yaml" 'deleted'

    # Row 6: duplicate id at head -> FAIL.
    cp "$TD/base.yaml" "$TD/dup.yaml"
    cat >>"$TD/dup.yaml" <<'EOF'
- id: A-3
  title: 'third entry'
  status: planned
  notes: null
EOF
    assert_row 'duplicate id at head' FAIL "$TD/dup.yaml" 'duplicate-id'

    # Row 7: head does not parse -> FAIL.
    cp "$TD/base.yaml" "$TD/badparse.yaml"
    printf '\tbad: [unterminated\n' >>"$TD/badparse.yaml"
    assert_row 'head does not parse as YAML' FAIL "$TD/badparse.yaml" 'head-unparsable'

    # Row 8: the remedy — trim() of the re-serialised head, re-checked -> PASS.
    cp "$TD/reserial.yaml" "$TD/reserial_trimmed.yaml"
    if ! python3 "$PY_LIB" trim --base "$TD/base.yaml" --file "$TD/reserial_trimmed.yaml" --write \
        >"$TD/trim.out" 2>&1
    then
        row=$((row + 1))
        printf 'FAIL  row %-2s trim of the re-serialised head\n' "$row"
        cat "$TD/trim.out" | sed 's|^|             |'
        fails=1
    else
        assert_row 'trim of the re-serialised head, re-checked' PASS "$TD/reserial_trimmed.yaml" 'added=1'
    fi

    # Row 9: the real history, if this checkout's object store can reach it.
    # `git cat-file -e` never MUTATES the repo, so this is safe to run
    # unconditionally; a shallow clone that never fetched the commit reports
    # it absent, and that is reported as SKIP, not silently counted as a
    # pass or a fail.
    row=$((row + 1))
    if git -C "$REPO_ROOT" cat-file -e 6d9ba274a^{commit} 2>/dev/null; then
        hist_out=$(run_check 6d9ba274a~1 6d9ba274a 2>&1)
        hist_rc=$?
        hist_reserial=$(printf '%s\n' "$hist_out" | grep -oE 'reserialised=[0-9]+' | tail -1 | cut -d= -f2)
        if [ "$hist_rc" -eq 1 ] && [ -n "$hist_reserial" ] && [ "$hist_reserial" -ge 100 ]; then
            # trim's --file is BOTH the git-show path (relative to a repo)
            # and the physical working file it rewrites, so testing it
            # against a non-current commit needs a real checkout of that
            # commit at the real relative path — a throwaway detached
            # worktree, removed again below. Read-only w.r.t. repo history:
            # it only checks out an existing commit.
            hist_wt="$TD/hist-wt"
            if git -C "$REPO_ROOT" worktree add --detach "$hist_wt" 6d9ba274a -q \
                    >"$TD/hist.err" 2>&1 \
                && (cd "$hist_wt" && python3 "$PY_LIB" trim --base 6d9ba274a~1 --write) \
                    >"$TD/hist_trim.out" 2>&1 \
                && (cd "$hist_wt" && python3 "$PY_LIB" check --base 6d9ba274a~1 --head "$ROADMAP_FILE") \
                    >"$TD/hist_check.out" 2>&1
            then
                printf 'ok    row %-2s the real 6d9ba274a re-serialisation (reserialised=%s), trimmed -> PASS\n' \
                    "$row" "$hist_reserial"
            else
                printf 'FAIL  row %-2s the trimmed 6d9ba274a result did not re-check clean\n' "$row"
                cat "$TD/hist.err" "$TD/hist_trim.out" "$TD/hist_check.out" 2>/dev/null | sed 's|^|             |'
                fails=1
            fi
            git -C "$REPO_ROOT" worktree remove --force "$hist_wt" >/dev/null 2>&1 || true
        else
            printf 'FAIL  row %-2s 6d9ba274a: wanted FAIL reserialised>=100, got rc=%s reserialised=%s\n' \
                "$row" "$hist_rc" "${hist_reserial:-<none>}"
            fails=1
        fi
    else
        printf 'SKIP  row %-2s history row (6d9ba274a unreachable in this object store)\n' "$row"
        row=$((row - 1))
    fi

    # Rows 10-12: a duplicate id already in BASE is baselined, not this PR's
    # fault (main carried PMAT-966 twice on 2026-09-05); only a duplicate that
    # GROWS at head is a violation; removing the extra copy is the remedy.
    cp "$TD/base.yaml" "$TD/base_dup.yaml"
    cat >>"$TD/base_dup.yaml" <<'EOF'
- id: A-2
  title: 'second entry, minted twice'
  status: planned
  notes: null
EOF
    cp "$TD/base_dup.yaml" "$TD/dup_kept.yaml"
    BASEF="$TD/base_dup.yaml"; assert_row 'pre-existing duplicate in base, unchanged at head' PASS "$TD/dup_kept.yaml" 'known duplicate-id: id=A-2 appears 2 times in base'; unset BASEF
    cp "$TD/base_dup.yaml" "$TD/dup_grown.yaml"
    cat >>"$TD/dup_grown.yaml" <<'EOF'
- id: A-2
  title: 'a third copy'
  status: planned
  notes: null
EOF
    BASEF="$TD/base_dup.yaml"; assert_row 'a base duplicate grown at head (2 -> 3)' FAIL "$TD/dup_grown.yaml" 'duplicate-id: id=A-2 appears 3 times in head (base had 2)'; unset BASEF
    head -n -4 "$TD/base_dup.yaml" >"$TD/dedup.yaml"   # drop the LATER copy of A-2 (its 4 lines), keep the first
    cat >>"$TD/dedup.yaml" <<'EOF'
- id: A-4
  title: 'second entry, minted twice'
  status: planned
  notes: null
EOF
    BASEF="$TD/base_dup.yaml"; assert_row 'the later copy re-minted (dedup PR): A-2 back to 1, A-4 added' PASS "$TD/dedup.yaml" 'added=1'; unset BASEF

    # Rows 13-14: base resolution in a shallow checkout — a merge-commit head
    # resolves to its first parent; a non-merge head is exit 2, never HEAD.
    row=$((row + 1))
    R="$TD/repo"; mkdir -p "$R"; ( cd "$R" && git init -q -b main . && git config user.email t@t && git config user.name t \
        && cp "$TD/base.yaml" r.yaml && git add r.yaml && git commit -qm base && git branch -q feat && git checkout -q feat \
        && cp "$TD/append.yaml" r.yaml && git commit -qam feat && git checkout -q main && git merge -q --no-ff -m merge feat \
        && git update-ref refs/remotes/origin/main "$(git rev-parse main)" )
    got=""
    got=$( cd "$R" && ROADMAP_DIFF_FORCE_SHALLOW=1 bash -c '. "$0" --lib-only; REPO_ROOT="$1"; resolve_base HEAD && printf "%s|%s" "$BASE_REF" "$BASE_HOW"' "$SELF" "$R" 2>/dev/null ) || true
    want=$( cd "$R" && git rev-parse 'HEAD^1' )
    case "$got" in "$want|first parent"*) printf 'ok    row %-2s shallow fallback: merge-commit head -> its first parent\n' "$row" ;;
        *) printf 'FAIL  row %-2s shallow fallback: wanted %s|first parent..., got %s\n' "$row" "$want" "$got"; fails=1 ;; esac
    row=$((row + 1))
    rc2=0; err2=$( cd "$R" && git checkout -q feat && ROADMAP_DIFF_FORCE_SHALLOW=1 bash -c '. "$0" --lib-only; REPO_ROOT="$1"; resolve_base HEAD' "$SELF" "$R" 2>&1 >/dev/null ) || rc2=$?
    case "$rc2:$err2" in
        0:*) printf 'FAIL  row %-2s shallow fallback: a non-merge head resolved a base (rc=0)\n' "$row"; fails=1 ;;
        *"is not a merge commit"*) printf 'ok    row %-2s shallow fallback: non-merge head is refused by name, never HEAD\n' "$row" ;;
        *) printf 'FAIL  row %-2s shallow fallback: refused for the wrong reason (rc=%s): %s\n' "$row" "$rc2" "$err2"; fails=1 ;;
    esac

    row=$((row + 1))
    ( cd "$R" && git checkout -q main && cp "$TD/base_dup.yaml" r.yaml && git commit -qam squash-shaped-queue-head ) 2>/dev/null   # main already holds append.yaml; a DIFFERENT file makes the commit real
    got3=$( cd "$R" && ROADMAP_DIFF_FORCE_SHALLOW=1 bash -c '. "$0" --lib-only; REPO_ROOT="$1"; resolve_base HEAD && printf "%s|%s" "$BASE_REF" "$BASE_HOW"' "$SELF" "$R" 2>/dev/null ) || true
    want3=$( cd "$R" && git rev-parse origin/main )
    case "$got3" in "$want3|single parent == origin/main tip"*) printf 'ok    row %-2s shallow fallback: single-parent head on the origin/main tip (merge_group squash head) -> that tip\n' "$row" ;;
        *) printf 'FAIL  row %-2s shallow fallback: wanted %s|single parent == origin/main tip..., got %s\n' "$row" "$want3" "$got3"; fails=1 ;; esac

    if [ "$fails" -ne 0 ]; then
        printf '\nSELF-TEST FAILED (%s/%s rows)\n' "$((row - fails + fails))" "$row"
        exit 1
    fi
    printf '\n%s/%s rows\n' "$row" "$row"
    exit 0
fi

# ---------------------------------------------------------------------------
# the real check
# ---------------------------------------------------------------------------
if ! git -C "$REPO_ROOT" rev-parse --verify -q origin/main >/dev/null; then
    printf '%s: origin/main is not resolvable here (no such remote-tracking ref).\n' "$PROG" >&2
    printf '    This is an environment gap, not a roadmap defect. Fetch it:\n' >&2
    printf '    git -C %s fetch origin main\n' "$REPO_ROOT" >&2
    exit 2
fi

HEAD_REF="${2:-HEAD}"
if [ -n "${1:-}" ]; then
    BASE_REF="$1"; BASE_HOW="argument"
else
    if ! resolve_base "$HEAD_REF"; then exit 2; fi
fi

printf '=== roadmap.yaml diff is additive: base=%s (%s) head=%s (%s) ===\n' "$BASE_REF" "$BASE_HOW" "$HEAD_REF" "$PROG"
if [ "$(git -C "$REPO_ROOT" rev-parse "$BASE_REF^{commit}")" = "$(git -C "$REPO_ROOT" rev-parse "$HEAD_REF^{commit}")" ]; then
    printf 'PASS  base and head are the same commit (a push/workflow_dispatch run on the branch tip): there is no PR diff to judge here; the judgement happened on the pull_request and merge_group runs\n'
    exit 0
fi
if out=$(run_check "$BASE_REF" "$HEAD_REF" 2>&1); then
    printf '%s\n' "$out"
    printf 'PASS\n'
    exit 0
fi
rc=$?
printf '%s\n' "$out"
if [ "$rc" -eq 2 ]; then
    printf '\n%s: usage/read error (see above).\n' "$PROG" >&2
    exit 2
fi
printf '\nPMAT-980 (#2874): a roadmap.yaml diff may only ADD entries or edit a\n'
printf 'ticket''s own lifecycle fields. Run scripts/roadmap_trim.py to collapse a\n'
printf 're-serialisation back to base bytes.\n'
exit 1
