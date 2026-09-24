#!/usr/bin/env bash
# check_census_derived.sh — contracts/census.json is DERIVED, and has one writer (#3569).
#
# THE DEFECT
# ----------
# The census was a tracked file that every pull request regenerated. Makefile
# `contracts:` rewrote it and ran `git diff --exit-code`, so any branch that
# added a contract had to commit a new census. Two branches that each added one
# contract then conflicted on the same `n_files` line, every fold resolved that
# conflict by hand, and a hand resolution is a census that no tool computed.
# The file was an output that behaved like an input.
#
# THE RULE
# --------
#   1. A pull request may not edit contracts/census.json. The release train
#      (T-0) is its only writer: a `release/X.Y.Z` branch, or a run with
#      CENSUS_WRITER=train. Any other diff against the comparand is refused, and
#      the refusal names the file.
#   2. CI computes the census FRESH (`make contracts` → `pv census`) and checks
#      its invariants. It no longer diffs that fresh census against the tracked
#      one, because the tracked one is now a train snapshot and is expected to lag.
#
# The comparand is the one every ratchet here uses (scripts/lib_baseline_ratchet.sh):
# the merge-base with origin/main, or the tip of origin/main. If neither resolves,
# the check FAILS. It is never degraded to comparing the branch against itself.
#
#   bash scripts/check_census_derived.sh                  # touch check + invariants of the tracked census
#   bash scripts/check_census_derived.sh --census FILE    # invariants of FILE only (make contracts: a fresh one)
#   bash scripts/check_census_derived.sh --self-test      # the case table; every RED row must go RED
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CENSUS_PATH="contracts/census.json"
# shellcheck source=scripts/lib_baseline_ratchet.sh
. "$REPO_ROOT/scripts/lib_baseline_ratchet.sh" || exit 2

# ── invariants ───────────────────────────────────────────────────────────────
# Each one is an identity `pv census` (crates/aprender-contracts-cli/src/commands/census.rs)
# holds by construction. A census that breaks one was not written by pv census,
# or was written by a pv census that is broken. Either way it is not a measurement.
invariant_program() { # the jq program: one message per broken identity
    cat <<'JQ'
    def num: type == "number" and . >= 0 and . == floor;
    def total(o): [o[]] | add // 0;
    [
      (if (.schema | type) == "string" and (.schema | length) > 0 then empty else "schema is not a non-empty string" end),
      (if .git_sha == null then empty else "git_sha is \(.git_sha|tojson), not null: a sha is unknowable for the commit that will contain it" end),
      (if (.n_files | num) and (.n_parsed | num) and (.n_parse_errors | num) and (.quarantined_n | num) then empty else "n_files/n_parsed/n_parse_errors/quarantined_n are not all non-negative integers" end),
      (if .n_files > 0 then empty else "n_files is \(.n_files): a zero census is a broken measurement" end),
      (if .n_parsed + .n_parse_errors == .n_files then empty else "n_parsed (\(.n_parsed)) + n_parse_errors (\(.n_parse_errors)) != n_files (\(.n_files)): a file was silently skipped" end),
      (if (.parse_errors | type) == "array" and (.parse_errors | length) == .n_parse_errors then empty else "len(parse_errors) != n_parse_errors (\(.n_parse_errors))" end),
      (if total(.by_kind) == .n_parsed then empty else "sum(by_kind) = \(total(.by_kind)) != n_parsed (\(.n_parsed))" end),
      (if total(.by_anchoring) == .n_parsed then empty else "sum(by_anchoring) = \(total(.by_anchoring)) != n_parsed (\(.n_parsed))" end),
      (if (.by_anchoring | keys | join(",")) == "class,instance,unanchored" then empty else "by_anchoring keys are \(.by_anchoring | keys | tojson), not the three levels of ONT-001 §0.0" end),
      (if total(.by_entity_type) <= (.by_anchoring.class + .by_anchoring.instance) then empty else "sum(by_entity_type) = \(total(.by_entity_type)) exceeds the anchored count (\(.by_anchoring.class + .by_anchoring.instance))" end),
      (if (.id_set_sha256 | type) == "string" and (.id_set_sha256 | test("^[0-9a-f]{64}$")) then empty else "id_set_sha256 is not 64 lowercase hex digits" end)
    ] | .[]
JQ
}

census_invariants() { # census_invariants FILE -> prints one line per violation; rc 1 on any
    local f="$1" out
    [ -s "$f" ] || { printf 'FAIL  %s is missing or empty: the census is UNMEASURED\n' "$f"; return 1; }
    out=$(jq -r "$(invariant_program)" "$f" 2>&1) || { printf 'FAIL  %s is not a census jq can read: %s\n' "$f" "$out"; return 1; }
    [ -z "$out" ] && return 0
    while IFS= read -r l; do printf 'FAIL  %s: %s\n' "$f" "$l"; done <<<"$out"
    return 1
}

# ── the writer exemption ─────────────────────────────────────────────────────
is_train() { # is_train BRANCH -> 0 iff this run is the census's one writer
    [ "${CENSUS_WRITER:-}" = train ] && return 0
    [[ "$1" =~ ^release/[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

current_branch() {
    local b="${GITHUB_HEAD_REF:-}"
    [ -n "$b" ] || b=$(git -C "$1" rev-parse --abbrev-ref HEAD 2>/dev/null) || b=""
    printf '%s' "$b"
}

# ── the touch check ──────────────────────────────────────────────────────────
touch_check() { # touch_check ROOT BRANCH -> 0 not touched (or train); 1 touched or unmeasurable
    local root="$1" branch="$2" res mode ref changed
    res=$(baseline_ratchet_resolve "$root" "$BASELINE_RATCHET_BASE_REF" "$CENSUS_PATH")
    mode=${res%%$'\t'*}; ref=${res##*$'\t'}
    case "$mode" in
        MERGEBASE | TIP) ;;
        *)
            printf 'FAIL  comparand %s resolved %s for %s: whether this branch edits the census is UNMEASURED, and that is not "untouched".\n' "$ref" "$mode" "$CENSUS_PATH"
            printf '      In CI: git fetch --no-tags --depth=1 origin +refs/heads/main:refs/remotes/origin/main\n'
            return 1 ;;
    esac
    # Committed changes AND the working tree, so a local run sees an unstaged edit too.
    changed=$(git -C "$root" diff --name-only "$ref" -- "$CENSUS_PATH") || {
        printf 'FAIL  git diff against %s failed: UNMEASURED\n' "$ref"; return 1; }
    if [ -z "$changed" ]; then
        printf 'ok    %s is untouched against %s (%s)\n' "$CENSUS_PATH" "${ref:0:12}" "$mode"
        return 0
    fi
    if is_train "$branch"; then
        printf 'ok    %s changed on <%s>: the release train is its one writer\n' "$CENSUS_PATH" "$branch"
        return 0
    fi
    printf 'FAIL  this branch <%s> edits %s against %s (%s).\n' "$branch" "$CENSUS_PATH" "${ref:0:12}" "$mode"
    printf '      The census is derived, and the release train (T-0) is its only writer (#3569).\n'
    printf '      CI computes a fresh one (make contracts). Drop the edit:\n'
    printf '          git checkout %s -- %s\n' "${ref:0:12}" "$CENSUS_PATH"
    return 1
}

# ── the case table ───────────────────────────────────────────────────────────
self_test() {
    local t pass=0 fail=0 good rc
    t=$(mktemp -d)
    case "$t" in /tmp/* | /var/tmp/* | "${TMPDIR:-/nonexistent}"/*) : ;; *) printf 'NO-GO: odd mktemp path %s\n' "$t" >&2; return 2 ;; esac
    row() { # row NAME GOT_RC WANT_RC
        if [ "$2" = "$3" ]; then pass=$((pass + 1)); printf '  ok    rc=%s  %s\n' "$2" "$1"
        else fail=$((fail + 1)); printf '  FAIL  rc=%s (wanted %s)  %s\n' "$2" "$3" "$1"; fi
    }
    printf 'check_census_derived self-test\n'

    # -- invariants. A good census, then one corruption per invariant: each must go RED.
    good='{"schema":"s1","git_sha":null,"n_files":5,"n_parsed":4,"n_parse_errors":1,"parse_errors":["x.yaml"],"quarantined_n":0,"by_kind":{"kernel":3,"pattern":1},"by_entity_type":{"k":2},"by_anchoring":{"unanchored":1,"class":2,"instance":1},"id_set_sha256":"'"$(printf '%064d' 0 | tr 0 a)"'","declared_external":[],"timing":{}}'
    printf '%s\n' "$good" > "$t/good.json"
    rc=0; census_invariants "$t/good.json" >/dev/null || rc=$?; row "a consistent census is GREEN" "$rc" 0
    rc=0; census_invariants "$REPO_ROOT/$CENSUS_PATH" >/dev/null || rc=$?; row "the tracked census is GREEN" "$rc" 0
    local name filter
    while IFS='|' read -r name filter; do
        jq -c "$filter" "$t/good.json" > "$t/bad.json"
        rc=0; census_invariants "$t/bad.json" >/dev/null || rc=$?; row "RED: $name" "$rc" 1
    done <<'CASES'
n_files one more than parsed + errors|.n_files += 1
n_parsed one less (a silent skip)|.n_parsed -= 1
parse_errors list disagrees with its count|.parse_errors = []
by_kind sums to one more than n_parsed|.by_kind.kernel += 1
by_anchoring sums to one less|.by_anchoring.class -= 1
by_anchoring lost a level|.by_anchoring |= del(.instance) | .by_anchoring.class += 1
entity types exceed the anchored count|.by_entity_type.k = 4
git_sha stamped|.git_sha = "abc"
id_set_sha256 truncated|.id_set_sha256 = "abc"
schema missing|del(.schema)
zero census|.n_files = 0 | .n_parsed = 0 | .n_parse_errors = 0 | .parse_errors = [] | .by_kind = {} | .by_anchoring = {"unanchored":0,"class":0,"instance":0} | .by_entity_type = {}
negative count|.quarantined_n = -1
CASES
    printf 'not json\n' > "$t/nj.json"
    rc=0; census_invariants "$t/nj.json" >/dev/null || rc=$?; row "RED: not JSON" "$rc" 1
    : > "$t/empty.json"
    rc=0; census_invariants "$t/empty.json" >/dev/null || rc=$?; row "RED: empty file" "$rc" 1

    # -- the touch check, in a scratch repo whose origin/main is a real ref.
    local r="$t/repo"
    mkdir -p "$r/contracts"
    git -C "$r" init -q -b main
    git -C "$r" config user.email t@t; git -C "$r" config user.name t; git -C "$r" config commit.gpgsign false; git -C "$r" config core.hooksPath /dev/null
    printf '%s\n' "$good" > "$r/$CENSUS_PATH"; printf 'a\n' > "$r/contracts/a.yaml"
    git -C "$r" add -A; git -C "$r" commit -qm base
    git -C "$r" update-ref refs/remotes/origin/main HEAD
    git -C "$r" checkout -qb feat
    printf 'b\n' > "$r/contracts/b.yaml"; git -C "$r" add -A; git -C "$r" commit -qm "add a contract, leave the census"
    rc=0; CENSUS_WRITER='' touch_check "$r" feat >/dev/null || rc=$?; row "a PR that adds a contract and leaves the census is GREEN" "$rc" 0
    jq -c '.n_files += 1 | .n_parsed += 1 | .by_kind.kernel += 1 | .by_anchoring.unanchored += 1' "$r/$CENSUS_PATH" > "$t/c"; cp "$t/c" "$r/$CENSUS_PATH"
    rc=0; CENSUS_WRITER='' touch_check "$r" feat > "$t/out" || rc=$?; row "RED: an UNCOMMITTED census edit on a PR" "$rc" 1
    git -C "$r" commit -qam "regenerate the census"
    rc=0; CENSUS_WRITER='' touch_check "$r" feat > "$t/out" || rc=$?; row "RED: a PR diff that edits the census (planted)" "$rc" 1
    grep -q "edits $CENSUS_PATH" "$t/out"; row "the refusal names $CENSUS_PATH" "$?" 0
    rc=0; CENSUS_WRITER='' touch_check "$r" release/0.70.0 >/dev/null || rc=$?; row "the release/X.Y.Z train may write it" "$rc" 0
    rc=0; CENSUS_WRITER=train touch_check "$r" feat >/dev/null || rc=$?; row "CENSUS_WRITER=train may write it" "$rc" 0
    rc=0; CENSUS_WRITER='' touch_check "$r" release/next >/dev/null || rc=$?; row "RED: release/<not a version> is not the train" "$rc" 1
    rc=0; CENSUS_WRITER='' touch_check "$r" fix/release/0.70.0 >/dev/null || rc=$?; row "RED: a branch that merely CONTAINS release/X.Y.Z is not the train" "$rc" 1
    rc=0; CENSUS_WRITER='' touch_check "$r" release/0.70.0-x >/dev/null || rc=$?; row "RED: release/X.Y.Z<suffix> is not the train" "$rc" 1
    rc=0; CENSUS_WRITER='' touch_check "$r" batch/0.70.0 >/dev/null || rc=$?; row "RED: a batch branch is not the train" "$rc" 1
    git -C "$r" update-ref -d refs/remotes/origin/main
    rc=0; CENSUS_WRITER='' touch_check "$r" feat >/dev/null || rc=$?; row "RED: no comparand is UNMEASURED, not untouched" "$rc" 1

    printf 'self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ -n "$t" ] && [ -d "$t" ] && rm -rf -- "$t"
    [ "$fail" -eq 0 ] && [ "$pass" -gt 0 ]
}

main() {
    command -v jq >/dev/null 2>&1 || { printf 'NO-GO: jq is required\n' >&2; return 2; }
    case "${1:-}" in
        --self-test) self_test; return $? ;;
        --census)
            [ -n "${2:-}" ] || { printf 'usage: %s --census FILE\n' "$(basename "$0")" >&2; return 2; }
            printf '== census invariants: %s ==\n' "$2"
            census_invariants "$2" && printf 'PASS  every pv census identity holds\n'
            return $? ;;
        '') ;;
        *) printf 'usage: %s [--census FILE | --self-test]\n' "$(basename "$0")" >&2; return 2 ;;
    esac
    local rc=0
    printf '== census is derived; the train is its one writer (#3569) ==\n'
    touch_check "$REPO_ROOT" "$(current_branch "$REPO_ROOT")" || rc=1
    census_invariants "$REPO_ROOT/$CENSUS_PATH" || rc=1
    [ "$rc" = 0 ] && printf 'PASS\n'
    return "$rc"
}

main "$@"
