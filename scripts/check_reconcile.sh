#!/usr/bin/env bash
# check_reconcile.sh <tag> [--prev-tag T] [--json OUT] [--self-test]
#
# APR-RELEASE-001 section 6, T-5 reconcile. Computes the five predicates and
# fails (exit 1) unless R1..R4 are ALL zero. The autopilot writes DONE only
# after this reads every predicate 0; it must never silently pass.
#
#   R1 fixed-but-open   open issues cited with a closing keyword by a commit
#                       in prev..tag on the current checkout.
#   R2 missing-closing  merged PRs in the window whose body fails
#                       check_pr_closes_issue.sh.
#   R3 dead branches    remote branches with no open PR, tip older than 14d.
#   R4 dirty stale PRs  open PRs mergeStateStatus=DIRTY, created before
#                       --prev-tag's date.
#   R5 ratio            issues opened vs closed in the window (recorded, not
#                       gated).
#
# Exit 0  = R1..R4 all zero.
# Exit 1  = at least one of R1..R4 is nonzero (receipt still written/printed).
# Exit 2  = environment failure: gh/git missing or unauthenticated, tag or
#           prev-tag not resolvable. Never a silent pass.
#
# The predicate logic is factored into functions that read from FILES, so
# --self-test never touches the network: it builds a fixture input directory
# per case and calls the same functions the live path calls.
#
# R1 reads git log as NUL-separated records from a FILE (never piped into the
# parser on the same stdin as the parser's own code) and hands that file to a
# short python3 helper for regex extraction, because bash cannot safely split
# arbitrary commit text on NUL.

set -euo pipefail

SELF_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
SCRIPT_DIR="$(dirname "$SELF_PATH")"
CLOSE_SCRIPT="${SCRIPT_DIR}/check_pr_closes_issue.sh"

usage() {
    printf 'usage: %s <tag> [--prev-tag TAG] [--json FILE] | --self-test\n' "$(basename "$0")" >&2
    exit 2
}

# ---------------------------------------------------------------------------
# predicates (pure functions over files)
# ---------------------------------------------------------------------------

# r1_fixed_but_open LOGFILE ISSUES_OPEN_FILE
# LOGFILE: NUL-separated commit records (sha, subject, body concatenated).
# ISSUES_OPEN_FILE: one open issue number per line (a tab-separated title is
# tolerated, only the first field is read).
# Prints the intersection: issue numbers that are open AND cited with a
# closing keyword by a commit in the window.
r1_fixed_but_open() {
    logfile="$1"
    issues_open_file="$2"

    cited="$(python3 -c '
import re, sys

CLOSE_RE = re.compile(
    r"(close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)"
    r"\s*:?\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)?#([0-9]+)",
    re.IGNORECASE,
)

path = sys.argv[1]
data = open(path, "rb").read()
nums = set()
for record in data.split(b"\x00"):
    if not record.strip():
        continue
    text = record.decode("utf-8", "replace")
    for m in CLOSE_RE.finditer(text):
        nums.add(m.group(3))
for n in sorted(nums, key=int):
    print(n)
' "$logfile")"

    open_nums="$(cut -f1 "$issues_open_file" 2>/dev/null | grep -vE '^[[:space:]]*$' | LC_ALL=C sort -u || true)"
    open_sp=" $(printf '%s' "$open_nums" | tr '\n' ' ') "

    while IFS= read -r num; do
        [ -z "$num" ] && continue
        case "$open_sp" in
            *" $num "*) printf '%s\n' "$num" ;;
        esac
    done <<EOF_CITED
$cited
EOF_CITED
    return 0
}

# r2_missing_closing_ref MERGED_PRS_JSON
# MERGED_PRS_JSON: a gh-shaped JSON array of {"number":N,"body":"..."}.
# Prints the PR numbers whose body fails check_pr_closes_issue.sh.
r2_missing_closing_ref() {
    merged_prs_file="$1"

    workdir="$(mktemp -d)" || return 1
    python3 -c '
import json, os, sys

data = json.load(open(sys.argv[1]))
outdir = sys.argv[2]
for pr in data:
    num = pr.get("number")
    body = pr.get("body") or ""
    with open(os.path.join(outdir, "%s.txt" % num), "w") as fh:
        fh.write(body)
' "$merged_prs_file" "$workdir"

    for bodyfile in "$workdir"/*.txt; do
        [ -e "$bodyfile" ] || continue
        num="$(basename "$bodyfile" .txt)"
        rc=0
        bash "$CLOSE_SCRIPT" --body "$bodyfile" >/dev/null 2>&1 || rc=$?
        if [ "$rc" -eq 1 ]; then
            printf '%s\n' "$num"
        fi
    done | LC_ALL=C sort -n -u

    rm -rf "${workdir:?}"
    return 0
}

# r3_dead_branches BRANCHES_FILE OPEN_PRS_JSON CUTOFF_EPOCH
# BRANCHES_FILE: "<branch>\t<iso8601-tip-date>" per line.
# OPEN_PRS_JSON: a gh-shaped JSON array of {"headRefName": "..."}.
# Prints branches with no open PR whose tip is older than CUTOFF_EPOCH.
r3_dead_branches() {
    branches_file="$1"
    open_prs_file="$2"
    cutoff_epoch="$3"

    open_refs="$(python3 -c '
import json, sys
data = json.load(open(sys.argv[1]))
for pr in data:
    ref = pr.get("headRefName", "")
    if ref:
        print(ref)
' "$open_prs_file")"
    open_sp=" $(printf '%s' "$open_refs" | tr '\n' ' ') "

    while IFS="$(printf '\t')" read -r branch date_str; do
        [ -z "$branch" ] && continue
        case "$open_sp" in
            *" $branch "*) continue ;;
        esac
        epoch=""
        epoch="$(date -u -d "$date_str" +%s 2>/dev/null || true)"
        [ -z "$epoch" ] && continue
        if [ "$epoch" -lt "$cutoff_epoch" ]; then
            printf '%s\n' "$branch"
        fi
    done < "$branches_file"
    return 0
}

# r4_dirty_stale_prs OPEN_PRS_JSON PREV_TAG_EPOCH
# OPEN_PRS_JSON: a gh-shaped JSON array of
# {"number":N,"mergeStateStatus":"DIRTY|...","createdAt":"iso8601"}.
# Prints PR numbers DIRTY and created before PREV_TAG_EPOCH.
r4_dirty_stale_prs() {
    open_prs_file="$1"
    prev_tag_epoch="$2"

    python3 -c '
import datetime
import json
import sys

data = json.load(open(sys.argv[1]))
prev_epoch = int(sys.argv[2])
for pr in data:
    if pr.get("mergeStateStatus") != "DIRTY":
        continue
    created = pr.get("createdAt", "")
    try:
        dt = datetime.datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        continue
    epoch = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
    if epoch < prev_epoch:
        print(pr.get("number"))
' "$open_prs_file" "$prev_tag_epoch"
    return 0
}

# r5_ratio ARRIVAL CLOSURE
r5_ratio() {
    arrival="$1"
    closure="$2"
    if [ "$arrival" -eq 0 ]; then
        printf '0\n'
        return 0
    fi
    python3 -c "print(round($closure / $arrival, 4))"
}

count_lines() {
    text="$1"
    if [ -z "$(printf '%s' "$text" | tr -d '[:space:]')" ]; then
        printf '0\n'
        return 0
    fi
    printf '%s\n' "$text" | grep -vE '^[[:space:]]*$' | wc -l | tr -d ' '
}

# build_json SHA PREV TAG R1_LIST R1C R2_LIST R2C R3_LIST R3C R4_LIST R4C ARRIVAL CLOSURE RATIO
build_json() {
    R1_LIST="$4" R2_LIST="$6" R3_LIST="$8" R4_LIST="${10}" \
    RJ_SHA="$1" RJ_PREV="$2" RJ_TAG="$3" \
    RJ_R1C="$5" RJ_R2C="$7" RJ_R3C="$9" RJ_R4C="${11}" \
    RJ_ARRIVAL="${12}" RJ_CLOSURE="${13}" RJ_RATIO="${14}" \
    python3 -c '
import json
import os


def lst(name):
    raw = os.environ.get(name, "")
    return [x for x in raw.splitlines() if x.strip()]


receipt = {
    "sha": os.environ["RJ_SHA"],
    "window": [os.environ["RJ_PREV"], os.environ["RJ_TAG"]],
    "R1": {"count": int(os.environ["RJ_R1C"]), "list": lst("R1_LIST")},
    "R2": {"count": int(os.environ["RJ_R2C"]), "list": lst("R2_LIST")},
    "R3": {"count": int(os.environ["RJ_R3C"]), "list": lst("R3_LIST")},
    "R4": {"count": int(os.environ["RJ_R4C"]), "list": lst("R4_LIST")},
    "R5": {
        "arrival": int(os.environ["RJ_ARRIVAL"]),
        "closure": int(os.environ["RJ_CLOSURE"]),
        "ratio": float(os.environ["RJ_RATIO"]),
    },
}
print(json.dumps(receipt))
'
}

# reconcile_from_dir DIR SHA TAG PREV_TAG PREV_EPOCH JSON_OUT
# Runs all five predicates against the files in DIR, writes/prints the
# receipt, and returns 0 iff R1..R4 are all zero.
reconcile_from_dir() {
    dir="$1"
    sha="$2"
    tag="$3"
    prev_tag="$4"
    prev_epoch="$5"
    json_out="$6"

    cutoff_epoch="$(date -u -d '-14 days' +%s)"  # bashrs disable-line=DET002

    r1_list="$(r1_fixed_but_open "${dir}/log.bin" "${dir}/issues-open.tsv")"
    r2_list="$(r2_missing_closing_ref "${dir}/merged-prs.json")"
    r3_list="$(r3_dead_branches "${dir}/branches.txt" "${dir}/open-prs.json" "$cutoff_epoch")"
    r4_list="$(r4_dirty_stale_prs "${dir}/open-prs.json" "$prev_epoch")"

    arrival="0"
    closure="0"
    if [ -f "${dir}/issues-window.json" ]; then
        arrival="$(python3 -c 'import json,sys;print(int(json.load(open(sys.argv[1])).get("opened",0)))' "${dir}/issues-window.json")"
        closure="$(python3 -c 'import json,sys;print(int(json.load(open(sys.argv[1])).get("closed",0)))' "${dir}/issues-window.json")"
    fi
    ratio="$(r5_ratio "$arrival" "$closure")"

    r1_count="$(count_lines "$r1_list")"
    r2_count="$(count_lines "$r2_list")"
    r3_count="$(count_lines "$r3_list")"
    r4_count="$(count_lines "$r4_list")"

    receipt="$(build_json "$sha" "$prev_tag" "$tag" "$r1_list" "$r1_count" "$r2_list" "$r2_count" "$r3_list" "$r3_count" "$r4_list" "$r4_count" "$arrival" "$closure" "$ratio")"

    if [ -n "$json_out" ]; then
        printf '%s\n' "$receipt" > "$json_out"
    fi
    printf '%s\n' "$receipt"

    printf 'R1(fixed-but-open)=%s R2(missing-closing-ref)=%s R3(dead-branches)=%s R4(dirty-stale-prs)=%s R5(ratio)=%s\n' \
        "$r1_count" "$r2_count" "$r3_count" "$r4_count" "$ratio" >&2

    if [ "$r1_count" -eq 0 ] && [ "$r2_count" -eq 0 ] && [ "$r3_count" -eq 0 ] && [ "$r4_count" -eq 0 ]; then
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# live input gathering (network; the only part --self-test never runs)
# ---------------------------------------------------------------------------

fetch_live_inputs() {
    tag="$1"
    prev_tag="$2"
    dir="$3"

    gh issue list --state open --limit 1000 --json number \
        | python3 -c 'import json,sys
for i in json.load(sys.stdin):
    print(i["number"])' \
        > "${dir}/issues-open.tsv" || return 1

    git log --format='%H%n%s%n%b%x00' "${prev_tag}..${tag}" > "${dir}/log.bin" || return 1

    prev_date="$(git log -1 --format=%aI "$prev_tag")"
    gh pr list --state merged --search "merged:>=${prev_date%%T*}" --json number,body --limit 500 \
        > "${dir}/merged-prs.json" || return 1

    gh pr list --state open --json number,headRefName,mergeStateStatus,createdAt --limit 500 \
        > "${dir}/open-prs.json" || return 1

    git fetch --quiet origin '+refs/heads/*:refs/remotes/origin/*' --prune || true
    git for-each-ref --format="$(printf '%%(refname:short)\t%%(committerdate:iso-strict)')" refs/remotes/origin/ \
        | sed -E 's#^origin/##' \
        > "${dir}/branches.txt" || true

    opened="$(gh issue list --search "created:>=${prev_date%%T*}" --json number --limit 1000 | python3 -c 'import json,sys;print(len(json.load(sys.stdin)))')"
    closed="$(gh issue list --state closed --search "closed:>=${prev_date%%T*}" --json number --limit 1000 | python3 -c 'import json,sys;print(len(json.load(sys.stdin)))')"
    printf '{"opened":%s,"closed":%s}\n' "$opened" "$closed" > "${dir}/issues-window.json"
    return 0
}

# ---------------------------------------------------------------------------
# self-test: fixture-only, no network. Every predicate gets a positive fixture
# and a twin negative fixture, proving the check is not vacuous.
# ---------------------------------------------------------------------------

self_test() {
    fails=0

    # R1: an open issue #42 fixed by a commit -> R1 = [42]. Twin: same commit,
    # but #42 is not in the open-issues file -> R1 = [].
    fixdir="$(mktemp -d)" || return 2
    printf '42\tSome bug\n' > "${fixdir}/issues-open.tsv"
    printf 'deadbeef\nfix: repair the thing\nFixes #42\n\x00' > "${fixdir}/log.bin"
    got="$(r1_fixed_but_open "${fixdir}/log.bin" "${fixdir}/issues-open.tsv")"
    if [ "$(count_lines "$got")" != "1" ] || [ "$got" != "42" ]; then
        printf 'FAIL R1-positive: expected [42], got [%s]\n' "$got" >&2
        fails=$((fails + 1))
    fi
    printf '99\tUnrelated\n' > "${fixdir}/issues-open-twin.tsv"
    got="$(r1_fixed_but_open "${fixdir}/log.bin" "${fixdir}/issues-open-twin.tsv")"
    if [ "$(count_lines "$got")" != "0" ]; then
        printf 'FAIL R1-twin: expected empty, got [%s]\n' "$got" >&2
        fails=$((fails + 1))
    fi
    rm -rf "${fixdir:?}"

    # R2: a merged PR that cites without closing -> R2 = [10]. Twin: PR body
    # closes properly -> R2 = [].
    fixdir="$(mktemp -d)" || return 2
    printf '[{"number":10,"body":"Refs #5"}]' > "${fixdir}/merged-prs.json"
    got="$(r2_missing_closing_ref "${fixdir}/merged-prs.json")"
    if [ "$got" != "10" ]; then
        printf 'FAIL R2-positive: expected [10], got [%s]\n' "$got" >&2
        fails=$((fails + 1))
    fi
    printf '[{"number":10,"body":"Closes #5"}]' > "${fixdir}/merged-prs-twin.json"
    got="$(r2_missing_closing_ref "${fixdir}/merged-prs-twin.json")"
    if [ "$(count_lines "$got")" != "0" ]; then
        printf 'FAIL R2-twin: expected empty, got [%s]\n' "$got" >&2
        fails=$((fails + 1))
    fi
    rm -rf "${fixdir:?}"

    # R3: a branch with no open PR, tip 20 days old -> R3 = [stale-branch].
    # Twin A: tip 5 days old -> excluded. Twin B: 20 days old but has an open
    # PR -> excluded.
    fixdir="$(mktemp -d)" || return 2
    stale_date="$(date -u -d '-20 days' +%Y-%m-%dT%H:%M:%SZ)"  # bashrs disable-line=DET002
    fresh_date="$(date -u -d '-5 days' +%Y-%m-%dT%H:%M:%SZ)"  # bashrs disable-line=DET002
    cutoff_epoch="$(date -u -d '-14 days' +%s)"
    printf 'stale-branch\t%s\n' "$stale_date" > "${fixdir}/branches.txt"
    printf '[]' > "${fixdir}/open-prs-empty.json"
    got="$(r3_dead_branches "${fixdir}/branches.txt" "${fixdir}/open-prs-empty.json" "$cutoff_epoch")"
    if [ "$got" != "stale-branch" ]; then
        printf 'FAIL R3-positive: expected [stale-branch], got [%s]\n' "$got" >&2
        fails=$((fails + 1))
    fi
    printf 'fresh-branch\t%s\n' "$fresh_date" > "${fixdir}/branches-fresh.txt"
    got="$(r3_dead_branches "${fixdir}/branches-fresh.txt" "${fixdir}/open-prs-empty.json" "$cutoff_epoch")"
    if [ "$(count_lines "$got")" != "0" ]; then
        printf 'FAIL R3-twin-fresh: expected empty, got [%s]\n' "$got" >&2
        fails=$((fails + 1))
    fi
    printf '[{"headRefName":"stale-branch"}]' > "${fixdir}/open-prs-covers.json"
    got="$(r3_dead_branches "${fixdir}/branches.txt" "${fixdir}/open-prs-covers.json" "$cutoff_epoch")"
    if [ "$(count_lines "$got")" != "0" ]; then
        printf 'FAIL R3-twin-has-pr: expected empty, got [%s]\n' "$got" >&2
        fails=$((fails + 1))
    fi
    rm -rf "${fixdir:?}"

    # R4: a DIRTY PR created before prev-tag's date -> R4 = [7]. Twin A: DIRTY
    # but created after prev-tag -> excluded. Twin B: CLEAN, same old date ->
    # excluded.
    fixdir="$(mktemp -d)" || return 2
    old_created="$(date -u -d '-25 days' +%Y-%m-%dT%H:%M:%SZ)"  # bashrs disable-line=DET002
    new_created="$(date -u -d '-5 days' +%Y-%m-%dT%H:%M:%SZ)"  # bashrs disable-line=DET002
    prev_epoch="$(date -u -d '-14 days' +%s)"
    printf '[{"number":7,"mergeStateStatus":"DIRTY","createdAt":"%s"}]' "$old_created" > "${fixdir}/open-prs.json"
    got="$(r4_dirty_stale_prs "${fixdir}/open-prs.json" "$prev_epoch")"
    if [ "$got" != "7" ]; then
        printf 'FAIL R4-positive: expected [7], got [%s]\n' "$got" >&2
        fails=$((fails + 1))
    fi
    printf '[{"number":7,"mergeStateStatus":"DIRTY","createdAt":"%s"}]' "$new_created" > "${fixdir}/open-prs-new.json"
    got="$(r4_dirty_stale_prs "${fixdir}/open-prs-new.json" "$prev_epoch")"
    if [ "$(count_lines "$got")" != "0" ]; then
        printf 'FAIL R4-twin-new: expected empty, got [%s]\n' "$got" >&2
        fails=$((fails + 1))
    fi
    printf '[{"number":7,"mergeStateStatus":"CLEAN","createdAt":"%s"}]' "$old_created" > "${fixdir}/open-prs-clean.json"
    got="$(r4_dirty_stale_prs "${fixdir}/open-prs-clean.json" "$prev_epoch")"
    if [ "$(count_lines "$got")" != "0" ]; then
        printf 'FAIL R4-twin-clean: expected empty, got [%s]\n' "$got" >&2
        fails=$((fails + 1))
    fi
    rm -rf "${fixdir:?}"

    # R5: ratio arithmetic, including the zero-arrival guard.
    got="$(r5_ratio 10 5)"
    if [ "$got" != "0.5" ]; then
        printf 'FAIL R5-ratio: expected 0.5, got [%s]\n' "$got" >&2
        fails=$((fails + 1))
    fi
    got="$(r5_ratio 0 0)"
    if [ "$got" != "0" ]; then
        printf 'FAIL R5-zero-arrival: expected 0, got [%s]\n' "$got" >&2
        fails=$((fails + 1))
    fi

    # End-to-end aggregation via reconcile_from_dir: an all-clean input dir
    # must exit 0 and write a receipt with every R1..R4 count 0; seeding one
    # fixed-but-open issue (the falsifier APR-RELEASE-001 section 6 names
    # explicitly) must turn it RED with exit 1 and R1.count = 1. This is the
    # mutation proof: a neutered predicate that always returns empty passes
    # both, which the RED assertion below catches.
    fixdir="$(mktemp -d)" || return 2
    printf '' > "${fixdir}/issues-open.tsv"
    printf 'deadbeef\nchore: bump deps\n\x00' > "${fixdir}/log.bin"
    printf '[]' > "${fixdir}/merged-prs.json"
    printf '' > "${fixdir}/branches.txt"
    printf '[]' > "${fixdir}/open-prs.json"
    printf '{"opened":2,"closed":1}\n' > "${fixdir}/issues-window.json"
    jsonout="$(mktemp)"
    rc=0
    reconcile_from_dir "$fixdir" "abc123" "v0.1.1" "v0.1.0" "$(date -u -d '-14 days' +%s)" "$jsonout" >/dev/null 2>/dev/null || rc=$?  # bashrs disable-line=DET002
    if [ "$rc" -ne 0 ]; then
        printf 'FAIL aggregate-clean: expected exit 0, got %s\n' "$rc" >&2
        fails=$((fails + 1))
    fi
    r1c="$(python3 -c 'import json;print(json.load(open("'"$jsonout"'"))["R1"]["count"])')"
    if [ "$r1c" != "0" ]; then
        printf 'FAIL aggregate-clean: expected R1.count=0, got %s\n' "$r1c" >&2
        fails=$((fails + 1))
    fi
    rm -f "$jsonout"

    printf '77\tReopened by mistake\n' > "${fixdir}/issues-open.tsv"
    printf 'cafef00d\nfix: actually done\nFixes #77\n\x00' > "${fixdir}/log.bin"
    jsonout="$(mktemp)"
    rc=0
    reconcile_from_dir "$fixdir" "abc123" "v0.1.1" "v0.1.0" "$(date -u -d '-14 days' +%s)" "$jsonout" >/dev/null 2>/dev/null || rc=$?  # bashrs disable-line=DET002
    if [ "$rc" -ne 1 ]; then
        printf 'FAIL aggregate-seeded: expected exit 1, got %s\n' "$rc" >&2
        fails=$((fails + 1))
    fi
    r1c="$(python3 -c 'import json;print(json.load(open("'"$jsonout"'"))["R1"]["count"])')"
    if [ "$r1c" != "1" ]; then
        printf 'FAIL aggregate-seeded: expected R1.count=1, got %s\n' "$r1c" >&2
        fails=$((fails + 1))
    fi
    rm -f "$jsonout"
    rm -rf "${fixdir:?}"

    # Env-failure path: a nonexistent tag must be exit 2, not a silent pass.
    # This is a local git rev-parse only; no network is involved.
    rc=0
    bash "$SELF_PATH" "does-not-exist-xyz-tag" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne 2 ]; then
        printf 'FAIL env-failure: expected exit 2 on missing tag, got %s\n' "$rc" >&2
        fails=$((fails + 1))
    fi

    if [ "$fails" -ne 0 ]; then
        printf 'self-test FAILED: %s case(s).\n' "$fails" >&2
        return 1
    fi
    printf 'self-test OK: 17 case(s).\n'
    return 0
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

main() {
    # No arguments (guard_tree.sh runs every cargo-free guard bare): the self-test IS the bare run.
    if [ $# -eq 0 ] || [ "${1:-}" = "--self-test" ]; then
        self_test
        exit $?
    fi

    tag="${1:-}"
    if [ -z "$tag" ]; then
        usage
    fi
    shift || true

    prev_tag=""
    json_out=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --prev-tag)
                prev_tag="${2:-}"
                shift 2
                ;;
            --json)
                json_out="${2:-}"
                shift 2
                ;;
            *)
                usage
                ;;
        esac
    done

    if ! command -v gh >/dev/null 2>&1; then
        printf 'ENV: gh is not on PATH.\n' >&2
        exit 2
    fi
    if ! command -v git >/dev/null 2>&1; then
        printf 'ENV: git is not on PATH.\n' >&2
        exit 2
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        printf 'ENV: python3 is not on PATH.\n' >&2
        exit 2
    fi

    if ! git rev-parse -q --verify "refs/tags/${tag}" >/dev/null 2>&1; then
        printf 'ENV: tag not found: %s\n' "$tag" >&2
        exit 2
    fi

    if [ -z "$prev_tag" ]; then
        prev_tag="$(git describe --tags --abbrev=0 "${tag}^" 2>/dev/null || true)"
        if [ -z "$prev_tag" ]; then
            printf 'ENV: could not derive --prev-tag from %s^; pass it explicitly.\n' "$tag" >&2
            exit 2
        fi
    fi

    if ! git rev-parse -q --verify "refs/tags/${prev_tag}" >/dev/null 2>&1; then
        printf 'ENV: prev-tag not found: %s\n' "$prev_tag" >&2
        exit 2
    fi

    if ! gh auth status >/dev/null 2>&1; then
        printf 'ENV: gh is not authenticated.\n' >&2
        exit 2
    fi

    input_dir="$(mktemp -d)" || exit 2
    trap 'rm -rf "${input_dir:?}"' EXIT

    if ! fetch_live_inputs "$tag" "$prev_tag" "$input_dir"; then
        printf 'ENV: failed to gather inputs from git/gh.\n' >&2
        exit 2
    fi

    sha="$(git rev-parse "$tag")"
    prev_epoch="$(git log -1 --format=%at "$prev_tag")"

    rc=0
    reconcile_from_dir "$input_dir" "$sha" "$tag" "$prev_tag" "$prev_epoch" "$json_out" || rc=$?
    exit "$rc"
}

main "$@"
