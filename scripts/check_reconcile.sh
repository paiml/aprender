#!/usr/bin/env bash
# check_reconcile.sh <tag> [--prev-tag T] [--rev SHA] [--json OUT] [--archive-dir DIR] [--self-test]
#
# APR-RELEASE-001 section 6, T-5 reconcile. Computes the five predicates and
# fails (exit 1) unless R1, R2, and R4 are ALL zero. R3 (dead branches) is
# non-blocking: dead branches are archived (git bundle + tips.txt + README.md),
# verified readable, and reported on stdout (exit 0 for R3 alone); branches are
# never deleted by this script.
#
#   R1 fixed-but-open   open issues cited with a closing keyword by a commit
#                       in prev..tag on the current checkout.
#   R2 missing-closing  merged PRs in the window whose body fails
#                       check_pr_closes_issue.sh.
#   R3 dead branches    remote branches with no open PR, tip older than 14d.
#                       release/** and single-level nightly/<name> heads (the paths the repository
#                       rulesets lock against deletion) are exempt.
#                       Non-blocking: archived to --archive-dir / R3_ARCHIVE_DIR
#                       (default ${RUNNER_TEMP:-${TMPDIR:-/tmp}}/r3-branch-archive-<tag>)
#                       and reported on stdout. Never deleted here.
#   R4 dirty stale PRs  open PRs mergeStateStatus=DIRTY, created before
#                       --prev-tag's date.
#   R5 ratio            issues opened vs closed in the window (recorded, not
#                       gated).
#
# --rev SHA: judge the window prev..SHA when <tag> does not exist yet (release.yml bump path,
#            before the tag is made). If <tag> exists it must point at SHA, else exit 2.
# --archive-dir DIR (or env R3_ARCHIVE_DIR): directory to write R3 git bundle,
#            tips.txt, and README.md when R3 > 0. Fails closed (exit 2) if the
#            archive cannot be written or read back.
#
# Exit 0  = R1, R2, R4 all zero (R3 archived + verified if nonzero).
# Exit 1  = at least one of R1, R2, R4 is nonzero (receipt still written/printed).
# Exit 2  = environment/archive failure: gh/git missing or unauthenticated, tag or
#           prev-tag not resolvable, or R3 archive write/read-back failed. Never a silent pass.
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
    printf 'usage: %s <tag> [--prev-tag TAG] [--rev SHA] [--json FILE] [--archive-dir DIR] | --self-test\n' "$(basename "$0")" >&2
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
' "$logfile")" || return 2

    # grep -v exits 1 on an empty list (no open issue): that is data, not a failed read; any other status is a failure.
    open_nums="$(cut -f1 "$issues_open_file" | { grep -vE '^[[:space:]]*$' || [ "$?" -eq 1 ]; } | LC_ALL=C sort -u)"
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
' "$merged_prs_file" "$workdir" || { rm -rf "${workdir:?}"; return 2; }

    failed=""
    for bodyfile in "$workdir"/*.txt; do
        [ -e "$bodyfile" ] || continue
        num="$(basename "$bodyfile" .txt)"
        rc=0
        bash "$CLOSE_SCRIPT" --body "$bodyfile" >/dev/null 2>&1 || rc=$?
        if [ "$rc" -eq 1 ]; then
            failed="${failed}${num}
"
        elif [ "$rc" -ne 0 ]; then
            rm -rf "${workdir:?}"
            return 2   # the body check itself errored: not a pass
        fi
    done
    rm -rf "${workdir:?}"
    printf '%s' "$failed" | grep -v '^$' | LC_ALL=C sort -n -u || [ "$?" -eq 1 ]
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
' "$open_prs_file")" || return 2
    open_sp=" $(printf '%s' "$open_refs" | tr '\n' ' ') "

    while IFS="$(printf '\t')" read -r branch date_str; do
        [ -z "$branch" ] && continue
        case "$open_sp" in
            *" $branch "*) continue ;;
        esac
        # R3B: release/** and single-level nightly/<name> heads cannot be deleted (deletion refused by the
        # repository rulesets release-locked, refs/heads/release/**, and nightly-locked, refs/heads/nightly/*
        # one level), so no archive-then-delete can clear them: exempt, never counted. A nested
        # nightly/a/b is NOT locked by nightly-locked, so it falls through and is judged like any branch.
        case "$branch" in
            nightly/*/*) ;;
            release/*|nightly/*) continue ;;
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
' "$open_prs_file" "$prev_tag_epoch" || return 2
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

# r3_verify_archive BUNDLE_FILE TIPS_FILE README_FILE EXPECTED_TIPS_FILE
# Verifies that the bundle, tips.txt, and README.md exist, are readable, pass
# `git bundle verify` inside a clean bare repo, and match all expected tips.
r3_verify_archive() {
    local bundle_file="$1" tips_file="$2" readme_file="$3" expected_tips_file="$4"
    local abs_bundle verify_repo heads_out

    [ -f "$bundle_file" ] && [ -r "$bundle_file" ] && [ -s "$bundle_file" ] || {
        printf 'ENV: r3 archive bundle missing or unreadable: %s\n' "$bundle_file" >&2
        return 2
    }
    [ -f "$tips_file" ] && [ -r "$tips_file" ] && [ -s "$tips_file" ] || {
        printf 'ENV: r3 archive tips.txt missing or unreadable: %s\n' "$tips_file" >&2
        return 2
    }
    [ -f "$readme_file" ] && [ -r "$readme_file" ] && [ -s "$readme_file" ] || {
        printf 'ENV: r3 archive README.md missing or unreadable: %s\n' "$readme_file" >&2
        return 2
    }

    abs_bundle="$(cd "$(dirname "$bundle_file")" && pwd)/$(basename "$bundle_file")" || return 2
    verify_repo="$(mktemp -d -p "${RUNNER_TEMP:-${TMPDIR:-/tmp}}")" || return 2
    if ! git init -q --bare --template= "$verify_repo" >/dev/null 2>&1; then
        rm -rf "${verify_repo:?}"
        return 2
    fi
    if ! git -C "$verify_repo" bundle verify "$abs_bundle" >/dev/null 2>&1; then
        rm -rf "${verify_repo:?}"
        printf 'ENV: r3 archive bundle failed git bundle verify: %s\n' "$bundle_file" >&2
        return 2
    fi
    heads_out="$(git -C "$verify_repo" bundle list-heads "$abs_bundle" 2>/dev/null)" || {
        rm -rf "${verify_repo:?}"
        printf 'ENV: r3 archive bundle list-heads failed: %s\n' "$bundle_file" >&2
        return 2
    }
    rm -rf "${verify_repo:?}"

    R3_HEADS="$heads_out" R3_BUNDLE_NAME="$(basename "$bundle_file")" \
    python3 - "$expected_tips_file" "$tips_file" "$readme_file" <<'PY_VERIFY' || return 2
import os, sys

expected_path, tips_path, readme_path = sys.argv[1:4]
expected = []
with open(expected_path, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) != 2:
                sys.stderr.write(f"ENV: malformed expected tip: {line}\n")
                sys.exit(2)
            expected.append((parts[0], parts[1]))

if not expected:
    sys.stderr.write("ENV: expected tips list is empty\n")
    sys.exit(2)

with open(tips_path, "r", encoding="utf-8") as fh:
    tips_lines = [l.strip() for l in fh if l.strip()]

tips_map = {}
for line in tips_lines:
    parts = line.split()
    if len(parts) != 2:
        sys.stderr.write(f"ENV: malformed tips.txt line: {line}\n")
        sys.exit(2)
    tips_map[parts[0]] = parts[1]

bundle_heads = {}
for line in os.environ.get("R3_HEADS", "").splitlines():
    parts = line.strip().split()
    if len(parts) == 2:
        bundle_heads[parts[1]] = parts[0]

readme_text = open(readme_path, "r", encoding="utf-8").read()
bundle_name = os.environ["R3_BUNDLE_NAME"]
if bundle_name not in readme_text or "git fetch" not in readme_text or "git push origin" not in readme_text:
    sys.stderr.write("ENV: README.md missing restore instructions\n")
    sys.exit(2)

for branch, sha in expected:
    if tips_map.get(branch) != sha:
        sys.stderr.write(f"ENV: tips.txt mismatch for {branch}: got {tips_map.get(branch)} want {sha}\n")
        sys.exit(2)
    ref = f"refs/remotes/origin/{branch}"
    if bundle_heads.get(ref) != sha:
        sys.stderr.write(f"ENV: bundle head mismatch for {ref}: got {bundle_heads.get(ref)} want {sha}\n")
        sys.exit(2)
    if branch not in readme_text or sha not in readme_text:
        sys.stderr.write(f"ENV: README.md missing branch/sha for {branch}\n")
        sys.exit(2)
PY_VERIFY
    return 0
}

# r3_archive_dead_branches R3_LIST ARCHIVE_DIR REPO_DIR TAG
# Archives every dead branch in R3_LIST as a git bundle in ARCHIVE_DIR with
# tips.txt and README.md, verifies read-back, and prints the bundle path.
# Never deletes any branch. Fails closed (returns 2) on any write/verify error.
r3_archive_dead_branches() {
    local r3_list="$1" archive_dir="$2" repo_dir="$3" tag="$4"
    local safe_tag bundle_name bundle_file tips_file readme_file expected_tips_file
    local branch sha ref count=0 now_utc
    local -a refs=()

    [ -n "$archive_dir" ] || { printf 'ENV: R3 archive directory is empty.\n' >&2; return 2; }
    git -C "$repo_dir" rev-parse --git-dir >/dev/null 2>&1 || {
        printf 'ENV: not a git repository for R3 archiving: %s\n' "$repo_dir" >&2
        return 2
    }
    mkdir -p "$archive_dir" 2>/dev/null || {
        printf 'ENV: cannot create R3 archive directory: %s\n' "$archive_dir" >&2
        return 2
    }

    expected_tips_file="$(mktemp -p "${RUNNER_TEMP:-${TMPDIR:-/tmp}}")" || return 2
    while IFS= read -r branch; do
        [ -z "$branch" ] && continue
        ref="refs/remotes/origin/${branch}"
        if ! git -C "$repo_dir" show-ref --verify --quiet "$ref"; then
            rm -f "${expected_tips_file:?}"
            printf 'ENV: dead branch ref not found in git repo: %s\n' "$ref" >&2
            return 2
        fi
        sha="$(git -C "$repo_dir" rev-parse --verify "${ref}^{commit}" 2>/dev/null || true)"
        if [ -z "$sha" ]; then
            rm -f "${expected_tips_file:?}"
            printf 'ENV: cannot resolve dead branch commit: %s\n' "$ref" >&2
            return 2
        fi
        refs+=("$ref")
        printf '%s %s\n' "$branch" "$sha" >> "$expected_tips_file" || {
            rm -f "${expected_tips_file:?}"
            return 2
        }
        count=$((count + 1))
    done <<EOF_R3_LIST
$r3_list
EOF_R3_LIST

    if [ "$count" -eq 0 ]; then
        rm -f "${expected_tips_file:?}"
        return 0
    fi

    safe_tag="$(printf '%s' "${tag:-unknown}" | tr '/[:space:]' '__')"
    bundle_name="r3-${safe_tag}-branches.bundle"
    bundle_file="${archive_dir}/${bundle_name}"
    tips_file="${archive_dir}/tips.txt"
    readme_file="${archive_dir}/README.md"

    rm -f "$bundle_file" 2>/dev/null || true
    if ! git -C "$repo_dir" bundle create "$bundle_file" "${refs[@]}" >/dev/null 2>&1; then
        rm -f "${expected_tips_file:?}" "$bundle_file" 2>/dev/null || true
        printf 'ENV: git bundle create failed at %s\n' "$bundle_file" >&2
        return 2
    fi

    if ! cp "$expected_tips_file" "$tips_file" 2>/dev/null; then
        rm -f "${expected_tips_file:?}"
        printf 'ENV: failed to write %s\n' "$tips_file" >&2
        return 2
    fi

    now_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"  # bashrs disable-line=DET002
    if ! {
        printf '# R3 dead-branch archive for %s (%s)\n\n' "$tag" "$now_utc"
        printf 'Bundle: `%s` (%s head(s)). Tips recorded in `tips.txt`.\n' "$bundle_name" "$count"
        printf 'Policy: stale remote branches (no open PR, tip older than 14 days) are archived and reported; branches are never deleted by check_reconcile.sh.\n\n'
        printf '## Restore\n\n'
        printf 'General: `git fetch <bundle> <ref>; git push origin <sha>:refs/heads/<name>`\n\n'
        printf '```sh\n'
        while IFS=' ' read -r branch sha; do
            [ -z "$branch" ] && continue
            printf 'git fetch %s refs/remotes/origin/%s && git push origin %s:refs/heads/%s\n' \
                "$bundle_name" "$branch" "$sha" "$branch"
        done < "$expected_tips_file"
        printf '```\n'
    } > "$readme_file" 2>/dev/null; then
        rm -f "${expected_tips_file:?}"
        printf 'ENV: failed to write %s\n' "$readme_file" >&2
        return 2
    fi

    if ! r3_verify_archive "$bundle_file" "$tips_file" "$readme_file" "$expected_tips_file"; then
        rm -f "${expected_tips_file:?}"
        return 2
    fi
    rm -f "${expected_tips_file:?}"
    printf '%s\n' "$bundle_file"
    return 0
}

# build_json SHA PREV TAG R1_LIST R1C R2_LIST R2C R3_LIST R3C R4_LIST R4C ARRIVAL CLOSURE RATIO [R3_ARCHIVED] [R3_ARCHIVE_DIR] [R3_BUNDLE]
build_json() {
    R1_LIST="$4" R2_LIST="$6" R3_LIST="$8" R4_LIST="${10}" \
    RJ_SHA="$1" RJ_PREV="$2" RJ_TAG="$3" \
    RJ_R1C="$5" RJ_R2C="$7" RJ_R3C="$9" RJ_R4C="${11}" \
    RJ_ARRIVAL="${12}" RJ_CLOSURE="${13}" RJ_RATIO="${14}" \
    RJ_R3_ARCHIVED="${15:-0}" RJ_R3_ARCHIVE_DIR="${16:-}" RJ_R3_BUNDLE="${17:-}" \
    python3 -c '
import json
import os


def lst(name):
    raw = os.environ.get(name, "")
    return [x for x in raw.splitlines() if x.strip()]


r3_obj = {"count": int(os.environ["RJ_R3C"]), "list": lst("R3_LIST")}
if os.environ.get("RJ_R3_ARCHIVED") == "1":
    r3_obj["archived"] = True
    r3_obj["archive_dir"] = os.environ.get("RJ_R3_ARCHIVE_DIR", "")
    r3_obj["bundle"] = os.environ.get("RJ_R3_BUNDLE", "")

receipt = {
    "sha": os.environ["RJ_SHA"],
    "window": [os.environ["RJ_PREV"], os.environ["RJ_TAG"]],
    "R1": {"count": int(os.environ["RJ_R1C"]), "list": lst("R1_LIST")},
    "R2": {"count": int(os.environ["RJ_R2C"]), "list": lst("R2_LIST")},
    "R3": r3_obj,
    "R4": {"count": int(os.environ["RJ_R4C"]), "list": lst("R4_LIST")},
    "R5": {
        "arrival": int(os.environ["RJ_ARRIVAL"]),
        "closure": int(os.environ["RJ_CLOSURE"]),
        "ratio": float(os.environ["RJ_RATIO"]),
    },
}
print(json.dumps(receipt))
'
}   # callers add || return 2

# reconcile_from_dir DIR SHA TAG PREV_TAG PREV_EPOCH JSON_OUT [ARCHIVE_DIR] [REPO_DIR]
# Runs all five predicates against the files in DIR, archives any R3 dead
# branches into ARCHIVE_DIR and verifies read-back, writes/prints the receipt
# plus stdout REPORT lines for R3, and returns 0 iff R1, R2, and R4 are all
# zero, 1 if any of R1, R2, R4 is nonzero, 2 if a predicate, R3 archive
# write/read-back, or receipt could not be computed (a crash is never a zero).
reconcile_from_dir() {
    dir="$1"
    sha="$2"
    tag="$3"
    prev_tag="$4"
    prev_epoch="$5"
    json_out="$6"
    archive_dir="${7:-}"
    repo_dir="${8:-.}"

    if [ -z "$archive_dir" ]; then
        if [ -n "${R3_ARCHIVE_DIR+x}" ]; then
            if [ -z "$R3_ARCHIVE_DIR" ]; then
                printf 'ENV: R3_ARCHIVE_DIR is set but empty.\n' >&2
                return 2
            fi
            archive_dir="$R3_ARCHIVE_DIR"
        else
            safe_tag="$(printf '%s' "${tag:-unknown}" | tr '/[:space:]' '__')"
            archive_dir="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/r3-branch-archive-${safe_tag}"
        fi
    fi

    cutoff_epoch="$(date -u -d '-14 days' +%s)"  # bashrs disable-line=DET002

    r1_list="$(r1_fixed_but_open "${dir}/log.bin" "${dir}/issues-open.tsv")" || return 2
    r2_list="$(r2_missing_closing_ref "${dir}/merged-prs.json")" || return 2
    r3_list="$(r3_dead_branches "${dir}/branches.txt" "${dir}/open-prs.json" "$cutoff_epoch")" || return 2
    r4_list="$(r4_dirty_stale_prs "${dir}/open-prs.json" "$prev_epoch")" || return 2

    arrival="0"
    closure="0"
    if [ -f "${dir}/issues-window.json" ]; then
        arrival="$(python3 -c 'import json,sys;print(int(json.load(open(sys.argv[1])).get("opened",0)))' "${dir}/issues-window.json")" || return 2
        closure="$(python3 -c 'import json,sys;print(int(json.load(open(sys.argv[1])).get("closed",0)))' "${dir}/issues-window.json")" || return 2
    fi
    ratio="$(r5_ratio "$arrival" "$closure")" || return 2

    r1_count="$(count_lines "$r1_list")"
    r2_count="$(count_lines "$r2_list")"
    r3_count="$(count_lines "$r3_list")"
    r4_count="$(count_lines "$r4_list")"

    r3_archived="0"
    r3_bundle=""
    if [ "$r3_count" -gt 0 ]; then
        r3_bundle="$(r3_archive_dead_branches "$r3_list" "$archive_dir" "$repo_dir" "$tag")" || return 2
        r3_archived="1"
    fi

    receipt="$(build_json "$sha" "$prev_tag" "$tag" "$r1_list" "$r1_count" "$r2_list" "$r2_count" "$r3_list" "$r3_count" "$r4_list" "$r4_count" "$arrival" "$closure" "$ratio" "$r3_archived" "$archive_dir" "$r3_bundle")" || return 2

    if [ -n "$json_out" ]; then
        printf '%s\n' "$receipt" > "$json_out"
    fi
    printf '%s\n' "$receipt"

    if [ "$r3_count" -gt 0 ]; then
        printf 'REPORT R3(dead-branches)=%s non-blocking archived=%s bundle=%s\n' \
            "$r3_count" "$archive_dir" "$r3_bundle"
        while IFS= read -r _r3_tip_line; do
            [ -z "$_r3_tip_line" ] && continue
            printf 'REPORT R3 dead-branch: %s\n' "$_r3_tip_line"
        done < "${archive_dir}/tips.txt"
    fi

    printf 'R1(fixed-but-open)=%s R2(missing-closing-ref)=%s R3(dead-branches)=%s R4(dirty-stale-prs)=%s R5(ratio)=%s\n' \
        "$r1_count" "$r2_count" "$r3_count" "$r4_count" "$ratio" >&2

    if [ "$r1_count" -eq 0 ] && [ "$r2_count" -eq 0 ] && [ "$r4_count" -eq 0 ]; then
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# live input gathering (network; the only part --self-test never runs)
# ---------------------------------------------------------------------------

fetch_live_inputs() {
    local tag="$1" prev_tag="$2" dir="$3" prev_date opened closed

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

    git fetch --quiet origin '+refs/heads/*:refs/remotes/origin/*' --prune || return 1
    git for-each-ref --format="$(printf '%%(refname:short)\t%%(committerdate:iso-strict)')" refs/remotes/origin/ \
        | sed -E 's#^origin/##' \
        > "${dir}/branches.txt" || return 1

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

    # PIN THE REFERENCE CLASSIFIER, or this table stops being a table.
    #
    # R2 below delegates to check_pr_closes_issue.sh, which now resolves each
    # cited "#N" and drops the ones that are pull requests or already-closed
    # issues (a PR you build on promises nothing; an issue already closed cannot
    # be left open forever). That made this file's fixtures LIVE: `Refs #5` is a
    # real, closed issue in this repository, so R2-positive expected [10] and
    # got [] — the fixture number had quietly become an API call.
    #
    # Pinned to a stub instead: every number in these fixtures is an OPEN issue,
    # which is what they were always written to mean. No network, no token,
    # deterministic. It is exported because r2_missing_closing_ref runs the
    # close script as a child process.
    _rc_stub="$(mktemp -d)/kindstub.sh" || return 2
    mkdir -p "$(dirname "$_rc_stub")"
    printf '#!/usr/bin/env bash\nprintf issue\n' > "$_rc_stub"
    chmod +x "$_rc_stub"
    PR_CLOSES_REF_KIND_CMD="$_rc_stub"
    export PR_CLOSES_REF_KIND_CMD

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
    # R2: the body check itself errors (rc 2) -> R2 returns 2, never a pass.
    _close_saved="$CLOSE_SCRIPT"
    printf '#!/usr/bin/env bash\nexit 2\n' > "${fixdir}/close-rc2.sh"
    CLOSE_SCRIPT="${fixdir}/close-rc2.sh"
    # r2_missing_closing_ref sets the global rc itself, so its status is read into its own variable.
    r2_status=0; r2_missing_closing_ref "${fixdir}/merged-prs-twin.json" >/dev/null || r2_status=$?
    CLOSE_SCRIPT="$_close_saved"
    if [ "$r2_status" -ne 2 ]; then
        printf 'FAIL R2-closes-rc2: expected exit 2, got %s\n' "$r2_status" >&2
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
    printf 'release/0.70\t%s\nnightly/probe\t%s\n' "$stale_date" "$stale_date" > "${fixdir}/branches-locked.txt"
    got="$(r3_dead_branches "${fixdir}/branches-locked.txt" "${fixdir}/open-prs-empty.json" "$cutoff_epoch")"
    if [ "$(count_lines "$got")" != "0" ]; then
        printf 'FAIL R3-twin-locked: expected empty, got [%s]\n' "$got" >&2
        fails=$((fails + 1))
    fi
    printf 'releasex/stale\t%s\nfix/nightly/stale\t%s\nnightly/a/b\t%s\n' "$stale_date" "$stale_date" "$stale_date" > "${fixdir}/branches-lookalike.txt"
    got="$(r3_dead_branches "${fixdir}/branches-lookalike.txt" "${fixdir}/open-prs-empty.json" "$cutoff_epoch")"
    if [ "$(count_lines "$got")" != "3" ]; then
        printf 'FAIL R3-twin-lookalike: expected 3 listed, got [%s]\n' "$got" >&2
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

    # N3: a predicate that cannot read its input is exit 2, never a zero count.
    fixdir="$(mktemp -d)"
    : > "${fixdir}/log.bin"; : > "${fixdir}/issues-open.tsv"; : > "${fixdir}/branches.txt"
    printf '{not json' > "${fixdir}/merged-prs.json"; printf '[]' > "${fixdir}/open-prs.json"
    rc=0
    reconcile_from_dir "$fixdir" "abc123" "v0.1.1" "v0.1.0" "$(date -u +%s)" "" >/dev/null 2>&1 || rc=$?  # bashrs disable-line=DET002
    if [ "$rc" -ne 2 ]; then printf 'FAIL malformed-merged-prs: expected exit 2, got %s\n' "$rc" >&2; fails=$((fails + 1)); fi
    printf '[]' > "${fixdir}/merged-prs.json"; printf 'oops' > "${fixdir}/open-prs.json"
    rc=0
    reconcile_from_dir "$fixdir" "abc123" "v0.1.1" "v0.1.0" "$(date -u +%s)" "" >/dev/null 2>&1 || rc=$?  # bashrs disable-line=DET002
    if [ "$rc" -ne 2 ]; then printf 'FAIL malformed-open-prs: expected exit 2, got %s\n' "$rc" >&2; fails=$((fails + 1)); fi
    printf '[]' > "${fixdir}/open-prs.json"
    rc=0
    reconcile_from_dir "$fixdir" "abc123" "v0.1.1" "v0.1.0" "$(date -u +%s)" "" >/dev/null 2>&1 || rc=$?  # bashrs disable-line=DET002
    if [ "$rc" -ne 0 ]; then printf 'FAIL empty-fixture-clean: expected exit 0, got %s\n' "$rc" >&2; fails=$((fails + 1)); fi
    rm -rf "${fixdir:?}"

    # --rev: a tag that is not made yet is judged at --rev (release.yml bump path). Every row runs with a stub gh on
    # PATH, so exit 2 can only come from the --rev / window logic, never from a missing gh.
    revdir="$(mktemp -d)"
    mkdir -p "${revdir}/bin" "${revdir}/repo"
    cat > "${revdir}/bin/gh" <<'EOF_GH'
#!/bin/sh
case "$*" in
    "auth status") exit 0 ;;
    "issue list --state open"*) printf '[{"number":5}]\n' ;;
    *) printf '[]\n' ;;
esac
EOF_GH
    chmod +x "${revdir}/bin/gh"
    (
        cd "${revdir}/repo" || exit 9
        git init -q --template= . && git config core.hooksPath /dev/null && git config user.email t@example.invalid && git config user.name t && git remote add origin "${revdir}/repo"
        git commit -q --allow-empty -m "base" && git tag v0.1.0
        git commit -q --allow-empty -m "work"
        git tag v0.2.0
    )
    rev_rc() { # args... -> rc of check_reconcile.sh run inside the fixture repo
        local r=0
        ( cd "${revdir}/repo" && PATH="${revdir}/bin:${PATH}" bash "$SELF_PATH" "$@" >/dev/null 2>&1 ) || r=$?
        return "$r"
    }
    rc=0; rev_rc v9.9.9-none --rev "not-a-commit-xyz" --prev-tag v0.1.0 || rc=$?
    if [ "$rc" -ne 2 ]; then printf 'FAIL rev-bogus: expected exit 2, got %s\n' "$rc" >&2; fails=$((fails + 1)); fi
    rc=0; rev_rc v0.3.0 --prev-tag v0.1.0 --rev || rc=$?
    if [ "$rc" -ne 2 ]; then printf 'FAIL rev-novalue: expected exit 2, got %s\n' "$rc" >&2; fails=$((fails + 1)); fi
    # An empty flag value is refused by the argument parser (usage, exit 2), not later by a downstream check.
    usage_rc() { # args... -> 0 iff the run exits 2 AND prints the usage line
        local out r=0
        out="$( cd "${revdir}/repo" && PATH="${revdir}/bin:${PATH}" bash "$SELF_PATH" "$@" 2>&1 >/dev/null )" || r=$?
        [ "$r" -eq 2 ] && case "$out" in *usage:*) return 0 ;; esac
        return 1
    }
    usage_rc v0.3.0 --rev "" --prev-tag v0.1.0 \
        || { printf 'FAIL rev-emptyvalue: expected usage + exit 2\n' >&2; fails=$((fails + 1)); }
    usage_rc v0.3.0 --rev HEAD --prev-tag "" \
        || { printf 'FAIL prev-tag-emptyvalue: expected usage + exit 2\n' >&2; fails=$((fails + 1)); }
    usage_rc v0.3.0 --rev HEAD --prev-tag v0.1.0 --json "" \
        || { printf 'FAIL json-emptyvalue: expected usage + exit 2\n' >&2; fails=$((fails + 1)); }
    usage_rc v0.3.0 --rev HEAD --prev-tag v0.1.0 --archive-dir "" \
        || { printf 'FAIL r3-usage-archive-emptyvalue: expected usage + exit 2\n' >&2; fails=$((fails + 1)); }
    rc=0; rev_rc v0.3.0 --prev-tag v0.1.0 --rev HEAD --archive-dir || rc=$?
    if [ "$rc" -ne 2 ]; then printf 'FAIL r3-usage-archive-novalue: expected exit 2, got %s\n' "$rc" >&2; fails=$((fails + 1)); fi
    rc=0; rev_rc v0.3.0 --rev HEAD --prev-tag v0.2.0 || rc=$?
    if [ "$rc" -ne 2 ]; then printf 'FAIL rev-empty-window: expected exit 2, got %s\n' "$rc" >&2; fails=$((fails + 1)); fi
    rc=0; rev_rc v0.3.0 --rev v0.1.0 --prev-tag v0.2.0 || rc=$?
    if [ "$rc" -ne 2 ]; then printf 'FAIL rev-not-ancestor: expected exit 2, got %s\n' "$rc" >&2; fails=$((fails + 1)); fi
    rc=0; rev_rc v0.2.0 --rev v0.1.0 --prev-tag v0.1.0 || rc=$?
    if [ "$rc" -ne 2 ]; then printf 'FAIL rev-tag-elsewhere: expected exit 2, got %s\n' "$rc" >&2; fails=$((fails + 1)); fi
    (
        cd "${revdir}/repo" || exit 9
        git commit -q --allow-empty -m "next"
    )
    rc=0; rev_rc v0.3.0 --rev HEAD --prev-tag v0.2.0 || rc=$?
    if [ "$rc" -ne 0 ]; then printf 'FAIL rev-clean: expected exit 0, got %s\n' "$rc" >&2; fails=$((fails + 1)); fi
    (
        cd "${revdir}/repo" || exit 9
        git commit -q --allow-empty -m "fix: x" -m "Fixes #5"
    )
    rc=0; rev_rc v0.3.0 --rev HEAD --prev-tag v0.2.0 || rc=$?
    if [ "$rc" -ne 1 ]; then printf 'FAIL rev-seeded: expected exit 1, got %s\n' "$rc" >&2; fails=$((fails + 1)); fi
    rm -rf "${revdir:?}"

    # #7373 R3 non-blocking contract: archive (bundle + tips.txt + README.md),
    # verify read-back, report to stdout, never delete branches, exit 0 for R3 alone,
    # keep R1/R2/R4 blocking, and fail closed on write/read-back errors.
    cases=35
    r3dir="$(mktemp -d -p "${RUNNER_TEMP:-${TMPDIR:-/tmp}}")" || return 2
    mkdir -p "${r3dir}/bin" "${r3dir}/repo"
    cat > "${r3dir}/bin/gh" <<'EOF_GH_R3'
#!/bin/sh
case "$*" in
    "auth status") exit 0 ;;
    "issue list --state open"*) printf '[{"number":5}]\n' ;;
    *) printf '[]\n' ;;
esac
EOF_GH_R3
    chmod +x "${r3dir}/bin/gh"
    old_git_date="$(date -u -d '-20 days' +%Y-%m-%dT%H:%M:%SZ)"  # bashrs disable-line=DET002
    (
        cd "${r3dir}/repo" || exit 9
        git init -q --template= -b main .
        git config core.hooksPath /dev/null
        git config user.email t@example.invalid
        git config user.name t
        git remote add origin "${r3dir}/repo"
        GIT_AUTHOR_DATE="$old_git_date" GIT_COMMITTER_DATE="$old_git_date" \
            git commit -q --allow-empty -m "base" && git tag v0.1.0
        git checkout -q -b feat/stale-work
        GIT_AUTHOR_DATE="$old_git_date" GIT_COMMITTER_DATE="$old_git_date" \
            git commit -q --allow-empty -m "stale feature work"
        git checkout -q -b stale-plain main
        GIT_AUTHOR_DATE="$old_git_date" GIT_COMMITTER_DATE="$old_git_date" \
            git commit -q --allow-empty -m "stale plain work"
        git checkout -q main
        git commit -q --allow-empty -m "chore: recent main commit"
    )
    sha_feat="$(git -C "${r3dir}/repo" rev-parse refs/heads/feat/stale-work)"
    sha_plain="$(git -C "${r3dir}/repo" rev-parse refs/heads/stale-plain)"

    # Row: empty R3_ARCHIVE_DIR env var fails closed (exit 2).
    rc=0
    ( cd "${r3dir}/repo" && R3_ARCHIVE_DIR="" PATH="${r3dir}/bin:${PATH}" bash "$SELF_PATH" v0.2.0 --rev HEAD --prev-tag v0.1.0 >/dev/null 2>&1 ) || rc=$?
    if [ "$rc" -ne 2 ]; then printf 'FAIL r3-env-archive-emptyvalue: expected exit 2, got %s\n' "$rc" >&2; fails=$((fails + 1)); fi
    cases=$((cases + 1))

    # Row (PROVES R3 FORMERLY BLOCKED): R3=2 alone with --archive-dir exits 0, archives + verifies + reports, never deletes.
    arch1="${r3dir}/archive-flag"
    json1="${r3dir}/receipt1.json"
    out1=""
    rc=0
    out1="$( cd "${r3dir}/repo" && PATH="${r3dir}/bin:${PATH}" bash "$SELF_PATH" v0.2.0 --rev HEAD --prev-tag v0.1.0 --json "$json1" --archive-dir "$arch1" 2>/dev/null )" || rc=$?
    if [ "$rc" -ne 0 ] \
        || ! grep -qF 'REPORT R3(dead-branches)=2' <<< "$out1" \
        || ! grep -qF "feat/stale-work ${sha_feat}" <<< "$out1" \
        || ! grep -qF "stale-plain ${sha_plain}" <<< "$out1" \
        || [ ! -s "${arch1}/tips.txt" ] \
        || [ ! -s "${arch1}/README.md" ] \
        || [ ! -s "${arch1}/r3-v0.2.0-branches.bundle" ]; then
        printf 'FAIL r3-nonblocking-archive-flag: rc=%s out=%s\n' "$rc" "$out1" >&2
        fails=$((fails + 1))
    else
        r3_check_ok=1
        python3 - "$json1" "$arch1" "$sha_feat" "$sha_plain" <<'PY_R3_CHECK' || r3_check_ok=0
import json, os, sys
jpath, adir, sf, sp = sys.argv[1:5]
d = json.load(open(jpath))
assert d["R3"]["count"] == 2, d["R3"]
assert d["R3"]["archived"] is True, d["R3"]
assert d["R3"]["archive_dir"] == adir, d["R3"]
tips = dict(l.split() for l in open(os.path.join(adir, "tips.txt")) if l.strip())
assert tips == {"feat/stale-work": sf, "stale-plain": sp}, tips
readme = open(os.path.join(adir, "README.md")).read()
assert "git fetch" in readme and "git push origin" in readme and sf in readme and sp in readme
PY_R3_CHECK
        bare_check="${r3dir}/bare-check"
        git init -q --bare --template= "$bare_check"
        git -C "$bare_check" bundle verify "${arch1}/r3-v0.2.0-branches.bundle" >/dev/null 2>&1 || r3_check_ok=0
        git -C "${r3dir}/repo" show-ref --verify --quiet refs/remotes/origin/feat/stale-work || r3_check_ok=0
        git -C "${r3dir}/repo" show-ref --verify --quiet refs/remotes/origin/stale-plain || r3_check_ok=0
        git -C "${r3dir}/repo" show-ref --verify --quiet refs/heads/feat/stale-work || r3_check_ok=0
        rm -rf "${bare_check:?}"
        if [ "$r3_check_ok" -ne 1 ]; then
            printf 'FAIL r3-nonblocking-archive-flag: archive verification or branch-preservation failed\n' >&2
            fails=$((fails + 1))
        fi
    fi
    cases=$((cases + 1))

    # Row: direct reconcile_from_dir with R3=1, R1=R2=R4=0 exits 0 (formerly exited 1).
    r3_fx="${r3dir}/fx"
    mkdir -p "$r3_fx"
    : > "${r3_fx}/issues-open.tsv"
    printf 'deadbeef\nchore: ok\n\x00' > "${r3_fx}/log.bin"
    printf '[]' > "${r3_fx}/merged-prs.json"
    printf '[]' > "${r3_fx}/open-prs.json"
    printf 'stale-plain\t%s\n' "$old_git_date" > "${r3_fx}/branches.txt"
    rc=0
    reconcile_from_dir "$r3_fx" "abc123" "v0.2.0" "v0.1.0" "$(date -u -d '-14 days' +%s)" "" "${r3dir}/arch-fx" "${r3dir}/repo" >/dev/null 2>&1 || rc=$?  # bashrs disable-line=DET002
    if [ "$rc" -ne 0 ]; then
        printf 'FAIL r3-reconcile-from-dir-formerly-blocked: expected exit 0, got %s\n' "$rc" >&2
        fails=$((fails + 1))
    fi
    cases=$((cases + 1))

    # Row: R3_ARCHIVE_DIR env var without --archive-dir works and archives.
    arch2="${r3dir}/archive-env"
    rc=0
    ( cd "${r3dir}/repo" && R3_ARCHIVE_DIR="$arch2" PATH="${r3dir}/bin:${PATH}" bash "$SELF_PATH" v0.2.0 --rev HEAD --prev-tag v0.1.0 >/dev/null 2>&1 ) || rc=$?
    if [ "$rc" -ne 0 ] || [ ! -s "${arch2}/r3-v0.2.0-branches.bundle" ] || [ ! -s "${arch2}/tips.txt" ] || [ ! -s "${arch2}/README.md" ]; then
        printf 'FAIL r3-archive-env-var: rc=%s\n' "$rc" >&2
        fails=$((fails + 1))
    fi
    cases=$((cases + 1))

    # Row: R3 + R1 still blocks (exit 1) while archiving R3.
    printf '5\tOpen bug\n' > "${r3_fx}/issues-open.tsv"
    printf 'deadbeef\nfix: bug\nFixes #5\n\x00' > "${r3_fx}/log.bin"
    rc=0
    reconcile_from_dir "$r3_fx" "abc123" "v0.2.0" "v0.1.0" "$(date -u -d '-14 days' +%s)" "" "${r3dir}/arch-r1" "${r3dir}/repo" >/dev/null 2>&1 || rc=$?  # bashrs disable-line=DET002
    if [ "$rc" -ne 1 ] || [ ! -s "${r3dir}/arch-r1/r3-v0.2.0-branches.bundle" ]; then
        printf 'FAIL r3-plus-r1-still-blocks: expected exit 1, got %s\n' "$rc" >&2
        fails=$((fails + 1))
    fi
    cases=$((cases + 1))

    # Row: R3 + R2 still blocks (exit 1).
    r3_fx_r2="${r3dir}/fx-r2"
    mkdir -p "$r3_fx_r2"
    : > "${r3_fx_r2}/issues-open.tsv"
    printf 'deadbeef\nchore: ok\n\x00' > "${r3_fx_r2}/log.bin"
    printf '[{"number":10,"body":"Refs #5"}]' > "${r3_fx_r2}/merged-prs.json"
    printf '[]' > "${r3_fx_r2}/open-prs.json"
    printf 'stale-plain\t%s\n' "$old_git_date" > "${r3_fx_r2}/branches.txt"
    rc=0
    reconcile_from_dir "$r3_fx_r2" "abc123" "v0.2.0" "v0.1.0" "$(date -u -d '-14 days' +%s)" "" "${r3dir}/arch-r2" "${r3dir}/repo" >/dev/null 2>&1 || rc=$?  # bashrs disable-line=DET002
    if [ "$rc" -ne 1 ]; then
        printf 'FAIL r3-plus-r2-still-blocks: expected exit 1, got %s\n' "$rc" >&2
        fails=$((fails + 1))
    fi
    cases=$((cases + 1))

    # Row: R3 + R4 still blocks (exit 1).
    r3_fx_r4="${r3dir}/fx-r4"
    mkdir -p "$r3_fx_r4"
    : > "${r3_fx_r4}/issues-open.tsv"
    printf 'deadbeef\nchore: ok\n\x00' > "${r3_fx_r4}/log.bin"
    printf '[]' > "${r3_fx_r4}/merged-prs.json"
    printf '[{"number":7,"mergeStateStatus":"DIRTY","createdAt":"%s"}]' "$old_git_date" > "${r3_fx_r4}/open-prs.json"
    printf 'stale-plain\t%s\n' "$old_git_date" > "${r3_fx_r4}/branches.txt"
    rc=0
    reconcile_from_dir "$r3_fx_r4" "abc123" "v0.2.0" "v0.1.0" "$(date -u -d '-14 days' +%s)" "" "${r3dir}/arch-r4" "${r3dir}/repo" >/dev/null 2>&1 || rc=$?  # bashrs disable-line=DET002
    if [ "$rc" -ne 1 ]; then
        printf 'FAIL r3-plus-r4-still-blocks: expected exit 1, got %s\n' "$rc" >&2
        fails=$((fails + 1))
    fi
    cases=$((cases + 1))

    # Row: unwritable archive path fails closed (exit 2).
    printf 'not a directory\n' > "${r3dir}/regular-file"
    rc=0
    ( cd "${r3dir}/repo" && PATH="${r3dir}/bin:${PATH}" bash "$SELF_PATH" v0.2.0 --rev HEAD --prev-tag v0.1.0 --archive-dir "${r3dir}/regular-file/sub" >/dev/null 2>&1 ) || rc=$?
    if [ "$rc" -ne 2 ]; then
        printf 'FAIL r3-fail-closed-unwritable-archive: expected exit 2, got %s\n' "$rc" >&2
        fails=$((fails + 1))
    fi
    cases=$((cases + 1))

    # Row: dead branch missing ref in git repo fails closed (exit 2).
    r3_fx_missing="${r3dir}/fx-missing"
    mkdir -p "$r3_fx_missing"
    : > "${r3_fx_missing}/issues-open.tsv"
    printf 'deadbeef\nchore: ok\n\x00' > "${r3_fx_missing}/log.bin"
    printf '[]' > "${r3_fx_missing}/merged-prs.json"
    printf '[]' > "${r3_fx_missing}/open-prs.json"
    printf 'ghost-branch\t%s\n' "$old_git_date" > "${r3_fx_missing}/branches.txt"
    rc=0
    reconcile_from_dir "$r3_fx_missing" "abc123" "v0.2.0" "v0.1.0" "$(date -u -d '-14 days' +%s)" "" "${r3dir}/arch-missing" "${r3dir}/repo" >/dev/null 2>&1 || rc=$?  # bashrs disable-line=DET002
    if [ "$rc" -ne 2 ]; then
        printf 'FAIL r3-fail-closed-missing-ref: expected exit 2, got %s\n' "$rc" >&2
        fails=$((fails + 1))
    fi
    cases=$((cases + 1))

    # Read-back fail-closed rows: corrupt bundle, mismatched tips.txt, missing README.md.
    arch_bad="${r3dir}/arch-bad"
    cp -r "$arch1" "$arch_bad"
    printf 'stale-plain %s\n' "$sha_plain" > "${r3dir}/expected-plain.txt"
    printf 'feat/stale-work %s\nstale-plain %s\n' "$sha_feat" "$sha_plain" > "${r3dir}/expected-both.txt"
    printf 'corrupt' > "${arch_bad}/r3-v0.2.0-branches.bundle"
    rc=0
    r3_verify_archive "${arch_bad}/r3-v0.2.0-branches.bundle" "${arch_bad}/tips.txt" "${arch_bad}/README.md" "${r3dir}/expected-both.txt" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne 2 ]; then
        printf 'FAIL r3-fail-closed-corrupt-bundle-readback: expected exit 2, got %s\n' "$rc" >&2
        fails=$((fails + 1))
    fi
    cases=$((cases + 1))

    cp "${arch1}/r3-v0.2.0-branches.bundle" "${arch_bad}/r3-v0.2.0-branches.bundle"
    printf 'feat/stale-work 0000000000000000000000000000000000000000\n' > "${arch_bad}/tips.txt"
    rc=0
    r3_verify_archive "${arch_bad}/r3-v0.2.0-branches.bundle" "${arch_bad}/tips.txt" "${arch_bad}/README.md" "${r3dir}/expected-both.txt" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne 2 ]; then
        printf 'FAIL r3-fail-closed-tips-mismatch-readback: expected exit 2, got %s\n' "$rc" >&2
        fails=$((fails + 1))
    fi
    cases=$((cases + 1))

    cp "${arch1}/tips.txt" "${arch_bad}/tips.txt"
    : > "${arch_bad}/README.md"
    rc=0
    r3_verify_archive "${arch_bad}/r3-v0.2.0-branches.bundle" "${arch_bad}/tips.txt" "${arch_bad}/README.md" "${r3dir}/expected-both.txt" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -ne 2 ]; then
        printf 'FAIL r3-fail-closed-missing-readme-readback: expected exit 2, got %s\n' "$rc" >&2
        fails=$((fails + 1))
    fi
    cases=$((cases + 1))

    # Mutants for #7373: every mutant must apply cleanly (count == 1) and be killed.
    r3_fx_clean="${r3dir}/fx-clean"
    mkdir -p "$r3_fx_clean"
    : > "${r3_fx_clean}/issues-open.tsv"
    printf 'deadbeef\nchore: ok\n\x00' > "${r3_fx_clean}/log.bin"
    printf '[]' > "${r3_fx_clean}/merged-prs.json"
    printf '[]' > "${r3_fx_clean}/open-prs.json"
    printf 'stale-plain\t%s\n' "$old_git_date" > "${r3_fx_clean}/branches.txt"

    make_r3_mutant() {
        local mname="$1" old="$2" new="$3"
        local dst="${r3dir}/mut-${mname}.sh"
        python3 - "$SELF_PATH" "$dst" "$old" "$new" <<'PY_MUT' || return 1
import sys
src, dst, old, new = sys.argv[1:5]
s = open(src, "r", encoding="utf-8").read()
if s.count(old) != 1:
    sys.stderr.write(f"mutant anchor count={s.count(old)} for {old!r}\n")
    sys.exit(1)
open(dst, "w", encoding="utf-8").write(s.replace(old, new, 1))
PY_MUT
    }

    run_r3_mutant() {
        local mname="$1" old="$2" new="$3" want_fail_rc="$4" mode="$5"
        local mfile="${r3dir}/mut-${mname}.sh" mrc=0 mout=""
        if ! make_r3_mutant "$mname" "$old" "$new"; then
            printf 'FAIL mutant-%s: anchor missing or non-unique\n' "$mname" >&2
            fails=$((fails + 1))
            cases=$((cases + 1))
            return
        fi
        case "$mode" in
            r3-only)
                ( CHECK_RECONCILE_SOURCE_ONLY=1; . "$mfile"; reconcile_from_dir "$r3_fx_clean" "abc123" "v0.2.0" "v0.1.0" "$(date -u -d '-14 days' +%s)" "" "${r3dir}/mut-arch-${mname}" "${r3dir}/repo" >/dev/null 2>&1 ) || mrc=$?  # bashrs disable-line=DET002
                [ "$mrc" -ne "$want_fail_rc" ] || { printf 'FAIL mutant-%s SURVIVED (rc=%s)\n' "$mname" "$mrc" >&2; fails=$((fails + 1)); }
                ;;
            unwritable)
                ( CHECK_RECONCILE_SOURCE_ONLY=1; . "$mfile"; reconcile_from_dir "$r3_fx_clean" "abc123" "v0.2.0" "v0.1.0" "$(date -u -d '-14 days' +%s)" "" "${r3dir}/regular-file/sub" "${r3dir}/repo" >/dev/null 2>&1 ) || mrc=$?  # bashrs disable-line=DET002
                [ "$mrc" -ne "$want_fail_rc" ] || { printf 'FAIL mutant-%s SURVIVED (rc=%s)\n' "$mname" "$mrc" >&2; fails=$((fails + 1)); }
                ;;
            corrupt-bundle)
                git -C "${r3dir}/repo" bundle create "${arch_bad}/r3-v0.2.0-branches.bundle" "^main" refs/remotes/origin/feat/stale-work refs/remotes/origin/stale-plain >/dev/null 2>&1
                cp "${arch1}/tips.txt" "${arch_bad}/tips.txt"
                cp "${arch1}/README.md" "${arch_bad}/README.md"
                ( CHECK_RECONCILE_SOURCE_ONLY=1; . "$mfile"; r3_verify_archive "${arch_bad}/r3-v0.2.0-branches.bundle" "${arch_bad}/tips.txt" "${arch_bad}/README.md" "${r3dir}/expected-both.txt" >/dev/null 2>&1 ) || mrc=$?
                [ "$mrc" -ne "$want_fail_rc" ] || { printf 'FAIL mutant-%s SURVIVED (rc=%s)\n' "$mname" "$mrc" >&2; fails=$((fails + 1)); }
                ;;
            report-stdout)
                mout="$( CHECK_RECONCILE_SOURCE_ONLY=1; . "$mfile"; reconcile_from_dir "$r3_fx_clean" "abc123" "v0.2.0" "v0.1.0" "$(date -u -d '-14 days' +%s)" "" "${r3dir}/mut-arch-${mname}" "${r3dir}/repo" 2>/dev/null )" || mrc=$?  # bashrs disable-line=DET002
                if [ "$mrc" -eq 0 ] && grep -qF 'REPORT R3(dead-branches)=' <<< "$mout"; then
                    printf 'FAIL mutant-%s SURVIVED\n' "$mname" >&2
                    fails=$((fails + 1))
                fi
                ;;
            r1)
                ( CHECK_RECONCILE_SOURCE_ONLY=1; . "$mfile"; reconcile_from_dir "$r3_fx" "abc123" "v0.2.0" "v0.1.0" "$(date -u -d '-14 days' +%s)" "" "${r3dir}/mut-arch-${mname}" "${r3dir}/repo" >/dev/null 2>&1 ) || mrc=$?  # bashrs disable-line=DET002
                [ "$mrc" -ne "$want_fail_rc" ] || { printf 'FAIL mutant-%s SURVIVED (rc=%s)\n' "$mname" "$mrc" >&2; fails=$((fails + 1)); }
                ;;
            r2)
                ( CHECK_RECONCILE_SOURCE_ONLY=1; . "$mfile"; reconcile_from_dir "$r3_fx_r2" "abc123" "v0.2.0" "v0.1.0" "$(date -u -d '-14 days' +%s)" "" "${r3dir}/mut-arch-${mname}" "${r3dir}/repo" >/dev/null 2>&1 ) || mrc=$?  # bashrs disable-line=DET002
                [ "$mrc" -ne "$want_fail_rc" ] || { printf 'FAIL mutant-%s SURVIVED (rc=%s)\n' "$mname" "$mrc" >&2; fails=$((fails + 1)); }
                ;;
            r4)
                ( CHECK_RECONCILE_SOURCE_ONLY=1; . "$mfile"; reconcile_from_dir "$r3_fx_r4" "abc123" "v0.2.0" "v0.1.0" "$(date -u -d '-14 days' +%s)" "" "${r3dir}/mut-arch-${mname}" "${r3dir}/repo" >/dev/null 2>&1 ) || mrc=$?  # bashrs disable-line=DET002
                [ "$mrc" -ne "$want_fail_rc" ] || { printf 'FAIL mutant-%s SURVIVED (rc=%s)\n' "$mname" "$mrc" >&2; fails=$((fails + 1)); }
                ;;
        esac
        cases=$((cases + 1))
    }

    run_r3_mutant "r3-formerly-blocked" \
        'if [ "$r1_count" -eq 0 ] && ''[ "$r2_count" -eq 0 ] && [ "$r4_count" -eq 0 ]; then' \
        'if [ "$r1_count" -eq 0 ] && [ "$r2_count" -eq 0 ] && [ "$r3_count" -eq 0 ] && [ "$r4_count" -eq 0 ]; then' \
        0 "r3-only"
    run_r3_mutant "r3-skip-archive" \
        'r3_bundle="$(r3_archive_dead_branches ''"$r3_list" "$archive_dir" "$repo_dir" "$tag")" || return 2' \
        'r3_bundle="skipped"' \
        2 "unwritable"
    run_r3_mutant "r3-skip-bundle-verify" \
        'if ! git -C "$verify_repo" ''bundle verify "$abs_bundle" >/dev/null 2>&1; then' \
        'if false; then' \
        2 "corrupt-bundle"
    run_r3_mutant "r3-drop-report" \
        "REPORT ""R3(dead-branches)=%s non-blocking archived=%s bundle=%s" \
        "MUTATED-REPORT R3" \
        0 "report-stdout"
    run_r3_mutant "r1-waived" \
        'if [ "$r1_count" -eq 0 ] && ''[ "$r2_count" -eq 0 ] && [ "$r4_count" -eq 0 ]; then' \
        'if [ "$r2_count" -eq 0 ] && [ "$r4_count" -eq 0 ]; then' \
        1 "r1"
    run_r3_mutant "r2-waived" \
        'if [ "$r1_count" -eq 0 ] && ''[ "$r2_count" -eq 0 ] && [ "$r4_count" -eq 0 ]; then' \
        'if [ "$r1_count" -eq 0 ] && [ "$r4_count" -eq 0 ]; then' \
        1 "r2"
    run_r3_mutant "r4-waived" \
        'if [ "$r1_count" -eq 0 ] && ''[ "$r2_count" -eq 0 ] && [ "$r4_count" -eq 0 ]; then' \
        'if [ "$r1_count" -eq 0 ] && [ "$r2_count" -eq 0 ]; then' \
        1 "r4"

    rm -rf "${r3dir:?}"

    if [ "$fails" -ne 0 ]; then
        printf 'self-test FAILED: %s case(s).\n' "$fails" >&2
        return 1
    fi
    printf 'self-test OK: %s case(s).\n' "$cases"
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
    rev=""
    archive_dir=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --prev-tag)
                { [ $# -ge 2 ] && [ -n "${2:-}" ]; } || usage
                prev_tag="${2:-}"
                shift 2
                ;;
            --json)
                { [ $# -ge 2 ] && [ -n "${2:-}" ]; } || usage
                json_out="${2:-}"
                shift 2
                ;;
            --rev)
                { [ $# -ge 2 ] && [ -n "${2:-}" ]; } || usage
                rev="${2:-}"
                shift 2
                ;;
            --archive-dir)
                { [ $# -ge 2 ] && [ -n "${2:-}" ]; } || usage
                archive_dir="${2:-}"
                shift 2
                ;;
            *)
                usage
                ;;
        esac
    done

    if [ -z "$archive_dir" ]; then
        if [ -n "${R3_ARCHIVE_DIR+x}" ]; then
            if [ -z "$R3_ARCHIVE_DIR" ]; then
                printf 'ENV: R3_ARCHIVE_DIR is set but empty.\n' >&2
                exit 2
            fi
            archive_dir="$R3_ARCHIVE_DIR"
        else
            safe_tag="$(printf '%s' "$tag" | tr '/[:space:]' '__')"
            archive_dir="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/r3-branch-archive-${safe_tag}"
        fi
    fi

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

    # end = the commit the window ends at: the tag's commit, or --rev when the tag is not made yet.
    end=""
    if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null 2>&1; then
        end="$(git rev-parse "${tag}^{commit}")"
        if [ -n "$rev" ] && [ "$(git rev-parse -q --verify "${rev}^{commit}" 2>/dev/null || true)" != "$end" ]; then
            printf 'ENV: tag %s exists at %s, not at --rev %s\n' "$tag" "$end" "$rev" >&2
            exit 2
        fi
    elif [ -n "$rev" ]; then
        end="$(git rev-parse -q --verify "${rev}^{commit}" 2>/dev/null || true)"
        if [ -z "$end" ]; then
            printf 'ENV: --rev not a commit: %s\n' "$rev" >&2
            exit 2
        fi
    else
        printf 'ENV: tag not found: %s\n' "$tag" >&2
        exit 2
    fi

    if [ -z "$prev_tag" ]; then
        prev_tag="$(git describe --tags --abbrev=0 "${end}^" 2>/dev/null || true)"
        if [ -z "$prev_tag" ]; then
            printf 'ENV: could not derive --prev-tag from %s^; pass it explicitly.\n' "$tag" >&2
            exit 2
        fi
    fi

    if ! git rev-parse -q --verify "refs/tags/${prev_tag}" >/dev/null 2>&1; then
        printf 'ENV: prev-tag not found: %s\n' "$prev_tag" >&2
        exit 2
    fi

    # The window must be a real, non-empty range: prev-tag an ancestor of the end commit and not the end commit itself.
    if ! git merge-base --is-ancestor "refs/tags/${prev_tag}" "$end" 2>/dev/null \
        || [ "$(git rev-parse "refs/tags/${prev_tag}^{commit}")" = "$end" ]; then
        printf 'ENV: prev-tag %s is not an ancestor of %s (or the window is empty).\n' "$prev_tag" "$end" >&2
        exit 2
    fi

    if ! gh auth status >/dev/null 2>&1; then
        printf 'ENV: gh is not authenticated.\n' >&2
        exit 2
    fi

    input_dir="$(mktemp -d)" || exit 2
    trap 'rm -rf "${input_dir:?}"' EXIT

    if ! fetch_live_inputs "$end" "$prev_tag" "$input_dir"; then
        printf 'ENV: failed to gather inputs from git/gh.\n' >&2
        exit 2
    fi

    sha="$end"
    prev_epoch="$(git log -1 --format=%at "$prev_tag")"

    rc=0
    reconcile_from_dir "$input_dir" "$sha" "$tag" "$prev_tag" "$prev_epoch" "$json_out" "$archive_dir" "." || rc=$?
    exit "$rc"
}

if [ "${CHECK_RECONCILE_SOURCE_ONLY:-0}" != "1" ]; then
    main "$@"
fi
