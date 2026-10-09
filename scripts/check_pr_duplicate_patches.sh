#!/usr/bin/env bash
# check_pr_duplicate_patches.sh - DUP-001 (#4536): a PR may not carry a change
# that another OPEN PR already carries.
#
# Why: the 0.70 train carried the same fixes in two and three open PRs at once
# (a fix on its owner PR, copied into a fold, cherry-picked again into the
# release car). Each copy is reviewed, queued and CI'd separately, and when
# they diverge nobody knows which one is the fix. Target: 0 duplicated
# changes across open PRs.
#
# How: every commit and every hunk of the PR is hashed with `git patch-id
# --stable`, which ignores the commit message, line numbers and whitespace but
# not the file path (the same edit to a different file is a different id).
# Hunks come from each commit in base...head (merges skipped) AND from the
# PR's cumulative diff against its merge-base, so a copy squashed into a bigger
# commit, or reached in two steps, still meets its twin. A commit's own id is
# taken over its counted hunks only. Any id shared with another open PR fails
# the check, naming both PRs, both shas, and the file + hunk.
#
# Not counted:
#   * generated outputs (GENERATED_PATHS, GENERATED_PREFIXES): two honest
#     regens write identical bytes;
#   * README.md hunks whose changed lines all sit inside a generated
#     <!-- X_START -->..<!-- X_END --> block (readme_sync.sh);
#   * whitespace-only hunks;
#   * a hunk the PR's base already gained since the PR forked, under any sha:
#     two stale PRs that both still carry a landed fix are behind, not copies.
#
# Exemption: a PR body line `stacked-on: #N` exempts that pair, both ways.
# There is no other exemption; a fold closes its sources, it does not sit
# beside them.
#
# Every diff is read under one pinned git configuration (gitp, DIFF_FLAGS, the
# pins scripts/lib/pr_review_patch_id.sh uses), so no runner's or developer's
# git config changes what is hashed. The engine runs as its own process
# (--engine) and any failure in it is rc 2: a run that could not read every
# PR is never reported as a pass (0) nor blamed on the PR as a duplicate (1).
#
# Usage:
#   check_pr_duplicate_patches.sh --pr N        check open PR N against every other open PR
#   check_pr_duplicate_patches.sh               same, N read from $GITHUB_EVENT_PATH; an event
#                                               with no pull_request in it is rc 2 (nothing was
#                                               measured, so there is nothing to pass)
#   check_pr_duplicate_patches.sh --all         every pair of open PRs, once (the nightly sweep)
#   check_pr_duplicate_patches.sh --self-test   the must-match / must-not-match case table
#
# Exit: 0 no duplicate; 1 duplicate found (or a case-table row failed);
#       2 usage error, or a PR whose data could not be read.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SELF="$HERE/$(basename "$0")"
TD=""  # global: the EXIT trap runs after the function that set it returned
REPO=""
DETAIL_PER_PAIR=20
GENERATED_PATHS="Cargo.lock docs/roadmaps/roadmap.yaml contracts/census.json contracts/contracts.nt contracts/shapes.ttl"
# pv-sat's witnesses: two PRs that regenerate one get the same cpu_ms hunk.
GENERATED_PREFIXES="contracts/witness/"
DIFF_FLAGS=(--no-color --no-ext-diff --no-textconv --no-renames --full-index
    --diff-algorithm=myers --indent-heuristic --inter-hunk-context=0 -U3
    --src-prefix=a/ --dst-prefix=b/ -O/dev/null)
# One line per `stacked-on: #N` in the body (any case), as a comma list of N.
TSV_JQ='.[] | [(.number | tostring), .headRefOid, .baseRefName,
    ([(.body // "") | split("\n")[]
      | select(test("^[[:space:]]*stacked-on:[[:space:]]*#[0-9]+[[:space:]]*$"; "i"))
      | capture("#(?<n>[0-9]+)").n | tonumber | tostring] | join(","))] | @tsv'

usage() {
    sed -n '/^# Usage:/,/^# Exit:/p' "$SELF" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

die2() {
    echo "DUP-001: $* -- refusing to pass unread" >&2
    exit 2
}

gitp() {
    git -C "$REPO" -c core.quotePath=true -c diff.noprefix=false -c diff.mnemonicPrefix=false \
        -c diff.suppressBlankEmpty=false -c diff.relative=false -c log.showSignature=false "$@"
}

# ------------------------------------------------------------------- engine --

# Splits `git log -p` / `git diff` output into hunks. Each hunk that counts
# becomes one patch in FEED ("commit <40-digit key>", the file header, the
# hunk) and one row in INDEX (key, label, path, hunk header). count=1 applies
# the not-counted rules; count=0 (a base's landed changes) keeps every hunk.
SPLIT_AWK=$(cat <<'AWK'
function load(rev,   cmd, line, i, depth, o, c) {
    if (rev in LOADED) return
    LOADED[rev] = 1
    cmd = "git -C \"$DUP_REPO\" cat-file blob '" rev ":README.md' 2>/dev/null"
    i = 0; depth = 0
    while ((cmd | getline line) > 0) {
        i++; o = count_start(line); c = count_end(line)
        if (o || c || depth) BLK[rev, i] = 1
        depth += o - c
        if (depth < 0) depth = 0
    }
    close(cmd)
}
function count_start(line,   n, pos, p) {
    n = 0; pos = 1
    while (match(substr(line, pos), /<!--[[:space:]]*[A-Z][A-Z0-9_]*_START/)) {
        p = pos + RSTART - 1
        if (substr(line, p + RLENGTH, 1) ~ /[A-Za-z0-9_]/) { pos = p + 1; continue }
        n++; pos = p + RLENGTH
    }
    return n
}
function count_end(line,   n, pos, p) {
    n = 0; pos = 1
    while (match(substr(line, pos), /[A-Z][A-Z0-9_]*_END[[:space:]]*-->/)) {
        p = pos + RSTART - 1
        if (p > 1 && substr(line, p - 1, 1) ~ /[A-Za-z0-9_]/) { pos = p + 1; continue }
        n++; pos = p + RLENGTH
    }
    return n
}
function readme_generated(   nl, L, k, h, o, nn, t) {
    nl = split(hunk, L, "\n")
    h = L[1]
    if (h !~ /^@@ -[0-9]+(,[0-9]+)? \+[0-9]+(,[0-9]+)? @@/) return 0
    sub(/^@@ -/, "", h); o = h + 0
    sub(/^[^+]*\+/, "", h); nn = h + 0
    load(oldr); load(newr)
    for (k = 2; k <= nl; k++) {
        t = substr(L[k], 1, 1)
        if (t == "-") { if (!((oldr, o) in BLK)) return 0; o++ }
        else if (t == "+") { if (!((newr, nn) in BLK)) return 0; nn++ }
        else if (t == " ") { o++; nn++ }
    }
    return 1
}
function blank_only(   nl, L, k, t, s, r, a) {
    nl = split(hunk, L, "\n"); r = ""; a = ""
    for (k = 2; k <= nl; k++) {
        t = substr(L[k], 1, 1)
        if (t != "-" && t != "+") continue
        s = substr(L[k], 2); gsub(/[[:space:]]/, "", s)
        if (t == "-") r = r s; else a = a s
    }
    return r == a
}
function flush(   i, key, hh) {
    if (!inhunk) return
    inhunk = 0
    if (count) {
        if (path in GEN) return
        for (i = 1; i <= npre; i++) if (index(path, PRE[i]) == 1) return
        if (blank_only()) return
        if (path == "README.md" && readme_generated()) return
    }
    key = sprintf("%040d", n++)
    printf "commit %s\n%s\n%s", key, hdr, hunk > feed
    hh = hhdr; gsub(/\t/, " ", hh)
    printf "%s\t%s\t%s\t%s\n", key, label, path, hh > index_
}
BEGIN {
    ng = split(gen_paths, G, " ")
    for (i = 1; i <= ng; i++) GEN[G[i]] = 1
    npre = split(gen_prefixes, PRE, " ")
    n = 0
}
/^commit / && NF == 2 && ($2 == "cumulative" || (length($2) == 40 && $2 !~ /[^0-9a-f]/)) {
    flush(); infile = 0; label = $2
    if (label == "cumulative") { oldr = mb; newr = headrev } else { oldr = label "^"; newr = label }
    next
}
/^diff --git / { flush(); infile = 1; hdr = $0; path = ""; next }
!infile { next }
/^@@/ { flush(); inhunk = 1; hunk = $0 "\n"; hhdr = $0; next }
inhunk { hunk = hunk $0 "\n"; next }
{
    hdr = hdr "\n" $0
    p = substr($0, 5)
    if (substr($0, 1, 4) == "+++ ") {
        if (substr(p, 1, 2) == "b/") path = substr(p, 3)
        else if (substr(p, 1, 3) == "\"b/") path = "\"" substr(p, 4)
    } else if (substr($0, 1, 4) == "--- " && path == "") {
        if (substr(p, 1, 2) == "a/") path = substr(p, 3)
        else if (substr(p, 1, 3) == "\"a/") path = "\"" substr(p, 4)
    }
}
END { flush() }
AWK
)

# split_hunks <count> <mb> <headrev> <raw> <feed> <index>
split_hunks() {
    LC_ALL=C awk -v count="$1" -v mb="$2" -v headrev="$3" -v feed="$5" -v index_="$6" \
        -v gen_paths="$GENERATED_PATHS" -v gen_prefixes="$GENERATED_PREFIXES" \
        "$SPLIT_AWK" "$4" || die2 "splitting $4 into hunks failed"
    touch "$5" "$6"
}

# Joins hunk ids to the index: drops a hunk with no id or one the base already
# landed, keeps the first place each id occurs, and writes each commit's
# counted hunks as one patch (keyed by the commit sha) for its commit-level id.
ASSEMBLE_AWK=$(cat <<'AWK'
BEGIN { FS = "\t" }
FILENAME == landed { LANDED[$1] = 1; next }
FILENAME == hpid { split($0, a, " "); PID[a[2]] = a[1]; next }
FILENAME == index_ {
    if (!($1 in PID) || (PID[$1] in LANDED)) next
    if (!(PID[$1] in SEEN)) { SEEN[PID[$1]] = 1; printf "%s\thunk\t%s\t%s\t%s\n", PID[$1], $2, $3, $4 > out }
    if ($2 != "cumulative") KEEP[$1] = $2
    next
}
FILENAME == feed {
    if (substr($0, 1, 7) == "commit " && length($0) == 47) {
        lab = (substr($0, 8) in KEEP) ? KEEP[substr($0, 8)] : ""
        if (lab != "" && !(lab in BUF)) { ORDER[++no] = lab; BUF[lab] = "" }
        next
    }
    if (lab != "") BUF[lab] = BUF[lab] $0 "\n"
    next
}
END {
    for (i = 1; i <= no; i++) printf "commit %s\n%s", ORDER[i], BUF[ORDER[i]] > cfeed
}
AWK
)

# ids_of <pr> <base-rev> <head-rev> <out>: one PR's ids, "pid level label path hunk"
# (tab-separated), one row per id, at the first place it occurs: hunks first.
ids_of() {
    local pr=$1 base=$2 head=$3 out=$4 w mb i
    local -a paths
    w="$(mktemp -d "$TD/ids.XXXXXX")"
    mb="$(gitp merge-base "$base" "$head")" || die2 "PR #$pr: git merge-base $base $head failed"
    gitp log -p --no-merges --right-only --format='commit %H' "${DIFF_FLAGS[@]}" "$base...$head" -- \
        > "$w/raw" || die2 "PR #$pr: git log $base...$head failed"
    printf 'commit cumulative\n' >> "$w/raw"
    gitp diff "${DIFF_FLAGS[@]}" "$mb" "$head" -- >> "$w/raw" || die2 "PR #$pr: git diff $mb $head failed"
    split_hunks 1 "$mb" "$head" "$w/raw" "$w/feed" "$w/index"
    gitp patch-id --stable < "$w/feed" > "$w/hpid" || die2 "PR #$pr: git patch-id failed"

    # What the base gained since the PR forked, on the PR's own paths, is no PR's duplicate.
    LC_ALL=C awk -F'\t' '$3 != "" { print $3 }' "$w/index" | LC_ALL=C sort -u > "$w/paths" \
        || die2 "PR #$pr: listing paths failed"
    mapfile -t paths < "$w/paths"
    : > "$w/lraw"
    if [ "${#paths[@]}" -gt 0 ]; then
        if LC_ALL=C grep -q '^"' "$w/paths"; then
            # A quoted path is not a pathspec as printed: read every path the base changed.
            gitp log -p --no-merges --format='commit %H' "${DIFF_FLAGS[@]}" "$mb..$base" -- \
                > "$w/lraw" || die2 "PR #$pr: git log $mb..$base failed"
        else
            for ((i = 0; i < ${#paths[@]}; i += 200)); do  # bounded argv
                gitp --literal-pathspecs log -p --no-merges --format='commit %H' "${DIFF_FLAGS[@]}" \
                    "$mb..$base" -- "${paths[@]:i:200}" >> "$w/lraw" \
                    || die2 "PR #$pr: git log $mb..$base failed"
            done
        fi
    fi
    split_hunks 0 "" "" "$w/lraw" "$w/lfeed" "$w/lindex"
    gitp patch-id --stable < "$w/lfeed" > "$w/lpid" || die2 "PR #$pr: git patch-id (landed) failed"
    cut -d' ' -f1 "$w/lpid" > "$w/landed" || die2 "PR #$pr: reading landed ids failed"

    LC_ALL=C awk -v landed="$w/landed" -v hpid="$w/hpid" -v index_="$w/index" -v feed="$w/feed" \
        -v out="$w/hunks" -v cfeed="$w/cfeed" "$ASSEMBLE_AWK" \
        "$w/landed" "$w/hpid" "$w/index" "$w/feed" || die2 "PR #$pr: assembling ids failed"
    touch "$w/hunks" "$w/cfeed"
    gitp patch-id --stable < "$w/cfeed" > "$w/cpid" || die2 "PR #$pr: git patch-id (commits) failed"
    LC_ALL=C awk -v hunks="$w/hunks" '
        FILENAME == hunks { HAVE[substr($0, 1, index($0, "\t") - 1)] = 1; print; next }
        { split($0, a, " "); if (!(a[1] in HAVE)) { HAVE[a[1]] = 1; printf "%s\tcommit\t%s\t\t\n", a[1], a[2] } }
    ' "$w/hunks" "$w/cpid" | LC_ALL=C sort > "$out" || die2 "PR #$pr: writing ids failed"
}

# find_dups <s> <o> <mine> <theirs> <rows>: one row per shared id, in id order.
find_dups() {
    LC_ALL=C awk -F'\t' -v S="$1" -v O="$2" -v theirs="$4" '
        FILENAME == theirs { T[$1] = $0; next }
        ($1 in T) {
            split(T[$1], b, "\t")
            k = $3 SUBSEP $4 SUBSEP $5 SUBSEP b[3]
            if (k in SEEN) next
            SEEN[k] = 1
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", S, O, $1, $2, $3, $4, $5, b[2], b[3]
        }
    ' "$4" "$3" >> "$5" || die2 "comparing PR #$1 with PR #$2 failed"
}

# engine <prs.tsv> <oid|pr-ref> <base-prefix> <subject|ALL> <rows>
# Writes the duplicate rows and exits 0; any failure exits non-zero, which the
# caller turns into rc 2. Runs as its own process so errexit is always live.
engine() {
    local tsv=$1 hmode=$2 bprefix=$3 subj=$4 rows=$5 n h b s i j
    local -a num head base stk
    REPO="${DUP_REPO:?engine needs DUP_REPO}"
    TD="$(mktemp -d)"
    trap 'rm -rf -- "${TD:?}"' EXIT
    while IFS=$'\t' read -r n h b s; do
        num+=("$n"); stk+=("$s")
        if [ "$hmode" = pr-ref ]; then head+=("refs/dup001/pr/$n"); else head+=("$h"); fi
        base+=("$bprefix$b")
    done < "$tsv"
    : > "$rows"
    ids() {  # ids <i>: cached ids file of PR i
        [ -f "$TD/pr.${num[$1]}" ] || ids_of "${num[$1]}" "${base[$1]}" "${head[$1]}" "$TD/pr.${num[$1]}"
    }
    pair() {  # pair <i> <j>: subject i against j, unless the pair is a declared stack
        case ",${stk[$1]}," in *",${num[$2]},"*) return 0 ;; esac
        case ",${stk[$2]}," in *",${num[$1]},"*) return 0 ;; esac
        ids "$1"; ids "$2"
        find_dups "${num[$1]}" "${num[$2]}" "$TD/pr.${num[$1]}" "$TD/pr.${num[$2]}" "$rows"
    }
    if [ "$subj" = ALL ]; then
        for ((i = 0; i < ${#num[@]}; i++)); do
            for ((j = i + 1; j < ${#num[@]}; j++)); do pair "$i" "$j"; done
        done
    else
        s=-1
        for ((i = 0; i < ${#num[@]}; i++)); do
            if [ "${num[$i]}" = "$subj" ]; then s=$i; fi
        done
        [ "$s" -ge 0 ] || die2 "PR #$subj is not in the open-PR list"
        for ((j = 0; j < ${#num[@]}; j++)); do [ "$j" -eq "$s" ] || pair "$s" "$j"; done
    fi
}

# render <rows>: detail rows (capped per pair), then one SUMMARY line per pair.
render() {
    local sumf
    sumf="$(mktemp "$TD/sum.XXXXXX")"
    LC_ALL=C awk -F'\t' -v cap="$DETAIL_PER_PAIR" -v sumf="$sumf" -v q="'" '
        {
            pair = $1 SUBSEP $2
            if (!(pair in SHOWN)) { NP++; PA[NP] = $1; PB[NP] = $2; KEY[NP] = pair }
            SHOWN[pair]++
            if (SHOWN[pair] <= cap) {
                tail = ($4 == "hunk") ? " " $6 " " $7 : ""
                printf "DUP-001: PR #%s %s %s%s duplicates PR #%s %s %s (patch-id %s)\n", \
                    $1, $4, substr($5, 1, 10), tail, $2, $8, substr($9, 1, 10), substr($3, 1, 12)
            }
            if ($4 == "commit") { C[pair]++; if ($5 == $9) SAME[pair]++ }
            else { H[pair]++; if (!((pair, $6) in F)) { F[pair, $6] = 1; NFILE[pair]++ } }
        }
        END {
            for (i = 1; i <= NP; i++) {
                p = KEY[i]; more = ""; stack = ""
                if (SHOWN[p] > cap) more = sprintf("; %d more row(s) not printed", SHOWN[p] - cap)
                if (SAME[p] > 0) stack = sprintf(" -- %d under the SAME sha: one branch is built on the" \
                    " other; declare %sstacked-on: #%s%s or rebase off it", SAME[p], q, PB[i], q)
                printf "%s\t%s\tDUP-001 SUMMARY: PR #%s vs PR #%s: %d commit(s), %d hunk(s) in %d file(s)%s%s\n", \
                    PA[i], PB[i], PA[i], PB[i], C[p] + 0, H[p] + 0, NFILE[p] + 0, more, stack > sumf
            }
        }
    ' "$1" || die2 "rendering the report failed"
    LC_ALL=C sort -t "$(printf '\t')" -k1,1n -k2,2n "$sumf" | cut -f3- || die2 "rendering the summary failed"
}

# to_tsv <open.json> <prs.tsv>: number, head oid, base branch, stacked-on list.
to_tsv() {
    jq -r "$TSV_JQ" "$1" > "$2" || die2 "cannot read the open-PR list"
    LC_ALL=C awk -F'\t' 'NF != 4 || $1 !~ /^[0-9]+$/ || $2 == "" || $3 == "" { bad = 1 } END { exit bad }' "$2" \
        || die2 "the open-PR list has a row without a number, head or base"
}

# judge <prs.tsv> <oid|pr-ref> <base-prefix> <subject|ALL>: prints the report;
# returns 0 (no duplicate) or 1 (duplicate); exits 2 when the engine failed.
judge() {
    local rows rc=0
    rows="$(mktemp "$TD/rows.XXXXXX")"
    DUP_REPO="$REPO" bash "$SELF" --engine "$1" "$2" "$3" "$4" "$rows" || rc=$?
    [ "$rc" -eq 0 ] || die2 "the engine stopped (rc $rc)"
    [ -s "$rows" ] || return 0
    render "$rows"
    return 1
}

# ---------------------------------------------------------------- real mode --

# read_open_prs: the open-PR list as $TD/prs.tsv, and every head and base fetched.
read_open_prs() {
    local -a refspecs
    if ! gh pr list --state open --limit 500 \
            --json number,headRefOid,baseRefName,body > "$TD/open.json"; then
        die2 "cannot list open PRs (gh failed)"
    fi
    [ "$(jq length "$TD/open.json")" -lt 500 ] || die2 "the open-PR list hit its 500-row limit"
    to_tsv "$TD/open.json" "$TD/prs.tsv"
    if [ "$(git -C "$REPO" rev-parse --is-shallow-repository)" = true ]; then
        git -C "$REPO" fetch --quiet --no-tags --unshallow origin \
            || die2 "cannot unshallow the checkout (git fetch failed)"
    fi
    # Heads by their pull ref, not the listed oid: a PR pushed between the list and
    # the fetch is read at its newest head instead of failing on a sha not fetched.
    mapfile -t refspecs < <(LC_ALL=C awk -F'\t' '
        !($3 in B) { B[$3] = 1; print "+refs/heads/" $3 ":refs/dup001/base/" $3 }
        { print "+refs/pull/" $1 "/head:refs/dup001/pr/" $1 }' "$TD/prs.tsv")
    [ "${#refspecs[@]}" -gt 0 ] || return 0
    git -C "$REPO" fetch --quiet --no-tags origin "${refspecs[@]}" \
        || die2 "cannot fetch the open PR heads (git fetch failed)"
}

real() {  # real <pr|ALL>
    local rc=0 n
    REPO="$(git rev-parse --show-toplevel)" || die2 "not inside a git checkout"
    TD="$(mktemp -d)"
    trap 'rm -rf -- "${TD:?}"' EXIT
    read_open_prs
    n="$(wc -l < "$TD/prs.tsv")"
    judge "$TD/prs.tsv" pr-ref refs/dup001/base/ "$1" || rc=$?
    if [ "$1" = ALL ]; then
        case "$rc" in
            0) echo "PASS: DUP-001 no two of the $n open PR(s) share a patch-id" ;;
            *) echo "FAIL: DUP-001 open PRs carry the same changes (above)." >&2 ;;
        esac
    else
        case "$rc" in
            0) echo "PASS: DUP-001 PR #$1 shares no patch-id with the $((n - 1)) other open PR(s)" ;;
            *) echo "FAIL: DUP-001 PR #$1 carries changes another open PR already carries (above)." \
                    "Drop the copy, close the PR it was folded from, or declare 'stacked-on: #N'." >&2 ;;
        esac
    fi
    exit "$rc"
}

pr_from_event() {
    local ev="${GITHUB_EVENT_PATH:-}"
    [ -n "$ev" ] && [ -f "$ev" ] || return 0
    jq -r '.pull_request.number // empty' "$ev" || die2 "cannot read $ev"
}

# ---------------------------------------------------------------- self-test --

# Each row builds branches in a scratch repo off `main`, runs the engine with
# the named subject (PR #1) and other (PR #2), and compares the exit status
# and the reported level. must-match rows expect 1; must-not-match rows 0.
self_test() {
    local r fail=0 total=0 fix
    TD="$(mktemp -d)"
    trap 'rm -rf -- "${TD:?}"' EXIT
    r="$TD/repo"
    REPO="$r"
    git init --quiet -b main "$r"
    g() { git -C "$r" -c user.name=t -c user.email=t@t -c commit.gpgsign=false \
              -c core.hooksPath=/dev/null "$@"; }
    commit() { g add -A && g commit --quiet -m "$1"; }
    seq 1 40 > "$r/a.txt"; seq 1 40 > "$r/b.txt"; seq 1 300 > "$r/c.txt"
    printf 'x\n<!-- CONTRACT_COUNT_START -->10<!-- CONTRACT_COUNT_END -->\ny\n' > "$r/README.md"
    cp "$r/README.md" "$r/notes.md"
    mkdir -p "$r/docs/roadmaps" "$r/contracts/witness"
    echo 'n: 1' > "$r/docs/roadmaps/roadmap.yaml"
    printf '{\n  "a": 1,\n  "b": 2,\n  "c": 3,\n  "cpu_ms": 2\n}\n' > "$r/contracts/witness/w.json"
    seq 1 10 > "$r/$(printf 'caf\303\251.txt')"
    commit init

    # A: the original fix (two hunks in a.txt)
    g checkout --quiet -b A main
    sed -i -e 's/^5$/five/' -e 's/^30$/thirty/' "$r/a.txt"; commit "fix: a"
    fix="$(g rev-parse HEAD)"
    # copy: the same commit, same message, on another branch
    g checkout --quiet -b copy main; g cherry-pick "$fix" >/dev/null
    # pick: cherry-picked, message reworded, plus an unrelated later commit
    g checkout --quiet -b pick main; g cherry-pick "$fix" >/dev/null
    g commit --quiet --amend >/dev/null -m "chore: totally different words"
    sed -i 's/^20$/twenty/' "$r/b.txt"; commit "other work"
    # edit: copy then edit -- the hunk at line 5 kept, line 30 changed, b.txt added
    g checkout --quiet -b edit main
    sed -i -e 's/^5$/five/' -e 's/^30$/THIRTY/' "$r/a.txt"; sed -i 's/^9$/nine/' "$r/b.txt"
    commit "fix: a, edited"
    # squash: the fix squashed into one larger commit with other work
    g checkout --quiet -b squash main
    sed -i -e 's/^5$/five/' -e 's/^30$/thirty/' "$r/a.txt"; sed -i 's/^33$/x33/' "$r/b.txt"
    commit "big squash"
    # regen: an identical generated regen on two PRs (roadmap + README block + witness)
    g checkout --quiet -b regen1 main
    echo 'n: 2' > "$r/docs/roadmaps/roadmap.yaml"
    sed -i 's/START -->10</START -->11</' "$r/README.md"; commit "regen"
    g checkout --quiet -b regen2 main
    echo 'n: 2' > "$r/docs/roadmaps/roadmap.yaml"
    sed -i 's/START -->10</START -->11</' "$r/README.md"
    sed -i 's/^12$/twelve/' "$r/b.txt"; commit "regen + own work"
    g checkout --quiet -b wit1 main; sed -i 's/"cpu_ms": 2/"cpu_ms": 6/' "$r/contracts/witness/w.json"
    sed -i 's/^14$/fourteen/' "$r/b.txt"; commit "witness"
    g checkout --quiet -b wit2 main; sed -i 's/"cpu_ms": 2/"cpu_ms": 6/' "$r/contracts/witness/w.json"
    sed -i 's/^36$/thirty-six/' "$r/b.txt"; commit "witness again"
    # readme-prose: the same README edit OUTSIDE a generated block IS a duplicate
    g checkout --quiet -b prose1 main; sed -i 's/^x$/intro/' "$r/README.md"; commit "prose"
    g checkout --quiet -b prose2 main; sed -i 's/^x$/intro/' "$r/README.md"; commit "prose again"
    # samefile-elsewhere: the same edit in a DIFFERENT file is not a duplicate
    g checkout --quiet -b elsewhere main; sed -i 's/^5$/five/' "$r/b.txt"; commit "b five"
    # unrelated work
    g checkout --quiet -b unrelated main; sed -i 's/^25$/x25/' "$r/b.txt"; commit "unrelated"
    # stack: B stacked on A carries A's commit by construction
    g checkout --quiet -b stack A; sed -i 's/^38$/x38/' "$r/b.txt"; commit "on top of A"
    # landed: A's fix already on main under another sha; a PR re-carrying it is
    # not a duplicate of a second PR that merged main
    g checkout --quiet -b main2 main; g cherry-pick "$fix" >/dev/null
    g commit --quiet --amend >/dev/null -m "landed"
    g checkout --quiet -b landedA main; g cherry-pick "$fix" >/dev/null
    g commit --quiet --amend >/dev/null -m "re-carried"
    g checkout --quiet -b landedB main2; sed -i 's/^2$/two/' "$r/b.txt"; commit "after landing"
    # many: 30 separate hunks carried by two PRs -- more than the detail cap
    g checkout --quiet -b many1 main; sed -i '0~10s/$/ edited/' "$r/c.txt"; commit many
    g checkout --quiet -b many2 main; sed -i '0~10s/$/ edited/' "$r/c.txt"; commit "many again"
    # landedC: a second stale PR that ALSO still carries the since-landed fix
    g checkout --quiet -b landedC main; g cherry-pick "$fix" >/dev/null
    g commit --quiet --amend >/dev/null -m "also re-carried"
    sed -i 's/^3$/three/' "$r/b.txt"; commit "own work"
    # twostep: the same end state as A's line-5 hunk, reached in two commits --
    # only the PR's cumulative diff carries the finished hunk
    g checkout --quiet -b twostep main; sed -i 's/^5$/fiv/' "$r/a.txt"; commit "step 1"
    sed -i 's/^fiv$/five/' "$r/a.txt"; commit "step 2"
    # block-elsewhere: a START/END block outside README.md is NOT generated
    g checkout --quiet -b blk1 main; sed -i 's/START -->10</START -->11</' "$r/notes.md"; commit blk
    g checkout --quiet -b blk2 main; sed -i 's/START -->10</START -->11</' "$r/notes.md"; commit "blk again"
    # whitespace-only: identical whitespace hunks on two PRs are ignored
    g checkout --quiet -b ws1 main; sed -i 's/^17$/17 /' "$r/a.txt"; commit ws
    g checkout --quiet -b ws2 main; sed -i 's/^17$/17 /' "$r/a.txt"; commit "ws again"
    # quoted: the same edit to a non-ASCII path (git prints it quoted) on two PRs
    g checkout --quiet -b q1 main; sed -i 's/^4$/four/' "$r/$(printf 'caf\303\251.txt')"; commit q
    g checkout --quiet -b q2 main; sed -i 's/^4$/four/' "$r/$(printf 'caf\303\251.txt')"; commit "q again"
    # qmain: the quoted-path fix landed under another sha (the landed read cannot
    # take a quoted path as a pathspec, so it must read the base unfiltered)
    g checkout --quiet -b qmain main; g cherry-pick "$(g rev-parse q1)" >/dev/null
    g commit --quiet --amend >/dev/null -m "q landed"
    g checkout --quiet main

    # spec <base> <head1> <body1> <head2> <body2> [<head3> <body3>]: gh-shaped JSON
    spec() {
        local b=$1; shift
        jq -n --arg b "$b" '$ARGS.positional | [range(0; length; 2) as $i
            | {number: ($i / 2 + 1), headRefOid: .[$i], baseRefName: $b, body: .[$i + 1]}]' \
            --args "$@"
    }
    ok() { echo "  ok   [$1] rc=$2"; }
    bad() { echo "  FAIL [$1] $2"; fail=$((fail + 1)); }

    # row <expect-rc> <expect-level|-> <name> <base> <subject> <subject-body> <other> <other-body>
    #     [<output must match ERE|-> [<output must NOT match ERE>]]
    row() {
        local want="$1" level="$2" name="$3" base="$4" s="$5" sb="$6" o="$7" ob="$8"
        local must="${9:--}" mustnt="${10:-}" out rc=0
        total=$((total + 1))
        spec "$base" "$(g rev-parse "$s")" "$sb" "$(g rev-parse "$o")" "$ob" > "$TD/open.json"
        to_tsv "$TD/open.json" "$TD/prs.tsv"
        out="$(judge "$TD/prs.tsv" oid refs/heads/ 1 2>&1)" || rc=$?
        if [ "$rc" != "$want" ]; then bad "$name" "rc=$rc want=$want: $out"; return; fi
        if [ "$level" != - ] && ! printf '%s\n' "$out" | grep -q "PR #1 $level .*duplicates PR #2"; then
            bad "$name" "no '$level' row naming both PRs: $out"; return
        fi
        if [ "$must" != - ] && ! printf '%s\n' "$out" | grep -Eq -- "$must"; then
            bad "$name" "output lacks /$must/: $out"; return
        fi
        if [ -n "$mustnt" ] && printf '%s\n' "$out" | grep -Eq -- "$mustnt"; then
            bad "$name" "output has /$mustnt/: $out"; return
        fi
        ok "$name" "$rc"
    }
    echo "must-match:"
    row 1 commit "pure copy"                        main copy "" A ""
    row 1 commit "cherry-pick, different message"   main pick "" A ""
    row 1 hunk   "copy then edit (hunk kept)"       main edit "" A ""
    row 1 hunk   "squashed into a bigger commit"    main squash "" A ""
    row 1 hunk   "README prose outside a block"     main prose2 "" prose1 ""
    row 1 commit "stack declared by the WRONG pr#"  main stack "stacked-on: #9" A "" \
        'SUMMARY: PR #1 vs PR #2: 1 commit\(s\), .* 1 under the SAME sha.*stacked-on: #2'
    row 1 commit "copy is summarized, not called a stack" main pick "" A "" \
        'SUMMARY: PR #1 vs PR #2: 1 commit\(s\), 2 hunk\(s\) in 1 file\(s\)$' 'SAME sha'
    row 1 hunk   "detail capped, summary counts all" main many2 "" many1 "" \
        'SUMMARY: PR #1 vs PR #2: 1 commit\(s\), 30 hunk\(s\) in 1 file\(s\); 11 more row\(s\) not printed'
    row 1 hunk   "same end state in two commits"    main twostep "" A ""
    row 1 hunk   "START/END block outside README"   main blk2 "" blk1 ""
    row 1 hunk   "quoted (non-ASCII) path"          main q2 "" q1 "" 'caf\\303\\251'
    echo "must-not-match:"
    row 0 - "identical generated regen"             main regen2 "" regen1 ""
    row 0 - "identical witness regen"               main wit2 "" wit1 ""
    row 0 - "same edit, different file"             main elsewhere "" A ""
    row 0 - "unrelated work"                        main unrelated "" A ""
    row 0 - "declared stack (subject body)"         main stack "stacked-on: #2" A ""
    row 0 - "declared stack (other body)"           main A "" stack "Stacked-on: #1"
    row 0 - "fix already on base under other sha"   main2 landedA "" landedB ""
    row 0 - "two stale PRs both carry a landed fix" main2 landedA "" landedC ""
    row 0 - "whitespace-only hunk"                  main ws2 "" ws1 ""
    row 0 - "landed fix on a quoted path"           qmain q2 "" q1 ""

    echo "output, engine and wrapper:"
    local n rc out
    # The cap: exactly DETAIL_PER_PAIR detail rows for a pair that has more.
    spec main "$(g rev-parse many2)" "" "$(g rev-parse many1)" "" > "$TD/open.json"
    to_tsv "$TD/open.json" "$TD/prs.tsv"
    n="$( (judge "$TD/prs.tsv" oid refs/heads/ 1 || true) | grep -c '^DUP-001: PR #1 ')" || true
    total=$((total + 1))
    if [ "$n" = 20 ]; then ok "detail rows capped at 20" "n=$n"; else bad "detail rows capped at 20" "n=$n"; fi
    # A head the repo does not have: the engine cannot read it, so 2, never 0 or 1.
    spec main "$(g rev-parse A)" "" 0123456789abcdef0123456789abcdef01234567 "" > "$TD/open.json"
    to_tsv "$TD/open.json" "$TD/prs.tsv"
    rc=0; out="$(judge "$TD/prs.tsv" oid refs/heads/ 1 2>&1)" || rc=$?
    total=$((total + 1))
    if [ "$rc" = 2 ]; then ok "unreadable head -> rc 2" "$rc"; else bad "unreadable head -> rc 2" "rc=$rc: $out"; fi
    # --all: every pair once; the declared stack is skipped, the copy is not.
    spec main "$(g rev-parse A)" "" "$(g rev-parse copy)" "" "$(g rev-parse stack)" "stacked-on: #1" \
        > "$TD/open.json"
    to_tsv "$TD/open.json" "$TD/prs.tsv"
    rc=0; out="$(judge "$TD/prs.tsv" oid refs/heads/ ALL 2>&1)" || rc=$?
    total=$((total + 1))
    if [ "$rc" = 1 ] && [[ "$out" == *"SUMMARY: PR #1 vs PR #2:"* ]] \
            && [[ "$out" == *"SUMMARY: PR #2 vs PR #3:"* ]] && [[ "$out" != *"PR #1 vs PR #3"* ]]; then
        ok "--all: every pair once, declared stack skipped" "$rc"
    else bad "--all: every pair once, declared stack skipped" "rc=$rc: $out"; fi

    # The real path end to end: gh (stubbed) lists the PRs, the heads come from
    # refs/pull/N/head on an `origin`, the bases from refs/heads.
    local o="$TD/origin.git" w="$TD/work" stub="$TD/bin"
    git init --quiet --bare -b main "$o"
    g push --quiet "$o" main "A:refs/pull/1/head" "copy:refs/pull/2/head" "unrelated:refs/pull/3/head"
    git clone --quiet "$o" "$w"
    mkdir -p "$stub"
    printf '#!/bin/sh\ncat "%s"\n' "$TD/gh.json" > "$stub/gh"; chmod +x "$stub/gh"
    spec main "$(g rev-parse A)" "" "$(g rev-parse copy)" "" "$(g rev-parse unrelated)" "" > "$TD/gh.json"
    rc=0; out="$(cd "$w" && PATH="$stub:$PATH" bash "$SELF" --pr 3 2>&1)" || rc=$?
    total=$((total + 1))
    if [ "$rc" = 0 ] && [[ "$out" == *"PASS: DUP-001 PR #3 shares no patch-id with the 2 other"* ]]; then
        ok "real path: clean PR passes" "$rc"; else bad "real path: clean PR passes" "rc=$rc: $out"; fi
    printf '{"pull_request":{"number":2}}' > "$TD/event.json"
    rc=0; out="$(cd "$w" && GITHUB_EVENT_PATH="$TD/event.json" PATH="$stub:$PATH" bash "$SELF" 2>&1)" || rc=$?
    total=$((total + 1))
    if [ "$rc" = 1 ] && [[ "$out" == *"PR #2 commit "*"duplicates PR #1 commit"* ]]; then
        ok "real path: PR read from the event, copy fails" "$rc"
    else bad "real path: PR read from the event, copy fails" "rc=$rc: $out"; fi
    rc=0; out="$(cd "$w" && PATH="$stub:$PATH" bash "$SELF" --all 2>&1)" || rc=$?
    total=$((total + 1))
    if [ "$rc" = 1 ] && [[ "$out" == *"SUMMARY: PR #1 vs PR #2:"* ]] && [[ "$out" != *"PR #3"* ]]; then
        ok "real path: --all names the one copied pair" "$rc"
    else bad "real path: --all names the one copied pair" "rc=$rc: $out"; fi
    # No PR under test (push, merge_group, a bare local run): nothing measured, rc 2.
    printf '{"ref":"refs/heads/main"}' > "$TD/event.json"
    rc=0; out="$(cd "$w" && GITHUB_EVENT_PATH="$TD/event.json" PATH="$stub:$PATH" bash "$SELF" 2>&1)" || rc=$?
    total=$((total + 1))
    if [ "$rc" = 2 ] && [[ "$out" == *"no pull request under test"* ]]; then
        ok "no PR in the event -> rc 2, says why" "$rc"
    else bad "no PR in the event -> rc 2, says why" "rc=$rc: $out"; fi
    # A PR the list does not hold: 2.
    rc=0; out="$(cd "$w" && PATH="$stub:$PATH" bash "$SELF" --pr 9 2>&1)" || rc=$?
    total=$((total + 1))
    if [ "$rc" = 2 ] && [[ "$out" == *"PR #9 is not in the open-PR list"* ]]; then
        ok "PR not in the open list -> rc 2" "$rc"; else bad "PR not in the open list -> rc 2" "rc=$rc: $out"; fi
    # gh cannot list the open PRs: refuse (2), never pass unread (0) or blame the PR (1).
    printf '#!/bin/sh\nexit 1\n' > "$stub/gh"
    rc=0; out="$(cd "$w" && PATH="$stub:$PATH" bash "$SELF" --pr 1 2>&1)" || rc=$?
    total=$((total + 1))
    if [ "$rc" = 2 ] && [[ "$out" == *"cannot list open PRs"* ]]; then
        ok "gh fails -> rc 2" "$rc"; else bad "gh fails -> rc 2" "rc=$rc: $out"; fi
    # gh answers but the fetch fails (this repo has no `origin`): 2.
    printf '#!/bin/sh\ncat "%s"\n' "$TD/gh.json" > "$stub/gh"
    rc=0; out="$(cd "$r" && PATH="$stub:$PATH" bash "$SELF" --pr 1 2>&1)" || rc=$?
    total=$((total + 1))
    if [ "$rc" = 2 ] && [[ "$out" == *"cannot fetch the open PR heads"* ]]; then
        ok "git fetch fails -> rc 2" "$rc"; else bad "git fetch fails -> rc 2" "rc=$rc: $out"; fi

    if [ "$fail" -ne 0 ]; then
        echo "FAIL: DUP-001 self-test $fail of $total row(s) failed"; exit 1
    fi
    echo "PASS: DUP-001 self-test $total/$total rows"
}

case "${1:-}" in
    --self-test) [ "$#" -eq 1 ] || usage; self_test ;;
    --engine) [ "$#" -eq 6 ] || usage; engine "$2" "$3" "$4" "$5" "$6" ;;
    --all) [ "$#" -eq 1 ] || usage; real ALL ;;
    --pr) [ "$#" -eq 2 ] || usage
          case "$2" in ''|*[!0-9]*) usage ;; esac
          real "$2" ;;
    -h|--help) usage ;;
    "") pr="$(pr_from_event)"
        if [ -z "$pr" ]; then
            echo "DUP-001: no pull request under test (event: ${GITHUB_EVENT_NAME:-none}); nothing was measured" >&2
            exit 2
        fi
        real "$pr" ;;
    *) usage ;;
esac
