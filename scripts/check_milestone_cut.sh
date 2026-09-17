#!/usr/bin/env bash
# check_milestone_cut.sh <milestone-title> [--repo O/R] [--json OUT] | --self-test
#
# APR-RELEASE-001 section 4 (T-0, T-3) and 06x-release-schedule.md section 4
# steps 1 and 4 (PMAT-3445, #3445). The milestone a release train cuts must
# hold ZERO open items -- issues AND pull requests -- at the instant of the cut.
#
# WHY: v0.68.0 was tagged 2026-09-17T06:35Z while milestone 0.68.0 held two
# open items, one of them #3091, a P1 defect reopened 85 minutes before the
# tag. The T-5 reconcile reads the milestone at one instant, so an item
# reopened after it is invisible; the autopilot's tag step read no milestone
# at all, and its only read was the close step, after publish.
#
# THE RULE: exit 0 only when open_issues == 0. Pull requests count because
# the milestone is closed on open_issues and GitHub counts them there. An item
# that does not ride this train is CARRIED by moving it to the next milestone
# (`gh issue edit N --milestone <next>`, or `gh pr edit`) with a
# `slipped_from: X.Y.0` comment -- never by leaving it open in this one.
#
# WHEN: at the freeze, before the train's bump PR opens, and again
# immediately before `git tag`, after the bump PR has merged. Never while the
# bump PR is open: it sits in the milestone and reads RED.
#
# Exit 0 = zero open items in the milestone.
# Exit 1 = at least one open item; each is named with its remedy.
# Exit 2 = cannot judge, never a silent pass: gh or python3 missing, gh
#          unauthenticated, a read failed, the title matches 0 or 2+
#          milestones (compared literally), the milestone has no items at all,
#          or the listed items disagree with the milestone's own open_issues.
#
# No arguments runs the self-test, because guard_tree.sh runs every guard
# bare. The self-test builds fixtures and stubs gh on PATH; it never touches
# the network.

set -euo pipefail

SELF_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

usage() {
    printf 'usage: %s <milestone-title> [--repo O/R] [--json OUT] | --self-test\n' "$(basename "$0")" >&2
    exit 2
}

# judge_from_dir DIR TITLE [JSON_OUT]
# DIR holds milestones.jsonl and items.jsonl, one JSON object per line: the
# shape `gh api --paginate --jq '.[]'` writes, so a read of more than one page
# is never a concatenation of arrays. Prints the verdict and returns 0, 1 or 2.
judge_from_dir() {
    python3 -c '
import json, sys

d, title, json_out = sys.argv[1], sys.argv[2], sys.argv[3]

def env(msg):
    print("ENV: " + msg, file=sys.stderr)
    sys.exit(2)

def read_lines(name):
    rows = []
    try:
        with open(d + "/" + name, encoding="utf-8") as f:
            for n, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except ValueError:
                    env("%s line %d is not a JSON object" % (name, n))
                if not isinstance(obj, dict):
                    env("%s line %d is not a JSON object" % (name, n))
                rows.append(obj)
    except OSError as e:
        env("cannot read %s: %s" % (name, e))
    return rows

matches = [m for m in read_lines("milestones.jsonl") if m.get("title") == title]
if len(matches) != 1:
    env("title \"%s\" matches %d milestone(s); exactly 1 is required" % (title, len(matches)))
m = matches[0]
number = m.get("number")
open_n = m.get("open_issues")
closed_n = m.get("closed_issues")
if not isinstance(open_n, int) or not isinstance(closed_n, int):
    env("milestone \"%s\" carries no integer open_issues/closed_issues" % title)
if open_n + closed_n == 0:
    env("milestone \"%s\" (#%s) has no items at all -- the wrong milestone was named" % (title, number))

items = read_lines("items.jsonl")
if len(items) != open_n:
    env("milestone \"%s\" (#%s) reports open_issues=%d but the read listed %d item(s)" % (title, number, open_n, len(items)))
for it in items:
    if it.get("state") != "open" or (it.get("milestone") or {}).get("number") != number:
        env("listed item #%s is not an open item of milestone #%s" % (it.get("number"), number))

rows = []
for it in sorted(items, key=lambda i: i.get("number", 0)):
    rows.append({
        "number": it.get("number"),
        "kind": "pr" if "pull_request" in it else "issue",
        "title": it.get("title", ""),
        "labels": [lb.get("name", "") for lb in (it.get("labels") or [])],
        "url": it.get("html_url", ""),
    })

verdict = "RED" if rows else "PASS"
if json_out:
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({"milestone": title, "number": number, "open": open_n, "closed": closed_n,
                   "items": rows, "verdict": verdict}, f, indent=2, sort_keys=True)
        f.write("\n")

if not rows:
    print("PASS  milestone %s (#%s): 0 open, %d closed -- the cut may proceed" % (title, number, closed_n))
    sys.exit(0)

for r in rows:
    print("#%s %s [%s] %s" % (r["number"], r["kind"], ",".join(r["labels"]), r["title"]))
for r in rows:
    print("  remedy #%s: close it, or carry it: gh %s edit %s --milestone <next> && gh %s comment %s --body \"slipped_from: %s\""
          % (r["number"], r["kind"], r["number"], r["kind"], r["number"], title))
print("RED   milestone %s (#%s): %d open item(s) -- no tag until each is closed or carried" % (title, number, len(rows)))
sys.exit(1)
' "$1" "$2" "${3:-}"
}

# fetch_live REPO TITLE DIR -- writes DIR/milestones.jsonl and DIR/items.jsonl.
# A title that does not resolve to exactly one milestone leaves items.jsonl
# empty; judge_from_dir then names the resolution failure (exit 2).
fetch_live() {
    repo="$1"
    title="$2"
    dir="$3"
    gh api --paginate --jq '.[]' "repos/${repo}/milestones?state=all&per_page=100" > "${dir}/milestones.jsonl" || return 1
    number="$(python3 -c '
import json, sys
nums = []
with open(sys.argv[1], encoding="utf-8") as f:
    for raw in f:
        if raw.strip():
            obj = json.loads(raw)
            if obj.get("title") == sys.argv[2]:
                nums.append(obj.get("number"))
print(nums[0] if len(nums) == 1 else "")
' "${dir}/milestones.jsonl" "$title")" || return 1
    if [ -z "$number" ]; then
        : > "${dir}/items.jsonl"
        return 0
    fi
    gh api --paginate --jq '.[]' "repos/${repo}/issues?milestone=${number}&state=open&per_page=100" > "${dir}/items.jsonl" || return 1
}

# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

ST_FAILS=0
ST_CASES=0

# st_check CASE WANT_RC GOT_RC OUTFILE [NEEDLE ...]
st_check() {
    c="$1"
    want="$2"
    got="$3"
    out="$4"
    shift 4
    ST_CASES=$((ST_CASES + 1))
    if [ "$got" != "$want" ]; then
        printf 'FAIL %s: expected exit %s, got %s\n' "$c" "$want" "$got" >&2
        sed 's/^/    /' "$out" >&2
        ST_FAILS=$((ST_FAILS + 1))
        return 0
    fi
    for needle in "$@"; do
        if ! grep -qF -- "$needle" "$out"; then
            printf 'FAIL %s: output lacks [%s]\n' "$c" "$needle" >&2
            sed 's/^/    /' "$out" >&2
            ST_FAILS=$((ST_FAILS + 1))
        fi
    done
}

# st_judge FIXDIR CASE WANT_RC TITLE [NEEDLE ...]
st_judge() {
    fx="$1"
    c="$2"
    want="$3"
    t="$4"
    shift 4
    rc=0
    judge_from_dir "$fx" "$t" "" > "${fx}/out" 2>&1 || rc=$?
    st_check "$c" "$want" "$rc" "${fx}/out" "$@"
}

# st_ms TITLE NUMBER OPEN CLOSED -- one milestones.jsonl line
st_ms() {
    printf '{"title":"%s","number":%s,"open_issues":%s,"closed_issues":%s}\n' "$1" "$2" "$3" "$4"
}

# st_item NUMBER MILESTONE_NUMBER KIND STATE LABEL -- one items.jsonl line
st_item() {
    pr=""
    if [ "$3" = pr ]; then
        pr=',"pull_request":{"url":"u"}'
    fi
    printf '{"number":%s,"state":"%s","milestone":{"number":%s},"title":"item %s","labels":[{"name":"%s"}]%s}\n' \
        "$1" "$4" "$2" "$1" "$5" "$pr"
}

self_test() {
    fx="$(mktemp -d)" || return 2

    # S1 zero open, some closed -> PASS
    st_ms M 7 0 5 > "${fx}/milestones.jsonl"
    : > "${fx}/items.jsonl"
    st_judge "$fx" S1 0 M "PASS  milestone M (#7): 0 open, 5 closed"

    # S2 one open issue -> RED, named
    st_ms M 7 1 5 > "${fx}/milestones.jsonl"
    st_item 10 7 issue open bug > "${fx}/items.jsonl"
    st_judge "$fx" S2 1 M "#10 issue [bug] item 10" "gh issue edit 10 --milestone <next>" "1 open item(s)"

    # S3 one open PR -> RED, named as pr, remedy through gh pr
    st_item 11 7 pr open release > "${fx}/items.jsonl"
    st_judge "$fx" S3 1 M "#11 pr [release] item 11" "gh pr edit 11 --milestone <next>"

    # S4 an issue and a PR -> both named with their labels
    st_ms M 7 2 5 > "${fx}/milestones.jsonl"
    { st_item 11 7 pr open release; st_item 10 7 issue open P1; } > "${fx}/items.jsonl"
    st_judge "$fx" S4 1 M "#10 issue [P1] item 10" "#11 pr [release] item 11" "2 open item(s)"

    # S5 an empty milestone is the wrong milestone, not a pass
    st_ms M 7 0 0 > "${fx}/milestones.jsonl"
    : > "${fx}/items.jsonl"
    st_judge "$fx" S5 2 M "has no items at all"

    # S6 the read lost an item
    st_ms M 7 3 5 > "${fx}/milestones.jsonl"
    { st_item 10 7 issue open a; st_item 11 7 issue open b; } > "${fx}/items.jsonl"
    st_judge "$fx" S6 2 M "open_issues=3 but the read listed 2"

    # S7 a stale zero count cannot pass over a listed open item
    st_ms M 7 0 5 > "${fx}/milestones.jsonl"
    st_item 10 7 issue open a > "${fx}/items.jsonl"
    st_judge "$fx" S7 2 M "open_issues=0 but the read listed 1"

    # S8 no milestone carries the title
    st_ms Other 7 0 5 > "${fx}/milestones.jsonl"
    : > "${fx}/items.jsonl"
    st_judge "$fx" S8 2 M "matches 0 milestone(s)"

    # S9 two milestones carry the title
    { st_ms M 7 0 5; st_ms M 8 0 2; } > "${fx}/milestones.jsonl"
    st_judge "$fx" S9 2 M "matches 2 milestone(s)"

    # S10 the title is compared literally: dots are not wildcards, a prefix is not a match
    { st_ms 0a69b0 1 1 0; st_ms 0.69.0 2 0 3; } > "${fx}/milestones.jsonl"
    : > "${fx}/items.jsonl"
    st_judge "$fx" S10a 0 0.69.0 "PASS  milestone 0.69.0 (#2)"
    st_judge "$fx" S10b 2 0.69 "matches 0 milestone(s)"

    # S11 a listed item from another milestone, or a closed one, is a bad read
    st_ms M 7 1 5 > "${fx}/milestones.jsonl"
    st_item 10 8 issue open a > "${fx}/items.jsonl"
    st_judge "$fx" S11a 2 M "is not an open item of milestone #7"
    st_item 10 7 issue closed a > "${fx}/items.jsonl"
    st_judge "$fx" S11b 2 M "is not an open item of milestone #7"

    # S12 more than one page of items (150) -> every one counted
    st_ms M 7 150 5 > "${fx}/milestones.jsonl"
    : > "${fx}/items.jsonl"
    i=1000
    while [ "$i" -lt 1150 ]; do
        st_item "$i" 7 issue open a >> "${fx}/items.jsonl"
        i=$((i + 1))
    done
    st_judge "$fx" S12 1 M "#1000 issue" "#1149 issue" "150 open item(s)"

    # S16 --json records the verdict and the items
    st_ms M 7 1 5 > "${fx}/milestones.jsonl"
    st_item 10 7 issue open bug > "${fx}/items.jsonl"
    rc=0
    judge_from_dir "$fx" M "${fx}/receipt.json" > "${fx}/out" 2>&1 || rc=$?
    python3 -c '
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
ok = r["verdict"] == "RED" and r["open"] == 1 and [i["number"] for i in r["items"]] == [10]
print("json-receipt " + ("ok" if ok else "WRONG: %r" % r))
' "${fx}/receipt.json" >> "${fx}/out" 2>&1 || true
    st_check S16 1 "$rc" "${fx}/out" "json-receipt ok"

    # The live path, with gh stubbed on PATH. SELF_PATH is re-executed so main()
    # and fetch_live() are exercised exactly as a release step calls them.
    stub="${fx}/stub"
    mkdir -p "$stub"

    # S13 gh absent -> ENV. The PATH holds only the tools the script needs before its gh check.
    for tool in dirname basename python3; do
        ln -s "$(command -v "$tool")" "${stub}/${tool}"
    done
    rc=0
    env PATH="$stub" "$BASH" "$SELF_PATH" M > "${fx}/out" 2>&1 || rc=$?
    st_check S13 2 "$rc" "${fx}/out" "gh is not on PATH"
    rm -f "${stub}/dirname" "${stub}/basename" "${stub}/python3"

    # S14 gh present but unauthenticated -> ENV
    printf '#!/bin/sh\n[ "$1" = auth ] && exit 1\nexit 0\n' > "${stub}/gh"
    chmod +x "${stub}/gh"
    rc=0
    PATH="${stub}:${PATH}" bash "$SELF_PATH" M > "${fx}/out" 2>&1 || rc=$?
    st_check S14 2 "$rc" "${fx}/out" "gh is not authenticated"

    # S15 a stub gh serving the S2 fixture. It refuses a read that does not
    # paginate as JSON lines, and an items query that is not state=open for
    # the resolved milestone number -- so the live query itself is under test.
    cat > "${stub}/gh" <<'STUB'
#!/bin/sh
case "$1" in
    auth) exit 0 ;;
    api) ;;
    *) echo "stub gh: unexpected verb: $*" >&2; exit 9 ;;
esac
case " $* " in
    *' --paginate --jq .[] '*) ;;
    *) echo "stub gh: read without --paginate --jq .[]: $*" >&2; exit 9 ;;
esac
case "$*" in
    *'repos/o/r/milestones?state=all&per_page=100'*) cat "$STUB_DIR/milestones.jsonl" ;;
    *'repos/o/r/issues?milestone=7&state=open&per_page=100'*) cat "$STUB_DIR/items.jsonl" ;;
    *) echo "stub gh: unexpected read: $*" >&2; exit 9 ;;
esac
STUB
    chmod +x "${stub}/gh"
    rc=0
    STUB_DIR="$fx" PATH="${stub}:${PATH}" bash "$SELF_PATH" M --repo o/r > "${fx}/out" 2>&1 || rc=$?
    st_check S15 1 "$rc" "${fx}/out" "#10 issue [bug] item 10"

    rm -rf "${fx:?}"
    if [ "$ST_FAILS" -ne 0 ]; then
        printf 'self-test FAILED: %s of %s check(s).\n' "$ST_FAILS" "$ST_CASES" >&2
        return 1
    fi
    printf 'self-test OK: %s case(s).\n' "$ST_CASES"
    return 0
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

main() {
    if [ $# -eq 0 ] || [ "${1:-}" = "--self-test" ]; then
        self_test
        exit $?
    fi
    case "$1" in
        -h|--help) sed -n '2,33p' "$SELF_PATH"; exit 0 ;;
        -*) usage ;;
    esac

    title="$1"
    shift
    repo="paiml/aprender"
    json_out=""
    while [ $# -gt 0 ]; do
        [ $# -ge 2 ] || usage
        case "$1" in
            --repo) repo="$2" ;;
            --json) json_out="$2" ;;
            *) usage ;;
        esac
        shift 2
    done

    if ! command -v gh >/dev/null 2>&1; then
        printf 'ENV: gh is not on PATH.\n' >&2
        exit 2
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        printf 'ENV: python3 is not on PATH.\n' >&2
        exit 2
    fi
    if ! gh auth status >/dev/null 2>&1; then
        printf 'ENV: gh is not authenticated.\n' >&2
        exit 2
    fi

    input_dir="$(mktemp -d)" || exit 2
    trap 'rm -rf "${input_dir:?}"' EXIT
    if ! fetch_live "$repo" "$title" "$input_dir"; then
        printf 'ENV: a gh read of %s failed.\n' "$repo" >&2
        exit 2
    fi
    rc=0
    judge_from_dir "$input_dir" "$title" "$json_out" || rc=$?
    exit "$rc"
}

main "$@"
