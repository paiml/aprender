#!/usr/bin/env bash
# carry_milestone_items.sh <milestone-title> [--repo O/R] [--dry-run] | --self-test
#
# #3459 part 2 (cop ruling 2026-09-24): only open ISSUES labelled `must-carry` block a release cut
# (check_milestone_cut.sh --must-carry). Every OTHER open item of the milestone is MOVED here, before
# the tag, so nothing is silently left behind:
#   * to the NEXT release's milestone when that release's epic ("EPIC: release train <next>", label
#     epic) references it (#N in its body);
#   * otherwise to the `backlog` milestone (the operator's backlog rule).
# Each move gets ONE comment: "slipped_from: <milestone> -- carried to <target> at the <milestone>
# cut by the release autopilot (<why>)". The release epic of THIS train is never moved: it closes
# after publish. The autopilot's cut_tag() runs this between the must-carry gate and the STRICT
# gate; the strict gate is what verifies the milestone is empty at the tag.
#
# REFUSES (exit 1, nothing moved) while any open must-carry issue remains: carrying around a
# blocker would hide the reason the cut must wait. Exit 2 = could not act (gh/python3 missing, a
# read or a write failed, no or ambiguous milestone, no next milestone, no backlog milestone). A
# partial carry is reported by name and is exit 2, never 0.
#
# --dry-run prints the moves without writing. --self-test stubs gh on PATH (no network) and runs
# the case table.
set -uo pipefail
SELF="$(cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")"
BACKLOG="backlog"

usage() { printf 'usage: %s <milestone-title> [--repo O/R] [--dry-run] | --self-test\n' "$(basename "$0")" >&2; exit 2; }

# plan DIR TITLE -> lines "MOVE <kind> <number> <target> <why>", or "BLOCK <number>" / "ENV <msg>"
plan() {
    python3 - "$1" "$2" "$BACKLOG" <<'PY'
import json, re, sys
d, title, backlog = sys.argv[1:]
def rows(name):
    out = []
    with open(d + "/" + name, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                out.append(json.loads(raw))
    return out
def semver(t):
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", t or "")
    return tuple(int(x) for x in m.groups()) if m else None
ms = rows("milestones.jsonl")
cur = [m for m in ms if m.get("title") == title]
if len(cur) != 1:
    print("ENV milestone %s matches %d milestone(s)" % (title, len(cur))); sys.exit(0)
if not any(m.get("title") == backlog for m in ms):
    print("ENV no '%s' milestone to carry into" % backlog); sys.exit(0)
v = semver(title)
if v is None:
    print("ENV milestone title %s is not X.Y.Z" % title); sys.exit(0)
later = sorted((semver(m["title"]), m["title"]) for m in ms
               if m.get("state") == "open" and semver(m.get("title")) and semver(m["title"]) > v)
if not later:
    print("ENV no open milestone after %s" % title); sys.exit(0)
nxt = later[0][1]
listed = set()
for e in rows("next_epics.jsonl"):
    names = [l.get("name") for l in (e.get("labels") or [])]
    t = e.get("title", "")
    rest = t[len("EPIC: release train " + nxt):]
    if "epic" in names and t.startswith("EPIC: release train " + nxt) and (rest == "" or rest[0].isspace()):
        listed |= {int(n) for n in re.findall(r"#(\d+)\b", e.get("body") or "")}
epic_prefix = "EPIC: release train " + title
for it in sorted(rows("items.jsonl"), key=lambda i: i.get("number", 0)):
    n = it.get("number")
    kind = "pr" if "pull_request" in it else "issue"
    labels = [l.get("name") for l in (it.get("labels") or [])]
    t = it.get("title", "")
    rest = t[len(epic_prefix):]
    if kind == "issue" and "epic" in labels and t.startswith(epic_prefix) and (rest == "" or rest[0].isspace()):
        continue                                   # this train's epic: closes after publish
    if kind == "issue" and "must-carry" in labels:
        print("BLOCK %s" % n); continue
    if n in listed:
        print("MOVE %s %s %s the %s epic lists it" % (kind, n, nxt, nxt))
    else:
        print("MOVE %s %s %s not must-carry, and not listed by the %s epic" % (kind, n, backlog, nxt))
PY
}

fetch() { # REPO TITLE DIR -> milestones.jsonl, items.jsonl, next_epics.jsonl
    local repo=$1 title=$2 dir=$3 number
    gh api --paginate --jq '.[]' "repos/${repo}/milestones?state=all&per_page=100" > "$dir/milestones.jsonl" || return 1
    number=$(python3 -c 'import json,sys
n=[json.loads(l)["number"] for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("title")==sys.argv[2]]
print(n[0] if len(n)==1 else "")' "$dir/milestones.jsonl" "$title") || return 1
    [ -n "$number" ] || { : > "$dir/items.jsonl"; : > "$dir/next_epics.jsonl"; return 0; }
    gh api --paginate --jq '.[]' "repos/${repo}/issues?milestone=${number}&state=open&per_page=100" > "$dir/items.jsonl" || return 1
    gh api --paginate --jq '.[]' "repos/${repo}/issues?labels=epic&state=open&per_page=100" > "$dir/next_epics.jsonl" || return 1
}

carry() { # REPO TITLE DRY
    local repo=$1 title=$2 dry=$3 dir p rc=0 moved=0 failed="" kind n target why
    dir=$(mktemp -d) || return 2
    fetch "$repo" "$title" "$dir" || { rm -rf -- "${dir:?}"; echo "ENV: a gh read of $repo failed" >&2; return 2; }
    p=$(plan "$dir" "$title") || { rm -rf -- "${dir:?}"; echo "ENV: the carry plan could not be computed" >&2; return 2; }
    rm -rf -- "${dir:?}"
    if grep -q '^ENV ' <<< "$p"; then sed -n 's/^ENV /ENV: /p' <<< "$p" >&2; return 2; fi
    if grep -q '^BLOCK ' <<< "$p"; then
        printf 'REFUSE: %s open must-carry issue(s) block the %s cut; NOTHING was carried: %s\n' \
            "$(grep -c '^BLOCK ' <<< "$p")" "$title" "$(sed -n 's/^BLOCK /#/p' <<< "$p" | tr '\n' ' ')"
        return 1
    fi
    while read -r _ kind n target why; do
        [ -n "${n:-}" ] || continue
        if [ "$dry" = 1 ]; then printf 'WOULD CARRY %s #%s -> %s (%s)\n' "$kind" "$n" "$target" "$why"; continue; fi
        if gh "$kind" edit "$n" --repo "$repo" --milestone "$target" > /dev/null \
           && gh "$kind" comment "$n" --repo "$repo" \
                --body "slipped_from: $title -- carried to $target at the $title cut by the release autopilot ($why)" > /dev/null; then
            printf 'CARRIED %s #%s -> %s (%s)\n' "$kind" "$n" "$target" "$why"; moved=$((moved + 1))
        else
            failed="$failed #$n"; rc=2
        fi
    done < <(grep '^MOVE ' <<< "$p")
    if [ "$rc" -ne 0 ]; then printf 'PARTIAL: %s carried, FAILED:%s -- the strict gate will be RED\n' "$moved" "$failed" >&2; return 2; fi
    printf 'DONE  %s item(s) carried out of %s\n' "$moved" "$title"
    return 0
}

self_test() {
    local d stub rc bad=0 out
    d=$(mktemp -d) || return 2
    case "$d" in /tmp/?*) ;; *) echo "self-test: bad temp dir $d"; return 2 ;; esac
    stub="$d/bin"; mkdir -p "$stub"
    # a stub gh: serves the fixture reads and RECORDS every write (edit/comment) to $STUB_DIR/writes
    cat > "$stub/gh" <<'STUB'
#!/usr/bin/env bash
case "$1" in
  api) case "$*" in
         *'milestones?state=all'*) cat "$STUB_DIR/milestones.jsonl" ;;
         *'issues?milestone=7&state=open'*) cat "$STUB_DIR/items.jsonl" ;;
         *'issues?labels=epic&state=open'*) cat "$STUB_DIR/epics.jsonl" ;;
         *) echo "stub gh: unexpected read $*" >&2; exit 9 ;;
       esac ;;
  issue|pr) [ "${FAIL_ON:-}" = "$3" ] && exit 1; echo "$*" >> "$STUB_DIR/writes" ;;
  *) echo "stub gh: unexpected $*" >&2; exit 9 ;;
esac
STUB
    chmod +x "$stub/gh"
    ms() { printf '{"title":"%s","number":%s,"state":"%s"}\n' "$1" "$2" "$3"; }
    item() { printf '{"number":%s,"state":"open","title":"%s","labels":[{"name":"%s"}]%s}\n' "$1" "$2" "$3" "${4:-}"; }
    fixture() { # base fixture: M=0.70.0 (#7), next 0.71.0 whose epic lists #12, backlog exists
        { ms 0.70.0 7 open; ms 0.71.0 9 open; ms 0.69.1 5 closed; ms backlog 11 open; } > "$d/milestones.jsonl"
        { item 20 "EPIC: release train 0.70.0 — schedule" epic
          item 10 "an unlabelled issue" P1
          item 12 "listed by the next epic" bug
          item 13 "a pull request" release ',"pull_request":{"url":"u"}'; } > "$d/items.jsonl"
        printf '{"number":40,"title":"EPIC: release train 0.71.0 — schedule","labels":[{"name":"epic"}],"body":"carries #12 and #99"}\n' > "$d/epics.jsonl"
        : > "$d/writes"
    }
    run() { STUB_DIR="$d" PATH="$stub:$PATH" bash "$SELF" 0.70.0 --repo o/r "$@" > "$d/out" 2>&1; }
    ok() { printf 'ok    %s\n' "$1"; }
    nok() { printf 'FAIL  %s\n' "$1"; sed 's/^/        /' "$d/out"; bad=1; }

    fixture; run; rc=$?
    if [ "$rc" = 0 ] && grep -q '^issue edit 12 --repo o/r --milestone 0.71.0$' "$d/writes" \
       && grep -q '^issue edit 10 --repo o/r --milestone backlog$' "$d/writes" \
       && grep -q '^pr edit 13 --repo o/r --milestone backlog$' "$d/writes" \
       && ! grep -q ' 20 ' "$d/writes" && [ "$(grep -c ' comment ' "$d/writes")" = 3 ] \
       && grep -q 'slipped_from: 0.70.0 -- carried to 0.71.0 at the 0.70.0 cut' "$d/writes"; then
        ok "every non-must-carry item moves: listed -> next release, others -> backlog, one comment each; the epic stays"
    else nok "the carry moved the wrong set (rc=$rc)"; cat "$d/writes"; fi

    fixture; item 30 "a must-carry blocker" must-carry >> "$d/items.jsonl"; run; rc=$?
    if [ "$rc" = 1 ] && [ ! -s "$d/writes" ] && grep -q 'REFUSE: 1 open must-carry issue(s) block the 0.70.0 cut; NOTHING was carried: #30' "$d/out"; then
        ok "an open must-carry issue REFUSES the carry, and nothing is written"
    else nok "a must-carry blocker did not refuse cleanly (rc=$rc)"; fi

    fixture; run --dry-run; rc=$?
    if [ "$rc" = 0 ] && [ ! -s "$d/writes" ] && grep -q 'WOULD CARRY issue #12 -> 0.71.0' "$d/out"; then
        ok "--dry-run writes nothing"
    else nok "--dry-run wrote or misplanned (rc=$rc)"; fi

    fixture; FAIL_ON=10 run; rc=$?
    if [ "$rc" = 2 ] && grep -q 'PARTIAL: .* FAILED: #10' "$d/out"; then
        ok "a failed write is a named PARTIAL, exit 2, never DONE"
    else nok "a failed write was not reported (rc=$rc)"; fi

    fixture; { ms 0.70.0 7 open; ms 0.71.0 9 open; } > "$d/milestones.jsonl"; run; rc=$?
    if [ "$rc" = 2 ] && [ ! -s "$d/writes" ] && grep -q "no 'backlog' milestone" "$d/out"; then
        ok "no backlog milestone is 'cannot act' (2), nothing written"
    else nok "a missing backlog milestone was not refused (rc=$rc)"; fi

    fixture; { ms 0.70.0 7 open; ms backlog 11 open; } > "$d/milestones.jsonl"; run; rc=$?
    if [ "$rc" = 2 ] && [ ! -s "$d/writes" ] && grep -q 'no open milestone after 0.70.0' "$d/out"; then
        ok "no next milestone is 'cannot act' (2), nothing written"
    else nok "a missing next milestone was not refused (rc=$rc)"; fi

    fixture; printf '{"number":41,"title":"EPIC: release train 0.71.0x — other","labels":[{"name":"epic"}],"body":"#10"}\n' > "$d/epics.jsonl"; run; rc=$?
    if [ "$rc" = 0 ] && grep -q '^issue edit 10 --repo o/r --milestone backlog$' "$d/writes" && grep -q '^issue edit 12 --repo o/r --milestone backlog$' "$d/writes"; then
        ok "another train's epic (prefix 0.71.0x) lists nothing for 0.71.0"
    else nok "a prefix-matching epic was read as the next release's (rc=$rc)"; fi

    rm -rf -- "${d:?}"
    [ "$bad" -eq 0 ] && { echo "SELF-TEST PASSED"; return 0; }
    echo "SELF-TEST FAILED"; return 1
}

[ $# -gt 0 ] || { self_test; exit $?; }
[ "${1:-}" = "--self-test" ] && { self_test; exit $?; }
case "$1" in -h|--help) sed -n '2,20p' "$SELF" | sed 's/^# \{0,1\}//'; exit 0 ;; -*) usage ;; esac
title=$1; shift; repo="paiml/aprender"; dry=0
while [ $# -gt 0 ]; do
    case "$1" in
        --repo) [ $# -ge 2 ] || usage; repo=$2; shift 2 ;;
        --dry-run) dry=1; shift ;;
        *) usage ;;
    esac
done
command -v gh > /dev/null || { echo "ENV: gh is not on PATH" >&2; exit 2; }
command -v python3 > /dev/null || { echo "ENV: python3 is not on PATH" >&2; exit 2; }
carry "$repo" "$title" "$dry"; exit $?
