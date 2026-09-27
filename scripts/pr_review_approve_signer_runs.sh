#!/usr/bin/env bash
# pr_review_approve_signer_runs.sh — approve the CI runs the receipt signer's own push
# is held on (PR-REVIEW-SKILL-002 v2 §4.3 CI signer; operator flow directive 2026-09-27).
#
# WHY. ci/sections.yml `pr-review-sign` commits the signature back to the PR branch
# with the job token. A push by github-actions[bot] creates the new head's
# `pull_request` runs HELD (conclusion action_required) under the repository's
# fork-approval policy, and nothing approves them: measured 2026-09-27, 8 held runs,
# every one actor github-actions[bot], on car/0.70.0, fold/b3-onto-main and
# batch/b1r-serve-perf-runtime. The required checks (`gate`, `workspace-test`) come
# from those runs, so every signed PR stalled until a human clicked "Approve".
#
# WHAT. After the signer pushes <sha>, approve exactly the runs that push created:
# head_sha == <sha>, event == pull_request, actor == github-actions[bot], held. A
# human's held run, another sha's, or another event's is never touched. No secret, no
# ruleset change: the job token already has actions: write (ci.yml x86-main is
# write-all).
#
# EXIT
#   0   every held signer run on <sha> was approved, or runs exist and none is held
#   1   an approve call failed (the message names the run and the HTTP status)
#   3   no run at all appeared on <sha> within the wait window
#   64  usage
#
#   pr_review_approve_signer_runs.sh <sha>     # needs GITHUB_TOKEN, GITHUB_REPOSITORY
#   pr_review_approve_signer_runs.sh --self-test
set -euo pipefail

PROG=pr_review_approve_signer_runs.sh
BOT='github-actions[bot]'
WAIT_S="${APPROVE_WAIT_S:-120}"
POLL_S="${APPROVE_POLL_S:-10}"
MUTANT="${APPROVE_MUTANT:-}"

# api METHOD PATH -> body on stdout; "HTTP <code>" on the last line of stderr.
api() {
    local method="$1" path="$2" out code
    out="$(mktemp)"
    code=$(curl -sS -o "$out" -w '%{http_code}' -X "$method" \
        -H "Authorization: Bearer ${GITHUB_TOKEN:?GITHUB_TOKEN is required}" \
        -H 'Accept: application/vnd.github+json' \
        "${GITHUB_API_URL:-https://api.github.com}$path") || code=000
    cat "$out"; rm -f "${out:?}"
    [ "${code:0:1}" = 2 ] || { echo "HTTP $code" >&2; return 1; }
}

# select_held SHA < runs-json -> one run id per line: the signer's held runs on SHA.
select_held() {
    python3 -c '
import json, sys
sha, bot, mutant = sys.argv[1], sys.argv[2], sys.argv[3]
for r in json.load(sys.stdin).get("workflow_runs", []):
    if mutant != "no-sha" and r.get("head_sha") != sha: continue
    if mutant != "no-event" and r.get("event") != "pull_request": continue
    if mutant != "no-actor" and (r.get("actor") or {}).get("login") != bot: continue
    if mutant != "no-held" and r.get("conclusion") != "action_required": continue
    print(r["id"])
' "$1" "$BOT" "$MUTANT"
}

count_runs() { python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("workflow_runs", [])))'; }

approve_for() {
    local sha="$1" repo="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
    local waited=0 seen_any=0 approved=" " body id ids quiet=0
    while :; do
        body=$(api GET "/repos/$repo/actions/runs?head_sha=$sha&per_page=100") || {
            echo "::error::$PROG: listing runs on $sha failed"; return 1; }
        [ "$(count_runs <<< "$body")" -gt 0 ] && seen_any=1
        ids=$(select_held "$sha" <<< "$body")
        local new=0
        for id in $ids; do
            case "$approved" in *" $id "*) continue ;; esac
            if api POST "/repos/$repo/actions/runs/$id/approve" >/dev/null; then
                echo "approved run $id (held signer push on $sha)"
                approved="$approved$id "; new=1
            else
                echo "::error::$PROG: approving run $id on $sha failed — approve it at https://github.com/$repo/actions/runs/$id"
                return 1
            fi
        done
        # Workflows on one push are created together but not atomically: after the
        # first approval, one quiet poll with nothing new ends the wait.
        if [ "$approved" != " " ]; then
            [ "$new" -eq 0 ] && quiet=$((quiet + 1))
            [ "$quiet" -ge 1 ] && break
        fi
        [ "$waited" -ge "$WAIT_S" ] && break
        sleep "$POLL_S"; waited=$((waited + POLL_S))
    done
    if [ "$approved" != " " ]; then return 0; fi
    if [ "$seen_any" -eq 1 ]; then echo "runs exist on $sha and none is held — nothing to approve"; return 0; fi
    echo "::error::$PROG: no run appeared on $sha within ${WAIT_S}s — the signed head may have no CI"
    return 3
}

# ---------------------------------------------------------------------------
# --self-test: the API is a fake that serves one fixture list and records approvals.
self_test() {
    local td fail=0 rows=0
    td="$(mktemp -d)"
    trap 'rm -rf "${td:?}"' RETURN
    local S=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa O=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    run_json() { printf '{"id":%s,"head_sha":"%s","event":"%s","actor":{"login":"%s"},"conclusion":%s}' "$@"; }
    api() {
        case "$1" in
            GET)  cat "$td/runs.json" ;;
            POST) local id="${2%/approve}"; id="${id##*/}"
                  [ -f "$td/deny" ] && { echo "HTTP 403" >&2; return 1; }
                  echo "$id" >> "$td/approved" ;;
        esac
    }
    # row NAME WANT_RC WANT_APPROVED(space list or -) [deny] -- run objects...
    row() {
        local name="$1" want_rc="$2" want_ap="$3"; shift 3
        local deny=0; [ "${1:-}" = deny ] && { deny=1; shift; }
        : > "$td/approved"; rm -f "$td/deny"; [ "$deny" -eq 1 ] && : > "$td/deny"
        local IFS=,; printf '{"workflow_runs":[%s]}' "$*" > "$td/runs.json"; unset IFS
        local rc=0
        GITHUB_REPOSITORY=o/r WAIT_S=0 POLL_S=0 approve_for "$S" > "$td/out" 2>&1 || rc=$?
        local got; got=$(sort -n "$td/approved" | tr '\n' ' ' | sed 's/ $//'); [ -n "$got" ] || got=-
        rows=$((rows + 1))
        if [ "$rc" != "$want_rc" ] || [ "$got" != "$want_ap" ]; then
            echo "FAIL $name: want rc=$want_rc approved=[$want_ap] got rc=$rc approved=[$got]"; fail=1
        fi
    }
    local H='"action_required"' OK='"success"' N=null
    run_rows() {
        fail=0; rows=0
        row "held signer runs on the sha are approved" 0 "1 2" \
            "$(run_json 1 $S pull_request "$BOT" "$H")" "$(run_json 2 $S pull_request "$BOT" "$H")"
        row "a human's held run is never approved" 0 "1" \
            "$(run_json 1 $S pull_request "$BOT" "$H")" "$(run_json 3 $S pull_request someone "$H")"
        row "a held run on another sha is never approved" 0 "1" \
            "$(run_json 1 $S pull_request "$BOT" "$H")" "$(run_json 4 $O pull_request "$BOT" "$H")"
        row "a held run of another event is never approved" 0 "1" \
            "$(run_json 1 $S pull_request "$BOT" "$H")" "$(run_json 5 $S pull_request_target "$BOT" "$H")"
        row "a run that is not held is never approved" 0 "1" \
            "$(run_json 1 $S pull_request "$BOT" "$H")" "$(run_json 6 $S pull_request "$BOT" "$OK")"
        row "runs exist, none held: nothing to approve, rc 0" 0 - \
            "$(run_json 7 $S pull_request "$BOT" $N)"
        row "no run at all: rc 3, never green" 3 -
        row "approve refused: rc 1, never green" 1 - deny \
            "$(run_json 1 $S pull_request "$BOT" "$H")"
    }
    echo "== $PROG --self-test =="
    run_rows
    [ "$fail" -eq 0 ] || return 1
    echo "self-test: $rows/$rows rows"
    # Planted mutants: each drops one filter and must turn some row RED.
    local m killed=0 total=0
    for m in no-sha no-event no-actor no-held; do
        total=$((total + 1))
        MUTANT="$m"; run_rows > "$td/mut" 2>&1; MUTANT=""
        if [ "$fail" -eq 1 ]; then killed=$((killed + 1)); else echo "SURVIVED mutant: $m"; fi
    done
    fail=0
    echo "mutants: $killed/$total RED"
    [ "$killed" -eq "$total" ]
}

case "${1:-}" in
    --self-test) self_test ;;
    '') echo "usage: $PROG <sha> | --self-test" >&2; exit 64 ;;
    *)  [[ $1 =~ ^[0-9a-f]{40}$ ]] || { echo "usage: $PROG <40-hex sha> | --self-test" >&2; exit 64; }
        approve_for "$1" ;;
esac
