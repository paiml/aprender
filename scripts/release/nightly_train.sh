#!/usr/bin/env bash
# nightly_train.sh — is main's head releasable, judged from the evidence the night already produced? (BLD-002 row R4)
#
# THE ROW. BLD-002 R4: "All lanes in parallel on main's head. Writes bundle(H) and prints RELEASABLE or NOT RELEASABLE
#   with the failing check and run." Planted falsifier: "Kill one lane: the result is NOT RELEASABLE naming it, never
#   silence." This is the entry point. It prints ONE line on stdout:
#     RELEASABLE H=<C>                                  every verdict lane is green on C
#     NOT RELEASABLE: <lane>, <run id|not_measured>     the first failing verdict lane, then (+N more)
#     NOT RELEASABLE: nightly-train, <step> ...         the train itself failed part-way (the EXIT trap prints it)
#   and every line ends with [C=<sha10> pin=<train>.<greens>.<redage>]: the commit judged and the shas of the three
#   scripts that judged it. H is the commit id of C, main's head at the moment of the read.
#
# READ, DON'T RE-RUN. Night 1 runs nothing. Each lane takes the result its existing scheduled producer (a workflow on
#   main and, optionally, a job-name pattern) recorded for C. A lane is
#     green         the newest completed run of its producer on C ran every matched job to success, at attempt 1;
#     red           a matched job ended failure, timed_out or startup_failure, or the run succeeded only at attempt 2
#                   or later (a retried green is not a green: FLOW-003 G4, CI retries 2 -> 0);
#     not_measured  no producer exists; no run on main for C (the newest is on another commit); the run is still in
#                   progress; no job matched or every matched job was skipped/cancelled; the attempt could not be
#                   read; or the read failed (rate limit, GraphQL error, network). not_measured is never a pass.
#   A run whose matched jobs were all skipped (nightly.yml's `reused` decision) measured nothing; the next older run on
#   C is taken instead.
#
# THE VERDICT covers exactly what release day enforces today: autopilot.sh steps wait..dryrun (deep build checks,
#   dogfood + preflight R5, the model matrix R7, readiness R8, the milestone cut, clean-room on the tag, the release
#   assets, preflight R1-R8, the publish dry run). Lanes nobody produces nightly yet are listed, so they print
#   not_measured: the verdict is NOT RELEASABLE until a producer exists for each. The other lanes are information
#   only, coverage included (operator ruling C291.1: the coverage floor does not gate this release). The lane table is a
#   constant in this file, never an option or an environment value: a verdict set a caller can shrink is theater.
#   --self-test pins it. Report-only: this script blocks nothing and no release script reads it.
#
# OUTPUTS under --out DIR (or $OUT), per UTC date D:
#     D/line         the line;
#     D/bundle.tsv   per lane: lane, kind, state, producer, run id, run head, conclusion, attempt, started, ended,
#                    hours lost, reason; then the tree of C, the command, the pins, the 6 h budget (info only);
#     D/reds.tsv     red and not_measured lanes ranked by hours lost (red_age.sh's measure of the current stretch);
#     D/raw/         the normalized read, so `--from D/raw` replays the judgement with no network;
#     history.tsv    one row per run in nightly_greens.sh's format; greens in a row = its streak for nightly-train.
#   With --inbox FILE (or $INBOX) one line of at most 300 bytes is appended and read back.
#   Rows carry event "${GITHUB_EVENT_NAME:-timer}": a user timer's runs never count as scheduled nights.
#
# GITHUB. At most MAX_CALLS REST/GraphQL calls per run: rate_limit (free), the workflow list (ETag; a 304 is free), ONE
#   GraphQL query (every producer's recent runs with their check runs, plus main's head and its check rollup), and one
#   run read per green verdict-lane candidate for run_attempt (GraphQL has no attempt field). Core remaining under
#   RATE_FLOOR makes no further call: the read failed, every producer lane is not_measured.
#
# EXIT  0 a verdict line was printed (RELEASABLE or NOT) · 2 the train failed part-way (trap line) · 3 caller error
#   With --exit-verdict, NOT RELEASABLE exits 1 (a CI job is green only on a releasable head).
# NO TOKEN. The train refuses to start (trap line "no-token") when a registry token is reachable: CARGO_REGISTRY_TOKEN
#   or CARGO_REGISTRIES_*_TOKEN set, or $CARGO_HOME/credentials(.toml). The installed unit points CARGO_HOME at an
#   empty directory in its bundle.
#
# USAGE
#   nightly_train.sh --out DIR [--inbox FILE] [--from RAWDIR] [--now YYYY-MM-DDTHH:MM:SSZ] [--pinned] [--exit-verdict]
#   nightly_train.sh --self-test   the case table (fixtures, no network)
#   nightly_train.sh --mutants     each planted mutant must turn the case table RED
#   nightly_train.sh --install --home DIR --out DIR [--inbox FILE] --train SHA --greens SHA --redage SHA
#                                  pin a bundle copied out of git and install the daily user timer (04:45 UTC)
set -uo pipefail
PROG="${0##*/}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$HERE/${BASH_SOURCE[0]##*/}"
REPO="paiml/aprender"
MAX_CALLS=10
RATE_FLOOR=1000
UNIT="aprender-nightly-train"
# lane;kind;release-day check;producer workflow;event pattern;job-name pattern[;required job names, default 1]
# ('-' producer: none yet). A run with fewer distinct matched job names than required measures nothing (void).
LANES='ci-main;verdict;ci / gate + workspace-test (merge, autopilot wait);.github/workflows/ci.yml;^push$;^(ci / gate|workspace-test)$;2
deep-doctests;verdict;autopilot deep: cargo test --doc --workspace;-;-;-
deep-nodefault;verdict;autopilot deep: cargo check --workspace --no-default-features;-;-;-
deep-examples;verdict;autopilot deep: cargo build --workspace --examples;.github/workflows/examples-nightly.yml;^schedule$;^examples$
deep-bins-build;verdict;autopilot deep: all-bins cargo build --locked --release;.github/workflows/nightly.yml;^schedule$;-unknown-linux-gnu on 
deep-bins-smoke;verdict;autopilot deep: nightly_manifest.py smoke;-;-;-
dogfood;verdict;dogfood.sh --phase pre-publish + preflight R5;-;-;-
models;verdict;models_t1.sh GPU-host ladder legs, preflight R7;-;-;-
readiness;verdict;release_readiness.sh, preflight R8;-;-;-
milestone;verdict;check_milestone_cut.sh --must-carry;-;-;-
cleanroom-cpu;verdict;clean-room (aprender) on the tag;-;-;-
cleanroom-gpu;verdict;b2-gpu.yml on the tag;-;-;-
assets;verdict;binary-release.yml + check_release_assets.sh;-;-;-
preflight;verdict;check_publish_preflight.sh R1-R8;-;-;-
publish-dryrun;verdict;rc_publish_gate.sh --verify + cascade-publish.sh --check;-;-;-
coverage;info;tag_coverage_gate.sh (C291.1: not gating);.github/workflows/coverage-nightly.yml;^schedule$;^coverage$
guards;info;guards-nightly;.github/workflows/guards-nightly.yml;^schedule$;
mutants;info;mutants-nightly;.github/workflows/mutants-nightly.yml;^schedule$;
toolchain-ceiling;info;toolchain-ceiling;.github/workflows/toolchain-ceiling.yml;^schedule$;
cuda;info;cuda-nightly;.github/workflows/cuda-nightly.yml;^schedule$;
silicon;info;silicon-nightly;.github/workflows/silicon-nightly.yml;^schedule$;
qwen-story;info;qwen-story-daily;.github/workflows/qwen-story-daily.yml;^schedule$;
beat-speed;info;beat-speed-nightly;.github/workflows/beat-speed-nightly.yml;^schedule$;
conleche;info;conleche-nightly;.github/workflows/conleche-nightly.yml;^schedule$;
bench;info;nightly-bench;.github/workflows/nightly-bench.yml;^schedule$;
book;info;book;.github/workflows/book.yml;^schedule$;
book-contracts;info;book-contracts;.github/workflows/book-contracts.yml;^schedule$;
install-script;info;install-script;.github/workflows/install-script.yml;^schedule$;
fleet-toolset;info;fleet-toolset;.github/workflows/fleet-toolset.yml;^schedule$;'
caller_error() { printf 'NOT RELEASABLE: nightly-train, caller error: %s\n' "$*"; exit 3; }
# ---------------------------------------------------------------- judgement (pure: files in, files out) ----------
# evaluate LANESFILE RAW MODE -> MODE=cand: run ids of green verdict candidates; MODE=final: lanes.tsv rows and
#   RAW/hist_redage.tsv. RAW holds C, read, runs.tsv, attempts.tsv.
evaluate() {
    local c rd
    c="$(cat "$2/C" 2>/dev/null)"; rd="$(cat "$2/read" 2>/dev/null)"
    [ -f "$2/attempts.tsv" ] || : > "$2/attempts.tsv"
    [ -f "$2/runs.tsv" ] || : > "$2/runs.tsv"
    awk -F '\t' -v OFS='\t' -v C="$c" -v RD="${rd:-failed: no read status}" -v MODE="$3" -v HIST="$2/hist_redage.tsv" '
    FILENAME == ARGV[1] { split($0, a, ";"); n++; L[n] = a[1]; K[n] = a[2]; CK[n] = a[3]; WF[n] = a[4]; EV[n] = a[5]; RE[n] = a[6]; NEED[n] = (a[7] ~ /^[0-9]+$/ ? a[7] + 0 : 1); next }
    FILENAME == ARGV[2] { AT[$1] = $2; next }
    {
        key = $3 SUBSEP $10 SUBSEP $13
        if (key in seen) next
        seen[key] = 1; r = $3
        if (!(r in RW)) { RW[r] = $2; REV[r] = $4; RBR[r] = $5; RH[r] = $6; RC[r] = $7; RST[r] = toupper($8); RCO[r] = toupper($9); nr++; RID[nr] = r }
        if ($10 != "") { nj[r]++; k = nj[r]; JN[r, k] = $10; JS[r, k] = toupper($11); JC[r, k] = toupper($12); JB[r, k] = $13; JE[r, k] = $14; JD[r, k] = $15 }
    }
    # lane state of run r for lane i: green | red | pending | void; sets WHY, ST, EN, DUP
    function lst(i, r,    k, b, fb, nm, pend, bad, ok, tot) {
        split("", b); split("", fb); ST = ""; EN = ""; DUP = 0; WHY = ""
        if (RST[r] != "COMPLETED") { WHY = "run " r " is " tolower(RST[r]); return "pending" }
        for (k = 1; k <= nj[r]; k++) if (JN[r, k] == "#truncated") { WHY = "run " r ": its job list was cut at the page size, so a failed job may be unseen"; return "unread" }
        for (k = 1; k <= nj[r]; k++) if (JN[r, k] ~ RE[i]) {
            if (!(JN[r, k] in b) || JB[r, k] > JB[r, b[JN[r, k]]]) b[JN[r, k]] = k
            if (JD[r, k] > 1) DUP = 1
            # every matched job votes, not only the newest of a name: two parallel legs that render the same name must not
            # let a later success hide an earlier failure (a re-run that hides one is attempt >= 2, red either way)
            if (JS[r, k] == "COMPLETED" && JC[r, k] ~ /^(FAILURE|TIMED_OUT|STARTUP_FAILURE)$/ && !((JN[r, k]) in fb)) { fb[JN[r, k]] = 1; bad = bad " " JN[r, k] "=" tolower(JC[r, k]) }
        }
        for (nm in b) {
            k = b[nm]; tot++
            if (ST == "" || (JB[r, k] != "" && JB[r, k] < ST)) ST = JB[r, k]
            if (JE[r, k] > EN) EN = JE[r, k]
            if (JS[r, k] != "COMPLETED") pend = pend " " nm
            else if (JC[r, k] == "SUCCESS") ok++
        }
        if (tot == 0) { WHY = "run " r ": no job matched"; return "void" }
        if (pend != "") { WHY = "run " r ": job(s) still running:" pend; return "pending" }
        if (bad != "") { WHY = "run " r ":" bad; return "red" }
        if (tot < NEED[i]) { WHY = "run " r ": " tot " of " NEED[i] " required job(s) present"; return "void" }
        if (ok == tot) return "green"
        WHY = "run " r ": matched job(s) skipped or cancelled"; return "void"
    }
    function out(i, state, r, att, why) {
        print L[i], K[i], state, (WF[i] == "-" ? "none" : WF[i] " " RE[i]), (r == "" ? "not_measured" : r), (r == "" ? "" : RH[r]), (r == "" ? "" : tolower(RCO[r])), att, ST, EN, why
    }
    END {
        if (MODE == "final") print "check", "run_id", "created_at", "branch", "event", "conclusion", "attempt" > HIST
        readok = (RD == "ok" && C ~ /^[0-9a-f]{40}$/)
        for (i = 1; i <= n; i++) {
            ST = ""; EN = ""
            if (WF[i] == "-") { if (MODE == "final") out(i, "not_measured", "", "", "no nightly producer on main yet"); continue }
            if (!readok) { if (MODE == "final") out(i, "not_measured", "", "", "read " RD); continue }
            # this lane'"'"'s runs on main, newest first
            m = 0; split("", S)
            for (q = 1; q <= nr; q++) { r = RID[q]; if (RW[r] == WF[i] && REV[r] ~ EV[i] && RBR[r] == "main") S[++m] = r }
            for (p = 2; p <= m; p++) for (q = p; q > 1 && RC[S[q]] > RC[S[q - 1]]; q--) { t = S[q]; S[q] = S[q - 1]; S[q - 1] = t }
            pick = ""; pst = ""; pwhy = ""; other = ""
            for (q = 1; q <= m; q++) {
                r = S[q]; s = lst(i, r)
                if (MODE == "final" && s != "pending" && s != "unread") {
                    ha = (r in AT && AT[r] ~ /^[0-9]+$/) ? AT[r] : (DUP ? 2 : 1)
                    hc = (s == "green" ? "success" : (s == "red" ? "failure" : "cancelled"))
                    print L[i], r, RC[r], "main", REV[r], hc, ha > HIST
                }
                if (RH[r] != C) { if (other == "") other = r " on " substr(RH[r], 1, 10); continue }
                if (pick == "" && s != "void") { pick = r; pst = s; pwhy = WHY; pS = ST; pE = EN; pD = DUP }
                else if (pick == "" && pwhy == "") pwhy = WHY
            }
            if (pick == "") {
                if (MODE == "final") out(i, "not_measured", "", "", (pwhy != "" ? pwhy : "no run on C") (other != "" ? "; newest run " other : (m == 0 ? "; no run on main" : "")))
                continue
            }
            ST = pS; EN = pE
            if (pst == "pending") { if (MODE == "final") out(i, "not_measured", pick, "", "in progress: " pwhy); continue }
            if (pst == "unread") { if (MODE == "final") out(i, "not_measured", pick, "", pwhy); continue }
            if (pst == "red") { if (MODE == "final") out(i, "red", pick, "", pwhy); continue }
            # green: the attempt decides
            if (K[i] == "verdict") {
                if (MODE == "cand") { print pick; continue }
                if (!(pick in AT) || AT[pick] !~ /^[0-9]+$/) { out(i, "not_measured", pick, "?", "attempt not read"); continue }
                att = AT[pick]
            } else {
                if (MODE == "cand") continue
                # no attempt is read for an info lane (the call budget): a repeated job name shows a re-run; otherwise the
                # attempt is inferred, and the bundle says so instead of claiming attempt 1
                if (pick in AT && AT[pick] ~ /^[0-9]+$/) att = AT[pick]
                else if (pD) att = 2
                else { out(i, "green", pick, "1?", "attempt inferred from the job list, not read: an info lane, it does not vote"); continue }
            }
            if (att > 1) out(i, "red", pick, att, "success only at attempt " att ": a retried green is not a green")
            else out(i, "green", pick, att, "")
        }
    }' <(printf '%s\n' "$1") "$2/attempts.tsv" "$2/runs.tsv"
}
# hours LANESOUT REDAGE_OUTPUT_FILE -> lane<TAB>hours lost (2 decimals) for every lane red_age measured
hours() {
    awk '{
        if (match($0, /REDAGE [^ ]+ [a-z_]+: red for (at least )?[0-9]+h[0-9]+m[0-9]+s/)) {
            s = substr($0, RSTART, RLENGTH); split(s, w, " "); d = s; sub(/.*red for (at least )?/, "", d)
            split(d, t, /[hms]/); printf "%s\t%.2f\n", w[2], t[1] + t[2] / 60 + t[3] / 3600
        } else if (match($0, /REDAGE [^ ]+ ok: green/)) { s = substr($0, RSTART, RLENGTH); split(s, w, " "); printf "%s\t0.00\n", w[2] }
    }' "$1"
}
# decide LANESFILE RAW NOW -> RAW/lanes.tsv, RAW/reds.tsv, RAW/line (the verdict, without the pin suffix)
decide() {
    local raw="$2" c
    c="$(cat "$raw/C" 2>/dev/null)"
    evaluate "$1" "$raw" final > "$raw/lanes.tsv" || return 1
    : > "$raw/redage.out"
    if [ -x "$HERE/red_age.sh" ] || [ -f "$HERE/red_age.sh" ]; then
        bash "$HERE/red_age.sh" --history "$raw/hist_redage.tsv" --as-of "$3" > "$raw/redage.out" 2>&1
    fi
    hours "$raw/redage.out" > "$raw/hours.tsv"
    # reds: rank by hours lost (unknown last), verdict before info, red before not_measured, then table order
    awk -F '\t' -v OFS='\t' 'FILENAME == ARGV[1] { H[$1] = $2; next }
        { n++; if ($3 == "green") next
          h = ($1 in H) ? H[$1] : "-"
          print (h == "-" ? -1 : h), ($2 == "verdict" ? 0 : 1), ($3 == "red" ? 0 : 1), n, $1, $2, $3, h, $5, $11 }' \
        "$raw/hours.tsv" "$raw/lanes.tsv" | sort -t "$(printf '\t')" -k1,1gr -k2,2n -k3,3n -k4,4n \
        | awk -F '\t' -v OFS='\t' 'BEGIN { print "rank", "lane", "kind", "state", "hours_lost", "run_id", "reason" } { print NR, $5, $6, $7, $8, $9, $10 }' > "$raw/reds.tsv"
    # the line: the first failing verdict lane (red before not_measured, then hours lost, then table order)
    awk -F '\t' -v C="$c" 'FILENAME == ARGV[1] { H[$1] = $2; next }
        $2 == "verdict" { nv++; if ($3 == "green") next
            bad++; h = ($1 in H) ? H[$1] + 0 : -1; s = ($3 == "red" ? 1 : 0)
            if (best == "" || s > bs || (s == bs && h > bh)) { best = $1; brun = ($3 == "red" ? $5 : "not_measured"); bs = s; bh = h } }
        END {
            if (nv == 0) { print "NOT RELEASABLE: nightly-train, the lane table has no verdict lane"; exit }
            if (bad == 0 && C ~ /^[0-9a-f]{40}$/) { print "RELEASABLE H=" C; exit }
            if (bad == 0) { print "NOT RELEASABLE: nightly-train, no commit read"; exit }
            printf "NOT RELEASABLE: %s, %s%s\n", best, brun, (bad > 1 ? " (+" bad - 1 " more)" : "")
        }' "$raw/hours.tsv" "$raw/lanes.tsv" > "$raw/line"
}
# greens HISTORY DATE -> the streak of nightly-train in HISTORY by nightly_greens.sh's rules, or not_measured
greens() {
    local o s
    [ -f "$HERE/nightly_greens.sh" ] || { printf 'not_measured (nightly_greens.sh absent)\n'; return 0; }
    o="$(bash "$HERE/nightly_greens.sh" --check nightly-train --history "$1" --as-of "$2" 2>&1)"
    s="$(printf '%s\n' "$o" | awk 'match($0, /streak=[0-9]+/) { print substr($0, RSTART + 7, RLENGTH - 7); exit }')"
    if [ -n "$s" ]; then printf '%s\n' "$s"; else printf 'not_measured\n'; fi
}
# ---------------------------------------------------------------- the read (network) -------------------------------
CALLS=0
call_ok() { [ "$CALLS" -lt "$MAX_CALLS" ] || return 1; CALLS=$((CALLS + 1)); }
# normalize GQL WORKFLOWS -> runs.tsv rows on stdout
# gql_query LANES WFJSON -> the one GraphQL query; empty when a producer workflow is missing from the list.
# jq 1.6 compatible (the timer PATH may find it first): no reserved words such as $or as variable names.
gql_query() {
    printf '%s\n' "$1" | awk -F ';' '$4 != "-" { print $4 }' | sort -u | jq -R -s --slurpfile wf "$2" -r --arg repo "$REPO" '
        split("\n") | map(select(length > 0)) as $want
        | ($wf | first | .workflows | map(select(.path as $x | $want | index($x))) ) as $hit
        | if ($hit | length) != ($want | length) then error("producer workflow missing from the list") else . end
        | ($repo | split("/")) as $own
        | "query { repository(owner: \"\($own | first)\", name: \"\($own | last)\") { defaultBranchRef { name target { ... on Commit { oid tree { oid } statusCheckRollup { contexts(first: 100) { pageInfo { hasNextPage } nodes { ... on CheckRun { name status conclusion startedAt completedAt checkSuite { status conclusion branch { name } workflowRun { databaseId event createdAt workflow { id } } } } } } } } } } } "
          + ([$hit | to_entries[] | "w\(.key): node(id: \"\(.value.node_id)\") { ... on Workflow { id runs(first: 12) { nodes { databaseId createdAt event checkSuite { status conclusion branch { name } commit { oid } checkRuns(first: 100, filterBy: {checkType: ALL}) { pageInfo { hasNextPage } nodes { name status conclusion startedAt completedAt } } } } } } }"] | join(" ")) + " }"' 2>/dev/null
}
normalize() {
    jq -r --slurpfile wf "$2" '
      ($wf | first | .workflows | map({key: .node_id, value: .path}) | from_entries) as $p
      | .data.repository.defaultBranchRef.target as $c
      | ( .data | to_entries[] | select(.key | test("^w[0-9]+$")) | .value as $w | ($w.runs.nodes // [])[] | . as $r
          | ($r.checkSuite.checkRuns.nodes // []) as $j
          | if ($j | length) == 0 then
              ["wf", ($p | .[$w.id]), $r.databaseId, $r.event, ($r.checkSuite.branch.name // ""), $r.checkSuite.commit.oid, $r.createdAt,
               $r.checkSuite.status, ($r.checkSuite.conclusion // ""), "", "", "", "", "", 0]
            else ($j | .[]) as $k
              | ["wf", ($p | .[$w.id]), $r.databaseId, $r.event, ($r.checkSuite.branch.name // ""), $r.checkSuite.commit.oid, $r.createdAt,
                 $r.checkSuite.status, ($r.checkSuite.conclusion // ""), $k.name, $k.status, ($k.conclusion // ""),
                 ($k.startedAt // ""), ($k.completedAt // ""), ([$j | .[] | select(.name == $k.name)] | length)]
            end,
            # a cut job list measures nothing: mark the run so the judge cannot read a partial set as green
            (if ($r.checkSuite.checkRuns.pageInfo.hasNextPage // false) then
              ["wf", ($p | .[$w.id]), $r.databaseId, $r.event, ($r.checkSuite.branch.name // ""), $r.checkSuite.commit.oid, $r.createdAt,
               $r.checkSuite.status, ($r.checkSuite.conclusion // ""), "#truncated", "COMPLETED", "TRUNCATED", "", "", 1]
             else empty end) ),
        ( ($c.statusCheckRollup.contexts.nodes // [])[] | select(.checkSuite.workflowRun != null)
          | .checkSuite.workflowRun.workflow.id as $wid | ["rollup", ($p | .[$wid]), .checkSuite.workflowRun.databaseId, .checkSuite.workflowRun.event,
             (.checkSuite.branch.name // ""), $c.oid, .checkSuite.workflowRun.createdAt, .checkSuite.status, (.checkSuite.conclusion // ""),
             .name, .status, (.conclusion // ""), (.startedAt // ""), (.completedAt // ""), 1] ),
        ( if ($c.statusCheckRollup.contexts.pageInfo.hasNextPage // false) then
            [($c.statusCheckRollup.contexts.nodes // [])[] | select(.checkSuite.workflowRun != null)] | unique_by(.checkSuite.workflowRun.databaseId) | .[]
            | .checkSuite.workflowRun.workflow.id as $wid | ["rollup", ($p | .[$wid]), .checkSuite.workflowRun.databaseId, .checkSuite.workflowRun.event,
               (.checkSuite.branch.name // ""), $c.oid, .checkSuite.workflowRun.createdAt, .checkSuite.status, (.checkSuite.conclusion // ""),
               "#truncated", "COMPLETED", "TRUNCATED", "", "", 1]
          else empty end )
      | map(tostring) | join("\t")' "$1"
}
# fetch RAW CACHE -> RAW/{C,tree,read,runs.tsv,attempts.tsv,graphql.json,workflows.json}; never fails: a failed read
#   is recorded in RAW/read and makes every producer lane not_measured
fetch() {
    local raw="$1" cache="$2" rem hdr st q a ids r
    printf 'failed: not read\n' > "$raw/read"; : > "$raw/C"; : > "$raw/tree"; : > "$raw/runs.tsv"; : > "$raw/attempts.tsv"
    rem="$(gh api rate_limit --jq .resources.core.remaining 2>/dev/null)"
    case "$rem" in ''|*[!0-9]*) printf 'failed: rate_limit unreadable\n' > "$raw/read"; return 0 ;; esac
    [ "$rem" -ge "$RATE_FLOOR" ] || { printf 'failed: core remaining %s under %s\n' "$rem" "$RATE_FLOOR" > "$raw/read"; return 0; }
    mkdir -p "$cache"
    call_ok || { printf 'failed: call budget\n' > "$raw/read"; return 0; }
    hdr=()
    [ -s "$cache/workflows.etag" ] && [ -s "$cache/workflows.json" ] && hdr=(-H "If-None-Match: $(cat "$cache/workflows.etag")")
    gh api -i "${hdr[@]}" "repos/$REPO/actions/workflows?per_page=100" > "$raw/workflows.http" 2>/dev/null
    st="$(awk 'NR == 1 { print $2; exit }' "$raw/workflows.http")"
    if [ "$st" = "200" ]; then
        awk 'f { print } /^\r?$/ { f = 1 }' "$raw/workflows.http" > "$cache/workflows.json"
        awk 'tolower($1) == "etag:" { sub(/\r$/, ""); sub(/^[^:]*: */, ""); print; exit }' "$raw/workflows.http" > "$cache/workflows.etag"
    elif [ "$st" = "304" ]; then CALLS=$((CALLS - 1))
    else printf 'failed: workflow list HTTP %s\n' "${st:-none}" > "$raw/read"; return 0; fi
    cp "$cache/workflows.json" "$raw/workflows.json" 2>/dev/null || { printf 'failed: no workflow list\n' > "$raw/read"; return 0; }
    # one GraphQL query: every producer workflow's recent runs, and main's head with its check rollup
    q="$(gql_query "$LANES" "$raw/workflows.json")"
    [ -n "$q" ] || { printf 'failed: a producer workflow is not in the workflow list\n' > "$raw/read"; return 0; }
    call_ok || { printf 'failed: call budget\n' > "$raw/read"; return 0; }
    gh api graphql -f query="$q" > "$raw/graphql.json" 2>/dev/null || { printf 'failed: GraphQL call\n' > "$raw/read"; return 0; }
    if jq -e '(.errors // []) | length > 0' "$raw/graphql.json" > /dev/null 2>&1; then printf 'failed: GraphQL errors\n' > "$raw/read"; return 0; fi
    jq -r '.data.repository.defaultBranchRef | select(.name == "main") | .target.oid // empty' "$raw/graphql.json" > "$raw/C" 2>/dev/null
    jq -r '.data.repository.defaultBranchRef.target.tree.oid // empty' "$raw/graphql.json" > "$raw/tree" 2>/dev/null
    grep -q -E '^[0-9a-f]{40}$' "$raw/C" || { printf 'failed: no head of main in the response\n' > "$raw/read"; return 0; }
    normalize "$raw/graphql.json" "$raw/workflows.json" > "$raw/runs.tsv" 2>/dev/null || { printf 'failed: response did not normalize\n' > "$raw/read"; return 0; }
    printf 'ok\n' > "$raw/read"
    # run_attempt for each green verdict candidate (GraphQL has none); an unread attempt stays unread -> not_measured
    ids="$(evaluate "$LANES" "$raw" cand | sort -u)"
    for r in $ids; do
        call_ok || break
        a="$(gh api "repos/$REPO/actions/runs/$r" --jq .run_attempt 2>/dev/null)"
        case "$a" in ''|*[!0-9]*) a="?" ;; esac
        printf '%s\t%s\n' "$r" "$a" >> "$raw/attempts.tsv"
    done
}
# ---------------------------------------------------------------- the run -----------------------------------------
STEP="start"; PRINTED=""; HISTDONE=""; OUTDIR=""; INBOXF=""; DAY=""; CSHA=""; PIN=""; PINNED=""; EXIT_VERDICT=""
step() { STEP="$1"; [ "${NIGHTLY_TRAIN_FAULT:-}" != "$1" ] || { STEP="$1 (planted fault)"; exit 1; }; }
# no_token -> 0 when no registry token is reachable: no CARGO_REGISTRY_TOKEN or CARGO_REGISTRIES_*_TOKEN in the
#   environment (set at all, even empty) and no credentials file under $CARGO_HOME (default ~/.cargo). The train
#   runs no cargo, but the rule is structural: it must be unable to upload, so it refuses to start where a token
#   exists. A flag alone is not a guard.
no_token() {
    local ch="${CARGO_HOME:-$HOME/.cargo}"
    env | awk -F '=' '$1 == "CARGO_REGISTRY_TOKEN" || $1 ~ /^CARGO_REGISTRIES_[A-Za-z0-9_]+_TOKEN$/ { f = 1 } END { exit f }' || return 1
    [ ! -e "$ch/credentials" ] && [ ! -e "$ch/credentials.toml" ]
}
# verdict_rc LINE -> 0 for RELEASABLE, 1 for anything else (--exit-verdict: a green job means a releasable head)
verdict_rc() { case "$1" in "RELEASABLE "*) return 0 ;; *) return 1 ;; esac; }
inbox_line() {   # inbox_line LINE GREENS -> append <= 300 bytes and read it back
    local l
    [ -n "$INBOXF" ] || return 0
    step inbox
    l="$(date -u +%H:%MZ) | aprender-a7 nightly-r4 | R4 | $1 | greens in a row: $2 | C=${CSHA:0:10}"
    l="$(printf '%s' "$l" | head -c 300)"
    printf '%s\n' "$l" >> "$INBOXF" 2>/dev/null \
        || printf '%s\n' "$l" | sg "$(stat -c %G -- "$INBOXF")" -c "tee -a $INBOXF > /dev/null"   # the file's own group: no name in the repo
    [ "$(tail -n 1 "$INBOXF" 2>/dev/null)" = "$l" ] || { printf 'inbox: read-back mismatch\n' >&2; return 1; }
}
on_exit() {
    local rc=$?
    # once the history row is written the run is recorded: a later failure (a closed stdout) rewrites nothing
    [ -z "$PRINTED$HISTDONE" ] || { [ -z "${DRILL:-}" ] || rm -f -- "${INBOXF:?}"; exit "$rc"; }
    local l="NOT RELEASABLE: nightly-train, $STEP failed (exit $rc) [C=${CSHA:0:10} pin=${PIN:-unpinned}]"
    printf '%s\n' "$l"
    if [ -n "$OUTDIR" ] && [ -n "$DAY" ] && mkdir -p "$OUTDIR/$DAY" 2>/dev/null; then
        printf '%s\n' "$l" > "$OUTDIR/$DAY/line"
        [ -f "$OUTDIR/history.tsv" ] || printf 'run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\n' > "$OUTDIR/history.tsv"
        printf 'nt-%s\t%s\tmain\t%s\tfailure\t%s\n' "${NOW//[-:]/}" "$NOW" "${GITHUB_EVENT_NAME:-timer}" "${GITHUB_RUN_ATTEMPT:-1}" >> "$OUTDIR/history.tsv"
    fi
    inbox_line "$l" "not_measured"
    [ -z "${DRILL:-}" ] || rm -f -- "${INBOXF:?}"
    exit 2
}
run_train() {
    local from="$1" raw line g concl budget pin_t pin_g pin_r hrow
    trap on_exit EXIT
    DAY="${NOW%%T*}"   # before any step that can fail, so every failure leaves its line and its history row
    step no-token
    no_token || exit 1
    step pin
    # the timer passes --pinned: then PIN and SHA256SUMS must both be there, so deleting them cannot skip the check
    if [ -n "$PINNED" ] || [ -f "$HERE/PIN" ]; then
        [ -f "$HERE/PIN" ] && [ -f "$HERE/SHA256SUMS" ] || { PIN="missing"; exit 1; }
        awk '$2 == "PIN" { f = 1 } END { exit !f }' "$HERE/SHA256SUMS" || { PIN="unhashed"; exit 1; }   # SHA256SUMS must cover PIN too
        pin_t="$(awk '$1 == "train" { print substr($2, 1, 10) }' "$HERE/PIN")"
        pin_g="$(awk '$1 == "greens" { print substr($2, 1, 10) }' "$HERE/PIN")"
        pin_r="$(awk '$1 == "redage" { print substr($2, 1, 10) }' "$HERE/PIN")"
        PIN="$pin_t.$pin_g.$pin_r"
        # what runs must be what was pinned: an edited bundle refuses (the trap line names the step)
        (cd "$HERE" && sha256sum --quiet -c SHA256SUMS) > /dev/null 2>&1 || { PIN="$PIN!modified"; exit 1; }
    else PIN="unpinned"; fi
    step tools
    for t in jq awk sort date; do command -v "$t" > /dev/null || exit 1; done
    [ -n "$from" ] || command -v gh > /dev/null || exit 1
    raw="$OUTDIR/$DAY/raw"
    step outdir
    mkdir -p "$raw" || exit 1
    step read
    if [ -n "$from" ]; then
        for f in C read runs.tsv attempts.tsv tree; do [ "$from/$f" -ef "$raw/$f" ] || cp "$from/$f" "$raw/$f" 2>/dev/null || : > "$raw/$f"; done
    else fetch "$raw" "$OUTDIR/cache"; fi
    CSHA="$(cat "$raw/C" 2>/dev/null)"
    step judge
    decide "$LANES" "$raw" "$NOW" || exit 1
    step rank
    [ -s "$raw/line" ] || exit 1
    cp "$raw/reds.tsv" "$OUTDIR/$DAY/reds.tsv" || exit 1
    step greens
    line="$(cat "$raw/line") [C=${CSHA:0:10} pin=$PIN]"
    case "$line" in "RELEASABLE "*) concl=success ;; *) if awk -F '\t' '$2 == "verdict" && $3 == "red" { f = 1 } END { exit !f }' "$raw/lanes.tsv"; then concl=failure; else concl=neutral; fi ;; esac
    # tonight's row goes into a copy first: history.tsv gets exactly one row per run, written after everything but stdout
    if [ -f "$OUTDIR/history.tsv" ]; then cp "$OUTDIR/history.tsv" "$raw/history.next" || exit 1
    else printf 'run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\n' > "$raw/history.next" || exit 1; fi
    hrow="$(printf 'nt-%s\t%s\tmain\t%s\t%s\t%s' "${NOW//[-:]/}" "$NOW" "${GITHUB_EVENT_NAME:-timer}" "$concl" "${GITHUB_RUN_ATTEMPT:-1}")"
    printf '%s\n' "$hrow" >> "$raw/history.next" || exit 1
    g="$(greens "$raw/history.next" "$DAY")"
    step bundle
    budget="$(awk -F '\t' -v C="$CSHA" '$6 == C && $9 != "" { if (s == "" || $9 < s) s = $9; if ($10 > e) e = $10 } END { print s "\t" e }' "$raw/lanes.tsv")"
    {
        printf 'lane\tkind\tstate\tproducer\trun_id\trun_head\tconclusion\tattempt\tstarted\tended\treason\thours_lost\n'
        awk -F '\t' -v OFS='\t' 'FILENAME == ARGV[1] { H[$1] = $2; next } { print $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, (($1 in H) ? H[$1] : "-") }' "$raw/hours.tsv" "$raw/lanes.tsv"
        printf '# C\t%s\n# tree\t%s\n# read\t%s\n# command\t%s\n# pin\ttrain=%s greens=%s redage=%s\n' "$CSHA" "$(cat "$raw/tree")" "$(cat "$raw/read")" "$PROG ${ARGS_SEEN:-}" "${pin_t:-unpinned}" "${pin_g:-}" "${pin_r:-}"
        printf '# budget\tfirst start %s, last end %s, %s (6 h budget; info only, not in the verdict)\n' "${budget%%	*}" "${budget##*	}" "$(span "${budget%%	*}" "${budget##*	}")"
        printf '# github calls\t%s\n# greens in a row\t%s\n# as of\t%s\n' "$CALLS" "$g" "$NOW"
    } > "$OUTDIR/$DAY/bundle.tsv" || exit 1
    step print
    printf '%s\n' "$line" > "$OUTDIR/$DAY/line" || exit 1
    step history
    [ -f "$OUTDIR/history.tsv" ] || printf 'run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\n' > "$OUTDIR/history.tsv" || exit 1
    printf '%s\n' "$hrow" >> "$OUTDIR/history.tsv" || exit 1
    HISTDONE=1
    printf '%s\n' "$line" || exit 1; PRINTED=1
    inbox_line "$line" "$g" || exit 1
    [ -z "$EXIT_VERDICT" ] || verdict_rc "$line" || exit 1
    exit 0
}
span() {   # span START END -> XhYYm, or not_measured
    local a b
    a="$(date -u -d "$1" +%s 2>/dev/null)"; b="$(date -u -d "$2" +%s 2>/dev/null)"
    if [ -z "$1" ] || [ -z "$2" ] || [ -z "$a" ] || [ -z "$b" ]; then printf 'not_measured'; return 0; fi
    printf '%dh%02dm' $(((b - a) / 3600)) $((((b - a) % 3600) / 60))
}
# ---------------------------------------------------------------- install (the pinned bundle and the user timer) ----
install_timer() {
    local home="" out="" inbox="" t="" g="" r="" dir unitdir p
    while [ $# -gt 0 ]; do
        case "$1" in
            --home) home="${2:-}"; shift 2 ;; --out) out="${2:-}"; shift 2 ;; --inbox) inbox="${2:-}"; shift 2 ;;
            --train) t="${2:-}"; shift 2 ;; --greens) g="${2:-}"; shift 2 ;; --redage) r="${2:-}"; shift 2 ;;
            *) caller_error "--install: unknown argument $1" ;;
        esac
    done
    [ -n "$home" ] && [ -n "$out" ] && [ -n "$t" ] && [ -n "$g" ] && [ -n "$r" ] || caller_error "--install needs --home --out --train --greens --redage"
    case "$home$out$inbox" in *[[:space:]%\\\"\']*) caller_error "--install: paths may not hold spaces, quotes, % or backslashes" ;; esac
    t="$(git rev-parse --verify --quiet "$t^{commit}")" || caller_error "--train: not a commit"
    g="$(git rev-parse --verify --quiet "$g^{commit}")" || caller_error "--greens: not a commit"
    r="$(git rev-parse --verify --quiet "$r^{commit}")" || caller_error "--redage: not a commit"
    [ -n "$(git branch -r --contains "$t" 2>/dev/null)" ] || caller_error "--train $t is not on any remote branch: push it first"
    [ -n "$(git branch -r --contains "$g" 2>/dev/null)" ] || caller_error "--greens $g is not on any remote branch"
    [ -n "$(git branch -r --contains "$r" 2>/dev/null)" ] || caller_error "--redage $r is not on any remote branch"
    command -v gh > /dev/null && command -v jq > /dev/null || caller_error "gh and jq must be on PATH: the unit PATH is built from them"
    [ "$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null)" = yes ] || caller_error "Linger is not yes: a user timer does not fire while logged out; nothing installed"
    dir="$home/b-${t:0:10}-${g:0:10}-${r:0:10}"
    mkdir -p "$dir" "$dir/cargo" "$out" || caller_error "cannot create $dir or $out"   # cargo: an empty CARGO_HOME for the unit
    chmod u+w "$dir"/*.sh 2>/dev/null   # a reinstall of the same pin overwrites the read-only copies
    git show "$t:scripts/release/nightly_train.sh" > "$dir/nightly_train.sh" || caller_error "no nightly_train.sh at $t"
    git show "$g:scripts/release/nightly_greens.sh" > "$dir/nightly_greens.sh" || caller_error "no nightly_greens.sh at $g"
    git show "$r:scripts/release/red_age.sh" > "$dir/red_age.sh" || caller_error "no red_age.sh at $r"
    printf 'train %s\ngreens %s\nredage %s\n' "$t" "$g" "$r" > "$dir/PIN"
    chmod 0555 "$dir"/*.sh
    (cd "$dir" && sha256sum nightly_train.sh nightly_greens.sh red_age.sh PIN > SHA256SUMS) || caller_error "cannot write SHA256SUMS"
    # the self-test runs under the unit's own PATH: a tool the timer resolves differently (jq 1.6) must fail here
    p="$(dirname "$(command -v gh)"):$(dirname "$(command -v jq)"):/usr/local/bin:/usr/bin:/bin"
    PATH="$p" bash "$dir/nightly_train.sh" --self-test > "$dir/self-test.out" 2>&1 || { tail -n 5 "$dir/self-test.out"; caller_error "the pinned self-test is RED; nothing installed"; }
    tail -n 1 "$dir/self-test.out"
    unitdir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
    mkdir -p "$unitdir" || caller_error "cannot create $unitdir"
    {
        printf '[Unit]\nDescription=aprender nightly evidence train (BLD-002 R4), report-only\n\n[Service]\nType=oneshot\nNice=10\nTimeoutStartSec=900\n'
        printf 'Environment=OUT=%s\nEnvironment=INBOX=%s\nEnvironment=PATH=%s\nEnvironment=CARGO_HOME=%s/cargo\nExecStart=/bin/bash %s/nightly_train.sh --pinned\n' "$out" "$inbox" "$p" "$dir" "$dir"
    } > "$unitdir/$UNIT.service"
    printf '[Unit]\nDescription=aprender nightly evidence train, daily 04:45 UTC\n\n[Timer]\nOnCalendar=*-*-* 04:45:00 UTC\nPersistent=true\nUnit=%s.service\n\n[Install]\nWantedBy=timers.target\n' "$UNIT" > "$unitdir/$UNIT.timer"
    systemctl --user daemon-reload || caller_error "daemon-reload failed"
    systemctl --user enable --now "$UNIT.timer" > /dev/null 2>&1 || caller_error "enable --now $UNIT.timer failed"
    printf 'installed %s/%s.{service,timer} -> %s\n' "$unitdir" "$UNIT" "$dir"
    systemctl --user list-timers "$UNIT.timer" --no-pager | head -n 2
    loginctl show-user "$(id -un)" -p Linger 2>/dev/null
}
# ---------------------------------------------------------------- the case table ----------------------------------
ST_LANES='v-a;verdict;check A;.github/workflows/a.yml;^schedule$;^job-a$
v-b;verdict;check B;.github/workflows/b.yml;^schedule$;
v-c;verdict;check C;.github/workflows/c.yml;^push$;^(gate|test)$;2
i-d;info;check D;.github/workflows/d.yml;^schedule$;'
ST_C="$(printf 'a%.0s' $(seq 40))"
ST_X="$(printf 'b%.0s' $(seq 40))"
# fixture DIR -> every lane green on C, attempt 1
fixture() {
    mkdir -p "$1"
    printf '%s\n' "$ST_C" > "$1/C"; printf 'ok\n' > "$1/read"; printf '%s\n' "$(printf 'c%.0s' $(seq 40))" > "$1/tree"
    {
        printf 'wf\t.github/workflows/a.yml\t101\tschedule\tmain\t%s\t2026-10-04T01:00:00Z\tCOMPLETED\tSUCCESS\tjob-a\tCOMPLETED\tSUCCESS\t2026-10-04T01:01:00Z\t2026-10-04T01:30:00Z\t1\n' "$ST_C"
        printf 'wf\t.github/workflows/a.yml\t101\tschedule\tmain\t%s\t2026-10-04T01:00:00Z\tCOMPLETED\tSUCCESS\tother\tCOMPLETED\tFAILURE\t2026-10-04T01:01:00Z\t2026-10-04T01:02:00Z\t1\n' "$ST_C"
        printf 'wf\t.github/workflows/b.yml\t201\tschedule\tmain\t%s\t2026-10-04T02:00:00Z\tCOMPLETED\tSUCCESS\tx\tCOMPLETED\tSUCCESS\t2026-10-04T02:01:00Z\t2026-10-04T02:40:00Z\t1\n' "$ST_C"
        printf 'rollup\t.github/workflows/c.yml\t301\tpush\tmain\t%s\t2026-10-03T03:00:00Z\tCOMPLETED\tSUCCESS\tgate\tCOMPLETED\tSUCCESS\t2026-10-03T03:01:00Z\t2026-10-03T04:00:00Z\t1\n' "$ST_C"
        printf 'rollup\t.github/workflows/c.yml\t301\tpush\tmain\t%s\t2026-10-03T03:00:00Z\tCOMPLETED\tSUCCESS\ttest\tCOMPLETED\tSUCCESS\t2026-10-03T03:01:00Z\t2026-10-03T04:30:00Z\t1\n' "$ST_C"
        printf 'wf\t.github/workflows/d.yml\t401\tschedule\tmain\t%s\t2026-10-04T03:00:00Z\tCOMPLETED\tSUCCESS\td\tCOMPLETED\tSUCCESS\t2026-10-04T03:01:00Z\t2026-10-04T03:20:00Z\t1\n' "$ST_C"
    } > "$1/runs.tsv"
    printf '101\t1\n201\t1\n301\t1\n' > "$1/attempts.tsv"
}
# st_decide DIR -> the line, the lanes and the reds, as one text
st_decide() { decide "$ST_LANES" "$1" 2026-10-04T06:00:00Z; cat "$1/line" "$1/lanes.tsv" "$1/reds.tsv"; }
self_test() {
    local tmp pass=0 fail=0 d o rc v
    tmp="$(mktemp -d)" || exit 3
    export INBOX="$tmp/inbox.md"   # whatever INBOX the caller set: a self-test never writes a real inbox
    # and no registry token: the rows run the train, which refuses to start where one is reachable
    export CARGO_HOME="$tmp/cargo"; mkdir -p "$CARGO_HOME"; unset CARGO_REGISTRY_TOKEN
    while read -r v; do unset "$v"; done < <(env | awk -F "=" '$1 ~ /^CARGO_REGISTRIES_[A-Za-z0-9_]+_TOKEN$/ { print $1 }')
    # row NAME EXPECT-RC NEEDLE FORBID -- CMD...
    row() {
        local name="$1" expect="$2" needle="$3" forbid="$4"; shift 5
        o="$("$@" 2>&1)"; rc=$?
        if [ "$rc" != "$expect" ]; then printf '  BROKE %-44s expected exit %s got %s\n%s\n' "$name" "$expect" "$rc" "$o"; fail=$((fail + 1)); return 0; fi
        case "$o" in *"$needle"*) ;; *) printf '  BROKE %-44s never said: %s\n%s\n' "$name" "$needle" "$o"; fail=$((fail + 1)); return 0 ;; esac
        if [ -n "$forbid" ]; then case "$o" in *"$forbid"*) printf '  BROKE %-44s said: %s\n' "$name" "$forbid"; fail=$((fail + 1)); return 0 ;; esac; fi
        printf '  ok    %-44s exit=%s\n' "$name" "$rc"; pass=$((pass + 1))
    }
    d="$tmp/green"; fixture "$d"
    row all_verdict_lanes_green_is_releasable 0 "RELEASABLE H=$ST_C" "NOT RELEASABLE" -- st_decide "$d"
    d="$tmp/removed"; fixture "$d"; sed -i '/\t201\t/d' "$d/runs.tsv"
    row one_lane_removed_names_it 0 "NOT RELEASABLE: v-b, not_measured" "RELEASABLE H=" -- st_decide "$d"
    d="$tmp/other"; fixture "$d"; sed -i "/\t101\t/s/$ST_C/$ST_X/" "$d/runs.tsv"
    row run_on_another_commit_is_not_measured 0 "newest run 101 on bbbbbbbbbb" "RELEASABLE H=" -- st_decide "$d"
    d="$tmp/running"; fixture "$d"; sed -i '/\t101\t/s/\tCOMPLETED\tSUCCESS\tjob-a/\tIN_PROGRESS\t\tjob-a/' "$d/runs.tsv"
    row run_in_progress_is_not_measured 0 "in progress: run 101 is in_progress" "RELEASABLE H=" -- st_decide "$d"
    d="$tmp/retry"; fixture "$d"; sed -i 's/^101\t1$/101\t2/' "$d/attempts.tsv"
    row green_at_attempt_2_is_not_green 0 "NOT RELEASABLE: v-a, 101" "RELEASABLE H=" -- st_decide "$d"
    d="$tmp/noread"; fixture "$d"; printf 'failed: core remaining 12 under 1000\n' > "$d/read"
    row failed_read_is_not_measured 0 "NOT RELEASABLE: v-a, not_measured (+2 more)" "RELEASABLE H=" -- st_decide "$d"
    d="$tmp/noatt"; fixture "$d"; sed -i 's/^201\t1$/201\t?/' "$d/attempts.tsv"
    row unread_attempt_is_not_measured 0 "attempt not read" "RELEASABLE H=" -- st_decide "$d"
    d="$tmp/info"; fixture "$d"; sed -i '/\t401\t/s/\tCOMPLETED\tSUCCESS\t2026-10-04T03:01/\tCOMPLETED\tFAILURE\t2026-10-04T03:01/' "$d/runs.tsv"
    row info_lane_red_leaves_the_verdict 0 "RELEASABLE H=$ST_C" "NOT RELEASABLE" -- st_decide "$d"
    d="$tmp/rank"; fixture "$d"
    sed -i -e '/\t101\t/s/2026-10-04T01:00:00Z/2026-10-02T01:00:00Z/' -e '/\t101\t/s/\tSUCCESS\tjob-a\tCOMPLETED\tSUCCESS/\tFAILURE\tjob-a\tCOMPLETED\tFAILURE/' \
        -e '/\t201\t/s/\tCOMPLETED\tSUCCESS\tx\tCOMPLETED\tSUCCESS/\tCOMPLETED\tFAILURE\tx\tCOMPLETED\tFAILURE/' "$d/runs.tsv"
    printf 'wf\t.github/workflows/a.yml\t100\tschedule\tmain\t%s\t2026-10-01T01:00:00Z\tCOMPLETED\tSUCCESS\tjob-a\tCOMPLETED\tSUCCESS\t\t\t1\n' "$ST_X" >> "$d/runs.tsv"
    printf 'wf\t.github/workflows/b.yml\t200\tschedule\tmain\t%s\t2026-10-03T01:00:00Z\tCOMPLETED\tSUCCESS\tx\tCOMPLETED\tSUCCESS\t\t\t1\n' "$ST_X" >> "$d/runs.tsv"
    row reds_ranked_by_hours_lost 0 "$(printf '1\tv-a\tverdict\tred\t53.00\t101')" "" -- st_decide "$d"
    row the_line_names_the_longest_red 0 "NOT RELEASABLE: v-a, 101 (+1 more)" "" -- cat "$d/line"
    printf 'run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\n' > "$tmp/hist"
    printf 'n%s\t2026-%s\tmain\tschedule\t%s\t1\n' 1 09-30T04:45:00Z success 2 10-01T04:45:00Z failure 3 10-02T04:45:00Z success 4 10-03T04:45:00Z success >> "$tmp/hist"
    printf 't5\t2026-10-04T04:45:00Z\tmain\ttimer\tsuccess\t1\n' >> "$tmp/hist"
    row greens_in_a_row_by_nightly_greens 0 "greens=2" "" -- eval 'printf "greens=%s\n" "$(greens "$tmp/hist" 2026-10-04)"'
    printf 'run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\nt1\t2026-10-03T04:45:00Z\tmain\ttimer\tsuccess\t1\n' > "$tmp/thist"
    row timer_runs_never_count_toward_b1 0 "greens=0" "" -- eval 'printf "greens=%s\n" "$(greens "$tmp/thist" 2026-10-04)"'
    d="$tmp/trap"; fixture "$d"
    row failure_part_way_prints_the_trap_line 2 "NOT RELEASABLE: nightly-train, judge (planted fault) failed" "RELEASABLE H=" -- \
        env NIGHTLY_TRAIN_FAULT=judge bash "$SCRIPT_PATH" --from "$d" --out "$tmp/out" --now 2026-10-04T06:00:00Z
    row the_trap_line_is_recorded 0 "NOT RELEASABLE: nightly-train, judge" "" -- cat "$tmp/out/2026-10-04/line"
    row verdict_lanes_are_release_day_checks 0 "verdict=ci-main deep-doctests deep-nodefault deep-examples deep-bins-build deep-bins-smoke dogfood models readiness milestone cleanroom-cpu cleanroom-gpu assets preflight publish-dryrun;coverage=info" "" -- \
        eval 'printf "verdict=%s;coverage=%s\n" "$(printf "%s\n" "$LANES" | awk -F ";" "\$2 == \"verdict\" { printf \"%s%s\", s, \$1; s = \" \" }")" "$(printf "%s\n" "$LANES" | awk -F ";" "\$1 == \"coverage\" { print \$2 }")"'
    st_norm() {
        printf '{"workflows":[{"node_id":"WA","path":".github/workflows/a.yml"},{"node_id":"WC","path":".github/workflows/c.yml"}]}\n' > "$tmp/wf.json"
        printf '%s' '{"data":{"repository":{"defaultBranchRef":{"name":"main","target":{"oid":"'"$ST_C"'","tree":{"oid":"t"},"statusCheckRollup":{"contexts":{"nodes":[{},{"name":"gate","status":"COMPLETED","conclusion":"SUCCESS","startedAt":"s","completedAt":"e","checkSuite":{"status":"COMPLETED","conclusion":"SUCCESS","branch":{"name":"main"},"workflowRun":{"databaseId":301,"event":"push","createdAt":"c","workflow":{"id":"WC"}}}}]}}}}},"w0":{"id":"WA","runs":{"nodes":[{"databaseId":101,"createdAt":"c1","event":"schedule","checkSuite":{"status":"COMPLETED","conclusion":"SUCCESS","branch":{"name":"main"},"commit":{"oid":"'"$ST_C"'"},"checkRuns":{"nodes":[{"name":"job-a","status":"COMPLETED","conclusion":"SUCCESS","startedAt":"s1","completedAt":"e1"},{"name":"job-a","status":"COMPLETED","conclusion":"FAILURE","startedAt":"s0","completedAt":"e0"}]}}}]}}}}' > "$tmp/gql.json"
        normalize "$tmp/gql.json" "$tmp/wf.json"
    }
    row normalize_maps_runs_jobs_and_the_rollup 0 "$(printf 'rollup\t.github/workflows/c.yml\t301\tpush\tmain\t%s\tc\tCOMPLETED\tSUCCESS\tgate' "$ST_C")" "" -- st_norm
    row normalize_counts_a_rerun_job 0 "$(printf 'job-a\tCOMPLETED\tSUCCESS\ts1\te1\t2')" "" -- st_norm
    row gql_query_names_every_producer 0 'w1: node(id: "WC")' "" -- gql_query "$(printf '%s\n' "$ST_LANES" | awk -F ';' '$4 ~ /[ac][.]yml$/')" "$tmp/wf.json"
    row gql_query_refuses_a_missing_producer 5 "" "query" -- gql_query "$ST_LANES" "$tmp/wf.json"
    d="$tmp/voidnew"; fixture "$d"
    printf 'wf\t.github/workflows/a.yml\t102\tschedule\tmain\t%s\t2026-10-04T05:00:00Z\tCOMPLETED\tCANCELLED\tjob-a\tCOMPLETED\tCANCELLED\t\t\t1\n' "$ST_C" >> "$d/runs.tsv"
    row a_void_newer_run_falls_to_the_older_green 0 "RELEASABLE H=$ST_C" "NOT RELEASABLE" -- st_decide "$d"
    d="$tmp/need"; fixture "$d"; sed -i '/\t301\t/{/\ttest\t/d}' "$d/runs.tsv"
    row a_missing_required_job_is_not_measured 0 "1 of 2 required job(s) present" "RELEASABLE H=" -- st_decide "$d"
    d="$tmp/cut"; fixture "$d"
    printf 'wf\t.github/workflows/a.yml\t101\tschedule\tmain\t%s\t2026-10-04T01:00:00Z\tCOMPLETED\tSUCCESS\t#truncated\tCOMPLETED\tTRUNCATED\t\t\t1\n' "$ST_C" >> "$d/runs.tsv"
    row a_cut_job_list_is_not_measured 0 "its job list was cut" "RELEASABLE H=" -- st_decide "$d"
    st_cut() {
        printf '%s' '{"data":{"repository":{"defaultBranchRef":{"name":"main","target":{"oid":"'"$ST_C"'","tree":{"oid":"t"},"statusCheckRollup":{"contexts":{"pageInfo":{"hasNextPage":false},"nodes":[]}}}}},"w0":{"id":"WA","runs":{"nodes":[{"databaseId":101,"createdAt":"c1","event":"schedule","checkSuite":{"status":"COMPLETED","conclusion":"SUCCESS","branch":{"name":"main"},"commit":{"oid":"'"$ST_C"'"},"checkRuns":{"pageInfo":{"hasNextPage":true},"nodes":[{"name":"job-a","status":"COMPLETED","conclusion":"SUCCESS","startedAt":"s","completedAt":"e"}]}}}]}}}}' > "$tmp/cut.json"
        normalize "$tmp/cut.json" "$tmp/wf.json"
    }
    row normalize_marks_a_cut_job_list 0 "$(printf '#truncated\tCOMPLETED\tTRUNCATED')" "" -- st_cut
    d="$tmp/noprod"; fixture "$d"
    row a_lane_without_a_producer_is_not_measured 0 "$(printf 'v-e\tverdict\tnot_measured')" "RELEASABLE H=" -- \
        eval 'decide "$ST_LANES
v-e;verdict;check E;-;-;-" "$d" 2026-10-04T06:00:00Z; cat "$d/line" "$d/lanes.tsv"'
    d="$tmp/green1"; fixture "$d"
    row stdout_is_exactly_one_line 0 "lines=1" "" -- \
        eval 'bash "$SCRIPT_PATH" --from "$d" --out "$tmp/o1" --now 2026-10-04T06:00:00Z | awk "END { print \"lines=\" NR }"'
    row a_fault_after_judging_leaves_one_history_row 0 "rows=1 failure=1" "" -- \
        eval 'NIGHTLY_TRAIN_FAULT=history bash "$SCRIPT_PATH" --from "$d" --out "$tmp/o3" --now 2026-10-04T06:00:00Z > /dev/null; awk -F "\t" "NR > 1 { n++; if (\$5 == \"failure\") f++ } END { print \"rows=\" n+0, \"failure=\" f+0 }" "$tmp/o3/history.tsv"'
    row a_fault_after_history_adds_no_second_row 0 "rows=1 failure=0" "" -- \
        eval 'NIGHTLY_TRAIN_FAULT=inbox bash "$SCRIPT_PATH" --from "$d" --out "$tmp/o4" --inbox "$tmp/ib4" --now 2026-10-04T06:00:00Z > /dev/null; awk -F "\t" "NR > 1 { n++; if (\$5 == \"failure\") f++ } END { print \"rows=\" n+0, \"failure=\" f+0 }" "$tmp/o4/history.tsv"'
    row a_closed_stdout_adds_no_second_row 0 "rows=1 failure=0" "" -- \
        eval 'bash "$SCRIPT_PATH" --from "$d" --out "$tmp/o5" --now 2026-10-04T06:00:00Z >&- 2> /dev/null; awk -F "\t" "NR > 1 { n++; if (\$5 == \"failure\") f++ } END { print \"rows=\" n+0, \"failure=\" f+0 }" "$tmp/o5/history.tsv"'
    row a_self_test_writes_only_its_own_inbox 0 "sandbox" "" -- \
        eval 'case "$INBOX" in "$tmp"/*) echo sandbox ;; *) echo "INBOX is $INBOX" ;; esac'
    row a_planted_fault_never_writes_the_named_inbox 0 "caller=0 lines=2" "" -- \
        eval 'INBOX="$tmp/c6.md" NIGHTLY_TRAIN_FAULT=bundle bash "$SCRIPT_PATH" --from "$d" --out "$tmp/o6" --now 2026-10-04T06:00:00Z > /dev/null; NIGHTLY_TRAIN_FAULT=bundle bash "$SCRIPT_PATH" --from "$d" --out "$tmp/o7" --inbox "$tmp/c7.md" --now 2026-10-04T06:00:00Z > /dev/null; printf "caller=%s lines=%s\n" "$(cat "$tmp/c6.md" "$tmp/c7.md" 2>/dev/null | wc -l)" "$(cat "$tmp/o6"/*/line "$tmp/o7"/*/line | grep -c "planted fault")"'
    d="$tmp/legs"; fixture "$d"
    printf 'wf\t.github/workflows/a.yml\t101\tschedule\tmain\t%s\t2026-10-04T01:00:00Z\tCOMPLETED\tSUCCESS\tjob-a\tCOMPLETED\tFAILURE\t2026-10-04T01:00:30Z\t2026-10-04T01:20:00Z\t2\n' "$ST_C" >> "$d/runs.tsv"
    row a_failed_parallel_leg_of_the_same_name_is_red 0 "NOT RELEASABLE: v-a, 101" "RELEASABLE H=" -- st_decide "$d"
    d="$tmp/infoatt"; fixture "$d"
    row an_info_lane_attempt_is_marked_inferred 0 "$(printf 'i-d\tinfo\tgreen\t.github/workflows/d.yml \t401\t%s\tsuccess\t1?' "$ST_C")" "" -- st_decide "$d"
    st_rcut() {
        printf '%s' '{"data":{"repository":{"defaultBranchRef":{"name":"main","target":{"oid":"'"$ST_C"'","tree":{"oid":"t"},"statusCheckRollup":{"contexts":{"pageInfo":{"hasNextPage":true},"nodes":[{"name":"gate","status":"COMPLETED","conclusion":"SUCCESS","startedAt":"s","completedAt":"e","checkSuite":{"status":"COMPLETED","conclusion":"SUCCESS","branch":{"name":"main"},"workflowRun":{"databaseId":301,"event":"push","createdAt":"c","workflow":{"id":"WC"}}}}]}}}}}}}' > "$tmp/rcut.json"
        normalize "$tmp/rcut.json" "$tmp/wf.json"
    }
    row normalize_marks_a_cut_rollup 0 "$(printf 'rollup\t.github/workflows/c.yml\t301\tpush\tmain\t%s\tc\tCOMPLETED\tSUCCESS\t#truncated\tCOMPLETED\tTRUNCATED' "$ST_C")" "" -- st_rcut
    # fetch against a stub gh: the refusals happen before any real call could be made
    mkdir -p "$tmp/bin"
    printf '%s\n' '#!/bin/bash' 'case "$*" in' \
        '  *rate_limit*) echo "${STUB_REM:-5000}" ;;' \
        '  *actions/workflows*) printf "HTTP/2.0 200 OK\r\netag: \"e1\"\r\n\r\n"; cat "$STUB_WF" ;;' \
        '  *graphql*) echo "{\"errors\":[{\"message\":\"stub\"}]}" ;;' \
        '  *) echo 1 ;;' 'esac' > "$tmp/bin/gh"; chmod +x "$tmp/bin/gh"
    printf '{"workflows":[%s]}\n' '{"node_id":"WA","path":".github/workflows/a.yml"},{"node_id":"WB","path":".github/workflows/b.yml"},{"node_id":"WC","path":".github/workflows/c.yml"},{"node_id":"WD","path":".github/workflows/d.yml"}' > "$tmp/wfall.json"
    st_fetch() { mkdir -p "$tmp/f$1"; PATH="$tmp/bin:$PATH" STUB_WF="$tmp/wfall.json" LANES="$ST_LANES" fetch "$tmp/f$1" "$tmp/f$1/cache"; cat "$tmp/f$1/read"; }
    row fetch_refuses_under_the_rate_floor 0 "failed: core remaining 999 under 1000" "" -- eval 'STUB_REM=999 st_fetch 1'
    row fetch_refuses_past_the_call_budget 0 "failed: call budget" "" -- eval 'MAX_CALLS=0; st_fetch 2'
    row fetch_refuses_a_graphql_error 0 "failed: GraphQL errors" "" -- eval 'CALLS=0; st_fetch 3'
    row an_inbox_read_back_mismatch_exits_non_zero 1 "inbox: read-back mismatch" "" -- \
        bash "$SCRIPT_PATH" --from "$d" --out "$tmp/o9" --inbox /dev/null --now 2026-10-04T06:00:00Z
    row a_closed_stdout_keeps_the_real_verdict 0 "NOT RELEASABLE: ci-main, not_measured (+14 more) [C=aaaaaaaaaa pin=" "nightly-train" -- cat "$tmp/o5/2026-10-04/line"
    row a_registry_token_refuses_to_start 2 "NOT RELEASABLE: nightly-train, no-token failed" "RELEASABLE H=" -- \
        env CARGO_REGISTRY_TOKEN=x bash "$SCRIPT_PATH" --from "$d" --out "$tmp/ot1" --now 2026-10-04T06:00:00Z
    row an_empty_named_registry_token_refuses 2 "NOT RELEASABLE: nightly-train, no-token failed" "RELEASABLE H=" -- \
        env CARGO_REGISTRIES_MIRROR_TOKEN= bash "$SCRIPT_PATH" --from "$d" --out "$tmp/ot2" --now 2026-10-04T06:00:00Z
    row a_credentials_file_refuses 2 "NOT RELEASABLE: nightly-train, no-token failed" "RELEASABLE H=" -- \
        eval 'mkdir -p "$tmp/ch"; : > "$tmp/ch/credentials.toml"; CARGO_HOME="$tmp/ch" bash "$SCRIPT_PATH" --from "$d" --out "$tmp/ot3" --now 2026-10-04T06:00:00Z'
    row exit_verdict_maps_releasable_to_0 0 "rc=0 rc=1" "" -- eval 'verdict_rc "RELEASABLE H=$ST_C"; a=$?; verdict_rc "NOT RELEASABLE: v-a, 101"; echo "rc=$a rc=$?"'
    row exit_verdict_not_releasable_exits_1 1 "NOT RELEASABLE: ci-main" "nightly-train" -- \
        bash "$SCRIPT_PATH" --from "$d" --out "$tmp/ov" --exit-verdict --now 2026-10-04T06:00:00Z
    mkdir -p "$tmp/pb" && cp "$SCRIPT_PATH" "$HERE/red_age.sh" "$HERE/nightly_greens.sh" "$tmp/pb/" 2>/dev/null; chmod u+w "$tmp/pb"/*.sh
    printf 'train %s\ngreens %s\nredage %s\n' "$ST_C" "$ST_C" "$ST_C" > "$tmp/pb/PIN"
    (cd "$tmp/pb" && sha256sum nightly_train.sh nightly_greens.sh red_age.sh PIN > SHA256SUMS) 2>/dev/null
    cp -r "$tmp/pb" "$tmp/pc"; chmod u+w "$tmp/pc"/*; rm -f -- "${tmp:?}/pc/PIN" "${tmp:?}/pc/SHA256SUMS"
    for x in pd pe pf; do cp -r "$tmp/pb" "$tmp/$x"; chmod u+w "$tmp/$x"/*; done
    printf 'train %s\n' "$ST_X" >> "$tmp/pd/PIN"
    (cd "$tmp/pf" && sha256sum nightly_train.sh nightly_greens.sh red_age.sh > SHA256SUMS) 2>/dev/null
    printf '# an edit after pinning\n' >> "$tmp/pb/nightly_train.sh"
    row an_edited_bundle_refuses 2 "NOT RELEASABLE: nightly-train, pin failed" "RELEASABLE H=" -- \
        bash "$tmp/pb/nightly_train.sh" --from "$d" --out "$tmp/o2" --now 2026-10-04T06:00:00Z
    row an_edited_bundle_leaves_its_line_and_one_row 0 "rows=1 failure=1 pin" "" -- \
        eval 'awk -F "\t" "NR > 1 { n++; if (\$5 == \"failure\") f++ } END { printf \"rows=%d failure=%d \", n, f }" "$tmp/o2/history.tsv"; grep -o "pin failed" "$tmp/o2/2026-10-04/line"'
    row a_pinned_run_of_an_intact_bundle_judges 0 "[C=aaaaaaaaaa pin=aaaaaaaaaa.aaaaaaaaaa.aaaaaaaaaa]" "nightly-train" -- \
        bash "$tmp/pe/nightly_train.sh" --pinned --from "$d" --out "$tmp/oe" --now 2026-10-04T06:00:00Z
    row a_pinned_run_without_pin_refuses 2 "NOT RELEASABLE: nightly-train, pin failed" "RELEASABLE H=" -- \
        bash "$tmp/pc/nightly_train.sh" --pinned --from "$d" --out "$tmp/oc" --now 2026-10-04T06:00:00Z
    row an_edited_pin_refuses 2 "NOT RELEASABLE: nightly-train, pin failed" "RELEASABLE H=" -- \
        bash "$tmp/pd/nightly_train.sh" --pinned --from "$d" --out "$tmp/od" --now 2026-10-04T06:00:00Z
    row sums_that_omit_pin_refuse 2 "NOT RELEASABLE: nightly-train, pin failed" "RELEASABLE H=" -- \
        bash "$tmp/pf/nightly_train.sh" --pinned --from "$d" --out "$tmp/of" --now 2026-10-04T06:00:00Z
    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + fail))"
    rm -rf -- "${tmp:?}"
    [ "$fail" -eq 0 ]
}
# the planted mutants: name, TAB, a sed expression applied to this file
NT_MUTANTS='m01_not_measured_counts_as_green	s/nv++; if (\$3 == "green") next/nv++; if ($3 != "red") next/
m02_run_on_any_commit	s/if (RH\[r\] != C) {/if (0) {/
m03_in_progress_run_counts	s/if (RST\[r\] != "COMPLETED") {/if (0) {/
m04_retried_green_is_green	s/if (att > 1) out/if (att > 9) out/
m05_failed_read_is_read	s/readok = (RD == "ok" \&\& /readok = (/
m06_unread_attempt_is_attempt_1	s/if (!(pick in AT) || AT\[pick\] !~ \/\^\[0-9\]+\$\/) {/if (0) {/
m07_no_trap	s/^    trap on_exit EXIT$/    :/
m08_info_lanes_vote	s/\$2 == "verdict" { nv++/{ nv++/
m09_ranked_by_fewest_hours	s/-k1,1gr/-k1,1g/
m10_greens_counts_total	s/match(\$0, \/streak=/match($0, \/total=/
m11_coverage_gates	s/^coverage;info;/coverage;verdict;/
m12_a_verdict_lane_dropped	/^milestone;verdict;/d
m13_rollup_dropped	s/select(.checkSuite.workflowRun != null)/select(false)/
m14_rerun_job_not_counted	s/(\[\$j | .\[\] | select(.name == \$k.name)\] | length)\]/1]/
m15_missing_producer_not_refused	s/if (\$hit | length) != (\$want | length) then/if false then/
m16_required_job_count_ignored	s/if (tot < NEED\[i\]) {/if (0) {/
m17_cut_job_list_is_read	s/if (JN\[r, k\] == "#truncated") {/if (0) {/
m18_void_run_is_picked	s/if (pick == "" \&\& s != "void") {/if (pick == "") {/
m19_edited_bundle_runs	s/sha256sum --quiet -c SHA256SUMS/true/
m20_producerless_lane_is_green	s/out(i, "not_measured", "", "", "no nightly producer on main yet")/out(i, "green", "", "", "x")/
m21_failure_row_after_history	s/\[ -z "\$PRINTED\$HISTDONE" \]/[ -z "$PRINTED" ]/
m22_self_test_inbox_inherited	s/^    export INBOX="\$tmp\/inbox.md"   # whatever/    : # whatever/
m23_drill_writes_named_inbox	s/then INBOXF="\$(mktemp)" || exit 3; DRILL=1;/then DRILL="";/
m24_one_leg_hides_another	/fb\[JN\[r, k\]\] = 1; bad = bad/d
m25_cut_rollup_read	s/if (\$c.statusCheckRollup.contexts.pageInfo.hasNextPage \/\/ false) then/if false then/
m26_day_set_late	/^    DAY=.*# before any step/d;s/^    step outdir$/    DAY="${NOW%%T*}"; step outdir/
m27_pinned_not_required	s/if \[ -n "\$PINNED" \] || \[ -f "\$HERE\/PIN" \]; then/if [ -f "$HERE\/PIN" ]; then/
m28_unhashed_pin_accepted	s/|| { PIN="unhashed"; exit 1; }/|| :/
m29_rate_floor_ignored	s/\[ "\$rem" -ge "\$RATE_FLOOR" \]/[ 1 ]/
m30_call_budget_ignored	s/call_ok() { \[ "\$CALLS" -lt "\$MAX_CALLS" \] || return 1;/call_ok() {/
m31_graphql_errors_read	s/(.errors \/\/ \[\]) | length > 0/false/
m32_inbox_mismatch_exits_0	s/inbox_line "\$line" "\$g" || exit 1/inbox_line "$line" "$g"/
m33_info_attempt_claimed	s/out(i, "green", pick, "1?", /out(i, "green", pick, "1", /
m34_token_guard_dropped	s/^    no_token || exit 1$/    :/
m35_credentials_file_ignored	s/\[ ! -e "\$ch\/credentials" \] \&\& \[ ! -e "\$ch\/credentials.toml" \]/true/
m36_named_registry_token_ignored	s/ || \$1 ~ \/\^CARGO_REGISTRIES_\[A-Za-z0-9_\]+_TOKEN\$\// /
m37_exit_verdict_ignored	s/^    \[ -z "\$EXIT_VERDICT" \] || verdict_rc "\$line" || exit 1$/    :/
m38_verdict_rc_always_red	s/"RELEASABLE "\*) return 0 ;;/"RELEASABLE "*) return 1 ;;/'
# each planted mutant must change the file, still parse, and turn at least one row RED
mutants() {
    local tmp pass=0 fail=0 name expr o rc
    tmp="$(mktemp -d)" || exit 3
    export INBOX="$tmp/inbox.md"   # the same for every mutant run
    cp "$HERE/red_age.sh" "$HERE/nightly_greens.sh" "$tmp/" 2>/dev/null
    # baseline: the unmutated copy must be green here, or every kill below proves nothing
    cp "$SCRIPT_PATH" "$tmp/nightly_train.sh"; chmod u+w "$tmp"/*.sh   # an installed bundle is 0555
    if ! bash "$tmp/nightly_train.sh" --self-test > /dev/null 2>&1; then printf '  BROKE %-44s the unmutated script is not green in the mutant dir\n' baseline; rm -rf -- "${tmp:?}"; return 1; fi
    while IFS='	' read -r name expr; do
        [ -n "$name" ] || continue
        sed -e "$expr" "$SCRIPT_PATH" > "$tmp/nightly_train.sh"
        if cmp -s "$SCRIPT_PATH" "$tmp/nightly_train.sh"; then printf '  BROKE %-44s changed nothing: its pattern no longer matches\n' "$name"; fail=$((fail + 1)); continue; fi
        if ! bash -n "$tmp/nightly_train.sh" 2>/dev/null; then printf '  BROKE %-44s does not parse\n' "$name"; fail=$((fail + 1)); continue; fi
        o="$(bash "$tmp/nightly_train.sh" --self-test 2>&1)"; rc=$?
        case "$rc:$o" in
            0:*) printf '  BROKE %-44s SURVIVED: the case table stayed green\n' "$name"; fail=$((fail + 1)) ;;
            *"  BROKE "*) printf '  ok    %-44s killed, %s row(s) broke\n' "$name" "$(printf '%s\n' "$o" | awk '/^  BROKE /{n++} END{print n+0}')"; pass=$((pass + 1)) ;;
            *) printf '  BROKE %-44s exit %s with no broken row: not a kill\n' "$name" "$rc"; fail=$((fail + 1)) ;;
        esac
    done < <(printf '%s\n' "$NT_MUTANTS")
    printf -- '--- %s/%s mutants killed ---\n' "$pass" "$((pass + fail))"
    rm -rf -- "${tmp:?}"
    [ "$fail" -eq 0 ]
}
# ---------------------------------------------------------------- main ----------------------------------------------
ARGS_SEEN="$*"
case "${1:-}" in
    --self-test) self_test; exit $? ;;
    --mutants) mutants; exit $? ;;
    --install) shift; install_timer "$@"; exit $? ;;
esac
FROM=""; NOW=""; OUTDIR="${OUT:-}"; INBOXF="${INBOX:-}"
# a planted fault is a drill: its line goes to a throwaway inbox, never the one the caller named
if [ -n "${NIGHTLY_TRAIN_FAULT:-}" ]; then INBOXF="$(mktemp)" || exit 3; DRILL=1; else DRILL=""; fi
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUTDIR="${2:-}"; shift 2 ;;
        --inbox) [ -n "$DRILL" ] || INBOXF="${2:-}"; shift 2 ;;
        --from) FROM="${2:-}"; shift 2 ;;
        --pinned) PINNED=1; shift ;;
        --exit-verdict) EXIT_VERDICT=1; shift ;;
        --now) NOW="${2:-}"; shift 2 ;;
        *) caller_error "unknown argument $1" ;;
    esac
done
[ -n "$OUTDIR" ] || caller_error "--out DIR (or OUT) is required"
[ -z "$FROM" ] || [ -d "$FROM" ] || caller_error "--from $FROM is not a directory"
[ -n "$NOW" ] || NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
[[ "$NOW" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] || caller_error "--now must be YYYY-MM-DDTHH:MM:SSZ"
run_train "$FROM"
