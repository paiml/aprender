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
#   scripts that judged it. H is the commit id of C. C is the head of main at the moment of the read, unless the head has
#   no measured run from some verdict producer: then C is the newest commit, at most MAX_LAG commits behind the head,
#   on which every verdict producer has a measured run, and the line adds " (judged <sha10>, head <sha10>, lag k
#   commits)". All lanes are judged on that one commit, never a per-lane mix. None within MAX_LAG = the head, which
#   reads not_measured for a lane with no run there; there is no other fallback (#4798).
#
# READ, DON'T RE-RUN. Night 1 runs nothing. Each lane takes the result its existing nightly producer (a workflow on
#   main and, optionally, a job-name pattern) recorded for C. A producer that chains from "Nightly pick" is read by
#   its workflow_run runs only (scripts/release/check_producers_chain.sh P6 keeps this table in step with the
#   triggers). Such a run is filed under main's head when the pick finished, while its tree is the night's pick
#   (nightly_c_checkout.sh), and the two differ when a merge lands while the pick runs or a pick is dispatched again
#   later (it keeps the older C). So the run counts only when the pick of its night is C. Its night is the night of
#   the newest Nightly pick run created at or before it, refs/heads/nightly/<N> for the window [N 12:00Z, N+1 12:00Z)
#   that holds that pick run's created_at: a pick rerun the next evening keeps its created_at and its producers build
#   the old night's pick, so the run's own creation time names the wrong night. An older pick run updated after the
#   newest one (a rerun that may be what chained the run), an unread pick or an unread pick-run list is not_measured, and
#   so is a full pick-run list (PICK_RUNS) whose oldest run is less than 30 days, the rerun limit, before the run.
#   A run created 12:00Z-17:59Z also needs the night before to be C (the scheduled pick fires at 20:30Z). A lane is
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
#   min(RATE_FLOOR, limit/5) makes no further call: the read failed, every producer lane is not_measured.
#
# EXIT  0 a verdict line was printed (RELEASABLE or NOT) · 2 the train failed part-way (trap line) · 3 caller error
#   With --exit-verdict, NOT RELEASABLE exits 1 (a CI job is green only on a releasable head).
# NO TOKEN. The train refuses to start (trap line "no-token") when a registry token is reachable: CARGO_REGISTRY_TOKEN
#   or CARGO_REGISTRIES_*_TOKEN set, or credentials(.toml) or a config token line under $CARGO_HOME or ~/.cargo. The
#   installed unit runs in a user namespace (PrivateUsers=) with an empty tmpfs over ~/.cargo and an empty CARGO_HOME;
#   install proves inside that sandbox that no token is reachable (--probe-no-token), or installs nothing.
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
MAX_LAG=6   # J (the commit judged) is at most this many commits behind the head of main
PICK_RUNS=100   # Nightly pick runs listed; a full list reaching back less than 30 days (the rerun limit) is not_measured
RATE_FLOOR=1000
UNIT="aprender-nightly-train"
# lane;kind;release-day check;producer workflow;event pattern;job-name pattern[;required job names, default 1]
# ('-' producer: none yet). A run with fewer distinct matched job names than required measures nothing (void).
LANES='ci-main;verdict;ci / gate + workspace-test (merge, autopilot wait);.github/workflows/ci.yml;^push$;^(ci / gate|workspace-test)$;2
deep-doctests;verdict;autopilot deep: cargo test --doc --workspace;-;-;-
deep-nodefault;verdict;autopilot deep: cargo check --workspace --no-default-features;-;-;-
deep-examples;verdict;autopilot deep: cargo build --workspace --examples;.github/workflows/examples-nightly.yml;^workflow_run$;^examples$
deep-bins-build;verdict;autopilot deep: all-bins cargo build --locked --release;.github/workflows/nightly.yml;^workflow_run$;-unknown-linux-gnu on 
deep-bins-smoke;verdict;autopilot deep: nightly_manifest.py smoke;-;-;-
dogfood;verdict;dogfood.sh --phase pre-publish + preflight R5;-;-;-
models;verdict;models_t1.sh GPU-host ladder legs, preflight R7;.github/workflows/models-nightly.yml;^schedule$;^models$
readiness;verdict;release_readiness.sh, preflight R8;-;-;-
milestone;verdict;check_milestone_cut.sh --must-carry;-;-;-
cleanroom-cpu;verdict;clean-room (aprender) on the tag;-;-;-
cleanroom-gpu;verdict;b2-gpu.yml on the tag;-;-;-
assets;verdict;binary-release.yml + check_release_assets.sh;-;-;-
preflight;verdict;check_publish_preflight.sh R1-R8;-;-;-
publish-dryrun;verdict;rc_publish_gate.sh --verify + cascade-publish.sh --check;-;-;-
coverage;info;tag_coverage_gate.sh (C291.1: not gating);.github/workflows/coverage-nightly.yml;^workflow_run$;^coverage$
guards;info;guards-nightly;.github/workflows/guards-nightly.yml;^workflow_run$;
mutants;info;mutants-nightly;.github/workflows/mutants-nightly.yml;^workflow_run$;
toolchain-ceiling;info;toolchain-ceiling;.github/workflows/toolchain-ceiling.yml;^workflow_run$;
cuda;info;cuda-nightly;.github/workflows/cuda-nightly.yml;^workflow_run$;
silicon;info;silicon-nightly;.github/workflows/silicon-nightly.yml;^workflow_run$;
qwen-story;info;qwen-story-daily;.github/workflows/qwen-story-daily.yml;^workflow_run$;
qwen-hunt;info;qwen-hunt-nightly;.github/workflows/qwen-hunt-nightly.yml;^workflow_run$;
beat-speed;info;beat-speed-nightly;.github/workflows/beat-speed-nightly.yml;^workflow_run$;
conleche;info;conleche-nightly;.github/workflows/conleche-nightly.yml;^workflow_run$;
bench;info;nightly-bench;.github/workflows/nightly-bench.yml;^workflow_run$;
book;info;book;.github/workflows/book.yml;^workflow_run$;
book-contracts;info;book-contracts;.github/workflows/book-contracts.yml;^workflow_run$;
install-script;info;install-script;.github/workflows/install-script.yml;^workflow_run$;
fleet-toolset;info;fleet-toolset;.github/workflows/fleet-toolset.yml;^schedule$;'
caller_error() { printf 'NOT RELEASABLE: nightly-train, caller error: %s\n' "$*"; exit 3; }
next_day() {   # next_day YYYY-MM-DD -> the day after, by arithmetic: no date(1), so no clock in reach (bashrs DET002)
    local y m d dim months=12   # named, so the month test is not a literal count (derive_identity R3)
    case "$1" in [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;; *) return 1 ;; esac
    y=$((10#${1:0:4})); m=$((10#${1:5:2})); d=$((10#${1:8:2}))
    case "$m" in
        1|3|5|7|8|10|12) dim=31 ;;
        4|6|9|11) dim=30 ;;
        2) dim=28; { [ $((y % 4)) -eq 0 ] && [ $((y % 100)) -ne 0 ]; } || [ $((y % 400)) -eq 0 ] && dim=29 ;;
        *) return 1 ;;
    esac
    [ "$d" -ge 1 ] && [ "$d" -le "$dim" ] || return 1
    d=$((d + 1)); [ "$d" -le "$dim" ] || { d=1; m=$((m + 1)); }
    [ "$m" -le "$months" ] || { m=1; y=$((y + 1)); }
    printf '%04d-%02d-%02d\n' "$y" "$m" "$d"
}
# ---------------------------------------------------------------- judgement (pure: files in, files out) ----------
# evaluate LANESFILE RAW MODE -> MODE=cand: run ids of green verdict candidates; MODE=final: lanes.tsv rows and
#   RAW/hist_redage.tsv. RAW holds C, read, runs.tsv, attempts.tsv.
evaluate() {
    local c rd
    c="$(cat "$2/C" 2>/dev/null)"; rd="$(cat "$2/read" 2>/dev/null)"
    [ -f "$2/attempts.tsv" ] || : > "$2/attempts.tsv"
    [ -f "$2/runs.tsv" ] || : > "$2/runs.tsv"
    [ -f "$2/picks.tsv" ] || : > "$2/picks.tsv"
    # a pick read that failed judges every lane not_measured (never the c310 fallback for an absent pick, #4798)
    [ -s "$2/pickfail" ] && [ "${rd:-}" = ok ] && rd="failed: pick read: $(head -n 1 "$2/pickfail")"
    # each picked night as the window of run creation times it owns, [N 12:00Z, N+1 12:00Z), and its C
    local n o e
    while IFS=$'\t' read -r n o; do
        case "$n" in [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;; *) continue ;; esac
        e="$(next_day "$n")" || continue
        printf '%sT12:00:00Z\t%sT12:00:00Z\t%s\n' "$n" "$e" "$o"
    done < "$2/picks.tsv" > "$2/pickwin.tsv"
    [ -f "$2/rank.tsv" ] || : > "$2/rank.tsv"
    awk -F '\t' -v OFS='\t' -v C="$c" -v PRF="$2/pickruns.tsv" -v PRN="$PICK_RUNS" -v RD="${rd:-failed: no read status}" -v MODE="$3" -v HIST="$2/hist_redage.tsv" -v JOUT="$2/judged" -v MAXLAG="$MAX_LAG" '
    FILENAME == ARGV[1] { split($0, a, ";"); n++; L[n] = a[1]; K[n] = a[2]; CK[n] = a[3]; WF[n] = a[4]; EV[n] = a[5]; RE[n] = a[6]; NEED[n] = (a[7] ~ /^[0-9]+$/ ? a[7] + 0 : 1); next }
    FILENAME == ARGV[2] { AT[$1] = $2; next }
    FILENAME == ARGV[3] { np++; PS[np] = $1; PE[np] = $2; PO[np] = $3; next }
    FILENAME == ARGV[4] { RKO[$1 + 0] = $2; NRK++; next }
    FILENAME == PRF { npr++; PRC[npr] = "" $1; PRU[npr] = "" $2; next }
    {
        key = $3 SUBSEP $10 SUBSEP $13
        if (key in seen) next
        seen[key] = 1; r = $3
        if (!(r in RW)) { RW[r] = $2; REV[r] = $4; RBR[r] = $5; RH[r] = $6; RC[r] = $7; RST[r] = toupper($8); RCO[r] = toupper($9); nr++; RID[nr] = r }
        if ($10 != "") { nj[r]++; k = nj[r]; JN[r, k] = $10; JS[r, k] = toupper($11); JC[r, k] = toupper($12); JB[r, k] = $13; JE[r, k] = $14; JD[r, k] = $15 }
    }
    # nightk(t) -> the picked night whose window holds creation time t (0: no pick read for it); nightc(t) -> its C;
    #   prevc(t) -> the C of the night before it
    function nightk(t,    k) { for (k = 1; k <= np; k++) if (t >= PS[k] && t < PE[k]) return k; return 0 }
    function nightc(t,    k) { k = nightk(t); return (k ? PO[k] : "") }
    function prevc(t,    k, j) { k = nightk(t); if (!k) return ""; for (j = 1; j <= np; j++) if (PE[j] == PS[k]) return PO[j]; return "" }
    # chainc(t) -> the C a chained run created at t measured: the night of the newest Nightly pick run created at or before t
    #   (a producer reads the created_at of its pick run, which a rerun keeps, so a rerun the next evening is the old night);
    #   "" with CWHY when no pick run is read at or before t, or an older pick run was updated after that one was created
    #   (a rerun of it may have fired these producers: which pick they measured is unknown, so it is not_measured), or the
    #   list is full (PRN runs) and its oldest run is less than 30 days before t: a run older than the list may be rerun
    #   for 30 days, and that rerun would be unseen
    function dz(y, m, d,    era, yoe, doy, doe, mp) {
        y -= (m <= 2); era = int(y / 400); yoe = y - era * 400; mp = (m + 9) % 12
        doy = int((153 * mp + 2) / 5) + d - 1; doe = yoe * 365 + int(yoe / 4) - int(yoe / 100) + doy
        return era * 146097 + doe - 719468
    }
    function sz(s) { return dz(substr(s, 1, 4) + 0, substr(s, 6, 2) + 0, substr(s, 9, 2) + 0) * 86400 + substr(s, 12, 2) * 3600 + substr(s, 15, 2) * 60 + substr(s, 18, 2) }
    function chainc(t,    k, b, o) {
        b = ""; CWHY = "no pick for that night"
        for (k = 1; k <= npr; k++) if (PRC[k] <= t && b < PRC[k]) b = "" PRC[k]
        if (b == "") { CWHY = "no pick run seen at or before it"; return "" }
        o = b; for (k = 1; k <= npr; k++) if (PRC[k] < o) o = "" PRC[k]
        if (npr >= PRN && sz(o) > sz(t) - 30 * 86400) { CWHY = "the " npr " pick runs listed reach back only to " o; return "" }
        for (k = 1; k <= npr; k++) if (b > PRC[k] && b < PRU[k]) { CWHY = "the pick run of " PRC[k] " was rerun after " b; return "" }
        return nightc(b)
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
        # J, the commit judged (#4798): the newest commit on main, at or before the read, on which EVERY verdict lane with
        #   a producer has a measured (green or red) run; C itself when it has them. One commit for all lanes, never a
        #   per-lane mix. The candidates are rank.tsv: the head of main (rank 0) and its history, newest first, read in the
        #   same query; LAG is the rank of J, the commits between J and the head. Only ranks 0..MAXLAG are candidates. None
        #   qualifies (or no rank list): J = C and every lane reads as it would on C, which is not_measured for a lane
        #   with no run there. Nothing falls back past MAXLAG.
        J = C; LAG = 0
        if (readok && RKO[0] == C) {
            split("", HAS); nprod = 0
            for (i = 1; i <= n; i++) {
                if (WF[i] == "-" || K[i] != "verdict") continue
                nprod++
                for (q = 1; q <= nr; q++) {
                    r = RID[q]
                    if (RW[r] == WF[i] && REV[r] ~ EV[i] && RBR[r] == "main") {
                        # a chained run stands for the pick of its night, a plain run for its own commit (see the lane loop below)
                        if (EV[i] == "^workflow_run$") { hm = chainc(RC[r]); if (substr(RC[r], 12, 2) >= "12" && substr(RC[r], 12, 2) < "18" && prevc(RC[r]) != hm) hm = "" }
                        else hm = RH[r]
                        s = lst(i, r); if (hm != "" && (s == "green" || s == "red")) HAS[hm, i] = 1 }
                }
            }
            for (k = 0; k <= MAXLAG && k < NRK && nprod > 0; k++) {
                h = RKO[k]; all = 1
                for (i = 1; i <= n && all; i++) if (WF[i] != "-" && K[i] == "verdict" && !((h, i) in HAS)) all = 0
                if (all) { J = h; LAG = k; break }
            }
        }
        if (MODE == "final") print J, C, LAG > JOUT
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
                # a plain run stands for the commit it ran on. A chained run is filed under the head of main when the pick finished,
                #   but its tree is the pick of its night (nightly_c_checkout.sh), so it stands for that pick and for no other commit:
                #   it counts only when that pick is J (#4798; a re-dispatched pick keeps an older C and still fires with the newer head)
                if (EV[i] != "^workflow_run$") { if (RH[r] != J) { if (other == "") other = r " on " substr(RH[r], 1, 10); continue } }
                else if ((pc = chainc(RC[r])) != J) { if (other == "") other = r " measured the pick " (pc == "" ? CWHY : substr(pc, 1, 10)); continue }
                # a run created 12:00Z-17:59Z may come from a pick that started before noon: the night before must be J too
                if (EV[i] == "^workflow_run$" && substr(RC[r], 12, 2) >= "12" && substr(RC[r], 12, 2) < "18" && (pc = prevc(RC[r])) != J) { if (other == "") other = r " may have measured the pick " (pc == "" ? "no pick for that night" : substr(pc, 1, 10)); continue }
                if (pick == "" && s != "void") { pick = r; pst = s; pwhy = WHY; pS = ST; pE = EN; pD = DUP }
                else if (pick == "" && pwhy == "") pwhy = WHY
            }
            if (pick == "") {
                if (MODE == "final") out(i, "not_measured", "", "", (pwhy != "" ? pwhy : "no run on the judged commit") (other != "" ? "; newest run " other : (m == 0 ? "; no run on main" : "")))
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
    }' <(printf '%s\n' "$1") "$2/attempts.tsv" "$2/pickwin.tsv" "$2/rank.tsv" "$2/pickruns.tsv" "$2/runs.tsv"
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
    local raw="$2" c j lag
    c="$(cat "$raw/C" 2>/dev/null)"
    evaluate "$1" "$raw" final > "$raw/lanes.tsv" || return 1
    # the judged commit J and its lag (#4798); the line names main's head and the lag whenever J is not the head
    read -r j _ lag < "$raw/judged" 2>/dev/null; [[ "$j" =~ ^[0-9a-f]{40}$ && "$lag" =~ ^[0-9]+$ ]] || { j="$c"; lag=0; }
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
    awk -F '\t' -v C="$j" -v HEAD="$c" -v LAG="${lag:-0}" 'FILENAME == ARGV[1] { H[$1] = $2; next }
        $2 == "verdict" { nv++; if ($3 == "green") next
            bad++; h = ($1 in H) ? H[$1] + 0 : -1; s = ($3 == "red" ? 1 : 0)
            if (best == "" || s > bs || (s == bs && h > bh)) { best = $1; brun = ($3 == "red" ? $5 : "not_measured"); bs = s; bh = h } }
        END {
            note = (C != HEAD && HEAD != "" ? " (judged " substr(C, 1, 10) ", head " substr(HEAD, 1, 10) ", lag " LAG " commits)" : "")
            if (nv == 0) { print "NOT RELEASABLE: nightly-train, the lane table has no verdict lane"; exit }
            if (bad == 0 && C ~ /^[0-9a-f]{40}$/) { print "RELEASABLE H=" C note; exit }
            if (bad == 0) { print "NOT RELEASABLE: nightly-train, no commit read"; exit }
            printf "NOT RELEASABLE: %s, %s%s%s\n", best, brun, (bad > 1 ? " (+" bad - 1 " more)" : ""), note
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
    printf '%s\n' "$1" | awk -F ';' '$4 != "-" { print $4 }' | sort -u | jq -R -s --slurpfile wf "$2" -r --arg repo "$REPO" --argjson hn "$((MAX_LAG + 1))" --argjson pn "$PICK_RUNS" '
        split("\n") | map(select(length > 0)) as $want
        | ($wf | first | .workflows | map(select(.path as $x | $want | index($x))) ) as $hit
        | if ($hit | length) != ($want | length) then error("producer workflow missing from the list") else . end
        | ($wf | first | .workflows | map(select(.path == ".github/workflows/nightly-pick.yml")) | first) as $pk
        | if $pk == null then error("the Nightly pick workflow is missing from the list") else . end
        | ($repo | split("/")) as $own
        | "query { repository(owner: \"\($own | first)\", name: \"\($own | last)\") { defaultBranchRef { name target { ... on Commit { oid tree { oid } statusCheckRollup { contexts(first: 100) { pageInfo { hasNextPage } nodes { ... on CheckRun { name status conclusion startedAt completedAt checkSuite { status conclusion branch { name } workflowRun { databaseId event createdAt workflow { id } } } } } } } history(first: \($hn)) { nodes { oid statusCheckRollup { contexts(first: 100) { pageInfo { hasNextPage } nodes { ... on CheckRun { name status conclusion startedAt completedAt checkSuite { status conclusion branch { name } workflowRun { databaseId event createdAt workflow { id } } } } } } } } } } } } nightly: refs(refPrefix: \"refs/heads/nightly/\", first: 40, orderBy: {field: ALPHABETICAL, direction: DESC}) { nodes { name target { oid } } } } "
          + ([$hit | to_entries[] | "w\(.key): node(id: \"\(.value.node_id)\") { ... on Workflow { id runs(first: 12) { nodes { databaseId createdAt event checkSuite { status conclusion branch { name } commit { oid } checkRuns(first: 100, filterBy: {checkType: ALL}) { pageInfo { hasNextPage } nodes { name status conclusion startedAt completedAt } } } } } } }"] | join(" "))
          + " pk: node(id: \"\($pk.node_id)\") { ... on Workflow { runs(first: \($pn)) { nodes { createdAt updatedAt } } } } }"' 2>/dev/null
}
# picks_read GQLJSON PICKS PICKFAIL -> PICKS (night<TAB>C), and PICKFAIL empty when the refs were read. An absent ref is a row
#   missing from a good read (PICKFAIL stays empty: the chained lane falls to c310's "no pick for that night"); a response whose
#   refs are not an array is a failed read (PICKFAIL holds the reason: every lane is not_measured, no fallback) (#4798)
picks_read() {
    : > "$3"
    if jq -e '(.data.repository.nightly.nodes | type) == "array"' "$1" > /dev/null 2>&1 && picks_of "$1" > "$2" 2>/dev/null; then :
    else : > "$2"; printf 'the nightly pick refs did not parse\n' > "$3"; fi
}
# pickruns_read GQLJSON PICKRUNS PICKFAIL -> PICKRUNS (created_at<TAB>updated_at of each Nightly pick run read). A chained run
#   is the night of the newest pick run created at or before it; a run list that is not an array is a failed read, written
#   to PICKFAIL: every lane is not_measured, as for the refs
pickruns_read() {
    jq -e '(.data.pk.runs.nodes | type) == "array"' "$1" > /dev/null 2>&1 && jq -r '.data.pk.runs.nodes[] | select((.createdAt | type) == "string" and (.updatedAt | type) == "string") | [.createdAt, .updatedAt] | @tsv' "$1" > "$2" 2>/dev/null && return 0
    printf '' > "$2"
    printf 'the Nightly pick runs did not parse\n' > "$3"
}
# picks_of GQLJSON -> night<TAB>C for each refs/heads/nightly/<YYYY-MM-DD> the query returned; other names are dropped
picks_of() { jq -r '(.data.repository.nightly.nodes // [])[] | select(.name | test("^[0-9]{4}-[0-9]{2}-[0-9]{2}$")) | [.name, .target.oid] | @tsv' "$1"; }
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
        # the head and the newer history of main (#4798): a push run on an older main commit can fall out of the 12 newest runs of its workflow
        ( [$c] + [($c.history.nodes // [])[] | select(.oid != $c.oid)] | .[] | . as $x
          | ( ($x.statusCheckRollup.contexts.nodes // [])[] | select(.checkSuite.workflowRun != null)
              | .checkSuite.workflowRun.workflow.id as $wid | ["rollup", ($p | .[$wid]), .checkSuite.workflowRun.databaseId, .checkSuite.workflowRun.event,
                 (.checkSuite.branch.name // ""), $x.oid, .checkSuite.workflowRun.createdAt, .checkSuite.status, (.checkSuite.conclusion // ""),
                 .name, .status, (.conclusion // ""), (.startedAt // ""), (.completedAt // ""), 1] ),
            ( if ($x.statusCheckRollup.contexts.pageInfo.hasNextPage // false) then
                [($x.statusCheckRollup.contexts.nodes // [])[] | select(.checkSuite.workflowRun != null)] | unique_by(.checkSuite.workflowRun.databaseId) | .[]
                | .checkSuite.workflowRun.workflow.id as $wid | ["rollup", ($p | .[$wid]), .checkSuite.workflowRun.databaseId, .checkSuite.workflowRun.event,
                   (.checkSuite.branch.name // ""), $x.oid, .checkSuite.workflowRun.createdAt, .checkSuite.status, (.checkSuite.conclusion // ""),
                   "#truncated", "COMPLETED", "TRUNCATED", "", "", 1]
              else empty end ) )
      | map(tostring) | join("\t")' "$1"
}
# rank_rows GQL -> "rank<TAB>oid", the head of main (rank 0) and then its history, newest first, the head once (#4798)
rank_rows() {
    jq -r '.data.repository.defaultBranchRef.target as $c | ([$c.oid] + [($c.history.nodes // [])[] | .oid | select(. != $c.oid)]) | to_entries[] | "\(.key)\t\(.value)"' "$1"
}
# fetch RAW CACHE -> RAW/{C,tree,read,runs.tsv,attempts.tsv,graphql.json,workflows.json}; never fails: a failed read
#   is recorded in RAW/read and makes every producer lane not_measured
fetch() {
    local raw="$1" cache="$2" lim rem fl hdr st q a ids r
    printf 'failed: not read\n' > "$raw/read"; : > "$raw/C"; : > "$raw/tree"; : > "$raw/runs.tsv"; : > "$raw/attempts.tsv"; : > "$raw/picks.tsv"; : > "$raw/rank.tsv"; : > "$raw/pickfail"; : > "$raw/pickruns.tsv"
    read -r lim rem <<< "$(gh api rate_limit --jq '"\(.resources.core.limit) \(.resources.core.remaining)"' 2>/dev/null)"
    case "$lim:$rem" in *[!0-9:]*|:*|*:) printf 'failed: rate_limit unreadable\n' > "$raw/read"; return 0 ;; esac
    # the floor is a fifth of the token's own hourly limit, capped at RATE_FLOOR: the fleet PAT (5000) keeps its 1000
    #   reserve, a job's github.token (1000/h, an allowance of its own) refuses under 200
    fl=$((lim / 5)); [ "$fl" -le "$RATE_FLOOR" ] || fl="$RATE_FLOOR"
    [ "$rem" -ge "$fl" ] || { printf 'failed: core remaining %s under %s\n' "$rem" "$fl" > "$raw/read"; return 0; }
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
    # the night's picks (refs/heads/nightly/<night>): a chained lane counts a run only when its night picked C
    #   "no pick for that night" (the ref is absent: a row missing from a good read) is not "pick read failed" (the refs came
    #   back unusable): the second is written to pickfail and judges every lane not_measured, it never falls back to the first
    picks_read "$raw/graphql.json" "$raw/picks.tsv" "$raw/pickfail"
    pickruns_read "$raw/graphql.json" "$raw/pickruns.tsv" "$raw/pickfail"
    grep -q -E '^[0-9a-f]{40}$' "$raw/C" || { printf 'failed: no head of main in the response\n' > "$raw/read"; return 0; }
    normalize "$raw/graphql.json" "$raw/workflows.json" > "$raw/runs.tsv" 2>/dev/null || { printf 'failed: response did not normalize\n' > "$raw/read"; return 0; }
    rank_rows "$raw/graphql.json" > "$raw/rank.tsv" 2>/dev/null || : > "$raw/rank.tsv"
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
#   environment (set at all, even empty), no credentials file and no config token line under $CARGO_HOME or ~/.cargo. The train
#   runs no cargo, but the rule is structural: it must be unable to upload, so it refuses to start where a token
#   exists. A flag alone is not a guard.
no_token() {
    local d
    if ! env | awk -F '=' '$1 == "CARGO_REGISTRY_TOKEN" || $1 ~ /^CARGO_REGISTRIES_[A-Za-z0-9_]+_TOKEN$/ { f = 1 } END { exit f }'; then
        echo "no-token: a registry token variable is set" >&2; return 1
    fi
    # CARGO_HOME and the uid's own ~/.cargo: pointing CARGO_HOME elsewhere does not make a token there unreachable
    for d in "${CARGO_HOME:-$HOME/.cargo}" "$HOME/.cargo"; do
        if [ -e "$d/credentials" ] || [ -e "$d/credentials.toml" ]; then
            echo "no-token: a credentials file under $d is reachable (the unit hides ~/.cargo; a manager that cannot, refuses)" >&2; return 1
        fi
        # a legacy token = line in the config reaches cargo as well
        if grep -qsE '^[[:space:]]*token[[:space:]]*=' "$d/config" "$d/config.toml"; then
            echo "no-token: a token line in the config under $d is reachable" >&2; return 1
        fi
    done
}
# verdict_rc LINE -> 0 for RELEASABLE, 1 for anything else (--exit-verdict: a green job means a releasable judged commit J, named with the head and the lag when J is not the head)
verdict_rc() { case "$1" in "RELEASABLE "*) return 0 ;; *) return 1 ;; esac; }
inbox_line() {   # inbox_line LINE GREENS -> append <= 300 bytes and read it back
    local l who="${NIGHTLY_TRAIN_REPORTER:-nightly-r4}"   # the reporter is a role, named by the caller, never written here
    [ -n "$INBOXF" ] || return 0
    step inbox
    case "$who" in *[!A-Za-z0-9._-]*) printf 'inbox: NIGHTLY_TRAIN_REPORTER %s is not [A-Za-z0-9._-]\n' "$who" >&2; return 1 ;; esac
    l="${NOW:11:5}Z | $who | R4 | $1 | greens in a row: $2 | C=${CSHA:0:10}"   # the clock read once, at entry
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
    local from="$1" raw line g concl budget pin_t pin_g pin_r hrow j HSHA
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
        for f in C read runs.tsv attempts.tsv tree picks.tsv rank.tsv pickfail pickruns.tsv; do [ "$from/$f" -ef "$raw/$f" ] || cp "$from/$f" "$raw/$f" 2>/dev/null || : > "$raw/$f"; done
    else fetch "$raw" "$OUTDIR/cache"; fi
    CSHA="$(cat "$raw/C" 2>/dev/null)"
    step judge
    decide "$LANES" "$raw" "$NOW" || exit 1
    # from here C is the commit judged (#4798); HSHA keeps main's head for the bundle
    HSHA="$CSHA"; read -r j _ _ < "$raw/judged" 2>/dev/null; [[ "${j:-}" =~ ^[0-9a-f]{40}$ ]] && CSHA="$j"
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
        printf '# head\t%s\n' "$HSHA"
        printf '# C\t%s\n# tree\t%s\n# read\t%s\n# command\t%s\n# pin\ttrain=%s greens=%s redage=%s\n' "$CSHA" "$(if [ "$CSHA" = "$HSHA" ]; then cat "$raw/tree"; else printf 'not read (the read returns the tree of the head only)'; fi)" "$(cat "$raw/read")" "$PROG ${ARGS_SEEN:-}" "${pin_t:-unpinned}" "${pin_g:-}" "${pin_r:-}"
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
span() {   # span START END -> XhYYm, or not_measured; parsed by awk (step_table.sh secs()), no clock read
    awk -v a="$1" -v b="$2" '
        function days(y, m, d,    era, yoe, doy, doe, mp) {
            y -= (m <= 2); era = int(y / 400); yoe = y - era * 400; mp = (m + 9) % 12
            doy = int((153 * mp + 2) / 5) + d - 1; doe = yoe * 365 + int(yoe / 4) - int(yoe / 100) + doy
            return era * 146097 + doe - 719468
        }
        function secs(s) { return days(substr(s, 1, 4) + 0, substr(s, 6, 2) + 0, substr(s, 9, 2) + 0) * 86400 + substr(s, 12, 2) * 3600 + substr(s, 15, 2) * 60 + substr(s, 18, 2) }
        function ok(s) { return s ~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z$/ }
        BEGIN {
            if (!ok(a) || !ok(b)) { printf "not_measured"; exit }
            d = secs(b) - secs(a)
            printf "%dh%02dm", int(d / 3600), int((d % 3600) / 60)
        }'
}
# ---------------------------------------------------------------- install (the pinned bundle and the user timer) ----
install_timer() {
    local home="" out="" inbox="" t="" g="" r="" dir unitdir p o v sb pr
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
    dir="$(realpath -m -s -- "$home/b-${t:0:10}-${g:0:10}-${r:0:10}")" || caller_error "--home: cannot canonicalise"
    out="$(realpath -m -s -- "$out")" || caller_error "--out: cannot canonicalise"
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
    # the unit's sandbox, one list for the unit file and the probe: a user namespace (a --user manager cannot mount
    #   without one), an empty read-only tmpfs over ~/.cargo, and an empty CARGO_HOME of its own
    sb=(PrivateUsers=yes "TemporaryFileSystem=$HOME/.cargo:ro" "Environment=CARGO_HOME=$dir/cargo")
    pr=(); for v in "${sb[@]}"; do pr+=(-p "$v"); done
    # the planted row that runs inside that sandbox: it must find no token, or nothing is installed
    o="$(systemd-run --user --wait --pipe --quiet --collect "${pr[@]}" /bin/bash "$dir/nightly_train.sh" --probe-no-token 2>&1)" \
        || caller_error "the unit sandbox does not hide every registry token ($o); nothing installed"
    printf 'sandbox probe: %s; without the sandbox: %s\n' "$o" "$(bash "$dir/nightly_train.sh" --probe-no-token 2>&1)"
    unitdir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
    mkdir -p "$unitdir" || caller_error "cannot create $unitdir"
    {
        printf '[Unit]\nDescription=aprender nightly evidence train (BLD-002 R4), report-only\n\n[Service]\nType=oneshot\nNice=10\nTimeoutStartSec=900\n'
        printf '%s\n' "${sb[@]}"
        printf 'Environment=OUT=%s\nEnvironment=INBOX=%s\nEnvironment=PATH=%s\nExecStart=/bin/bash %s/nightly_train.sh --pinned\n' "$out" "$inbox" "$p" "$dir"
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
ST_Y="$(printf 'd%.0s' $(seq 40))"
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
    printf '2026-10-03\t%s\n2026-10-04\t%s\n' "$ST_C" "$ST_X" > "$1/picks.tsv"   # run 201 (02:00Z on the 4th) is night 2026-10-03
    # the Nightly pick runs, created_at and updated_at: one at noon each day, so a run belongs to the night whose window holds it
    printf '2026-10-03T12:00:00Z\t2026-10-03T12:05:00Z\n2026-10-04T12:00:00Z\t2026-10-04T12:05:00Z\n' > "$1/pickruns.tsv"
    printf '0\t%s\n' "$ST_C" > "$1/rank.tsv"
}
# ranks DIR SHA... -> DIR/rank.tsv: main's head and history, newest first (rank 0 = the head, aaaa...)
ranks() { local d="$1" k=0 s; shift; : > "$d/rank.tsv"; for s in "$@"; do printf '%s\t%s\n' "$k" "$s" >> "$d/rank.tsv"; k=$((k + 1)); done; }
# st_decide DIR -> the line, the lanes and the reds, as one text
st_decide() { decide "$ST_LANES" "$1" 2026-10-04T06:00:00Z; cat "$1/line" "$1/lanes.tsv" "$1/reds.tsv"; }
self_test() {
    local tmp pass=0 fail=0 d o rc v
    tmp="$(mktemp -d)" || exit 3
    export INBOX="$tmp/inbox.md"   # whatever INBOX the caller set: a self-test never writes a real inbox
    # and no registry token: the rows run the train, which refuses to start where one is reachable
    export CARGO_HOME="$tmp/cargo" HOME="$tmp/home"; mkdir -p "$CARGO_HOME" "$HOME"; unset CARGO_REGISTRY_TOKEN
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
    st_chain() { decide "$(printf '%s\n' "$ST_LANES" | sed -e '/^v-b;/s/\^schedule\$/^workflow_run$/')" "$1" 2026-10-04T06:00:00Z; cat "$1/line"; }
    d="$tmp/chain"; fixture "$d"; sed -i '/\t201\t/s/\tschedule\t/\tworkflow_run\t/' "$d/runs.tsv"
    row a_chained_lane_reads_its_workflow_run_run 0 "RELEASABLE H=$ST_C" "NOT RELEASABLE" -- st_chain "$d"
    d="$tmp/chainsched"; fixture "$d"
    row a_chained_lane_ignores_a_schedule_run 0 "NOT RELEASABLE: v-b, not_measured" "RELEASABLE H=" -- st_chain "$d"
    st_chainl() { st_chain "$1"; cat "$1/lanes.tsv"; }
    d="$tmp/repick"; fixture "$d"; sed -i '/\t201\t/s/\tschedule\t/\tworkflow_run\t/' "$d/runs.tsv"; sed -i "1s/$ST_C/$ST_X/" "$d/picks.tsv"
    row a_repicked_night_never_credits_main_head 0 "201 measured the pick bbbbbbbbbb" "RELEASABLE H=" -- st_chainl "$d"
    d="$tmp/nopick"; fixture "$d"; sed -i '/\t201\t/s/\tschedule\t/\tworkflow_run\t/' "$d/runs.tsv"; : > "$d/picks.tsv"
    row a_chained_run_without_a_read_pick_is_not_measured 0 "201 measured the pick no pick for that night" "RELEASABLE H=" -- st_chainl "$d"
    d="$tmp/pickfail"; fixture "$d"; printf 'the nightly pick refs did not parse\n' > "$d/pickfail"
    row a_failed_pick_read_is_not_measured_never_a_fallback 0 "failed: pick read: the nightly pick refs did not parse" "RELEASABLE H=" -- st_decide "$d"
    d="$tmp/ownnight"; fixture "$d"; sed -i '/\t201\t/s/\tschedule\t/\tworkflow_run\t/; s/\t2026-10-04T02:00:00Z\t/\t2026-10-04T13:00:00Z\t/' "$d/runs.tsv"
    row a_chained_run_is_judged_by_its_own_night 0 "201 measured the pick bbbbbbbbbb" "RELEASABLE H=" -- st_chainl "$d"
    d="$tmp/atnoon"; fixture "$d"; sed -i '/\t201\t/s/\tschedule\t/\tworkflow_run\t/; s/\t2026-10-04T02:00:00Z\t/\t2026-10-04T12:00:00Z\t/' "$d/runs.tsv"
    row a_run_at_noon_belongs_to_the_new_night 0 "201 measured the pick bbbbbbbbbb" "RELEASABLE H=" -- st_chainl "$d"
    d="$tmp/opennight"; fixture "$d"; sed -i '/\t201\t/s/\tschedule\t/\tworkflow_run\t/; s/\t2026-10-04T02:00:00Z\t/\t2026-10-03T12:00:00Z\t/' "$d/runs.tsv"; printf '2026-10-02\t%s\n' "$ST_C" >> "$d/picks.tsv"
    row a_run_at_noon_opens_its_night 0 "RELEASABLE H=$ST_C" "NOT RELEASABLE" -- st_chain "$d"
    d="$tmp/straddle"; fixture "$d"; sed -i '/\t201\t/s/\tschedule\t/\tworkflow_run\t/; s/\t2026-10-04T02:00:00Z\t/\t2026-10-04T17:59:00Z\t/' "$d/runs.tsv"
    printf '2026-10-03\t%s\n2026-10-04\t%s\n' "$ST_X" "$ST_C" > "$d/picks.tsv"
    row a_run_after_noon_needs_the_night_before_too 0 "201 may have measured the pick bbbbbbbbbb" "RELEASABLE H=" -- st_chainl "$d"
    sed -i 's/\t2026-10-04T17:59:00Z\t/\t2026-10-04T18:00:00Z\t/' "$d/runs.tsv"
    row a_run_from_six_pm_needs_its_own_night_only 0 "RELEASABLE H=$ST_C" "NOT RELEASABLE" -- st_chain "$d"
    # dupx DIR: every plain run also exists on X (ids +1000), so X has a measured run on each plain lane (#4798 J search); run 201, the chained lane, stays on C only
    dupx() { awk -F '\t' -v OFS='\t' -v X="$ST_X" -v CHAINED=201 '{ print } $3 != CHAINED { $3 = $3 + 1000; $6 = X; print }' "$1/runs.tsv" > "$1/runs.new" && mv "$1/runs.new" "$1/runs.tsv"; printf '1101\t1\n1201\t1\n1301\t1\n1401\t1\n' >> "$1/attempts.tsv"; ranks "$1" "$ST_C" "$ST_X"; }
    d="$tmp/jchain"; fixture "$d"; sed -i '/\t201\t/s/\tschedule\t/\tworkflow_run\t/' "$d/runs.tsv"; dupx "$d"; printf '2026-10-03\t%s\n' "$ST_X" > "$d/picks.tsv"
    row the_search_credits_a_chained_run_to_its_pick_not_the_head 0 "RELEASABLE H=$ST_X (judged bbbbbbbbbb, head aaaaaaaaaa, lag 1 commits)" "NOT RELEASABLE" -- st_chain "$d"
    d="$tmp/jstraddle"; fixture "$d"; sed -i '/\t201\t/s/\tschedule\t/\tworkflow_run\t/; s/\t2026-10-04T02:00:00Z\t/\t2026-10-04T13:00:00Z\t/' "$d/runs.tsv"; dupx "$d"; printf '2026-10-03\t%s\n2026-10-04\t%s\n' "$ST_C" "$ST_X" > "$d/picks.tsv"
    row the_search_skips_a_commit_whose_chained_run_straddles_two_picks 0 "NOT RELEASABLE: v-b" "(judged" -- st_chain "$d"
    # a pick rerun the next evening keeps its created_at, so its producers built the old night's pick: they are created on
    #   the new night, but the newest pick run created before them is the old one (measured: one pick built, another judged green)
    st_rr() { printf '2026-10-05\t%s\n2026-10-06\t%s\n' "$ST_X" "$ST_C" > "$1/picks.tsv"; printf '2026-10-05T20:30:00Z\t%s\n2026-10-06T20:30:00Z\t2026-10-06T20:35:00Z\n' "$2" > "$1/pickruns.tsv"; st_chainl "$1"; }
    d="$tmp/rerun"; fixture "$d"; sed -i '/\t201\t/s/\tschedule\t/\tworkflow_run\t/; s/\t2026-10-04T02:00:00Z\t/\t2026-10-06T19:05:00Z\t/' "$d/runs.tsv"
    row a_rerun_pick_next_evening_credits_the_old_night 0 "201 measured the pick bbbbbbbbbb" "RELEASABLE H=" -- st_rr "$d" 2026-10-06T19:00:00Z
    d="$tmp/rerun2"; fixture "$d"; sed -i '/\t201\t/s/\tschedule\t/\tworkflow_run\t/; s/\t2026-10-04T02:00:00Z\t/\t2026-10-06T21:05:00Z\t/' "$d/runs.tsv"
    row a_rerun_after_the_next_pick_is_not_measured 0 "201 measured the pick the pick run of 2026-10-05T20:30:00Z was rerun after 2026-10-06T20:30:00Z" "RELEASABLE H=" -- st_rr "$d" 2026-10-06T21:00:00Z
    d="$tmp/rerun3"; fixture "$d"; sed -i '/\t201\t/s/\tschedule\t/\tworkflow_run\t/; s/\t2026-10-04T02:00:00Z\t/\t2026-10-06T21:05:00Z\t/' "$d/runs.tsv"
    row a_pick_run_not_rerun_judges_its_own_night 0 "RELEASABLE H=$ST_C" "NOT RELEASABLE" -- st_rr "$d" 2026-10-05T20:35:00Z
    # a full list (100 runs, the two picks and 98 from DAY): a rerun of a run older than the list is unseen (measured: green
    #   on A from B's builds), so the list must reach back 30 days, the rerun limit
    st_rrf() {
        printf '2026-10-05\t%s\n2026-10-06\t%s\n' "$ST_X" "$ST_C" > "$1/picks.tsv"
        awk -v D="$2" 'BEGIN { print "2026-10-05T20:30:00Z\t2026-10-05T20:35:00Z"; print "2026-10-06T20:30:00Z\t2026-10-06T20:35:00Z"
            for (i = 0; i < 98; i++) { s = sprintf("%sT%02d:%02d:00Z", D, int(i / 60), i % 60); print s "\t" s } }' > "$1/pickruns.tsv"
        st_chainl "$1"
    }
    row a_full_pick_run_list_under_30_days_is_not_measured 0 "201 measured the pick the 100 pick runs listed reach back only to 2026-10-01T00:00:00Z" "RELEASABLE H=" -- st_rrf "$d" 2026-10-01
    row a_full_pick_run_list_over_30_days_judges 0 "RELEASABLE H=$ST_C" "NOT RELEASABLE" -- st_rrf "$d" 2026-09-06
    d="$tmp/noprun"; fixture "$d"; sed -i '/\t201\t/s/\tschedule\t/\tworkflow_run\t/' "$d/runs.tsv"; : > "$d/pickruns.tsv"
    row a_chained_run_without_a_read_pick_run_is_not_measured 0 "201 measured the pick no pick run seen at or before it" "RELEASABLE H=" -- st_chainl "$d"
    st_prr() {
        printf '%s' "$1" > "$tmp/gqlrr.json"; : > "$tmp/prr.fail"
        pickruns_read "$tmp/gqlrr.json" "$tmp/prr.tsv" "$tmp/prr.fail"
        cat "$tmp/prr.tsv" "$tmp/prr.fail"
    }
    row pickruns_read_keeps_each_run 0 "2026-10-05T20:35:00Z" "did not parse" -- st_prr '{"data":{"pk":{"runs":{"nodes":[{"createdAt":"2026-10-05T20:30:00Z","updatedAt":"2026-10-05T20:35:00Z"}]}}}}'
    row a_pick_run_list_that_is_not_read_failed 0 "the Nightly pick runs did not parse" "2026" -- st_prr '{"data":{"pk":null}}'
    d="$tmp/schedpick"; fixture "$d"; sed -i "s/$ST_C/$ST_X/" "$d/picks.tsv"
    row a_schedule_lane_ignores_the_picks 0 "RELEASABLE H=$ST_C" "NOT RELEASABLE" -- st_decide "$d"
    st_picks() { printf '%s' '{"data":{"repository":{"nightly":{"nodes":[{"name":"2026-10-03","target":{"oid":"c3"}},{"name":"probe","target":{"oid":"p"}}]}}}}' > "$tmp/gqlp.json"; picks_of "$tmp/gqlp.json"; }
    st_pr() { printf '%s' "$2" > "$tmp/gqlr.json"; picks_read "$tmp/gqlr.json" "$tmp/pr.tsv" "$tmp/pr.fail"; printf 'rows=%s fail=%s\n' "$(wc -l < "$tmp/pr.tsv")" "$(cat "$tmp/pr.fail")"; }
    row an_absent_pick_is_a_good_read 0 "rows=0 fail=" "did not parse" -- st_pr x '{"data":{"repository":{"nightly":{"nodes":[]}}}}'
    row a_pick_read_with_no_refs_field_failed 0 "fail=the nightly pick refs did not parse" "rows=1" -- st_pr x '{"data":{"repository":{"nightly":null}}}'
    row picks_read_keeps_the_dated_rows 0 "rows=1 fail=" "did not parse" -- st_pr x '{"data":{"repository":{"nightly":{"nodes":[{"name":"2026-10-03","target":{"oid":"c3"}}]}}}}'
    row picks_of_reads_dated_nights_only 0 "$(printf '2026-10-03\tc3')" "probe" -- st_picks
    d="$tmp/fromchain"; fixture "$d"; sed -i '/\t201\t/s/\tschedule\t/\tworkflow_run\t/' "$d/runs.tsv"
    st_a_from_run_reads_the_picks() { (LANES="$(printf '%s\n' "$ST_LANES" | sed -e '/^v-b;/s/\^schedule\$/^workflow_run$/')"; NOW=2026-10-04T06:00:00Z; OUTDIR="$tmp/ofc"; INBOXF=""; run_train "$d"); }
    row a_from_run_reads_the_picks 0 "RELEASABLE H=$ST_C" "NOT RELEASABLE" -- st_a_from_run_reads_the_picks
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
    st_greens_in_a_row_by_nightly_greens() { printf "greens=%s\n" "$(greens "$tmp/hist" 2026-10-04)"; }
    row greens_in_a_row_by_nightly_greens 0 "greens=2" "" -- st_greens_in_a_row_by_nightly_greens
    printf 'run_id\tcreated_at\tbranch\tevent\tconclusion\tattempt\nt1\t2026-10-03T04:45:00Z\tmain\ttimer\tsuccess\t1\n' > "$tmp/thist"
    st_timer_runs_never_count_toward_b1() { printf "greens=%s\n" "$(greens "$tmp/thist" 2026-10-04)"; }
    row timer_runs_never_count_toward_b1 0 "greens=0" "" -- st_timer_runs_never_count_toward_b1
    d="$tmp/trap"; fixture "$d"
    row failure_part_way_prints_the_trap_line 2 "NOT RELEASABLE: nightly-train, judge (planted fault) failed" "RELEASABLE H=" -- \
        env NIGHTLY_TRAIN_FAULT=judge bash "$SCRIPT_PATH" --from "$d" --out "$tmp/out" --now 2026-10-04T06:00:00Z
    row the_trap_line_is_recorded 0 "NOT RELEASABLE: nightly-train, judge" "" -- cat "$tmp/out/2026-10-04/line"
    st_verdict_lanes_are_release_day_checks() { printf "verdict=%s;coverage=%s\n" "$(printf "%s\n" "$LANES" | awk -F ";" "\$2 == \"verdict\" { printf \"%s%s\", s, \$1; s = \" \" }")" "$(printf "%s\n" "$LANES" | awk -F ";" "\$1 == \"coverage\" { print \$2 }")"; }
    row verdict_lanes_are_release_day_checks 0 "verdict=ci-main deep-doctests deep-nodefault deep-examples deep-bins-build deep-bins-smoke dogfood models readiness milestone cleanroom-cpu cleanroom-gpu assets preflight publish-dryrun;coverage=info" "" -- \
        st_verdict_lanes_are_release_day_checks
    st_norm() {
        printf '{"workflows":[{"node_id":"WA","path":".github/workflows/a.yml"},{"node_id":"WC","path":".github/workflows/c.yml"},{"node_id":"WP","path":".github/workflows/nightly-pick.yml"}]}\n' > "$tmp/wf.json"
        printf '%s' '{"data":{"repository":{"defaultBranchRef":{"name":"main","target":{"oid":"'"$ST_C"'","tree":{"oid":"t"},"statusCheckRollup":{"contexts":{"nodes":[{},{"name":"gate","status":"COMPLETED","conclusion":"SUCCESS","startedAt":"s","completedAt":"e","checkSuite":{"status":"COMPLETED","conclusion":"SUCCESS","branch":{"name":"main"},"workflowRun":{"databaseId":301,"event":"push","createdAt":"c","workflow":{"id":"WC"}}}}]}}}}},"w0":{"id":"WA","runs":{"nodes":[{"databaseId":101,"createdAt":"c1","event":"schedule","checkSuite":{"status":"COMPLETED","conclusion":"SUCCESS","branch":{"name":"main"},"commit":{"oid":"'"$ST_C"'"},"checkRuns":{"nodes":[{"name":"job-a","status":"COMPLETED","conclusion":"SUCCESS","startedAt":"s1","completedAt":"e1"},{"name":"job-a","status":"COMPLETED","conclusion":"FAILURE","startedAt":"s0","completedAt":"e0"}]}}}]}}}}' > "$tmp/gql.json"
        normalize "$tmp/gql.json" "$tmp/wf.json"
    }
    row normalize_maps_runs_jobs_and_the_rollup 0 "$(printf 'rollup\t.github/workflows/c.yml\t301\tpush\tmain\t%s\tc\tCOMPLETED\tSUCCESS\tgate' "$ST_C")" "" -- st_norm
    row normalize_counts_a_rerun_job 0 "$(printf 'job-a\tCOMPLETED\tSUCCESS\ts1\te1\t2')" "" -- st_norm
    row gql_query_names_every_producer 0 'w1: node(id: "WC")' "" -- gql_query "$(printf '%s\n' "$ST_LANES" | awk -F ';' '$4 ~ /[ac][.]yml$/')" "$tmp/wf.json"
    row gql_query_refuses_a_missing_producer 5 "" "query" -- gql_query "$ST_LANES" "$tmp/wf.json"
    row gql_query_reads_the_nightly_picks 0 'nightly: refs(refPrefix: "refs/heads/nightly/"' "" -- gql_query "$(printf '%s\n' "$ST_LANES" | awk -F ';' '$4 ~ /[ac][.]yml$/')" "$tmp/wf.json"
    row gql_query_reads_the_pick_runs 0 'pk: node(id: "WP") { ... on Workflow { runs(first: 100) { nodes { createdAt updatedAt' "" -- gql_query "$(printf '%s\n' "$ST_LANES" | awk -F ';' '$4 ~ /[ac][.]yml$/')" "$tmp/wf.json"
    # the depth of each top-level field in the query: GitHub refuses refs nested in defaultBranchRef (measured: "Field
    #   'refs' doesn't exist on type 'Ref'", every lane not_measured), and a fixture answer never sees the nesting
    st_gql_depth() {
        gql_query "$(printf '%s\n' "$ST_LANES" | awk -F ';' '$4 ~ /[ac][.]yml$/')" "$tmp/wf.json" | awk -v RS='\001' '{
            for (i = 1; i <= length($0); i++) { c = substr($0, i, 1); d += (c == "{") - (c == "}")
                if (substr($0, i, 9) == " nightly:" || substr($0, i, 5) == " pk: " || substr($0, i, 5) == " w0: ") printf "%s=%d ", substr($0, i + 1, 2), d }
            printf "end=%d\n", d }'
    }
    row gql_query_nests_each_field_where_github_has_it 0 "ni=2 w0=1 pk=1 end=0" "" -- st_gql_depth
    st_gql_nopick() { sed -e 's/,{"node_id":"WP","path":".github\/workflows\/nightly-pick.yml"}//' "$tmp/wf.json" > "$tmp/wfnp.json"; gql_query "$(printf '%s\n' "$ST_LANES" | awk -F ';' '$4 ~ /[ac][.]yml$/')" "$tmp/wfnp.json"; }
    row gql_query_refuses_without_the_pick_workflow 5 "" "query" -- st_gql_nopick
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
    st_hist() {
        local ctx='{"pageInfo":{"hasNextPage":false},"nodes":[{"name":"gate","status":"COMPLETED","conclusion":"SUCCESS","startedAt":"s","completedAt":"e","checkSuite":{"status":"COMPLETED","conclusion":"SUCCESS","branch":{"name":"main"},"workflowRun":{"databaseId":RID,"event":"push","createdAt":"c","workflow":{"id":"WC"}}}}]}'
        printf '%s' '{"data":{"repository":{"defaultBranchRef":{"name":"main","target":{"oid":"'"$ST_C"'","tree":{"oid":"t"},"statusCheckRollup":{"contexts":'"${ctx/RID/301}"'},"history":{"nodes":[{"oid":"'"$ST_C"'","statusCheckRollup":{"contexts":'"${ctx/RID/301}"'}},{"oid":"'"$ST_X"'","statusCheckRollup":{"contexts":'"${ctx/RID/302}"'}}]}}}}}}' > "$tmp/hist.json"
        normalize "$tmp/hist.json" "$tmp/wf.json" | cut -f1,3,6 | sort | uniq -c
    }
    # #4798: a push run on an older main commit is read from the history rollup, and the head is not read twice
    row normalize_reads_main_history_rollups 0 "$(printf '1 rollup\t302\t%s' "$ST_X")" "$(printf '2 rollup\t301')" -- st_hist
    st_rank() { rank_rows "$tmp/hist.json"; }
    row rank_rows_are_the_head_then_the_history_once 0 "$(printf '0\t%s\n1\t%s' "$ST_C" "$ST_X")" "$(printf '2\t')" -- st_rank
    d="$tmp/rankhead"; fixture "$d"; sed -i "s/$ST_C/$ST_X/g" "$d/runs.tsv"; ranks "$d" "$ST_Y" "$ST_X"
    row a_rank_list_for_another_head_is_ignored 0 "NOT RELEASABLE: v-a, not_measured (+2 more)" "RELEASABLE H=" -- st_decide "$d"
    d="$tmp/noprod"; fixture "$d"
    st_a_lane_without_a_producer_is_not_measured() { decide "$ST_LANES
v-e;verdict;check E;-;-;-" "$d" 2026-10-04T06:00:00Z; cat "$d/line" "$d/lanes.tsv"; }
    row a_lane_without_a_producer_is_not_measured 0 "$(printf 'v-e\tverdict\tnot_measured')" "RELEASABLE H=" -- \
        st_a_lane_without_a_producer_is_not_measured
    d="$tmp/green1"; fixture "$d"
    st_stdout_is_exactly_one_line() { bash "$SCRIPT_PATH" --from "$d" --out "$tmp/o1" --now 2026-10-04T06:00:00Z | awk "END { print \"lines=\" NR }"; }
    row stdout_is_exactly_one_line 0 "lines=1" "" -- \
        st_stdout_is_exactly_one_line
    st_a_fault_after_judging_leaves_one_history_row() { NIGHTLY_TRAIN_FAULT=history bash "$SCRIPT_PATH" --from "$d" --out "$tmp/o3" --now 2026-10-04T06:00:00Z > /dev/null; awk -F "\t" "NR > 1 { n++; if (\$5 == \"failure\") f++ } END { print \"rows=\" n+0, \"failure=\" f+0 }" "$tmp/o3/history.tsv"; }
    row a_fault_after_judging_leaves_one_history_row 0 "rows=1 failure=1" "" -- \
        st_a_fault_after_judging_leaves_one_history_row
    st_a_fault_after_history_adds_no_second_row() { NIGHTLY_TRAIN_FAULT=inbox bash "$SCRIPT_PATH" --from "$d" --out "$tmp/o4" --inbox "$tmp/ib4" --now 2026-10-04T06:00:00Z > /dev/null; awk -F "\t" "NR > 1 { n++; if (\$5 == \"failure\") f++ } END { print \"rows=\" n+0, \"failure=\" f+0 }" "$tmp/o4/history.tsv"; }
    row a_fault_after_history_adds_no_second_row 0 "rows=1 failure=0" "" -- \
        st_a_fault_after_history_adds_no_second_row
    st_a_closed_stdout_adds_no_second_row() { bash "$SCRIPT_PATH" --from "$d" --out "$tmp/o5" --now 2026-10-04T06:00:00Z >&- 2> /dev/null; awk -F "\t" "NR > 1 { n++; if (\$5 == \"failure\") f++ } END { print \"rows=\" n+0, \"failure=\" f+0 }" "$tmp/o5/history.tsv"; }
    row a_closed_stdout_adds_no_second_row 0 "rows=1 failure=0" "" -- \
        st_a_closed_stdout_adds_no_second_row
    st_a_self_test_writes_only_its_own_inbox() { case "$INBOX" in "$tmp"/*) echo sandbox ;; *) echo "INBOX is $INBOX" ;; esac; }
    row a_self_test_writes_only_its_own_inbox 0 "sandbox" "" -- \
        st_a_self_test_writes_only_its_own_inbox
    st_a_planted_fault_never_writes_the_named_inbox() { INBOX="$tmp/c6.md" NIGHTLY_TRAIN_FAULT=bundle bash "$SCRIPT_PATH" --from "$d" --out "$tmp/o6" --now 2026-10-04T06:00:00Z > /dev/null; NIGHTLY_TRAIN_FAULT=bundle bash "$SCRIPT_PATH" --from "$d" --out "$tmp/o7" --inbox "$tmp/c7.md" --now 2026-10-04T06:00:00Z > /dev/null; printf "caller=%s lines=%s\n" "$(cat "$tmp/c6.md" "$tmp/c7.md" 2>/dev/null | wc -l)" "$(cat "$tmp/o6"/*/line "$tmp/o7"/*/line | grep -c "planted fault")"; }
    row a_planted_fault_never_writes_the_named_inbox 0 "caller=0 lines=2" "" -- \
        st_a_planted_fault_never_writes_the_named_inbox
    d="$tmp/legs"; fixture "$d"
    printf 'wf\t.github/workflows/a.yml\t101\tschedule\tmain\t%s\t2026-10-04T01:00:00Z\tCOMPLETED\tSUCCESS\tjob-a\tCOMPLETED\tFAILURE\t2026-10-04T01:00:30Z\t2026-10-04T01:20:00Z\t2\n' "$ST_C" >> "$d/runs.tsv"
    row a_failed_parallel_leg_of_the_same_name_is_red 0 "NOT RELEASABLE: v-a, 101" "RELEASABLE H=" -- st_decide "$d"
    # J (#4798): main moved after the producers ran, so the head has no run; every verdict producer ran on one commit.
    d="$tmp/moved"; fixture "$d"; sed -i "s/$ST_C/$ST_X/g" "$d/runs.tsv"; ranks "$d" "$ST_C" "$ST_X"
    row a_moved_head_judges_the_commit_every_producer_ran_on 0 "RELEASABLE H=$ST_X (judged bbbbbbbbbb, head aaaaaaaaaa, lag 1 commits)" "NOT RELEASABLE" -- st_decide "$d"
    # one commit for every lane, never a per-lane mix: v-a ran on X only, the rest on Y only
    d="$tmp/split"; fixture "$d"; sed -i "s/$ST_C/$ST_Y/g; /\t101\t/s/$ST_Y/$ST_X/" "$d/runs.tsv"; ranks "$d" "$ST_C" "$ST_X" "$ST_Y"
    row producers_split_across_commits_are_not_measured 0 "NOT RELEASABLE: v-a, not_measured (+2 more)" "RELEASABLE H=" -- st_decide "$d"
    # a newer commit that only some producers ran on does not win; the lag counts it and the head
    d="$tmp/partial"; fixture "$d"; sed -i "s/$ST_C/$ST_X/g" "$d/runs.tsv"; ranks "$d" "$ST_C" "$ST_Y" "$ST_X"
    printf 'wf\t.github/workflows/a.yml\t102\tschedule\tmain\t%s\t2026-10-04T05:00:00Z\tCOMPLETED\tSUCCESS\tjob-a\tCOMPLETED\tSUCCESS\t2026-10-04T05:01:00Z\t2026-10-04T05:30:00Z\t1\n' "$ST_Y" >> "$d/runs.tsv"
    row a_newer_partial_commit_does_not_win 0 "RELEASABLE H=$ST_X (judged bbbbbbbbbb, head aaaaaaaaaa, lag 2 commits)" "NOT RELEASABLE" -- st_decide "$d"
    # a commit with no run at all still counts in the lag: the lag is the history position, not the commits seen in runs
    d="$tmp/gap"; fixture "$d"; sed -i "s/$ST_C/$ST_X/g" "$d/runs.tsv"; ranks "$d" "$ST_C" "$ST_Y" "$ST_X"
    row a_commit_without_runs_counts_in_the_lag 0 "(judged bbbbbbbbbb, head aaaaaaaaaa, lag 2 commits)" "NOT RELEASABLE" -- st_decide "$d"
    # newest = nearest the head in history, not the latest run time: the older commit's runs started later
    d="$tmp/order"; fixture "$d"; sed -i "s/$ST_C/$ST_X/g" "$d/runs.tsv"; ranks "$d" "$ST_C" "$ST_X" "$ST_Y"
    awk -F '\t' -v OFS='\t' -v y="$ST_Y" '{ $3 = $3 + 1000; $6 = y; $7 = "2026-10-05T01:00:00Z"; print }' "$d/runs.tsv" >> "$d/runs.tsv.y" && cat "$d/runs.tsv.y" >> "$d/runs.tsv"
    row the_nearest_commit_wins_not_the_latest_run 0 "H=$ST_X (judged bbbbbbbbbb, head aaaaaaaaaa, lag 1 commits)" "NOT RELEASABLE" -- st_decide "$d"
    # a red on J is a red, named with the head and the lag
    d="$tmp/movedred"; fixture "$d"; sed -i "s/$ST_C/$ST_X/g; /\t201\t/s/\tx\tCOMPLETED\tSUCCESS\t/\tx\tCOMPLETED\tFAILURE\t/" "$d/runs.tsv"; ranks "$d" "$ST_C" "$ST_X"
    row a_red_on_the_judged_commit_is_red 0 "NOT RELEASABLE: v-b, 201 (judged bbbbbbbbbb, head aaaaaaaaaa, lag 1 commits)" "RELEASABLE H=" -- st_decide "$d"
    # the lookback is capped at MAX_LAG commits behind the head: rank 6 is judged, rank 7 is not_measured, not a stale verdict
    d="$tmp/edge"; fixture "$d"; sed -i "s/$ST_C/$ST_X/g" "$d/runs.tsv"; ranks "$d" "$ST_C" "$ST_Y" "$ST_Y" "$ST_Y" "$ST_Y" "$ST_Y" "$ST_X"
    row a_judged_commit_at_the_lag_cap_is_judged 0 "RELEASABLE H=$ST_X (judged bbbbbbbbbb, head aaaaaaaaaa, lag 6 commits)" "NOT RELEASABLE" -- st_decide "$d"
    d="$tmp/stale"; fixture "$d"; sed -i "s/$ST_C/$ST_X/g" "$d/runs.tsv"; ranks "$d" "$ST_C" "$ST_Y" "$ST_Y" "$ST_Y" "$ST_Y" "$ST_Y" "$ST_Y" "$ST_X"
    row a_judged_commit_past_the_lag_cap_is_not_measured 0 "NOT RELEASABLE: v-a, not_measured (+2 more)" "RELEASABLE H=" -- st_decide "$d"
    # no history read (an old raw, a failed history): only the head is a candidate, as before
    d="$tmp/norank"; fixture "$d"; sed -i "s/$ST_C/$ST_X/g" "$d/runs.tsv"; : > "$d/rank.tsv"
    row no_history_judges_only_the_head 0 "NOT RELEASABLE: v-a, not_measured (+2 more)" "RELEASABLE H=" -- st_decide "$d"
    st_moved_bundle() {
        local m="$tmp/mb/raw" t='\t'
        mkdir -p "$m"; printf '0\t%s\n1\t%s\n' "$ST_C" "$ST_X" > "$m/rank.tsv"; printf '%s\n' "$ST_C" > "$m/C"; printf 'ok\n' > "$m/read"; printf 'cccc\n' > "$m/tree"; printf '2026-10-03\t%s\n' "$ST_X" > "$m/picks.tsv"; printf '2026-10-03T20:30:00Z\t2026-10-03T20:35:00Z\n' > "$m/pickruns.tsv"; printf '501\t1\n502\t1\n503\t1\n504\t1\n' > "$m/attempts.tsv"
        {
            printf "rollup${t}.github/workflows/ci.yml${t}501${t}push${t}main${t}%s${t}2026-10-03T23:00:00Z${t}COMPLETED${t}SUCCESS${t}ci / gate${t}COMPLETED${t}SUCCESS${t}${t}${t}1\n" "$ST_X"
            printf "rollup${t}.github/workflows/ci.yml${t}501${t}push${t}main${t}%s${t}2026-10-03T23:00:00Z${t}COMPLETED${t}SUCCESS${t}workspace-test${t}COMPLETED${t}SUCCESS${t}${t}${t}1\n" "$ST_X"
            printf "wf${t}.github/workflows/examples-nightly.yml${t}502${t}workflow_run${t}main${t}%s${t}2026-10-04T01:00:00Z${t}COMPLETED${t}SUCCESS${t}examples${t}COMPLETED${t}SUCCESS${t}${t}${t}1\n" "$ST_X"
            printf "wf${t}.github/workflows/nightly.yml${t}503${t}workflow_run${t}main${t}%s${t}2026-10-04T01:10:00Z${t}COMPLETED${t}SUCCESS${t}x86_64-unknown-linux-gnu on box${t}COMPLETED${t}SUCCESS${t}${t}${t}1\n" "$ST_X"
            printf "wf${t}.github/workflows/models-nightly.yml${t}504${t}schedule${t}main${t}%s${t}2026-10-04T01:20:00Z${t}COMPLETED${t}SUCCESS${t}models${t}COMPLETED${t}SUCCESS${t}${t}${t}1\n" "$ST_X"   # the models lane has a producer since #4868
        } > "$m/runs.tsv"
        bash "$SCRIPT_PATH" --out "$tmp/mbout" --from "$m" --now 2026-10-04T06:00:00Z > /dev/null 2>&1
        cat "$tmp/mbout/2026-10-04/line"; grep -E '^(ci-main|deep-examples|deep-bins-build)	|^# (head|C|tree)	' "$tmp/mbout/2026-10-04/bundle.tsv"
    }
    row a_moved_head_bundle_names_judged_and_head 0 "(judged bbbbbbbbbb, head aaaaaaaaaa, lag 1 commits) [C=bbbbbbbbbb pin=" "$(printf '# tree\tcccc')" -- st_moved_bundle
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
        '  *rate_limit*) echo "${STUB_LIM-5000} ${STUB_REM:-5000}" ;;' \
        '  *actions/workflows*) printf "HTTP/2.0 200 OK\r\netag: \"e1\"\r\n\r\n"; cat "$STUB_WF" ;;' \
        '  *graphql*) echo "{\"errors\":[{\"message\":\"stub\"}]}" ;;' \
        '  *) echo 1 ;;' 'esac' > "$tmp/bin/gh"; chmod +x "$tmp/bin/gh"
    printf '{"workflows":[%s]}\n' '{"node_id":"WA","path":".github/workflows/a.yml"},{"node_id":"WB","path":".github/workflows/b.yml"},{"node_id":"WC","path":".github/workflows/c.yml"},{"node_id":"WD","path":".github/workflows/d.yml"},{"node_id":"WP","path":".github/workflows/nightly-pick.yml"}' > "$tmp/wfall.json"
    st_fetch() {
        local f; f="$(realpath -m -s -- "$tmp/f$1")"
        mkdir -p "$f"
        PATH="$tmp/bin:$PATH" STUB_WF="$tmp/wfall.json" LANES="$ST_LANES" fetch "$f" "$f/cache"
        cat "$f/read"
    }
    st_fetch_refuses_under_the_rate_floor() { STUB_REM=999 st_fetch 1; }
    row fetch_refuses_under_the_rate_floor 0 "failed: core remaining 999 under 1000" "" -- st_fetch_refuses_under_the_rate_floor
    st_a_job_token_under_a_fifth_refuses() { STUB_LIM=1000 STUB_REM=199 st_fetch 4; }
    row a_job_token_under_a_fifth_refuses 0 "failed: core remaining 199 under 200" "" -- st_a_job_token_under_a_fifth_refuses
    st_a_job_token_at_a_fifth_reads() { CALLS=0; STUB_LIM=1000 STUB_REM=200 st_fetch 5; }
    row a_job_token_at_a_fifth_reads 0 "failed: GraphQL errors" "under" -- st_a_job_token_at_a_fifth_reads
    st_an_unreadable_limit_refuses() { STUB_LIM= STUB_REM=5000 st_fetch 6; }
    row an_unreadable_limit_refuses 0 "failed: rate_limit unreadable" "" -- st_an_unreadable_limit_refuses
    st_fetch_refuses_past_the_call_budget() { MAX_CALLS=0; st_fetch 2; }
    row fetch_refuses_past_the_call_budget 0 "failed: call budget" "" -- st_fetch_refuses_past_the_call_budget
    st_fetch_refuses_a_graphql_error() { CALLS=0; st_fetch 3; }
    row fetch_refuses_a_graphql_error 0 "failed: GraphQL errors" "" -- st_fetch_refuses_a_graphql_error
    row an_inbox_read_back_mismatch_exits_non_zero 1 "inbox: read-back mismatch" "" -- \
        bash "$SCRIPT_PATH" --from "$d" --out "$tmp/o9" --inbox /dev/null --now 2026-10-04T06:00:00Z
    row a_closed_stdout_keeps_the_real_verdict 0 "NOT RELEASABLE: ci-main, not_measured (+14 more) [C=aaaaaaaaaa pin=" "nightly-train" -- cat "$tmp/o5/2026-10-04/line"
    row a_registry_token_refuses_to_start 2 "NOT RELEASABLE: nightly-train, no-token failed" "RELEASABLE H=" -- \
        env CARGO_REGISTRY_TOKEN=x bash "$SCRIPT_PATH" --from "$d" --out "$tmp/ot1" --now 2026-10-04T06:00:00Z
    row an_empty_named_registry_token_refuses 2 "NOT RELEASABLE: nightly-train, no-token failed" "RELEASABLE H=" -- \
        env CARGO_REGISTRIES_MIRROR_TOKEN= bash "$SCRIPT_PATH" --from "$d" --out "$tmp/ot2" --now 2026-10-04T06:00:00Z
    st_a_credentials_file_refuses() { mkdir -p "$tmp/ch"; : > "$tmp/ch/credentials.toml"; CARGO_HOME="$tmp/ch" bash "$SCRIPT_PATH" --from "$d" --out "$tmp/ot3" --now 2026-10-04T06:00:00Z; }
    row a_credentials_file_refuses 2 "NOT RELEASABLE: nightly-train, no-token failed" "RELEASABLE H=" -- \
        st_a_credentials_file_refuses
    st_exit_verdict_maps_releasable_to_0() { verdict_rc "RELEASABLE H=$ST_C"; a=$?; verdict_rc "NOT RELEASABLE: v-a, 101"; echo "rc=$a rc=$?"; }
    row exit_verdict_maps_releasable_to_0 0 "rc=0 rc=1" "" -- st_exit_verdict_maps_releasable_to_0
    row exit_verdict_not_releasable_exits_1 1 "NOT RELEASABLE: ci-main" "nightly-train" -- \
        bash "$SCRIPT_PATH" --from "$d" --out "$tmp/ov" --exit-verdict --now 2026-10-04T06:00:00Z
    st_a_legacy_config_token_refuses() { mkdir -p "$tmp/cc"; printf "[registry]\ntoken = \"x\"\n" > "$tmp/cc/config.toml"; CARGO_HOME="$tmp/cc" bash "$SCRIPT_PATH" --from "$d" --out "$tmp/ot4" --now 2026-10-04T06:00:00Z; }
    row a_legacy_config_token_refuses 2 "NOT RELEASABLE: nightly-train, no-token failed" "RELEASABLE H=" -- \
        st_a_legacy_config_token_refuses
    st_a_home_credentials_file_refuses_whatever_cargo_home() { mkdir -p "$tmp/hh/.cargo"; : > "$tmp/hh/.cargo/credentials"; HOME="$tmp/hh" bash "$SCRIPT_PATH" --from "$d" --out "$tmp/ot5" --now 2026-10-04T06:00:00Z; }
    row a_home_credentials_file_refuses_whatever_cargo_home 2 "NOT RELEASABLE: nightly-train, no-token failed" "RELEASABLE H=" -- \
        st_a_home_credentials_file_refuses_whatever_cargo_home
    st_a_home_config_token_refuses_whatever_cargo_home() { mkdir -p "$tmp/hc/.cargo"; printf "token=\"x\"\n" > "$tmp/hc/.cargo/config"; HOME="$tmp/hc" bash "$SCRIPT_PATH" --probe-no-token; }
    row a_home_config_token_refuses_whatever_cargo_home 1 "no-token: a token line in the config under $tmp/hc/.cargo" "clear" -- \
        st_a_home_config_token_refuses_whatever_cargo_home
    row the_probe_is_clear_where_no_token_is_reachable 0 "no-token: clear" "" -- bash "$SCRIPT_PATH" --probe-no-token
    d="$tmp/evg"; fixture "$d"
    st_exit_verdict_releasable_run_exits_0() { (LANES="$ST_LANES"; EXIT_VERDICT=1; NOW=2026-10-04T06:00:00Z; OUTDIR="$tmp/oev"; INBOXF=""; run_train "$d"); }
    row exit_verdict_releasable_run_exits_0 0 "RELEASABLE H=$ST_C" "NOT RELEASABLE" -- \
        st_exit_verdict_releasable_run_exits_0
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
    st_an_edited_bundle_leaves_its_line_and_one_row() { awk -F "\t" "NR > 1 { n++; if (\$5 == \"failure\") f++ } END { printf \"rows=%d failure=%d \", n, f }" "$tmp/o2/history.tsv"; grep -o "pin failed" "$tmp/o2/2026-10-04/line"; }
    row an_edited_bundle_leaves_its_line_and_one_row 0 "rows=1 failure=1 pin" "" -- \
        st_an_edited_bundle_leaves_its_line_and_one_row
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
m02_run_on_any_commit	s/if (RH\[r\] != J) {/if (0) {/
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
m25_cut_rollup_read	s/if (\$x.statusCheckRollup.contexts.pageInfo.hasNextPage \/\/ false) then/if false then/
m26_day_set_late	/^    DAY=.*# before any step/d;s/^    step outdir$/    DAY="${NOW%%T*}"; step outdir/
m27_pinned_not_required	s/if \[ -n "\$PINNED" \] || \[ -f "\$HERE\/PIN" \]; then/if [ -f "$HERE\/PIN" ]; then/
m28_unhashed_pin_accepted	s/|| { PIN="unhashed"; exit 1; }/|| :/
m29_rate_floor_ignored	s/\[ "\$rem" -ge "\$fl" \]/[ 1 ]/
m30_call_budget_ignored	s/call_ok() { \[ "\$CALLS" -lt "\$MAX_CALLS" \] || return 1;/call_ok() {/
m31_graphql_errors_read	s/(.errors \/\/ \[\]) | length > 0/false/
m32_inbox_mismatch_exits_0	s/inbox_line "\$line" "\$g" || exit 1/inbox_line "$line" "$g"/
m33_info_attempt_claimed	s/out(i, "green", pick, "1?", /out(i, "green", pick, "1", /
m34_token_guard_dropped	s/^    no_token || exit 1$/    :/
m35_credentials_file_ignored	s/if \[ -e "\$d\/credentials" \] || \[ -e "\$d\/credentials.toml" \]; then/if false; then/
m36_named_registry_token_ignored	s/ || \$1 ~ \/\^CARGO_REGISTRIES_\[A-Za-z0-9_\]+_TOKEN\$\// /
m37_exit_verdict_ignored	s/^    \[ -z "\$EXIT_VERDICT" \] || verdict_rc "\$line" || exit 1$/    :/
m38_verdict_rc_always_red	s/"RELEASABLE "\*) return 0 ;;/"RELEASABLE "*) return 1 ;;/
m39_exit_verdict_always_red	s/^    \[ -z "\$EXIT_VERDICT" \] || verdict_rc "\$line" || exit 1$/    [ -z "$EXIT_VERDICT" ] || exit 1/
m40_config_token_ignored	s/^        if grep -qsE /        if false \&\& grep -qsE /
m41_home_cargo_unchecked	s/ "\$HOME\/.cargo"; do$/; do/
m42_floor_ignores_the_limit	s/fl=\$((lim \/ 5))/fl=$RATE_FLOOR/
m43_any_event_counts	s/ && REV\[r\] ~ EV\[i\]//
m44_chain_ignores_the_pick	s/(pc = chainc(RC\[r\])) != J)/0)/
m45_any_night_counts	s/if (t >= PS\[k\] && t < PE\[k\]) return k/if (PO[k] == C) return k/
m46_window_end_inclusive	s/t < PE\[k\]/t <= PE[k]/
m47_window_start_exclusive	s/t >= PS\[k\]/t > PS[k]/
m48_every_lane_checks_the_pick	s/if (EV\[i\] != "\^workflow_run\$") {/if (0) {/
m49_from_skips_the_picks	s/ tree picks.tsv rank.tsv pickfail pickruns.tsv; do/ tree rank.tsv pickfail pickruns.tsv; do/
m50_picks_any_ref_name	s/ | select(.name | test("^\[0-9\]{4}-\[0-9\]{2}-\[0-9\]{2}\$"))//
m51_query_drops_the_picks	s/ nightly: refs(refPrefix/ unread: refs(refPrefix/
m52_no_straddle_check	s/ && (pc = prevc(RC\[r\])) != J)/ \&\& 0)/
m53_straddle_window_ends_early	s/substr(RC\[r\], 12, 2) < "18"/substr(RC\[r\], 12, 2) < "13"/
m54_prev_is_the_same_night	s/if (PE\[j\] == PS\[k\]) return PO\[j\]/if (j == k) return PO[j]/
m55_judge_only_the_head	s/if (all) { J = h; LAG = k; break }/if (all \&\& k == 0) { J = h; LAG = k; break }/
m56_lag_cap_ignored	s/k <= MAXLAG \&\& k < NRK/k < NRK/
m57_lag_not_printed	s/note = (C != HEAD \&\& /note = (0 \&\& /
m58_a_partial_commit_counts	s/!((h, i) in HAS)) all = 0/0) all = 0/
m59_bundle_keeps_the_head	s/\[\[ "\${j:-}" =~ \^\[0-9a-f\]{40}\$ \]\] \&\& CSHA="\$j"/:/
m60_history_not_read	s/\[\$c\] + \[(\$c.history.nodes \/\/ \[\])\[\] | select(.oid != \$c.oid)\]/[$c]/
m61_head_read_twice	s/ | select(.oid != \$c.oid)\]/]/
m62_cap_off_by_one	s/k <= MAXLAG/k < MAXLAG/
m63_rank_head_unchecked	s/readok \&\& RKO\[0\] == C/readok/
m64_oldest_commit_wins	s/if (all) { J = h; LAG = k; break }/if (all) { J = h; LAG = k }/
m65_rank_history_dropped	s/\[(\$c.history.nodes \/\/ \[\])\[\] | .oid | select(. != \$c.oid)\]/[]/
m66_failed_pick_read_falls_back	s/\[ -s "\$2\/pickfail" \] \&\& \[ "\${rd:-}" = ok \] \&\& rd=/false \&\& rd=/
m67_failed_pick_read_is_an_absent_pick	s/nodes | type) == "array"/nodes | type) != "zzz"/
m68_search_files_a_chained_run_under_the_head	s/hm = chainc(RC\[r\]);/hm = RH[r];/
m69_search_ignores_the_straddle	s/ \&\& prevc(RC\[r\]) != hm) hm = ""/ \&\& 0) hm = ""/
m70_a_chained_run_by_its_own_creation	s/chainc(RC/nightc(RC/g
m71_an_older_pick_rerun_is_ignored	s/if (b > PRC\[k\] \&\& b < PRU\[k\])/if (0)/
m72_an_unread_pick_run_list_passes	s/^    printf .the Nightly pick runs did not parse.n. > "\$3"$/    true/
m73_the_pick_runs_are_not_queried	s/ + " pk: node(id: / + " pq: node(id: /
m74_the_pick_workflow_may_be_missing	s/if \$pk == null then error/if false then error/
m75_from_skips_the_pick_runs	s/ pickfail pickruns.tsv; do/ pickfail; do/
m76_a_full_pick_run_list_is_trusted	s/if (npr >= PRN \&\& sz(o)/if (0 \&\& sz(o)/
m77_the_page_size_is_not_passed	s/ -v PRN="[^"]*"//
m78_the_query_reads_twenty	s/runs(first: \\(\$pn))/runs(first: 20)/
m79_the_rerun_limit_is_31_days	s/sz(t) - 30 \* 86400/sz(t) - 31 * 86400/
m80_the_oldest_run_is_the_newest	s/if (PRC\[k\] < o) o = "" PRC\[k\]/if (0) o = ""/
m81_refs_nested_in_the_branch	s/} } nightly: refs/} nightly: refs/; s/{ name target { oid } } } } "/{ name target { oid } } } } } "/'
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
    --probe-no-token) no_token && echo "no-token: clear"; exit $? ;;
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
