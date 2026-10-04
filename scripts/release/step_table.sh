#!/usr/bin/env bash
# step_table.sh -- the release step table (BLD-002 R0). For every step of a release: how many
# runs, their results, whether it passed first time, how long it ran, how long it waited for its
# inputs and who decided; then the rolled first-pass yield (RTY) over every release given.
# THE RULE  A duration exists only where both of its ends were measured. A step with no measured
#           duration prints UNMEASURED: nothing is estimated, interpolated or filled in.
# THE INPUT --steps: a TSV of step rows, 15 columns, one or more files.
#           --questions: a TSV of operator questions, 9 columns, optional.
#           Every time cell is empty or a UTC instant, YYYY-MM-DDTHH:MMZ or YYYY-MM-DDTHH:MM:SSZ,
#           with its source in the next column. "~", "ETA" and "by" text is refused, so an
#           estimate can never enter a table.
# STEPS     S01 version bump; S02 PR CI on the release commit; S03 merge; S04 clean-room job;
#           S05 and S06 model ladders on the two GPU hosts; S07 CRUX smoke; S08 CPU-host gates;
#           S09 dogfood (S09c is its coverage stage, inside S09 and kept out of the yield);
#           S10 pre-tag check; S11 tag; S12 crates.io publish; S13 GitHub release.
#           X01..X99 are extra work on the release path; --extras prints them one per row.
# FIRST     A step passed first time only if every run of it says yes. Any no makes it no;
#           otherwise any UNMEASURED, or no run at all, makes it UNMEASURED.
# WAIT      From the newest end of an input step at or before the step's last start, to that
#           start. UNMEASURED when an input run's end is unknown and could fall in between, or
#           when a run with no start could be the last one.
# RTY       y = yes / (yes + no) over the releases, for each step measured at least once; RTY
#           is the product of the y. A step never measured is listed, never counted as 1 or 0.
# EXIT      0 the tables; 1 REFUSED: file:line, step and rule for every bad row, and no table;
#           3 caller error
# USAGE     step_table.sh --steps FILE [--steps FILE ...] [--questions FILE ...] [--extras]
#           step_table.sh --selftest | --mutants | -h
set -uo pipefail

PROG="${0##*/}"
SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/${BASH_SOURCE[0]##*/}"
caller_error() { printf 'FAIL  STEPTABLE %s: caller error: %s\n' "$PROG" "$*"; exit 3; }

# Calendar arithmetic and the cell rules both readers share. A time is a UTC instant written
# YYYY-MM-DDTHH:MMZ or YYYY-MM-DDTHH:MM:SSZ that names a real date; anything else is not a time.
TIME_AWK='
    # civil date -> days since 1970-01-01 (H. Hinnant)
    function days(y, m, d,    era, yoe, doy, doe, mp) {
        y -= (m <= 2)
        era = int(y / 400)
        yoe = y - era * 400
        mp = (m + 9) % 12
        doy = int((153 * mp + 2) / 5) + d - 1
        doe = yoe * 365 + int(yoe / 4) - int(yoe / 100) + doy
        return era * 146097 + doe - 719468
    }
    # days since 1970-01-01 -> YYYY-MM-DD
    function civil(z,    era, doe, yoe, y, doy, mp, d, m) {
        z += 719468
        era = int(z / 146097)
        doe = z - era * 146097
        yoe = int((doe - int(doe / 1460) + int(doe / 36524) - int(doe / 146096)) / 365)
        y = yoe + era * 400
        doy = doe - (365 * yoe + int(yoe / 4) - int(yoe / 100))
        mp = int((5 * doy + 2) / 153)
        d = doy - int((153 * mp + 2) / 5) + 1
        m = (mp < 10) ? mp + 3 : mp - 9
        return sprintf("%04d-%02d-%02d", y + (m <= 2), m, d)
    }
    function dayno(s) { return days(substr(s, 1, 4) + 0, substr(s, 6, 2) + 0, substr(s, 9, 2) + 0) }
    # a real calendar date, written YYYY-MM-DD, year 0001 or later
    function isdate(s,    m, d) {
        if (s !~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]$/) return 0
        m = substr(s, 6, 2) + 0; d = substr(s, 9, 2) + 0
        if (substr(s, 1, 4) + 0 < 1 || m < 1 || m > 12 || d < 1 || d > 31) return 0
        return (civil(dayno(s)) == s)
    }
    # a real UTC instant, to the minute or to the second
    function istime(s,    n) {
        if (s ~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]Z$/) n = 0
        else if (s ~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z$/) n = substr(s, 18, 2) + 0
        else return 0
        if (!isdate(substr(s, 1, 10))) return 0
        return (substr(s, 12, 2) + 0 <= 23 && substr(s, 15, 2) + 0 <= 59 && n <= 59)
    }
    # seconds since 1970-01-01T00:00:00Z of an instant istime() accepted
    function secs(s) { return dayno(s) * 86400 + substr(s, 12, 2) * 3600 + substr(s, 15, 2) * 60 + ((length(s) == length("1970-01-01T00:00:00Z")) ? substr(s, 18, 2) : 0) }
    function mins(x) { return sprintf("%.1f", x / 60) }
    # a table cell never carries the column separator, and an empty cell prints -
    function cell(s) { gsub(/\|/, "/", s); return (s == "") ? "-" : s }
    # sorts a[1..n] ascending in place and returns the median; n >= 1
    function median(a, n,    i, j, t) {
        for (i = 2; i <= n; i++) {
            t = a[i]
            for (j = i - 1; j >= 1 && a[j] > t; j--) a[j + 1] = a[j]
            a[j + 1] = t
        }
        return (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2
    }
    function refuse(rule) { printf "REFUSED %s:%d %s: %s\n", FILENAME, FNR, ($2 == "" ? "-" : $2), rule; bad++ }
    # an empty time is unknown; a time must be a UTC instant and must name its source
    function timecell(t, src, col) {
        if (t == "") return 1
        if (!istime(t)) { refuse(col " " t " is not a UTC instant YYYY-MM-DDTHH:MM[:SS]Z: an estimate is not a time"); return 0 }
        if (src == "") { refuse(col " " t " has no source: a time nobody can re-read is not measured"); return 0 }
        return 1
    }
'

# The step reader: the table of each release, its extras and the RTY; or only REFUSED lines and exit 1.
steps_awk() {
    local extras="$1"
    shift
    awk -v extras="$extras" "$TIME_AWK"'
    BEGIN {
        FS = "\t"
        want = "release\tstep_id\tstep\tlane\tstart_utc\tstart_src\tend_utc\tend_src\tresult\tattempts\tattempts_src\tfirst_pass\tdecided_by\tdecided_src\tnote"
        shown = want; gsub(/\t/, " ", shown); nf = split(want, hcol, "\t")
        ncanon = split("S01 S02 S03 S04 S05 S06 S07 S08 S09 S09c S10 S11 S12 S13", canon, " ")
        split("version bump|PR CI on the release commit|merge queue / merge|clean-room job|model ladder, GPU host 1|model ladder, GPU host 2|CRUX smoke|CPU-host gates|dogfood|dogfood coverage stage (inside S09)|pre-tag check|tag|crates.io publish + index check|GitHub release + assets", label, "|")
        for (i = 1; i <= ncanon; i++) name[canon[i]] = label[i]
        split("pass fail refused stopped running not-run", rkind, " ")
        need["S02"] = "S01"; need["S03"] = "S02"
        need["S04"] = "S01"; need["S05"] = "S01"; need["S06"] = "S01"; need["S07"] = "S01"; need["S08"] = "S01"
        need["S09"] = "S05 S06 S07"; need["S09c"] = "S05 S06 S07"
        need["S10"] = "S04 S08 S09"; need["S11"] = "S10"; need["S12"] = "S11"; need["S13"] = "S12"
    }
    FNR == 1 {
        hbad = ($0 != want)
        if (hbad) { printf "REFUSED %s:1 -: the header is not the step header: %s\n", FILENAME, shown; bad++ }
        next
    }
    hbad { next }
    NF != nf { refuse("a step row has " nf " tab-separated fields, as the header does, this one has " NF); next }
    {
        if ($1 !~ /^[0-9A-Za-z.-]+$/) refuse("release " $1 " is not a release name [0-9A-Za-z.-]")
        if ($2 !~ /^(S[0-9][0-9]c?|X[0-9][0-9])$/ || ($2 ~ /^S/ && !($2 in name))) refuse("step_id " $2 " is neither a step S01..S13 or S09c nor an extra X01..X99")
        if ($3 == "") refuse("the step name is empty")
        stok = timecell($5, $6, "start_utc")
        enok = timecell($7, $8, "end_utc")
        if (stok && enok && $5 != "" && $7 != "" && secs($7) < secs($5)) refuse("end_utc " $7 " is before start_utc " $5)
        if ($9 ~ /^running@/) {
            t = substr($9, 9)
            if (!istime(t)) refuse("result " $9 ": running@ wants the UTC instant the run was seen running")
            else if (stok && $5 != "" && secs(t) < secs($5)) refuse("result " $9 " is before start_utc " $5)
            if ($7 != "") refuse("result " $9 " has an end_utc: a run that ended is not running")
        } else if ($9 == "not-run") {
            if ($5 != "" || $7 != "") refuse("result not-run has a time: a step that never ran has no start and no end")
        } else if ($9 !~ /^(pass|fail|refused|stopped)$/) refuse("result " $9 " is not pass, fail, refused, stopped, not-run or running@<UTC instant>")
        if ($12 !~ /^(yes|no|UNMEASURED)$/) refuse("first_pass " $12 " is not yes, no or UNMEASURED")
        if ($13 != "" && $13 !~ /^(operator:C[0-9]+|cop|quorum:.+|worker:.+|script:.+)$/) refuse("decided_by " $13 " is not operator:C<n>, cop, quorum:<x>, worker:<x> or script:<x>")
        if ($13 != "" && $14 == "") refuse("decided_by " $13 " has no decided_src")
        n++
        rst[n] = $5; ren[n] = $7; rres[n] = $9; rfp[n] = $12; rdb[n] = $13; rid[n] = $2; rname[n] = $3
        if (!($1 in seenrel)) { seenrel[$1] = 1; rels[++nrel] = $1 }
        at[$1, $2, ++cnt[$1, $2]] = n
        if ($2 ~ /^X/) xs[$1] = xs[$1] " " n
    }
    # the instant a running row was seen running, or empty
    function seen(r) { return (rres[r] ~ /^running@/) ? substr(rres[r], 9) : "" }
    # who decided, grouped by kind in order of appearance: "operator: C1, C2; cop"
    function decided(rel, c,    j, r, i, k, v, nk, out) {
        split("", dseen); split("", dvals); split("", dkind); nk = 0
        for (j = 1; j <= cnt[rel, c]; j++) {
            r = at[rel, c, j]
            if (rdb[r] == "" || (rdb[r] in dseen)) continue
            dseen[rdb[r]] = 1
            i = index(rdb[r], ":")
            k = i ? substr(rdb[r], 1, i - 1) : rdb[r]
            v = i ? substr(rdb[r], i + 1) : ""
            if (!(k in dvals)) { dkind[++nk] = k; dvals[k] = "" }
            if (v != "") dvals[k] = dvals[k] (dvals[k] == "" ? "" : ", ") v
        }
        out = ""
        for (i = 1; i <= nk; i++) out = out (i > 1 ? "; " : "") dkind[i] (dvals[dkind[i]] == "" ? "" : ": " dvals[dkind[i]])
        return cell(out)
    }
    # the wait before the last run of c: from the newest input end at or before its start
    function waited(rel, c,    k, j, r, L, Ls, ub, np, pl, p, q, m, e, emax, have_e) {
        k = cnt[rel, c]; L = 0
        for (j = 1; j <= k; j++) {
            r = at[rel, c, j]
            if (rres[r] != "not-run" && rst[r] != "" && (!L || secs(rst[r]) > secs(rst[L]))) L = r
        }
        if (!L) return "UNMEASURED"
        Ls = secs(rst[L])
        for (j = 1; j <= k; j++) {
            r = at[rel, c, j]
            if (rres[r] == "not-run" || rst[r] != "") continue
            ub = (ren[r] != "") ? ren[r] : seen(r)
            if (ub == "" || secs(ub) > Ls) return "UNMEASURED"
        }
        have_e = 0
        np = split(need[c], pl, " ")
        for (p = 1; p <= np; p++) for (q = 1; q <= cnt[rel, pl[p]]; q++) {
            m = at[rel, pl[p], q]
            if (rres[m] == "not-run") continue
            if (ren[m] == "") {
                if (seen(m) != "" && secs(seen(m)) >= Ls) continue
                if (rst[m] != "" && secs(rst[m]) > Ls) continue
                return "UNMEASURED"
            }
            e = secs(ren[m])
            if (e <= Ls && (!have_e || e > emax)) { emax = e; have_e = 1 }
        }
        return have_e ? mins(Ls - emax) : "UNMEASURED"
    }
    # one canonical step of one release: prints its row and leaves its first-pass value in FP
    function step(rel, c,    k, j, r, runs, rs, fno, funm, nd, med, runc, lo, hi, have, lb, e, l, span, wait) {
        k = cnt[rel, c] + 0
        if (k == 0) {
            FP = "UNMEASURED"
            printf "| %s | %s | 0 | no row | UNMEASURED | UNMEASURED | UNMEASURED | %s | - |\n", c, name[c], (c == "S01" ? "n/a" : "UNMEASURED")
            return
        }
        split("", rc); split("", dv)
        runs = fno = funm = nd = have = lb = 0
        for (j = 1; j <= k; j++) {
            r = at[rel, c, j]
            rc[(rres[r] ~ /^running@/) ? "running" : rres[r]]++
            if (rres[r] == "not-run") continue
            runs++
            if (rfp[r] == "no") fno = 1
            if (rfp[r] == "UNMEASURED") funm = 1
            if (rst[r] != "" && ren[r] != "") { nd++; dv[nd] = secs(ren[r]) - secs(rst[r]) }
            e = (rst[r] != "") ? rst[r] : ((ren[r] != "") ? ren[r] : seen(r))
            l = (ren[r] != "") ? ren[r] : ((seen(r) != "") ? seen(r) : rst[r])
            if (rst[r] == "" || ren[r] == "") lb = 1
            if (e == "") continue
            if (!have || secs(e) < lo) lo = secs(e)
            if (!have || secs(l) > hi) hi = secs(l)
            have = 1
        }
        FP = fno ? "no" : ((funm || runs == 0) ? "UNMEASURED" : "yes")
        rs = ""
        for (j = 1; j <= 6; j++) if (rc[rkind[j]]) rs = rs (rs == "" ? "" : ", ") rc[rkind[j]] " " rkind[j]
        runc = "UNMEASURED"
        if (nd) { med = median(dv, nd); runc = sprintf("%s (%s-%s; %d/%d)", mins(med), mins(dv[1]), mins(dv[nd]), nd, runs) }
        span = have ? ((lb ? ">=" : "") mins(hi - lo)) : "UNMEASURED"
        wait = (c == "S01") ? "n/a" : waited(rel, c)
        printf "| %s | %s | %d | %s | %s | %s | %s | %s | %s |\n", c, name[c], runs, rs, FP, runc, span, wait, decided(rel, c)
    }
    # the X rows of one release: a count line, and with --extras one table row each
    function extra(rel,    nx, xl, i, r, runs, meas, y, no, u, d) {
        nx = split(xs[rel], xl, " ")
        runs = meas = y = no = u = 0
        for (i = 1; i <= nx; i++) {
            r = xl[i]
            if (rres[r] == "not-run") continue
            runs++
            if (rst[r] != "" && ren[r] != "") meas++
            if (rfp[r] == "yes") y++
            else if (rfp[r] == "no") no++
            else u++
        }
        printf "STEPTABLE-X %s rows=%d runs=%d measured=%d yes=%d no=%d unmeasured=%d\n", rel, nx, runs, meas, y, no, u
        if (!(extras + 0) || !nx) return
        print ""
        print "| extra | name | result | first pass | run min | decided by |"
        print "|---|---|---|---|---|---|"
        for (i = 1; i <= nx; i++) {
            r = xl[i]
            d = (rst[r] != "" && ren[r] != "") ? mins(secs(ren[r]) - secs(rst[r])) : "UNMEASURED"
            printf "| %s | %s | %s | %s | %s | %s |\n", rid[r], cell(rname[r]), rres[r], rfp[r], d, cell(rdb[r])
        }
    }
    function release(rel,    i, c, y, no, u, sh) {
        printf "\n## Release %s\n\n", rel
        print "| step | name | runs | results | first pass | run min | span min | wait min | decided by |"
        print "|---|---|---|---|---|---|---|---|---|"
        y = no = u = 0
        for (i = 1; i <= ncanon; i++) {
            c = canon[i]
            step(rel, c)
            if (c == "S09c") continue
            fpat[rel, c] = FP
            if (FP == "yes") y++
            else if (FP == "no") no++
            else u++
        }
        sh = y "/" (y + no)
        printf "STEPTABLE %s first_pass yes=%d no=%d unmeasured=%d share=%s\n", rel, y, no, u, sh
        extra(rel)
    }
    # y per step over the releases, and their product; S09c is inside S09 and is not counted
    function rty(    i, j, c, v, y, no, u, yi, ys, prod, m, zero, unm) {
        print ""
        print "## Rolled first-pass yield"
        print ""
        print "| step | name | yes | no | unmeasured | y |"
        print "|---|---|---|---|---|---|"
        prod = 1; m = 0; zero = ""; unm = ""
        for (i = 1; i <= ncanon; i++) {
            c = canon[i]
            if (c == "S09c") continue
            y = no = u = 0
            for (j = 1; j <= nrel; j++) {
                v = fpat[rels[j], c]
                if (v == "yes") y++
                else if (v == "no") no++
                else u++
            }
            ys = "-"
            if (y + no) { yi = y / (y + no); prod *= yi; m++; ys = sprintf("%.3f", yi); if (!y) zero = zero " " c }
            else unm = unm " " c
            printf "| %s | %s | %d | %d | %d | %s |\n", c, name[c], y, no, u, ys
        }
        printf "RTY %s over %d of %d steps; y=0:%s; unmeasured:%s\n", (m ? sprintf("%.3f", prod) : "UNMEASURED"), m, ncanon - 1, (zero == "" ? " none" : zero), (unm == "" ? " none" : unm)
    }
    END {
        if (bad) exit 1
        if (!n) { print "REFUSED -: no step rows: a table of nothing proves nothing"; exit 1 }
        print "# Release step table"
        for (i = 1; i <= nrel; i++) release(rels[i])
        rty()
    }
    ' "$@"
}

# The question reader: per release, each question's wait, their sum and their union; or REFUSED.
questions_awk() {
    awk "$TIME_AWK"'
    BEGIN {
        FS = "\t"
        want = "release\tq_id\tasked_utc\tasked_src\ttopic\tanswered_utc\tanswered_src\truling\tnote"
        shown = want; gsub(/\t/, " ", shown); nf = split(want, hcol, "\t")
    }
    FNR == 1 {
        hbad = ($0 != want)
        if (hbad) { printf "REFUSED %s:1 -: the header is not the question header: %s\n", FILENAME, shown; bad++ }
        next
    }
    hbad { next }
    NF != nf { refuse("a question row has " nf " tab-separated fields, as the header does, this one has " NF); next }
    {
        if ($1 !~ /^[0-9A-Za-z.-]+$/) refuse("release " $1 " is not a release name [0-9A-Za-z.-]")
        if ($2 == "") refuse("q_id is empty")
        if ($5 == "") refuse("the topic is empty")
        aok = timecell($3, $4, "asked_utc")
        wok = timecell($6, $7, "answered_utc")
        if (aok && wok && $3 != "" && $6 != "" && secs($6) < secs($3)) refuse("answered_utc " $6 " is before asked_utc " $3)
        n++
        qid[n] = $2; qa[n] = $3; qw[n] = $6; qt[n] = $5
        if (!($1 in seenrel)) { seenrel[$1] = 1; rels[++nrel] = $1 }
        at[$1, ++cnt[$1]] = n
    }
    # one release: its questions, how long each waited, the sum and the union of the waits
    function asked(rel,    k, j, r, ans, noans, nd, lb, sum, w, i, ts, te, cs, ce, uni, med, medc, sumc, unic) {
        printf "\n## Questions %s\n\n", rel
        print "| question | asked | answered | wait min | topic |"
        print "|---|---|---|---|---|"
        split("", dv); split("", is_); split("", ie_)
        k = cnt[rel]; ans = nd = lb = sum = 0; noans = ""
        for (j = 1; j <= k; j++) {
            r = at[rel, j]
            if (qw[r] != "") ans++
            else noans = noans " " qid[r]
            w = "UNMEASURED"
            if (qa[r] != "" && qw[r] != "") {
                nd++; is_[nd] = secs(qa[r]); ie_[nd] = secs(qw[r]); dv[nd] = ie_[nd] - is_[nd]; sum += dv[nd]; w = mins(dv[nd])
            } else lb = 1
            printf "| %s | %s | %s | %s | %s |\n", cell(qid[r]), cell(qa[r]), cell(qw[r]), w, cell(qt[r])
        }
        medc = sumc = unic = "UNMEASURED"
        if (nd) {
            for (i = 2; i <= nd; i++) {
                ts = is_[i]; te = ie_[i]
                for (j = i - 1; j >= 1 && is_[j] > ts; j--) { is_[j + 1] = is_[j]; ie_[j + 1] = ie_[j] }
                is_[j + 1] = ts; ie_[j + 1] = te
            }
            uni = 0; cs = is_[1]; ce = ie_[1]
            for (i = 2; i <= nd; i++) {
                if (is_[i] <= ce) { if (ie_[i] > ce) ce = ie_[i] }
                else { uni += ce - cs; cs = is_[i]; ce = ie_[i] }
            }
            uni += ce - cs
            med = median(dv, nd)
            medc = mins(med) " wait_min=" mins(dv[1]) " wait_max=" mins(dv[nd])
            sumc = (lb ? ">=" : "") mins(sum)
            unic = (lb ? ">=" : "") mins(uni)
        }
        printf "STEPTABLE-Q %s asked=%d answered=%d no_answer_time=%d no_answer_ids=%s measured=%d wait_median=%s sum_min=%s union_min=%s\n", rel, k, ans, k - ans, (noans == "" ? "-" : substr(noans, 2)), nd, medc, sumc, unic
    }
    END {
        if (bad) exit 1
        if (!n) { print "REFUSED -: no question rows"; exit 1 }
        for (i = 1; i <= nrel; i++) asked(rels[i])
    }
    ' "$@"
}

main() {
    local -a steps=() questions=()
    local extras=0 f out_s out_q rc_s rc_q
    while [ $# -gt 0 ]; do
        case "$1" in
            --steps|--questions)
                [ $# -ge 2 ] || caller_error "$1 needs a file"
                f="$2"
                if [ ! -f "$f" ] || [ ! -r "$f" ]; then caller_error "$1 $f is not a readable file"; fi
                # awk reads an operand written name=value as an assignment and "-" as stdin: a relative path gets a "./".
                case "$f" in /*) : ;; *) f="./$f" ;; esac
                if [ "$1" = --steps ]; then steps+=("$f"); else questions+=("$f"); fi
                shift 2 ;;
            --extras) extras=1; shift ;;
            *) caller_error "unknown option '$1' (try -h)" ;;
        esac
    done
    [ "${#steps[@]}" -gt 0 ] || caller_error "no --steps file"
    rc_s=0; out_s="$(steps_awk "$extras" "${steps[@]}")" || rc_s=$?
    rc_q=0; out_q=''
    if [ "${#questions[@]}" -gt 0 ]; then out_q="$(questions_awk "${questions[@]}")" || rc_q=$?; fi
    if [ "$rc_s" -gt 1 ] || [ "$rc_q" -gt 1 ]; then caller_error "awk failed: steps exit $rc_s, questions exit $rc_q"; fi
    if [ "$rc_s" -eq 0 ] && [ "$rc_q" -eq 0 ]; then
        printf '%s\n' "$out_s"
        if [ -n "$out_q" ]; then printf '%s\n' "$out_q"; fi
        return 0
    fi
    # A refusal prints the REFUSED lines of every reader and no table at all.
    if [ "$rc_s" -ne 0 ]; then printf '%s\n' "$out_s"; fi
    if [ "$rc_q" -ne 0 ]; then printf '%s\n' "$out_q"; fi
    return 1
}

selftest_cleanup() { case "${tmp:-}" in ''|/) return 0 ;; *) rm -rf -- "${tmp:?}" ;; esac; }

selftest() {
    local tmp pass=0 fail=0 D=2030-01-01T G
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    # SEC011: validate before rm -rf. $tmp is this function's own mktemp -d (checked above).
    trap selftest_cleanup RETURN
    row() { # name expect-rc needle -- script-args ... ; a needle that starts with ! must NOT be said
        local name="$1" expect="$2" needle="$3" o rc=0
        shift 3
        if [ "${1:-}" = -- ]; then shift; fi
        o="$(bash "$SCRIPT_PATH" "$@" < /dev/null 2>&1)" || rc=$?
        if [ "$rc" != "$expect" ]; then printf '  BROKE %-60s expected exit %s got %s\n%s\n' "$name" "$expect" "$rc" "$o"; fail=$((fail + 1)); return 0; fi
        case "$needle" in
            '!'*)
                case "$o" in
                    *"${needle#!}"*) printf '  BROKE %-60s exit %s but said: %s\n%s\n' "$name" "$rc" "${needle#!}" "$o"; fail=$((fail + 1)) ;;
                    *) printf '  ok    %-60s exit=%s\n' "$name" "$rc"; pass=$((pass + 1)) ;;
                esac ;;
            *)
                case "$o" in
                    *"$needle"*) printf '  ok    %-60s exit=%s\n' "$name" "$rc"; pass=$((pass + 1)) ;;
                    *) printf '  BROKE %-60s exit %s but never said: %s\n%s\n' "$name" "$rc" "$needle" "$o"; fail=$((fail + 1)) ;;
                esac ;;
        esac
    }
    tsv() { local IFS=$'\t'; printf '%s\n' "$*"; }
    sh_() { tsv release step_id step lane start_utc start_src end_utc end_src result attempts attempts_src first_pass decided_by decided_src note; }
    qh_() { tsv release q_id asked_utc asked_src topic answered_utc answered_src ruling note; }
    srow() { # release id start end result first_pass decided_by: a well-formed row, every time sourced
        local ss='' es='' ds=''
        if [ -n "$3" ]; then ss=log; fi
        if [ -n "$4" ]; then es=log; fi
        if [ -n "$7" ]; then ds=log; fi
        tsv "$1" "$2" "name of $2" lane "$3" "$ss" "$4" "$es" "$5" 1 log "$6" "$7" "$ds" ''
    }
    one() { # stem, then the 15 cells of one row: the header and that row
        local f="$tmp/$1.tsv"
        shift
        { sh_; tsv "$@"; } > "$f"
    }

    echo "--- the tables (two synthetic releases, 2030) ---"
    G="$tmp/g.tsv"
    {
        sh_
        srow 9.9.0 S01 "${D}00:00:30Z" "${D}00:02:00Z" pass yes operator:C1
        srow 9.9.0 S02 "${D}00:10Z" "${D}00:40Z" fail no script:ci
        srow 9.9.0 S02 "${D}01:00Z" "${D}01:20Z" pass no script:ci
        srow 9.9.0 S03 "${D}01:30Z" "${D}01:40Z" pass yes cop
        srow 9.9.0 S04 '' "${D}00:50Z" pass yes worker:w1
        srow 9.9.0 S04 "${D}00:20Z" "${D}00:45Z" pass yes operator:C2
        srow 9.9.0 S08 "${D}00:05Z" '' fail no ''
        srow 9.9.0 S08 "${D}00:30Z" "${D}00:35Z" pass yes ''
        srow 9.9.0 X01 "${D}00:00Z" "${D}00:30Z" refused yes 'script:a|b'
        srow 9.9.1 S01 "${D}02:00Z" "${D}02:01Z" pass yes operator:C3
        srow 9.9.1 S02 "${D}02:05Z" "${D}02:35Z" pass yes script:ci
        srow 9.9.1 S03 '' '' not-run UNMEASURED ''
        srow 9.9.1 S08 "${D}02:10Z" "${D}02:20Z" pass yes ''
    } > "$G"
    row 'seconds count: 00:00:30 to 00:02:00 is 1.5 min' 0 '| S01 | version bump | 1 | 1 pass | yes | 1.5 (1.5-1.5; 1/1) | 1.5 | n/a | operator: C1 |' -- --steps "$G"
    row 'two runs: median, range, span, wait from the input end' 0 '| S02 | PR CI on the release commit | 2 | 1 pass, 1 fail | no | 25.0 (20.0-30.0; 2/2) | 70.0 | 58.0 | script: ci |' -- --steps "$G"
    row 'the wait runs from the NEWEST input end before the start' 0 '| S03 | merge queue / merge | 1 | 1 pass | yes | 10.0 (10.0-10.0; 1/1) | 10.0 | 10.0 | cop |' -- --steps "$G"
    row 'a run with no start: span is >=, wait UNMEASURED' 0 '| S04 | clean-room job | 2 | 2 pass | yes | 25.0 (25.0-25.0; 1/2) | >=30.0 | UNMEASURED | worker: w1; operator: C2 |' -- --steps "$G"
    row 'a step with no row is UNMEASURED, never 0' 0 '| S05 | model ladder, GPU host 1 | 0 | no row | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | - |' -- --steps "$G"
    row 'one no makes the step no; a missing end is no duration' 0 '| S08 | CPU-host gates | 2 | 1 pass, 1 fail | no | 5.0 (5.0-5.0; 1/2) | >=30.0 | 28.0 | - |' -- --steps "$G"
    row 'share = yes/(yes+no); UNMEASURED is not counted' 0 'STEPTABLE 9.9.0 first_pass yes=3 no=2 unmeasured=8 share=3/5' -- --steps "$G"
    row 'a not-run step is UNMEASURED' 0 '| S03 | merge queue / merge | 0 | 1 not-run | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | - |' -- --steps "$G"
    row 'the second release has its own share' 0 'STEPTABLE 9.9.1 first_pass yes=3 no=0 unmeasured=10 share=3/3' -- --steps "$G"
    row 'y of a step is over the releases' 0 '| S02 | PR CI on the release commit | 1 | 1 | 0 | 0.500 |' -- --steps "$G"
    row 'RTY is the product of y over measured steps' 0 'RTY 0.250 over 5 of 13 steps; y=0: none; unmeasured: S05 S06 S07 S09 S10 S11 S12 S13' -- --steps "$G"
    row 'the extras are counted' 0 'STEPTABLE-X 9.9.0 rows=1 runs=1 measured=1 yes=1 no=0 unmeasured=0' -- --steps "$G"
    row 'no extras table without --extras' 0 '!| extra |' -- --steps "$G"
    row 'with --extras, one row each; a | in a cell becomes /' 0 '| X01 | name of X01 | refused | yes | 30.0 | script:a/b |' -- --steps "$G" --extras

    echo "--- unknown ends, running rows and the wait ---"
    {
        sh_
        srow 9.9.2 S01 '' '' pass yes operator:C9
        srow 9.9.2 S09 "${D}03:00Z" '' "running@${D}03:45Z" UNMEASURED ''
    } > "$tmp/k0.tsv"
    row 'a run with no times: run and span UNMEASURED, never 0' 0 '| S01 | version bump | 1 | 1 pass | yes | UNMEASURED | UNMEASURED | n/a | operator: C9 |' -- --steps "$tmp/k0.tsv"
    row 'running@: the span is a lower bound and nothing ran to an end' 0 '| S09 | dogfood | 1 | 1 running | UNMEASURED | UNMEASURED | >=45.0 | UNMEASURED | - |' -- --steps "$tmp/k0.tsv"
    {
        sh_
        srow 9.9.3 S01 "${D}00:00Z" "${D}00:01Z" pass yes operator:C1
        srow 9.9.3 S05 "${D}00:05Z" '' stopped no ''
        srow 9.9.3 S06 "${D}00:02Z" "${D}00:30Z" pass yes ''
        srow 9.9.3 S07 "${D}00:02Z" '' "running@${D}02:00Z" UNMEASURED ''
        srow 9.9.3 S09 "${D}01:00Z" "${D}01:30Z" pass yes ''
    } > "$tmp/w1.tsv"
    row 'an input run with an unknown end makes the wait UNMEASURED' 0 '| S09 | dogfood | 1 | 1 pass | yes | 30.0 (30.0-30.0; 1/1) | 30.0 | UNMEASURED | - |' -- --steps "$tmp/w1.tsv"
    {
        sh_
        srow 9.9.3 S01 "${D}00:00Z" "${D}00:01Z" pass yes operator:C1
        srow 9.9.3 S05 "${D}00:05Z" "${D}00:40Z" pass yes ''
        srow 9.9.3 S06 "${D}00:02Z" "${D}00:30Z" pass yes ''
        srow 9.9.3 S07 "${D}00:02Z" '' "running@${D}02:00Z" UNMEASURED ''
        srow 9.9.3 S09 "${D}01:00Z" "${D}01:30Z" pass yes ''
    } > "$tmp/w2.tsv"
    row 'an input still running at the start is not an end before it' 0 '| S09 | dogfood | 1 | 1 pass | yes | 30.0 (30.0-30.0; 1/1) | 30.0 | 20.0 | - |' -- --steps "$tmp/w2.tsv"
    { sh_; srow 9.9.4 S01 2028-02-28T23:00Z 2028-03-01T01:00Z pass yes operator:C1; } > "$tmp/leap.tsv"
    row 'a leap day counts: 02-28T23:00 to 03-01T01:00 in 2028 is 26 h' 0 '| S01 | version bump | 1 | 1 pass | yes | 1560.0 (1560.0-1560.0; 1/1) | 1560.0 | n/a | operator: C1 |' -- --steps "$tmp/leap.tsv"

    echo "--- refused: an estimate never enters a table ---"
    one est 9.9.0 S01 n lane '~2030-01-01T00:00Z' log '' '' pass 1 log yes '' '' ''
    row 'R0 FALSIFIER: an estimated time ~ is REFUSED' 1 'est.tsv:2 S01: start_utc ~2030-01-01T00:00Z is not a UTC instant' -- --steps "$tmp/est.tsv"
    row 'R0 FALSIFIER: a refused input prints no table' 1 '!| step |' -- --steps "$tmp/est.tsv"
    row 'one bad file withholds every table, the good ones too' 1 '!STEPTABLE ' -- --steps "$G" --steps "$tmp/est.tsv"
    one eta 9.9.0 S02 n lane "${D}00:00Z" log 'ETA 2030-01-01T01:00Z' log pass 1 log yes '' '' ''
    row 'an ETA is not a time' 1 'end_utc ETA 2030-01-01T01:00Z is not a UTC instant' -- --steps "$tmp/eta.tsv"
    one by 9.9.0 S02 n lane 'by 2030-01-01T01:00Z' log '' '' pass 1 log yes '' '' ''
    row 'a by-time is not a time' 1 'start_utc by 2030-01-01T01:00Z is not a UTC instant' -- --steps "$tmp/by.tsv"
    one nosrc 9.9.0 S02 n lane "${D}00:00Z" '' '' '' pass 1 log yes '' '' ''
    row 'a time with no source is refused' 1 'start_utc 2030-01-01T00:00Z has no source' -- --steps "$tmp/nosrc.tsv"
    one back 9.9.0 S02 n lane "${D}01:00Z" log "${D}00:30Z" log pass 1 log yes '' '' ''
    row 'an end before its start is refused' 1 'end_utc 2030-01-01T00:30Z is before start_utc' -- --steps "$tmp/back.tsv"
    one feb29 9.9.0 S02 n lane 2030-02-29T00:00Z log '' '' pass 1 log no '' '' ''
    row 'a date that does not exist is refused (2030 has no leap day)' 1 'start_utc 2030-02-29T00:00Z is not a UTC instant' -- --steps "$tmp/feb29.tsv"
    one h24 9.9.0 S02 n lane 2030-01-01T24:00Z log '' '' pass 1 log no '' '' ''
    row 'hour 24 is refused' 1 'start_utc 2030-01-01T24:00Z is not a UTC instant' -- --steps "$tmp/h24.tsv"
    one res 9.9.0 S02 n lane '' '' '' '' passed 1 log yes '' '' ''
    row 'an unknown result is refused' 1 'result passed is not pass' -- --steps "$tmp/res.tsv"
    one runend 9.9.0 S02 n lane "${D}00:00Z" log "${D}00:30Z" log "running@${D}00:20Z" 1 log no '' '' ''
    row 'a running row with an end is refused' 1 'has an end_utc: a run that ended is not running' -- --steps "$tmp/runend.tsv"
    one runback 9.9.0 S02 n lane "${D}01:00Z" log '' '' "running@${D}00:30Z" 1 log no '' '' ''
    row 'running@ before the start is refused' 1 'result running@2030-01-01T00:30Z is before start_utc' -- --steps "$tmp/runback.tsv"
    one notrun 9.9.0 S02 n lane "${D}01:00Z" log '' '' not-run 1 log no '' '' ''
    row 'a not-run row with a time is refused' 1 'result not-run has a time' -- --steps "$tmp/notrun.tsv"
    one fp 9.9.0 S02 n lane '' '' '' '' pass 1 log maybe '' '' ''
    row 'first_pass is yes, no or UNMEASURED' 1 'first_pass maybe is not yes, no or UNMEASURED' -- --steps "$tmp/fp.tsv"
    one who 9.9.0 S02 n lane '' '' '' '' pass 1 log no boss log ''
    row 'decided_by names its kind' 1 'decided_by boss is not operator:C<n>' -- --steps "$tmp/who.tsv"
    one whosrc 9.9.0 S02 n lane '' '' '' '' pass 1 log no cop '' ''
    row 'decided_by needs a source' 1 'decided_by cop has no decided_src' -- --steps "$tmp/whosrc.tsv"
    one s14 9.9.0 S14 n lane '' '' '' '' pass 1 log no '' '' ''
    row 'a step id outside the list is refused, never dropped' 1 'step_id S14 is neither' -- --steps "$tmp/s14.tsv"
    { sh_; tsv 9.9.0 S02 n lane '' '' '' '' pass 1 log no '' ''; } > "$tmp/short.tsv"
    row 'a short row is refused' 1 'this one has 14' -- --steps "$tmp/short.tsv"
    { tsv release step_id step; tsv 9.9.0 S01 n; } > "$tmp/hdr.tsv"
    row 'a wrong header is refused' 1 'hdr.tsv:1 -: the header is not the step header' -- --steps "$tmp/hdr.tsv"
    {
        sh_
        tsv 9.9.0 S01 n lane '~x' log '' '' pass 1 log yes '' '' ''
        tsv 9.9.0 S02 n lane '' '' '' '' passed 1 log no '' '' ''
    } > "$tmp/two.tsv"
    row 'every bad row is reported, not only the first' 1 'two.tsv:3 S02: result passed' -- --steps "$tmp/two.tsv"
    sh_ > "$tmp/empty.tsv"
    row 'a header with no rows is refused' 1 'no step rows' -- --steps "$tmp/empty.tsv"

    echo "--- questions ---"
    {
        qh_
        tsv 9.9.0 A1 "${D}00:00Z" log 'topic one' "${D}00:30Z" log C1 ''
        tsv 9.9.0 A2 "${D}00:20Z" log 'topic two' "${D}01:00Z" log '' ''
        tsv 9.9.0 A3 '' '' 'topic three' '' '' '' ''
        tsv 9.9.0 A4 '' '' 'topic four' "${D}02:00Z" log C2 ''
    } > "$tmp/q.tsv"
    row 'overlapping waits count once in the union; missing times make >=' 0 'STEPTABLE-Q 9.9.0 asked=4 answered=3 no_answer_time=1 no_answer_ids=A3 measured=2 wait_median=35.0 wait_min=30.0 wait_max=40.0 sum_min=>=70.0 union_min=>=60.0' -- --steps "$G" --questions "$tmp/q.tsv"
    { qh_; tsv 9.9.0 B1 "${D}01:00Z" log t "${D}00:30Z" log '' ''; } > "$tmp/qback.tsv"
    row 'an answer before its question is refused' 1 'answered_utc 2030-01-01T00:30Z is before asked_utc' -- --steps "$G" --questions "$tmp/qback.tsv"
    { qh_; tsv 9.9.0 B2 '~2030-01-01T01:00Z' log t '' '' '' ''; } > "$tmp/qest.tsv"
    row 'an estimated question time is refused' 1 'asked_utc ~2030-01-01T01:00Z is not a UTC instant' -- --steps "$G" --questions "$tmp/qest.tsv"
    row 'a refused question file withholds the step tables too' 1 '!| step |' -- --steps "$G" --questions "$tmp/qest.tsv"

    echo "--- caller errors ---"
    row 'no --steps' 3 'no --steps file' --
    row '--steps with no file' 3 'needs a file' -- --steps
    row 'a file that does not exist' 3 'is not a readable file' -- --steps "$tmp/none.tsv"
    row 'an unknown option' 3 'unknown option' -- --bogus

    printf -- '--- %s/%s rows ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

# Planted defects: each must turn a row of the case table RED, or the case table proves nothing.
mutants() {
    local tmp pass=0 fail=0 name expr copy o rc
    tmp="$(mktemp -d)"
    case "$tmp" in ?*/tmp.?*) : ;; *) echo "mktemp gave '${tmp:-}', refusing"; return 2 ;; esac
    # SEC011: validate before rm -rf. $tmp is this function's own mktemp -d (checked above).
    trap selftest_cleanup RETURN
    while read -r name expr; do
        [ -n "$name" ] || continue
        copy="$tmp/$name.sh"
        sed -e "$expr" "$SCRIPT_PATH" > "$copy"
        if cmp -s "$SCRIPT_PATH" "$copy"; then
            printf '  BROKE %-34s changed nothing: its pattern no longer matches\n' "$name"; fail=$((fail + 1)); continue
        fi
        if ! bash -n "$copy" 2>/dev/null; then
            printf '  BROKE %-34s does not parse: a RED from it would prove nothing\n' "$name"; fail=$((fail + 1)); continue
        fi
        rc=0; o="$(bash "$copy" --selftest < /dev/null 2>&1)" || rc=$?
        case "$rc:$o" in
            0:*) printf '  BROKE %-34s SURVIVED: the case table stayed green\n' "$name"; fail=$((fail + 1)) ;;
            *"syntax error"*|*"awk: "*) printf '  BROKE %-34s the mutant awk does not run: a RED from it proves nothing\n' "$name"; fail=$((fail + 1)) ;;
            *"  BROKE "*) printf '  ok    %-34s killed, %s row(s) broke\n' "$name" "$(printf '%s\n' "$o" | awk '/^  BROKE /{n++} END{print n+0}')"; pass=$((pass + 1)) ;;
            *) printf '  BROKE %-34s exit %s with no broken row: not a kill\n%s\n' "$name" "$rc" "$o"; fail=$((fail + 1)) ;;
        esac
    done <<'MUTANTS'
istime_accepts_anything     /^TIME_AWK='$/,/^'$/s/^    function istime(s,    n) {$/    function istime(s,    n) { return (s != "")/
source_not_required         /^TIME_AWK='$/,/^'$/s/if (src == "") {/if (0) {/
seconds_ignored             /^TIME_AWK='$/,/^'$/s/ + ((length(s) == length("1970-01-01T00:00:00Z")) ? substr(s, 18, 2) : 0) }/ + 0 }/
leap_years_ignored          /^TIME_AWK='$/,/^'$/s/doe = yoe \* 365 + int(yoe \/ 4) - int(yoe \/ 100) + doy/doe = yoe * 365 + doy/
end_before_start_accepted   /^steps_awk() {$/,/^}$/s/secs($7) < secs($5)/0/
notrun_with_time_accepted   /^steps_awk() {$/,/^}$/s/if ($5 != "" || $7 != "") refuse/if (0) refuse/
missing_end_read_as_zero    /^steps_awk() {$/,/^}$/s/if (rst\[r\] != "" && ren\[r\] != "") { nd++/if (rst[r] != "") { nd++/
first_pass_from_last_run    /^steps_awk() {$/,/^}$/s/if (rfp\[r\] == "no") fno = 1/fno = (rfp[r] == "no")/
span_not_a_lower_bound      /^steps_awk() {$/,/^}$/s/(lb ? ">=" : "") mins(hi - lo)/mins(hi - lo)/
nostart_run_ignored         /^steps_awk() {$/,/^}$/s/if (ub == "" || secs(ub) > Ls) return "UNMEASURED"/if (0) return "UNMEASURED"/
input_unknown_end_ignored   /^steps_awk() {$/,/^}$/s/^                return "UNMEASURED"$/                continue/
wait_from_oldest_input_end  /^steps_awk() {$/,/^}$/s/(!have_e || e > emax)/(!have_e || e < emax)/
share_counts_unmeasured     /^steps_awk() {$/,/^}$/s/sh = y "\/" (y + no)$/sh = y "\/" (y + no + u)/
rty_is_a_sum                /^steps_awk() {$/,/^}$/s/prod \*= yi/prod += yi/
union_is_the_sum            /^questions_awk() {$/,/^}$/s/if (is_\[i\] <= ce)/if (0)/
question_sum_not_lower_bound /^questions_awk() {$/,/^}$/s/sumc = (lb ? ">=" : "")/sumc = ("")/
MUTANTS
    printf -- '--- %s/%s mutants killed ---\n' "$pass" "$((pass + fail))"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --selftest) selftest ;;
    --mutants) mutants ;;
    -h|--help) awk 'NR > 1 && /^set -uo pipefail$/ { exit } NR > 1' "$0" ;;
    *) main "$@" ;;
esac
