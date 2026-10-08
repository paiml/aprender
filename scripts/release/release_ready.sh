#!/usr/bin/env bash
# release_ready.sh — one list in code of what merge, tag and publish require (BLD-001 row 4, BLD-002 R9, #4688).
#
# THE LIST is contracts/release-ready-v1.yaml, block `requirements:`, one requirement per line:
#   - {id: RR-…, applies_to: [merge|tag|publish, …], anchor: "…", checker: "path[#needle]", provenance: "…",
#      producer: "<nightly_train lane>"[, key: "…", value: "…"]}
#   `producer` is required when applies_to holds publish (BLD-001 row 10). `key`/`value` state one fact the
#   requirement fixes (a mode, a threshold, a set), so two places that say different things are seen.
#   `seven: <key>` marks the GATE that carries one of the seven entries a release always has (SEVEN below).
#
# THE SURFACES (fixed here, never an option: a surface set a caller can shrink is theater). Each extractor reads
# one file of the tree and prints the requirements that file states, as anchors:
#   merge    .github/workflows/ci.yml           job `gate`: every `needs:` job and every `for pair in` section
#   merge    CLAUDE.md                          the "Required status checks:" contexts
#   tag      scripts/release/autopilot.sh       every checker a step up to and including `tag` invokes
#   tag      scripts/release/autopilot.sh       the T-1 lanes (`for s in …; do run_step "$s"`): each lane's `t1_<step>`
#                                               body, under its own step name
#   tag      scripts/dogfood.sh + Cargo.toml    every row the pre-publish dogfood can FAIL: `mark X FAIL`, `gate X`,
#                                               `ci_mark X`, a `pre-publish:…) printf 'X FAIL`, and every
#                                               [package.metadata.dogfood].gates script (`declared:X`). A row inside a
#                                               branch that only runs post-publish cannot stop a tag and is not read;
#                                               nor are comments, heredoc bodies and `-c '…'` programs
#   publish  scripts/release/autopilot.sh       every checker a step after `tag` up to `dryrun` invokes
#   publish  scripts/check_publish_preflight.sh every rule R<n> of its RULES header
#   publish  scripts/cascade-publish.sh         every `if ! <gate>` the cascade refuses on, and every top-level
#                                               `if [ … ]` block before the upload loop (first top-level `for …; do`)
#                                               that exits 1..9
#   publish  scripts/release/publish_strict.sh  every `|| die "…"` / `&& die "…"` refusal before its first `cargo
#                                               publish` (the spec's T-4 executor), named by its message
#   Out: a verdict that runs after an upload (a final verification, autopilot's `cascade incomplete`) is an
#   outcome of publishing, not a requirement to start it.
#   A checker inside a step is: `bash scripts/X.sh [--flag]`, `python3 scripts/X.py [sub]`, `cargo test|check|build|metadata`
#   with its first two flags, a workflow it dispatches or reads (`gh-workflow X.yml`), `git merge-base
#   --is-ancestor`, and the same inside any top-level function the step calls. A quoted literal (a die message)
#   is not a call.
#   Claims (key = value, for contradictions): the R8 wrapper's committed mode (release_readiness.sh DEFAULT_MODE
#   vs the preflight's RULES text), autopilot's step list (STEPS=() vs its header), the coverage floor (Makefile
#   vs CLAUDE.md), the publish executor (the first script autopilot's `cascade` step runs vs the script the
#   spec's T-4 row names), and every key/value a list entry states.
#
# THE VERDICT fails on three things, and prints one row per finding:
#   UNLISTED       a surface states a requirement (anchor) that no list entry carries
#   ORPHANED       a list entry whose checker file is absent, whose #needle is not in it, or whose anchor no
#                  surface states any more
#   CONTRADICTION  one key carries two or more values across the claims and the list
#   then one line: `release-ready: unlisted=U orphaned=O contradictions=C tree=<sha|nogit>`.
# A row of a GATE (`parent: <id>`, e.g. one dogfood row under the dogfood step) is on the list and is checked like
# any entry; `--gates` prints it under its GATE, and the total `gates=N` counts GATEs only: one grain.
# A row of a GATE (`parent: <id>`, e.g. one dogfood row under the dogfood step) is on the list and is checked like
# any entry; `--gates` prints it under its GATE and the total `gates=N` counts GATEs only, one grain.
#
# PROBE (--probe). TWO GitHub reads, because both enforce on main: the ruleset rules on its branch
#   (`repos/R/rules/branches/main`) and the classic branch protection (`repos/R/branches/main/protection/
#   required_status_checks`). The union of their required contexts is compared with the list's
#   `merge:required-context/*` anchors. Read-only:
#   changing the required checks is the operator's.
#
# EXIT  0 zero findings (probe: sets equal) · 1 a finding (probe: sets differ) · 2 not_measured (a surface is
#       missing or unreadable; the probe read failed) · 3 caller error (bad arguments, a malformed list line)
# LAB today: no gate calls this script (it becomes a GATE only after three green nights and an operator yes).
#
# USAGE
#   release_ready.sh [--root DIR] [--list FILE]     the verdict on a tree (default: this repo, its list)
#   release_ready.sh --found [--root DIR]           the anchors the surfaces state (the in-repo inventory)
#   release_ready.sh --gates [--root DIR] [--list FILE]   the GATE list: every requirement, one row each
#   release_ready.sh --probe [--root DIR] [--list FILE]   GH (default gh) · RR_REPO (default paiml/aprender)
#   release_ready.sh --budget [--root DIR] [--list FILE]  GATEs per stage vs the caps merge=10 tag=5 publish=5,
#                                                         and every one of SEVEN carried (rc 1: OVER or MISSING)
#   release_ready.sh --selftest                     the case table (fixture trees, no network)
#   release_ready.sh --mutants                      each planted mutant must turn the case table RED
set -uo pipefail

SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/${BASH_SOURCE[0]##*/}"
DEFAULT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LIST_REL=contracts/release-ready-v1.yaml
SEVEN="cleanroom-tag-commit workspace-tests crux-smoke supply-chain provenance binaries-before-public no-secret-in-crates"
CI_REL=.github/workflows/ci.yml
DOC_REL=CLAUDE.md
AUTO_REL=scripts/release/autopilot.sh
PRE_REL=scripts/check_publish_preflight.sh
CAS_REL=scripts/cascade-publish.sh
STR_REL=scripts/release/publish_strict.sh
SPEC_REL=docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md
RRW_REL=scripts/release/release_readiness.sh
MK_REL=Makefile
DF_REL=scripts/dogfood.sh
CARGO_REL=Cargo.toml

caller_error() { printf 'release-ready: caller error: %s\n' "$*"; exit 3; }
nm() { printf 'release-ready: not_measured: %s\n' "$*"; exit 2; }

# ---------------------------------------------------------------- extractors (file in, "anchor<TAB>where" out) ----
# ci_gate_found FILE: job `gate`'s needs (flow form only; any other form is unreadable = rc 2) and its sections
ci_gate_found() {
    awk -v F="$2" '
        /^  gate:/ { g = 1; next }
        g && /^  [A-Za-z0-9_-]+:/ { g = 0 }
        g && /^    needs:/ {
            s = $0; if (s !~ /\[.*\]/) { bad = 1; next }
            sub(/^[^[]*\[/, "", s); sub(/\].*/, "", s); n = split(s, a, /[ ,]+/)
            for (i = 1; i <= n; i++) if (a[i] != "") { printf "merge:ci.yml/gate/needs/%s\t%s:%d\n", a[i], F, FNR; k++; kn++ }
        }
        g && /for pair in / {
            s = $0; sub(/.*for pair in /, "", s); sub(/;.*/, "", s); n = split(s, a, / +/)
            for (i = 1; i <= n; i++) if (a[i] != "") { x = a[i]; sub(/^[^:]*:/, "", x); printf "merge:ci.yml/gate/section/%s\t%s:%d\n", x, F, FNR; k++; kp++ }
        }
        END { if (bad || kn == 0 || kp == 0) exit 2 }
    ' "$1"
}

# doc_contexts_found FILE: the "Required status checks:" line's back-quoted contexts
doc_contexts_found() {
    awk -v F="$2" '
        /Required status checks:/ {
            s = $0; sub(/.*Required status checks:/, "", s); sub(/\. .*/, "", s)
            while (match(s, /`[^`]+`/)) { printf "merge:required-context/%s\t%s:%d\n", substr(s, RSTART + 1, RLENGTH - 2), F, FNR; s = substr(s, RSTART + RLENGTH); k++ }
        }
        END { if (k == 0) exit 2 }
    ' "$1"
}

# autopilot_found FILE: per step up to dryrun, the checkers it (and the functions it calls) invokes.
# A line that names more scripts/*.sh|*.py than the call forms below parse (`bash "$R"/scripts/x.sh`,
# `python3 "$R/scripts/x.py"`, `./scripts/x.sh`) is not_measured, never skipped: a call form this parser
# cannot read would otherwise never show as UNLISTED.
autopilot_found() {
    awk -v F="$2" '
        { L[NR] = $0 }
        function invs(line, out,   s, t, w, n, i, f, c, nr, np) {
            out = ""
            if (line ~ /^[ \t]*#/) return ""
            s = line
            while (match(s, /scripts\/[A-Za-z0-9_\/.-]+\.(sh|py)/)) { nr++; s = substr(s, RSTART + RLENGTH) }
            s = line
            while (match(s, /bash +"?(\$\{?[A-Za-z_]+\}?\/|\.\/)?scripts\/[A-Za-z0-9_\/.-]+\.sh"?( +--[a-z][a-z-]*)?/)) {
                w = substr(s, RSTART, RLENGTH); s = substr(s, RSTART + RLENGTH); np++
                gsub(/"/, "", w); sub(/\$\{?[A-Za-z_]+\}?\//, "", w); sub(/ \.\//, " ", w); gsub(/ +/, " ", w); out = out "\n" w
            }
            s = line
            while (match(s, /python3 +scripts\/[A-Za-z0-9_\/.-]+\.py( +[a-z][a-z-]*)?/)) {
                w = substr(s, RSTART, RLENGTH); s = substr(s, RSTART + RLENGTH); np++; gsub(/ +/, " ", w); out = out "\n" w
            }
            if (nr > np) UNP = 1
            # a quoted literal (a die message) is not a call: pair the quotes left to right, keep a `$(` inside one
            s = ""; t = line
            while (match(t, /"[^"]*"/)) { w = substr(t, RSTART, RLENGTH); s = s substr(t, 1, RSTART - 1) (index(w, "$(") ? w : "\"\""); t = substr(t, RSTART + RLENGTH) }
            s = s t
            while (match(s, /cargo +(test|check|build|metadata)( +[^ ;|&>)]+)*/)) {
                w = substr(s, RSTART, RLENGTH); s = substr(s, RSTART + RLENGTH)
                n = split(w, a, / +/); c = a[1] " " a[2]; f = 0
                for (i = 3; i <= n && f < 2; i++) if (a[i] ~ /^--/) { c = c " " a[i]; f++ }
                out = out "\n" c
            }
            s = line
            while (match(s, /(--workflow|workflow run) +[A-Za-z0-9_.-]+\.yml/)) {
                w = substr(s, RSTART, RLENGTH); s = substr(s, RSTART + RLENGTH); sub(/.* /, "", w); out = out "\ngh-workflow " w
            }
            if (line ~ /git merge-base --is-ancestor/) out = out "\ngit merge-base --is-ancestor"
            return out
        }
        function unp(ln) { if (UNP) { UL[++nu] = F ":" ln; UNP = 0 } }
        END {
            for (i = 1; i <= NR; i++) if (L[i] ~ /^STEPS=\(/) { s = L[i]; sub(/^STEPS=\(/, "", s); sub(/\).*/, "", s); ns = split(s, ST, / +/) }
            if (!ns) exit 2
            ph = "tag"; stop = 0
            for (j = 1; j <= ns; j++) { PH[ST[j]] = stop ? "" : ph; if (ST[j] == "tag") ph = "publish"; if (ST[j] == "dryrun") stop = 1 }
            nf = 0
            for (i = 1; i <= NR; i++) if (L[i] ~ /^[A-Za-z_][A-Za-z0-9_]*\(\) *\{/ && L[i] !~ /\}[ \t]*$/) {
                fn = L[i]; sub(/\(.*/, "", fn); FS0[fn] = i
                for (e = i + 1; e <= NR && L[e] !~ /^\}/; e++) ; FE[fn] = e; FN[++nf] = fn
            }
            for (i = 1; i <= NR; i++) {
                if (L[i] !~ /^if run_step [a-z]+; then/) continue
                st = L[i]; sub(/^if run_step /, "", st); sub(/;.*/, "", st)
                if (!(st in PH)) { unk = 1; continue }
                if (PH[st] == "") continue
                delete seen
                for (e = i + 1; e <= NR && L[e] !~ /^fi/; e++) {
                    o = invs(L[e]); unp(e); m = split(o, A, "\n")
                    for (q = 2; q <= m; q++) if (!(A[q] in seen)) { seen[A[q]] = 1; printf "%s:autopilot/%s/%s\t%s:%d\n", PH[st], st, A[q], F, e; k++ }
                    for (r = 1; r <= nf; r++) if (L[e] !~ /^[ \t]*#/ && match(L[e], "(^|[^A-Za-z0-9_])" FN[r] "([^A-Za-z0-9_]|$)"))
                        for (b = FS0[FN[r]] + 1; b < FE[FN[r]]; b++) {
                            o = invs(L[b]); unp(b); m = split(o, A, "\n")
                            for (q = 2; q <= m; q++) if (!(A[q] in seen)) { seen[A[q]] = 1; printf "%s:autopilot/%s/%s\t%s:%d\n", PH[st], st, A[q], F, b; k++ }
                        }
                }
            }
            # parallel lanes: `for s in a b c; do run_step "$s"` launched as `"t1_$s" &`; each step is its t1_<step> body
            for (i = 1; i <= NR; i++) {
                if (L[i] !~ /for s in [a-z ]+; [d]o run_step "\$s"/) continue
                ls = L[i]; sub(/.*for s in /, "", ls); sub(/;.*/, "", ls); nl = split(ls, LN, / +/)
                for (j = 1; j <= nl; j++) {
                    st = LN[j]; fn = "t1_" st
                    if (!(st in PH) || !(fn in FS0)) { unk = 1; continue }
                    if (PH[st] == "") continue
                    delete seen
                    for (e = FS0[fn] + 1; e < FE[fn]; e++) {
                        o = invs(L[e]); unp(e); m = split(o, A, "\n")
                        for (q = 2; q <= m; q++) if (!(A[q] in seen)) { seen[A[q]] = 1; printf "%s:autopilot/%s/%s\t%s:%d\n", PH[st], st, A[q], F, e; k++ }
                    }
                }
            }
            # the top-level lines between steps (outside every step and function) run on every train: step `commit`
            for (i = 1; i <= NR; i++) {
                if (L[i] ~ /^if run_step [a-z]+; then/) { st = L[i]; sub(/^if run_step /, "", st); sub(/;.*/, "", st)
                    if (!first) first = i; if (PH[st] == "" && !last) last = i
                    for (e = i + 1; e <= NR && L[e] !~ /^fi/; e++) IN[e] = 1; IN[i] = 1; IN[e] = 1 }
            }
            for (r = 1; r <= nf; r++) for (b = FS0[FN[r]]; b <= FE[FN[r]]; b++) IN[b] = 1
            if (!last) last = NR + 1
            delete seen
            for (i = first + 1; i < last; i++) if (!(i in IN)) {
                o = invs(L[i]); unp(i); m = split(o, A, "\n")
                for (q = 2; q <= m; q++) if (!(A[q] in seen)) { seen[A[q]] = 1; printf "tag:autopilot/commit/%s\t%s:%d\n", A[q], F, i; k++ }
            }
            for (i = 1; i <= nu; i++) printf "unparsed call form at %s\n", UL[i] > "/dev/stderr"
            if (nu || unk || k == 0) exit 2
        }
    ' "$1"
}

# preflight_found FILE: the rules of the RULES header
preflight_found() {
    awk -v F="$2" '
        /^# RULES/ { r = 1; next }
        r && /^#[ \t]*$/ { r = 0 }
        r && /^#   R[0-9]+  / { x = $2; printf "publish:preflight/%s\t%s:%d\n", x, F, FNR; k++ }
        END { if (k == 0) exit 2 }
    ' "$1"
}

# cascade_found FILE: every `if ! <function|bash script>` the cascade refuses on. A bare word counts only when the
# file defines it as a function, so `if ! git …`, `if ! grep …` and `if ! gh …` (reads, not gates) are not anchors.
# Also every top-level `if [ … ]` block before the upload loop (the first top-level `for …; do`) that exits 1..9, named
# by its condition (quotes dropped). No upload loop, or a block that never closes, is not_measured.
cascade_found() {
    awk -v F="$2" '
        NR == FNR { if ($0 ~ /^[A-Za-z_][A-Za-z0-9_]*\(\)/) { f = $0; sub(/\(.*/, "", f); DEF[f] = 1 }; next }
        /^[ \t]*#/ { next }
        /^for .*; do[ \t]*$/ { if (inb) unc = 1; loop = 1 }
        /^if / && inb { unc = 1 }
        !loop && /^if \[/ {
            blk = $0; sub(/;[ \t]*then.*/, "", blk); gsub(/"/, "", blk); bl = FNR; ex = ($0 ~ /exit [1-9]/)
            if ($0 ~ /;[ \t]*fi[ \t;]*$/) { if (ex) { printf "publish:cascade/%s\t%s:%d\n", blk, F, bl; k++ } }
            else inb = 1
            next
        }
        inb && /^fi/ { if (ex) { printf "publish:cascade/%s\t%s:%d\n", blk, F, bl; k++ }; inb = 0 }
        inb && /exit [1-9]/ { ex = 1 }
        /^[ \t]*if ! / {
            s = $0; sub(/^[ \t]*if ! /, "", s)
            if (s ~ /^bash /) { if (match(s, /scripts\/[A-Za-z0-9_\/.-]+\.sh/)) { printf "publish:cascade/%s\t%s:%d\n", substr(s, RSTART, RLENGTH), F, FNR; k++ } }
            else { w = s; sub(/[ ;].*/, "", w); if (w in DEF) { printf "publish:cascade/%s\t%s:%d\n", w, F, FNR; k++ } }
        }
        END { if (inb || unc) printf "unclosed top-level if at %s:%d\n", F, bl > "/dev/stderr"; if (k == 0 || !loop || inb || unc) exit 2 }
    ' "$1" "$1"
}

# strict_found FILE: every `|| die "…"` / `&& die "…"` before the first `cargo publish`, named by its message: the
# text up to the first `$(`, `: ` or closing quote, `$VAR` read as VAR, as at most 8 lowercase words. A `die` that
# names no word of its own (`die "$msg"`), a `die` without that form, or a name met twice is not_measured.
strict_found() {
    awk -v F="$2" '
        /^[ \t]*#/ { next }
        /cargo publish/ { cut = 1 }
        cut { next }
        /^[ \t]*die\(\)/ { next }
        /(^|[^A-Za-z0-9_])die[ \t]/ {
            s = $0
            if (!match(s, /(\|\||&&)[ \t]*die "/)) { UL[++nu] = F ":" FNR; next }
            s = substr(s, RSTART + RLENGTH)
            if (match(s, /\$\(|: |"/)) s = substr(s, 1, RSTART - 1)
            t = s; gsub(/\$\{?[A-Za-z0-9_]+\}?/, "", t); if (t !~ /[A-Za-z]/) { UL[++nu] = F ":" FNR; next }
            gsub(/\$\{?/, "", s); gsub(/\}/, "", s)
            s = tolower(s); gsub(/[^a-z0-9]+/, "-", s); gsub(/^-+|-+$/, "", s)
            n = split(s, w, "-"); s = ""; for (i = 1; i <= n && i <= 8; i++) s = s (i > 1 ? "-" : "") w[i]
            if (s == "") { UL[++nu] = F ":" FNR; next }
            if (s in SEEN) { printf "refusal named twice: %s at %s and %s:%d\n", s, SEEN[s], F, FNR > "/dev/stderr"; dup = 1; next }
            SEEN[s] = F ":" FNR
            printf "publish:strict/%s\t%s:%d\n", s, F, FNR; k++
        }
        END {
            for (i = 1; i <= nu; i++) printf "unparsed refusal at %s\n", UL[i] > "/dev/stderr"
            if (!cut) printf "no cargo publish in %s: the upload line is the boundary\n", F > "/dev/stderr"
            if (!cut || nu || dup || k == 0) exit 2
        }
    ' "$1"
}

# dogfood_found DOGFOOD CARGO: every row `dogfood.sh --phase pre-publish` can mark FAIL, named tag:dogfood/<row>. A
# row is `mark <row> FAIL`, `gate <row> <cmd>` or `ci_mark <row>` in statement position, or a `pre-publish:` arm of
# version_row that prints `<row> FAIL`; plus `declared:<name>` for each [package.metadata.dogfood].gates entry of the
# root manifest (any declared gate FAILs on a non-zero exit). A site counts unless an enclosing `if` branch cannot run
# in pre-publish: one that tests `"$DOGFOOD_PHASE" = post-publish`, or an elif/else after a branch that tests
# `= pre-publish`. Embedded python (heredocs, `python3 -c '…'`) is skipped. An unbalanced if/fi, no declared gate or
# no row at all is not_measured.
dogfood_found() {
    awk -v F="$3" -v C="$4" -v Q="'" '
        NR == FNR {
            if ($0 ~ /^\[package\.metadata\.dogfood\]/) { g = 1; next }
            if (g && /^\[/) g = 0
            if (g && match($0, /^[ \t]*"scripts\/[A-Za-z0-9_\/.-]+\.sh"/)) {
                n = substr($0, RSTART, RLENGTH); sub(/^[ \t]*"/, "", n); sub(/"$/, "", n); sub(/.*\//, "", n); sub(/\.sh$/, "", n)
                printf "tag:dogfood/declared:%s\t%s:%d\n", n, C, FNR; k++; nd++
            }
            next
        }
        HD != "" { u = $0; gsub(/[ \t]/, "", u); if (u == HD) HD = ""; next }
        SQ { if (index($0, Q)) SQ = 0; next }
        /^[ \t]*#/ { next }
        $0 ~ ("-c " Q "[ \t]*$") { SQ = 1; next }
        $0 !~ /<<</ && match($0, "<<-?" Q "?[A-Za-z_]+" Q "?") { HD = substr($0, RSTART, RLENGTH); gsub("[<" Q "-]", "", HD) }
        {
            line = $0; t = line; sub(/^[ \t]+/, "", t)
            if (t ~ /^if /) { d++; X[d] = (t ~ /"\$DOGFOOD_PHASE" = post-publish/); P[d] = (t ~ /"\$DOGFOOD_PHASE" = pre-publish/) }
            else if (t ~ /^elif /) { if (!d) bad = 1; X[d] = P[d] || (t ~ /"\$DOGFOOD_PHASE" = post-publish/); if (t ~ /"\$DOGFOOD_PHASE" = pre-publish/) P[d] = 1 }
            else if (t ~ /^else([ \t]|$)/) { if (!d) bad = 1; X[d] = P[d] }
            ex = 0; for (i = 1; i <= d; i++) if (X[i]) ex = 1
            if (!ex) {
                s = line
                while (match(s, /(^[ \t]*|([t]hen|else|[d]o|;|&&|\|\||\{) +)(mark +[a-z][a-z0-9-]* +FAIL|gate +[a-z][a-z0-9-]* +[^ ]|ci_mark +[a-z][a-z0-9-]* )/)) {
                    w = substr(s, RSTART, RLENGTH); s = substr(s, RSTART + RLENGTH); sub(/^.*(mark|gate|ci_mark) +/, "", w); sub(/ .*/, "", w)
                    if (!(w in SEEN)) { SEEN[w] = 1; printf "tag:dogfood/%s\t%s:%d\n", w, F, FNR; k++ }
                }
                if (match(t, "^pre-publish:[a-z-]+\\) +printf " Q "[a-z][a-z0-9-]* FAIL ")) {
                    w = substr(t, RSTART, RLENGTH); sub("^[^ ]+ +printf " Q, "", w); sub(/ .*/, "", w)
                    if (!(w in SEEN)) { SEEN[w] = 1; printf "tag:dogfood/%s\t%s:%d\n", w, F, FNR; k++ }
                }
            }
            if (t ~ /^fi([ \t;]|$)/ || (t ~ /^(if|elif|else) / && t ~ /;[ \t]*fi[ \t;]*$/)) { if (!d) bad = 1; else d-- }
        }
        END {
            if (d || bad) printf "unbalanced if/fi in %s\n", F > "/dev/stderr"
            if (!nd) printf "no [package.metadata.dogfood].gates in %s\n", C > "/dev/stderr"
            if (bad || d || !nd || k == 0) exit 2
        }
    ' "$2" "$1"
}

# claims_raw ROOT: "key<TAB>value<TAB>where" for every fact two places can disagree on
claims_raw() {
    local root=$1
    awk -v F="$RRW_REL" '/^DEFAULT_MODE=/ { v = $0; sub(/^DEFAULT_MODE=/, "", v); sub(/[ \t#].*/, "", v); printf "publish.R8.mode\t%s\t%s:%d\n", v, F, FNR }' "$root/$RRW_REL"
    awk -v F="$PRE_REL" '
        /^# RULES/ { r = 1 } r && /^#[ \t]*$/ { r = 0 }
        r { t = t " " $0; if (!at) at = FNR }
        END { gsub(/#/, " ", t); gsub(/[ \t]+/, " ", t)
              if (match(t, /DEFAULT_MODE is `[a-z]+`/)) { s = substr(t, RSTART, RLENGTH); sub(/^DEFAULT_MODE is `/, "", s); sub(/`$/, "", s); printf "publish.R8.mode\t%s\t%s:%d\n", s, F, at } }
    ' "$root/$PRE_REL"
    awk -v F="$AUTO_REL" '
        /^STEPS=\(/ { s = $0; sub(/^STEPS=\(/, "", s); sub(/\).*/, "", s); printf "autopilot.steps\t%s\t%s:%d\n", s, F, FNR }
        /^#   steps: / { s = $0; sub(/^#   steps: /, "", s); sub(/[ \t]+$/, "", s); printf "autopilot.steps\t%s\t%s:%d\n", s, F, FNR }
    ' "$root/$AUTO_REL"
    awk -v F="$AUTO_REL" '
        /^if run_step cascade; then/ { c = 1; next } c && /^fi/ { c = 0 }
        c && !/^[ \t]*#/ && match($0, /scripts\/[A-Za-z0-9_\/.-]+\.sh/) { v = substr($0, RSTART, RLENGTH); sub(/.*\//, "", v); printf "publish.executor\t%s\t%s:%d\n", v, F, FNR; exit }
    ' "$root/$AUTO_REL"
    awk -v F="$SPEC_REL" '/^\| \*\*T-4 Publish\*\* \|/ && match($0, /`[A-Za-z0-9_.-]+\.sh`/) { v = substr($0, RSTART + 1, RLENGTH - 2); printf "publish.executor\t%s\t%s:%d\n", v, F, FNR; exit }' "$root/$SPEC_REL"
    awk -v F="$MK_REL" '/^COV_FLOOR *:= *[0-9]+/ { v = $0; sub(/.*:= */, "", v); sub(/[^0-9].*/, "", v); printf "coverage.floor\t%s\t%s:%d\n", v, F, FNR }' "$root/$MK_REL"
    awk -v F="$DOC_REL" '{ s = $0; while (match(s, /COV_FLOOR := [0-9]+/)) { v = substr(s, RSTART, RLENGTH); sub(/.*:= /, "", v); printf "coverage.floor\t%s\t%s:%d\n", v, F, FNR; s = substr(s, RSTART + RLENGTH) } }' "$root/$DOC_REL"
}

# claims ROOT: claims_raw, and rc 2 unless every claim has both of its sides. A side that stops matching (a reworded
# comment, a moved variable) would otherwise drop out and leave the other side agreeing with nothing.
claims() {
    local out
    out=$(claims_raw "$1")
    printf '%s\n' "$out" | awk -F '\t' -v W="publish.R8.mode:$RRW_REL publish.R8.mode:$PRE_REL autopilot.steps:$AUTO_REL autopilot.steps:$AUTO_REL coverage.floor:$MK_REL coverage.floor:$DOC_REL publish.executor:$AUTO_REL publish.executor:$SPEC_REL" '
        NF >= 3 { f = $3; sub(/:[0-9]+$/, "", f); N[$1 ":" f]++ }
        END { n = split(W, w, " "); for (i = 1; i <= n; i++) need[w[i]]++
              for (x in need) if (N[x] < need[x]) { printf "claim side missing: %s (want %d, found %d)\n", x, need[x], N[x] + 0 > "/dev/stderr"; bad = 1 }
              exit bad ? 2 : 0 }' || return 2
    printf '%s\n' "$out"
}

# found ROOT: every anchor the surfaces state; rc 2 when a surface is missing or unreadable
found() {
    local root=$1 f
    for f in "$CI_REL" "$DOC_REL" "$AUTO_REL" "$PRE_REL" "$CAS_REL" "$STR_REL" "$SPEC_REL" "$RRW_REL" "$MK_REL" "$DF_REL" "$CARGO_REL"; do
        [ -f "$root/$f" ] || { printf 'surface missing: %s\n' "$f" >&2; return 2; }
    done
    ci_gate_found "$root/$CI_REL" "$CI_REL" || { printf 'surface unreadable: %s job gate\n' "$CI_REL" >&2; return 2; }
    doc_contexts_found "$root/$DOC_REL" "$DOC_REL" || { printf 'surface unreadable: %s required checks\n' "$DOC_REL" >&2; return 2; }
    autopilot_found "$root/$AUTO_REL" "$AUTO_REL" || { printf 'surface unreadable: %s steps\n' "$AUTO_REL" >&2; return 2; }
    preflight_found "$root/$PRE_REL" "$PRE_REL" || { printf 'surface unreadable: %s RULES\n' "$PRE_REL" >&2; return 2; }
    cascade_found "$root/$CAS_REL" "$CAS_REL" || { printf 'surface unreadable: %s gates\n' "$CAS_REL" >&2; return 2; }
    strict_found "$root/$STR_REL" "$STR_REL" || { printf 'surface unreadable: %s refusals\n' "$STR_REL" >&2; return 2; }
    dogfood_found "$root/$DF_REL" "$root/$CARGO_REL" "$DF_REL" "$CARGO_REL" || { printf 'surface unreadable: %s rows\n' "$DF_REL" >&2; return 2; }
}

# ---------------------------------------------------------------- the list ----------------------------------------
# parse_list FILE -> "id<TAB>applies<TAB>anchor<TAB>checker<TAB>producer<TAB>key<TAB>value<TAB>line<TAB>parent<TAB>seven";
# rc 3 on a bad line
parse_list() {
    awk -v SEVEN="$SEVEN" '
        BEGIN { n = split(SEVEN, a, " "); for (i = 1; i <= n; i++) S7[a[i]] = 1 }
        /^requirements:/ { r = 1; next }
        r && /^[^ #]/ { r = 0 }
        r && /^  - \{/ {
            s = $0; sub(/^  - \{/, "", s); if (s !~ /\}[ \t]*$/) { printf "line %d: not one {…} per line\n", NR > "/dev/stderr"; bad = 1; next }
            sub(/\}[ \t]*$/, "", s); delete V
            while (s != "") {
                sub(/^[ ,]+/, "", s); if (s == "") break
                if (!match(s, /^[a-z_]+: */)) { printf "line %d: expected key at: %s\n", NR, s > "/dev/stderr"; bad = 1; break }
                k = substr(s, 1, RLENGTH); sub(/: *$/, "", k); s = substr(s, RLENGTH + 1)
                if (s ~ /^"/) { if (!match(s, /^"[^"]*"/)) { bad = 1; break } v = substr(s, 2, RLENGTH - 2) }
                else if (s ~ /^\[/) { if (!match(s, /^\[[^]]*\]/)) { bad = 1; break } v = substr(s, 2, RLENGTH - 2); gsub(/ /, "", v) }
                else { match(s, /^[^,]*/); v = substr(s, 1, RLENGTH); sub(/[ \t]+$/, "", v) }
                if (k in V) { printf "line %d: key %s twice\n", NR, k > "/dev/stderr"; bad = 1 }
                V[k] = v; s = substr(s, RLENGTH + 1)
            }
            if (V["id"] == "" || V["applies_to"] == "" || V["anchor"] == "" || V["provenance"] == "") { printf "line %d: id, applies_to, anchor and provenance are required\n", NR > "/dev/stderr"; bad = 1; next }
            if (V["applies_to"] !~ /^(merge|tag|publish)(,(merge|tag|publish))*$/) { printf "line %d: applies_to must be a subset of merge, tag, publish\n", NR > "/dev/stderr"; bad = 1; next }
            if (V["applies_to"] ~ /publish/ && V["producer"] == "") { printf "line %d: %s applies to publish and names no nightly producer\n", NR, V["id"] > "/dev/stderr"; bad = 1; next }
            if ((V["key"] == "") != (V["value"] == "")) { printf "line %d: key and value come together\n", NR > "/dev/stderr"; bad = 1; next }
            if (V["id"] in ID) { printf "line %d: id %s twice\n", NR, V["id"] > "/dev/stderr"; bad = 1; next }
            if (V["seven"] != "" && !(V["seven"] in S7)) { printf "line %d: seven %s is not one of the seven\n", NR, V["seven"] > "/dev/stderr"; bad = 1; next }
            if (V["seven"] != "" && V["parent"] != "") { printf "line %d: a row carries no seven (%s)\n", NR, V["id"] > "/dev/stderr"; bad = 1; next }
            ID[V["id"]] = 1; if (V["parent"] != "") { PA[V["id"]] = V["parent"]; PL[V["id"]] = NR }
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\n", V["id"], V["applies_to"], V["anchor"], V["checker"], V["producer"], V["key"], V["value"], NR, V["parent"], V["seven"]
        }
        END {
            # a row of a GATE names a GATE: an id on the list that is not itself a row of another
            for (x in PA) if (!(PA[x] in ID) || (PA[x] in PA)) { printf "line %d: parent %s of %s is not a GATE on the list\n", PL[x], PA[x], x > "/dev/stderr"; bad = 1 }
            if (bad) exit 3
        }
    ' "$1"
}

# ---------------------------------------------------------------- the verdict -------------------------------------
verdict() {
    local root=$1 list=$2 tmp rc tree
    tmp=$(mktemp -d) || nm "mktemp"
    found "$root" > "$tmp/found" 2> "$tmp/err"; rc=$?
    [ "$rc" = 0 ] || { cat "$tmp/err"; rm -rf -- "${tmp:?}"; nm "a surface cannot be read"; }
    claims "$root" > "$tmp/claims" 2> "$tmp/err" || { cat "$tmp/err"; rm -rf -- "${tmp:?}"; nm "a claim has lost a side"; }
    : > "$tmp/list"
    if [ -f "$list" ]; then
        parse_list "$list" > "$tmp/list" 2> "$tmp/err"; rc=$?
        [ "$rc" = 0 ] || { cat "$tmp/err"; rm -rf -- "${tmp:?}"; caller_error "malformed list $list"; }
    else
        printf 'list absent: %s (every stated requirement is unlisted)\n' "$list"
    fi
    awk -F'\t' -v ROOT="$root" -v LISTF="${list#"$root"/}" -v FD="$tmp/found" -v CL="$tmp/claims" '
        BEGIN {
            while ((getline l < FD) > 0) { split(l, a, "\t"); if (!(a[1] in FW)) { FW[a[1]] = a[2]; FO[++nf] = a[1] } }
            while ((getline l < CL) > 0) { split(l, a, "\t"); C[++nc] = a[1] "\t" a[2] "\t" a[3] }
        }
        {
            id = $1; LA[$3] = id; n++
            if ($6 != "") C[++nc] = $6 "\t" $7 "\t" LISTF ":" $8 " (" id ")"
            ck = $4; nd = ""; if (index(ck, "#")) { nd = substr(ck, index(ck, "#") + 1); ck = substr(ck, 1, index(ck, "#") - 1) }
            if (ck == "" || ck == "-") { print "ORPHANED " id ": no checker"; o++; next }
            if ((getline x < (ROOT "/" ck)) <= 0) { print "ORPHANED " id ": checker " ck " is absent"; o++; next }
            close(ROOT "/" ck)
            if (nd != "") { hit = 0; while ((getline x < (ROOT "/" ck)) > 0) if (index(x, nd)) { hit = 1; break } close(ROOT "/" ck)
                            if (!hit) { print "ORPHANED " id ": " ck " does not contain \"" nd "\""; o++; next } }
            if (!($3 in FW)) { print "ORPHANED " id ": no surface states " $3 " any more"; o++ }
        }
        END {
            for (i = 1; i <= nf; i++) if (!(FO[i] in LA)) { print "UNLISTED " FO[i] "  (" FW[FO[i]] ")"; u++ }
            for (i = 1; i <= nc; i++) { split(C[i], a, "\t"); if (!((a[1], a[2]) in KV)) { KV[a[1], a[2]] = a[3]; NV[a[1]]++; if (NV[a[1]] == 1) KO[++nk] = a[1]; VL[a[1]] = VL[a[1]] " | " a[2] " @ " a[3] } }
            for (i = 1; i <= nk; i++) if (NV[KO[i]] > 1) { print "CONTRADICTION " KO[i] ":" substr(VL[KO[i]], 3); c++ }
            printf "release-ready: unlisted=%d orphaned=%d contradictions=%d listed=%d stated=%d\n", u, o, c, n, nf
            exit (u + o + c) ? 1 : 0
        }
    ' "$tmp/list"; rc=$?
    tree=$(git -C "$root" rev-parse --short=10 HEAD 2> /dev/null) || tree=nogit
    printf 'tree=%s list=%s\n' "$tree" "${list#"$root"/}"
    rm -rf -- "${tmp:?}"
    return "$rc"
}

# ---------------------------------------------------------------- the GATE list ------------------------------------
# gates LIST: print the list as the GATE list, one row per requirement. A check is a GATE when it can block a merge
# or a release; a check not on this list is LAB. A list that is absent, malformed or empty prints nothing to trust.
gates() {
    local list=$1 out
    [ -f "$list" ] || nm "no list $list"
    out=$(parse_list "$list") || caller_error "malformed list $list"
    [ -n "$out" ] || nm "the list $list names no requirement"
    printf 'GATE list from %s. A check is a GATE when it can block a merge or a release; anything not listed is LAB.\n' "${list##*/}"
    printf 'A GATE with rows (parent: on the list) prints them indented under it; the total counts GATEs, never rows.\n'
    printf '%s\n' "$out" | awk -F'\t' '
        { R[NR] = $0; if ($9 != "") { K[$9] = K[$9] "\n" NR; nk[$9]++ } }
        END {
            for (i = 1; i <= NR; i++) { split(R[i], a, "\t"); if (a[9] != "") continue
                printf "%-8s %-15s %-62s %s%s\n", a[1], a[2], a[3], a[4], nk[a[1]] ? "  (" nk[a[1]] " rows)" : ""; n++
                m = split(K[a[1]], c, "\n"); for (j = 2; j <= m; j++) { split(R[c[j]], b, "\t"); printf "  %-8s row of %-8s %-55s %s\n", b[1], a[1], b[3], b[4] } }
            printf "gates=%d\n", n
        }'
}

# ---------------------------------------------------------------- the budget ---------------------------------------
# budget LIST: GATEs (no parent) per stage against the caps, fixed here; an entry counts in every stage it names.
# Every key of SEVEN must ride on a GATE.
budget() {
    local list=$1 out
    [ -f "$list" ] || nm "no list $list"
    out=$(parse_list "$list") || caller_error "malformed list $list"
    [ -n "$out" ] || nm "the list $list names no requirement"
    printf '%s\n' "$out" | awk -F'\t' -v SEVEN="$SEVEN" '
        $9 == "" { n = split($2, s, ","); for (i = 1; i <= n; i++) E[s[i]]++; if ($10 != "") H[$10] = 1 }
        END {
            CAP["merge"] = 10; CAP["tag"] = 5; CAP["publish"] = 5
            printf "entries merge=%d tag=%d publish=%d total=%d\n", E["merge"], E["tag"], E["publish"], E["merge"] + E["tag"] + E["publish"]
            split("merge tag publish", st, " "); for (i = 1; i <= 3; i++) if (E[st[i]] > CAP[st[i]]) { printf "OVER %s %d/%d\n", st[i], E[st[i]], CAP[st[i]]; bad = 1 }
            n = split(SEVEN, k, " "); for (i = 1; i <= n; i++) if (!(k[i] in H)) { print "MISSING " k[i]; bad = 1 }
            exit bad
        }'
}

# ---------------------------------------------------------------- the probe (two GitHub reads) --------------------
probe() {
    local root=$1 list=$2 gh=${GH:-gh} repo=${RR_REPO:-paiml/aprender} got bp want lo go
    [ -f "$list" ] || caller_error "no list $list"
    want=$(parse_list "$list" | awk -F'\t' '$3 ~ /^merge:required-context\// { sub(/^merge:required-context\//, "", $3); print $3 }' | LC_ALL=C sort -u) \
        || caller_error "malformed list $list"
    got=$("$gh" api "repos/$repo/rules/branches/main" --jq '.[] | select(.type == "required_status_checks") | .parameters.required_status_checks[].context' 2> /dev/null) \
        || { printf 'release-ready probe: not_measured: the read of %s rules failed\n' "$repo"; return 2; }
    bp=$("$gh" api "repos/$repo/branches/main/protection/required_status_checks" --jq '.contexts[]' 2> /dev/null) \
        || { printf 'release-ready probe: not_measured: the read of %s branch protection failed\n' "$repo"; return 2; }
    # each surface must answer on its own: an empty side would hide behind the union
    [ -n "$(printf '%s\n' "$got" | awk 'NF')" ] || { printf 'release-ready probe: not_measured: the ruleset on %s main carries no required_status_checks rule\n' "$repo"; return 2; }
    [ -n "$(printf '%s\n' "$bp" | awk 'NF')" ] || { printf 'release-ready probe: not_measured: the branch protection on %s main requires no context\n' "$repo"; return 2; }
    got=$(printf '%s\n%s\n' "$got" "$bp")
    got=$(printf '%s\n' "$got" | awk 'NF' | LC_ALL=C sort -u)
    lo=$(LC_ALL=C comm -23 <(printf '%s\n' "$want") <(printf '%s\n' "$got") | paste -sd, -)
    go=$(LC_ALL=C comm -13 <(printf '%s\n' "$want") <(printf '%s\n' "$got") | paste -sd, -)
    printf 'release-ready probe: list=[%s] github=[%s] list-only=[%s] github-only=[%s]\n' \
        "$(printf '%s' "$want" | paste -sd, -)" "$(printf '%s' "$got" | paste -sd, -)" "$lo" "$go"
    [ -z "$lo$go" ]
}

# ---------------------------------------------------------------- the case table ----------------------------------
# fixture DIR: a tree whose seven surfaces state 17 requirements (3 of them rows of FX-T2), all listed (green)
fixture() {
    local d=$1
    mkdir -p "$d/.github/workflows" "$d/scripts/release" "$d/contracts"
    cat > "$d/$CI_REL" <<'YML'
jobs:
  x86-main:
    runs-on: x
    # sections: guard-tree
  gate:
    needs: [x86-main]
    steps:
      - run: |
          for pair in X86:guard-tree; do echo "$pair"; done
YML
    printf '%s\n' '**`main` is protected.** Required status checks: `ci / gate` + `workspace-test`. Direct pushes blocked.' \
        'The floor is `COV_FLOOR := 89`.' > "$d/$DOC_REL"
    cat > "$d/$AUTO_REL" <<'SH'
#   steps: wait deep tag cleanroom dryrun cascade
STEPS=(wait deep tag cleanroom dryrun cascade)
helper() {
  bash scripts/helper_gate.sh || die "helper"
}
t1_deep() {
  cargo metadata --locked --no-deps > "$AP/m.json" || die "T-1 cargo metadata failed"
}
if run_step wait; then
  echo waiting
fi
T1_LANES=(); for s in deep; do run_step "$s" && T1_LANES+=("$s"); done
for s in "${T1_LANES[@]}"; do "t1_$s" & done
git merge-base --is-ancestor "$MC" origin/main || die "not on main"
if run_step tag; then
  helper
  # bash scripts/commented_out.sh
fi
if run_step cleanroom; then
  gh workflow run clean-room.yml
fi
if run_step dryrun; then
  bash scripts/cascade-publish.sh --check
fi
if run_step cascade; then
  bash scripts/release/publish_strict.sh "$V"
  bash scripts/after_dryrun.sh
fi
SH
    cat > "$d/$PRE_REL" <<'SH'
# RULES
#   R1  the tree is clean.
#   R8  readiness: the wrapper's committed DEFAULT_MODE
#       is `enforce`.
#
echo "FAIL  R1 dirty"; echo "FAIL  R8 not ready"
SH
    cat > "$d/$CAS_REL" <<'SH'
new_refusal() {
  return 0
}
clean_room_gate() {
  return 0
}
if ! gh api x > /dev/null; then echo read; fi
if ! clean_room_gate "$ROOT"; then exit 1; fi
if ! git diff --quiet; then exit 1; fi
if [ "$N" -lt 1 ]; then
  echo "Refusing to publish nothing"
  exit 1
fi
# upload loop
for x in a; do
  :
done
if [ "$OK" = 0 ]; then
  exit 1
fi
# --check
SH
    cat > "$d/$STR_REL" <<'SH'
die() { echo "$*"; exit 1; }
[ "$(git rev-parse HEAD)" = "$T" ] || die "HEAD is not $T"
# [ -s r ] || die "commented out"
cargo publish -p x
[ "$n" -eq 1 ] || die "final verification: $n live"
SH
    mkdir -p "$d/${SPEC_REL%/*}"
    printf '%s\n' '| **T-4 Publish** | `publish_strict.sh` from a detached checkout of the tag |' > "$d/$SPEC_REL"
    printf '%s\n' 'DEFAULT_MODE=enforce' > "$d/$RRW_REL"
    printf '%s\n' 'COV_FLOOR := 89' > "$d/$MK_REL"
    printf '%s\n' 'exit 0' > "$d/scripts/helper_gate.sh"
    printf '%s\n' '[package.metadata.dogfood]' 'gates = [' '    "scripts/check_x.sh",' ']' '[other]' > "$d/$CARGO_REL"
    cat > "$d/$DF_REL" <<'SH'
mark git-clean FAIL "dirty"
gate clippy cargo clippy -- -D warnings
if [ "$DOGFOOD_PHASE" = post-publish ]; then
  mark release-assets FAIL "post-publish only"
fi
awk -f - <<'AWK'
mark in-heredoc FAIL
AWK
# mark commented FAIL
SH
    cat > "$d/$LIST_REL" <<'YML'
name: fixture
requirements:
  - {id: FX-M1, applies_to: [merge], anchor: "merge:ci.yml/gate/needs/x86-main", checker: ".github/workflows/ci.yml#  x86-main:", provenance: "fixture"}
  - {id: FX-M2, applies_to: [merge], anchor: "merge:ci.yml/gate/section/guard-tree", checker: ".github/workflows/ci.yml#guard-tree", provenance: "fixture"}
  - {id: FX-M3, applies_to: [merge], anchor: "merge:required-context/ci / gate", checker: ".github/workflows/ci.yml#  gate:", provenance: "fixture"}
  - {id: FX-M4, applies_to: [merge], anchor: "merge:required-context/workspace-test", checker: "CLAUDE.md#workspace-test", provenance: "fixture"}
  - {id: FX-T1, applies_to: [tag], anchor: "tag:autopilot/commit/git merge-base --is-ancestor", checker: "scripts/release/autopilot.sh#--is-ancestor", provenance: "fixture"}
  - {id: FX-T2, applies_to: [tag], anchor: "tag:autopilot/tag/bash scripts/helper_gate.sh", checker: "scripts/helper_gate.sh", provenance: "fixture"}
  - {id: FX-T3, applies_to: [tag], anchor: "tag:autopilot/deep/cargo metadata --locked --no-deps", checker: "scripts/release/autopilot.sh#cargo metadata --locked", provenance: "fixture"}
  - {id: FX-D1, applies_to: [tag], parent: FX-T2, anchor: "tag:dogfood/git-clean", checker: "scripts/dogfood.sh#mark git-clean FAIL", provenance: "fixture"}
  - {id: FX-D2, applies_to: [tag], parent: FX-T2, anchor: "tag:dogfood/clippy", checker: "scripts/dogfood.sh#gate clippy ", provenance: "fixture"}
  - {id: FX-D3, applies_to: [tag], parent: FX-T2, anchor: "tag:dogfood/declared:check_x", checker: "Cargo.toml#scripts/check_x.sh", provenance: "fixture"}
  - {id: FX-P1, applies_to: [publish], anchor: "publish:autopilot/cleanroom/gh-workflow clean-room.yml", checker: "scripts/release/autopilot.sh#clean-room.yml", provenance: "fixture", producer: "cleanroom-cpu"}
  - {id: FX-P2, applies_to: [publish], anchor: "publish:autopilot/dryrun/bash scripts/cascade-publish.sh --check", checker: "scripts/cascade-publish.sh#--check", provenance: "fixture", producer: "publish-dryrun"}
  - {id: FX-P3, applies_to: [publish], anchor: "publish:preflight/R1", checker: "scripts/check_publish_preflight.sh#FAIL  R1 ", provenance: "fixture", producer: "preflight"}
  - {id: FX-P4, applies_to: [tag, publish], anchor: "publish:preflight/R8", checker: "scripts/check_publish_preflight.sh#FAIL  R8 ", provenance: "fixture", producer: "readiness", key: "publish.R8.mode", value: "enforce"}
  - {id: FX-P5, applies_to: [publish], anchor: "publish:cascade/clean_room_gate", checker: "scripts/cascade-publish.sh#clean_room_gate() {", provenance: "fixture", producer: "cleanroom-cpu"}
  - {id: FX-P6, applies_to: [publish], anchor: "publish:strict/head-is-not-t", checker: "scripts/release/publish_strict.sh#HEAD is not $T", provenance: "fixture", producer: "publish-dryrun"}
  - {id: FX-P7, applies_to: [publish], anchor: "publish:cascade/if [ $N -lt 1 ]", checker: "scripts/cascade-publish.sh#Refusing to publish nothing", provenance: "fixture", producer: "publish-dryrun"}
YML
}

# plant DIR FILE TEXT: append TEXT as a line of FILE (relative to DIR)
plant() { printf '%s\n' "$3" >> "$1/$2"; }
# edit DIR FILE SED: one sed program on FILE; a program that changes nothing is a broken case, not a pass
edit() {
    local before
    before=$(cksum < "$1/$2")
    sed -i "$3" "$1/$2"
    [ "$(cksum < "$1/$2")" != "$before" ] || { printf 'CASE ERROR: sed %s changed nothing in %s\n' "$3" "$2"; return 1; }
}

# mutate CASE DIR: the planted change of each row (rows are named by what they plant)
mutate() {
    local c=$1 d=$2
    case "$c" in
        green|green-comment-mention|green-after-dryrun|green-gh-read) : ;;
        unlisted-step-script) edit "$d" "$AUTO_REL" '/^  gh workflow run clean-room.yml/a\  bash scripts/new_gate.sh' ;;
        unlisted-function-body) edit "$d" "$AUTO_REL" '/^  bash scripts\/helper_gate.sh/a\  bash scripts/new_in_helper.sh' ;;
        unlisted-between-steps) edit "$d" "$AUTO_REL" '/^git merge-base/a\bash scripts/new_preamble.sh --check || die x' ;;
        unlisted-cargo) edit "$d" "$AUTO_REL" '/^  echo waiting/a\  cargo test --doc --workspace' ;;
        unlisted-preflight-rule) edit "$d" "$PRE_REL" '/^#   R1 /a\#   R9  a new rule.' ;;
        unlisted-gate-needs) edit "$d" "$CI_REL" 's/needs: \[x86-main\]/needs: [x86-main, new-job]/' ;;
        unlisted-gate-section) edit "$d" "$CI_REL" 's/X86:guard-tree;/X86:guard-tree X86:new-section;/' ;;
        unlisted-cascade-gate) plant "$d" "$CAS_REL" 'if ! new_refusal "$ROOT"; then exit 1; fi' ;;
        unlisted-context) edit "$d" "$DOC_REL" 's/`workspace-test`/`workspace-test` + `new-context`/' ;;
        orphaned-checker-deleted) rm -f -- "${d:?}/scripts/helper_gate.sh" ;;
        orphaned-needle-gone) edit "$d" "$PRE_REL" 's/FAIL  R1 dirty/true/' ;;
        orphaned-anchor-gone) edit "$d" "$PRE_REL" '/^#   R1 /d' ;;
        orphaned-no-checker) edit "$d" "$LIST_REL" 's|checker: "scripts/helper_gate.sh", ||' ;;
        contradiction-two-entries) plant "$d" "$LIST_REL" '  - {id: FX-X1, applies_to: [merge], anchor: "merge:ci.yml/gate/needs/x86-main", checker: "Makefile", provenance: "fixture", key: "coverage.floor", value: "90"}' ;;
        contradiction-r8-doc) edit "$d" "$PRE_REL" 's/is `enforce`/is `report`/' ;;
        contradiction-steps-doc) edit "$d" "$AUTO_REL" 's/^#   steps: wait deep tag /#   steps: wait tag /' ;;
        contradiction-floor) edit "$d" "$DOC_REL" 's/COV_FLOOR := 89/COV_FLOOR := 88/' ;;
        nm-surface-missing) rm -f -- "${d:?}/$AUTO_REL" ;;
        nm-claim-surface-missing) rm -f -- "${d:?}/$MK_REL" ;;
        nm-needs-block-form) edit "$d" "$CI_REL" 's/needs: \[x86-main\]/needs:/' ;;
        nm-gate-no-sections) edit "$d" "$CI_REL" '/for pair in/d' ;;
        nm-claim-side-reworded) edit "$d" "$PRE_REL" 's/is `enforce`\./is the enforce mode./' ;;
        unlisted-dotslash) edit "$d" "$AUTO_REL" '/^  echo waiting/a\  bash ./scripts/dotslash_gate.sh' ;;
        unlisted-camel-function) edit "$d" "$AUTO_REL" 's/^helper() {/Helper2() {/; s/^  helper$/  Helper2/; /^  bash scripts\/helper_gate.sh/a\  bash scripts/camel_gate.sh' ;;
        nm-unknown-step) edit "$d" "$AUTO_REL" 's/^if run_step dryrun;/if run_step dryrnu;/' ;;
        nm-unparsed-quoted-root) edit "$d" "$AUTO_REL" '/^  echo waiting/a\  bash "$R"/scripts/quoted_gate.sh' ;;
        nm-unparsed-python-root) edit "$d" "$AUTO_REL" '/^  echo waiting/a\  python3 "$R/scripts/root_gate.py"' ;;
        nm-unparsed-exec) edit "$d" "$AUTO_REL" '/^  echo waiting/a\  ./scripts/exec_gate.sh' ;;
        unlisted-strict-refusal) edit "$d" "$STR_REL" '/^cargo publish/i\[ -s r ] || die "no receipt for $T: run T-4"' ;;
        nm-strict-unparsed) edit "$d" "$STR_REL" '/^cargo publish/i\[ -s r ] || die "$msg"' ;;
        nm-strict-no-upload) edit "$d" "$STR_REL" '/^cargo publish/d' ;;
        nm-strict-dup) edit "$d" "$STR_REL" '/^cargo publish/i\[ -n "$T" ] && die "HEAD is not $T"' ;;
        unlisted-cascade-if-block) edit "$d" "$CAS_REL" '/^# upload loop/i\if [ -z "$TOKEN" ]; then exit 3; fi' ;;
        green-cascade-if-after-loop) edit "$d" "$CAS_REL" '/^# --check/i\if [ -z "$LATE" ]; then exit 1; fi' ;;
        contradiction-executor) edit "$d" "$AUTO_REL" 's|bash scripts/release/publish_strict.sh|bash scripts/release/cascade-drain.sh|' ;;
        nm-executor-spec-reworded) edit "$d" "$SPEC_REL" 's/T-4 Publish/T-4 Upload/' ;;
        nm-spec-missing) rm -f -- "${d:?}/$SPEC_REL" ;;
        nm-cascade-no-loop) edit "$d" "$CAS_REL" 's/^for x in a/while false/' ;;
        nm-cascade-unclosed-if) edit "$d" "$CAS_REL" '/^# upload loop/i\if [ -z "$OPEN" ]; then' ;;
        nm-cascade-if-spans-loop) edit "$d" "$CAS_REL" '/^# upload loop/i\if [ -z "$OPEN" ]; then' && edit "$d" "$CAS_REL" '/^if \[ "$OK" = 0 \]; then/d' ;;
        malformed-no-producer) edit "$d" "$LIST_REL" 's|, producer: "preflight"||' ;;
        malformed-bad-phase) edit "$d" "$LIST_REL" 's/applies_to: \[merge\], anchor: "merge:ci.yml\/gate\/needs/applies_to: [deploy], anchor: "merge:ci.yml\/gate\/needs/' ;;
        malformed-dup-id) plant "$d" "$LIST_REL" '  - {id: FX-M1, applies_to: [merge], anchor: "x", checker: "Makefile", provenance: "fixture"}' ;;
        unlisted-dogfood-row) plant "$d" "$DF_REL" 'mark new-row FAIL "x"' ;;
        unlisted-dogfood-chained) plant "$d" "$DF_REL" '[ -s r ] || gate chained-row test -s r' ;;
        unlisted-dogfood-pre-printf) plant "$d" "$DF_REL" "  pre-publish:present) printf 'pre-row FAIL %s\\n' x ;;" ;;
        unlisted-declared-gate) edit "$d" "$CARGO_REL" '/check_x.sh/a\    "scripts/check_y.sh",' ;;
        green-dogfood-else-of-pre) plant "$d" "$DF_REL" 'if [ "$DOGFOOD_PHASE" = pre-publish ]; then' && plant "$d" "$DF_REL" '  :' \
            && plant "$d" "$DF_REL" 'else' && plant "$d" "$DF_REL" '  mark post-row FAIL "x"' && plant "$d" "$DF_REL" 'fi' ;;
        unlisted-lane-cargo) edit "$d" "$AUTO_REL" '/^  cargo metadata --locked/a\  cargo test --doc --workspace' ;;
        nm-lane-no-function) edit "$d" "$AUTO_REL" 's/^t1_deep() {/t1_deeep() {/' ;;
        nm-dogfood-missing) rm -f -- "${d:?}/$DF_REL" ;;
        nm-dogfood-unbalanced) plant "$d" "$DF_REL" 'if [ -n "$X" ]; then' ;;
        nm-dogfood-no-declared) edit "$d" "$CARGO_REL" '/check_x.sh/d' ;;
        malformed-parent-is-row) edit "$d" "$LIST_REL" 's/id: FX-D2, applies_to: \[tag\], parent: FX-T2/id: FX-D2, applies_to: [tag], parent: FX-D1/' ;;
        malformed-parent-unknown) edit "$d" "$LIST_REL" 's/id: FX-D2, applies_to: \[tag\], parent: FX-T2/id: FX-D2, applies_to: [tag], parent: FX-T9/' ;;
        *) printf 'CASE ERROR: unknown case %s\n' "$c"; return 1 ;;
    esac
}

# budget_list M T P [DROP]: a list of M merge, T tag and P publish GATEs; the keys of SEVEN ride on the first seven,
# except DROP
budget_list() {
    local st j i=0 sv
    local -a ks
    read -ra ks <<< "$SEVEN"
    printf 'requirements:\n'
    for st in "merge $1" "tag $2" "publish $3"; do
        for ((j = 1; j <= ${st#* }; j++)); do
            sv=${ks[$i]:-}; i=$((i + 1)); [ "$sv" != "${4:-}" ] || sv=""
            printf '  - {id: B-%s%d, applies_to: [%s], anchor: "x", provenance: "fixture", producer: "p"%s}\n' "${st%% *}" "$j" "${st%% *}" "${sv:+, seven: $sv}"
        done
    done
}

# the table: case · expected exit · a pattern the output must carry
CASES='green 0 unlisted=0 orphaned=0 contradictions=0 listed=17 stated=17
green-comment-mention 0 stated=17
green-after-dryrun 0 stated=17
green-gh-read 0 stated=17
unlisted-step-script 1 UNLISTED publish:autopilot/cleanroom/bash scripts/new_gate.sh
unlisted-function-body 1 UNLISTED tag:autopilot/tag/bash scripts/new_in_helper.sh
unlisted-between-steps 1 UNLISTED tag:autopilot/commit/bash scripts/new_preamble.sh --check
unlisted-cargo 1 UNLISTED tag:autopilot/wait/cargo test --doc --workspace
unlisted-preflight-rule 1 UNLISTED publish:preflight/R9
unlisted-gate-needs 1 UNLISTED merge:ci.yml/gate/needs/new-job
unlisted-gate-section 1 UNLISTED merge:ci.yml/gate/section/new-section
unlisted-cascade-gate 1 UNLISTED publish:cascade/new_refusal
unlisted-context 1 UNLISTED merge:required-context/new-context
orphaned-checker-deleted 1 ORPHANED FX-T2: checker scripts/helper_gate.sh is absent
orphaned-needle-gone 1 ORPHANED FX-P3: scripts/check_publish_preflight.sh does not contain
orphaned-anchor-gone 1 ORPHANED FX-P3: no surface states publish:preflight/R1
orphaned-no-checker 1 ORPHANED FX-T2: no checker
contradiction-two-entries 1 CONTRADICTION coverage.floor: 89
contradiction-r8-doc 1 CONTRADICTION publish.R8.mode
contradiction-steps-doc 1 CONTRADICTION autopilot.steps
contradiction-floor 1 CONTRADICTION coverage.floor
nm-surface-missing 2 not_measured
nm-claim-surface-missing 2 surface missing: Makefile
nm-needs-block-form 2 not_measured
nm-gate-no-sections 2 not_measured
nm-claim-side-reworded 2 not_measured
unlisted-dotslash 1 UNLISTED tag:autopilot/wait/bash scripts/dotslash_gate.sh
unlisted-camel-function 1 UNLISTED tag:autopilot/tag/bash scripts/camel_gate.sh
nm-unknown-step 2 not_measured
nm-unparsed-quoted-root 2 unparsed call form at
nm-unparsed-python-root 2 unparsed call form at
nm-unparsed-exec 2 unparsed call form at
unlisted-strict-refusal 1 UNLISTED publish:strict/no-receipt-for-t
nm-strict-unparsed 2 unparsed refusal at
nm-strict-no-upload 2 not_measured
nm-strict-dup 2 refusal named twice
unlisted-cascade-if-block 1 UNLISTED publish:cascade/if [ -z $TOKEN ]
green-cascade-if-after-loop 0 stated=17
contradiction-executor 1 CONTRADICTION publish.executor
nm-executor-spec-reworded 2 not_measured
nm-spec-missing 2 surface missing: docs/specifications/APR-RELEASE-001
nm-cascade-no-loop 2 not_measured
nm-cascade-unclosed-if 2 unclosed top-level if at
nm-cascade-if-spans-loop 2 unclosed top-level if at
malformed-no-producer 3 names no nightly producer
malformed-bad-phase 3 applies_to must be a subset
malformed-dup-id 3 id FX-M1 twice
unlisted-dogfood-row 1 UNLISTED tag:dogfood/new-row
unlisted-dogfood-chained 1 UNLISTED tag:dogfood/chained-row
unlisted-dogfood-pre-printf 1 UNLISTED tag:dogfood/pre-row
unlisted-declared-gate 1 UNLISTED tag:dogfood/declared:check_y
green-dogfood-else-of-pre 0 stated=17
unlisted-lane-cargo 1 UNLISTED tag:autopilot/deep/cargo test --doc --workspace
nm-lane-no-function 2 not_measured
nm-dogfood-missing 2 surface missing: scripts/dogfood.sh
nm-dogfood-unbalanced 2 unbalanced if/fi
nm-dogfood-no-declared 2 no [package.metadata.dogfood].gates
malformed-parent-is-row 3 parent FX-D1 of FX-D2 is not a GATE
malformed-parent-unknown 3 parent FX-T9 of FX-D2 is not a GATE'

selftest() {
    local tmp c want pat out rc pass=0 fail=0 n=0 line mx xl b1 b2 b3 b4
    tmp=$(mktemp -d) || nm "mktemp"
    fixture "$tmp/base"
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        c=${line%% *}; line=${line#* }; want=${line%% *}; pat=${line#* }; n=$((n + 1))
        cp -R "$tmp/base" "$tmp/$c"
        if ! mutate "$c" "$tmp/$c"; then printf 'FAIL  %-28s the case did not apply\n' "$c"; fail=$((fail + 1)); continue; fi
        out=$(bash "$SCRIPT_PATH" --root "$tmp/$c" 2>&1); rc=$?
        if [ "$rc" = "$want" ] && [[ "$out" == *"$pat"* ]]; then pass=$((pass + 1))
        else fail=$((fail + 1)); printf 'FAIL  %-28s want rc=%s and "%s"; got rc=%s: %s\n' "$c" "$want" "$pat" "$rc" "$(printf '%s' "$out" | tr '\n' ' ' | cut -c1-200)"; fi
    done <<< "$CASES"
    # the probe: a stub gh answers the two reads. STUB_CTX is the ruleset's contexts, STUB_BP the classic
    # protection's (unset: the same as STUB_CTX). STUB_FAIL=1 fails both reads, STUB_FAIL=bp only the second.
    mkdir -p "$tmp/bin"
    cat > "$tmp/bin/gh" <<'STUB'
#!/usr/bin/env bash
case "$*" in
    *protection*) [ -n "${STUB_FAIL:-}" ] && exit 1; printf '%s\n' "${STUB_BP-$STUB_CTX}" | tr , '\n' ;;
    *) [ "${STUB_FAIL:-}" = 1 ] && exit 1; printf '%s\n' "$STUB_CTX" | tr , '\n' ;;
esac
STUB
    chmod +x "$tmp/bin/gh"
    # row: case rc pattern|ruleset contexts|protection contexts ("-" = the same)|STUB_FAIL
    for line in 'probe-equal 0 list-only=[] github-only=[]|ci / gate,workspace-test|-|' \
                'probe-github-extra 1 github-only=[extra]|ci / gate,workspace-test,extra|-|' \
                'probe-list-extra 1 list-only=[workspace-test]|ci / gate|-|' \
                'probe-union-of-both 1 list-only=[] github-only=[gate]|gate,workspace-test|ci / gate,workspace-test|' \
                'probe-read-failed 2 not_measured|ci / gate|-|1' \
                'probe-protection-failed 2 branch protection failed|ci / gate,workspace-test|-|bp' \
                'probe-ruleset-empty 2 the ruleset on||ci / gate,workspace-test|' \
                'probe-protection-empty 2 requires no context|ci / gate,workspace-test||'; do
        c=${line%% *}; line=${line#* }; want=${line%% *}; line=${line#* }; pat=${line%%|*}; line=${line#*|}; n=$((n + 1))
        ctx=${line%%|*}; line=${line#*|}; bp=${line%%|*}; sf=${line#*|}
        if [ "$bp" = - ]; then
            out=$(GH="$tmp/bin/gh" STUB_CTX="$ctx" STUB_FAIL="$sf" bash "$SCRIPT_PATH" --probe --root "$tmp/base" 2>&1); rc=$?
        else
            out=$(GH="$tmp/bin/gh" STUB_CTX="$ctx" STUB_BP="$bp" STUB_FAIL="$sf" bash "$SCRIPT_PATH" --probe --root "$tmp/base" 2>&1); rc=$?
        fi
        if [ "$rc" = "$want" ] && [[ "$out" == *"$pat"* ]]; then pass=$((pass + 1))
        else fail=$((fail + 1)); printf 'FAIL  %-28s want rc=%s and "%s"; got rc=%s: %s\n' "$c" "$want" "$pat" "$rc" "$(printf '%s' "$out" | tr '\n' ' ' | cut -c1-200)"; fi
    done
    # --gates: the fixture's list prints its rows; an empty list and an absent one are not_measured, never a list.
    mkdir -p "$tmp/empty/contracts"; printf 'requirements:\n' > "$tmp/empty/$LIST_REL"
    for line in "gates-list 0 FX-M1 $tmp/base" "gates-rows 0 row of FX-T2 $tmp/base" "gates-total 0 gates=14 $tmp/base" "gates-empty 2 names no requirement $tmp/empty" "gates-absent 2 no list $tmp/none"; do
        c=${line%% *}; line=${line#* }; want=${line%% *}; line=${line#* }; pat=${line% *}; n=$((n + 1))
        out=$(bash "$SCRIPT_PATH" --gates --root "$tmp/base" --list "${line##* }/$LIST_REL" 2>&1); rc=$?
        if [ "$rc" = "$want" ] && [[ "$out" == *"$pat"* ]]; then pass=$((pass + 1))
        else fail=$((fail + 1)); printf 'FAIL  %-28s want rc=%s and "%s"; got rc=%s: %s\n' "$c" "$want" "$pat" "$rc" "$(printf '%s' "$out" | tr '\n' ' ' | cut -c1-200)"; fi
    done
    # --budget: case rc pattern|M T P [DROP] for budget_list|one more line
    for line in 'budget-green 0 entries merge=10 tag=5 publish=5 total=20|10 5 5|' \
                'budget-merge-over 1 OVER merge 11/10|11 5 5|' \
                'budget-tag-over 1 OVER tag 6/5|10 6 5|' \
                'budget-publish-over 1 OVER publish 6/5|10 5 6|' \
                'budget-seven-dropped 1 MISSING crux-smoke|10 5 5 crux-smoke|' \
                'budget-seven-unknown 3 seven nope is not one of the seven|10 5 5|  - {id: B-X, applies_to: [merge], anchor: "x", provenance: "fixture", seven: nope}' \
                'budget-seven-on-row 3 a row carries no seven|10 5 5|  - {id: B-X, applies_to: [merge], parent: B-merge1, anchor: "x", provenance: "fixture", seven: crux-smoke}' \
                'budget-row-uncounted 0 entries merge=10 tag=5 publish=5 total=20|10 5 5|  - {id: B-X, applies_to: [merge, tag, publish], parent: B-merge1, anchor: "x", provenance: "fixture", producer: "p"}' \
                'budget-two-stages 0 entries merge=10 tag=5 publish=5 total=20|9 4 5|  - {id: B-X, applies_to: [merge, tag], anchor: "x", provenance: "fixture"}' \
                'budget-empty 2 names no requirement|0 0 0|' \
                'budget-absent 2 no list|-|'; do
        c=${line%% *}; line=${line#* }; want=${line%% *}; line=${line#* }; pat=${line%%|*}; line=${line#*|}; n=$((n + 1))
        mx=${line%%|*}; xl=${line#*|}; read -r b1 b2 b3 b4 <<< "$mx"
        if [ "$mx" != - ]; then { budget_list "$b1" "$b2" "$b3" "${b4:-}"; [ -z "$xl" ] || printf '%s\n' "$xl"; } > "$tmp/$c.yaml"; fi
        out=$(bash "$SCRIPT_PATH" --budget --root "$tmp/base" --list "$tmp/$c.yaml" 2>&1); rc=$?
        if [ "$rc" = "$want" ] && [[ "$out" == *"$pat"* ]]; then pass=$((pass + 1))
        else fail=$((fail + 1)); printf 'FAIL  %-28s want rc=%s and "%s"; got rc=%s: %s\n' "$c" "$want" "$pat" "$rc" "$(printf '%s' "$out" | tr '\n' ' ' | cut -c1-200)"; fi
    done
    rm -rf -- "${tmp:?}"
    printf 'release-ready selftest: %d/%d rows pass\n' "$pass" "$n"
    [ "$fail" = 0 ] && [ "$pass" -gt 0 ]
}

# ---------------------------------------------------------------- mutants -----------------------------------------
# Each mutant drops one check of this script; the case table must turn RED on it. A mutant whose text is not
# found is an ERROR (the patch did not apply), never a survivor.
MUTANTS='M01 drop UNLISTED@@if (!(FO[i] in LA)) {@@if (0) {
M02 drop checker-absent@@if ((getline x < (ROOT "/" ck)) <= 0)@@if (0)
M03 drop needle check@@if (!hit) {@@if (0) {
M04 drop stale-anchor@@if (!($3 in FW))@@if (0)
M05 drop CONTRADICTION@@if (NV[KO[i]] > 1)@@if (0)
M06 not_measured passes@@nm() { printf '"'"'release-ready: not_measured: %s\n'"'"' "$*"; exit 2; }@@nm() { printf '"'"'release-ready: not_measured: %s\n'"'"' "$*"; exit 0; }
M07 drop producer rule@@V["applies_to"] ~ /publish/ && V["producer"] == ""@@0
M08 drop function bodies@@for (r = 1; r <= nf; r++) if (L[e]@@for (r = 1; r <= 0; r++) if (L[e]
M09 drop between-steps@@if (!(i in IN)) {@@if (0) {
M10 drop no-checker@@if (ck == "" || ck == "-")@@if (0)
M11 probe always equal@@[ -z "$lo$go" ]@@true
M12 drop unknown-step@@if (!(st in PH)) { unk = 1; continue }@@if (!(st in PH)) { continue }
M13 surface-missing passes@@[ -f "$root/$f" ] || { printf '"'"'surface missing: %s\n'"'"' "$f" >&2; return 2; }@@[ -f "$root/$f" ] || continue
M14 gate sections optional@@if (bad || kn == 0 || kp == 0)@@if (bad || kn == 0)
M15 claim side optional@@exit bad ? 2 : 0 }@@exit 0 }
M16 cascade takes any word@@if (w in DEF)@@if (w != "")
M17 dot-slash unseen@@(\$\{?[A-Za-z_]+\}?\/|\.\/)?scripts@@(\$\{?[A-Za-z_]+\}?\/)?scripts
M18 lowercase functions only@@if (L[i] ~ /^[A-Za-z_][A-Za-z0-9_]*\(\) *\{/@@if (L[i] ~ /^[a-z_]+\(\) *\{/
M19 drop protection read@@got=$(printf '"'"'%s\n%s\n'"'"' "$got" "$bp")@@got=$(printf '"'"'%s\n'"'"' "$got")
M20 empty protection passes@@[ -n "$(printf '"'"'%s\n'"'"' "$bp" | awk '"'"'NF'"'"')" ] ||@@true ||
M21 unparsed call form skipped@@if (nr > np) UNP = 1@@if (0) UNP = 1
M22 strict refusals after the upload read@@        cut { next }@@        0 { next }
M23 unparsed strict refusal passes@@if (!cut || nu || dup || k == 0) exit 2@@if (!cut || dup || k == 0) exit 2
M24 cascade if-blocks ignored@@!loop && /^if \[/ {@@0 && /^if \[/ {
M25 executor claim side optional@@ publish.executor:$AUTO_REL publish.executor:$SPEC_REL"@@"
M26 variable-only refusal named@@if (t !~ /[A-Za-z]/)@@if (0)
M27 refusal named twice passes@@dup = 1; next }@@next }
M28 no upload loop passes@@if (k == 0 || !loop || inb || unc) exit 2@@if (k == 0 || inb || unc) exit 2
M29 if-block spanning the loop passes@@{ if (inb) unc = 1; loop = 1 }@@{ loop = 1 }
M30 empty GATE list printed@@[ -n "$out" ] || nm "the list $list names no requirement"@@true
M31 dogfood surface unread@@dogfood_found "$root/$DF_REL" "$root/$CARGO_REL" "$DF_REL" "$CARGO_REL" ||@@true ||
M32 post-publish branch counted@@ex = 0; for (i = 1; i <= d; i++) if (X[i]) ex = 1@@ex = 0
M33 else of pre-publish counted@@X[d] = P[d] }@@X[d] = 0 }
M34 heredoc body scanned@@HD != "" { u = $0;@@0 { u = $0;
M35 unbalanced dogfood passes@@if (bad || d || !nd || k == 0) exit 2@@if (!nd || k == 0) exit 2
M36 no declared gates passes@@if (bad || d || !nd || k == 0) exit 2@@if (bad || d || k == 0) exit 2
M37 declared gates unread@@printf "tag:dogfood/declared:%s\t%s:%d\n", n, C, FNR; k++; nd++@@nd++
M38 parallel lanes unseen@@if (L[i] !~ /for s in [a-z ]+; [d]o run_step "\$s"/) continue@@continue
M39 lane without function passes@@if (!(st in PH) || !(fn in FS0)) { unk = 1; continue }@@if (!(st in PH) || !(fn in FS0)) { continue }
M40 quoted literal is a call@@(index(w, "$(") ? w : "\"\"")@@w
M41 parent validation dropped@@if (!(PA[x] in ID) || (PA[x] in PA))@@if (0)
M42 rows counted in the total@@if (a[9] != "") continue@@if (a[9] != "") { n++; continue }
M43 chained dogfood calls unseen@@([t]hen|else|[d]o|;|&&|\|\||\{) +)(mark@@([t]hen|else|[d]o|;|&&|\{) +)(mark
M44 pre-publish printf rows unseen@@if (match(t, "^pre-publish:[a-z-]+\\) +printf " Q "[a-z][a-z0-9-]* FAIL "))@@if (0)
M45 merge cap 11@@CAP["merge"] = 10@@CAP["merge"] = 11
M46 tag cap 6@@CAP["tag"] = 5@@CAP["tag"] = 6
M47 publish cap 6@@CAP["publish"] = 5@@CAP["publish"] = 6
M48 at the cap is over@@if (E[st[i]] > CAP[st[i]])@@if (E[st[i]] >= CAP[st[i]])
M49 a seven dropped@@ crux-smoke supply-chain@@ supply-chain
M50 rows counted as entries@@$9 == "" { n = split($2@@1 { n = split($2
M51 MISSING dropped@@if (!(k[i] in H))@@if (0)
M52 unknown seven passes@@if (V["seven"] != "" && !(V["seven"] in S7))@@if (0)
M53 seven on a row passes@@if (V["seven"] != "" && V["parent"] != "")@@if (0)'

mutants() {
    local tmp line id name from to killed=0 total=0 err=0
    tmp=$(mktemp -d) || nm "mktemp"
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        id=${line%% *}; line=${line#* }; name=${line%%@@*}; line=${line#*@@}; from=${line%%@@*}; to=${line#*@@}
        total=$((total + 1))
        RR_FROM=$from RR_TO=$to awk '{ i = index($0, ENVIRON["RR_FROM"]); if (i && !done) { $0 = substr($0, 1, i - 1) ENVIRON["RR_TO"] substr($0, i + length(ENVIRON["RR_FROM"])); done = 1 } print }
            END { if (!done) exit 1 }' "$SCRIPT_PATH" > "$tmp/m.sh" && ! cmp -s "$SCRIPT_PATH" "$tmp/m.sh" && bash -n "$tmp/m.sh" \
            || { printf 'ERROR    %s %s: the patch did not apply (or broke the syntax)\n' "$id" "$name"; err=$((err + 1)); continue; }
        if bash "$tmp/m.sh" --selftest > "$tmp/out" 2>&1; then printf 'SURVIVED %s %s\n' "$id" "$name"
        elif ! grep -q -e '^FAIL' "$tmp/out"; then printf 'ERROR    %s %s: the case table went red with no FAIL row (a crash is not a kill)\n' "$id" "$name"; err=$((err + 1))
        else killed=$((killed + 1)); printf 'killed   %s %s (%s rows red)\n' "$id" "$name" "$(grep -c -e '^FAIL' "$tmp/out")"; fi
    done <<< "$MUTANTS"
    rm -rf -- "${tmp:?}"
    printf 'release-ready mutants: killed=%d total=%d errors=%d\n' "$killed" "$total" "$err"
    [ "$killed" = "$total" ] && [ "$err" = 0 ]
}

# ---------------------------------------------------------------- main ------------------------------------------
main() {
    local mode=verdict root=$DEFAULT_ROOT list=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --root) [ $# -ge 2 ] || caller_error "--root needs DIR"; root=$2; shift 2 ;;
            --list) [ $# -ge 2 ] || caller_error "--list needs FILE"; list=$2; shift 2 ;;
            --found) mode=found; shift ;;
            --claims) mode=claims; shift ;;
            --gates) mode=gates; shift ;;
            --budget) mode=budget; shift ;;
            --probe) mode=probe; shift ;;
            --selftest) mode=selftest; shift ;;
            --mutants) mode=mutants; shift ;;
            -h|--help) sed -n '2,/^set -uo/p' "$SCRIPT_PATH" | sed '$d'; exit 0 ;;
            *) caller_error "unknown argument $1" ;;
        esac
    done
    [ -d "$root" ] || caller_error "no root $root"
    [ -n "$list" ] || list=$root/$LIST_REL
    case "$mode" in
        found) found "$root" ;;
        claims) claims "$root" ;;
        gates) gates "$list" ;;
        budget) budget "$list" ;;
        probe) probe "$root" "$list" ;;
        selftest) selftest ;;
        mutants) mutants ;;
        *) verdict "$root" "$list" ;;
    esac
}

[ "${RR_SOURCED:-0}" = 1 ] || main "$@"
