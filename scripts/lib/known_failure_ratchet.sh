# known_failure_ratchet.sh -- the known-failure ratchet, as a sourced library (#4689).
#
# WHAT IT JUDGES. A release ships with some failures that are already known: each one is on a list,
#   keyed by (model sha256, leg, check), with its ticket. A leg is a GPU architecture or a runner
#   label. The ratchet sorts every failure in tonight's evidence bundle into exactly one class:
#     KNOWN     the key is on the list. It is still a failure: printed RED with its ticket, never green.
#     NEW       the key passed in the last release's bundle and does not pass here. RED.
#     UNLISTED  the key fails here, is not on the list, and the last release's bundle did not show it
#               passing (or there is no such bundle). RED: a failure nobody has ruled on.
#   The list itself is shrink-only. It is RED when a row was added or edited against the list at the
#   base ref (a new failure needs a ruling, not a list edit), when a listed key passes here (a fixed
#   entry left on the list: STALE, it must be deleted), when a row has no ticket, and when a key is
#   listed twice. The ratchet never writes the list, and it never excuses a failure.
# NOT MEASURED IS NEVER A PASS. A missing or unreadable head bundle, a listed key the head did not
#   measure, or a key that passed in the base bundle and is absent here, makes the verdict
#   NOT_MEASURED unless something is RED. RED outranks NOT_MEASURED; GREEN needs neither.
#
# FORMATS (tab-separated; '#' lines and blank lines are ignored; a first column named sha256 is a header)
#   list     sha256  leg  check  ticket  [model ...]         ticket is #<digits>
#   bundle   DIR/checks.tsv: sha256  leg  check  state  [...] state is pass, fail or not_measured
#
# OPTION-NEUTRAL. This file is sourced: it sets no shell option and never exits. Every function
#   fails by return status:  . scripts/lib/known_failure_ratchet.sh || exit 1
#
# FUNCTIONS
#   kfr_list FILE OUT            valid rows -> OUT (sha, leg, check, ticket); FAIL lines; 1 on a bad row
#   kfr_bundle DIR OUT           measured rows -> OUT (sha, leg, check, state); 0 ok, 2 not_measured
#   kfr_judge LIST LBASE HEAD BASE
#                                LIST is a list file; LBASE a list file or "none" (no list at the base
#                                ref yet: BOOTSTRAP); HEAD a bundle dir; BASE a bundle dir or "none".
#                                Prints its findings and ONE verdict line, starting "KFR GREEN",
#                                "KFR RED" or "KFR NOT_MEASURED". Returns 0, 1 or 2.

kfr_list() { # FILE OUT
    [ -f "$1" ] && [ -r "$1" ] || { printf 'FAIL  KFR list %s is not a readable file\n' "$1"; return 1; }
    LC_ALL=C awk -F '\t' -v OFS='\t' -v out="$2" '
        function bad(why) { printf "FAIL  KFR list line %d: %s\n", NR, why; rc = 1 }
        BEGIN { rc = 0; printf "" > out }
        /^#/ || /^[[:space:]]*$/ { next }
        $1 == "sha256" { next }
        {
            if ($1 !~ /^[0-9a-f]{64}$/) { bad("sha256 \"" $1 "\" is not 64 lowercase hex"); next }
            if ($2 !~ /^[A-Za-z0-9_.-]+$/) { bad("leg \"" $2 "\" is not a GPU architecture or runner label"); next }
            if ($3 !~ /^[A-Za-z0-9_.:\/-]+$/) { bad("check \"" $3 "\" is empty or malformed"); next }
            if ($4 == "") { bad("entry " $1 " " $2 " " $3 " has no ticket: a known failure ships with its ticket"); next }
            if ($4 !~ /^#[0-9]+$/) { bad("ticket \"" $4 "\" is not #<number>"); next }
            k = $1 OFS $2 OFS $3
            if (k in seen) { bad("key " substr($1, 1, 12) " " $2 " " $3 " is listed twice (lines " seen[k] " and " NR ")"); next }
            seen[k] = NR
            print $1, $2, $3, $4 > out
        }
        END { exit rc }' "$1"
}

kfr_bundle() { # DIR OUT
    [ -d "$1" ] || { printf 'NM    KFR bundle %s does not exist: not_measured\n' "$1"; return 2; }
    [ -f "$1/checks.tsv" ] && [ -r "$1/checks.tsv" ] || { printf 'NM    KFR bundle %s has no readable checks.tsv: not_measured\n' "$1"; return 2; }
    LC_ALL=C awk -F '\t' -v OFS='\t' -v out="$2" -v dir="$1" '
        function nm(why) { printf "NM    KFR bundle %s line %d: %s: the bundle is unreadable, not_measured\n", dir, NR, why; rc = 2 }
        BEGIN { rc = 0; n = 0; printf "" > out }
        /^#/ || /^[[:space:]]*$/ { next }
        $1 == "sha256" { next }
        {
            if ($1 !~ /^[0-9a-f]{64}$/ || $2 !~ /^[A-Za-z0-9_.-]+$/ || $3 !~ /^[A-Za-z0-9_.:\/-]+$/) { nm("malformed key"); next }
            if ($4 != "pass" && $4 != "fail" && $4 != "not_measured") { nm("unknown state \"" $4 "\""); next }
            k = $1 OFS $2 OFS $3
            if ((k in st) && st[k] != $4) { nm("key " substr($1, 1, 12) " " $2 " " $3 " is both " st[k] " and " $4); next }
            if (!(k in st)) n++
            st[k] = $4
        }
        END {
            if (rc == 0 && n == 0) { printf "NM    KFR bundle %s measured no check: not_measured\n", dir; rc = 2 }
            if (rc == 0) for (k in st) print k, st[k] > out
            exit rc
        }' "$1/checks.tsv"
}

kfr_judge() { # LIST LBASE HEAD BASE
    local t red=0 nm=0 rc
    t="$(mktemp -d "${TMPDIR:-/tmp}/kfr.XXXXXX")" || { printf 'KFR NOT_MEASURED: no temp dir\n'; return 2; }
    kfr_list "$1" "$t/list" || red=1
    : > "$t/lbase"
    if [ "$2" = none ]; then
        printf '!     KFR BOOTSTRAP: no list at the base ref yet; the shrink-only floor arms when the list lands\n'
    else
        kfr_list "$2" "$t/lbase" > "$t/lbase.log" || { printf 'NM    KFR the list at the base ref is unreadable: shrink-only not measured\n'; nm=1; }
    fi
    : > "$t/head"
    kfr_bundle "$3" "$t/head"; rc=$?
    [ "$rc" -eq 0 ] || { nm=1; printf 'NM    KFR head bundle: not_measured, so no failure here is judged\n'; : > "$t/head"; touch "$t/head.nm"; }
    : > "$t/base"
    if [ "$4" = none ]; then
        printf '!     KFR no last-release bundle: a failure not on the list is UNLISTED, never excused\n'
    else
        kfr_bundle "$4" "$t/base"; rc=$?
        [ "$rc" -eq 0 ] || { nm=1; printf 'NM    KFR base bundle: not_measured, so NEW cannot be told from UNLISTED\n'; : > "$t/base"; }
    fi
    LC_ALL=C awk -F '\t' -v OFS='\t' -v lbase_none="$([ "$2" = none ] && echo 1 || echo 0)" \
        -v head_nm="$([ -f "$t/head.nm" ] && echo 1 || echo 0)" -v red="$red" -v nm="$nm" '
        FILENAME == ARGV[1] { k = $1 OFS $2 OFS $3; L[k] = $4; LT[k] = $0; nl++; next }
        FILENAME == ARGV[2] { LB[$0] = 1; nlb++; next }
        FILENAME == ARGV[3] { k = $1 OFS $2 OFS $3; H[k] = $4; next }
        FILENAME == ARGV[4] { k = $1 OFS $2 OFS $3; B[k] = $4; next }
        function name(k,   a) { split(k, a, OFS); return substr(a[1], 1, 12) " " a[2] " " a[3] }
        END {
            # shrink-only: every list row must already be on the list at the base ref, byte for byte
            if (!lbase_none) for (k in L) if (!(LT[k] in LB)) {
                printf "FAIL  KFR ADDED %s %s: not on the list at the base ref -- the list is shrink-only; a new failure needs a ruling, not a list edit\n", name(k), L[k]; red = 1
            }
            if (!head_nm) {
                for (k in L) {
                    if (!(k in H) || H[k] == "not_measured") { printf "NM    KFR %s %s: listed, not measured on the head\n", name(k), L[k]; nm = 1 }
                    else if (H[k] == "pass") { printf "FAIL  KFR STALE %s %s: listed, and it passes here -- delete the entry, a known failure must not outlive its defect\n", name(k), L[k]; red = 1 }
                    else { printf "KNOWN %s %s: fails, as listed -- counted RED with its ticket, never green\n", name(k), L[k]; known++ }
                }
                for (k in H) if (H[k] == "fail" && !(k in L)) {
                    if (B[k] == "pass") { printf "FAIL  KFR NEW %s: passed in the last release bundle and fails here\n", name(k); new++; red = 1 }
                    else { printf "FAIL  KFR UNLISTED %s: fails here, is not on the list, and no last-release pass to compare -- needs a ruling\n", name(k); unl++; red = 1 }
                }
                for (k in B) if (B[k] == "pass" && (!(k in H) || H[k] == "not_measured")) {
                    printf "NM    KFR %s: passed in the last release bundle, not measured on the head\n", name(k); nm = 1
                }
            }
            s = sprintf("listed=%d known_failing=%d new=%d unlisted=%d", nl, known, new, unl)
            if (red) { printf "KFR RED: %s\n", s; exit 1 }
            if (nm) { printf "KFR NOT_MEASURED: %s\n", s; exit 2 }
            printf "KFR GREEN: %s\n", s; exit 0
        }' "$t/list" "$t/lbase" "$t/head" "$t/base"
    rc=$?
    rm -rf -- "${t:?}"
    return "$rc"
}
