#!/usr/bin/env bash
# deep_nodefault_verdict.sh -- the verdict on `cargo check --workspace --no-default-features` (#4871).
#
# Callers: .github/workflows/deep-nightly.yml job `deep-nodefault` (a nightly-train lane) and
# scripts/release/autopilot.sh step `deep` (T-1). Both run cargo, keep its log and exit status, and
# hand them here; neither counts errors itself any more.
#
# THE DEFECT. Both callers counted errors inline as
#     grep '^error' -A3 LOG | grep '^\s+--> ' | grep -vc 'crates/aprender-distribute/'
# and went red only when that count was non-zero. It counts `-->` LOCATIONS, not errors, so a cargo
# that failed without a located diagnostic -- "error: failed to load manifest", "error: could not
# compile `aprender-core`" whose diagnostics were cut, a killed cargo, an empty log -- counted 0 and
# read GREEN. The exit status was printed and never consulted.
#
# THE RULE. Every line starting `error:` or `error[E....]:` is one error. It is EXPLAINED as the
# standing #3176 class (recorded, not blocking) only when
#   (a) it is a diagnostic whose first `-->` within the next 3 lines is under crates/aprender-distribute/
#       (anchored at the start of the path: crates/aprender-distribute-x/ and docs/aprender-distribute/
#       are outside), or
#   (b) it is exactly the summary  error: could not compile `aprender-distribute` ...
# A `thread '...' panicked` line (a rustc ICE or a cargo panic) is one error, always OUTSIDE.
# Every other error line is OUTSIDE. `warning:` lines never count, wherever they point.
#   GREEN  no error outside, and rc == 0, or rc == 101 (cargo's compile-failure exit) with at least
#          one error line to explain it
#   RED    any error outside; rc != 0 with zero error lines; or rc not in {0, 101} (killed, terminated:
#          the run did not finish, so its log explains nothing)
#   exit 2 LOG missing or unreadable, or RC not a non-negative integer (not_measured; callers read
#          any non-zero as red)
#
# Output: one line,
#   DEEP --no-default-features rc=<rc> errors_total=<n> errors_outside_aprender-distribute=<m> unexplained=<k>
# where k is 1 when rc != 0 and the log holds no error line, or rc is not 0 or 101, else 0.
#
# Usage:
#   deep_nodefault_verdict.sh LOG RC                the verdict (exit 0 green, 1 red, 2 not_measured)
#   deep_nodefault_verdict.sh --self-test [--impl old]
#                                                   the case table; `--impl old` runs it against the
#                                                   pre-#4871 inline rule and must FAIL (the defect)
#   deep_nodefault_verdict.sh --mutants             each planted wrong rule must break a case row
# Exit codes of --self-test / --mutants: 0 all rows pass / all mutants killed, 1 a row broke / a mutant
# survived, 2 environment (a mutant edit did not apply), 3 usage.
set -uo pipefail

SELF="${BASH_SOURCE[0]}"
CARGO_COMPILE_FAILED=101   # cargo's exit status when compilation fails (an exit code, not a count)
IMPL=new   # m:old-impl   (only `--self-test --impl old` changes it; no environment override)

# count LOG -> prints "<errors_total> <errors_outside>"
count_new() {
    awk '
    { sub(/\r$/, ""); gsub(/\033\[[0-9;]*[A-Za-z]/, ""); L[NR] = $0 }
    END {
        tot = 0; out = 0
        for (i = 1; i <= NR; i++) {
            if (L[i] ~ /^thread .* panicked/) { tot++; out++; continue }   # m:panic-ignored
            if (L[i] !~ /^error(\[E[0-9]+\])?:/) continue   # m:ignore-E-form m:warning-counts
            tot++
            if (L[i] ~ /^error: could not compile `aprender-distribute`( |$)/) continue   # m:any-could-not-compile m:summary-unanchored
            loc = ""
            for (j = i + 1; j <= i + 3 && j <= NR; j++) {
                if (L[j] ~ /^(error|warning)/) break   # m:lookahead-no-stop
                if (L[j] ~ /^[ \t]*--> /) { loc = L[j]; sub(/^[ \t]*--> /, "", loc); break }
            }
            if (index(loc, "crates/aprender-distribute/") == 1) continue   # m:unanchored-path
            out++
        }
        printf "%d %d\n", tot, out
    }' "$1"
}

# The pre-#4871 inline rule, kept only so --self-test --impl old can show the defect it had.
count_old() {
    local other
    other=$(grep -E '^error' -A3 "$1" | grep -E '^\s+--> ' | grep -vc 'crates/aprender-distribute/' || true)
    printf '%s %s\n' "$other" "$other"
}

# verdict LOG RC -> prints the DEEP line; rc 0 green, 1 red, 2 not_measured
verdict() {
    local log=${1:-} rc=${2:-} tot out k=0 counts
    case "$rc" in ''|*[!0-9]*) printf 'DEEP --no-default-features NOT_MEASURED: rc %s is not an integer\n' "${rc:-<empty>}"; return 2 ;; esac
    if [ ! -f "$log" ] || [ ! -r "$log" ]; then   # m:missing-log
        printf 'DEEP --no-default-features NOT_MEASURED: no readable log %s\n' "${log:-<empty>}"; return 2
    fi
    if [ "$IMPL" = old ]; then counts=$(count_old "$log"); else counts=$(count_new "$log"); fi || return 2
    read -r tot out <<< "$counts"
    # main's old rule never read rc; --impl old reproduces that exactly.
    if [ "$IMPL" != old ] && [ "$rc" != 0 ] && [ "$tot" = 0 ]; then k=1; fi   # m:zero-errors-rule
    # cargo exits 101 when compilation fails; any other non-zero (137 killed, 143 terminated, ...) means the
    # run did not finish, so its log cannot explain the failure, whatever errors it holds.
    if [ "$IMPL" != old ] && [ "$rc" != 0 ] && [ "$rc" != "$CARGO_COMPILE_FAILED" ]; then k=1; fi   # m:abnormal-rc
    printf 'DEEP --no-default-features rc=%s errors_total=%s errors_outside_aprender-distribute=%s unexplained=%s\n' \
        "$rc" "$tot" "$out" "$k"
    if [ "$out" = 0 ] && [ "$k" = 0 ]; then return 0; fi
    return 1
}

# ---------------------------------------------------------------- case table
PASS=0; FAIL=0; RAN=" "
# The rows the table must hold, named here and not counted from the table: a row deleted from the
# table is MISSING (red), and a row added there without a name here is UNLISTED (red).
ROW_IDS="C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C18"
# row NAME WANT_RC LOGFILE RC DESCRIPTION
row() {
    local name=$1 want=$2 log=$3 rc=$4 desc=$5 got line
    RAN="$RAN$name "
    line=$(verdict "$log" "$rc"); got=$?
    if [ "$got" = "$want" ]; then PASS=$((PASS + 1)); printf 'ok     %-4s want=%s got=%s  %s\n' "$name" "$want" "$got" "$desc"
    else FAIL=$((FAIL + 1)); printf 'BROKE  %-4s want=%s got=%s  %s\n         %s\n' "$name" "$want" "$got" "$desc" "$line"; fi
}

selftest() {
    # d stays global: the EXIT trap runs after this function returned, when a local is gone.
    d=$(mktemp -d) || return 2
    trap 'rm -rf -- "${d:?}"' EXIT

    cat > "$d/clean.log" <<'LOG'
    Checking aprender-core v0.70.0 (crates/aprender-core)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1m 02s
LOG
    cat > "$d/dist-only.log" <<'LOG'
    Checking aprender-distribute v0.70.0 (crates/aprender-distribute)
error[E0433]: failed to resolve: use of undeclared crate or module `tokio`
  --> crates/aprender-distribute/src/net.rs:12:5
   |
12 |     tokio::spawn(async move {
error: cannot find macro `info` in this scope
  --> crates/aprender-distribute/src/lib.rs:40:9
   |
error: could not compile `aprender-distribute` (lib) due to 2 previous errors
LOG
    { cat "$d/dist-only.log"; printf "%s\n" "thread 'rustc' panicked at compiler/rustc_middle/src/ty/mod.rs:1:1:"; } > "$d/dist-panic.log"
    printf "%s\n" "error: linking with \`cc\` failed: exit status: 1" "warning: unused import: \`std::io\`" "  --> crates/aprender-distribute/src/net.rs:3:5" > "$d/stop-warn.log"
    printf "%s\n" "error: failed to run custom build command for \`aprender-core\`" "error[E0433]: failed to resolve: use of undeclared crate or module \`tokio\`" "  --> crates/aprender-distribute/src/net.rs:12:5" > "$d/stop-error.log"
    : > "$d/empty.log"
    cat > "$d/manifest.log" <<'LOG'
error: failed to load manifest for workspace member `crates/aprender-core`

Caused by:
  failed to parse manifest at `crates/aprender-core/Cargo.toml`
LOG
    cat > "$d/core-summary.log" <<'LOG'
    Checking aprender-core v0.70.0 (crates/aprender-core)
error: could not compile `aprender-core` (lib) due to 1 previous error
LOG
    cat > "$d/dist-plus-core.log" <<'LOG'
error[E0433]: failed to resolve: use of undeclared crate or module `tokio`
  --> crates/aprender-distribute/src/net.rs:12:5
   |
error[E0412]: cannot find type `Tensor` in this scope
  --> crates/aprender-core/src/nn/mod.rs:88:17
   |
error: could not compile `aprender-distribute` (lib) due to 1 previous error
error: could not compile `aprender-core` (lib) due to 1 previous error
LOG
    cat > "$d/killed.log" <<'LOG'
    Checking aprender-core v0.70.0 (crates/aprender-core)
warning: unused import: `std::fmt`
 --> crates/aprender-core/src/lib.rs:3:5
  |
LOG
    cat > "$d/warn-outside.log" <<'LOG'
warning: unused variable: `x`
  --> crates/aprender-core/src/lib.rs:9:9
   |
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 40s
LOG
    cat > "$d/lookalike-crate.log" <<'LOG'
error[E0425]: cannot find value `n` in this scope
  --> crates/aprender-distribute-x/src/lib.rs:5:13
   |
error: could not compile `aprender-distribute` (lib) due to 1 previous error
LOG
    cat > "$d/lookalike-docs.log" <<'LOG'
error: unknown start of token: \u{201c}
  --> docs/aprender-distribute/example.rs:1:1
   |
error: could not compile `aprender-distribute` (example "e") due to 1 previous error
LOG
    cat > "$d/lookalike-summary.log" <<'LOG'
error: could not compile `aprender-distribute-x` (lib) due to 1 previous error
LOG
    cat > "$d/truncated.log" <<'LOG'
error[E0433]: failed to resolve: use of undeclared crate or module `tokio`
  --> crates/aprender-distribute/src/net.rs:12:5
   |
error: could not compile `aprender-distribute` (lib) due to 1 previous error
error[E0599]: no method named `foo` found for struct `Matrix`
  --> crates/aprender-core/src/primitives/matrix.rs:201:14
LOG

    row C1  0 "$d/clean.log"             0   "clean, rc 0"
    row C2  0 "$d/dist-only.log"         101 "rc 101, only #3176 diagnostics + could not compile aprender-distribute (standing class)"
    row C3  1 "$d/empty.log"             101 "rc 101, empty log (unexplained failure)"
    row C4  1 "$d/manifest.log"          101 "rc 101, failed to load manifest, no --> location"
    row C5  1 "$d/core-summary.log"      101 "rc 101, could not compile aprender-core alone, no located diagnostic"
    row C6  1 "$d/dist-plus-core.log"    101 "rc 101, a distribute diagnostic plus an aprender-core diagnostic"
    row C7  1 "$d/killed.log"            137 "rc 137 (killed), only warnings"
    row C8  0 "$d/warn-outside.log"      0   "a warning: pointing outside never counts"
    row C9  1 "$d/lookalike-crate.log"   101 "crates/aprender-distribute-x/ is outside"
    row C10 1 "$d/lookalike-docs.log"    101 "docs/aprender-distribute/ is outside"
    row C11 1 "$d/lookalike-summary.log" 101 "could not compile aprender-distribute-x is outside"
    row C12 1 "$d/truncated.log"         101 "an error[E...] outside whose summary line is cut is still outside"
    row C13 2 "$d/no-such.log"           0   "missing LOG = not_measured"
    row C14 2 "$d/clean.log"             abc "non-integer RC = not_measured"
    row C15 1 "$d/dist-only.log"         137 "rc 137 (killed) with only #3176 errors: an unfinished run is red"
    row C16 1 "$d/dist-panic.log"        101 "rc 101, only #3176 errors plus a panicked thread: a panic is outside"
    row C17 1 "$d/stop-warn.log"         101 "an unlocated error followed by a warning into distribute: the warning does not explain it"
    row C18 1 "$d/stop-error.log"        101 "an unlocated error followed by a distribute error: the next error does not explain it"   # m:drop-row

    printf 'deep_nodefault_verdict.sh --self-test (impl=%s): %s/%s rows pass\n' "$IMPL" "$PASS" "$((PASS + FAIL))"
    # Every named row ran, and every row that ran is named (ROW_IDS above).
    local id ids_ok=1
    for id in $ROW_IDS; do
        case "$RAN" in *" $id "*) ;; *) printf 'MISSING %s: named in ROW_IDS, not run by the table\n' "$id"; ids_ok=0 ;; esac
    done
    for id in $RAN; do
        case " $ROW_IDS " in *" $id "*) ;; *) printf 'UNLISTED %s: run by the table, not named in ROW_IDS\n' "$id"; ids_ok=0 ;; esac
    done
    [ "$FAIL" = 0 ] && [ "$ids_ok" = 1 ] && [ -n "$ROW_IDS" ] && return 0
    return 1
}

# ---------------------------------------------------------------- mutants
# One wrong rule per row: NAME, then a sed -E substitution applied ONLY to this script's line tagged
# `# m:NAME` (the table's own rows carry no tag, so they are never edited). An edit that does not
# change the copy is an environment error (exit 2), never a survivor. Each planted copy's --self-test
# must break a row.
#   zero-errors-rule       rc != 0 with no error line is read green again
#   unanchored-path        any path CONTAINING aprender-distribute is explained
#   old-impl               main's pre-#4871 inline rule: count only --> locations, ignore rc
#   any-could-not-compile  "could not compile" of ANY crate is explained
#   ignore-E-form          only `error:` lines count; `error[E....]:` is not an error
#   warning-counts         `warning:` lines count as errors
#   summary-unanchored     "could not compile `aprender-distribute-x`" is explained
#   missing-log            a missing LOG is read as an empty one instead of not_measured
#   abnormal-rc            a killed or terminated cargo (rc not 0 or 101) whose log holds only #3176 errors is green
#   panic-ignored          a panicked thread (rustc ICE, cargo panic) next to only #3176 errors is green
#   lookahead-no-stop      the --> look-ahead runs past the next error/warning line and borrows its location
#   drop-row               a row deleted from the table (C18) is no longer missed
mutants() {
    local name expr killed=0 total=0 first
    d=$(mktemp -d) || return 2
    trap 'rm -rf -- "${d:?}"' EXIT
    while read -r name expr; do
        [ -n "$name" ] || continue
        total=$((total + 1))
        sed -E "/# (m:[a-zA-Z-]+ )*m:${name}( |\$)/${expr}" "$SELF" > "$d/$name.sh"
        if cmp -s "$d/$name.sh" "$SELF"; then printf 'ERROR  mutant %s did not apply\n' "$name"; return 2; fi
        bash "$d/$name.sh" --self-test > "$d/$name.out" 2>&1
        if [ $? != 0 ]; then
            killed=$((killed + 1))
            first=$(grep -E '^(BROKE|MISSING|UNLISTED)' "$d/$name.out" | awk '{printf "%s%s", sep, $2; sep=","}')
            printf 'killed    %-22s broken rows: %s\n' "$name" "${first:-<none>}"
        else
            printf 'SURVIVED  %s\n' "$name"
        fi
    done <<'MUTANTS'
zero-errors-rule      s/then k=1; fi/then k=0; fi/
unanchored-path       s#index\(loc, "crates/aprender-distribute/"\) == 1#loc ~ /aprender-distribute/#
old-impl              s/^IMPL=new/IMPL=old/
any-could-not-compile s#could not compile `aprender-distribute`\( \|\$\)/#could not compile /#
ignore-E-form         s#\^error\(\\\[E\[0-9\]\+\\\]\)\?:#^error:#
warning-counts        s#/\^error\(#/^(error|warning)(#
summary-unanchored    s#`aprender-distribute`\( \|\$\)/#`aprender-distribute/#
missing-log           s#^( +)if \[ ! -f.*$#\1[ -r "$log" ] || log=/dev/null; if false; then#
abnormal-rc           s/then k=1; fi/then k=0; fi/
panic-ignored         s/.*//
lookahead-no-stop     s/.*//
drop-row              s/.*//
MUTANTS
    printf 'deep_nodefault_verdict.sh --mutants: %s/%s killed\n' "$killed" "$total"
    [ "$killed" = "$total" ] && [ "$total" -ge 5 ]
}

case "${1:-}" in
    --self-test)
        case "${2:-}" in
            '') ;;
            --impl) case "${3:-}" in old|new) IMPL=$3 ;; *) echo "usage: --self-test [--impl old|new]" >&2; exit 3 ;; esac ;;
            *) echo "usage: --self-test [--impl old|new]" >&2; exit 3 ;;
        esac
        selftest; exit $? ;;
    --mutants) mutants; exit $? ;;
    -h|--help) sed -n '2,38p' "$SELF"; exit 0 ;;
    '') echo "usage: deep_nodefault_verdict.sh LOG RC | --self-test [--impl old] | --mutants" >&2; exit 3 ;;
    *) if [ $# -ne 2 ]; then echo "usage: deep_nodefault_verdict.sh LOG RC" >&2; exit 3; fi
       verdict "$1" "$2" ;;
esac
