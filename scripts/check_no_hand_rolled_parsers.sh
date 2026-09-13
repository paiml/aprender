#!/usr/bin/env bash
# check_no_hand_rolled_parsers.sh — every binary must parse its arguments with
# clap derive. Hand-rolling argv is banned.
#
# WHY THIS EXISTS
# ---------------
# A hand-rolled `match args[1]` parser in `simular` silently DROPPED `--seed`:
#
#   * an unknown flag fell through a `_ => i += 1` catch-all and vanished
#   * `--seed` given with no value was discarded
#   * `--seed notanumber` became `None`, i.e. the default, instead of an error
#   * `verify --runs N` was only honoured at argv[3]
#
# None of that is visible from the outside. The command exits 0 and does the
# wrong thing, which is the worst failure mode a CLI has. The project rule is
# therefore absolute: argument parsing must be DECLARATIVE (clap derive), so the
# accepted grammar is data rather than control flow.
#
# Found by `dogfood_surfaces.sh` and then confirmed by this scan, four binaries
# were still hand-rolled after simular was converted:
#
#   aprender-compute-xtask    --help exited 1
#   aprender-ptx-debug        (converted in #2520)
#   aprender-qa-certify       apr-qa-readme-sync
#   aprender-zram-generator   --help printed 0 BYTES, and an unknown flag was
#                             accepted at exit 0 -- so a typo'd flag was treated
#                             as one of its DIRECTORY arguments
#
# WHY A STRUCTURAL SCAN AND NOT ONLY A BEHAVIOURAL PROBE
# -----------------------------------------------------
# dogfood_surfaces.sh probes behaviour: it feeds each binary an unknown flag and
# demands rejection. That catches the worst hand-rolled parsers but not a
# careful one that happens to reject unknown flags today and regresses later.
# This scan bans the CONSTRUCT, so the guarantee does not depend on a probe
# happening to hit the broken case. The two are complementary and both run.
#
# The universe comes from `cargo metadata`, never from a glob of src/bin or a
# grep for [[bin]] -- auto-discovered binaries have no [[bin]] stanza, and
# `autobins = false` deletes ones that do. A guard whose universe is built from
# the wrong side is the recurring defect this repo keeps paying for.
#
#   bash scripts/check_no_hand_rolled_parsers.sh              # check
#   bash scripts/check_no_hand_rolled_parsers.sh --self-test  # case table
#   bash scripts/check_no_hand_rolled_parsers.sh --update     # re-baseline (shrink only)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${REPO_ROOT}/scripts/hand_rolled_parsers_baseline.txt"

# Packages that build a binary AND read argv directly without clap deriving a
# Parser. One package name per line, sorted.
hand_rolled_in() {
    local root="$1"
    ( cd "$root" && cargo metadata --no-deps --format-version 1 2>/dev/null ) | python3 -c '
import json,sys,os,re,glob
md=json.load(sys.stdin)
out=[]
for p in md["packages"]:
    if not any("bin" in t["kind"] for t in p["targets"]):
        continue
    d=os.path.dirname(p["manifest_path"])
    src=""
    for f in glob.glob(d+"/src/**/*.rs", recursive=True):
        try:
            src+=open(f, errors="ignore").read()
        except OSError:
            pass
    # Declarative parsing means CLAP derive. Two signals, both required:
    #   1. the package actually depends on clap
    #   2. the source derives clap Parser
    #
    # An earlier version accepted a bare ::parse() as proof of clap. That is a
    # FALSE NEGATIVE and it hid the worst offender in the repo: the simular
    # main.rs calls run_cli(Args::parse()) where Args::parse is its OWN
    # hand-rolled function. The guard reported simular as clap while it was
    # still doing match args[1] with .parse().ok().unwrap_or(default) on
    # --seed, the exact defect the rule exists to ban. Row 4 of the case table
    # pins this. NOTE: no apostrophes in this block, it is inside a
    # single-quoted heredoc and one would terminate it.
    deps = set()
    for k in ("dependencies",):
        for dep in p.get(k, []) or []:
            deps.add(dep.get("name",""))
    has_clap = "clap" in deps
    derives_parser = re.search(r"derive\([^)]*\bParser\b", src) is not None
    declarative = has_clap and derives_parser
    reads_argv  = re.search(r"env::args\(\)", src) is not None
    if reads_argv and not declarative:
        out.append(p["name"])
for n in sorted(set(out)):
    print(n)
'
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    fails=0
    TD="$(mktemp -d)" || exit 1
    case "$TD" in /tmp/*|/var/folders/*) : ;; *) printf 'bad tmp\n'; exit 1 ;; esac
    trap 'rm -rf "${TD:?}"' EXIT

    # $4 = "clap" to give the fixture a real clap dependency. A crate that
    # derives Parser necessarily depends on clap, so a fixture that derives it
    # WITHOUT the dep is not a realistic clap crate -- and row 3 correctly
    # failed until this was added.
    make_crate() {
        local dir="$1" name="$2" body="$3" dep="${4:-}"
        mkdir -p "$dir/src"
        {
            printf '[package]\n'
            printf 'name = "%s"\n' "$name"
            printf 'version = "0.0.0"\n'
            printf 'edition = "2021"\n'
            if [ "$dep" = "clap" ]; then
                printf '\n[dependencies]\n'
                printf 'clap = { version = "4", features = ["derive"] }\n'
            fi
        } > "$dir/Cargo.toml"
        printf '%s\n' "$body" > "$dir/src/main.rs"
    }

    # Row 1: a hand-rolled parser must be REPORTED.
    W1="$TD/w1"
    make_crate "$W1" hand-rolled-probe \
'use std::env;
fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("run") => println!("run"),
        _ => println!("usage"),
    }
}'
    got="$(hand_rolled_in "$W1" | tr '\n' ' ')"
    if [ "$got" = "hand-rolled-probe " ]; then
        printf 'ok    row 1 a hand-rolled argv parser is reported\n'
    else
        printf 'FAIL  row 1 got [%s], expected hand-rolled-probe\n' "$got"; fails=1
    fi

    # Row 2 is the control. Without it row 1 passes even if this reported EVERY
    # binary crate it saw, and the guard could never go green.
    W2="$TD/w2"
    make_crate "$W2" clap-probe \
'use clap::Parser;
#[derive(Parser)]
struct Cli { #[arg(long)] name: String }
fn main() { let _c = Cli::parse(); }' clap
    if [ -z "$(hand_rolled_in "$W2")" ]; then
        printf 'ok    row 2 a clap-derive CLI is NOT reported\n'
    else
        printf 'FAIL  row 2 reported a clap CLI: %s\n' "$(hand_rolled_in "$W2" | tr '\n' ' ')"; fails=1
    fi

    # Row 3: reading argv is fine as long as parsing is declarative -- plenty of
    # clap CLIs also touch env::args() for a program name or a passthrough.
    W3="$TD/w3"
    make_crate "$W3" clap-and-argv-probe \
'use clap::Parser;
use std::env;
#[derive(Parser)]
struct Cli { #[arg(long)] name: String }
fn main() {
    let _prog = env::args().next();
    let _c = Cli::parse();
}' clap
    if [ -z "$(hand_rolled_in "$W3")" ]; then
        printf 'ok    row 3 a clap CLI that also reads argv is NOT reported\n'
    else
        printf 'FAIL  row 3 false positive on a clap CLI that reads argv\n'; fails=1
    fi

    # Row 4 is the one that matters most: a crate whose own type has a `parse()`
    # method must NOT be mistaken for clap. This is the real simular defect --
    # `run_cli(Args::parse())` where Args::parse is hand-written. An earlier
    # detector accepted any `::parse()` and so reported the repo's worst
    # hand-rolled parser as compliant.
    W4="$TD/w4"
    make_crate "$W4" fake-parse-probe \
'use std::env;
struct Args { verbose: bool }
impl Args {
    fn parse() -> Self {
        let a: Vec<String> = env::args().collect();
        Args { verbose: a.iter().any(|s| s == "-v") }
    }
}
fn main() { let _a = Args::parse(); }'
    got="$(hand_rolled_in "$W4" | tr '\n' ' ')"
    if [ "$got" = "fake-parse-probe " ]; then
        printf 'ok    row 4 a hand-rolled Args::parse() is NOT mistaken for clap\n'
    else
        printf 'FAIL  row 4 got [%s]; a hand-rolled parse() passed as clap\n' "$got"; fails=1
    fi

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (4/4)\n'
    exit 0
fi

printf '=== every binary must parse argv with clap derive (check_no_hand_rolled_parsers.sh) ===\n'

TOTAL=$( ( cd "$REPO_ROOT" && cargo metadata --no-deps --format-version 1 2>/dev/null ) | python3 -c '
import json,sys
md=json.load(sys.stdin)
print(sum(1 for p in md["packages"] if any("bin" in t["kind"] for t in p["targets"])))
')

# Vacuity: an enumeration that found no binary crates would report zero
# hand-rolled parsers and look like a pass.
if [ "${TOTAL:-0}" -lt 15 ]; then
    printf '\nFAIL (vacuity): found %s binary crate(s), expected 15+.\n' "${TOTAL:-0}"
    printf 'The ENUMERATION is broken, not the code. Fix it rather than this floor.\n'
    exit 1
fi

FOUND="$(hand_rolled_in "$REPO_ROOT")"
count=$(printf '%s\n' "$FOUND" | grep -c . || true)

printf '%s binary crate(s) scanned, %s hand-rolled\n' "$TOTAL" "$count"

if [ "${1:-}" = "--update" ]; then
    printf '%s\n' "$FOUND" | grep . > "$BASELINE" || : > "$BASELINE"
    printf 'baseline set to %s\n' "$count"
    exit 0
fi

# THE RATCHET IS A PROPERTY OF THE DIFF, NOT OF THE TREE.
#
# Everything above compares the scan against the baseline AS IT STANDS IN THE
# WORKING TREE, and that is not a ratchet. NEW (a finding with no entry) and
# STALE (an entry with no finding) are the only two properties a working tree
# can answer, and a commit that appends one line AND lands the matching
# violation satisfies both at once: not new, because it is baselined; not
# stale, because the finding is real.
#
# Measured, not argued: appending one entry cloned from this file's own last
# real entry returned rc=0 from this guard, under its own words:
#     "--update # re-baseline (shrink only)"
# Twelve guards in scripts/ failed the same probe.
#
# So growth is now compared against merge-base(HEAD, origin/main), falling
# back to the origin/main TIP because CI checks out shallow — a ref this
# branch cannot rewrite, and never the branch against itself.
# shellcheck source=scripts/lib_baseline_ratchet.sh
. "${REPO_ROOT}/scripts/lib_baseline_ratchet.sh" || exit 1
baseline_ratchet_check "${REPO_ROOT}" scripts/hand_rolled_parsers_baseline.txt set || exit 1

if [ ! -f "$BASELINE" ]; then
    printf 'FAIL: %s missing. Run --update once to establish it.\n' "$BASELINE"
    exit 1
fi
baseline_count=$(grep -cvE '^\s*(#|$)' "$BASELINE" || true)

if [ "$count" -gt "$baseline_count" ]; then
    printf '\nFAIL: hand-rolled parsers grew %s -> %s.\n\n' "$baseline_count" "$count"
    comm -13 <(grep -vE '^\s*(#|$)' "$BASELINE" | LC_ALL=C sort) \
             <(printf '%s\n' "$FOUND" | grep .) | sed 's|^|  NEW: |'
    printf '\nUse clap derive. A hand-rolled parser accepts what it should reject\n'
    printf 'and silently defaults what it cannot parse -- see the header.\n'
    exit 1
fi

if [ "$count" -lt "$baseline_count" ]; then
    printf '\nImproved: %s -> %s. Run --update to record it.\n' "$baseline_count" "$count"
fi

printf 'PASS (ratcheted)\n'
exit 0
