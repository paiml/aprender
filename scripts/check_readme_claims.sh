#!/usr/bin/env bash
# FALSIFY-README-001..004: verify README.md quantitative claims against
# live repository state. Bound by `contracts/readme-claims-v1.yaml`.
#
# Usage:
#   bash scripts/check_readme_claims.sh                     # all claims
#   bash scripts/check_readme_claims.sh --claim <name>      # one claim
#   bash scripts/check_readme_claims.sh --regen             # print new numbers for manual README edit
#
# Claims:
#   crate_count        → `cargo metadata --no-deps` members == README "N workspace crates"
#   contract_count     → `find contracts/ -name '*.yaml' | wc -l` == README "M provable contracts"
#   cli_command_count  → `apr --help` subcmd count == README "K CLI commands"
#   cookbook_link      → README.md mentions `apr-cookbook`

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
README="${README_PATH:-$REPO_ROOT/README.md}"   # README_PATH: a fixture, for --self-test
# G-11 (PMAT-1062, driver R2 2026-09-06): the README's counts are a RATCHET, not an
# equality. A row PR that adds a contract may leave the README LAGGING (claimed <
# measured); it may never OVERSTATE (claimed > measured). The orchestrator docs commit
# regenerates the counts after each merge and verifies them with --exact.
EXACT="${README_EXACT:-0}"
compare_count() { # compare_count <falsify id> <name> <claimed> <measured> <what measured is>
  local fid=$1 name=$2 claimed=$3 measured=$4 what=$5
  if [[ "$claimed" -gt "$measured" ]]; then
    echo "FAIL $fid $name: README claims $claimed, $what has $measured — the README may lag, never overstate" >&2; return 1
  fi
  if [[ "$claimed" -lt "$measured" ]]; then
    if [[ "$EXACT" = 1 ]]; then echo "FAIL $fid $name: README claims $claimed, $what has $measured (--exact: the orchestrator docs commit regenerates counts)" >&2; return 1; fi
    echo "PASS $fid $name: $measured (README lags at $claimed; the orchestrator docs commit regenerates it)"; return 0
  fi
  echo "PASS $fid $name: $measured"
}

# A cargo exit is classified before any verdict names the README. See
# scripts/cargo_classify.sh. The case table is armed on the normal path because
# no workflow invokes a --self-test here, and a case table nothing runs is the
# vacuous-scan class; re-mutated in this scope rather than inheriting another
# guard's green.
. "$REPO_ROOT/scripts/cargo_classify.sh" || exit 1
cargo_classify_selftest --quiet || exit 1

# The comparand resolver, and it is the ONLY one in this repository (threat
# model A2 / APR-RATCHET-D2-001 §4, docs/audits/threat-model-bse-03-ratchets.md).
# A guard that resolves its own comparand can be told to resolve HEAD; this one
# is shared with every baseline ratchet in the tree and its case table lives in
# scripts/check_baseline_ratchets.sh.
. "$REPO_ROOT/scripts/lib_baseline_ratchet.sh" || exit 1

# BSE-03 phase A: the README's contract count is DERIVED. scripts/readme_sync.sh
# writes the text between these markers and a human writes neither the markers
# nor the number. They are INLINE (both on one line) because a marker on its own
# line ends a GFM table and opens an HTML block, and the count is stated inside
# README.md's metrics table.
CONTRACT_BLOCK_START='<!-- CONTRACT_COUNT_START -->'
CONTRACT_BLOCK_END='<!-- CONTRACT_COUNT_END -->'

if [[ ! -f "$README" ]]; then
  echo "error: $README not found" >&2
  exit 2
fi

# --- measurements (authoritative, from filesystem / live apr binary) ---

measured_crate_count() {
  # `cargo metadata --no-deps` - the WORKSPACE members, which is what
  # "workspace crates" means. NOT `find crates/ -type d`.
  #
  # Those two numbers genuinely differ, and the directory count is the wrong
  # one: 82 directories, 81 with a Cargo.toml, 78 workspace members (4 are
  # `exclude`d from the workspace, 1 has no manifest). README.md:43 documents
  # the correct method AND warns against the directory count in the same
  # sentence - this function was using exactly the method the README told it
  # not to, so it reported the README as drifted while the README was right.
  # Wiring it in that state would have forced README to claim 82 workspace
  # crates, which is false. A gate that enforces the wrong answer is worse
  # than no gate.
  #
  # cargo's stderr used to go to /dev/null and its exit status was never read,
  # so a `cargo metadata` that DIED left `measured` empty and the caller
  # printed "README claims 78, cargo metadata --no-deps has  workspace
  # members" -- a verdict about the README, from a measurement that never
  # happened. Same class as the facade gate that blocked every PR on
  # 2026-08-27. Now: one invocation, rc read directly, ENV named as ENV.
  local md err rc
  md="$(mktemp)"; err="$(mktemp)"
  (cd "$REPO_ROOT" && cargo metadata --no-deps --format-version 1 > "$md" 2> "$err")
  rc=$?
  if [ "$rc" -ne 0 ] || [ ! -s "$md" ]; then
    if [ "$( classify_cargo_failure "$err" )" = 'ENV' ]; then
      report_cargo_env_failure "$err" 'the workspace crate count' >&2
    else
      echo "FAIL: cargo metadata exited $rc and the crate count could not be measured." >&2
      sed 's/^/  | /' "$err" >&2
    fi
    rm -f "$md" "$err"
    return 1
  fi
  python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["packages"]))' "$md"
  rm -f "$md" "$err"
}

measured_contract_count() {
  find "$REPO_ROOT/contracts" -name "*.yaml" | wc -l | tr -d ' '
}

measured_cli_command_count() {
  # Counted from the COMMAND CONTRACT, not by building and running apr.
  #
  # This used to be `cargo run --quiet -p apr-cli --bin apr -- --help`. Two
  # problems, both of which bit on 2026-08-19:
  #
  #   1. ci.yml runs this script in a bare `run:` step whose own comment says
  #      "Text-only, no build." A full `cargo run` of apr-cli is not text-only:
  #      it took 14 minutes in guard-runner-labels and then failed, and cargo is
  #      not reliably on PATH for raw run: steps on these runners (the same
  #      note appears in coverage-nightly.yml).
  #   2. When it failed, the caller did `|| return $?` with NO message, so the
  #      job went red printing nothing at all for FALSIFY-README-003 -- between
  #      a PASS for 002 and a PASS for 004. A red gate with no stated reason.
  #
  # The contract is the designated registry for this surface
  # (contracts/apr-cli-commands-v1.yaml, §commands), and its equivalence to the
  # real binary is ALREADY enforced elsewhere: FALSIFY-CLI-001 asserts every
  # listed command responds to --help, FALSIFY-CLI-002 asserts every command in
  # `apr --help` is listed. Those run in `cargo test -p apr-cli --test
  # cli_commands`, gated on ci.yml's integration line. So reading the contract
  # here preserves the guarantee and drops the build; if the two ever diverge,
  # CLI-001/002 fail, which is where that defect belongs.
  #
  # Parse the YAML. `grep -c '^  - name:'` reports 111 because other same-indent
  # `name:` keys exist in the file -- the contract says so in its own prose.
  #
  # `help` is EXCLUDED, as before: clap generates it automatically, it is not
  # one of apr's commands and has no chapter, contract or implementation.
  # Counting it once made this check report 104 against a README claiming 103 --
  # the README was right and this function was off by exactly clap's freebie.
  # The contract does not list `help`, so this is now true by construction.
  python3 -c '
import sys, yaml
with open(sys.argv[1]) as fh:
    doc = yaml.safe_load(fh)
cmds = doc.get("commands") or []
names = {c.get("name") for c in cmds if isinstance(c, dict) and c.get("name")}
names.discard("help")
if not names:
    sys.exit(1)          # empty parse is a FAILED measurement, never a zero count
print(len(names))
' "$REPO_ROOT/contracts/apr-cli-commands-v1.yaml"
}

measured_cookbook_link_present() {
  # True if README mentions apr-cookbook anywhere (link, path, etc.)
  if grep -Fq "apr-cookbook" "$README"; then
    echo 1
  else
    echo 0
  fi
}

# --- claim extractors (read the README) ---

claimed_crate_count() {
  # Look for pattern "**N** workspace crates"
  local crate_re='\*\*[0-9]+\*\* workspace crates'
  local num_re='[0-9]+'
  grep -oE "$crate_re" "$README" | grep -oE "$num_re" | head -1
}

# EVERY contract count the README AUTHORS, one per line, deduplicated. The
# GENERATED block is stripped first, so what this returns is only what a human
# wrote by hand -- the two are judged by different rules (a generated number may
# not lag; an authored one may, G-11).
#
# This used to match only `**M** provable contracts` -- the bold table form --
# and then `head -1`. The README carried THREE different counts and the guard
# saw one of them: 1771 in the table (correct), 1158 at line 225 ("# 1158
# provable YAML contracts", wrong by 613) and 1767 at line 256 ("1767 contracts
# across inference, training..."). A drift detector that reads one of three
# claims reports claim discipline it is not providing.
#
# Now: any number immediately preceding "contract(s)", optionally through one
# qualifier word ("provable YAML contracts"), with markdown bold stripped.
contract_block_strip() {
  # README.md with every generated block REMOVED; what is left is authored prose
  sed -E "s|${CONTRACT_BLOCK_START}[^<]*${CONTRACT_BLOCK_END}||g" "$README"
}

contract_block_counts() {
  # the body of each generated block, one per line, deduplicated
  grep -oE "${CONTRACT_BLOCK_START}[^<]*${CONTRACT_BLOCK_END}" "$README" \
    | sed -E "s|${CONTRACT_BLOCK_START}||; s|${CONTRACT_BLOCK_END}||" \
    | sort -u
}

claimed_contract_counts() {
  contract_block_strip \
    | grep -oiE '[0-9]+\*{0,2}( +[a-z]+){0,2} +contracts?\b' \
    | grep -oE '^[0-9]+' \
    | sort -un
}

claimed_cli_command_count() {
  # Look for pattern "**K** CLI commands"
  local cli_re='\*\*[0-9]+\*\* CLI commands'
  local num_re='[0-9]+'
  grep -oE "$cli_re" "$README" | grep -oE "$num_re" | head -1
}

# --- check runners ---

check_crate_count() {
  local measured claimed
  # `|| return 1` is required: without it the assignment swallows the ENV/vacuity
  # verdict above and the comparison proceeds against an empty measurement.
  measured=$(measured_crate_count) || return 1
  claimed=$(claimed_crate_count)
  if [[ -z "$claimed" ]]; then
    echo "FAIL FALSIFY-README-001 crate_count: README lacks '**N** workspace crates' claim (pattern mismatch)" >&2
    return 1
  fi
  compare_count FALSIFY-README-001 crate_count "$claimed" "$measured" "cargo metadata --no-deps"
}

# The measurement of ONE revision (BSE-03, D2 / APR-RATCHET-D2-001 §1): the
# count is a property of a TREE, read from the object store. The file on disk is
# never the comparand, and no literal anywhere carries the answer.
#
# `git archive <rev> -- contracts` is the pristine materialisation the normaliser
# names, taken as a tar STREAM rather than extracted to a scratch checkout: the
# count is a property of the listing, extracting 15 MB per revision twice a run
# buys nothing, and a stream cannot be contaminated by the working tree at all.
measure_contract_count_rev() { # measure_contract_count_rev <rev>
  local rev="$1" n=""
  n=$(git -C "$REPO_ROOT" archive --format=tar "$rev" -- contracts 2>/dev/null \
        | tar -tf - 2>/dev/null \
        | grep -c '\.yaml$') || n=""
  # 0 is a FAILED measurement, never a count. The preflight has already proved
  # the revision carries contracts/, so an empty listing means the instrument
  # broke -- and "0 violations over 0 files" is this fleet's signature defect.
  if [ -z "$n" ] || [ "$n" -eq 0 ]; then return 1; fi
  printf '%s\n' "$n"
}

# FALSIFY-README-002, BSE-03 phase A. The verdict is a function of
# (comparand SHA, merge SHA) and of the README's claim -- never of a number
# stored anywhere a pull request can rewrite.
#
#   * the count is GENERATED into a CONTRACT_COUNT block by
#     scripts/readme_sync.sh, so a block that disagrees with the merge tree is
#     RED by EQUALITY: a generated number cannot legitimately lag;
#   * a number a human wrote outside the block keeps the G-11 ratchet (may lag,
#     may never overstate, --exact for the orchestrator docs commit);
#   * no block and no literal is GREEN only because the generator regenerates it
#     HERE, deterministically, with the bytes printed -- never because a claim
#     is absent (threat model P7);
#   * an unresolvable comparand is RED at PREFLIGHT, before the measurement it
#     would invalidate (P5).
check_contract_count() {
  local resolution mode ref base_sha merge_sha base_count merge_count disk delta
  local blocks literals block nblocks target base_short merge_short base_date p1 p2 pn

  # --- PREFLIGHT: the comparand, before a single file is counted ----------
  resolution=$(baseline_ratchet_resolve "$REPO_ROOT" "$BASELINE_RATCHET_BASE_REF" contracts)
  mode=${resolution%%$'\t'*}
  ref=${resolution##*$'\t'}
  case "$mode" in
    UNRESOLVABLE)
      { printf 'FAIL FALSIFY-README-002 contract_count: PREFLIGHT — cannot resolve the comparand ref <%s>, so the count is UNMEASURED against anything. That is not "no drift", and it is not degraded to comparing this branch against itself.\n' "$ref"
        printf '     In CI, before this guard runs:  git fetch --no-tags --depth=1 origin +refs/heads/main:refs/remotes/origin/main\n'; } >&2
      return 1 ;;
    ABSENT|BOOTSTRAP)
      printf 'FAIL FALSIFY-README-002 contract_count: PREFLIGHT — <%s> carries no contracts/ tree, so there is no comparand to diff against. A missing comparand is not "no drift".\n' "$ref" >&2
      return 1 ;;
  esac
  base_sha=$(git -C "$REPO_ROOT" rev-parse --verify --quiet "${ref}^{commit}") || base_sha=""
  merge_sha=$(git -C "$REPO_ROOT" rev-parse --verify --quiet 'HEAD^{commit}') || merge_sha=""
  if [ -z "$base_sha" ] || [ -z "$merge_sha" ]; then
    printf 'FAIL FALSIFY-README-002 contract_count: PREFLIGHT — <%s> and HEAD did not both resolve to a commit (base=%q merge=%q).\n' "$ref" "$base_sha" "$merge_sha" >&2
    return 1
  fi

  # --- measurement: ONE instrument, TWO revisions ------------------------
  base_count=$(measure_contract_count_rev "$base_sha") || base_count=""
  merge_count=$(measure_contract_count_rev "$merge_sha") || merge_count=""   # RATCHET-MUTATION-POINT — the tree under test is measured from the OBJECT STORE, never read from a file on disk (the registered mutation of scripts/tests/ratchet_semantics_test.sh replaces exactly this line)
  if [ -z "$base_count" ] || [ -z "$merge_count" ]; then
    printf 'FAIL FALSIFY-README-002 contract_count: MEASUREMENT FAILED (base=%q merge=%q). A tree that could not be counted is a broken check, not a README drift; do not "fix" the README.\n' "$base_count" "$merge_count" >&2
    return 1
  fi
  disk=$(measured_contract_count)
  delta=$(( merge_count - base_count ))
  base_short=$(git -C "$REPO_ROOT" rev-parse --short "$base_sha" 2>/dev/null) || base_short="$base_sha"
  merge_short=$(git -C "$REPO_ROOT" rev-parse --short "$merge_sha" 2>/dev/null) || merge_short="$merge_sha"
  base_date=$(git -C "$REPO_ROOT" show -s --format=%cI "$base_sha" 2>/dev/null) || base_date="<no date>"

  printf 'FALSIFY-README-002 contract_count — D2 comparand diff (BSE-03): the verdict is a function of two REVISIONS and the README claim, of nothing else on disk\n'
  printf '  comparand   %-12s %s  %s\n' "$mode" "$base_short" "$base_date"
  printf '  merge       %-12s %s  (HEAD)\n' 'HEAD' "$merge_short"
  printf '  contracts   base=%s  merge=%s  delta=%s\n' "$base_count" "$merge_count" "$(printf '%+d' "$delta")"
  printf '  polarity    the verdict is about TRUTH, not DIRECTION: a count that FELL vs the comparand is an IMPROVEMENT, and a README stating the fallen count is GREEN. There is no lower bound and nothing to delete.\n'
  if [ "$BASELINE_RATCHET_BASE_REF" != "origin/main" ]; then
    printf '  OVERRIDDEN  comparand set via BASELINE_RATCHET_BASE_REF=%s — NOT a protected ref\n' "$BASELINE_RATCHET_BASE_REF"
  fi
  if [ "$disk" != "$merge_count" ]; then
    printf '  UNCOMMITTED the working tree holds %s contracts/*.yaml, the merge tree %s holds %s. A claim stating the WORKING tree is accepted HERE and re-verified against the merge tree in CI, whose checkout is pristine and where the two are equal by construction.\n' \
      "$disk" "$merge_short" "$merge_count"
  fi

  # Which tree the claim is judged against. Both candidates are TREES; neither
  # is a stored literal, and the working tree is only reachable when it actually
  # differs from the merge commit (never in CI).
  pick_target() { # pick_target <claimed>
    if [ "$disk" != "$merge_count" ] && [ "$1" = "$disk" ]; then printf '%s\n' "$disk"; else printf '%s\n' "$merge_count"; fi
  }

  # An authored literal keeps the G-11 ratchet; there must be at most one.
  check_contract_literals() { # check_contract_literals <literals>
    local lits="$1" nlit=0 tgt
    [ -n "$lits" ] || return 0
    nlit=$(printf '%s' "$lits" | grep -c .) || nlit=0
    if [ "$nlit" -gt 1 ]; then
      printf 'FAIL FALSIFY-README-002 contract_count: the README carries several different authored contract counts: %s — a drift the reader cannot resolve, whatever the tree says.\n' \
        "$(printf '%s' "$lits" | tr '\n' ' ')" >&2
      return 1
    fi
    tgt=$(pick_target "$lits")
    compare_count FALSIFY-README-002 contract_count "$lits" "$tgt" "the merge tree $merge_short (git archive <rev> -- contracts, *.yaml)"
  }

  blocks=$(contract_block_counts) || blocks=""
  literals=$(claimed_contract_counts) || literals=""

  if [ -n "$blocks" ]; then
    nblocks=$(printf '%s' "$blocks" | grep -c .) || nblocks=0
    if [ "$nblocks" -gt 1 ]; then
      printf 'FAIL FALSIFY-README-002 contract_count: the README carries several different CONTRACT_COUNT blocks: %s. One generator, one number.\n' \
        "$(printf '%s' "$blocks" | tr '\n' ' ')" >&2
      return 1
    fi
    block="$blocks"
    if ! printf '%s' "$block" | grep -qE '^[0-9]+$'; then
      printf 'FAIL FALSIFY-README-002 contract_count: the CONTRACT_COUNT block holds %q, which is not a number. It is generated: run `make readme-sync`.\n' "$block" >&2
      return 1
    fi
    target=$(pick_target "$block")
    if [ "$block" != "$target" ]; then
      printf 'FAIL FALSIFY-README-002 contract_count: the CONTRACT_COUNT block states %s, the merge tree carries %s — the block is GENERATED, not authored, so this is an EQUALITY and not a ratchet (a generated number cannot legitimately lag). Run: make readme-sync\n' \
        "$block" "$merge_count" >&2
      return 1
    fi
    check_contract_literals "$literals" || return 1
    printf 'PASS FALSIFY-README-002 contract_count: %s (CONTRACT_COUNT block, derived by scripts/readme_sync.sh; %s block(s) agree with the merge tree %s)\n' \
      "$block" "$nblocks" "$merge_short"
    return 0
  fi

  if [ -n "$literals" ]; then
    check_contract_literals "$literals" || return 1
    return 0
  fi

  # No block, no literal: the claim is DERIVED. This is GREEN only because the
  # generator produces it here, twice, byte for byte -- an absent claim on its
  # own proves nothing and inverting the old "makes no claim" FAIL without the
  # generator would have deleted the check (threat model, open question 1).
  p1=$(bash "$REPO_ROOT/scripts/readme_sync.sh" --print 2>/dev/null) || p1=""
  p2=$(bash "$REPO_ROOT/scripts/readme_sync.sh" --print 2>/dev/null) || p2=""
  if [ -z "$p1" ] || [ "$p1" != "$p2" ]; then
    printf 'FAIL FALSIFY-README-002 contract_count: the README states no contract count and scripts/readme_sync.sh --print is not byte-stable across two runs (%q then %q). An absent claim is GREEN only when the generator can regenerate it deterministically.\n' "$p1" "$p2" >&2
    return 1
  fi
  pn=$(printf '%s' "$p1" | sed -E "s|${CONTRACT_BLOCK_START}||; s|${CONTRACT_BLOCK_END}||")
  if [ "$pn" != "$(pick_target "$pn")" ]; then
    printf 'FAIL FALSIFY-README-002 contract_count: the generator would write %s, the merge tree carries %s — the generator and the tree under test disagree.\n' "$pn" "$merge_count" >&2
    return 1
  fi
  printf 'PASS FALSIFY-README-002 contract_count: DERIVED — the README states no count, and scripts/readme_sync.sh regenerates it deterministically (two runs, byte-identical). It would write: %s\n' "$p1"
}

check_cli_command_count() {
  local measured claimed rc
  # Never swallow a failed measurement. The previous form was
  #     measured=$(measured_cli_command_count) || return $?
  # which returned SILENTLY -- no PASS, no FAIL, no diagnostic -- so a broken
  # measurement produced a red job with nothing printed for this check at all.
  # A measurement that cannot run is its own failure mode and must say so.
  measured=$(measured_cli_command_count); rc=$?
  if [[ "$rc" -ne 0 || -z "$measured" ]]; then
    echo "FAIL FALSIFY-README-003 cli_command_count: MEASUREMENT FAILED (rc=$rc) —" \
         "could not count commands in contracts/apr-cli-commands-v1.yaml." \
         "This is a broken check, not a README drift; do not 'fix' the README." >&2
    return 1
  fi
  claimed=$(claimed_cli_command_count)
  if [[ -z "$claimed" ]]; then
    echo "FAIL FALSIFY-README-003 cli_command_count: README lacks '**K** CLI commands' claim" >&2
    return 1
  fi
  compare_count FALSIFY-README-003 cli_command_count "$claimed" "$measured" "contracts/apr-cli-commands-v1.yaml"
}

check_cookbook_link() {
  local measured
  measured=$(measured_cookbook_link_present)
  if [[ "$measured" != "1" ]]; then
    echo "FAIL FALSIFY-README-004 cookbook_link: README does not mention apr-cookbook" >&2
    return 1
  fi
  echo "PASS FALSIFY-README-004 cookbook_link: present"
}

# --- dispatcher ---

mode="all"
claim=""
for arg in "$@"; do
  case "$arg" in
    --claim) mode="one" ;;
    --regen) mode="regen" ;;
    --exact) EXACT=1 ;;
    --self-test) mode="selftest" ;;
    crate_count|contract_count|cli_command_count|cookbook_link) claim="$arg" ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done


# FALSIFY-README-005: the install line may not advertise a backend the DEFAULT
# feature set does not compile in.
#
# README.md said `cargo install aprender  # CPU + wgpu (default)` for three
# releases while root Cargo.toml said `default = ["cli"]` and
# `cli = ["dep:apr-cli"]` -- no GPU backend of any kind. That sentence is why
# #2696 stayed invisible: a user who reads "wgpu (default)" and then passes
# --gpu has every reason to expect it to work, and the published binary silently
# ran on CPU at 15.7 tok/s. The defect was not only that --gpu was ignored, it
# was that the docs promised the backend that would have honoured it.
#
# Read from Cargo.toml, not from a remembered string, so it tracks the manifest
# instead of drifting beside it.
check_install_line() {
  local default_feats install_line backend bad=0
  default_feats=$(sed -n 's/^default = \[\(.*\)\]/\1/p' Cargo.toml | head -1 | tr -d '" ')
  install_line=$(grep -m1 '^cargo install aprender  *#' README.md || true)
  if [ -z "$install_line" ]; then
    echo "FAIL FALSIFY-README-005 install_line: no 'cargo install aprender  #' line found"
    return 1
  fi
  for backend in wgpu cuda gpu metal rocm; do
    case ",$default_feats," in *",$backend,"*) continue ;; esac
    # A NEGATION IS NOT A CLAIM. The honest replacement line reads
    # "no GPU backend is compiled in", which mentions GPU precisely in order to
    # deny it -- and the first version of this check flagged it, which would
    # have forced the docs to avoid the clearest available wording. Only an
    # affirmative mention counts, so a `no <backend>` / `without <backend>` /
    # `not <backend>` is skipped. Both directions are in the case table below.
    if grep -qiE "(no|not|without|never)[[:space:]]+$backend" <<< "$install_line" ; then
      continue
    fi
    if grep -qiE "(^|[^a-z])$backend([^a-z]|$)" <<< "$install_line" ; then
      printf 'FAIL FALSIFY-README-005 install_line: advertises %s, but default = [%s]\n' \
             "$backend" "$default_feats"
      printf '       %s\n' "$install_line"
      bad=1
    fi
  done
  [ "$bad" -eq 0 ] || return 1
  printf 'PASS FALSIFY-README-005 install_line: claims no backend absent from default = [%s]\n' "$default_feats"
}

case "$mode" in
  selftest)
    # G-11 case table: the counts are a ratchet (lag allowed, overstatement RED; --exact for the orchestrator)
    TD=$(mktemp -d "${TMPDIR:-/tmp}/readme-selftest.XXXXXX")
    safe_rm_scratch() { local victim=${1:-} must=${2:-}; [ -n "$victim" ] || return 0; [ -n "$must" ] || return 0; [ "$victim" != "/" ] || return 0
      case "$victim" in *"$must"*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;; *) return 0 ;; esac; }
    cleanup() { safe_rm_scratch "$TD" 'readme-selftest.'; }
    trap cleanup EXIT
    mc=$(measured_crate_count) || { echo "FAIL self-test: cannot measure the crate count" >&2; exit 1; }
    cc=$(measured_contract_count)
    fx() { printf '# apr\n\n**%s** workspace crates, **%s** provable contracts.\n%s\n' "$1" "$2" "${3:-}" > "$TD/README.md"; }
    n=0; red=0
    row() { # row <want rc> <label> <claim> [<extra env>]
      local want=$1 label=$2 claim=$3 env=${4:-} rc=0
      n=$((n + 1))
      env README_PATH="$TD/README.md" $env bash "$0" --claim "$claim" >"$TD/out.$n" 2>&1 || rc=$?
      if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
      else printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"; sed 's/^/        /' "$TD/out.$n"; red=1; fi
    }
    fx "$mc" "$cc";                    row 0 "claims equal the measurement: PASS"                              crate_count
    fx "$mc" "$cc";                    row 0 "equal, --exact: PASS"                                            contract_count README_EXACT=1
    fx "$((mc - 1))" "$((cc - 1))";    row 0 "README lags by one (a row PR added a crate): PASS, lag reported" crate_count
    grep -q 'lags at' "$TD/out.$n" || { printf 'FAIL  row %-2s the lag was not reported\n' "$n"; red=1; }
    fx "$((mc - 1))" "$((cc - 1))";    row 1 "lags by one under --exact (the orchestrator docs commit): RED"  contract_count README_EXACT=1
    fx "$((mc + 1))" "$((cc + 1))";    row 1 "README OVERSTATES the crate count by one: RED (the registered mutation)" crate_count
    fx "$mc" "$((cc + 1))";            row 1 "README overstates the contract count: RED"                       contract_count
    fx "$mc" "$cc" "and $((cc - 1)) contracts elsewhere"; row 1 "two different contract counts in one README: RED (self-contradiction)" contract_count
    printf '%s/%s rows\n' "$((n - red))" "$n"
    [ "$red" = 0 ] || exit 1
    exit 0
    ;;
  regen)
    echo "workspace members:     $(measured_crate_count)"
    echo "contracts/ *.yaml:     $(measured_contract_count)"
    if cli=$(measured_cli_command_count 2>/dev/null); then
      echo "apr --help subcmds:    $cli"
    else
      echo "apr --help subcmds:    <apr binary not available>"
    fi
    echo "apr-cookbook link:     $(measured_cookbook_link_present)"
    ;;
  one)
    case "$claim" in
      crate_count)       check_crate_count ;;
      contract_count)    check_contract_count ;;
      cli_command_count) check_cli_command_count ;;
      cookbook_link)     check_cookbook_link ;;
      install_line)      check_install_line ;;
      *) echo "--claim requires one of: crate_count, contract_count, cli_command_count, cookbook_link, install_line" >&2; exit 2 ;;
    esac
    ;;
  all)
    fail=0
    check_crate_count       || fail=1
    check_contract_count    || fail=1
    check_cli_command_count || fail=1
    check_cookbook_link     || fail=1
    check_install_line      || fail=1
    exit "$fail"
    ;;
esac
