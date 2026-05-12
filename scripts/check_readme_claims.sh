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
#   crate_count        → `ls crates/ | wc -l` == README "N workspace crates"
#   contract_count     → `find contracts/ -name '*.yaml' | wc -l` == README "M provable contracts"
#   cli_command_count  → `apr --help` subcmd count == README "K CLI commands"
#   cookbook_link      → README.md mentions `apr-cookbook`

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
README="$REPO_ROOT/README.md"

if [[ ! -f "$README" ]]; then
  echo "error: $README not found" >&2
  exit 2
fi

# --- measurements (authoritative, from filesystem / live apr binary) ---

measured_crate_count() {
  find "$REPO_ROOT/crates" -maxdepth 1 -mindepth 1 -type d | wc -l | tr -d ' '
}

measured_contract_count() {
  find "$REPO_ROOT/contracts" -name "*.yaml" | wc -l | tr -d ' '
}

measured_cli_command_count() {
  # Count subcommands in `apr --help` output at HEAD (not a stale
  # `cargo install aprender` binary on PATH — that reflects a prior
  # crates.io version, which may differ from the repo's current state).
  #
  # Use `cargo run --quiet -p apr-cli --bin apr` so the number tracks
  # the committed source. Slow on cold build; fast after dev-profile warm.
  local subcmd_line_re='^  [a-z][-a-z0-9]* '
  (cd "$REPO_ROOT" && cargo run --quiet -p apr-cli --bin apr -- --help 2>&1) \
    | grep -cE "$subcmd_line_re" | tr -d ' '
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

claimed_contract_count() {
  # Look for pattern "**M** provable contracts"
  local contract_re='\*\*[0-9]+\*\* provable contracts'
  local num_re='[0-9]+'
  grep -oE "$contract_re" "$README" | grep -oE "$num_re" | head -1
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
  measured=$(measured_crate_count)
  claimed=$(claimed_crate_count)
  if [[ -z "$claimed" ]]; then
    echo "FAIL FALSIFY-README-001 crate_count: README lacks '**N** workspace crates' claim (pattern mismatch)" >&2
    return 1
  fi
  if [[ "$measured" != "$claimed" ]]; then
    echo "FAIL FALSIFY-README-001 crate_count: README claims $claimed, filesystem has $measured" >&2
    return 1
  fi
  echo "PASS FALSIFY-README-001 crate_count: $measured"
}

check_contract_count() {
  local measured claimed
  measured=$(measured_contract_count)
  claimed=$(claimed_contract_count)
  if [[ -z "$claimed" ]]; then
    echo "FAIL FALSIFY-README-002 contract_count: README lacks '**M** provable contracts' claim" >&2
    return 1
  fi
  if [[ "$measured" != "$claimed" ]]; then
    echo "FAIL FALSIFY-README-002 contract_count: README claims $claimed, filesystem has $measured" >&2
    return 1
  fi
  echo "PASS FALSIFY-README-002 contract_count: $measured"
}

check_cli_command_count() {
  local measured claimed
  measured=$(measured_cli_command_count) || return $?
  claimed=$(claimed_cli_command_count)
  if [[ -z "$claimed" ]]; then
    echo "FAIL FALSIFY-README-003 cli_command_count: README lacks '**K** CLI commands' claim" >&2
    return 1
  fi
  if [[ "$measured" != "$claimed" ]]; then
    echo "FAIL FALSIFY-README-003 cli_command_count: README claims $claimed, apr --help lists $measured" >&2
    return 1
  fi
  echo "PASS FALSIFY-README-003 cli_command_count: $measured"
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
    crate_count|contract_count|cli_command_count|cookbook_link) claim="$arg" ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

case "$mode" in
  regen)
    echo "crates/ count:         $(measured_crate_count)"
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
      *) echo "--claim requires one of: crate_count, contract_count, cli_command_count, cookbook_link" >&2; exit 2 ;;
    esac
    ;;
  all)
    fail=0
    check_crate_count       || fail=1
    check_contract_count    || fail=1
    check_cli_command_count || fail=1
    check_cookbook_link     || fail=1
    exit "$fail"
    ;;
esac
