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
README="$REPO_ROOT/README.md"

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
  (cd "$REPO_ROOT" && cargo metadata --no-deps --format-version 1 2>/dev/null) \
    | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["packages"]))'
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
  #
  # `help` is EXCLUDED. clap generates it automatically; it is not one of
  # apr's commands, has no chapter, no contract and no implementation in
  # crates/apr-cli/src/commands/. Counting it made this check report 104
  # against a README claiming 103 - the README was right and this function
  # was off by exactly clap's freebie.
  local subcmd_line_re='^  [a-z][-a-z0-9]* '
  (cd "$REPO_ROOT" && cargo run --quiet -p apr-cli --bin apr -- --help 2>&1) \
    | grep -E "$subcmd_line_re" \
    | grep -vE '^  help( |$)' \
    | wc -l | tr -d ' '
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

# EVERY contract count the README claims, one per line, deduplicated.
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
claimed_contract_counts() {
  grep -oiE '[0-9]+\*{0,2}( +[a-z]+){0,2} +contracts?\b' "$README" \
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
  measured=$(measured_crate_count)
  claimed=$(claimed_crate_count)
  if [[ -z "$claimed" ]]; then
    echo "FAIL FALSIFY-README-001 crate_count: README lacks '**N** workspace crates' claim (pattern mismatch)" >&2
    return 1
  fi
  if [[ "$measured" != "$claimed" ]]; then
    echo "FAIL FALSIFY-README-001 crate_count: README claims $claimed, cargo metadata --no-deps has $measured workspace members" >&2
    return 1
  fi
  echo "PASS FALSIFY-README-001 crate_count: $measured"
}

check_contract_count() {
  local measured claimed rc=0 n=0
  measured=$(measured_contract_count)
  claimed=$(claimed_contract_counts)
  if [[ -z "$claimed" ]]; then
    echo "FAIL FALSIFY-README-002 contract_count: README makes no contract-count claim" >&2
    return 1
  fi
  # EVERY claim must agree with the filesystem, not just the first one found.
  # The README is also not allowed to contradict itself: three different counts
  # in one file is a drift the reader cannot resolve.
  while IFS= read -r c; do
    [[ -n "$c" ]] || continue
    n=$((n + 1))
    if [[ "$measured" != "$c" ]]; then
      echo "FAIL FALSIFY-README-002 contract_count: README claims $c, filesystem has $measured" >&2
      grep -niE "\\b$c\\*{0,2}( +[a-z]+){0,2} +contracts?\\b" "$README" | sed 's|^|       |' >&2
      rc=1
    fi
  done <<< "$claimed"
  [[ "$rc" -eq 0 ]] || return 1
  echo "PASS FALSIFY-README-002 contract_count: $measured ($n claim(s) checked)"
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
