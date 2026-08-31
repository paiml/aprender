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

# A cargo exit is classified before any verdict names the README. See
# scripts/cargo_classify.sh. The case table is armed on the normal path because
# no workflow invokes a --self-test here, and a case table nothing runs is the
# vacuous-scan class; re-mutated in this scope rather than inheriting another
# guard's green.
. "$REPO_ROOT/scripts/cargo_classify.sh" || exit 1
cargo_classify_selftest --quiet || exit 1

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
  # `|| return 1` is required: without it the assignment swallows the ENV/vacuity
  # verdict above and the comparison proceeds against an empty measurement.
  measured=$(measured_crate_count) || return 1
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
  if [[ "$measured" != "$claimed" ]]; then
    echo "FAIL FALSIFY-README-003 cli_command_count: README claims $claimed," \
         "contracts/apr-cli-commands-v1.yaml lists $measured commands" >&2
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
