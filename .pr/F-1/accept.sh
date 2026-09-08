#!/usr/bin/env bash
# F-1 accept.sh — every A_i in one call (I5). The live matrix leg needs a model on the
# host and an explicitly named binary; it is SKIPPED loudly, never silently.
set -uo pipefail; cd "$(dirname "$0")/../.."; rc=0
run() { printf '== %s\n' "$*"; "$@"; local r=$?; printf 'rc=%s\n' "$r"; [ "$r" = 0 ] || rc=1; }
expect_fail() { printf '== (must FAIL) %s\n' "$*"; if "$@"; then printf 'rc=0 (wanted non-zero)\n'; rc=1; else printf 'rc=%s (as required)\n' "$?"; fi; }
CARGO=/home/noah/.cargo/bin/cargo
run python3 scripts/make_sharded_safetensors.py --self-test
run bash scripts/check_format_command_matrix.sh --self-test
run env "$CARGO" test -p apr-cli --lib -- sharded_index_resolves an_existing_unrecognised resolve_never_answers a_truncated_file_is the_three_single_file odd_extension
. scripts/pv_bin.sh >/dev/null 2>&1 && run "$PV" validate contracts/patterns/format-command-honesty-v1.yaml
run bash scripts/check_guards_are_wired.sh
if [ -n "${APR_BIN:-}" ] && [ -x "${APR_BIN:-}" ]; then
    run bash scripts/check_format_command_matrix.sh --matrix --apr "$APR_BIN"
else
    printf '== live 4-cell matrix: set APR_BIN=<apr built from this tree> on a host holding qwen2.5-coder-0.5b-instruct-safetensors (not a pass)\n'
fi
exit "$rc"
