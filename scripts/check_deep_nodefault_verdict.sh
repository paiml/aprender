#!/usr/bin/env bash
# check_deep_nodefault_verdict.sh -- the case table and the mutants of
# scripts/release/deep_nodefault_verdict.sh, the --no-default-features verdict that the
# deep-nightly `deep-nodefault` lane and autopilot's T-1 deep step both call (#4871).
#
# WHY A SEPARATE FILE. guard_tree.sh's universe is `git ls-files 'scripts/check_*.sh'`, a flat
# glob on scripts/, so a case table living in scripts/release/ is run by nothing on a PR. The
# deep-nightly job runs the verdict, not its table, and only once a night on main. This wrapper
# puts the table and the mutants on every PR through guard_tree.sh, with no workflow edit.
#
# Hermetic: literal log fixtures, no toolchain, no network.
#
#   check_deep_nodefault_verdict.sh   run the subject's case table, then its mutants (both must pass)
set -uo pipefail
case "${1:-}" in
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    '') ;;
    *) echo "usage: bash scripts/check_deep_nodefault_verdict.sh   (no arguments)" >&2; exit 2 ;;
esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
SUBJECT="$ROOT/scripts/release/deep_nodefault_verdict.sh"
if [ ! -r "$SUBJECT" ]; then echo "check_deep_nodefault_verdict: ENV no $SUBJECT -- the guard did not run" >&2; exit 2; fi

rc=0
bash "$SUBJECT" --self-test || rc=1
bash "$SUBJECT" --mutants || rc=1
echo "check_deep_nodefault_verdict: $([ "$rc" = 0 ] && echo PASS || echo FAIL)"
exit "$rc"
