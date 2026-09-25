#!/usr/bin/env bash
# check_verb_classification.sh -- contracts/apr-verb-classification-v1.yaml is
# what the CLI registry produces today (gh#3597 done_when 1 and 5).
#
# The classification is DERIVED from contracts/apr-cli-commands-v1.yaml by
# scripts/derive_verb_classification.py. The deriver has a --check mode, but a
# --check mode that nothing runs checks nothing: a verb added to the registry
# would leave the committed file one row short, and "0 unclassified" would
# go on reading true. This wrapper exists so guard_tree.sh (whose universe is
# `git ls-files 'scripts/check_*.sh'`) runs that --check on every PR.
#
# --self-test drops one verb from a copy of the committed classification and
# requires the check to reject it, so a deriver whose --check always passes
# fails this guard instead of passing it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DERIVE="$ROOT/scripts/derive_verb_classification.py"
OUT="$ROOT/contracts/apr-verb-classification-v1.yaml"

usage() {
    echo "usage: $0 [--self-test]"
    echo "  (no args)    fail if $OUT is not what the registry derives today"
    echo "  --self-test  a classification missing one verb must be rejected"
}

run_check() {
    python3 "$DERIVE" --check
}

self_test() {
    local rc
    BACKUP="$(mktemp)"
    cp "$OUT" "$BACKUP"
    # Restore on ANY exit, including a kill mid-test: the mutation is planted
    # in the tracked file itself, because the deriver reads a fixed path.
    trap 'cp "$BACKUP" "$OUT"; rm -f "$BACKUP"' EXIT
    python3 - "$OUT" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
# Drop the first verb row: the first "- verb:" item and its body up to the next item.
m = re.search(r"(?ms)^(\s*)- verb: .*?(?=^\1- verb: )", s)
if not m:
    sys.exit("self-test: no '- verb:' row found to drop -- the file shape changed")
open(p, "w").write(s[:m.start()] + s[m.end():])
PY
    rc=0
    run_check >/dev/null 2>&1 || rc=$?
    cp "$BACKUP" "$OUT"
    if [ "$rc" -eq 0 ]; then
        echo "FAIL self-test: --check accepted a classification with a verb dropped"
        return 1
    fi
    echo "ok  self-test: a dropped verb is rejected (rc=$rc)"
    run_check
}

case "${1:-}" in
    "") run_check ;;
    --self-test) self_test ;;
    -h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
