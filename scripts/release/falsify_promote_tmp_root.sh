#!/usr/bin/env bash
# RS-5: the promote_nightly / carry_milestone falsifiers build fixtures (a cloned git template)
# under a temp root. On the shared /tmp a concurrent job's sweep deleted .git/objects mid-copy
# (#6698 merge_group 41645998153), so the root is RUNNER_TEMP. --mutants must kill a bare mktemp.
set -euo pipefail
cd "$(dirname "$0")/../.."
FILES=(scripts/release/falsify_promote_nightly.sh scripts/release/falsify_carry_milestone.sh)
ok() { grep -qE '^TMP=\$\(mktemp -d -p "\$\{RUNNER_TEMP:-' "$1"; }
mut() {
    local d f rc=0
    d=$(mktemp -d -p "${RUNNER_TEMP:-${TMPDIR:-/tmp}}") || exit 2
    trap 'rm -rf -- "${d:?}"' RETURN
    for f in "${FILES[@]}"; do
        sed -E 's#mktemp -d -p "[^"]*"#mktemp -d#' "$f" > "$d/m"
        if ok "$d/m"; then echo "MUTANT SURVIVED: $f"; rc=1; fi
    done
    echo "mutants: $([ "$rc" = 0 ] && echo killed || echo SURVIVED)"
    return "$rc"
}
if [ "${1:-}" = --mutants ]; then mut; exit; fi
for f in "${FILES[@]}"; do ok "$f" || { echo "FAIL: $f fixture root not under RUNNER_TEMP" >&2; exit 1; }; done
echo "PASS: fixture roots under RUNNER_TEMP"
