#!/usr/bin/env bash
# Falsifier: planted undated PROPOSED spec must be RED, dated one GREEN, empty dir rc 2, real tree GREEN.
set -u
S="$(dirname "$0")/check_spec_proposed_dated.sh"; t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
printf '# X\n**Status:** PROPOSED\n' >"$t/a.md"
bash "$S" "$t" >/dev/null; [ $? -eq 1 ] || { echo "FAIL planted violation not RED"; exit 1; }
printf '# X\n**Status:** Proposed\n**Date:** 2026-05-01\n' >"$t/a.md"
bash "$S" "$t" >/dev/null || { echo "FAIL dated spec not GREEN"; exit 1; }
mkdir "$t/e"; bash "$S" "$t/e" >/dev/null; [ $? -eq 2 ] || { echo "FAIL empty dir not rc 2"; exit 1; }
bash "$S" "$(dirname "$0")/../docs/specifications" || { echo "FAIL real tree RED"; exit 1; }
echo PASS
