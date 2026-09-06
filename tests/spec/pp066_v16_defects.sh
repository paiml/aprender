#!/usr/bin/env bash
# tests/spec/pp066_v16_defects.sh — the RED-first test of PP-066 spec v1.6
# (PMAT-988, DAG row SPEC-1.6, #2903).
#
# Each row asserts that ONE corrected sentence is present in the spec. Against
# the v1.5 text (docs/specifications/PP-066-release-spec.md at 42be1560b) every
# row is RED; against v1.6 every row is GREEN. Reverting one correction turns
# exactly its row RED — that is the registered mutation of the docs row.
#
#   bash tests/spec/pp066_v16_defects.sh [<spec.md>]      # default docs/specifications/PP-066-release-spec.md
#   bash tests/spec/pp066_v16_defects.sh --v15-red        # proves the table is RED on the v1.5 text (git show 42be1560b:...)
#
# Rows 1-14 are the S0 ledger's numbered defects (docs/audits/pp-066-s0-ledger.md
# §"Spec defects"); row 15 is the §4 "credited first" single reading the plan
# quorum of 2026-09-05 asked for (docs/audits/pp-066-plan-quorum.md, A2).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPEC="${1:-$ROOT/docs/specifications/PP-066-release-spec.md}"
V15_SHA=42be1560b

if [ "${1:-}" = "--v15-red" ]; then
    TMP=$(mktemp "${TMPDIR:-/tmp}/pp066-v15.XXXXXX.md")
    git -C "$ROOT" show "$V15_SHA:docs/specifications/PP-066-release-spec.md" > "$TMP"
    rc=0; bash "$0" "$TMP" > /dev/null 2>&1 || rc=$?
    rm -f -- "$TMP"
    if [ "$rc" -ne 0 ]; then echo "ok    the v1.5 text ($V15_SHA) is RED under this table (rc=$rc)"; exit 0; fi
    echo "FAIL  the v1.5 text passes this table — the table cannot fail and proves nothing"; exit 1
fi
[ -f "$SPEC" ] || { echo "pp066_v16_defects: $SPEC is missing" >&2; exit 2; }

n=0; red=0
row() { # row <defect#> <label> <literal that must be present>
    n=$((n + 1))
    if grep -qF -- "$3" "$SPEC"; then printf 'ok    row %-2s (defect %-2s) %s\n' "$n" "$1" "$2"
    else printf 'FAIL  row %-2s (defect %-2s) %s\n        missing: %s\n' "$n" "$1" "$2" "$3"; red=1; fi
}
row 1  "F-9/§10: the tracked master is v3.1 with row 22"              "the tracked \`docs/specifications/PP-LLAMA-001-MASTER.md\` is v3.1 with row 22 at line 364"
row 2  "§3 S0-1: the grep matches the bold row id"                    "grep -n '^| \\*\\*22\\*\\*' docs/specifications/PP-LLAMA-001-MASTER.md"
row 2  "§3 S0-2: pmat work list has no --status all"                  "has no \`--status all\`"
row 2  "§3 S0-12: there is no metal feature"                          "there is **no** \`metal\` feature"
row 2  "§3 S0-15: cli is the root facade's feature"                   "\`cli\` is the root facade's feature; \`apr-cli\` has none"
row 3  "§3/§9: twenty-three premises, 23/23"                          "**23/23** Step-0 premises"
row 4  "§10: snapshot date 2026-09-05 (v1.6)"                         "snapshot at 2026-09-05 (v1.6)"
row 5  "D-7: FeatureDisabled => 9 is at error.rs:106"                 "crates/apr-cli/src/error.rs:106"
row 6  "S0-11/T-2: the residual clamp is the wgpu pipeline at :717"   "only the wgpu pipeline still hardcodes \`512\` at \`finetune.rs:717\`"
row 7  "S0-13: api/handlers.rs does not exist"                        "\`crates/aprender-serve/src/api/handlers.rs\` does not exist"
row 8  "C1's check is a command, not a ≥ sentence"                    "[ \"\$(grep -c CONFORMANT evidence/parity/LEDGER.md)\" -ge 2 ]"
row 8  "the \\| escaping note points at the DAG"                       "a \`\\|\` inside a command cell is markdown-table escaping"
row 9  "R-6/§8: public-key scheme, no THIRD scheme"                   "no **third** signing scheme"
row 10 "F-26: registry: true is 500"                                  "500 of the 1,258 top-level contracts self-exempt"
row 11 "S0-12/B-M/MIN-05: no Metal dispatcher, correction withdrawn"  "**Correction withdrawn (S0-12, v1.6):**"
row 12 "Track I: Appendix E carries no pmat work add lines"           "master Appendix E is a landing checklist and carries no \`pmat work add\` lines"
row 13 "Track P: improve-provable-contracts.md is not in the tree"    "is **not in the tree** (issue #2870 is its tracking ticket"
row 14 "§4 andon: the expiry parse finding and I-26"                  "DAG row I-26 fixes the parse and wires the RED"
row 15 "§4 credited first has exactly one reading"                    "C0 is a *precondition of crediting*, not of *working*"
row 15 "§4 names the quorum's other reading as the falsified one"     "read the v1.5 sentence as temporal precedence; this is the falsifier of that reading"
row 16 "G-4: the checker is in-repo, pmat has no such rule"           "\`pmat comply check\` has no \`--rule obligation-dag\` at pmat 3.37.0"
row 17 "expiry moves are recorded as §12 amendments"                  "2026-09-26 (v1.6: 09-19 → 09-26, zero slack vs P-0.3; §12 rule)"
printf '%s/%s rows\n' "$((n - red))" "$n"
[ "$red" = 0 ] || exit 1
