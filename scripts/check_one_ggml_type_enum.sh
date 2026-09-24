#!/usr/bin/env bash
# check_one_ggml_type_enum.sh — the workspace declares ONE ggml tensor-type enum.
#
# WHY (PMAT-3430, PP-QUANT-001 M1). aprender declared three, and they disagreed
# about which ids exist, about spelling, and about sizes: aprender-serve's
# `GgmlQuantType` (16 ids), aprender-core's `GgmlType` (12), aprender-compute's
# `GgmlType` (15), against 35 live ids upstream. Nothing compared any of them to
# ggml, so NVFP4 (40), Q1_0 (41) and Q2_0 (42) appeared upstream with no signal
# in this tree. M1 collapsed them into `trueno_quant::GgmlType`; this refuses the
# fourth.
#
# THE ENUM IS THE UNIT, NOT THE NAME. A re-export (`pub use trueno_quant::GgmlType`)
# is fine and expected — there are several. What may not exist twice is a
# DEFINITION (`pub enum GgmlType {`). The two are different lines and this guard
# distinguishes them, which is the whole reason it can be green at all.
#
# ROOT FROM $0, NOT FROM git. `git rev-parse --show-toplevel` fails inside the CI
# container ("dubious ownership in repository at '/workspace'": the tree is uid
# 1000, the container is root), and locating our own siblings never needed git.
# Measured: that exact line turned workspace-test-shard (3/3) red on this
# ticket's first CI run (#3586 generalises it).
#
#   bash scripts/check_one_ggml_type_enum.sh              # check
#   bash scripts/check_one_ggml_type_enum.sh --self-test  # the case table
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd)
root=${here%/scripts}

# THE PATTERN, and it ships a case table because this repo's guard regexes have
# been wrong five times and a table caught every one while review caught none.
# A definition is `pub enum <Name> {` at the start of a line (leading spaces
# allowed for a nested one); `//` anywhere before it disqualifies the line, so a
# commented-out enum is not a definition.
ENUM_DEF_RE='^[[:space:]]*pub[[:space:]]+enum[[:space:]]+(GgmlType|GgmlQuantType)[[:space:]]*\{'

find_definitions() {
    # $1 = tree to scan. Prints "path:line:text" per definition.
    grep -rnE "$ENUM_DEF_RE" "$1" --include='*.rs' 2>/dev/null || true
}

self_test() {
    td=$(mktemp -d)
    trap 'rm -rf "${td:?}"' RETURN
    mkdir -p "$td/src"
    fails=0
    ran=0

    row() { # row <label> <want-count> <file-body>
        ran=$((ran + 1))
        printf '%s\n' "$3" > "$td/src/case.rs"
        got=$(find_definitions "$td/src" | wc -l | tr -d ' ')
        if [ "$got" = "$2" ]; then
            printf '  PASS  %-56s matched %s\n' "$1" "$got"
        else
            printf '  FAIL  %-56s matched %s, wanted %s\n' "$1" "$got" "$2"
            fails=$((fails + 1))
        fi
    }

    printf 'check_one_ggml_type_enum self-test\n'
    # MUST MATCH — the three definitions this ticket removed, in their real shapes.
    row "serve's GgmlQuantType definition"        1 'pub enum GgmlQuantType {'
    row "core's GgmlType definition"              1 'pub enum GgmlType {'
    row "an indented (nested) definition"         1 '    pub enum GgmlType {'
    row "definition with the brace spaced off"    1 'pub enum GgmlType  {'
    # MUST NOT MATCH — everything that merely mentions the name.
    row "a re-export"                             0 'pub use trueno_quant::GgmlType;'
    row "a re-export under the old alias"         0 'pub use trueno_quant::GgmlType as GgmlQuantType;'
    row "a commented-out definition"              0 '// pub enum GgmlType {'
    row "an indented commented-out definition"    0 '    // pub enum GgmlType {'
    row "a doc comment naming the enum"           0 '/// pub enum GgmlType { — the old shape'
    row "a DIFFERENT enum with a similar name"    0 'pub enum GgmlTypeError {'
    row "the metadata value-type enum"            0 'pub enum GgufValueType {'
    row "the metadata value enum"                 0 'pub enum GgufValue {'
    row "APR's own on-disk quant enum"            0 'pub enum QuantType {'
    row "a private enum of the same name"         0 'enum GgmlType {'
    row "a use of the type in a signature"        0 'fn f(t: GgmlType) -> GgmlType { t }'

    printf '\n%s case(s), %s failure(s)\n' "$ran" "$fails"
    [ "$fails" -eq 0 ] || return 1
    return 0
}

if [ "${1:-}" = "--self-test" ]; then
    self_test
    exit $?
fi

defs=$(find_definitions "$root/crates")
n=$(printf '%s' "$defs" | grep -c . || true)

printf '== one ggml tensor-type enum (PMAT-3430) ==\n'
if [ "$n" -eq 1 ]; then
    printf 'ok    exactly 1 definition:\n%s\n' "$(printf '%s' "$defs" | sed 's/^/        /')"
    exit 0
fi

if [ "$n" -eq 0 ]; then
    # Zero is not a pass. A pattern that matches nothing is the vacuity this
    # fleet keeps re-finding; the enum exists, so zero means the pattern broke.
    printf 'FAIL  no ggml tensor-type enum definition found at all.\n'
    printf '      The enum exists (crates/aprender-quant/src/ggml_type.rs), so this is a\n'
    printf '      BROKEN PATTERN, not a clean tree. Run --self-test.\n'
    exit 1
fi

printf 'FAIL  %s ggml tensor-type enum definitions; the workspace declares ONE (#3430).\n' "$n"
printf '%s\n' "$(printf '%s' "$defs" | sed 's/^/        /')"
printf '      A consumer wants `pub use trueno_quant::GgmlType;`, not its own copy.\n'
exit 1
