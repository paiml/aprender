#!/usr/bin/env bash
# check_sourced_libs_option_neutral.sh - a sourced script may not change the
# caller's shell options.
#
# THE CLASS. `set` inside a sourced file mutates the SOURCING shell, not a
# child. scripts/apr_bin.sh opened with `set -euo pipefail`; scripts/qwen-story.sh
# sources it and had deliberately chosen `set -uo pipefail` WITHOUT `-e`,
# because its whole design is to run every beat and tally failures. The source
# silently turned errexit on underneath it, so the first non-zero command
# anywhere aborted the run: the nightly story died after six lines, inside an
# ADVISORY analyser hunt, and the log said nothing about why.
#
# The leak is invisible in review - both files are individually correct, and
# `set -euo pipefail` is the thing you are normally praised for writing. Only
# the COMBINATION is wrong, which is why this is a mechanical check rather than
# a style note.
#
# A sourced library must be option-neutral and signal failure by RETURN STATUS:
#     . scripts/apr_bin.sh || exit 1
#
# Exit 0 = every sourced library leaves the caller's options alone.
# Exit 1 = at least one sets shell options at file scope.
#
# `--self-test` proves the check can still turn RED, by running it against a
# reconstruction of the exact pre-fix apr_bin.sh header.

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

SEARCH_DIR="scripts"

# `set` with the errexit / nounset / pipefail family, at column 0 (file scope).
# Indented `set` lines live inside functions or blocks, where the option change
# is at least visible to the reader of that function.
SETOPT_RE='^set[[:space:]]+[-+]'

# Which files are actually SOURCED by another SCRIPT? Only those can leak: a
# script that is merely executed (`bash scripts/foo.sh`) gets its own shell.
#
# Deliberately scoped to script-to-script sourcing, and deliberately anchored on
# `.`/`source` as the FIRST token of a line. An earlier draft also scanned
# .github/workflows/*.yml for `scripts/*.sh` near a `.` or `source`, and
# promptly flagged check_format_sovereignty.sh - because ci.yml contains the
# COMMENT "# catch dev-dep cycles. See scripts/check_format_sovereignty.sh",
# where the sentence-ending period parsed as the source builtin. A workflow
# `run:` block is its own throwaway shell anyway, so a `set` leak there cannot
# outlive the block. Narrow and sound beats broad and wrong.
sourced_basenames() {
    grep -rhoE '^[[:space:]]*(\.|source)[[:space:]]+[^;&|#]+' "$SEARCH_DIR"/*.sh 2>/dev/null \
        | grep -oE '[A-Za-z0-9_.-]+\.sh' | sort -u
}

scan_file() {
    local f="$1"
    grep -nE "$SETOPT_RE" "$f" 2>/dev/null || true
}

violations=0
checked=0

for base in $(sourced_basenames); do
    f="$SEARCH_DIR/$base"
    [ -f "$f" ] || continue
    # A file that sources itself is not interesting; skip self-references.
    checked=$((checked + 1))
    hits=$(scan_file "$f")
    if [ -n "$hits" ]; then
        printf 'OPTION-LEAK %s is sourced by another script but sets shell options:\n' "$f" >&2
        printf '%s\n' "$hits" | sed 's/^/           /' >&2
        violations=$((violations + 1))
    fi
done

# --self-test: reconstruct the pre-fix header and prove the matcher rejects it.
# Without this, a matcher that silently stopped matching would report success.
if [ "${1:-}" = "--self-test" ]; then
    # Fed to grep on stdin rather than written to temp files: identical to
    # scanning a file with this content, minus a mktemp, a trap and an
    # `rm -rf "$tmp"` that bashrs rightly flags (SEC011) in a script whose only
    # job is to read files.
    bad_lib=$(printf '#!/usr/bin/env bash\n# a sourceable helper\n\nset -euo pipefail\n\nfoo() { :; }\n')
    good_lib=$(printf '#!/usr/bin/env bash\n# a sourceable helper\n\nfoo() { :; }\n')

    bad_hits=$(printf '%s\n' "$bad_lib" | grep -nE "$SETOPT_RE" || true)
    good_hits=$(printf '%s\n' "$good_lib" | grep -nE "$SETOPT_RE" || true)

    if [ -z "$bad_hits" ]; then
        printf 'SELF-TEST FAILED: the matcher no longer flags `set -euo pipefail`.\n' >&2
        exit 1
    fi
    if [ -n "$good_hits" ]; then
        printf 'SELF-TEST FAILED: the matcher flags an option-neutral library.\n' >&2
        exit 1
    fi
    printf 'self-test OK: rejects the pre-fix header, accepts an option-neutral one.\n'
fi

if [ "$violations" -gt 0 ]; then
    printf '\n%s sourced librar(y/ies) mutate the caller shell (%s checked).\n' \
        "$violations" "$checked" >&2
    printf 'Remove the file-scope `set` and fail by return status instead:\n' >&2
    printf '    # in the library, at file scope:\n' >&2
    printf '    some_check || { return 1 2>/dev/null || exit 1; }\n' >&2
    printf '    # at the call site:\n' >&2
    printf '    . scripts/the_lib.sh || exit 1\n' >&2
    exit 1
fi

# Fail closed: a discovery step that found nothing must not report success.
MIN_EXPECTED="${MIN_EXPECTED:-1}"
if [ "$checked" -lt "$MIN_EXPECTED" ]; then
    printf 'ERROR: examined %s sourced librar(y/ies), expected >= %s - discovery has gone blind.\n' \
        "$checked" "$MIN_EXPECTED" >&2
    exit 1
fi

printf 'OK: %s sourced librar(y/ies) checked, none mutates the caller shell.\n' "$checked"
