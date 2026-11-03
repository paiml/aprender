#!/usr/bin/env bash
# check_no_bash_c_rm.sh - no `bash -c` / `sh -c` wrapper may contain `rm`.
#
# THE CLASS. The agent permission checker cannot parse a quoted `-c` script,
# so "This shell -c script runs rm and could not be checked" stalls the session
# until someone clicks (aprender-f5 sat 3h on exactly that, 2026-11-03). The
# shape is the defect: cp a probe to a temp name, run it under `bash -c`, `rm`
# it in the same string. The fix is a committed script run by path, with its
# temp files from `mktemp -d` under $TMPDIR and cleanup in a `trap` (the rm
# lives in the script file, never inside a quoted `-c` string).
#
# TIER. LAB (C324): not a merge or release gate, so it lives in scripts/lab/ and runs nightly on an idle machine.
#
# Scans scripts/ and ci/explicit-test-commands.d/. Exit 0 = clean,
# 1 = at least one wrapper contains rm, 2 = cannot measure.
# `--self-test` plants a wrapper containing rm and proves the scan goes RED.

set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

# Prints "file:line: text" for every -c wrapper whose quoted body has a word `rm`.
scan_file() {
    awk -v F="$1" '
    function flush(   body) {
        body = acc; acc = ""
        if (body ~ /(^|[^[:alnum:]_.\/-])rm([[:space:]]|$)/) printf "%s:%d: %s\n", F, startln, substr(body, 1, 100)
    }
    {
        line = $0
        if (!inq) {
            if (match(line, /(^|[[:space:];|&(])(bash|sh|zsh|dash)[[:space:]]+(-[a-zA-Z]*)?c[[:space:]]+["\047]/)) {
                rest = substr(line, RSTART + RLENGTH - 1)
                q = substr(rest, 1, 1); rest = substr(rest, 2)
                inq = 1; startln = NR; acc = ""
                line = rest
            } else next
        }
        # inside a quoted body: take text up to the closing quote (double: skip \" escapes)
        n = length(line)
        for (i = 1; i <= n; i++) {
            ch = substr(line, i, 1)
            if (q == "\"" && ch == "\\") { acc = acc ch substr(line, i + 1, 1); i++; continue }
            if (ch == q) { flush(); inq = 0; break }
            acc = acc ch
        }
        if (inq) acc = acc " "
    }
    END { if (inq) flush() }' "$1"
}

scan_tree() {
    local root=$1 f
    while IFS= read -r f; do
        case "$f" in
        */check_no_bash_c_rm.sh) continue ;; # this file spells the pattern in its own docs and self-test
        esac
        scan_file "$f"
    done < <(find "$root"/scripts "$root"/ci/explicit-test-commands.d -type f \( -name '*.sh' -o -name '*.cmd' -o -name '*.md' \) 2>/dev/null | sort)
}

self_test() {
    local d rc=0 out
    d=$(mktemp -d "${TMPDIR:-/tmp}/bashc-rm.XXXXXX") || return 2
    trap 'rm -rf -- "$d"' RETURN
    mkdir -p "$d/scripts" "$d/ci/explicit-test-commands.d"
    # must-flag cases (the f5 shape, a multi-line body, sh -c, double quotes)
    printf '%s\n' 'cd x && bash -c "cp $P ./probe_tmp.sh && bash ./probe_tmp.sh; rm -f ./probe_tmp.sh"' > "$d/scripts/bad1.sh"
    printf '%s\n' "sh -c 'a=1" "rm -rf \"\$a\"'" > "$d/scripts/bad2.sh"
    printf '%s\n' "env X=1 bash -lc 'echo hi; rm x'" > "$d/ci/explicit-test-commands.d/bad3.cmd"
    # must-not-flag cases (no rm; rm outside the string; rm inside a word; a path)
    printf '%s\n' "bash -c 'echo hi'" "rm -f x" "bash -c 'git perform --thing; ls /usr/bin/rmdir-x'" "bash -c 'echo firm'" "bash scripts/probe.sh" > "$d/scripts/good.sh"
    out=$(scan_tree "$d") || true
    for want in bad1.sh bad2.sh bad3.cmd; do
        printf '%s\n' "$out" | grep -q "$want" || { echo "FAIL self-test: planted $want not flagged"; rc=1; }
    done
    if printf '%s\n' "$out" | grep -q "good.sh"; then echo "FAIL self-test: clean file flagged: $(printf '%s\n' "$out" | grep good.sh)"; rc=1; fi
    [ "$rc" -eq 0 ] && echo "self-test: PASS"
    return "$rc"
}

case "${1:-}" in
--self-test) self_test; exit $? ;;
"") ;;
*) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac

out=$(scan_tree .)
if [ -n "$out" ]; then
    printf '%s\n' "$out"
    echo "FAIL: bash -c / sh -c wrapper contains rm; use a committed script run by path with mktemp -d + trap cleanup"
    exit 1
fi
echo "OK: no bash -c / sh -c wrapper contains rm"
