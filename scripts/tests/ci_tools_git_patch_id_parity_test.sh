#!/usr/bin/env bash
# ci_tools_git_patch_id_parity_test.sh -- PMAT-4800: the identical-output case table for
# `aprender-ci-tools git-patch-id` against scripts/lib/git_patch_id.py (C301: no Python in
# the build; the .py is the EXTERNAL VALIDATOR here, never a build step).
#
# Every case runs one diff through both sides in one mode; stdout must be byte-identical
# and the exit status equal. Inputs are hand-written edge cases plus real diffs made in a
# scratch repo (the pr_review_patch_id.sh self-test fixture, multi-commit `git log -p` and
# `git format-patch` streams). The bin is then also compared with native `git patch-id`
# on PATH: every input and mode on git >= 2.40, the text-only inputs on older git.
#
# Its own script rather than a section of ci_tools_py_parity_test.sh: the sibling ports on
# this crate each bump that file's EXPECTED_CASES, and a shared counter is a merge conflict.
#
# Not vacuous (L25): the run fails if fewer cases ran than the table declares, and a
# planted mismatch must be reported as one before any real case counts.
#
# Usage: scripts/tests/ci_tools_git_patch_id_parity_test.sh   (CI_TOOLS_BIN=<path> to skip the build)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python3}"
GPID_PY="$ROOT/scripts/lib/git_patch_id.py"
MODES=(--stable --verbatim --unstable "")
EXPECTED_CASES=111

BIN="${CI_TOOLS_BIN:-}"
if [[ -z "$BIN" ]]; then
    cargo build -q -p aprender-ci-tools
    tdir="$(cargo metadata --no-deps --format-version 1 | grep -o '"target_directory":"[^"]*"' | cut -d'"' -f4)"
    BIN="$tdir/debug/aprender-ci-tools"
fi
if [[ ! -x "$BIN" ]]; then
    echo "FAIL: no aprender-ci-tools binary at $BIN" >&2
    exit 1
fi
BIN="$(realpath -e -- "$BIN")"

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp:?}"' EXIT
pass=0
fail=0

# side OUT IN CMD...: stdout to OUT, print the exit status.
side() {
    local out="$1" in="$2" rc=0
    shift 2
    "$@" <"$in" >"$out" 2>/dev/null || rc=$?
    echo "$rc"
}

# compare NAME IN MODE: .py vs bin, one mode ("" = no argument).
compare() {
    local name="$1" in="$2" mode="$3" a b
    local args=()
    [[ -n "$mode" ]] && args=("$mode")
    a="$(side "$tmp/py.out" "$in" "$PY" "$GPID_PY" "${args[@]}")"
    b="$(side "$tmp/rs.out" "$in" "$BIN" git-patch-id "${args[@]}")"
    if [[ "$a" == "$b" ]] && cmp -s "$tmp/py.out" "$tmp/rs.out"; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "MISMATCH: $name ${mode:-<none>} (py rc=$a, rs rc=$b)" >&2
        diff "$tmp/py.out" "$tmp/rs.out" | head -n 6 >&2 || true
    fi
}

# all_modes NAME IN: the case in every mode.
all_modes() {
    local m
    for m in "${MODES[@]}"; do
        compare "$1" "$2" "$m"
    done
}

# --- L25: a planted mismatch must be seen as one. -------------------------------------
printf 'diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n' >"$tmp/plant.diff"
real_bin="$BIN"
printf '#!/bin/sh\necho 0000 0000\n' >"$tmp/liar"
chmod +x "$tmp/liar"
BIN="$tmp/liar"
compare plant "$tmp/plant.diff" --stable 2>/dev/null
BIN="$real_bin"
if [[ "$fail" -ne 1 || "$pass" -ne 0 ]]; then
    echo "FAIL: the planted mismatch was not reported (pass=$pass fail=$fail)" >&2
    exit 1
fi
pass=0
fail=0

# --- Hand-written edge cases (each in all four modes). ---------------------------------
hw() { # NAME PRINTF_FORMAT
    printf "$2" >"$tmp/hw.diff"
    all_modes "$1" "$tmp/hw.diff"
}
P='diff --git a/x b/x\nindex 1111111111111111111111111111111111111111..2222222222222222222222222222222222222222 100644\n--- a/x\n+++ b/x\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n'
hw "empty input" ''
hw "no diff at all" 'hello\nworld\n'
hw "one patch" "$P"
hw "no final newline" 'diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b'
hw "CRLF throughout" 'diff --git a/x b/x\r\n--- a/x\r\n+++ b/x\r\n@@ -1 +1 @@\r\n-a \r\n+a\r\n'
hw "lone CR splits a line" 'diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,2 +1,2 @@\n-a\r-b\n+a\r+c\n'
hw "VT/FF stay in the line" 'diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\vb\n+a\fb\n'
hw "two patches, commit oids" "commit 0123456789abcdef0123456789abcdef01234567\n${P}commit ABCDEF0123456789ABCDEF0123456789ABCDEF01\n${P}"
hw "From oid (format-patch)" "From 0123456789abcdef0123456789abcdef01234567 Mon Sep 17 00:00:00 2001\nSubject: x\n\n---\n${P}-- \n2.34.1\n"
hw "bare 40-hex line" "${P}fedcba9876543210fedcba9876543210fedcba98\n${P}"
hw "39-hex is not an oid" "${P}fedcba9876543210fedcba9876543210fedcba9\n${P}"
hw "no-newline marker" 'diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n\\ No newline at end of file\n+b\n'
hw "short backslash line" 'diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n\\ short\n+b\n'
hw "binary marker" 'diff --git a/b b/b\nindex 1111111111111111111111111111111111111111..2222222222222222222222222222222222222222 100644\nBinary files a/b and b/b differ\n'
hw "binary then text" "diff --git a/b b/b\nindex 1111111111111111111111111111111111111111..2222222222222222222222222222222222222222\nGIT binary patch\nliteral 3\nKcmZQzU|;|M0RR91\n\n${P}"
hw "index without .." 'diff --git a/x b/x\nindex deadbeef\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\nBinary files differ\n'
hw "unparsable hunk header" 'diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n@@ -x +y @@\n-c\n+d\n'
hw "second hunk" 'diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n@@ -9,2 +9,2 @@\n c\n-d\n+e\n'
hw "trailing junk ends the patch" "${P}junk line\n+not hashed\n"
hw "header line not alpha" 'diff --git a/x b/x\n1 weird\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n'
hw "non-UTF-8 bytes" 'diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-\377\376\n+\200\n'

# --- Real diffs from a scratch repo. ---------------------------------------------------
(
    cd "$tmp" && git init -q ws && cd ws || exit 2
    ci() { git -c user.name=t -c user.email=t@t -c core.hooksPath=/dev/null commit -qm "$1"; }
    # The pr_review_patch_id.sh self-test fixture, byte for byte.
    printf 'a \n\tb\r\nc' >f
    printf '\000\001bin' >b.bin
    git add . && ci 1 || exit 2
    printf 'a  \n\tb\nc\n' >f
    printf '\000\002bin' >b.bin
    echo new >g
    git add . && ci 2 || exit 2
    git diff --no-color --full-index HEAD^ HEAD >"$tmp/ws.diff"
    git diff --no-color --full-index --binary HEAD^ HEAD >"$tmp/wsbin.diff"
    # More history: renames, deletes, mode changes, many hunks.
    seq 1 200 >n && git add n && ci 3 || exit 2
    sed -i -e 's/^1\(.\)$/X\1/' -e '150d' n && git add n && ci 4 || exit 2
    git mv g h && chmod +x f && git add . && ci 5 || exit 2
    git rm -q b.bin && ci 6 || exit 2
    git log -p --no-color --full-index >"$tmp/log.diff"
    git log -p --no-color --full-index --binary --reverse >"$tmp/logrev.diff"
    git format-patch -q --stdout --root >"$tmp/fp.diff"
    # Commits 3..5 only: hunks, a rename and a mode change, no binary hunk.
    git log -p --no-color --full-index HEAD~4..HEAD~1 >"$tmp/text.diff"
) || { echo "FAIL: could not build the scratch repo diffs" >&2; exit 1; }

for f in ws wsbin log logrev fp text; do
    # Inside `( ) ||` set -e is off, so a failed git leaves an empty file that both
    # sides would "agree" on.
    if [[ ! -s "$tmp/$f.diff" ]]; then
        echo "FAIL: scratch repo diff $f.diff is empty" >&2
        exit 1
    fi
    all_modes "repo:$f" "$tmp/$f.diff"
done

# The bad input: an unknown mode is refused by both with status 2 and no stdout.
for bad in --bogus stable -s; do
    a="$(side "$tmp/py.out" "$tmp/ws.diff" "$PY" "$GPID_PY" "$bad")"
    b="$(side "$tmp/rs.out" "$tmp/ws.diff" "$BIN" git-patch-id "$bad")"
    if [[ "$a" == 2 && "$b" == 2 ]] && cmp -s "$tmp/py.out" "$tmp/rs.out" && [[ ! -s "$tmp/rs.out" ]]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "MISMATCH: bad mode $bad (py rc=$a, rs rc=$b)" >&2
    fi
done

ran=$((pass + fail))
echo "git-patch-id parity: $pass passed, $fail failed, $ran of $EXPECTED_CASES declared"
if [[ "$fail" -ne 0 ]]; then
    exit 1
fi
if [[ "$ran" -ne "$EXPECTED_CASES" ]]; then
    echo "FAIL: ran $ran cases, the table declares $EXPECTED_CASES" >&2
    exit 1
fi

# --- Native git on PATH: its own verdict, never part of the count above. ----------------
# The port follows git 2.53. A git without --verbatim (< 2.40) hashes binary hunks by
# an older rule (measured on 2.34.1: every diff with a binary hunk differs, every
# text-only one matches), so there only the text-only inputs are compared and the
# binary-bearing ones are NOT_MEASURED, never a pass.
native=0
native_fail=0
native_cmp() { # MODE DIFF
    local a b
    a="$(git patch-id "$1" <"$2")"
    b="$("$BIN" git-patch-id "$1" <"$2")"
    native=$((native + 1))
    if [[ -z "$a" || "$a" != "$b" ]]; then
        native_fail=$((native_fail + 1))
        echo "MISMATCH: native ${2##*/} $1 native=[$a] bin=[$b]" >&2
    fi
}
if git patch-id --verbatim </dev/null >/dev/null 2>&1; then
    native_modes=(--stable --unstable --verbatim)
    native_diffs=(ws wsbin log logrev fp text plant)
    native_want=21
    unmeasured=""
else
    native_modes=(--stable --unstable)
    native_diffs=(text plant)
    native_want=4
    unmeasured="; binary-bearing ws wsbin log logrev fp and --verbatim: NOT_MEASURED (git < 2.40)"
fi
for m in "${native_modes[@]}"; do
    for d in "${native_diffs[@]}"; do
        native_cmp "$m" "$tmp/$d.diff"
    done
done
echo "native: $((native - native_fail)) of $native equal to $(git --version)$unmeasured"
if [[ "$native_fail" -ne 0 || "$native" -ne "$native_want" ]]; then
    exit 1
fi
