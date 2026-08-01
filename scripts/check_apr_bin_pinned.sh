#!/usr/bin/env bash
# check_apr_bin_pinned.sh - no CI surface may invoke a bare `apr`.
#
# THE CLASS. `cargo install --path crates/apr-cli --force` writes to
# ~/.cargo/bin. If anything earlier on PATH holds an older `apr`, a bare `apr`
# invocation runs THAT one. qwen-story-daily did exactly this: it installed
# 0.61.0 and then executed a 24-day-old 0.60.0 from ~/.local/bin, so every beat
# validated stale code while reporting green.
#
# scripts/apr_bin.sh makes that DETECTABLE at runtime (it compares the binary's
# embedded git SHA against HEAD). This script makes it UNREINTRODUCIBLE: any new
# bare `apr` invocation on a CI surface fails the PR that adds it.
#
# An invocation is PINNED when it goes through one of:
#   "$APR" / ${APR} / $APR_BIN     - resolved by scripts/apr_bin.sh
#   an explicit path               - ./target/release/apr, ~/.cargo/bin/apr, /abs/apr
#   cargo run ... --bin apr        - built from the current source by definition
#
# Exit 0 = every CI-surface invocation is pinned. Exit 1 = at least one is bare.

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

# CI surfaces only. Developer convenience scripts that CI never runs are out of
# scope on purpose - the invariant being protected is "what CI executes was
# built from the commit under test", not "nobody may ever type apr".
WORKFLOW_GLOB=".github/workflows"

# Scripts reachable from a workflow. Derived, not hand-listed, so a newly wired
# script is covered the moment the workflow names it.
ci_scripts() {
    grep -rhoE 'scripts/[A-Za-z0-9_.-]+\.sh' "$WORKFLOW_GLOB"/*.yml 2>/dev/null \
        | sort -u \
        | while IFS= read -r s; do
            [ -f "$s" ] && printf '%s\n' "$s"
          done
}

MIN_EXPECTED="${MIN_EXPECTED:-1}"

violations=0
scanned=0

# A line invokes apr "bare" when `apr` appears as a command word - i.e. at the
# start of a command - and is not preceded by $, /, or - (which would make it
# $APR, a path, or part of apr-cli / apr-corpus-ingest).
# `apr` in COMMAND POSITION - the only place it can actually launch a binary.
# That means: at the start of a line, right after a shell separator
# (; & | && ||), or after a YAML `run:`. Anchoring on command position rather
# than scanning line content is what keeps prose out:
#   - name: Pillar-1 - apr vs scikit-learn ...   (a label)
#   emit_pass "B2 apr qa"                        (a message)
# Two earlier drafts got this wrong in opposite directions - one missed
# `- run: apr qa` entirely, the next flagged ten step names. Both were caught
# by the case table below, not by reading.
BARE_APR='(^|[;&|]|&&|\|\||run:)[[:space:]]*apr[[:space:]]+[a-z]'

# An absolute path whose last component is `apr`. Anchored on a leading `/`,
# `~/` or `$HOME/` so relative `target/release/apr` (correct inside a checkout)
# is untouched, and requiring the path to END at `apr` so `apr-cli`,
# `aprender-*` and `.../apr_bin.sh` do not match.
# The leading anchor is load-bearing and was wrong in the first draft: without
# it, `[A-Za-z0-9_.$/-]*` happily matched the `/apr` inside RELATIVE
# `target/release/apr`, flagging correct code. Verified against a 12-case table
# (4 absolute forms must match, 8 relative/`$APR`/`--bin apr`/prose forms must
# not). This regex class has now been gotten wrong four times in this repo; if
# you change it, re-run the table rather than reading it.
ABS_APR='(^|[[:space:]"'"'"'=(])(/|~/|\$HOME/)[A-Za-z0-9_.$/-]*/apr([[:space:]"'"'"']|$)'

check_file() {
    local f="$1" n=0
    scanned=$((scanned + 1))
    while IFS= read -r hit; do
        local lineno text trimmed
        lineno="${hit%%:*}"
        text="${hit#*:}"
        trimmed=$(printf '%s' "$text" | sed 's/^[[:space:]]*//')

        # Comments are documentation, not invocations.
        case "$trimmed" in
            '#'*) continue ;;
            *) ;;
        esac
        # YAML metadata is prose, not shell. beat-speed-nightly.yml has ten
        # step names reading "Pillar-1 - apr vs scikit-learn ..."; flagging
        # those would make the check fire on its own labels.
        if printf '%s' "$trimmed" \
            | grep -qE '^-?[[:space:]]*(name|description|title|summary|if|id|uses|shell|working-directory):'; then
            continue
        fi
        # Already pinned?
        case "$text" in
            *'$APR'*|*'${APR'*|*'target/release/apr'*|*'target/debug/apr'*|*'.cargo/bin/apr'*|*'--bin apr'*)
                continue ;;
            # qwen-story.sh's run_cmd substitutes a leading bare `apr` with
            # "$APR", so its call sites are pinned by the wrapper.
            *'run_cmd '*)
                continue ;;
            *) ;;
        esac
        printf 'BARE-APR %s:%s\n' "$f" "$lineno"
        printf '         %s\n' "$trimmed"
        n=$((n + 1))
    done < <(grep -nE "$BARE_APR" "$f" 2>/dev/null || true)

    # SECOND CLASS: an ABSOLUTE hardcoded apr path.
    #
    # This is the other half of the same defect, and the `case` above would wave
    # it straight through: `/mnt/nvme-raid0/targets/aprender/release/apr` ends in
    # `target/release/apr`, so it matched the "already pinned" list. It is not
    # pinned to anything - it names one machine's build output, which on
    # 2026-08-01 was 6 days and TWO MINOR VERSIONS stale while docs still called
    # it canonical. A release smoke-test read it and reported a meaningless pass.
    #
    # There is no correct absolute path to hardcode: `.cargo/config.toml`
    # redirects cargo's target-dir and is gitignored, so the main checkout builds
    # to /mnt/nvme-raid0/coverage/aprender while a fresh worktree builds to
    # <worktree>/target. Any absolute path is right in one and silently wrong in
    # the other. Use `. scripts/apr_bin.sh || exit 1`, which asks cargo.
    while IFS= read -r hit; do
        local lineno text trimmed
        lineno="${hit%%:*}"
        text="${hit#*:}"
        trimmed=$(printf '%s' "$text" | sed 's/^[[:space:]]*//')
        case "$trimmed" in '#'*) continue ;; *) ;; esac
        # apr_bin.sh itself documents these paths in its own comments.
        case "$f" in */apr_bin.sh|*/check_apr_bin_pinned.sh) continue ;; *) ;; esac
        printf 'ABS-APR  %s:%s\n' "$f" "$lineno"
        printf '         %s\n' "$trimmed"
        n=$((n + 1))
    done < <(grep -nE "$ABS_APR" "$f" 2>/dev/null || true)
    violations=$((violations + n))
}

for f in "$WORKFLOW_GLOB"/*.yml; do
    [ -f "$f" ] || continue
    check_file "$f"
done

while IFS= read -r s; do
    [ -n "$s" ] || continue
    check_file "$s"
done < <(ci_scripts)

if [ "$violations" -gt 0 ]; then
    printf '\n%s bare `apr` invocation(s) on CI surfaces (%s file(s) scanned).\n' \
        "$violations" "$scanned" >&2
    printf 'A bare `apr` runs whatever PATH resolves - which is how a 24-day-old\n' >&2
    printf 'binary validated a gate merged the day before. Pin it:\n' >&2
    printf '  . scripts/apr_bin.sh    # exports $APR, asserts it was built from HEAD\n' >&2
    printf '  "$APR" qa model.gguf\n' >&2
    exit 1
fi

# Fail closed: a scanner that examined nothing must not report success.
if [ "$scanned" -lt "$MIN_EXPECTED" ]; then
    printf 'ERROR: scanned %s file(s), expected >= %s - the file discovery has gone blind.\n' \
        "$scanned" "$MIN_EXPECTED" >&2
    exit 1
fi

printf 'OK: %s CI-surface file(s) scanned, every `apr` invocation is pinned.\n' "$scanned"
