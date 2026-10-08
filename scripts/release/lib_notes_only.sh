#!/usr/bin/env bash
# lib_notes_only.sh -- when a release commit's saved results stand for a notes-only successor (#4939).
#
# A release commit M was measured: deep, dogfood (the R5 receipt) and models (the CRUX-smoke GO) all
# name M. A later commit N that only rewrites the release notes would rerun all of it (about 90 min)
# although no byte that any lane builds or reads changed. notes_only M N says when that is so, and
# it is the ONE predicate every consumer uses (cut_tag, R5 in check_publish_preflight.sh, and the
# autopilot's worktree move), so the three cannot disagree on what "notes-only" means.
#
# notes_only M N holds iff ALL of:
#   - M and N are commits, M != N, and M is an ancestor of N;
#   - `git diff --raw --no-renames M N` is exactly ONE line: CHANGELOG.md, status M, a regular file
#     (mode 100644) at both ends. A rename, a copy, an added or deleted CHANGELOG, a symlink, an exec
#     bit, or any second path is not notes-only;
#   - nothing in N's tree reads CHANGELOG.md at build time (an include_str!/include_bytes! of it, or a
#     build.rs naming it): then the notes ARE a build input and the saved results did not measure them.
# The Cargo version is equal at both ends by construction: no Cargo file is in the diff. That is what
# keeps this clear of #3708, where the parent's GO was reused across a version change.
#
# It reuses MEASUREMENTS, never the notes: a consumer that reads CHANGELOG content (the release-notes
# file the tag carries) reads it again at N.
#
# SOURCED, so OPTION-NEUTRAL: no `set` here (check_sourced_libs_option_neutral.sh).
#     . "$REPO_ROOT/scripts/release/lib_notes_only.sh" || exit 2
#     notes_only "$M" "$N" && echo "M's results stand for N"
# Tested by scripts/check_release_notes_only.sh (a must-match / must-not-match case table, mutants).

# notes_only M N [GITDIR]: status 0 iff N is M plus a release-notes edit and nothing else.
notes_only() {
    local m=${1:-} n=${2:-} g=${3:-.} raw
    m=$(git -C "$g" rev-parse -q --verify "${m}^{commit}" 2>/dev/null) || return 1
    n=$(git -C "$g" rev-parse -q --verify "${n}^{commit}" 2>/dev/null) || return 1
    [ "$m" != "$n" ] || return 1
    git -C "$g" merge-base --is-ancestor "$m" "$n" 2>/dev/null || return 1
    raw=$(git -C "$g" diff --raw --no-renames --no-abbrev "$m" "$n" 2>/dev/null) || return 1
    case "$raw" in
        *$'\n'*) return 1 ;;
        ":100644 100644 "*" M"$'\t'"CHANGELOG.md") ;;
        *) return 1 ;;
    esac
    # git grep: 0 = found (a build reads the notes), 1 = none, >1 = could not look. Only 1 passes.
    git -C "$g" grep -q -E 'include_(str|bytes)!\([^)]*CHANGELOG' "$n" -- '*.rs' 2>/dev/null
    [ $? = 1 ] || return 1
    git -C "$g" grep -q -F 'CHANGELOG' "$n" -- 'build.rs' '**/build.rs' 2>/dev/null
    [ $? = 1 ] || return 1
    return 0
}

# notes_only_base N SHA...: prints the first SHA (resolved to a full id) that IS N or that N is a
# notes-only successor of; status 1 when none is. Short ids are resolved in GITDIR (default .),
# and an ambiguous or unknown one is skipped, never guessed.
notes_only_base() {
    local n=${1:-} s full nf
    shift || return 1
    nf=$(git rev-parse -q --verify "${n}^{commit}" 2>/dev/null) || return 1
    for s in "$@"; do
        full=$(git rev-parse -q --verify "${s}^{commit}" 2>/dev/null) || continue
        if [ "$full" = "$nf" ] || notes_only "$full" "$nf"; then printf '%s\n' "$full"; return 0; fi
    done
    return 1
}
