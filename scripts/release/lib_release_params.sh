#!/usr/bin/env bash
# lib_release_params.sh -- the release train's identity, DERIVED, never written down (#3618).
#
# The release scripts were ported from an out-of-tree 0.68.2 bundle (#3599) and
# kept its constants: V=0.68.2, MS=12, EPIC=3477, LAST_TAG=v0.68.1, and AP set to
# an agent worktree on the operator box's RAID mount. Cutting 0.69 through them would
# have targeted the old version and an out-of-tree directory, and the path was
# invisible to check_hardcoded_paths.sh, which does not match that mount (#3592).
# scripts/check_release_scripts_derive_identity.sh now refuses both shapes.
#
# Now ONE input -- the version, passed by the operator -- and everything else is
# read from where it already lives:
#   V         the argument (X.Y.Z), refused otherwise
#   T         "v$V"
#   AP        the train's state directory: <main checkout>/target/release-train/<T>,
#             where <main checkout> is the directory holding the repo's COMMON git dir,
#             so every worktree of the repo resolves the SAME AP (prepare_bump.sh writes
#             release_notes.md there and autopilot.sh reads it). target/ is ignored.
#             RELEASE_AP overrides it at run time; it is never a literal in the tree.
#   milestone the one milestone titled exactly "$V"            (release_milestone_number)
#   epic      the one `epic`-labelled item in that milestone titled
#             "EPIC: release train $V ..."                     (release_epic_number;
#             RELEASE_EPIC overrides, digits only)
#   last tag  the highest v* tag strictly below $T              (release_last_tag)
# Each derivation that cannot find EXACTLY one answer returns 2 and says why:
# an ambiguous identity is not a default.
#
# SOURCED, so OPTION-NEUTRAL: no `set` here (check_sourced_libs_option_neutral.sh);
# every entry point reports by return status.
#     . "$REPO_ROOT/scripts/release/lib_release_params.sh" || exit 2
#     release_params "$1" || exit 2

RELEASE_REPO=paiml/aprender
RELEASE_INFRA=paiml/infra

# release_state_root <repo-root> -> the main checkout (parent of the common .git dir), or the root itself
release_state_root() {
    local root=$1 common=""
    common=$(git -C "$root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || common=""
    case "$common" in
        */.git) printf '%s\n' "${common%/.git}" ;;
        *)      printf '%s\n' "$root" ;;
    esac
}

# release_params <version> [<repo-root>] -> sets V T AP REPO INFRA; rc 2 on a bad version
release_params() {
    local v=${1:-} root=${2:-}
    [[ $v =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
        || { printf 'release: version "%s" is not X.Y.Z -- pass the version being cut as the first argument\n' "$v" >&2; return 2; }
    [ -n "$root" ] || root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 2
    V=$v
    T="v$V"
    REPO=$RELEASE_REPO
    INFRA=$RELEASE_INFRA
    AP="${RELEASE_AP:-$(release_state_root "$root")/target/release-train/$T}"
    mkdir -p "$AP" || { printf 'release: cannot create the state dir %s\n' "$AP" >&2; return 2; }
    return 0
}

# release_milestone_number -> the number of the one milestone titled exactly $V
release_milestone_number() {
    local out n
    out=$(gh api "repos/$RELEASE_REPO/milestones?state=all&per_page=100" --paginate \
          --jq ".[] | select(.title == \"$V\") | .number") \
        || { printf 'release: reading milestones failed\n' >&2; return 2; }
    n=$(printf '%s\n' "$out" | grep -c '[0-9]') || n=0
    [ "$n" = 1 ] || { printf 'release: %s milestone(s) titled "%s"; exactly 1 is required\n' "$n" "$V" >&2; return 2; }
    printf '%s\n' "$out"
}

# release_epic_number -> RELEASE_EPIC, or the one epic-labelled item in milestone $V titled "EPIC: release train $V ..."
release_epic_number() {
    local out n
    if [ -n "${RELEASE_EPIC:-}" ]; then
        case "$RELEASE_EPIC" in *[!0-9]*) printf 'release: RELEASE_EPIC=%s is not an issue number\n' "$RELEASE_EPIC" >&2; return 2 ;; esac
        printf '%s\n' "$RELEASE_EPIC"; return 0
    fi
    out=$(gh issue list --repo "$RELEASE_REPO" --milestone "$V" --label epic --state all --limit 100 \
          --json number,title --jq ".[] | select(.title == \"EPIC: release train $V\" or (.title | startswith(\"EPIC: release train $V \"))) | .number") \
        || { printf 'release: reading milestone %s failed\n' "$V" >&2; return 2; }
    n=$(printf '%s\n' "$out" | grep -c '[0-9]') || n=0
    [ "$n" = 1 ] || { printf 'release: %s item(s) in milestone %s titled "EPIC: release train %s ..."; exactly 1 is required (or set RELEASE_EPIC)\n' "$n" "$V" "$V" >&2; return 2; }
    printf '%s\n' "$out"
}

# release_last_tag <repo-root> -> the highest vX.Y.Z tag strictly below $T
release_last_tag() {
    local root=$1 t
    while IFS= read -r t; do
        case "$t" in *-*) continue ;; esac                       # no pre-release tags
        [ "$t" = "$T" ] && continue
        if printf '%s\n%s\n' "$t" "$T" | sort -V -C; then printf '%s\n' "$t"; return 0; fi
    done < <(git -C "$root" tag --list 'v[0-9]*' --sort=-v:refname)
    printf 'release: no tag below %s in %s\n' "$T" "$root" >&2
    return 2
}

# release_train_version [<repo-root>] -> "V MILESTONE LAST": the train the nightly measures (#4672, the rule
# of #4723). V is the LOWEST open milestone titled X.Y.Z strictly above LAST, the newest final vX.Y.Z tag.
# Not "the one open milestone" and not "the lowest open": released versions keep open milestones and stale
# ones sit empty, so either would name the wrong train. No final tag, a failed milestone read, or no
# milestone above LAST returns 2 and says why: the nightly reads that as NOT_MEASURED.
# Option-neutral like the rest of this file: every `$( )` that can be empty carries its own `||`, so a
# caller under `set -eo pipefail` (a workflow step) gets rc 2, never an exit from inside the lib.
release_train_version() {
    local root=${1:-} last ms t n best="" bestn=""
    [ -n "$root" ] || root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 2
    last=$(git -C "$root" tag -l 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sed 's/^v//' | sort -V | tail -n 1) || last=""
    [ -n "$last" ] || { printf 'release: no final vX.Y.Z tag in %s: the train cannot be named\n' "$root" >&2; return 2; }
    ms=$(gh api "repos/$RELEASE_REPO/milestones?state=open&per_page=100" --paginate --jq '.[] | "\(.title) \(.number)"') \
        || { printf 'release: reading the open milestones failed\n' >&2; return 2; }
    while read -r t n; do
        [[ $t =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || continue
        [ "$t" != "$last" ] && [ "$(printf '%s\n%s\n' "$last" "$t" | sort -V | tail -n 1)" = "$t" ] || continue
        if [ -z "$best" ] || [ "$(printf '%s\n%s\n' "$best" "$t" | sort -V | head -n 1)" = "$t" ]; then best=$t; bestn=$n; fi
    done <<< "$ms"
    [ -n "$best" ] || { printf 'release: no open X.Y.Z milestone above v%s: the train cannot be named\n' "$last" >&2; return 2; }
    printf '%s %s %s\n' "$best" "$bestn" "$last"
}
