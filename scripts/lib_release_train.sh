#!/usr/bin/env bash
# lib_release_train.sh — "is this run the release train?", ONE definition (#3569, GEN-001 #4526).
#
# Two derived artifacts have exactly one writer, the release train (T-0):
#   * contracts/census.json                       (scripts/check_census_derived.sh)
#   * the README's CONTRACT_COUNT block            (scripts/check_readme_derived.sh,
#                                                   and the equality rule in scripts/check_readme_claims.sh)
# Both guards source this file so the train is decided in one place. A second copy of
# the rule is how two guards come to disagree about who the writer is.
#
# A branch NAME is chosen by whoever pushes it, so on a pull_request the name alone
# is not the train: a fork can call its branch release/1.0.0. The train is a branch
# of THIS repository, so a PR whose head repo is not its base repo never is.
# Sourced, never executed: no `set` here, the caller owns its shell options.

is_fork_pr() { # rc 0 iff the event payload names a head repo other than the base repo
    local ev="${GITHUB_EVENT_PATH:-}" head base
    # FAIL CLOSED: a pull_request run whose payload cannot be read cannot prove it is not a fork.
    if [ -z "$ev" ] || [ ! -r "$ev" ]; then
        case "${GITHUB_EVENT_NAME:-}" in pull_request*) return 0 ;; *) return 1 ;; esac
    fi
    head=$(jq -r '.pull_request.head.repo.full_name // empty' "$ev" 2>/dev/null) || head=""
    base=$(jq -r '.pull_request.base.repo.full_name // empty' "$ev" 2>/dev/null) || base=""
    [ -n "$head" ] && [ "$head" != "$base" ]
}

is_train() { # is_train BRANCH -> 0 iff this run is the one writer of the derived artifacts
    is_fork_pr && return 1   # FIRST: no env a fork PR can set outranks where it came from
    [ "${CENSUS_WRITER:-}" = train ] && return 0
    [[ "$1" =~ ^release/[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

# GITHUB_HEAD_REF on a pull_request; GITHUB_REF_NAME on a push (actions/checkout
# leaves a detached HEAD, where rev-parse --abbrev-ref says only "HEAD").
current_branch() { # current_branch ROOT
    local b="${GITHUB_HEAD_REF:-}"
    [ -n "$b" ] || b="${GITHUB_REF_NAME:-}"
    [ -n "$b" ] || b=$(git -C "$1" rev-parse --abbrev-ref HEAD 2>/dev/null) || b=""
    printf '%s' "$b"
}
