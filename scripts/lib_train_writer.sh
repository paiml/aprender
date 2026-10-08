#!/usr/bin/env bash
# lib_train_writer.sh — who may write a DERIVED file: the release train, and no one else.
#
# Sourced, never run. Option-neutral (no `set`): it fails by return status
# (scripts/check_sourced_libs_option_neutral.sh). Moved unchanged out of
# check_census_derived.sh (#3569) so every derived file shares ONE writer
# predicate: contracts/census.json there, the README count blocks in
# check_generated_counts_untouched.sh and check_readme_claims.sh (#4526 GEN-001).
#
#   is_train BRANCH     rc 0 iff this run is the train: a release/X.Y.Z branch of
#                       this repository, or CENSUS_WRITER=train; never a fork PR
#   current_branch ROOT the branch this run is for (PR source, push ref, or HEAD)
#   is_fork_pr          rc 0 iff the event payload names a head repo other than the base


# ── the writer exemption ─────────────────────────────────────────────────────
# A branch NAME is chosen by whoever pushes it, so on a pull_request the name alone
# is not the train: a fork can call its branch release/1.0.0. The train is a branch
# of THIS repository, so a PR whose head repo is not its base repo never is.
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

is_train() { # is_train BRANCH -> 0 iff this run is the one writer of derived files
    is_fork_pr && return 1   # FIRST: no env a fork PR can set outranks where it came from
    [ "${CENSUS_WRITER:-}" = train ] && return 0
    [[ "$1" =~ ^release/[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

# GITHUB_HEAD_REF on a pull_request; GITHUB_REF_NAME on a push (actions/checkout
# leaves a detached HEAD, where rev-parse --abbrev-ref says only "HEAD").
current_branch() {
    local b="${GITHUB_HEAD_REF:-}"
    [ -n "$b" ] || b="${GITHUB_REF_NAME:-}"
    [ -n "$b" ] || b=$(git -C "$1" rev-parse --abbrev-ref HEAD 2>/dev/null) || b=""
    printf '%s' "$b"
}
