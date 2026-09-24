#!/usr/bin/env bash
# github_snapshot.sh — write one tracked GitHub snapshot for extract:json (ONT-4f, aprender#4330).
#
#   scripts/github_snapshot.sh repo         <owner>/<repo>
#   scripts/github_snapshot.sh issue        <owner>/<repo> <number>
#   scripts/github_snapshot.sh pull-request <owner>/<repo> <number>
#   scripts/github_snapshot.sh milestone    <owner>/<repo> <number>
#
# Writes evidence/github/<type>/<ref-slug>.json and prints its path. The ref is `<owner>/<repo>@<sha>` for a repo
# (its default-branch commit) and `<owner>/<repo>#<n>@<updated_at>` for the rest; the slug maps `/` -> `__`,
# `#` -> `--`, `:` -> `-` (json.rs `ref_slug`). Every field is picked by name, so a snapshot carries only scalar
# keys and no nested object extract:json would have to map. The extractor never calls the API: these files are
# its input, and a file whose name disagrees with its own sha / updated_at is refused.
set -euo pipefail

usage() {
    printf 'usage: %s repo|issue|pull-request|milestone <owner>/<repo> [<number>]\n' "$0" >&2
    exit 2
}

[ "$#" -ge 2 ] || usage
kind="$1"
repo="$2"
num="${3:-}"
case "$kind" in
    repo) [ -z "$num" ] || usage ;;
    issue | pull-request | milestone) [ -n "$num" ] || usage ;;
    *) usage ;;
esac

root="$(git rev-parse --show-toplevel)"
tmp="$(mktemp)"
cleanup() { rm -f "$tmp"; }
trap cleanup EXIT

case "$kind" in
    repo)
        sha="$(gh api "repos/$repo/commits/HEAD" --jq .sha)"
        gh api "repos/$repo" --jq "{full_name, default_branch, sha: \"$sha\", visibility, html_url}" >"$tmp"
        ;;
    issue)
        gh api "repos/$repo/issues/$num" --jq "{repository: \"$repo\", number, title, state, updated_at,
            milestone_number: .milestone.number, html_url}" >"$tmp"
        ;;
    pull-request)
        gh api "repos/$repo/pulls/$num" --jq "{repository: \"$repo\", number, title,
            state: (if .merged_at then \"merged\" else .state end), merged_at, updated_at,
            base_repo: .base.repo.full_name, base_ref: .base.ref, head_sha: .head.sha, html_url}" >"$tmp"
        ;;
    milestone)
        gh api "repos/$repo/milestones/$num" --jq "{repository: \"$repo\", number, title, state, updated_at,
            html_url}" >"$tmp"
        ;;
esac

if [ "$kind" = repo ]; then
    ref="$repo@$(jq -r .sha "$tmp")"
else
    ref="$repo#$num@$(jq -r .updated_at "$tmp")"
fi
slug="${ref//\//__}"
slug="${slug//#/--}"
slug="${slug//:/-}"
dir="$root/evidence/github/$kind"
mkdir -p "$dir"
jq -S . "$tmp" >"$dir/$slug.json"
printf '%s\n' "evidence/github/$kind/$slug.json"
