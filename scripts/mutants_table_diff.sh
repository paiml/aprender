#!/usr/bin/env bash
# mutants_table_diff.sh -- the PR diff the survivor table is bound to (#4587 A').
#
# mutants-shard (every matrix job) and mutants-table (the verifier) each run this on a checkout of the PR HEAD sha,
# so they compute the SAME bytes: `git diff <merge-base>...HEAD`, where the merge-base of the head with the base
# branch does not move when the base branch does. The checker compares sha256 of this file with every shard's.
#
#   bash scripts/mutants_table_diff.sh <base-ref> <out-diff>
# exit 0 written . 1 no merge-base / git failure . 2 usage
set -euo pipefail
[ "$#" -eq 2 ] || { echo "usage: $0 <base-ref> <out-diff>" >&2; exit 2; }
base=$1 out=$2
git fetch --no-tags origin "$base"
mb=$(git merge-base HEAD "origin/$base") || { echo "RED   no merge-base of HEAD with origin/$base" >&2; exit 1; }
git diff "$mb"...HEAD > "$out"
echo "head $(git rev-parse HEAD) merge-base $mb diff sha256 $(sha256sum "$out" | cut -d' ' -f1) ($(wc -l < "$out") lines)"
