#!/usr/bin/env bash
# mutants_table_shard.sh -- one CI shard of the survivor table (#4587 A', operator 2026-09-29 06:36Z).
#
# For a PR listed in ci/mutants-table-refs.txt the one-job `mutants` section cannot judge the diff (it holds ~60
# mutants in its hour), so the `mutants-shard` matrix judges it in N parts and `mutants-table` verifies the union
# with scripts/ci/mutants_survivor_table.py. This script is ONE part. It writes, under OUT:
#   list.txt       the FULL universe (`cargo mutants --list` of the whole diff, same flags as mutants_diff_gate.sh);
#                  every shard writes it, and the checker refuses shards whose universes differ
#   outcomes.json  cargo-mutants' outcomes for this shard's slice (--shard K-1/N, cargo-mutants counts from 0)
#   meta.json      shard, shards, head sha (git rev-parse HEAD), sha256 of the diff, run id/attempt
# It judges nothing: survivors are the checker's call. It is RED only when it could not produce the three files --
# a list that fails, or a run that wrote no outcomes.json, is a dead shard, never "0 mutants".
#
# --timeout 900 per mutant, not the section's 300: a mutant that times out is counted SURVIVED by the checker, so
# a longer timeout can only turn a contention TIMEOUT into a measured caught/missed. It judges nothing more leniently.
#
#   bash scripts/mutants_table_shard.sh <diff> <K> <N> <jobs> <out> [--exclude-crate NAME]...
# exit 0 files written . 1 dead shard . 2 usage
# Seam (tests only): MUTANTS_GATE_CARGO replaces `cargo`.
set -uo pipefail
[ "$#" -ge 5 ] || { echo "usage: $0 <diff> <K> <N> <jobs> <out> [--exclude-crate NAME]..." >&2; exit 2; }
diff=$1 k=$2 n=$3 jobs=$4 out=$5; shift 5
cargo=${MUTANTS_GATE_CARGO:-cargo}
case "$k$n$jobs" in *[!0-9]*) echo "K, N and jobs must be integers" >&2; exit 2 ;; esac
[ "$k" -ge 1 ] && [ "$k" -le "$n" ] || { echo "K must be in 1..N" >&2; exit 2; }
ex=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --exclude-crate) ex+=(--exclude "crates/$2/**"); shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -s "$diff" ] || { echo "RED   empty or missing diff $diff: a listed PR always has one"; exit 1; }
mkdir -p "$out" || exit 1
head_sha=$(git -c safe.directory="*" rev-parse HEAD) || { echo "RED   no git HEAD"; exit 1; }
diff_sha=$(sha256sum "$diff" | cut -d' ' -f1)
printf '{"shard": %d, "shards": %d, "head_sha": "%s", "diff_sha256": "%s", "run_id": "%s", "run_attempt": "%s"}\n' \
  "$k" "$n" "$head_sha" "$diff_sha" "${GITHUB_RUN_ID:-local}" "${GITHUB_RUN_ATTEMPT:-0}" > "$out/meta.json"
"$cargo" mutants --workspace "${ex[@]}" --in-diff "$diff" --list > "$out/list.txt" 2> "$out/list.err"; rc=$?
[ "$rc" -eq 0 ] || { echo "RED   cargo mutants --list exited $rc -- $(tail -1 "$out/list.err")"; exit 1; }
echo "shard $k/$n of $(grep -c . "$out/list.txt") listed mutant(s), head $head_sha, diff sha256 $diff_sha, -j $jobs"
"$cargo" mutants --workspace "${ex[@]}" --in-diff "$diff" --shard "$((k - 1))/$n" \
  --no-times --timeout 900 -j "$jobs" --output "$out/run" -- --lib; rc=$?
oc="$out/run/mutants.out/outcomes.json"
[ -f "$oc" ] || { echo "RED   cargo mutants exited $rc and wrote no outcomes.json: the shard died"; exit 1; }
cp "$oc" "$out/outcomes.json" || exit 1
echo "shard $k/$n done (cargo mutants rc $rc; the checker judges the outcomes)"
