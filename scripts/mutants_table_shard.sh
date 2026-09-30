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
# Feature-gated files (ci/mutants-feature-groups.txt): each row runs as its own `cargo mutants -p PKG --features F
# --file FILE --shard K-1/N` and FILE is excluded from the default run; the outcomes are merged into outcomes.json.
# F may be empty: the row then exists for its TEST ARGS (a file whose tests live outside --lib, #4588).
# The universe (list.txt) is still the default --list of the whole diff, so a row that tests nothing leaves its
# mutants MISSING in the checker. A row whose --list for this shard is non-empty but wrote no outcomes is a dead shard.
#
#   bash scripts/mutants_table_shard.sh <diff> <K> <N> <jobs> <out> [--exclude-crate NAME]...
# exit 0 files written . 1 dead shard . 2 usage
# Seams (tests only): MUTANTS_GATE_CARGO replaces `cargo`; MUTANTS_FEATURE_GROUPS replaces the groups file.
set -uo pipefail
[ "$#" -ge 5 ] || { echo "usage: $0 <diff> <K> <N> <jobs> <out> [--exclude-crate NAME]..." >&2; exit 2; }
diff=$1 k=$2 n=$3 jobs=$4 out=$5; shift 5
cargo=${MUTANTS_GATE_CARGO:-cargo}
groups=${MUTANTS_FEATURE_GROUPS:-$(cd "$(dirname "$0")/.." && pwd)/ci/mutants-feature-groups.txt}
[ -f "$groups" ] || { echo "RED   feature-groups file $groups is missing"; exit 1; }
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
gx=() rows=()
while IFS='|' read -r g_pkg g_feat g_file g_args; do
  case "$g_pkg" in ''|'#'*) continue ;; esac
  [ -n "$g_file" ] && [ -n "$g_args" ] || { echo "RED   malformed feature-group row: $g_pkg|$g_feat|$g_file|$g_args"; exit 1; }
  case "$g_feat" in *cuda*) echo "RED   feature group $g_pkg asks for cuda: no runner has a GPU"; exit 1 ;; esac
  gx+=(--exclude "$g_file"); rows+=("$g_pkg|$g_feat|$g_file|$g_args")
done < "$groups"
# --copy-vcs true: the scratch copy keeps .git, so a test that compares HEAD with origin/main (the contracts lint
# refinement gate) runs as it does in workspace-test. Without it that test fails the unmutated baseline and the
# shard tests nothing (run 36538406766: every default-run baseline FAILED -> 3527 MISSING).
vcs=(--copy-vcs true)
"$cargo" mutants --workspace "${ex[@]}" "${gx[@]}" --in-diff "$diff" --shard "$((k - 1))/$n" "${vcs[@]}" \
  --no-times --timeout 900 -j "$jobs" --output "$out/run" -- --lib; rc=$?
oc="$out/run/mutants.out/outcomes.json"
[ -f "$oc" ] || { echo "RED   cargo mutants exited $rc and wrote no outcomes.json: the shard died"; exit 1; }
parts=("$oc") i=0
for row in "${rows[@]}"; do
  i=$((i + 1))
  IFS='|' read -r g_pkg g_feat g_file g_args <<< "$row"
  read -r -a targs <<< "$g_args"
  gsel=(-p "$g_pkg" --file "$g_file" --in-diff "$diff" --shard "$((k - 1))/$n")
  [ -z "$g_feat" ] || gsel+=(--features "$g_feat")
  "$cargo" mutants "${gsel[@]}" --list > "$out/group-$i.list" 2> "$out/group-$i.err" \
    || { echo "RED   feature group $g_pkg ($g_feat) --list failed -- $(tail -1 "$out/group-$i.err")"; exit 1; }
  gn=$(grep -c . "$out/group-$i.list")
  [ "$gn" -gt 0 ] || { echo "group $i $g_pkg ($g_feat): 0 mutants in this shard"; continue; }
  "$cargo" mutants "${gsel[@]}" "${vcs[@]}" --no-times --timeout 900 -j "$jobs" --output "$out/group-$i" -- "${targs[@]}"; grc=$?
  goc="$out/group-$i/mutants.out/outcomes.json"
  [ -f "$goc" ] || { echo "RED   feature group $g_pkg ($g_feat) exited $grc and wrote no outcomes.json for $gn mutant(s)"; exit 1; }
  echo "group $i $g_pkg --features $g_feat: $gn mutant(s), cargo mutants rc $grc"
  parts+=("$goc")
done
# perl + core JSON::PP, not python: the sovereign-ci image the shards run in ships no python3.
perl -MJSON::PP -e 'my $o = shift; my @all; for my $p (@ARGV) { open(my $f, "<", $p) or die "$p: $!"; local $/;
  push @all, @{ JSON::PP->new->decode(<$f>)->{outcomes} } } open(my $w, ">", $o) or die "$o: $!";
  print $w JSON::PP->new->encode({ outcomes => \@all })' "$out/outcomes.json" "${parts[@]}" \
  || { echo "RED   could not merge outcomes"; exit 1; }
echo "shard $k/$n done (default run rc $rc, ${#rows[@]} feature group(s); the checker judges the outcomes)"
