#!/usr/bin/env bash
# quick_tier_shard.sh -- hand one workspace-test shard its share of the quick tier's nextest calls
#
# The quick tier ran every selected crate on shard 1 while shards 2 and 3 sat idle (36 / 3 / 3 min on a
# merge group). stdin is quick_tier_groups.sh's output (one line per nextest call, crates space-separated);
# this script keeps, in every line, the crates whose ordinal in the whole listing is SHARD-1 mod N, and drops
# a line that ends up empty. A subset of a call's crates is still a call that holds no crate together with
# one that raises it, so the feature-unification rule of quick_tier_groups.sh holds.
# Over the N shards every crate appears exactly once; the run step and the Σ-executed step both filter
# the same listing the same way, so each shard owes exactly what it ran.
#
# USAGE  quick_tier_shard.sh SHARD N < calls     SHARD in 1..N
#        quick_tier_shard.sh --self-test         the case table
# EXIT   0 printed / table green · 1 table broke · 3 caller error
set -euo pipefail

filter() { # $1 shard, $2 n; stdin calls
  perl -e '
    my ($s, $n) = @ARGV; my $i = 0;
    while (my $l = <STDIN>) {
      chomp $l; my @keep;
      for my $c (split " ", $l) { push @keep, $c if $i % $n == $s - 1; $i++; }
      print join(" ", @keep), "\n" if @keep;
    }' "$1" "$2"
}

if [ "${1:-}" = "--self-test" ]; then
  fail=0
  chk() { # name want shard n input
    got="$(printf '%b' "$5" | filter "$3" "$4" | tr '\n' '|')"
    [ "$got" = "$2" ] || { echo "RED $1: want '$2' got '$got'"; fail=1; }
  }
  in='a b c\nd e\nf\n'
  chk "shard1 of 3" "a|d|" 1 3 "$in"
  chk "shard2 of 3" "b|e|" 2 3 "$in"
  chk "shard3 of 3" "c|f|" 3 3 "$in"
  chk "n=1 keeps all" "a b c|d e|f|" 1 1 "$in"
  chk "empty line dropped" "a|" 1 3 "a\n"
  chk "shard beyond list is empty" "" 3 3 "a\n"
  # every crate exactly once across shards
  all="$(for s in 1 2 3; do printf '%b' "$in" | filter "$s" 3; done | tr ' ' '\n' | sort | tr '\n' ' ')"
  [ "$all" = "a b c d e f " ] || { echo "RED partition: got '$all'"; fail=1; }
  [ "$fail" = 0 ] && echo "quick_tier_shard: 7 cases green"
  exit "$fail"
fi

s="${1:-}"; n="${2:-}"
case "$s$n" in ''|*[!0-9]*) echo "usage: quick_tier_shard.sh SHARD N | --self-test" >&2; exit 3 ;; esac
[ "$s" -ge 1 ] && [ "$s" -le "$n" ] || { echo "shard $s not in 1..$n" >&2; exit 3; }
filter "$s" "$n"
