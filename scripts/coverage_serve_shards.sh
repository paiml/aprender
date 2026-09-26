#!/usr/bin/env bash
# Split aprender-serve's lib tests into per-module shards for `make coverage` (#4023).
#
# Why: one process running all ~16k aprender-serve lib tests builds up memory across tests
# (#4028): 30 GB peak single-threaded and 45 GB at 22 threads, measured on gx10, while each
# top-level module alone stays small. On yoga's 28 GB box earlyoom SIGTERMed it under llvm-cov
# (coverage-nightly run 35868368976), so the nightly measured nothing for the largest crate.
# Running the suite as several processes, one module group each, keeps every process small;
# the `--no-report` runs' profiles are merged by one final `cargo llvm-cov report`.
#
# Usage: coverage_serve_shards.sh <test-list> <skips-file> <out-dir> [<solo-file>]
#   <test-list>   the binary's `--list` output (`path: test` lines; anything else is ignored)
#   <skips-file>  scripts/coverage-skips.txt (exact paths; #-comments and blanks ignored)
#   <out-dir>     receives shard-NN-<module|pack>.txt, one exact test path per line, and
#                 solo-NN.txt (one test each) for every entry of <solo-file>
#   <solo-file>   scripts/coverage-solo.txt: tests run in their own process and measured.
#                 An entry that is not a listed test is an error, so a rename cannot hide it.
#
# Modules with >= ALONE tests get a shard of their own; the rest are packed, largest first,
# into shards of at most PACK tests. A module is never split, so each process holds one
# module's state. The partition is CHECKED before writing: every listed test not skipped is
# in exactly one shard. A partition that loses or duplicates a test exits 1 and writes nothing.
#
# Ported from coverage_serve_shards.py (CB-200, car #4429): same arguments, files and summary
# line. Plain POSIX awk + C-locale sort, so it runs on the mac hosts' BSD awk too.
set -euo pipefail
export LC_ALL=C

ALONE=1000
PACK=2000
# Modules that build up memory in ONE process even at 4 threads on yoga (a real RTX 4060): the
# instrumented `gpu` shard was SIGTERMed at 25.9 GB (22 threads, run 35881004821) and at 26.5 GB
# (4 threads, run 35885731831). They are chunked into processes of at most DEEP_CHUNK tests.
DEEP="gpu"
DEEP_CHUNK=200

die() { echo "coverage_serve_shards: $*" >&2; exit 1; }

# Entries of a skips/solo file: trimmed, blanks and #-comments dropped.
entries() { sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' "$1" | grep -v -e '^$' -e '^#' || true; }

# Lines of $2 that are not lines of $1, in $2's order.
minus() { awk 'FILENAME == ARGV[1] { drop[$0] = 1; next } !($0 in drop)' "$1" "$2"; }

# The shard plan for the sorted non-solo tests in $1: `index<TAB>label<TAB>test` lines.
plan() {
  awk -F'::' '{ n[$1]++ } END { for (m in n) print n[m] "\t" m }' "$1" | sort -t "$(printf '\t')" -k1,1nr -k2,2 |
    awk -v alone="$ALONE" -v pack="$PACK" -v deep="$DEEP" -v chunk="$DEEP_CHUNK" -f <(plan_awk) - "$1"
}

plan_awk() {
  cat <<'AWK'
BEGIN { FS = "\t" }
FILENAME == "-" { order[++nm] = $2; next }
{ split($0, p, "::"); m = p[1]; t[m, ++c[m]] = $0 }
function emit(label, m, from, to,   i) { for (i = from; i <= to; i++) print s "\t" label "\t" t[m, i]; s++ }
function flush(   k) { if (np == 0) return; for (k = 1; k <= np; k++) print s "\tpack\t" pk[k]; s++; np = 0 }
function add(m,   i) { for (i = 1; i <= c[m]; i++) pk[++np] = t[m, i] }
END {
  s = 0; np = 0
  for (j = 1; j <= nm; j++) {
    m = order[j]
    if (m == deep) { for (f = 1; f <= c[m]; f += chunk) emit(sprintf("%s.%02d", m, (f - 1) / chunk), m, f, (f + chunk - 1 < c[m]) ? f + chunk - 1 : c[m]) }
    else if (c[m] >= alone) emit(m, m, 1, c[m])
    else { if (np > 0 && np + c[m] > pack) flush(); add(m) }
  }
  flush()
}
AWK
}

# Every wanted test is in exactly one shard or solo file.
check_partition() {
  local placed="$W/placed" dup
  { cut -f3 "$W/plan"; cat "$W/solo"; } | sort > "$placed"
  dup=$(uniq -d "$placed" | wc -l | tr -d ' ')
  cmp -s <(sort -u "$W/wanted") <(sort -u "$placed") && [ "$dup" = 0 ] ||
    die "partition is wrong ($(comm -23 <(sort -u "$W/wanted") <(sort -u "$placed") | wc -l | tr -d ' ') missing, $dup duplicated)"
}

write_shards() {
  local out=$1
  mkdir -p "$out"
  find "$out" -maxdepth 1 -type f \( -name 'shard-*.txt' -o -name 'solo-*.txt' \) -exec rm -f {} +
  awk -F'\t' -v out="$out" 'NR == 1 || $1 != last { if (f) close(f); f = sprintf("%s/shard-%02d-%s.txt", out, $1, $2); last = $1 } { print $3 > f }' "$W/plan"
  awk -v out="$out" '{ f = sprintf("%s/solo-%02d.txt", out, NR - 1); print > f; close(f) }' "$W/solo"
}

summary() {
  local listed wanted solo
  listed=$(wc -l < "$W/listed" | tr -d ' '); wanted=$(wc -l < "$W/wanted" | tr -d ' '); solo=$(wc -l < "$W/solo" | tr -d ' ')
  awk -F'\t' -v w="$wanted" -v k="$((listed - wanted))" -v so="$solo" '
    NR == 1 || $1 != last { if (NR > 1) n[++ns] = cur; lab[ns + 1] = $2; cur = 0; last = $1 } { cur++ }
    END { if (NR > 0) n[++ns] = cur; line = ""; for (i = 1; i <= ns; i++) line = line (i > 1 ? ", " : "") lab[i] "=" n[i]
      printf "coverage_serve_shards: %d tests (%d skipped) in %d shards + %d solo: %s\n", w, k, ns, so, line }' "$W/plan"
}

main() {
  [ $# -eq 3 ] || [ $# -eq 4 ] || { sed -n '11,17s/^# \{0,1\}//p' "$0" >&2; exit 1; }
  W=$(mktemp -d); trap 'rm -rf "${W:?}"' EXIT
  sed -n 's/: test$//p' "$1" > "$W/listed"
  entries "$2" > "$W/skips"
  minus "$W/skips" "$W/listed" > "$W/wanted"
  [ -s "$W/wanted" ] || die "the test list is empty; refusing to write zero shards"
  : > "$W/solo"; [ $# -eq 4 ] && entries "$4" > "$W/solo"
  local unknown; unknown=$(minus "$W/wanted" "$W/solo" | awk '{ printf "%s'\''%s'\''", (NR > 1 ? ", " : ""), $0 }')
  [ -z "$unknown" ] || die "solo entries that are not listed tests: [$unknown]"
  minus "$W/solo" "$W/wanted" | sort > "$W/rest"
  plan "$W/rest" > "$W/plan"
  check_partition
  write_shards "$3"
  summary
}

main "$@"
