#!/usr/bin/env bash
# unsloth_finetune_throughput.sh — the committed T2 measurement command (0.72, row R5).
#
# beat-unsloth-finetune-throughput-v1 1.1.0 names this script. It runs the two sides
# (apr, incumbent = Unsloth) interleaved on ONE GPU in one session, each run under the
# fleet GPU queue (gpu-q: the lock goes around the run, never around cargo), then hands
# every receipt that exists to unsloth_ft_verdict.py, which alone decides.
#
#   unsloth_finetune_throughput.sh --model Qwen3.5-4B --gpu 0 --out evidence/beat-unsloth-ft/<version>/
#       [--runs 3] [--dry-run]
#       [--planted-half-targets | --planted-no-fla | --planted-incumbent-adamw8bit]
#
# Order is ABBA (run k odd: apr then incumbent; k even: incumbent then apr), so neither
# side always runs on the warmer GPU.
#
# Each side command is called as
#   <cmd> --side apr|incumbent --model M --gpu N --run K --receipt PATH [planted flag]
# and must write PATH (one JSON receipt, fields per unsloth_ft_verdict.py). A run that
# exits non-zero or writes no receipt is logged and skipped; the verdict then has fewer
# runs and says NOT_MEASURED. A failed run is never retried into a pass.
#
# Planted falsifiers go to the side they plant on:
#   --planted-half-targets           apr side        expect exit 1 SAME-WORK FAIL
#   --planted-no-fla                 incumbent side  expect exit 2 INCUMBENT_SLOW_PATH
#   --planted-incumbent-adamw8bit    incumbent side  expect exit 1 SAME-WORK FAIL
#
# Environment (each has a default; tests override them):
#   APR_FT_SIDE_CMD        apr side runner          (default scripts/bench/unsloth_ft_apr_side.sh)
#   INCUMBENT_FT_SIDE_CMD  incumbent side runner    (default scripts/bench/unsloth_ft_incumbent_side.sh)
#   GPUQ                   GPU queue wrapper        (default gpu-q; never empty: no unlocked GPU run)
#   GPUQ_PRIO              gpu-q priority 0-9       (default 5)
#   RUN_TIMEOUT            seconds per run          (default 3600)
#   APR_TRAIN_ACTIVE_MARKER if this file exists the train is active and nothing runs
#                          (default /tmp/apr-train-active [A]; a fleet-wide marker is a ticket after 0.70.1)
#   VERDICT                verdict script           (default scripts/bench/unsloth_ft_verdict.py)
#
# Exit: 0 PASS, 1 FAIL, 2 NOT_MEASURED (the verdict's codes), 3 refused (train active,
# a side runner missing, or no GPU queue), 64 usage. --dry-run writes plan.txt, prints
# the commands, runs nothing and exits 0.
# No -e: a failed run must be logged and skipped, not end the session; every status
# that matters is read explicitly (rc, PIPESTATUS).
set -uo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
model=""
gpu=""
out=""
runs=3
dry=0
planted=""

usage() { echo "usage: $0 --model M --gpu N --out DIR [--runs K] [--dry-run] [--planted-...]" >&2; exit 64; }

while [ $# -gt 0 ]; do
  case "$1" in
    --model) [ $# -ge 2 ] || usage; model=$2; shift 2 ;;
    --gpu) [ $# -ge 2 ] || usage; gpu=$2; shift 2 ;;
    --out) [ $# -ge 2 ] || usage; out=$2; shift 2 ;;
    --runs) [ $# -ge 2 ] || usage; runs=$2; shift 2 ;;
    --dry-run) dry=1; shift ;;
    --planted-half-targets|--planted-no-fla|--planted-incumbent-adamw8bit)
      if [ -n "$planted" ]; then echo "$0: one planted falsifier per session, got $planted and $1" >&2; exit 64; fi
      planted=$1; shift ;;
    *) echo "$0: unknown argument $1" >&2; usage ;;
  esac
done
[ -n "$model" ] && [ -n "$gpu" ] && [ -n "$out" ] || usage
case "$runs" in ''|*[!0-9]*|0) echo "$0: --runs must be a positive integer" >&2; exit 64 ;; esac
case "$gpu" in ''|*[!0-9]*) echo "$0: --gpu must be a device index" >&2; exit 64 ;; esac

apr_cmd=${APR_FT_SIDE_CMD:-$root/scripts/bench/unsloth_ft_apr_side.sh}
inc_cmd=${INCUMBENT_FT_SIDE_CMD:-$root/scripts/bench/unsloth_ft_incumbent_side.sh}
gpuq=${GPUQ-gpu-q}
prio=${GPUQ_PRIO:-5}
run_timeout=${RUN_TIMEOUT:-3600}
train_active=${APR_TRAIN_ACTIVE_MARKER:-/tmp/apr-train-active}
verdict=${VERDICT:-$root/scripts/bench/unsloth_ft_verdict.py}

# The planted flag a side receives, or nothing.
planted_for() {
  case "$1:$planted" in
    apr:--planted-half-targets) echo "$planted" ;;
    incumbent:--planted-no-fla|incumbent:--planted-incumbent-adamw8bit) echo "$planted" ;;
  esac
}

# The plan: one line per run, "<k> <side>", ABBA.
plan() {
  k=1
  while [ "$k" -le "$runs" ]; do
    if [ $((k % 2)) -eq 1 ]; then printf '%s apr\n%s incumbent\n' "$k" "$k"
    else printf '%s incumbent\n%s apr\n' "$k" "$k"; fi
    k=$((k + 1))
  done
}

mkdir -p "$out/receipts" "$out/logs" || { echo "$0: cannot create $out" >&2; exit 64; }
plan > "$out/plan.txt"
printf 'model=%s gpu=%s runs=%s planted=%s gpuq=%s prio=%s\n' \
  "$model" "$gpu" "$runs" "${planted:-none}" "$gpuq" "$prio" >> "$out/plan.txt"

if [ "$dry" -eq 1 ]; then
  if [ -e "$train_active" ]; then echo "dry-run: train-active ($train_active) is set; a real run would refuse (exit 3)"; fi
  while read -r k side; do
    if [ "$side" = apr ]; then cmd=$apr_cmd; else cmd=$inc_cmd; fi
    echo "dry-run: $gpuq --prio $prio -- timeout $run_timeout $cmd --side $side --model $model --gpu $gpu --run $k --receipt $out/receipts/$side-$k.json $(planted_for "$side")"
  done < <(plan)
  exit 0
fi

if [ -e "$train_active" ]; then
  echo "$0: REFUSED: train-active ($train_active) is set; no GPU run while the train is active" >&2
  exit 3
fi
[ -n "$gpuq" ] || { echo "$0: REFUSED: GPUQ is empty; no unlocked GPU run" >&2; exit 3; }
for c in "$apr_cmd" "$inc_cmd"; do
  [ -x "$c" ] || { echo "$0: REFUSED: side runner $c is missing or not executable" >&2; exit 3; }
done

while read -r k side; do
  if [ "$side" = apr ]; then cmd=$apr_cmd; else cmd=$inc_cmd; fi
  receipt=$out/receipts/$side-$k.json
  log=$out/logs/$side-$k.log
  rm -f "${receipt:?}"
  flag=$(planted_for "$side")
  "$gpuq" --prio "$prio" -- timeout "$run_timeout" "$cmd" --side "$side" --model "$model" \
    --gpu "$gpu" --run "$k" --receipt "$receipt" ${flag:+"$flag"} > "$log" 2>&1 < /dev/null
  rc=$?
  if [ "$rc" -ne 0 ] || [ ! -s "$receipt" ]; then
    echo "run $k $side: rc=$rc, skipped (see $log)" >&2
    rm -f "${receipt:?}"
  fi
done < <(plan)

apr_r=()
inc_r=()
for f in "$out"/receipts/apr-*.json; do [ -e "$f" ] && apr_r+=("$f"); done
for f in "$out"/receipts/incumbent-*.json; do [ -e "$f" ] && inc_r+=("$f"); done

python3 "$verdict" --apr "${apr_r[@]}" --incumbent "${inc_r[@]}" | tee "$out/verdict.json"
exit "${PIPESTATUS[0]}"
