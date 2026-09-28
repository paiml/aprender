#!/usr/bin/env bash
# KTEST-05 / L5: nightly compute-sanitizer receipt for apr's CUDA kernels (memcheck, racecheck,
# initcheck, synccheck) on a small model with batched prefill forced on and 2 decode tokens.
#
# An unfiltered racecheck OOMed at 57.9G on GB10 (#4590), so racecheck runs with a kernel-name filter
# and the caller should wrap this in `systemd-run --user -p MemoryMax=…`. The filter is applied to
# racecheck ONLY: initcheck does not track writes made by kernels the filter excludes, so a filtered
# initcheck reports every read of a GEMV output as "uninitialized" (measured 2026-09-28: a producer/
# consumer fixture gave 0 errors unfiltered, 32 with --kernel-name regex=rope (fixtures/initcheck_filter_artifact.cu); on apr it produced
# 50 false reads at rope_neox_indirect's LDG [x] on the 4090).
# A tool whose SUMMARY line is missing is RED (fail closed): a crashed or killed sanitizer is not 0.
#
# Usage: cuda_sanitizer_receipt.sh <apr> <model.gguf> <outdir> [tool ...]
#        cuda_sanitizer_receipt.sh --self-test
# Env:   KTEST_SAN_FILTER (default below), KTEST_SAN_FILTER_TOOLS (default racecheck), KTEST_SAN_TIMEOUT (s, default 1500), COMPUTE_SANITIZER
# Out:   <outdir>/receipt.tsv  tool  count  kind  rc  verdict   + <outdir>/receipt.json
# Exit:  0 all tools clean, 1 any finding or missing summary, 2 setup error.
set -uo pipefail
FILTER=${KTEST_SAN_FILTER:-regex=(scatter|rope|attn|attention|prefill|batch|rmsnorm|kv)}

# parse_summary <tool> <log> -> "count kind" (kind = errors|hazards); returns 1 when no summary line
parse_summary() {
  local tool=$1 log=$2 line
  if [ "$tool" = racecheck ]; then
    line=$(grep -E 'RACECHECK SUMMARY: [0-9]+ hazards? displayed' "$log" | tail -n 1)
    [ -n "$line" ] || return 1
    # hazards displayed (N errors, M warnings): errors are the gate, warnings are counted too
    echo "$line" | sed -E 's/.*SUMMARY: ([0-9]+) hazards? displayed \(([0-9]+) errors?, ([0-9]+) warnings?\).*/\1 hazards/'
  else
    line=$(grep -E 'ERROR SUMMARY: [0-9]+ errors?' "$log" | tail -n 1)
    [ -n "$line" ] || return 1
    echo "$line" | sed -E 's/.*ERROR SUMMARY: ([0-9]+) errors?.*/\1 errors/'
  fi
}

self_test() {
  local fails=0 tmp t
  tmp=$(mktemp -d) || return 2
  while IFS='|' read -r name tool text want; do
    printf '%b\n' "$text" > "$tmp/log"
    got=$(parse_summary "$tool" "$tmp/log" || echo MISSING)
    if [ "$got" = "$want" ]; then t=ok; else t=FAIL; fails=$((fails+1)); fi
    printf '%-4s %-30s want=%-12s got=%s\n' "$t" "$name" "'$want'" "'$got'"
  done <<'CASES'
memcheck-clean|memcheck|========= ERROR SUMMARY: 0 errors|0 errors
memcheck-one|memcheck|========= ERROR SUMMARY: 1 error|1 errors
memcheck-many|memcheck|========= ERROR SUMMARY: 1234 errors|1234 errors
memcheck-killed-no-summary|memcheck|========= Invalid __global__ read of size 4\n=========     at kern+0x40|MISSING
memcheck-last-line-wins|memcheck|========= ERROR SUMMARY: 0 errors\n========= ERROR SUMMARY: 3 errors|3 errors
racecheck-clean|racecheck|========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)|0 hazards
racecheck-hazard|racecheck|========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)|1 hazards
racecheck-many|racecheck|========= RACECHECK SUMMARY: 12 hazards displayed (8 errors, 4 warnings)|12 hazards
racecheck-ignores-error-summary|racecheck|========= ERROR SUMMARY: 0 errors|MISSING
synccheck-clean|synccheck|========= ERROR SUMMARY: 0 errors|0 errors
empty-log|initcheck||MISSING
CASES
  rm -rf "${tmp:?}"
  echo "self-test: $fails failing case(s)"; [ "$fails" -eq 0 ]
}

[ "${1:-}" = --self-test ] && { self_test; exit $?; }
[ $# -ge 3 ] || { echo "usage: $0 <apr> <model> <outdir> [tool ...] | --self-test" >&2; exit 2; }
APR=$1; M=$2; O=$3; shift 3
TOOLS=("$@"); [ ${#TOOLS[@]} -gt 0 ] || TOOLS=(memcheck racecheck initcheck synccheck)
CS=${COMPUTE_SANITIZER:-$(command -v compute-sanitizer || echo /usr/local/cuda/bin/compute-sanitizer)}
[ -x "$CS" ] || { echo "SETUP: no compute-sanitizer" >&2; exit 2; }
mkdir -p "$O" || exit 2
{ "$APR" --version; sha256sum "$APR" | cut -c1-16; "$CS" --version | tail -n 1
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader; } > "$O/version.txt" 2>&1
printf 'tool\tcount\tkind\trc\tverdict\tfilter\n' > "$O/receipt.tsv"
worst=0
for tool in "${TOOLS[@]}"; do
  filt=(); case " ${KTEST_SAN_FILTER_TOOLS:-racecheck} " in *" $tool "*) filt=(--kernel-name "$FILTER");; esac
  BATCHED_PREFILL=1 timeout "${KTEST_SAN_TIMEOUT:-1500}" "$CS" --tool "$tool" --print-limit 50 \
    "${filt[@]}" --log-file "$O/$tool.log" \
    "$APR" run "$M" --prompt "What is 2+2? Answer with one number." --max-tokens 2 --temperature 0 \
    > "$O/$tool.out" 2> "$O/$tool.err"; rc=$?
  if s=$(parse_summary "$tool" "$O/$tool.log"); then
    n=${s% *}; kind=${s#* }
    if [ "$n" -eq 0 ]; then v=CLEAN; else v=RED; worst=1; fi
  else n=NA; kind=NA; v=RED_NO_SUMMARY; worst=1; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tool" "$n" "$kind" "$rc" "$v" "${filt[1]:-none}" >> "$O/receipt.tsv"
done
python3 - "$O" <<'PY'
import csv, json, sys, datetime, pathlib
o = pathlib.Path(sys.argv[1]); v = (o / "version.txt").read_text().splitlines()
rows = list(csv.DictReader(open(o / "receipt.tsv"), delimiter="\t"))
json.dump({"schema": "ktest-05-sanitizer-receipt-v1",
           "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "apr_version": v[0] if v else None, "apr_sha256_16": v[1] if len(v) > 1 else None,
           "sanitizer": v[2] if len(v) > 2 else None, "device": v[3] if len(v) > 3 else None,
           "tools": rows, "clean": all(r["verdict"] == "CLEAN" for r in rows)},
          open(o / "receipt.json", "w"), indent=1)
PY
column -t "$O/receipt.tsv"
exit "$worst"
