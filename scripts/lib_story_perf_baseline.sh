# lib_story_perf_baseline.sh - the qwen story's PINNED throughput baseline.
#
# Sourced by scripts/qwen-story.sh and driven directly by
# scripts/check_story_perf_baseline.sh. Run as a script it pins a baseline:
#
#     bash scripts/lib_story_perf_baseline.sh pin <model.apr>
#
# Why (#4715): `apr qa` compares throughput against ~/.cache/apr/qa-reports/<model>.json
# and then OVERWRITES that file with the run it just made. The file records no commit,
# no binary and no load. One night on a busy host (3.6 tok/s at load 11.6; the same
# binary reads 7.59 idle) became the reference for the next night, and the night after
# that compared against the contaminated number. A baseline that every run rewrites is
# not a baseline.
#
# So the baseline here is PINNED:
#   - `pin` writes it once, on an idle host, recording the commit sha, the binary's
#     sha256, the host and the load. It refuses to overwrite: re-pinning is a person
#     removing the file by hand, never a run.
#   - `perf_baseline_judge` compares a run against it and never writes it. A run that
#     cannot be compared - no baseline, another host, a busy host, no executed
#     throughput gate - is not_measured, never pass (L25).
#
# Every refusal is ONE line ending in a `# R-<NAME>` marker, so the case table can
# delete it and prove the table turns red without it.
#
# A SOURCED library must be option-neutral: no `set` here. Failures are return codes.

PERF_BASELINE_DIR="${APR_PERF_BASELINE_DIR:-$HOME/.local/share/apr/perf-baseline}"
# Above this 1-minute load the host is busy and a tok/s reading is not a measurement.
PERF_MAX_LOAD="${APR_PERF_MAX_LOAD:-4.0}"
# A drop larger than this fraction of the pinned tok/s fails. Idle run-to-run spread on
# the same binary was measured at ~2% (7.59 vs 7.73), so 10% is noise-safe and a real
# 20% loss cannot hide under it.
PERF_MAX_DROP="${APR_PERF_MAX_DROP:-0.10}"
PERF_LOADAVG="${APR_PERF_LOADAVG:-/proc/loadavg}"

perf_load1() { awk '{print $1; exit}' "$PERF_LOADAVG" 2>/dev/null; }
perf_host() { printf '%s\n' "${APR_PERF_HOST:-$(hostname)}"; }
perf_gt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 > b + 0) }'; }
perf_baseline_file() { printf '%s/%s.json\n' "$PERF_BASELINE_DIR" "$(basename "${1%.*}")"; }

# The executed throughput gate's tok/s, or nothing.
perf_tps() {
  jq -r '[.gates[]? | select(.name == "throughput" and .skipped == false and .value != null) | .value][0] // empty' "$1" 2>/dev/null
}

# 0 when an `apr qa --json` report passes by apr qa's OWN rule (`gates_pass` in
# crates/apr-cli/src/commands/qa.rs: every gate `passed || skipped`) with ONE gate left
# out: performance_regression, which compares against the self-overwriting cache and is
# replaced by the judge below. A skip is `passed: false, skipped: true` (#3965), so reading
# `.passed` alone would fail B2 on every legitimately skipped gate. At least one gate
# must have EXECUTED: an all-skipped report is nothing measured.
qa_gates_pass_except_regression() {
  jq -e '[.gates[]? | select(.name != "performance_regression")] as $g
    | ($g | map(select(.skipped == false)) | length > 0) and ($g | map(.passed or .skipped) | all)' "$1" >/dev/null 2>&1
}

# perf_baseline_judge <model> <qa-report.json> <load1 sampled before the run>
# Sets PERF_VERDICT (pass|fail|not_measured) and PERF_REASON.
# Returns 0 pass, 1 fail, 3 not_measured. Never writes the baseline.
perf_baseline_judge() {
  local model="$1" report="$2" load="$3" base host cur bsha btps bhost
  base="$(perf_baseline_file "$model")"
  host="$(perf_host)"
  PERF_VERDICT=not_measured
  [ -f "$base" ] || { PERF_REASON="no pinned baseline at $base (pin one on an idle host: bash scripts/lib_story_perf_baseline.sh pin $model)"; return 3; } # R-MISSING
  btps="$(jq -r '.throughput_tps // empty' "$base" 2>/dev/null)"
  bhost="$(jq -r '.host // empty' "$base" 2>/dev/null)"
  bsha="$(jq -r '.sha // empty' "$base" 2>/dev/null)"
  [ -n "$btps" ] && perf_gt "$btps" 0 || { PERF_REASON="baseline $base carries no throughput_tps"; return 3; } # R-BADBASE
  [ "$bhost" = "$host" ] || { PERF_REASON="baseline was pinned on host '$bhost', this run is on '$host': tok/s across hosts is not comparable"; return 3; } # R-HOST
  [ -n "$load" ] || { PERF_REASON="the load before the run was not read ($PERF_LOADAVG)"; return 3; } # R-NOLOAD
  perf_gt "$load" "$PERF_MAX_LOAD" && { PERF_REASON="load1 $load > $PERF_MAX_LOAD before the run: a busy host's tok/s is not a measurement"; return 3; } # R-LOAD
  cur="$(perf_tps "$report")"
  [ -n "$cur" ] || { PERF_REASON="the report has no executed throughput gate"; return 3; } # R-NOTPS
  local drop
  drop="$(awk -v b="$btps" -v c="$cur" 'BEGIN { printf "%.4f", (b - c) / b }')"
  perf_gt "$drop" "$PERF_MAX_DROP" && { PERF_VERDICT=fail; PERF_REASON="throughput $cur tok/s vs pinned $btps (sha $bsha): drop $drop > $PERF_MAX_DROP at load1 $load"; return 1; } # R-DROP
  PERF_VERDICT=pass
  PERF_REASON="throughput $cur tok/s vs pinned $btps (sha $bsha): drop $drop <= $PERF_MAX_DROP at load1 $load"
  return 0
}

# perf_baseline_pin <model>: run `apr qa --json` once on an idle host and pin it.
# Returns 0 pinned, 2 refused (already pinned), 3 refused (cannot measure).
perf_baseline_pin() {
  local model="$1" base load host report tps ver sha bin_sha
  base="$(perf_baseline_file "$model")"
  [ ! -e "$base" ] || { PERF_REASON="$base is pinned; a run never rewrites it. Remove it by hand to re-pin"; return 2; } # R-PINNED
  load="$(perf_load1)"
  [ -n "$load" ] || { PERF_REASON="cannot read the load ($PERF_LOADAVG)"; return 3; }
  perf_gt "$load" "$PERF_MAX_LOAD" && { PERF_REASON="load1 $load > $PERF_MAX_LOAD: pin on an idle host"; return 3; } # R-PINLOAD
  [ -n "${APR:-}" ] && [ -x "$APR" ] || { PERF_REASON="APR is not set to an executable apr (source scripts/apr_bin.sh)"; return 3; }
  host="$(perf_host)"
  report="$(mktemp)"
  "$APR" qa "$model" --json >"$report" 2>/dev/null
  tps="$(perf_tps "$report")"
  [ -n "$tps" ] || { rm -f "${report:?}"; PERF_REASON="apr qa produced no executed throughput gate"; return 3; }
  ver="$("$APR" --version 2>/dev/null | head -1)"
  sha="$(printf '%s\n' "$ver" | sed -n 's/.*(\([0-9a-f]\{7,\}\)).*/\1/p')"
  bin_sha="$(sha256sum "$APR" | awk '{print $1}')"
  mkdir -p "$PERF_BASELINE_DIR" || { rm -f "${report:?}"; PERF_REASON="cannot create $PERF_BASELINE_DIR"; return 3; }
  jq -n --arg model "$(basename "$model")" --argjson tps "$tps" --arg sha "$sha" --arg version "$ver" \
    --arg apr_sha256 "$bin_sha" --arg host "$host" --arg load1 "$load" \
    --arg pinned_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{model: $model, throughput_tps: $tps, sha: $sha, version: $version, apr_sha256: $apr_sha256,
      host: $host, load1: ($load1 | tonumber), pinned_at: $pinned_at}' >"$base.tmp" \
    && mv -n "$base.tmp" "$base"
  rm -f "${report:?}" "${base:?}.tmp"
  [ -f "$base" ] || { PERF_REASON="could not write $base"; return 3; }
  PERF_REASON="pinned $base: $tps tok/s, sha $sha, load1 $load"
  return 0
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-}" in
    pin)
      [ -n "${2:-}" ] || { echo "usage: $0 pin <model.apr>" >&2; exit 64; }
      if [ -z "${APR:-}" ]; then
        . "$(dirname "$0")/apr_bin.sh" || exit 1
      fi
      perf_baseline_pin "$2"; rc=$?
      printf '%s\n' "$PERF_REASON"
      exit "$rc"
      ;;
    *) echo "usage: $0 pin <model.apr>" >&2; exit 64 ;;
  esac
fi
