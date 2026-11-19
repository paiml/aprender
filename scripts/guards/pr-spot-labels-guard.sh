#!/bin/sh
# pr-spot-labels-guard.sh -- the pool-routing opt-in may move ONLY the three aggregator jobs.
#
# ci.yml's `workspace-test`, `ci-gate` and `gate` jobs may route a pull_request run to the shared
# spot pool when the repo variable PR_SPOT_OPTIN is "1"; unset, they keep the clean-room labels.
# They only read `needs.*` results on a pull_request (no checkout, no secret, no cache, no host
# mount). Nothing else may name a spot (burst) label, and nothing that signs, publishes or
# releases may ever resolve to one.
#
# Judged on the PARSED documents (yq explode: anchors, merge keys, escapes, flow style, quoted
# keys all resolved), never on raw lines. Rules, per file under the workflow directory:
#   1. a job whose runs-on is an allowlisted (ci.yml, job) must equal that job's ONE canonical
#      expression, have permissions exactly {contents: read}, and name no secrets or id-token;
#   2. no other job, and no workflow-level key, may mention burst or PR_SPOT_OPTIN anywhere;
#   3. any other runs-on / reusable `with.runs_on` holding `${{` must be one of the known expressions
#      (pr-light, ci-unwedge, the macOS probe list), or `fromJSON(matrix.labels)` over a matrix with
#      no `${{` of its own;
#   4. ci.yml has no workflow-level secrets and no permissions wider than contents: read.
# A file that does not parse, an empty directory, or a shape not modelled is RED (cannot judge).
#
# Usage: pr-spot-labels-guard.sh [WORKFLOW_DIR]
# Exit: 0 clean, 1 violation, 2 usage / no yq / nothing scanned.
set -eu

dir="${1:-.github/workflows}"
[ -d "$dir" ] || { echo "guard: no such directory: $dir" >&2; exit 2; }
command -v yq >/dev/null 2>&1 || { echo "guard: yq not found; cannot parse the workflows" >&2; exit 2; }

q="'"
spot='["self-hosted","burst-x64","burst-public"]'
canon() { printf '%s' "\${{ (github.event_name == ${q}pull_request${q} && vars.PR_SPOT_OPTIN == ${q}1${q}) && fromJSON(${q}${spot}${q}) || fromJSON(${q}$1${q}) }}"; }
want_for() {
  case "$1" in
    workspace-test) canon '["self-hosted","Linux","clean-room"]' ;;
    ci-gate | gate) canon '["self-hosted","Linux","X64","clean-room"]' ;;
    *) return 1 ;;
  esac
}
known_a="\${{ vars.PR_LIGHT_POOL == ${q}pr-x64${q} && fromJSON(${q}[\"self-hosted\",\"Linux\",\"X64\",\"pr-x64\"]${q}) || fromJSON(${q}[\"self-hosted\",\"Linux\",\"X64\",\"clean-room\"]${q}) }}"
known_b="\${{ github.event.schedule == ${q}*/10 0-5,20-23 * * *${q} && fromJSON(${q}[\"self-hosted\", \"X64\", \"Linux\", \"clean-room\"]${q}) || fromJSON(${q}[\"self-hosted\", \"Linux\", \"clean-room\", \"ARM64\"]${q}) }}"
known_c='["self-hosted","macOS","ARM64","apple-silicon","${{ inputs.runner_label }}"]'

EXPR='explode(.) | select(kind == "map") | ((.jobs // {} | to_entries | .[] | ["J", .key, "=" + ((.value."runs-on" // "") | (select(kind == "scalar") // to_json(0))), "=" + ((.value.with.runs_on // "") | (select(kind == "scalar") // to_json(0))), "=" + ((.value.permissions // "") | (select(kind == "scalar") // to_json(0))), "=" + ((.value.strategy // "") | to_json(0)), ("=" + (.value | del(."runs-on") | to_json(0)))] | join("\t")), (["T", "=" + (del(.jobs) | to_json(0))] | join("\t")))'

tab=$(printf '\t')
list=$(mktemp)
out=$(mktemp)
[ -n "$list" ] && [ -f "$list" ] && [ -n "$out" ] && [ -f "$out" ] || exit 2
trap 'rm -f "${list:?}" "${out:?}"' EXIT
find "$dir" \( -type f -o -type l \) \( -name '*.yml' -o -name '*.yaml' \) -print | sort > "$list"

hits() { printf '%s' "$1" | grep -qiE 'burst|PR_SPOT_OPTIN'; }

scanned=0
bad=0
fail() { echo "FAIL $1" >&2; bad=$((bad + 1)); }

while IFS= read -r f; do
  scanned=$((scanned + 1))
  base=${f#"$dir"/}
  if [ ! -f "$f" ] || ! yq e "$EXPR" "$f" > "$out" 2> /dev/null; then
    fail "$f: does not parse as a YAML map document, so its runners cannot be judged"
    continue
  fi
  while IFS="$tab" read -r kind job ro wr perm strat rest; do
    case "$kind" in
      T)
        top="${job#=}"
        if hits "$top"; then fail "$f: a workflow-level key names a spot label or the opt-in variable"; fi
        if [ "$base" = ci.yml ]; then
          if printf '%s' "$top" | grep -qiE 'secrets|id-token|write'; then fail "$f: workflow-level key names secrets, id-token or write"; fi
        fi
        continue
        ;;
      J) ;;
      *) fail "$f: unmodelled record"; continue ;;
    esac
    ro="${ro#=}" wr="${wr#=}" perm="${perm#=}" strat="${strat#=}" rest="${rest#=}"
    allowed=""
    if [ "$base" = ci.yml ] && w=$(want_for "$job"); then allowed=$w; fi
    if [ -n "$allowed" ]; then
      [ "$ro" = "$allowed" ] || fail "$f job $job: runs-on is not its one canonical value: $ro"
      [ "$perm" = '{"contents":"read"}' ] || fail "$f job $job: permissions are not exactly contents: read: $perm"
      if printf '%s' "$rest$strat" | grep -qiE 'secrets|id-token|burst|PR_SPOT_OPTIN'; then fail "$f job $job: names secrets, id-token or a spot label outside runs-on"; fi
      [ -z "$wr" ] || fail "$f job $job: a reusable-workflow call cannot be moved"
      continue
    fi
    if hits "$ro$wr$perm$strat$rest"; then fail "$f job $job: names a spot label or the opt-in variable and is not allowlisted"; fi
    for v in "$ro" "$wr"; do
      case "$v" in
        *'${{'*)
          [ "$v" = "$known_a" ] && continue
          [ "$v" = "$known_b" ] && continue
          [ "$v" = "$known_c" ] && continue
          if [ "$v" = '${{ fromJSON(matrix.labels) }}' ] && ! printf '%s' "$strat" | grep -q '\${{'; then continue; fi
          fail "$f job $job: runs-on expression is not a known one, cannot judge: $v"
          ;;
      esac
    done
  done < "$out"
done < "$list"

echo "pr-spot-labels-guard: scanned $scanned workflow file(s), $bad violation(s)"
[ "$scanned" -gt 0 ] || exit 2
[ "$bad" -eq 0 ]
