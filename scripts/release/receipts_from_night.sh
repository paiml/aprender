#!/usr/bin/env bash
# receipts_from_night.sh -- build the promote_nightly.sh receipts for ONE sha from evidence that already exists (N26, PMAT-5306)
#
#   bash scripts/release/receipts_from_night.sh --sha <40-hex> --out DIR --models-dir D --cleanroom-json F \
#        --dryrun-rc N --dryrun-commit <sha>
#
#   --models-dir D       a models-crux/<sha>/ directory of the nightly-evidence branch: needs models-t1.log and verdict
#   --cleanroom-json F   ONE document {"run_id","job","conclusion","tested_shas"} read from the paiml/infra clean-room run
#   --dryrun-rc N        exit status of `rc_publish_gate.sh --verify <worktree of sha>`; --dryrun-commit the sha it ran on
#
# Pure over files: it fetches and runs nothing. A lane is written ONLY when its evidence is green for exactly this sha.
# A red, missing, unreadable or other-sha lane writes NO file, so promote_nightly.sh refuses it as "missing": absent is
# never green, and this script has no way to write a red receipt or a waiver.
# exit: 0 all three receipts written - 1 at least one lane not written (each reason printed) - 2 usage
set -uo pipefail
PROG=receipts_from_night
sha="" out="" mdir="" crj="" drc="" dcm=""
usage() { printf '%s: %s\n' "$PROG" "$*" >&2; exit 2; }
while [ $# -gt 0 ]; do
    [ $# -ge 2 ] || usage "$1 needs a value"
    case "$1" in
        --sha) sha=$2 ;; --out) out=$2 ;; --models-dir) mdir=$2 ;; --cleanroom-json) crj=$2 ;;
        --dryrun-rc) drc=$2 ;; --dryrun-commit) dcm=$2 ;; *) usage "unknown argument: $1" ;;
    esac
    shift 2
done
[[ $sha =~ ^[0-9a-f]{40}$ ]] || usage "--sha must be a full 40-hex sha"
[ -n "$out" ] || usage "--out is required"
command -v jq > /dev/null 2>&1 || usage "jq is not installed"
mkdir -p -- "$out" || usage "cannot create $out"
rm -f -- "${out:?}/clean-room.json" "${out:?}/models.json" "${out:?}/publish-dry-run.json"
bad=0
vget() { awk -v k="$1" 'index($0, k "=") == 1 { print substr($0, length(k) + 2); exit }'; }
no() { printf '%s: NOT WRITTEN %s -- %s\n' "$PROG" "$1" "$2" >&2; bad=1; }

# models: the exact go-line autopilot cut_tag greps, at this sha9, AND the judge's own verdict green with rc 0
lane_models() {
    local log="$mdir/models-t1.log" ver="$mdir/verdict" go
    [ -s "$log" ] && [ -s "$ver" ] || { no models "no models-t1.log / verdict in '$mdir'"; return; }
    [ "$(vget state < "$ver")" = green ] && [ "$(vget t1_rc < "$ver")" = 0 ] && [ "$(vget commit < "$ver")" = "$sha" ] || { no models "verdict (first-line keys) is not state=green, t1_rc=0, commit=sha"; return; }
    go="MODELS GO (CRUX smoke) on lambda and gx10 at ${sha:0:9}:"
    grep -qF "$go" "$log" || { no models "models-t1.log has no '$go'"; return; }
    if grep -q '^MODELS .*FAIL' "$log"; then no models "models-t1.log carries a FAIL line"; return; fi
    jq -n --arg s "$sha" --arg g "$go" '{lane:"models",sha:$s,status:"green",go:$g,hosts:["gx10","lambda"]}' > "$out/models.json"
}
# clean-room: run on exactly this sha (tested_shas == [sha]), job 'clean-room (aprender)', conclusion success, numeric run id
lane_cleanroom() {
    [ -s "${crj:-/nonexistent}" ] || { no clean-room "no clean-room run summary"; return; }
    jq -s -e --arg s "$sha" 'length==1 and (.[0] | (.run_id|tostring|test("\\A[1-9][0-9]*\\z")) and .job=="clean-room (aprender)" and .conclusion=="success" and .tested_shas==[$s])' "$crj" > /dev/null 2>&1 \
        || { no clean-room "run is not job 'clean-room (aprender)' success with tested_shas exactly [$sha]"; return; }
    jq --arg s "$sha" '{lane:"clean-room",sha:$s,status:"green",run_id:(.run_id|tostring),job,conclusion,tested_shas}' "$crj" > "$out/clean-room.json"
}
# publish dry-run: rc_publish_gate.sh --verify exited 0 on this very sha
lane_dryrun() {
    if [ "$drc" = 0 ] && [ "$dcm" = "$sha" ]; then
        jq -n --arg s "$sha" '{lane:"publish-dry-run",sha:$s,status:"green"}' > "$out/publish-dry-run.json"
    else no publish-dry-run "rc='${drc:-unset}' commit='${dcm:-unset}' (need rc 0 on $sha)"; fi
}
lane_models; lane_cleanroom; lane_dryrun
[ "$bad" = 0 ] && printf '%s: wrote 3 receipts for %s in %s\n' "$PROG" "${sha:0:9}" "$out"
exit "$bad"
