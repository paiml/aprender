#!/usr/bin/env bash
# ledger.sh AP MC TAG VERSION STATUS -> $AP/<sha9>-lambda-vector-train.json, the release train's ledger
# record (APR-RELEASE-001 section 4), and one line "ledger record <path>" on stdout.
#
# The bash+jq twin of the ledger.py it replaces (#4352: python3 is out of the release path). Same keys
# in the same order, same values: every STATUS read is a jq (oniguruma) regex over the whole file, matched line
# by line so `^` is a line anchor exactly as python's re.M made it, and a run id is the LAST match, not the first
# (STATUS is append-only and a step may be re-run: measured 2026-09-20 on v0.68.2, the first-match
# version recorded the FAILED clean-room run 35471384263 and installer rc=1 from the attempt that died
# on a missing receipts dir, while the green run 35495206021 and rc=0 sat later in the same file. A
# receipt that reports the failed attempt is worse than no receipt).
#
# Fails (rc 1, no record) where ledger.py crashed: an unreadable STATUS, and a perf041 marker that
# exists but is not one JSON object in UTF-8.
set -uo pipefail
[ $# -eq 5 ] || { echo "usage: ledger.sh AP MC TAG VERSION STATUS" >&2; exit 2; }
ap=$1 mc=$2 t=$3 v=$4 status=$5
command -v jq > /dev/null 2>&1 || { echo "ledger.sh: jq not found" >&2; exit 1; }
[ -f "$status" ] && [ -r "$status" ] || { echo "ledger.sh: cannot read $status" >&2; exit 1; }

# the perf041 marker: none, or one JSON object; anything else fails
marker="$ap/wt/evidence/perf041/lambda/marker.json"
m='{}'
if [ -e "$marker" ]; then
    [ -f "$marker" ] && iconv -f UTF-8 -t UTF-8 < "$marker" > /dev/null 2>&1 \
        && [ "$(head -c 3 "$marker" | od -An -tx1 | tr -d ' \n')" != efbbbf ] \
        && m=$(jq -c -n 'input as $d | if ([inputs] | length) > 0 then error("extra data")
                         elif ($d | type) != "object" then error("not an object") else $d end' < "$marker") \
        || { echo "ledger.sh: the perf041 marker $marker is not one JSON object" >&2; exit 1; }
fi
# the witness is built over {} too: no marker gives every field null, as ledger.py's lambda did
m=$(printf '%s' "$m" | jq -c '{host, cc, commit, started_utc, status, sha256,
    nightly_producer: "cuda-nightly.yml gx10 NIGHTLY-RED since 2026-09-12 (#3096, 0.70); release witness taken on lambda sm_89"}') \
    || exit 1

# python's datetime.now(timezone.utc).isoformat(): microseconds and +00:00
written=$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00 2> /dev/null)
case $written in
    *T*.[0-9][0-9][0-9][0-9][0-9][0-9]+00:00) ;;
    *) written=$(perl -MPOSIX=strftime -MTime::HiRes=gettimeofday \
        -e '($s,$u)=gettimeofday; printf "%s.%06d+00:00", strftime("%Y-%m-%dT%H:%M:%S", gmtime($s)), $u') \
        || { echo "ledger.sh: no clock" >&2; exit 1; } ;;
esac

out="$ap/${mc:0:9}-lambda-vector-train.json"
jq -an --rawfile s "$status" --arg ap "$ap" --arg mc "$mc" --arg t "$t" --arg v "$v" \
    --arg written "$written" --argjson m "$m" '
    # re.M: `^` starts every line. oniguruma anchors `^` at the string start only, so match per line
    # (split on \n, as re.M does; no pattern here can span a newline, so per-line matching is equal)
    ($s | split("\n")) as $lines |
    def grab($rx): [($lines | .[]) | match($rx) | .captures[0].string] | last;
    def seen($rx): any(($lines | .[]); test($rx));
    {spec: "APR-RELEASE-001", section: "4", job: "release-train", host: "lambda-vector", host_class: "lambda",
     source: (($ap | sub("/+$"; "") | split("/") | .[-1] // "") + "/STATUS"), sha: $mc, tag: $t, version: $v,
     published: seen("^\\S+ PUBLISHED "),
     asset_run_id: grab("^\\S+ ASSET RUN (\\d+)"), cleanroom_run_id: grab("^\\S+ CLEANROOM RUN (\\d+)"),
     assets_on_release: seen("^\\S+ ASSETS all present"),
     preflight_pass: seen("^\\S+ PREFLIGHT PASS"),
     dogfood_attempts: ([($lines | .[]) | select(test("^\\S+ START autopilot"))] | length), attended_minutes: 0,
     steps: (reduce ("DEEP GO", "DOGFOOD GO", "TAGGED", "CLEANROOM GREEN", "ASSETS all present", "PREFLIGHT PASS",
                     "PUBLISHED", "INSTALL cargo", "HOST gx10", "HOST yoga", "INSTALLER intel", "INSTALLER gx10") as $k
             ({}; .[$k] = seen("^\\S+ " + $k))),
     installer_receipts: (reduce ("intel", "gx10") as $h ({}; .[$h] = grab("^\\S+ INSTALLER " + $h + " rc=(\\d+)"))),
     pp26_witness: $m,
     unmeasured: ["t4_wall_minutes: derive from STATUS timestamps CASCADE..PUBLISHED"],
     written: $written}' > "$out.tmp" && mv -- "$out.tmp" "$out" || { echo "ledger.sh: no record written" >&2; exit 1; }
echo "ledger record $out"
