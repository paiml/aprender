#!/usr/bin/env bash
# tag_coverage_gate.sh — did coverage-nightly measure the release commit and hold COV_FLOOR? (#3690, #4734)
#
#   bash scripts/release/tag_coverage_gate.sh TAG SHA        # judge the nightly receipt for SHA
#   bash scripts/release/tag_coverage_gate.sh --resolve SHA  # the same judgement, before the tag
#   bash scripts/release/tag_coverage_gate.sh --self-test
#
# WHY. #3676 took coverage off the PR/queue/push path, and nothing on the release path read
# it, so a red coverage reached the crates.io cascade anyway (#3690). autopilot.sh runs this
# gate before preflight, and a missing, stale, unmeasured or below-floor coverage stops it.
#
# WHAT IS READ (#4734). Not the tag push's ci.yml section. On v0.70.1 its sov.coverage was
# `success` having built only the root facade, run 0 tests and printed no percentage, so a
# green verdict there is not a measurement. coverage-nightly measures the workspace and
# leaves coverage-receipt-<sha>.json (artifact `coverage-receipt`, scripts/coverage_receipt.sh)
# keyed by the commit it measured. This gate takes the NEWEST completed coverage-nightly run
# whose commit H is either SHA itself or a commit that SHA is a version-only bump of, and
# reads that run's receipt. A newer run on a qualifying commit always wins over an older one,
# so an older passing receipt cannot be picked over a newer failing one.
#
# VERSION-ONLY (#4735). H is an ancestor of SHA and every path changed in H..SHA is one of:
#   - a Cargo file bump-version.sh rewrites (SURFACE, measured from a real bump), MODIFIED, in
#     which every removed and added line is the same once its `version = "..."` value is blanked;
#   - CHANGELOG.md, MODIFIED;
#   - an ADDED evidence/dogfood/models/<V>/<host>.json, V the version Cargo.toml sets at SHA:
#     prepare_bump.sh commits the model-ladder receipts for the version being cut (#3708);
#   - an ADDED evidence/crux/<V>/prompt-certification.json or prompt-certification-inventory.json,
#     V as above: under the standing release policy prepare_bump.sh carries the CRUX prompt
#     certification forward into the version being cut, and those two names are all it writes.
# A changed dependency, path or source line, a modified or deleted receipt or certification,
# another version's directory, or another file name there is not a bump.
#
# BEFORE THE TAG (#4691). autopilot.sh cut_tag() runs `--resolve SHA` ahead of `git tag`, so a
# release commit with no qualifying receipt stops with no tag and no GitHub release made public.
# It resolves the receipt for SHA, not a ci.yml job name: the tag push no longer runs a coverage
# job worth waiting for (#4734). Preflight runs `TAG SHA` again after the tag.
#
# WHAT IS JUDGED. The receipt must be schema coverage-receipt/v1, for H, status measured,
# with passed > 0 tests, total > 0 lines, covered <= total, a pct that is covered/total
# truncated to two decimals, and pct >= COV_FLOOR as the Makefile at SHA sets it (the floor
# `make coverage` enforces). No qualifying run, no receipt in it, a receipt for another
# commit, no %, 0 tests, or gh failing refuses: Unknown is not a pass (L25). A receipt from
# before #4734 has no test count and refuses until a nightly writes one.
#
# ENV  GH (default gh) · GIT (git, run in the current directory) · TCG_REPO (paiml/aprender).
# EXIT 0 the commit's coverage was measured at or above COV_FLOOR · 1 anything else · 2 usage.
set -uo pipefail
PROG=tag_coverage_gate
SELF=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")
GH=${GH:-gh}
GIT=${GIT:-git}
REPO=${TCG_REPO:-paiml/aprender}
WF='coverage-nightly.yml'
ART='coverage-receipt'
SURFACE='^(Cargo\.toml|Cargo\.lock|CHANGELOG\.md|crates/[^/]+/Cargo\.toml|crates/facades/[^/]+/Cargo\.toml|crates/facades/Cargo\.lock)$'

# Pure. tcg_judge H FLOOR SCHEMA SHA STATUS PCT PASSED COVERED TOTAL REASON -> ok | bad <why>.
# Absent JSON fields arrive as the word null.
tcg_judge() {
    local h=$1 floor=$2 schema=$3 sha=$4 st=$5 pct=$6 passed=$7 cov=$8 tot=$9 why=${10:-} bp want
    [ "$schema" = coverage-receipt/v1 ] || { echo "bad the receipt's schema is '$schema', not coverage-receipt/v1"; return; }
    [ "$sha" = "$h" ] || { echo "bad the receipt is for $sha, not the run's commit $h"; return; }
    [ "$st" = measured ] || { echo "bad NOT_MEASURED: the nightly receipt says '$why' (L25: not a pass)"; return; }
    [[ $passed =~ ^[1-9][0-9]*$ ]] || { echo "bad NOT_MEASURED: the receipt counts '$passed' tests run (L25: not a pass)"; return; }
    [[ $pct =~ ^[0-9]+\.[0-9]{2}$ ]] || { echo "bad NOT_MEASURED: the receipt carries no line-coverage % ('$pct')"; return; }
    [[ $tot =~ ^[1-9][0-9]*$ ]] || { echo "bad NOT_MEASURED: the receipt measured '$tot' lines"; return; }
    [[ $cov =~ ^[0-9]+$ ]] && [ "$cov" -le "$tot" ] || { echo "bad NOT_MEASURED: the receipt reports $cov covered of $tot lines"; return; }
    bp=$((cov * 10000 / tot)); want=$(printf '%d.%02d' $((bp / 100)) $((bp % 100)))
    [ "$pct" = "$want" ] || { echo "bad the receipt's pct $pct is not its $cov/$tot lines ($want)"; return; }
    if awk -v p="$pct" -v f="$floor" 'BEGIN { exit !(p + 0 >= f + 0) }'; then echo ok
    else echo "bad coverage measured $pct%, below COV_FLOOR $floor%"; fi
}

# floor_at SHA -> COV_FLOOR as the Makefile at SHA sets it (one integer), or ''.
floor_at() {
    local v
    v=$("$GIT" show "$1:Makefile" 2>/dev/null | sed -nE 's/^COV_FLOOR := ([0-9]+)[[:space:]]*$/\1/p')
    [[ $v =~ ^[0-9]+$ ]] && echo "$v"
}

# blank_versions -> stdin's lines with every version = "..." value blanked, sorted
blank_versions() { sed -E 's/version = "[^"]*"/version = ""/g' | sort; }

# version_at SHA -> the [workspace.package] version Cargo.toml sets at SHA, or ''
version_at() {
    "$GIT" show "$1:Cargo.toml" 2>/dev/null | awk '
        /^\[/ { inw = ($0 == "[workspace.package]"); next }
        inw && /^version[ \t]*=/ { v = $0; sub(/^[^"]*"/, "", v); sub(/".*/, "", v); print v; exit }'
}

# version_only H SHA -> rc 0 iff SHA is H, or H plus a version-only bump
version_only() {
    local h=$1 sha=$2 v ev ec ns st p cargo='' d
    [ "$h" = "$sha" ] && return 0
    "$GIT" merge-base --is-ancestor "$h" "$sha" 2>/dev/null || return 1
    v=$(version_at "$sha"); [[ $v =~ ^[0-9A-Za-z.+-]+$ ]] || return 1
    ev="^evidence/dogfood/models/${v//./\\.}/[^/]+\\.json$"
    ec="^evidence/crux/${v//./\\.}/prompt-certification(-inventory)?\\.json$"
    ns=$("$GIT" diff --no-renames --name-status "$h" "$sha" 2>/dev/null) || return 1
    while IFS=$'\t' read -r st p; do
        if [ -z "$st" ]; then :
        elif [[ $p =~ $SURFACE ]]; then [ "$st" = M ] || return 1
            [[ $p == CHANGELOG.md ]] || cargo+="$p"$'\n'
        elif [[ $p =~ $ev || $p =~ $ec ]]; then [ "$st" = A ] || return 1
        else return 1; fi
    done <<< "$ns"
    [ -n "$cargo" ] || return 0
    # shellcheck disable=SC2086 # one path per line, none with spaces (the SURFACE list)
    d=$("$GIT" diff -U0 "$h" "$sha" -- $cargo 2>/dev/null) || return 1
    [ "$(grep -E '^-' <<< "$d" | grep -vE '^--- ' | cut -c2- | blank_versions)" = \
      "$(grep -E '^\+' <<< "$d" | grep -vE '^\+\+\+ ' | cut -c2- | blank_versions)" ]
}

# find_run SHA -> "<run id> <commit>" of the newest completed nightly on SHA or on a commit SHA
# is a version-only bump of; '' if none; rc 1 if gh failed or answered garbage.
find_run() {
    local runs id h
    runs=$("$GH" run list --repo "$REPO" --workflow "$WF" --status completed --limit 30 \
        --json databaseId,headSha,createdAt 2>/dev/null) || return 1
    runs=$(jq -er 'sort_by(.createdAt) | reverse | .[] | "\(.databaseId) \(.headSha)"' <<< "$runs" 2>/dev/null) \
        || jq -e 'length == 0' <<< "$runs" > /dev/null 2>&1 || return 1
    while read -r id h; do
        [ -n "$id" ] || continue
        if version_only "$h" "$1"; then echo "$id $h"; return 0; fi
    done <<< "$runs"
}

# receipt RUN_ID H -> the receipt's fields, tab separated (null for absent, - for empty), or ''
# if gh failed or the artifact holds no receipt for H.
receipt() {
    local d f out=''
    d=$(mktemp -d) || return 1
    if "$GH" run download "$1" --repo "$REPO" -n "$ART" -D "$d" > /dev/null 2>&1; then
        f="$d/coverage-receipt-$2.json"
        [ -s "$f" ] && out=$(jq -r '[.schema, .sha, .status, .pct, .passed, .covered, .total, .reason]
            | map(if . == null then "null" elif . == "" then "-" else tostring end) | @tsv' "$f" 2>/dev/null)
    fi
    rm -rf -- "${d:?}"
    printf '%s' "$out"
}

# gate WHAT SHA STOP -> judge the receipt for SHA. WHAT names the subject ("v0.70.2 at SHA"),
# STOP what a refusal stops. rc 0 measured at or above COV_FLOOR, 1 anything else.
gate() {
    local what=$1 sha=$2 stop=$3 floor found id h r v schema rsha st pct passed cov tot why
    floor=$(floor_at "$sha")
    if [ -z "$floor" ]; then echo "FAIL  no COV_FLOOR in the Makefile at $sha -- the floor is unknown, $stop"; return 1; fi
    found=$(find_run "$sha") || { echo "FAIL  gh could not list $WF runs -- Unknown is not a pass, $stop"; return 1; }
    if [ -z "$found" ]; then echo "FAIL  NOT_MEASURED: no completed $WF run on $sha or a version-only parent of it -- $stop"; return 1; fi
    id=${found%% *}; h=${found#* }
    r=$(receipt "$id" "$h")
    if [ -z "$r" ]; then echo "FAIL  NOT_MEASURED: $WF run $id left no readable receipt for $h -- $stop"; return 1; fi
    IFS=$'\t' read -r schema rsha st pct passed cov tot why <<< "$r"
    v=$(tcg_judge "$h" "$floor" "$schema" "$rsha" "$st" "$pct" "$passed" "$cov" "$tot" "$why")
    if [ "$v" = ok ]; then
        echo "ok    coverage $pct% >= COV_FLOOR $floor% ($passed tests) measured on $h by $WF run $id, for $what"; return 0
    fi
    echo "FAIL  ${v#bad } ($WF run $id on $h, for $what) -- $stop"
    return 1
}

self_test() {
    local fail=0 got want why d rc H=1111111111111111111111111111111111111111
    local schema sha st pct passed cov tot reason
    echo "$PROG self-test: judge table"
    while IFS='|' read -r want schema sha st pct passed cov tot reason why; do
        got=$(tcg_judge "$H" 89 "$schema" "$sha" "$st" "$pct" "$passed" "$cov" "$tot" "$reason")
        if [ "${got%% *}" = "$want" ]; then echo "  ok   $why"; else echo "  FAIL $why: wanted $want, got '$got'"; fail=1; fi
    done <<EOF
ok|coverage-receipt/v1|$H|measured|90.00|87772|9|10|-|measured above the floor
ok|coverage-receipt/v1|$H|measured|89.00|1|89|100|-|measured exactly at the floor
bad|coverage-receipt/v1|$H|measured|88.99|87772|8899|10000|-|below the floor
bad|coverage-receipt/v1|$H|not_measured|null|87772|null|null|no TOTAL line in the coverage log|status not_measured
bad|coverage-receipt/v1|$H|measured|null|87772|9|10|-|no %
bad|coverage-receipt/v1|$H|measured|90.00|0|9|10|-|0 tests
bad|coverage-receipt/v1|$H|measured|90.00|null|9|10|-|no test count (a receipt from before #4734)
bad|coverage-receipt/v1|2222222222222222222222222222222222222222|measured|90.00|87772|9|10|-|a receipt for another commit
bad|coverage-receipt/v2|$H|measured|90.00|87772|9|10|-|an unknown schema
bad|coverage-receipt/v1|$H|measured|100.00|87772|0|0|-|0 lines
bad|coverage-receipt/v1|$H|measured|110.00|87772|11|10|-|more lines covered than exist
bad|coverage-receipt/v1|$H|measured|99.00|87772|9|10|-|a pct that is not its covered/total
EOF
    # End to end through a real git history and the real jq filters, with a stub gh:
    # `run list` -> runs.json, `run download ID ... -D DIR` -> copies art/ID/* into DIR.
    d=$(mktemp -d) || return 1
    cat > "$d/gh" <<'STUB'
#!/usr/bin/env bash
[ -e "$FIX/down" ] && { echo "gh: HTTP 503" >&2; exit 1; }
case "$1 $2" in
    run\ list) cat "$FIX/runs.json" ;;
    run\ download)
        id=$3; dir=''; while [ "$#" -gt 0 ]; do [ "$1" = -D ] && dir=$2; shift; done
        [ -d "$FIX/art/$id" ] || { echo "no artifact" >&2; exit 1; }
        cp "$FIX/art/$id/"* "$dir/" 2>/dev/null; exit 0 ;;
esac
STUB
    chmod +x "$d/gh"
    # history: C (floor 89) -> B = C + version-only bump -> X = B + a code change; D = C + a
    # dependency change beside a version bump; N = C without COV_FLOOR; A = an unrelated root.
    # #4735, each B + one more change: EA adds a model-ladder receipt for the version cut (0.1.1),
    # EM modifies one C already held, EO adds one for another version, EN adds a file off the surface,
    # ED deletes CHANGELOG.md.
    local g="$d/repo" C B X D N A EA EM EO EN ED CA CM CD CO CN
    git init -q "$g" && git -C "$g" config user.email t@t && git -C "$g" config user.name t \
        && git -C "$g" config core.hooksPath /dev/null || return 1
    printf 'COV_FLOOR := 89\n' > "$g/Makefile"; mkdir -p "$g/crates/a" "$g/evidence/dogfood/models/0.1.1"
    printf '[workspace]\nmembers = ["crates/a"]\n\n[workspace.package]\nversion = "0.1.0"\n' > "$g/Cargo.toml"
    printf '[package]\nname = "a"\nversion = "0.1.0"\n\n[dependencies]\nb = { path = "../b", version = "0.1.0" }\n' > "$g/crates/a/Cargo.toml"
    printf '[[package]]\nname = "a"\nversion = "0.1.0"\n' > "$g/Cargo.lock"; printf 'fn f() {}\n' > "$g/lib.rs"; printf '# log\n' > "$g/CHANGELOG.md"
    printf '{"host":"early"}\n' > "$g/evidence/dogfood/models/0.1.1/early.json"
    git -C "$g" add -A && git -C "$g" commit -qm C && C=$(git -C "$g" rev-parse HEAD)
    sed -i 's/0\.1\.0/0.1.1/' "$g/Cargo.toml" "$g/crates/a/Cargo.toml" "$g/Cargo.lock"; printf '# log\n## 0.1.1\n' > "$g/CHANGELOG.md"
    git -C "$g" commit -qam B && B=$(git -C "$g" rev-parse HEAD)
    # on_b NAME FILE TEXT -> commit FILE=TEXT on top of B, print the commit
    on_b() { git -C "$g" checkout -q "$B" && mkdir -p "$(dirname "$g/$2")" && printf '%s\n' "$3" > "$g/$2" \
        && git -C "$g" add -A && git -C "$g" commit -qm "$1" && git -C "$g" rev-parse HEAD; }
    EA=$(on_b EA evidence/dogfood/models/0.1.1/intel.json '{"host":"intel"}')
    EM=$(on_b EM evidence/dogfood/models/0.1.1/early.json '{"host":"rewritten"}')
    EO=$(on_b EO evidence/dogfood/models/0.1.0/intel.json '{"host":"intel"}')
    EN=$(on_b EN docs/notes.md 'not a version')
    ED=$(git -C "$g" checkout -q "$B" && git -C "$g" rm -q CHANGELOG.md && git -C "$g" commit -qm ED && git -C "$g" rev-parse HEAD)
    # P7 D3, the CRUX prompt certification prepare_bump.sh carries into the version cut: CA adds
    # both files for 0.1.1 on B; CM modifies, CD deletes, one CA already holds; CO adds one for
    # another version, CN another name beside them.
    CA=$(on_b CA evidence/crux/0.1.1/prompt-certification-inventory.json '{"prompts":[]}' >/dev/null \
        && printf '{"apr_commit":"x"}\n' > "$g/evidence/crux/0.1.1/prompt-certification.json" \
        && git -C "$g" add -A && git -C "$g" commit -qm CA && git -C "$g" rev-parse HEAD)
    CM=$(git -C "$g" checkout -q "$CA" && printf '{"apr_commit":"y"}\n' > "$g/evidence/crux/0.1.1/prompt-certification.json" \
        && git -C "$g" commit -qam CM && git -C "$g" rev-parse HEAD)
    CD=$(git -C "$g" checkout -q "$CA" && git -C "$g" rm -q evidence/crux/0.1.1/prompt-certification.json \
        && git -C "$g" commit -qm CD && git -C "$g" rev-parse HEAD)
    CO=$(on_b CO evidence/crux/0.1.0/prompt-certification.json '{"apr_commit":"x"}')
    CN=$(on_b CN evidence/crux/0.1.1/prompt-certification-extra.json '{"apr_commit":"x"}')
    git -C "$g" checkout -q "$B"
    printf 'fn f() { g() }\n' > "$g/lib.rs"; git -C "$g" commit -qam X && X=$(git -C "$g" rev-parse HEAD)
    git -C "$g" checkout -q "$C" && sed -i -e 's|path = "../b"|path = "../evil"|' -e 's/0\.1\.0/0.1.1/g' "$g/crates/a/Cargo.toml"
    git -C "$g" commit -qam D && D=$(git -C "$g" rev-parse HEAD)
    git -C "$g" checkout -q "$C" && printf 'all:\n' > "$g/Makefile"; git -C "$g" commit -qam N && N=$(git -C "$g" rev-parse HEAD)
    git -C "$g" checkout -q --orphan other && git -C "$g" commit -qm A && A=$(git -C "$g" rev-parse HEAD)
    # rec SHA PCT PASSED COVERED TOTAL [STATUS] -> a receipt in the producer's form
    rec() { printf '{"schema":"coverage-receipt/v1","sha":"%s","floor":89,"pct":%s,"covered":%s,"total":%s,"status":"%s","reason":null,"passed":%s}' \
        "$1" "$2" "$4" "$5" "${6:-measured}" "$3"; }
    # e2e WANT_RC WHY SHA RUNS [ID:FILE-SHA:JSON | down ...] -- RUNS is "id:sha:createdAt,..."
    e2e() {
        local want=$1 why=$2 sha=$3 runs=$4 a rid rest fsha js first=1
        shift 4
        rm -rf -- "${d:?}/art" "${d:?}/down"; mkdir -p "$d/art"
        { printf '['
          for a in ${runs//,/ }; do
              [ "$first" = 1 ] || printf ','; first=0
              printf '{"databaseId":%s,"headSha":"%s","createdAt":"%s"}' "${a%%:*}" "$(cut -d: -f2 <<< "$a")" "$(cut -d: -f3- <<< "$a")"
          done; printf ']'; } > "$d/runs.json"
        for a in "$@"; do
            if [ "$a" = down ]; then : > "$d/down"; continue; fi
            rid=${a%%:*}; rest=${a#*:}; fsha=${rest%%:*}; js=${rest#*:}
            mkdir -p "$d/art/$rid"; printf '%s\n' "$js" > "$d/art/$rid/coverage-receipt-$fsha.json"
        done
        (cd "$g" && FIX=$d GH=$d/gh bash "$SELF" "$mode" "$sha") > "$d/out" 2>&1; rc=$?
        if [ "$rc" = "$want" ]; then echo "  ok   $why"; else echo "  FAIL $why: wanted rc $want, got $rc: $(cat "$d/out")"; fail=1; fi
    }
    local mode=v9.9.9
    local T1=2026-10-01T03:00:00Z T2=2026-10-02T03:00:00Z OK
    echo "$PROG self-test: end to end"
    OK=$(rec "$C" 90.00 87772 9 10)
    e2e 0 "e2e: a receipt for the release commit at or above COV_FLOOR passes" "$C" "7:$C:$T1" "7:$C:$OK"
    e2e 0 "e2e: a receipt for C passes for a version-only bump of C (Cargo versions + CHANGELOG)" "$B" "7:$C:$T1" "7:$C:$OK"
    e2e 1 "e2e: plant missing receipt -- no nightly run at all refuses" "$C" ""
    e2e 1 "e2e: plant missing receipt -- a run whose artifact holds no receipt refuses" "$C" "7:$C:$T1"
    e2e 1 "e2e: plant another sha -- a nightly on an unrelated commit refuses" "$C" "7:$A:$T1" "7:$A:$(rec "$A" 90.00 87772 9 10)"
    e2e 1 "e2e: plant another sha -- a receipt naming another commit refuses" "$C" "7:$C:$T1" "7:$C:$(rec "$A" 90.00 87772 9 10)"
    e2e 1 "e2e: plant another sha -- a run on C whose receipt file is for another commit refuses" "$C" "7:$C:$T1" "7:$A:$(rec "$A" 90.00 87772 9 10)"
    e2e 1 "e2e: plant no % -- a not_measured receipt refuses" "$C" "7:$C:$T1" "7:$C:$(rec "$C" null 87772 null null not_measured)"
    e2e 1 "e2e: plant 0 tests -- a receipt that ran 0 tests refuses" "$C" "7:$C:$T1" "7:$C:$(rec "$C" 90.00 0 9 10)"
    e2e 1 "e2e: plant 0 tests -- a receipt with no test count refuses" "$C" "7:$C:$T1" \
        "7:$C:{\"schema\":\"coverage-receipt/v1\",\"sha\":\"$C\",\"floor\":89,\"pct\":90.00,\"covered\":9,\"total\":10,\"status\":\"measured\",\"reason\":null}"
    e2e 1 "e2e: plant gh failing -- gh failing refuses (Unknown is not a pass)" "$C" "7:$C:$T1" "7:$C:$OK" down
    e2e 1 "e2e: a receipt below COV_FLOOR refuses" "$C" "7:$C:$T1" "7:$C:$(rec "$C" 88.99 87772 8899 10000)"
    e2e 1 "e2e: a receipt for C does not cover a CODE change after C" "$X" "7:$C:$T1" "7:$C:$OK"
    e2e 1 "e2e: a receipt for C does not cover a dependency change hidden in a version bump" "$D" "7:$C:$T1" "7:$C:$OK"
    e2e 1 "e2e: a receipt for a LATER commit does not cover an earlier one" "$C" "7:$X:$T1" "7:$X:$(rec "$X" 90.00 87772 9 10)"
    e2e 1 "e2e: the newest qualifying run wins -- a newer failing receipt is not skipped for an older passing one" "$B" \
        "7:$C:$T1,8:$B:$T2" "8:$B:$(rec "$B" 80.00 87772 8 10)" "7:$C:$OK"
    e2e 0 "e2e: an unrelated newer run is skipped for the qualifying one" "$B" "8:$A:$T2,7:$C:$T1" "7:$C:$OK"
    e2e 1 "e2e: no COV_FLOOR at the release commit refuses" "$N" "7:$N:$T1" "7:$N:$(rec "$N" 90.00 87772 9 10)"
    echo "$PROG self-test: the version surface (#4735)"
    e2e 0 "e2e: an ADDED model-ladder receipt for the version cut rides on the bump" "$EA" "7:$C:$T1" "7:$C:$OK"
    e2e 1 "e2e: a MODIFIED model-ladder receipt is not a bump" "$EM" "7:$C:$T1" "7:$C:$OK"
    e2e 1 "e2e: a receipt in ANOTHER version's dir is not a bump" "$EO" "7:$C:$T1" "7:$C:$OK"
    e2e 1 "e2e: any other path is not a bump" "$EN" "7:$C:$T1" "7:$C:$OK"
    e2e 1 "e2e: a DELETED surface file is not a bump" "$ED" "7:$C:$T1" "7:$C:$OK"
    e2e 1 "e2e: a receipt for a later version-only bump does not cover the commit before it" "$C" "7:$B:$T1" "7:$B:$(rec "$B" 90.00 87772 9 10)"
    echo "$PROG self-test: the carried CRUX prompt certification (P7 D3)"
    e2e 0 "e2e: an ADDED prompt certification and inventory for the version cut ride on the bump" "$CA" "7:$C:$T1" "7:$C:$OK"
    e2e 1 "e2e: a MODIFIED prompt certification is not a bump" "$CM" "7:$CA:$T1" "7:$CA:$(rec "$CA" 90.00 87772 9 10)"
    e2e 1 "e2e: a DELETED prompt certification is not a bump" "$CD" "7:$CA:$T1" "7:$CA:$(rec "$CA" 90.00 87772 9 10)"
    e2e 1 "e2e: a prompt certification in ANOTHER version's dir is not a bump" "$CO" "7:$C:$T1" "7:$C:$OK"
    e2e 1 "e2e: another file name beside the certification is not a bump" "$CN" "7:$C:$T1" "7:$C:$OK"
    echo "$PROG self-test: --resolve SHA, before the tag (#4691)"
    mode="--resolve"
    e2e 0 "e2e: --resolve passes on a release commit whose receipt holds the floor" "$B" "7:$C:$T1" "7:$C:$OK"
    e2e 1 "e2e: --resolve refuses with no receipt for the release commit" "$B" ""
    e2e 1 "e2e: --resolve refuses a receipt below COV_FLOOR" "$C" "7:$C:$T1" "7:$C:$(rec "$C" 88.99 87772 8899 10000)"
    e2e 1 "e2e: --resolve refuses when gh fails" "$C" "7:$C:$T1" "7:$C:$OK" down
    mode=v9.9.9
    printf 'rate limited' > "$d/runs.json"; rm -f -- "${d:?}/down"
    if (cd "$g" && FIX=$d GH=$d/gh bash "$SELF" v9.9.9 "$C") > /dev/null 2>&1; then echo "  FAIL e2e: gh answering garbage passed"; fail=1
    else echo "  ok   e2e: gh answering garbage refuses"; fi
    rm -rf -- "${d:?}"
    [ "$fail" -eq 0 ] && { echo "$PROG self-test: PASS"; return 0; }
    echo "$PROG self-test: FAIL"; return 1
}

case "${1:-}" in
    --self-test) self_test ;;
    -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d'; exit 0 ;;
    --resolve) [ "$#" -eq 2 ] || { echo "usage: $0 --resolve SHA" >&2; exit 2; }
       gate "the release commit $2" "$2" "no tag, nothing carried" ;;
    *) [ "$#" -eq 2 ] || { echo "usage: $0 TAG SHA | --resolve SHA | --self-test" >&2; exit 2; }
       gate "$1 at $2" "$2" "no preflight, no cascade" ;;
esac
