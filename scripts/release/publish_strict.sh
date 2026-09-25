#!/usr/bin/env bash
# publish_strict.sh <version> [--plan] — the crates.io cascade under the operator's 2026-09-17
#   authorization (first run 0.68.2). The version is the one argument; T and the state dir AP are
#   derived (#3618, lib_release_params.sh), never literals.
#   one crate per `cargo publish` call, in the tag's own TIERS order (scripts/cascade-publish.sh);
#   STOP on the first non-zero (partial publishes are recorded, never rolled back);
#   never --allow-dirty; a crates.io transient (429/5xx/timeout) is retried <=3 times with backoff,
#   same inputs; anything else stops. Runs only from a detached checkout whose HEAD == the tag.
#   publish_strict.sh <version> --plan   prints the ordered plan and uploads nothing
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" || { echo "cannot resolve the repo root from $0" >&2; exit 2; }
# shellcheck source=scripts/release/lib_release_params.sh
. "$REPO_ROOT/scripts/release/lib_release_params.sh" || exit 2
# shellcheck source=scripts/lib/cascade_universe.sh
. "$REPO_ROOT/scripts/lib/cascade_universe.sh" || exit 2

# ---- the two JSON reads, in jq (#4352: python3 is out of the release path) ----
# ps_order_violations ORDER...: stdin is `cargo metadata --no-deps`. Prints every `pkg<-dep`
# where both are in ORDER and dep comes AFTER pkg -- normal, build and versioned dev-deps; an
# unversioned dev-dep (req "*") is not a publish edge. Space-joined on one line (an empty line
# when there are none); rc 1 when there is any, and non-zero on unreadable or empty metadata,
# a second JSON document after it, or a dev-dep without "req" (python's json.load / KeyError).
ps_order_violations() {
  jq -rn '($ARGS.positional) as $o
    | (reduce range($o | length) as $i ({}; .[$o[$i]] = $i)) as $pos
    | (input as $m | if ([inputs] | length) > 0 then error("extra data after the metadata") else $m end) as $m
    | [$m.packages[] as $p | select($pos | has($p.name))
       | $p.dependencies[] as $d
       | select(($pos | has($d.name)) and $d.name != $p.name
                and ($d.kind != "dev" or ($d | if has("req") then .req else error("dev-dep without req") end) != "*") and $pos[$d.name] > $pos[$p.name])
       | "\($p.name)<-\($d.name)"]
    | join(" "), (if length > 0 then "" | halt_error(1) else empty end)' --args "$@"
}
# ps_index_has VERSION: stdin is a sparse-index file (one JSON object per line). rc 0 when a line's
# "vers" is exactly VERSION. Lines are read in order and the scan stops at the first match: a
# blank line is skipped, a malformed or non-object line BEFORE the match is non-zero (not live),
# one after it is never read -- python's lazy any() over json.loads, case for case.
ps_index_has() {
  jq -Rne --arg v "$1" '(first(inputs | select(test("\\S")) | fromjson
      | if type == "object" then . else error("not an object") end
      | select(.vers == $v)) | true) // false' > /dev/null
}
if [ "${1:-}" = "--self-test" ]; then
  # I/O rows: the two jq reads and the universe (a fake `cargo` on PATH serves the metadata).
  # The parity table against the python they replaced is in #4352 (order 130/130, index 200/200,
  # universe 120/132 -- the 12 are python's traceback rc 1 on unreadable metadata, here rc 2).
  ps_cu() { # ps_cu CASE [--names]: cascade_universe over fixture CASE
    PATH="$PS_T/bin:$PATH" PS_CASE=$1 cascade_universe ${2:+"$2"} /r
  }
  ps_io_rows() {
    local m='{"packages":[{"name":"a","dependencies":[{"name":"b","req":"^1"},{"name":"a","req":"*"}]},
      {"name":"b","dependencies":[{"name":"c","kind":"dev","req":"*"},{"name":"d","kind":"dev","req":"^1"}]},
      {"name":"c","dependencies":[{"name":"x","kind":"build","req":"^1"}]}]}'
    row 1 "a<-b" "a needs b and b comes after it -> violation"            ps_order_violations "$m" a b c
    row 0 ""     "b before a; c after b is an unversioned dev-dep -> clean" ps_order_violations "$m" b c a
    row 1 "b<-d" "a VERSIONED dev-dep after its crate is a publish edge"    ps_order_violations "$m" b d a
    row 0 ""     "a dep outside ORDER (x) is not judged"                    ps_order_violations "$m" c b a
    row 1 "-"    "empty metadata is a refusal, never a clean order"         ps_order_violations "" a b
    row 1 "-"    "a second JSON document is a refusal (json.load)"          ps_order_violations "$m"$'\n{}' b c a
    row 1 "-"    "a dev-dep without req is a refusal (KeyError)" \
      ps_order_violations '{"packages":[{"name":"a","dependencies":[{"name":"b","kind":"dev"}]}]}' b a
    local ix=$'{"vers":"1.0.0"}\n\n{"vers":"1.0.1","yanked":true}\n'
    row 0 "-" "the version is on the index -> live"                         ps_index_has "$ix" 1.0.1
    row 1 "-" "a prefix is not the version"                                 ps_index_has "$ix" 1.0
    row 1 "-" "an empty index file -> not live"                             ps_index_has "" 1.0.0
    row 1 "-" "a malformed line before the match -> not live"               ps_index_has $'{bad\n{"vers":"1.0.0"}' 1.0.0
    row 0 "-" "a malformed line after the match is never read"              ps_index_has $'{"vers":"1.0.0"}\n{bad' 1.0.0
    row 1 "-" "a number is not the version string"                          ps_index_has '{"vers":1}' 1
    row 0 "75" "the universe: 72 root + 3 facade crates, one row each"      ps_cu_count "" ok
    row 0 "c-0	0.1.0	/r/c-0/Cargo.toml	/r" "a row is name, version, manifest, workspace root" ps_cu_first "" ok
    row 0 "72" "publish = false is not in the universe"                     ps_cu_count "" nopub
    row 1 "-" "one name from two manifests -> refused"                      ps_cu_count "" dup
    row 1 "-" "fewer than CU_MIN_CRATES -> refused, never a short cascade"  ps_cu_count "" few
    row 1 "-" "cargo metadata exiting non-zero -> refused, even with JSON on stdout"  ps_cu_count "" fail
  }
  ps_cu_count() { local o; o=$(ps_cu "$1") || return 1; printf '%s\n' "$o" | wc -l | tr -d ' '; }
  ps_cu_first() { local o; o=$(ps_cu "$1") || return 1; printf '%s\n' "$o" | head -1; }
  ps_fixtures() { # 72 root crates c-0..c-71 (+/-) under /r, 3 facades under /r/crates/facades
    local d=$1; mkdir -p "$d/bin" || return 1
    printf '%s\n' '#!/usr/bin/env bash' \
      'case ${*: -1} in */crates/facades/Cargo.toml) t=fac ;; *) t=root ;; esac' \
      'cat "$PS_T/$PS_CASE.$t" || exit 101; [ "$PS_CASE" != fail ] || exit 101' > "$d/bin/cargo"
    chmod +x "$d/bin/cargo"
    gen() { jq -n --argjson n "$2" --arg p "$3" --arg r "$4" \
      "{packages: [range(\$n) | {name: \"\(\$p)-\(.)\", version: \"0.1.\(.)\", manifest_path: \"\(\$r)/\(\$p)-\(.)/Cargo.toml\"}]} ${5:+| $5}" > "$d/$1"; }
    gen ok.root 72 c /r && gen ok.fac 3 f /r/crates/facades &&
    gen nopub.root 72 c /r '.packages[0].publish = [] | .packages[1].publish = null' && gen nopub.fac 1 f /r/crates/facades &&
    gen dup.root 72 c /r && gen dup.fac 3 c /r/crates/facades &&
    gen few.root 66 c /r && gen few.fac 3 f /r/crates/facades &&
    gen fail.root 72 c /r && gen fail.fac 3 f /r/crates/facades  # served, but cargo exits 101
  }
  ps_self_test() {
    local fail=0 n=0 got rc
    row() { # row WANT_RC WANT_OUT LABEL FN INPUT ARGS...
      local want_rc=$1 want_out=$2 label=$3 fn=$4 input=$5; shift 5
      n=$((n+1)); got=$(printf '%s' "$input" | "$fn" "$@" 2>/dev/null); rc=$?
      [ "$rc" -ne 0 ] && rc=1
      if [ "$rc" = "$want_rc" ] && { [ "$want_out" = "-" ] || [ "$got" = "$want_out" ]; }; then :; else
        fail=1; printf 'FAIL %s: rc=%s out=[%s] want rc=%s out=[%s]\n' "$label" "$rc" "$got" "$want_rc" "$want_out"; fi
    }
    PS_T=$(mktemp -d) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf -- '$PS_T'" RETURN
    export PS_T
    ps_fixtures "$PS_T" || { echo "FAIL: fixtures"; return 1; }
    ps_io_rows
    [ -n "${PS_NO_MUTANTS:-}" ] && return "$fail"
    # Mutants: each breaks one rule in a copy of this script (or of the universe lib) and the
    # rows above must go red. The anchors are asserted unique, so a mutant cannot land in a row.
    local me="$REPO_ROOT/scripts/release/publish_strict.sh" lib="$REPO_ROOT/scripts/lib/cascade_universe.sh"
    local spec f a b src tail t k=0 mark='if [ "${1:-}" = "--self-test" ]; then'
    while IFS='~' read -r f a b; do
      # this file's mutants land only above the self-test (its rows and this table hold the anchors too)
      k=$((k+1)); tail=""
      if [ "$f" = lib ]; then src=$(cat "$lib"); else src=$(cat "$me"); tail=${src#*"$mark"}; src=${src%%"$mark"*}; fi
      t=$PS_T/mut$k; mkdir -p "$t/scripts/release" "$t/scripts/lib"
      cp "$REPO_ROOT/scripts/release/lib_release_params.sh" "$t/scripts/release/"
      cp "$me" "$t/scripts/release/publish_strict.sh"; cp "$lib" "$t/scripts/lib/cascade_universe.sh"
      spec=${src//"$a"/}; [ "${#spec}" -eq "$(( ${#src} - ${#a} ))" ] || { fail=1; echo "FAIL mutant $k: anchor not unique: $a"; continue; }
      if [ "$f" = lib ]; then printf '%s\n' "${src/"$a"/"$b"}" > "$t/scripts/lib/cascade_universe.sh"
      else printf '%s\n' "${src/"$a"/"$b"}$mark$tail" > "$t/scripts/release/publish_strict.sh"; fi
      if PS_NO_MUTANTS=1 bash "$t/scripts/release/publish_strict.sh" --self-test > /dev/null 2>&1; then
        fail=1; echo "FAIL mutant $k survived: $a -> $b"
      fi
    done <<'MUT'
me~and ($d.kind != "dev" or~and (true or
me~> $pos[$p.name])~< $pos[$p.name])
me~if ([inputs] | length) > 0 then~if false then
me~first(inputs | select(test("\\S")) | fromjson~first(inputs | fromjson
me~select(.vers == $v)) | true)~select(.vers | startswith($v))) | true)
lib~select(.publish != [])~.
lib~if has($r[0]) and .[$r[0]][1] != $r[2] then~if false then
lib~if ($seen | length) < $min then~if false then
lib~--manifest-path "$manifest" 2> /dev/null) \~--manifest-path "$manifest" 2> /dev/null || true) \
MUT
    [ "$fail" = 0 ] && printf 'publish_strict self-test: %d rows + %d mutants PASS\n' "$n" "$k"
    return "$fail"
  }
  ps_self_test; exit $?
fi
release_params "${1:-}" "$REPO_ROOT" || { echo "usage: publish_strict.sh <version> [--plan]" >&2; exit 2; }
shift
WT="$AP/wt"
TSV="$AP/publish-timestamps.tsv"; LOGD="$AP/publish-logs"; STATUS="$AP/STATUS"
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$STATUS"; }  # bashrs disable-line=DET002
die() { say "STOP publish: $*"; exit 1; }
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
unset CARGO_REGISTRY_TOKEN
cd "$WT" || die "no $WT"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "refs/tags/$T^{commit}")" ] || die "HEAD is not $T"
git symbolic-ref -q HEAD > /dev/null && die "checkout is not detached"
[ -z "$(git status --porcelain)" ] || die "tree dirty: $(git status --porcelain | head -3 | tr '\n' ' ')"
[ -e .cargo/config.toml ] && die ".cargo/config.toml present in the publish tree"
[ -s "$HOME/.cargo/credentials.toml" ] || die "no publish token on this host (precondition 5)"
[ "${1:-}" = "--plan" ] || [ -s "$AP/cleanroom-run-id" ] || die "no green clean-room run id recorded for $T (rule 7, #3335)"
[ "${1:-}" = "--plan" ] || [ -s "$AP/b2gpu-run-id" ] || die "no green B2-gpu run id recorded for $T (rule 14)"
[ "${1:-}" = "--plan" ] || [ -s "$AP/dryrun-receipt-commit" ] || die "no committed dry-run receipt (T-4)"

# order: NOT the tag's TIERS — measured 2026-09-17, TIERS is not topological (47 non-dev
# violations; it only ever worked through the drain's retries). publish-order.txt is derived from
# `cargo metadata` at the tag (normal + build + versioned dev-deps, acyclic); the facades, which
# version independently and resolve their upstream from the registry, go last. Re-proved below.
# An empty crate name is refused BY LINE NUMBER, before any set check (#3696): a blank line used to
# reach the checks below as "" and stop with "not in the universe:  " -- naming nothing.
blank=$(awk '/^[[:space:]]*$/ { printf "%s%d", (n++ ? ", " : ""), NR }' "$WT/scripts/release/publish-order.txt")
[ -z "$blank" ] || die "blank line $blank in publish-order.txt: an empty crate name is not a crate"
mapfile -t ORDER < <(cat "$WT/scripts/release/publish-order.txt"; printf '%s\n' provable-contracts provable-contracts-macros provable-contracts-cli)
declare -A EXPECT MANIFEST ROOTWS
while IFS=$'\t' read -r n v m w; do [ -n "$n" ] && { EXPECT[$n]=$v; MANIFEST[$n]=$m; ROOTWS[$n]=$w; }; done < <(cascade_universe "$WT")
# The ORDER must equal the universe as a SET (#3657): no crate missing, none extra, none twice.
# The size is N, read from the universe. It is never a literal: 0.68.2's `74` would have
# stopped a later cascade for a reason unrelated to publish safety
# (check_release_scripts_derive_identity.sh R3).
N=${#EXPECT[@]}
[ "$N" -gt 0 ] || die "the universe is empty: cascade_universe enumerated nothing at $WT"
dup=$(printf '%s\n' "${ORDER[@]}" | sort | uniq -d | tr '\n' ' ')
[ -z "$dup" ] || die "crate(s) twice in the publish order: $dup"
extra=$(for c in "${ORDER[@]}"; do [ -n "${EXPECT[$c]:-}" ] || printf '%s\n' "$c"; done | sort | tr '\n' ' ')
[ -z "$extra" ] || die "in the publish order but not in the universe: $extra"
declare -A INORDER; for c in "${ORDER[@]}"; do INORDER[$c]=1; done
missing=$(for c in "${!EXPECT[@]}"; do [ -n "${INORDER[$c]:-}" ] || printf '%s\n' "$c"; done | sort | tr '\n' ' ')
[ -z "$missing" ] || die "in the universe but not in the publish order, so never uploaded: $missing"

cargo metadata --format-version 1 --no-deps 2>/dev/null | ps_order_violations "${ORDER[@]}" > "$AP/order-check.txt" || die "publish order is not a dependency order: $(cat "$AP/order-check.txt")"

live() { # live <crate> <version>: 0 when exactly that version is on the sparse index
  local c=$1 v=$2 p n=${#1}
  case $n in 1) p="1/$c";; 2) p="2/$c";; 3) p="3/${c:0:1}/$c";; *) p="${c:0:2}/${c:2:2}/$c";; esac
  curl -sf --retry 3 --retry-delay 2 -H 'User-Agent: aprender-release-train' "https://index.crates.io/$p" \
    | ps_index_has "$v"
}
if [ "${1:-}" = "--plan" ]; then
  i=0; for c in "${ORDER[@]}"; do i=$((i+1)); if live "$c" "${EXPECT[$c]}"; then s=LIVE; else s=TODO; fi; printf '%2d %s %s %s\n' $i "$c" "${EXPECT[$c]}" $s; done; exit 0
fi

mkdir -p "$LOGD"; [ -f "$TSV" ] || printf 'crate\tversion\tstart_utc\tend_utc\tresult\tattempts\n' > "$TSV"
say "CASCADE START $V strict ($N crates, dependency order derived at the tag)"
i=0
for c in "${ORDER[@]}"; do
  i=$((i+1)); v=${EXPECT[$c]}
  if live "$c" "$v"; then printf '%s\t%s\t-\t%s\talready-live\t0\n' "$c" "$v" "$(date -u +%FT%TZ)" >> "$TSV"; continue; fi  # bashrs disable-line=DET002
  sel=(-p "$c"); [ "${ROOTWS[$c]}" != "$WT" ] && sel=(--manifest-path "${MANIFEST[$c]}")
  t0=$(date -u +%FT%TZ); a=0  # bashrs disable-line=DET002
  while :; do
    a=$((a+1)); log="$LOGD/$(printf '%02d' $i)-$c.attempt$a.log"
    cargo publish "${sel[@]}" --locked > "$log" 2>&1; rc=$?
    [ $rc -eq 0 ] && break
    if live "$c" "$v"; then rc=0; break; fi   # uploaded, then the wait-for-index timed out
    if [ $a -lt 3 ] && grep -qiE '\b(429|500|502|503|504)\b|too many requests|timed? ?out|connection (reset|closed)|temporar' "$log"; then
      say "TRANSIENT $c attempt $a rc=$rc: $(grep -iE 'error|429|50[0-9]' "$log" | head -1 | cut -c1-160); backoff $((a*120))s"
      sleep $((a*120)); continue
    fi
    printf '%s\t%s\t%s\t%s\tFAILED rc=%s\t%s\n' "$c" "$v" "$t0" "$(date -u +%FT%TZ)" "$rc" "$a" >> "$TSV"  # bashrs disable-line=DET002
    say "PUBLISHED so far: $(awk -F'\t' '$5=="published"' "$TSV" | wc -l) crate(s); see $TSV"
    die "crate $i/$N $c rc=$rc after $a attempt(s): $(grep -iE '^error|caused by' "$log" | head -2 | tr '\n' ' ' | cut -c1-300) ($log)"
  done
  for _ in $(seq 1 30); do live "$c" "$v" && break; sleep 10; done
  live "$c" "$v" || die "$c $v: cargo publish rc=0 but the version is not on the index after 5 min"
  printf '%s\t%s\t%s\t%s\tpublished\t%s\n' "$c" "$v" "$t0" "$(date -u +%FT%TZ)" "$a" >> "$TSV"  # bashrs disable-line=DET002
  [ -z "$(git status --porcelain)" ] || die "tree dirty after publishing $c"
done
n=0; for c in "${ORDER[@]}"; do live "$c" "${EXPECT[$c]}" && n=$((n+1)); done
[ "$n" -eq "$N" ] || die "final verification: $n/$N live"
say "CASCADE END $V: $N/$N live on crates.io"
