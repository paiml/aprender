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
mapfile -t ORDER < <(cat "$WT/scripts/release/publish-order.txt"; printf '%s\n' provable-contracts provable-contracts-macros provable-contracts-cli)
declare -A EXPECT MANIFEST ROOTWS
while IFS=$'\t' read -r n v m w; do [ -n "$n" ] && { EXPECT[$n]=$v; MANIFEST[$n]=$m; ROOTWS[$n]=$w; }; done < <(python3 scripts/lib/cascade_universe.py "$WT")
# The ORDER must equal the universe as a SET (#3657): no crate missing, none extra, none twice.
# The size is N, read from the universe. It is never a literal: 0.68.2's `74` would have
# stopped a later cascade for a reason unrelated to publish safety
# (check_release_scripts_derive_identity.sh R3).
N=${#EXPECT[@]}
[ "$N" -gt 0 ] || die "the universe is empty: cascade_universe.py enumerated nothing at $WT"
dup=$(printf '%s\n' "${ORDER[@]}" | sort | uniq -d | tr '\n' ' ')
[ -z "$dup" ] || die "crate(s) twice in the publish order: $dup"
extra=$(for c in "${ORDER[@]}"; do [ -n "${EXPECT[$c]:-}" ] || printf '%s\n' "$c"; done | sort | tr '\n' ' ')
[ -z "$extra" ] || die "in the publish order but not in the universe: $extra"
declare -A INORDER; for c in "${ORDER[@]}"; do INORDER[$c]=1; done
missing=$(for c in "${!EXPECT[@]}"; do [ -n "${INORDER[$c]:-}" ] || printf '%s\n' "$c"; done | sort | tr '\n' ' ')
[ -z "$missing" ] || die "in the universe but not in the publish order, so never uploaded: $missing"

cargo metadata --format-version 1 --no-deps 2>/dev/null | python3 -c '
import json,sys
order=sys.argv[1:]; pos={c:i for i,c in enumerate(order)}; bad=[]
for p in json.load(sys.stdin)["packages"]:
    if p["name"] not in pos: continue
    for d in p["dependencies"]:
        if d["name"] in pos and d["name"]!=p["name"] and (d.get("kind")!="dev" or d["req"]!="*") and pos[d["name"]]>pos[p["name"]]:
            bad.append(p["name"]+"<-"+d["name"])
print(" ".join(bad)); sys.exit(1 if bad else 0)' "${ORDER[@]}" > "$AP/order-check.txt" || die "publish order is not a dependency order: $(cat "$AP/order-check.txt")"

live() { # live <crate> <version>: 0 when exactly that version is on the sparse index
  local c=$1 v=$2 p n=${#1}
  case $n in 1) p="1/$c";; 2) p="2/$c";; 3) p="3/${c:0:1}/$c";; *) p="${c:0:2}/${c:2:2}/$c";; esac
  curl -sf --retry 3 --retry-delay 2 -H 'User-Agent: aprender-release-train' "https://index.crates.io/$p" \
    | python3 -c 'import json,sys; v=sys.argv[1]; sys.exit(0 if any(json.loads(l).get("vers")==v for l in sys.stdin if l.strip()) else 1)' "$v"
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
