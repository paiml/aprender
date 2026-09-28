#!/usr/bin/env bash
# publish_strict.sh <version> [--plan] — the crates.io cascade under the operator's 2026-09-17
#   authorization (first run 0.68.2). The version is the one argument; T and the state dir AP are
#   derived (#3618, lib_release_params.sh), never literals.
#   one crate per `cargo publish` call, in the tag's own TIERS order (scripts/cascade-publish.sh);
#   STOP on the first non-zero (partial publishes are recorded, never rolled back);
#   never --allow-dirty; a crates.io transient (429/5xx/timeout) is retried <=3 times with backoff,
#   same inputs; anything else stops. Runs only from a detached checkout whose HEAD == the tag.
#   publish_strict.sh <version> --plan   prints the ordered plan and uploads nothing
#   --only a,b   publish ONLY these crates (#4587 B2a, pv 0.69.2 on its own). The universe and order
#                checks still run on the whole cascade; the selection is then refused unless every
#                universe crate it depends on and does not publish is already LIVE at its version.
#   --tag NAME   the tag HEAD must equal, when it is not v<version>: a scoped tag (pv-v0.69.2) that
#                no v* workflow trigger matches. It must end in -v<version>; its state dir is its own.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" || { echo "cannot resolve the repo root from $0" >&2; exit 2; }
# shellcheck source=scripts/release/lib_release_params.sh
. "$REPO_ROOT/scripts/release/lib_release_params.sh" || exit 2
USAGE="usage: publish_strict.sh <version> [--plan] [--only crate,crate] [--tag NAME]"
release_params "${1:-}" "$REPO_ROOT" || { echo "$USAGE" >&2; exit 2; }
shift
PLAN=""; ONLY=""; TAG=$T
while [ $# -gt 0 ]; do
  case $1 in
    --plan) PLAN=1; shift ;;
    --only) [ -n "${2:-}" ] || { echo "$USAGE" >&2; exit 2; }; ONLY=$2; shift 2 ;;
    --tag) [ -n "${2:-}" ] || { echo "$USAGE" >&2; exit 2; }; TAG=$2; shift 2 ;;
    *) echo "$USAGE" >&2; exit 2 ;;
  esac
done
if [ "$TAG" != "$T" ]; then
  [[ $TAG =~ ^[a-z][a-z0-9-]*-v${V//./\\.}$ ]] || { echo "--tag $TAG does not end in -$T: a scoped tag names the version it cuts" >&2; exit 2; }
  AP="${AP%/*}/$TAG"; mkdir -p "$AP" || { echo "cannot create $AP" >&2; exit 2; }
fi
WT="$AP/wt"
TSV="$AP/publish-timestamps.tsv"; LOGD="$AP/publish-logs"; STATUS="$AP/STATUS"
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$STATUS"; }  # bashrs disable-line=DET002
die() { say "STOP publish: $*"; exit 1; }
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
unset CARGO_REGISTRY_TOKEN
cd "$WT" || die "no $WT"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "refs/tags/$TAG^{commit}")" ] || die "HEAD is not $TAG"
git symbolic-ref -q HEAD > /dev/null && die "checkout is not detached"
[ -z "$(git status --porcelain)" ] || die "tree dirty: $(git status --porcelain | head -3 | tr '\n' ' ')"
[ -e .cargo/config.toml ] && die ".cargo/config.toml present in the publish tree"
[ -s "$HOME/.cargo/credentials.toml" ] || die "no publish token on this host (precondition 5)"
[ -n "$PLAN" ] || [ -s "$AP/cleanroom-run-id" ] || die "no green clean-room run id recorded for $TAG (rule 7, #3335)"
[ -n "$PLAN" ] || [ -s "$AP/b2gpu-run-id" ] || die "no green B2-gpu run id recorded for $TAG (rule 14)"
[ -n "$PLAN" ] || [ -s "$AP/dryrun-receipt-commit" ] || die "no committed dry-run receipt (T-4)"
# The recorded id is not the evidence; the run is (#4587 B2a). Fail closed unless the infra run's
# `clean-room (aprender)` job is green AND it names the tag's own commit. The proof is the result
# artifact when the job uploaded one; otherwise the job's own log line from assert-tested-ref.sh, read
# through the API. Run 36422691130 tested the tag green and uploaded nothing: the container wrote no
# result csv, and the upload step ignores a missing file. That assert step fails the job on any other
# commit, so a green job whose log says `tested-sha: <tag sha>` tested the tag.
if [ -z "$PLAN" ]; then
  crun=""; IFS= read -r crun < "$AP/cleanroom-run-id"
  tsha=$(git rev-parse "refs/tags/$TAG^{commit}")
  job=$(gh run view "$crun" --repo "$INFRA" --json jobs --jq '.jobs[] | select(.name=="clean-room (aprender)") | "\(.databaseId) \(.conclusion)"' 2>/dev/null | head -1)
  jid=${job%% *}; jc=${job#* }
  [ "$jc" = success ] || die "clean-room run $crun: job clean-room (aprender) is '${jc:-absent}', not success"
  rm -rf "${AP:?}/cleanroom-artifact"
  if gh run download "$crun" --repo "$INFRA" -n result-aprender -D "$AP/cleanroom-artifact" > /dev/null 2>&1; then
    grep -rqF "$tsha" "$AP/cleanroom-artifact" || die "clean-room run $crun did not test $TAG ($tsha): its artifact names another commit"
    proof="artifact result-aprender"
  else
    gh api "repos/$INFRA/actions/jobs/$jid/logs" > "$AP/cleanroom-job.log" 2> /dev/null \
      || die "clean-room run $crun: no result-aprender artifact and no job log to prove which commit it tested"
    grep -qE "Z +tested-sha: $tsha\$" "$AP/cleanroom-job.log" \
      || die "clean-room run $crun did not test $TAG ($tsha): no artifact, and its job log has no tested-sha line for it"
    proof="job $jid log (no artifact uploaded)"
  fi
  say "CLEANROOM VERIFIED run $crun tested $TAG ($tsha): $proof"
fi

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
if [ -n "$ONLY" ]; then
  mapfile -t SEL < <(tr ',' '\n' <<< "$ONLY" | sed '/^$/d')
  declare -A INSEL; for c in "${SEL[@]}"; do [ -n "${EXPECT[$c]:-}" ] || die "--only $c: not in the universe"; INSEL[$c]=1; done
  # a crate we do not publish must already be live at the version the selection requires of it
  mapfile -t NEED < <(cargo metadata --format-version 1 --no-deps 2>/dev/null | python3 -c '
import json,sys
sel=set(sys.argv[1:])
for p in json.load(sys.stdin)["packages"]:
    if p["name"] not in sel: continue
    for d in p["dependencies"]:
        if d["name"] not in sel and (d.get("kind")!="dev" or d["req"]!="*"): print(d["name"])' "${SEL[@]}" | sort -u)
  for d in "${NEED[@]}"; do
    [ -n "${EXPECT[$d]:-}" ] || continue
    live "$d" "${EXPECT[$d]}" || die "--only leaves out $d, which it needs at ${EXPECT[$d]}, and that is not live"
  done
  KEPT=(); for c in "${ORDER[@]}"; do [ -n "${INSEL[$c]:-}" ] && KEPT+=("$c"); done
  ORDER=("${KEPT[@]}"); N=${#ORDER[@]}
fi
if [ -n "$PLAN" ]; then
  i=0; for c in "${ORDER[@]}"; do i=$((i+1)); if live "$c" "${EXPECT[$c]}"; then s=LIVE; else s=TODO; fi; printf '%2d %s %s %s\n' $i "$c" "${EXPECT[$c]}" $s; done; exit 0
fi

mkdir -p "$LOGD"; [ -f "$TSV" ] || printf 'crate\tversion\tstart_utc\tend_utc\tresult\tattempts\n' > "$TSV"
say "CASCADE START $V ($TAG) strict (${ONLY:+only $ONLY: }$N crates, dependency order derived at the tag)"
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
