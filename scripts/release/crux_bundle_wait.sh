#!/usr/bin/env bash
# crux_bundle_wait.sh -- wait for, fetch, and verify a commit's CRUX bundle from nightly-evidence.
# Shared by publish_tagged.sh and one_command.sh so both release paths wait cleanly for an in-flight
# producer (such as the host release-crux and nightly-publish units) and fail closed with a named
# reason on absent, red, tampered, or mismatched bundles.
# Usage:
#   crux_bundle_wait.sh --version <X.Y.Z> --sha <40-hex> --out <dir>
#       [--remote origin] [--branch nightly-evidence]
#       [--wait-tries N] [--poll SECONDS] [--require-cert]
set -uo pipefail
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }  # bashrs disable-line=DET002
die() { say "STOP $*"; exit 1; }
usage() { echo "STOP crux_bundle_wait: $*" >&2; exit 2; }

ver="" sha="" out="" remote=origin branch=nightly-evidence
wait_tries="${CRUX_WAIT_TRIES:-30}" poll="${CRUX_WAIT_POLL:-60}" req_cert=0
while [ $# -gt 0 ]; do
  case "$1" in
    --version) [ $# -ge 2 ] || usage "--version needs a value"; ver=$2; shift 2 ;;
    --sha) [ $# -ge 2 ] || usage "--sha needs a value"; sha=$2; shift 2 ;;
    --out) [ $# -ge 2 ] || usage "--out needs a value"; out=$2; shift 2 ;;
    --remote) [ $# -ge 2 ] || usage "--remote needs a value"; remote=$2; shift 2 ;;
    --branch) [ $# -ge 2 ] || usage "--branch needs a value"; branch=$2; shift 2 ;;
    --wait-tries) [ $# -ge 2 ] || usage "--wait-tries needs a value"; wait_tries=$2; shift 2 ;;
    --poll) [ $# -ge 2 ] || usage "--poll needs a value"; poll=$2; shift 2 ;;
    --require-cert) req_cert=1; shift ;;
    *) usage "unknown argument: $1" ;;
  esac
done

[[ $ver =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || usage "bad version '$ver'"
[[ $sha =~ ^[0-9a-f]{40}$ ]] || usage "bad sha '$sha'"
[ -n "$out" ] || usage "--out is required"
[[ $wait_tries =~ ^[0-9]+$ ]] || usage "bad --wait-tries '$wait_tries'"
[[ $poll =~ ^[0-9]+$ ]] || usage "bad --poll '$poll'"
case "$out" in *..*) die "unsafe output path '$out'" ;; esac

if [ "$req_cert" = 1 ]; then
  [ -s "evidence/crux/$ver/prompt-certification.json" ] \
    || die "commit $sha carries no evidence/crux/$ver/prompt-certification.json"
fi

b="models-crux/$sha"
git fetch -q "$remote" "+refs/heads/$branch:refs/remotes/$remote/$branch" \
  || die "cannot fetch $branch from $remote"

try=0
while ! git cat-file -e "$remote/$branch:$b/verdict" 2>/dev/null; do
  if [ "$try" -ge "$wait_tries" ]; then
    if [ "$wait_tries" -gt 0 ]; then
      die "no CRUX bundle for $sha on $branch after $wait_tries poll(s) ($b/verdict absent)"
    fi
    die "no CRUX bundle for $sha on $branch ($b/verdict absent)"
  fi
  try=$((try + 1))
  say "waiting for CRUX bundle $b on $branch (poll $try/$wait_tries, next in ${poll}s)"
  [ "$poll" = 0 ] || sleep "$poll"
  git fetch -q "$remote" "+refs/heads/$branch:refs/remotes/$remote/$branch" \
    || die "cannot fetch $branch from $remote"
done

if [ -e "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
  die "output dir $out is not empty"
fi
install -d -- "$out" || die "no scratch dir for the bundle"
listed_tree=$(git ls-tree --name-only "$remote/$branch:$b") || die "cannot list $b on $remote/$branch"
[ -n "$listed_tree" ] || die "bundle $b is empty"
while IFS= read -r f; do
  [ -n "$f" ] || continue
  case "$f" in */*|*..*) die "bundle $b: unsafe entry '$f'" ;; esac
  [ "$(git cat-file -t "$remote/$branch:$b/$f" 2>/dev/null)" = blob ] || die "bundle $b: $f is not a file"
  git show "$remote/$branch:$b/$f" > "$out/$f" || die "cannot read $b/$f"
done <<< "$listed_tree"

[ -s "$out/SHA256SUMS" ] || die "bundle $b has no SHA256SUMS"
env -C "$out" sha256sum --check --strict --quiet SHA256SUMS || die "bundle $b: SHA256SUMS does not hold"
for f in "$out"/*; do
  [ -e "$f" ] || continue
  f=${f##*/}; [ "$f" = SHA256SUMS ] && continue
  grep -q -F -x -e "$f" <(sed 's/^[0-9a-f]\{64\}  //' "$out/SHA256SUMS") \
    || die "bundle $b: $f is not covered by SHA256SUMS"
done
for want in "commit=$sha" "version=$ver" "measure=crux" "state=green"; do
  grep -q -F -x -e "$want" "$out/verdict" || die "bundle $b: verdict has no line '$want'"
done
say "CRUX bundle $b: checksums hold, verdict green for $ver at $sha"
