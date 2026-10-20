#!/usr/bin/env bash
# FALSIFY-5561: rc.1, rc.2 and autopilot raced together -> exactly one proceeds; nothing unmeasured is green.
# `gh` is a stub: $W/runs ("id status" lines), $W/exists ("release:<tag>" / "tag:<tag>" / "claim:<tag>"),
# $W/gh-err (every gh call fails with that text), $W/runs-fail (run list fails).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$here/../..
guard=$here/../release/release_exclusive.sh
W=$(mktemp -d); trap 'rm -rf "${W:?}"' EXIT
mkdir "$W/bin"
cat > "$W/bin/gh" <<'S'
#!/usr/bin/env bash
if [ -f "$W/gh-err" ]; then cat "$W/gh-err" >&2; exit 1; fi
nf() { echo "HTTP 404: Not Found" >&2; exit 1; }
case "$1 $2" in
  "release view") grep -qx "release:$3" "$W/exists" || nf ;;
  "run list") [ -f "$W/runs-fail" ] && exit 1
              q=; while [ $# -gt 0 ]; do [ "$1" = --jq ] && q=$2; shift; done
              awk '{ printf "{\"databaseId\":%s,\"status\":\"%s\"}\n", $1, $2 }' "$W/runs" | jq -s "." | jq -r "$q" ;;
  "api repos/o/r/commits/HEAD") echo deadbeef ;;
  "api -X") if [ "$3" = POST ]; then tag=; for a in "$@"; do case $a in ref=refs/release-claims/*) tag=${a##*/} ;; esac; done
              if grep -qx "claim:$tag" "$W/exists"; then echo "Reference already exists" >&2; exit 1; fi
              echo "claim:$tag" >> "$W/exists"
              else ref=${@: -1}; ref=${ref##*/}; grep -vx "claim:$ref" "$W/exists" > "$W/e2" || true; mv "$W/e2" "$W/exists"; fi ;;
  "api repos/o/r/git/ref/tags/"*) grep -qx "tag:${2##*/}" "$W/exists" || nf ;;
esac
S
chmod +x "$W/bin/gh"; export W PATH="$W/bin:$PATH" GITHUB_REPOSITORY=o/r
fail=0
ok() { "$@" || { fail=1; echo "FAIL (want proceed): $*"; }; }
no() { if "$@"; then fail=1; echo "FAIL (want refuse): $*"; fi; }
run() { bash "$guard" "$@" > /dev/null 2>&1; }
reset() { : > "$W/exists"; : > "$W/runs"; rm -f "${W:?}/gh-err" "${W:?}/runs-fail"; }

reset
# rc.1 (100) and rc.2 (101) in flight, then autopilot: only 100 may proceed (and claims).
printf '100 in_progress\n101 queued\n' > "$W/runs"
ok run v0.71.0 100
no run v0.71.0 101
no run v0.71.0
# Every non-completed status blocks, pending and waiting included; completed does not.
for st in queued in_progress waiting requested pending; do
  reset; printf '5 %s\n' "$st" > "$W/runs"; no run v0.71.0 9
done
reset; printf '5 completed\n' > "$W/runs"; ok run v0.71.0 9
# Lone run and an idle autopilot proceed.
reset; printf '100 in_progress\n' > "$W/runs"; ok run v0.71.0 100
reset; ok run v0.71.0
# Existing draft/release or tag refuses everyone.
reset; echo release:v0.71.0 > "$W/exists"; no run v0.71.0 100; no run v0.71.0
reset; echo tag:v0.71.0 > "$W/exists"; no run v0.71.0 100
# The claim: a second caller that passed the checks loses at the atomic ref; --release frees it.
reset; ok run v0.71.0; no run v0.71.0
bash "$guard" --release v0.71.0; ok run v0.71.0
# Unmeasured is not green: exit 2 for a failed run list and for a non-404 gh error.
reset; touch "$W/runs-fail"
set +e; bash "$guard" v0.71.0 100 > /dev/null 2>&1; rc=$?; set -e
[ "$rc" = 2 ] || { fail=1; echo "FAIL: unreadable run list gave rc=$rc, want 2"; }
reset; echo "HTTP 502: Bad Gateway" > "$W/gh-err"
set +e; bash "$guard" v0.71.0 100 > /dev/null 2>&1; rc=$?; set -e
[ "$rc" = 2 ] || { fail=1; echo "FAIL: gh 502 gave rc=$rc, want 2"; }
# Wiring: autopilot guards before it cuts the tag; release.yml plan calls the guard.
a=$(grep -n 'release_exclusive.sh' "$root/scripts/release/autopilot.sh" | head -1 | cut -d: -f1)
c=$(grep -n 'cut_tag "\$V"' "$root/scripts/release/autopilot.sh" | cut -d: -f1 | awk -v a="$a" '$1 > a { print; exit }')
{ [ -n "$a" ] && [ -n "$c" ] && [ "$a" -lt "$c" ]; } || { fail=1; echo "FAIL: autopilot guard (line ${a:-none}) is not before cut_tag (${c:-none})"; }
grep -q 'release_exclusive.sh "\$tag"' "$root/.github/workflows/release.yml" || { fail=1; echo "FAIL: release.yml plan does not call the guard"; }
grep -q 'release_exclusive.sh --release' "$root/.github/workflows/release.yml" || { fail=1; echo "FAIL: release.yml summary does not free the claim"; }
[ "$fail" = 0 ] && echo "release_exclusive_test: PASS"
exit "$fail"
