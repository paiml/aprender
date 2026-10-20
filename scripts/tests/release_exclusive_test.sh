#!/usr/bin/env bash
# FALSIFY-5561: rc.1, rc.2 and autopilot raced together -> exactly one proceeds.
# `gh` is a stub reading $W/runs (active release.yml run ids) and $W/exists (tags/releases that exist).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
guard=$here/../release/release_exclusive.sh
W=$(mktemp -d); trap 'rm -rf "${W:?}"' EXIT
mkdir "$W/bin"
cat > "$W/bin/gh" <<'S'
#!/usr/bin/env bash
case "$1 $2" in
  "release view") grep -qx "release:$3" "$W/exists" ;;
  "api repos/"*) grep -qx "tag:${2##*/}" "$W/exists" ;;
  "run list") st=; while [ $# -gt 0 ]; do [ "$1" = --status ] && st=$2; shift; done
              [ -f "$W/runs-fail" ] && exit 1
              awk -v s="$st" '$2 == s { print $1 }' "$W/runs" ;;
esac
S
chmod +x "$W/bin/gh"; export W PATH="$W/bin:$PATH" GITHUB_REPOSITORY=o/r
: > "$W/exists"; fail=0
ok() { "$@" || { fail=1; echo "FAIL (want proceed): $*"; }; }
no() { if "$@"; then fail=1; echo "FAIL (want refuse): $*"; fi; }
run() { bash "$guard" "$@" > /dev/null 2>&1; }   # prints nothing; the exit code is the verdict

# rc.1 (run 100) and rc.2 (run 101) in flight, then autopilot: only run 100 may proceed.
printf '100 in_progress\n101 queued\n' > "$W/runs"
ok run v0.71.0 100
no run v0.71.0 101
no run v0.71.0
# Lone run proceeds; autopilot proceeds when nothing is active.
printf '100 in_progress\n' > "$W/runs"; ok run v0.71.0 100
: > "$W/runs"; ok run v0.71.0
# An existing draft/release or tag refuses everyone.
echo release:v0.71.0 > "$W/exists"; no run v0.71.0 100; no run v0.71.0
echo tag:v0.71.0 > "$W/exists"; no run v0.71.0 100
# A run list that cannot be read is not a green: exit 2.
: > "$W/exists"; touch "$W/runs-fail"
set +e; bash "$guard" v0.71.0 100 > /dev/null 2>&1; rc=$?; set -e
[ "$rc" = 2 ] || { fail=1; echo "FAIL: unreadable run list gave rc=$rc, want 2"; }
[ "$fail" = 0 ] && echo "release_exclusive_test: PASS"
exit "$fail"
