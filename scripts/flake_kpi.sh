#!/usr/bin/env bash
# flake_kpi.sh -- the nightly flake KPI: how many tests passed only on a retry (#6446).
#
# nextest records a test that failed, was retried and then passed as a <testcase>
# holding one or more <flakyFailure>/<flakyError> children. Each such test is a flake:
# it is quarantined and owes ONE ticket. A junit that is missing or unreadable is
# UNMEASURED (rc 2), never a count of 0.
#
#   flake_kpi.sh <junit.xml>   print "flake_kpi retried_pass=N" then one "FLAKE <classname>::<name>" per test
#   flake_kpi.sh --self-test   the case table below (committed fixtures, no cargo)
# rc: 0 measured (N may be 0) · 2 unmeasured (no file, no <testsuites>) · 1 self-test failed
set -uo pipefail

kpi() {
  local f="$1"
  [ -r "$f" ] || { echo "UNMEASURED: cannot read '$f'" >&2; return 2; }
  grep -q '<testsuites\|<testsuite ' "$f" || { echo "UNMEASURED: '$f' has no <testsuite>" >&2; return 2; }
  awk '
    function attr(s, k,   m) { if (match(s, k "=\"[^\"]*\"")) { m = substr(s, RSTART + length(k) + 2, RLENGTH - length(k) - 3); return m } return "" }
    /<testcase / { cls = attr($0, "classname"); nm = attr($0, "name"); flaky = 0 }
    /<flakyFailure|<flakyError/ { if (!flaky) { flaky = 1; n++; names[n] = cls "::" nm } }
    END { printf "flake_kpi retried_pass=%d\n", n; for (i = 1; i <= n; i++) print "FLAKE " names[i] }
  ' "$f"
}

self_test() {
  local d rc=0 out
  d="$(mktemp -d "${TMPDIR:-/tmp}/flake_kpi.XXXXXX")" || return 2
  trap 'rm -r "$d"' EXIT
  printf '%s\n' '<testsuites><testsuite name="a"><testcase name="t1" classname="c"/></testsuite></testsuites>' > "$d/clean.xml"
  printf '%s\n' '<testsuites><testsuite name="a">' \
    '<testcase name="t1" classname="c"><flakyFailure type="x"/><flakyFailure type="x"/></testcase>' \
    '<testcase name="t2" classname="c"/>' \
    '<testcase name="t3" classname="d"><flakyError type="y"/></testcase>' \
    '</testsuite></testsuites>' > "$d/flaky.xml"
  printf 'garbage\n' > "$d/bad.xml"
  out="$(kpi "$d/clean.xml")"; [ "$out" = "flake_kpi retried_pass=0" ] || { echo "FAIL clean: $out"; rc=1; }
  out="$(kpi "$d/flaky.xml")"
  [ "$(printf '%s\n' "$out" | sed -n 1p)" = "flake_kpi retried_pass=2" ] || { echo "FAIL flaky count: $out"; rc=1; }
  printf '%s\n' "$out" | grep -qx 'FLAKE c::t1' || { echo "FAIL flaky name t1"; rc=1; }
  printf '%s\n' "$out" | grep -qx 'FLAKE d::t3' || { echo "FAIL flaky name t3"; rc=1; }
  kpi "$d/bad.xml" >/dev/null 2>&1; [ $? -eq 2 ] || { echo "FAIL bad file not unmeasured"; rc=1; }
  kpi "$d/missing.xml" >/dev/null 2>&1; [ $? -eq 2 ] || { echo "FAIL missing file not unmeasured"; rc=1; }
  [ $rc -eq 0 ] && echo "flake_kpi self-test: 6 rows PASS"
  return $rc
}

case "${1:-}" in
  --self-test) self_test ;;
  "") echo "usage: flake_kpi.sh <junit.xml> | --self-test" >&2; exit 2 ;;
  *) kpi "$1" ;;
esac
