#!/usr/bin/env bash
#
# check_surface_audit_cites.sh — verify the `evidence_path` cites of the dogfood
# surface ledger point at real code (#7370).
#
# Before this, a cite could be shifted, point at the wrong symbol or run past the
# end of the file and every dogfood check still exited 0.
#
#   C1 parse    a quoted CSV field holding commas is ONE field (row 76 used to
#               shift every later column, so a whole file went unchecked)
#   C2 exists   the cited file exists
#   C3 range    every cited line N / range N-M satisfies 1 <= N <= M <= lines
#   C4 symbol   for a row whose cite CHANGED against the comparand (same
#               binary+feature, different evidence_path), the normalised last
#               word of the feature (parenthesised tail and :params dropped;
#               letters and digits only, case-folded) must occur on the FIRST
#               cited line, so a shift by one line is RED. Unchanged legacy rows
#               and brand-new rows are held to C2/C3 only: about a sixth of the
#               old ledger cites the line of a field or attribute, not of the
#               name, and a symbol rule over them would be red on main on day one.
#
# Usage: check_surface_audit_cites.sh [--root DIR] [--base REF]
#   --root   repo root holding docs/audits/surface_audit.csv (default: this repo)
#   --base   ref whose ledger is the comparand for C4 (default: none = C2/C3 only)
# Exit 0 clean, 1 findings (printed as `FAIL: ...`), 2 could not run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) ROOT="${2:?--root needs a dir}"; shift 2 ;;
    --base) BASE="${2:?--base needs a ref}"; shift 2 ;;
    *) printf 'usage: %s [--root DIR] [--base REF]\n' "$0" >&2; exit 2 ;;
  esac
done
LEDGER="docs/audits/surface_audit.csv"
[ -f "$ROOT/$LEDGER" ] || { printf 'FAIL: cannot read %s\n' "$ROOT/$LEDGER" >&2; exit 2; }

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
HAVE_BASE=0
if [ -n "$BASE" ] && git -C "$ROOT" show "${BASE}:${LEDGER}" > "$TMP" 2>/dev/null; then
  HAVE_BASE=1
fi
# awk needs a non-empty first file for the two-file NR==FNR idiom.
[ -s "$TMP" ] || printf 'header\n' > "$TMP"

rc=0
awk -v root="$ROOT" -v have_base="$HAVE_BASE" '
function csv(line, out,    n, i, c, q, cur) {
  n = 0; cur = ""; q = 0
  for (i = 1; i <= length(line); i++) {
    c = substr(line, i, 1)
    if (q) {
      if (c == "\"" && substr(line, i + 1, 1) == "\"") { cur = cur "\""; i++ }
      else if (c == "\"") q = 0
      else cur = cur c
    } else if (c == "\"") q = 1
    else if (c == ",") { out[++n] = cur; cur = "" }
    else cur = cur c
  }
  out[++n] = cur
  return n
}
function norm(s) { s = tolower(s); gsub(/[^a-z0-9]/, "", s); return s }
function symbol(feat,    t, n, a, i) {
  sub(/[ ]*\(.*$/, "", feat)
  n = split(feat, a, /[ \/]+/)
  for (i = n; i >= 1; i--) {
    t = a[i]
    if (t ~ /^[:{<-]/) continue
    t = norm(t)
    if (length(t) >= 3) return t
  }
  return ""
}
function bad(msg) { print "FAIL: ledger line " FNR " (" f[1] " / " f[2] "): " msg; rc = 1 }
NR == FNR {
  if (FNR > 1 && csv($0, g) == 10) base[g[1] SUBSEP g[2]] = g[9]
  next
}
FNR == 1 { next }
{
  delete f
  nf = csv($0, f)
  if (nf != 10) { bad("expected 10 CSV fields, read " nf); next }
  ev = f[9]
  if (!match(ev, /^[^ :]+\.(rs|toml|yml|yaml|md|sh|py)/)) next
  file = substr(ev, 1, RLENGTH); rest = substr(ev, RLENGTH + 1); sub(/ .*$/, "", rest)
  path = root "/" file
  if (!(file in lines)) {
    if ((getline probe < path) < 0) lines[file] = -1
    else {
      n = 1; txt[file, 1] = probe
      while ((getline probe < path) > 0) { n++; txt[file, n] = probe }
      lines[file] = n
    }
    close(path)
  }
  if (lines[file] < 0) { bad("cited file " file " does not exist"); next }
  if (rest == "") next
  if (rest !~ /^:[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$/) { bad("malformed cite " ev); next }
  np = split(substr(rest, 2), parts, ",")
  for (i = 1; i <= np; i++) {
    m = split(parts[i], se, "-"); lo = se[1] + 0; hi = (m == 2 ? se[2] + 0 : lo)
    if (lo < 1 || lo > hi || hi > lines[file]) {
      bad("cite " file ":" parts[i] " is outside " file " (" lines[file] " lines)"); next
    }
  }
  if (!have_base) next
  key = f[1] SUBSEP f[2]
  if (!(key in base) || base[key] == ev) next
  split(substr(rest, 2), fp, /[,-]/); first = fp[1] + 0
  sym = symbol(f[2])
  if (sym == "") next
  if (index(norm(txt[file, first]), sym) == 0)
    bad("cite " file ":" first " does not name \"" sym "\" (line reads: " substr(txt[file, first], 1, 60) ")")
}
BEGIN { rc = 0 }
END { exit rc }
' "$TMP" "$ROOT/$LEDGER" || rc=$?

if [ "$rc" -eq 0 ]; then
  printf 'OK: surface_audit.csv evidence_path cites verified (exists, in range%s)\n' \
    "$([ "$HAVE_BASE" -eq 1 ] && printf ', symbol on changed rows')"
fi
exit "$rc"
