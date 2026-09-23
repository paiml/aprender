#!/usr/bin/env bash
# check_no_rust_truncation.sh -- #3916: no NEW Rust truncation of a value read later.
#
# #3904 built this guard for Python's `[:N]`. Rust truncates with `.chars().take(N)`, and
# #3904 cannot see any of it -- found when a `.chars().take(100)` in the golden gate's own
# failure reason cost a diagnosis during #3914. Same rule, same key scheme, same amnesty
# shape; a different idiom and its own measured scope.
#
# THE PATTERN'S CASE TABLE RUNS ON EVERY INVOCATION, not only under --self-test. A net over
# 584 raw `.take(N)` hits is the kind that rots silently, and the table is the only thing
# that can tell a correct net from one that stopped selecting.
#
#   scripts/lib/rust_truncation_scan.py     the scan, and why the scope is what it is
#   scripts/lib/rust_truncation_cases.txt   must-match / must-not-match, every row REAL
#   scripts/rust_truncation_baseline.txt    the enumeration, shrink-only
#
# Usage:
#   bash scripts/check_no_rust_truncation.sh              # enforce
#   bash scripts/check_no_rust_truncation.sh --list       # print the scan
#   bash scripts/check_no_rust_truncation.sh --self-test  # the guard's own case table
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCAN="$ROOT/scripts/lib/rust_truncation_scan.py"
CASES="$ROOT/scripts/lib/rust_truncation_cases.txt"
BASELINE="${BASELINE_PATH:-"$ROOT/scripts/rust_truncation_baseline.txt"}"

enforce() { # enforce <root> <baseline> <cases>
  python3 - "$1" "$2" "$SCAN" "$3" <<'PY'
import hashlib, importlib.util, os, re, sys

root, baseline, scan_path, cases_path = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("rust_truncation_scan", scan_path)
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
fail = []


def _loud(x, n):
    """The guard's report is read later too. #3904's guard shipped truncating the offending
    line at 100 chars -- silently, in the one string a human reads to act on it -- and its
    own scan caught that. This guard was copied from that shape and REINTRODUCED the same
    defect in five places; #3904's scan caught it again. Say how much was dropped."""
    x = str(x)
    return x if len(x) <= n else "%s ... and %d more chars" % (x[:n], len(x) - n)


# THE PATTERN'S OWN TABLE, first and on every run. A guard whose selector has drifted
# reports a clean tree for the wrong reason, and nothing downstream can tell.
tbl_rows = 0
for raw in open(cases_path):
    if raw.startswith("#") or not raw.strip():
        continue
    verdict, line = raw.rstrip("\n").split("\t", 1)
    tbl_rows += 1
    if T.selects(line) != (verdict == "MATCH"):
        fail.append(("table", "want %s, got %s: %s" % (verdict, T.selects(line), _loud(line.strip(), 96))))
if tbl_rows < 20:
    fail.append(("table", "only %d case row(s) -- the table cannot discriminate this net" % tbl_rows))

rows = T.scan(root)

# ANTI-VACUITY. check_package_includes.sh is the standing proof of the alternative: it
# greps include!( in a two-file facade holding zero, prints "OK: All 0 ... are included",
# cannot fail, and is still wired into tier3. A guard must first be able to say NO.
if len(rows) < 10:
    print("FAIL no_rust_truncation: the scan found %d candidate(s) -- too few to be this tree." % len(rows))
    print("     A count that low is a claim about the INSTRUMENT, not about the code.")
    raise SystemExit(1)
if not os.path.exists(baseline) or os.path.getsize(baseline) == 0:
    print("FAIL no_rust_truncation: baseline %s missing or empty -- every row would read as new" % baseline)
    raise SystemExit(1)

base = {}
for ln in open(baseline):
    if ln.startswith("#") or not ln.strip():
        continue
    parts = ln.rstrip("\n").split("\t")
    if len(parts) < 3:
        fail.append(("malformed", "row is not <class>TAB<key>TAB<text>: %s" % _loud(repr(ln), 70))); continue
    base[parts[1]] = (parts[0], parts[2])

# TAMPER: the key hashes the text, so a row edited to look innocent breaks its own key.
for key, (cls, text) in sorted(base.items()):
    bits = key.split("|")
    if len(bits) != 3:
        fail.append(("malformed", "key is not <file>|<hash8>|<idx>: %s" % key)); continue
    want = hashlib.sha1(text.encode()).hexdigest()[:8]
    if bits[1] != want:
        fail.append(("tampered", "%s -- recorded text hashes to %s, not the key's %s" % (key, want, bits[1])))

# CLASSES ARE CHECKED, never taken on trust. An unverified class is an exemption keyword.
COUNTED = re.compile(r"more chars|and \{|\+\{len|\{\} more|more\]")
MARKED = re.compile(r"\.\.\.")
for key, (cls, text) in sorted(base.items()):
    if cls == "loud" and not COUNTED.search(text):
        fail.append(("not-loud", "%s is classed `loud` but never says how much it dropped:\n            %s" % (key, _loud(text, 96))))
    if cls == "marked" and not MARKED.search(text):
        fail.append(("not-marked", "%s is classed `marked` but appends no ellipsis:\n            %s" % (key, _loud(text, 96))))

scanned = {k: (loc, text) for k, loc, text in rows}
for key, loc, text in rows:
    if key not in base:
        fail.append(("new", "%s\n            %s" % (loc, _loud(text, 96))))

# STALE, with #3880's distinction: a row whose FILE is gone is NOT APPLICABLE; a row whose
# file is present but whose truncation is gone is stale and must be pruned, or the baseline
# stops shrinking and a future truncation hides behind a row excusing nothing.
for key in sorted(base):
    path = key.split("|")[0]
    if not os.path.exists(os.path.join(root, path)):
        continue
    if key not in scanned:
        fail.append(("stale", key))

if not fail:
    disp = sum(1 for c, _ in base.values() if c == "display")
    print("PASS no_rust_truncation: %d candidate(s), all baselined; %d still in the display class; "
          "pattern table %d/%d (#3916)" % (len(rows), disp, tbl_rows, tbl_rows))
    raise SystemExit(0)

head = {
 "table":     "the pattern's own case table no longer holds -- the SELECTOR has drifted, so a\n     clean tree here would mean nothing:",
 "new":       "a NEW Rust truncation of a value a human reads later -- say how much was dropped,\n     or do not truncate. If it is an id prefix or dies immediately, baseline it WITH its class:",
 "stale":     "baseline row(s) whose truncation is GONE -- prune them:",
 "not-loud":  "row(s) claiming `loud` without a count:",
 "not-marked":"row(s) claiming `marked` without an ellipsis:",
 "tampered":  "baseline row(s) whose text no longer matches its own key:",
 "malformed": "baseline row(s) that do not parse:",
}
for kind in ("table", "new", "stale", "not-loud", "not-marked", "tampered", "malformed"):
    items = [m for k, m in fail if k == kind]
    if items:
        print("FAIL no_rust_truncation: " + head[kind])
        for m in items:
            print("        " + m)
raise SystemExit(1)
PY
}

case "${1:-}" in
  --list) python3 "$SCAN" "$ROOT"; exit 0 ;;
  --self-test) ;;
  "") enforce "$ROOT" "$BASELINE" "$CASES"; exit $? ;;
  *) echo "unknown arg: ${1}" >&2; exit 2 ;;
esac

TD=$(mktemp -d) || exit 2
trap 'rm -rf "${TD:?}"' EXIT INT TERM
n=0; bad=0
row() { n=$((n + 1)); local want=$1 label=$2 r=$3 b=$4 c=$5 rc=0
  enforce "$r" "$b" "$c" > "$TD/out.$n" 2>&1 || rc=$?
  if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
  else printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"; sed 's/^/        /' "$TD/out.$n"; bad=1; fi
}
says() { grep -q -- "$2" "$TD/out.$1" || { printf 'FAIL  row %-2s did not name it: %s\n' "$1" "$2"; bad=1; }; }

row 0 "the real tree, baselined" "$ROOT" "$BASELINE" "$CASES"

cp -r "$ROOT/crates" "$TD/crates" 2>/dev/null; mkdir -p "$TD/scripts"
grep -v '^#' "$BASELINE" > "$TD/rows.tsv"; { echo '# hdr'; cat "$TD/rows.tsv"; } > "$TD/plain.txt"

# A NEW truncation must be refused. Composed so this file never matches its own scan.
printf 'let preview: String = reason.chars()%s(120).collect();\n' '.take' > "$TD/crates/planted_new_3916.rs"
row 1 "a NEW display truncation: refused" "$TD" "$TD/plain.txt" "$CASES"
says 2 "planted_new_3916"
rm -f "$TD/crates/planted_new_3916.rs"

# A DRIFTED SELECTOR must be refused even when the tree is clean.
sed 's/^SKIP\tfor (tok, logit) in indexed/MATCH\tfor (tok, logit) in indexed/' "$CASES" > "$TD/cases_bad.txt"
row 1 "the pattern table no longer holds: refused" "$TD" "$TD/plain.txt" "$TD/cases_bad.txt"
says 3 "SELECTOR has drifted"

# `marked` must be earned.
python3 - "$TD/rows.tsv" "$TD/fakemark.txt" <<'PY'
import sys, hashlib
out, done = [], False
for ln in open(sys.argv[1]):
    c, k, t = ln.rstrip("\n").split("\t", 2)
    if not done and c == "display":
        k = "%s|%s|%s" % (k.split("|")[0], hashlib.sha1(t.encode()).hexdigest()[:8], k.split("|")[2])
        c, done = "marked", True
    out.append("%s\t%s\t%s\n" % (c, k, t))
open(sys.argv[2], "w").writelines(["# hdr\n"] + out)
PY
row 1 "a silent row RELABELLED \`marked\`: refused" "$TD" "$TD/fakemark.txt" "$CASES"
says 4 "appends no ellipsis"

# Tamper: edit a row's text, leave its key.
python3 - "$TD/rows.tsv" "$TD/tamper.txt" <<'PY'
import sys
out, done = [], False
for ln in open(sys.argv[1]):
    c, k, t = ln.rstrip("\n").split("\t", 2)
    if not done and c == "display": t, done = "let x = 1;", True
    out.append("%s\t%s\t%s\n" % (c, k, t))
open(sys.argv[2], "w").writelines(["# hdr\n"] + out)
PY
row 1 "baseline text edited under its own key: refused" "$TD" "$TD/tamper.txt" "$CASES"
says 5 "no longer matches its own key"

# Stale, and the #3880 distinction.
printf 'display\tcrates/gone_3916.rs|deadbeef|0\tlet p: String = x.chars().take(9).collect();\n' > /dev/null
python3 - "$TD/rows.tsv" "$TD/stale.txt" <<'PY'
import sys, hashlib
t = "let p: String = x.chars().take(9).collect();"
k = "crates/gone_3916.rs|%s|0" % hashlib.sha1(t.encode()).hexdigest()[:8]
open(sys.argv[2], "w").writelines(["# hdr\n", "display\t%s\t%s\n" % (k, t)] + open(sys.argv[1]).readlines())
PY
printf 'pub fn f() {}\n' > "$TD/crates/gone_3916.rs"
row 1 "a STALE row (file present, truncation gone): refused" "$TD" "$TD/stale.txt" "$CASES"
says 6 "gone_3916"
rm -f "$TD/crates/gone_3916.rs"
row 0 "an INAPPLICABLE row (file absent) is not stale" "$TD" "$TD/stale.txt" "$CASES"

mkdir -p "$TD/empty/crates" "$TD/empty/scripts"
row 1 "a tree with no candidates: refused as an instrument failure" "$TD/empty" "$BASELINE" "$CASES"
says 8 "claim about the INSTRUMENT"

printf '%s/%s rows\n' "$((n - bad))" "$n"
[ "$bad" = 0 ] || exit 1
