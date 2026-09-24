#!/usr/bin/env bash
# check_no_silent_truncation.sh -- no NEW truncation of a value a human reads later (#3904).
#
# #3872 removed four diagnostic-truncating slices from the model ladder. Those were the four
# that had been FOUND; nobody enumerated the surface, so the 5th sat untouched in the
# ladder's own cells judge -- a cell verdict's reason cut at 80 characters on its way into
# the only line a human reads, four lines above the honest pattern. Fixing six instances
# without a guard just waits for the seventh. That is CLAUDE.md's own rule -- enumerate the
# decision surfaces, then check coverage -- left unapplied to the fix that quotes it.
#
# THE RULE IS NOT "NEVER TRUNCATE", IT IS "NEVER TRUNCATE SILENTLY".
# THE DIVIDING LINE is whether a human reads the string later, DETACHED from what produced
# it. A die() is fine; a receipt is not.
#
# The candidate set, the key scheme and the classes are documented where they live:
#   scripts/lib/truncation_scan.py         (one scan, shared by enforce and self-test)
#   scripts/silent_truncation_baseline.txt (the enumeration, shrink-only)
#
# Usage:
#   bash scripts/check_no_silent_truncation.sh              # enforce
#   bash scripts/check_no_silent_truncation.sh --list       # print the scan
#   bash scripts/check_no_silent_truncation.sh --self-test  # the case table
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCAN="$ROOT/scripts/lib/truncation_scan.py"
BASELINE="${BASELINE_PATH:-"$ROOT/scripts/silent_truncation_baseline.txt"}"

enforce() { # enforce <root> <baseline> -> 0 clean, 1 refused
  python3 - "$1" "$2" "$SCAN" <<'PY'
import hashlib, importlib.util, os, re, sys

root, baseline, scan_path = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("truncation_scan", scan_path)
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)

def _loud(s, n):
    """The guard's own output is read later too. Its first cut truncated the offending
    line at 100 chars -- silently, in the one string a human reads to decide what to
    do about it. Found by its own scan, which is the only reason this note exists."""
    s = str(s)
    return s if len(s) <= n else f"{s[:n]} ... and {len(s) - n} more chars"


rows = T.scan(root)
fail = []

# ANTI-VACUITY, before any verdict. A scan that matches nothing reports "0 violations"
# forever. check_package_includes.sh is the standing proof: it greps include!( in a
# two-file facade holding zero such directives, prints "OK: All 0 ... are included", and
# CANNOT FAIL. It is still wired into tier3. A guard must first be able to say NO.
if len(rows) < 10:
    print("FAIL no_silent_truncation: the scan found %d candidate(s) -- too few to be this tree." % len(rows))
    print("     A count that low is a claim about the INSTRUMENT, not about the code.")
    raise SystemExit(1)

if not os.path.exists(baseline) or os.path.getsize(baseline) == 0:
    print("FAIL no_silent_truncation: baseline %s missing or empty -- every row would read as new" % baseline)
    raise SystemExit(1)

base = {}
for ln in open(baseline):
    if ln.startswith("#") or not ln.strip():
        continue
    parts = ln.rstrip("\n").split("\t")
    if len(parts) < 3:
        fail.append(("malformed", "baseline row is not <class>TAB<key>TAB<text>: %r" % _loud(ln.rstrip(), 70)))
        continue
    base[parts[1]] = (parts[0], parts[2])

# TAMPER. The key hashes the text, so a row edited to look innocent no longer keys itself.
# Without this the text column is decoration and `loud` could be claimed by retyping a line.
for key, (cls, text) in sorted(base.items()):
    bits = key.split("|")
    if len(bits) != 3:
        fail.append(("malformed", "key is not <file>|<hash8>|<idx>: %s" % key)); continue
    want = hashlib.sha1(text.encode()).hexdigest()[:8]
    if bits[1] != want:
        fail.append(("tampered", "%s -- recorded text hashes to %s, not the key's %s" % (key, want, bits[1])))

# LOUD is CHECKED, never taken on trust. An exemption class nobody verifies is just a
# keyword that turns a violation green, which is the seam we keep refusing.
LOUD = re.compile(r"more chars|\.\.\. and |and \{len|more\]|\+\{len|more before|before this")
for key, (cls, text) in sorted(base.items()):
    if cls == "loud" and not LOUD.search(text):
        fail.append(("not-loud", "%s is classed `loud` but its line never says how much it dropped:\n            %s" % (key, _loud(text, 100))))

scanned = {k: (loc, text) for k, loc, text in rows}

# NEW -- the ratchet.
for key, loc, text in rows:
    if key not in base:
        fail.append(("new", "%s\n            %s" % (loc, _loud(text, 100))))

# STALE, with #3880's distinction: "nothing matched" and "what it covered now passes" are
# both zero-usage and only ONE is a defect. A row whose FILE is gone is NOT APPLICABLE; a
# row whose file is present but whose slice is gone is stale and must be pruned, or the
# baseline stops shrinking and a future truncation hides behind a row excusing nothing.
for key in sorted(base):
    path = key.split("|")[0]
    if not os.path.exists(os.path.join(root, path)):
        continue
    if key not in scanned:
        fail.append(("stale", key))

if not fail:
    disp = sum(1 for c, _ in base.values() if c == "display")
    print("PASS no_silent_truncation: %d candidate(s), all baselined; %d still in the display class (#3904)"
          % (len(rows), disp))
    raise SystemExit(0)

order = ["new", "stale", "not-loud", "tampered", "malformed"]
head = {
  "new":      "a NEW truncation of a value a human reads later -- say how much was dropped\n     (`... and N more chars`), or do not truncate. If it dies immediately, or is an id\n     prefix or a fixed-width number, add it to the baseline WITH its class:",
  "stale":    "baseline row(s) whose slice is GONE -- prune them:",
  "not-loud": "row(s) claiming the `loud` class without earning it:",
  "tampered": "baseline row(s) whose text no longer matches its own key:",
  "malformed":"baseline row(s) that do not parse:",
}
for kind in order:
    items = [m for k, m in fail if k == kind]
    if items:
        print("FAIL no_silent_truncation: " + head[kind])
        for m in items:
            print("        " + m)
raise SystemExit(1)
PY
}

case "${1:-}" in
  --list) python3 "$SCAN" "$ROOT"; exit 0 ;;
  --self-test) ;;
  "") enforce "$ROOT" "$BASELINE"; exit $? ;;
  *) echo "unknown arg: ${1}" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- the case table
TD=$(mktemp -d) || exit 2
# A plant without a trap is restored only when nothing goes wrong, and the whole point of
# a plant is that something is going wrong. `${TD:?}` rather than a `case` guard: bashrs
# tracks taint syntactically, so a case guard does not clear SEC011 (the repo's other
# guards all settled on this shape).
trap 'rm -rf "${TD:?}"' EXIT INT TERM

n=0; bad=0
row() { # row <want rc> <label> <root> <baseline>
  n=$((n + 1)); local want=$1 label=$2 r=$3 b=$4 rc=0
  enforce "$r" "$b" > "$TD/out.$n" 2>&1 || rc=$?
  if [ "$rc" = "$want" ]; then
    printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
  else
    printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"
    sed 's/^/        /' "$TD/out.$n"; bad=1
  fi
}
says() { grep -q -- "$2" "$TD/out.$1" || { printf 'FAIL  row %-2s did not name it: %s\n' "$1" "$2"; bad=1; }; }

row 0 "the real tree, baselined" "$ROOT" "$BASELINE"

cp -r "$ROOT/scripts" "$TD/scripts"
grep -v '^#' "$BASELINE" > "$TD/rows.tsv"
mk() { { echo "# hdr"; cat "$TD/rows.tsv"; } > "$1"; }   # a baseline + whatever is prepended
mk "$TD/plain.txt"

# A NEW truncation must be refused. Planted in a COPY; the tree is never mutated.
# Composed, never written literally: this file must contain ZERO matches of its own scan,
# or it would need an exemption -- and an exemption on the scanner is the one hole nothing
# else can see. (The scan also skips truncation_scan.py for the same reason.)
printf 'print(f"reason: {str(x)%s64]}")\n' '[:' > "$TD/scripts/planted_new.py"
row 1 "a NEW display truncation: refused" "$TD" "$TD/plain.txt"
says 2 "planted_new.py"
rm -f "$TD/scripts/planted_new.py"   # each row starts from a clean plant

# LINE SHIFT MUST NOT CHURN. This is the row that keeps the baseline honest: under
# file:line keying, inserting a helper renumbered every row below it into one NEW
# violation plus one STALE fix, and a reviewer facing that churn regenerates the baseline
# -- laundering any genuine truncation in the same commit.
printf '#\n#\n#\n#\n#\n#\n%s' "$(cat "$TD/scripts/lib/perf_receipt.py")" > "$TD/scripts/lib/perf_receipt.py.new"
mv "$TD/scripts/lib/perf_receipt.py.new" "$TD/scripts/lib/perf_receipt.py"
row 0 "6 lines inserted above 40+ rows: NO churn" "$TD" "$TD/plain.txt"
cp "$ROOT/scripts/lib/perf_receipt.py" "$TD/scripts/lib/perf_receipt.py"

# A baseline row whose slice is gone must be pruned...
# The key must hash its own text or the TAMPER rule fires first and row 5 never gets to
# test staleness -- a fixture that fails for the wrong reason proves nothing.
printf 'display\tscripts/gone_3904.py|%s|0\tprint(f"x")\n' 'a3d9b50f' > "$TD/stale.txt"
cat "$TD/rows.tsv" >> "$TD/stale.txt"
printf 'x = 1\n' > "$TD/scripts/gone_3904.py"
row 1 "a STALE row (file present, slice gone): refused" "$TD" "$TD/stale.txt"
says 4 "gone_3904"

# ...but a row whose FILE is absent is NOT APPLICABLE, not stale (#3880's distinction).
rm -f "$TD/scripts/gone_3904.py"
row 0 "an INAPPLICABLE row (file absent) is not stale" "$TD" "$TD/stale.txt"

# `loud` must be earned.
sed 's/^loud\(.*more chars.*\)$/loud\1/' "$TD/rows.tsv" > /dev/null
python3 - "$TD/rows.tsv" "$TD/fakeloud.txt" <<'PY'
import sys, hashlib
src, dst = sys.argv[1], sys.argv[2]
out, done = [], False
for ln in open(src):
    c, k, t = ln.rstrip("\n").split("\t", 2)
    if not done and c == "display":
        # a real silent truncation, relabelled `loud` -- key rehashed so it is NOT a tamper
        k = "%s|%s|%s" % (k.split("|")[0], hashlib.sha1(t.encode()).hexdigest()[:8], k.split("|")[2])
        c, done = "loud", True
    out.append("%s\t%s\t%s\n" % (c, k, t))
open(dst, "w").writelines(["# hdr\n"] + out)
PY
row 1 "a silent row RELABELLED \`loud\`: refused" "$TD" "$TD/fakeloud.txt"
says 6 "never says how much it dropped"

# Tamper: edit a row's text to look innocent, leave its key alone.
python3 - "$TD/rows.tsv" "$TD/tamper.txt" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
out, done = [], False
for ln in open(src):
    c, k, t = ln.rstrip("\n").split("\t", 2)
    if not done and c == "display":
        t, done = "print('nothing to see here')", True
    out.append("%s\t%s\t%s\n" % (c, k, t))
open(dst, "w").writelines(["# hdr\n"] + out)
PY
row 1 "baseline text edited under its own key: refused" "$TD" "$TD/tamper.txt"
says 7 "no longer matches its own key"

# Anti-vacuity: a scan that finds nothing must refuse.
mkdir -p "$TD/empty/scripts" "$TD/empty/crates"
row 1 "a tree with no candidates: refused as an instrument failure" "$TD/empty" "$BASELINE"
says 8 "claim about the INSTRUMENT"

# THE IDENTIFIER-PREFIX CLASS (#4046, cop ruling A'; truncation_scan.id_prefix_only).
# An id prefix of literal width 1-16 on a name that SAYS it is an id is exempt; each
# must-RED below plants the nearest thing that is not, and must still be refused.
# Composed, never literal (see row 1). Rows 9-13, appended so rows 1-8 keep their numbers.
plant() { printf '%s\n' "$1" > "$TD/scripts/planted_id.py"; }
plant "$(printf 'print(f"model {model_sha%s12]} measured")' '[:')"
row 0 "an identifier prefix (model_sha, literal width 12) is the id class, not a truncation" "$TD" "$TD/plain.txt"
plant "$(printf 'print(f"why {why_text%s12]}")' '[:')"
row 1 "the same width on a NON-id name is still a display truncation" "$TD" "$TD/plain.txt"
says 10 "planted_id.py"
plant "$(printf 'print(f"model {model_sha%sn]}")' '[:')"
row 1 "a VARIABLE-width id slice is still a truncation" "$TD" "$TD/plain.txt"
plant "$(printf 'print(f"model {model_sha%s40]}")' '[:')"
row 1 "an id slice WIDER than 16 is prose, not a prefix: still a truncation" "$TD" "$TD/plain.txt"
plant "$(printf 'print(f"{model_sha%s12]} {reason%s80]}")' '[:' '[:')"
row 1 "an id prefix BESIDE a display slice: the display slice still counts" "$TD" "$TD/plain.txt"
# THE COMPUTED-LOUD CLASS (#4046; truncation_scan.loud_on_line). Rows 14-16.
plant "$(printf 'print(s if len(s) <= n else f"{s%sn]} … and {len(s) - n} more chars")' '[:')"
row 0 "a slice beside a COMPUTED drop count on the same line is loud, not silent" "$TD" "$TD/plain.txt"
plant "$(printf 'print(f"{why_text%s80]} … and 80 more chars")' '[:')"
row 1 "a LITERAL drop count can lie: still a truncation" "$TD" "$TD/plain.txt"
plant "$(printf 'n_more = len(s) - 80  # … and n more chars\nprint(f"{why_text%s80]}")' '[:')"
row 1 "the marker on a DIFFERENT line does not make the slice loud" "$TD" "$TD/plain.txt"
rm -f "$TD/scripts/planted_id.py"

printf '%s/%s rows\n' "$((n - bad))" "$n"
[ "$bad" = 0 ] || exit 1
