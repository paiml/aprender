#!/usr/bin/env bash
# check_comparator_pin_citations.sh -- exactly one llama.cpp comparator pin is CITED in-tree; every
# other sha is dated history or a fixture value, listed by file, or it is RED (#3741, #3563 part B).
#
# WHY. scripts/llama_pin.toml pins the comparator (build_commit, d1d3c3396 since 2026-09-15), and
# scripts/llama_bin.sh refuses to RESOLVE any other build. Nothing refused a CITATION: a committed
# receipt, threshold row or how-to could go on claiming a current baseline against the previous
# pin, and two did (evidence/parity/thresholds.yaml named it as "the pin in llama_pin.toml"
# after the file had said d1d3c3396 for a week). A bump is a decision with a named cost -- every
# ratio measured against the old pin becomes incomparable -- and the cost is visible only if every
# citation of the old pin is dated history or a fixture, never a live claim.
#
# THE MECHANISM
#   llama_pin.toml `superseded_commits` names every pin ever replaced. The LEDGER,
#   scripts/comparator_pin_citations.txt, is the enumeration #3741 asked for, committed: one line per
#   file that cites a superseded pin, "<path><TAB><historical|fixture><TAB><why>".
#   R1  an unlisted file that cites a superseded pin is RED, naming file:line -- a planted receipt
#       citing an old pin is exactly this
#   R2  a ledger entry whose file no longer cites any superseded pin is RED (stale: delete the line;
#       that is how the ledger shrinks)
#   R3  a historical entry carries the date of what it records: YYYY-MM-DD in the file, in its path,
#       or in the ledger line's why
#   R4  the ledger may not GROW against origin/main (shrink-only; scripts/check_baseline_ratchets.sh
#       classifies it `set`)
#   R5  superseded_commits is declared, non-empty, and never contains build_commit
#   A citation is the commit's first 7 hex digits followed by hex (git's abbreviation floor), so
#   the 8-digit, 9-digit and 40-digit forms of the previous pin all count.
#
#   bash scripts/check_comparator_pin_citations.sh              # the tree
#   bash scripts/check_comparator_pin_citations.sh --self-test  # the case table, a mutant per row
# exit 0 green; 1 RED; 2 ENV (nothing was judged).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
PIN_REL="scripts/llama_pin.toml"
LEDGER_REL="scripts/comparator_pin_citations.txt"

env_die() { printf 'ENV   %s -- nothing was judged, not a pass\n' "$*" >&2; exit 2; }
for t in git python3 grep; do command -v "$t" > /dev/null 2>&1 || env_die "no $t"; done

# pin_value FILE KEY -> the quoted value of KEY in a llama_pin.toml (the same shape llama_bin.sh reads)
pin_value() { sed -n "s/^[[:space:]]*$2[[:space:]]*=[[:space:]]*\"\\(.*\\)\"[[:space:]]*$/\\1/p" "$1" | head -n 1; }

# judge ROOT [MAIN_LEDGER_FILE|-] -> prints ok/FAIL rows; rc 0/1; rc 2 when nothing could be judged
judge() {
    local r=$1 main=${2:--} pin ledger build sup rc
    pin="$r/$PIN_REL"; ledger="$r/$LEDGER_REL"
    [ -f "$pin" ] || { printf 'FAIL  R5 no %s\n' "$PIN_REL"; return 1; }
    build=$(pin_value "$pin" build_commit); sup=$(pin_value "$pin" superseded_commits)
    [ -n "$build" ] || { printf 'FAIL  R5 %s declares no build_commit\n' "$PIN_REL"; return 1; }
    [ -n "$sup" ] || { printf 'FAIL  R5 %s declares no superseded_commits: a pin with no history is a pin that was never bumped, and this one was\n' "$PIN_REL"; return 1; }
    case " $sup " in *" $build "*) printf 'FAIL  R5 build_commit %s is listed as superseded by itself\n' "$build"; return 1 ;; esac
    [ -f "$ledger" ] || { printf 'FAIL  R1 no %s: the enumeration is the ledger\n' "$LEDGER_REL"; return 1; }
    python3 - "$r" "$ledger" "$main" "$build" "$PIN_REL" $sup <<'PY'
import os, re, subprocess, sys
root, ledger, main, build, pin_rel = sys.argv[1:6]
sup = sys.argv[6:]
pats = [re.compile(r"(?<![0-9a-f])" + re.escape(s[:7]) + r"[0-9a-f]*") for s in sup]
date_re = re.compile(r"20[0-9]{2}-[01][0-9]-[0-3][0-9]")
bad = 0
def entries(path):
    out = {}
    for n, l in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
        l = l.rstrip("\n")
        if not l.strip() or l.lstrip().startswith("#"): continue
        f = l.split("\t")
        if len(f) != 3 or f[1] not in ("historical", "fixture") or not f[2].strip():
            print(f"FAIL  ledger line {n} is not <path>TAB<historical|fixture>TAB<why>: {l!r}"); sys.exit(1)
        out[f[0]] = (f[1], f[2])
    return out
led = entries(ledger)
tracked = subprocess.run(["git", "-C", root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"], capture_output=True).stdout.decode(errors="replace").split("\0")
tracked = sorted({t for t in tracked if t})
if len(tracked) < 1: print("FAIL  the tracked universe is empty"); sys.exit(1)
citing = {}
for rel in tracked:
    if rel in (os.path.relpath(ledger, root), pin_rel) or rel.startswith("docs/roadmaps/") or rel.startswith("docs/audits/quorum-"):
        continue  # the ledger and the pin file declare the old pins; fragments and quorum records quote issues, which name them
    p = os.path.join(root, rel)
    try:
        with open(p, "rb") as h:
            raw = h.read()
    except OSError:
        continue
    if b"\0" in raw[:4096]: continue
    text = raw.decode("utf-8", errors="replace")
    if not any(s[:7] in text for s in sup): continue
    lines = []
    for i, line in enumerate(text.splitlines(), 1):
        if any(pt.search(line) for pt in pats): lines.append(i)
    if lines: citing[rel] = lines
for rel, lines in sorted(citing.items()):
    if rel not in led:
        print(f"FAIL  R1 {rel}:{lines[0]} cites a superseded comparator pin and is not on {os.path.relpath(ledger, root)} "
              f"({len(lines)} line(s)). A live claim against a superseded comparator is the defect; dated history or a fixture value is listed by file, with why.")
        bad = 1
for rel, (cls, why) in sorted(led.items()):
    if rel not in citing:
        print(f"FAIL  R2 stale ledger entry {rel} ({cls}): the file no longer cites a superseded pin (or is not tracked) -- delete the line"); bad = 1
        continue
    if cls == "historical":
        text = open(os.path.join(root, rel), encoding="utf-8", errors="replace").read()
        if not (date_re.search(text) or date_re.search(rel) or date_re.search(why)):
            print(f"FAIL  R3 {rel} is ledgered as historical but neither it, its path nor its ledger line carries a date (YYYY-MM-DD)"); bad = 1
if main != "-":
    try:
        mled = entries(main)
    except SystemExit:
        mled = None
    if mled is not None and len(led) > len(mled):
        print(f"FAIL  R4 the ledger GREW against origin/main ({len(mled)} -> {len(led)}): it may only shrink"); bad = 1
if not bad:
    print(f"ok    build_commit {build}; superseded {' '.join(sup)}; {len(citing)} file(s) cite a superseded pin, every one ledgered "
          f"({sum(1 for c,_ in led.values() if c=='historical')} historical, {sum(1 for c,_ in led.values() if c=='fixture')} fixture); no stale entry")
sys.exit(bad)
PY
}

# ---- the case table --------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    TMP=$(mktemp -d) || exit 2
    cleanup() { case "${TMP:-}" in ''|/) return 0 ;; *) [ -d "$TMP" ] && rm -rf -- "$TMP" ;; esac; }
    trap cleanup EXIT
    export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_AUTHOR_NAME=fx GIT_AUTHOR_EMAIL=fx@x GIT_COMMITTER_NAME=fx GIT_COMMITTER_EMAIL=fx@x
    fails=0; rows=0
    # the superseded shas, BUILT so this file cites none of them (its own R1 would read it otherwise)
    OLD="39173""bcac"; OLD2="60b06""ab9a"; CUR="d1d3c""3396"
    # fixture NAME -> $TMP/NAME: a tracked tree with the pin file, a dated historical record, a fixture,
    # and the ledger listing both
    fixture() {
        local d="$TMP/$1"
        mkdir -p "$d/scripts" "$d/evidence/parity" "$d/tests/fx" || return 2
        printf '[comparator]\nbuild_commit = "%s"\nsuperseded_commits = "%s %s"\n' "$CUR" "$OLD" "$OLD2" > "$d/scripts/llama_pin.toml"
        printf 'measured 2026-08-24 against llama.cpp %s: 0.59x\n' "$OLD" > "$d/evidence/parity/old-ratio.md"
        printf '{"comparator_sha": "%s0123456789abcdef"}\n' "$OLD" > "$d/tests/fx/case.json"
        printf 'evidence/parity/old-ratio.md\thistorical\ta ratio measured before the 2026-09-15 bump, dated in the file\ntests/fx/case.json\tfixture\ta sample value in a self-test\n' > "$d/scripts/comparator_pin_citations.txt"
        git -C "$d" init -q && git -C "$d" add -A && git -C "$d" commit -q -m fx || return 2
    }
    row() { rows=$((rows + 1)); if [ "$2" = 0 ]; then printf 'ok    %s\n' "$1"; elif [ "$2" = 2 ]; then printf 'ENV   %s\n' "$1"; exit 2; else printf 'FAIL  %s: %s\n' "$1" "$3"; fails=$((fails + 1)); fi; }
    expect() { # NAME FIXTURE-DIR WANT-RC NEEDLE [MAIN-LEDGER]
        local out rc; out=$(judge "$2" "${5:--}" 2>&1); rc=$?
        [ "$rc" = "$3" ] && grep -qF -- "$4" <<< "$out"; row "$1" $? "rc=$rc (want $3): $(tr '\n' '|' <<< "$out" | cut -c1-300)"
    }
    fixture c1 || exit 2
    expect 'C1 control: dated history + a fixture, both ledgered -> green' "$TMP/c1" 0 'every one ledgered'
    fixture c2 && printf '{"comparator": "llama.cpp %s", "ratio": 0.6}\n' "$OLD" > "$TMP/c2/evidence/parity/new-receipt.json" && git -C "$TMP/c2" add -A
    expect 'C2 a PLANTED receipt citing the old pin, unlisted -> R1 RED naming it' "$TMP/c2" 1 'R1 evidence/parity/new-receipt.json:1 cites a superseded'
    fixture c3 && printf 'comparator %s (8 digits)\n' "${OLD%?}" > "$TMP/c3/evidence/parity/abbrev.md" && git -C "$TMP/c3" add -A
    expect 'C3 an 8-digit abbreviation of the old pin is a citation -> R1' "$TMP/c3" 1 'R1 evidence/parity/abbrev.md:1'
    fixture c4 && printf 'nothing here\n' > "$TMP/c4/evidence/parity/old-ratio.md"
    expect 'C4 a ledger entry whose file no longer cites -> R2 stale' "$TMP/c4" 1 'R2 stale ledger entry evidence/parity/old-ratio.md'
    fixture c5 && printf 'against llama.cpp %s: 0.59x (no date anywhere)\n' "$OLD" > "$TMP/c5/evidence/parity/old-ratio.md" \
        && printf 'evidence/parity/old-ratio.md\thistorical\tundated\ntests/fx/case.json\tfixture\ta sample value\n' > "$TMP/c5/scripts/comparator_pin_citations.txt"
    expect 'C5 a historical entry with no date in file, path or ledger -> R3' "$TMP/c5" 1 'R3 evidence/parity/old-ratio.md is ledgered as historical'
    fixture c5b && printf 'against llama.cpp %s: 0.59x\n' "$OLD" > "$TMP/c5b/evidence/parity/old-ratio.md" \
        && printf 'evidence/parity/old-ratio.md\thistorical\tmeasured 2026-08-24 (dated here)\ntests/fx/case.json\tfixture\ta sample value\n' > "$TMP/c5b/scripts/comparator_pin_citations.txt"
    expect 'C5b the date on the ledger line satisfies R3' "$TMP/c5b" 0 'every one ledgered'
    fixture c6 && printf 'evidence/parity/old-ratio.md\thistorical\tdated 2026-08-24\n' > "$TMP/c6/main-ledger.txt"
    expect 'C6 a ledger that GREW against main (1 -> 2) -> R4' "$TMP/c6" 1 'R4 the ledger GREW against origin/main (1 -> 2)' "$TMP/c6/main-ledger.txt"
    fixture c7 && printf '[comparator]\nbuild_commit = "%s"\nsuperseded_commits = "%s %s"\n' "$CUR" "$OLD" "$CUR" > "$TMP/c7/scripts/llama_pin.toml"
    expect 'C7 build_commit listed as superseded -> R5' "$TMP/c7" 1 "R5 build_commit $CUR is listed as superseded by itself"
    fixture c8 && printf '[comparator]\nbuild_commit = "%s"\n' "$CUR" > "$TMP/c8/scripts/llama_pin.toml"
    expect 'C8 no superseded_commits declared -> R5' "$TMP/c8" 1 'declares no superseded_commits'
    fixture c9 && printf 'evidence/parity/old-ratio.md\thistorical\n' > "$TMP/c9/scripts/comparator_pin_citations.txt"
    expect 'C9 a malformed ledger line -> RED, never a pass' "$TMP/c9" 1 'is not <path>TAB<historical|fixture>TAB<why>'
    fixture c10 && printf 'cites %s only, the pin of record\n' "$CUR" > "$TMP/c10/evidence/parity/current.md" && git -C "$TMP/c10" add -A
    expect 'C10 a citation of the CURRENT pin needs no ledger -> green' "$TMP/c10" 0 'every one ledgered'

    # MUTANTS: each must turn its row RED. The judge is re-sourced from a mutated copy of this file.
    mutant() { # NAME OLD NEW WANT-RC NEEDLE FIXTURE-DIR [MAIN-LEDGER] -- killed iff the row's expectation no longer holds
        local name=$1 old=$2 new=$3 want=$4 needle=$5; shift 5
        python3 - "$0" "$TMP/mut-$name.sh" "$old" "$new" <<'PY' || { printf 'FAIL  mutant %s: anchor not found exactly once\n' "$name"; fails=$((fails + 1)); return; }
import sys
src, dst, old, new = sys.argv[1:5]
s = open(src).read()
# judge() precedes the self-test, whose mutant calls quote the same anchors: mutate the FIRST occurrence only
if s.count(old) < 1: sys.exit(1)
open(dst, "w").write(s.replace(old, new, 1))
PY
        local out rc
        out=$( . "$TMP/mut-$name.sh" --source-only; judge "$@" 2>&1 ); rc=$?
        rows=$((rows + 1))
        if [ "$rc" = "$want" ] && grep -qF -- "$needle" <<< "$out"; then printf 'FAIL  mutant %s SURVIVED (the row still holds)\n' "$name"; fails=$((fails + 1))
        else printf 'ok    mutant %s killed\n' "$name"; fi
    }
    mutant drop-r1 '    if rel not in led:
        print(f"FAIL  R1' '    if False:
        print(f"FAIL  R1' 1 'R1 evidence/parity/new-receipt.json:1' "$TMP/c2"
    mutant drop-r2 '    if rel not in citing:
        print(f"FAIL  R2' '    if False:
        print(f"FAIL  R2' 1 'R2 stale ledger entry' "$TMP/c4"
    mutant drop-r3 '        if not (date_re.search(text) or date_re.search(rel) or date_re.search(why)):' '        if False:' 1 'R3 evidence/parity/old-ratio.md' "$TMP/c5"
    mutant drop-r4 '    if mled is not None and len(led) > len(mled):' '    if False:' 1 'R4 the ledger GREW' "$TMP/c6" "$TMP/c6/main-ledger.txt"
    mutant full-sha-only 'pats = [re.compile(r"(?<![0-9a-f])" + re.escape(s[:7]) + r"[0-9a-f]*") for s in sup]' 'pats = [re.compile(r"(?<![0-9a-f])" + re.escape(s) + r"(?![0-9a-f])") for s in sup]' 1 'R1 evidence/parity/abbrev.md:1' "$TMP/c3"

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED (%s of %s)\n' "$fails" "$rows"; exit 1; }
    printf '\nSELF-TEST PASSED (%s rows and mutants)\n' "$rows"
    exit 0
fi
# a mutated copy is sourced by the self-test for its judge() only
[ "${1:-}" = "--source-only" ] && return 0 2> /dev/null

# ---- the tree --------------------------------------------------------------------------------
printf '=== one comparator pin is cited; every other is dated history or a fixture, by file (#3741) ===\n'
MAINL=$(mktemp) || exit 2
if git -C "$ROOT" cat-file -e "origin/main:$LEDGER_REL" 2> /dev/null; then
    git -C "$ROOT" show "origin/main:$LEDGER_REL" > "$MAINL"
else
    printf '!     BOOTSTRAP: %s is not on origin/main yet, so the shrink-only floor has no comparand this run\n' "$LEDGER_REL"
    rm -f -- "$MAINL"; MAINL=-
fi
judge "$ROOT" "$MAINL"; rc=$?
[ "$MAINL" = - ] || rm -f -- "$MAINL"
[ "$rc" -eq 0 ] && printf 'PASS\n' || printf 'FAIL  (rc=%s)\n' "$rc"
exit "$rc"
