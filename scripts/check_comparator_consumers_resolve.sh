#!/usr/bin/env bash
# check_comparator_consumers_resolve.sh -- every llama.cpp comparator consumer resolves through
# scripts/llama_bin.sh: no binary by bare name or literal path, no PATH probe, no list of candidate
# paths, no second resolver (#3740, #3563 part C).
#
# WHY. #3563(A) made scripts/llama_bin.sh resolve and PROVE the pinned llama.cpp on lambda and gx10
# (a version+sha line, the cmake cache, a prefix match on the commit). A proof is worth nothing to a
# consumer that goes around it. The audit on #3740 (2026-09-21) found every SHELL consumer already
# sourcing it, and three Rust consumers not: a `which llama-server` probe in `apr showcase` (#3773
# owns that file), a `which llama-cli` plus a hard-coded candidate list in the Qwen3-MoE argmax
# parity test, and a PATH default in the bench library (kept for outside users, documented as
# unverified -- cop ruling 4). On lambda, PATH's llama.cpp is a different commit whose llama-cli
# dies on an undefined symbol, so "whatever PATH has" is not a comparator.
#
# THE RULES (over the tree)
#   R1  SHELL EXEC    no llama.cpp binary in COMMAND POSITION by bare name or literal path, in
#                     scripts/**/*.sh and crates/*/scripts/*.sh. The pinned build is reached through
#                     $LLAMA_CLI / $LLAMA_SERVER / $LLAMA_BENCH, which only llama_bin.sh sets. Command
#                     position is decided by check_comparator_one_client.sh's walker (quotes, $( ),
#                     heredoc bodies, comments, continuations), EXTRACTED from that file at run time
#                     with only its predicate swapped: one walker, two predicates, so a fix to the
#                     walker reaches both guards (cop ruling 5 on #3740).
#   R1b SHELL PROBE   no `which` / `command -v` / `type -P` of a llama.cpp binary on a code line
#                     (heredoc bodies and comments are data).
#   R2  RUST PROBE    no `Command::new("which"|"command"|"type")` whose argument is a llama.cpp binary.
#   R3  RUST EXEC     no `Command::new("llama-...")`, bare or as a literal path.
#   R4  RUST LIST     no `const`/`static` `&[&str]` list holding a path to a llama.cpp binary.
#   R5  BASELINE      scripts/comparator_consumer_baseline.txt lists the KNOWN violations
#                     ("<rule><TAB><path><TAB>#<issue>"). An unlisted violation is RED; a listed entry
#                     that no longer matches a violation is RED (stale: delete it -- that is how the
#                     list shrinks); and the list may not grow against origin/main.
#
# THE RELEASE GATE. scripts/dogfood_comparator_env_tests.sh runs the env-gated Rust tests WITH the
# pinned build at pre-publish and fails on a skip (cop's condition on #3740). The case table below
# drives it with a stub resolver and a stub test runner, a mutant per row.
#
#   bash scripts/check_comparator_consumers_resolve.sh              # the tree
#   bash scripts/check_comparator_consumers_resolve.sh --self-test  # the case table
# exit 0 green; 1 RED; 2 ENV (a subject or an anchor is missing -- nothing was judged).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
ONE_CLIENT="$ROOT/scripts/check_comparator_one_client.sh"
GATE="$ROOT/scripts/dogfood_comparator_env_tests.sh"
BASELINE_REL="scripts/comparator_consumer_baseline.txt"
BINS='cli|server|bench|tokenize|completion|perplexity'

env_die() { printf 'ENV   %s -- nothing was judged, not a pass\n' "$*" >&2; exit 2; }
for f in "$ONE_CLIENT" "$GATE"; do [ -r "$f" ] || env_die "no $f"; done
for t in awk python3 git; do command -v "$t" > /dev/null 2>&1 || env_die "no $t"; done

TMP=$(mktemp -d) || exit 2
# SEC011: validate before rm -rf. An empty or '/' value must never reach it.
cleanup() { case "${TMP:-}" in ''|/) return 0 ;; *) [ -d "$TMP" ] && rm -rf -- "$TMP" ;; esac; }
trap cleanup EXIT

# ---- R1's walker: check_comparator_one_client.sh's, with the predicate swapped -------------------
# Its predicate flags llama-bench and $LLAMA_BENCH (the bench binary is banned outright there). Here
# every llama.cpp binary is banned BY NAME OR PATH, and the $LLAMA_* variables are the sanctioned way
# in, so the swapped predicate matches only the literal.
WALKER="$TMP/walker.awk"
python3 - "$ONE_CLIENT" "$WALKER" "$BINS" <<'PY' || env_die "the walker could not be extracted from check_comparator_one_client.sh (its AWK heredoc or its isbench() moved)"
import re, sys
src, out, bins = sys.argv[1:4]
s = open(src).read()
m = re.search(r"cat > \"\$AWK_WALKER\" <<'AWK'\n(.*?)\nAWK\n", s, re.S)
if not m:
    sys.exit(3)
pred = ("function isbench(w,   s) {\n    s = w\n    gsub(/\"/, \"\", s)\n    gsub(/'/, \"\", s)\n"
        "    if (s ~ /(^|\\/)llama-(" + bins + ")$/) return 1\n    return 0\n}\n")
w, n = re.subn(r"function isbench\(w,   s\) \{.*?\n\}\n", lambda _m: pred, m.group(1), count=1, flags=re.S)
if n != 1:
    sys.exit(4)
open(out, "w").write(w + "\n")
PY

# ---- the scan ------------------------------------------------------------------------------------
# universe ROOT -> repo-relative shell and Rust files, tracked UNION untracked (an untracked file gets
# no free pass), build dirs pruned
universe() {
    local r=$1
    { git -C "$r" ls-files 'scripts/*.sh' 'scripts/**/*.sh' 'crates/*/scripts/*.sh' '*.rs' 'crates/**/*.rs' 2> /dev/null || true
      find "$r" \( -name .git -o -name target -o -name 'target_*' -o -name node_modules -o -name vendor \) -prune -o \
          -type f \( -name '*.rs' -o -name '*.sh' \) -print 2> /dev/null | sed "s|^$r/||" \
          | grep -E '^(scripts/|crates/[^/]+/scripts/|.*\.rs$)' || true
    } | LC_ALL=C sort -u
}

# scan ROOT -> "<rule>\t<path>:<line>\t<text>" for every violation; rc 2 when a walker run failed
scan() {
    local r=$1 rel out rc bad=0 uni
    uni=$(mktemp) || return 2
    universe "$r" > "$uni"
    while IFS= read -r rel; do
        [ -f "$r/$rel" ] || continue
        case "$rel" in
            *.sh)
                grep -qE "llama-($BINS)" "$r/$rel" || continue
                out=$(awk -f "$WALKER" "$r/$rel" 2> "$TMP/walk.err"); rc=$?
                if [ "$rc" -ne 0 ] || [ -s "$TMP/walk.err" ]; then
                    printf 'WALKER-ERROR on %s: rc=%s %s\n' "$rel" "$rc" "$(tr '\n' ' ' < "$TMP/walk.err")" >&2; bad=1; continue
                fi
                [ -z "$out" ] || printf '%s\n' "$out" | while IFS=$'\t' read -r ln text; do printf 'R1\t%s:%s\t%s\n' "$rel" "$ln" "$text"; done ;;
        esac
    done < "$uni"
    python3 - "$r" "$BINS" "$uni" <<'PY' || bad=1
import re, sys
root, bins, uni = sys.argv[1:4]
B = r"(?:[^\"'\s]*/)?llama-(?:" + bins + r")"
probe_sh = re.compile(r"(?:^|[\s;|&(`])(?:which|command\s+-[vV]|type\s+-[pPa]+)\s+[\"']?" + B + r"\b")
probe_rs = re.compile(r"Command::new\(\s*\"(?:which|command|type)\"\s*\)[^;]{0,400}?\.args?\(\s*(?:&?\[)?\s*\"" + B + r"\"", re.S)
exec_rs = re.compile(r"Command::new\(\s*\"" + B + r"\"\s*\)")
list_rs = re.compile(r"(?:const|static)\s+\w+\s*:\s*&(?:'static\s+)?\[\s*&(?:'static\s+)?str\s*\]\s*=\s*&\[(.*?)\];", re.S)
path_lit = re.compile(r"\"[^\"]*/llama-(?:" + bins + r")\"")
heredoc = re.compile(r"<<-?\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?")
def line_of(text, pos): return text.count("\n", 0, pos) + 1
for rel in open(uni).read().split():
    p = f"{root}/{rel}"
    try:
        text = open(p, errors="replace").read()
    except OSError:
        continue
    if "llama-" not in text:
        continue
    if rel.endswith(".sh"):
        end = None
        for i, line in enumerate(text.splitlines(), 1):
            if end is not None:
                if line.strip() == end: end = None
                continue
            m = heredoc.search(line)
            code = line.split(" #", 1)[0] if not line.lstrip().startswith("#") else ""
            if code and probe_sh.search(code):
                print(f"R1b\t{rel}:{i}\t{line.strip()}")
            if m and "<<<" not in line:
                end = m.group(1)
    elif rel.endswith(".rs"):
        code = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
        for m in probe_rs.finditer(code):
            print(f"R2\t{rel}:{line_of(code, m.start())}\t{code[m.start():m.end()].splitlines()[0].strip()}")
        for m in exec_rs.finditer(code):
            print(f"R3\t{rel}:{line_of(code, m.start())}\t{m.group(0)}")
        for m in list_rs.finditer(code):
            for lit in path_lit.finditer(m.group(1)):
                print(f"R4\t{rel}:{line_of(code, m.start(1) + lit.start())}\t{lit.group(0)}")
PY
    rm -f -- "$uni"
    return $(( bad ? 2 : 0 ))
}

# judge VIOLATIONS BASELINE [MAIN_BASELINE] -> R5; prints ok/FAIL rows, rc 0/1
judge() {
    local viol=$1 base=$2 main=${3:-} rc=0
    python3 - "$viol" "$base" "$main" <<'PY' || rc=1
import sys
viol, base, main = sys.argv[1:4]
def entries(p):
    out = []
    try:
        for l in open(p):
            l = l.rstrip("\n")
            if not l.strip() or l.lstrip().startswith("#"): continue
            f = l.split("\t")
            if len(f) != 3 or not f[2].startswith("#"):
                print(f"FAIL  R5 malformed baseline line (want <rule>TAB<path>TAB#<issue>): {l!r}"); sys.exit(1)
            out.append((f[0], f[1], f[2]))
    except FileNotFoundError:
        pass
    return out
v = [l.rstrip("\n").split("\t", 2) for l in open(viol) if l.strip()]
b = entries(base)
known = {(r, p) for r, p, _ in b}
bad = 0
for r, loc, text in v:
    path = loc.rsplit(":", 1)[0]
    if (r, path) in known:
        print(f"known {r} {loc}  (baseline: {next(i for rr, pp, i in b if (rr, pp) == (r, path))})")
    else:
        print(f"FAIL  {r} {loc}: {text[:160]}"); bad = 1
live = {(r, loc.rsplit(':', 1)[0]) for r, loc, _ in v}
for r, p, i in b:
    if (r, p) not in live:
        print(f"FAIL  R5 stale baseline entry {r} {p} ({i}): no such violation any more -- delete the line"); bad = 1
if main:
    mb = entries(main)
    if len(b) > len(mb):
        print(f"FAIL  R5 the baseline GREW against origin/main ({len(mb)} -> {len(b)}): it may only shrink"); bad = 1
if not bad:
    print(f"ok    {len(v)} violation(s), every one owned by a baseline entry; {len(b)} entry(ies), none stale")
sys.exit(bad)
PY
    return "$rc"
}

# ---- the case table ------------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    fails=0; rows=0
    ok_row() { rows=$((rows + 1)); if [ "$2" = 0 ]; then printf 'ok    %s\n' "$1"; else printf 'FAIL  %s: %s\n' "$1" "$3"; fails=$((fails + 1)); fi; }
    # scan_row NAME WANT-RULE|clean RELPATH CONTENT -> one file in a fresh tree, scanned
    scan_row() {
        local name=$1 want=$2 rel=$3 content=$4 fx out
        fx="$TMP/scan-$rows"; mkdir -p "$fx/$(dirname "$rel")"; printf '%s\n' "$content" > "$fx/$rel"
        out=$(scan "$fx" 2>&1)
        if [ "$want" = clean ]; then
            [ -z "$out" ]; ok_row "$name" $? "expected clean, got: $out"
        else
            grep -q "^$want	$rel:" <<< "$out"; ok_row "$name" $? "expected $want, got: ${out:-<nothing>}"
        fi
    }
    # The probe fixtures are BUILT, so this file's own lines are not probes for its own R1b to find.
    WH="wh""ich"; CV="command"" -v"
    scan_row 'S1 shell: a llama.cpp binary by bare name in command position -> R1' R1 scripts/a.sh 'llama-cli -m m.gguf -p hi'
    scan_row 'S2 shell: the pinned variable -> clean' clean scripts/a.sh '"$LLAMA_CLI" -m m.gguf -p hi'
    scan_row 'S3 shell: a literal path in command position -> R1' R1 scripts/a.sh 'out=$(/opt/llama.cpp/build/bin/llama-server --port 1)'
    scan_row 'S4 shell: the name as an argument (grep) -> clean' clean scripts/a.sh 'grep -qE "llama-cli|llama-server" f.sh'
    HD=$'cat > f <<"EOF"\nllama-server -m m\n'"$WH"$' llama-cli\nEOF'
    scan_row 'S5 shell: a heredoc body -> clean' clean scripts/a.sh "$HD"
    scan_row 'S6 shell: a PATH probe with which -> R1b' R1b scripts/a.sh "p=\$($WH llama-cli)"
    scan_row 'S7 shell: a PATH probe with command -v -> R1b' R1b scripts/lib/b.sh "LLAMA=\$($CV llama-server) || exit 1"
    scan_row 'S8 shell: a comment -> clean' clean scripts/a.sh "# never \`$WH llama-cli\`: llama-cli -m m is what we avoid"
    scan_row 'S9 rust: a which probe -> R2' R2 crates/x/tests/t.rs $'fn f() {\n    let o = Command::new("which")\n        .arg("llama-cli")\n        .output();\n}'
    scan_row 'S10 rust: Command::new of a bare name -> R3' R3 crates/x/src/lib.rs 'fn f() { let _ = std::process::Command::new("llama-server").arg("-m"); }'
    scan_row 'S11 rust: Command::new of a literal path -> R3' R3 crates/x/src/lib.rs 'fn f() { let _ = Command::new("/opt/llama.cpp/llama-cli"); }'
    scan_row 'S12 rust: a const list of candidate paths -> R4' R4 crates/x/tests/t.rs $'const C: &[&str] = &[\n    "/usr/local/bin/llama-cli",\n    "/usr/bin/llama-cli",\n];'
    scan_row 'S13 rust: the pinned env -> clean' clean crates/x/tests/t.rs 'fn f() -> Option<std::ffi::OsString> { std::env::var_os("LLAMA_CLI") } // Command::new("which").arg("llama-cli") was the old way'
    scan_row 'S14 rust: a config literal in a unit test (no spawn, no list) -> clean' clean crates/x/src/t.rs 'let c = LlamaCppConfig::new("/usr/local/bin/llama-cli");'
    # R5, the baseline ratchet
    printf 'R2\tcrates/x/tests/t.rs:3\tCommand::new("which")\n' > "$TMP/v1"
    printf 'R2\tcrates/x/tests/t.rs\t#9001\n' > "$TMP/b1"
    judge "$TMP/v1" "$TMP/b1" > "$TMP/j" 2>&1; ok_row 'B1 a listed violation passes' $? "$(cat "$TMP/j")"
    : > "$TMP/b0"
    judge "$TMP/v1" "$TMP/b0" > "$TMP/j" 2>&1; [ $? = 1 ] && grep -q 'FAIL  R2 crates/x/tests/t.rs:3' "$TMP/j"; ok_row 'B2 an unlisted violation is RED' $? "$(cat "$TMP/j")"
    : > "$TMP/v0"
    judge "$TMP/v0" "$TMP/b1" > "$TMP/j" 2>&1; [ $? = 1 ] && grep -q 'stale baseline entry R2 crates/x/tests/t.rs' "$TMP/j"; ok_row 'B3 a stale entry is RED (delete it)' $? "$(cat "$TMP/j")"
    printf 'R2\tcrates/x/tests/t.rs\t#9001\nR3\tcrates/y/src/l.rs\t#9002\n' > "$TMP/b2"
    printf 'R3\tcrates/y/src/l.rs:1\tCommand::new("llama-cli")\n' >> "$TMP/v1"
    judge "$TMP/v1" "$TMP/b2" "$TMP/b1" > "$TMP/j" 2>&1; [ $? = 1 ] && grep -q 'GREW against origin/main (1 -> 2)' "$TMP/j"; ok_row 'B4 a baseline that grows against main is RED' $? "$(cat "$TMP/j")"
    printf 'R2\tcrates/x/tests/t.rs\n' > "$TMP/b3"
    judge "$TMP/v1" "$TMP/b3" > "$TMP/j" 2>&1; [ $? = 1 ] && grep -q 'malformed baseline line' "$TMP/j"; ok_row 'B5 an entry naming no issue is RED' $? "$(cat "$TMP/j")"

    # the release gate, driven with a stub resolver (scripts/llama_bin.sh) and a stub test runner, under a
    # hermetic PATH: the host's own gpu-q and GPU lock are never touched by the table
    SB="$TMP/gate-bin"; mkdir -p "$SB"
    cat > "$SB/cargo" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FX_LOG"
case "$*" in *--no-run*) exit "${FX_BUILD_RC:-0}" ;; esac
case "${FX_TEST:-ok}" in
  ok)   printf 'running 1 test\ntest f_qw3_moe_parity_002_argmax_vs_llama_cpp ... ok\n' ;;
  skip) printf 'F-QW3-MOE-PARITY-002: skipped — $LLAMA_CLI is unset, so there is no PINNED llama-cli\n' >&2
        printf 'running 1 test\ntest f_qw3_moe_parity_002_argmax_vs_llama_cpp ... ok\n' ;;
  none) printf 'running 0 tests\n' ;;
  fail) printf 'test f_qw3_moe_parity_002_argmax_vs_llama_cpp ... FAILED\n'; exit 101 ;;
esac
STUB
    cat > "$SB/gpu-q" <<'STUB'
#!/usr/bin/env bash
printf 'gpu-q %s %s\n' "$1" "$2" >> "$FX_LOG"; shift 3; exec "$@"
STUB
    chmod +x "$SB/cargo" "$SB/gpu-q"
    SBNQ="$TMP/gate-bin-noq"; mkdir -p "$SBNQ"; ln -s "$SB/cargo" "$SBNQ/cargo"
    # gate_row NAME GATE-SCRIPT WANT-RC NEEDLE [ENV=VAL ...]
    gate_row() {
        local name=$1 g=$2 want=$3 needle=$4 d out rc; shift 4
        d="$TMP/gate-$rows"; mkdir -p "$d/scripts"; cp -- "$g" "$d/scripts/dogfood_comparator_env_tests.sh"
        printf '#!/bin/sh\n' > "$d/llama-cli"; chmod +x "$d/llama-cli"
        # the stub resolver: sets what llama_bin.sh exports, returns what a row picks
        printf 'LLAMA_COMPLETION="${FX_LLAMA_CLI-%s}"; LLAMA_BUILD="version: fixture"; export LLAMA_COMPLETION LLAMA_BUILD\nreturn "${FX_PIN_RC:-0}"\n' \
            "$d/llama-cli" > "$d/scripts/llama_bin.sh"
        # PATH is set AFTER the row's variables: FX_PATH is one of them
        out=$( export FX_LOG="$d/log"; for kv in "$@"; do export "${kv?}"; done; export PATH="${FX_PATH:-$SBNQ}:/usr/bin:/bin"
               bash "$d/scripts/dogfood_comparator_env_tests.sh" 2>&1 ); rc=$?
        [ "$rc" = "$want" ] && grep -qF -- "$needle" <<< "$out"; ok_row "$name" $? "rc=$rc (want $want): $(tr '\n' '|' <<< "$out")"
        GATE_LAST="$d"
    }
    gate_row 'D1 the gate passes a test that ran against the pinned build' "$GATE" 0 'PASS  1 comparator-consumer test(s)'
    gate_row 'D2 a SKIP is a FAIL' "$GATE" 1 'SKIPPED with the pinned build set' FX_TEST=skip
    gate_row 'D3 a test that did not run is a FAIL' "$GATE" 1 'did not run' FX_TEST=none
    gate_row 'D4 a failing test is a FAIL' "$GATE" 1 'exited 101' FX_TEST=fail
    gate_row 'D5 an unresolved pin is a FAIL' "$GATE" 1 'does not resolve' FX_PIN_RC=1
    gate_row 'D6 no LLAMA_COMPLETION exported is a FAIL' "$GATE" 1 'exported no executable LLAMA_COMPLETION' FX_LLAMA_CLI=
    gate_row 'D7 the run queues through gpu-q at release priority 1' "$GATE" 0 'PASS' "FX_PATH=$SB"
    grep -qx 'gpu-q --prio 1' "$GATE_LAST/log"; ok_row 'D7b the gpu-q call is --prio 1' $? "$(cat "$GATE_LAST/log")"

    # MUTANTS: each must turn its row RED
    mut() { # NAME SRC DST OLD NEW
        python3 - "$2" "$3" "$4" "$5" <<'PY' || { printf 'FAIL  mutant %s: anchor not found exactly once\n' "$1"; fails=$((fails + 1)); return 1; }
import sys
src, dst, old, new = sys.argv[1:5]
s = open(src).read()
if s.count(old) != 1: sys.exit(1)
open(dst, "w").write(s.replace(old, new))
PY
    }
    killed() { rows=$((rows + 1)); if [ "$2" != 0 ]; then printf 'ok    mutant %s killed\n' "$1"; else printf 'FAIL  mutant %s SURVIVED\n' "$1"; fails=$((fails + 1)); fi; }
    MUT_N=0; GATE_TRY_LAST=""
    gate_try() { # GATE-SCRIPT WANT-RC NEEDLE ENV... -> 0 when the row would be green
        local g=$1 want=$2 needle=$3 d out rc; shift 3
        MUT_N=$((MUT_N + 1)); d="$TMP/mut-$MUT_N"; GATE_TRY_LAST=$d; mkdir -p "$d/scripts"; cp -- "$g" "$d/scripts/dogfood_comparator_env_tests.sh"
        printf '#!/bin/sh\n' > "$d/llama-cli"; chmod +x "$d/llama-cli"
        printf 'LLAMA_COMPLETION="${FX_LLAMA_CLI-%s}"; LLAMA_BUILD="version: fixture"; export LLAMA_COMPLETION LLAMA_BUILD\nreturn "${FX_PIN_RC:-0}"\n' "$d/llama-cli" > "$d/scripts/llama_bin.sh"
        # PATH is set AFTER the row's variables: FX_PATH is one of them
        out=$( export FX_LOG="$d/log"; for kv in "$@"; do export "${kv?}"; done; export PATH="${FX_PATH:-$SBNQ}:/usr/bin:/bin"
               bash "$d/scripts/dogfood_comparator_env_tests.sh" 2>&1 ); rc=$?
        [ "$rc" = "$want" ] && grep -qF -- "$needle" <<< "$out"
    }
    mut drop-skip-check "$GATE" "$TMP/m1.sh" "    if grep -q 'skipped —' \"\$log\"; then" "    if false; then" \
        && { gate_try "$TMP/m1.sh" 1 'SKIPPED with the pinned build set' FX_TEST=skip; killed "drop-skip-check (D2)" $?; }
    mut drop-ran-check "$GATE" "$TMP/m2.sh" "    elif ! grep -qE \"^test \${name} \\.\\.\\. ok\$\" \"\$log\"; then" "    elif false; then" \
        && { gate_try "$TMP/m2.sh" 1 'did not run' FX_TEST=none; killed "drop-ran-check (D3)" $?; }
    mut drop-pin-check "$GATE" "$TMP/m3.sh" 'if [ "$pin_rc" -ne 0 ]; then' 'if false; then' \
        && { gate_try "$TMP/m3.sh" 1 'does not resolve' FX_PIN_RC=1; killed "drop-pin-check (D5)" $?; }
    mut no-gpuq "$GATE" "$TMP/m4.sh" 'command -v gpu-q > /dev/null 2>&1 && WRAP="gpu-q --prio $prio --"' ':' \
        && { gate_try "$TMP/m4.sh" 0 'PASS' "FX_PATH=$SB" && grep -qx 'gpu-q --prio 1' "$GATE_TRY_LAST/log"; killed "no-gpuq (D7b)" $?; }
    # the scan's own rules, mutated in the walker/scan text
    cp -- "$WALKER" "$TMP/walker.real"
    python3 - "$ONE_CLIENT" "$WALKER" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
open(sys.argv[2], "w").write(re.search(r"cat > \"\$AWK_WALKER\" <<'AWK'\n(.*?)\nAWK\n", s, re.S).group(1) + "\n")
PY
    fx="$TMP/scan-walker"; rel=scripts/a.sh; mkdir -p "$fx/$(dirname "$rel")"; printf 'llama-cli -m m.gguf -p hi\n' > "$fx/$rel"
    out=$(scan "$fx" 2>&1); grep -q '^R1	scripts/a.sh:' <<< "$out"; killed "one-client's own predicate (S1: it flags only llama-bench)" $?
    cp -- "$TMP/walker.real" "$WALKER"
    for m in 'probe_rs.finditer(code)|[].__iter__()|S9|crates/x/tests/t.rs|R2|fn f() {\n    let o = Command::new("which")\n        .arg("llama-cli")\n        .output();\n}' \
             'list_rs.finditer(code)|[].__iter__()|S12|crates/x/tests/t.rs|R4|const C: &[&str] = &[\n    "/usr/local/bin/llama-cli",\n];' \
             "probe_sh.search(code)|False|S6|scripts/a.sh|R1b|p=\$($WH llama-cli)"; do
        IFS='|' read -r old new rowname rel rule content <<< "$m"
        declare -f scan | python3 -c 'import sys; s=sys.stdin.read(); o,n=sys.argv[1:3]; sys.exit(1) if s.count(o)!=1 else print(s.replace(o,n))' "$old" "$new" > "$TMP/scan_mut.sh" \
            || { printf 'FAIL  scan mutant for %s: anchor gone\n' "$rowname"; fails=$((fails + 1)); continue; }
        fx="$TMP/mut-$rowname"; mkdir -p "$fx/$(dirname "$rel")"; printf '%b\n' "$content" > "$fx/$rel"
        out=$( . "$TMP/scan_mut.sh"; scan "$fx" 2>&1 ); grep -q "^$rule	$rel:" <<< "$out"; killed "scan without $rule ($rowname)" $?
    done
    declare -f judge | python3 -c 'import sys; s=sys.stdin.read(); o="if (r, p) not in live:"; sys.exit(1) if s.count(o)!=1 else print(s.replace(o,"if False:"))' > "$TMP/judge_mut.sh" \
        && { out=$( . "$TMP/judge_mut.sh"; judge "$TMP/v0" "$TMP/b1" 2>&1 ); [ $? = 1 ] && grep -q 'stale baseline entry' <<< "$out"; killed "judge without the stale check (B3)" $?; }

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED (%s of %s)\n' "$fails" "$rows"; exit 1; }
    printf '\nSELF-TEST PASSED (%s rows and mutants)\n' "$rows"
    exit 0
fi

# ---- the tree ------------------------------------------------------------------------------------
printf '=== every llama.cpp comparator consumer resolves through scripts/llama_bin.sh (#3740) ===\n'
scan "$ROOT" > "$TMP/viol"; src=$?
[ "$src" -eq 0 ] || { printf 'FAIL  the scan could not walk every file (see WALKER-ERROR above); nothing is judged\n'; exit 1; }
nuni=$(universe "$ROOT" | grep -c .)
[ "$nuni" -gt 0 ] || env_die "the file universe is empty"
git -C "$ROOT" show "origin/main:$BASELINE_REL" > "$TMP/main_baseline" 2> /dev/null || : > "$TMP/main_baseline"
if [ -s "$TMP/main_baseline" ] || git -C "$ROOT" cat-file -e "origin/main:$BASELINE_REL" 2> /dev/null; then
    judge "$TMP/viol" "$ROOT/$BASELINE_REL" "$TMP/main_baseline"; rc=$?
else
    printf '!     BOOTSTRAP: %s is not on origin/main yet, so the ratchet has no floor this run\n' "$BASELINE_REL"
    judge "$TMP/viol" "$ROOT/$BASELINE_REL"; rc=$?
fi
printf '      %s file(s) in the universe\n' "$nuni"
[ "$rc" -eq 0 ] && printf 'PASS\n' || printf 'FAIL  (rc=%s)\n' "$rc"
exit "$rc"
