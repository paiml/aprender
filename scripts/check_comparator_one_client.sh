#!/usr/bin/env bash
# PERF-019 / I-15 — ONE CLIENT, BOTH SERVERS.
#
# THE RULE (APR-PERF-GATE-001 §4.4.8, finding X7). The comparator is
# `llama-server` driven by OUR client, never llama.cpp's own bench binary. That
# binary "does not separate PP from TG under concurrent load; metrics
# intertwine" — so a ratio built from it is not apr-vs-llama.cpp, it is
# our-client-vs-their-client with two servers attached. The client decides
# concurrency, streaming, warmup and where the clock starts; swapping it swaps
# the measurement while the receipt still calls the result a server ratio.
#
# WHAT THIS GUARD BANS. Executing llama.cpp's bench binary from anything on the
# comparator path. `scripts/parity_host_receipt.sh` drives BOTH servers with
# `apr test llm bench` (lines 101 and 117 at the time of writing) and reaches
# the comparator only through `$LLAMA_SERVER`. This keeps that true.
#
# WHY A COMMAND-POSITION WALKER AND NOT A grep. The banned token has to appear
# in the guards that ban it, in the pin resolver that locates the build tree by
# it, and in the case tables that prove those guards can fail. A grep for the
# token reddens all of them, and the usual escape — exempting a file by name —
# is how a guard stops seeing the file it most needs to watch. So the predicate
# asks the only question that distinguishes them: is the token in COMMAND
# POSITION? The walker tracks quote state, command substitution and heredoc
# bodies, and tests only the word that would be EXECUTED.
#
# That makes every legitimate mention green BY CONSTRUCTION, with no allowlist:
#
#   grep -qE "...|llama-bench|..."           command word is `grep`
#   printf 'no llama-server beside ...'      command word is `printf`
#   LLAMA_BENCH="$candidate"                 an assignment, not a command
#   export LLAMA_BENCH LLAMA_BUILD           command word is `export`
#   chmod +x "$1/llama-bench"                command word is `chmod`
#   LLAMA_BIN="$(command -v llama-bench)"    `command -v` LOOKS UP, never runs
#   <<'CASES' ... llama-bench ... CASES      heredoc body, not shell
#   # prose naming the banned binary         a comment
#
# and this file passes its own gate for exactly those reasons rather than
# because it is spelled out somewhere. The selftest asserts that.
#
#   bash scripts/check_comparator_one_client.sh            # gate
#   bash scripts/check_comparator_one_client.sh --selftest # case table
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AWK_WALKER="$(mktemp)" || exit 2
CASE_DIR=""
cleanup() {
    rm -f "$AWK_WALKER"
    if [ -n "$CASE_DIR" ]; then
        rm -rf "${CASE_DIR:?}"
    fi
    return 0
}
trap cleanup EXIT

# The walker lives in a QUOTED heredoc, so this file's own copy of the pattern
# sits in a heredoc body — skipped by the same rule that exempts every other
# guard's fixtures. The mechanism is applied to itself, not worked around.
cat > "$AWK_WALKER" <<'AWK'
# Is this word the comparator bench binary, once quoting is removed?
function isbench(w,   s) {
    s = w
    gsub(/"/, "", s)
    gsub(/'/, "", s)
    if (s ~ /^\$\{?LLAMA_BENCH\}?$/)   return 1   # "$LLAMA_BENCH", ${LLAMA_BENCH}
    if (s ~ /(^|\/)llama-bench$/)      return 1   # bare, ./relative, /absolute
    return 0
}

# Words that pass command position along to the next word. Flags are NOT
# skipped, deliberately: `command -v llama-bench` is a PATH LOOKUP, not an
# execution, and stopping at `-v` is what keeps it green.
function istransfer(w) {
    return (w == "if" || w == "while" || w == "until" || w == "then" ||
            w == "do" || w == "else" || w == "elif" || w == "!" ||
            w == "time" || w == "exec" || w == "eval" || w == "command" ||
            w == "sudo" || w == "nohup" || w == "env" || w == "{" ||
            w == "-" || w == "@" || w == "run:")
}
function isassign(w) { return (w ~ /^[A-Za-z_][A-Za-z0-9_]*=/) }

BEGIN { hd = ""; contline = 0; cont_q = 0; cont_cmdpos = 1 }

{
    # A heredoc body is DATA, not shell. This is what makes every guard's
    # must-match fixture green without naming the guard.
    if (hd != "") {
        t = $0
        sub(/^[ \t]+/, "", t); sub(/[ \t]+$/, "", t)
        if (t == hd) hd = ""
        next
    }

    line = $0
    n = length(line)
    # A backslash-continued line is the SAME command. Resetting command
    # position at every physical line said that `"$LLAMA_BENCH" "$LLAMA_BUILD"`
    # on the second line of a wrapped printf was an invocation — the guard's
    # own self-reference proof caught it against scripts/llama_bin.sh:180.
    if (contline) { q = cont_q; cmdpos = cont_cmdpos } else { q = 0; cmdpos = 1 }
    depth = 0; pend = ""; sawcomment = 0
    i = 1
    while (i <= n) {
        c = substr(line, i, 1)

        if (q == 1) { if (c == "'") q = 0; i++; continue }
        if (q == 2) {
            if (c == "\\") { i += 2; continue }
            if (c == "$" && substr(line, i + 1, 1) == "(") {
                depth++; stack[depth] = 2; q = 0; cmdpos = 1; i += 2; continue
            }
            if (c == "\"") q = 0
            i++; continue
        }

        # unquoted. A quote at COMMAND POSITION opens a word, not a string to
        # skip: `"$LLAMA_BENCH" -m ...` is the most common real invocation and
        # the first draft skipped straight past it because the word began with
        # a quote character. Elsewhere a quoted run is argument text, and
        # skipping it is what keeps a `|`-separated ban list inside a grep from
        # reading as a pipeline.
        if (c == "\\") { i += 2; continue }
        if (c == "'" && !cmdpos) { q = 1; i++; continue }
        if (c == "\"" && !cmdpos) { q = 2; i++; continue }
        if (c == "#" && (i == 1 || substr(line, i - 1, 1) ~ /[ \t]/)) { sawcomment = 1; break }
        if (c == "$" && substr(line, i + 1, 1) == "(") {
            depth++; stack[depth] = 0; q = 0; cmdpos = 1; i += 2; continue
        }
        if (c == ")") {
            if (depth > 0) { q = stack[depth]; depth-- } else { cmdpos = 1 }
            i++; continue
        }
        if (c == "<" && substr(line, i + 1, 1) == "<") {
            if (substr(line, i + 2, 1) == "<") { i += 3; continue }   # here-STRING
            j = i + 2
            if (substr(line, j, 1) == "-") j++
            while (substr(line, j, 1) ~ /[ \t]/) j++
            d = ""
            while (j <= n && substr(line, j, 1) !~ /[ \t;|&<>()]/) { d = d substr(line, j, 1); j++ }
            gsub(/"/, "", d); gsub(/'/, "", d); gsub(/\\/, "", d)
            if (d != "") pend = d
            i = j; continue
        }
        if (c ~ /[;|&(`{}]/) { cmdpos = 1; i++; continue }
        if (c ~ /[ \t]/) { i++; continue }

        # A word starts here. Consume it, honouring quotes inside it.
        w = ""; wq = 0
        while (i <= n) {
            c = substr(line, i, 1)
            # `$(` ends the word wherever it appears, quoted or not, so
            # `out=$("$LLAMA_BENCH" -m m)` yields the assignment prefix and then
            # hands command position to the substitution instead of swallowing
            # the whole line as one quoted token.
            if (c == "$" && substr(line, i + 1, 1) == "(") break
            if (wq == 0 && c ~ /[ \t;|&()`<>]/) break
            if (wq == 0 && c == "\"") wq = 2
            else if (wq == 0 && c == "'") wq = 1
            else if (wq == 2 && c == "\"") wq = 0
            else if (wq == 1 && c == "'") wq = 0
            w = w c; i++
        }
        # A separator this loop does not consume (a bare `<` or `>`) would
        # otherwise leave `i` where it was and spin forever. The first draft
        # did exactly that and hung on the first file containing a redirect.
        if (w == "") { i++; continue }
        if (cmdpos) {
            if (isassign(w) || istransfer(w)) { cmdpos = 1 }
            else {
                if (isbench(w)) printf "%d\t%s\n", FNR, $0
                cmdpos = 0
            }
        }
    }
    # An ODD number of trailing backslashes continues the line; an even number
    # is an escaped backslash and ends it. A comment cannot be continued.
    nb = 0; k = n
    while (k >= 1 && substr(line, k, 1) == "\\") { nb++; k-- }
    if (nb % 2 == 1 && !sawcomment) {
        contline = 1; cont_q = q; cont_cmdpos = cmdpos
    } else {
        contline = 0
    }
    if (pend != "") hd = pend
}
AWK

walk_shell() { awk -f "$AWK_WALKER" "$1" 2>/dev/null; }

# A make recipe is shell WRAPPED in make syntax. Two things have to go before
# the shell walker sees it, or `\t@$(LLAMA_BENCH) -m $(MODEL)` reads as the
# command `@$` followed by a substitution: the recipe-prefix characters `@`
# and `-`, and make's `$(VAR)` — which is a variable reference, not a command
# substitution. Line numbers are preserved because nothing is added or removed.
walk_make() {
    sed -e 's/\$(\([A-Za-z_][A-Za-z0-9_]*\))/$\1/g' -e 's/^\(\t\)[@+-]*/\1/' "$1" 2>/dev/null \
      | awk -f "$AWK_WALKER" 2>/dev/null
}

# Python and Rust cannot be walked as shell. There the comparator binary can
# only enter as a STRING LITERAL naming it — `Command::new("llama-bench")`,
# `subprocess.run(["/opt/llama.cpp/llama-bench", ...])`. Prose that merely says
# the word is not a literal equal to it, so a doc comment stays green and so
# does the reference to llama.cpp's compare-llama-bench.py in bench_receipt.py.
LITERAL_RE='("|'"'"')([^"'"'"']*/)?llama-bench("|'"'"')'
walk_literal() {
    sed -e 's://.*$::' -e 's:^[[:space:]]*#.*$::' "$1" 2>/dev/null \
      | grep -nE "$LITERAL_RE" || true
}

# UNIVERSE: tracked UNION working tree. A `git ls-files`-only universe hands an
# untracked script a free pass, which is how three earlier guards went blind.
universe() {
    { git -C "$ROOT" ls-files \
          'scripts/*.sh' 'scripts/**/*.sh' 'crates/*/scripts/*.sh' \
          'scripts/*.py' 'scripts/**/*.py' \
          '.github/workflows/*.yml' '.github/workflows/*.yaml' \
          'Makefile' '*.mk' 'crates/*/Makefile' \
          '*.rs' 'crates/**/*.rs' 2>/dev/null || true
      find "$ROOT/scripts" \( -name '*.sh' -o -name '*.py' \) -type f 2>/dev/null \
          | sed "s|^$ROOT/||" || true
      find "$ROOT/.github/workflows" \( -name '*.yml' -o -name '*.yaml' \) -type f 2>/dev/null \
          | sed "s|^$ROOT/||" || true
      # crates/*/scripts is a real comparator surface —
      # crates/aprender-serve/scripts/bench-server-matrix.sh lives there — and
      # covering it by `git ls-files` alone would hand an UNTRACKED script in
      # it the same free pass that blinded three earlier guards.
      find "$ROOT/crates" -mindepth 2 -maxdepth 3 -path '*/scripts/*' \
          \( -name '*.sh' -o -name '*.py' \) -type f 2>/dev/null \
          | sed "s|^$ROOT/||" || true
    } | LC_ALL=C sort -u
}

# PRE-FILTER, and it is a STRICT SUPERSET, not a heuristic. Both predicates can
# only fire on a word containing the literal `llama-bench` or `LLAMA_BENCH`
# (walk_make normalises `$(LLAMA_BENCH)` to `$LLAMA_BENCH`, which still does).
# A file holding neither substring is unflaggable, so skipping it changes no
# verdict — it only avoids ~20,000 sed/grep/awk spawns per run across the Rust
# tree, which took a single gate run past a minute on a loaded box.
#
# The VACUITY check still counts the FULL universe, so narrowing the walk set
# can never be mistaken for a collapsed universe.
candidates() {
    local uni="$1"
    tr '\n' '\0' < "$uni" | xargs -0 -r grep -lIiE -e 'llama[-_]bench' -- 2>/dev/null | LC_ALL=C sort -u
}

scan() {
    local base="$1" rel f hits n=0 scanned=0 uni cand
    uni="$(mktemp)"; cand="$(mktemp)"
    universe > "$uni"
    scanned=$(grep -c . "$uni")
    ( cd "$base" && candidates "$uni" ) > "$cand"
    # POKA-YOKE FOR THE FILTER ITSELF. If xargs or grep fails — a BSD xargs with
    # no `-r`, a locale surprise, a permissions error — `candidates` returns
    # nothing, every file is skipped, and the gate reports OK over a full
    # universe. That is the vacuous-pass class this repo keeps closing, moved
    # one level down. This file always contains the token (it bans it), and it
    # is always in the universe, so its absence from the candidate set can only
    # mean the filter is broken.
    if ! grep -qxF 'scripts/check_comparator_one_client.sh' "$cand"; then
        printf '  FILTER-BROKEN  the pre-filter did not return this guard itself\n' >&2
        rm -f "$uni" "$cand"
        printf 'FILTER_BROKEN %s\n' "$scanned"
        return 0
    fi
    while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        f="$base/$rel"
        [ -f "$f" ] || continue
        case "$rel" in
            *.rs|*.py)          hits="$(walk_literal "$f")" ;;
            Makefile|*/Makefile|*.mk) hits="$(walk_make "$f")" ;;
            *)                  hits="$(walk_shell "$f")" ;;
        esac
        if [ -n "$hits" ]; then
            while IFS= read -r h; do
                [ -n "$h" ] || continue
                printf '  COMPARATOR-BENCH  %s:%s\n' "$rel" "$h" >&2
                n=$((n + 1))
            done <<< "$hits"
        fi
    done < "$cand"
    rm -f "$uni" "$cand"
    printf '%s %s\n' "$n" "$scanned"
}

gate() {
    echo "=== one client, both servers (PERF-019 / I-15) ==="
    local out n scanned
    out="$(scan "$ROOT")"
    n="${out%% *}"; scanned="${out##* }"
    if [ "$n" = "FILTER_BROKEN" ]; then
        echo "  files=$scanned  invocations=<not computed>"
        echo "FAIL: the candidate pre-filter returned nothing usable, so every"
        echo "      file was skipped. This would have reported OK over an"
        echo "      unexamined tree. Check that 'xargs -0 -r grep -lIiE' works"
        echo "      on this host."
        return 1
    fi
    echo "  files=$scanned  invocations=$n"
    # VACUITY: a sweep over nothing is not a pass.
    if [ "$scanned" -lt 200 ]; then
        echo "FAIL: scanned only $scanned file(s); the universe collapsed."
        return 1
    fi
    if [ "$n" -gt 0 ]; then
        echo "FAIL: the comparator is driven by the llama.cpp bench binary."
        echo "      4.4.8/X7: it does not separate PP from TG under concurrent"
        echo "      load. Drive llama-server with 'apr test llm bench', the same"
        echo "      client the apr side uses (scripts/parity_host_receipt.sh)."
        return 1
    fi
    echo "OK"
}

# ---------------------------------------------------------------------------
# A guard seen only passing is not evidence. Every row below is a file this
# walker must classify. The must-not-match half is copied from shapes that
# actually exist in scripts/llama_bin.sh, scripts/check_llama_pin.sh,
# scripts/check_no_competing_harnesses.sh and check_no_fabricated_baselines.sh.
# ---------------------------------------------------------------------------
selftest() {
    CASE_DIR="$(mktemp -d)" || exit 2
    case "$CASE_DIR" in /tmp/*|/var/folders/*) : ;; *) echo "refusing $CASE_DIR"; exit 2 ;; esac
    local pass=0 fail=0

    _row() { # name, expect(detect|ignore), extension, body
        local name="$1" want="$2" ext="$3" body="$4" got=ignore probe hits
        probe="$CASE_DIR/probe.$ext"
        printf '%s\n' "$body" > "$probe"
        case "$ext" in
            rs|py) hits="$(walk_literal "$probe")" ;;
            mk)    hits="$(walk_make "$probe")" ;;
            *)     hits="$(walk_shell "$probe")" ;;
        esac
        if [ -n "$hits" ]; then
            got=detect
        fi
        if [ "$got" = "$want" ]; then
            printf '  ok    %-46s expect=%s\n' "$name" "$want"; pass=$((pass + 1))
        else
            printf '  BROKE %-46s expected %s got %s\n' "$name" "$want" "$got"; fail=$((fail + 1))
        fi
    }

    echo "=== must-match: an invocation on the comparator path ==="
    _row "pinned var, quoted"       detect sh  '"$LLAMA_BENCH" -m "$MODEL" -n 128 -o json'
    _row "pinned var, bare"         detect sh  '$LLAMA_BENCH -m model.gguf'
    _row "braced var"               detect sh  '${LLAMA_BENCH} -m model.gguf'
    _row "relative path"            detect sh  './llama-bench -m model.gguf -p 0 -n 128 -r 5 -o json'
    _row "bare name on PATH"        detect sh  'llama-bench -m model.gguf -n 128'
    _row "absolute build path"      detect sh  '/opt/llama.cpp/build/bin/llama-bench -m m.gguf'
    _row "inside command subst"     detect sh  'out=$("$LLAMA_BENCH" -m m.gguf -o json)'
    _row "after a pipe"             detect sh  'echo x | llama-bench -m m.gguf'
    _row "after &&"                 detect sh  'cd "$d" && ./llama-bench -m m.gguf'
    _row "inside a for loop body"   detect sh  'for c in 1 4; do "$LLAMA_BENCH" -m m -r "$c"; done'
    _row "env-prefixed"             detect sh  'CUDA_VISIBLE_DEVICES=0 ./llama-bench -m m.gguf'
    _row "exec-ed"                  detect sh  'exec /usr/local/bin/llama-bench -m m.gguf'
    _row "workflow run: step"       detect yml '        run: ./llama-bench -m m.gguf -o json'
    _row "workflow list-form step"  detect yml '      - run: llama-bench -m m.gguf'
    _row "makefile recipe"          detect mk  '	@$(LLAMA_BENCH) -m $(MODEL) -n 128'
    _row "rust Command::new"        detect rs  'let out = Command::new("llama-bench").arg("-m").output()?;'
    _row "rust absolute path"       detect rs  'Command::new("/opt/llama.cpp/llama-bench")'
    _row "python subprocess"        detect py  'subprocess.run(["llama-bench", "-m", model], check=True)'
    _row "python path literal"      detect py  "subprocess.check_output(['/opt/llama.cpp/bin/llama-bench'])"

    echo "=== must-not-match: the self-reference trap (guards, pins, prose) ==="
    _row "a ban list inside grep"   ignore sh  'grep -qE "serve run|llama-server|llama-cli|llama-bench|vllm serve" "$1"'
    _row "prose in a comment"       ignore sh  '# 4.4.8 forbids driving the comparator with llama-bench at all'
    _row "trailing comment"         ignore sh  'true   # see llama-bench, retired by I-15'
    _row "a printf message"         ignore sh  "printf 'FAIL  no llama-server beside the pinned llama-bench\\n' >&2"
    _row "assignment of the path"   ignore sh  'LLAMA_BENCH="$llama_bin_candidate"'
    _row "export of the name"       ignore sh  'export LLAMA_BENCH LLAMA_BUILD LLAMA_CLI LLAMA_SERVER'
    _row "the pin env var"          ignore sh  'llama_bin_candidate="${LLAMA_BENCH_PATH:-}"'
    _row "command -v LOOKS UP"      ignore sh  'LLAMA_BIN="$(command -v llama-bench)"'
    _row "a stub written by a test" ignore sh  'printf "#!/bin/sh\\nexit 1\\n" > "$1/llama-bench"'
    _row "chmod on a stub"          ignore sh  'chmod +x "$1/llama-bench"'
    _row "a case-table argument"    ignore sh  'run_case "pinned" "abcdef1" "$td/good/llama-bench" 0'
    _row "resolver uses the SERVER" ignore sh  '"$LLAMA_SERVER" -m "$MODEL" --port "$p" -ngl "$ngl"'
    _row "our client drives it"     ignore sh  '"$APR" test llm bench --url "http://127.0.0.1:$lport" --stream'
    _row "rust doc comment"         ignore rs  '/// Verified llama-bench baselines (RTX 4090, tg64):'
    _row "rust line comment"        ignore rs  '// llama-bench is banned on the comparator path'
    _row "python comment"           ignore py  '# adopted from llama.cpp compare-llama-bench.py'
    _row "python prose docstring"   ignore py  '"""Ratios never come from llama-bench; see 4.4.8."""'
    _row "workflow step name"       ignore yml '      - name: llama-bench must not drive the comparator'

    # Multi-line fixtures, written from heredocs so the file itself carries no
    # escaped-quote thicket. `_probe` classifies whatever probe.sh currently is.
    _probe() { # name, expect(detect|ignore)
        local got=ignore hits
        hits="$(walk_shell "$CASE_DIR/probe.sh")"
        if [ -n "$hits" ]; then
            got=detect
        fi
        if [ "$got" = "$2" ]; then
            printf '  ok    %-46s expect=%s\n' "$1" "$2"; pass=$((pass + 1))
        else
            printf '  BROKE %-46s expected %s got %s\n' "$1" "$2" "$got"; fail=$((fail + 1))
        fi
    }

    echo "=== a backslash continuation is the SAME command ==="
    # Command position must CARRY across a continuation...
    cat > "$CASE_DIR/probe.sh" <<'CONT_CMD'
CUDA_VISIBLE_DEVICES=0 \
    ./llama-bench -m m.gguf
CONT_CMD
    _probe "continuation keeps command position" detect
    # ...and so must ARGUMENT position. This is scripts/llama_bin.sh:179-180
    # verbatim, and the row exists because the self-reference proof below
    # reported it as an invocation, not because anyone anticipated it.
    cat > "$CASE_DIR/probe.sh" <<'CONT_ARG'
printf 'ok    llama.cpp pinned: %s\n      build: %s\n' \
       "$LLAMA_BENCH" "$LLAMA_BUILD" ;;
CONT_ARG
    _probe "continuation keeps argument position" ignore

    echo "=== the heredoc rule, asserted rather than assumed ==="
    # HEREDOC BODIES ARE DATA. This is the mechanism that keeps the other
    # guards green without an allowlist, so it is proved in both directions.
    cat > "$CASE_DIR/probe.sh" <<'HD_BODY'
cat > f <<'CASES'
./llama-bench -m m.gguf
CASES
HD_BODY
    _probe "heredoc body is data" ignore
    # ...and the line AFTER a heredoc closes must still be walked, or the skip
    # becomes a hole big enough to hide the invocation in.
    cat > "$CASE_DIR/probe.sh" <<'HD_AFTER'
cat > f <<'CASES'
text
CASES
./llama-bench -m m.gguf
HD_AFTER
    _probe "walking resumes after the heredoc" detect
    # A here-STRING is not a heredoc; treating it as one would skip the file.
    cat > "$CASE_DIR/probe.sh" <<'HD_STRING'
grep -q x <<< "$out"
./llama-bench -m m.gguf
HD_STRING
    _probe "here-string is not a heredoc" detect

    echo "=== the self-reference proof ==="
    # This guard names the banned binary dozens of times and must classify
    # ITSELF clean, structurally, with no filename exemption anywhere.
    cp "$ROOT/scripts/check_comparator_one_client.sh" "$CASE_DIR/probe.sh"
    _probe "this guard passes its own predicate" ignore
    # The five in-tree files that carry the token on purpose: the two guards
    # whose ban lists name it, the pin case table that builds stubs called it,
    # the resolver that locates a build tree by it, and the receipt producer
    # whose header explains why it is not used. If any of them reds, the answer
    # is a better predicate, never an allowlist entry.
    for sib in check_no_competing_harnesses.sh check_no_fabricated_baselines.sh \
               check_llama_pin.sh llama_bin.sh parity_host_receipt.sh; do
        cp "$ROOT/scripts/$sib" "$CASE_DIR/probe.sh"
        _probe "$sib" ignore
    done

    printf '  %d passed, %d broken\n' "$pass" "$fail"
    [ "$fail" = 0 ]
}

case "${1:-}" in
    --selftest) selftest ;;
    "")         gate ;;
    *)          echo "usage: $0 [--selftest]" >&2; exit 2 ;;
esac
