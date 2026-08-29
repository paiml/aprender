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
# Scratch for the walkers. WALK_ERR holds ONE invocation's stderr, WALK_FAIL
# accumulates every invocation that failed, and WALK_LOG gets one line per
# invocation that COMPLETED. The last two are files rather than variables on
# purpose: every walk runs inside a `$( )` subshell, so a counter kept in a
# variable would be discarded along with the subshell — which is how a walker
# can die on every file and still leave the parent believing it swept the tree.
WALK_ERR="$(mktemp)" || exit 2
WALK_FAIL="$(mktemp)" || exit 2
WALK_LOG="$(mktemp)" || exit 2
WALK_TMP="$(mktemp)" || exit 2
CASE_DIR=""
PROBE_DIR=""
cleanup() {
    rm -f "$AWK_WALKER" "$WALK_ERR" "$WALK_FAIL" "$WALK_LOG" "$WALK_TMP"
    if [ -n "$CASE_DIR" ]; then
        rm -rf "${CASE_DIR:?}"
    fi
    if [ -n "$PROBE_DIR" ]; then
        rm -rf "${PROBE_DIR:?}"
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

# THE WALKER'S OWN EVIDENCE IS NEVER DISCARDED.
#
# `awk -f "$AWK_WALKER" "$1" 2>/dev/null` was the hole. A walker that is
# missing, empty, unparseable or bypassed emits NO LINES, and no lines is
# exactly what a clean file emits — so its failure was indistinguishable from
# its success, and the one diagnostic that could tell them apart went to
# /dev/null. Measured on this file before the fix: pointing AWK_WALKER at a
# path that does not exist still printed "files=10469  invocations=0" and "OK".
#
# So every invocation now records three things: its exit status, its stderr,
# and — on success only — one line in WALK_LOG that no other code path can
# write. rc is read from the assignment directly, never through a pipe.
_walk_failed() {   # tool, file, rc
    printf '  WALKER-ERROR  %s on %s: rc=%s %s\n' "$1" "$2" "$3" \
        "$(tr '\n' ' ' < "$WALK_ERR")" >&2
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$WALK_FAIL"
    return 1
}

walk_shell() {
    local out rc
    out="$(awk -f "$AWK_WALKER" "$1" 2>"$WALK_ERR")"; rc=$?
    if [ "$rc" -ne 0 ] || [ -s "$WALK_ERR" ]; then _walk_failed awk "$1" "$rc"; return 1; fi
    printf 'shell\t%s\n' "$1" >> "$WALK_LOG"
    printf '%s' "$out"
}

# A make recipe is shell WRAPPED in make syntax. Two things have to go before
# the shell walker sees it, or `\t@$(LLAMA_BENCH) -m $(MODEL)` reads as the
# command `@$` followed by a substitution: the recipe-prefix characters `@`
# and `-`, and make's `$(VAR)` — which is a variable reference, not a command
# substitution. Line numbers are preserved because nothing is added or removed.
walk_make() {
    local out rc
    sed -e 's/\$(\([A-Za-z_][A-Za-z0-9_]*\))/$\1/g' -e 's/^\(\t\)[@+-]*/\1/' "$1" \
        > "$WALK_TMP" 2>"$WALK_ERR"; rc=$?
    if [ "$rc" -ne 0 ] || [ -s "$WALK_ERR" ]; then _walk_failed sed "$1" "$rc"; return 1; fi
    # Two commands, not a pipeline: `out=$(a | b)` leaves PIPESTATUS describing
    # the assignment, so the sed half of the old pipeline could fail unseen.
    out="$(awk -f "$AWK_WALKER" "$WALK_TMP" 2>"$WALK_ERR")"; rc=$?
    if [ "$rc" -ne 0 ] || [ -s "$WALK_ERR" ]; then _walk_failed awk "$1" "$rc"; return 1; fi
    printf 'make\t%s\n' "$1" >> "$WALK_LOG"
    printf '%s' "$out"
}

# Python and Rust cannot be walked as shell. There the comparator binary can
# only enter as a STRING LITERAL naming it — `Command::new("llama-bench")`,
# `subprocess.run(["/opt/llama.cpp/llama-bench", ...])`. Prose that merely says
# the word is not a literal equal to it, so a doc comment stays green and so
# does the reference to llama.cpp's compare-llama-bench.py in bench_receipt.py.
LITERAL_RE='("|'"'"')([^"'"'"']*/)?llama-bench("|'"'"')'
walk_literal() {
    local out rc
    sed -e 's://.*$::' -e 's:^[[:space:]]*#.*$::' "$1" > "$WALK_TMP" 2>"$WALK_ERR"; rc=$?
    if [ "$rc" -ne 0 ] || [ -s "$WALK_ERR" ]; then _walk_failed sed "$1" "$rc"; return 1; fi
    out="$(grep -nE "$LITERAL_RE" "$WALK_TMP" 2>"$WALK_ERR")"; rc=$?
    # grep rc=1 is "no match", the ordinary answer here; rc>=2 is a real error.
    # The old `|| true` swallowed both, so a grep that could not run read as a
    # clean file. Stderr is checked at every rc, including 0.
    if [ "$rc" -gt 1 ] || [ -s "$WALK_ERR" ]; then _walk_failed grep "$1" "$rc"; return 1; fi
    printf 'literal\t%s\n' "$1" >> "$WALK_LOG"
    printf '%s' "$out"
}

# ---------------------------------------------------------------------------
# THE WALKER, PROVED ENGAGED — the second sentinel.
#
# The pre-filter sentinel further down proves the FILTER returned files. It
# says nothing whatever about the walker that reads them, and the first version
# of this guard treated the one as proof of the other. It is not: the filter
# can hand over 200 candidates to a walker that is absent, empty, blind, mute
# or bypassed, and the sweep then prints "invocations=0 / OK" over a tree it
# never read. Five mutations of the walker alone were measured against that
# version and every one of them PASSED (rc=0) with byte-identical output.
#
# An exit-status check alone does not close it either: emptying the awk program
# leaves awk exiting 0 and printing nothing. So the walker has to produce
# something only a WORKING walker can produce. Each walker is run here over a
# fixture whose classification is known exactly, and its own output is compared
# against the expected string — which carries the LINE NUMBER the walker itself
# computed. The fixtures' first lines name the same token in prose, so a walker
# that flags everything fails the probe as surely as one that flags nothing.
#
# The fixtures are written with printf from single-quoted format strings, so
# the token sits in ARGUMENT position and this file still passes its own gate.
# ---------------------------------------------------------------------------
_probe_check() {   # label, expected, got
    if [ "$3" = "$2" ]; then
        return 0
    fi
    printf '  WALKER-DEAD  %s walker returned [%s], expected [%s]\n' "$1" "$3" "$2" >&2
    return 1
}

prove_walkers_engaged() {
    local d exp got bad=0
    PROBE_DIR="$(mktemp -d)" || return 1
    d="$PROBE_DIR"
    case "$d" in
        /tmp/*|/var/folders/*) : ;;
        *) printf '  WALKER-DEAD  refusing probe dir %s\n' "$d" >&2; return 1 ;;
    esac

    # shell: line 1 is prose and must be ignored, line 2 is an invocation.
    printf '# prose naming llama-bench is not an invocation\n./llama-bench -m probe.gguf -n 8\n' > "$d/probe.sh"
    exp="$(printf '2\t%s' "$(sed -n '2p' "$d/probe.sh")")"
    got="$(walk_shell "$d/probe.sh")"
    _probe_check shell "$exp" "$got" || bad=1

    # make: the recipe on line 3, after prefix stripping and $(VAR) rewriting.
    printf '# prose naming llama-bench in a makefile\nbench:\n\t@$(LLAMA_BENCH) -m $(MODEL) -n 8\n' > "$d/probe.mk"
    exp="$(printf '3\t\t$LLAMA_BENCH -m $MODEL -n 8')"
    got="$(walk_make "$d/probe.mk")"
    _probe_check make "$exp" "$got" || bad=1

    # literal: line 1 is a comment stripped by sed, line 2 is a string literal.
    printf '// llama-bench named in a comment stays green\nlet _ = Command::new("llama-bench").arg("-m");\n' > "$d/probe.rs"
    exp="$(printf '2:%s' "$(sed -n '2p' "$d/probe.rs")")"
    got="$(walk_literal "$d/probe.rs")"
    _probe_check literal "$exp" "$got" || bad=1

    # A walker that failed OUTRIGHT already filed itself; report that too, so a
    # probe that somehow matched anyway cannot launder an errored invocation.
    if [ -s "$WALK_FAIL" ]; then
        printf '  WALKER-DEAD  %s walker invocation(s) errored during the probe\n' \
            "$(grep -c . "$WALK_FAIL")" >&2
        bad=1
    fi
    [ "$bad" = 0 ]
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
      # ...and the same free pass was still open on the two types the globs
      # above claim REPO-WIDE rather than under a directory: `*.rs` and the
      # makefiles. Measured, not assumed: dropping
      #     fn d() { let _ = Command::new("llama-bench").arg("-m"); }
      # into crates/aprender-core/src/ left the gate at rc=0 while UNTRACKED
      # and went to rc=1 the moment `git add -N` put it in ls-files' output.
      # Same file, same walker, same predicate — only the universe differed.
      # Pruned at the build dirs, which hold no hand-written comparator and
      # would otherwise dominate the walk.
      find "$ROOT" \( -name .git -o -name target -o -name 'target_*' \
                     -o -name node_modules -o -name vendor -o -name .venv \) -prune -o \
          -type f \( -name '*.rs' -o -name 'Makefile' -o -name '*.mk' \) -print 2>/dev/null \
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
    local base="$1" rel f hits n=0 scanned=0 ncand=0 walked=0 uni cand
    uni="$(mktemp)"; cand="$(mktemp)"
    universe > "$uni"
    scanned=$(grep -c . "$uni")
    ( cd "$base" && candidates "$uni" ) > "$cand"
    ncand=$(grep -c . "$cand")
    # The probe above ran the walkers three times. Those invocations prove the
    # walker is alive; they are not coverage of the TREE, so the ledger starts
    # empty here and every line in it from now on is a file that was walked.
    : > "$WALK_LOG"
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
        printf 'FILTER_BROKEN %s 0 0\n' "$scanned"
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
    walked=$(grep -c . "$WALK_LOG")
    # A walker that dies PART WAY through leaves the count short while the
    # verdict still reads "invocations=0". The caller compares the two.
    if [ -s "$WALK_FAIL" ]; then
        printf 'WALKER_ERROR %s %s %s\n' "$scanned" "$ncand" "$walked"
        return 0
    fi
    printf '%s %s %s %s\n' "$n" "$scanned" "$ncand" "$walked"
}

gate() {
    echo "=== one client, both servers (PERF-019 / I-15) ==="
    local out n scanned ncand walked
    # STAGE 1 of 3. Nothing below this line means anything until the walker is
    # known to work, because every way it can break is silent.
    if ! prove_walkers_engaged; then
        echo "FAIL: the walker did not classify its own probe fixtures. The"
        echo "      sweep below would have reported 'invocations=0' over a tree"
        echo "      it never read — a missing, empty, blind or bypassed walker"
        echo "      emits no lines, and no lines is what a clean file emits."
        return 1
    fi
    echo "  walker: ENGAGED (shell, make and literal probes classified exactly)"
    out="$(scan "$ROOT")"
    read -r n scanned ncand walked <<< "$out"
    if [ "$n" = "FILTER_BROKEN" ]; then
        echo "  files=$scanned  invocations=<not computed>"
        echo "FAIL: the candidate pre-filter returned nothing usable, so every"
        echo "      file was skipped. This would have reported OK over an"
        echo "      unexamined tree. Check that 'xargs -0 -r grep -lIiE' works"
        echo "      on this host."
        return 1
    fi
    echo "  filter: ENGAGED (this guard is in its own candidate set)"
    if [ "$n" = "WALKER_ERROR" ]; then
        echo "  files=$scanned  candidates=$ncand  walked=$walked  invocations=<not computed>"
        echo "FAIL: a walker invocation errored mid-sweep. Its stderr is above."
        echo "      The verdict is withheld: an errored walk returns no lines,"
        echo "      which is indistinguishable from a clean file."
        return 1
    fi
    echo "  files=$scanned  candidates=$ncand  walked=$walked  invocations=$n"
    # STAGE 3. Every candidate the filter produced must have been WALKED. This
    # is the arithmetic the first version never did: it counted the universe,
    # never the files a walker actually read.
    if [ "$walked" -lt 1 ] || [ "$walked" != "$ncand" ]; then
        echo "FAIL: the filter offered $ncand candidate(s) and a walker read"
        echo "      $walked of them. A file that is never walked cannot be"
        echo "      reported clean."
        return 1
    fi
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
    # The case table is written entirely in terms of the walkers, so a dead
    # walker would turn every must-match row BROKE and every must-not-match row
    # ok. Prove it first and say so, rather than reading the mixture.
    echo "=== the walker is alive (probe fixtures, exact output) ==="
    if ! prove_walkers_engaged; then
        echo "  BROKE the walker cannot classify its own fixtures"
        return 1
    fi
    echo "  ok    shell, make and literal walkers returned the expected lines"
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
