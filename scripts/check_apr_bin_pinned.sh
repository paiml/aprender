#!/usr/bin/env bash
# check_apr_bin_pinned.sh - no execution surface may resolve `apr` through PATH
# or through a hardcoded absolute path.
#
# THE CLASS. `cargo install --path crates/apr-cli --force` writes to
# ~/.cargo/bin. If anything earlier on PATH holds an older `apr`, a bare `apr`
# invocation runs THAT one. qwen-story-daily did exactly this: it installed
# 0.61.0 and then executed a 24-day-old 0.60.0 from ~/.local/bin, so every beat
# validated stale code while reporting green.
#
# scripts/apr_bin.sh makes that DETECTABLE at runtime (it compares the binary's
# embedded git SHA against HEAD). This script makes it UNREINTRODUCIBLE: any new
# unpinned `apr` invocation on an execution surface fails the PR that adds it.
#
# An invocation is PINNED when it goes through one of:
#   "$APR" / ${APR} / $APR_BIN     - resolved by scripts/apr_bin.sh
#   a checkout-relative path       - ./target/release/apr, ${ROOT}/target/...
#   cargo run ... --bin apr        - built from the current source by definition
#
# Exit 0 = every invocation on every scanned surface is pinned.
# Exit 1 = at least one is not.
#
# ---------------------------------------------------------------------------
# SCOPE (#2358). The previous revision scanned workflows, the scripts a workflow
# NAMES, and the Makefile. Three whole surfaces sat outside it, and all three
# held live violations:
#
#   1. scripts/ reachable only INDIRECTLY. Discovery was one level deep - grep
#      the workflows for `scripts/*.sh`. scripts/dogfood-book.sh runs
#      scripts/check_book_examples_executable.sh, which resolved its binary with
#      `command -v apr` and fell back to a hardcoded /mnt path. Never scanned,
#      because no workflow names it by name. The fix is to stop deriving: EVERY
#      scripts/**/*.sh is in scope. "Reachable from CI" is not a property you can
#      compute with one grep, and guessing it wrong fails silently.
#   2. .claude/skills/**. The dogfood skill is the surface where the RELEASE
#      decision is made, and its bash fences invoked a bare `apr` 43 times.
#      #2361 already taught this once (a user-scope skill shadowed the repo's
#      release-certifying skill); the skills were still unscanned.
#   3. The PATH-resolution class itself, which the old regexes could not see at
#      all - see CLASS 3 below. check_book_cli_parity.sh IS in the old scope and
#      still shipped `APR="$(which apr)"`, because laundering PATH into $APR
#      made every later use look pinned.
#   4. RUST SOURCE (#2465 finding 4). Everything above is shell. The MCP server
#      is Rust, and three of its spawn sites passed the literal "apr" to a
#      subprocess helper - so `apr mcp` from a 0.63.0 build ran whatever `apr`
#      $PATH held, which is the exact field defect #2384/#2424 were about.
#      Two of the three sat DIRECTLY BENEATH a module comment in apr_bin.rs
#      asserting that all eight subprocess tools resolved. Prose is not a
#      guard; see CLASS 4.
#
# Extending a guard's SCOPE requires re-mutating in the NEW scope; the old proof
# does not transfer. `--self-test` injects a violation into each surface in that
# surface's own syntax and asserts this script turns RED. Run it, don't read it.
# ---------------------------------------------------------------------------

set -euo pipefail

SELF_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.." || exit 1

# ---------------------------------------------------------------------------
# CLASS 1: `apr` in COMMAND POSITION - the only place it can launch a binary.
#
# Command position means: start of line, after a shell separator (; & | && ||),
# inside a substitution or subshell ( `(` covers both `(` and `$(` ), after a
# backtick, or after a YAML `run:`. Anchoring on command position rather than
# scanning line content is what keeps prose out:
#   - name: Pillar-1 - apr vs scikit-learn ...   (a label)
#   emit_pass "B2 apr qa"                        (a message)
# Two early drafts got this wrong in opposite directions - one missed
# `- run: apr qa`, the next flagged ten step names. Both were caught by the case
# table, not by reading.
#
# `@?` covers Makefile recipe lines, which start with a TAB and usually make's
# silent-prefix `@`. Without it a recipe `\t@apr qa model.apr` slipped straight
# through.
#
# WRAP covers command prefixes that do NOT change the fact that `apr` is the
# command being launched: `timeout 30 apr qa`, `nohup apr distill`,
# `APR_LOG=1 apr run`. The dogfood skill hid 19 invocations behind `timeout N`
# alone, every one of them invisible to the un-wrapped pattern.
WRAP='([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+|timeout[[:space:]]+[0-9]+[a-z]?[[:space:]]+|nohup[[:space:]]+|time[[:space:]]+|sudo[[:space:]]+|nice[[:space:]]+|env[[:space:]]+|exec[[:space:]]+|stdbuf[[:space:]]+-[^[:space:]]+[[:space:]]+)'
#
# A BACKTICK is deliberately NOT an opener, and this is a measured call rather
# than an oversight. Once markdown is in scope the backtick is overwhelmingly a
# code-span delimiter, not command substitution: every ``apr`` occurrence
# after a backtick in this tree - all 10 of them - is prose (script header
# comments, `printf '**Tool:** \`apr qualify\`'`, and a book-example string
# reading `'apr rm <id>   # from \`apr list\` output'`). Zero are the legacy
# `cmd` substitution form. Treating it as an opener produced one false positive
# and caught nothing. RESIDUAL GAP, stated rather than hidden: a genuine
# OUT=`apr qa model` would slip past. `$( )` - the form this tree actually uses,
# and the one shellcheck requires - is covered, via `(` below.
# `!` + backtick IS an opener: that pair is unambiguously "run this" in a skill,
# never a markdown code span (a span opens on a bare backtick). So the inline
# command form is covered without the false positives a bare backtick brings.
OPEN='(^|[;&|(]|&&|\|\||run:|!`)'
# The trailing class exists to require an ARGUMENT, so that bare `apr` and the
# crate name `apr-cli` do not match. It was `[a-z]` - a subcommand and nothing
# else - and each character missing from it was a silent hole:
#   `-`        `apr --version`   the single most load-bearing bare invocation
#                                there is, since reading a version off the
#                                wrong binary IS the defect. Four live ones,
#                                one of them inside qwen-story-daily's own
#                                anti-staleness step.
#   `"` `'` `$` `apr "$cmd" --help`, `apr $cmd $MODEL`  - the dogfood skill
#                                drives whole subcommand grids this way.
#   `/` `~` `.` `apr ./model.apr`, `apr ~/models/x.gguf`
# Widening it is not a loosening: the OPENER is what keeps prose out, and it is
# unchanged. This class has been wrong five times; re-run --self-test.
BARE_APR="${OPEN}[[:space:]]*@?[[:space:]]*${WRAP}*apr[[:space:]]+[a-z\"'\$/~.-]"

# ---------------------------------------------------------------------------
# CLASS 2: an ABSOLUTE hardcoded `apr` path.
#
# The other half of the same defect, and the class-1 "already pinned" allowlist
# waves it straight through: `/mnt/nvme-raid0/targets/aprender/release/apr` ends
# in `target/release/apr`. It is not pinned to anything - it names one machine's
# build output, which on 2026-08-01 was 6 days and TWO MINOR VERSIONS stale
# while docs still called it canonical. A release smoke-test read it and
# reported a meaningless pass.
#
# There is no correct absolute path to hardcode: `.cargo/config.toml` redirects
# cargo's target-dir and is gitignored, so the main checkout builds to
# /mnt/nvme-raid0/coverage/aprender while a fresh worktree builds to
# <worktree>/target. Any absolute path is right in one and silently wrong in the
# other. Use `. scripts/apr_bin.sh || exit 1`, which asks cargo.
#
# The leading anchor is load-bearing: without it `[A-Za-z0-9_.$/-]*` matches the
# `/apr` inside RELATIVE `target/release/apr`, flagging correct code. `:-` and
# `:=` are openers because `${APR:-/home/noah/.cargo/bin/apr}` is a hardcoded
# absolute path wearing a parameter-expansion costume - that exact line was live
# in check_book_cli_parity.sh, INSIDE the old scope, invisible for two
# independent reasons (the `-` before `/home` was not an opener, and the `}`
# after `apr` was not a terminator). `}` and `)` are therefore terminators.
#
# This regex class has now been gotten wrong five times in this repo. If you
# change it, re-run `--self-test` rather than reading it.
ABS_APR='(^|[[:space:]"'"'"'=(`]|:-|:=)(/|~/|\$HOME/)[A-Za-z0-9_.$/-]*/apr([[:space:]"'"'"'})]|$)'

# ---------------------------------------------------------------------------
# CLASS 3: resolving `apr` through PATH into a variable.
#
# `APR="$(which apr)"` is the ORIGINAL defect with the guard's own approved
# costume on. The binary is still whatever PATH holds - but every subsequent
# `"$APR" qa` matches the "already pinned" allowlist, so the old checker read
# the file, found nothing, and passed. Three CI-adjacent scripts shipped this:
# check_book_cli_parity.sh (a gate), check_book_examples_executable.sh (a gate),
# qualify-matrix.sh.
#
# `command -v apr` inside an `echo` is a DIAGNOSTIC, not a resolution -
# qwen-story-daily prints it precisely to name the shadowing binary. The
# echo/printf exemption below covers that.
PATHRES='(command[[:space:]]+-v[[:space:]]+apr|which[[:space:]]+apr|type[[:space:]]+(-[A-Za-z]+[[:space:]]+)?apr)([[:space:]]|[;)&|]|$)'

MIN_EXPECTED="${MIN_EXPECTED:-80}"
# Fail-closed floor for the Rust surface, and its own knob: the shell floor
# counts a few hundred files, this one counts ten thousand, so one number
# cannot serve both.
RUST_MIN_EXPECTED="${RUST_MIN_EXPECTED:-2000}"

violations=0
scanned=0
rust_scanned=0

# ---------------------------------------------------------------------------
# CLASS 4: RUST SOURCE handing a literal "apr" to something that RUNS it.
#
# #2465 finding 4. Three live sites, none of them visible to classes 1-3
# because those only ever read shell:
#
#   crates/aprender-mcp/src/tools/serve.rs     spawn_and_confirm("apr", ...)
#   crates/aprender-mcp/src/tools/run.rs       stream_with_sink("apr", ...)
#   crates/aprender-mcp/src/tools/finetune.rs  stream_with_sink("apr", ...)
#
# WHY A NAME LIST WOULD HAVE FAILED. The obvious implementation is "flag
# Command::new(\"apr\") plus a list of known spawn helpers". That list is
# exactly what was already wrong: `Command::new` appears in NONE of the three
# files. serve.rs is one hop from it, and run.rs / finetune.rs are TWO -
# `stream_with_sink` forwards to `subprocess::spawn_streaming`, which is the
# one that calls `Command::new(program)`. Anyone maintaining a hand list would
# have listed the primitives and missed all three, which is how they shipped.
#
# So the launcher set is DERIVED from the source on every run:
#
#   SEED  a fn that passes one of its OWN PARAMETERS to Command::new. That is
#         the mechanical definition of "runs a caller-supplied program".
#         `Command::new(apr_binary())` is NOT a seed - it pins internally, so
#         no caller can hand it a bare name. That distinction is the whole
#         point, and it falls out of the rule rather than being asserted.
#   EDGE  a fn that forwards one of its OWN PARAMETERS as the first argument
#         of a call. Closed to a fixpoint, this walks helper chains of any
#         depth - it is what reaches `stream_with_sink` from `Command::new`.
#   HIT   a call whose first argument is the literal "apr" and whose callee is
#         in the launcher set, plus `Command::new(... "apr" ...)` directly.
#
# The closure is computed PER CRATE. Rust resolves a bare call name inside the
# crate that can see it, and so must this: repo-wide, one generic seed named
# `execute` dragged 4,800 names into the set (`add`, `validate`, `run`), at
# which point `chat_stream_objects("apr", "hi")` - a model LABEL in an ollama
# compat test - would have been flagged as a spawn. Per crate the set is 77
# names across the tree and 6 inside aprender-mcp, with zero false positives.
#
# RESIDUAL GAPS, stated rather than hidden. A cross-crate `pub` helper called
# from another crate is not resolved (name lookup stops at the crate). A first
# argument on a different line from the callee is not matched. `let p = "apr";
# Command::new(p)` launders the literal through a local. `Command::new(&self.bin)`
# stores the program on a struct first. Each is a real hole; none of them is
# the shape of any spawn in this tree today, and CLASS 4 catching the shapes
# that ARE here is worth more than a perfect checker that does not exist.
#
# Cost: ~5s, dominated by the fixpoint. It reads ~500 of the ~10k .rs files -
# only those that mention `Command::new(`, a launcher name, or the literal
# "apr" - which is sound because every class above requires one of those
# substrings on the line.
rust_scan_program() {
    cat <<'RUST_SCAN_AWK'
function trim(s) { gsub(/^[ \t]+/, "", s); gsub(/[ \t]+$/, "", s); return s }

# Cut the line at a `//` comment. `://` (a URL) is not a comment opener.
# Load-bearing: subprocess.rs documents this very defect in a doc comment
# reading `Command::new("apr")`, and a scanner that reads prose as code would
# flag the fix's own explanation.
function strip_comment(s) {
    if (match(s, /(^|[^:\/])\/\//)) return substr(s, 1, RSTART + RLENGTH - 3)
    return s
}

# A bare call name resolves inside its crate; group by that, not repo-wide.
function crate_of(path,   n, a) {
    n = split(path, a, "/")
    if (n < 2) return "."
    if (a[1] == "crates") return a[2]
    return a[1]
}

# Parameter names of the captured signature: `ident:` with a SINGLE colon, so
# `std::path::PathBuf` contributes nothing and `program: P` contributes
# `program`. Type parameters are upper-case and excluded by the leading class.
function collect_params(sig,   rest, pos, tok) {
    delete params
    rest = sig
    while (match(rest, /[A-Za-z_][A-Za-z0-9_]*[ \t]*:/)) {
        tok = substr(rest, RSTART, RLENGTH)
        pos = RSTART + RLENGTH
        if (substr(rest, pos, 1) != ":") {
            sub(/[ \t]*:$/, "", tok)
            if (tok ~ /^[a-z_]/) params[tok] = 1
        }
        rest = substr(rest, pos)
    }
}

FNR == 1 { cur = ""; capturing = 0; depth = 0; sig = ""; delete params; pending = ""; CR = crate_of(FILENAME) }

{
    code = strip_comment($0)
    if (code ~ /^[ \t]*$/) next

    # Signatures span lines here (`pub fn spawn_streaming<P: AsRef<OsStr>, F>(`
    # ... `) -> ToolCallResult`), so capture by paren depth rather than by line.
    if (capturing) {
        sig = sig " " code
        tmp = code
        depth += gsub(/\(/, "(", tmp) - gsub(/\)/, ")", tmp)
        if (depth <= 0) { capturing = 0; collect_params(sig) }
    } else if (match(code, /(^|[^A-Za-z0-9_])fn[ \t]+[a-z_][A-Za-z0-9_]*/)) {
        head = substr(code, RSTART, RLENGTH)
        sub(/^[^A-Za-z0-9_]/, "", head)
        sub(/^fn[ \t]+/, "", head)
        cur = head
        rest = substr(code, RSTART + RLENGTH)
        p = index(rest, "(")
        if (p > 0) {
            sig = substr(rest, p)
            tmp = sig
            depth = gsub(/\(/, "(", tmp) - gsub(/\)/, ")", tmp)
            if (depth <= 0) collect_params(sig); else capturing = 1
        } else { sig = ""; delete params }
    }

    # SEED
    if (cur != "" && match(code, /Command::new[ \t]*\([ \t]*[&*]*[ \t]*[A-Za-z_][A-Za-z0-9_]*/)) {
        arg = substr(code, RSTART, RLENGTH)
        sub(/^Command::new[ \t]*\([ \t]*[&*]*[ \t]*/, "", arg)
        if (arg in params) print "SEED\t" CR "\t" cur
    }

    # EDGE
    if (cur != "" && code ~ /\(/) {
        rest = code
        while (match(rest, /[A-Za-z_][A-Za-z0-9_]*[ \t]*\([ \t]*[&*]*[ \t]*[A-Za-z_][A-Za-z0-9_]*[ \t]*[,)]/)) {
            seg = substr(rest, RSTART, RLENGTH)
            rest = substr(rest, RSTART + RLENGTH)
            callee = seg; sub(/[ \t]*\(.*$/, "", callee)
            arg = seg
            sub(/^[A-Za-z_][A-Za-z0-9_]*[ \t]*\([ \t]*[&*]*[ \t]*/, "", arg)
            sub(/[ \t]*[,)]$/, "", arg)
            if (callee ~ /^(if|while|for|match|return|let|in|fn|as|move)$/) continue
            if (arg in params) print "EDGE\t" CR "\t" callee "\t" cur
        }
    }

    # HIT, direct: the std spawn primitive, whatever wraps the literal
    # (`Command::new("apr")`, `Command::new(OsStr::new("apr"))`).
    if (code ~ /Command::new[ \t]*\([^)]*"apr"/) print "CMDNEW\t" CR "\t" FILENAME "\t" FNR "\t" trim($0)

    # HIT, candidate: `<callee>("apr", ...)`. Whether it is a violation is
    # decided against the derived launcher set, not here.
    #
    # `X::new("apr")` CONSTRUCTS a value - `OsStr::new("apr")`,
    # `PathBuf::from("apr")` (the documented tier-3 fallback in apr_bin.rs) -
    # and the one launcher spelled `new` is Command::new, handled above.
    rest = code
    while (match(rest, /[A-Za-z_][A-Za-z0-9_]*[ \t]*\([ \t]*"apr"[ \t]*[,)]/)) {
        seg = substr(rest, RSTART, RLENGTH)
        rest = substr(rest, RSTART + RLENGTH)
        callee = seg; sub(/[ \t]*\(.*$/, "", callee)
        if (callee == "new") continue
        print "CALL\t" CR "\t" FILENAME "\t" FNR "\t" callee "\t" trim($0)
    }

    # A call whose ARGUMENTS start on the next line - `callee(` at end of line,
    # `"apr",` under it. Not a hypothetical: rustfmt writes serve.rs`s call
    # that way as soon as the argument list grows, so re-mutating the real fix
    # produced exactly this shape and the single-line matcher above stayed
    # GREEN on a live bare-"apr" spawn. One line of lookahead closes it; a
    # third line of separation still slips (stated, not hidden).
    if (pending != "" && match(code, /^[ \t]*"apr"[ \t]*[,)]/)) {
        if (pending == "Command::new") print "CMDNEW\t" CR "\t" FILENAME "\t" FNR "\t" trim($0)
        else print "CALL\t" CR "\t" FILENAME "\t" FNR "\t" pending "\t" trim($0)
    }
    if (pending != "" && match(code, /^[ \t]*[&*]*[ \t]*[A-Za-z_][A-Za-z0-9_]*[ \t]*[,)]/)) {
        arg = trim(substr(code, RSTART, RLENGTH))
        sub(/[ \t]*[,)]$/, "", arg); sub(/^[&*]+[ \t]*/, "", arg)
        if (cur != "" && (arg in params) && pending != "Command::new") {
            print "EDGE\t" CR "\t" pending "\t" cur
        }
        if (cur != "" && (arg in params) && pending == "Command::new") {
            print "SEED\t" CR "\t" cur
        }
    }
    pending = ""
    if (code ~ /Command::new[ \t]*\([ \t]*$/) pending = "Command::new"
    else if (match(code, /[A-Za-z_][A-Za-z0-9_]*[ \t]*\([ \t]*$/)) {
        seg = trim(substr(code, RSTART, RLENGTH))
        sub(/[ \t]*\($/, "", seg)
        if (seg != "new" && seg !~ /^(if|while|for|match|return|let|in|fn|as|move)$/) pending = seg
    }
}
RUST_SCAN_AWK
}

# Every .rs in the tree. `target/` is build output; `.claude/` is pruned
# because worktrees live under `.claude/worktrees/<id>/` in the main checkout,
# and scanning them would report another branch's code as this branch's
# violations.
rust_surface_files() {
    find . \( -name target -o -name .git -o -name node_modules -o -name .claude \) -prune \
        -o -name '*.rs' -type f -print 2>/dev/null | sed 's|^\./||' | sort
}

# Same crate_of as the awk, for filtering the candidate file list.
rust_crate_of() {
    awk -F'/' '{ if (NF < 2) print "."; else if ($1 == "crates") print $2; else print $1 }'
}

# ---------------------------------------------------------------------------
# CLASS-4 BASELINE - pre-existing violations, enumerated, at the moment the
# class was added (#2465). RATCHET: this list may only SHRINK. A file not
# listed here fails the build; a listed file with MORE hits fails; a listed
# file with FEWER hits fails too, telling you to lower the number. That last
# rule is what stops the list becoming permanent furniture.
#
# These are NOT waivers on the merits. Every one is the same defect as the
# three #2465 fixed, and two of them are worse: aprender-qa-runner's
# `get_apr_cli_version()` reports a version read off a $PATH binary, which is
# precisely the provenance defect of the 0.63.0 dogfood, and
# falsify_mcp_e2e_001 asserts the MCP tool matches "the CLI" while resolving
# the two sides through DIFFERENT binaries. They are listed rather than fixed
# because each needs its own falsifier (a resolver for qa-runner, a pinned
# CARGO_BIN_EXE for the tests) and #2465 finding 4 is scoped to aprender-mcp.
rust_baseline() {
    cat <<'RUST_BASELINE'
2	crates/apr-cli/tests/falsification_bert_326_embed_parity.rs
2	crates/apr-cli/tests/falsification_bert_326_hf_parity.rs
1	crates/aprender-core/examples/gpu_fallback_dogfood.rs
1	crates/aprender-mcp/tests/falsify_mcp_e2e_001.rs
1	crates/aprender-qa-runner/src/diagnostics_markdown_gen.rs
2	crates/aprender-qa-runner/src/executor_tools_backend_equivalence.rs
1	crates/aprender-qa-runner/src/executor_serve_lifecycle.rs
1	crates/aprender-qa-runner/src/provenance_utilities.rs
RUST_BASELINE
}

# Global, not a `local`: the EXIT trap below runs after the function has
# returned, and a trap that dereferences a dead local is an `unbound variable`
# abort under `set -u` - which this script hit, AFTER printing OK.
RUST_TMP=""
cleanup_rust_tmp() {
    if [ -n "${RUST_TMP:-}" ] && [ "$RUST_TMP" != / ] && [ -d "$RUST_TMP" ]; then
        rm -rf "$RUST_TMP"
    fi
}

check_rust_surface() {
    local tmp scan rs_list names round before after

    RUST_TMP="$(mktemp -d)"
    trap cleanup_rust_tmp EXIT
    tmp="$RUST_TMP"
    scan="$tmp/scan.awk"
    rust_scan_program > "$scan"

    rs_list="$tmp/rs.list"
    rust_surface_files > "$rs_list"
    rust_scanned="$(wc -l < "$rs_list" | tr -d ' ')"
    if [ "$rust_scanned" -eq 0 ]; then
        : > "$tmp/violations"
        return 0
    fi

    # Phase A - seeds. Only a file that mentions Command::new can hold one.
    : > "$tmp/launchers"
    if xargs grep -l 'Command::new(' < "$rs_list" > "$tmp/cmdnew.files" 2>/dev/null &&
        [ -s "$tmp/cmdnew.files" ]; then
        xargs awk -f "$scan" < "$tmp/cmdnew.files" |
            awk -F'\t' '$1 == "SEED" { print $2 "\t" $3 }' | sort -u > "$tmp/launchers"
    fi

    # Phase B - forwarding closure, per crate, to a fixpoint.
    for round in 1 2 3 4 5 6 7 8; do
        before="$(wc -l < "$tmp/launchers" | tr -d ' ')"
        [ "$before" -gt 0 ] || break
        names="$(cut -f2 "$tmp/launchers" | sort -u | paste -sd'|' -)"
        cut -f1 "$tmp/launchers" | sort -u > "$tmp/crates"
        awk 'NR == FNR { c[$0] = 1; next }
             { k = $0; n = split(k, a, "/");
               key = (n < 2) ? "." : ((a[1] == "crates") ? a[2] : a[1]);
               if (key in c) print }' "$tmp/crates" "$rs_list" > "$tmp/cand.files"
        : > "$tmp/fwd.files"
        if [ -s "$tmp/cand.files" ]; then
            xargs grep -lE "(^|[^A-Za-z0-9_])(${names})[ 	]*\(" < "$tmp/cand.files" \
                > "$tmp/fwd.files" 2>/dev/null || true
        fi
        [ -s "$tmp/fwd.files" ] || break
        xargs awk -f "$scan" < "$tmp/fwd.files" | awk -F'\t' '$1 == "EDGE"' | sort -u > "$tmp/edges"
        awk -F'\t' -v LF="$tmp/launchers" '
            BEGIN { while ((getline l < LF) > 0) { split(l, f, "\t"); L[f[1] SUBSEP f[2]] = 1 } }
            ($2 SUBSEP $3) in L { L[$2 SUBSEP $4] = 1 }
            END { for (k in L) { split(k, f, SUBSEP); print f[1] "\t" f[2] } }
        ' "$tmp/edges" | sort -u > "$tmp/launchers.next"
        mv "$tmp/launchers.next" "$tmp/launchers"
        after="$(wc -l < "$tmp/launchers" | tr -d ' ')"
        [ "$before" != "$after" ] || break
    done

    # Phase C - call sites. Every class requires the literal on the line.
    : > "$tmp/sites"
    if xargs grep -l '"apr"' < "$rs_list" > "$tmp/apr.files" 2>/dev/null &&
        [ -s "$tmp/apr.files" ]; then
        xargs awk -f "$scan" < "$tmp/apr.files" |
            awk -F'\t' '$1 == "CALL" || $1 == "CMDNEW"' > "$tmp/sites"
    fi

    awk -F'\t' -v LF="$tmp/launchers" '
        BEGIN { while ((getline l < LF) > 0) { split(l, f, "\t"); L[f[1] SUBSEP f[2]] = 1 } }
        $1 == "CMDNEW" { print $3 "\t" $4 "\t" $5; next }
        $1 == "CALL" && (($2 SUBSEP $5) in L) { print $3 "\t" $4 "\t" $6 }
    ' "$tmp/sites" | sort -u > "$tmp/violations"

    rust_baseline > "$tmp/baseline"

    # Reconciliation goes to a FILE, not into a process substitution: a `<(awk
    # ...)` that dies takes its diagnostic to stderr and hands the loop an
    # empty stream, so the guard reports zero violations and prints OK. That
    # is precisely the shape of failure this whole script exists to prevent,
    # and the first draft of this function shipped it. Redirected into a file
    # under `set -e`, an awk that dies stops the script instead.
    awk -F'\t' -v BL="$tmp/baseline" -v RL="$rs_list" '
        BEGIN {
            S = sprintf("%c", 31)
            while ((getline l < BL) > 0) { split(l, bf, "\t"); B[bf[2]] = bf[1] }
            while ((getline l < RL) > 0) P[l] = 1
        }
        { cnt[$1]++; if (!($1 in firstline)) { firstline[$1] = $2; firsttext[$1] = $3 } }
        END {
            for (v in cnt) {
                seen[v] = 1
                if (!(v in B))          print "NEW" S v S S firstline[v] S firsttext[v]
                else if (cnt[v] > B[v]) print "GREW" S v S cnt[v] S B[v] S firstline[v] ": " firsttext[v]
                else if (cnt[v] < B[v]) print "RATCHET" S v S cnt[v] S B[v] S
            }
            for (b in B) if (!(b in seen) && (b in P)) print "FIXED" S b S S S
        }
    ' "$tmp/violations" > "$tmp/reconciled"

    # A baseline entry whose file is absent
    # from the scanned tree is SKIPPED, not failed: the --self-test probes run
    # this scanner over three-file trees, and failing there would make every
    # probe go RED for a reason that has nothing to do with what it probes.
    # Staleness of the paths themselves is checked by --self-test, against the
    # real repository.
    # US (0x1f), not TAB. Tab is IFS *whitespace*, so bash collapses a run of
    # them into one delimiter and drops empty fields: with `\t` the NEW record
    # `NEW<TAB>file<TAB><TAB>line<TAB>text` arrived as four fields and the
    # report printed the source line where the line NUMBER belongs. A
    # non-whitespace IFS preserves empty fields.
    while IFS=$'\037' read -r kind file detail line text; do
        case "$kind" in
            NEW)
                report 'RUST-APR' "$file" "$line" "$text"
                ;;
            GREW)
                printf 'RUST-APR %s: %s violation(s), baseline allows %s\n' "$file" "$detail" "$line"
                printf '         newest: %s\n' "$text"
                violations=$((violations + 1))
                ;;
            RATCHET)
                printf 'RUST-APR %s: %s violation(s) left, baseline still says %s.\n' \
                    "$file" "$detail" "$line" >&2
                printf '         Lower the number in rust_baseline() - the list may only shrink.\n' >&2
                violations=$((violations + 1))
                ;;
            FIXED)
                printf 'RUST-APR %s: no violations left; remove its rust_baseline() entry.\n' \
                    "$file" >&2
                violations=$((violations + 1))
                ;;
        esac
    done < "$tmp/reconciled"
}

# ---------------------------------------------------------------------------
# Surfaces. Everything that can EXECUTE apr, with no reachability guessing.
#
# docs/ and book/ are deliberately NOT here. `apr run model.apr` in a model card
# is instruction to a human who installed the binary; there is one `apr` on
# their machine and pinning prose to a worktree path would be nonsense. The
# invariant is about surfaces that run apr and then REPORT A VERDICT.
surface_files() {
    local f
    for f in .github/workflows/*.yml .github/workflows/*.yaml; do
        [ -f "$f" ] && printf '%s\n' "$f"
    done
    [ -f Makefile ] && printf '%s\n' Makefile
    find scripts -name '*.sh' -type f 2>/dev/null | sort
    find .claude/skills -name '*.md' -type f 2>/dev/null | sort
}

# Lines a surface can EXECUTE, as `lineno:text`.
#
# For markdown (skills) that is the content of bash fences only. A skill's prose
# says things like "run apr qa first" at the start of a line, which is command
# position by syntax and documentation by intent. Scanning the fences keeps the
# check on the text an agent actually pastes into a shell.
# The `/apr/` pre-filter is sound, not a shortcut: all three classes require the
# literal substring `apr` somewhere on the line, so a line without it cannot be a
# violation. It takes the scan from ~30k bash-regex evaluations to a few hundred
# (12.6s -> under a second), which matters because this runs on every PR.
emit_lines() {
    local f="$1"
    case "$f" in
        *.md)
            # Two executable forms in a skill, and only the first is obvious:
            #   1. ```bash fences  - what an agent pastes into a shell.
            #   2. !`...` lines    - the skill front-matter's inline command
            #                        substitution, which Claude Code RUNS when
            #                        the skill loads. apr-dogfood used one to
            #                        report "Installed apr version" from a bare
            #                        `apr --version`, i.e. the PATH binary, in
            #                        the header of the protocol that certifies
            #                        releases.
            awk '
                /^[[:space:]]*```/ {
                    if (infence) { infence = 0 }
                    else if ($0 ~ /```(bash|sh|shell|console)[[:space:]]*$/) { infence = 1 }
                    next
                }
                infence && /apr/ { printf "%d:%s\n", NR, $0; next }
                /!`/ && /apr/ { printf "%d:%s\n", NR, $0 }
            ' "$f"
            ;;
        *)
            awk '/apr/ { printf "%d:%s\n", NR, $0 }' "$f"
            ;;
    esac
}

# Strip leading whitespace, into $LTRIM. A bash-regex capture rather than the
# nested `${t#"${t%%[![:space:]]*}"}` idiom: same result, one expansion, and it
# does not fork a process per line (this scans ~30k lines).
ltrim() {
    [[ $1 =~ ^[[:space:]]*(.*)$ ]]
    LTRIM="${BASH_REMATCH[1]}"
}

# Quoted text handed to echo/printf is a MESSAGE, not a command. Stripping the
# quoted segments (and only for echo/printf lines) keeps
#   echo 'MODEL_DIR is a FILE; apr publish needs a directory'
# out while keeping
#   echo '{"jsonrpc":"2.0"}' | apr mcp
# in - the pipe and the invocation are OUTSIDE the quotes.
demessage() {
    local t="$1"
    case "$t" in
        echo\ *|printf\ *|echo|printf)
            printf '%s' "$t" | sed "s/'[^']*'//g; s/\"[^\"]*\"//g"
            ;;
        *) printf '%s' "$t" ;;
    esac
}

report() {
    printf '%s %s:%s\n' "$1" "$2" "$3"
    printf '         %s\n' "$4"
    violations=$((violations + 1))
}

check_file() {
    local f="$1" hit lineno text trimmed probe self=0
    scanned=$((scanned + 1))
    # This file quotes deliberate violations verbatim in its case table, and
    # apr_bin.sh names the stale absolute paths it exists to detect. Both are
    # data, not invocations. (check_pass_grep_anchored.sh exempts itself for the
    # same reason.) The case table is what proves these two are still honest.
    case "$f" in */apr_bin.sh|*/check_apr_bin_pinned.sh) self=1 ;; esac
    [ "$self" -eq 1 ] && return 0
    while IFS= read -r hit; do
        lineno="${hit%%:*}"
        text="${hit#*:}"
        ltrim "$text"; trimmed="$LTRIM"

        # Comments are documentation, not invocations.
        case "$trimmed" in '#'*) continue ;; esac

        # YAML metadata is prose, not shell. beat-speed-nightly.yml has ten step
        # names reading "Pillar-1 - apr vs scikit-learn ..."; flagging those
        # would make the check fire on its own labels.
        case "$trimmed" in
            name:*|-\ name:*|description:*|title:*|summary:*|if:*|-\ if:*|id:*|uses:*|shell:*|working-directory:*)
                continue ;;
        esac

        probe="$(demessage "$trimmed")"

        # CLASS 1 --------------------------------------------------------
        if [[ $probe =~ $BARE_APR ]]; then
            case "$text" in
                *'$APR'*|*'${APR'*|*'target/release/apr'*|*'target/debug/apr'*|*'.cargo/bin/apr'*|*'--bin apr'*)
                    : ;;
                # qwen-story.sh's run_cmd substitutes a leading bare `apr` with
                # "$APR", so its call sites are pinned by the wrapper.
                *'run_cmd '*) : ;;
                *) report 'BARE-APR' "$f" "$lineno" "$trimmed" ;;
            esac
        fi

        # CLASS 2 --------------------------------------------------------
        if [[ $trimmed =~ $ABS_APR ]]; then
            report 'ABS-APR ' "$f" "$lineno" "$trimmed"
        fi

        # CLASS 3 --------------------------------------------------------
        # apr_bin.sh probes `type -aP apr` on purpose, to NAME the shadows;
        # it is exempted wholesale above.
        if [[ $probe =~ $PATHRES ]]; then
            report 'PATH-APR' "$f" "$lineno" "$trimmed"
        fi
    done < <(emit_lines "$f")
}

# ---------------------------------------------------------------------------
# --self-test: the must-match / must-not-match case table, plus one mutation
# per SURFACE in that surface's own syntax.
#
# The surface half is not decoration. When this guard grew from "workflows" to
# "workflows + Makefile", the Makefile recipe form `\t@apr qa` was still
# invisible - the scope moved, the proof did not, and only re-mutating inside
# the Makefile found it. Every surface added since gets its own probe here.
if [ "${1:-}" = "--self-test" ]; then
    fails=0

    # -- regex case table ---------------------------------------------------
    must_match_bare=(
        'apr qa model.apr'
        '  apr run model.gguf --prompt hi'
        '- run: apr qa model.apr'
        '	@apr qa model.apr'
        '	apr validate model.apr'
        'cd /tmp && apr qa model.apr'
        'foo; apr lint model.apr'
        'echo hi | apr mcp'
        'OUT=$(apr inspect model.apr)'
        'OUT=$(timeout 10 apr inspect model.apr)'
        '  echo -n x && timeout 30 apr run $M'
        'nohup apr distill teacher.apr &'
        'APR_LOG=1 apr run model.apr'
        'env APR_LOG=1 apr run model.apr'
        'echo {} | apr mcp'
        '- Installed apr version: !`apr --version 2>/dev/null`'
        'apr --version | grep -qE unknown'
        'apr "$cmd" --help 2>&1 | grep -qi not-implemented'
        'apr $cmd $MODEL 2>&1 | head -1'
        'apr ./model.apr qa'
        'apr ~/models/x.gguf qa'
    )
    must_not_match_bare=(
        '"$APR" qa model.apr'
        '${APR} qa model.apr'
        '"$APR_BIN" qa model.apr'
        './target/release/apr qa model.apr'
        '"${REPO_ROOT}/target/release/apr" qa model.apr'
        'cargo run --bin apr -- qa model.apr'
        '- name: Pillar-1 - apr vs scikit-learn parity'
        '# apr qa model.apr'
        'emit_pass "B2 apr qa"'
        'run_cmd apr qa model.apr'
        'cargo install apr-cli --force'
        'echo "  apr pull ${REPO} && apr qa <path>"'
        "echo 'MODEL_DIR is a FILE; apr publish needs a directory'"
        'aprender-core builds apr eventually'
        'ls scripts/apr_bin.sh'
        # Markdown/prose code spans. A backtick is not an opener - see OPEN.
        "EXAMPLE[rm]='apr rm <id>   # from \`apr list\` output'"
        "printf '**Tool:** \`apr qualify\` (11-gate smoke test)\\n'"
    )
    must_match_abs=(
        '/mnt/nvme-raid0/targets/aprender/release/apr publish x'
        'APR="${APR:-/home/noah/.cargo/bin/apr}"'
        'APR_BINARY="${APR_BINARY:-/mnt/nvme-raid0/targets/aprender/release/apr}"'
        'elif [ -x /mnt/nvme-raid0/targets/aprender/release/apr ]; then'
        'APR_BIN=/mnt/nvme-raid0/targets/aprender/release/apr'
        '~/.cargo/bin/apr --version'
        '$HOME/.cargo/bin/apr --version'
    )
    must_not_match_abs=(
        './target/release/apr --version'
        'APR_BIN="${APR_BIN:-${REPO_DIR}/target/release/apr}"'
        'readonly APR_BIN="${PROJECT_ROOT}/target/release/apr"'
        'cp "target/${{ matrix.target }}/release/apr" "$ARCHIVE/"'
        'APR_BIN_PATH="${CARGO_HOME:-$HOME/.cargo}/bin/apr"'
        'bash scripts/apr_bin.sh'
        'cargo run --bin apr'
        'ls /mnt/nvme-raid0/targets/aprender/release/apr-cli'
    )
    must_match_pathres=(
        'APR="$(which apr)"'
        'APR_BIN="$(command -v apr)"'
        'if command -v apr >/dev/null 2>&1; then'
        'if ! command -v apr; then'
    )
    must_not_match_pathres=(
        'echo "apr --version: $GOT ($(command -v apr))"'
        'echo "::error::PATH resolves to $(command -v apr)."'
        'command -v cargo >/dev/null'
        'which apr-cli'
        'command -v aprender'
    )

    probe_case() {
        local re="$1" line="$2" want="$3" label="$4" p t
        # Mirror check_file's own pre-filters exactly, or the table would be
        # testing a different pipeline than the one that ships.
        ltrim "$line"; t="$LTRIM"
        p="$(demessage "$t")"
        case "$t" in '#'*) p='' ;; esac
        case "$t" in name:*|-\ name:*) p='' ;; esac
        if [ "$want" = match ]; then
            if ! [[ $p =~ $re ]]; then
                printf 'CASE-TABLE FAIL [%s] expected MATCH, got none: %s\n' "$label" "$line" >&2
                fails=$((fails + 1))
            fi
        else
            if [[ $p =~ $re ]]; then
                printf 'CASE-TABLE FAIL [%s] expected NO match, got one: %s\n' "$label" "$line" >&2
                fails=$((fails + 1))
            fi
        fi
    }

    for c in "${must_match_bare[@]}"; do probe_case "$BARE_APR" "$c" match bare; done
    for c in "${must_not_match_bare[@]}"; do
        # The allowlist is part of the class-1 decision, so apply it here too.
        case "$c" in
            *'$APR'*|*'${APR'*|*'target/release/apr'*|*'target/debug/apr'*|*'.cargo/bin/apr'*|*'--bin apr'*|*'run_cmd '*)
                continue ;;
        esac
        probe_case "$BARE_APR" "$c" nomatch bare
    done
    for c in "${must_match_abs[@]}"; do probe_case "$ABS_APR" "$c" match abs; done
    for c in "${must_not_match_abs[@]}"; do probe_case "$ABS_APR" "$c" nomatch abs; done
    for c in "${must_match_pathres[@]}"; do probe_case "$PATHRES" "$c" match pathres; done
    for c in "${must_not_match_pathres[@]}"; do probe_case "$PATHRES" "$c" nomatch pathres; done

    # -- per-surface mutation ----------------------------------------------
    TMPROOT=$(mktemp -d)
    cleanup_selftest() {
        if [ -n "${TMPROOT:-}" ] && [ "$TMPROOT" != / ] && [ -d "$TMPROOT" ]; then
            rm -rf "$TMPROOT"
        fi
    }
    trap cleanup_selftest EXIT

    # A FRESH tree per probe. The first draft reused one directory and cleaned it
    # with `rm -rf "$TMP"/*`, which does not match dotfiles - so `.github/` and
    # `.claude/` survived, every later probe inherited the FIRST probe's
    # violation, and all of them "turned RED" without their own mutation ever
    # being read. Probes that pass for a reason other than the thing they probe
    # are the exact failure mode this script exists to prevent; the guard's own
    # test is not exempt. Only the prose probe (which must stay GREEN) noticed.
    mk_tree() {
        local d
        d=$(mktemp -d "$TMPROOT/probe.XXXXXX")
        mkdir -p "$d/scripts"
        cp "$SELF_PATH" "$d/scripts/check_apr_bin_pinned.sh"
        : > "$d/Makefile"
        printf '%s' "$d"
    }

    # A probe tree holds no Rust, so its CLASS-4 floor is 0. Leaving the real
    # floor in place would fail every probe on "the scan has gone blind" - a
    # RED for a reason unrelated to what the probe injects, which is how a
    # probe starts passing without ever testing anything. `rust-blind` below
    # exercises the floor on purpose.
    surface_probe() {
        local label="$1" path="$2" content="$3" d
        d=$(mk_tree)
        mkdir -p "$d/$(dirname "$path")"
        printf '%s\n' "$content" > "$d/$path"
        if (cd "$d" && MIN_EXPECTED=1 RUST_MIN_EXPECTED=0 \
            bash scripts/check_apr_bin_pinned.sh >/dev/null 2>&1); then
            printf 'SURFACE-PROBE FAIL [%s]: guard stayed GREEN with a violation in %s\n' \
                "$label" "$path" >&2
            fails=$((fails + 1))
        fi
    }

    # CLASS 4 needs whole Rust FILES, not one line: the derivation has to find
    # a `Command::new(<param>)` and walk the call chain to the literal.
    # `want` is `red` (the guard must flag it) or `green` (it must not).
    # A RED probe must go red FOR THE CLASS. Exit status alone is not enough
    # and this is not hypothetical: with `check_rust_surface` deleted, all
    # three red probes still "passed", because a tree with no Rust trips the
    # fail-closed floor and exits 1. Three probes that could never fail, in
    # the very commit that added them. So the assertion is on the RUST-APR
    # finding naming the file, which only CLASS 4 can emit.
    rust_probe() {
        local label="$1" want="$2" d rc out
        shift 2
        d=$(mk_tree)
        while [ "$#" -gt 1 ]; do
            mkdir -p "$d/$(dirname "$1")"
            printf '%s\n' "$2" > "$d/$1"
            shift 2
        done
        rc=0
        out="$(cd "$d" && MIN_EXPECTED=1 RUST_MIN_EXPECTED=1 \
            bash scripts/check_apr_bin_pinned.sh 2>&1)" || rc=$?
        if [ "$want" = red ]; then
            if [ "$rc" -eq 0 ]; then
                printf 'RUST-PROBE FAIL [%s]: guard stayed GREEN with a bare "apr" spawn\n' "$label" >&2
                fails=$((fails + 1))
            elif ! printf '%s' "$out" | grep -q '^RUST-APR '; then
                printf 'RUST-PROBE FAIL [%s]: guard exited %s but reported no RUST-APR finding -\n' \
                    "$label" "$rc" >&2
                printf '           it went red for some OTHER reason, so this probe proves nothing:\n' >&2
                printf '%s\n' "$out" | sed 's/^/           /' >&2
                fails=$((fails + 1))
            fi
        elif [ "$rc" -ne 0 ]; then
            printf 'RUST-PROBE FAIL [%s]: guard flagged Rust that spawns nothing\n' "$label" >&2
            printf '%s\n' "$out" | sed 's/^/           /' >&2
            fails=$((fails + 1))
        fi
    }

    # Each violation is written in the SURFACE'S OWN syntax. The Makefile case is
    # the reason this section exists: a tab-and-@ recipe line is not a shell line
    # and the pattern that covered every workflow was blind to it.
    surface_probe 'workflow'  '.github/workflows/w.yml' \
        '      - name: x
        run: apr qa model.apr'
    surface_probe 'makefile'  'Makefile' \
        'qa:
	@apr qa model.apr'
    surface_probe 'script-indirect' 'scripts/never-named-by-a-workflow.sh' \
        '#!/usr/bin/env bash
apr qa model.apr'
    surface_probe 'script-nested' 'scripts/pub/deep.sh' \
        '#!/usr/bin/env bash
/mnt/nvme-raid0/targets/aprender/release/apr publish x'
    surface_probe 'skill-fence' '.claude/skills/x/SKILL.md' \
        'Run the gate:

```bash
apr qa model.apr
```'
    surface_probe 'skill-wrapped' '.claude/skills/x/SKILL.md' \
        '```bash
timeout 30 apr run $MODEL --max-tokens 4
```'
    surface_probe 'skill-inline-bang' '.claude/skills/x/SKILL.md' \
        '- Installed apr version: !`apr --version 2>/dev/null || echo none`'
    surface_probe 'pathres-launder' 'scripts/launder.sh' \
        '#!/usr/bin/env bash
APR="$(which apr)"
"$APR" qa model.apr'
    surface_probe 'abs-default' 'scripts/absdefault.sh' \
        '#!/usr/bin/env bash
APR="${APR:-/home/noah/.cargo/bin/apr}"'

    # A skill's PROSE is not an execution surface; only its bash fences are.
    # This asserts the fence filter, so the guard cannot start policing English.
    prose_dir=$(mk_tree)
    mkdir -p "$prose_dir/.claude/skills/x"
    printf 'apr qa model.apr is the first tool to reach for.\n' \
        > "$prose_dir/.claude/skills/x/SKILL.md"
    if ! (cd "$prose_dir" && MIN_EXPECTED=1 RUST_MIN_EXPECTED=0 \
        bash scripts/check_apr_bin_pinned.sh >/dev/null 2>&1); then
        printf 'SURFACE-PROBE FAIL [skill-prose]: guard flagged markdown prose outside a bash fence\n' >&2
        fails=$((fails + 1))
    fi

    # -- CLASS 4: re-mutation in the NEW scope ------------------------------
    # The whole point of #2360's lesson. These are not the shell probes with
    # different text; they inject Rust into a Rust surface, in the shapes the
    # three #2465 sites actually had.

    # The std primitive, direct.
    rust_probe 'rust-command-new' red \
        'crates/x/src/lib.rs' 'use std::process::Command;
pub fn go() {
    let _ = Command::new("apr").arg("qa").output();
}'

    # ONE hop - serve.rs`s shape. `spawn_it` is a name this script has never
    # heard of; it is a launcher because it hands a PARAMETER to Command::new.
    rust_probe 'rust-derived-launcher' red \
        'crates/x/src/lib.rs' 'use std::ffi::OsStr;
use std::process::Command;
pub fn spawn_it<P: AsRef<OsStr>>(program: P, args: &[String]) -> bool {
    Command::new(program).args(args).status().is_ok()
}
pub fn go(argv: &[String]) -> bool {
    spawn_it("apr", argv)
}'

    # TWO hops, across FILES - run.rs / finetune.rs. This is the shape #2424
    # missed: neither file contains `Command::new`, and `stream_with_sink` is
    # not a name any list would have carried.
    rust_probe 'rust-forwarder-chain' red \
        'crates/x/src/sub.rs' 'use std::ffi::OsStr;
use std::process::Command;
pub fn spawn_streaming<P: AsRef<OsStr>>(program: P, args: &[&str]) -> bool {
    Command::new(program).args(args).status().is_ok()
}' \
        'crates/x/src/tool.rs' 'use crate::sub::spawn_streaming;
pub fn stream_with_sink(program: &str, args: &[&str]) -> bool {
    spawn_streaming(program, args)
}
pub fn go(argv: &[&str]) -> bool {
    stream_with_sink("apr", argv)
}'

    # Arguments on the NEXT line. rustfmt produces this the moment the call
    # gets long, and re-mutating serve.rs`s real fix produced it verbatim -
    # the guard stayed green on a live bare-"apr" spawn until the lookahead
    # above existed. The mutation found this; reading the pattern did not.
    rust_probe 'rust-multiline-arg' red \
        'crates/x/src/lib.rs' 'use std::ffi::OsStr;
use std::process::Command;
pub fn spawn_it<P: AsRef<OsStr>>(program: P, args: &[String], port: u16) -> bool {
    let _ = port;
    Command::new(program).args(args).status().is_ok()
}
pub fn go(argv: &[String]) -> bool {
    spawn_it(
        "apr",
        argv,
        8080,
    )
}'

    # Must stay GREEN, all in one file so a single false positive fails it:
    # a model LABEL that happens to read "apr" (ollama-compat tests pass one
    # to `chat_stream_objects`), the CONSTRUCTORS `OsStr::new` / `PathBuf::from`
    # (the latter IS apr_bin.rs`s documented tier-3 fallback, and flagging it
    # would make the resolver unable to contain its own fallback), a DOC
    # COMMENT quoting the defect verbatim - subprocess.rs really does carry
    # that line - and a helper that pins internally, which no caller can
    # redirect. Repo-wide (no per-crate scoping) the first of these was a
    # live false positive.
    rust_probe 'rust-not-a-spawn' green \
        'crates/x/src/lib.rs' 'use std::ffi::OsStr;
use std::path::PathBuf;
use std::process::Command;

/// Must execute the resolved binary, not `Command::new("apr")`.
pub fn apr_binary() -> PathBuf {
    std::env::var_os("APR_BIN").map_or_else(|| PathBuf::from("apr"), PathBuf::from)
}
pub fn run_apr(args: &[&str]) -> bool {
    Command::new(apr_binary()).args(args).status().is_ok()
}
pub fn label(model: &str, text: &str) -> String {
    format!("{model}:{text}")
}
pub fn demo() -> String {
    let _ = OsStr::new("apr");
    label("apr", "the capital of France is Paris")
}'

    # Fail-closed: a scanner that found no Rust must not report success. The
    # floor is what turns a broken `find` into a RED instead of a green run
    # over nothing.
    blind_dir=$(mk_tree)
    blind_rc=0
    blind_out="$(cd "$blind_dir" && MIN_EXPECTED=1 RUST_MIN_EXPECTED=1 \
        bash scripts/check_apr_bin_pinned.sh 2>&1)" || blind_rc=$?
    if [ "$blind_rc" -eq 0 ]; then
        printf 'RUST-PROBE FAIL [rust-blind]: guard passed having scanned ZERO .rs files\n' >&2
        fails=$((fails + 1))
    elif ! printf '%s' "$blind_out" | grep -q 'Rust discovery has gone blind'; then
        printf 'RUST-PROBE FAIL [rust-blind]: exited %s, but not on the blind-scan check\n' \
            "$blind_rc" >&2
        fails=$((fails + 1))
    fi

    # The baseline names real paths. A renamed or deleted file would leave an
    # entry that can never be satisfied, and the scan skips absent files (see
    # check_rust_surface), so nothing else would ever notice.
    while IFS=$'\t' read -r _count path; do
        [ -n "$path" ] || continue
        if [ ! -f "$path" ]; then
            printf 'BASELINE FAIL: rust_baseline() names %s, which does not exist\n' "$path" >&2
            fails=$((fails + 1))
        fi
    done < <(rust_baseline)

    if [ "$fails" -gt 0 ]; then
        printf '\nself-test FAILED with %s case(s).\n' "$fails" >&2
        exit 1
    fi
    printf 'self-test OK: %s regex cases, 9 shell surface probes, 6 Rust probes.\n' \
        "$(( ${#must_match_bare[@]} + ${#must_not_match_bare[@]} + ${#must_match_abs[@]} \
             + ${#must_not_match_abs[@]} + ${#must_match_pathres[@]} + ${#must_not_match_pathres[@]} ))"
    exit 0
fi

while IFS= read -r f; do
    [ -n "$f" ] || continue
    check_file "$f"
done < <(surface_files)

check_rust_surface

if [ "$violations" -gt 0 ]; then
    printf '\n%s unpinned `apr` reference(s) on execution surfaces (%s shell, %s Rust file(s) scanned).\n' \
        "$violations" "$scanned" "$rust_scanned" >&2
    printf 'A bare `apr` runs whatever PATH resolves - which is how a 24-day-old\n' >&2
    printf 'binary validated a gate merged the day before. An absolute path names\n' >&2
    printf 'one machine. `$(which apr)` is PATH wearing a pinned costume. Pin it:\n' >&2
    printf '  . scripts/apr_bin.sh    # exports $APR, asserts it was built from HEAD\n' >&2
    printf '  "$APR" qa model.gguf\n' >&2
    printf 'In Rust, spawn crate::apr_bin::apr_binary() - never the literal "apr",\n' >&2
    printf 'which is `$PATH` again with a type on it (#2465).\n' >&2
    exit 1
fi

# Fail closed: a scanner that examined nothing must not report success.
if [ "$scanned" -lt "$MIN_EXPECTED" ]; then
    printf 'ERROR: scanned %s file(s), expected >= %s - the file discovery has gone blind.\n' \
        "$scanned" "$MIN_EXPECTED" >&2
    exit 1
fi
if [ "$rust_scanned" -lt "$RUST_MIN_EXPECTED" ]; then
    printf 'ERROR: scanned %s .rs file(s), expected >= %s - the Rust discovery has gone blind.\n' \
        "$rust_scanned" "$RUST_MIN_EXPECTED" >&2
    exit 1
fi

printf 'OK: %s execution-surface + %s Rust file(s) scanned, every `apr` reference is pinned.\n' \
    "$scanned" "$rust_scanned"
