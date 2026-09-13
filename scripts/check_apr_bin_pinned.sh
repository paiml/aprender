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

# ---------------------------------------------------------------------------
# CLASS 4/5: the same two defects, for `pv`.
#
# `pv` on PATH was 0.49.0 while the tree was 0.63.0, and they DISAGREED on the
# gate that matters: strict-test-binding reported 253 refs / 51 missing under the
# stale binary vs 371 / 27 under HEAD. Both surfaces where the RELEASE decision
# is made were using the stale one -- dogfood_surfaces.sh printed
# `pv present (pv 0.49.0)` into the release receipt AS EVIDENCE OF CORRECTNESS,
# and Makefile `contracts:` ran a bare `pv lint contracts/` as the gate.
#
# Same OPEN/WRAP machinery as `apr`, deliberately: that class was wrong five
# times and the openers are what keep prose out. Do not hand-tune these; add a
# case-table row and re-run --self-test.
#
# No ABS_PV class: unlike apr there is no hardcoded absolute pv path in the tree,
# and pv_bin.sh asks cargo rather than naming a path. If one ever appears, the
# apr ABS class is the template.
# OPEN_PV extends OPEN with shell KEYWORDS. `if pv validate ...; then` is the
# live form at scripts/dogfood-book.sh:92,102 and OPEN does not admit it: `^`
# then `if ` is not whitespace, so the bare invocation sailed through. Found by
# probing the pattern against the known-bad line rather than by trusting a green
# run -- the guard reported "every reference is pinned" while two were not.
# OPEN itself is left ALONE: it backs BARE_APR, whose 62-case table has been
# gotten wrong five times, and widening it is not this change's business.
# Split across two lines (with the same resulting string) because a
# same-line "while ... then" substring, even fully inside a quoted regex
# literal, trips bashrs's naive SC2135 "use do not then" heuristic.
OPEN_PV_KW="if|elif|while|until"
OPEN_PV_KW="${OPEN_PV_KW}|then|do|else"
OPEN_PV="(${OPEN}|(^|[[:space:];&|(])(${OPEN_PV_KW})[[:space:]]+)"
BARE_PV="${OPEN_PV}[[:space:]]*@?[[:space:]]*${WRAP}*pv[[:space:]]+[a-z\"'\$/~.-]"
PATHRES_PV='(command[[:space:]]+-v[[:space:]]+pv|which[[:space:]]+pv|type[[:space:]]+(-[A-Za-z]+[[:space:]]+)?pv|require_tool[[:space:]]+pv)([[:space:]]|[;)&|]|$)'

MIN_EXPECTED="${MIN_EXPECTED:-80}"

violations=0
scanned=0

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
                infence && /apr|pv/ { printf "%d:%s\n", NR, $0; next }
                /!`/ && /apr|pv/ { printf "%d:%s\n", NR, $0 }
            ' "$f"
            ;;
        *)
            # WIDENED to /apr|pv/. This pre-filter fed ONLY lines containing
            # "apr" to the pattern checks, so when the pv classes were added they
            # were inert on any line without the word "apr" in it -- including
            # the one that mattered most, Makefile `@pv lint contracts/`, the
            # release gate. The guard reported "every reference is pinned" while
            # that line was bare. Caught by mutating the Makefile and confirming
            # the mutation ENGAGED (line 352 rewritten, guard still rc=0), not by
            # reading the code. A filter upstream of a correct pattern is the
            # same class as a correct pattern that is never run.
            awk '/apr|pv/ { printf "%d:%s\n", NR, $0 }' "$f"
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
# PV-specific probe: ALWAYS strip quoted strings.
#
# demessage() below strips them only for lines starting with echo/printf, which
# was enough for `apr` because prose rarely reads "apr <lowercase-word>". It is
# NOT enough for `pv`: this repo's own reporters (ok/bad/warn/pass/fail/step)
# take message strings, and `ok "$label validates (pv validate)"` is prose that
# matched BARE_PV and produced a false positive at dogfood_surfaces.sh:221.
#
# Stripping cannot hide a real bare invocation: `pv validate "$c"` still reads
# `pv validate` after the quotes go. It does blind us to the rare `pv "$@"`
# form, which is the correct trade -- a false positive on every reporter line
# would get this guard disabled, and a disabled guard catches nothing.
demessage_pv() {
    printf '%s' "$1" | sed "s/'[^']*'//g; s/\"[^\"]*\"//g"
}

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
    local f="$1" hit lineno text trimmed probe probe_pv self=0
    scanned=$((scanned + 1))
    # This file quotes deliberate violations verbatim in its case table, and
    # apr_bin.sh names the stale absolute paths it exists to detect. Both are
    # data, not invocations. (check_pass_grep_anchored.sh exempts itself for the
    # same reason.) The case table is what proves these two are still honest.
    case "$f" in */apr_bin.sh|*/pv_bin.sh|*/check_apr_bin_pinned.sh) self=1 ;; esac
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
        probe_pv="$(demessage_pv "$trimmed")"
        if [[ $probe_pv =~ $BARE_PV ]]; then
            report "$f" "$lineno" "BARE-PV" "$trimmed"
        fi
        if [[ $probe_pv =~ $PATHRES_PV ]]; then
            report "$f" "$lineno" "PATHRES-PV" "$trimmed"
        fi
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

    # PV rows. probe_case uses demessage(); check_file uses demessage_pv() for
    # the pv classes, so a pv row driven through probe_case would test a pipeline
    # that does not ship -- the exact trap probe_case's own comment warns about.
    probe_case_pv() {
        local re="$1" line="$2" want="$3" label="$4" p t
        ltrim "$line"; t="$LTRIM"
        p="$(demessage_pv "$t")"
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
    must_match_pv=(
        $'\t@pv lint contracts/'   # real TAB: the Makefile recipe form
        'if pv validate contracts/x.yaml; then'
        'out=$(pv lint "$ROOT/contracts/" 2>&1)'
        '        run: pv validate contracts/x.yaml'
        'while pv status x.yaml; do'
    )
    must_not_match_pv=(
        $'\t@. scripts/pv_bin.sh && "$$PV" lint contracts/'
        'out=$("$PV" validate "$c" 2>&1)'
        '# Use pv (not bash) for contract validation'
        'ok "$label validates (pv validate)"'
        'bad "pv lint contracts/ FAILED: $out"'
        'if "$PV" validate contracts/x.yaml; then'
    )
    must_match_pathres_pv=( 'PV=$(command -v pv)' 'require_tool pv "x"' 'which pv' )
    must_not_match_pathres_pv=( '. scripts/pv_bin.sh || exit 1' 'echo "resolved $PV"' )
    for c in "${must_match_pv[@]}";          do probe_case_pv "$BARE_PV" "$c" match bare-pv; done
    for c in "${must_not_match_pv[@]}";      do probe_case_pv "$BARE_PV" "$c" nomatch bare-pv; done
    for c in "${must_match_pathres_pv[@]}";  do probe_case_pv "$PATHRES_PV" "$c" match pathres-pv; done
    for c in "${must_not_match_pathres_pv[@]}"; do probe_case_pv "$PATHRES_PV" "$c" nomatch pathres-pv; done

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
        if [ -n "$TMPROOT" ] && [ "$TMPROOT" != / ] && [ -d "$TMPROOT" ]; then
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

    surface_probe() {
        local label="$1" path="$2" content="$3" d
        d=$(mk_tree)
        mkdir -p "$d/$(dirname "$path")"
        printf '%s\n' "$content" > "$d/$path"
        if (cd "$d" && MIN_EXPECTED=1 bash scripts/check_apr_bin_pinned.sh >/dev/null 2>&1); then
            printf 'SURFACE-PROBE FAIL [%s]: guard stayed GREEN with a violation in %s\n' \
                "$label" "$path" >&2
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
    if ! (cd "$prose_dir" && MIN_EXPECTED=1 bash scripts/check_apr_bin_pinned.sh >/dev/null 2>&1); then
        printf 'SURFACE-PROBE FAIL [skill-prose]: guard flagged markdown prose outside a bash fence\n' >&2
        fails=$((fails + 1))
    fi

    if [ "$fails" -gt 0 ]; then
        printf '\nself-test FAILED with %s case(s).\n' "$fails" >&2
        exit 1
    fi
    printf 'self-test OK: %s regex cases and 9 surface probes.\n' \
        "$(( ${#must_match_bare[@]} + ${#must_not_match_bare[@]} + ${#must_match_abs[@]} \
             + ${#must_not_match_abs[@]} + ${#must_match_pathres[@]} + ${#must_not_match_pathres[@]} \
             + ${#must_match_pv[@]} + ${#must_not_match_pv[@]} \
             + ${#must_match_pathres_pv[@]} + ${#must_not_match_pathres_pv[@]} ))"
    exit 0
fi

while IFS= read -r f; do
    [ -n "$f" ] || continue
    check_file "$f"
done < <(surface_files)

if [ "$violations" -gt 0 ]; then
    printf '\n%s unpinned `apr` reference(s) on execution surfaces (%s file(s) scanned).\n' \
        "$violations" "$scanned" >&2
    printf 'A bare `apr` runs whatever PATH resolves - which is how a 24-day-old\n' >&2
    printf 'binary validated a gate merged the day before. An absolute path names\n' >&2
    printf 'one machine. `$(which apr)` is PATH wearing a pinned costume. Pin it:\n' >&2
    printf '  . scripts/apr_bin.sh    # exports $APR, asserts it was built from HEAD\n' >&2
    printf '  "$APR" qa model.gguf\n' >&2
    exit 1
fi

# Fail closed: a scanner that examined nothing must not report success.
if [ "$scanned" -lt "$MIN_EXPECTED" ]; then
    printf 'ERROR: scanned %s file(s), expected >= %s - the file discovery has gone blind.\n' \
        "$scanned" "$MIN_EXPECTED" >&2
    exit 1
fi

printf 'OK: %s execution-surface file(s) scanned, every `apr` reference is pinned.\n' "$scanned"
