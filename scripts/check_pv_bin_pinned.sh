#!/usr/bin/env bash
# check_pv_bin_pinned.sh - no execution surface may resolve `pv` through PATH
# or through a hardcoded absolute path.
#
# THE CLASS. This is the `apr` defect (see check_apr_bin_pinned.sh) with a
# different binary, and it was LIVE when this guard was written. Measured
# 2026-08-20 in a worktree at origin/main 773a39da1:
#
#   /home/noah/.cargo/bin/pv       pv 0.49.0   installed 2026-06-13
#   crates/aprender-contracts-cli  pv 0.63.0   HEAD
#
#   $ /home/noah/.cargo/bin/pv lint contracts --strict-test-binding \
#         --format json --no-cache | jq -c '.gates[]|select(.name=="strict-test-binding").detail'
#   {"type":"verify","total_refs":253,"existing":202,"missing":51}
#   $ cargo run -q -p aprender-contracts-cli --bin pv -- lint contracts \
#         --strict-test-binding --format json --no-cache | jq -c '...'
#   {"type":"verify","total_refs":371,"existing":344,"missing":27}
#
# 118 test references the stale binary cannot see; 24 "missing" it invents.
# `validate` and `lint` agree, which is what makes this dangerous: the surfaces
# that used bare `pv` looked fine every time anyone checked them by hand.
#
# The three live violations at the time of writing were not incidental:
#   scripts/dogfood_surfaces.sh   the RELEASE-certifying surface sweep, which
#                                 ran `require_tool pv` (a PATH probe), then
#                                 `pv validate` and `pv lint contracts/`
#   scripts/dogfood-book.sh       two `pv validate` gates
#   Makefile:352                  `\t@pv lint contracts/` inside `make contracts`,
#                                 the HARD release gate the dogfood protocol
#                                 looks for -- and the exact `\t@` recipe form
#                                 that stayed invisible to the apr guard until
#                                 someone re-mutated inside the Makefile.
#
# scripts/pv_bin.sh makes staleness IMPOSSIBLE at runtime (it asks cargo to
# build, and names cargo's output). This script makes it UNREINTRODUCIBLE.
#
# An invocation is PINNED when it goes through one of:
#   "$PV" / ${PV} / $PV_BIN        - resolved by scripts/pv_bin.sh
#   $(PV_BIN) in the Makefile      - `cargo run ... --bin pv --`
#   a checkout-relative path       - ./target/release/pv, ${ROOT}/target/...
#   cargo run/build ... --bin pv   - built from the current source by definition
#
# Exit 0 = every invocation on every scanned surface is pinned.
# Exit 1 = at least one is not.
#
# WHY BASH AND NOT pv ITSELF. CLAUDE.md forbids re-implementing in bash what pv
# already does, and that rule is respected here: pv validates YAML CONTRACTS. It
# has no source-scanning subcommand, and none of its 42 subcommands reads shell,
# Makefile or YAML-workflow syntax. Teaching a contract engine to lint bash
# recipe lines would invert its domain, and the in-tree precedent for exactly
# this invariant is a bash guard (check_apr_bin_pinned.sh), wired the same way.
# The contract-shaped part of the job -- "does pv's own verdict match CI's" --
# already belongs to scripts/check_contract_test_binding.sh, which runs pv.
#
# Extending a guard's SCOPE requires re-mutating in the NEW scope; the old proof
# does not transfer. `--self-test` injects a violation into each surface in that
# surface's own syntax and asserts this script turns RED. Run it, don't read it.

set -euo pipefail

SELF_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.." || exit 1

# ---------------------------------------------------------------------------
# CLASS 1: `pv` in COMMAND POSITION - the only place it can launch a binary.
#
# Command position means: start of line, after a shell separator (; & | && ||),
# inside a substitution or subshell (`(` covers both `(` and `$(`), or after a
# YAML `run:`. Anchoring on command position rather than line content is what
# keeps prose out - the word "pv" appears in ~200 comment lines in this tree.
#
# `@?` covers Makefile recipe lines, which start with a TAB and usually make's
# silent-prefix `@`. This is not defensive programming: `\t@pv lint contracts/`
# was a LIVE violation at Makefile:352, and it is the precise form that survived
# the apr guard's first scope extension. It is in the case table below.
#
# WRAP covers prefixes that do not change which binary is launched:
# `timeout 60 pv lint`, `PV_NO_CACHE=1 pv validate`.
#
# WRAP also covers the shell KEYWORDS that introduce a command: `if pv validate
# ... ; then`. This is not hypothetical padding - it is the form at
# scripts/dogfood-book.sh:92 and :102, two of the live violations this guard was
# written for, and the case table caught its absence on the first run. The apr
# guard has no keyword alternative and would miss `if apr qa model.apr; then`
# for the same reason; noted rather than silently copied.
WRAP='([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+|(if|elif|then|else|while|until|do)[[:space:]]+|![[:space:]]+|timeout[[:space:]]+[0-9]+[a-z]?[[:space:]]+|nohup[[:space:]]+|time[[:space:]]+|sudo[[:space:]]+|nice[[:space:]]+|env[[:space:]]+|exec[[:space:]]+|stdbuf[[:space:]]+-[^[:space:]]+[[:space:]]+)'
#
# A BACKTICK is deliberately NOT an opener, for the reason measured in the apr
# guard: once markdown is in scope a backtick is overwhelmingly a code-span
# delimiter. `!` + backtick IS an opener - that pair is unambiguously "run this"
# in a skill front-matter, never a code span.
OPEN='(^|[;&|(]|&&|\|\||run:|!`)'
# The trailing class requires an ARGUMENT, so the bare word `pv`, the directory
# `.pv/`, and prose like "install pv" cannot match. `-` is in the class because
# `pv --version` is the single most load-bearing bare invocation there is:
# reading a version off the wrong binary IS the defect.
BARE_PV="${OPEN}[[:space:]]*@?[[:space:]]*${WRAP}*pv[[:space:]]+[a-z\"'\$/~.-]"

# ---------------------------------------------------------------------------
# CLASS 2: an ABSOLUTE hardcoded `pv` path.
#
# There is no correct absolute path to hardcode. Measured on this box: the same
# worktree reports target_directory = <worktree>/target under a plain bash or
# zsh, and /mnt/nvme-raid0/targets/aprender under the agent shell, because the
# user profile defines a `cargo` SHELL FUNCTION that derives a per-project
# target dir from `git config remote.origin.url`. An absolute path is therefore
# right in one shell and silently wrong in the other, before you even consider
# the second checkout. Ask cargo: `. scripts/pv_bin.sh || exit 1`.
#
# The leading anchor is load-bearing: without it `[A-Za-z0-9_.$/-]*` matches the
# `/pv` inside RELATIVE `target/release/pv`, flagging correct code. `:-` and
# `:=` are openers because `${PV:-/home/noah/.cargo/bin/pv}` is a hardcoded
# absolute path wearing a parameter-expansion costume - the form that was live
# in check_book_cli_parity.sh for `apr` and invisible for two reasons at once.
# `}` and `)` are therefore terminators.
#
# Note `.pv/` (the lint trend-state directory, e.g. `.pv/lint-previous.json`)
# cannot match: the pattern requires a literal `/` immediately before `pv`, and
# that path has `/.pv`. It is in the must-not-match table.
ABS_PV='(^|[[:space:]"'"'"'=(`]|:-|:=)(/|~/|\$HOME/)[A-Za-z0-9_.$/-]*/pv([[:space:]"'"'"'})]|$)'

# ---------------------------------------------------------------------------
# CLASS 3: resolving `pv` through PATH, with or without a variable.
#
# `PV="$(which pv)"` is the original defect wearing the guard's own approved
# costume: the binary is still whatever PATH holds, but every later `"$PV" lint`
# matches the class-1 allowlist, so a checker that only looked at invocations
# would read the file, find nothing, and pass.
#
# `require_tool pv` is included because it is THIS repo's spelling of the same
# thing. dogfood_surfaces.sh:247 called `require_tool pv "contract validation
# must use pv, not a python YAML walk"`, whose body is `command -v "$tool"` -
# PATH resolution through a variable, which no generic regex can see. It then
# printed the resolved version into the release receipt, so the sweep reported
# `pv present (pv 0.49.0)` as evidence of correctness.
#
# `command -v pv` inside an `echo` is a DIAGNOSTIC, not a resolution; the
# echo/printf exemption below covers that.
PATHRES='(command[[:space:]]+-v[[:space:]]+pv|which[[:space:]]+pv|type[[:space:]]+(-[A-Za-z]+[[:space:]]+)?pv|require_tool[[:space:]]+pv)([[:space:]]|[;)&|]|$)'

MIN_EXPECTED="${MIN_EXPECTED:-80}"

violations=0
scanned=0

# ---------------------------------------------------------------------------
# Surfaces. Everything that can EXECUTE pv, with no reachability guessing.
#
# EVERY scripts/**/*.sh is in scope, not just the ones a workflow names. That
# was learned the expensive way on the apr side: discovery was one grep deep,
# and scripts/dogfood-book.sh -> check_book_examples_executable.sh sat outside
# it holding a live violation. "Reachable from CI" is not a property you can
# compute with one grep, and guessing it wrong fails silently.
#
# docs/ and book/ are deliberately NOT here. `pv validate model.yaml` in a book
# chapter is instruction to a reader who installed the crate. The invariant is
# about surfaces that run pv and then REPORT A VERDICT.
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
# For markdown (skills) that is the content of bash fences only, plus the
# front-matter's `!`...`` inline command substitution, which Claude Code RUNS
# when the skill loads. A skill's prose says "run pv lint first" at the start of
# a line - command position by syntax, documentation by intent.
#
# The `/pv/` pre-filter is sound, not a shortcut: all three classes require the
# literal substring `pv` on the line, so a line without it cannot be a
# violation. It keeps the scan under a second over ~30k lines.
emit_lines() {
    local f="$1"
    case "$f" in
        *.md)
            awk '
                /^[[:space:]]*```/ {
                    if (infence) { infence = 0 }
                    else if ($0 ~ /```(bash|sh|shell|console)[[:space:]]*$/) { infence = 1 }
                    next
                }
                infence && /pv/ { printf "%d:%s\n", NR, $0; next }
                /!`/ && /pv/ { printf "%d:%s\n", NR, $0 }
            ' "$f"
            ;;
        *)
            awk '/pv/ { printf "%d:%s\n", NR, $0 }' "$f"
            ;;
    esac
}

# Strip leading whitespace, into $LTRIM. A bash-regex capture rather than a
# nested expansion: same result, and it does not fork a process per line.
ltrim() {
    [[ $1 =~ ^[[:space:]]*(.*)$ ]]
    LTRIM="${BASH_REMATCH[1]}"
}

# Quoted text handed to a REPORTING call is a MESSAGE, not a command. Stripping
# the quoted segments keeps
#   printf 'Running pv lint %s --strict-test-binding ...\n' "$DIR"
# out - live at check_contract_test_binding.sh:323 - while keeping
#   echo "$YAML" | pv validate -
# in, since the pipe and the invocation are OUTSIDE the quotes.
#
# The apr guard hardcodes `echo|printf`. That list is not enough here and the
# first real run proved it: scripts/dogfood_surfaces.sh:221 reads
#   ok "$label validates (pv validate)"
# and the parenthesis inside the MESSAGE is a command-position opener, so the
# guard flagged a string. `ok`/`bad`/`emit_pass` are this repo's reporting
# helpers and any future script will invent another name, so the rule is on
# SHAPE rather than on a name list: `<identifier> <quote>` is a call whose first
# argument is a string literal.
#
# Deliberately NOT "strip every quoted segment on every line": that erases the
# quoted body of `bash -c "..."`, and it is the difference between a rule that
# describes prose and one that describes shell.
#
# RESIDUAL GAP, stated rather than hidden. A quote is not an OPENER, so
# `bash -c "pv lint contracts/"` is invisible to class 1 - the `pv` sits after a
# `"`. Making a quote an opener was tried and rejected: it turns
# `grep -q "pv lint" file` into a violation, and there is no shape rule that
# separates the two. Zero occurrences of the `bash -c` form exist in this tree
# today; the class-3 rules still catch the resolution half of the defect.
demessage() {
    local t="$1"
    case "$t" in
        echo\ *|printf\ *|echo|printf)
            printf '%s' "$t" | sed "s/'[^']*'//g; s/\"[^\"]*\"//g"
            return ;;
    esac
    if [[ $t =~ ^[A-Za-z_][A-Za-z0-9_]*[[:space:]]+[\"\'] ]]; then
        printf '%s' "$t" | sed "s/'[^']*'//g; s/\"[^\"]*\"//g"
        return
    fi
    printf '%s' "$t"
}

report() {
    printf '%s %s:%s\n' "$1" "$2" "$3"
    printf '        %s\n' "$4"
    violations=$((violations + 1))
}

check_file() {
    local f="$1" hit lineno text trimmed probe self=0
    scanned=$((scanned + 1))
    # These two quote deliberate violations verbatim - the case table below, and
    # pv_bin.sh's header naming the stale binary it exists to refuse. Both are
    # data, not invocations. (check_apr_bin_pinned.sh exempts itself the same
    # way, and the case table is what proves the exemption stays honest.)
    case "$f" in */pv_bin.sh|*/check_pv_bin_pinned.sh) self=1 ;; esac
    [ "$self" -eq 1 ] && return 0
    while IFS= read -r hit; do
        lineno="${hit%%:*}"
        text="${hit#*:}"
        ltrim "$text"; trimmed="$LTRIM"

        # Comments are documentation, not invocations.
        case "$trimmed" in '#'*) continue ;; esac

        # YAML metadata is prose, not shell.
        case "$trimmed" in
            name:*|-\ name:*|description:*|title:*|summary:*|if:*|-\ if:*|id:*|uses:*|shell:*|working-directory:*)
                continue ;;
        esac

        probe="$(demessage "$trimmed")"

        # CLASS 1 --------------------------------------------------------
        if [[ $probe =~ $BARE_PV ]]; then
            case "$text" in
                *'$PV'*|*'${PV'*|*'$(PV_BIN)'*|*'target/release/pv'*|*'target/debug/pv'*|*'--bin pv'*)
                    : ;;
                *) report 'BARE-PV' "$f" "$lineno" "$trimmed" ;;
            esac
        fi

        # CLASS 2 --------------------------------------------------------
        if [[ $trimmed =~ $ABS_PV ]]; then
            report 'ABS-PV ' "$f" "$lineno" "$trimmed"
        fi

        # CLASS 3 --------------------------------------------------------
        if [[ $probe =~ $PATHRES ]]; then
            report 'PATH-PV' "$f" "$lineno" "$trimmed"
        fi
    done < <(emit_lines "$f")
}

# ---------------------------------------------------------------------------
# --self-test: the must-match / must-not-match case table, plus one mutation
# per SURFACE in that surface's own syntax.
#
# CLAUDE.md: "Guard regexes ship a case table" - the apr patterns were wrong
# five times, every one caught by the table and none by review.
if [ "${1:-}" = "--self-test" ]; then
    fails=0

    # -- regex case table ---------------------------------------------------
    must_match_bare=(
        'pv lint contracts/'
        '  pv validate contracts/softmax-kernel-v1.yaml'
        '- run: pv lint contracts/'
        '	@pv lint contracts/ 2>&1 | tail -5'
        '	pv validate contracts/x.yaml'
        'cd /tmp && pv status contracts/x.yaml'
        'foo; pv audit contracts/'
        'echo hi | pv validate -'
        'out=$(pv validate "$contract" 2>&1)'
        'out=$(timeout 60 pv lint "$REPO_ROOT/contracts/" 2>&1)'
        'if pv validate contracts/apr-book-completeness-v1.yaml > /dev/null; then'
        'if ! pv validate contracts/x.yaml; then'
        'while pv status "$c" >/dev/null; do'
        'elif pv audit contracts/; then'
        'PV_NO_CACHE=1 pv lint contracts/'
        'env PV_NO_CACHE=1 pv lint contracts/'
        'pv --version'
        'pv "$sub" --help 2>&1 | head -1'
        'pv $sub contracts/x.yaml'
        'pv ./contracts/x.yaml'
        'pv ~/contracts/x.yaml'
        '- pv version: !`pv --version 2>/dev/null`'
    )
    must_not_match_bare=(
        '"$PV" lint contracts/'
        '${PV} validate contracts/x.yaml'
        '"$PV_BIN" lint contracts/'
        './target/debug/pv lint contracts/'
        '"${REPO_ROOT}/target/release/pv" lint contracts/'
        'cargo run -q -p aprender-contracts-cli --bin pv -- lint contracts/'
        'PV_BIN := cargo run --release -p aprender-contracts-cli --bin pv --'
        '$(PV_BIN) validate "$$contract" || exit 1; \'
        '- name: pv lint gate'
        '# pv lint contracts/'
        'ok "pv lint contracts/ passed"'
        "printf 'Running pv lint %s --strict-test-binding ...\\n' \"\$CONTRACT_DIR\""
        'cargo install aprender-contracts-cli'
        'cat .pv/lint-previous.json'
        'rm -f .pv/lint-cache.json'
        'ls scripts/pv_bin.sh'
        'the pv binary is built by this checkout'
        'if pvcheck --version; then'
        'if [ -x "$PV" ]; then'
        # Live at scripts/dogfood_surfaces.sh:221/223. The `(` inside the
        # MESSAGE is a command-position opener; only demessage's shape rule
        # keeps these out. Delete that rule and these two turn RED.
        'ok "$label validates (pv validate)"'
        'bad "$label FAILED pv validate: $(printf %s "$out" | tail -1)"'
        'emit_pass "contracts (pv lint clean)"'
    )
    must_match_abs=(
        '/home/noah/.cargo/bin/pv lint contracts/'
        'PV="${PV:-/home/noah/.cargo/bin/pv}"'
        'PV_BINARY="${PV_BINARY:-/mnt/nvme-raid0/targets/aprender/debug/pv}"'
        'elif [ -x /mnt/nvme-raid0/targets/aprender/debug/pv ]; then'
        'PV_BIN=/mnt/nvme-raid0/targets/aprender/release/pv'
        '~/.cargo/bin/pv --version'
        '$HOME/.cargo/bin/pv --version'
    )
    must_not_match_abs=(
        './target/debug/pv --version'
        'PV_BIN="${PV_BIN:-${REPO_DIR}/target/debug/pv}"'
        'readonly PV_BIN="${PROJECT_ROOT}/target/release/pv"'
        'bash scripts/pv_bin.sh'
        'cargo run --bin pv'
        'cat /home/noah/src/aprender/.pv/lint-previous.json'
        'ls /mnt/nvme-raid0/targets/aprender/debug/pv-cli'
    )
    must_match_pathres=(
        'PV="$(which pv)"'
        'PV_BIN="$(command -v pv)"'
        'if command -v pv >/dev/null 2>&1; then'
        'if ! command -v pv; then'
        'require_tool pv     "contract validation must use pv"'
    )
    must_not_match_pathres=(
        'echo "pv --version: $GOT ($(command -v pv))"'
        'command -v cargo >/dev/null'
        'which pmat'
        'require_tool bashrs "shell linting must use bashrs"'
        'command -v pvcheck'
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

    for c in "${must_match_bare[@]}"; do probe_case "$BARE_PV" "$c" match bare; done
    for c in "${must_not_match_bare[@]}"; do
        # The allowlist is part of the class-1 decision, so apply it here too.
        case "$c" in
            *'$PV'*|*'${PV'*|*'$(PV_BIN)'*|*'target/release/pv'*|*'target/debug/pv'*|*'--bin pv'*)
                continue ;;
        esac
        probe_case "$BARE_PV" "$c" nomatch bare
    done
    for c in "${must_match_abs[@]}"; do probe_case "$ABS_PV" "$c" match abs; done
    for c in "${must_not_match_abs[@]}"; do probe_case "$ABS_PV" "$c" nomatch abs; done
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

    # A FRESH tree per probe. The apr guard's first draft reused one directory
    # and cleaned it with `rm -rf "$TMP"/*`, which does not match dotfiles - so
    # `.github/` and `.claude/` survived, every later probe inherited the FIRST
    # probe's violation, and all of them "turned RED" without their own mutation
    # ever being read. A probe that passes for a reason other than the thing it
    # probes is the exact failure this file exists to prevent.
    mk_tree() {
        local d
        d=$(mktemp -d "$TMPROOT/probe.XXXXXX")
        mkdir -p "$d/scripts"
        cp "$SELF_PATH" "$d/scripts/check_pv_bin_pinned.sh"
        : > "$d/Makefile"
        printf '%s' "$d"
    }

    surface_probe() {
        local label="$1" path="$2" content="$3" d
        d=$(mk_tree)
        mkdir -p "$d/$(dirname "$path")"
        printf '%s\n' "$content" > "$d/$path"
        if (cd "$d" && MIN_EXPECTED=1 bash scripts/check_pv_bin_pinned.sh >/dev/null 2>&1); then
            printf 'SURFACE-PROBE FAIL [%s]: guard stayed GREEN with a violation in %s\n' \
                "$label" "$path" >&2
            fails=$((fails + 1))
        fi
    }

    # Each violation is written in the SURFACE'S OWN syntax. The Makefile case is
    # the reason this section exists: a tab-and-@ recipe line is not a shell line
    # and it is the form that was actually live at Makefile:352.
    surface_probe 'workflow'  '.github/workflows/w.yml' \
        '      - name: contracts
        run: pv lint contracts/'
    surface_probe 'makefile'  'Makefile' \
        'contracts:
	@pv lint contracts/ 2>&1 | tail -5'
    surface_probe 'script-indirect' 'scripts/never-named-by-a-workflow.sh' \
        '#!/usr/bin/env bash
pv validate contracts/x.yaml'
    surface_probe 'script-nested' 'scripts/pub/deep.sh' \
        '#!/usr/bin/env bash
/home/noah/.cargo/bin/pv lint contracts/'
    surface_probe 'skill-fence' '.claude/skills/x/SKILL.md' \
        'Run the gate:

```bash
pv lint contracts/
```'
    surface_probe 'skill-inline-bang' '.claude/skills/x/SKILL.md' \
        '- Installed pv version: !`pv --version 2>/dev/null || echo none`'
    surface_probe 'pathres-launder' 'scripts/launder.sh' \
        '#!/usr/bin/env bash
PV="$(which pv)"
"$PV" lint contracts/'
    surface_probe 'require-tool' 'scripts/sweep.sh' \
        '#!/usr/bin/env bash
require_tool pv "contract validation must use pv"'
    surface_probe 'abs-default' 'scripts/absdefault.sh' \
        '#!/usr/bin/env bash
PV="${PV:-/home/noah/.cargo/bin/pv}"'

    # A NON-VACUITY control in the other direction: the guard must stay GREEN on
    # a tree whose pv usage is correctly pinned. Without this, a regex that
    # matched everything would pass all nine probes above and be useless.
    green_dir=$(mk_tree)
    mkdir -p "$green_dir/.github/workflows"
    printf '%s\n' '#!/usr/bin/env bash
. scripts/pv_bin.sh || exit 1
"$PV" lint contracts/
"$PV" validate contracts/x.yaml' > "$green_dir/scripts/pinned.sh"
    printf '%s\n' 'contracts:
	@cargo run -q -p aprender-contracts-cli --bin pv -- lint contracts/' \
        > "$green_dir/Makefile"
    printf '%s\n' 'name: w
jobs:
  j:
    steps:
      - run: cargo run -q -p aprender-contracts-cli --bin pv -- lint contracts/' \
        > "$green_dir/.github/workflows/w.yml"
    if ! (cd "$green_dir" && MIN_EXPECTED=1 bash scripts/check_pv_bin_pinned.sh >/dev/null 2>&1); then
        printf 'SURFACE-PROBE FAIL [pinned-green]: guard flagged correctly pinned usage\n' >&2
        (cd "$green_dir" && MIN_EXPECTED=1 bash scripts/check_pv_bin_pinned.sh >&2 || true)
        fails=$((fails + 1))
    fi

    # A skill's PROSE is not an execution surface; only its bash fences are.
    prose_dir=$(mk_tree)
    mkdir -p "$prose_dir/.claude/skills/x"
    printf 'pv lint contracts/ is the first tool to reach for.\n' \
        > "$prose_dir/.claude/skills/x/SKILL.md"
    if ! (cd "$prose_dir" && MIN_EXPECTED=1 bash scripts/check_pv_bin_pinned.sh >/dev/null 2>&1); then
        printf 'SURFACE-PROBE FAIL [skill-prose]: guard flagged markdown prose outside a bash fence\n' >&2
        fails=$((fails + 1))
    fi

    # The vacuity guard itself must be able to fail: a scanner that examined
    # nothing must not report success. Probe it with a tree of zero surfaces.
    empty_dir=$(mktemp -d "$TMPROOT/empty.XXXXXX")
    mkdir -p "$empty_dir/scripts"
    cp "$SELF_PATH" "$empty_dir/scripts/check_pv_bin_pinned.sh"
    if (cd "$empty_dir" && bash scripts/check_pv_bin_pinned.sh >/dev/null 2>&1); then
        printf 'SURFACE-PROBE FAIL [vacuity]: guard passed with a near-empty surface set\n' >&2
        fails=$((fails + 1))
    fi

    if [ "$fails" -gt 0 ]; then
        printf '\nself-test FAILED with %s case(s).\n' "$fails" >&2
        exit 1
    fi
    printf 'self-test OK: %s regex cases, 9 violation probes, 2 must-stay-green probes, 1 vacuity probe.\n' \
        "$(( ${#must_match_bare[@]} + ${#must_not_match_bare[@]} + ${#must_match_abs[@]} \
             + ${#must_not_match_abs[@]} + ${#must_match_pathres[@]} + ${#must_not_match_pathres[@]} ))"
    exit 0
fi

while IFS= read -r f; do
    [ -n "$f" ] || continue
    check_file "$f"
done < <(surface_files)

if [ "$violations" -gt 0 ]; then
    printf '\n%s unpinned `pv` reference(s) on execution surfaces (%s file(s) scanned).\n' \
        "$violations" "$scanned" >&2
    printf 'A bare `pv` runs whatever PATH resolves. On this box that is pv 0.49.0,\n' >&2
    printf '68 days stale, which reports 253 test refs / 51 missing where HEAD pv\n' >&2
    printf 'reports 371 / 27 on the SAME tree. An absolute path names one machine\n' >&2
    printf 'and one shell. `$(which pv)` is PATH wearing a pinned costume. Pin it:\n' >&2
    printf '  . scripts/pv_bin.sh    # exports $PV, built from this checkout by cargo\n' >&2
    printf '  "$PV" lint contracts/\n' >&2
    exit 1
fi

# Fail closed: a scanner that examined nothing must not report success.
if [ "$scanned" -lt "$MIN_EXPECTED" ]; then
    printf 'ERROR: scanned %s file(s), expected >= %s - the file discovery has gone blind.\n' \
        "$scanned" "$MIN_EXPECTED" >&2
    exit 1
fi

printf 'OK: %s execution-surface file(s) scanned, every `pv` reference is pinned.\n' "$scanned"
