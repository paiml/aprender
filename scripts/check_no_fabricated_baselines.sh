#!/usr/bin/env bash
#
# check_no_fabricated_baselines.sh — a comparator baseline may be MEASURED or
# ABSENT, never asserted (F12, aprender#2679 / #2672 / #2706 PERF-008).
#
# F12 — FABRICATED MEASUREMENT. A value carrying the form, units and
# provenance-shape of a measurement, produced without the measurement having
# been taken. It is distinct from its neighbours: not F7 (nothing passes
# falsely — the number may even be correct for some past run) and not F9
# (there is no coupled oracle; there is no oracle at all). The harm is that it
# is indistinguishable from evidence at the point of consumption, and it
# survives review because it LOOKS like the thing it replaces.
#
# Found 2026-08-24 in scripts/benchmark-2x-ollama.sh:27-29 —
#   OLLAMA_BASELINE="${OLLAMA_BASELINE:-291}"
#   OLLAMA_SINGLE="${OLLAMA_SINGLE:-120}"
#   OLLAMA_CPU="${OLLAMA_CPU:-15}"
# with ollama never invoked, and the same three literals emitted into JSON by
# scripts/benchmark-matrix.sh:396 as `ollama_baselines`.
#
# THE RULE: a shell variable naming a competitor's performance may not carry a
# numeric literal default. Deleting the four scripts removes the instances;
# this bans the CONSTRUCT, so the pattern cannot return under a new name.
#
# ---------------------------------------------------------------------------
# PERF-008 (#2706 §5) — SHAPE, NOT LITERAL.
#
# The spec marked this guard WEAK, and the reason generalises: "a guard that
# knows the literal 291 proves string equality, not shape recognition". So the
# question is not whether the guard catches the three lines it was born from.
# It is which SPELLINGS of the same construct it cannot see at all.
#
# That was measured rather than argued. Sixteen fabrication shapes were put to
# the pre-PERF-008 pattern; it matched THREE:
#
#     MATCH  "ollama_baseline": 318          (JSON key)
#     MATCH  OLLAMA_TPS=318                  (bare uppercase assignment)
#     MATCH  OLLAMA_BASELINE=291 run_bench   (env-prefixed command)
#     miss   ollama_baseline = 291           TOML
#     miss   ollama_baseline: 291            YAML
#     miss   OLLAMA_BASELINES=(291 318)      array
#     miss   declare -a OLLAMA_BASELINES=(291 120 15)
#     miss   OLLAMA_BASELINE=$((291))        arithmetic
#     miss   : "${OLLAMA_BASELINE:=291}"     assign-default (`:=`, not `:-`)
#     miss   ollama_baseline=291             lowercase shell variable
#     miss   OLLAMA_BASELINE="${BASE}91"     assembled from parts
#     miss   OLLAMA_BASELINE = 291           python
#     miss   ollama_baseline() { echo 291; } function returning a literal
#     miss   ollama) BASELINE=291 ;;         case arm
#     miss   let ollama_baseline = 225.0;    Rust
#
# 3 of 16. A construct ban that sees one spelling is a ban on one spelling —
# the sentence was already in this file, one paragraph up, describing the JSON
# hole. It was true of eleven more shapes than it admitted.
#
# What this pass covers, what it does not, and why, is set out at each pattern.
# The two shapes deliberately NOT claimed are recorded under RESIDUAL below —
# named, not quietly dropped.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

rc=0
printf -- '--- no fabricated comparator baselines (F12) ------------------------\n'

# The competitor list is UNCHANGED by PERF-008 and is deliberately so: this pass
# widens the SHAPE axis, and widening the NAME axis at the same time would leave
# neither one mutation-tested in isolation. Residual, recorded: check_no_claim_
# literals.sh's comparator list also carries SGLang, TensorRT, LMDeploy,
# TurboMind and FasterTransformer; this one does not yet.
COMP_UC='OLLAMA|LLAMA|LLAMACPP|VLLM|TGI|PYTORCH|TORCH'
COMP_LC='ollama|llama|llamacpp|vllm|tgi|pytorch|torch'
COMP='('"$COMP_UC"'|'"$COMP_LC"')'

# ---------------------------------------------------------------------------
# PATTERN_VAR — a competitor-named SHELL variable assigned a numeric literal.
#
# The `:-N}` default form is the one actually found; plain `=N` too. TWO shapes
# at first, because the first pattern written here caught only one of the two
# live instances. `benchmark-2x-ollama.sh` asserts via a shell variable;
# `benchmark-matrix.sh:396` emitted the same three numbers as a JSON literal:
#   JSON_RESULTS+='],"ollama_baselines":{"gpu_batched":291,"gpu_single":120,"cpu":15}}'
#
# The suffix ALLOWLIST was the wrong shape and the case table caught it: the
# real line `OLLAMA_CPU=15` carries no baseline-ish suffix, so a pattern keyed
# on BASELINE|TPS|THROUGHPUT would have missed one of the three live
# instances. Inverted: ANY competitor-prefixed variable assigned a numeric
# literal is suspect, MINUS an explicit denylist of configuration suffixes
# (a timeout or a port is a setting, not a measurement).
#
# A DECLARATION KEYWORD IS NOT AN ESCAPE HATCH. The anchor was `^[[:space:]]*`
# followed directly by the variable name, so `readonly OLLAMA_BASELINE=291` and
# `export OLLAMA_BASELINE=291` matched NOTHING — and `export` is the likelier
# form in a shell script than a bare assignment. The guard read as strict and
# was blind to the two spellings a real fabrication would most plausibly use.
# Caught by running the case table, not by reading the pattern: the plain
# `OLLAMA_TPS=163` fixture passed while `readonly OLLAMA_BASELINE=291` sailed
# through in the same sweep.
#
# RC/STATUS/CODE/PID/FD were added after scripts/llama_bin.sh's `LLAMA_PIN_RC=3`
# — a RETURN CODE — was flagged as a fabricated baseline. It surfaced only when
# PARITY-009 and PARITY-005 met in the cumulative stack head, which is what a
# cumulative head is for: two branches each green alone, one false positive
# together.
#
# PERF-008 adds, on the RHS: `:=` beside `:-`, array initialisers, arithmetic
# expansion, and one form of assembly-from-parts. On the LHS: lowercase names.
# Lowercase is not a stylistic nicety — a shell FUNCTION-local baseline is
# conventionally lowercase, and the whole point of banning the construct is
# that it must not return under a new name.
DECL='((readonly|export|local|declare|typeset)[[:space:]]+(-[a-zA-Z]+[[:space:]]+)?)?'
VNAME="$COMP"'_[A-Za-z0-9_]*'
# RHS alternatives, each a way to write "this competitor did N":
#   1  ${X:-291} / ${X:=291}   parameter default, both operators
#   2  291 / "291"             bare literal
#   3  (291 318)               array initialiser, first element numeric
#   4  $((291))                arithmetic expansion of a literal
#   5  "${BASE}91" / "2${X}"   assembled from an expansion and a digit run
VRHS='("?\$\{[A-Za-z0-9_]+:[-=][0-9.]+\}"?'\
'|"?[0-9.]+"?'\
'|\([[:space:]]*"?[0-9.]'\
'|"?\$\(\([[:space:]]*[0-9.]'\
'|"\$\{[A-Za-z0-9_]+\}[0-9]|"[0-9]+\$\{[A-Za-z0-9_]+\}")'
PATTERN_VAR='^[[:space:]]*(:[[:space:]]+"?\$\{)?'"$DECL$VNAME"'[[:space:]]*=[[:space:]]*'"$VRHS"
# The `: "${OLLAMA_BASELINE:=291}"` idiom puts the variable INSIDE an expansion,
# so the line does not begin with the name. Matched by its own alternative
# rather than by relaxing the anchor — relaxing it is what let `readonly`
# through last time, in reverse.
PATTERN_ASSIGN_DEFAULT='\$\{'"$VNAME"':[-=][0-9.]+\}'

# A UNIT IS NOT A SETTING. `MS`, `SECONDS` and `SECS` were on this denylist and
# PERF-008 removed them: `ollama_baseline_seconds: 30` and `ollama_ms = 14.05`
# are fabricated LATENCIES, and exempting them because they name their unit
# exempts exactly the class the guard exists for. `OLLAMA_TIMEOUT_SECONDS=30`
# stays exempt on TIMEOUT, which is the word that makes it a setting.
#
# THE `(^|[^A-Za-z0-9])` PREFIX IS LOAD-BEARING, and it is the same defect the
# Rust name-denylist hit one screen down. Without it the denylist matches a
# SUBSTRING anywhere on the line: `de<CODE>_tps`, `ti<MIN>g`, `sou<RC>e`. Every
# one of those silently exempts a true positive, and it exempts them invisibly —
# the guard reports ok, with a smaller number. Requiring a name boundary is what
# makes this a suffix denylist rather than a substring lottery.
#
# Matched case-insensitively, because PERF-008 admits lowercase variable names
# and a denylist that only knows OLLAMA_TRIALS while the pattern catches
# ollama_trials is a denylist with a hole shaped like the widening that
# introduced it. That hole was real and the case table below is what found it.
CONFIG_SUFFIX='(^|[^A-Za-z0-9])(TIMEOUT|PORT|RETRIES|RETRY|LIMIT|MAX|MIN|SIZE|COUNT|WORKERS|THREADS|RC|STATUS|CODE|PID|FD|LEVEL|VERSION|TRIALS|SEED|ITERS|ITERATIONS)[A-Za-z0-9_]*[[:space:]]*[=:]'

PATTERN_JSON='"('"$COMP_LC"')_[a-z0-9_]*(baseline|bench|tps|throughput|speed|latency)[a-z0-9_]*"[[:space:]]*:[[:space:]]*(\{[^}]*[0-9][^}]*\}|[0-9.]+)'

# ---------------------------------------------------------------------------
# PATTERN_CFG — the same construct in a CONFIG or PYTHON file, where `=` and `:`
# carry surrounding whitespace and the name is conventionally lowercase.
#
#   ollama_baseline = 291        TOML / Python
#   ollama_tps: 318              YAML
#   ollama_baselines = [291, 318]
#
# WHY THIS IS A SEPARATE PATTERN AND NOT A RELAXATION OF PATTERN_VAR. In shell,
# `OLLAMA_TPS = 291` is not an assignment at all — it is the command `OLLAMA_TPS`
# with two arguments, and the construct that really looks like it is a COMPARISON:
#   [ "$OLLAMA_TPS" = 291 ]
# Allowing optional spaces around `=` in the shell pattern would flag that, and a
# comparison against a measured value is the opposite of a fabrication. So the
# space-tolerant form is admitted only in the universes where it is an assignment.
#
# A baseline-ish word is REQUIRED here, unlike PATTERN_VAR. The shell universe is
# 134 hand-written files where every competitor-prefixed variable is suspect; the
# config universe contains workflow YAML and cargo TOML where `llama_layers: 32`
# is a model parameter, not a claim about a competitor's speed.
CFG_MEASURE='(baseline|bench|tps|tok_s|toks|throughput|speed|latency|parity|ms|p50|p95|p99)'
PATTERN_CFG='^[[:space:]]*(-[[:space:]]+)?"?('"$COMP_LC"')[a-z0-9_]*'"$CFG_MEASURE"'[a-z0-9_]*"?[[:space:]]*[:=][[:space:]]*(\[[[:space:]]*)?[0-9]+(\.[0-9]+)?[[:space:]]*(,|\]|#|$)'

PATTERN="($PATTERN_VAR)|($PATTERN_ASSIGN_DEFAULT)|($PATTERN_JSON)"

# THIS FILE IS EXCLUDED FROM ITS OWN SCAN, and the reason is not convenience.
# Its case table deliberately CONTAINS the forbidden construct — that is what a
# must-match fixture IS. Scanning itself would make the guard permanently red
# against its own proof of discrimination.
#
# It passed for a while and then began failing, which is the interesting part:
# the universe is `git ls-files`, so while this file was UNTRACKED it was not
# scanned at all. The moment it was committed it began matching its own
# fixtures. That is the third instance of the tracked-only-universe shape in
# this epic (SHIM-2644-03; check_bench_threshold.sh in PARITY-008/009), so the
# universe below also unions the working tree — a new offender must not get a
# free pass merely by being uncommitted.
SELF="check_no_fabricated_baselines.sh"

# FULL-LINE COMMENTS ARE NOT CODE, and a guard that cannot tell the difference
# reds its own neighbours. This one did: check_no_claim_literals.sh:7 documents
# the very constructs banned here — a bare competitor-prefixed assignment, and
# the matching JSON key — inside a comment explaining why the two guards differ.
# Matching it made a sibling guard's DOCUMENTATION a fabricated baseline, and
# the FAIL text offers no lever except widening the allowlist, so the pressure
# was to weaken the guard to describe it.
#
# Only lines whose first non-space character is `#` are dropped, and the true
# line number survives (grep -n runs first, the filter runs on its output). A
# trailing comment on a real assignment — `OLLAMA_TPS=163  # measured` — is
# still scanned and still caught, because the code is on that line too. Widening
# this to strip `#` to end-of-line would blind the guard to exactly that.
#
# `//` joins `#` for the config universe: JSON5/JSONC and the Rust-adjacent
# fixtures under scripts/ comment with slashes.
scan_file() {
    grep -nE "$1" "$2" 2>/dev/null | grep -vE '^[0-9]+:[[:space:]]*(#|//)'
}

# hits_in — SIGPIPE-SAFE. Never `scan_file ... | grep -q`.
#
# `grep -q` exits at its FIRST match. Under `set -o pipefail` the upstream greps
# then take SIGPIPE, exit 141, and pipefail hands the PIPELINE that 141 — so the
# `if` reads FALSE even though the pattern matched. The direction matters: this
# is a silent FALSE NEGATIVE, a free pass for a real fabrication, not a noisy
# false alarm.
#
# It is input-size dependent, which is why nothing noticed. On the one-line
# fixtures in the case table below, upstream finishes before `-q` exits and the
# hit is reported correctly; on a file large enough to fill the 64KB pipe buffer
# it is lost every time. Measured on a 200k-line file: the `-q` form missed the
# hit 5 runs out of 5, the capture-then-test form found it 5 out of 5. The
# construct was inherited from the pre-PERF-008 guard, where the same line sat
# on the only path that reports a violation.
#
# The regression case at the end of this file rebuilds a >64KB file and asserts
# the hit still surfaces, because a fix with no failing test is a comment.
hits_in() { # hits_in <pattern> <file>  -> prints the offending lines, if any
    scan_file "$1" "$2" | grep -viE "$CONFIG_SUFFIX"
}

sweep() { # sweep <label> <pattern> <min-files> <file-list-on-stdin>
    local label="$1" pat="$2" floor="$3" n=0 f
    local -a found=()
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        [ "${f##*/}" = "$SELF" ] && continue   # parameter expansion, not a fork per file
        n=$((n + 1))
        if [ -n "$(hits_in "$pat" "$f")" ]; then
            found+=("$f")
        fi
    done
    # VACUITY, PER UNIVERSE. A scan over zero files sweeps clean and means
    # nothing — and a universe that silently resolves to nothing is the exact
    # trap check_no_claim_literals.sh documents: the `universe: N` line goes up
    # while coverage does not. Each universe carries its own floor so that
    # adding one cannot dilute the check into a pass.
    if [ "$n" -lt "$floor" ]; then
        printf 'FAIL  %s: scanned only %s file(s), floor is %s. The universe\n' "$label" "$n" "$floor"
        printf '      collapsed; a clean sweep over nothing is not a pass.\n'
        return 1
    fi
    if [ "${#found[@]}" -gt 0 ]; then
        printf 'FAIL  %s: fabricated comparator baseline(s):\n' "$label"
        for f in "${found[@]}"; do
            printf '      %s\n' "$f"
            hits_in "$pat" "$f" | sed 's/^/        /'
        done
        return 1
    fi
    printf 'ok    %-8s %3s file(s), no asserted competitor baseline\n' "$label" "$n"
    return 0
}

sweep shell "$PATTERN" 100 < <(
    { git ls-files 'scripts/*.sh' 'crates/*/scripts/*.sh' 2>/dev/null
      find scripts -maxdepth 2 -type f -name '*.sh' 2>/dev/null
    } | LC_ALL=C sort -u
) || rc=1

# THE CONFIG UNIVERSE, and why it is not a speculative widening. scripts/lib/
# holds bench_receipt.py and bench_threshold.py — the ONE receipt validator and
# the threshold derivation the whole epic turns on — plus .github/workflows/
# where the gates are actually invoked. Every one of those files was outside a
# shell-only guard, so a fabricated baseline in the validator that decides
# whether a receipt is honest was invisible to the guard that bans fabricated
# baselines. Rule 6 applies and was obeyed: the mutation was re-run IN this
# scope, not carried over from the shell one. See the mutation log in the PR.
sweep config "$PATTERN_CFG|$PATTERN_JSON" 30 < <(
    { git ls-files 'scripts/*.py' 'scripts/*.toml' 'scripts/*.yaml' 'scripts/*.yml' \
                   '.github/workflows/*.yml' '.github/workflows/*.yaml' 2>/dev/null
      find scripts .github/workflows -maxdepth 2 -type f \
           \( -name '*.py' -o -name '*.toml' -o -name '*.yaml' -o -name '*.yml' \) 2>/dev/null
    } | LC_ALL=C sort -u
) || rc=1

# ---------------------------------------------------------------------------
# THE RUST SITES — a SHRINK-ONLY LEDGER, and explicitly NOT a construct ban.
#
# Read this before trusting the row it prints.
#
# §9 named "the `225.0 // Ollama parity` literals in crates/aprender-serve/src/
# gguf/tests/parity*.rs" and put the count at "15+" with a `[C]` marker. Counted
# on main at ce712eae0, the spec was wrong in BOTH directions and wrong about
# the location:
#
#   225.0 in aprender-serve src/tests/examples          22 lines
#   ...of which are competitor-NAMED bindings            2   (parity016d, parity021c)
#   ...in parity*.rs at all, on any value                4
#   competitor-named bindings = bare literal, crates/**  29   <- the true count
#   "default <competitor> baseline" announcements         2
#
# So 225.0 is not the shape; parity*.rs is not the place. TWENTY of the 29 are
# in crates/aprender-serve/examples/ and benches/, which §9 does not mention,
# and one is in a different crate entirely (aprender-core/examples/
# ch22_vs_llamacpp.rs). Chasing the literal would have found 2 of 29.
#
# WHY A LEDGER AND NOT A BAN. A construct ban over Rust would be born RED
# against 31 live sites, and a guard born red is deleted or ignored within the
# week. The ledger commits the true count TODAY so the number cannot grow while
# the deletions are scheduled — §5's actual instruction: "commit the true count
# as the shrink-only baseline now, or the guard will be quietly widened later."
#
# WHAT THIS IS NOT. It is not a claim that a shell regex understands Rust. It
# recognises ONE syntactic shape — `let|const|static <name-containing-a-
# competitor> = <bare numeric literal>;` — and it will miss a fabrication
# assembled through a const table, a builder, or a match arm. Deleting the 31
# sites is PERF-008-RUST, a separate ticket, and closing it retires this block
# rather than tightening it.
RUST_LEDGER="scripts/fabricated_baseline_rust_sites.txt"
RUST_BIND='(let|const|static)[[:space:]]+(mut[[:space:]]+)?[A-Za-z0-9_]*('"$COMP"')[A-Za-z0-9_]*([[:space:]]*:[[:space:]]*[A-Za-z0-9_]+)?[[:space:]]*=[[:space:]]*[0-9]+(\.[0-9]+)?(_?f(32|64)|_?[iu](8|16|32|64|size))?[[:space:]]*[;,]'
RUST_DEFAULT='default[[:space:]]+[A-Za-z.]*('"$COMP"')[A-Za-z.]*[[:space:]]+baseline'
# The denylist is applied to the BOUND NAME ONLY, and that is load-bearing. Run
# against the whole `path:line:text`, it silently dropped three true positives:
# `beat_ollama_deCODE_...rs` matched CODE, `tests_f083_tiMINg_f084.rs` matched
# MIN, and `llamacpp_deCODE_tps` matched CODE. A denylist over a haystack that
# includes the file path is a denylist over the directory tree.
RUST_NAME_DENY='(^|_)(trials?|runs?|timeout|port|retries|retry|seconds|secs|limit|max|min|size|count|workers|threads|rc|status|code|pid|fd|level|version|index|idx|seed|iters?|iterations?)(_|$)'

# `--untracked` IS NOT DECORATION. Plain `git grep` scans tracked files only, so
# a brand-new .rs file carrying a fabricated baseline is invisible until someone
# commits it — the tracked-only-universe free pass this repo has now hit four
# times (SHIM-2644-03, check_bench_threshold.sh in PARITY-008/009, and this
# guard's own shell universe, which is why the sweep above unions a find).
# Measured before fixing: `git grep` rc=1 on an untracked fabrication, `git grep
# --untracked` rc=0. The shell universe was already immune; the Rust block was
# born with the hole, one screen away from the comment explaining it.
#
# FULL-LINE `//` COMMENTS ARE DROPPED HERE TOO, and for the reason this epic has
# now hit three times: a guard's own documentation quoting the banned pattern
# reddens a sibling. gpu_showcase_benchmark.rs:463 is literally
# `// Use default Ollama baseline from spec` sitting one line above the println
# that does it. Ledgering the comment as well as the code would mean any future
# Rust comment EXPLAINING this ban becomes a violation of it. The println on 464
# is the fabrication; the comment is a description of one.
rust_hits() {
    { git grep --untracked -nIE "$RUST_BIND" -- 'crates/**/*.rs' 2>/dev/null || true; } \
    | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(//|/\*|\*)' \
    | while IFS= read -r ln; do
        nm=$(printf '%s\n' "${ln#*:*:}" \
             | sed -E 's/.*(let|const|static)[[:space:]]+(mut[[:space:]]+)?([A-Za-z0-9_]+).*/\3/' \
             | tr 'A-Z' 'a-z')
        printf '%s\n' "$nm" | grep -qE "$RUST_NAME_DENY" && continue
        printf '%s\n' "$ln"
      done
    { git grep --untracked -nIE "$RUST_DEFAULT" -- 'crates/**/*.rs' 2>/dev/null || true; } \
    | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(//|/\*|\*)'
}

rl_new=0
rl_known=0
if [ -f "$RUST_LEDGER" ]; then
    all_rust=$(rust_hits | LC_ALL=C sort -u)
    while IFS= read -r ln; do
        [ -n "$ln" ] || continue
        loc="${ln%%:*}:$(printf '%s' "$ln" | cut -d: -f2)"
        if grep -qxF "$loc" "$RUST_LEDGER"; then
            rl_known=$((rl_known + 1))
        else
            printf 'FAIL  NEW fabricated baseline in Rust: %s\n' "$(printf '%s' "$ln" | cut -c1-140)"
            rl_new=$((rl_new + 1))
        fi
    done <<< "$all_rust"

    # THE RATCHET DIRECTION. Growth is a failure; shrinkage is the goal and must
    # be BANKED, not merely tolerated — an entry left in the ledger after its
    # site is deleted silently re-admits a fabrication at that location later.
    rl_stale=0
    while IFS= read -r loc; do
        [ -n "$loc" ] || continue
        case "$loc" in '#'*) continue ;; esac
        case " $all_rust " in *"$loc:"*) : ;; *) rl_stale=$((rl_stale + 1)) ;; esac
    done < "$RUST_LEDGER"

    if [ "$rl_new" -gt 0 ]; then
        printf '      A comparator baseline is MEASURED or ABSENT, never asserted.\n'
        printf '      The Rust ledger is SHRINK-ONLY: %s is not an\n' "$RUST_LEDGER"
        printf '      allowlist to append to. Delete the literal, or derive it.\n'
        rc=1
    else
        printf 'ok    rust     %s ledgered site(s), 0 new (shrink-only, PERF-008-RUST)\n' "$rl_known"
    fi
    if [ "$rl_stale" -gt 0 ]; then
        printf 'REPORT %s ledger entry(ies) no longer match — prune them, or the\n' "$rl_stale"
        printf '       ratchet re-admits a fabrication at that location for free.\n'
    fi
else
    printf 'FAIL  %s is missing. The Rust site count is\n' "$RUST_LEDGER"
    printf '      UNMEASURED without it, and an unmeasured ratchet is not a ratchet.\n'
    rc=1
fi

if [ "$rc" -ne 0 ]; then
    printf '      Invoke the comparator and record its output, or record the\n'
    printf '      absence explicitly so the consuming gate can treat it as RED.\n'
fi

# ---------------------------------------------------------------------------
# CASE TABLE, not a single sentinel. Every guard regex in this repo that was
# wrong -- and the pinning walker was wrong sixteen times -- was caught by a
# must-match/must-not-match table and none by reading the pattern. This one was
# already wrong once: the first version missed the JSON-literal spelling, and
# PERF-008 found it missing eleven more.
#
# NOTE THE LITERALS USED. §5's objection to the pre-PERF-008 table was that it
# rehearsed the numbers the guard was born from — 291, 120, 15 — so it proved
# string equality rather than shape recognition. Every row below that could use
# the birth numbers uses DIFFERENT ones (137, 318, 407, 44.5), and the shipped
# mutation is `${OLLAMA_BASELINE:-137}` for the same reason.
ctl=$(mktemp -d) || exit 2
cat > "$ctl/must_match" <<'CASES'
OLLAMA_BASELINE="${OLLAMA_BASELINE:-137}"
OLLAMA_CPU=44
LLAMA_TPS="318"
JSON+='"ollama_baselines":{"gpu_batched":137,"gpu_single":44,"cpu":7}'
printf '{"llamacpp_throughput": 40.7}'
readonly OLLAMA_BASELINE=407
export OLLAMA_TPS=318
CASES
# PERF-008 additions, kept in their own fixture so a regression names the shape.
cat > "$ctl/must_match_shapes" <<'CASES'
OLLAMA_BASELINES=(137 318)
declare -a OLLAMA_BASELINES=(137 44 7)
OLLAMA_BASELINE=$((137))
: "${OLLAMA_BASELINE:=137}"
ollama_baseline=137
    local llama_tps=44.5
OLLAMA_BASELINE="${BASE}37"
OLLAMA_BASELINE="1${SUFFIX}"
llamacpp_decode_tps=407
CASES
cat > "$ctl/must_not_match" <<'CASES'
OLLAMA_BASELINE="$(measure_ollama)"
OLLAMA_URL="http://localhost:11434"
echo "ollama baseline is measured, not asserted"
LLAMA_BIN="$(command -v llama-bench)"
LLAMA_PIN_RC=3
OLLAMA_EXIT_CODE=1
LLAMA_LOG_LEVEL=2
JSON+='"ollama_baselines":null'
readonly OLLAMA_TIMEOUT_SECONDS=30
OLLAMA_BASELINES=("$(measure_ollama)")
ollama_trials=5
[ "$OLLAMA_TPS" = 318 ]
OLLAMA_BASELINE="${OLLAMA_BASELINE:?measure it}"
CASES
# The CONFIG universe has its own table: PATTERN_CFG is a different pattern over
# a different file set, and Rule 6 says the shell proof does not transfer.
cat > "$ctl/cfg_must_match" <<'CASES'
ollama_baseline = 137
ollama_tps: 318
  llamacpp_throughput: 40.7
ollama_baselines = [137, 318]
    - vllm_latency_ms: 44
"ollama_baseline": 318
ollama_baseline_seconds: 30
CASES
cat > "$ctl/cfg_must_not_match" <<'CASES'
ollama_layers = 32
llama_context_size: 4096
ollama_baseline = measured_value
ollama_url = "http://localhost:11434"
gpu_layers: 99
ollama_baseline_timeout: 30
CASES

tbl_bad=0
run_tbl() { # run_tbl <file> <pattern> <expect match|nomatch>
    local file="$1" pat="$2" want="$3" line got
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        printf '%s\n' "$line" > "$ctl/one"
        if [ -n "$(grep -E "$pat" "$ctl/one" | grep -viE "$CONFIG_SUFFIX")" ]; then got=match; else got=nomatch; fi
        if [ "$got" != "$want" ]; then
            printf 'FAIL  want %-7s got %-7s : %s\n' "$want" "$got" "$line"
            tbl_bad=1
        fi
    done < "$file"
}
run_tbl "$ctl/must_match"          "$PATTERN"     match
run_tbl "$ctl/must_match_shapes"   "$PATTERN"     match
run_tbl "$ctl/must_not_match"      "$PATTERN"     nomatch
run_tbl "$ctl/cfg_must_match"      "$PATTERN_CFG|$PATTERN_JSON" match
run_tbl "$ctl/cfg_must_not_match"  "$PATTERN_CFG|$PATTERN_JSON" nomatch

# The Rust ledger's detector gets a table too, on the same terms.
cat > "$ctl/rust_must_match" <<'CASES'
    let ollama_baseline = 137.0;
const OLLAMA_BASELINE_TOKS: f64 = 318.0;
    let llamacpp_tps: f64 = 407.0;
    let ollama_toks = 44.5f32;
        println!("Using default Ollama baseline (318 tok/s from spec)");
    let llamacpp_decode_tps = 407.0_f64;
CASES
cat > "$ctl/rust_must_not_match" <<'CASES'
    let ollama_tok_s = measure_ollama()?;
const OLLAMA_TRIALS: usize = 5;
    let ollama_ratio = ours / theirs;
    let decode_tps = 407.0;
CASES
while IFS= read -r line; do
    [ -n "$line" ] || continue
    printf '%s\n' "$line" > "$ctl/one"
    got=nomatch
    if grep -qE "$RUST_BIND|$RUST_DEFAULT" "$ctl/one"; then
        nm=$(printf '%s\n' "$line" \
             | sed -E 's/.*(let|const|static)[[:space:]]+(mut[[:space:]]+)?([A-Za-z0-9_]+).*/\3/' \
             | tr 'A-Z' 'a-z')
        grep -qE "$RUST_NAME_DENY" <<< "$nm" || got=match
        grep -qE "$RUST_DEFAULT" "$ctl/one" && got=match
    fi
    [ "$got" = match ] || { printf 'FAIL  rust want match   got %s : %s\n' "$got" "$line"; tbl_bad=1; }
done < "$ctl/rust_must_match"
while IFS= read -r line; do
    [ -n "$line" ] || continue
    printf '%s\n' "$line" > "$ctl/one"
    got=nomatch
    if grep -qE "$RUST_BIND|$RUST_DEFAULT" "$ctl/one"; then
        nm=$(printf '%s\n' "$line" \
             | sed -E 's/.*(let|const|static)[[:space:]]+(mut[[:space:]]+)?([A-Za-z0-9_]+).*/\3/' \
             | tr 'A-Z' 'a-z')
        grep -qE "$RUST_NAME_DENY" <<< "$nm" || got=match
        grep -qE "$RUST_DEFAULT" "$ctl/one" && got=match
    fi
    [ "$got" = nomatch ] || { printf 'FAIL  rust want nomatch got %s : %s\n' "$got" "$line"; tbl_bad=1; }
done < "$ctl/rust_must_not_match"

# SIGPIPE REGRESSION CASE. Every fixture above is one line long, and on a
# one-line file the `-q` form and the capture form behave identically — so the
# whole table above stays green against a revert to `| grep -q`. This case is
# the only one that can tell them apart.
#
# WHAT HAS TO BE BIG IS THE UPSTREAM OUTPUT, NOT THE FILE. The first version of
# this fixture padded with non-matching lines: a 293KB file, and `grep -nE`
# emitted exactly ONE line from it. One line never fills the pipe buffer, no
# writer ever blocks, no SIGPIPE — and the deliberate revert to `grep -q` sailed
# through GREEN. The fixture proved nothing and said "ok" while doing it, which
# is the failure it exists to catch, committed into the catcher.
#
# So the padding MATCHES. `grep -q` exits on line 2 while the writer still has
# thousands of matching lines to push, the writer takes SIGPIPE, and pipefail
# reports 141 for a pipeline whose reader succeeded.
big="$ctl/big_sigpipe.sh"
{ printf '#!/usr/bin/env bash\n'
  printf 'OLLAMA_BASELINE=137\n'
  awk 'BEGIN{for(i=0;i<8000;i++) printf "OLLAMA_TPS=%d\n", i}'
} > "$big"
upstream_bytes=$(scan_file "$PATTERN" "$big" | wc -c)
if [ "$upstream_bytes" -lt 65536 ]; then
    printf 'FAIL  sigpipe fixture: upstream emits only %s bytes. Under the 64KB pipe\n' "$upstream_bytes"
    printf '      buffer nothing blocks, so this case cannot distinguish `grep -q`\n'
    printf '      from capture-then-test and proves nothing. Pad with MATCHING lines.\n'
    tbl_bad=1
elif [ -z "$(hits_in "$PATTERN" "$big")" ]; then
    printf 'FAIL  SIGPIPE regression: a fabrication on line 2 was NOT reported from a\n'
    printf '      file whose scan emits %s bytes. `scan_file | grep -q` under pipefail\n' "$upstream_bytes"
    printf '      returns 141 when the reader exits early — a silent free pass.\n'
    tbl_bad=1
fi

if [ "$tbl_bad" -eq 0 ]; then
    printf 'ok    sigpipe: hit found though upstream emits %s bytes (>64KB buffer)\n' "$upstream_bytes"
    printf 'ok    case table: %s shell must-match, %s must-not-match, %s cfg, %s rust — all correct\n' \
        "$(( $(grep -c . "$ctl/must_match") + $(grep -c . "$ctl/must_match_shapes") ))" \
        "$(grep -c . "$ctl/must_not_match")" \
        "$(( $(grep -c . "$ctl/cfg_must_match") + $(grep -c . "$ctl/cfg_must_not_match") ))" \
        "$(( $(grep -c . "$ctl/rust_must_match") + $(grep -c . "$ctl/rust_must_not_match") ))"
else
    rc=1
fi
rm -rf "${ctl:?}"

# ---------------------------------------------------------------------------
# RESIDUAL — shapes measured as MISSED and deliberately NOT claimed. Recorded
# here because an unlisted gap reads as coverage, which is the F12 defect
# applied to the guard itself.
#
#   INDIRECT ASSEMBLY.  BASE="1"; BASE="${BASE}37"; OLLAMA_BASELINE="$BASE"
#     The competitor-named variable never touches a literal. Undecidable for a
#     line-oriented matcher; it needs dataflow. The DIRECT form
#     (`OLLAMA_BASELINE="${BASE}37"`) is covered and is in the table.
#
#   MULTI-LINE FUNCTION BODY.  ollama_baseline() { \n echo 137 \n }
#     The single-line form is caught by PATTERN_VAR only when the echo shares
#     the line. A body spanning lines is out of reach without a parser.
#
#   CASE ARM.  ollama) BASELINE=137 ;;
#     The competitor name is the LABEL and the variable is generic, so neither
#     the name nor the value alone is suspicious. A pattern keyed on
#     "competitor word anywhere on a line with a numeric assignment" was tried
#     and rejected: it flags `# ollama uses 4 threads` and every prose line in
#     a script, and a guard that flags everything is as broken as one that
#     flags nothing.
#
#   RUST, beyond the one binding shape — see the ledger block above.
#
printf '\n'
if [ "$rc" -eq 0 ]; then
    printf 'PASS  no comparator baseline is asserted rather than measured.\n'
else
    printf 'FAIL  see rows above (#2679, #2706 PERF-008).\n'
fi
exit "$rc"
