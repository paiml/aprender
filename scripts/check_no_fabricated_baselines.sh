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

# ---------------------------------------------------------------------------
# MODES, AND WHY AN UNKNOWN ARGUMENT IS FATAL.
#
# `--selftest` (the repo convention, cf. check_no_claim_literals.sh:140) runs the
# case tables alone: the regexes and the Rust and docs bans are exercised against
# fixtures, with no repository scan. The plain invocation — what ci.yml:927 runs
# — does the scan AND the tables.
#
# Before this pass the script read no arguments at all, so `--selftest` "worked"
# by being IGNORED: it ran the full scan and exited 0, and so did `--slftest`,
# and so did `--no-really-dont-check-anything`. Measured on the parent commit:
# `bash scripts/check_no_fabricated_baselines.sh --this-flag-does-not-exist`
# exited 0. A flag that is silently ignored is a flag that does not exist, and
# the caller cannot tell the difference between "the mode ran" and "the mode is
# not implemented" — this repo has already paid for that once, when a misspelled
# nextest key meant NO timeout and killed the merge queue.
MODE=full
case "${1:-}" in
    '')         : ;;
    --selftest) MODE=selftest ;;
    *)  printf 'usage: %s [--selftest]\n' "${0##*/}" >&2
        printf 'refusing to run: unknown argument %s. Exiting 2 rather than\n' "$1" >&2
        printf 'silently running something the caller did not ask for.\n' >&2
        exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
    printf 'refusing to run: %s takes at most one argument, got %s.\n' "${0##*/}" "$#" >&2
    exit 2
fi

TMPD=$(mktemp -d) || exit 2
trap 'rm -rf "${TMPD:?}"' EXIT

rc=0
printf -- '--- no fabricated comparator baselines (F12) [mode: %s] -------------\n' "$MODE"

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
# It is input-size dependent, which is why nothing noticed — and the ONSET IS
# FAR LOWER THAN THE PIPE BUFFER, which is why "64KB" was the wrong number to
# reason with. Measured against a verbatim reconstruction of the parent
# construct (`scan_file "$f" | grep -qvE "$CONFIG_SUFFIX"`, bash, pipefail), 20
# runs per size, counting how often the hit was LOST:
#
#     upstream bytes   3584  7384  9284 10234 11184 13084 39786 165786
#     hit lost (of 20)    0     0     0     2    19    20    20     20
#
# The producer here is itself a two-stage pipe (`grep -nE | grep -vE`), so what
# has to fill is the FIRST stage's buffered output plus one pipe, not 64KB: loss
# begins around 10KB and is total by 13KB. Between 9KB and 11KB it is a RACE —
# the same input, the same binary, lost 2/20 then 19/20 — so a guard sitting in
# that band reds and greens at random on identical input.
#
# zsh does NOT reproduce it; this is a bash + `set -o pipefail` behaviour, and
# the reproduction must be run under bash or it reports the defect as absent.
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

# THE SCANS RUN IN `full` MODE ONLY. `--selftest` exercises the rules against
# fixtures; it deliberately does not touch the repository, so it stays valid on a
# tree that is mid-edit and can be run by anyone changing a regex here.
if [ "$MODE" = full ]; then
    sweep shell "$PATTERN" 100 < <(
        { git ls-files 'scripts/*.sh' 'crates/*/scripts/*.sh' 2>/dev/null
          find scripts -maxdepth 2 -type f -name '*.sh' 2>/dev/null
        } | LC_ALL=C sort -u
    ) || rc=1
fi

# THE CONFIG UNIVERSE, and why it is not a speculative widening. scripts/lib/
# holds bench_receipt.py and bench_threshold.py — the ONE receipt validator and
# the threshold derivation the whole epic turns on — plus .github/workflows/
# where the gates are actually invoked. Every one of those files was outside a
# shell-only guard, so a fabricated baseline in the validator that decides
# whether a receipt is honest was invisible to the guard that bans fabricated
# baselines. Rule 6 applies and was obeyed: the mutation was re-run IN this
# scope, not carried over from the shell one. See the mutation log in the PR.
if [ "$MODE" = full ]; then
    sweep config "$PATTERN_CFG|$PATTERN_JSON" 30 < <(
        { git ls-files 'scripts/*.py' 'scripts/*.toml' 'scripts/*.yaml' 'scripts/*.yml' \
                       '.github/workflows/*.yml' '.github/workflows/*.yaml' 2>/dev/null
          find scripts .github/workflows -maxdepth 2 -type f \
               \( -name '*.py' -o -name '*.toml' -o -name '*.yaml' -o -name '*.yml' \) 2>/dev/null
        } | LC_ALL=C sort -u
    ) || rc=1
fi

# ---------------------------------------------------------------------------
# THE RUST SITES — a BAN since #3773. Until then this block was a SHRINK-ONLY
# LEDGER (scripts/fabricated_baseline_rust_sites.txt, 36 coordinates, PERF-008-RUST)
# whose stated purpose was to hold the count while the deletions were scheduled.
# #3773 did the deletions: every non-test site was deleted, measured through the
# pinned comparator, or labelled with its dated receipt, so the ledger's reason to
# exist ended and it is retired into the ban it was always meant to become.
#
# WHY THE LEDGER'S SHAPE WAS NOT ENOUGH. `apr showcase` printed
# `35.0 + generate_jitter()` tok/s as llama.cpp's throughput (#3773), bound to a
# variable called `tps` inside `fn run_llama_cpp_bench`. The ledger recognised
# one shape — a binding whose NAME names a competitor — and `tps` names nothing,
# so the most visible fabrication in the tree was invisible to the guard built for
# it. Its Ollama twin, `.map_or(200.0, …)` as a parse-failure fallback, was a
# second shape the ledger could not see. So the ban keeps the old shape and adds
# the one that was missed:
#
#   R1  `let|const|static <name naming a competitor> = <bare numeric literal>`
#       (the ledger's shape, name denylist unchanged);
#   R2  "default <competitor> baseline" announcements;
#   R3  inside a `fn` whose NAME names a competitor: a throughput-named binding
#       (`tps`, `tok_s`, `toks`, `throughput`, `ttft`, `baseline`, …) initialised
#       from a non-zero numeric literal, or a `.map_or(N` / `.unwrap_or(N` numeric
#       fallback. A zero initialiser is an accumulator, not a claim.
#
# A RECEIPT, NOT AN ALLOWLIST. A site is legal iff a `receipt:` comment carrying a
# YYYY-MM-DD date sits on the same line or within the 3 lines above — a dated,
# receipted historical figure (crates/aprender-core/examples/ch22_vs_llamacpp.rs
# cites the bootstrap JSON its number came from). There is no file of exempt
# coordinates for an edit to append to.
#
# TEST FILES ARE OUT OF SCOPE, by ruling (cop on #3773: "test fixtures stay"): a
# path with a `tests/` component, a file named test*.rs / *_test(s).rs /
# tests_*.rs, or the lines after a `#[cfg(test)]` module header. Examples and
# benches ARE in scope: they print numbers a user reads.
#
# ONE DECISION FUNCTION, CALLED BY BOTH THE SCAN AND THE CASE TABLE (the lesson
# of the table that once graded a copy of the rule): `rust_ban_scan` below is the
# only place R1-R3, the receipt and the test exclusion exist.
#
# WHAT THIS IS NOT. It is not a Rust parser. The function context is the last
# `fn <name>` seen above a line, which is exact for the shapes above and
# approximate for closures; a fabrication assembled through a const table, a
# builder or a match arm is RESIDUAL (listed at the end of this file).
# THE PATH LIST IS A FILE, NOT ARGV. The universe is ~10k paths; passed as
# arguments that is ~0.6 MB of argv, inside Linux's limit on a quiet shell and
# not on a CI container whose environment is large. An E2BIG there would fail
# the exec, print nothing — and nothing is what a clean tree prints.
rust_ban_scan() { # rust_ban_scan <file-of-paths>  -> "HIT\tpath:line:text" | "RECEIPTED\tpath:line"
    COMP="$COMP" python3 - "$1" <<'PY'
import os, re, sys
COMP = os.environ["COMP"]
comp = re.compile(COMP)
BIND = re.compile(r"(let|const|static)\s+(mut\s+)?([A-Za-z0-9_]+)(\s*:\s*[A-Za-z0-9_]+)?\s*=\s*[0-9]+(\.[0-9]+)?(_?f(32|64)|_?[iu](8|16|32|64|size))?\s*[;,]")
DEFAULT = re.compile(r"default\s+[A-Za-z.]*" + COMP + r"[A-Za-z.]*\s+baseline")
NAME_DENY = re.compile(r"(^|_)(trials?|runs?|timeout|port|retries|retry|seconds|secs|limit|max|min|size|count|workers|threads|rc|status|code|pid|fd|level|version|index|idx|seed|iters?|iterations?)(_|$)")
TPUT_NAME = re.compile(r"(^|_)(tps|tok_s|toks|tok_per_s|tokens_per_sec|throughput|ttft|ttft_ms|baseline)(_|$)")
LIT_INIT = re.compile(r"(let|const|static)\s+(mut\s+)?([A-Za-z0-9_]+)(\s*:\s*[A-Za-z0-9_]+)?\s*=\s*([0-9]+(\.[0-9]+)?)(_?f(32|64))?\b")
LIT_BIND = re.compile(r"(let|const|static)\s+(mut\s+)?([A-Za-z0-9_]+)")
TPUT_WORD = re.compile(r"(\btps\b|_tps\b|tok_s|\btoks\b|tok/s|tokens_per_sec|throughput|\bttft|baseline)", re.I)
FALLBACK = re.compile(r"\.(map_or|unwrap_or)\(\s*([0-9]+(\.[0-9]+)?)(_?f(32|64))?\s*[,)]")
FN = re.compile(r"\bfn\s+([A-Za-z0-9_]+)")
RECEIPT = re.compile(r"receipt:.*\b20[0-9]{2}-[0-9]{2}-[0-9]{2}\b")
TEST_PATH = re.compile(r"(^|/)tests?/|(^|/)(tests?_[^/]*|[^/]*_tests?|test_[^/]*)\.rs$")
COMMENT = re.compile(r"^\s*(//|/\*|\*)")
CFG_TEST = re.compile(r"^#\[cfg\(test\)\]\s*$")

def nonzero(v):
    return float(v) != 0.0

with open(sys.argv[1], encoding="utf-8") as fh:
    paths = [p for p in fh.read().split("\n") if p]
for path in paths:
    if TEST_PATH.search(path):
        continue
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    except OSError:
        continue
    fn = ""
    for i, line in enumerate(lines):
        if CFG_TEST.match(line) and any(l.lstrip().startswith("mod ") for l in lines[i + 1:i + 3]):
            break  # the rest of the file is a test module
        m = FN.search(line)
        if m and not COMMENT.match(line):
            fn = m.group(1).lower()
        if COMMENT.match(line):
            continue
        hit = False
        b = BIND.search(line)
        if b and comp.search(b.group(3)) and not NAME_DENY.search(b.group(3).lower()):
            hit = True  # R1
        if DEFAULT.search(line):
            hit = True  # R2
        if fn and comp.search(fn):
            li = LIT_INIT.search(line)
            if li and TPUT_NAME.search(li.group(3).lower()) and nonzero(li.group(5)):
                hit = True  # R3 binding
        # A fallback is a COMPETITOR THROUGHPUT fallback when the statement it
        # sits in (this line or the 3 above) is about throughput, and either the
        # fn or that statement names a competitor: `let llamacpp_tps = …
        # .map_or(200.0, …)` in `fn check_point_41` names it in the binding, not
        # the fn. A default temperature or buffer size is a setting, not a claim.
        stmt = "\n".join(lines[max(0, i - 3):i + 1])
        fb = FALLBACK.search(line)
        # "About throughput" is decided by what the fallback STANDS IN FOR: the
        # value it defaults (`.map_or(318.0, |s| s.mean_throughput)`, on this
        # line) or the binding it initialises (`let tps = … .map_or(200.0, …)`).
        # A `min_speedup.unwrap_or(0.2)` next to a comment about Ollama is a
        # ratio threshold, not a competitor's throughput.
        binds_tput = any(
            (m := LIT_BIND.search(l)) and TPUT_NAME.search(m.group(3).lower())
            for l in lines[max(0, i - 3):i + 1])
        if (fb and nonzero(fb.group(2))
                and (TPUT_WORD.search(line) or binds_tput)
                and ((fn and comp.search(fn)) or comp.search(stmt))):
            hit = True  # R3 fallback
        if not hit:
            continue
        window = lines[max(0, i - 3):i + 1]
        if any(RECEIPT.search(w) for w in window):
            print(f"RECEIPTED\t{path}:{i + 1}")
        else:
            print(f"HIT\t{path}:{i + 1}:{line.strip()[:140]}")
PY
}

rust_universe() {
    { git ls-files --cached --others --exclude-standard -- 'crates/*.rs' 'src/*.rs' 2>/dev/null; } \
        | LC_ALL=C sort -u
}

rust_ban_sweep() {
    local n hits receipted
    rust_universe > "$TMPD/rust_universe"
    n=$(grep -c . "$TMPD/rust_universe" || true)
    # A universe that silently resolves to nothing reports a perfectly clean tree.
    if [ "$n" -lt 1000 ]; then
        printf 'FAIL  rust     scanned only %s file(s), floor is 1000. The universe\n' "$n"
        printf '               query resolved to (almost) nothing — a vacuous PASS.\n'
        rc=1
        return
    fi
    # A scanner that dies (no python3, a syntax error) prints nothing, and
    # nothing is exactly what a clean tree prints. Its status is the verdict.
    if ! rust_ban_scan "$TMPD/rust_universe" > "$TMPD/rust_scan"; then
        printf 'FAIL  rust     the scanner itself failed; the tree is UNMEASURED.\n'
        rc=1
        return
    fi
    hits=$(grep -c '^HIT' "$TMPD/rust_scan" || true)
    receipted=$(grep -c '^RECEIPTED' "$TMPD/rust_scan" || true)
    if [ "$hits" -gt 0 ]; then
        while IFS=$'\t' read -r _ where; do
            printf 'FAIL  rust     unreceipted competitor throughput: %s\n' "$where"
        done < <(grep '^HIT' "$TMPD/rust_scan")
        printf '      A comparator figure is MEASURED (the pinned comparator env,\n'
        printf '      scripts/llama_bin.sh) or RECEIPTED (a `// receipt: <path> (YYYY-MM-DD)`\n'
        printf '      comment within 3 lines above), or it is deleted. There is no ledger\n'
        printf '      to append to (#3773).\n'
        rc=1
        return
    fi
    printf 'ok    rust     %s file(s), 0 unreceipted competitor throughput, %s receipted\n' \
        "$n" "$receipted"
}

[ "$MODE" = full ] && rust_ban_sweep

# ---------------------------------------------------------------------------
# THE PUBLISHED DOCS — the same ban, on the pages a user reads (#3773).
#
# crates/apr-cli/README.md is the crates.io page for the `apr` binary, and it
# said "2.9x faster than Ollama" with nothing behind it; a sample-output block
# further down said "755+ tok/s (2.6x Ollama)". check_no_claim_literals.sh bans
# ratio claims over book/, docs/ and the ROOT README — and its universe never
# included crates/*/README.md, which is where those two sat. So this guard owns
# the published competitor figure directly, over the surfaces the cop's ruling
# named: the root README.md, every crates/*/README.md, and docs/BEATS.md.
#
#   D1  a line naming a competitor (ollama, llama.cpp / llama-server /
#       llama-bench / llama-cli, vllm, tgi, sglang, tensorrt, pytorch, unsloth)
#       that carries a THROUGHPUT figure — `N tok/s` (t/s, tokens/s), or a ratio
#       fastened to the competitor's name (`2.6x Ollama`, `1.109× ollama`,
#       `Ollama 1.2×`).
#   D2  a figure in a TABLE CELL under a column whose HEADER names a competitor.
#       main's apr-cli README carried `| Mode | Throughput | vs Ollama | Memory |`
#       over rows like `| GPU (batched) | ~850 tok/s | 2.9x | 1.9 GB |`: the row
#       names no competitor, so D1 alone reads every one of them as clean.
#
# Bare "llama" is deliberately NOT a competitor here: `TinyLlama 1.1B: 90 tok/s`
# is apr's own number on a Llama-family model, and a docs rule that reads model
# names as competitors reds every model card in the tree.
#
# A RECEIPT ON THE SAME LINE: a YYYY-MM-DD date AND a pointer to what was
# measured (a .json/.yaml path, evidence/, results/, contracts/, or a
# `beat-…-vN` contract id). Same line, not the paragraph: in a table the rows
# share a paragraph, and a window that wide lets one row's receipt license its
# neighbour — measured on BEATS.md, where the llama.cpp row's receipt would have
# "receipted" the Ollama row above it.
#
# Code blocks are scanned: a sample-output block on a crates.io page is
# published text, and one of the two README sites above was in one.
docs_universe() {
    { git ls-files --cached --others --exclude-standard -- \
          ':(glob)README.md' ':(glob)crates/*/README.md' 'docs/BEATS.md' 2>/dev/null; } \
        | LC_ALL=C sort -u
}

docs_ban_scan() { # docs_ban_scan <file-of-paths>  -> "HIT\tpath:line:text" | "RECEIPTED\tpath:line"
    python3 - "$1" <<'PY'
import re, sys
COMP = r"(ollama|llama\.cpp|llama-server|llama-bench|llama-cli|llamacpp|vllm|tgi|sglang|tensorrt(-llm)?|pytorch|unsloth)"
comp = re.compile(r"\b" + COMP + r"\b", re.I)
TPUT = re.compile(r"\d[\d,]*(\.\d+)?\+?\s*(tok/s|tokens?/s(ec)?|t/s)\b", re.I)
RATIO = re.compile(
    r"\d+(\.\d+)?\s*[x×]\**\s+(faster\s+than\s+|vs\.?\s+|over\s+)?\**" + COMP + r"\b"
    r"|\b" + COMP + r"\**\s+~?\d+(\.\d+)?\s*[x×]", re.I)
DATE = re.compile(r"\b20[0-9]{2}-[0-9]{2}-[0-9]{2}\b")
POINTER = re.compile(r"(\.json\b|\.ya?ml\b|evidence/|results/|contracts/|\bbeat-[a-z0-9-]+-v[0-9]+\b)")
SEP = re.compile(r"^\s*\|?\s*:?-{3,}")
CELL_FIG = re.compile(r"\d+(\.\d+)?\s*[x×]|^\W*~?\d+(\.\d+)?(\W|$)", re.I)

def cells(row):
    return [c.strip() for c in row.strip().strip("|").split("|")]

with open(sys.argv[1], encoding="utf-8") as fh:
    paths = [p for p in fh.read().split("\n") if p]
for path in paths:
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    except OSError:
        continue
    comp_cols = []  # D2: columns of the current table whose HEADER names a competitor
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            comp_cols = []
        elif i + 1 < len(lines) and SEP.match(lines[i + 1]) and not SEP.match(line):
            comp_cols = [j for j, c in enumerate(cells(line)) if comp.search(c)]
            continue
        d2 = False
        if comp_cols and not SEP.match(line):
            row = cells(line)
            d2 = any(j < len(row) and (CELL_FIG.search(row[j]) or TPUT.search(row[j]))
                     for j in comp_cols)
        d1 = comp.search(line) and (TPUT.search(line) or RATIO.search(line))
        if not (d1 or d2):
            continue
        if DATE.search(line) and POINTER.search(line):
            print(f"RECEIPTED\t{path}:{i + 1}")
        else:
            print(f"HIT\t{path}:{i + 1}:{line.strip()[:140]}")
PY
}

docs_ban_sweep() {
    local n hits receipted
    docs_universe > "$TMPD/docs_universe"
    n=$(grep -c . "$TMPD/docs_universe" || true)
    # The universe is small by nature (one root README, one per crate, BEATS),
    # so the floor is low; what it catches is the query resolving to nothing.
    if [ "$n" -lt 20 ]; then
        printf 'FAIL  docs     scanned only %s file(s), floor is 20. The universe\n' "$n"
        printf '               query resolved to (almost) nothing — a vacuous PASS.\n'
        rc=1
        return
    fi
    if ! docs_ban_scan "$TMPD/docs_universe" > "$TMPD/docs_scan"; then
        printf 'FAIL  docs     the scanner itself failed; the docs are UNMEASURED.\n'
        rc=1
        return
    fi
    hits=$(grep -c '^HIT' "$TMPD/docs_scan" || true)
    receipted=$(grep -c '^RECEIPTED' "$TMPD/docs_scan" || true)
    if [ "$hits" -gt 0 ]; then
        while IFS=$'\t' read -r _ where; do
            printf 'FAIL  docs     unreceipted competitor throughput: %s\n' "$where"
        done < <(grep '^HIT' "$TMPD/docs_scan")
        printf '      A published competitor figure carries its date AND what it was\n'
        printf '      measured from (a receipt path or the beat contract) on the same\n'
        printf '      line, or it is deleted (#3773).\n'
        rc=1
        return
    fi
    printf 'ok    docs     %s file(s), 0 unreceipted competitor throughput, %s receipted\n' \
        "$n" "$receipted"
}

[ "$MODE" = full ] && docs_ban_sweep

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
ctl="$TMPD/cases"
mkdir -p "$ctl" || exit 2
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

# THE RUST BAN'S TABLE CALLS THE SHIPPED FUNCTION on fixture FILES, because
# R3 is a function-context rule and a one-line fixture has no function. Each row
# is a file and its expected verdict. The literals avoid the ones the ban was
# born from (35.0, 318, 200), for the reason given above the shell table.
#
# THE FIXTURES ARE SCANNED BY RELATIVE PATH from inside their own directory. The
# test-path exclusion keys on a `tests/` component, so an absolute path under a
# TMPDIR that happened to contain one would exclude EVERY fixture and the
# must-HIT rows would fail loudly — correct, but for the wrong reason.
rs="$ctl/rust"
mkdir -p "$rs/tests" || exit 2
: > "$ctl/rust_want"
rs_case() { # rs_case <relative file> <expect HIT|RECEIPTED|none> <content, \n-escaped>
    printf '%b' "$3" > "$rs/$1"
    printf '%s\t%s\n' "$1" "$2" >> "$ctl/rust_want"
}
# R1 — a competitor-named binding to a bare literal (the ledger's shape).
rs_case r1_let.rs         HIT 'fn main() {\n    let ollama_baseline = 137.0;\n}\n'
rs_case r1_const.rs       HIT 'const OLLAMA_BASELINE_TOKS: f64 = 407.0;\n'
rs_case r1_typed.rs       HIT 'fn main() {\n    let llamacpp_tps: f64 = 407.0;\n}\n'
rs_case r1_suffix.rs      HIT 'fn main() {\n    let llamacpp_decode_tps = 407.0_f64;\n}\n'
# R2 — the announcement.
rs_case r2_default.rs     HIT 'fn main() {\n    println!("Using default Ollama baseline (137 tok/s from spec)");\n}\n'
# R3 — the #3773 shapes: a generic name inside a competitor-named fn.
rs_case r3_bind.rs        HIT 'fn run_llama_cpp_bench() -> f64 {\n    let tps = 44.0 + jitter();\n    tps\n}\n'
rs_case r3_ttft.rs        HIT 'fn bench_vllm() {\n    let ttft_ms = 44.5;\n}\n'
rs_case r3_fallback.rs    HIT 'fn ollama_tps(r: &str) -> f64 {\n    parse(r).map_or(137.0, |v| v.tps)\n}\n'
rs_case r3_stmt.rs        HIT 'fn check_point_41(&self) -> bool {\n    let llamacpp_tps = self\n        .llamacpp_stats\n        .as_ref()\n        .map_or(407.0, |s| s.mean_throughput);\n    true\n}\n'
# Not a claim: measured, a setting, an accumulator, a ratio, a comment.
rs_case n_measured.rs     none 'fn main() {\n    let ollama_tok_s = measure_ollama()?;\n}\n'
rs_case n_trials.rs       none 'const OLLAMA_TRIALS: usize = 5;\n'
rs_case n_ratio.rs        none 'fn main() {\n    let ollama_ratio = ours / theirs;\n}\n'
rs_case n_generic.rs      none 'fn decode() {\n    let decode_tps = 407.0;\n}\n'
rs_case n_accum.rs        none 'fn run_llama_cpp_bench() {\n    let mut tps = 0.0;\n}\n'
rs_case n_zero_fb.rs      none 'fn run_ollama() {\n    let tps = r.eval_count.map_or(0.0, |c| c as f64);\n}\n'
rs_case n_setting.rs      none 'fn run_ollama(temp: Option<f32>) {\n    let t = temp.unwrap_or(0.7);\n}\n'
rs_case n_ctx.rs          none 'fn ollama_config() {\n    let ctx = x.unwrap_or(4096);\n}\n'
rs_case n_threshold.rs    none 'fn gate() {\n    // Ollama parity floor\n    let min = args.min_speedup.unwrap_or(0.2);\n}\n'
rs_case n_comment.rs      none '// let ollama_baseline = 137.0;\n'
# Test code is out of scope, by ruling: a tests/ path, a tests_* file, a
# #[cfg(test)] module. The same line OUTSIDE the module is in scope.
rs_case tests/r1_let.rs   none 'fn f() {\n    let ollama_baseline = 137.0;\n}\n'
rs_case tests_fixture.rs  none 'fn f() {\n    let ollama_baseline = 137.0;\n}\n'
rs_case n_cfgtest.rs      none 'fn real() {}\n\n#[cfg(test)]\nmod tests {\n    fn f() {\n        let ollama_baseline = 137.0;\n    }\n}\n'
rs_case cfgtest_above.rs  HIT  'fn real() {\n    let ollama_baseline = 137.0;\n}\n\n#[cfg(test)]\nmod tests {}\n'
# A receipt is dated and near. Undated, or too far above, is not a receipt.
rs_case rc_dated.rs       RECEIPTED '// receipt: evidence/bootstrap.json (2026-04-05)\nconst LLAMA_CPP_TPS: f64 = 407.1;\n'
rs_case rc_window.rs      RECEIPTED '// receipt: evidence/bootstrap.json (2026-04-05)\n//\n//\nconst LLAMA_CPP_TPS: f64 = 407.1;\n'
rs_case rc_undated.rs     HIT '// receipt: evidence/bootstrap.json\nconst LLAMA_CPP_TPS: f64 = 407.1;\n'
rs_case rc_far.rs         HIT '// receipt: evidence/bootstrap.json (2026-04-05)\n//\n//\n//\nconst LLAMA_CPP_TPS: f64 = 407.1;\n'
cut -f1 "$ctl/rust_want" > "$ctl/rust_list"
if ( cd "$rs" && rust_ban_scan "$ctl/rust_list" ) > "$ctl/rust_out"; then
    while IFS=$'\t' read -r f want; do
        got=none
        grep -qF "RECEIPTED"$'\t'"$f:" "$ctl/rust_out" && got=RECEIPTED
        grep -qF "HIT"$'\t'"$f:" "$ctl/rust_out" && got=HIT
        if [ "$got" != "$want" ]; then
            printf 'FAIL  rust want %-9s got %-9s : %s\n' "$want" "$got" "$f"
            tbl_bad=1
        fi
    done < "$ctl/rust_want"
else
    printf 'FAIL  rust table: the scanner itself failed, so every rust row is UNTESTED.\n'
    tbl_bad=1
fi
rs_rows=$(grep -c . "$ctl/rust_want")

# THE DOCS BAN'S TABLE, on the same terms: fixture files through the shipped
# `docs_ban_scan`. The two README sites #3773 removed are rows here, with the
# literals changed (2.9x -> 3.4x, 2.6x -> 1.7x) for the reason given above.
ds="$ctl/docs"
mkdir -p "$ds" || exit 2
: > "$ctl/docs_want"
ds_case() { # ds_case <file> <expect HIT|RECEIPTED|none> <content, \n-escaped>
    printf '%b' "$3" > "$ds/$1"
    printf '%s\t%s\n' "$1" "$2" >> "$ctl/docs_want"
}
ds_case d_headline.md     HIT  '- **Speed**: 3.4x faster than Ollama on GPU\n'
ds_case d_sample.md       HIT  '```\nPerformance: 811+ tok/s (1.7x Ollama)\n```\n'
ds_case d_tps.md          HIT  '| llama.cpp CUDA | 407 tok/s |\n'
ds_case d_suffix_ratio.md HIT  'apr is at **1.3×** ollama on this host.\n'
ds_case d_comp_first.md   HIT  'vs Ollama 1.2x on decode\n'
ds_case d_date_only.md    HIT  'llama.cpp 407 tok/s, measured 2026-04-05\n'
ds_case d_pointer_only.md HIT  'llama.cpp 407 tok/s (`results/bootstrap.json`)\n'
ds_case d_row_leak.md     HIT  '| Ollama | 407 tok/s | - |\n| llama.cpp | 399 tok/s | 2026-04-05 `results/b.json` |\n'
ds_case d_receipted.md    RECEIPTED 'llama.cpp 407 tok/s (receipt: `results/bootstrap.json`, 2026-04-05)\n'
ds_case d_contract.md     RECEIPTED '| apr **1.02×** ollama (2026-07-31) | `beat-ollama-decode-throughput-speed-v1` |\n'
ds_case d2_header.md      HIT  '| Mode | Throughput | vs Ollama | Memory |\n|------|------|------|------|\n| CPU | ~15 tok/s | - | 1.1 GB |\n| GPU (batched) | ~811 tok/s | 3.4x | 1.9 GB |\n'
ds_case d2_bare_ratio.md  HIT  '| Engine | vs llama.cpp |\n|---|---|\n| apr | 0.82 |\n'
ds_case d2_receipted.md   RECEIPTED '| Engine | vs llama.cpp |\n|---|---|\n| apr | 0.82 (2026-04-06, `results/bootstrap.json`) |\n'
ds_case d2_no_comp.md     none '| Mode | Throughput | Memory |\n|------|------|------|\n| GPU (batched) | ~811 tok/s | 1.9 GB |\n'
ds_case d2_other_col.md   none '| Engine | Notes | Memory |\n|---|---|---|\n| apr | same GGUF as Ollama | 1.9 GB |\n'
ds_case d2_table_ends.md  none '| Engine | vs Ollama |\n|---|---|\n| apr | - |\n\n| Mode | Speed |\n|---|---|\n| GPU | 3.4x |\n'
ds_case d2_prose_after.md none '| vs Ollama | Mode |\n|---|---|\n| - | CPU |\n\n2.5x more memory is held by the batched mode.\n'
ds_case d_model_name.md   none '| Llama-3.2 1B Q4_K | 91 tok/s |\n'
ds_case d_no_figure.md    none 'apr runs the same GGUF as Ollama and llama.cpp.\n'
ds_case d_other_ratio.md  none 'Cold-start vs PyTorch: apr ~140× faster (ratio 0.007)\n'
ds_case d_own_tps.md      none 'apr decodes at 407 tok/s on this host.\n'
cut -f1 "$ctl/docs_want" > "$ctl/docs_list"
if ( cd "$ds" && docs_ban_scan "$ctl/docs_list" ) > "$ctl/docs_out"; then
    while IFS=$'\t' read -r f want; do
        got=none
        grep -qF "RECEIPTED"$'\t'"$f:" "$ctl/docs_out" && got=RECEIPTED
        grep -qF "HIT"$'\t'"$f:" "$ctl/docs_out" && got=HIT
        if [ "$got" != "$want" ]; then
            printf 'FAIL  docs want %-9s got %-9s : %s\n' "$want" "$got" "$f"
            tbl_bad=1
        fi
    done < "$ctl/docs_want"
else
    printf 'FAIL  docs table: the scanner itself failed, so every docs row is UNTESTED.\n'
    tbl_bad=1
fi
ds_rows=$(grep -c . "$ctl/docs_want")

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
#
# THE 65536 FLOOR BELOW IS A MARGIN, NOT THE THRESHOLD. Measured onset is ~10KB
# (curve above); the fixture emits ~165KB, roughly 16x past the point where loss
# is total. The floor is kept at the pipe-buffer size because it is the one
# number that cannot get smaller on a different libc, and being over-strict on a
# fixture costs nothing.
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
    printf 'ok    case table: %s shell must-match, %s must-not-match, %s cfg, %s rust,\n' \
        "$(( $(grep -c . "$ctl/must_match") + $(grep -c . "$ctl/must_match_shapes") ))" \
        "$(grep -c . "$ctl/must_not_match")" \
        "$(( $(grep -c . "$ctl/cfg_must_match") + $(grep -c . "$ctl/cfg_must_not_match") ))" \
        "$rs_rows"
    printf '               %s docs — all correct\n' "$ds_rows"
else
    rc=1
fi

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
#   DOCS, a figure split across lines — `(1.371× median,` on one line and
#     `412.3 vs 300.7 tok/s` on the next, with the competitor named on a third.
#     D1 is a line rule; BEATS.md's history paragraphs wrap exactly like that.
#     They are dated prose about a withdrawn claim, and a paragraph window
#     was rejected above because it lets one table row receipt its neighbour.
#
#   RUST, beyond R1-R3 — a fabrication assembled through a const table, a
#     builder or a match arm, or a generic-named literal in a fn whose name
#     does not name the competitor. See the ban block above.
#
printf '\n'
# A SELFTEST PASS IS NOT A CLEAN TREE, and it must not be printable as one. The
# two modes reach the same `exit 0`, so the only thing separating "the rules
# discriminate" from "the repository holds no fabricated baseline" is this line.
if [ "$rc" -ne 0 ]; then
    printf 'FAIL  see rows above (#2679, #2706 PERF-008).\n'
elif [ "$MODE" = selftest ]; then
    printf 'PASS  case table only. The rules discriminate; NO REPOSITORY SCAN was\n'
    printf '      run, so this says nothing about the tree. Run with no arguments.\n'
else
    printf 'PASS  no comparator baseline is asserted rather than measured.\n'
fi
exit "$rc"
