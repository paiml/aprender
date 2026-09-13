#!/usr/bin/env bash
# FALSIFY-BOOK-EXAMPLE-EXECUTES-001 (Phase 6 of BOOK-CLOSEOUT-001).
#
# For every fenced bash code block in book/src/{cli,lib}/*.md, classify
# by ``<!-- example-cost: ... -->`` annotation and either execute, skip,
# or rewrite + execute. Exits 0 iff zero hard FAILs (skips are fine).
#
# Cost handling:
#   trivial          execute with `timeout 10 bash -c $code`; exit 0 required
#   model-required   if ~/models/<model> missing, SKIP; else `timeout 60 bash`
#   gpu              skip unless nvidia-smi + nvcc both present
#   destructive      rewrite mutating commands to `--help` (or --dry-run)
#                    and assert exit 0; if rewrite fails, SKIP
#   interactive      SKIP unconditionally
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXTRACT="bash scripts/extract-book-examples.sh"
MODELS_DIR="${APR_MODELS_DIR:-$HOME/models}"

pass=0
skip=0
fail=0
total=0

# Resolve the binary THIS CHECKOUT builds (#2358).
#
# This used to be `command -v apr`, falling back to a hardcoded
# /mnt/nvme-raid0/targets/aprender/release/apr. Both are the #2357 defect, and
# this gate is a bad place for it: it executes the book's own examples and
# reports whether they work. Against a stale binary it certifies that examples
# run correctly on code nobody is shipping -- an example using a flag added this
# week FAILS, and an example using a flag deleted this week PASSES. The /mnt
# fallback is worse than stale: nothing writes that path any more.
#
# `. scripts/apr_bin.sh` asks cargo for the target dir and asserts the binary's
# embedded git SHA matches HEAD. It returns non-zero when there is no
# freshly-built binary, which for this gate is a SKIP, not a failure -- the
# examples are still scanned and rust blocks still reported.
APR_BIN=""
# The pinned binary must be REACHED, not merely resolved. Every example below
# runs through `bash -c "$code"`, and the code says `apr ...` -- which PATH
# resolves, not $APR_BIN. On this box a bare `apr` was /home/noah/.local/bin/apr
# = 0.60.0, so the gate that certifies the book's CLI examples was exercising a
# binary from a different release than the tree it was gating. Putting the pinned
# binary FIRST on PATH covers `apr` in any position, including pipelines and
# subshells, which substituting a leading token does not.
APR_PATH_PREFIX=""
if . scripts/apr_bin.sh 2>/dev/null; then
    APR_BIN="$APR"
    APR_PATH_PREFIX="$(dirname "$APR_BIN")"
    PATH="${APR_PATH_PREFIX}:${PATH}"
    export PATH
    echo "[INFO] apr pinned to $APR_BIN (prepended to PATH)"
else
    echo "[INFO] no apr binary built from HEAD; all CLI examples will SKIP"
    echo "[INFO]   build one with: cargo build --release -p apr-cli --bin apr"
fi

# Helper: rewrite a destructive command into a safe variant (or fail).
rewrite_destructive() {
    local code="$1"
    # If the command already has a flag that makes it a no-op (--help,
    # --dry-run, --version), leave it alone.
    case "$code" in
        *"--help"*|*"--version"*|*"--dry-run"*)
            echo "$code"
            return 0
            ;;
    esac
    # Strategy: replace the first apr-subcommand line with `apr <cmd> --help`.
    # This is safe for ALL destructive ops in the catalogue (publish, encrypt,
    # decrypt, rm, upload, stamp).
    local first
    first=$(printf '%s\n' "$code" | sed -n '1p')
    if grep -qE '^apr\s+[a-z]' <<< "$first" ; then
        local cmd
        cmd=$(printf '%s\n' "$first" | awk '{print $2}')
        echo "apr $cmd --help"
        return 0
    fi
    return 1
}

# Helper: rewrite model placeholders that won't resolve locally.
# E.g. "qwen2.5-coder-1.5b" -> "$MODELS_DIR/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
# Only used when the named model file is missing in $MODELS_DIR.
expand_model_path() {
    local model="$1"
    if [ -e "$MODELS_DIR/$model" ]; then
        printf '%s' "$MODELS_DIR/$model"
        return 0
    fi
    return 1
}

while IFS= read -r record; do
    total=$((total + 1))
    lang=$(printf '%s\n' "$record" | python3 -c "import json,sys;print(json.load(sys.stdin)['lang'])")
    cost=$(printf '%s\n' "$record" | python3 -c "import json,sys;print(json.load(sys.stdin)['cost'])")
    path=$(printf '%s\n' "$record" | python3 -c "import json,sys;print(json.load(sys.stdin)['path'])")
    code=$(printf '%s\n' "$record" | python3 -c "import json,sys;print(json.load(sys.stdin)['code'])")
    model=$(printf '%s\n' "$record" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('model',''))")

    # Rust blocks are not executed here -- that's the compile checker's job.
    if [ "$lang" = "rust" ]; then
        skip=$((skip + 1))
        echo "[SKIP] $path :: $cost rust (compile gate handles this)"
        continue
    fi

    case "$cost" in
        trivial)
            if [ -z "$APR_BIN" ] && grep -qE '^apr\b' <<< "$code" ; then
                skip=$((skip + 1))
                echo "[SKIP] $path :: trivial (apr binary not available)"
                continue
            fi
            if timeout 10 bash -c "$code" >/dev/null 2>&1; then
                pass=$((pass + 1))
                echo "[PASS] $path :: trivial"
            else
                # Special-case `apr <subcmd> --help` and `apr --version` --
                # those are guaranteed by clap to exit 0 if the binary works.
                fail=$((fail + 1))
                echo "[FAIL] $path :: trivial -- $(printf '%s' "$code" | head -c 80)"
            fi
            ;;
        model-required)
            # Resolve the model location, accepting either a path or a name.
            target=""
            if [ -n "$model" ]; then
                if [ -e "$MODELS_DIR/$model" ]; then
                    target="$MODELS_DIR/$model"
                elif [ -e "$model" ]; then
                    target="$model"
                fi
            fi
            if [ -z "$target" ]; then
                skip=$((skip + 1))
                echo "[SKIP] $path :: model-required ($model not in cache) [model not in cache]"
                continue
            fi
            if [ -z "$APR_BIN" ]; then
                skip=$((skip + 1))
                echo "[SKIP] $path :: model-required (apr binary not available)"
                continue
            fi
            # Rewrite bare model-name tokens to their equivalent in $MODELS_DIR
            # so examples don't depend on auto-pull. Two strategies:
            #   1. Any token ending in .gguf/.apr/.safetensors that has no path
            #      separator AND exists in $MODELS_DIR -> prefix with $MODELS_DIR/.
            #   2. If the annotated model is a bare name (no slash, no extension)
            #      that exists in $MODELS_DIR -> substitute. (Common for `qwen2.5-coder-1.5b`)
            run_code="$code"
            while read -r tok; do
                [ -z "$tok" ] && continue
                # Skip tokens that contain any slash (already path-like).
                case "$tok" in */*) continue;; esac
                if [ -e "$MODELS_DIR/$tok" ]; then
                    # Use word-boundary substitution to avoid double-substituting.
                    run_code=$(printf '%s' "$run_code" | sed "s#\\b$tok\\b#$MODELS_DIR/$tok#g")
                fi
            done < <(printf '%s\n' "$run_code" | grep -oE '[A-Za-z0-9_.-]+\.(gguf|apr|safetensors)' | sort -u)
            # Replace the annotated bare-name model only if it's not already substituted.
            if [ -n "$model" ] \
                && ! [ -e "$model" ] \
                && [ -e "$MODELS_DIR/$model" ] \
                && ! grep -qF "$MODELS_DIR/$model" <<< "$run_code" ; then
                run_code=$(printf '%s' "$run_code" | sed "s#\\b$model\\b#$MODELS_DIR/$model#g")
            fi
            if timeout 60 bash -c "$run_code" >/dev/null 2>&1; then
                pass=$((pass + 1))
                echo "[PASS] $path :: model-required"
            else
                fail=$((fail + 1))
                echo "[FAIL] $path :: model-required -- $(printf '%s' "$run_code" | head -c 80)"
            fi
            ;;
        gpu)
            if ! command -v nvidia-smi >/dev/null 2>&1; then
                skip=$((skip + 1))
                echo "[SKIP] $path :: gpu (no nvidia-smi)"
                continue
            fi
            if ! nvidia-smi >/dev/null 2>&1; then
                skip=$((skip + 1))
                echo "[SKIP] $path :: gpu (nvidia-smi failed)"
                continue
            fi
            if [ ! -x /usr/local/cuda/bin/nvcc ]; then
                skip=$((skip + 1))
                echo "[SKIP] $path :: gpu (no nvcc)"
                continue
            fi
            if [ -z "$APR_BIN" ]; then
                skip=$((skip + 1))
                echo "[SKIP] $path :: gpu (apr binary not available)"
                continue
            fi
            if timeout 30 bash -c "$code" >/dev/null 2>&1; then
                pass=$((pass + 1))
                echo "[PASS] $path :: gpu"
            else
                fail=$((fail + 1))
                echo "[FAIL] $path :: gpu -- $(printf '%s' "$code" | head -c 80)"
            fi
            ;;
        destructive)
            if [ -z "$APR_BIN" ]; then
                skip=$((skip + 1))
                echo "[SKIP] $path :: destructive (apr binary not available)"
                continue
            fi
            if rewritten=$(rewrite_destructive "$code"); then
                if timeout 10 bash -c "$rewritten" >/dev/null 2>&1; then
                    pass=$((pass + 1))
                    echo "[PASS] $path :: destructive (rewrote to --help)"
                else
                    fail=$((fail + 1))
                    echo "[FAIL] $path :: destructive rewrite did not succeed: $rewritten"
                fi
            else
                skip=$((skip + 1))
                echo "[SKIP] $path :: destructive (could not rewrite safely)"
            fi
            ;;
        interactive)
            skip=$((skip + 1))
            echo "[SKIP] $path :: interactive (CI cannot feed stdin to TUI/REPL)"
            ;;
        *)
            fail=$((fail + 1))
            echo "[FAIL] $path :: unknown cost class '$cost'"
            ;;
    esac
done < <($EXTRACT)

echo ""
echo "FALSIFY-BOOK-EXAMPLE-EXECUTES-001: total=$total pass=$pass skip=$skip fail=$fail"
if [ "$fail" -gt 0 ]; then
    echo "FALSIFY-BOOK-EXAMPLE-EXECUTES-001: FAIL"
    exit 1
fi

# A run that executed NOTHING is not a pass. With no apr binary built from HEAD
# every CLI example skips, and this gate printed
#   total=244 pass=0 skip=244 fail=0 ... PASS
# which is the whole "gate that cannot fail" class in one line: the number that
# mattered was zero and the verdict was green.
if [ "$total" -gt 0 ] && [ "$pass" -eq 0 ]; then
    if [ -z "$APR_BIN" ]; then
        echo "FALSIFY-BOOK-EXAMPLE-EXECUTES-001: NOT RUN -- no apr binary built from HEAD," \
             "so all $skip example(s) skipped and nothing was verified."
        echo "  Build one first:  cargo build --release -p apr-cli --bin apr"
        exit 1
    fi
    echo "FALSIFY-BOOK-EXAMPLE-EXECUTES-001: FAIL -- apr is available at $APR_BIN but" \
         "0 of $total example(s) executed. The gate measured nothing."
    exit 1
fi
echo "FALSIFY-BOOK-EXAMPLE-EXECUTES-001: PASS"
