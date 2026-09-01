#!/usr/bin/env bash
# dogfood-use.sh — USE the release binary on real models, and fail if it misbehaves.
#
# Contract (from the dogfood pre-release skill):
#   env BIN=<built release binary> WORK=<scratch dir> bash scripts/dogfood-use.sh
#   exit 0 = the tool did its job on real data; non-zero = it did not.
#
# This is deliberately NOT a test re-run. `cargo test` already ran in an earlier
# gate. This step exists because a released tool nobody actually executed is not
# dogfooded — v0.63.0 exists precisely because a nightly "validated" 24-day-old
# code while reporting green, and a release smoke-test read a five-hour-old
# binary and reported a meaningless pass.
#
# So the first thing this asserts is WHICH BINARY IT RAN.
#
# Models are looked up in $MODELS_DIR (default ~/models). Each block SKIPs if its
# model is absent, so this runs on a machine without the 30GB registry — but it
# FAILS at the end if nothing was exercised at all, because "no models present"
# must never read as "dogfooded".

set -uo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
WORK="${WORK:-$(mktemp -d)}"
PASS=0
SKIP=0
FAILED=0

ok()   { PASS=$((PASS+1)); printf '  \033[32mPASS\033[0m %s\n' "$*"; }
skip() { SKIP=$((SKIP+1)); printf '  \033[33mSKIP\033[0m %s\n' "$*"; }
bad()  { FAILED=$((FAILED+1)); printf '  \033[31mFAIL\033[0m %s\n' "$*"; }

# ── 0. WHICH BINARY. Resolve and prove provenance before trusting any result ──
if [ -z "${BIN:-}" ] || [ ! -x "${BIN:-}" ]; then
    # The skill's own extraction looks for an executable named after the CRATE
    # (`aprender`), but this crate's binary is `apr` — so BIN can arrive empty.
    # Fall back to the repo's resolver, which asks cargo and proves SHA == HEAD.
    # shellcheck source=scripts/apr_bin.sh
    if . "$(dirname "$0")/apr_bin.sh"; then
        BIN="$APR"
    else
        printf 'FATAL: no usable apr binary (BIN unset and apr_bin.sh could not resolve one)\n' >&2
        exit 1
    fi
fi

VERSION_LINE=$("$BIN" --version 2>&1)
printf 'dogfood-use: %s\n  %s\n' "$BIN" "$VERSION_LINE"

# Provenance: the binary must carry the commit under test. A dogfood run against
# a binary built from something else proves nothing about this release.
if HEAD_SHA=$(git rev-parse --short HEAD 2>/dev/null); then
    case "$VERSION_LINE" in
        *"$HEAD_SHA"*) ok "binary reports HEAD ($HEAD_SHA)" ;;
        *) bad "binary does NOT report HEAD ($HEAD_SHA) — refusing to dogfood a foreign build: $VERSION_LINE" ;;
    esac
fi

WS_VERSION=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
case "$VERSION_LINE" in
    *"$WS_VERSION"*) ok "binary reports workspace version $WS_VERSION" ;;
    *) bad "binary reports '$VERSION_LINE', workspace is $WS_VERSION" ;;
esac

# ── 1. inspect: read a real GGUF's metadata ──────────────────────────────────
GGUF="$MODELS_DIR/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
if [ -f "$GGUF" ]; then
    OUT=$(timeout 120 "$BIN" inspect --json "$GGUF" 2>/dev/null); EC=$?
    ARCH=$(printf '%s' "$OUT" | jq -r '.architecture // empty' 2>/dev/null)
    if [ "$EC" -eq 0 ] && [ "$ARCH" = "qwen2" ]; then
        ok "inspect GGUF -> arch=$ARCH"
    else
        bad "inspect GGUF: exit=$EC arch='$ARCH' (expected 0 / qwen2)"
    fi

    OUT=$(timeout 120 "$BIN" tensors --json "$GGUF" 2>/dev/null); EC=$?
    N=$(printf '%s' "$OUT" | jq -r '.tensor_count // (.|length) // 0' 2>/dev/null)
    if [ "$EC" -eq 0 ] && [ "${N:-0}" -gt 100 ]; then
        ok "tensors GGUF -> $N tensors"
    else
        bad "tensors GGUF: exit=$EC n=$N (expected 0 / >100)"
    fi
else
    skip "GGUF absent at $GGUF"
fi

# ── 2. run: actually generate tokens and check the answer is right ───────────
# A factual prompt with a wide argmax margin — per #2359, a bare greeting sits on
# a near-tie and flips between backends, which would make this gate a coin flip.
APR_MODEL="$MODELS_DIR/qwen2.5-coder-1.5b-instruct-q4k.apr"
if [ -f "$APR_MODEL" ]; then
    OUT=$(timeout 300 "$BIN" run "$APR_MODEL" --prompt "What is the capital of France?" \
            --max-tokens 24 2>/dev/null); EC=$?
    if [ "$EC" -eq 0 ] && grep -qi "paris" <<< "$OUT" ; then
        ok "run APR -> answered 'Paris'"
    else
        bad "run APR: exit=$EC, no 'Paris' in output: $(printf '%s' "$OUT" | tail -2 | tr '\n' ' ')"
    fi

    # qa is the product's own falsifiable gate suite — the strongest single
    # assertion available that the binary works end to end on a real model.
    OUT=$(timeout 600 "$BIN" qa "$APR_MODEL" 2>&1); EC=$?
    if grep -q "ALL GATES PASSED" <<< "$OUT" ; then
        ok "qa APR -> ALL GATES PASSED"
    else
        bad "qa APR: exit=$EC, no 'ALL GATES PASSED' — $(printf '%s' "$OUT" | grep -E '✗|FAIL' | head -2 | tr '\n' ' ')"
    fi
else
    skip "APR model absent at $APR_MODEL"
fi

# ── 3. round-trip: export and read back what we wrote ────────────────────────
if [ -f "$APR_MODEL" ]; then
    OUT_GGUF="$WORK/roundtrip.gguf"
    timeout 300 "$BIN" export "$APR_MODEL" --format gguf -o "$OUT_GGUF" >/dev/null 2>&1; EC=$?
    if [ "$EC" -eq 0 ] && [ -s "$OUT_GGUF" ]; then
        A=$(timeout 120 "$BIN" inspect --json "$OUT_GGUF" 2>/dev/null | jq -r '.architecture // empty' 2>/dev/null)
        if [ -n "$A" ]; then
            ok "export -> gguf, and the result reads back (arch=$A)"
        else
            bad "export produced a file apr cannot read back"
        fi
    elif [ "$EC" -eq 5 ]; then
        # Documented clean validation error, not a crash (see qwen-story B4).
        ok "export declined cleanly (exit 5, no panic)"
    else
        bad "export: exit=$EC"
    fi
fi

# ── verdict ──────────────────────────────────────────────────────────────────
printf '\n  %s pass / %s fail / %s skip\n' "$PASS" "$FAILED" "$SKIP"

# Nothing exercised is NOT a pass. Only the provenance checks run without models,
# and a binary that merely reports its own version has not been dogfooded.
if [ "$PASS" -le 2 ] && [ "$FAILED" -eq 0 ]; then
    printf 'FAIL: no model was exercised (MODELS_DIR=%s) — provenance alone is not dogfooding.\n' \
        "$MODELS_DIR" >&2
    exit 1
fi

[ "$FAILED" -eq 0 ] || exit 1
exit 0
