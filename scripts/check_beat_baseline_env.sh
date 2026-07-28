#!/usr/bin/env bash
# Preflight: the incumbent baseline (scikit-learn + numpy under `uv`) must
# actually import before we time anything against it.
#
# WHY THIS EXISTS. Every Pillar-1/2/3 speed beat is a RELATIVE ratio: we time
# apr and we time the incumbent on the same host in the same run. If the
# incumbent cannot even be imported, the beat does not measure a regression -
# it measures nothing, and it does so while printing a red X that looks exactly
# like a lost beat. On 2026-07-27/28 the whole nightly lane died this way:
#
#   sklearn timing failed: ModuleNotFoundError: No module named 'numpy'
#
# ROOT CAUSE (diagnosed on intel-clean-room-10 / mac-server, uv 0.11.23). uv
# caches the resolved `--with` environment as a venv under
# `~/.cache/uv/archive-v0/<hash>/`. One of those archives had lost its
# `pyvenv.cfg` (its `bin/` was Jul 26, its `lib/` Jul 2 - a torn write or a
# disk-pressure sweep; this host has a history of both). uv still counted the
# requirements as satisfied - so it printed no "Installed N packages" and did
# no install - but with no `pyvenv.cfg` it could not use the archive as the
# base for the ephemeral env, and silently fell back to `/usr/bin/python3`.
# The archive's site-packages therefore never joined `sys.path`:
#
#   healthy:  .../builds-v0/<tmp>/lib/python3.10/site-packages
#             .../archive-v0/<hash>/lib/python3.10/site-packages   <-- the --with env
#             /usr/lib/python3/dist-packages
#   poisoned: .../builds-v0/<tmp>/lib/python3.10/site-packages
#             /usr/lib/python3/dist-packages                       <-- overlay GONE
#
# So python ran, found no numpy, and exited in under a second. Neither
# `uv cache clean <pkg>` nor `uv run --refresh` repairs it: both operate on
# wheel/package caches, and the corruption is in the cached *environment*.
# `rm -rf`-ing that one archive fixed it immediately (verified: "Installed 5
# packages in 33ms" -> numpy 2.2.6 / sklearn 1.7.2).
#
# WHAT THIS DOES. Behaviour first, layout second - we do not want a gate that
# encodes a guess about uv's cache internals:
#   1. Try the import. Pass -> done, no mutation.
#   2. Fail -> remove only archives with the exact poison signature (venv-shaped
#      - has bin/python3 - but missing pyvenv.cfg). Healthy archives always
#      carry pyvenv.cfg + CACHEDIR.TAG, so this is precise, not a blanket wipe.
#   3. Retry. Still failing -> full `uv cache clean` (last resort, self-healing
#      at the cost of one re-download).
#   4. Retry. Still failing -> exit 1 with the diagnosis, because at that point
#      it is not a poisoned cache and a human needs the sys.path dump.
#
# Same shape as the EACCES self-heal added to guard-runner-labels in #2311:
# detect a known-recurring runner-state fault, repair it, prove the repair.
set -euo pipefail

PROBE='import numpy, sklearn; print("BASELINE_OK numpy=%s sklearn=%s" % (numpy.__version__, sklearn.__version__))'

if ! command -v uv >/dev/null 2>&1; then
    echo "✗ check_beat_baseline_env: uv not on PATH - the speed beats cannot time the incumbent" >&2
    exit 1
fi

echo "check_beat_baseline_env: $(uv --version)"

try_probe() {
    uv run --with scikit-learn --with numpy python3 -c "$PROBE" 2>&1
}

# Step 1 - ALWAYS sweep poisoned env archives, before probing.
#
# This is deliberately proactive rather than reactive. Measured on mac-server
# 2026-07-28: 16 of 19 cached environments were missing pyvenv.cfg. The beats
# use several different incumbents (scikit-learn, torch, transformers, unsloth),
# each with its OWN cached environment, so a probe that only exercises the
# scikit-learn env would pass while the torch env stayed broken - and we would
# rediscover this one incumbent at a time. A poisoned archive is definitively
# unusable (uv cannot use it as a venv base), so removing it is never a loss:
# the worst case is one re-download.
CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
ARCHIVES="$CACHE_DIR/archive-v0"
removed=0
if [ -d "$ARCHIVES" ]; then
    # find, not `ls` - archive names are content hashes, so iterate directories
    # directly. A cached *environment* is venv-shaped (bin/python3 exists); a
    # cached *package* is just an unpacked wheel and is skipped by that test.
    while IFS= read -r archive; do
        [ -n "$archive" ] || continue
        # A cached *environment* is venv-shaped and healthy ones always carry
        # pyvenv.cfg (plus CACHEDIR.TAG); a cached *package* is just an unpacked
        # wheel with no bin/python3 and is skipped. So "has bin/python3 but no
        # pyvenv.cfg" is the precise poison signature, verified against both a
        # healthy and the poisoned archive on mac-server.
        if [ -e "$archive/bin/python3" ] && [ ! -f "$archive/pyvenv.cfg" ]; then
            # Belt and braces before an rm -rf: the path must be non-empty, must
            # not be the filesystem root, and must still live strictly UNDER the
            # cache root. An empty or unanchored $archive here is catastrophic,
            # so validate immediately before the destructive call, not earlier.
            unsafe=0
            [ -z "$archive" ] && unsafe=1
            [ "$archive" = "/" ] && unsafe=1
            case "$archive" in
                "$ARCHIVES"/?*) : ;;
                *) unsafe=1 ;;
            esac
            if [ "$unsafe" -ne 0 ] || [ -z "$archive" ]; then
                echo "  refusing to remove unsafe/unanchored path: '$archive'" >&2
            elif [ -n "$archive" ] && [ "$archive" != "/" ]; then
                echo "  removing poisoned env archive (venv-shaped, no pyvenv.cfg): $archive" >&2
                rm -rf "${archive:?refusing to rm an empty path}"
                removed=$((removed + 1))
            fi
        fi
    done < <(find "$ARCHIVES" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)
fi
echo "swept $removed poisoned env archive(s) from $ARCHIVES"

# Step 2 - prove the incumbent actually imports.
if out=$(try_probe) && [[ "$out" == *BASELINE_OK* ]]; then
    echo "  $out"
    echo "✓ check_beat_baseline_env: baseline importable (swept $removed poisoned archive(s))"
    exit 0
fi

echo "⚠ baseline import FAILED even after sweeping $removed poisoned archive(s)" >&2
echo "  probe output: ${out}" >&2

# Step 3 - last resort: nuke the whole uv cache. Costs one re-download.
echo "⚠ falling back to full 'uv cache clean'" >&2
uv cache clean >&2 || true

if out=$(try_probe) && [[ "$out" == *BASELINE_OK* ]]; then
    echo "✓ check_beat_baseline_env: baseline restored by full cache clean"
    exit 0
fi

# Not a cache fault. Give the operator everything needed to diagnose.
echo "::error::check_beat_baseline_env: scikit-learn/numpy baseline is NOT importable after cache repair" >&2
echo "  Every speed beat is a ratio against this baseline; without it the lane measures nothing." >&2
echo "  final probe output: ${out}" >&2
echo "  --- diagnostics ---" >&2
uv run --with scikit-learn --with numpy python3 \
    -c 'import sys; print("executable:", sys.executable); [print("  path:", p) for p in sys.path]' >&2 2>&1 || true
exit 1
