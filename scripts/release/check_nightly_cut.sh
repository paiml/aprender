#!/usr/bin/env bash
# check_nightly_cut.sh -- RQ-2 (ruling A, aprender-a7 2026-10-04): may release <V> cut AT the commit
# last night's release-ladders nightly measured, and skip running deep / dogfood / models today?
#
# The nightly (scripts/release/release_ladders_nightly.sh) leaves, per measured sha S:
#   <root>/index.tsv           one row per run: <S>\t<version>\t<deep rc>\t<dogfood rc>\t<models rc>\t<finished epoch>
#   <root>/<S>/dogfood/receipt.json      the dogfood receipt whose `commit` is S
#   <root>/<S>/models-t1/<host>.json     models_t1.sh's per-host receipts (lambda, gx10)
#
# This script picks the cut; it does NOT judge the receipts' contents. Release day re-runs the SAME
# judges on them (R5 --receipt-only, check_model_ladder.sh, readiness), so nothing here can admit
# what those refuse. It only refuses to point them at the wrong commit. Case table:
# scripts/release/check_nightly_cut_table.sh (HANDOFF.md §3 + §6 F5).
#
#   check_nightly_cut.sh --root DIR --version V --bump SHA [--repo DIR] [--now EPOCH] [--max-age-h H]
#
# stdout, last line: CUT <sha> | FALLBACK <reason> | STOP <reason>
# EXIT 0 CUT · 1 STOP (no tag, a finding) · 2 FALLBACK (run the steps today, ruling B) · 64 usage
set -uo pipefail

ROOT="" V="" BUMP="" REPO="." NOW="" MAXH=30
while [ $# -gt 0 ]; do
    case "$1" in
        --root) ROOT="${2:-}"; shift 2 ;;
        --version) V="${2:-}"; shift 2 ;;
        --bump) BUMP="${2:-}"; shift 2 ;;
        --repo) REPO="${2:-}"; shift 2 ;;
        --now) NOW="${2:-}"; shift 2 ;;
        --max-age-h) MAXH="${2:-}"; shift 2 ;;
        *) echo "usage: $0 --root DIR --version V --bump SHA [--repo DIR] [--now EPOCH] [--max-age-h H]" >&2; exit 64 ;;
    esac
done
[ -n "$ROOT" ] && [ -n "$V" ] && [ -n "$BUMP" ] || { echo "usage: --root, --version and --bump are required" >&2; exit 64; }
[ -n "$NOW" ] || NOW=$(date -u +%s)

stop() { echo "STOP $*"; exit 1; }
fallback() { echo "FALLBACK $*"; exit 2; }

# jf FILE KEY -> the top-level string KEY, or empty (missing, unreadable, not an object)
jf() { jq -r --arg k "$2" 'if type == "object" and (.[$k] | type) == "string" then .[$k] else empty end' -- "$1" 2>/dev/null; }

# rows 11, 12: no index means not measured, never a pass
[ -f "$ROOT/index.tsv" ] || fallback "no nightly index at $ROOT/index.tsv (row 11/12: nightly absent or not run)"

# rows 7, 8: only rows at V whose sha is the bump or descends from it are this release's code.
# Newest by finish time (row 2); an index row is a pointer, the receipts below are the evidence.
best="" best_t=-1
while IFS=$'\t' read -r s ver drc frc mrc t; do
    [ "$ver" = "$V" ] || continue
    case "$s" in *[!0-9a-f]* | '') continue ;; esac
    [ "${#s}" = 40 ] || continue
    case "$t" in *[!0-9]* | '') continue ;; esac
    git -C "$REPO" merge-base --is-ancestor "$BUMP" "$s" 2>/dev/null || continue
    if [ "$t" -gt "$best_t" ]; then best="$s"$'\t'"$drc"$'\t'"$frc"$'\t'"$mrc"; best_t="$t"; fi
done < "$ROOT/index.tsv"
[ -n "$best" ] || fallback "no nightly row at $V on or after the bump ${BUMP:0:9} (rows 7/8)"
IFS=$'\t' read -r S DRC FRC MRC <<< "$best"

# row 9: a stale row is a stale host/toolchain state
age=$(( NOW - best_t ))
[ "$age" -le $(( MAXH * 3600 )) ] || fallback "newest row ${S:0:9} is $(( age / 3600 ))h old, over ${MAXH}h (row 9)"

# row 3: a red step at V is a finding -- STOP, and no older green row is looked for
for p in "deep:$DRC" "dogfood:$FRC" "models:$MRC"; do
    [ "${p#*:}" = 1 ] && stop "nightly ${p%%:*} RED (rc 1) at ${S:0:9} for $V (row 3)"
done
# rows 4, 15: a decline (2), a lock-busy 75, anything not 0 is not measured
for p in "deep:$DRC" "dogfood:$FRC" "models:$MRC"; do
    [ "${p#*:}" = 0 ] || fallback "nightly ${p%%:*} rc ${p#*:} at ${S:0:9}: not measured (rows 4/15)"
done

# row 17: deep leaves no receipt, so its evidence is its own GO line in the run's STATUS, at S (quorum r1 F1).
# An index rc 0 without it is a bare pointer: not measured.
grep -qF " DEEP GO at $S " "$ROOT/$S/STATUS" 2> /dev/null || fallback "no 'DEEP GO at ${S:0:9}' line in $ROOT/$S/STATUS (row 17)"

# rows 5, 6, 13: the dogfood receipt exists, is bound to S, to V and to the pre-publish phase
r="$ROOT/$S/dogfood/receipt.json"
c=$(jf "$r" commit)
[ -n "$c" ] || fallback "dogfood receipt missing or unreadable: $r (row 5)"
[ "$c" = "$S" ] || stop "dogfood receipt commit ${c:0:9} != measured sha ${S:0:9} (row 6)"
[ "$(jf "$r" version)" = "$V" ] || stop "dogfood receipt version '$(jf "$r" version)' != $V (row 13)"
[ "$(jf "$r" phase)" = pre-publish ] || stop "dogfood receipt phase '$(jf "$r" phase)' != pre-publish (row 13)"

# rows 5, 6, 10, 13: BOTH hosts' model receipts, each bound to S and V. One missing = full fallback.
for h in lambda gx10; do   # models_t1.sh LOCAL_HOST + REMOTE_HOST
    m="$ROOT/$S/models-t1/$h.json"
    a=$(jf "$m" apr_sha)
    [ -n "$a" ] || fallback "models receipt missing or unreadable: $m (rows 5/10: no partial reuse)"
    [ "$a" = "$S" ] || stop "models $h apr_sha ${a:0:9} != measured sha ${S:0:9} (row 6)"
    [ "$(jf "$m" version)" = "$V" ] || stop "models $h version '$(jf "$m" version)' != $V (row 13)"
done

# rows 1, 2
echo "CUT $S"
exit 0
