#!/usr/bin/env bash
# check_nightly_cut_table.sh -- planted rows for scripts/release/check_nightly_cut.sh (RQ-2, ruling A;
# HANDOFF.md §3 + §6 F5). Builds a throwaway repo (pre -> bump -> post) and a nightly root per row,
# runs the real script, and checks "<exit>|<first word of the verdict>". Then plants mutants that
# drop a binding or a STOP and requires a row to catch each, so the table is proven able to see them.
#
# EXIT 0 every row holds and every mutant is caught · 1 otherwise
#
#   bash scripts/release/check_nightly_cut_table.sh
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SUT="$ROOT/scripts/release/check_nightly_cut.sh"
TD="$(mktemp -d)"
trap 'rm -rf "${TD:?}"' EXIT
V=9.9.0
NOW=2000000000

G="$TD/repo"
gc() { git -C "$G" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q --allow-empty -m "$1"; }
git -C "$G" init -q 2> /dev/null || { mkdir -p "$G" && git -C "$G" init -q; }
gc pre;  PRE=$(git -C "$G" rev-parse HEAD)
gc bump; BUMP=$(git -C "$G" rev-parse HEAD)
gc post; POST=$(git -C "$G" rev-parse HEAD)
git -C "$G" checkout -q -b side "$PRE"
gc side; SIDE=$(git -C "$G" rev-parse HEAD)   # descends from pre, not from the bump

fail=0
row() { # row <name> <got> <want>
    if [ "$2" = "$3" ]; then echo "ok    $1"; else echo "FAIL  $1: got '$2', want '$3'"; fail=1; fi
}

# plant <dir> <sha> [version] [phase] [commit] [apr_sha] -- a GO set of receipts for <sha>
plant() {
    local d="$1" s="$2" ver="${3:-$V}" ph="${4:-pre-publish}" c="${5:-$2}" a="${6:-$2}" h
    mkdir -p "$d/$s/dogfood" "$d/$s/models-t1"
    printf '2026-01-01T00:00:00Z DEEP GO at %s (doctests, examples, all bins green)\n' "$s" > "$d/$s/STATUS"
    printf '{"commit":"%s","version":"%s","phase":"%s","verdict":"GO"}\n' "$c" "$ver" "$ph" > "$d/$s/dogfood/receipt.json"
    for h in lambda gx10; do
        printf '{"host":"%s","version":"%s","sha":"%s","apr_sha":"%s"}\n' "$h" "$ver" "$s" "$a" > "$d/$s/models-t1/$h.json"
    done
}
idx() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$2" "$3" "$4" "$5" "$6" "$7" >> "$1/index.tsv"; } # idx <dir> s v d f m t

# run <sut> <dir> -> "<exit>|<verdict word>"
run() {
    local rc=0 out
    out=$(bash "$1" --root "$2" --version "$V" --bump "$BUMP" --repo "$G" --now "$NOW" 2>&1) || rc=$?
    echo "$rc|$(tail -n 1 <<< "$out" | cut -d' ' -f1)"
}

fresh() { D=$(mktemp -d "$TD/r.XXXXXX"); }   # unique: table runs in a subshell per mutant
H=$((NOW - 6 * 3600))   # finished six hours ago

table() { # table <sut> <label>
    local s="$1" l="$2"
    fresh; plant "$D" "$BUMP"; idx "$D" "$BUMP" "$V" 0 0 0 "$H"
    row "$l 1 bump measured, all GO, receipts bound -> CUT" "$(run "$s" "$D")" "0|CUT"
    fresh; plant "$D" "$BUMP"; plant "$D" "$POST"; idx "$D" "$BUMP" "$V" 0 0 0 $((H - 3600)); idx "$D" "$POST" "$V" 0 0 0 "$H"
    row "$l 2 two green rows -> CUT the newest" "$(run "$s" "$D")" "0|CUT"
    row "$l 2 ... and it is the post-bump sha" "$(bash "$s" --root "$D" --version "$V" --bump "$BUMP" --repo "$G" --now "$NOW" | tail -n 1)" "CUT $POST"
    fresh; plant "$D" "$BUMP"; plant "$D" "$POST"; idx "$D" "$BUMP" "$V" 0 0 0 $((H - 3600)); idx "$D" "$POST" "$V" 0 0 1 "$H"
    row "$l 3 newest row models rc 1 -> STOP, no older green row" "$(run "$s" "$D")" "1|STOP"
    fresh; plant "$D" "$BUMP"; idx "$D" "$BUMP" "$V" 1 0 0 "$H"
    row "$l 3 deep rc 1 -> STOP" "$(run "$s" "$D")" "1|STOP"
    fresh; plant "$D" "$BUMP"; idx "$D" "$BUMP" "$V" 0 2 0 "$H"
    row "$l 4 dogfood rc 2 (decline) -> FALLBACK" "$(run "$s" "$D")" "2|FALLBACK"
    fresh; idx "$D" "$BUMP" "$V" 0 0 0 "$H"
    row "$l 5 index rc 0 but no receipts -> FALLBACK" "$(run "$s" "$D")" "2|FALLBACK"
    fresh; plant "$D" "$BUMP"; echo 'not json' > "$D/$BUMP/dogfood/receipt.json"; idx "$D" "$BUMP" "$V" 0 0 0 "$H"
    row "$l 5 dogfood receipt unreadable -> FALLBACK" "$(run "$s" "$D")" "2|FALLBACK"
    fresh; plant "$D" "$BUMP" "$V" pre-publish "$PRE"; idx "$D" "$BUMP" "$V" 0 0 0 "$H"
    row "$l 6 dogfood receipt commit != measured sha -> STOP" "$(run "$s" "$D")" "1|STOP"
    fresh; plant "$D" "$BUMP" "$V" pre-publish "$BUMP" "$PRE"; idx "$D" "$BUMP" "$V" 0 0 0 "$H"
    row "$l 6 models apr_sha != measured sha -> STOP" "$(run "$s" "$D")" "1|STOP"
    fresh; plant "$D" "$PRE" 9.8.0; idx "$D" "$PRE" 9.8.0 0 0 0 "$H"
    row "$l 7 only a pre-bump row at the old version -> FALLBACK" "$(run "$s" "$D")" "2|FALLBACK"
    fresh; plant "$D" "$SIDE"; idx "$D" "$SIDE" "$V" 0 0 0 "$H"
    row "$l 8 row at V not descended from the bump -> FALLBACK" "$(run "$s" "$D")" "2|FALLBACK"
    fresh; plant "$D" "$BUMP"; idx "$D" "$BUMP" "$V" 0 0 0 $((NOW - 31 * 3600))
    row "$l 9 row 31h old (bound 30h) -> FALLBACK" "$(run "$s" "$D")" "2|FALLBACK"
    fresh; plant "$D" "$BUMP"; rm -f "$D/$BUMP/models-t1/gx10.json"; idx "$D" "$BUMP" "$V" 0 0 0 "$H"
    row "$l 10 gx10 receipt missing -> FALLBACK (no partial reuse)" "$(run "$s" "$D")" "2|FALLBACK"
    fresh; rmdir "$D"
    row "$l 11 nightly root absent -> FALLBACK" "$(run "$s" "$D")" "2|FALLBACK"
    fresh
    row "$l 12 root present, nightly never wrote an index -> FALLBACK" "$(run "$s" "$D")" "2|FALLBACK"
    fresh; plant "$D" "$BUMP" 9.8.0; idx "$D" "$BUMP" "$V" 0 0 0 "$H"
    row "$l 13 receipt version != V under a row at V -> STOP" "$(run "$s" "$D")" "1|STOP"
    fresh; plant "$D" "$BUMP" "$V" post-publish; idx "$D" "$BUMP" "$V" 0 0 0 "$H"
    row "$l 13 dogfood phase != pre-publish -> STOP" "$(run "$s" "$D")" "1|STOP"
    fresh; plant "$D" "$BUMP"; idx "$D" "$BUMP" "$V" 0 0 75 "$H"
    row "$l 15 models rc 75 (apr-gpu.lock busy) -> FALLBACK" "$(run "$s" "$D")" "2|FALLBACK"
    fresh; plant "$D" "$BUMP"; : > "$D/$BUMP/STATUS"; idx "$D" "$BUMP" "$V" 0 0 0 "$H"
    row "$l 17 deep rc 0 but no DEEP GO line -> FALLBACK" "$(run "$s" "$D")" "2|FALLBACK"
    fresh; plant "$D" "$BUMP"; echo "x DEEP GO at $PRE (y)" > "$D/$BUMP/STATUS"; idx "$D" "$BUMP" "$V" 0 0 0 "$H"
    row "$l 17 DEEP GO line for another sha -> FALLBACK" "$(run "$s" "$D")" "2|FALLBACK"
    fresh; plant "$D" "$BUMP"; idx "$D" "${BUMP:0:9}" "$V" 0 0 0 "$H"
    row "$l   short sha in the index is not a row -> FALLBACK" "$(run "$s" "$D")" "2|FALLBACK"
}

table "$SUT" real

# mutants: each must turn at least one row of the real table
mutant() { # mutant <label> <sed expr>
    local m="$TD/mutant.sh" out
    sed "$2" "$SUT" > "$m"
    if cmp -s "$SUT" "$m"; then echo "FAIL  mutant $1: the sed did not change the script (pattern drifted)"; fail=1; return; fi
    out=$(fail=0; table "$m" "  [$1]" 2>&1 | grep -c '^FAIL' || true)
    if [ "$out" -gt 0 ]; then echo "ok    mutant $1 caught by $out row(s)"; else echo "FAIL  mutant $1 survived every row"; fail=1; fi
}
mutant no-commit-binding  's/^\[ "\$c" = "\$S" \] || stop/true || stop/'
mutant no-apr-sha-binding 's/^    \[ "\$a" = "\$S" \] || stop/    true || stop/'
mutant no-red-stop        's/\[ "\${p#\*:}" = 1 \] && stop/false \&\& stop/'
mutant no-ancestry        's/merge-base --is-ancestor "\$BUMP" "\$s" 2>\/dev\/null || continue/true/'
mutant no-age-bound       's/^\[ "\$age" -le/[ 0 -le/'
mutant oldest-row-wins    's/"\$t" -gt "\$best_t"/"$best_t" -lt 0/'
mutant no-deep-evidence   's/^grep -qF " DEEP GO at \$S " "\$ROOT\/\$S\/STATUS" 2> \/dev\/null || fallback/true || fallback/'

exit "$fail"
