#!/usr/bin/env bash
# check_bump_carries_cert.sh -- a version bump must land WITH its CRUX prompt certification (PMAT-7029)
#
#   bash scripts/release/check_bump_carries_cert.sh [--root DIR]   # the tree's workspace version V
#   bash scripts/release/check_bump_carries_cert.sh --self-test    # case table + mutants
#
# WHY. models_t1.sh (crux mode) answers "MODELS NO-GO: no prompt certification for V" when
# evidence/crux/V/prompt-certification.json is absent at the measured commit. The nightly relay then reads
# not_measured on EVERY commit after the bump, and promote_scheduler.sh (the nightly-green -> tag path) has no
# green to promote until a certification PR lands. prepare_bump.sh carries it (pb_carry_cert), but a bump cut by
# hand with bump-version.sh skipped it: 0.72.4, 0.74.0. This check runs in PR CI, so such a bump is red in its own PR.
# Reads only files; compiles nothing. rc 0 ok / not covered, 1 RED (certification absent), 2 not measured.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2

has_json() { [ -s "$1" ] && { ! command -v jq > /dev/null 2>&1 || jq -e . "$1" > /dev/null 2>&1; }; } # non-empty, and parseable when jq exists

judge() { # ROOT -> 0 ok, 1 red, 2 not measured
    local root=$1 v lad rc
    v=$(awk '/^\[workspace\.package\]/{w=1;next} /^\[/{w=0} w && /^version[ ]*=/{gsub(/[" ]/,"",$0); sub(/^version=/,"",$0); print; exit}' "$root/Cargo.toml" 2> /dev/null)
    [[ $v =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "not measured: no X.Y.Z [workspace.package] version in $root/Cargo.toml"; return 2; }
    . "$root/scripts/lib/release_policy.sh" || { echo "not measured: cannot source release_policy.sh"; return 2; }
    # Called in this shell (stdout to a file), not in $( ), so RP_WHY survives (release_policy.sh header).
    local out; out=$(mktemp) || { echo "not measured: mktemp failed"; return 2; }
    RP_WHY=""
    release_policy_ladder "$root/contracts/model-capability-ladder-v1.yaml" "$v" > "$out"; rc=$?
    lad=$(< "$out"); rm -f -- "${out:?}"
    [ -z "$lad" ] || [ "$lad" = "$root/contracts/model-capability-ladder-v1.yaml" ] || rm -f -- "${lad:?}"
    [ "$rc" = 0 ] || { echo "not measured: release policy rc $rc: ${RP_WHY:-no reason given}"; return 2; }
    # release_policy_ladder ran in a subshell above, so its RP_APPLIES is not visible: re-derive coverage here.
    local since; since=$(awk '/^ *release_policy:/{p=1;next} p && /^ *since:/{gsub(/[" ]/,"",$2); print $2; exit}' "$root/contracts/model-capability-ladder-v1.yaml")
    [[ $since =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "not measured: no release_policy.since in the ladder"; return 2; }
    if ! rp_ge "$v" "$since"; then echo "ok: the standing release policy (since $since) does not cover $v"; return 0; fi
    if has_json "$root/evidence/crux/$v/prompt-certification.json" && has_json "$root/evidence/crux/$v/prompt-certification-inventory.json"; then
        echo "ok: evidence/crux/$v carries the prompt certification and its inventory"; return 0
    fi
    echo "RED: workspace version $v is covered by the standing release policy but evidence/crux/$v/{prompt-certification.json,prompt-certification-inventory.json} is absent; the models nightly cannot be measured at this commit. Carry it: scripts/release/prepare_bump.sh does (pb_carry_cert), or copy the newest earlier version's pair and set apr_commit"
    return 1
}

if [ "${1:-}" = --self-test ] || [ "${1:-}" = --self-test-table ]; then
    T=$(mktemp -d) || exit 2; trap 'rm -rf -- "${T:?}"' EXIT
    # CHECK_BUMP_REPO lets a mutated copy living in $T find the real tree.
    REPO=${CHECK_BUMP_REPO:-$(cd "$HERE/../.." && pwd)} || exit 2
    fail=0
    mk() { # name version cert(0/1) -> fixture root
        local r="$T/$1"; mkdir -p "$r/scripts" "$r/contracts" "$r/evidence/crux" || exit 2
        cp -r "$REPO/scripts/lib" "$r/scripts/lib" && cp "$REPO/contracts/model-capability-ladder-v1.yaml" "$r/contracts/" || exit 2
        printf '[workspace.package]\nversion = "%s"\n' "$2" > "$r/Cargo.toml"
        if [ "$3" = 1 ]; then mkdir -p "$r/evidence/crux/$2" && echo '{}' > "$r/evidence/crux/$2/prompt-certification.json" && echo '{}' > "$r/evidence/crux/$2/prompt-certification-inventory.json"; fi
        printf '%s' "$r"
    }
    row() { # name want root
        local out rc; out=$(judge "$3"); rc=$?
        if [ "$rc" = "$2" ]; then echo "PASS  $1"; else echo "FAIL  $1 (rc=$rc want=$2): $out"; fail=1; fi
    }
    NROWS=10
    row "covered version with certification" 0 "$(mk a 0.74.0 1)"
    row "covered version WITHOUT certification" 1 "$(mk b 0.74.0 0)"
    row "pre-policy version needs none" 0 "$(mk c 0.70.9 0)"
    d=$(mk d 0.75.0 1); rm -f "${d:?}/evidence/crux/0.75.0/prompt-certification-inventory.json"
    row "inventory missing is red" 1 "$d"
    d=$(mk e 0.74.0 1); printf '[workspace.package]\n' > "$d/Cargo.toml"
    row "no workspace version is not measured" 2 "$d"
    d=$(mk f 0.75.1 0); mkdir -p "$d/evidence/crux/0.74.0"; echo '{}' > "$d/evidence/crux/0.74.0/prompt-certification.json"; echo '{}' > "$d/evidence/crux/0.74.0/prompt-certification-inventory.json"
    row "certification of another version is red" 1 "$d"
    d=$(mk g 0.74.0 1); : > "$d/evidence/crux/0.74.0/prompt-certification.json"
    row "empty certification file is red" 1 "$d"
    d=$(mk h 0.74.0 1); echo 'not json' > "$d/evidence/crux/0.74.0/prompt-certification-inventory.json"
    row "unparseable inventory file is red" 1 "$d"
    d=$(mk i 0.74.0 1); rm -f "${d:?}/contracts/model-capability-ladder-v1.yaml"
    row "unreadable ladder is not measured" 2 "$d"
    d=$(mk j 0.74.0 1); echo 'garbage' > "$d/contracts/model-capability-ladder-v1.yaml"
    row "malformed ladder is not measured" 2 "$d"
    if [ "$1" = --self-test ] && [ "$fail" = 0 ]; then
        n=0 miss=0
        # CONTROL: the unmutated copy, run from $T, must pass every row. Otherwise no mutant verdict means anything.
        cp "$0" "$T/ctl.sh"
        ctl=$(CHECK_BUMP_REPO=$REPO bash "$T/ctl.sh" --self-test-table 2>&1); crc=$?
        cn=$(printf '%s\n' "$ctl" | grep -c '^PASS ')
        if [ "$crc" != 0 ] || [ "$cn" != "$NROWS" ]; then echo "CONTROL FAILED: unmutated copy rc=$crc passing rows=$cn want $NROWS" >&2; exit 1; fi
        echo "control: unmutated copy rc 0, $cn/$NROWS rows pass"
        # A mutant is killed only if it ran every row (no crash) and at least one assertion row printed FAIL.
        mut() { n=$((n + 1)); sed "$2" "$0" > "$T/m.sh"
            if cmp -s "$0" "$T/m.sh"; then echo "MUTANT $1 did not apply" >&2; miss=$((miss + 1)); return; fi
            local mo mrc rows fails
            mo=$(CHECK_BUMP_REPO=$REPO bash "$T/m.sh" --self-test-table 2>&1); mrc=$?
            rows=$(printf '%s\n' "$mo" | grep -c -E '^(PASS|FAIL) ')
            fails=$(printf '%s\n' "$mo" | grep -c '^FAIL ')
            if [ "$mrc" = 1 ] && [ "$rows" = "$NROWS" ] && [ "$fails" -ge 1 ]; then return; fi
            if [ "$mrc" = 0 ]; then echo "MUTANT $1 SURVIVED" >&2; else echo "MUTANT $1 died by crash, not by an assertion (rc=$mrc rows=$rows)" >&2; fi
            miss=$((miss + 1)); }
        mut cert-not-required 's/if has_json "\$root\/evidence\/crux\/\$v\/prompt-certification.json" \&\& has_json/if true || has_json/'
        mut inventory-ignored 's/ \&\& has_json "\$root\/evidence\/crux\/\$v\/prompt-certification-inventory.json"; then/; then/'
        mut policy-always-off 's/if ! rp_ge "\$v" "\$since"; then/if true; then/'
        mut wrong-version-dir '/^    if has_json/s/"\$root\/evidence\/crux\/\$v\/\([^"]*\)"/"$root"\/evidence\/crux\/*\/\1/g'
        mut empty-file-accepted 's/^has_json() .*/has_json() { [ -f "$1" ]; }/'
        mut rp-why-unset 's/\${RP_WHY:-no reason given}/$RP_WHYX/'
        echo "mutants: $n run, $miss missed"
        [ "$miss" = 0 ] || fail=1
    fi
    exit "$fail"
fi

root=$(cd "$HERE/../.." && pwd) || exit 2
[ "${1:-}" = --root ] && root=${2:?--root DIR}
judge "$root"
exit $?
