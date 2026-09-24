#!/usr/bin/env bash
# asset_version_check.sh — does a release asset belong to its tag? (#4275, RC-DOGFOOD-001)
#
#   bash scripts/release/asset_version_check.sh TAG COMMIT "VERSION_LINE"
#   bash scripts/release/asset_version_check.sh --self-test
#
# TAG is vX.Y.Z or vX.Y.Z-rc.N. COMMIT is the full sha the tag points at.
# VERSION_LINE is the first line of the asset's `apr --version`: `apr X.Y.Z (<sha>)`.
#
# The printed version must equal the tag EXACTLY, -rc.N included (operator 2026-09-24: "we
# need actual version numbers", "version number needs release canidate info in it"). The rc
# commit's Cargo.toml says X.Y.Z; binary-release.yml runs stamp_rc_version.sh on the tag tree
# before building, so an rc asset's CARGO_PKG_VERSION is X.Y.Z-rc.N. The v0.69.3-rc.2 asset,
# built before that, printed `apr 0.69.3 (v0.69.3+no-git)`: refused twice over, since the sha
# is required too. The version names WHICH rc; only the sha binds it to the tag's commit
# (`vX.Y.Z+no-git`, what every container build reported before binary-release.yml passed
# APR_GIT_SHA_OVERRIDE, is refused).
#
# A PROMOTED final (#4286, scripts/release/promote_rc.sh) ships the rc's bytes unchanged,
# so on a final tag vX.Y.Z the asset prints `apr X.Y.Z-rc.N (<sha>)`: the rc it was promoted
# from. That is accepted, and ONLY that: the same X.Y.Z, a well-formed -rc.N, and the sha
# must still be the final tag's commit. An rc built from another commit, or an rc of
# another version, is refused as before.
#
# The smoke this replaces tested `case "$v" in *"${TAG#v}"*)`: a substring match. It
# refused every rc (0.69.3 does not contain 0.69.3-rc.1) and accepted 0.69.30 for
# v0.69.3. The version here is compared whole.
#
# Exit: 0 the asset belongs to the tag, 1 it does not, 2 usage.
set -uo pipefail
PROG=asset_version_check

# Pure. Prints `ok <version> at <sha>` or `bad <reason>`.
avc_decide() {
    local tag=$1 commit=$2 line=$3 want got sha
    local promoted=""
    if [[ $tag =~ ^v([0-9]+\.[0-9]+\.[0-9]+)(-rc\.[0-9]+)?$ ]]; then
        want=${tag#v}
    else
        echo "bad tag '$tag' is not vX.Y.Z or vX.Y.Z-rc.N"; return
    fi
    if [[ ! $commit =~ ^[0-9a-f]{40}$ ]]; then
        echo "bad commit '$commit' is not a full 40-hex sha"; return
    fi
    if [[ $line =~ ^apr\ ([0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?)\ \((.*)\)$ ]]; then
        got=${BASH_REMATCH[1]}; sha=${BASH_REMATCH[3]}
    else
        echo "bad version line '$line' is not 'apr X.Y.Z[-rc.N] (<sha>)'"; return
    fi
    if [ "$got" != "$want" ]; then
        # a final tag carrying its own rc's bytes (#4286): X.Y.Z-rc.N on vX.Y.Z, nothing looser.
        # On an rc tag want is X.Y.Z-rc.N, so the X.Y.Z equality below already refuses there.
        if [[ $got =~ ^([0-9]+\.[0-9]+\.[0-9]+)-rc\.[0-9]+$ && ${BASH_REMATCH[1]} == "$want" ]]; then
            promoted=" (promoted from v$got)"
        else
            echo "bad asset reports $got; tag $tag wants $want"; return
        fi
    fi
    if [[ ! $sha =~ ^[0-9a-f]{7,40}$ ]]; then
        echo "bad asset carries no commit ('$sha'), so nothing binds it to $tag"; return
    fi
    if [ "${commit#"$sha"}" = "$commit" ]; then
        echo "bad asset was built from $sha; tag $tag is $commit"; return
    fi
    echo "ok $got at $sha$promoted"
}

self_test() {
    local fail=0 got want tag line c=7ff50ec2a1f671ad031ba427a275b1f983031d66
    echo "$PROG self-test: case table"
    # want<TAB>tag<TAB>commit<TAB>version line<TAB>why
    while IFS=$'\t' read -r want tag commit line why; do
        got=$(avc_decide "$tag" "$commit" "$line")
        if [ "${got%% *}" = "$want" ]; then echo "  ok   $why"; else echo "  FAIL $why: wanted $want, got '$got'"; fail=1; fi
    done <<EOF
ok	v0.69.3	$c	apr 0.69.3 (7ff50ec2a)	a final tag matches its version
ok	v0.69.3-rc.1	$c	apr 0.69.3-rc.1 (7ff50ec2a)	an rc asset prints its rc (stamp_rc_version.sh)
ok	v0.69.3-rc.12	$c	apr 0.69.3-rc.12 (7ff50ec)	a two-digit rc and a 7-hex short sha
bad	v0.69.3-rc.1	$c	apr 0.69.3 (7ff50ec2a)	an rc asset printing bare X.Y.Z is refused (never bare 0.69.3 on an rc)
bad	v0.69.3-rc.2	$c	apr 0.69.3-rc.1 (7ff50ec2a)	rc.1's version on the rc.2 tag
bad	v0.69.3-rc.1	$c	apr 0.69.3-rc.12 (7ff50ec2a)	rc.12 is not rc.1 (compared whole)
ok	v0.69.3	$c	apr 0.69.3-rc.2 (7ff50ec2a)	a promoted final prints the rc it was promoted from (#4286)
bad	v0.69.3	$c	apr 0.69.3-rc.2 (0badc0de1)	a promoted final's rc must be built from the final tag's commit
bad	v0.69.3	$c	apr 0.69.4-rc.1 (7ff50ec2a)	an rc of another version on the final tag
bad	v0.69.3	$c	apr 0.69.3-rc (7ff50ec2a)	an rc suffix without its number
bad	v0.69.3	$c	apr 0.69.3-rc.2x (7ff50ec2a)	an rc suffix with trailing junk
bad	v0.69.3	$c	apr 0.69.3-beta.1 (7ff50ec2a)	a pre-release that is not an rc on the final tag
bad	v0.69.3-rc.2	$c	apr 0.69.3 (v0.69.3+no-git)	the measured v0.69.3-rc.2 asset (2026-09-24): no rc, no sha
ok	v0.69.3	$c	apr 0.69.3 ($c)	a full-length sha
bad	v0.69.3-rc.1	$c	apr 0.69.2 (7ff50ec2a)	an rc tag does not match the previous version
bad	v0.69.3	$c	apr 0.69.30 (7ff50ec2a)	0.69.30 is not 0.69.3 (the old substring glob said it was)
bad	v0.69.3	$c	apr 0.69.3 (0badc0de1)	a sha that is not the tag's commit
bad	v0.69.3	$c	apr 0.69.3 (v0.69.3+no-git)	an asset built without a sha cannot be bound to a commit
bad	v0.69.3	$c	apr 0.69.3 ()	an empty sha
bad	v0.69.3	$c	apr 0.69.3 (7ff50e)	a 6-hex sha is too short to bind anything
bad	v0.69.3-beta	$c	apr 0.69.3 (7ff50ec2a)	a tag that is neither vX.Y.Z nor vX.Y.Z-rc.N
bad	v0.69.3	7ff50ec2a	apr 0.69.3 (7ff50ec2a)	the expected commit must be a full sha
bad	v0.69.3	$c	pv 0.69.3 (7ff50ec2a)	a line that is not apr's
EOF
    if [ "$fail" -eq 0 ]; then echo "$PROG self-test: PASS"; return 0; fi
    echo "$PROG self-test: FAIL"; return 1
}

main() {
    case "${1:-}" in
        --self-test) self_test; return ;;
        -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}"; return 0 ;;
    esac
    if [ "$#" -ne 3 ]; then echo "$PROG: usage: TAG COMMIT \"VERSION_LINE\" | --self-test" >&2; return 2; fi
    local verdict
    verdict=$(avc_decide "$1" "$2" "$3")
    echo "$PROG: $verdict"
    [ "${verdict%% *}" = ok ]
}

main "$@"
