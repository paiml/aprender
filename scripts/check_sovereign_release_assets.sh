#!/usr/bin/env bash
# check_sovereign_release_assets.sh -- cross-repo release-asset contract (aprender#4328 C6)
#
# CONTRACT
# --------
# Every binary a sovereign repo releases ships, on its latest non-prerelease
# GitHub release, ONE archive per required platform, and each archive has a
# sibling `<archive>.sha256`:
#
#   linux-x86_64   x86_64-unknown-linux-musl, or an old-glibc build whose
#                  triple carries the floor: x86_64-unknown-linux-gnu.2.NN
#                  with NN <= 35. A plain `-gnu` archive does NOT count --
#                  it is built on the runner's glibc and fails to start on
#                  older hosts (the gap bashrs#385 / copia#66 are instances of).
#   linux-aarch64  aarch64-unknown-linux-gnu or aarch64-unknown-linux-musl
#   darwin-arm64   aarch64-apple-darwin
#
# Archive names are `<bin>-[v]<version>-<triple>[-<variant>].<ext>`, the
# layout forjar and pmat already ship. `<variant>` (e.g. `-cpu`, `-cuda`)
# is accepted; another binary's archive never satisfies this one
# (`pv-…` is not `pmat-…`).
#
# SCOPE vs check_release_assets.sh: that script asserts aprender's OWN tag
# carries its exact sixteen apr/pv assets (binary-release.yml, C13). This one
# asserts the platform floor across EVERY sovereign repo's latest release.
#
# MODES
# -----
#   check_sovereign_release_assets.sh               offline case table (== --self-test).
#                                          This is what guard_tree runs bare:
#                                          it never touches the network, so a
#                                          sibling repo's release cannot turn
#                                          an aprender PR red.
#   check_sovereign_release_assets.sh --self-test   same
#   check_sovereign_release_assets.sh --live [--repo OWNER/REPO] [--tag TAG]
#                                          query `gh release view` for every
#                                          manifest repo (or one), print one
#                                          row per binary x platform
#   check_sovereign_release_assets.sh --assets-file FILE --bins b1,b2
#                                          check a newline list of asset names
#
# EXIT
# ----
#   0   every required asset present
#   10  RED: at least one asset or .sha256 missing (or a repo could not be
#       queried -- fail closed). Distinct from 1/2/126/127 so a crash in this
#       script can never read as a verdict, and vice versa.
#   2   usage error

set -uo pipefail

RED_EXIT=10

# repo  binaries (comma-separated). The universe of sovereign binaries whose
# releases this contract covers. Add a repo here when it starts cutting
# binary releases.
MANIFEST='paiml/aprender apr,pv
paiml/bashrs bashrs
paiml/paiml-mcp-agent-toolkit pmat
paiml/copia copia
paiml/forjar forjar
paiml/pzsh pzsh
paiml/rmedia rmedia'

usage() {
    sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
}

# check_assets BINS  < asset names on stdin
# Prints `ok|RED  <bin>  <platform>  <detail>` rows; returns 0 or RED_EXIT.
check_assets() {
    python3 -c "$CHECK_PY" "$1"
}

# The checker. Held in a variable and passed with -c: a `python3 - <<PY`
# heredoc would BE stdin, and the asset list piped in would never be read.
CHECK_PY=$(cat <<'PY'
import re, sys
bins = [b for b in sys.argv[1].split(",") if b]
names = set(l.strip() for l in sys.stdin if l.strip())
ARCH = r"(?:tar\.gz|tgz|tar\.xz|tar\.zst|zip)"
PLATFORMS = [
    ("linux-x86_64", r"x86_64-unknown-linux-(?:musl|gnu\.2\.(?:[0-9]|[12][0-9]|3[0-5]))"),
    ("linux-aarch64", r"aarch64-unknown-linux-(?:gnu|musl)"),
    ("darwin-arm64", r"aarch64-apple-darwin"),
]
red = False
if not bins:
    print("RED  -  -  empty binary list")
    sys.exit(10)
for b in bins:
    for plat, triple in PLATFORMS:
        rx = re.compile(r"^%s-v?[0-9][0-9A-Za-z.+~]*-%s(?:-[a-z0-9]+)?\.%s$"
                        % (re.escape(b), triple, ARCH))
        hits = sorted(n for n in names if rx.match(n))
        signed = [n for n in hits if n + ".sha256" in names]
        if signed:
            print("ok   %s  %s  %s" % (b, plat, signed[0]))
        elif hits:
            red = True
            print("RED  %s  %s  %s has no .sha256" % (b, plat, hits[0]))
        else:
            red = True
            print("RED  %s  %s  no archive" % (b, plat))
sys.exit(10 if red else 0)
PY
)

# One fixture case: name, expected exit, bins, asset list.
self_test() {
    local fails=0 n=0
    run_case() {
        local name="$1" want="$2" bins="$3" assets="$4" got
        n=$((n + 1))
        printf '%s\n' "$assets" | check_assets "$bins" >/dev/null
        got=$?
        if [ "$got" -ne "$want" ]; then
            printf 'FAIL case %-34s want=%s got=%s\n' "$name" "$want" "$got"
            fails=$((fails + 1))
        fi
    }
    local full='tool-1.2.3-x86_64-unknown-linux-musl.tar.gz
tool-1.2.3-x86_64-unknown-linux-musl.tar.gz.sha256
tool-1.2.3-aarch64-unknown-linux-gnu.tar.gz
tool-1.2.3-aarch64-unknown-linux-gnu.tar.gz.sha256
tool-1.2.3-aarch64-apple-darwin.tar.gz
tool-1.2.3-aarch64-apple-darwin.tar.gz.sha256'
    run_case full-set-green 0 tool "$full"
    # Each required platform removed turns RED (the RED-turning mutations).
    run_case no-x86_64 10 tool "$(printf '%s\n' "$full" | grep -v x86_64)"
    run_case no-aarch64-linux 10 tool "$(printf '%s\n' "$full" | grep -v aarch64-unknown-linux)"
    run_case no-darwin 10 tool "$(printf '%s\n' "$full" | grep -v apple-darwin)"
    run_case missing-sha256 10 tool "$(printf '%s\n' "$full" | grep -v 'darwin.tar.gz.sha256')"
    run_case only-sha256-no-archive 10 tool "$(printf '%s\n' "$full" | grep -v 'darwin.tar.gz$')"
    run_case plain-gnu-x86_64-is-not-enough 10 tool "$(printf '%s\n' "$full" | sed 's/x86_64-unknown-linux-musl/x86_64-unknown-linux-gnu/')"
    run_case old-glibc-2.17-ok 0 tool "$(printf '%s\n' "$full" | sed 's/x86_64-unknown-linux-musl/x86_64-unknown-linux-gnu.2.17/')"
    run_case glibc-2.39-too-new 10 tool "$(printf '%s\n' "$full" | sed 's/x86_64-unknown-linux-musl/x86_64-unknown-linux-gnu.2.39/')"
    run_case v-prefixed-version 0 tool "$(printf '%s\n' "$full" | sed 's/tool-1/tool-v1/')"
    run_case cuda-variant-ok 0 tool "$(printf '%s\n' "$full" | sed 's/\.tar\.gz/-cuda.tar.gz/')"
    run_case other-bin-does-not-count 10 tool "$(printf '%s\n' "$full" | sed 's/^tool-/toolx-/')"
    run_case prefix-bin-does-not-count 10 to "$full"
    run_case zero-assets 10 tool ''
    run_case empty-bin-list 10 '' "$full"
    run_case second-bin-missing 10 tool,pv "$full"
    run_case two-bins-green 0 tool,pv "$full
$(printf '%s\n' "$full" | sed 's/^tool-/pv-/')"
    if [ "$fails" -ne 0 ]; then
        echo "self-test: $fails/$n case(s) FAILED"
        return "$RED_EXIT"
    fi
    echo "self-test: $n/$n cases pass"
}

live() {
    local only_repo="$1" tag="$2" repo bins assets rc worst=0
    command -v gh >/dev/null 2>&1 || { echo "RED  gh not on PATH"; return "$RED_EXIT"; }
    while read -r repo bins; do
        [ -n "$repo" ] || continue
        [ -z "$only_repo" ] || [ "$repo" = "$only_repo" ] || continue
        if [ -n "$tag" ]; then
            assets="$(gh release view "$tag" -R "$repo" --json assets -q '.assets[].name' 2>/dev/null)"; rc=$?
        else
            assets="$(gh release view -R "$repo" --json assets -q '.assets[].name' 2>/dev/null)"; rc=$?
        fi
        if [ "$rc" -ne 0 ]; then
            echo "RED  $repo  -  -  release query failed (fail closed)"
            worst=$RED_EXIT
            continue
        fi
        printf '%s\n' "$assets" | check_assets "$bins" | sed "s|^|$repo  |"
        rc=${PIPESTATUS[1]}
        [ "$rc" -eq 0 ] || worst=$RED_EXIT
    done <<<"$MANIFEST"
    return "$worst"
}

mode=self-test repo='' tag='' assets_file='' bins=''
while [ $# -gt 0 ]; do
    case "$1" in
        --self-test) mode=self-test ;;
        --live) mode=live ;;
        --repo) repo="${2:?--repo needs OWNER/REPO}"; shift ;;
        --tag) tag="${2:?--tag needs a tag}"; shift ;;
        --assets-file) mode=file; assets_file="${2:?--assets-file needs a path}"; shift ;;
        --bins) bins="${2:?--bins needs b1,b2}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

case "$mode" in
    self-test) self_test ;;
    live) live "$repo" "$tag" ;;
    file)
        [ -n "$bins" ] || { echo "--assets-file needs --bins" >&2; exit 2; }
        check_assets "$bins" <"$assets_file"
        ;;
esac
