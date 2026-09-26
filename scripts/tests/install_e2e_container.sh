#!/usr/bin/env bash
# install_e2e_container.sh — `curl | bash` of scripts/install.sh in a CLEAN container, judged from outside (#3546).
#
# install_test.sh runs install.sh on the runner itself, whose HOME, PATH and tools are not a user's. This runs
# the one-liner as the README gives it, inside a fresh debian container holding nothing but curl, as a fresh
# non-root user, and checks four things against facts it resolves ITSELF on the host — never against what
# install.sh printed about itself:
#
#   1. channel   the tag the channel resolves to is the one the GitHub API says it is (and, with --expect-tag,
#                the release being published: a release gate must install THAT release, not the previous one);
#   2. checksum  the published .sha256 matches the asset, and install.sh reported verifying that same digest;
#   3. bytes     the apr on PATH inside the container is byte-identical to the one in the asset;
#   4. version   `apr --version` names the tag's version AND a prefix of the tag's commit sha. `+no-git` (a
#                build that could not see its commit) FAILS: two rcs of one version otherwise print the same.
#
# Usage:
#   bash scripts/tests/install_e2e_container.sh stable|rc [--expect-tag vX.Y.Z[-rc.N]] [--script-url URL]
#   bash scripts/tests/install_e2e_container.sh --self-test     # the version comparator's case table
# Without --script-url the checkout's scripts/install.sh is fed to the container on stdin and fetched there
# with curl from file:// (/opt of a container that exists for this one run) — the PR's installer through the same `curl | bash` pipe, with no bind mount
# (ephemeral runners talk to a sibling dockerd that cannot see the runner's files).
#
# Exit codes (scripts/tests convention): 0 pass, 1 a check failed, 2 environment (no docker / network / tool).
set -euo pipefail

GH_TOKEN="${GH_TOKEN:-}"
REPO="${APR_E2E_REPO:-paiml/aprender}"
IMAGE="${APR_E2E_IMAGE:-debian:bookworm-slim}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

die_env() {
    printf 'ENV   %s\n' "$1" >&2
    exit 2
}
fail=0
check() { # check <status> <label>
    if [ "$1" -eq 0 ]; then printf 'ok    %s\n' "$2"; else
        printf 'FAIL  %s\n' "$2"
        fail=1
    fi
}

# version_ok <apr --version output> <tag> <tag commit sha, full>: 0 when the output carries the tag's cargo
# version as a word (an rc tag's `-rc.N` is not in the crate version) and a 7+ hex prefix of the commit.
version_ok() {
    local out="$1" tag="$2" sha="$3" ver tok
    ver="${tag#v}"
    ver="${ver%%-rc.*}"
    case "$out" in *"+no-git"*) return 1 ;; *) ;; esac
    case " $out " in *" ${ver} "*) ;; *) return 1 ;; esac
    while read -r tok; do
        if [ "${sha#"$tok"}" != "$sha" ]; then return 0; fi
    done < <(printf '%s\n' "$out" | grep -oE '[0-9a-f]{7,40}')
    return 1
}

self_test() {
    local sha=d8a6df53a6c7104ef4eabfd2284908f93f1d6a3f bad=0 got want out tag
    while IFS='|' read -r want out tag; do
        if version_ok "$out" "$tag" "$sha"; then got=0; else got=1; fi
        if [ "$got" -eq "$want" ]; then printf 'ok    self-test %-32s %-14s -> %s\n' "$out" "$tag" "$want"; else
            printf 'FAIL  self-test %-32s %-14s -> %s (wanted %s)\n' "$out" "$tag" "$got" "$want"
            bad=1
        fi
    done <<'EOF'
0|apr 0.69.1 (d8a6df53a)|v0.69.1
0|apr 0.69.3 (d8a6df53a6c7)|v0.69.3-rc.2
1|apr 0.69.1 (v0.69.1+no-git)|v0.69.1
1|apr 0.69.1 (15c3032fc)|v0.69.1
1|apr 0.69.2 (d8a6df53a)|v0.69.1
1|apr 0.69.10 (d8a6df53a)|v0.69.1
1|apr 0.69.1 (d8a6df)|v0.69.1
1|apr 0.69.1|v0.69.1
EOF
    return "$bad"
}

if [ "${1:-}" = "--self-test" ]; then
    self_test
    exit $?
fi

CHANNEL="${1:-}"
[ $# -gt 0 ] && shift
case "$CHANNEL" in
    stable | rc) ;;
    *)
        printf 'usage: %s {stable,rc} [--expect-tag TAG] [--script-url URL]  or  %s --self-test\n' "$0" "$0" >&2
        exit 2
        ;;
esac
EXPECT_TAG=''
SCRIPT_URL=''
while [ $# -gt 0 ]; do
    case "$1" in
        --expect-tag)
            EXPECT_TAG="${2:?--expect-tag needs a tag}"
            shift 2
            ;;
        --script-url)
            SCRIPT_URL="${2:?--script-url needs a URL}"
            shift 2
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

for tool in curl python3 sha256sum tar; do
    command -v "$tool" >/dev/null 2>&1 || die_env "no $tool"
done
DOCKER=(docker)
docker ps >/dev/null 2>&1 || DOCKER=(sudo -n docker)
"${DOCKER[@]}" ps >/dev/null 2>&1 || die_env "no usable docker"
case "$(uname -m)" in
    x86_64) TARGET=x86_64-unknown-linux-gnu ;;
    aarch64 | arm64) TARGET=aarch64-unknown-linux-gnu ;;
    *) die_env "no apr asset for $(uname -m)" ;;
esac

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK:?}"' EXIT

# GitHub REST, authenticated when a token is present (the runners share one rate limit). The header goes
# through a file so the token never reaches argv.
: >"$WORK/headers"
if [ -n "$GH_TOKEN" ]; then printf 'Authorization: Bearer %s\n' "$GH_TOKEN" >"$WORK/headers"; fi
api() { curl -fsSL -H "@$WORK/headers" -H 'Accept: application/vnd.github+json' "https://api.github.com/repos/${REPO}/$1"; }

read -r -d '' NEWEST_RC_PY <<'EOF' || true
import json, re, sys
tags = [r["tag_name"] for r in json.load(sys.stdin) if re.fullmatch(r"v\d+\.\d+\.\d+(-rc\.\d+)?", r["tag_name"])]
print(tags[0] if tags else "")
EOF

# ── 1. what the channel must resolve to, decided here, not by install.sh ──
if [ "$CHANNEL" = stable ]; then
    TAG="$(api releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')" ||
        die_env "GitHub API (releases/latest)"
else
    # Newest-first, prereleases included; the first vX.Y.Z or vX.Y.Z-rc.N (never the rolling `nightly`).
    TAG="$(api 'releases?per_page=50' | python3 -c "$NEWEST_RC_PY")" || die_env "GitHub API (releases)"
fi
[ -n "$TAG" ] || die_env "no $CHANNEL release found"
SHA="$(api "commits/${TAG}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')" ||
    die_env "GitHub API (commits/$TAG)"
printf 'info  channel %s -> %s @ %s (%s, %s)\n' "$CHANNEL" "$TAG" "$SHA" "$TARGET" "$IMAGE"
if [ -n "$EXPECT_TAG" ]; then
    st=0
    [ "$TAG" = "$EXPECT_TAG" ] || st=1
    check "$st" "channel $CHANNEL resolves to the release under test ($EXPECT_TAG; got $TAG)"
fi

# ── 2. the asset and its published checksum, fetched independently ──
ASSET="apr-${TAG}-${TARGET}-cpu"
BASE="https://github.com/${REPO}/releases/download/${TAG}"
curl -fsSL -o "$WORK/a.tgz" "$BASE/${ASSET}.tar.gz" || die_env "download ${ASSET}.tar.gz"
curl -fsSL -o "$WORK/a.sha256" "$BASE/${ASSET}.tar.gz.sha256" || die_env "download ${ASSET}.tar.gz.sha256"
PUBLISHED="$(awk '{print $1}' "$WORK/a.sha256")"
ACTUAL="$(sha256sum "$WORK/a.tgz" | awk '{print $1}')"
st=0
[ "$PUBLISHED" = "$ACTUAL" ] || st=1
check "$st" "published sha256 matches ${ASSET}.tar.gz ($ACTUAL)"
tar -xzf "$WORK/a.tgz" -C "$WORK"
BIN_SHA="$(sha256sum "$WORK/${ASSET}/apr" | awk '{print $1}')"

# ── 3. the one-liner, in a clean container, as a fresh non-root user ──
if [ -n "$SCRIPT_URL" ]; then
    SRC=/dev/null
    FETCH="$SCRIPT_URL"
else
    SRC="$ROOT/scripts/install.sh"
    FETCH=file:///opt/install.sh
fi
RC=0
"${DOCKER[@]}" run --rm -i -e "CHANNEL=$CHANNEL" -e "FETCH=$FETCH" "$IMAGE" bash -c '
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get -qq update >/dev/null
apt-get -qq install -y --no-install-recommends curl ca-certificates >/dev/null
cat > /opt/install.sh
chmod 644 /opt/install.sh
useradd -m -s /bin/bash tester
su tester -c "set -o pipefail; export PATH=\$HOME/.local/bin:\$PATH; curl -fsSL \"$FETCH\" | bash -s -- --channel $CHANNEL --cpu && APR=\$(IFS=:; for d in \$PATH; do [ -x \"\$d/apr\" ] && { echo \"\$d/apr\"; break; }; done) && echo APR_PATH=\$APR && echo APR_VERSION=\$(\"\$APR\" --version) && echo APR_BIN_SHA=\$(sha256sum \"\$APR\")" # bashrs disable-line=SEC008,SEC015 (curl | bash IS the path under test, in a throwaway container)
' <"$SRC" >"$WORK/run.log" 2>&1 || RC=$? # bashrs disable-line=SEC008,SEC015 (the script above; see its comment)
sed 's/^/      | /' "$WORK/run.log" | tail -n 25
check "$RC" "curl-pipe-bash -s -- --channel $CHANNEL --cpu exits 0 in a clean $IMAGE (rc=$RC)"
field() { sed -n "s/^$1=//p" "$WORK/run.log" | tail -n 1; }
st=0
grep -q "sha256 ${PUBLISHED}" "$WORK/run.log" || st=1
check "$st" "install.sh reported verifying the published digest"
st=0
TESTER_BIN=/home/tester/.local/bin  # where install.sh puts it for user `tester`
[ "$(field APR_PATH)" = "$TESTER_BIN/apr" ] || st=1
check "$st" "apr on PATH is the fresh install ($(field APR_PATH))"
st=0
[ "$(field APR_BIN_SHA | awk '{print $1}')" = "$BIN_SHA" ] || st=1
check "$st" "installed apr is byte-identical to the asset's ($BIN_SHA)"
V="$(field APR_VERSION)"
st=0
version_ok "$V" "$TAG" "$SHA" || st=1
check "$st" "apr --version '$V' names $TAG and its commit ${SHA:0:9}"

exit "$fail"
