#!/usr/bin/env bash
# nightly_fetch_manifest.sh OUT -- the previous nightly-manifest.json, or an
# EMPTY OUT when the release has none yet (HTTP 404: the first run).
# Any other failure exits nonzero. Reading a network error as "no manifest"
# would make merge forget every arch's last green build (#4189).
set -euo pipefail
out="${1:?usage: nightly_fetch_manifest.sh OUT}"
url="${MANIFEST_URL:?MANIFEST_URL unset}"
if ! code=$(curl -sSL --retry 3 --retry-all-errors -o "$out" -w '%{http_code}' "$url"); then
  echo "::error::fetching $url failed (curl could not complete the request)" >&2
  exit 1
fi
case "$code" in
  200) echo "previous manifest: $(wc -c < "$out") bytes" ;;
  404) : > "$out"; echo "no previous manifest (HTTP 404): first run" ;;
  *) echo "::error::fetching $url: HTTP $code" >&2; exit 1 ;;
esac
