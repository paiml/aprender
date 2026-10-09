#!/usr/bin/env bash
# written by pre-push-tags.sh --install
# The pre-push hook of this clone: the tag guard first, then the pre-push that was here before
# the install (pre-push.chained), on the same stdin. A missing guard refuses the push.
set -euo pipefail
d="$(cd "$(dirname "$0")" && pwd)"
in="$(cat; printf x)"
in="${in%x}"
printf '%s' "$in" | bash "$d/pre-push-tags" "$@"
if [ -x "$d/pre-push.chained" ]; then
    printf '%s' "$in" | "$d/pre-push.chained" "$@"
fi
