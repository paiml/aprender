#!/usr/bin/env bash
# kit_receipt_run.sh KIT_TAR VERSION HOST - unpack the host-receipt kit into a private
# mktemp -d under $TMPDIR, run host_receipt.sh from it, and remove the dir on exit.
# A committed script run by path replaces the `bash -c '... rm -rf ...'` wrapper in
# autopilot.sh, which the agent permission checker cannot parse (check_no_bash_c_rm.sh).
set -uo pipefail
[ "$#" -eq 3 ] || { echo "usage: $0 KIT_TAR VERSION HOST" >&2; exit 2; }
d=$(mktemp -d "${TMPDIR:-/tmp}/kit-receipt.XXXXXX") || exit 2
trap 'if [ -n "$d" ] && [ "$d" != "/" ]; then rm -rf -- "$d"; fi' EXIT
tar -C "$d" -xf "$1" || exit 2
bash "$d/scripts/release/host_receipt.sh" "$2" "$3" --gpu-prio 1
