#!/usr/bin/env bash
# #6798: a failing nightly dogfood step must name the failing gate and the reason.
# Plants a NO-GO log and a log that died with no verdict; both must be explained.
set -uo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
exp="$here/dogfood_nightly_explain.sh"
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
fail=0
chk() { # name, want-substring, got
  case "$3" in *"$2"*) echo "ok   $1" ;; *) echo "FAIL $1: wanted '$2' in: $3"; fail=1 ;; esac
}
printf 'noise\nNO-GO — these gates are RED:\n  · fmt                      cargo fmt dirty\nVERDICT: NO-GO\n' > "$t/nogo.log"
out=$(bash "$exp" "$t/nogo.log" 1 2>&1)
chk nogo-gate "fmt" "$out"; chk nogo-reason "cargo fmt dirty" "$out"; chk nogo-rc "rc=1" "$out"
printf 'building...\nsome tool crashed\n' > "$t/crash.log"
out=$(bash "$exp" "$t/crash.log" 1 2>&1)
chk crash-says-noverdict "no VERDICT line" "$out"; chk crash-tail "some tool crashed" "$out"
out=$(bash "$exp" "$t/missing.log" 1 2>&1)
chk missing-log "unreadable" "$out"
exit "$fail"
