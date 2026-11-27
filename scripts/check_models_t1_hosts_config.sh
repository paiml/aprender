#!/usr/bin/env bash
# check_models_t1_hosts_config.sh -- PMAT-7497 (#7497): models_t1.sh takes its host pair from the tracked
# scripts/release/models_t1_hosts.conf, parsed as data, never a literal and never sourced. No env names the
# conf and none prints the pair. The behaviour rows (conf line not run, refusals, no exit 0) are in
# scripts/check_release_models_t1.sh. Exit 0 all pass, 1 any fail.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
T1="$root/scripts/release/models_t1.sh"
CONF="$root/scripts/release/models_t1_hosts.conf"
fail=0
ok() { printf 'PASS %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1"; fail=1; }

if grep -qE '^(LOCAL_HOST|REMOTE_HOST)=[a-z0-9-]+$' "$T1"; then bad "models_t1.sh still hardcodes a host literal"; else ok "models_t1.sh has no host literal"; fi
[ -s "$CONF" ] && ok "models_t1_hosts.conf exists" || bad "models_t1_hosts.conf missing"
if grep -qE 'MODELS_T1_HOSTS_FILE|MODELS_T1_PRINT_HOSTS' "$T1"; then bad "models_t1.sh has a conf override or print hook"; else ok "no conf override, no print hook"; fi
if grep -qE '^[[:space:]]*(\.|source)[[:space:]]+"?\$' "$T1"; then bad "models_t1.sh sources a variable path"; else ok "models_t1.sh sources nothing by variable"; fi
for k in LOCAL_HOST REMOTE_HOST; do
    grep -qE "^$k=[a-z0-9-]+$" "$CONF" && ok "conf sets $k" || bad "conf lacks $k"
done
if grep -vE '^(#.*|LOCAL_HOST=[a-z0-9-]+|REMOTE_HOST=[a-z0-9-]+|)$' "$CONF" | grep -q .; then bad "conf has a line that is not data"; else ok "conf is only KEY=VALUE data and comments"; fi
exit "$fail"
