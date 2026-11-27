#!/usr/bin/env bash
# check_models_t1_hosts_config.sh -- PMAT-7497 (#7497): models_t1.sh takes its host pair from
# scripts/release/models_t1_hosts.conf (override MODELS_T1_HOSTS_FILE), never a literal.
# Operator 2026-11-27 1336Z: crux runs only on gx10, kfocus, jetson. Exit 0 all pass, 1 any fail.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
T1="$root/scripts/release/models_t1.sh"
CONF="$root/scripts/release/models_t1_hosts.conf"
fail=0
ok() { printf 'PASS %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1"; fail=1; }

if grep -qE '^(LOCAL_HOST|REMOTE_HOST)=[a-z0-9-]+$' "$T1"; then bad "models_t1.sh still hardcodes a host literal"; else ok "models_t1.sh has no host literal"; fi
[ -s "$CONF" ] && ok "models_t1_hosts.conf exists" || bad "models_t1_hosts.conf missing"
grep -q 'MODELS_T1_HOSTS_FILE' "$T1" && ok "models_t1.sh honours MODELS_T1_HOSTS_FILE" || bad "no MODELS_T1_HOSTS_FILE override"
for k in LOCAL_HOST REMOTE_HOST; do
    grep -qE "^$k=[a-z0-9-]+$" "$CONF" && ok "conf sets $k" || bad "conf lacks $k"
done
# behaviour: a conf naming other hosts is what the script reports
tmp=$(mktemp -d /mnt/nvme-raid0/tmp/t1hosts.XXXXXX)
printf 'LOCAL_HOST=kfocus\nREMOTE_HOST=jetson\n' > "$tmp/h.conf"
out=$(MODELS_T1_HOSTS_FILE="$tmp/h.conf" MODELS_T1_PRINT_HOSTS=1 bash "$T1" 2>&1 || true)
case $out in *"hosts kfocus jetson"*) ok "override conf drives the host pair" ;; *) bad "override conf ignored: $out" ;; esac
rm -rf "${tmp:?}"
exit "$fail"
