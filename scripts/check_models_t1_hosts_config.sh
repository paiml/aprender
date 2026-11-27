#!/usr/bin/env bash
# check_models_t1_hosts_config.sh -- PMAT-7497 (#7497): models_t1.sh takes its host pair from the tracked
# scripts/release/models_t1_hosts.conf, parsed as data, never a literal and never sourced. No env names the
# conf and none prints the pair. This checks the TRACKED conf as data: only comments, blanks and the two keys,
# each key exactly once, each value in the operator's set (REMOTE_HOST gx10|kfocus|jetson, never lambda;
# LOCAL_HOST lambda|gx10|kfocus|jetson), and two different hosts. Its own --self-test (also run first on every
# plain run) feeds it a bad conf per defect class and a padded one, each of which must FAIL, and good ones,
# which must PASS. The behaviour rows (conf line not run, refusals, identity, no exit 0) are in
# scripts/check_release_models_t1.sh. Exit 0 all pass, 1 any fail, 2 the self-test could not run.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
T1="$root/scripts/release/models_t1.sh"
CONF="$root/scripts/release/models_t1_hosts.conf"
# the one line that names the conf: beside the script, from no env
CONF_LINE='mt_conf="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models_t1_hosts.conf"'

# check_conf FILE -> one "FAIL ..." or "PASS ..." line per check; exit 1 on any FAIL
check_conf() {
    local f=$1 cf=0 k n l r
    [ -f "$f" ] && [ -s "$f" ] || { printf 'FAIL conf missing or empty\n'; return 1; }
    printf 'PASS conf exists and is not empty\n'
    # grep -v with -q: a match is a line that is not data; no pipe, so no SIGPIPE can hide one (#7498 round 2)
    if grep -qvE '^(#.*|LOCAL_HOST=[a-z0-9][a-z0-9-]*|REMOTE_HOST=[a-z0-9][a-z0-9-]*|)$' "$f"; then printf 'FAIL conf has a line that is not data\n'; cf=1
    else printf 'PASS conf is only KEY=VALUE data and comments\n'; fi
    for k in LOCAL_HOST REMOTE_HOST; do
        n=$(grep -cE "^$k=" "$f")
        if [ "$n" = 1 ]; then printf 'PASS conf sets %s exactly once\n' "$k"; else printf 'FAIL conf sets %s %s time(s), not once\n' "$k" "$n"; cf=1; fi
    done
    l=$(sed -n 's/^LOCAL_HOST=//p' "$f" | head -n 1); r=$(sed -n 's/^REMOTE_HOST=//p' "$f" | head -n 1)
    case $l in
        lambda|gx10|kfocus|jetson) printf 'PASS LOCAL_HOST=%s is one of lambda gx10 kfocus jetson\n' "$l" ;;
        *) printf "FAIL LOCAL_HOST='%s' is not one of lambda gx10 kfocus jetson\n" "$l"; cf=1 ;;
    esac
    case $r in
        gx10|kfocus|jetson) printf 'PASS REMOTE_HOST=%s is one of gx10 kfocus jetson\n' "$r" ;;
        *) printf "FAIL REMOTE_HOST='%s' is not one of gx10 kfocus jetson (never lambda)\n" "$r"; cf=1 ;;
    esac
    if [ -n "$l" ] && [ "$l" != "$r" ]; then printf 'PASS LOCAL_HOST != REMOTE_HOST\n'
    else printf "FAIL LOCAL_HOST == REMOTE_HOST ('%s'): one host judged twice\n" "$l"; cf=1; fi
    return "$cf"
}

self_test() {
    local d st=0 name body want got i
    d=$(mktemp -d -p "${RUNNER_TEMP:-${TMPDIR:-/tmp}}") || return 2
    while IFS='~' read -r want name body; do
        [ -n "$name" ] || continue
        printf '%b' "$body" > "$d/$name.conf"
        if check_conf "$d/$name.conf" > "$d/$name.out"; then got=PASS; else got=FAIL; fi
        if [ "$got" = "$want" ]; then printf 'ok   self-test %-8s -> %s\n' "$name" "$want"
        else printf 'FAIL self-test %-8s wanted %s, got %s\n' "$name" "$want" "$got"; st=1; fi
    done <<'CASES'
PASS~good~# c\nLOCAL_HOST=lambda\nREMOTE_HOST=gx10\n
PASS~moved~LOCAL_HOST=kfocus\n\nREMOTE_HOST=jetson\n
FAIL~same~LOCAL_HOST=gx10\nREMOTE_HOST=gx10\n
FAIL~swapped~LOCAL_HOST=gx10\nREMOTE_HOST=lambda\n
FAIL~foreign~LOCAL_HOST=foo\nREMOTE_HOST=bar\n
FAIL~badlocal~LOCAL_HOST=intel\nREMOTE_HOST=gx10\n
FAIL~dup~LOCAL_HOST=lambda\nREMOTE_HOST=gx10\nLOCAL_HOST=kfocus\n
FAIL~nondata~LOCAL_HOST=lambda\nREMOTE_HOST=gx10\ntouch x\n
FAIL~nokey~LOCAL_HOST=lambda\n
FAIL~dash~LOCAL_HOST=lambda\nREMOTE_HOST=-gx10\n
FAIL~empty~
CASES
    # one bad line after 5000 good comment lines: the shape a pipe into grep -q could pass by SIGPIPE (round 2)
    { for ((i = 0; i < 5000; i++)); do echo '# pad'; done; printf 'LOCAL_HOST=lambda\nREMOTE_HOST=gx10\nrm -rf x\n'; } > "$d/padded.conf"
    if check_conf "$d/padded.conf" > /dev/null; then echo "FAIL self-test padded   wanted FAIL, got PASS"; st=1; else echo "ok   self-test padded   -> FAIL"; fi
    rm -rf -- "${d:?}"
    return "$st"
}

self_test; st=$?
[ "$st" -eq 0 ] || { echo "FAIL the checker's own self-test (rc $st): a bad conf would pass"; exit 1; }
[ "${1:-}" = --self-test ] && exit 0
fail=0
check_conf "$CONF" || fail=1
ok() { printf 'PASS %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1"; fail=1; }
if grep -qE '^(LOCAL_HOST|REMOTE_HOST)=[a-z0-9-]+$' "$T1"; then bad "models_t1.sh still hardcodes a host literal"; else ok "models_t1.sh has no host literal"; fi
if grep -qE 'MODELS_T1_HOSTS_FILE|MODELS_T1_PRINT_HOSTS' "$T1"; then bad "models_t1.sh has a conf override or print hook"; else ok "no conf override, no print hook"; fi
if grep -qE '^[[:space:]]*(\.|source)[[:space:]]+"?\$' "$T1"; then bad "models_t1.sh sources a variable path"; else ok "models_t1.sh sources nothing by variable"; fi
if grep -qxF -- "$CONF_LINE" "$T1"; then ok "models_t1.sh names the conf beside itself, from no env"; else bad "models_t1.sh does not name its conf with the fixed line"; fi
exit "$fail"
