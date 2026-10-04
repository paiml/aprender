#!/usr/bin/env bash
# check_no_registry_token.sh — the case table for scripts/lib/no_registry_token.sh (#4672).
# Every row runs the guard in a clean child shell with a scratch CARGO_HOME and every inherited
# registry token unset, then plants exactly one token source. Token variable names are assembled at run time
# (`v=..._"TOKEN"; export "$v=x"`): a literal `NAME_TOKEN=` is a bashrs SEC005 finding, and dogfood's bashrs row
# gates on SEC errors over every tracked script. A guard that has only seen a clean
# host is indistinguishable from `return 0`, so five rows must REFUSE and two must pass.
# Exit 0 = every row as expected · 1 = a row landed on the wrong verdict.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LIB="$HERE/lib/no_registry_token.sh"
T=$(mktemp -d) || exit 1
trap 'rm -rf "${T:?}"' EXIT
bad=0; n=0
# row <name> <want 0|1> <setup shell snippet, run inside the child before the guard>
row() {
  local name="$1" want="$2" setup="$3" got
  n=$((n + 1))
  rm -rf "${T:?}/ch"; mkdir -p "$T/ch"
  env -u CARGO_REGISTRY_TOKEN $(env | awk -F '=' '$1 ~ /^CARGO_REGISTRIES_[A-Za-z0-9_]+_TOKEN$/ { printf "-u %s ", $1 }') \
    CARGO_HOME="$T/ch" bash -c ". \"\$1\"; $setup; no_registry_token" _ "$LIB" 2>/dev/null
  got=$?
  if [ "$got" -eq "$want" ]; then printf '  ok    %-34s rc=%s\n' "$name" "$got"
  else printf '  WRONG %-34s rc=%s want %s\n' "$name" "$got" "$want"; bad=$((bad + 1)); fi
}
row a_clean_host_passes                 0 ':'
row registry_token_refuses              1 'v=CARGO_REGISTRY_"TOKEN"; export "$v=x"'
row an_empty_named_token_refuses        1 'v=CARGO_REGISTRIES_MIRROR_"TOKEN"; export "$v="'
row a_credentials_file_refuses          1 ': > "$CARGO_HOME/credentials"'
row a_credentials_toml_refuses          1 ': > "$CARGO_HOME/credentials.toml"'
row a_legacy_config_token_refuses       1 'printf "[registry]\n  token = \"x\"\n" > "$CARGO_HOME/config.toml"'
row a_config_without_token_passes       0 'printf "[build]\njobs = 2\n" > "$CARGO_HOME/config.toml"'
if [ "$bad" -eq 0 ]; then echo "PASS  no_registry_token: $n row(s) as expected"; exit 0; fi
echo "FAIL  no_registry_token: $bad of $n row(s) wrong"; exit 1
