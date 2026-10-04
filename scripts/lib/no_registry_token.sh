# no_registry_token.sh — sourced. A job that must be UNABLE to publish proves it before it starts.
#
# no_registry_token -> 0 when no crates.io-style registry token is reachable from this shell:
#   * no CARGO_REGISTRY_TOKEN and no CARGO_REGISTRIES_<NAME>_TOKEN in the environment (set at all,
#     even empty: an empty variable is still a token slot something can fill);
#   * no credentials or credentials.toml under $CARGO_HOME (default ~/.cargo);
#   * no legacy `token =` line in $CARGO_HOME/config or config.toml.
# Returns 1 and prints the first reason on stderr otherwise. A `--no-publish` flag is not this guard:
# a flag says what the caller intends, this says what the caller CAN do.
# The rule is the one the nightly train's no-token refusal applies (#4672); it lives here so every
# scheduled producer calls the same function. Option-neutral: it sets no shell option and never
# exits — the caller decides (`no_registry_token || exit 2`).
# Case table: scripts/check_no_registry_token.sh.
no_registry_token() {
  local ch="${CARGO_HOME:-$HOME/.cargo}" v f
  v=$(env | awk -F '=' '$1 == "CARGO_REGISTRY_TOKEN" || $1 ~ /^CARGO_REGISTRIES_[A-Za-z0-9_]+_TOKEN$/ { print $1; exit }')
  if [ -n "$v" ]; then echo "no-token: $v is set in the environment" >&2; return 1; fi
  for f in "$ch/credentials" "$ch/credentials.toml"; do
    if [ -e "$f" ]; then echo "no-token: a credentials file exists at $f" >&2; return 1; fi
  done
  if grep -qsE '^[[:space:]]*token[[:space:]]*=' "$ch/config" "$ch/config.toml"; then
    echo "no-token: a token = line is in $ch/config(.toml)" >&2; return 1
  fi
  return 0
}
