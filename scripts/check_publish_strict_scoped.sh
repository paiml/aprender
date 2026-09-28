#!/usr/bin/env bash
# check_publish_strict_scoped.sh -- case table for publish_strict.sh --only / --tag and its
# fail-closed clean-room check (#4587 B2a: pv 0.69.2 publishes on its own).
#
# Runs publish_strict.sh from THIS tree against a scratch clone of HEAD, tagged with a scoped tag.
# Nothing reaches the network or crates.io: curl, gh and `cargo publish` are shims.
#   curl         only aprender-contracts-macros is "live" (at its own version, as on crates.io)
#   gh           GH_MODE=green|red|noart|wrongsha decides the clean-room run it reports
#   cargo        `publish` exits 1 (reaching it proves every gate before it passed); the rest is real
# Exit 0 when every row matches, 1 otherwise.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PS="$ROOT/scripts/release/publish_strict.sh"
# the toolchain binary, not a PATH wrapper: a wrapper that re-resolves `cargo` on PATH finds the shim
REAL_CARGO=$(rustup which cargo 2>/dev/null) || REAL_CARGO=$(command -v cargo) || { echo "no cargo on PATH" >&2; exit 2; }
REAL_CARGO_HOME=${CARGO_HOME:-$HOME/.cargo}
REAL_RUSTUP_HOME=${RUSTUP_HOME:-$HOME/.rustup}
META=$("$REAL_CARGO" metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" 2>/dev/null)
ver_of() { python3 -c 'import json,sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"]==sys.argv[1]))' "$1" <<< "$META"; }
VER=$(ver_of aprender-contracts-cli)
MACROS_VER=$(ver_of aprender-contracts-macros)   # the one crate the pv pair needs and does not publish
SCOPED="pv-v$VER"
TMP=$(mktemp -d); trap 'rm -rf "${TMP:?}"' EXIT
WT="$TMP/ap/$SCOPED/wt"
git clone -q --no-checkout "$ROOT" "$WT"
git -C "$WT" -c core.hooksPath=/dev/null checkout -q --detach "$(git -C "$ROOT" rev-parse HEAD)"
git -C "$WT" tag "$SCOPED"
TSHA=$(git -C "$WT" rev-parse HEAD)
mkdir -p "$TMP/home/.cargo" "$TMP/cargohome/bin"
printf '[registry]\ntoken = "not-a-token"\n' > "$TMP/home/.cargo/credentials.toml"
cat > "$TMP/cargohome/bin/curl" <<EOF
#!/usr/bin/env bash
case "\$*" in */aprender-contracts-macros*) echo '{"name":"aprender-contracts-macros","vers":"$MACROS_VER"}' ;; *) exit 22 ;; esac
EOF
cat > "$TMP/cargohome/bin/cargo" <<EOF
#!/usr/bin/env bash
[ "\${1:-}" = publish ] && { echo "shim: cargo publish refused"; exit 1; }
CARGO_HOME="$REAL_CARGO_HOME" exec "$REAL_CARGO" "\$@"
EOF
cat > "$TMP/cargohome/bin/gh" <<EOF
#!/usr/bin/env bash
case "\$1 \$2" in
  "run view") case \$GH_MODE in red) echo failure;; *) echo success;; esac ;;
  "run download")
    [ "\$GH_MODE" = noart ] && exit 1
    while [ \$# -gt 0 ]; do [ "\$1" = -D ] && d=\$2; shift; done
    mkdir -p "\$d"
    if [ "\$GH_MODE" = wrongsha ]; then echo "aprender,pass,0000000000000000000000000000000000000000" > "\$d/aprender.csv"
    else echo "aprender,pass,$TSHA" > "\$d/aprender.csv"; fi ;;
  *) exit 1 ;;
esac
EOF
chmod +x "$TMP/cargohome/bin/"*
AP="$TMP/ap/$SCOPED"
printf '1\n' > "$AP/cleanroom-run-id"; printf '2\n' > "$AP/b2gpu-run-id"; printf 'abc\n' > "$AP/dryrun-receipt-commit"

fails=0
row() { # row <want-rc> <must-contain> <label> <GH_MODE> <args...>
  local want=$1 needle=$2 label=$3 mode=$4 out rc; shift 4
  out=$(cd "$WT" && HOME="$TMP/home" CARGO_HOME="$TMP/cargohome" RUSTUP_HOME="$REAL_RUSTUP_HOME" \
        PATH="$TMP/cargohome/bin:$PATH" GH_MODE=$mode RELEASE_AP="$TMP/ap/v$VER" bash "$PS" "$VER" "$@" 2>&1); rc=$?
  if [ "$rc" = "$want" ] && grep -qF -- "$needle" <<< "$out"; then printf 'ok    %s\n' "$label"
  else printf 'FAIL  %s: rc=%s (want %s), wanted "%s" in:\n%s\n' "$label" "$rc" "$want" "$needle" "$(tail -3 <<< "$out")"; fails=$((fails+1)); fi
}
set +e
row 0 "aprender-contracts-cli $VER TODO" "--plan --only the pv pair --tag scoped: both listed" green \
    --plan --only aprender-contracts,aprender-contracts-cli --tag "$SCOPED"
row 2 "does not end in" "--tag naming another version -> refused" green --plan --tag "pv-v$VER.1"
row 2 "does not end in" "--tag whose dots are not dots -> refused" green --plan --tag "pv-v${VER//./x}"
row 2 "usage:" "--only with no list -> usage" green --plan --only
row 1 "not in the universe" "--only a crate outside the universe -> STOP" green --plan --only no-such-crate --tag "$SCOPED"
row 1 "which it needs at $VER, and that is not live" "--only the cli without the lib it needs -> STOP" green \
    --plan --only aprender-contracts-cli --tag "$SCOPED"
row 1 "crate 1/2 aprender-contracts rc=1" "green clean-room on the tag sha -> reaches cargo publish" green \
    --only aprender-contracts,aprender-contracts-cli --tag "$SCOPED"
row 1 "not success" "clean-room job red -> STOP before any publish" red \
    --only aprender-contracts,aprender-contracts-cli --tag "$SCOPED"
row 1 "no result-aprender artifact" "clean-room with no artifact -> STOP" noart \
    --only aprender-contracts,aprender-contracts-cli --tag "$SCOPED"
row 1 "did not test $SCOPED" "clean-room that tested another commit -> STOP" wrongsha \
    --only aprender-contracts,aprender-contracts-cli --tag "$SCOPED"
set -e
[ "$fails" -eq 0 ] && { echo PASS; exit 0; }
echo "FAIL: $fails row(s)"; exit 1
