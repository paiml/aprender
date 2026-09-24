#!/usr/bin/env bash
# package_tarball_build.sh — every publishable crate's test code compiles FROM ITS PUBLISHED TARBALL (#4114).
#
# WHY. check_package_includes.sh reads source: it resolves include!() targets and compares them with
# `cargo package --list`. Since the 0.69.1 workaround for #4048 it skips `#[cfg(test)]`-only modules,
# and it passed aprender-serve 0.69.1. That .crate (sha256 22a710f0…), unpacked, fails
# `cargo test --lib --no-run` with 3 errors, all of them includes the reader skipped:
#   fusion_call_site_guard_3985.rs  include_str!("../../../contracts/kernel-fusion-v1.yaml")
#   tokenizer_tests_unk_3609.rs     include_bytes! of two tests/fixtures/gguf-header-slices/* files
#                                   (the package excludes /tests/)
# A reader of the source can only agree with itself. This gate measures the ARTIFACT, and it is the
# verdict. The source-side reader stays as the fast path.
#
# WHAT IT DOES
#   1. `cargo package -p <crate> --no-verify` for every publishable workspace crate
#      (scripts/lib/publishable_crates.py), in a private target dir, so no stale .crate is read.
#   2. Unpacks every .crate OUTSIDE the repository (`tar -m`: mtimes are NOW, never the archive's --
#      cargo's freshness check is by mtime, so an archived mtime older than a warm target dir's
#      artifact would reuse a build of DIFFERENT sources of the same name+version: a false green,
#      measured in this gate's own case table) and makes them ONE workspace
#      (scripts/lib/tarball_workspace.py). Every sibling is patched to its unpacked tarball through
#      [patch.crates-io]: before a publish the siblings are not on crates.io at this version. The
#      patch is printed, not hidden. The repository's Cargo.lock and rust-toolchain.toml are copied
#      in, so third-party versions and the compiler match the release.
#   3. `cargo build --workspace --tests --keep-going` in that workspace: every target with test=true,
#      compiled in test mode -- the lib's unit tests (#[cfg(test)]) and the integration tests a crate
#      still ships (cargo drops the targets whose files are excluded). That is what
#      `cargo test --no-run` compiles, but `cargo test` has no --keep-going, and one crate's error must
#      not hide the next crate's.
#   4. scripts/lib/tarball_build_errors.py names every error by crate and file:line.
#
# OPTIONS
#   --root DIR            the repository to package (default: this script's repo)
#   --crate-file F.crate  use F instead of the freshly packaged crate of the same name (repeatable).
#                         This is how a PUBLISHED artifact is judged, e.g. the negative control.
#   --allow-dirty         package uncommitted changes too (a developer measuring a branch). The
#                         release run never passes it: a dirty tree there is "could not check" (2).
#   --negative-control    fetch the published aprender-serve 0.69.1 .crate (sha256 pinned), judge it
#                         with --root at a v0.69.1 tree, and require RED naming the 3 includes above.
#                         Exit 0 = the gate SEES the defect; 1 = it went green on a known-broken crate.
#                         Needs the network and --root at a 0.69.1 checkout.
#   TARBALL_BUILD_TARGET_DIR  the build's target dir (default: ${CARGO_TARGET_DIR:-<root>/target}/tarball-build)
#
# COST. One workspace test build (dependencies are shared across all crates, and a warm target dir is
# reused), plus about 1 min of packaging. Release-time only: [package.metadata.dogfood].gates runs it
# in `scripts/dogfood.sh`. The PR-time half is its case table in check_package_includes.sh --self-test
# (a one-crate fixture: a clean tarball is green, and a planted #[cfg(test)] include of an excluded
# file is RED).
#
# Exit: 0 every tarball compiles · 1 a tarball does not (named) · 2 could not check (not a pass).
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CRATE_FILES=(); NEG=0; DIRTY=()
NEG_URL="https://static.crates.io/crates/aprender-serve/aprender-serve-0.69.1.crate"
NEG_SHA="22a710f0bbce7c0e67a90f255e51cec37a4f0390ce9c2561fa3750a7c89e8d9a"
NEG_NEEDLES=("kernel-fusion-v1.yaml" "gguf-header-slices" "fusion_call_site_guard_3985.rs" "tokenizer_tests_unk_3609.rs")

while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$(cd -- "$2" && pwd)" || { echo "  cannot check: no directory $2" >&2; exit 2; }; shift 2 ;;
    --crate-file) CRATE_FILES+=("$2"); shift 2 ;;
    --allow-dirty) DIRTY=(--allow-dirty); shift ;;
    --negative-control) NEG=1; shift ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "unknown argument '$1'" >&2; exit 2 ;;
  esac
done
for t in cargo python3 tar sha256sum; do
  command -v "$t" > /dev/null || { echo "  cannot check: $t is not on PATH" >&2; exit 2; }
done
[ -f "$ROOT/Cargo.toml" ] || { echo "  cannot check: no Cargo.toml at $ROOT" >&2; exit 2; }

T=$(mktemp -d) || { echo "  cannot check: mktemp failed" >&2; exit 2; }
case "$T" in /tmp/?*) ;; *) echo "  cannot check: expected a temp dir under /tmp, got '$T'" >&2; exit 2 ;; esac
cleanup() {
  case "${T:-}" in
    /tmp/?*) [ -d "$T" ] && rm -rf -- "$T" ;;
  esac
}
trap cleanup EXIT
BUILD_TARGET="${TARBALL_BUILD_TARGET_DIR:-${CARGO_TARGET_DIR:-$ROOT/target}/tarball-build}"

if [ "$NEG" = 1 ]; then
  command -v curl > /dev/null || { echo "  cannot check: curl is not on PATH" >&2; exit 2; }
  neg_crate="$T/aprender-serve-0.69.1.crate"
  curl -fsSL --retry 3 -A "aprender package_tarball_build (#4114)" -o "$neg_crate" "$NEG_URL" \
    || { echo "  cannot check: could not fetch $NEG_URL" >&2; exit 2; }
  got=$(sha256sum "$neg_crate" | cut -d' ' -f1)
  [ "$got" = "$NEG_SHA" ] || { echo "  cannot check: $NEG_URL sha256 $got, pinned $NEG_SHA" >&2; exit 2; }
  echo "negative control: the published aprender-serve 0.69.1 (.crate sha256 ${NEG_SHA:0:16}…) judged with --root $ROOT"
  out=$(TARBALL_BUILD_TARGET_DIR="$BUILD_TARGET" bash "$0" --root "$ROOT" --crate-file "$neg_crate" 2>&1); rc=$?
  # the markers must appear in the SUBSTITUTED crate's own RED block, not anywhere in the output
  serve_block=$(awk '/^RED   aprender-serve-0\.69\.1 /{p=1; print; next} /^(RED|PASS|FAIL|ERROR) /{p=0} p' <<< "$out")
  grep -E '^(SUBSTITUTED|PATCH|FAIL|PASS) ' <<< "$out"
  printf '%s\n' "$serve_block"
  missing=()
  for n in "${NEG_NEEDLES[@]}"; do grep -qF -- "$n" <<< "$serve_block" || missing+=("$n"); done
  if [ "$rc" = 1 ] && [ "${#missing[@]}" -eq 0 ]; then
    echo "NEGATIVE CONTROL OK: the published aprender-serve 0.69.1 is RED in its own block, naming all four markers"; exit 0
  fi
  echo "NEGATIVE CONTROL FAILED: rc=$rc (want 1); unnamed: ${missing[*]:-none}"; exit 1
fi

echo "== package_tarball_build: every publishable crate of $ROOT, built from its .crate =="
meta="$(cd "$ROOT" && cargo metadata --no-deps --format-version 1 2>/dev/null)" \
  || { echo "  cannot check: cargo metadata failed in $ROOT" >&2; exit 2; }
names="$(printf '%s' "$meta" | python3 "$SCRIPT_DIR/lib/publishable_crates.py" | cut -f1)"
[ -n "$names" ] || { echo "  cannot check: no publishable crate in $ROOT (vacuous)" >&2; exit 2; }
pargs=()
while IFS= read -r n; do [ -n "$n" ] && pargs+=(-p "$n"); done <<< "$names"

# 1. package, into a private target dir so only THIS run's .crate files exist there
if ! (cd "$ROOT" && CARGO_TARGET_DIR="$T/pkg-target" cargo package "${pargs[@]}" --no-verify "${DIRTY[@]}" > "$T/package.log" 2>&1); then
  echo "  cannot check: cargo package failed:"; grep -E '^error' -A6 "$T/package.log" | head -n 20; exit 2
fi
mkdir -p "$T/ws/pkgs"
n_crate=0
for f in "$T"/pkg-target/package/*.crate; do
  [ -f "$f" ] || continue
  tar -xzmf "$f" -C "$T/ws/pkgs" || { echo "  cannot check: could not unpack $f" >&2; exit 2; }
  n_crate=$((n_crate + 1))
done
[ "$n_crate" -eq "$((${#pargs[@]} / 2))" ] \
  || { echo "  cannot check: packaged $n_crate .crate file(s) for $((${#pargs[@]} / 2)) publishable crate(s)" >&2; exit 2; }

# --crate-file: the given artifact REPLACES the packaged crate of the same name
for f in "${CRATE_FILES[@]}"; do
  [ -f "$f" ] || { echo "  cannot check: no crate file $f" >&2; exit 2; }
  top=$(tar -tzf "$f" | head -n 1); top=${top%%/*}
  cname=${top%-*}
  case "$cname" in ''|*/*|*..*) echo "  cannot check: $f unpacks to '$top'" >&2; exit 2 ;; esac
  # the packaged crate of the SAME name (name = dir minus its last -<version>; "aprender" never
  # matches "aprender-serve-0.69.1", whose name part is "aprender-serve")
  for old in "$T"/ws/pkgs/*; do
    ob=${old##*/}
    if [ -d "$old" ] && [ "${ob%-*}" = "$cname" ]; then rm -rf -- "$old"; fi
  done
  tar -xzmf "$f" -C "$T/ws/pkgs" || { echo "  cannot check: could not unpack $f" >&2; exit 2; }
  echo "SUBSTITUTED $top from $f (not the freshly packaged crate)"
done

# 2. one workspace of tarballs, siblings patched to their unpacked copies
rows="$(python3 "$SCRIPT_DIR/lib/tarball_workspace.py" "$T/ws")" || { echo "  cannot check: could not write the tarball workspace" >&2; exit 2; }
cp "$ROOT/Cargo.lock" "$T/ws/Cargo.lock" 2>/dev/null
[ -f "$ROOT/rust-toolchain.toml" ] && cp "$ROOT/rust-toolchain.toml" "$T/ws/"
echo "PATCH: $(grep -c . <<< "$rows") crate(s) resolved from their unpacked tarballs via [patch.crates-io] — before a publish the siblings are not on crates.io at this version; nothing is read from $ROOT's sources"

# 3. the build
echo "building: cargo build --workspace --tests --keep-going (target $BUILD_TARGET)"
(cd "$T/ws" && CARGO_TARGET_DIR="$BUILD_TARGET" cargo build --workspace --tests --keep-going --message-format short > "$T/build.log" 2>&1)
brc=$?

# 4. the verdict, by crate
report="$(python3 "$SCRIPT_DIR/lib/tarball_build_errors.py" "$T/build.log")"; erc=$?
if [ "$brc" -eq 0 ] && [ "$erc" -eq 0 ]; then
  echo "PASS  all $n_crate published tarball(s) compile their tests (cargo build --tests)"; exit 0
fi
if [ "$erc" -eq 1 ]; then
  printf '%s\n' "$report"
  echo "FAIL  a published tarball does not compile its own tests (cargo rc $brc)"; exit 1
fi
echo "  cannot check: cargo rc $brc and no error could be attributed to a tarball (report rc $erc):"
[ -n "$report" ] && printf '%s\n' "$report"
tail -n 15 "$T/build.log"; exit 2
