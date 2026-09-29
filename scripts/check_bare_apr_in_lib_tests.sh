#!/usr/bin/env bash
# check_bare_apr_in_lib_tests.sh -- no unit-test source in the crates whose lib
# tests spawn `apr` may name the bare program "apr" (L25, 0.70.1 row #1).
#
# WHY. Under the required workspace-test nextest run (`--workspace --lib`,
# see ci.yml) 42 tests in aprender-qa-runner (40) and aprender-mcp (2) spawned the bare
# name "apr", i.e. whatever $PATH held. They passed with the fleet 0.69 binary,
# with a 0.70 binary, and with no apr at all. A test that cannot fail is not a
# test (L25).
#
# The fix is runtime first: in cfg(test) builds both crates resolve `apr` to
# either the pinned real binary (APR_BIN absolute + APR_BIN_SHA256 match) or a
# sha256-checked stub, and every spawn site panics on a bare "apr"
# (crate::apr_bin::guard_program). This guard is the static half: it catches
# the literal before a test run does, including in code paths no test reaches.
#
# SCOPE. ALL of src/ in the two crates, *test files included* -- the existing
# check_mcp_no_bare_apr_literal.sh skips `_tests.rs` and so never looked at the
# 42. Comment lines are ignored. A line may opt out only with the trailing
# marker `// L25-PLANTED`, which exists for the deliberate should_panic tests
# that prove the runtime guard fires.
#
#   bash scripts/check_bare_apr_in_lib_tests.sh              # check the tree
#   bash scripts/check_bare_apr_in_lib_tests.sh --self-test  # case table + planted RED
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

SCOPE=(crates/aprender-qa-runner/src crates/aprender-mcp/src)

# A spawn-shaped call, or a binary field/variable, given the literal "apr".
PAT='((Command::new|with_binary|run_program|spawn_cancellable|spawn_streaming|spawn_and_confirm|stream_with_sink|run_apr_streaming|guard_program|map_apr)\([[:space:]]*"apr"[[:space:]]*[,)]|(apr_binary|binary)[[:space:]]*[:=][[:space:]]*"apr"[[:space:]]*(\.|,|;|$))'

# Reads "path:line:text" rows on stdin, prints the violating ones.
violations() {
  grep -E "^[^:]+:[0-9]+:.*$PAT" | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' | grep -vF '// L25-PLANTED' || true
}

# scan ROOT -> violation rows under ROOT's scope; fails closed on a missing dir.
scan() {
  local root=$1 d
  for d in "${SCOPE[@]}"; do
    [ -d "$root/$d" ] || { printf 'MISSING %s\n' "$d"; return 0; }
  done
  (cd "$root" && grep -rnE --include='*.rs' "$PAT" "${SCOPE[@]}" 2> /dev/null | violations)
}

self_test() {
  local fail=0 line want got
  # must-match (1) / must-not-match (0)
  while IFS='|' read -r want line; do
    [ -n "$line" ] || continue
    got=$(printf 'f.rs:1:%s\n' "$line" | violations | wc -l)
    if [ "$got" != "$want" ]; then
      printf 'CASE FAIL want=%s got=%s: %s\n' "$want" "$got" "$line"
      fail=1
    fi
  done << 'CASES'
1|    let out = Command::new("apr").arg("qa").output();
1|    let out = std::process::Command::new( "apr" )
1|    let r = RealCommandRunner::with_binary("apr");
1|    let r = run_program("apr", &["--version"]);
1|    spawn_cancellable("apr", &args, &rx, 10);
1|        apr_binary: "apr".to_string(),
1|        binary: "apr".into(),
1|    let apr_binary = "apr";
1|    let _ = guard_program("apr");
0|    let _ = guard_program("apr"); // L25-PLANTED
0|    // These use Command::new("apr") directly and will fail
0|    //! spawn via `Command::new("apr")` from inside the
0|    let r = RealCommandRunner::with_binary("/nonexistent/apr");
0|    let r = RealCommandRunner::with_binary(crate::apr_bin::default_apr_binary());
0|    let r = Command::new("apr-cli");
0|    let r = Command::new("aprx");
0|        format: "apr".to_string(),
0|    let p = workspace.join("apr");
0|    Format::Apr => ("apr", "apr", "G0-FORMAT-APR-001"),
0|    validate_comparison(&prov, "gguf", "apr")
CASES

  # Planted RED: a copy of the scope, clean -> GREEN; one bare call planted in a
  # test file -> RED naming exactly that line; the scope deleted -> RED.
  local tmp
  tmp=$(mktemp -d) || return 1
  mkdir -p "$tmp/crates"
  cp -r crates/aprender-qa-runner "$tmp/crates/" && cp -r crates/aprender-mcp "$tmp/crates/"
  rm -rf "${tmp:?}/crates/aprender-qa-runner/target" "${tmp:?}/crates/aprender-mcp/target"
  got=$(scan "$tmp")
  if [ -n "$got" ]; then printf 'PLANT FAIL clean copy is not GREEN:\n%s\n' "$got"; fail=1; fi
  printf '\n#[test]\nfn planted() {\n    let _ = std::process::Command::new("apr").output();\n}\n' \
    >> "$tmp/crates/aprender-qa-runner/src/command_tests_part_a.rs"
  got=$(scan "$tmp")
  if ! printf '%s\n' "$got" | grep -q 'command_tests_part_a.rs:.*Command::new("apr")'; then
    printf 'PLANT FAIL planted Command::new("apr") not caught: %s\n' "$got"; fail=1
  fi
  rm -rf "${tmp:?}/crates/aprender-mcp/src"
  got=$(scan "$tmp")
  if ! printf '%s\n' "$got" | grep -q '^MISSING crates/aprender-mcp/src'; then
    printf 'PLANT FAIL missing scope not RED: %s\n' "$got"; fail=1
  fi
  rm -rf "${tmp:?}"

  if [ "$fail" = 0 ]; then printf 'OK  self-test: case table + planted RED\n'; fi
  return "$fail"
}

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  "") ;;
  *) printf 'usage: %s [--self-test]\n' "$0" >&2; exit 2 ;;
esac

hits=$(scan .)
if [ -n "$hits" ]; then
  printf 'FAIL  a lib test source names the bare program "apr" (resolved through $PATH):\n'
  printf '%s\n' "$hits" | sed 's/^/        /'
  printf '      Use crate::apr_bin::default_apr_binary() (qa-runner) or\n'
  printf '      crate::apr_bin::apr_binary() (mcp), or an explicit absolute path.\n'
  exit 1
fi
printf 'OK  no bare "apr" in %s\n' "${SCOPE[*]}"
