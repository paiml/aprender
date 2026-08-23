#!/usr/bin/env bash
# check_hermetic_stdin_tests.sh — aprender#2307 (class guard for the #2607 test)
#
# THE CLASS: a test that asserts something about the fd 0 it INHERITED.
#
# `cargo nextest` gives every test its own process with /dev/null on fd 0.
# A plain `cargo test` runs tests as THREADS of one process that inherited the
# caller's stdin — a pipe or a redirected file under CI. So an assertion about
# the inherited fd 0 is green on the required `workspace-test` lane (nextest)
# and red on `make coverage` / `make tier3` (plain cargo test). That divergence
# hid the #2607 test's environment dependence until it blocked the 0.64.0
# coverage measurement — the failure was invisible to every required check.
#
# THE RULE: if a file asserts on this process' stdin, that same file must
# establish fd 0 itself — re-exec the test binary with `Stdio::null()` on fd 0
# (or otherwise own fd 0) and make the assertion in the child. Reading stdin is
# fine; ASSERTING on what a harness happened to hand you is not.
#
# Deliberately NOT covered (know the limits before trusting a green run):
#   * granularity is per FILE, not per assertion — a file that legitimately
#     re-execs somewhere is not re-checked assertion by assertion;
#   * `#[cfg(test)]` blocks are not parsed, so a production `assert!` about
#     stdin is flagged too (which is also worth flagging);
#   * only fd 0 is modelled. An inherited-fd-1/2 assertion is a sibling defect
#     this guard does not see.
#
# Usage:
#   scripts/check_hermetic_stdin_tests.sh              # scan the repo
#   scripts/check_hermetic_stdin_tests.sh --self-test  # run the case table
#   scripts/check_hermetic_stdin_tests.sh --scan DIR   # scan one directory
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Emits "path:line: <source line>" for every assertion made about the stdin
# this process inherited, in a file that never establishes fd 0 itself.
scan_file() {
    local file="$1"
    # A file that hands fd 0 to a child it spawns owns its own stdin: exempt.
    if grep -q 'Stdio::null()' "$file"; then
        return 0
    fi
    # An assertion and the fd-0 expression it is about are rarely on one line:
    # rustfmt breaks `assert!(...)` open, and the `/dev/stdin` stat usually sits
    # a line ABOVE its assert (`if let Ok(meta) = metadata("/dev/stdin")`). So
    # the window is +/- 3 lines in BOTH directions, not a forward scan.
    awk -v path="$file" '
        { line[NR] = $0 }
        function is_stdin_expr(s) {
            # `std::io::stdin()` / `io::stdin()`: fd 0 of THIS process.
            if (s ~ /io::stdin\(\)/) return 1
            # A filesystem call ON /dev/stdin. The bare string is not enough —
            # `assert!(is_stdin("/dev/stdin"))` classifies a path, it never
            # touches fd 0.
            if (s ~ /(metadata|symlink_metadata|read_link|canonicalize|File::open|OpenOptions)[^;]*"\/dev\/stdin"/) return 1
            return 0
        }
        function is_assert(s) {
            return s ~ /assert!\(|assert_eq!\(|assert_ne!\(|debug_assert!\(|debug_assert_eq!\(/
        }
        END {
            for (i = 1; i <= NR; i++) {
                if (!is_stdin_expr(line[i])) continue
                lo = i - 3; if (lo < 1) lo = 1
                hi = i + 3; if (hi > NR) hi = NR
                for (j = lo; j <= hi; j++) {
                    if (is_assert(line[j])) {
                        printf "%s:%d: %s\n", path, i, line[i]
                        break
                    }
                }
            }
        }
    ' "$file"
}

scan_dir() {
    local dir="$1"
    local file
    local found=0
    while IFS= read -r file; do
        [ -f "$file" ] || continue
        if out="$(scan_file "$file")" && [ -n "$out" ]; then
            printf '%s\n' "$out"
            found=1
        fi
    done < <(find "$dir" -type f -name '*.rs' | sort)
    return "$found"
}

self_test() {
    local tmp
    tmp="$(mktemp -d)"
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" EXIT
    local failures=0

    # --- MUST MATCH (the defect this guard exists for) -------------------
    cat > "$tmp/must_match_same_line.rs" <<'EOF'
#[test]
fn t() {
    assert!(!std::io::IsTerminal::is_terminal(&std::io::stdin()), "boom");
}
EOF
    cat > "$tmp/must_match_wrapped.rs" <<'EOF'
#[test]
fn t() {
    assert!(
        !std::io::IsTerminal::is_terminal(&std::io::stdin()),
        "harness attached a terminal to stdin"
    );
}
EOF
    cat > "$tmp/must_match_dev_stdin.rs" <<'EOF'
#[test]
fn t() {
    let meta = std::fs::metadata("/dev/stdin").expect("stat");
    assert!(
        !kind_can_carry_input(&meta.file_type()),
        "harness attached a pipe to stdin"
    );
}
EOF
    # The /dev/stdin form above puts the path BEFORE the assert; catch the
    # common shape where it follows one too.
    cat > "$tmp/must_match_dev_stdin_after.rs" <<'EOF'
#[test]
fn t() {
    assert_eq!(
        std::fs::metadata("/dev/stdin").is_ok(),
        true
    );
}
EOF

    # --- MUST NOT MATCH --------------------------------------------------
    cat > "$tmp/must_not_match_production_branch.rs" <<'EOF'
pub fn run() {
    // Production code READING fd 0 is fine — it makes no claim about it.
    if std::io::stdin().is_terminal() {
        repl();
    }
}
EOF
    cat > "$tmp/must_not_match_reader_arg.rs" <<'EOF'
#[test]
fn t() {
    // Asserting on an OWNED reader is the falsifiable form we want.
    let mut empty = std::io::BufReader::new(std::io::empty());
    assert!(!reader_has_input(&mut empty));
}
EOF
    cat > "$tmp/must_not_match_hermetic.rs" <<'EOF'
#[test]
fn t() {
    if std::env::var_os("CHILD").is_some() {
        assert!(
            !std::io::IsTerminal::is_terminal(&std::io::stdin()),
            "child stdin is a terminal"
        );
        return;
    }
    let out = std::process::Command::new(exe)
        .stdin(std::process::Stdio::null())
        .output()
        .expect("re-exec");
    assert!(out.status.success());
}
EOF
    cat > "$tmp/must_not_match_path_classifier.rs" <<'EOF'
#[test]
fn t() {
    // The literal is DATA here — a path classifier, no syscall, no fd 0.
    assert!(is_stdin("/dev/stdin"));
    assert!(!is_stdout("/dev/stdin"));
}
EOF
    cat > "$tmp/must_not_match_far_apart.rs" <<'EOF'
#[test]
fn t() {
    assert!(cfg.enabled);
    let a = 1;
    let b = 2;
    let c = 3;
    let _ = std::io::stdin();
}
EOF

    local f base
    while IFS= read -r f; do
        base="$(basename "$f")"
        out="$(scan_file "$f" || true)"
        case "$base" in
            must_match_*)
                if [ -z "$out" ]; then
                    echo "SELF-TEST FAIL: $base should have been flagged, was not"
                    failures=$((failures + 1))
                fi
                ;;
            must_not_match_*)
                if [ -n "$out" ]; then
                    echo "SELF-TEST FAIL: $base must not be flagged, got: $out"
                    failures=$((failures + 1))
                fi
                ;;
        esac
    done < <(find "$tmp" -type f -name '*.rs' | sort)

    if [ "$failures" -ne 0 ]; then
        echo "check_hermetic_stdin_tests.sh: SELF-TEST FAILED ($failures case(s))"
        return 1
    fi
    echo "check_hermetic_stdin_tests.sh: self-test OK (9 cases: 4 must-match, 5 must-not-match)"
    return 0
}

main() {
    case "${1:---repo}" in
        --self-test)
            self_test
            ;;
        --scan)
            if scan_dir "${2:?--scan needs a directory}"; then
                echo "OK: no inherited-stdin assertions under ${2}"
            else
                echo "FAIL: assertion(s) about inherited fd 0 above (aprender#2307)"
                return 1
            fi
            ;;
        --repo)
            if scan_dir "$REPO_ROOT/crates" && scan_dir "$REPO_ROOT/src"; then
                echo "OK: no test asserts on the fd 0 it inherited (aprender#2307)"
            else
                cat <<'MSG'
FAIL: the assertion(s) above are about the fd 0 this process INHERITED.
      They pass under `cargo nextest` (own process, /dev/null on fd 0) and fail
      under a plain `cargo test` with a pipe or file on stdin — so they are
      green on the required lane and red on `make coverage` (aprender#2307).
      Fix: re-exec the test binary with `Stdio::null()` on fd 0 and assert in
      the child, as crates/aprender-orchestrate/src/agent/code_tests.rs does.
MSG
                return 1
            fi
            ;;
        *)
            echo "usage: $0 [--repo|--self-test|--scan DIR]" >&2
            return 2
            ;;
    esac
}

main "$@"
