#!/usr/bin/env bash
# check_cargo_install_private_root.sh - no self-hosted job may install a cargo
# tool into the SHARED ~/.cargo/bin.
#
# THE CLASS (aprender#2353). mac-server hosts 16 runners under ONE user, so
# every self-hosted job shares a single $HOME/.cargo/bin. `cargo install`
# replaces a binary at that path; a concurrent job that is mid-run and execs the
# same path gets whatever the filesystem has at that instant. On 2026-07-31 the
# Coverage Nightly died with
#
#   could not execute process `/home/noah/.cargo/bin/cargo-llvm-cov` (never executed)
#   No such file or directory (os error 2)
#
# ENOENT on an absolute path means the path did not exist at exec time - which
# nothing but a concurrent writer to that path can produce. It reported an EMPTY
# coverage figure into the tracking issue: a missing measurement, the same class
# as project_coverage_floor_measurement_broken.
#
# THE RULE. A job that runs on a self-hosted runner and installs a cargo tool
# must route the install to a per-run private root, either
#   * `cargo install ... --root <dir>`, or
#   * a job/step `CARGO_INSTALL_ROOT:` (or `CARGO_HOME:`) that is NOT under
#     ~/.cargo.
# and, once it has opted into a private root, must not then put $HOME/.cargo/bin
# back at the FRONT of PATH - which would resolve the shared copy anyway and
# make the private root decorative.
#
# Out of scope on purpose:
#   * GitHub-hosted runners (`ubuntu-latest`): ephemeral home per job, no
#     sharing, nothing to race.
#   * `cargo install` inside `docker run`: the container has its own CARGO_HOME
#     (/usr/local/cargo in sovereign-ci), not the host's.
#
# Exit 0 = every self-hosted install is private (or explicitly exempt below).
# Exit 1 = at least one writes the shared path.
#
# `--self-test` runs the must-match/must-not-match case table for the line
# classifiers. Per the house rule, a guard regex ships a case table: re-run the
# table rather than re-reading the pattern.

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

WORKFLOW_DIR=".github/workflows"
MIN_EXPECTED_JOBS="${MIN_EXPECTED_JOBS:-8}"

# ---------------------------------------------------------------------------
# EXEMPTIONS - closed list, keyed "<workflow file>:<job id>", each with a reason.
# A new self-hosted `cargo install` fails by default; adding it here is a
# deliberate, reviewable act.
#
# qwen-story-daily.yml:story - installs `apr` itself into ~/.cargo/bin ON
#   PURPOSE: scripts/apr_bin.sh and the story's PATH pinning both expect the
#   freshly built binary at that path, and the very next lines assert the
#   running binary's embedded SHA equals HEAD (so a clobbered install fails
#   closed and loudly rather than silently measuring the wrong build). It
#   carries the same race exposure as everything else on that box; moving it
#   needs apr_bin.sh to move with it and is tracked separately, NOT fixed here.
# ---------------------------------------------------------------------------
is_exempt() {
    case "$1" in
        "qwen-story-daily.yml:story") return 0 ;;
        *) return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Line classifiers. Each is probed by --self-test.
# ---------------------------------------------------------------------------

# `cargo install` in COMMAND POSITION - start of a line, after a shell
# separator, or straight after a YAML `run:`. Anchoring on command position is
# what keeps prose and comments ("# cargo install writes to ~/.cargo/bin") out.
RE_INSTALL='(^|[;&|]|&&|\|\||run:)[[:space:]]*(sudo[[:space:]]+)?cargo[[:space:]]+install([[:space:]]|$)'

is_install() {
    grep -qE "$RE_INSTALL" <<< "$1"
}

# An explicit `--root <dir>` / `--root=<dir>` on the install itself.
has_explicit_root() {
    grep -qE -- '--root([[:space:]]|=)' <<< "$1"
}

# A CARGO_INSTALL_ROOT / CARGO_HOME declaration (YAML `KEY: value`, `export
# KEY=value`, or a bare `KEY=value` prefix) whose value is NOT the shared
# ~/.cargo. `CARGO_HOME: /tmp/cargo-home-security-...` (what the security job
# already does) and `CARGO_INSTALL_ROOT: /tmp/apr-cov-tools-...` both qualify;
# `CARGO_INSTALL_ROOT="$HOME/.cargo"` does not.
declares_private_root() {
    grep -qE '(CARGO_INSTALL_ROOT|CARGO_HOME)[[:space:]]*[:=]' <<< "$1" || return 1
    if grep -qE '(\$HOME|\$\{HOME\}|~)/\.cargo' <<< "$1" ; then
        return 1
    fi
    return 0
}

# `docker run` opens a chain: the `cargo install` may be several backslash
# continuations further down (ci.yml's mutants job is exactly that shape). The
# caller clears the chain on the first line that does not end in a backslash.
opens_docker_chain() {
    grep -qE '(^|[[:space:]])docker[[:space:]]+run([[:space:]]|$)' <<< "$1"
}

continues_line() {
    grep -qE '\\[[:space:]]*$' <<< "$1"
}

# First component of an `export PATH=...` assignment, or empty if the line is
# not such an assignment.
path_first_entry() {
    printf '%s\n' "$1" \
        | sed -n 's/.*export[[:space:]][[:space:]]*PATH=["'"'"']\{0,1\}\([^:"'"'"']*\).*/\1/p'
}

# A PATH assignment is FRONT-SHARED when its first component is a $HOME dir or
# the inherited $PATH (which on these runners contains ~/.cargo/bin). Such a
# line cancels a private install root.
path_front_is_shared() {
    local first
    first="$(path_first_entry "$1")"
    [ -n "$first" ] || return 1
    case "$first" in
        '$HOME'/*|'${HOME}'/*|'~'/*|'$PATH'|'${PATH}') return 0 ;;
        *) return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Self-test: must-match / must-not-match table.
# ---------------------------------------------------------------------------
self_test() {
    local fails=0

    probe() { # probe <fn> <expect 0|1> <line>
        local fn="$1" want="$2" line="$3" got=0
        "$fn" "$line" || got=1
        if [ "$got" != "$want" ]; then
            printf 'CASE-FAIL %s expected=%s got=%s on: %s\n' "$fn" "$want" "$got" "$line" >&2
            fails=$((fails + 1))
        fi
    }

    # is_install MUST match
    probe is_install 0 '          cargo install cargo-llvm-cov --locked'
    probe is_install 0 '        run: cargo install pmat --locked'
    probe is_install 0 '  command -v pmat >/dev/null 2>&1 || cargo install pmat --locked || true'
    probe is_install 0 '            cargo install cargo-mutants --locked'
    probe is_install 0 '          cargo install --path crates/apr-cli --force'
    # is_install MUST NOT match
    probe is_install 1 '          # cargo install writes to ~/.cargo/bin'
    probe is_install 1 '        name: Install cargo-mutants'
    probe is_install 1 '          echo "cargo installed already"'
    probe is_install 1 '          cargo build --release'
    probe is_install 1 '          cargo-install-helper --run'

    # has_explicit_root
    probe has_explicit_root 0 'cargo install cargo-llvm-cov --locked --root "$COV_TOOLS"'
    probe has_explicit_root 0 'cargo install pmat --root=/tmp/x'
    probe has_explicit_root 1 'cargo install pmat --locked'
    probe has_explicit_root 1 'cargo install pmat --rooted'

    # declares_private_root
    probe declares_private_root 0 '      CARGO_HOME: /tmp/cargo-home-security-intel-clean-room-14'
    probe declares_private_root 0 '      CARGO_INSTALL_ROOT: /tmp/apr-cov-tools-123-1'
    probe declares_private_root 0 '          export CARGO_INSTALL_ROOT=/tmp/tools'
    # The shape this repo now uses. $RUNNER_TEMP is under the runner's own
    # _work tree and is wiped per JOB, so it is private without the
    # half-persistence of a /tmp root (paiml/infra shared-$HOME guard,
    # rule cargo-install-boot-volatile).
    probe declares_private_root 0 '      CARGO_INSTALL_ROOT: ${{ runner.temp }}/apr-cov-tools'
    probe declares_private_root 0 '          export CARGO_INSTALL_ROOT="$RUNNER_TEMP/tools"'
    probe declares_private_root 1 '          export CARGO_INSTALL_ROOT="$HOME/.cargo"'
    probe declares_private_root 1 '      CARGO_HOME: ~/.cargo'
    probe declares_private_root 1 '          # CARGO_INSTALL_ROOT would help here'
    probe declares_private_root 1 '          export PATH="$HOME/.cargo/bin:$PATH"'

    # opens_docker_chain
    probe opens_docker_chain 0 '          docker run --rm \'
    probe opens_docker_chain 1 '          if docker image inspect "$IMAGE" > /dev/null 2>&1'
    probe opens_docker_chain 1 '          echo "docker running"'

    # path_front_is_shared
    probe path_front_is_shared 0 '          export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"'
    probe path_front_is_shared 0 '          export PATH="$PATH:$COV_TOOLS/bin"'
    probe path_front_is_shared 0 "          export PATH=~/.cargo/bin:\$PATH"
    probe path_front_is_shared 1 '          export PATH="$COV_TOOLS/bin:$HOME/.cargo/bin:$PATH"'
    probe path_front_is_shared 1 '          export PATH="/tmp/tools/bin:$PATH"'
    probe path_front_is_shared 1 '          echo "PATH is $PATH"'

    if [ "$fails" -gt 0 ]; then
        printf '\nSELF-TEST FAILED: %s failing probe(s).\n' "$fails" >&2
        return 1
    fi
    printf 'OK: classifier case table passes.\n'
    return 0
}

# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

# Emit "<start>\t<end>\t<job id>" for every job in a workflow file.
job_ranges() {
    awk '
        /^jobs:[[:space:]]*$/ { injobs=1; next }
        /^[A-Za-z]/ && injobs && job { print start "\t" NR-1 "\t" job; job=""; injobs=0; next }
        /^[A-Za-z]/ { injobs=0; next }
        injobs && /^  [A-Za-z0-9_.-]+:[[:space:]]*$/ {
            if (job) print start "\t" NR-1 "\t" job
            job=$0
            sub(/^  /,"",job)
            sub(/:.*$/,"",job)
            start=NR
        }
        END { if (job) print start "\t" NR "\t" job }
    ' "$1"
}

violations=0
jobs_scanned=0
installs_seen=0

check_job() {
    local file="$1" job="$2" start="$3" end="$4"
    local key="${file##*/}:$job"
    local selfhosted=0 privroot=0
    local line trimmed docker_chain=0 lineno

    # Pass 1: job-wide facts (runs-on, private-root declarations anywhere in the
    # job - job env, step env, or an inline export).
    while IFS= read -r line; do
        if grep -q 'runs-on:' <<< "$line" && grep -q 'self-hosted' <<< "$line"; then
            selfhosted=1
        fi
        trimmed="$(printf '%s' "$line" | sed 's/^[[:space:]]*//')"
        case "$trimmed" in '#'*) continue ;; *) ;; esac
        if declares_private_root "$line"; then
            privroot=1
        fi
    done < <(sed -n "${start},${end}p" "$file")

    jobs_scanned=$((jobs_scanned + 1))
    [ "$selfhosted" = 1 ] || return 0

    # Pass 2: the installs themselves, plus the PATH-order rule.
    lineno=$((start - 1))
    while IFS= read -r line; do
        lineno=$((lineno + 1))
        trimmed="$(printf '%s' "$line" | sed 's/^[[:space:]]*//')"
        case "$trimmed" in
            '#'*)
                continues_line "$line" || docker_chain=0
                continue
                ;;
            *) ;;
        esac

        if opens_docker_chain "$line"; then
            docker_chain=1
        fi

        if is_install "$line"; then
            installs_seen=$((installs_seen + 1))
            if [ "$docker_chain" = 1 ]; then
                : # containerized: the container owns its own CARGO_HOME
            elif has_explicit_root "$line" || [ "$privroot" = 1 ]; then
                : # private root
            elif is_exempt "$key"; then
                printf 'EXEMPT   %s:%s (%s) shared-root install, allowlisted\n' \
                    "$file" "$lineno" "$key"
            else
                printf 'SHARED-INSTALL %s:%s (job %s)\n' "$file" "$lineno" "$job"
                printf '               %s\n' "$trimmed"
                violations=$((violations + 1))
            fi
        fi

        # A private root that PATH then front-runs with $HOME/.cargo/bin is
        # decorative: the shared copy still wins resolution.
        if [ "$privroot" = 1 ] && path_front_is_shared "$line"; then
            printf 'SHARED-PATH-FIRST %s:%s (job %s)\n' "$file" "$lineno" "$job"
            printf '                  %s\n' "$trimmed"
            violations=$((violations + 1))
        fi

        continues_line "$line" || docker_chain=0
    done < <(sed -n "${start},${end}p" "$file")
}

main() {
    local f start end job
    for f in "$WORKFLOW_DIR"/*.yml; do
        [ -f "$f" ] || continue
        while IFS="$(printf '\t')" read -r start end job; do
            [ -n "$job" ] || continue
            check_job "$f" "$job" "$start" "$end"
        done < <(job_ranges "$f")
    done

    if [ "$violations" -gt 0 ]; then
        printf '\n%s shared-~/.cargo/bin exposure(s) on self-hosted jobs.\n' "$violations" >&2
        printf 'mac-server runs 16 runners under one $HOME, so `cargo install` there\n' >&2
        printf 'replaces a binary that another running job is about to exec (aprender#2353:\n' >&2
        printf 'cargo-llvm-cov, ENOENT, empty coverage figure). Give the job a private root:\n' >&2
        printf '  env:\n' >&2
        printf '    CARGO_INSTALL_ROOT: ${{ runner.temp }}/<tool>\n' >&2
        printf '  ... and put "$CARGO_INSTALL_ROOT/bin" FIRST on PATH.\n' >&2
        exit 1
    fi

    # Fail closed: a scanner that examined nothing must not report success.
    if [ "$jobs_scanned" -lt "$MIN_EXPECTED_JOBS" ]; then
        printf 'ERROR: scanned %s job(s), expected >= %s - job discovery has gone blind.\n' \
            "$jobs_scanned" "$MIN_EXPECTED_JOBS" >&2
        exit 1
    fi

    printf 'OK: %s workflow job(s) scanned, %s cargo install(s); no self-hosted job writes the shared ~/.cargo/bin.\n' \
        "$jobs_scanned" "$installs_seen"
}

case "${1:-}" in
    --self-test) self_test ;;
    *) main ;;
esac
