#!/usr/bin/env bash
# rc_publish_gate.sh — can this release candidate reach crates.io? (#4287, RC-DOGFOOD-001 §4.5)
#
#   bash scripts/release/rc_publish_gate.sh ROOT     # judge the checked-out rc tree at ROOT
#   bash scripts/release/rc_publish_gate.sh --verify ROOT   # also COMPILE every tarball (T-4)
#   bash scripts/release/rc_publish_gate.sh --self-test
#
# WHY AT THE RC. v0.68.0 and v0.69.0 never reached crates.io. v0.68.0 went GitHub-only
# when clean-room B2 went red AT PUBLISH: aprender-core's lib tests named `entrenar`, a
# path-only dev-dep that `cargo publish` deletes (#3425). The fix was a whole hotfix
# cycle. The same defects are visible on the commit, long before the tag, so the rc cut
# (rc-cut.yml, #4285) runs this before it writes the tag: red here means no rc.
#
# THREE CHECKS, each the tool that already owns its rule, pointed at ROOT:
#   graph     check_publish_preflight.sh --graph-only: R6, no versioned sibling
#             dev-dependency on a cycle (PMAT-955, #3468). R1/R3/R4/R5/R7 describe the
#             upload itself and stay at publish.
#   pathonly  check_pathonly_devdeps_unused_in_src.sh --rc-gate ROOT: src/ uses no
#             dev-dep the publish strips, judged against THIS checkout's triage (the
#             default branch), never the rc tree's own baseline, which masked #3425.
#   package   `cargo package --no-verify --locked` over every publishable member in one
#             invocation, so unpublished sibling versions resolve through cargo's local
#             overlay: the cascade's manifests, includes and version pins, without an
#             upload. Every selected crate must print a Packaged line.
#             --verify drops --no-verify: cargo unpacks each tarball and builds its lib and
#             bins against the overlay, which is what `cargo publish --dry-run` does per
#             crate, for the whole cascade at once. That is the only pre-upload check that
#             compiles the tarballs (the autopilot dryrun step used to run `--check`, a
#             version report, and never stopped). Verify does NOT build tests or benches:
#             test-in-tarball defects (#4192/#4193/#4129/#4130) stay with the tarball lane.
#             Run it on the BUMPED tree: at an already-published version (batch/0.70.0 still
#             said 0.69.0 while crates.io had 0.69.1) `^0.69.0` resolves the registry's
#             newer copy over the overlay and aprender-orchestrate red on a stale realizar
#             API -- a false red. Measured 2026-09-24 at 0.70.0: 72/72 green, 18 min warm.
#
# ROOT is a clean checkout of the rc commit. The checks run from THIS file's checkout,
# so a release branch cannot weaken the gate that judges it.
#
# EXIT  0 all three green · 1 a check found a publish defect · 2 a check could not measure
#       (a defect found outranks a check that could not run: 1 wins over 2).
set -uo pipefail
PROG=rc_publish_gate
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$(cd -- "$HERE/.." && pwd)"

# Pure: the verdict of the three check exit codes. Prints `green`, `red` or `env`.
rpg_verdict() {
    local rc red=0 env=0
    for rc in "$@"; do
        case "$rc" in
            0) ;;
            1) red=1 ;;
            *) env=1 ;;
        esac
    done
    if [ "$red" = 1 ]; then echo red
    elif [ "$env" = 1 ] || [ "$#" -ne 3 ]; then echo env
    else echo green
    fi
}

# Pure: exit 0 when AUTOPILOT's `if run_step tag` block runs `--verify "$WT"` and its
# `die "publish dry-run` BEFORE the cut_tag call (the dry run is ahead of the tag), else 1.
wiring_ahead_of_tag() {
    awk '/^if run_step tag; then/ { t = 1; next }
         t && /^fi/ { t = 0 }
         t && /^ +cut_tag / { if (v && d) ok = 1; t = 0 }
         t && /^ +bash scripts\/release\/rc_publish_gate\.sh --verify "\$WT"/ { v = 1 }
         t && v && /die "publish dry-run/ { d = 1 }
         END { exit !ok }' "$1"
}

# Pure: `cargo metadata` JSON on stdin -> every member whose `publish` is not `[]`, sorted,
# one per line; one empty line when there is none. jq, not python3 (#4352).
rpg_publishable() {
    jq -r '[.packages[] | select(.publish != []) | .name] | sort | if . == [] then "" else .[] end'
}

# Pure: `cargo metadata` JSON on stdin -> the target directory, or nothing when it is not
# an absolute path (package_walk reads nothing as ENV: it will not guess what to empty).
rpg_target_dir() {
    jq -r '.target_directory // "" | select(startswith("/"))'
}

# Pure: `cargo metadata` JSON on stdin -> `<name>-<version>` of every publishable member, one per
# line (the directory name cargo unpacks an overlay source under). Empty when there is none.
rpg_member_dirs() {
    jq -r '[.packages[] | select(.publish != []) | "\(.name)-\(.version)"] | sort | .[]'
}

# rpg_overlay_units_gone TARGET NAME-VERSION...: 0 when no dep-info under TARGET (any profile or
# triple, build-script units included) still names an overlay source dir of a member, 1 when one
# does or the scan could not run. It never reads cargo's directory layout: a unit is judged by the
# source path it was compiled from.
rpg_overlay_units_gone() {
    local tdir=$1 md esc alt=''
    shift
    [ "${tdir#/}" != "$tdir" ] && [ "$#" -gt 0 ] || return 1
    [ -d "$tdir" ] || return 0
    for md in "$@"; do
        esc=$(printf '%s' "$md" | sed 's/[][\.*^$+?(){}|/]/\\&/g') || return 1
        alt="${alt:+$alt|}$esc"
    done
    grep -RlqE --include='*.d' "/registry/src/-[^/]*/($alt)/" "$tdir"
    case $? in 1) return 0 ;; *) return 1 ;; esac
}

self_test() {
    local fail=0 got got2 want args why
    echo "$PROG self-test: case table"
    while IFS='|' read -r want args why; do
        # shellcheck disable=SC2086  # args is a space-separated list of exit codes on purpose
        got=$(rpg_verdict $args)
        if [ "$got" = "$want" ]; then echo "  ok   $why"; else echo "  FAIL $why: wanted $want, got $got"; fail=1; fi
    done <<'EOF'
green|0 0 0|all three green
red|1 0 0|the graph refused
red|0 1 0|the path-only rule refused (the #3425 shape)
red|0 0 1|the package walk refused
env|2 0 0|a check that could not measure is not a pass
red|1 2 0|a defect found outranks a check that could not run
env|0 0 101|an unexpected exit code is not a pass
env|0 0|a missing check is not a pass
EOF
    got=$(echo '{"packages":[{"name":"z","publish":null},{"name":"a","publish":["crates-io"]},{"name":"m","publish":[]}]}' | rpg_publishable | tr '\n' ,)
    if [ "$got" = "a,z," ]; then echo "  ok   publishable: publish=[] dropped, the rest sorted"
    else echo "  FAIL publishable: wanted a,z, got $got"; fail=1; fi
    got=$(echo '{"packages":[{"name":"m","publish":[]}]}' | rpg_publishable | wc -l)
    if [ "$got" = 1 ]; then echo "  ok   publishable: none -> one empty line (package_walk reads it as ENV)"
    else echo "  FAIL publishable: none gave $got lines"; fail=1; fi
    # MECHANISM: a fake cargo records what package_walk asked for. --verify must drop
    # --no-verify (else the "dry-run" compiles nothing), and the default must keep it.
    local d; d=$(mktemp -d) || return 1
    # Every walk below runs against a CARGO_HOME inside $d: the self-test never reads or clears the
    # caller's cargo home, and does not depend on the caller's CARGO_HOME or HOME being absolute.
    local had_ch=${CARGO_HOME+x} old_ch=${CARGO_HOME:-}
    export CARGO_HOME="$d/ch"; mkdir -p "$CARGO_HOME"
    printf '%s\n' '#!/usr/bin/env bash' \
        'if [ "$1" = metadata ]; then echo "{\"target_directory\":\"$FAKE_TARGET\",\"packages\":[{\"name\":\"${FAKE_M:-a}\",\"version\":\"0.0.0\",\"publish\":null},{\"name\":\"b\",\"version\":\"0.0.0\",\"publish\":[]}]}"; exit 0; fi' \
        'fake_clean_pkg() {' \
        '  local f h prof=debug' \
        '  [ -z "$2" ] || prof=release' \
        '  for f in "$FAKE_TARGET/$prof"/.fingerprint/"$1"-* "$FAKE_TARGET/$prof"/build/"$1"-*; do' \
        '    [ -e "$f" ] || continue' \
        '    h=${f##*-}' \
        '    rm -rf "${f:?}" "${FAKE_TARGET:?}/$prof"/deps/*-"${h:?}".*' \
        '  done' \
        '}' \
        'if [ "$1" = clean ]; then' \
        '  case " $* " in *" --offline "*) echo "error: no matching package named x found (offline)" >&2; exit 101 ;; esac' \
        '  [ -z "${FAKE_CLEAN_RC:-}" ] || { echo "error: fake clean refusal" >&2; exit "$FAKE_CLEAN_RC"; }' \
        '  [ -z "${FAKE_CLEAN_NOOP:-}" ] || exit 0' \
        '  shift' \
        '  rel=; for a in "$@"; do [ "$a" != --release ] || rel=1; done' \
        '  while [ $# -gt 1 ]; do' \
        '    [ "$1" != -p ] || fake_clean_pkg "$2" "$rel"' \
        '    shift' \
        '  done' \
        '  exit 0' \
        'fi' \
        'echo "$*" >> "$FAKE_CARGO_LOG"' \
        'if [ -e "$FAKE_TARGET/package/tmp-registry/stale.crate" ]; then echo "error: no hash listed for a v0.0.0"; exit 101; fi' \
        'if [ -e "$CARGO_HOME/registry/src/-h/a-0.0.0/planted" ]; then echo "error: verified a leftover unpacked source"; exit 101; fi' \
        'if grep -rlqE --include="*.d" "/registry/src/-[^/]*/${FAKE_M:-a}-0\.0\.0/" "$FAKE_TARGET"; then echo "error: reused a compiled overlay unit"; exit 101; fi' \
        'echo "   Packaged 1 file"; [ -z "${FAKE_CARGO_RC:-}" ] || { echo "error: fake refusal"; exit "$FAKE_CARGO_RC"; }' > "$d/cargo"
    chmod +x "$d/cargo"; : > "$d/Cargo.toml"
    export FAKE_TARGET="$d/t"
    # #6878: an overlay tarball left by an earlier run must be gone before cargo runs (the fake
    # cargo refuses like the real one when it is still there); the build cache must survive.
    mkdir -p "$d/t/package/tmp-registry" "$d/t/debug"; : > "$d/t/package/tmp-registry/stale.crate"; : > "$d/t/debug/keep"
    if FAKE_CARGO_LOG=$d/args PATH="$d:$PATH" package_walk "$d" --verify > /dev/null \
        && [ ! -e "$d/t/package" ] && [ -e "$d/t/debug/keep" ]; then
        echo "  ok   the walk starts from an empty <target>/package and keeps the build cache (#6878)"
    else echo "  FAIL a stale overlay tarball survived into the walk, or the build cache did not"; fail=1; fi
    # #6880: an unpacked overlay source left in CARGO_HOME by an earlier commit must not be what
    # gets verified (the fake cargo refuses when it is still there). Only the member's overlay dir
    # goes: a sibling overlay crate that is not a publishable member and the real registry copy stay.
    mkdir -p "$CARGO_HOME/registry/src/-h/a-0.0.0" "$CARGO_HOME/registry/src/-h/zz-1.0.0" "$CARGO_HOME/registry/src/-h/b-0.0.0" "$CARGO_HOME/registry/src/index.crates.io-x/a-0.0.0"
    : > "$CARGO_HOME/registry/src/-h/a-0.0.0/planted"
    if FAKE_CARGO_LOG=$d/args PATH="$d:$PATH" package_walk "$d" --verify > /dev/null \
        && [ ! -e "$CARGO_HOME/registry/src/-h/a-0.0.0" ] && [ -d "$CARGO_HOME/registry/src/-h/zz-1.0.0" ] && [ -d "$CARGO_HOME/registry/src/-h/b-0.0.0" ] \
        && [ -d "$CARGO_HOME/registry/src/index.crates.io-x/a-0.0.0" ]; then
        echo "  ok   a leftover unpacked member source is removed before the walk; other sources stay (#6880)"
    else echo "  FAIL a leftover unpacked overlay source was verified, or an unrelated source was removed"; fail=1; fi
    # #6880 (compiled unit): a unit compiled from the overlay source by an earlier commit stays fresh
    # in a kept target dir (cargo keys a registry unit by id, not mtime). cargo's own clean drops the
    # member's units; the gate never guesses their directory names. One row per shape the first
    # attempt (dep-info stem as fingerprint name, deps/ only) left in place, each planted alone beside
    # an unrelated unit that must stay.
    st_unit() { # LABEL MEMBER FINGERPRINT-DIR DEP-INFO (the last two relative to <target>/debug)
        local label=$1 m=$2 fp=$3 dep=$4 t=$d/t/debug
        rm -rf -- "${d:?}/t/debug"
        mkdir -p "$t/deps" "$t/.fingerprint/$fp" "$t/.fingerprint/zz-9" "$(dirname -- "$t/$dep")"
        echo "x: $CARGO_HOME/registry/src/-h/$m-0.0.0/src/lib.rs" > "$t/$dep"
        echo "zz-9.d: $CARGO_HOME/registry/src/-h/zz-1.0.0/src/lib.rs" > "$t/deps/zz-9.d"
        if FAKE_M=$m FAKE_CARGO_LOG=$d/args PATH="$d:$PATH" package_walk "$d" --verify > /dev/null \
            && [ ! -e "$t/.fingerprint/$fp" ] && [ ! -e "$t/$dep" ] && [ -d "$t/.fingerprint/zz-9" ] && [ -e "$t/deps/zz-9.d" ]; then
            echo "  ok   compiled overlay unit: $label is invalidated; the unrelated unit stays (#6880)"
        else echo "  FAIL compiled overlay unit: $label survived into the walk, or an unrelated unit was removed"; fail=1; fi
    }
    st_unit "plain package name" a a-aa11 deps/a-aa11.d
    st_unit "hyphenated package (lib stem h_p)" h-p h-p-bb22 deps/h_p-bb22.d
    st_unit "custom [lib] name" c-q c-q-cc33 deps/provable_x-cc33.d
    st_unit "build-script unit" h-p h-p-dd44 build/h-p-dd44/build_script_build-dd44.d
    # Fail closed: a clean that removed nothing while a stale unit exists, and a clean that failed,
    # are ENV (rc 2), never a green walk.
    mkdir -p "$d/t/debug/deps"; echo "x: $CARGO_HOME/registry/src/-h/h-p-0.0.0/src/lib.rs" > "$d/t/debug/deps/h_p-ee55.d"
    got=$(FAKE_M=h-p FAKE_CLEAN_NOOP=1 FAKE_CARGO_LOG=$d/args PATH="$d:$PATH" package_walk "$d" --verify; echo "rc=$?")
    case "$got" in
        ENV*"rc=2") echo "  ok   a clean that removed nothing while a stale unit exists is ENV" ;;
        *) echo "  FAIL a no-op clean with a stale unit gave: $got"; fail=1 ;;
    esac
    got=$(FAKE_M=h-p FAKE_CLEAN_RC=101 FAKE_CARGO_LOG=$d/args PATH="$d:$PATH" package_walk "$d" --verify; echo "rc=$?")
    case "$got" in
        ENV*"rc=2") echo "  ok   a cargo clean that fails is ENV" ;;
        *) echo "  FAIL a failed cargo clean gave: $got"; fail=1 ;;
    esac
    case "$got" in
        *"fake clean refusal"*) echo "  ok   a failed cargo clean shows cargo's own stderr" ;;
        *) echo "  FAIL a failed cargo clean hid its stderr: $got"; fail=1 ;;
    esac
    # the clean must not run --offline (a cold cargo home or a new dependency needs the index): the
    # fake refuses --offline like cargo 1.93.0 did, so the plain walk above already goes red on it.
    # --release units: a stale unit under the release profile must go too.
    rm -rf -- "${d:?}/t/debug"; mkdir -p "$d/t/release/deps" "$d/t/release/.fingerprint/a-ff66"
    echo "x: $CARGO_HOME/registry/src/-h/a-0.0.0/src/lib.rs" > "$d/t/release/deps/a-ff66.d"
    if FAKE_CARGO_LOG=$d/args PATH="$d:$PATH" package_walk "$d" --verify > /dev/null \
        && [ ! -e "$d/t/release/.fingerprint/a-ff66" ] && [ ! -e "$d/t/release/deps/a-ff66.d" ]; then
        echo "  ok   a stale release-profile overlay unit is invalidated too (#6880)"
    else echo "  FAIL a stale release-profile overlay unit survived into the walk"; fail=1; fi
    # An unscannable or symlinked target is never "gone": the scan must fail closed on a directory
    # it cannot read, and must follow a symlinked subdirectory.
    rm -rf -- "${d:?}/t/release"
    rpg_overlay_units_gone "$d/t" a-0.0.0; got=$?
    if [ "$got" -eq 0 ]; then echo "  ok   scan: a clean target is gone"
    else echo "  FAIL scan: a clean target gave $got"; fail=1; fi
    mkdir -p "$d/real/deps"; echo "x: $CARGO_HOME/registry/src/-h/a-0.0.0/src/lib.rs" > "$d/real/deps/a-ff66.d"
    ln -s "$d/real" "$d/t/linked"
    rpg_overlay_units_gone "$d/t" a-0.0.0; got=$?
    if [ "$got" -eq 1 ]; then echo "  ok   scan: a stale unit behind a symlinked subdirectory is found"
    else echo "  FAIL scan: a stale unit behind a symlink gave $got"; fail=1; fi
    rm -f -- "${d:?}/t/linked"
    if [ "$(id -u)" -ne 0 ]; then
        mkdir -p "$d/t/locked"; chmod 000 "$d/t/locked"
        rpg_overlay_units_gone "$d/t" a-0.0.0; got=$?
        got2=$(FAKE_M=a FAKE_CARGO_LOG=$d/args PATH="$d:$PATH" package_walk "$d" --verify; echo "rc=$?")
        chmod 755 "$d/t/locked"; rmdir -- "${d:?}/t/locked"
        if [ "$got" -eq 1 ]; then echo "  ok   scan: an unreadable target subdirectory is not gone"
        else echo "  FAIL scan: an unreadable target subdirectory gave $got"; fail=1; fi
        case "$got2" in
            ENV*"rc=2") echo "  ok   an unscannable target is ENV in the walk" ;;
            *) echo "  FAIL an unscannable target gave: $got2"; fail=1 ;;
        esac
    fi
    rm -rf -- "${d:?}/t/debug"
    # The production case: CARGO_HOME unset or empty, so the cargo home is $HOME/.cargo. The
    # leftover there must go too, and nothing may be cleared straight under $HOME.
    mkdir -p "$d/home/.cargo/registry/src/-h/a-0.0.0" "$d/home/registry/src/-h/a-0.0.0"
    if CARGO_HOME='' HOME="$d/home" FAKE_CARGO_LOG=$d/args PATH="$d:$PATH" package_walk "$d" --verify > /dev/null \
        && [ ! -e "$d/home/.cargo/registry/src/-h/a-0.0.0" ] && [ -d "$d/home/registry/src/-h/a-0.0.0" ]; then
        echo "  ok   with CARGO_HOME empty the leftover under \$HOME/.cargo is removed, nothing else"
    else echo "  FAIL with CARGO_HOME empty the leftover under \$HOME/.cargo survived, or \$HOME itself was cleared"; fail=1; fi
    # No absolute cargo home is ENV, never a walk that cleared nothing: a relative CARGO_HOME, and
    # CARGO_HOME and HOME both empty (which used to give "/.cargo" and pass the check).
    got=$(CARGO_HOME=relative FAKE_CARGO_LOG=$d/args PATH="$d:$PATH" package_walk "$d"; echo "rc=$?")
    case "$got" in
        ENV*"rc=2") echo "  ok   a CARGO_HOME that is not absolute is ENV" ;;
        *) echo "  FAIL a relative CARGO_HOME gave: $got"; fail=1 ;;
    esac
    got=$(CARGO_HOME='' HOME='' FAKE_CARGO_LOG=$d/args PATH="$d:$PATH" package_walk "$d"; echo "rc=$?")
    case "$got" in
        ENV*"rc=2") echo "  ok   CARGO_HOME and HOME both empty is ENV, not a walk under /.cargo" ;;
        *) echo "  FAIL empty CARGO_HOME and HOME gave: $got"; fail=1 ;;
    esac
    got=$(FAKE_TARGET=relative FAKE_CARGO_LOG=$d/args PATH="$d:$PATH" package_walk "$d"; echo "rc=$?")
    case "$got" in
        ENV*"rc=2") echo "  ok   a target directory that is not absolute is ENV: nothing is emptied" ;;
        *) echo "  FAIL a relative target directory gave: $got"; fail=1 ;;
    esac
    got=$(FAKE_CARGO_RC=101 FAKE_CARGO_LOG=$d/args PATH="$d:$PATH" package_walk "$d" | sed -n 's/^ *full cargo log: //p')
    if [ -n "$got" ] && grep -q 'fake refusal' "$got"; then echo "  ok   a red walk keeps cargo's full log and names it"
    else echo "  FAIL a red walk left no cargo log (got: ${got:-nothing})"; fail=1; fi
    [ -z "$got" ] || rm -f -- "${got:?}"
    : > "$d/args"
    FAKE_CARGO_LOG=$d/args PATH="$d:$PATH" package_walk "$d" --verify > /dev/null
    if grep -qx 'package -p a --locked' "$d/args"; then echo "  ok   --verify packages the publishable member with verify on"
    else echo "  FAIL --verify ran: $(cat "$d/args")"; fail=1; fi
    : > "$d/args"; FAKE_CARGO_LOG=$d/args PATH="$d:$PATH" package_walk "$d" > /dev/null
    if grep -qx 'package -p a --no-verify --locked' "$d/args"; then echo "  ok   the default walk stays --no-verify (rc-cut cost)"
    else echo "  FAIL default ran: $(cat "$d/args")"; fail=1; fi
    got=$(FAKE_CLEAN_RC=101 FAKE_CARGO_LOG=$d/args PATH="$d:$PATH" package_walk "$d" > /dev/null; echo "rc=$?")
    if [ "$got" = rc=0 ]; then echo "  ok   the default walk does not run cargo clean (no overlay unit is compiled without --verify)"
    else echo "  FAIL the default walk ran cargo clean: $got"; fail=1; fi
    rm -rf -- "${d:?}"; unset FAKE_TARGET
    if [ -n "$had_ch" ]; then export CARGO_HOME="$old_ch"; else unset CARGO_HOME; fi
    # WIRING: the autopilot's tag step runs --verify on the release commit AHEAD of the tag: the
    # --verify line and its `die "publish dry-run` sit inside `if run_step tag`, before cut_tag.
    if wiring_ahead_of_tag "$HERE/autopilot.sh"; then echo "  ok   autopilot runs --verify ahead of the tag and stops on red"
    else echo "  FAIL autopilot.sh does not run rc_publish_gate.sh --verify with a die ahead of cut_tag"; fail=1; fi
    # The same predicate must refuse the old order (the dry run after the tag), or it proves nothing.
    d=$(mktemp -d) || return 2
    printf '%s\n' 'if run_step tag; then' '  cut_tag "$V" "$T" "$MC"' 'fi' 'if run_step dryrun; then' \
        '  bash scripts/release/rc_publish_gate.sh --verify "$WT" > x 2>&1; rc=$?' '  [ $rc -eq 0 ] || die "publish dry-run refused"' 'fi' > "$d/after.sh"
    if wiring_ahead_of_tag "$d/after.sh"; then echo "  FAIL the wiring check accepts a dry run after the tag"; fail=1
    else echo "  ok   the wiring check refuses a dry run after the tag"; fi
    rm -rf -- "${d:?}"
    if [ "$fail" -eq 0 ]; then echo "$PROG self-test: PASS"; return 0; fi
    echo "$PROG self-test: FAIL"; return 1
}

# package_walk ROOT [--verify] -> 0 every publishable member packaged, 1 cargo refused, 2 cannot measure
package_walk() {
    local root=$1 mode=--no-verify log n got rc=0 meta tdir
    [ "${2:-}" = --verify ] && mode=''
    local -a sel=()
    meta=$(cargo metadata --no-deps --offline --format-version 1 --manifest-path "$root/Cargo.toml" 2>/dev/null)
    mapfile -t sel < <(printf '%s\n' "$meta" | rpg_publishable)
    n=${#sel[@]}
    if [ "$n" -eq 0 ] || [ -z "${sel[0]}" ]; then
        echo "ENV   package: cargo metadata names no publishable member in $root"
        return 2
    fi
    local -a args=()
    local c
    for c in "${sel[@]}"; do args+=(-p "$c"); done
    # EMPTY OVERLAY (#6878). cargo writes the cascade's tarballs to <target>/package/tmp-registry
    # and rewrites a tarball that is already there IN PLACE, without truncating. A CI runner
    # exports a CARGO_TARGET_DIR that outlives the job, so a tarball that came out shorter than
    # the one an earlier run left kept the old tail: sha256(file) != the index cksum, and verify
    # died on the first dependent that unpacked it ("no hash listed for aprender-contracts").
    # Only <target>/package goes; the build cache beside it stays.
    tdir=$(printf '%s\n' "$meta" | rpg_target_dir)
    if [ -z "$tdir" ]; then
        echo "ENV   package: cargo metadata names no absolute target directory for $root"
        return 2
    fi
    if ! rm -rf -- "${tdir:?}/package"; then
        echo "ENV   package: cannot empty ${tdir}/package"
        return 2
    fi
    # LEFTOVER UNPACKED SOURCE (#6880). cargo does not unpack an overlay crate again when
    # <CARGO_HOME>/registry/src/-<hash>/<crate>-<version> exists, so a CARGO_HOME that outlives the
    # job (the nightly walks one version until the next bump) verifies the source an earlier commit
    # left there. Remove the members' own overlay dirs only (glob `-*` = the local overlay, never
    # index.crates.io-*); a failed removal is ENV.
    local ch=${CARGO_HOME:-${HOME:+${HOME}/.cargo}} md m
    if [ "${ch#/}" = "$ch" ]; then
        echo "ENV   package: no absolute CARGO_HOME to clear leftover overlay sources from"
        return 2
    fi
    while IFS= read -r m; do
        [ -n "$m" ] || continue
        for md in "$ch"/registry/src/-*/"$m"; do
            [ -d "$md" ] || continue
            rm -rf -- "${md:?}" || { echo "ENV   package: cannot remove leftover $md"; return 2; }
        done
    done < <(printf '%s\n' "$meta" | rpg_member_dirs)
    # COMPILED OVERLAY UNIT (#6880, measured on cargo 1.93.0). cargo fingerprints a registry-source
    # unit by package id and version only, with no mtime check, so the unit an earlier commit compiled
    # from the overlay source stays "fresh" in a kept CARGO_TARGET_DIR although the source dir above
    # was unpacked anew: verify linked the OLD sibling API (rc 0 for a broken API, false red for a new
    # one). cargo drops a member's lib, build-script and run units itself (`cargo clean -p`; the
    # version qualifier is ignored, so the workspace copy goes too), dev and release. We do not guess
    # its directory names: afterwards any dep-info that still names a member overlay source (another
    # profile or triple, a clean that removed nothing) is ENV, never green.
    # --verify only: a --no-verify walk compiles no overlay source, so it has no stale unit to
    # reuse and must not pay for a rebuild of every member or an online lockfile resolve.
    if [ -z "$mode" ]; then
        local mds=()
        mapfile -t mds < <(printf '%s\n' "$meta" | rpg_member_dirs)
        # No --offline: `cargo clean -p` resolves the whole lockfile, so a cold CARGO_HOME or a new
        # dependency needs the index (cargo package below is online too). Its stderr is kept.
        local clog; clog=$(mktemp) || return 2
        if ! (cd -- "$root" && cargo clean "${args[@]}" --locked && cargo clean "${args[@]}" --release --locked) > "$clog" 2>&1; then
            echo "ENV   package: cargo clean could not drop the compiled overlay units of the members:"
            tail -n 8 "$clog" | sed "s/^/        /"
            rm -f -- "${clog:?}"; return 2
        fi
        rm -f -- "${clog:?}"
        if ! rpg_overlay_units_gone "$tdir" "${mds[@]}"; then
            echo "ENV   package: a compiled unit under $tdir still reads a member overlay source after cargo clean"
            return 2
        fi
    fi
    log=$(mktemp) || return 2
    (cd -- "$root" && cargo package "${args[@]}" ${mode:+"$mode"} --locked) > "$log" 2>&1 || rc=$?
    got=$(grep -cE '^ +Packaged ' "$log")
    if [ "$rc" -eq 0 ] && [ "$got" -eq "$n" ]; then
        echo "ok    package: $got/$n publishable members packaged (cargo package ${mode:-with verify} --locked)"
        rm -f -- "$log"; return 0
    fi
    echo "FAIL  package: cargo package exited $rc with $got/$n members packaged:"
    # the crate cargo was on when it stopped: verify errors name the tarball, not the member
    grep -E '^ +(Packaging|Verifying) ' "$log" | tail -n 1 | sed 's/^ */        last: /'
    grep -E '^(error|Caused by)' -A 4 "$log" | head -n 20 | sed 's/^/        /'
    # a red keeps what cargo said: the 20 lines above did not show the cause of #6878
    echo "        full cargo log: $log"
    return 1
}

main() {
    case "${1:-}" in
        --self-test) self_test; return ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; return 0 ;;
        '') echo "$PROG: usage: [--verify] ROOT | --self-test" >&2; return 2 ;;
    esac
    local verify='' g p k verdict root
    if [ "$1" = --verify ]; then verify=--verify; shift; fi
    root=${1:?usage: $PROG [--verify] ROOT}
    [ -f "$root/Cargo.toml" ] || { echo "$PROG: $root has no Cargo.toml" >&2; return 2; }
    root="$(cd -- "$root" && pwd)"
    echo "=== $PROG: can $root reach crates.io? ==="
    PUBLISH_PREFLIGHT_ROOT="$root" bash "$SCRIPTS/check_publish_preflight.sh" --graph-only; g=$?
    bash "$SCRIPTS/check_pathonly_devdeps_unused_in_src.sh" --rc-gate "$root"; p=$?
    package_walk "$root" $verify; k=$?
    verdict=$(rpg_verdict "$g" "$p" "$k")
    echo "$PROG: graph=$g pathonly=$p package=$k -> $verdict"
    case "$verdict" in
        green) return 0 ;;
        red) return 1 ;;
        *) return 2 ;;
    esac
}

main "$@"
