#!/usr/bin/env bash
# pre_push_local.sh -- run, before an epic push, the checks CI's x86-main would run on it.
#
# ONE script, bash only, in this order:
#   1 preflight    every file the later steps need is present (a missing one is rc 127 later, say so now)
#   2 guard-tree   scripts/ci_guards.sh guard-tree
#   3 guard-cargo  scripts/ci_guards.sh guard-cargo
#   4 vendored-schemas  scripts/check_vendored_schemas.sh
#   5 quick tier   nextest over the crates the diff against the base touches (CI's quick_tier_groups split)
#   6 ont witness  pv-sat rewrites contracts/witness; a witness the tree does not carry is RED: commit it
# Every step runs after a red one (not after a red preflight: it stops the run, FAIL); the summary lists each with its seconds. Exit 0 only if
# every step passed, 1 if any failed, 3 on caller error. A step that cannot run here is FAIL, never SKIP,
# except quick tier with no touched crate (CI skips it the same way) and the explicit QUICK=0.
#
# USAGE  bash scripts/pre_push_local.sh [--base REF] [--only 'a|b'] [--no-quick]
#   --base REF   diff base for the quick tier crate selection (default origin/main)
#   --only RE    run only steps whose name matches RE (names: preflight guard-tree guard-cargo
#                vendored-schemas quick-tier ont-witness)
#   --prove      prove the harness: a red step is FAIL, a failed quick-tier selection is rc 1
#   CRATES="a b" env overrides the quick tier selection (package names)
#   TARGET_DIR   cargo target dir (default /mnt/nvme-raid0/targets/pre-push-<worktree name>)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 3
base=origin/main; only=""; quick=1; selftest=0
while [ $# -gt 0 ]; do
  case "$1" in
    --base) base="${2:?--base needs a ref}"; shift 2 ;;
    --only) only="${2:?--only needs a regex}"; shift 2 ;;
    --no-quick) quick=0; shift ;;
    --prove) selftest=1; shift ;;
    -h | --help) sed -n 2,22p "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "pre_push_local: unknown arg $1" >&2; exit 3 ;;
  esac
done
TMPD="$(mktemp -d)" || exit 3; trap 'rm -rf "${TMPD:?}"' EXIT   # every scratch file lives here, removed on exit
NVME="${PREPUSH_NVME:-/mnt/nvme-raid0}"   # the one fast-disk root (cache, tool pins, target dirs)
PINNED_TOOLS_DIR="${PINNED_TOOLS_DIR:-$HOME/.local/pinned-tools}"
export CARGO_TARGET_DIR="${TARGET_DIR:-$NVME/targets/pre-push-$(basename "$ROOT")}"

names=(); results=(); secs=()
want() { [ -z "$only" ] || [[ "$1" =~ $only ]]; }
run_step() { # name fn
  want "$1" || return 0
  local t0 rc; t0=$SECONDS; echo "=== $1"
  "$2"; rc=$?
  names+=("$1"); results+=("$rc"); secs+=("$(( SECONDS - t0 ))")
  echo "=== $1 rc=$rc"
}

# make_clone SRC DEST: a private shallow clone of SRC with this tree's HEAD checked out; never touches ROOT's gitdir.
make_clone() {
  rm -rf "${2:?}"
  git clone -q --depth=50 --no-checkout "$1" "$2" || return 1
  git -C "$2" fetch -q --depth=50 "$ROOT" HEAD || return 1
  git -C "$2" checkout -q --detach FETCH_HEAD || return 1
}

# --prove: the harness must turn a red step into FAIL and a failed selection into rc 1 (a pass case and a fail case each).
run_selftest() {
  local bad=0 meta sel
  ok() { [ "$2" = "$3" ] && echo "self-test PASS: $1" || { echo "self-test FAIL: $1 (got $2, want $3)"; bad=1; }; }
  names=(); results=(); secs=(); only=''
  run_step st-pass true > /dev/null; run_step st-fail false > /dev/null
  ok "run_step records a passing step as 0" "${results[0]}" 0
  ok "run_step records a failing step as 1" "${results[1]}" 1
  meta="$TMPD/pv-meta"; sel="$TMPD/pv-sel"
  cargo metadata --no-deps --format-version 1 > "$meta" || { echo "self-test FAIL: cargo metadata"; return 1; }
  select_crates "$meta" HEAD "$sel" > /dev/null 2>&1; ok "select_crates passes on an empty diff" "$?" 0
  ok "an empty diff selects nothing" "$(wc -c < "$sel" | tr -d ' ')" 0
  select_crates "$meta" "no-such-ref-$$" "$sel" > /dev/null 2>&1; ok "select_crates fails on an unresolvable base" "$?" 1
  # clone isolation: building the private clone and deepening it must leave the shared gitdir byte-identical
  local gd before after cl="$TMPD/iso-clone"
  gd="$(git rev-parse --git-common-dir)"
  before="$({ cat "$gd/shallow" 2> /dev/null; git for-each-ref; } | sha256sum)"
  make_clone "file://$ROOT" "$cl" > /dev/null 2>&1; ok "make_clone builds a private clone" "$?" 0
  git -C "$cl" fetch -q --deepen=5 "file://$ROOT" > /dev/null 2>&1
  after="$({ cat "$gd/shallow" 2> /dev/null; git for-each-ref; } | sha256sum)"
  ok "clone + deepen leave the shared gitdir (shallow, refs) unchanged" "$after" "$before"
  [ "$(git -C "$cl" rev-parse --absolute-git-dir)" != "$(git rev-parse --absolute-git-dir)" ]; ok "the clone has its own gitdir" "$?" 0
  return "$bad"
}

step_preflight() {
  local f miss=0
  for f in scripts/ci_guards.sh scripts/guard_tree.sh scripts/tests/guard_tree_test.sh \
    scripts/check_vendored_schemas.sh scripts/contracts_gate.sh scripts/pv_bin.sh \
    scripts/ci/quick_tier_groups.sh scripts/lib/ci_guard_steps.sh; do
    [ -f "$f" ] || { echo "preflight: missing $f (is this checkout current?)"; miss=1; }
  done
  # The runner image carries these on PATH; a bare shell (fat_driver, ssh, cron) often does not, and the
  # first guard that needs one dies rc 127. Put the usual homes on PATH, then name every missing tool here.
  for f in "$HOME/.cargo/bin" "$HOME/.local/bin" /usr/local/bin; do
    if [ -d "$f" ]; then case ":$PATH:" in *":$f:"*) ;; *) PATH="$f:$PATH" ;; esac; fi
  done
  export PATH
  # The runner has the tools.toml pins; a dev box is usually ahead (lambda: pmat 3.42.0, bashrs 7.4.2 vs 3.41.1 /
  # 7.4.1 -> check_tool_versions FAIL). Find a binary AT the pin, link it into a private dir, put that first.
  local pinbin="$CARGO_TARGET_DIR/pinned-bin" tool want have c
  mkdir -p "$pinbin"
  for tool in pmat bashrs; do
    want="$(awk -v t="[$tool]" '$0==t{s=1;next} /^\[/{s=0} s&&/^version/{gsub(/[" ]/,"",$0);sub(/version=/,"");print;exit}' tools.toml)"
    [ -n "$want" ] || { echo "preflight: no [$tool] version in tools.toml"; miss=1; continue; }
    have=""
    for c in "$PINNED_TOOLS_DIR/$tool-$want/bin/$tool" "$HOME/.local/$tool/$want/bin/$tool" \
      "$HOME/.forjar/cache/cargo/$tool-$want-$(uname -m)/bin/$tool" "$NVME/build/aprender/tools/bin/$tool" \
      "$NVME/tmp/tool-pins/$tool-$want/bin/$tool" "$(command -v "$tool" || true)"; do
      [ -x "$c" ] && [ "$(timeout 20 "$c" --version 2> /dev/null | head -1 | awk '{print $NF}')" = "$want" ] && { have="$c"; break; }
    done
    if [ -n "$have" ]; then ln -sf "$have" "$pinbin/$tool"; echo "preflight: $tool pinned $want -> $have"
    else echo "preflight: MISSING $tool $want (tools.toml pin); none of the known pin dirs holds it. Install it into \$PINNED_TOOLS_DIR/$tool-$want/bin (fleet forjar stack-tool-$tool)"; miss=1; fi
  done
  PATH="$pinbin:$PATH"; export PATH
  for f in git cargo rustc jq bashrs pmat pv pv-sat shellcheck check-jsonschema python3 perl awk sed xargs setsid unshare docker timeout; do
    if command -v "$f" > /dev/null 2>&1; then printf 'preflight: %-17s %s\n' "$f" "$(command -v "$f")"
    else echo "preflight: MISSING tool $f (not on PATH=$PATH)"; miss=1; fi
  done
  cargo nextest --version > /dev/null 2>&1 || { echo "preflight: cargo-nextest missing"; miss=1; }
  git rev-parse --verify -q "$base" > /dev/null || { echo "preflight: base $base not found (git fetch origin main)"; miss=1; }
  return "$miss"
}
# Cache key: sha(Cargo.lock + the pinned toolchain's rustc -vV). Target dir and CARGO_HOME are keyed by it and shared by
# every worktree, so a second worktree on the same lock starts warm instead of cold (#5983).
cache_key() { { cat "$ROOT/Cargo.lock"; rustc -vV; } | sha256sum | cut -c1-12; }
PREPUSH_CACHE="${PREPUSH_CACHE:-$NVME/targets/pre-push-cache}"

step_guard_tree() {
  # guard-tree judges the COMMITTED HEAD in a private clone: pin_build_base moves origin/main and a shallow checkout needs
  # a deepen, and neither may write into the shared gitdir (.git/shallow / refs of every worktree).
  local gt="$CARGO_TARGET_DIR/gt-clone"
  make_clone "$(git remote get-url origin)" "$gt" || return 1
  cd "$gt" || return 1
  echo "guard-tree: private clone $gt at $(git rev-parse --short HEAD) (committed state; uncommitted edits are not judged)"
  bash scripts/ci/pin_build_base.sh || return 1
  git merge-base origin/main HEAD > /dev/null 2>&1 || git fetch -q --deepen=400 origin main || return 1
  bash scripts/ci_guards.sh guard-tree
}

step_guard_cargo() {
  # m5 runs cargo in a container that mounts ~/.cargo, then compares with a host build; so the step runs in the runner's
  # build env (no CARGO_TARGET_DIR / CARGO_INCREMENTAL / CARGO_PROFILE_* / RUSTFLAGS: a dev shell sets them and m5 then
  # sees 157 units rebuilt instead of 12) and builds into <repo>/target. That target is seeded from, and saved back to,
  # the keyed cache; CARGO_HOME is the keyed cache's (a ~/.cargo/registry symlink to /dev/shm is invisible to the container).
  local key cache t0 dl_s rc seed_s run_s units comp_s
  key="$(cache_key)" || return 1
  cache="$PREPUSH_CACHE/$key"; mkdir -p "$cache/target" "$cache/cargo-home" "$ROOT/target"
  echo "guard-cargo: cache key $key ($cache)"
  t0=$SECONDS
  if [ -f "$cache/target/.warm" ]; then
    cp -a --reflink=auto -n "$cache/target/." "$ROOT/target/" || echo "guard-cargo: seed copy failed; building cold"; echo "guard-cargo: target seeded from cache"
  else echo "guard-cargo: cold cache for key $key"; fi
  seed_s=$((SECONDS - t0)); t0=$SECONDS
  (export CARGO_HOME="$cache/cargo-home"; unset CARGO_TARGET_DIR CARGO_INCREMENTAL CARGO_BUILD_STRATEGY RUSTFLAGS $(compgen -e | grep "^CARGO_PROFILE_"); cargo fetch --locked > /dev/null 2>&1)
  dl_s=$((SECONDS - t0)); t0=$SECONDS
  (export CARGO_HOME="$cache/cargo-home"; unset CARGO_TARGET_DIR CARGO_INCREMENTAL CARGO_BUILD_STRATEGY RUSTFLAGS $(compgen -e | grep "^CARGO_PROFILE_"); bash scripts/ci_guards.sh guard-cargo)
  rc=$?
  run_s=$((SECONDS - t0))
  units="$(sed -n 's/^host cargo build after the container build: \([0-9]*\) compile commands.*/\1/p' target/ci-guards-local/guard-cargo.m5.log 2> /dev/null | tail -1)"
  comp_s="$(sed -n 's/^host cargo build after the container build: [0-9]* compile commands, \([0-9]*\) s.*/\1/p' target/ci-guards-local/guard-cargo.m5.log 2> /dev/null | tail -1)"
  echo "guard-cargo split: seed ${seed_s}s, download(cargo fetch) ${dl_s}s, guards ${run_s}s of which m5 host compile ${comp_s:-n/a}s for ${units:-n/a} units (CI shape: 12), rc=$rc"
  if [ "$rc" = 0 ]; then
    t0=$SECONDS; rm -f "${cache:?}/target/.warm"
    if cp -a --reflink=auto "$ROOT/target/." "$cache/target/"; then
      rm -rf "${cache:?}/target/ci-guards-local"; touch "$cache/target/.warm"; echo "guard-cargo: cache saved ($((SECONDS - t0))s)"
    else echo "guard-cargo: cache save FAILED; cache left cold"; fi
  fi
  return "$rc"
}
step_vendored() { bash scripts/check_vendored_schemas.sh; }

# select_crates META BASE OUT: write the workspace members the diff BASE..HEAD touches. Every stage is checked: a failed
# merge-base / diff / jq is rc 1 (never an empty selection that the caller would report as "skipped").
select_crates() {
  local meta="$1" b="$2" out="$3" mb files f
  mb="$(git merge-base "$b" HEAD 2> /dev/null)" || { mb="$b"; echo "quick tier: no merge-base with $b (shallow history); diffing against $b itself"; }
  files="$TMPD/files"
  git diff --name-only "$mb" HEAD > "$files" || { echo "quick tier: git diff $mb HEAD failed"; return 1; }
  : > "$out"
  while IFS= read -r f; do
    # a touched file under a member's directory selects that member
    jq -r --arg f "$ROOT/$f" '.packages[] | (.manifest_path | sub("/Cargo.toml$"; "")) as $d
      | select($f | startswith($d + "/")) | .name' "$meta" >> "$out" || { echo "quick tier: jq failed on $f"; return 1; }
  done < "$files"
  sort -u -o "$out" "$out" || return 1
}

step_quick() {
  [ "$quick" = 1 ] || { echo "quick tier: disabled by --no-quick"; return 0; }
  local meta calls sel
  meta="$TMPD/meta"; calls="$TMPD/calls"; sel="$TMPD/sel"
  cargo metadata --no-deps --format-version 1 > "$meta" || { echo "quick tier: cargo metadata failed"; return 1; }
  if [ -n "${CRATES:-}" ]; then
    tr ' ' '\n' <<< "$CRATES" | sed '/^$/d' > "$sel"; [ "${PIPESTATUS[0]}${PIPESTATUS[1]}" = 00 ] || { echo "quick tier: CRATES selection failed"; return 1; }
  else
    select_crates "$meta" "$base" "$sel" || return 1
  fi
  if [ ! -s "$sel" ]; then
    echo "quick tier: no crate selected by the diff against $base -- skipped, as CI skips it"
    return 0
  fi
  echo "quick tier crates: $(tr '\n' ' ' < "$sel")"
  mapfile -t crates < "$sel"
  bash scripts/ci/quick_tier_groups.sh --metadata "$meta" "${crates[@]}" > "$calls" || return 1
  [ -s "$calls" ] || { echo "quick tier: crates selected but the split printed no nextest call"; return 1; }
  local rc=0 line pkgs c
  while IFS= read -r line; do
    pkgs=(); for c in $line; do pkgs+=(-p "$c"); done
    echo "nextest call: ${pkgs[*]}"
    cargo nextest run --profile ci --no-fail-fast "${pkgs[@]}" --lib --tests || rc=1
  done < "$calls"
  return "$rc"
}

step_witness() {
  bash scripts/contracts_gate.sh consistency
}

if [ "$selftest" = 1 ]; then run_selftest; exit $?; fi

run_step preflight step_preflight
if [ "${results[0]:-0}" != 0 ] && want preflight; then
  echo "pre_push_local: preflight failed, every later step is skipped and the run is FAIL (they would only report rc 127)"
else
  # guard-tree (no cargo) runs beside guard-cargo (container + cargo build): independent, so wall time is the max, not the sum (#5983)
  rm -rf "${ROOT:?}/target/ci-guards-local"; mkdir -p "$ROOT/target"
  gt_log="$TMPD/gt.log"; gt_t0=$SECONDS
  if want guard-tree; then ( step_guard_tree > "$gt_log" 2>&1; echo $? > "$gt_log.rc" ) & gt_pid=$!; fi
  run_step guard-cargo step_guard_cargo
  if [ -n "${gt_pid:-}" ]; then
    wait "$gt_pid"; echo "=== guard-tree (ran in parallel with guard-cargo)"; tail -n 40 "$gt_log"
    gt_rc="$(cat "$gt_log.rc" 2> /dev/null || echo 1)"
    names+=(guard-tree); results+=("$gt_rc"); secs+=("$((SECONDS - gt_t0))"); echo "=== guard-tree rc=$gt_rc"
    rm -f "${gt_log:?}" "${gt_log:?}.rc"
  fi
  run_step vendored-schemas step_vendored
  run_step quick-tier step_quick
  run_step ont-witness step_witness
fi

echo; echo "pre_push_local summary (base $base, head $(git rev-parse --short HEAD)):"
fail=0
for i in "${!names[@]}"; do
  [ "${results[$i]}" = 0 ] && v=PASS || { v=FAIL; fail=1; }
  printf '  %-18s %s  %ss\n' "${names[$i]}" "$v" "${secs[$i]}"
done
[ "${#names[@]}" -gt 0 ] || { echo "  nothing ran"; exit 3; }
exit "$fail"
