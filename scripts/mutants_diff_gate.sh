#!/usr/bin/env bash
# mutants_diff_gate.sh -- the PR mutation gate (#4142): mutate the PR diff across the WORKSPACE and judge it, and
# never pass on anything it did not judge.
#
# WHY. Until #4142 the `mutants` job ran `cargo mutants --in-diff pr.diff -- --lib` at the workspace ROOT without
# `--workspace`. cargo-mutants then mutates only the package in the cwd, the near-empty root facade, so every diff
# under crates/** listed ZERO mutants and the job printed "0 mutants in diff. Pass." (measured on #3772, #3737,
# #3707 and #3706). The blocking gate had never mutated a line of crate code. `.cargo/mutants.toml` cannot fix
# that: cargo-mutants 27 refuses `workspace = true` ("unknown field"). The flag is the only way.
#
# RULES, in order (operator ruling via the cop, 2026-09-24: "go with recommended"):
#   1. LIST with --workspace. A list that fails is RED: the gate cannot say what it would judge.
#   2. 0 mutants listed -> pass, saying so. (A diff of tests, docs or data has nothing to mutate.)
#   3. more than --cap mutants -> RED "too many to judge": the job's hour holds about 60 at -j 4. Split the PR.
#      The nightly run (mutants-nightly.yml, --cap 0 = no cap) judges main's day of merges in full.
#   4. RUN with --workspace -j N (copies, not --in-place). No outcomes.json is RED, whatever the exit code: the
#      list said there were mutants, so a run that wrote nothing died.
#   5. An unparseable outcomes.json is RED (not being able to measure is not measuring zero), and so is a
#      TESTED count that differs from the LISTED count.
#   6. missed + timeout > --max-missed is RED, naming each survivor.
#   0. (operator 2026-09-27, verbatim "exempt the car") a release car -- head ref car/* pushed to THIS repo, not a
#      fork -- is not judged here: `--exempt-ref REF HEAD_REPO BASE_REPO` exits 0 and says so, and mutants-nightly.yml
#      (uncapped) judges the car's content once it lands on main. Every other PR keeps the cap. The repo check is
#      load-bearing: a fork chooses its own branch name and could call it car/anything.
#   --exclude-crate NAME (repeatable) keeps crates/NAME/** out of mutation, for the crates CI's workspace-test also
#   excludes (aprender-gpu, aprender-cuda-edge, aprender-compute: hardware, or a harness that segfaults at exit on a
#   clean pass, so cargo-mutants would read its baseline as failed). The exclusion is printed, never silent.
#
#   7. (#4621, operator E1) a survivor in a file the build never compiled is NOT_MEASURED: RED, or with
#      --defer-unbuilt FILE written sorted to FILE for a feature shard (the section passes; the `gate` job is RED
#      unless that shard measured exactly this set in the same run). No local receipt counts.
#   Feature shard (mutants-cuda): --features PKG/FEAT[,..] --gated-only mutates only the diff's files that the
#   feature build compiles and the default build does not (both MEASURED by dep-info) and writes that list, sorted,
#   to <out>/measured.txt.
#
#   bash scripts/mutants_diff_gate.sh <diff> [--cap N] [--jobs J] [--max-missed M] [--out DIR] [--exclude-crate NAME]...
#   bash scripts/mutants_diff_gate.sh --exempt-ref <head-ref> <head-repo> <base-repo>   # exit 0 exempt . 1 judge it
#        [--timeout S] [--defer-unbuilt FILE] [--features PKG/FEAT[,..] [--gated-only]] [--test-names FILE]
#   --test-names FILE runs only the lib tests named in FILE (--exact, one thread): the shard's cuda-only set. A
#   smaller test set can only leave MORE survivors, so it never makes the gate easier to pass.
#   bash scripts/mutants_diff_gate.sh --self-test
# exit 0 judged and within --max-missed . 1 RED . 2 usage
# Seam (the self-test only): MUTANTS_GATE_CARGO replaces `cargo`.
set -uo pipefail
TIMEOUT=300 DEFER="" FEATURES="" GATED_ONLY=0 NAMES=""

gate() { # gate <diff> <cap> <jobs> <max-missed> <out> [excluded crate...]
  local diff=$1 cap=$2 jobs=$3 max=$4 out=$5 cargo=${MUTANTS_GATE_CARGO:-cargo} n rc oc c
  shift 5
  local -a ex=()
  for c in "$@"; do ex+=(--exclude "crates/$c/**"); done
  # written first: an early pass defers nothing, and says so. Its directory may not exist yet: CI hands
  # mutants-gate/not_measured.txt to a fresh checkout, and the redirect failed before --out was made (#4620 x86-main)
  [ -z "$DEFER" ] || { mkdir -p "$(dirname "$DEFER")" && : > "$DEFER"; } || return 1
  [ -s "$diff" ] || { echo "ok    empty diff: nothing to mutate"; return 0; }
  mkdir -p "$out" || return 1
  local -a ta=(--lib)
  # Under --features, --lib must reach the BUILD as well. cargo-mutants' baseline is `cargo test --no-run`,
  # which takes only --cargo-arg; without it every feature-gated integration target is compiled, and one
  # that does not build fails the baseline (#4621, yoga run 36553822552). Passed once as --cargo-arg it
  # also reaches `cargo test`, so the test args drop theirs: a second --lib is a cargo usage error and
  # every mutant "fails" before a test runs (#4621, yoga run 36547274574).
  [ -z "$FEATURES" ] || { ex+=(--features "$FEATURES" --cargo-arg --lib); ta=(); echo "features: $FEATURES"; }
  if [ -n "$NAMES" ]; then
    [ -s "$NAMES" ] || { echo "RED   --test-names $NAMES is empty: a run that selects no test catches nothing"; return 1; }
    mapfile -t c < "$NAMES"
    ta+=(-- --exact --test-threads 1 "${c[@]}")
    echo "tests: the ${#c[@]} named in $NAMES, --exact, one thread"
  fi
  if [ "$GATED_ONLY" = 1 ]; then
    : > "$out/measured.txt" || return 1
    gated_files "$diff" "$out" || return 1
    if [ "${#GATED[@]}" -eq 0 ]; then
      echo "ok    0 feature-gated files in the diff: measured=0"
      return 0
    fi
    echo "gated files: ${GATED[*]}"
    for c in "${GATED[@]}"; do ex+=(--file "$c"); done
  fi
  [ "$#" -gt 0 ] && echo "excluded from mutation (judged by their own CI steps): $*"
  "$cargo" mutants --workspace "${ex[@]}" --in-diff "$diff" --list > "$out/list.txt" 2> "$out/list.err"; rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "RED   cargo mutants --list exited $rc: the gate cannot say what it would judge -- $(tail -1 "$out/list.err")"
    return 1
  fi
  n=$(grep -c . "$out/list.txt")
  if [ "$GATED_ONLY" = 1 ]; then LC_ALL=C sort "$out/list.txt" > "$out/measured.txt" || return 1; fi
  if [ "$n" -eq 0 ]; then
    echo "ok    0 mutants in the diff, listed across the workspace"
    return 0
  fi
  if [ "$cap" -gt 0 ] && [ "$n" -gt "$cap" ]; then
    echo "RED   too many to judge: $n mutants in the diff > cap $cap -- split the PR (mutants-nightly judges main in full)"
    return 1
  fi
  echo "mutating $n mutant(s) across the workspace, -j $jobs"
  "$cargo" mutants --workspace "${ex[@]}" --no-times --timeout "$TIMEOUT" -j "$jobs" --in-diff "$diff" --output "$out" -- "${ta[@]}"; rc=$?
  oc="$out/mutants.out/outcomes.json"
  if [ ! -f "$oc" ]; then
    echo "RED   cargo mutants exited $rc and wrote no outcomes.json although $n mutant(s) were listed: the run died"
    return 1
  fi
  judge "$oc" "$n" "$max" "$out"
}

exempt_ref() { # exempt_ref <head-ref> <head-repo> <base-repo> -- 0 iff a car of THIS repo (rule 0)
  [ -n "$2" ] && [ "$2" = "$3" ] || return 1
  [[ $1 == car/?* ]] || return 1
  echo "car/*: deferred to mutants-nightly (uncapped, operator 2026-09-27)"
}

field() { # field <json> <key> -> the first `"key": <int>` (top-level summary keys; outcome entries carry none)
  grep -oE "\"$2\": ?[0-9]+" "$1" | head -1 | grep -oE '[0-9]+$'
}

judge() { # judge <outcomes.json> <listed> <max-missed> <out>
  local oc=$1 n=$2 max=$3 out=$4 total missed timeout
  total=$(field "$oc" total_mutants); missed=$(field "$oc" missed); timeout=$(field "$oc" timeout)
  if [ -z "$total" ] || [ -z "$missed" ] || [ -z "$timeout" ]; then
    echo "RED   cannot parse total_mutants/missed/timeout out of $oc -- did cargo-mutants change its JSON shape?"
    return 1
  fi
  if [ "$total" -ne "$n" ]; then
    echo "RED   cargo mutants tested $total mutant(s) but listed $n: what was judged is not what the diff holds"
    return 1
  fi
  echo "judged $total mutant(s): missed=$missed timeout=$timeout (max allowed $max)"
  if [ $((missed + timeout)) -gt "$max" ]; then
    unbuilt_split "$out" || return 1
    if [ "$((missed + timeout - UNBUILT_N))" -gt "$max" ] || [ "$UNBUILT_N" -eq 0 ]; then
      echo "RED   $((missed + timeout)) mutant(s) survived or timed out on the diff (> $max): add tests that kill them"
      cat "$out/mutants.out/missed.txt" "$out/mutants.out/timeout.txt" 2> /dev/null | sed 's/^/  /'
      return 1
    fi
    if [ -n "$DEFER" ]; then
      printf '%s' "$UNBUILT" | LC_ALL=C sort > "$DEFER" || return 1
      echo "DEFERRED $UNBUILT_N NOT_MEASURED mutant(s) to the feature shard ($DEFER); the gate job is RED unless it measures this set"
      sed 's/^/  NOT_MEASURED  /' "$DEFER"
      return 0
    fi
    not_measured "$out" || return 1
  fi
  echo "ok    every mutant in the diff was caught (or within --max-missed)"
}

# #4621 (cop ruling 2026-09-29 06:19Z). cargo-mutants tests a mutant with `cargo test -p <pkg> --lib` at the
# package's DEFAULT features, so a mutant inside `#[cfg(feature = "cuda")]` code is never compiled: its run is the
# baseline, and it comes back MISSED or, on a slow suite, TIMEOUT (PR #4620: all 7 mutants were in cfg(cuda) code,
# 0 `cuda::` tests ran of 16149, 1 missed at 292 s and 6 timed out at 300.1 s). Such a mutant was never measured, so it
# is neither a survivor nor a pass. The compiled set is MEASURED, not guessed from cfg attributes: `cargo check -p
# <pkg> --lib` in a private target dir writes dep-info (*.d) naming every source file the compiler read.
unbuilt_split() { # unbuilt_split <out> -> UNBUILT (lines) and UNBUILT_N; RED (1) when it cannot measure
  local out=$1 cargo=${MUTANTS_GATE_CARGO:-cargo} f d pkg line
  local -A pkgs=() built=()
  UNBUILT=""; UNBUILT_N=0
  while IFS= read -r line; do
    f=${line%%:*}
    case "$f" in crates/*/*) d=${f#crates/}; d=crates/${d%%/*} ;; *) d=. ;; esac
    pkg=$(sed -n 's/^name *= *"\(.*\)"/\1/p' "$d/Cargo.toml" 2> /dev/null | head -1)
    [ -n "$pkg" ] || { echo "RED   cannot name the package that owns $f: the gate cannot tell whether it was compiled"; return 1; }
    pkgs[$pkg]=1
  done < <(cat "$out/mutants.out/missed.txt" "$out/mutants.out/timeout.txt" 2> /dev/null)
  [ "${#pkgs[@]}" -gt 0 ] || return 0   # counts without named survivors: judged as survivors, never as unbuilt
  for pkg in "${!pkgs[@]}"; do
    "$cargo" check -p "$pkg" --lib $(pkg_features "$pkg") --target-dir "$out/depinfo" > "$out/depinfo.log" 2>&1 \
      || { echo "RED   cargo check -p $pkg --lib failed: the gate cannot measure which files its build compiled -- $(tail -1 "$out/depinfo.log")"; return 1; }
  done
  while IFS= read -r f; do built[$f]=1; done < <(cat "$out"/depinfo/debug/deps/*.d 2> /dev/null \
    | tr ' ' '\n' | sed -n 's/:$//; s|^'"$PWD"'/||; /\.rs$/p' | sort -u)
  [ "${#built[@]}" -gt 0 ] || { echo "RED   the dep-info under $out/depinfo names no source file: nothing measured is not nothing compiled"; return 1; }
  while IFS= read -r line; do
    [ -n "${built[${line%%:*}]:-}" ] && continue
    UNBUILT+="$line"$'\n'; UNBUILT_N=$((UNBUILT_N + 1))
  done < <(cat "$out/mutants.out/missed.txt" "$out/mutants.out/timeout.txt" 2> /dev/null)
  return 0
}

pkg_features() { # pkg_features <pkg> -> "--features a,b" for the FEATURES entries PKG/FEAT this package owns
  local f fs=""
  for f in ${FEATURES//,/ }; do [ "${f%%/*}" = "$1" ] && fs+="${fs:+,}${f#*/}"; done
  [ -z "$fs" ] || printf -- '--features %s' "$fs"
}

# The feature shard's scope: files in the diff that `cargo check --features` compiles and the default build does
# not, each set MEASURED from dep-info in its own private target dir.
gated_files() { # gated_files <diff> <out> -> GATED (array)
  local diff=$1 out=$2 cargo=${MUTANTS_GATE_CARGO:-cargo} pkg f
  local -A dflt=() feat=()
  GATED=()
  for pkg in $(for f in ${FEATURES//,/ }; do printf '%s\n' "${f%%/*}"; done | sort -u); do
    "$cargo" check -p "$pkg" --lib --target-dir "$out/dep-default" > "$out/dep-default.log" 2>&1 \
      || { echo "RED   cargo check -p $pkg --lib failed -- $(tail -1 "$out/dep-default.log")"; return 1; }
    "$cargo" check -p "$pkg" --lib $(pkg_features "$pkg") --target-dir "$out/dep-feature" > "$out/dep-feature.log" 2>&1 \
      || { echo "RED   cargo check -p $pkg --lib $(pkg_features "$pkg") failed -- $(tail -1 "$out/dep-feature.log")"; return 1; }
  done
  while IFS= read -r f; do dflt[$f]=1; done < <(depinfo_files "$out/dep-default")
  while IFS= read -r f; do feat[$f]=1; done < <(depinfo_files "$out/dep-feature")
  [ "${#feat[@]}" -gt 0 ] || { echo "RED   the feature build's dep-info names no source file: nothing measured is not nothing compiled"; return 1; }
  while IFS= read -r f; do
    if [ -n "${feat[$f]:-}" ] && [ -z "${dflt[$f]:-}" ]; then GATED+=("$f"); fi
  done < <(sed -n 's|^+++ b/||p' "$diff" | sort -u)
  return 0
}

depinfo_files() { # depinfo_files <target-dir> -> repo-relative .rs paths its dep-info names
  cat "$1"/debug/deps/*.d 2> /dev/null | tr ' ' '\n' | sed -n 's/:$//; s|^'"$PWD"'/||; /\.rs$/p' | sort -u
}

# A never-compiled mutant is NOT_MEASURED (L25): never MISSED, never a pass, and no local receipt turns it green.
# Operator ruling E1 (2026-09-29 07:55Z): cuda-gated mutants are measured only by a CI shard on a CUDA runner;
# a not_measured count > 0 on changed gated code is RED. Local evidence does not count.
not_measured() { # not_measured <out>
  local line
  echo "NOT_MEASURED $UNBUILT_N mutant(s) sit in code this build never compiles (a cfg feature off by default):"
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    echo "  NOT_MEASURED  $line"
  done <<< "$UNBUILT"
  echo "RED   not_measured=$UNBUILT_N on changed feature-gated code: measure it in a CI shard that builds the feature (a CUDA runner for cuda); local runs do not count"
  return 1
}

self_test() {
  local T bad=0 SELF
  SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
  T=$(mktemp -d) || return 2
  # shellcheck disable=SC2064
  trap "rm -rf -- '$T'" RETURN
  cat > "$T/cargo" <<'STUB'
#!/usr/bin/env bash
# A fake `cargo mutants`: STUB_LIST (lines), STUB_LIST_RC, STUB_OUTCOMES (json text or "none"), STUB_MISSED.
echo "$*" >> "$STUB_ARGS"
case " $* " in
  *" --list "*) printf '%b' "${STUB_LIST:-}"; exit "${STUB_LIST_RC:-0}" ;;
  " check "*) td=""; prev=""; for a in "$@"; do [ "$prev" = "--target-dir" ] && td=$a; prev=$a; done
    di=${STUB_DEPINFO:-}; case " $* " in *" --features "*) di=${STUB_DEPINFO_FEATURE:-$di} ;; esac
    mkdir -p "$td/debug/deps"; printf '%b' "$di" > "$td/debug/deps/x.d"; exit "${STUB_CHECK_RC:-0}" ;;
esac
out=""; prev=""
for a in "$@"; do [ "$prev" = "--output" ] && out=$a; prev=$a; done
[ "${STUB_OUTCOMES:-none}" = none ] && exit "${STUB_RUN_RC:-0}"
mkdir -p "$out/mutants.out"
printf '%s' "$STUB_OUTCOMES" > "$out/mutants.out/outcomes.json"
printf '%b' "${STUB_MISSED:-}" > "$out/mutants.out/missed.txt"
printf '%b' "${STUB_TIMEOUT:-}" > "$out/mutants.out/timeout.txt"
exit "${STUB_RUN_RC:-0}"
STUB
  chmod +x "$T/cargo"
  printf 'diff --git a/x.rs b/x.rs\n+fn f() {}\n' > "$T/pr.diff"
  # a two-package tree: `root` owns src/, `foo` owns crates/foo/; foo's gated.rs is never in the dep-info
  mkdir -p "$T/tree/crates/foo/src" "$T/tree/evidence/mutants-local"
  # a committed local "receipt" must change nothing (operator E1): it is planted so a gate that reads it goes RED
  printf '[package]\nname = "root"\n' > "$T/tree/Cargo.toml"
  printf '[package]\nname = "foo"\n' > "$T/tree/crates/foo/Cargo.toml"
  printf 'CAUGHT crates/foo/src/gated.rs:7:5: replace g -> bool with true\n' > "$T/tree/evidence/mutants-local/r.txt"
  row() { # row <name> <want rc> <needle> [ENV=VAL...] -- runs THIS script's gate against the stub
    local name=$1 want=$2 needle=$3 out rc; shift 3
    : > "$T/args-$name"   # a fresh log per row: a stale one would answer for a mutant that never ran
    out=$(cd "$T/tree" && env MUTANTS_GATE_CARGO="$T/cargo" STUB_ARGS="$T/args-$name" \
          STUB_DEPINFO="$T/tree/target/x: $T/tree/src/x.rs $T/tree/crates/foo/src/built.rs\n" "$@" \
          bash "$GATE_SCRIPT" "${GATE_DIFF:-$T/pr.diff}" --cap 3 --jobs 2 --max-missed "${MAXM:-0}" --out "$T/out-$name" ${EXCL:-} ${GATE_OPTS:-} 2>&1); rc=$?
    if [ "$rc" = "$want" ] && grep -qF -- "$needle" <<< "$out"; then echo "ok    $name"
    else echo "FAIL  $name -- rc $rc (want $want): $(tr '\n' ' ' <<< "$out" | cut -c1-220)"; fi
  }
  OK2='{"total_mutants": 2, "missed": 0, "caught": 2, "timeout": 0, "unviable": 0, "outcomes": [{"summary": "CaughtMutant"}]}'
  table() {
    row zero-listed-passes 0 "0 mutants in the diff" STUB_LIST=""
    row all-caught-passes 0 "every mutant in the diff was caught" STUB_LIST='a\nb\n' STUB_OUTCOMES="$OK2"
    if grep -q -- "--workspace --in-diff .* --list" "$T/args-all-caught-passes" 2> /dev/null \
       && grep -q -- "--workspace --no-times .* -j 2 " "$T/args-all-caught-passes" 2> /dev/null; then
      echo "ok    workspace-on-list-and-run"
    else echo "FAIL  workspace-on-list-and-run -- $(tr '\n' '|' < "$T/args-all-caught-passes" 2> /dev/null)"; fi
    EXCL="--exclude-crate aprender-compute" row excluded-crate-reaches-list-and-run 0 "excluded from mutation (judged by their own CI steps): aprender-compute" \
        STUB_LIST='a\nb\n' STUB_OUTCOMES="$OK2"
    if [ "$(grep -c -- "--exclude crates/aprender-compute/\*\*" "$T/args-excluded-crate-reaches-list-and-run" 2> /dev/null)" = 2 ]; then
      echo "ok    exclusion-on-list-and-run"
    else echo "FAIL  exclusion-on-list-and-run -- $(tr '\n' '|' < "$T/args-excluded-crate-reaches-list-and-run" 2> /dev/null)"; fi
    row list-failure-is-red 1 "--list exited 101" STUB_LIST_RC=101
    row over-cap-is-red 1 "too many to judge: 4 mutants in the diff > cap 3" STUB_LIST='a\nb\nc\nd\n'
    row missed-is-red-and-named 1 "  src/x.rs:1: replace f with ()" STUB_LIST='a\nb\n' \
        STUB_OUTCOMES='{"total_mutants":2,"missed":1,"caught":1,"timeout":0}' STUB_MISSED='src/x.rs:1: replace f with ()\n'
    row timeout-counts-as-uncaught 1 "1 mutant(s) survived or timed out" STUB_LIST='a\nb\n' \
        STUB_OUTCOMES='{"total_mutants": 2, "missed": 0, "caught": 1, "timeout": 1}'
    MAXM=1 row missed-within-tolerance-passes 0 "missed=1" STUB_LIST='a\nb\n' \
        STUB_OUTCOMES='{"total_mutants": 2, "missed": 1, "caught": 1, "timeout": 0}'
    row no-outcomes-is-red-even-at-rc-0 1 "wrote no outcomes.json although 2 mutant(s)" STUB_LIST='a\nb\n' STUB_OUTCOMES=none
    row unparseable-outcomes-is-red 1 "cannot parse" STUB_LIST='a\nb\n' STUB_OUTCOMES='{"total": 2}'
    row tested-count-mismatch-is-red 1 "tested 1 mutant(s) but listed 2" STUB_LIST='a\nb\n' \
        STUB_OUTCOMES='{"total_mutants": 1, "missed": 0, "caught": 1, "timeout": 0}'
    # the REAL cargo-mutants 27 shape (captured from a run, trimmed): pretty-printed, "outcomes" FIRST with nested
    # phase results, the summary keys AFTER it. The pre-#4142 inline parser read nothing out of this file.
    row real-outcomes-shape-is-parsed 1 "judged 2 mutant(s): missed=1 timeout=0" STUB_LIST='a\nb\n' \
        STUB_OUTCOMES="$(cat "$(dirname "$SELF")/mutants_diff_gate.real-outcomes.json")"
    # #4621: a survivor in a file the build never compiled is NOT_MEASURED, never MISSED and never a pass
    row unbuilt-with-local-receipt-is-red 1 "NOT_MEASURED  crates/foo/src/gated.rs:7:5" STUB_LIST='a\nb\n' \
        STUB_OUTCOMES='{"total_mutants":2,"missed":1,"caught":1,"timeout":0}' STUB_MISSED='crates/foo/src/gated.rs:7:5: replace g -> bool with true\n'
    row unbuilt-timeout-is-red 1 "RED   not_measured=1 on changed feature-gated code" STUB_LIST='a\nb\n' \
        STUB_OUTCOMES='{"total_mutants":2,"missed":0,"caught":1,"timeout":1}' STUB_TIMEOUT='crates/foo/src/gated.rs:9:1: replace h with ()\n'
    row built-survivor-beside-unbuilt-is-red 1 "  crates/foo/src/built.rs:3:1: replace k with ()" STUB_LIST='a\nb\n' \
        STUB_OUTCOMES='{"total_mutants":2,"missed":2,"caught":0,"timeout":0}' \
        STUB_MISSED='crates/foo/src/gated.rs:7:5: replace g -> bool with true\ncrates/foo/src/built.rs:3:1: replace k with ()\n'
    row depinfo-empty-is-red 1 "names no source file" STUB_LIST='a\nb\n' STUB_DEPINFO="" \
        STUB_OUTCOMES='{"total_mutants":2,"missed":1,"caught":1,"timeout":0}' STUB_MISSED='crates/foo/src/gated.rs:7:5: replace g -> bool with true\n'
    row check-failure-is-red 1 "cargo check -p foo --lib failed" STUB_LIST='a\nb\n' STUB_CHECK_RC=101 \
        STUB_OUTCOMES='{"total_mutants":2,"missed":1,"caught":1,"timeout":0}' STUB_MISSED='crates/foo/src/gated.rs:7:5: replace g -> bool with true\n'
    # the CPU gate hands its NOT_MEASURED set to the feature shard, sorted, and says so; built survivors stay RED
    GATE_OPTS="--defer-unbuilt $T/defer.txt" row unbuilt-deferred-passes-and-is-written 0 "DEFERRED 1 NOT_MEASURED" STUB_LIST='a\nb\n' \
        STUB_OUTCOMES='{"total_mutants":2,"missed":1,"caught":1,"timeout":0}' STUB_MISSED='crates/foo/src/gated.rs:7:5: replace g -> bool with true\n'
    if [ "$(cat "$T/defer.txt" 2> /dev/null)" = "crates/foo/src/gated.rs:7:5: replace g -> bool with true" ]; then echo "ok    deferred-set-written"
    else echo "FAIL  deferred-set-written -- $(cat "$T/defer.txt" 2> /dev/null)"; fi
    # the deferred file's directory need not exist: CI passes mutants-gate/not_measured.txt on a fresh checkout
    rm -rf -- "${T:?}/fresh"   # the table runs once per mutant on one $T: a stale directory would hide the mutant
    GATE_OPTS="--defer-unbuilt $T/fresh/dir/defer3.txt" row defer-into-a-missing-dir 0 "0 mutants in the diff" STUB_LIST=""
    [ -f "$T/fresh/dir/defer3.txt" ] && echo "ok    defer-dir-created" || echo "FAIL  defer-dir-created -- no $T/fresh/dir/defer3.txt"
    GATE_OPTS="--defer-unbuilt $T/defer2.txt" row deferred-built-survivor-is-red 1 "  crates/foo/src/built.rs:3:1: replace k with ()" STUB_LIST='a\nb\n' \
        STUB_OUTCOMES='{"total_mutants":2,"missed":2,"caught":0,"timeout":0}' \
        STUB_MISSED='crates/foo/src/gated.rs:7:5: replace g -> bool with true\ncrates/foo/src/built.rs:3:1: replace k with ()\n'
    # the feature shard mutates only the files the feature build adds, and records what it measured
    printf 'diff --git a/crates/foo/src/gated.rs b/crates/foo/src/gated.rs\n+++ b/crates/foo/src/gated.rs\n+fn g() {}\n+++ b/crates/foo/src/built.rs\n+fn k() {}\n' > "$T/gated.diff"
    GATE_DIFF="$T/gated.diff" GATE_OPTS="--features foo/cuda --gated-only" row gated-only-mutates-the-gated-file 0 "gated files: crates/foo/src/gated.rs" \
        STUB_LIST='z\na\n' STUB_OUTCOMES="$OK2" \
        STUB_DEPINFO_FEATURE="$T/tree/target/x: $T/tree/crates/foo/src/built.rs $T/tree/crates/foo/src/gated.rs\n"
    if grep -q -- "--file crates/foo/src/gated.rs" "$T/args-gated-only-mutates-the-gated-file" && ! grep -q -- "--file crates/foo/src/built.rs" "$T/args-gated-only-mutates-the-gated-file" \
       && grep -q -- "check -p foo --lib --features cuda" "$T/args-gated-only-mutates-the-gated-file" \
       && ! grep -q -- "--lib.*--lib" "$T/args-gated-only-mutates-the-gated-file" \
       && grep -q -- "--cargo-arg --lib" "$T/args-gated-only-mutates-the-gated-file" \
       && [ "$(tr '\n' ' ' < "$T/out-gated-only-mutates-the-gated-file/measured.txt" 2> /dev/null)" = "a z " ]; then echo "ok    gated-scope-and-measured-set"
    else echo "FAIL  gated-scope-and-measured-set -- $(tr '\n' '|' < "$T/args-gated-only-mutates-the-gated-file" 2> /dev/null)"; fi
    printf 'm::t_one\nm::t_two\n' > "$T/names.txt"; : > "$T/nonames.txt"
    GATE_OPTS="--test-names $T/names.txt" row test-names-reach-libtest 0 "tests: the 2 named" STUB_LIST='a\nb\n' STUB_OUTCOMES="$OK2"
    if grep -q -- "-- --lib -- --exact --test-threads 1 m::t_one m::t_two" "$T/args-test-names-reach-libtest"; then echo "ok    test-names-on-the-run"
    else echo "FAIL  test-names-on-the-run -- $(tr '\n' '|' < "$T/args-test-names-reach-libtest" 2> /dev/null)"; fi
    GATE_OPTS="--test-names $T/nonames.txt" row empty-test-names-is-red 1 "is empty: a run that selects no test" STUB_LIST='a\nb\n' STUB_OUTCOMES="$OK2"
    GATE_DIFF="$T/gated.diff" GATE_OPTS="--features foo/cuda --gated-only" row gated-only-with-nothing-gated-passes 0 "0 feature-gated files in the diff: measured=0"
    : > "$T/empty.diff"
    GATE_DIFF="$T/empty.diff" row empty-diff-passes 0 "empty diff" STUB_LIST='a\n'
    xrow() { # xrow <name> <want rc> <head-ref> <head-repo> <base-repo> -- rule 0, the car exemption
      local name=$1 want=$2 rc=0; shift 2
      bash "$GATE_SCRIPT" --exempt-ref "$@" > "$T/x-$name" 2>&1 || rc=$?
      if [ "$rc" = "$want" ] && { [ "$want" != 0 ] || grep -qF "deferred to mutants-nightly" "$T/x-$name"; }; then echo "ok    $name"
      else echo "FAIL  $name -- rc $rc (want $want): $(tr '\n' ' ' < "$T/x-$name" | cut -c1-160)"; fi
    }
    xrow car-is-exempt 0 car/0.70.0 paiml/aprender paiml/aprender
    xrow non-car-is-judged 1 feat/x paiml/aprender paiml/aprender
    xrow nested-car-is-judged 1 fix/car/0.70.0 paiml/aprender paiml/aprender
    xrow car-prefix-word-is-judged 1 cargo/x paiml/aprender paiml/aprender
    xrow bare-car-is-judged 1 car/ paiml/aprender paiml/aprender
    xrow fork-car-is-judged 1 car/0.70.0 someone/aprender paiml/aprender
    xrow unknown-repo-is-judged 1 car/0.70.0 "" ""
  }
  GATE_SCRIPT=$SELF
  local o; o=$(table); printf '%s\n' "$o"; grep -q '^FAIL' <<< "$o" && bad=1
  # every rule deleted in a copy of THIS script must break the row that names it (code above self_test only)
  while IFS='~' read -r label must old new; do
    [ -n "$label" ] || continue
    python3 -c 'import sys
s = open(sys.argv[1]).read(); code, cut, rest = s.partition("\nself_test() {")
assert code.count(sys.argv[3]) == 1, "anchor"
open(sys.argv[2], "w").write(code.replace(sys.argv[3], sys.argv[4]) + cut + rest)' "$SELF" "$T/m.sh" "$old" "$new" 2> /dev/null \
      || { echo "FAIL  mutant $label did not apply"; bad=1; continue; }
    GATE_SCRIPT="$T/m.sh"; o=$(table); GATE_SCRIPT=$SELF
    if grep -q "^FAIL  $must " <<< "$o"; then echo "ok    mutant $label killed by $must"
    else echo "FAIL  mutant $label SURVIVED $must"; bad=1; fi
  done <<'MUT'
no-workspace-on-list~workspace-on-list-and-run~"$cargo" mutants --workspace "${ex[@]}" --in-diff "$diff" --list~"$cargo" mutants "${ex[@]}" --in-diff "$diff" --list
no-workspace-on-run~workspace-on-list-and-run~"$cargo" mutants --workspace "${ex[@]}" --no-times~"$cargo" mutants "${ex[@]}" --no-times
list-rc-ignored~list-failure-is-red~  if [ "$rc" -ne 0 ]; then~  if false; then
cap-ignored~over-cap-is-red~  if [ "$cap" -gt 0 ] && [ "$n" -gt "$cap" ]; then~  if false; then
no-outcomes-passes~no-outcomes-is-red-even-at-rc-0~    echo "RED   cargo mutants exited $rc and wrote no outcomes.json although $n mutant(s) were listed: the run died"~    echo "RED   cargo mutants exited $rc and wrote no outcomes.json although $n mutant(s) were listed: the run died"; return 0
compact-json-only-parser~real-outcomes-shape-is-parsed~  grep -oE "\"$2\": ?[0-9]+" "$1" | head -1 | grep -oE '[0-9]+$'~  grep -oE "\"$2\":[0-9]+" "$1" | head -1 | grep -oE '[0-9]+$'
missing-field-is-zero~unparseable-outcomes-is-red~  if [ -z "$total" ] || [ -z "$missed" ] || [ -z "$timeout" ]; then~  total=${total:-2}; missed=${missed:-0}; timeout=${timeout:-0}; if false; then
count-mismatch-ok~tested-count-mismatch-is-red~  if [ "$total" -ne "$n" ]; then~  if false; then
timeout-not-counted~timeout-counts-as-uncaught~  if [ $((missed + timeout)) -gt "$max" ]; then~  if [ "$missed" -gt "$max" ]; then
exclusion-dropped-on-run~exclusion-on-list-and-run~  "$cargo" mutants --workspace "${ex[@]}" --no-times~  "$cargo" mutants --workspace --no-times
exclusion-dropped-on-list~exclusion-on-list-and-run~  "$cargo" mutants --workspace "${ex[@]}" --in-diff "$diff" --list~  "$cargo" mutants --workspace --in-diff "$diff" --list
car-glob-widened~nested-car-is-judged~  [[ $1 == car/?* ]] || return 1~  [[ $1 == *car/?* ]] || return 1
car-prefix-unanchored~car-prefix-word-is-judged~  [[ $1 == car/?* ]] || return 1~  [[ $1 == car?* ]] || return 1
fork-not-checked~fork-car-is-judged~  [ -n "$2" ] && [ "$2" = "$3" ] || return 1~  :
empty-repo-accepted~unknown-repo-is-judged~  [ -n "$2" ] && [ "$2" = "$3" ] || return 1~  [ "$2" = "$3" ] || return 1
exempt-never~car-is-exempt~  [[ $1 == car/?* ]] || return 1~  return 1
unbuilt-is-a-pass~unbuilt-timeout-is-red~  echo "RED   not_measured=$UNBUILT_N~  return 0; echo "RED   not_measured=$UNBUILT_N
local-receipt-honoured~unbuilt-with-local-receipt-is-red~    echo "  NOT_MEASURED  $line"~    grep -qxF -- "CAUGHT $line" evidence/mutants-local/*.txt 2> /dev/null && continue; echo "  NOT_MEASURED  $line"
built-survivor-hidden~built-survivor-beside-unbuilt-is-red~    if [ "$((missed + timeout - UNBUILT_N))" -gt "$max" ] || [ "$UNBUILT_N" -eq 0 ]; then~    if [ "$UNBUILT_N" -eq 0 ]; then
empty-depinfo-means-all-unbuilt~depinfo-empty-is-red~  [ "${#built[@]}" -gt 0 ] || {~  true || {
names-dropped~test-names-on-the-run~    ta+=(-- --exact --test-threads 1 "${c[@]}")~    :
empty-names-accepted~empty-test-names-is-red~    [ -s "$NAMES" ] || { echo "RED   --test-names~    true || { echo "RED   --test-names
defer-dir-not-made~defer-into-a-missing-dir~{ mkdir -p "$(dirname "$DEFER")" && : > "$DEFER"; }~{ : > "$DEFER"; }
defer-ignored~unbuilt-deferred-passes-and-is-written~    if [ -n "$DEFER" ]; then~    if false; then
deferred-set-unsorted~deferred-set-written~      printf '%s' "$UNBUILT" | LC_ALL=C sort > "$DEFER" || return 1~      : > "$DEFER"
gated-includes-default-built~gated-scope-and-measured-set~    if [ -n "${feat[$f]:-}" ] && [ -z "${dflt[$f]:-}" ]; then GATED+=("$f"); fi~    if [ -n "${feat[$f]:-}" ]; then GATED+=("$f"); fi
measured-unsorted~gated-scope-and-measured-set~  if [ "$GATED_ONLY" = 1 ]; then LC_ALL=C sort "$out/list.txt" > "$out/measured.txt" || return 1; fi~  if [ "$GATED_ONLY" = 1 ]; then cp "$out/list.txt" "$out/measured.txt" || return 1; fi
features-dropped-from-check~gated-scope-and-measured-set~      || { echo "RED   cargo check -p $pkg --lib failed -- $(tail -1 "$out/dep-default.log")"; return 1; }~      || { echo "RED   cargo check -p $pkg --lib failed -- $(tail -1 "$out/dep-default.log")"; return 1; }; FEATURES=""
check-rc-ignored~check-failure-is-red~    "$cargo" check -p "$pkg" --lib $(pkg_features "$pkg") --target-dir "$out/depinfo" > "$out/depinfo.log" 2>&1 \~    true \
lib-passed-twice~gated-scope-and-measured-set~--cargo-arg --lib); ta=(); echo~--cargo-arg --lib); echo
lib-missing-from-build~gated-scope-and-measured-set~ex+=(--features "$FEATURES" --cargo-arg --lib); ta=();~ex+=(--features "$FEATURES");
survivors-unnamed~missed-is-red-and-named~    cat "$out/mutants.out/missed.txt" "$out/mutants.out/timeout.txt" 2> /dev/null | sed 's/^/  /'~    :
MUT
  echo "mutants_diff_gate self-test: $([ "$bad" = 0 ] && echo PASS || echo FAIL)"
  return "$bad"
}

main() {
  local diff="" cap=60 jobs=4 max=0 out=mutants-gate
  local -a excl=()
  [ "${1:-}" = "--self-test" ] && { self_test; exit $?; }
  if [ "${1:-}" = "--exempt-ref" ]; then
    [ $# -eq 4 ] || { echo "usage: mutants_diff_gate.sh --exempt-ref <head-ref> <head-repo> <base-repo>" >&2; exit 2; }
    exempt_ref "$2" "$3" "$4"; exit $?
  fi
  while [ $# -gt 0 ]; do
    case "$1" in
      --cap) cap=$2; shift 2 ;;
      --jobs) jobs=$2; shift 2 ;;
      --max-missed) max=$2; shift 2 ;;
      --out) out=$2; shift 2 ;;
      --exclude-crate) excl+=("$2"); shift 2 ;;
      --timeout) TIMEOUT=$2; shift 2 ;;
      --defer-unbuilt) DEFER=$2; shift 2 ;;
      --features) FEATURES=$2; shift 2 ;;
      --gated-only) GATED_ONLY=1; shift ;;
      --test-names) NAMES=$2; shift 2 ;;
      -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
      -*) echo "mutants_diff_gate: unknown option $1" >&2; exit 2 ;;
      *) diff=$1; shift ;;
    esac
  done
  [ "$GATED_ONLY" = 0 ] || [ -n "$FEATURES" ] || { echo "mutants_diff_gate: --gated-only needs --features" >&2; exit 2; }
  [ -n "$diff" ] || { echo "usage: mutants_diff_gate.sh <diff> [--cap N] [--jobs J] [--max-missed M] [--out DIR]" >&2; exit 2; }
  for v in "$cap" "$jobs" "$max"; do [[ "$v" =~ ^[0-9]+$ ]] || { echo "mutants_diff_gate: not a count: $v" >&2; exit 2; }; done
  gate "$diff" "$cap" "$jobs" "$max" "$out" "${excl[@]}"
}

main "$@"
