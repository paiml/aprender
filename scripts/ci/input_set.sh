#!/usr/bin/env bash
# FLOW-003 QM-10 (#4528): the build-and-test input set I of the repo at a base commit, written to
# ci/input-set.json by ONE writer, the nightly. Spec: docs/specifications/FLOW-003-queue-and-release-cycle-model.md
# §T.2 Definition 14 and §11.4 (contract ci-input-set-v1). The tier router (QM-11) reads it; this script
# never routes.
#
#   I = I_dep ∪ F ∪ C
#   I_dep  every repo path in rustc dep-info (target/**/*.d) of a full build on base          -> .dep
#   F      every repo path a test process opened or stat'd                                    -> .read
#          every directory a test process ENUMERATED (getdents64), recorded as `dir/**`       -> .dirPrefix
#          every repo path a test process looked up and did NOT find (ENOENT/ENOTDIR)         -> .absent
#   C      manifests, lockfile, toolchain file, .cargo/, workflows, the package of every build
#          script with no rerun-if-changed, every proc-macro package                          -> .config
#
# Paths are repo-relative. A directory entry is written `dir/**` (the whole prefix, including paths
# that do not exist yet: INV-DIR). `**` alone is the repo root: nothing is docs-only then.
#
# usage:
#   input_set.sh trace   <root> <trace.log> -- <cmd...>   run cmd under strace (the syscalls of R2-9)
#   input_set.sh parse   <root> <trace.log>...            F as tab lines: R|A|D <tab> repo path
#   input_set.sh dep     <root> <target-dir>              I_dep, one repo path per line
#   input_set.sh config  <root>                           C, one entry per line
#   input_set.sh assemble <root> <target-dir> <base-sha> <out.json> <trace.log>...
#   input_set.sh merge   <out.json> <in.json>...          union of F over nightlies; dep/config from the first
#   input_set.sh check   <input-set.json>                 the ci-input-set-v1 shape (exit 1 names the fault)
#   input_set.sh covers  <input-set.json> <path>...       exit 0 iff every path is in I; prints the hit
#   input_set.sh base    <input-set.json> <repo> <base>   INV-BASE: exit 0 iff every commit in n..base
#                                                         touches no path of I; else exit 10, naming them
#   input_set.sh --self-test                             fixture falsifiers F1..F5 + controls, then planted mutants
#
# exit: 0 ok · 1 a fault (named on stderr) · 10 INV-BASE refresh required · 64 usage
set -euo pipefail

SYSCALLS='openat,open,stat,lstat,newfstatat,statx,access,faccessat,faccessat2,getdents64'
# the kinds the contract's cis:traceSyscalls may name (lstat/faccessat* are recorded under stat/access)
CONTRACT_SYSCALLS='["openat","open","newfstatat","statx","access","getdents64"]'
MIN_NIGHTLIES=7
SELF=$(realpath "$0")
# INPUT_SET_MUTATE plants one weakened mechanism for the self-test (see self_test_mutants); unset in real runs.
MUTATE=${INPUT_SET_MUTATE:-}
[ "$MUTATE" = drop-getdents64 ] && SYSCALLS=${SYSCALLS%,getdents64}

die() { printf 'input_set: %s\n' "$1" >&2; exit "${2:-1}"; }
usage() { sed -n '/^# usage:/,/^# exit:/p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 64; }

# The root as the tracer may see it: logical and physical (a symlinked /mnt/nvme-raid0 is both).
roots() {
  local r
  r=$(cd "$1" && pwd -L)
  printf '%s\n' "$r"
  (cd "$1" && pwd -P)
}

cmd_trace() {
  [ $# -ge 4 ] && [ "$3" = "--" ] || usage
  local root=$1 log=$2
  shift 3
  command -v strace >/dev/null || die "strace not found: F cannot be traced, so no input set"
  (cd "$root" && strace -f -y -qq -e trace="$SYSCALLS" -o "$log" -- "$@")
}

# The parser. strace -y prints every dirfd with its path (`AT_FDCWD</cwd>`, `3</dir>`), so each record
# resolves on its own line; a syscall split across `<unfinished ...>` / `<... resumed>` is rejoined per pid.
# A non-*at syscall with a relative path resolves against the last cwd its pid printed; one that cannot
# is a fault (fail closed: an unresolved read is a hole in F).
PARSE_AWK='
function norm(p,   n, a, i, k, st, out) {
  n = split(p, a, "/"); k = 0
  for (i = 1; i <= n; i++) {
    if (a[i] == "" || a[i] == ".") continue
    if (a[i] == "..") { if (k > 0) k--; continue }
    st[++k] = a[i]
  }
  out = ""
  for (i = 1; i <= k; i++) out = out "/" st[i]
  return out == "" ? "/" : out
}
function rel(p,   i, r) {
  for (i = 1; i <= nroot; i++) {
    r = root[i]
    if (p == r) return "."
    if (substr(p, 1, length(r) + 1) == r "/") return substr(p, length(r) + 2)
  }
  return ""
}
function fdpath(s,   i, j) {
  i = index(s, "<"); if (i == 0) return ""
  j = index(substr(s, i + 1), ">"); if (j == 0) return ""
  return substr(s, i + 1, j - 1)
}
function qstr(s,   i, j, c, out) {
  i = index(s, "\""); if (i == 0) return "\001"
  out = ""
  for (j = i + 1; j <= length(s); j++) {
    c = substr(s, j, 1)
    if (c == "\\") { out = out substr(s, j + 1, 1); j++; continue }
    if (c == "\"") return out
    out = out c
  }
  return "\001"
}
function emit(kind, abs,   r) {
  r = rel(norm(abs))
  if (r == "" ) return
  if (r == "target" || substr(r, 1, 7) == "target/") return
  if (tdrel != "" && (r == tdrel || substr(r, 1, length(tdrel) + 1) == tdrel "/")) return
  print kind "\t" r
}
function handle(pid, name, args, res,   base, p, abs, kind, isat) {
  if (match(args, /^AT_FDCWD</)) cwd[pid] = fdpath(args)
  if (name == "getdents64") {
    base = fdpath(args)
    if (base != "" && res !~ /^-1/) emit("D", base)
    return
  }
  isat = (name ~ /at2?$/ || name == "statx")
  p = qstr(args)
  if (p == "\001") return
  if (substr(p, 1, 1) == "/") abs = p
  else {
    base = isat ? fdpath(args) : cwd[pid]
    if (base == "") { unresolved++; if (unresolved <= 5) printf "unresolved relative path (pid %s, %s): %s\n", pid, name, p > "/dev/stderr"; return }
    abs = (p == "") ? base : base "/" p
  }
  if (res ~ /^-1 (ENOENT|ENOTDIR)/ && mutate != "drop-enoent") kind = "A"
  else if (res ~ /^-1/) return
  else kind = "R"
  emit(kind, abs)
}
BEGIN { nroot = split(roots, root, "\n") }
{
  line = $0
  if (!match(line, /^[0-9]+ +/)) next
  pid = substr(line, 1, RLENGTH); sub(/ +$/, "", pid)
  line = substr(line, RLENGTH + 1)
  if (line ~ /^<\.\.\. [a-z0-9_]+ resumed>/) {
    if (!(pid in pend)) next
    sub(/^<\.\.\. [a-z0-9_]+ resumed>/, "", line)
    line = pend[pid] line
    delete pend[pid]
  } else if (line ~ /<unfinished \.\.\.>$/) {
    sub(/ *<unfinished \.\.\.>$/, "", line)
    pend[pid] = line
    next
  }
  if (!match(line, /^[a-z0-9_]+\(/)) next
  name = substr(line, 1, RLENGTH - 1)
  args = substr(line, RLENGTH + 1)
  n = split(line, parts, /\) += +/)
  if (n < 2) next
  handle(pid, name, args, parts[n])
}
END { if (unresolved > 0) { printf "input_set: %d unresolved relative path(s): F would have a hole\n", unresolved > "/dev/stderr"; exit 1 } }
'

cmd_parse() {
  [ $# -ge 2 ] || usage
  local root=$1 tdrel=""
  shift
  [ -n "${INPUT_SET_TARGET_DIR:-}" ] && tdrel=$(realpath -m --relative-to="$root" "$INPUT_SET_TARGET_DIR")
  awk -v roots="$(roots "$root")" -v tdrel="$tdrel" -v mutate="$MUTATE" "$PARSE_AWK" "$@" | LC_ALL=C sort -u
}

# rustc dep-info is make syntax: `out: dep dep …` with `\ ` for a space; trailing `dep:` rules repeat deps.
cmd_dep() {
  [ $# -eq 2 ] || usage
  local root=$1 td=$2
  [ -d "$td" ] || die "no target dir $td: I_dep needs a full build first"
  local n
  n=$(find "$td" -name '*.d' -type f | head -1 | wc -l)
  [ "$n" -gt 0 ] || die "no dep-info (*.d) under $td: I_dep would be empty"
  find "$td" -name '*.d' -type f -print0 \
    | xargs -0 cat \
    | awk -v roots="$(roots "$root")" -v mutate="$MUTATE" '
      function norm(p,   n, a, i, k, st, out) {
        n = split(p, a, "/"); k = 0
        for (i = 1; i <= n; i++) { if (a[i] == "" || a[i] == ".") continue; if (a[i] == "..") { if (k > 0) k--; continue }; st[++k] = a[i] }
        out = ""; for (i = 1; i <= k; i++) out = out "/" st[i]
        return out
      }
      # rustc runs from the workspace root, so a relative dep-info path (the source of a workspace member,
      # `app/src/../../docs/x.md`) is relative to it; an absolute one (registry, cargo-level .d) is as is.
      function rel(p,   i, r) {
        p = norm(substr(p, 1, 1) == "/" ? p : root[1] "/" p)
        for (i = 1; i <= nroot; i++) { r = root[i]; if (substr(p, 1, length(r) + 1) == r "/") return substr(p, length(r) + 2) }
        return ""
      }
      BEGIN { nroot = split(roots, root, "\n") }
      {
        gsub(/\\ /, "\001")
        c = index($0, ": "); if (c == 0) { if ($0 ~ /:$/) next; else next }
        n = split(substr($0, c + 2), t, " ")
        for (i = 1; i <= n; i++) { p = t[i]; gsub("\001", " ", p); r = rel(p); if (r != "" && r !~ /^target\// && (mutate != "drop-depinfo" || r ~ /\.rs$/)) print r }
      }' \
    | LC_ALL=C sort -u
}

# C. A package whose build.rs prints no rerun-if-changed reruns on ANY change in the package, and a
# proc-macro runs inside every dependent's compile: both are whole-package entries.
cmd_config() {
  [ $# -eq 1 ] || usage
  local root=$1
  (
    cd "$root"
    git ls-files -- 'Cargo.toml' '*/Cargo.toml' 'Cargo.lock' 'rust-toolchain.toml' 'rust-toolchain'
    printf '%s\n' '.cargo/**' '.github/workflows/**' '.config/**'
    cargo metadata --no-deps --format-version 1 \
      | jq -r --arg root "$(pwd -P)/" '
          .packages[]
          | (.manifest_path | ltrimstr($root) | rtrimstr("Cargo.toml")) as $dir
          | .targets[]
          | select(any(.kind[]; . == "custom-build" or . == "proc-macro"))
          | [(.kind | join(",")), $dir, (.src_path | ltrimstr($root))] | @tsv' \
      | while IFS=$'\t' read -r kind dir src; do
          if [ "$kind" = "proc-macro" ] || ! grep -q 'rerun-if-changed' "$src" 2>/dev/null; then
            printf '%s**\n' "$dir"
          fi
        done
  ) | LC_ALL=C sort -u
}

lines_json() { jq -Rn '[inputs | select(length > 0)]'; }

cmd_assemble() {
  [ $# -ge 5 ] || usage
  local root=$1 td=$2 base=$3 out=$4
  shift 4
  [[ "$base" =~ ^[0-9a-f]{40}$ ]] || die "base sha must be 40 hex, got '$base'"
  local tmp
  tmp=$(mktemp -d)
  # shellcheck disable=SC2064
  trap "rm -rf -- '${tmp:?}'" EXIT
  cmd_dep "$root" "$td" > "$tmp/dep"
  INPUT_SET_TARGET_DIR=$td cmd_parse "$root" "$@" > "$tmp/f"
  cmd_config "$root" > "$tmp/c"
  awk -F'\t' '$1 == "R" { print $2 }' "$tmp/f" > "$tmp/read"
  awk -F'\t' '$1 == "A" { print $2 }' "$tmp/f" > "$tmp/absent"
  awk -F'\t' '$1 == "D" { print ($2 == "." ? "**" : $2 "/**") }' "$tmp/f" | LC_ALL=C sort -u > "$tmp/dir"
  jq -n --arg base "$base" --argjson sys "$CONTRACT_SYSCALLS" \
    --slurpfile dep <(lines_json < "$tmp/dep") \
    --slurpfile read <(lines_json < "$tmp/read") \
    --slurpfile dir <(lines_json < "$tmp/dir") \
    --slurpfile absent <(lines_json < "$tmp/absent") \
    --slurpfile config <(lines_json < "$tmp/c") \
    '{schema: "ci-input-set-v1", baseSha: $base, dep: $dep[0], read: $read[0], dirPrefix: $dir[0],
      absent: $absent[0], config: $config[0], traceSyscalls: $sys, nightlies: 1, nightlyShas: [$base]}' > "$out"
  cmd_check "$out" --allow-few
}

# F is unioned over nightlies; I_dep and C describe the newest base only (they are rebuilt, not traced).
cmd_merge() {
  [ $# -ge 2 ] || usage
  local out=$1
  shift
  jq -s '
    .[0] as $new
    | $new + {
        read: ([.[].read[]] | unique), dirPrefix: ([.[].dirPrefix[]] | unique), absent: ([.[].absent[]] | unique),
        nightlyShas: ([.[].nightlyShas[]] | unique), nightlies: ([.[].nightlyShas[]] | unique | length)
      }' "$@" > "$out"
  cmd_check "$out" --allow-few
}

cmd_check() {
  [ $# -ge 1 ] || usage
  local f=$1 few=${2:-}
  local fault
  fault=$(jq -r --argjson min "$MIN_NIGHTLIES" --arg few "$few" '
    def strs: type == "array" and all(.[]; type == "string" and length > 0);
    ["schema","baseSha","dep","read","dirPrefix","absent","config","traceSyscalls","nightlies","nightlyShas"] as $keys
    | if type != "object" then "not an object"
      elif ((keys - $keys) | length) > 0 then "closed shape: unknown key(s) \(keys - $keys)"
      elif (($keys - keys) | length) > 0 then "missing key(s) \($keys - keys)"
      elif (.baseSha | test("^[0-9a-f]{40}$") | not) then "baseSha is not 40 hex"
      elif (.dep | strs and length >= 1 | not) then "dep: needs >= 1 path (I_dep is never empty)"
      elif (.read | strs and length >= 1 | not) then "read: needs >= 1 path (a traced test run reads something)"
      elif (.dirPrefix | strs | not) then "dirPrefix: not a list of paths"
      elif (.absent | strs | not) then "absent: not a list of paths"
      elif (.config | strs and length >= 1 | not) then "config: needs >= 1 entry"
      elif (.traceSyscalls | index("getdents64") == null) then "traceSyscalls lacks getdents64: directory listings would escape F (R2-9)"
      elif (.nightlies | type != "number" or . < 1) then "nightlies is not a positive integer"
      elif ($few == "" and .nightlies < $min) then "nightlies \(.nightlies) < \($min): F is not yet unioned over 7 nightlies"
      else empty end' "$f") || die "$f is not JSON"
  [ -z "$fault" ] || die "$f: $fault"
}

# Is a repo path in I? dep/read/absent are exact; config and dirPrefix entries ending `**` are prefixes.
IN_I_JQ='
  def inI($p):
    ([.dep[], .read[], .absent[]] | index($p)) != null
    or any((.config[], .dirPrefix[]);
           if endswith("**") then (rtrimstr("**")) as $pre | ($pre == "" or ($p | startswith($pre)))
           else . == $p end);
'
cmd_covers() {
  [ $# -ge 2 ] || usage
  local f=$1 miss=0 p
  shift
  for p in "$@"; do
    if jq -e --arg p "$p" "$IN_I_JQ inI(\$p)" "$f" >/dev/null; then printf 'in I: %s\n' "$p"
    else printf 'not in I: %s\n' "$p"; miss=1; fi
  done
  return "$miss"
}

# INV-BASE (R2-8): I(n) stands in for I(b) only if no commit in n..b touched a path of I(n). A commit that
# did (say, one adding `include_str!("../docs/x.md")`) changes what the build reads, so the router must
# refresh I_dep on b; this names every such commit and path.
cmd_base() {
  [ $# -eq 3 ] || usage
  local f=$1 repo=$2 b=$3 n c p hit=0
  [ "$MUTATE" = base-noop ] && return 0
  n=$(jq -r .baseSha "$f")
  git -C "$repo" merge-base --is-ancestor "$n" "$b" || die "input set base $n is not an ancestor of $b: no n..b range"
  while read -r c; do
    while IFS= read -r p; do
      [ -n "$p" ] || continue
      if jq -e --arg p "$p" "$IN_I_JQ inI(\$p)" "$f" >/dev/null; then
        printf 'refresh: %s touches %s (in I)\n' "${c:0:10}" "$p"
        hit=1
      fi
    done < <(git -C "$repo" diff-tree --no-commit-id --name-only -r --root "$c")
  done < <(git -C "$repo" rev-list --reverse "$n..$b")
  [ "$hit" -eq 0 ] || exit 10
}


# --self-test: a fixture workspace carrying one planted instance of each §11.4 falsifier, plus controls.
# Every row must hold on the real script, and each planted mutant (INPUT_SET_MUTATE) must turn its row RED.
# fixture git: no user config, no global hooks (a pre-commit hook would run inside the fixture)
stg() { local d=$1; shift; git -C "$d" -c user.name=t -c user.email=t@t -c commit.gpgsign=false -c core.hooksPath=/dev/null "$@"; }
st_fixture() {
  local r=$1
  mkdir -p "$r/app/src" "$r/app/tests" "$r/bs/src" "$r/docs/specifications"
  printf '[workspace]\nmembers = ["app", "bs"]\nresolver = "2"\n' > "$r/Cargo.toml"
  printf '[package]\nname = "app"\nversion = "0.1.0"\nedition = "2021"\n' > "$r/app/Cargo.toml"
  printf '[package]\nname = "bs"\nversion = "0.1.0"\nedition = "2021"\nbuild = "build.rs"\n' > "$r/bs/Cargo.toml"
  printf 'fn main() {}\n' > "$r/bs/build.rs"
  printf 'pub fn f() {}\n' > "$r/bs/src/lib.rs"
  # F1 include_str! of a docs file -> dep
  printf 'pub const X: &str = include_str!("../../docs/x.md");\n' > "$r/app/src/lib.rs"
  cat > "$r/app/tests/t.rs" <<'RS'
#[test]
fn reads_and_probes() {
    // F2 a test reads a docs file -> read
    let _ = std::fs::read_to_string("../docs/y.md").expect("y.md");
    // F3 a test enumerates a docs directory -> dirPrefix docs/specifications/**
    let n = std::fs::read_dir("../docs/specifications").expect("dir").count();
    assert!(n > 0);
    // F4 a test probes a path that does not exist -> absent
    assert!(!std::path::Path::new("../docs/absent.md").exists());
    assert!(!app::X.is_empty());
}
RS
  printf 'x\n' > "$r/docs/x.md"
  printf 'y\n' > "$r/docs/y.md"
  printf 'a\n' > "$r/docs/specifications/a.md"
  printf 'u\n' > "$r/docs/unrelated.md"
  git -C "$r" init -q
  stg "$r" add -A
  stg "$r" commit -qm nightly
}

self_test() {
  local tmp r td log out bin fails=0
  tmp=$(mktemp -d)
  # shellcheck disable=SC2064
  trap "rm -rf -- '${tmp:?}'" EXIT
  r=$tmp/repo td=$tmp/target log=$tmp/trace.log out=$tmp/input-set.json
  st_fixture "$r"
  bin=$(cd "$r" && CARGO_TARGET_DIR=$td cargo test -q -p app --test t --no-run --message-format=json 2>/dev/null \
        | jq -r 'select(.reason == "compiler-artifact" and .target.name == "t") | .executable // empty' | tail -1)
  [ -x "$bin" ] || die "self-test: fixture test binary did not build"
  # The traced process is the test binary with cargo's cwd for it (the package root), as nextest runs it.
  "$SELF" trace "$r/app" "$log" -- "$bin" -q >/dev/null 2>&1 || die "self-test: the traced fixture test failed"
  "$SELF" assemble "$r" "$td" "$(git -C "$r" rev-parse HEAD)" "$out" "$log" || die "self-test: assemble failed"

  row() { # row <id> <description> <command...>: GREEN iff the command succeeds
    local id=$1 what=$2
    shift 2
    if "$@" >/dev/null 2>&1; then printf 'GREEN %s %s\n' "$id" "$what"
    else printf 'RED   %s %s\n' "$id" "$what"; fails=$((fails + 1)); fi
  }
  has() { jq -e --arg p "$2" ".$1 | index(\$p) != null" "$out"; }
  not() { ! "$@"; }
  row F1 'include_str! of docs/x.md is in dep' has dep docs/x.md
  row F2 'a test reading docs/y.md puts it in read' has read docs/y.md
  row F3 'read_dir of docs/specifications puts docs/specifications/** in dirPrefix' has dirPrefix 'docs/specifications/**'
  row F3b 'so a NEW file there is in I (not docs-only)' "$SELF" covers "$out" docs/specifications/new.md
  row F4 'Path::exists of docs/absent.md puts it in absent' has absent docs/absent.md
  row C1 'a build script without rerun-if-changed puts its package in config' has config 'bs/**'
  row C2 'the lockfile-free fixture still lists its manifests' has config app/Cargo.toml
  row K1 'control: docs/unrelated.md is NOT in I' not "$SELF" covers "$out" docs/unrelated.md
  row K2 'control: target/ paths never enter I' not jq -e '[.dep[], .read[]] | any(startswith("target/"))' "$out"
  row K3 'one nightly is not a valid input set (nightlies >= 7)' not "$SELF" check "$out"
  jq --arg z "$(printf '%039d' 0)" '.nightlyShas = [range(7) | $z + tostring]' "$out" > "$tmp/n0.json"
  row K4 'the union of 7 nightlies is valid' "$SELF" merge "$tmp/n7.json" "$out" "$tmp/n0.json"
  row K4b '... and passes the full check' "$SELF" check "$tmp/n7.json"
  # F5 INV-BASE: a docs-only commit keeps I(n); a code commit that include_str!s a docs file forces a refresh.
  printf 'v\n' >> "$r/docs/unrelated.md"
  stg "$r" commit -qam docs-only
  row K5 'control: a docs-only commit in n..b keeps I(n)' "$SELF" base "$out" "$r" HEAD
  printf 'z\n' > "$r/docs/z.md"
  printf 'pub const Z: &str = include_str!("../../docs/z.md");\n' >> "$r/app/src/lib.rs"
  stg "$r" add -A
  stg "$r" commit -qm code
  local rc=0
  "$SELF" base "$out" "$r" HEAD >/dev/null 2>&1 || rc=$?
  row F5 'a code commit in n..b that include_str!s a docs file forces a refresh (exit 10)' test "$rc" -eq 10
  [ "$fails" -eq 0 ] || { printf 'self-test: %d row(s) RED\n' "$fails" >&2; return 1; }
}

# Each mutant weakens one mechanism; the self-test must fail on it. A mutant that leaves every row GREEN
# means the row that should catch it is vacuous.
self_test_mutants() {
  local m out rc
  for m in drop-getdents64 drop-enoent drop-depinfo base-noop; do
    rc=0
    out=$(INPUT_SET_MUTATE=$m "$SELF" --self-test-once 2>&1) || rc=$?
    if [ "$rc" -eq 0 ]; then printf 'mutant %s: survived (every row GREEN)\n' "$m" >&2; return 1; fi
    printf 'mutant %s: killed by %s\n' "$m" "$(printf '%s\n' "$out" | awk '/^RED/ {print $2}' | paste -sd, -)"
  done
}

[ $# -ge 1 ] || usage
sub=$1
shift
case "$sub" in
  trace) cmd_trace "$@" ;;
  parse) cmd_parse "$@" ;;
  dep) cmd_dep "$@" ;;
  config) cmd_config "$@" ;;
  assemble) cmd_assemble "$@" ;;
  merge) cmd_merge "$@" ;;
  check) cmd_check "$@" ;;
  covers) cmd_covers "$@" ;;
  base) cmd_base "$@" ;;
  --self-test) self_test && self_test_mutants ;;
  --self-test-once) self_test ;;
  *) usage ;;
esac
