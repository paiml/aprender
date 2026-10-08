#!/usr/bin/env bash
# fetch_p6.sh -- `git fetch` for the CI sections, read up to 3 times under P6
# (#4936, operator C343 #9).
#
# WHY THIS EXISTS
# ---------------
# Shard, determinism and x86-main sections died on `git fetch` with
# "RPC failed; curl 92 HTTP/2 stream 5 was not closed cleanly: CANCEL (err 8)"
# or "curl 56 Recv failure: Connection reset by peer", on every host alike.
# The tier step's depth-1 fetch of main ran 376 s and died, although main's
# objects were already in the section clone: the read did not answer, it was
# not too big. Every section fetch was a single read, so one stalled
# connection was a red section.
#
# P6: a read that did not answer (non-zero exit, HTTP 5xx, timeout, empty
# body) is read at most 3 times in all. A read that answered is never re-read:
# a missing ref, an unadvertised object, a 401/403/404 and a usage error are
# answers. Each read that did not answer prints one status line. A read that
# stalls is cut (under LOW_SPEED_LIMIT bytes/s for LOW_SPEED_TIME s), so "did
# not answer" never hangs a section.
#
#   bash scripts/ci/fetch_p6.sh <git fetch args...>     # in place of `git fetch`
#   bash scripts/ci/fetch_p6.sh --check-wiring          # no bare `git fetch` in ci/sections.yml
#   bash scripts/ci/fetch_p6.sh --self-test
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SECTIONS_REL="ci/sections.yml"
HELPER_REL="scripts/ci/fetch_p6.sh"
P6_READS=3
LOW_SPEED_LIMIT=1000
LOW_SPEED_TIME=60
# Seconds before re-read n is (n * SLEEP_S). The case table plants a `sleep`
# that records its argument and returns at once.
SLEEP_S=10

# answered <rc> <stderr-file> -> 0 when the read answered and a re-read cannot
# change the answer. git runs under LC_ALL=C, so these are git's own texts.
# rc 129 is git's usage error: the read never left this host.
# "could not read Username" is a 401 with no credentials to offer.
answered() {
  [ "$1" -eq 129 ] && return 0
  grep -qiE "couldn't find remote ref|not our ref|unadvertised object|Repository not found|does not appear to be a git repository|not a git repository|Authentication failed|could not read Username|The requested URL returned error: 40[134]" "$2"
}

# p6_fetch <git fetch args...> -> git fetch, read up to P6_READS times.
# rc = the last read's rc. git's stderr reaches the caller's stderr.
p6_fetch() {
  local err n rc why
  err="$(mktemp)" || return 2
  for ((n = 1; n <= P6_READS; n++)); do
    LC_ALL=C git -c "http.lowSpeedLimit=$LOW_SPEED_LIMIT" -c "http.lowSpeedTime=$LOW_SPEED_TIME" \
      fetch "$@" 2> "$err"
    rc=$?
    cat "$err" >&2
    if [ "$rc" -eq 0 ] || answered "$rc" "$err"; then
      rm -f -- "${err:?}"
      return "$rc"
    fi
    why="$(grep -E '^(error|fatal):' "$err" | head -n 1)"
    if [ "$n" -lt "$P6_READS" ]; then
      printf 'fetch_p6: read %d/%d did not answer (rc=%d: %s); reading again\n' \
        "$n" "$P6_READS" "$rc" "${why:-no error line}"
      sleep "$((n * SLEEP_S))"
    else
      printf 'fetch_p6: read %d/%d did not answer (rc=%d: %s); no answer under P6\n' \
        "$n" "$P6_READS" "$rc" "${why:-no error line}"
    fi
  done
  rm -f -- "${err:?}"
  return "$rc"
}

# A git command whose subcommand is fetch. git's global options may sit
# between them (-C <dir>, -c <k=v>, --git-dir=<d>, --no-pager), quoted or not,
# and any run of blanks separates words. The self-test's must-match and
# must-not-match forms are this pattern's case table.
FETCH_RE='(^|[^[:alnum:]_./-])"?git"?([[:space:]]+(-[cC][[:space:]]+("[^"]*"|[^[:space:]]+)|--?[^[:space:]]+))*[[:space:]]+fetch([[:space:];&|)]|$)'

# check_wiring <sections.yml> -> one FAIL line per bare `git fetch` outside a
# comment. rc 0 = every fetch goes through the helper; 1 = a bare fetch;
# 2 = unreadable, or no helper call found (a check that read nothing).
check_wiring() {
  [ -r "$1" ] || return 2
  local bad=0 line
  while IFS= read -r line; do
    printf 'FAIL: %s fetches without P6 re-reads -- use bash %s: %s\n' \
      "$SECTIONS_REL" "$HELPER_REL" "$line"
    bad=1
  done < <(grep -nE "$FETCH_RE" "$1" | grep -vE '^[0-9]+:[[:space:]]*#')
  grep -qF "bash $HELPER_REL " "$1" || return 2
  return "$bad"
}

self_test() {
  printf '=== case table: fetch_p6.sh ===\n'
  local tmp fails=0 rc real_git
  tmp="$(mktemp -d)" || return 1
  trap 'rm -rf "${tmp:?}"' RETURN
  real_git="$(command -v git)"
  row() {  # row <label> <want> <got>
    if [ "$2" = "$3" ]; then
      printf '  ok   %-70s %s\n' "$1" "$3"
    else
      printf '  FAIL %-70s want=%s got=%s\n' "$1" "$2" "$3"
      fails=$((fails + 1))
    fi
  }
  # A planted git: counts its reads, records its args, answers per SHIM_MODE.
  mkdir -p "$tmp/bin"
  cat > "$tmp/bin/git" <<'SHIM'
#!/usr/bin/env bash
n=$(( $(cat "$SHIM_DIR/count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$SHIM_DIR/count"
printf '%s\n' "$*" >> "$SHIM_DIR/args"
printf '%s\n' "${LC_ALL:-unset}" > "$SHIM_DIR/lc"
cancel() { printf '%s\n' 'error: RPC failed; curl 92 HTTP/2 stream 5 was not closed cleanly: CANCEL (err 8)' 'fatal: early EOF' >&2; exit 128; }
case "$SHIM_MODE" in
  ok) exit 0 ;;
  cancel-twice) [ "$n" -le 2 ] && cancel; exit 0 ;;
  never) printf '%s\n' 'error: RPC failed; curl 56 Recv failure: Connection reset by peer' >&2; exit 128 ;;
  http500-once) [ "$n" -le 1 ] && { printf '%s\n' 'fatal: unable to access: The requested URL returned error: 500' >&2; exit 128; }; exit 0 ;;
  silent-once) [ "$n" -le 1 ] && exit 1; exit 0 ;;
  missing) printf '%s\n' "fatal: couldn't find remote ref refs/heads/nope" >&2; exit 128 ;;
  notfound) printf '%s\n' 'remote: Repository not found.' 'fatal: repository not found' >&2; exit 128 ;;
  noauth) printf '%s\n' "fatal: could not read Username for 'https://github.com': terminal prompts disabled" >&2; exit 128 ;;
  norepo) printf '%s\n' 'fatal: not a git repository (or any of the parent directories): .git' >&2; exit 128 ;;
  usage) printf '%s\n' 'usage: git fetch [<options>] [<repository> [<refspec>...]]' >&2; exit 129 ;;
  real) exec "$SHIM_REAL" "$@" ;;
esac
SHIM
  # A planted sleep: records how long each re-read would wait, returns at once.
  printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$1" >> "$SHIM_DIR/sleeps"\n' > "$tmp/bin/sleep"
  chmod +x "$tmp/bin/git" "$tmp/bin/sleep"
  # The caller runs under LC_ALL=POSIX, not C: a runner that already exports LC_ALL=C would
  # otherwise pass the locale row with the helper's own LC_ALL=C removed.
  run() {  # run <mode> <fetch args...> -> rc; reads in $tmp/<mode>/count
    local mode="$1"; shift
    mkdir -p "$tmp/$mode"
    ( cd "$tmp/client" && SLEEP_S=7 && LC_ALL=POSIX PATH="$tmp/bin:$PATH" SHIM_DIR="$tmp/$mode" SHIM_MODE="$mode" \
        SHIM_REAL="$real_git" p6_fetch "$@" ) \
      > "$tmp/$mode/out" 2> "$tmp/$mode/err"
  }
  reads() { cat "$tmp/$1/count" 2>/dev/null || echo 0; }
  status_lines() { grep -c '^fetch_p6: read ' "$tmp/$1/out"; }

  # A real remote for the rows that run real git.
  "$real_git" init -q --bare "$tmp/remote.git"
  "$real_git" init -q "$tmp/seed"
  "$real_git" -C "$tmp/seed" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -q --allow-empty -m seed
  "$real_git" -C "$tmp/seed" push -q "$tmp/remote.git" HEAD:refs/heads/main
  "$real_git" init -q "$tmp/client"
  "$real_git" -C "$tmp/client" remote add origin "$tmp/remote.git"

  run ok origin main; rc=$?
  row 'a read that answers at once: rc 0' 0 "$rc"
  row 'a read that answers at once: read once, no status line' '1 0' "$(reads ok) $(status_lines ok)"
  row 'every read is cut on a stall (lowSpeedLimit and lowSpeedTime set)' 1 \
    "$(grep -c "^-c http.lowSpeedLimit=$LOW_SPEED_LIMIT -c http.lowSpeedTime=$LOW_SPEED_TIME fetch origin main\$" "$tmp/ok/args")"
  row "git runs under LC_ALL=C, not the caller's locale, so the answer texts are git's own" C "$(cat "$tmp/ok/lc")"

  run cancel-twice --no-tags --depth=1 origin +refs/heads/main:refs/remotes/origin/main; rc=$?
  row 'curl 92 CANCEL twice, then an answer (#4934): rc 0' 0 "$rc"
  row 'curl 92 CANCEL twice, then an answer: 3 reads, 2 status lines' '3 2' "$(reads cancel-twice) $(status_lines cancel-twice)"
  row 'each status line names its read and the error' 1 \
    "$(grep -c '^fetch_p6: read 2/3 did not answer (rc=128: error: RPC failed; curl 92' "$tmp/cancel-twice/out")"

  run never origin main; rc=$?
  row 'curl 56 reset on every read: the step still fails (rc 128)' 128 "$rc"
  row 'curl 56 reset on every read: exactly 3 reads, never a 4th' 3 "$(reads never)"
  row 'curl 56 reset on every read: the last line says no answer' 1 \
    "$(tail -n 1 "$tmp/never/out" | grep -c '^fetch_p6: read 3/3 did not answer .*no answer under P6$')"
  row 'curl 56 reset on every read: waits n*SLEEP_S before re-read n+1, none after' '7 14' \
    "$(paste -sd' ' "$tmp/never/sleeps" 2>/dev/null)"
  row 'outside the table the backoff step is SLEEP_S=10' 10 "$SLEEP_S"

  run http500-once origin main; rc=$?
  row 'HTTP 500 once, then an answer: rc 0 after 2 reads' '0 2' "$rc $(reads http500-once)"
  run silent-once origin main; rc=$?
  row 'non-zero exit with no error line, then an answer: 2 reads' '0 2' "$rc $(reads silent-once)"

  run missing origin nope; rc=$?
  row 'a missing ref is an answer: rc 128, read once' '128 1' "$rc $(reads missing)"
  row "git's stderr still reaches the caller" 1 "$(grep -c "couldn't find remote ref" "$tmp/missing/err")"
  run notfound origin main; rc=$?
  row 'repository not found is an answer: read once' '128 1' "$rc $(reads notfound)"
  run noauth origin main; rc=$?
  row 'a 401 with no credentials (could not read Username) is an answer: read once' '128 1' "$rc $(reads noauth)"
  run norepo origin main; rc=$?
  row 'not a git repository is an answer: read once' '128 1' "$rc $(reads norepo)"
  run usage --bogus origin main; rc=$?
  row "git's usage error (rc 129) never left the host: read once" '129 1' "$rc $(reads usage)"

  run real origin +refs/heads/main:refs/remotes/origin/main; rc=$?
  row 'real git, a ref that exists: rc 0, read once' '0 1' "$rc $(reads real)"
  row 'real git, a ref that exists: the ref is fetched' 1 \
    "$("$real_git" -C "$tmp/client" rev-parse -q --verify refs/remotes/origin/main >/dev/null && echo 1 || echo 0)"
  rm -f -- "${tmp:?}/real/count"
  run real origin +refs/heads/nope:refs/remotes/origin/nope; rc=$?
  row "real git, a missing ref: fails, read once (git's own wording is an answer)" '1 1' \
    "$([ "$rc" -ne 0 ] && echo 1 || echo 0) $(reads real)"

  # Wiring: no bare `git fetch` in the sections file.
  printf '          bash %s --no-tags --depth=1 origin x\n          # git fetch in a comment is prose\n' \
    "$HELPER_REL" > "$tmp/ok.yml"
  check_wiring "$tmp/ok.yml" > /dev/null; row 'wiring: every fetch through the helper, a comment ignored' 0 "$?"
  # FETCH_RE's case table: each form sits beside a helper call, alone.
  local form i=0
  while IFS= read -r form; do
    i=$((i + 1))
    printf '          bash %s origin x\n          %s\n' "$HELPER_REL" "$form" > "$tmp/m$i.yml"
    check_wiring "$tmp/m$i.yml" > /dev/null; row "wiring: RED on: $form" 1 "$?"
  done <<'MUST_MATCH'
git fetch --no-tags --depth=1 origin main
git -C "$ws" fetch origin main
git -C "$a b" fetch origin main
git -c http.version=HTTP/1.1 fetch origin
git --no-pager fetch origin
git --git-dir=.git fetch origin
git  fetch origin
cd "$ws" && git fetch; echo done
"git" fetch origin
MUST_MATCH
  while IFS= read -r form; do
    i=$((i + 1))
    printf '          bash %s origin x\n          %s\n' "$HELPER_REL" "$form" > "$tmp/m$i.yml"
    check_wiring "$tmp/m$i.yml" > /dev/null; row "wiring: clean on: $form" 0 "$?"
  done <<'MUST_NOT_MATCH'
bash scripts/ci/fetch_p6.sh --no-tags --deepen=1 origin
git log --grep fetch origin/main
git status && echo fetch
echo "the git fetched it"
legit fetch
MUST_NOT_MATCH
  printf '          git status\n' > "$tmp/none.yml"
  check_wiring "$tmp/none.yml" > /dev/null; row 'wiring: no helper call found is not clean' 2 "$?"
  check_wiring "$tmp/absent.yml" > /dev/null; row 'wiring: an unreadable sections file is not clean' 2 "$?"

  if [ "$fails" -gt 0 ]; then
    printf '\nFAIL: %s case(s) failed. The helper does not do what P6 says.\n' "$fails"
    return 1
  fi
  printf 'PASS: all cases behave as declared.\n'
  return 0
}

case "${1:-}" in
  --self-test)
    self_test
    exit $?
    ;;
  --check-wiring)
    printf '=== every git fetch in %s reads under P6 (fetch_p6.sh) ===\n' "$SECTIONS_REL"
    check_wiring "$REPO_ROOT/$SECTIONS_REL"
    rc=$?
    if [ "$rc" -eq 2 ]; then
      printf 'FAIL: %s is unreadable or calls no %s -- a check that read nothing certifies nothing.\n' \
        "$SECTIONS_REL" "$HELPER_REL"
      exit 1
    fi
    [ "$rc" -eq 0 ] || exit 1
    printf 'PASS: %s fetch call(s) in %s, each through the helper.\n' \
      "$(grep -cF "bash $HELPER_REL " "$REPO_ROOT/$SECTIONS_REL")" "$SECTIONS_REL"
    exit 0
    ;;
  "")
    printf 'usage: %s <git fetch args...> | --check-wiring | --self-test\n' "$HELPER_REL" >&2
    exit 2
    ;;
  *)
    p6_fetch "$@"
    exit $?
    ;;
esac
