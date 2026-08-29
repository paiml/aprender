#!/usr/bin/env bash
# perf_receipt_sign.sh — the HOST half of APR-PERF-GATE-001 v2.2 section 4.9.1.
#
#   scripts/perf_receipt_sign.sh --receipt R.json --key-id ID [--keyring K] [--out S.json]
#   scripts/perf_receipt_sign.sh --selftest
#
# WHY THIS EXISTS
# ---------------
# Section 4.9.1, verbatim:
#
#   Per `APR-QUALITY-001` v1.8 section 0.7 J3: hosts push signed receipts
#   (forjar cron), and the blocking job runs anywhere and verifies signature +
#   freshness. Cite J3; do not re-litigate.
#
#   The staleness arm is what makes it a gate. Without
#   `receipt.commit` contains `commit-under-test`, `evidence/` is a
#   declared-state artifact.
#
# lambda-4090 is do-not-revive, gx10 and mini are not general CI runners, and
# intel's perf label must not contend with the clean-room release gate. So the
# gate cannot be a job that runs ON the host that measures. What reaches the
# gate is a FILE, and until this script existed a receipt was a file anyone
# could write: nothing bound its `commit` field to the commit under test and
# nothing bound the document to the host that produced it.
#
# This script is the producing end. `scripts/perf_gate.sh --phase release`
# is the verifying end. `scripts/lib/receipt_sig.py` is the single authority
# both of them call, so the payload construction cannot drift between them.
#
# WHAT THE FORJAR DEPLOYMENT WOULD BE  (NOT IMPLEMENTED ON THIS BRANCH)
# ---------------------------------------------------------------------
# `forjar` config lives in a separate repository (`paiml/infra`) and is not
# touched here. Written down so the remaining step is a deployment rather than
# a design:
#
#   1. Key material. One 256-bit key per host, `key_id` = `<host>-<serial>`
#      (`gx10-2026a`). Generate with `openssl rand -hex 32`. The host gets a
#      one-line keyring at `/etc/apr/perf-receipt.key` mode 0400 owned by the
#      bench user; the verifier gets the same rows as a CI secret. Rotation is
#      a new serial, never an edit in place: `load_keyring` refuses a duplicate
#      `key_id` precisely so a rotated key cannot keep validating silently.
#   2. `machines/<host>/forjar.yaml` gains a `perf-receipt` unit + timer that
#      runs the section 4.4 harness, then this script, then pushes the signed
#      file to `evidence/perf/<host>/<workload>/<commit>.json`.
#   3. `forjar apply`, then `make -C machines/<host> deploy-systemd-units` and
#      `verify-systemd-units`. Repo edits are inert until deployed -- the
#      2026-04-26 ENOSPC outage was a 7-day silent desync of exactly this kind.
#   4. On `intel` the timer takes the dedicated single-agent label, NOT one of
#      `intel-clean-room-{1..16}`, and shares the clean-room concurrency group
#      (section 4.9.2, I-7).
#   5. mini is macOS: `launchd`, not `systemd`, and its bash is 3.2. This script
#      is written to 3.2 (no `declare -A`, no `${var^^}`, no `readarray`).
#
# The half that IS implemented and proven here is signing and verification.
# Nothing below has ever run on a fleet host; that claim is not made.
#
# WHAT THIS DELIBERATELY WILL NOT DO
# ----------------------------------
# It will not sign a receipt for a host its key is not scoped to, it will not
# invent a `commit`, and it will not mint a key when the keyring is missing.
# Each of those would produce a document indistinguishable from a real one,
# which is worse than refusing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIGLIB="$ROOT/scripts/lib/receipt_sig.py"
DEFAULT_KEYRING="${APR_PERF_RECEIPT_KEYRING:-}"

die() { printf 'perf_receipt_sign: %s\n' "$*" >&2; exit 2; }

sign_receipt() {
  # receipt, key_id, keyring, out
  local receipt="$1" key_id="$2" keyring="$3" out="$4"
  [ -f "$receipt" ] || die "receipt not found: $receipt"
  [ -n "$keyring" ] || die "no keyring: pass --keyring or set APR_PERF_RECEIPT_KEYRING"
  python3 "$SIGLIB" --sign --in "$receipt" --out "$out" \
    --key-id "$key_id" --keyring "$keyring"
}

# ------------------------------------------------------------- selftest -----
# Both polarities of every row. A signer whose refusals are untested is a
# signer that stamps anything, and a signer whose happy path is untested is a
# gate that can never pass -- this epic has shipped both shapes.
selftest() {
  local tmp pass=0 fail=0 rc=0
  tmp="$(mktemp -d)"
  case "$tmp" in
    /tmp/*|/var/folders/*) : ;;
    *) die "mktemp -d gave ${tmp:-<empty>}, refusing to rm -rf it" ;;
  esac

  # Throwaway 256-bit selftest keys. Never deployed anywhere: the fleet keys
  # come from `openssl rand -hex 32` and live only on the hosts.
  local keyA keyB kr
  keyA=a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1
  keyB=b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2
  kr="$tmp/keyring"
  {
    printf '# perf_receipt_sign selftest keyring\n'
    printf 'lambda-selftest %s\n' "$keyA"
    printf 'gx10-selftest %s\n' "$keyB"
  } > "$kr"

  printf '{"commit":"deadbeef","provenance":{"host":"lambda"},"bands":[]}\n' \
    > "$tmp/lambda.json"
  printf '{"commit":"deadbeef","provenance":{"host":"gx10"},"bands":[]}\n' \
    > "$tmp/gx10.json"
  printf '{"provenance":{"host":"lambda"},"bands":[]}\n' > "$tmp/nocommit.json"

  _row() { # name, expect(pass|fail), receipt, key_id, keyring, out
    local got
    if sign_receipt "$3" "$4" "$5" "$6" >/dev/null 2>&1; then got=pass; else got=fail; fi
    if [ "$got" = "$2" ]; then
      printf '  ok    %-34s expect=%s\n' "$1" "$2"
      pass=$((pass + 1))
    else
      printf '  BROKE %-34s expected %s got %s\n' "$1" "$2" "$got"
      fail=$((fail + 1))
    fi
  }

  _row signs_own_host          pass "$tmp/lambda.json"   lambda-selftest "$kr" "$tmp/s1.json"
  _row refuses_foreign_host    fail "$tmp/gx10.json"     lambda-selftest "$kr" "$tmp/s2.json"
  _row refuses_commitless      fail "$tmp/nocommit.json" lambda-selftest "$kr" "$tmp/s3.json"
  _row refuses_unknown_key_id  fail "$tmp/lambda.json"   lambda-rotated  "$kr" "$tmp/s4.json"
  _row refuses_missing_keyring fail "$tmp/lambda.json"   lambda-selftest "$tmp/absent" "$tmp/s5.json"

  # The output actually verifies. Without this row the four refusals above
  # would still pass on a signer that emitted garbage.
  local repo c0
  repo="$tmp/repo"
  mkdir -p "$repo"
  git -C "$repo" init -q --template= >/dev/null 2>&1
  git -C "$repo" config user.email selftest@example.invalid
  git -C "$repo" config user.name 'perf-receipt selftest'
  git -C "$repo" config commit.gpgsign false
  git -C "$repo" config core.hooksPath "$tmp/nohooks"
  printf 'x\n' > "$repo/f"
  git -C "$repo" add -A
  git -C "$repo" commit -q -m c0
  c0="$(git -C "$repo" rev-parse HEAD)"

  printf '{"commit":"%s","provenance":{"host":"lambda"},"bands":[]}\n' "$c0" \
    > "$tmp/real.json"
  _row signs_real_commit pass "$tmp/real.json" lambda-selftest "$kr" "$tmp/real.signed.json"

  local out
  out="$(python3 "$SIGLIB" --verify "$tmp/real.signed.json" --host lambda --commit "$c0" --keyring "$kr" --git-dir "$repo" 2>&1)" && rc=0 || rc=$?
  if [ "$rc" = 0 ]; then
    printf '  ok    %-34s expect=pass\n' signed_output_verifies
    pass=$((pass + 1))
  else
    printf '  BROKE %-34s verifier said: %s\n' signed_output_verifies "$out"
    fail=$((fail + 1))
  fi

  # The unsigned original must NOT verify. Same file, same host, same commit --
  # the only difference is the signature, so this isolates it.
  out="$(python3 "$SIGLIB" --verify "$tmp/real.json" --host lambda --commit "$c0" --keyring "$kr" --git-dir "$repo" 2>&1)" && rc=0 || rc=$?
  case "$out" in
    *UNSIGNED*)
      if [ "$rc" = 0 ]; then
        printf '  BROKE %-34s named UNSIGNED but exited 0\n' unsigned_original_rejected
        fail=$((fail + 1))
      else
        printf '  ok    %-34s expect=UNSIGNED\n' unsigned_original_rejected
        pass=$((pass + 1))
      fi
      ;;
    *)
      printf '  BROKE %-34s expected UNSIGNED, got: %s\n' unsigned_original_rejected "$out"
      fail=$((fail + 1))
      ;;
  esac

  # DISCRIMINATION: a no-op re-sign with the same pinned timestamp is
  # byte-identical, so re-running the cron does not churn evidence/.
  python3 "$SIGLIB" --sign --in "$tmp/real.json" --out "$tmp/d1.json" \
    --key-id lambda-selftest --keyring "$kr" --signed-at 2026-08-29T00:00:00Z >/dev/null
  python3 "$SIGLIB" --sign --in "$tmp/real.json" --out "$tmp/d2.json" \
    --key-id lambda-selftest --keyring "$kr" --signed-at 2026-08-29T00:00:00Z >/dev/null
  if cmp -s "$tmp/d1.json" "$tmp/d2.json"; then
    printf '  ok    %-34s expect=identical\n' resign_is_deterministic
    pass=$((pass + 1))
  else
    printf '  BROKE %-34s two signings of one receipt differ\n' resign_is_deterministic
    fail=$((fail + 1))
  fi

  printf '  %d passed, %d broken\n' "$pass" "$fail"
  rm -rf "${tmp:?refusing to rm an empty path}"
  [ "$fail" = 0 ]
}

main() {
  local receipt="" key_id="" keyring="$DEFAULT_KEYRING" out=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --selftest) selftest; return $? ;;
      --receipt) receipt="$2"; shift 2 ;;
      --key-id) key_id="$2"; shift 2 ;;
      --keyring) keyring="$2"; shift 2 ;;
      --out) out="$2"; shift 2 ;;
      *) die "unknown argument: $1" ;;
    esac
  done
  [ -n "$receipt" ] && [ -n "$key_id" ] \
    || die "usage: perf_receipt_sign.sh --receipt R.json --key-id ID [--keyring K] [--out S.json]"
  [ -n "$out" ] || out="${receipt%.json}.signed.json"
  sign_receipt "$receipt" "$key_id" "$keyring" "$out"
}

main "$@"
