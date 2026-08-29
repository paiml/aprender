#!/usr/bin/env python3
"""Receipt signature + staleness — APR-PERF-GATE-001 v2.2 §4.9.1, I-10 (PERF-007).

Requires Python 3.6+ (hashlib, hmac, json, subprocess only). No third-party
dependency, because this runs on four hosts including a macOS box on bash 3.2.

WHAT §4.9.1 SAYS, VERBATIM
-------------------------
    Per `APR-QUALITY-001` v1.8 §0.7 J3: **hosts push signed receipts**
    (forjar cron), and the blocking job runs anywhere and verifies
    **signature + freshness**. Cite J3; do not re-litigate.

    The **staleness arm is what makes it a gate**. Without
    `receipt.commit ⊇ commit-under-test`, `evidence/` is a declared-state
    artifact.

and §4.10 I-10, whose registered mutation is "present a receipt one commit
stale":

    | I-10 | Receipt signature valid **and** `receipt.commit ⊇ commit-under-test`
    | present a receipt one commit stale |

Two hosts are not CI runners and the only fully-comparated one is
do-not-revive, so the gate cannot run *on* the host that measures. What arrives
at the gate is a FILE. Before this module a receipt was a file anyone could
write: nothing bound its `commit` field to the commit under test, and nothing
bound the whole document to the host that produced it.

THE TWO FAILURES ARE DIFFERENT AND MUST READ DIFFERENTLY
--------------------------------------------------------
`scripts/apr_bin.sh` conflated STALE with WRONG-TREE and it cost real time: a
person told "your binary is old" rebuilds, when the true fault was that they
were looking at a different checkout entirely. So:

  * STALE  -> "your evidence is older than the code" -> RE-MEASURE.
  * WRONG-HOST / FORGED / UNKNOWN-KEY -> "your evidence is about something
    else" -> re-measure would not help; find the right receipt.

Every verdict below carries a machine-readable code precisely so the two
classes cannot be collapsed into one "receipt bad" message.

A MISSING SIGNATURE IS RED, NEVER "NOT APPLICABLE"
--------------------------------------------------
The recurring defect in this epic is a check that cannot fail. So there is no
"unsigned receipts are exempt" path, no "keyring absent so skip", and no
"cannot reach git so assume fresh". Every one of those is a FAILURE code below.
The only legal skip is the phase one, and it lives in `perf_gate.sh`: §4.5's
Arm C table scopes this rule to `release`, not `merge`.

THREAT MODEL, STATED RATHER THAN IMPLIED
----------------------------------------
`hmac-sha256` with a per-host 256-bit key. The private half lives on the
producing host (forjar-deployed); the verifier holds the same bytes in order to
check them. This is SYMMETRIC, so it does exactly one thing and no more: it
makes a receipt unforgeable **by anyone who does not hold a host key**, which
covers the case this gate exists for — a receipt hand-written, hand-edited or
copied forward into `evidence/` inside a pull request. It does NOT defend
against a holder of the verification keyring. Upgrading to an asymmetric
scheme is a change of `alg` and `_verify_value` only: the signed payload is
constructed independently of the algorithm, and `signature.alg` is covered by
it, so a downgrade to a weaker `alg` cannot be smuggled past a verifier that
does not accept it.

FRESHNESS IS A COMMIT RELATION, NOT A CLOCK
-------------------------------------------
`receipt.commit ⊇ commit-under-test` is an ancestry test with no continuous
threshold in it, which is why it is armed here. A wall-clock maximum age is a
**policy** number under `perf-matrix.yaml`'s GROUNDING RULE and would need an
author and a rationale; none has been decided, so none is invented. The age is
REPORTED (`signed_at`, plus its age in days) and nothing fails on it. That is
recorded here rather than left implicit: the load-bearing arm is armed, the
unarmed one is named.

Usage:
    receipt_sig.py --sign   --in R.json --out S.json --key-id ID --keyring K
    receipt_sig.py --verify R.json --host H --commit SHA --keyring K [--git-dir D]
    receipt_sig.py --selftest
Exit:
    0 ok - 1 rejected - 2 usage/read error
"""
import datetime
import hashlib
import hmac
import json
import os
import subprocess
import sys

DOMAIN = "APR-PERF-GATE-001/receipt-signature/v1"
SIGNATURE_KEY = "signature"
ALGORITHMS = ("hmac-sha256",)
# 256-bit, inherited from SHA-256's output width. Not a tunable: a short key is
# the whole scheme's strength, and "usually long enough" is how it stops being.
KEY_HEX_LEN = 64
SIGNATURE_FIELDS = ("alg", "key_id", "signed_at", "commit", "host",
                    "body_sha256", "value")
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class SigError(Exception):
    """A signing-side refusal. Carries a code the caller prints verbatim."""

    def __init__(self, code, message):
        Exception.__init__(self, "%s %s" % (code, message))
        self.code = code
        self.message = message


# --------------------------------------------------------------- payload ---

def canonical_body(receipt):
    """The bytes a signature covers: the receipt WITHOUT its signature object,
    serialised deterministically.

    Both sign and verify canonicalise a *parsed* document, so the result does
    not depend on how the producer spaced or ordered its JSON. `allow_nan` is
    off: `json` will happily emit a bare `NaN`, which is not JSON, and a
    receipt carrying one must raise here rather than acquire a signature.
    """
    body = {k: v for k, v in receipt.items() if k != SIGNATURE_KEY}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def body_digest(receipt):
    """Lowercase hex SHA-256 of `canonical_body`."""
    return hashlib.sha256(canonical_body(receipt)).hexdigest()


def build_payload(alg, key_id, signed_at, commit, host, body_sha256):
    """The exact byte string that gets signed.

    Domain-separated, and it covers every header field. `signed_at` in
    particular MUST be in here: if it were not, a stale receipt could be made
    to look freshly produced by editing one string, and the freshness half of
    §4.9.1 would be decorative.
    """
    lines = [
        DOMAIN,
        "alg=%s" % alg,
        "key_id=%s" % key_id,
        "signed_at=%s" % signed_at,
        "commit=%s" % commit,
        "host=%s" % host,
        "body_sha256=%s" % body_sha256,
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _mac(alg, key_bytes, payload):
    if alg != "hmac-sha256":
        raise SigError("ALG-UNKNOWN", "alg=%r is not one of %s"
                                      % (alg, list(ALGORITHMS)))
    return hmac.new(key_bytes, payload, hashlib.sha256).hexdigest()


# --------------------------------------------------------------- keyring ---

def _is_hex(value, length):
    return (isinstance(value, str) and len(value) == length
            and all(c in "0123456789abcdef" for c in value))


def load_keyring(path):
    """`key_id <64-hex>` per line, `#` comments. Raises SigError.

    A duplicate `key_id` is refused rather than last-wins: two rows claiming
    one identity means the verifier cannot say which host signed, and silently
    picking one is how a rotated key keeps validating.
    """
    if not path:
        raise SigError("NO-KEYRING",
                       "no keyring path given -- a verifier holding no keys "
                       "cannot verify anything, and that is not a pass. Set "
                       "APR_PERF_RECEIPT_KEYRING.")
    if not os.path.isfile(path):
        raise SigError("NO-KEYRING",
                       "keyring %r does not exist -- the verifier holds no "
                       "keys. This is a failure, never a skip." % path)
    keys = {}
    with open(path, encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            _keyring_row(keys, path, lineno, raw)
    if not keys:
        raise SigError("NO-KEYRING",
                       "keyring %r holds zero keys -- an empty keyring "
                       "verifies nothing and must not read as a pass" % path)
    return keys


# Refusal texts live here rather than inline so each check below stays one
# readable line. They are the message a person gets at 3am; they are not
# decoration.
_M_ROW = "%s: expected `key_id <64-hex>`"
_M_HEX = ("%s: key for %r is not 64 lowercase hex characters (a 256-bit key, "
          "inherited from SHA-256)")
_M_DUP = ("%s: duplicate key_id %r -- two rows for one identity means the "
          "verifier cannot say which host signed")


def _keyring_row(keys, path, lineno, raw):
    """Parse one keyring line into `keys`. Blank and comment lines are skipped;
    everything else must be exactly `key_id <64-hex>`."""
    line = raw.split("#", 1)[0].strip()
    if not line:
        return
    parts = line.split()
    where = "%s:%d" % (path, lineno)
    if len(parts) != 2:
        raise SigError("KEYRING-MALFORMED", _M_ROW % where)
    _keyring_entry(keys, where, parts[0], parts[1])


def _keyring_entry(keys, where, key_id, material):
    """One well-formed row: the key is a 256-bit hex string and its identity is
    not already claimed."""
    if not _is_hex(material, KEY_HEX_LEN):
        raise SigError("KEYRING-MALFORMED", _M_HEX % (where, key_id))
    if key_id in keys:
        raise SigError("KEYRING-MALFORMED", _M_DUP % (where, key_id))
    keys[key_id] = material


def key_id_host(key_id):
    """The host a key_id is scoped to: everything before the first `-`.

    Binds key to host structurally so a receipt claiming `host: lambda` cannot
    be signed by gx10's key while naming gx10's key_id.
    """
    return key_id.split("-", 1)[0] if "-" in key_id else ""


# ---------------------------------------------------------------- signing ---

def sign(receipt, key_id, key_hex, signed_at=None):
    """Return a copy of `receipt` carrying a `signature` object.

    Refuses at SOURCE if the key is not scoped to the host the receipt claims.
    Mislabelling is cheapest to catch here, where the host still knows what it
    is; at the verifier it is indistinguishable from an attack.
    """
    commit, host = _sign_preflight(receipt, key_id, key_hex)
    if signed_at is None:
        signed_at = _utcnow().strftime(TIMESTAMP_FORMAT)
    _parse_timestamp(signed_at)
    alg = ALGORITHMS[0]
    digest = body_digest(receipt)
    value = _mac(alg, bytes.fromhex(key_hex),
                 build_payload(alg, key_id, signed_at, commit, host, digest))
    signed = {k: v for k, v in receipt.items() if k != SIGNATURE_KEY}
    signed[SIGNATURE_KEY] = {
        "alg": alg,
        "key_id": key_id,
        "signed_at": signed_at,
        "commit": commit,
        "host": host,
        "body_sha256": digest,
        "value": value,
    }
    return signed


_M_OBJ = "receipt must be a JSON object"
_M_NOCOMMIT = ("receipt.commit is absent or empty -- there is nothing for the "
               "staleness arm to compare, so this receipt cannot be signed")
_M_NOHOST = ("provenance.host is absent -- an unattributed receipt cannot be "
             "signed for a host")
_M_KEYHOST = ("key_id %r is scoped to host %r but the receipt claims host %r "
              "-- refusing to sign another host's evidence")
_M_KEYLEN = "key for %r is not 64 lowercase hex characters"


def _sign_preflight(receipt, key_id, key_hex):
    """Everything the signer refuses BEFORE producing an attestation.

    Mislabelling is cheapest to catch here, where the host still knows what it
    is; at the verifier it is indistinguishable from an attack.
    """
    if not isinstance(receipt, dict):
        raise SigError("RECEIPT-MALFORMED", _M_OBJ)
    commit, host = _sign_target(receipt)
    if key_id_host(key_id) != host:
        raise SigError("KEY-HOST-MISMATCH",
                       _M_KEYHOST % (key_id, key_id_host(key_id), host))
    if not _is_hex(key_hex, KEY_HEX_LEN):
        raise SigError("KEY-MALFORMED", _M_KEYLEN % key_id)
    return commit, host


def _sign_target(receipt):
    """What the receipt says it is about: (commit, host). Neither has a
    default -- a signature over an invented one attests to nothing."""
    commit = receipt.get("commit")
    if not isinstance(commit, str) or not commit.strip():
        raise SigError("NO-COMMIT", _M_NOCOMMIT)
    prov = receipt.get("provenance")
    host = prov.get("host") if isinstance(prov, dict) else None
    if not isinstance(host, str) or not host.strip():
        raise SigError("NO-HOST", _M_NOHOST)
    return commit, host


def _utcnow():
    """Naive UTC. `utcnow()` is deprecated from 3.12; this is the replacement
    that keeps the value naive so it subtracts cleanly from a parsed stamp."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _parse_timestamp(text):
    try:
        return datetime.datetime.strptime(text, TIMESTAMP_FORMAT)
    except (TypeError, ValueError):
        raise SigError("TIMESTAMP-MALFORMED",
                       "signed_at=%r is not %s" % (text, TIMESTAMP_FORMAT))


# ------------------------------------------------------------- containment ---

def commit_contains(git_dir, ancestor, descendant):
    """Does `descendant`'s history contain `ancestor`? (`⊇`, §4.9.1.)

    Returns (verdict, detail) where verdict is one of YES / NO / UNDECIDABLE.
    UNDECIDABLE is deliberately NOT folded into either answer: a verifier that
    cannot see the commit a receipt names knows nothing about freshness, and
    guessing in either direction is how a gate stops being one.
    """
    for name, rev in (("commit-under-test", ancestor),
                      ("receipt.commit", descendant)):
        proc = subprocess.run(["git", "-C", git_dir, "cat-file", "-e",
                               "%s^{commit}" % rev],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            return ("UNDECIDABLE",
                    "%s %r is not a commit this checkout can resolve (git "
                    "cat-file exit %d) -- containment is undecidable here, "
                    "which is not the same as fresh. Fetch it, or run the "
                    "verifier somewhere that has it."
                    % (name, rev, proc.returncode))
    proc = subprocess.run(["git", "-C", git_dir, "merge-base",
                           "--is-ancestor", ancestor, descendant],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 0:
        return ("YES", "")
    if proc.returncode == 1:
        return ("NO", "")
    return ("UNDECIDABLE",
            "git merge-base --is-ancestor exited %d: %s"
            % (proc.returncode, proc.stderr.decode("utf-8", "replace").strip()))


# ------------------------------------------------------------ verification ---

def _fail(out, code, message):
    out.append((code, message))


def _check_shape(sig, out):
    """The signature object is present and complete, or nothing else applies."""
    if sig is None:
        _fail(out, "UNSIGNED",
              "receipt carries no `signature` object. A receipt is a file "
              "anyone can write; unsigned, it binds to no host and no commit. "
              "This is a FAILURE, not `signature not applicable`.")
        return False
    if not isinstance(sig, dict):
        _fail(out, "SIGNATURE-MALFORMED", "`signature` is not an object")
        return False
    missing = [k for k in SIGNATURE_FIELDS if not sig.get(k)]
    if missing:
        _fail(out, "SIGNATURE-MALFORMED",
              "signature is missing %s" % ", ".join(missing))
        return False
    if sig["alg"] not in ALGORITHMS:
        _fail(out, "ALG-UNKNOWN",
              "signature.alg=%r is not one of %s -- a verifier that does not "
              "know the algorithm has not verified anything"
              % (sig["alg"], list(ALGORITHMS)))
        return False
    return True


def _check_identity(receipt, sig, host_expected, out):
    """WHOSE evidence is this? Every failure here says `about something else`,
    never `re-measure` -- re-measuring the wrong host produces the same file."""
    prov = receipt.get("provenance")
    body_host = prov.get("host") if isinstance(prov, dict) else None
    commit = receipt.get("commit")
    ok = True
    if sig.get("host") != body_host:
        _fail(out, "SIGNED-IDENTITY-MISMATCH",
              "signature was made over host=%r but the receipt body claims "
              "host=%r -- these are different documents"
              % (sig.get("host"), body_host))
        ok = False
    if sig.get("commit") != commit:
        _fail(out, "SIGNED-IDENTITY-MISMATCH",
              "signature was made over commit=%r but the receipt body claims "
              "commit=%r" % (sig.get("commit"), commit))
        ok = False
    if key_id_host(sig["key_id"]) != body_host:
        _fail(out, "KEY-HOST-MISMATCH",
              "key_id=%r is scoped to host %r, receipt claims host=%r -- one "
              "host's key cannot attest another host's numbers"
              % (sig["key_id"], key_id_host(sig["key_id"]), body_host))
        ok = False
    if host_expected is not None and body_host != host_expected:
        _fail(out, "WRONG-HOST",
              "this receipt is from host %r; the gate was asked about %r. Your "
              "evidence is about something else -- re-measuring will not fix "
              "it, finding the right receipt will." % (body_host, host_expected))
        ok = False
    return ok


def _verify_value(receipt, sig, keys, out):
    """Was this exact document signed by the key it names?"""
    key_id = sig["key_id"]
    material = keys.get(key_id)
    if material is None:
        _fail(out, "UNKNOWN-KEY",
              "no key %r in the keyring. The verifier cannot check this "
              "receipt, so it does not pass. Known key ids: %s"
              % (key_id, sorted(keys)))
        return False
    digest = body_digest(receipt)
    if sig["body_sha256"] != digest:
        _fail(out, "FORGED",
              "signature.body_sha256=%s but the receipt body hashes to %s -- "
              "the document was modified after it was signed"
              % (sig["body_sha256"], digest))
        return False
    payload = build_payload(sig["alg"], key_id, sig["signed_at"],
                            sig["commit"], sig["host"], digest)
    expect = _mac(sig["alg"], bytes.fromhex(material), payload)
    if not hmac.compare_digest(expect, sig["value"]):
        _fail(out, "FORGED",
              "signature does not verify under key %r. Either the header "
              "(alg/key_id/signed_at/commit/host) was edited after signing, or "
              "the receipt was signed by something that does not hold this "
              "host's key." % key_id)
        return False
    return True


def _verify_freshness(receipt, commit_under_test, git_dir, out, report):
    """§4.9.1's staleness arm: `receipt.commit ⊇ commit-under-test`."""
    receipt_commit = receipt.get("commit")
    verdict, detail = commit_contains(git_dir, commit_under_test, receipt_commit)
    if verdict == "YES":
        report.append("PASS ArmC-fresh receipt.commit=%s contains "
                      "commit-under-test=%s" % (receipt_commit, commit_under_test))
        return True
    if verdict == "NO":
        _fail(out, "STALE",
              "receipt.commit=%s does NOT contain commit-under-test=%s. This "
              "measurement describes older code, so it is not evidence about "
              "this commit -- RE-MEASURE on a tree that contains it. (Not an "
              "identity problem: the host and the signature are fine.)"
              % (receipt_commit, commit_under_test))
        return False
    _fail(out, "COMMIT-UNDECIDABLE", detail)
    return False


def verify_receipt(receipt, host, commit_under_test, keyring_path,
                   git_dir, today=None):
    """Full §4.9.1 verification. Returns (failures, report_lines).

    `failures` is a list of (code, message); empty means the receipt passed.
    Ordering is load-bearing: forgery is decided BEFORE freshness, because a
    document whose signature does not verify has no trustworthy `commit` field
    for the staleness arm to read.
    """
    out = []
    report = []
    sig = receipt.get(SIGNATURE_KEY)
    if not _check_shape(sig, out):
        return out, report
    try:
        keys = load_keyring(keyring_path)
    except SigError as exc:
        _fail(out, exc.code, exc.message)
        return out, report
    identity_ok = _check_identity(receipt, sig, host, out)
    try:
        value_ok = _verify_value(receipt, sig, keys, out)
    except SigError as exc:
        _fail(out, exc.code, exc.message)
        value_ok = False
    if not (identity_ok and value_ok):
        return out, report
    report.append("PASS ArmC-sig %s signed by %s over body %s"
                  % (sig["alg"], sig["key_id"], sig["body_sha256"][:12]))
    report.append(_age_line(sig["signed_at"], today))
    _verify_freshness(receipt, commit_under_test, git_dir, out, report)
    return out, report


def _age_line(signed_at, today=None):
    """REPORT, never FAIL. A wall-clock maximum age is a policy number under
    perf-matrix.yaml's GROUNDING RULE -- it needs an author and a rationale, and
    none has been decided, so none is invented here. The number is surfaced so
    the decision has something to be made against."""
    stamp = _parse_timestamp(signed_at)
    now = (datetime.datetime.strptime(today, TIMESTAMP_FORMAT) if today
           else _utcnow())
    age_days = (now - stamp).total_seconds() / 86400.0
    return ("REPORT ArmC-sig signed_at=%s age=%.2f days (no wall-clock bound "
            "is armed: that is a POLICY number and none is decided -- "
            "freshness is enforced as commit containment)"
            % (signed_at, age_days))


# ----------------------------------------------------------------- selftest ---
# Every row states the mutation and the code it must produce. Polarity is
# checked in BOTH directions: a correctly signed, fresh receipt must PASS, and
# each mutation must fail with ITS OWN code -- not merely "fail", because a
# stale receipt reported as WRONG-HOST sends the reader to the wrong fix.

_SELFTEST_KEY_A = "a1" * 32
_SELFTEST_KEY_B = "b2" * 32


def _selftest_receipt(commit, host="lambda"):
    return {
        "spec": "APR-PERF-GATE-001 v2.2 §4.4",
        "commit": commit,
        "workload": "W1",
        "provenance": {"host": host, "compute_class": "cuda",
                       "binary_sha256": "0" * 64},
        "bands": [{"concurrency": 1, "aggregate_tok_per_sec": 100.0}],
    }


def _git(repo, *args):
    proc = subprocess.run(["git", "-C", repo] + list(args),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError("git %s failed: %s"
                           % (" ".join(args),
                              proc.stderr.decode("utf-8", "replace")))
    return proc.stdout.decode("utf-8").strip()


def _selftest_repo(tmp):
    """Two commits, C0 then C1. `--template=` keeps the caller's git templates
    and hooks out of a throwaway repo."""
    repo = os.path.join(tmp, "repo")
    os.makedirs(repo)
    _git(repo, "init", "-q", "--template=")
    _git(repo, "config", "user.email", "selftest@example.invalid")
    _git(repo, "config", "user.name", "perf-gate selftest")
    _git(repo, "config", "commit.gpgsign", "false")
    shas = []
    for i in (0, 1):
        with open(os.path.join(repo, "f%d" % i), "w", encoding="utf-8") as fh:
            fh.write("%d\n" % i)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "c%d" % i)
        shas.append(_git(repo, "rev-parse", "HEAD"))
    return repo, shas[0], shas[1]


def _selftest_cases(repo, c0, c1, keyring):
    """(name, receipt, host, commit_under_test, keyring, expected_code)."""
    fresh = sign(_selftest_receipt(c1), "lambda-selftest", _SELFTEST_KEY_A,
                 signed_at="2026-08-29T00:00:00Z")
    stale = sign(_selftest_receipt(c0), "lambda-selftest", _SELFTEST_KEY_A,
                 signed_at="2026-08-29T00:00:00Z")

    forged_body = json.loads(json.dumps(fresh))
    forged_body["bands"][0]["aggregate_tok_per_sec"] = 999.0

    forged_digest = json.loads(json.dumps(fresh))
    forged_digest["bands"][0]["aggregate_tok_per_sec"] = 999.0
    forged_digest["signature"]["body_sha256"] = body_digest(forged_digest)

    forged_when = json.loads(json.dumps(fresh))
    forged_when["signature"]["signed_at"] = "2026-08-28T00:00:00Z"

    forged_commit = json.loads(json.dumps(stale))
    forged_commit["signature"]["commit"] = c1

    wrong_key = sign(_selftest_receipt(c1), "lambda-selftest", _SELFTEST_KEY_B,
                     signed_at="2026-08-29T00:00:00Z")

    unknown_key = sign(_selftest_receipt(c1), "lambda-rotated", _SELFTEST_KEY_A,
                       signed_at="2026-08-29T00:00:00Z")

    other_host = sign(_selftest_receipt(c1, host="gx10"), "gx10-selftest",
                      _SELFTEST_KEY_B, signed_at="2026-08-29T00:00:00Z")

    unsigned = _selftest_receipt(c1)

    no_alg = json.loads(json.dumps(fresh))
    no_alg["signature"]["alg"] = "rot13"

    unknown_commit = sign(_selftest_receipt("0" * 40), "lambda-selftest",
                          _SELFTEST_KEY_A, signed_at="2026-08-29T00:00:00Z")

    return [
        # The discrimination case. If this ever fails, every row below is
        # meaningless -- they would all be "failing" for a reason unrelated to
        # what they mutate.
        ("signed_fresh_receipt_passes", fresh, "lambda", c1, keyring, None),
        ("signed_fresh_covers_ancestor", fresh, "lambda", c0, keyring, None),
        ("one_commit_stale", stale, "lambda", c1, keyring, "STALE"),
        ("unsigned_is_red", unsigned, "lambda", c1, keyring, "UNSIGNED"),
        ("forged_body", forged_body, "lambda", c1, keyring, "FORGED"),
        ("forged_body_and_digest", forged_digest, "lambda", c1, keyring, "FORGED"),
        ("forged_signed_at", forged_when, "lambda", c1, keyring, "FORGED"),
        ("forged_commit_in_header", forged_commit, "lambda", c1, keyring,
         "SIGNED-IDENTITY-MISMATCH"),
        ("signed_by_another_key", wrong_key, "lambda", c1, keyring, "FORGED"),
        ("key_id_not_in_keyring", unknown_key, "lambda", c1, keyring, "UNKNOWN-KEY"),
        ("receipt_from_another_host", other_host, "lambda", c1, keyring, "WRONG-HOST"),
        ("unknown_algorithm", no_alg, "lambda", c1, keyring, "ALG-UNKNOWN"),
        ("commit_not_in_checkout", unknown_commit, "lambda", c1, keyring,
         "COMMIT-UNDECIDABLE"),
        ("keyring_absent_is_red", fresh, "lambda", c1,
         os.path.join(repo, "no-such-keyring"), "NO-KEYRING"),
        ("keyring_path_unset_is_red", fresh, "lambda", c1, None, "NO-KEYRING"),
    ]


def _run_selftest():
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="receipt-sig-selftest-")
    passed = broken = 0
    try:
        repo, c0, c1 = _selftest_repo(tmp)
        keyring = os.path.join(tmp, "keyring")
        with open(keyring, "w", encoding="utf-8") as fh:
            fh.write("# selftest keyring\n")
            fh.write("lambda-selftest %s\n" % _SELFTEST_KEY_A)
            fh.write("gx10-selftest %s\n" % _SELFTEST_KEY_B)
        passed, broken = _run_verify_table(_selftest_cases(repo, c0, c1,
                                                            keyring), repo)
        sign_ok, sign_broken = _selftest_signer_refusals(c1)
        passed += sign_ok
        broken += sign_broken
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  %d passed, %d broken" % (passed, broken))
    return 0 if broken == 0 else 1


def _run_verify_table(cases, repo):
    """Run the verification case table. Returns (passed, broken).

    Each row asserts its own CODE, not merely that it failed: a stale receipt
    reported as WRONG-HOST would send the reader to the wrong fix.
    """
    passed = broken = 0
    for name, receipt, host, cut, keyring, want in cases:
        fails, _report = verify_receipt(receipt, host, cut, keyring, repo,
                                        today="2026-08-29T00:00:00Z")
        got = fails[0][0] if fails else None
        if got == want:
            passed += 1
            print("  ok    %-34s expect=%s" % (name, want or "PASS"))
        else:
            broken += 1
            print("  BROKE %-34s expected %s got %s (%s)"
                  % (name, want or "PASS", got,
                     fails[0][1] if fails else "no failure"))
    return passed, broken


def _selftest_signer_refusals(commit):
    """The signer refuses at source. Returns (passed, broken)."""
    passed = broken = 0
    rows = [
        ("sign_refuses_foreign_host",
         _selftest_receipt(commit, host="gx10"), "lambda-selftest",
         _SELFTEST_KEY_A, "KEY-HOST-MISMATCH"),
        ("sign_refuses_commitless_receipt",
         {"provenance": {"host": "lambda"}}, "lambda-selftest",
         _SELFTEST_KEY_A, "NO-COMMIT"),
        ("sign_refuses_short_key",
         _selftest_receipt(commit), "lambda-selftest", "ab", "KEY-MALFORMED"),
    ]
    for name, receipt, key_id, key, want in rows:
        try:
            sign(receipt, key_id, key)
            got = None
        except SigError as exc:
            got = exc.code
        if got == want:
            passed += 1
            print("  ok    %-34s expect=%s" % (name, want))
        else:
            broken += 1
            print("  BROKE %-34s expected %s got %s" % (name, want, got))
    return passed, broken


# --------------------------------------------------------------------- cli ---

def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def _mode_sign(argv):
    src = _arg(argv, "--in")
    dst = _arg(argv, "--out")
    key_id = _arg(argv, "--key-id")
    keyring = _arg(argv, "--keyring", os.environ.get("APR_PERF_RECEIPT_KEYRING"))
    if not src or not dst or not key_id:
        sys.stderr.write("usage: receipt_sig.py --sign --in R --out S "
                         "--key-id ID [--keyring K]\n")
        return 2
    try:
        keys = load_keyring(keyring)
        material = keys.get(key_id)
        if material is None:
            raise SigError("UNKNOWN-KEY",
                           "no key %r in %r -- this host cannot sign as that "
                           "identity" % (key_id, keyring))
        signed = sign(_read_json(src), key_id, material,
                      signed_at=_arg(argv, "--signed-at"))
    except SigError as exc:
        sys.stderr.write("REFUSED %s %s\n" % (exc.code, exc.message))
        return 1
    except (OSError, ValueError) as exc:
        sys.stderr.write("cannot read %s: %s\n" % (src, exc))
        return 2
    with open(dst, "w", encoding="utf-8") as handle:
        json.dump(signed, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("signed %s -> %s (key_id=%s body_sha256=%s)"
          % (src, dst, key_id, signed[SIGNATURE_KEY]["body_sha256"]))
    return 0


def _mode_verify(argv):
    path = _arg(argv, "--verify")
    host = _arg(argv, "--host")
    commit = _arg(argv, "--commit")
    keyring = _arg(argv, "--keyring", os.environ.get("APR_PERF_RECEIPT_KEYRING"))
    git_dir = _arg(argv, "--git-dir", os.getcwd())
    if not path or not host or not commit:
        sys.stderr.write("usage: receipt_sig.py --verify R --host H --commit "
                         "SHA [--keyring K] [--git-dir D]\n")
        return 2
    try:
        receipt = _read_json(path)
    except (OSError, ValueError) as exc:
        sys.stderr.write("cannot read %s: %s\n" % (path, exc))
        return 2
    fails, report = verify_receipt(receipt, host, commit, keyring, git_dir)
    for line in report:
        print(line)
    for code, message in fails:
        print("FAIL ArmC-sig %s: %s" % (code, message))
    return 1 if fails else 0


def main(argv):
    if "--selftest" in argv:
        return _run_selftest()
    if "--sign" in argv:
        return _mode_sign(argv)
    if "--verify" in argv:
        return _mode_verify(argv)
    sys.stderr.write(__doc__.split("Usage:", 1)[-1])
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
