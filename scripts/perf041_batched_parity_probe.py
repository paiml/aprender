#!/usr/bin/env python3
"""PP-26 batch-invariance witness (FALSIFY-CB-006), at TOKEN level.

WHAT CHANGED AND WHY
--------------------
The previous revision of this probe compared the first 40 CHARACTERS of the
decoded text of two non-streaming completions, at `max_tokens=400` with no
`seed` and no `ignore_eos`. Three things were wrong with that, all of them
recorded in contracts/continuous-batching-v1.yaml's own measurement blocks:

  1. 40 characters is roughly ten tokens. PP-26 requires agreement to a
     DECLARED divergence point (>= 64 tokens, `witness.min_agree_tokens` in
     scripts/perf-matrix.yaml). A ten-token witness cannot decide a 64-token
     rule, and a tokenizer round-trip can both mask a divergence (two token
     sequences decoding to the same prefix) and mint one.
  2. The sampler was not pinned, so `completion_tokens` varied 286 vs 319 on
     two consecutive m=1 calls of the same greedy prompt. A comparison whose
     two sides generated different numbers of tokens is not a comparison.
  3. IT COULD NOT FAIL ON A QUIET BOX. A band in which no `m > 1` batch formed
     was `continue`d, `failures` stayed 0, and the run printed GREEN and exited
     0 having witnessed nothing at all. That is the blacklist-clause class:
     the rule fired only on the branch it named and fell open on the
     complement. The scheduler's default batch window is 0 ms, so "no batch
     formed" is the LIKELY nightly outcome, not the exotic one.

  4. Batch attribution was cumulative: `max_batch_formed` re-read the WHOLE
     server log on every band, so the c=4 band was credited with a batch that
     formed during c=2.

This revision fixes all four:

  * every request carries `temperature`, `seed`, `ignore_eos` and
    `max_tokens` from `scripts/perf-matrix.yaml` (`protocol.sampler`,
    `protocol.n_predict`) and `stream: true`;
  * the comparison is per SSE content chunk. One content chunk is the server's
    delta for one generated token (crates/aprender-serve/src/api/
    chat_completions_stream.rs `streaming_text_deltas`, one push per token id).
    Where a multi-byte codepoint spans two token ids the deltas group, which
    makes the chunk count a LOWER bound on the token count -- and a sample
    whose chunk count is not exactly `n_predict` is REFUSED (PP-28), so the
    grouping case is reported as UNMEASURABLE rather than silently compared;
  * `divergence_at` is the index of the first differing chunk; a band PASSes
    iff `divergence_at >= declared_min`;
  * engagement is attributed PER BAND: the server log's byte offset is
    recorded before the band is fired and only the bytes after it are scanned;
    a band at `c > 1` with `m_formed < 2` is UNMEASURABLE, never a pass;
  * the run exits 2 (never 0) when no `c > 1` band was measured, or when any
    band was UNMEASURABLE.

EXIT CODES
    0  PASS         every band measured and agreeing to the declared point
    1  FAIL         a code defect: a batched slot diverged before that point
    2  UNMEASURABLE the run could not decide (env, model, harness, no batch)

A guard that names a code cause for a box it could not evaluate has fired
three times in this repo in one day, so 1 and 2 are kept apart. Both are RED
in CI: PP-26 says an absent witness makes the band INVALID-CORRECTNESS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import threading
import urllib.error
import urllib.request

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_UNMEASURABLE = 2

WITNESS_VERSION = 1
MARKER = "PP-26-WITNESS: "

# `[PMAT-044] Batch m=N done`, emitted by
# crates/aprender-serve/src/api/cuda_batch_scheduler.rs.
BATCH_RE = re.compile(r"Batch m=(\d+) done")

# Recorded, not inherited. PERF-053 sets CUBLAS_GEMM_THRESHOLD at step scope and
# contracts/continuous-batching-v1.yaml records that knob (with FP8_DECODE=0)
# making the m=1 reference garbage -- so the witness states which knobs were
# live, including the ones it expects to be unset (recorded as null).
ENV_KEYS = (
    "CUDA_BATCH_WINDOW_MS",
    "CUBLAS_GEMM_THRESHOLD",
    "FP8_DECODE",
    "FP8_PREFILL",
    "BATCHED_PREFILL",
    "MULTI_PROMPT_PREFILL",
    "APR_DECODE_GEMM",
    "ITERATION_SCHEDULER",
    "FUSED_GATE_UP",
)

# Fallbacks. PP-33 says every number a gate compares against lives in
# perf-matrix.yaml; these exist only so the probe can still RUN (and say so
# loudly) on a tree whose matrix has not been extended yet. Using one prints a
# line containing the exact words "matrix block absent".
FALLBACK_MIN_AGREE = 64
FALLBACK_SEED = 0
FALLBACK_TEMPERATURE = 0.0
FALLBACK_N_PREDICT = 128
FALLBACK_LADDER = (1, 4, 8, 16)

DEFAULT_PROMPT = "Write an essay on compilers."
REQUEST_TIMEOUT_S = 300.0


# --------------------------------------------------------------------------
# policy (perf-matrix.yaml)
# --------------------------------------------------------------------------
def load_matrix(path: str) -> tuple[dict, list[str]]:
    """The matrix as a dict, plus notes describing anything that was missing."""
    notes: list[str] = []
    try:
        import yaml  # PyYAML; installed on every host that runs this lane.
    except ImportError:
        notes.append(
            "matrix block absent: PyYAML is not importable, so "
            f"{path} could not be read"
        )
        return {}, notes
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except OSError as exc:
        notes.append(f"matrix block absent: cannot read {path} ({exc})")
        return {}, notes
    except yaml.YAMLError as exc:
        notes.append(f"matrix block absent: {path} does not parse ({exc})")
        return {}, notes
    if not isinstance(loaded, dict):
        notes.append(f"matrix block absent: {path} is not a mapping")
        return {}, notes
    return loaded, notes


def _block(matrix: dict, key: str) -> dict:
    value = matrix.get(key)
    return value if isinstance(value, dict) else {}


def resolve_policy(matrix: dict, notes: list[str], path: str) -> dict:
    """declared_min, sampler and n_predict -- from the matrix, or announced."""
    witness = _block(matrix, "witness")
    protocol = _block(matrix, "protocol")
    sampler = _block(protocol, "sampler")
    ladder_block = _block(matrix, "ladder")

    declared_min = witness.get("min_agree_tokens")
    if not isinstance(declared_min, int) or isinstance(declared_min, bool):
        notes.append(
            f"matrix block absent: {path} witness.min_agree_tokens -- "
            f"falling back to declared_min={FALLBACK_MIN_AGREE} (PP-33 debt)"
        )
        declared_min = FALLBACK_MIN_AGREE

    seed = sampler.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        notes.append(
            f"matrix block absent: {path} protocol.sampler.seed -- "
            f"falling back to seed={FALLBACK_SEED} (PP-33 debt)"
        )
        seed = FALLBACK_SEED

    temperature = sampler.get("temperature")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        notes.append(
            f"matrix block absent: {path} protocol.sampler.temperature -- "
            f"falling back to temperature={FALLBACK_TEMPERATURE} (PP-33 debt)"
        )
        temperature = FALLBACK_TEMPERATURE

    ignore_eos = sampler.get("ignore_eos")
    if not isinstance(ignore_eos, bool):
        notes.append(
            f"matrix block absent: {path} protocol.sampler.ignore_eos -- "
            "falling back to ignore_eos=true (PP-33 debt)"
        )
        ignore_eos = True

    n_predict = protocol.get("n_predict")
    if not isinstance(n_predict, int) or isinstance(n_predict, bool):
        notes.append(
            f"matrix block absent: {path} protocol.n_predict -- "
            f"falling back to n_predict={FALLBACK_N_PREDICT} (PP-33 debt)"
        )
        n_predict = FALLBACK_N_PREDICT

    declared = ladder_block.get("declared")
    if not (isinstance(declared, list) and declared
            and all(isinstance(c, int) and not isinstance(c, bool) and c >= 1
                    for c in declared)):
        notes.append(
            f"matrix block absent: {path} ladder.declared -- "
            f"falling back to ladder={list(FALLBACK_LADDER)} (PP-33 debt)"
        )
        declared = list(FALLBACK_LADDER)

    return {
        "declared_min": declared_min,
        "n_predict": n_predict,
        "ladder": list(declared),
        "sampler": {
            "temperature": float(temperature),
            "seed": int(seed),
            "ignore_eos": bool(ignore_eos),
            "n_predict": int(n_predict),
        },
    }


# --------------------------------------------------------------------------
# HTTP / SSE
# --------------------------------------------------------------------------
def request_body(prompt: str, sampler: dict) -> bytes:
    return json.dumps(
        {
            "model": "q",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": sampler["n_predict"],
            "temperature": sampler["temperature"],
            "seed": sampler["seed"],
            "ignore_eos": sampler["ignore_eos"],
            "stream": True,
        }
    ).encode()


def parse_sse(reader) -> tuple[list[str], str | None, int | None]:
    """(content chunks, finish_reason, usage.completion_tokens or None).

    One content chunk == the delta the server emitted for one generated token.
    The role-only opening chunk and the finish-reason chunk carry no content
    and are not counted.
    """
    chunks: list[str] = []
    finish: str | None = None
    usage_tokens: int | None = None
    for raw in reader:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        usage = event.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
            usage_tokens = usage["completion_tokens"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                chunks.append(content)
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return chunks, finish, usage_tokens


def stream_completion(url: str, prompt: str, sampler: dict) -> dict:
    """One streaming completion, as chunks. Never raises; errors are data."""
    req = urllib.request.Request(
        url,
        data=request_body(prompt, sampler),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            chunks, finish, usage_tokens = parse_sse(resp)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "chunks": [],
                "finish_reason": None, "completion_tokens": None,
                "token_count_source": None}
    # `usage` on the terminal SSE chunk is PP-27 / §12 row 0b work and is not
    # emitted today; until it is, the chunk count IS the token count by the
    # one-chunk-per-token property documented at the top of this file. Which
    # one was used is recorded rather than assumed.
    if usage_tokens is not None:
        return {"error": None, "chunks": chunks, "finish_reason": finish,
                "completion_tokens": usage_tokens, "token_count_source": "usage"}
    return {"error": None, "chunks": chunks, "finish_reason": finish,
            "completion_tokens": len(chunks), "token_count_source": "sse_chunks"}


def fire_band(url: str, prompt: str, sampler: dict, c: int) -> list[dict]:
    """c simultaneous streaming requests; results in slot order."""
    out: dict[int, dict] = {}
    lock = threading.Lock()

    def go(i: int) -> None:
        result = stream_completion(url, prompt, sampler)
        with lock:
            out[i] = result

    threads = [threading.Thread(target=go, args=(i,)) for i in range(c)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [out[i] for i in sorted(out)]


# --------------------------------------------------------------------------
# engagement
# --------------------------------------------------------------------------
def log_offset(server_log: str) -> int:
    try:
        return os.path.getsize(server_log)
    except OSError:
        return 0


def max_batch_formed(server_log: str, offset: int) -> int:
    """Largest m in `Batch m=N done` AFTER `offset` bytes.

    Scanning from an offset is what makes engagement attributable to THIS
    band. The previous revision re-read the whole file, so a batch formed
    during c=2 credited every later band.
    """
    try:
        with open(server_log, "rb") as handle:
            handle.seek(offset)
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return 0
    sizes = [int(m) for m in BATCH_RE.findall(text)]
    return max(sizes) if sizes else 0


# --------------------------------------------------------------------------
# comparison (pure)
# --------------------------------------------------------------------------
def divergence_index(got: list[str], ref: list[str]) -> int:
    """Index of the first differing chunk; len(ref) when equal over its length."""
    limit = min(len(got), len(ref))
    for i in range(limit):
        if got[i] != ref[i]:
            return i
    if len(got) != len(ref):
        return limit
    return limit


def evaluate_band(c: int, m_formed: int, samples: list[dict],
                  ref_chunks: list[str], declared_min: int,
                  n_predict: int) -> dict:
    """One band's verdict. Pure: no I/O, so the case table can drive it."""
    slots: list[dict] = []
    refusals: list[str] = []
    for i, sample in enumerate(samples):
        slot = {"i": i, "completion_tokens": sample.get("completion_tokens"),
                "finish_reason": sample.get("finish_reason"),
                "token_count_source": sample.get("token_count_source"),
                "agree_to": None, "refused": None}
        if sample.get("error"):
            slot["refused"] = sample["error"]
            refusals.append(f"slot {i}: {sample['error']}")
        elif sample.get("completion_tokens") != n_predict:
            slot["refused"] = "short"
            refusals.append(
                f"slot {i}: completion_tokens={sample.get('completion_tokens')} "
                f"!= n_predict={n_predict} (PP-28)"
            )
        else:
            slot["agree_to"] = divergence_index(sample["chunks"], ref_chunks)
        slots.append(slot)

    if refusals:
        return {"c": c, "m_formed": m_formed, "result": "UNMEASURABLE",
                "divergence_at": None, "declared_min": declared_min,
                "reason": "; ".join(refusals), "slots": slots}
    # A batch is by definition two or more sequences decoded in one step, so
    # `m_formed < 2` is not a threshold — it is the word "batch". At c=1 no
    # batch can form and none is required: PP-26 exempts c=1, whose witness is
    # the m=1 reference's agreement with itself.
    if c > 1 and m_formed < 2:
        return {"c": c, "m_formed": m_formed, "result": "UNMEASURABLE",
                "divergence_at": None, "declared_min": declared_min,
                "reason": (f"no batch with m>1 formed in this band's window "
                           f"(max m={m_formed}); the batched path was never "
                           f"exercised, so nothing about it was witnessed"),
                "slots": slots}
    if not slots:
        return {"c": c, "m_formed": m_formed, "result": "UNMEASURABLE",
                "divergence_at": None, "declared_min": declared_min,
                "reason": "no samples returned", "slots": slots}
    divergence_at = min(slot["agree_to"] for slot in slots)
    result = "PASS" if divergence_at >= declared_min else "FAIL"
    return {"c": c, "m_formed": m_formed, "result": result,
            "divergence_at": divergence_at, "declared_min": declared_min,
            "reason": None, "slots": slots}


def exit_code(bands: list[dict]) -> int:
    """1 on any FAIL; 2 on any UNMEASURABLE or a vacuous run; else 0.

    "Vacuous" is the case the old probe returned 0 for: no band at c > 1 was
    ever MEASURED, so the batched path was not witnessed at all.
    """
    if any(band["result"] == "FAIL" for band in bands):
        return EXIT_FAIL
    measured_batched = any(band["c"] > 1 and band["result"] in ("PASS", "FAIL")
                           for band in bands)
    if any(band["result"] == "UNMEASURABLE" for band in bands):
        return EXIT_UNMEASURABLE
    if not measured_batched:
        return EXIT_UNMEASURABLE
    return EXIT_PASS


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def sha256_file(path: str | None) -> str | None:
    if not path:
        return None
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def env_block() -> dict:
    return {key: os.environ.get(key) for key in ENV_KEYS}


def print_marker(host: str, band: dict) -> None:
    divergence = band["divergence_at"]
    print(f"{MARKER}host={host} c={band['c']} m_formed={band['m_formed']} "
          f"result={band['result']} "
          f"divergence_at={'none' if divergence is None else divergence}")


def run_probe(url: str, server_log: str, prompt: str, ladder: list[int],
              policy: dict, host: str, provenance: dict,
              notes: list[str]) -> tuple[int, dict]:
    sampler = policy["sampler"]
    declared_min = policy["declared_min"]
    n_predict = policy["n_predict"]

    witness = {
        "witness_version": WITNESS_VERSION,
        "probe": "perf041",
        "host": host,
        "commit": provenance.get("commit"),
        "binary_sha256": provenance.get("binary_sha256"),
        "model": {"path": provenance.get("model_path"),
                  "sha256": provenance.get("model_sha256")},
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "sampler": dict(sampler),
        "declared_min": declared_min,
        "env": env_block(),
        "notes": list(notes),
        "reference": {"tokens": None, "stable": False,
                      "token_count_source": None},
        "bands": [],
    }
    for note in notes:
        print(f"perf041: {note}")

    print("== m=1 reference (positive control) ==")
    first = stream_completion(url, prompt, sampler)
    second = stream_completion(url, prompt, sampler)
    for label, sample in (("ref#1", first), ("ref#2", second)):
        print(f"  {label} error={sample['error']} "
              f"finish={sample['finish_reason']} "
              f"tokens={sample['completion_tokens']} "
              f"source={sample['token_count_source']}")
    witness["reference"]["tokens"] = first["completion_tokens"]
    witness["reference"]["token_count_source"] = first["token_count_source"]

    for label, sample in (("ref#1", first), ("ref#2", second)):
        if sample["error"]:
            witness["reference"]["reason"] = f"{label}: {sample['error']}"
            print(f"UNMEASURABLE: the m=1 reference could not be taken "
                  f"({label}: {sample['error']}). Not a code verdict.")
            return EXIT_UNMEASURABLE, witness
        if sample["completion_tokens"] != n_predict:
            witness["reference"]["reason"] = (
                f"{label}: completion_tokens={sample['completion_tokens']} "
                f"!= n_predict={n_predict}")
            print(f"UNMEASURABLE: {label} returned "
                  f"{sample['completion_tokens']} tokens against a pinned "
                  f"n_predict={n_predict}; with ignore_eos on the wire this is "
                  f"a harness/server disagreement (PP-28), not a batching "
                  f"verdict.")
            return EXIT_UNMEASURABLE, witness

    ref_agreement = divergence_index(second["chunks"], first["chunks"])
    witness["reference"]["self_divergence_at"] = ref_agreement
    if ref_agreement < declared_min:
        witness["reference"]["reason"] = (
            f"the two m=1 references diverge at chunk {ref_agreement} "
            f"< declared_min {declared_min}")
        print(f"UNMEASURABLE: the m=1 reference is not reproducible on this "
              f"box/model — two solo calls diverge at token {ref_agreement}, "
              f"below the declared point {declared_min}. A batched difference "
              f"could not be attributed to batching. Not a code verdict.")
        return EXIT_UNMEASURABLE, witness
    witness["reference"]["stable"] = True
    ref_chunks = first["chunks"]
    print(f"  reference stable to chunk {ref_agreement} "
          f"(declared_min={declared_min})")

    for c in ladder:
        print(f"== concurrency {c} ==")
        offset = log_offset(server_log)
        samples = fire_band(url, prompt, sampler, c)
        m_formed = max_batch_formed(server_log, offset)
        band = evaluate_band(c, m_formed, samples, ref_chunks, declared_min,
                             n_predict)
        witness["bands"].append(band)
        for slot in band["slots"]:
            print(f"  [{slot['i']}] tokens={slot['completion_tokens']} "
                  f"finish={slot['finish_reason']} "
                  f"agree_to={slot['agree_to']} refused={slot['refused']}")
        if band["reason"]:
            print(f"  {band['result']}: {band['reason']}")
        print_marker(host, band)

    rc = exit_code(witness["bands"])
    print()
    if rc == EXIT_FAIL:
        print("FALSIFY-CB-006 RED: a batched slot diverged from the m=1 "
              "reference before the declared point on identical greedy input.")
    elif rc == EXIT_UNMEASURABLE:
        print("PP-26 UNMEASURABLE: no c>1 band was measured, or a band could "
              "not be decided. This is NOT a pass — an absent witness makes "
              "the band INVALID-CORRECTNESS (PP-26, P-4).")
    else:
        print("FALSIFY-CB-006 GREEN: every batched slot agrees with the m=1 "
              "reference to the declared point.")
    return rc, witness


def write_witness(path: str | None, witness: dict) -> None:
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(witness, handle, indent=2, sort_keys=True)
        handle.write("\n")


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------
SELFTEST_NAMES = (
    "witness_constant_token_m3",
    "witness_no_batch_formed_is_unmeasurable",
    "witness_identical_128_ok",
    "witness_cross_band_log_not_credited",
    "witness_short_reference_is_unmeasurable",
    # The two rows below exist because the five above did NOT kill every
    # mutant. Removing the per-slot PP-28 refusal (evaluate_band) and removing
    # the vacuity rule (exit_code) both left the table 5/5 GREEN, which is the
    # "a gate that cannot fail" class this probe is being rewritten to close.
    # Measured, not argued -- see the PR body's mutation transcript.
    "witness_short_slot_is_unmeasurable",
    "witness_c1_only_is_not_a_pass",
)


def _replay_server(plan, log_path: str):
    """A threading HTTP server replaying canned SSE, appending to a fake log.

    `plan(n)` is called with the 0-based global request index and returns
    `(chunks, log_line_or_None)`. Appending the log line from inside the
    handler is what a real server does, so the probe's before-band offset and
    after-band scan are exercised for real rather than simulated.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    counter = {"n": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):  # noqa: D102 - silence the access log
            return

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's name
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            with lock:
                index = counter["n"]
                counter["n"] += 1
            chunks, log_line = plan(index)
            if log_line:
                with lock, open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(log_line + "\n")
                    handle.flush()
            body = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n']
            for chunk in chunks:
                event = {"choices": [{"delta": {"content": chunk}}]}
                body.append(("data: " + json.dumps(event) + "\n\n").encode())
            body.append(
                b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n')
            body.append(b"data: [DONE]\n\n")
            payload = b"".join(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _canned(n: int, seed: str) -> list[str]:
    """n distinct chunks -- a stand-in for a coherent 128-token continuation."""
    return [f"{seed}{i} " for i in range(n)]


def _selftest_case(name: str, plan, ladder: list[int], declared_min: int,
                   n_predict: int) -> tuple[int, dict]:
    import tempfile

    tmp = tempfile.mkdtemp(prefix="perf041-selftest-")
    log_path = os.path.join(tmp, "server.log")
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("server starting\n")
    server, _thread = _replay_server(plan, log_path)
    host, port = server.server_address[0], server.server_address[1]
    url = f"http://{host}:{port}/v1/chat/completions"
    policy = {
        "declared_min": declared_min,
        "n_predict": n_predict,
        "ladder": ladder,
        "sampler": {"temperature": 0.0, "seed": 0, "ignore_eos": True,
                    "n_predict": n_predict},
    }
    try:
        rc, witness = run_probe(url, log_path, DEFAULT_PROMPT, ladder, policy,
                                f"selftest-{name}",
                                {"commit": "selftest", "binary_sha256": None,
                                 "model_path": None, "model_sha256": None},
                                [])
    finally:
        server.shutdown()
        server.server_close()
    return rc, witness


def selftest() -> int:
    """The case table. Names are §6/PP-29 contract; do not rename casually."""
    n_predict = 8          # the shape, not the size: 8 chunks stand in for 128
    declared_min = 4       # ... and 4 for the matrix's 64
    passed = 0
    broken = 0

    def row(name: str, expect_rc: int, got_rc: int, extra_ok: bool = True,
            extra: str = "") -> None:
        nonlocal passed, broken
        if got_rc == expect_rc and extra_ok:
            print(f"  ok    {name:<40} expect=exit{expect_rc}")
            passed += 1
        else:
            print(f"  BROKE {name:<40} expected exit{expect_rc} got exit{got_rc}"
                  f"{(' ' + extra) if extra else ''}")
            broken += 1

    good = _canned(n_predict, "tok")
    garbage = ["!"] * n_predict

    # 1. #2753's signature: `[PMAT-044] Batch m=3 done` in the log and every
    #    slot returning the same constant token. MUST be exit 1.
    def plan_constant(index: int):
        if index < 2:
            return good, None
        return garbage, ("[PMAT-044] Batch m=3 done" if index == 2 else None)

    rc, _ = _selftest_case("witness_constant_token_m3", plan_constant, [3],
                           declared_min, n_predict)
    row("witness_constant_token_m3", EXIT_FAIL, rc)

    # 2. THE ANTI-FAIL-OPEN CASE. Slots are byte-identical to the reference, so
    #    the comparison would pass -- but no m>1 batch ever formed, so the
    #    batched path was not exercised and there is nothing to pass. The old
    #    probe returned 0 here.
    def plan_no_batch(index: int):
        return good, ("[PMAT-044] Batch m=1 done" if index == 2 else None)

    rc, _ = _selftest_case("witness_no_batch_formed_is_unmeasurable",
                           plan_no_batch, [4], declared_min, n_predict)
    row("witness_no_batch_formed_is_unmeasurable", EXIT_UNMEASURABLE, rc)

    # 3. The must-not-fire fixture: m=1 and an m=4 batch agree for the whole
    #    generation.
    def plan_ok(index: int):
        return good, ("[PMAT-044] Batch m=4 done" if index == 2 else None)

    rc, _ = _selftest_case("witness_identical_128_ok", plan_ok, [4],
                           declared_min, n_predict)
    row("witness_identical_128_ok", EXIT_PASS, rc)

    # 4. A batch formed during the c=2 window must NOT credit the c=4 window.
    #    Two refs, then 2 slots at c=2, then 4 slots at c=4.
    def plan_cross(index: int):
        return good, ("[PMAT-044] Batch m=2 done" if index == 2 else None)

    rc, witness = _selftest_case("witness_cross_band_log_not_credited",
                                 plan_cross, [2, 4], declared_min, n_predict)
    by_c = {band["c"]: band for band in witness["bands"]}
    credited_ok = (by_c.get(2, {}).get("m_formed") == 2
                   and by_c.get(4, {}).get("m_formed") == 0
                   and by_c.get(4, {}).get("result") == "UNMEASURABLE")
    row("witness_cross_band_log_not_credited", EXIT_UNMEASURABLE, rc,
        credited_ok,
        extra=f"(c=2 m_formed={by_c.get(2, {}).get('m_formed')}, "
              f"c=4 m_formed={by_c.get(4, {}).get('m_formed')} "
              f"result={by_c.get(4, {}).get('result')})")

    # 5. A reference that did not reach n_predict is refused (PP-28), not
    #    compared against.
    def plan_short(index: int):
        if index < 2:
            return good[:-1], None
        return good, ("[PMAT-044] Batch m=4 done" if index == 2 else None)

    rc, _ = _selftest_case("witness_short_reference_is_unmeasurable",
                           plan_short, [4], declared_min, n_predict)
    row("witness_short_reference_is_unmeasurable", EXIT_UNMEASURABLE, rc)

    # 6. The reference is fine and an m=4 batch DID form, but the batched
    #    slots stopped short of n_predict. Comparing a 7-chunk answer against
    #    an 8-chunk reference and calling the shared prefix "agreement" is the
    #    length confound PP-28 exists to refuse. Without this row, deleting the
    #    per-slot refusal in evaluate_band leaves the table 5/5 GREEN.
    def plan_short_slot(index: int):
        if index < 2:
            return good, None
        return good[:-1], ("[PMAT-044] Batch m=4 done" if index == 2 else None)

    rc, _ = _selftest_case("witness_short_slot_is_unmeasurable",
                           plan_short_slot, [4], declared_min, n_predict)
    row("witness_short_slot_is_unmeasurable", EXIT_UNMEASURABLE, rc)

    # 7. A ladder with no c>1 band witnesses nothing about batching, however
    #    green each of its bands is. Without this row, deleting the vacuity
    #    rule in exit_code leaves the table 5/5 GREEN -- the exact defect
    #    (`return 0` on a run that formed no batch) this rewrite removes.
    def plan_c1_only(_index: int):
        return good, None

    rc, witness_c1 = _selftest_case("witness_c1_only_is_not_a_pass",
                                    plan_c1_only, [1], declared_min, n_predict)
    c1_passed = any(band["c"] == 1 and band["result"] == "PASS"
                    for band in witness_c1["bands"])
    row("witness_c1_only_is_not_a_pass", EXIT_UNMEASURABLE, rc, c1_passed,
        extra="(the c=1 band itself must still be PASS, so the exit code is "
              "carried by the vacuity rule and not by a band verdict)")

    print(f"  {passed} passed, {broken} broken")
    return 0 if broken == 0 else 1


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PP-26 token-level batch-invariance witness (FALSIFY-CB-006)")
    parser.add_argument("--url")
    parser.add_argument("--server-log")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--ladder", type=int, nargs="+", default=None,
                        help="bands to fire; default: perf-matrix ladder.declared")
    parser.add_argument("--matrix", default=None,
                        help="path to scripts/perf-matrix.yaml")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="write witness.json here")
    parser.add_argument("--host", default=None)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--binary", default=None,
                        help="the apr binary under test (sha256 recorded)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--list-selftests", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_selftests:
        for name in SELFTEST_NAMES:
            print(name)
        return 0
    if args.selftest:
        return selftest()

    if not args.url or not args.server_log:
        parser.error("--url and --server-log are required outside --selftest")

    here = os.path.dirname(os.path.abspath(__file__))
    matrix_path = args.matrix or os.path.join(here, "perf-matrix.yaml")
    matrix, notes = load_matrix(matrix_path)
    policy = resolve_policy(matrix, notes, matrix_path)
    ladder = args.ladder if args.ladder else policy["ladder"]

    host = args.host or os.environ.get("PERF041_HOST") or socket.gethostname()
    provenance = {
        "commit": args.commit or os.environ.get("PERF041_COMMIT"),
        "binary_sha256": sha256_file(args.binary),
        "model_path": args.model,
        "model_sha256": sha256_file(args.model),
    }
    rc, witness = run_probe(args.url, args.server_log, args.prompt, ladder,
                            policy, host, provenance, notes)
    witness["exit"] = rc
    write_witness(args.json_out, witness)
    return rc


if __name__ == "__main__":
    sys.exit(main())
