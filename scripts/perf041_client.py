#!/usr/bin/env python3
"""PERF-041 closed-loop measurement client (APR-PERF-GATE-001 v2.2 §4.4).

Why this exists rather than reusing scripts/perf000_serialization_probe.sh:
that script fires N requests once, waits for all of them, and divides wall
times. It takes no warmup per band, no quiesce, ONE replicate, and -- the part
that matters -- it never counts a single token. `wall(c)/wall(1)` measures
serialization only if both bands generated the same number of tokens, and a
harness that does not count them cannot know. This one counts them and reports
the token-normalised index beside the wall-clock one, so a divergence in
generation length shows up as a gap between the two rather than as
"serialization".

§4.4.1 client model: closed-loop, fixed concurrency c, external HTTP.
§4.4.3 metrics: agg_tok_s is wall-clock aggregate, never a mean of rates.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field

TIMEOUT_S = 120.0  # §4.4.3: hard per-request timeout
QUIESCE_S = 5.0    # §4.4.2: warmup gate


def one_request(url: str, prompt: str, max_tokens: int) -> tuple[float, float, int]:
    """Issue one non-streaming completion. Returns (start, end, completion_tokens)."""
    body = json.dumps(
        {
            "model": "q",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        payload = json.load(resp)
    end = time.monotonic()
    # §4.4.6: token counting is declared, not assumed. If the server does not
    # report it, the sample is refused rather than silently replaced by a guess
    # -- a ratio between two bands counted differently is not a ratio.
    usage = payload.get("usage") or {}
    if "completion_tokens" not in usage:
        raise RuntimeError("response carries no usage.completion_tokens (§4.4.6)")
    return start, end, int(usage["completion_tokens"])


@dataclass
class BandState:
    """Shared mutable state for one band's workers."""

    url: str
    prompt: str
    max_tokens: int
    warmup_per_worker: int
    concurrency: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop: threading.Event = field(default_factory=threading.Event)
    warmed: threading.Event = field(default_factory=threading.Event)
    warmed_count: int = 0
    sampling_start: float = 0.0
    samples: list[tuple[float, float, int]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def record_warm(self) -> None:
        with self.lock:
            self.warmed_count += 1
            if self.warmed_count >= self.concurrency:
                self.warmed.set()

    def record_sample(self, sample: tuple[float, float, int]) -> None:
        with self.lock:
            self.samples.append(sample)

    def record_error(self, exc: BaseException) -> None:
        with self.lock:
            self.errors.append(repr(exc))
        self.stop.set()
        self.warmed.set()

    def count(self) -> int:
        with self.lock:
            return len(self.samples)


def _warm_up(band: BandState) -> None:
    for _ in range(band.warmup_per_worker):
        one_request(band.url, band.prompt, band.max_tokens)
    band.record_warm()
    band.warmed.wait(timeout=TIMEOUT_S)


def _sample_until_stopped(band: BandState) -> None:
    while not band.stop.is_set():
        sample = one_request(band.url, band.prompt, band.max_tokens)
        if sample[0] >= band.sampling_start:
            band.record_sample(sample)


def worker(band: BandState) -> None:
    """One closed-loop client: warm up, then re-issue on every completion."""
    try:
        _warm_up(band)
        _sample_until_stopped(band)
    except BaseException as exc:  # noqa: BLE001 - recorded, never swallowed
        band.record_error(exc)


def _await_termination(band: BandState, min_samples: int, min_seconds: float) -> None:
    """§4.4.2: terminate on whichever bound is satisfied LAST."""
    band_start = time.monotonic()
    hard_stop = band_start + min_seconds + TIMEOUT_S * 2
    while not band.stop.is_set():
        elapsed = time.monotonic() - band_start
        if band.count() >= min_samples and elapsed >= min_seconds:
            break
        if time.monotonic() >= hard_stop:
            break
        time.sleep(0.1)
    band.stop.set()


def _summarize(c: int, got: list[tuple[float, float, int]]) -> dict:
    first_start = min(s for s, _, _ in got)
    last_end = max(e for _, e, _ in got)
    wall = last_end - first_start
    total_tokens = sum(t for _, _, t in got)
    latencies = sorted(e - s for s, e, _ in got)
    tok_counts = sorted(t for _, _, t in got)
    return {
        "c": c,
        "completed": len(got),
        "wall_s": wall,
        "total_tokens": total_tokens,
        # §4.4.3: wall-clock aggregate, not the mean of per-request rates.
        "agg_tok_s": total_tokens / wall if wall > 0 else 0.0,
        "latency_p50_s": statistics.median(latencies),
        "latency_min_s": latencies[0],
        "latency_max_s": latencies[-1],
        "tokens_p50": statistics.median(tok_counts),
        "tokens_min": tok_counts[0],
        "tokens_max": tok_counts[-1],
    }


def run_band(url: str, c: int, prompt: str, max_tokens: int, min_samples: int,
             min_seconds: float, warmup_per_worker: int) -> dict:
    """One band at concurrency c. Closed-loop: each worker re-issues on completion."""
    band = BandState(
        url=url, prompt=prompt, max_tokens=max_tokens,
        warmup_per_worker=warmup_per_worker, concurrency=c,
    )
    threads = [threading.Thread(target=worker, args=(band,), daemon=True)
               for _ in range(c)]
    for t in threads:
        t.start()

    # §4.4.2 warmup gate: sampling begins after every worker has completed its
    # warmup AND a 5 s quiesce has elapsed.
    band.warmed.wait(timeout=TIMEOUT_S * 2)
    time.sleep(QUIESCE_S)
    band.sampling_start = time.monotonic()

    _await_termination(band, min_samples, min_seconds)
    for t in threads:
        t.join(timeout=TIMEOUT_S)

    if band.errors:
        return {"c": c, "error": band.errors[0], "completed": band.count()}
    with band.lock:
        got = list(band.samples)
    if not got:
        return {"c": c, "error": "no samples", "completed": 0}
    return _summarize(c, got)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--prompt", default="Write an essay on compilers.")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--min-seconds", type=float, default=60.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    band = run_band(
        url=args.url,
        c=args.concurrency,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        # §4.4.2: max(30, 8 x c) sampled requests, 2 x c warmup.
        min_samples=max(30, 8 * args.concurrency),
        min_seconds=args.min_seconds,
        warmup_per_worker=2,
    )
    band["label"] = args.label
    band["max_tokens"] = args.max_tokens
    print(json.dumps(band))
    return 1 if "error" in band else 0


if __name__ == "__main__":
    sys.exit(main())
