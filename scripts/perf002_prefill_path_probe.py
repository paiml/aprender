#!/usr/bin/env python3
"""PP-LLAMA-001 §12 row 2 — decompose prefill wall time into fixed + per-token.

The measurement half is `perf002_prefill_path_probe.sh`; this file is the part
that DECIDES, and the part that can be proven without a GPU.

WHAT §9 #1 CLAIMS AND WHAT THIS TESTS
-------------------------------------
§9 #1: Blackwell prefill is a per-token serial loop —
`gpu_profile.rs:517-531`, `select_prefill_path`: `cc >= SM12X_MIN_CC => Serial`.
The size was measured once, on gx10, from a single fixed point: 16.75 s at 513
prompt tokens = 32.65 ms per prompt token, "fitted over 28 of 30 samples".

§10's registered prediction, verbatim:

    prefill wall time linear at ~32.6 ms per prompt token on the default arm;
    collapses to ~0.35 s at 512 prompt tokens under BATCHED_PREFILL=1
    KILL IF: flat in prompt length, or no collapse

A single (513 tokens, 16.75 s) point cannot distinguish "linear at 32.65 ms per
token" from "a 16.75 s fixed cost that does not depend on prompt length at all"
— the two fit that point equally well and imply opposite defects. That is why
this row exists and why the probe sweeps SEVERAL prompt lengths: the slope is
the claim, and one point has no slope.

WHY REFUSALS, NOT A BEST-EFFORT FIT
-----------------------------------
A decomposition that always returns a number would let this probe report
"32.6 ms per token" from noise, from two points on top of each other, or from a
downward-sloping cloud. The four refusals below are the shapes where the fit
would be arithmetic without being a measurement. Each one names ITSELF in the
output, so a caller never has to infer which rule fired:

  too_few_distinct_x   fewer than 2 distinct prompt-token counts — no slope
                       exists. This is the single-point shape §9 #1 was
                       originally sized from.
  negative_slope       prefill getting FASTER with more tokens is not the
                       mechanism under test; it means the workload or the
                       harness moved, and a negative "ms per token" reported
                       as a per-token cost would be nonsense.
  r2_below_bound       the points do not lie on a line, so `fixed + slope*n`
                       is the wrong model and its coefficients describe
                       nothing.
  bimodal              TWO lines, not one. This is a REPORT, not a rejection:
                       the fit is emitted per mode. §9 #1's own sizing threw
                       away 2 of 30 samples to get its line; a probe that
                       silently did the same would hide a path switch.

A fifth refusal, `implausibly_fast`, guards the HARNESS rather than the fit, and
it is here because a draft of this probe needed it: see decompose().

FOUR HARNESS DEFECTS THIS PROBE HIT, SO THE NEXT PERSON DOES NOT RE-DERIVE THEM
------------------------------------------------------------------------------
All four were found by running it on gx10 on 2026-09-20, and all four produced
a CONFIDENT WRONG ANSWER rather than an error:

  1. curl's `time_starttransfer` is when the response HEADERS begin, not the
     first token. A streaming server sends headers before it has prefilled
     anything, so the first draft measured connection setup: ~0.6 ms samples,
     r2 ~ 1, and a verdict of "the default arm is FLAT". TTFT is now read off
     the SSE stream (first chunk carrying content) in measure(), and
     `implausibly_fast` catches the class.
  2. The first request after `serve run` goes healthy pays CUDA context
     creation, autotune and graph capture: 0.76 s at 64 tokens against 0.13 s
     at 128, i.e. a NEGATIVE slope. One warm-up per arm is issued and
     discarded.
  3. A repeated prompt measures the PREFIX CACHE, not the prefill path
     (`paged_kv/mod_quantized_paged.rs::find_longest_prefix`,
     `scheduler/chunked_prefill.rs::record_prefix_cache_hit`). With one prompt
     per rung the default arm was BIMODAL: 10 of 15 samples flat and shapeless
     (0.045 ms/token, r2 0.37 — hit latency does not depend on prompt length)
     and 5 on a clean line at 8.97 ms/token. Each sample now carries a unique
     nonce; `--repeat-prompt` reproduces the bimodal result on demand.
  4. The r2 floor belongs to the DEFAULT arm's justification and was wrongly
     applied to the batched one. A serial per-token loop is a sum of n
     identical steps and is nearly exactly linear; a batched prefill is a fixed
     overhead plus a small per-token term with real variance and has no reason
     to reach 0.95. The probe reported UNMEASURABLE about the clearest result
     in the run (40 ms at 64 tokens to ~99 ms at 513, against the default arm's
     4720 ms). The collapse is now read off the MEASUREMENT, which is what §10
     actually predicts — a time at a length, not a line.

None of these is an error in the probe. They are UNMEASURABLE verdicts about
the run, which is why the driver maps them to exit 2 and never to a defect
claim: a guard that names a code cause for a box it could not evaluate has
fired three times in this repo in one day (see the sibling probe's header).
"""

from __future__ import annotations

import argparse
import json
import sys

# R2 floor for accepting a single-line model. 0.95 is not a magic number: the
# claim under test is a SERIAL PER-TOKEN LOOP, whose wall time is a sum of n
# near-identical steps and is therefore very nearly exactly linear. A cloud that
# only reaches 0.9 is not that mechanism, and reporting a per-token cost from it
# would be describing something else. Overridable so the driver can record what
# it used rather than hard-coding a belief.
DEFAULT_R2_MIN = 0.95

# Two modes are "separated" when the gap between their mean per-token costs is
# at least this many times the larger within-mode spread. At 3.0 the split has
# to be visible rather than arithmetic: two clusters whose spreads nearly touch
# stay ONE fit, and the r2 floor then decides whether that fit is usable.
DEFAULT_MODE_SEPARATION = 3.0

# A prefill faster than this cannot be a prefill. The smallest rung on the
# ladder is 64 tokens; even a fully batched GPU prefill of 64 tokens is
# milliseconds, and the serial path under test is ~2 s at that length. 5 ms is
# two orders of magnitude below the fastest plausible answer and three below the
# claim — a floor that only a broken harness can trip. See the comment in
# decompose(); this exists because a draft of this probe tripped it.
DEFAULT_PREFILL_FLOOR_S = 0.005


def _fit(xs: list[float], ys: list[float]) -> dict:
    """Ordinary least squares y = fixed + slope*x, with R^2.

    Returns the coefficients and the sample count. No refusal logic: the caller
    owns that, so this stays a piece of arithmetic that can be read on its own.
    """
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    fixed = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (fixed + slope * x)) ** 2 for x, y in zip(xs, ys))
    # A perfectly flat y (ss_tot == 0) is a real outcome, not a division to
    # guard against silently: every sample identical means the fit explains
    # everything there is to explain, so R^2 is 1 by definition.
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {
        "n": n,
        "fixed_s": fixed,
        "slope_s_per_token": slope,
        "r2": r2,
    }


def _modes(xs: list[float], ys: list[float], separation: float) -> list[list[int]]:
    """Split sample indices by per-token cost into 1 or 2 modes.

    One dimension, two clusters, so this is a sort and a cut rather than
    k-means: for every gap between consecutive sorted per-token costs, ask
    whether splitting there separates the two sides by `separation` times the
    larger side's spread. The widest qualifying gap wins.

    Samples at x == 0 carry no per-token cost and are held out of the clustering
    (they still enter the fit); a zero-token prefill is the intercept, not a
    mode.
    """
    rates = [(ys[i] / xs[i], i) for i in range(len(xs)) if xs[i] > 0]
    if len(rates) < 4:
        # Two modes need at least two points each to have a spread at all.
        return [list(range(len(xs)))]
    rates.sort()
    best: tuple[float, int] | None = None
    for cut in range(1, len(rates)):
        lo = [r for r, _ in rates[:cut]]
        hi = [r for r, _ in rates[cut:]]
        if len(lo) < 2 or len(hi) < 2:
            continue
        gap = min(hi) - max(lo)
        spread = max(max(lo) - min(lo), max(hi) - min(hi))
        # An exactly-zero spread on both sides means two delta functions: any
        # non-zero gap separates them.
        if spread == 0:
            if gap > 0 and (best is None or gap > best[0]):
                best = (gap, cut)
            continue
        if gap >= separation * spread and (best is None or gap > best[0]):
            best = (gap, cut)
    if best is None:
        return [list(range(len(xs)))]
    cut = best[1]
    lo_idx = sorted(i for _, i in rates[:cut])
    hi_idx = sorted(i for _, i in rates[cut:])
    zero_idx = [i for i in range(len(xs)) if xs[i] <= 0]
    # The intercept samples join the LOW mode: they are the cheapest possible
    # prefill, and putting them in both would double-count them.
    return [sorted(lo_idx + zero_idx), hi_idx]


def decompose(
    samples: list[dict],
    r2_min: float = DEFAULT_R2_MIN,
    separation: float = DEFAULT_MODE_SEPARATION,
    floor_s: float = DEFAULT_PREFILL_FLOOR_S,
) -> dict:
    """Decompose samples into fixed + per-token cost, or REFUSE and say why.

    `samples` is a list of {"prompt_tokens": int, "prefill_s": float}.

    Always returns a dict carrying `refused` (bool) and `rule` (the name of the
    rule that fired, or None). A refusal is a statement about the RUN, never
    about the code under test.
    """
    xs = [float(s["prompt_tokens"]) for s in samples]
    ys = [float(s["prefill_s"]) for s in samples]

    # THE HARNESS CHECK, BEFORE ANY FIT.
    #
    # Measured on gx10 2026-09-20 with a first draft that timed prefill as
    # curl's `time_starttransfer`: every sample came back at ~0.6 ms, the fit was
    # a beautiful straight line (r2 ~ 1) with a slope of 0.00005 ms/token, and
    # the verdict was PREDICTION_KILLED — "the default arm is flat". It was not
    # flat. `time_starttransfer` on a streaming endpoint measures when the
    # RESPONSE HEADERS start, which the server sends before it has prefilled
    # anything, so the probe had measured connection setup and reported it as a
    # property of the CUDA prefill path.
    #
    # None of the four refusals below catches that: the data really is linear,
    # really has positive slope, really has two distinct x, really is unimodal.
    # A false FINDING about serve code would have gone to the spec owner.
    #
    # So: a sample faster than the floor cannot be a prefill of that many tokens
    # on any path, and the probe says the HARNESS is wrong rather than the code.
    too_fast = [s for s, y in zip(samples, ys) if y < floor_s]
    if too_fast:
        return {
            "refused": True,
            "rule": "implausibly_fast",
            "detail": (
                f"{len(too_fast)} of {len(ys)} samples are under {floor_s * 1000:.1f} ms "
                f"(fastest {min(ys) * 1000:.3f} ms at "
                f"{int(min(s['prompt_tokens'] for s in too_fast))} tokens). "
                "No prefill of that many tokens takes that long on any path — the "
                "harness is timing something other than the first token (headers, "
                "connection setup, a cached response). This is a statement about "
                "the RUN, not about the code under test."
            ),
            "floor_s": floor_s,
            "n": len(ys),
        }

    distinct = sorted(set(xs))
    if len(distinct) < 2:
        return {
            "refused": True,
            "rule": "too_few_distinct_x",
            "detail": (
                f"{len(distinct)} distinct prompt-token count(s) "
                f"({[int(x) for x in distinct]}); a slope needs at least 2"
            ),
            "n": len(xs),
        }

    groups = _modes(xs, ys, separation)
    if len(groups) > 1:
        fits = []
        for idx in groups:
            fit = _fit([xs[i] for i in idx], [ys[i] for i in idx])
            fit["prompt_tokens"] = sorted({int(xs[i]) for i in idx})
            fits.append(fit)
        return {
            "refused": True,
            "rule": "bimodal",
            "detail": (
                "per-token cost splits into 2 modes at "
                f"{separation}x the within-mode spread; one line would describe "
                "neither. Per-mode fits are reported."
            ),
            "n": len(xs),
            "modes": fits,
        }

    fit = _fit(xs, ys)
    if fit["slope_s_per_token"] < 0:
        return {
            "refused": True,
            "rule": "negative_slope",
            "detail": (
                f"slope {fit['slope_s_per_token'] * 1000:.3f} ms/token is "
                "negative; prefill cannot get cheaper with more tokens, so the "
                "workload or the harness moved during the sweep"
            ),
            **fit,
        }
    if fit["r2"] < r2_min:
        return {
            "refused": True,
            "rule": "r2_below_bound",
            "detail": (
                f"r2 {fit['r2']:.4f} < {r2_min}; the samples do not lie on a "
                "line, so fixed + slope*n is the wrong model"
            ),
            "r2_min": r2_min,
            **fit,
        }
    return {"refused": False, "rule": None, "r2_min": r2_min, **fit}


def verdict(default_fit: dict, batched_fit: dict | None,
            batched_samples: list[dict] | None, args) -> dict:
    """Apply §10's registered kill conditions to two decomposed arms.

    `flat in prompt length` and `no collapse` are the only two ways the
    prediction dies. Everything else is either a confirmation of §9 #1's
    mechanism or an unmeasurable run.
    """
    if default_fit.get("refused"):
        return {
            "status": "UNMEASURABLE",
            "reason": f"default arm refused: {default_fit['rule']}",
        }

    slope_ms = default_fit["slope_s_per_token"] * 1000.0
    # "Flat in prompt length" is the kill condition, so it needs a threshold
    # that a REAL serial loop could never sit under. The claim is 32.65 ms per
    # token; `--flat-below-ms` defaults to a tenth of that. Anything slower than
    # the threshold is linear enough to be the mechanism; anything under it is
    # flat, and the fixed-cost story is the right one instead.
    if slope_ms < args.flat_below_ms:
        return {
            "status": "PREDICTION_KILLED",
            "reason": (
                f"default arm is FLAT in prompt length: {slope_ms:.3f} ms/token "
                f"< {args.flat_below_ms} ms/token. §9 #1 sizes the defect as a "
                "per-token serial loop; a flat arm means the 16.75 s at 513 "
                "tokens is a FIXED cost and §9 #1 is scoped wrong."
            ),
            "slope_ms_per_token": slope_ms,
        }

    if batched_samples is None:
        return {
            "status": "UNMEASURABLE",
            "reason": "the BATCHED_PREFILL=1 arm did not run, so the collapse half is unmeasured",
            "slope_ms_per_token": slope_ms,
        }

    # THE COLLAPSE IS READ OFF THE MEASUREMENT, NOT OFF A FIT.
    #
    # §10's prediction is "collapses to ~0.35 s at 512 prompt tokens", which is a
    # claim about a measured time at a length, not about a line. Requiring the
    # batched arm to BE a line before the collapse can be judged imports the
    # default arm's justification into an arm it does not apply to: a serial
    # per-token loop is a sum of n identical steps and so is very nearly exactly
    # linear, but a batched prefill is a fixed overhead plus a small per-token
    # term with real variance, and it has no reason to reach r2 0.95.
    #
    # Measured on gx10 2026-09-20: the batched arm ran 40 ms at 64 tokens to
    # ~99 ms at 513 — an unmistakable collapse from the default arm's 4720 ms —
    # and the fit's r2 fell below the floor purely on within-rung spread. The
    # probe reported UNMEASURABLE about the clearest result in the run. The fit
    # is still reported because a batched arm that is NOT collapsing is worth
    # describing; it is no longer a precondition for answering the question.
    at = args.collapse_at_tokens
    near = [s for s in batched_samples
            if abs(float(s["prompt_tokens"]) - at) <= args.collapse_tolerance_tokens]
    if not near:
        return {
            "status": "UNMEASURABLE",
            "reason": (
                f"no batched sample within {args.collapse_tolerance_tokens} tokens of "
                f"{at}; the ladder does not reach the length the prediction names"
            ),
            "slope_ms_per_token": slope_ms,
        }
    times = sorted(float(s["prefill_s"]) for s in near)
    # Median, not mean: one slow first-of-rung sample (autotune, graph capture)
    # should not decide a collapse either way.
    measured = times[len(times) // 2]
    if measured > args.collapse_below_s:
        return {
            "status": "PREDICTION_KILLED",
            "reason": (
                f"NO COLLAPSE: BATCHED_PREFILL=1 measured {measured:.3f} s at "
                f"~{at} prompt tokens (median of {len(near)}), above the "
                f"{args.collapse_below_s} s the prediction requires"
            ),
            "slope_ms_per_token": slope_ms,
            "batched_measured_s_at": {str(at): measured},
            "batched_fit_rule": (batched_fit or {}).get("rule"),
        }
    return {
        "status": "MECHANISM_CONFIRMED",
        "reason": (
            f"default arm linear at {slope_ms:.3f} ms/token (r2 "
            f"{default_fit['r2']:.4f}, n={default_fit['n']}); BATCHED_PREFILL=1 "
            f"measured {measured:.3f} s at ~{at} tokens (median of {len(near)})"
        ),
        "slope_ms_per_token": slope_ms,
        "batched_measured_s_at": {str(at): measured},
        "batched_fit_rule": (batched_fit or {}).get("rule"),
    }


# --------------------------------------------------------------------------
# selftest
#
# Every refusal gets a fixture that fires it AND a control that does not, so a
# rule that stopped working is distinguishable from an input that stopped
# triggering it. The last case is the one that matters most: a clean linear
# sweep must NOT refuse, or the four refusals above would be satisfied by a
# decompose() that rejects everything.
# --------------------------------------------------------------------------
def _selftest() -> int:
    failures = []

    def check(name: str, got: str | None, want: str | None) -> None:
        if got == want:
            print(f"  ok   {name}")
        else:
            failures.append(name)
            print(f"  FAIL {name}: rule={got!r}, wanted {want!r}")

    # 1. one distinct x — the single-point shape §9 #1 was sized from
    single = [{"prompt_tokens": 513, "prefill_s": 16.75}] * 4
    check("one distinct prompt length is refused", decompose(single)["rule"],
          "too_few_distinct_x")

    # 2. negative slope
    falling = [{"prompt_tokens": n, "prefill_s": 20.0 - 0.01 * n}
               for n in (64, 128, 256, 512)]
    check("a negative slope is refused", decompose(falling)["rule"],
          "negative_slope")

    # 3. r2 below the bound — a rising but shapeless cloud
    noisy = [{"prompt_tokens": n, "prefill_s": y} for n, y in
             ((64, 1.0), (128, 9.0), (256, 2.0), (512, 11.0), (600, 3.0))]
    check("a cloud that is not a line is refused", decompose(noisy)["rule"],
          "r2_below_bound")

    # 4. bimodal — two per-token costs, e.g. a path switch mid-sweep
    bimodal = [{"prompt_tokens": n, "prefill_s": 0.0326 * n}
               for n in (64, 96, 128, 160)]
    bimodal += [{"prompt_tokens": n, "prefill_s": 0.0007 * n}
                for n in (256, 320, 384, 448)]
    out = decompose(bimodal)
    check("two per-token costs are reported per mode", out["rule"], "bimodal")
    if out.get("rule") == "bimodal" and len(out.get("modes", [])) != 2:
        failures.append("bimodal emits two fits")
        print("  FAIL bimodal emits two fits")
    elif out.get("rule") == "bimodal":
        print("  ok   bimodal emits two fits")

    # 5. implausibly_fast — the harness guard, planted with the EXACT shape that
    #    fooled the first draft on gx10: ~0.6 ms samples, positive slope,
    #    r2 ~ 1, two distinct x, unimodal. Every other rule passes it; only the
    #    floor catches it. Without this case the four rules above are a set that
    #    provably misses a whole class of wrong answer.
    headers_not_tokens = [{"prompt_tokens": 64, "prefill_s": 0.00063},
                          {"prompt_tokens": 128, "prefill_s": 0.000633},
                          {"prompt_tokens": 256, "prefill_s": 0.00064},
                          {"prompt_tokens": 512, "prefill_s": 0.00065}]
    check("a harness timing headers, not tokens, is refused",
          decompose(headers_not_tokens)["rule"], "implausibly_fast")
    # and the anti-vacuity half: that fixture must survive every OTHER rule, or
    # it is not evidence that the floor is what caught it.
    without_floor = decompose(headers_not_tokens, floor_s=0.0)
    if without_floor.get("refused"):
        failures.append("the floor is what catches it")
        print(f"  FAIL the floor is what catches it: {without_floor['rule']} did")
    else:
        print("  ok   with the floor removed, every other rule passes it")

    # 6. THE CONTROL. A clean serial-loop sweep at the sized 32.65 ms/token must
    #    pass, or every rule above is satisfied by refusing everything.
    clean = [{"prompt_tokens": n, "prefill_s": 0.12 + 0.03265 * n}
             for n in (64, 128, 256, 384, 513)]
    got = decompose(clean)
    check("a clean linear sweep is NOT refused", got["rule"], None)
    if not got.get("refused"):
        slope_ms = got["slope_s_per_token"] * 1000
        if abs(slope_ms - 32.65) > 0.01:
            failures.append("the recovered slope is the planted one")
            print(f"  FAIL the recovered slope is the planted one: {slope_ms:.3f}")
        else:
            print(f"  ok   recovers the planted slope ({slope_ms:.3f} ms/token)")

    if failures:
        print(f"perf002-decompose --selftest: NO-GO ({len(failures)} failed)")
        return 1
    print("perf002-decompose --selftest: OK (8 cases)")
    return 0


def measure(url: str, tokens: list[int], repeats: int, timeout: float,
            unique_prompts: bool = True) -> int:
    """Sweep the ladder, printing `tokens<TAB>seconds_to_first_token` per sample.

    TIME TO FIRST CONTENT TOKEN, read off the SSE stream — not curl's
    `time_starttransfer`, which is when the response HEADERS begin. A streaming
    server sends headers before it has prefilled anything, so timing them
    measures connection setup and reports it as a property of the prefill path.
    A draft of this probe did exactly that on gx10 and produced a confident
    PREDICTION_KILLED from 0.6 ms samples. `implausibly_fast` now catches that
    class, but the right fix is to measure the right thing here.

    A prompt is one repeated token so LENGTH is the only thing that varies:
    content-varying prompts confound tokenizer behaviour with the path.
    """
    import time
    import urllib.request

    # ONE DISCARDED WARM-UP PER ARM, and it is not politeness.
    #
    # Measured on gx10 2026-09-20: the first request after `serve run` becomes
    # healthy pays CUDA context creation, kernel autotune and graph capture. On
    # a 64/128 ladder that landed as 0.760 s at 64 tokens and 0.132 s at 128 —
    # a NEGATIVE slope, which decompose() correctly refused rather than
    # reporting as a collapse. The refusal did its job; the harness should not
    # have handed it that shape. The warm-up is discarded, never recorded, so it
    # cannot enter a fit.
    warm = json.dumps({
        "model": "m", "stream": True, "max_tokens": 1,
        "messages": [{"role": "user", "content": "warm"}],
    }).encode()
    try:
        req = urllib.request.Request(
            url, data=warm, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for _ in resp:
                pass
    except OSError:
        # A failed warm-up is not fatal: the ladder below will produce no
        # samples and the caller will refuse for that reason, which is a more
        # accurate complaint than one about the warm-up.
        pass

    sample_id = 0
    for n in tokens:
        for _ in range(repeats):
            sample_id += 1
            # A DISTINCT PROMPT PER SAMPLE, and this is the default on purpose.
            #
            # Measured on gx10 2026-09-20 with a repeated prompt: the default
            # arm came back BIMODAL over the whole ladder — 10 of 15 samples on
            # a flat, shapeless line (slope 0.045 ms/token, r2 0.37) and 5 on a
            # clean per-token line (8.97 ms/token, r2 1.0000). Both modes span
            # every rung, so it is not a path switch at some length; the same
            # prompt length landed in either mode depending on the request. A
            # prompt or prefix cache explains that exactly, including the r2:
            # a cache hit's latency does not depend on prompt length.
            #
            # A probe for the PREFILL PATH must not be able to measure a cache.
            # Each sample's first token is a unique nonce, so no two requests
            # share a prefix, at the cost of one token of prompt.
            filler = " ".join(["token"] * max(n - 1, 1))
            content = (f"n{sample_id}x {filler}" if unique_prompts
                       else " ".join(["token"] * n))
            body = json.dumps({
                "model": "m",
                "stream": True,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": content}],
            }).encode()
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            start = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    ttft = None
                    for raw in resp:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except ValueError:
                            continue
                        delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                        # The FIRST chunk carrying content is the first token.
                        # A role-only opening chunk is not one.
                        if delta.get("content"):
                            ttft = time.monotonic() - start
                            break
            except OSError:
                # A dropped sample is dropped, never recorded as a zero: a zero
                # would be a fabricated fast prefill, which is the exact thing
                # the floor exists to catch.
                continue
            if ttft is not None:
                print(f"{n}\t{ttft:.6f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="prove every refusal fires, and that a clean sweep does not")
    ap.add_argument("--measure", metavar="URL",
                    help="sweep the ladder against this chat-completions URL")
    ap.add_argument("--tokens", help="space-separated ladder, with --measure")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--repeat-prompt", action="store_true",
                    help="reuse one prompt per rung — measures the CACHE, not the path; "
                         "kept so the bimodal gx10 result can be reproduced")
    ap.add_argument("--samples", help="JSON file: {default: [...], batched: [...]}")
    ap.add_argument("--json", help="write the decomposition here")
    ap.add_argument("--r2-min", type=float, default=DEFAULT_R2_MIN)
    ap.add_argument("--mode-separation", type=float, default=DEFAULT_MODE_SEPARATION)
    ap.add_argument("--prefill-floor-s", type=float, default=DEFAULT_PREFILL_FLOOR_S,
                    help="a sample under this is a broken harness, not a fast path")
    ap.add_argument("--flat-below-ms", type=float, default=3.265,
                    help="slope under this is FLAT and kills the prediction")
    ap.add_argument("--collapse-at-tokens", type=int, default=512)
    ap.add_argument("--collapse-below-s", type=float, default=0.35)
    ap.add_argument("--collapse-tolerance-tokens", type=int, default=8,
                    help="how near a ladder rung must be to --collapse-at-tokens to count")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if args.measure:
        if not args.tokens:
            ap.error("--tokens is required with --measure")
        return measure(args.measure, [int(t) for t in args.tokens.split()],
                       args.repeats, args.timeout, not args.repeat_prompt)
    if not args.samples:
        ap.error("--samples is required unless --selftest")

    with open(args.samples, encoding="utf-8") as handle:
        data = json.load(handle)

    default_fit = decompose(data.get("default") or [], args.r2_min,
                            args.mode_separation, args.prefill_floor_s)
    batched_raw = data.get("batched")
    batched_fit = (decompose(batched_raw, args.r2_min, args.mode_separation,
                             args.prefill_floor_s)
                   if batched_raw else None)

    record = {
        "arms": {"default": default_fit, "batched": batched_fit},
        "verdict": verdict(default_fit, batched_fit, batched_raw, args),
        "thresholds": {
            "r2_min": args.r2_min,
            "mode_separation": args.mode_separation,
            "prefill_floor_s": args.prefill_floor_s,
            "flat_below_ms_per_token": args.flat_below_ms,
            "collapse_at_tokens": args.collapse_at_tokens,
            "collapse_below_s": args.collapse_below_s,
        },
    }
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(text)
    sys.stdout.write(text)

    status = record["verdict"]["status"]
    return {"MECHANISM_CONFIRMED": 0, "PREDICTION_KILLED": 1}.get(status, 2)


if __name__ == "__main__":
    sys.exit(main())
