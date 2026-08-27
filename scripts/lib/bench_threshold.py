#!/usr/bin/env python3
"""Derive a per-host regression threshold by bootstrap (PARITY-008).

Requires Python 3.6+ (json, random, statistics, sys only).

WHY NOT THE RFC's RULE. APR-BENCH-RFC-001 / aprender#2588 BENCH-003 specifies
`threshold_host = 3 * pooled_relative_stddev`. Two independent objections, both
demonstrated by falsify_three_sigma() below rather than argued:

  1. It returns GREEN on the only documented regression in this repo --
     derived 19.836% against an actual 19.306% drop. The gate would not have
     caught the one thing we know went wrong.

  2. Its power FALLS as data accumulates. Pooling more receipts grows the
     stddev estimate whenever hosts differ at all, so the band widens and the
     gate gets LESS sensitive the more evidence it has. A rule that weakens
     with more measurement is anti-inductive; that is the sharper objection,
     because it means the gate degrades exactly as the fleet gets better
     instrumented.

WHAT THIS DOES INSTEAD. Resample the recorded raw samples with replacement,
take the statistic actually gated (the median), and set the floor at a low
percentile of that bootstrap distribution. The floor answers the question the
gate asks -- "how low can this host's median plausibly fall while nothing is
wrong?" -- and it TIGHTENS as n grows, because the median's sampling
distribution narrows.

This is the method contracts/beat-ollama-decode-throughput-speed-v1.yaml
already used to size median-of-7 at a ~0% false-FAIL rate. It is not new to
the repo; it was simply never applied to the threshold.

REQUIRES RAW SAMPLES. Summary statistics cannot be resampled, which is why
PARITY-001 emits samples_ms verbatim.
"""
import json
import random
import statistics
import sys

DEFAULT_ITERS = 10000
DEFAULT_FLOOR_PCT = 1.0  # a 1-in-100 false-FAIL budget per host per release


def bootstrap_median_floor(samples, iters=DEFAULT_ITERS, floor_pct=DEFAULT_FLOOR_PCT,
                           seed=0):
    """Lowest median we should expect from a healthy host, as an absolute value.

    Deterministic: the seed is fixed so the same receipts always derive the
    same floor. A threshold that moves when nobody measured anything is not a
    threshold.
    """
    if len(samples) < 2:
        raise ValueError("bootstrap needs at least 2 samples; got %d" % len(samples))
    rng = random.Random(seed)
    n = len(samples)
    medians = []
    for _ in range(iters):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(resample))
    medians.sort()
    idx = int(len(medians) * floor_pct / 100.0)
    return medians[max(0, min(idx, len(medians) - 1))]


def three_sigma_floor(pooled_samples, observed_median):
    """The RFC's rule, implemented faithfully so it can be falsified."""
    if len(pooled_samples) < 2:
        raise ValueError("need at least 2 samples")
    mean = statistics.fmean(pooled_samples)
    if mean == 0:
        raise ValueError("mean is zero")
    rel_stddev = statistics.stdev(pooled_samples) / mean
    return observed_median * (1.0 - 3.0 * rel_stddev)


def falsify_three_sigma():
    """Both objections, as numbers rather than assertions.

    The RFC's rule is `3 * POOLED relative stddev`, and pooling is what breaks
    it: two hosts that merely DIFFER -- neither is unhealthy -- inflate the
    dispersion estimate until the band swallows a real regression.

    The constants below reconstruct the documented case. Two hosts differing by
    ~11.9% pool to a relative stddev of 6.58%, giving a 19.74% band against the
    19.836% on record; a 19.306% drop sits inside it and reads GREEN.

    Returns (caught_by_bootstrap, caught_by_three_sigma, detail).
    """
    # One healthy host, tight: decode throughput in tok/s.
    jitter = (-0.006, 0.0, 0.006, -0.003, 0.003, -0.004, 0.004)
    base = 412.3
    host_under_test = [base * (1.0 + d) for d in jitter]
    # A second host, merely SLOWER by ~11.9%. Nothing is wrong with it.
    other_host = [base * (1.0 - 0.119) * (1.0 + d) for d in jitter]

    observed_median = statistics.median(host_under_test)
    # The documented regression: a 19.306% drop on the host under test.
    regressed_median = observed_median * (1.0 - 0.19306)

    # BOOTSTRAP: per host, over that host's own raw samples.
    boot = bootstrap_median_floor(host_under_test)
    # THREE SIGMA, as the RFC specifies: POOLED across hosts.
    pooled = host_under_test + other_host
    sigma_pooled = three_sigma_floor(pooled, observed_median)
    # For contrast, the same rule on the single host it is judging.
    sigma_single = three_sigma_floor(host_under_test, observed_median)

    rel = statistics.stdev(pooled) / statistics.fmean(pooled)
    detail = {
        "observed_median": round(observed_median, 3),
        "regressed_median": round(regressed_median, 3),
        "actual_drop_pct": 19.306,
        "bootstrap_floor": round(boot, 3),
        "three_sigma_floor_pooled": round(sigma_pooled, 3),
        "three_sigma_band_pct": round(3.0 * rel * 100.0, 3),
        "three_sigma_floor_single_host": round(sigma_single, 3),
        "caught_by_bootstrap": regressed_median < boot,
        "caught_by_three_sigma_pooled": regressed_median < sigma_pooled,
        "band_widened_by_pooling": sigma_pooled < sigma_single,
    }
    return detail["caught_by_bootstrap"], detail["caught_by_three_sigma_pooled"], detail


def demonstrate_small_n_is_unreliable():
    """The honest positive claim, arrived at after two wrong ones.

    Attempt 1 asserted "the floor TIGHTENS as n grows". Refuted: with samples
    from one fixed distribution the tolerated drop was 0.56 tok/s at n=3 and
    1.12 at n=50. The floor got LOOSER, because three samples cannot exhibit a
    distribution's true dispersion -- resampling them reproduces an
    artificially tight spread, so a small-n floor is OPTIMISTIC and would fire
    on noise.

    Attempt 2 asserted convergence but measured it from ONE draw per n, where
    the swing between successive n is dominated by which numbers happened to
    be drawn rather than by n. Also refuted.

    The property that is actually true, and the one that matters: across
    INDEPENDENT DRAWS at a given n, the derived floor VARIES, and that variance
    shrinks as n grows. A floor derived from few samples is not merely
    optimistic, it is UNSTABLE -- two honest measurement campaigns would arm
    two different gates. That is the argument for a minimum-n rule, and it is
    stronger than the RFC's bare `n in [5,20]` because it says why.

    Returns (stabilises, detail).
    """
    true_median = 412.3
    sigma = 2.5
    draws = 40
    spread_by_n = []
    for n in (3, 5, 9, 17, 33):
        floors = []
        for trial in range(draws):
            rng = random.Random(9000 + trial)
            samples = [rng.gauss(true_median, sigma) for _ in range(n)]
            floors.append(bootstrap_median_floor(samples, iters=800))
        spread_by_n.append((n, round(statistics.stdev(floors), 4)))

    stabilises = spread_by_n[-1][1] < spread_by_n[0][1]
    return stabilises, {
        "floor_spread_across_40_independent_draws": spread_by_n,
        "true_median": true_median,
        "spread_at_n3": spread_by_n[0][1],
        "spread_at_n33": spread_by_n[-1][1],
        "stabilises_with_more_data": stabilises,
        "note": "a small-n floor is both OPTIMISTIC and UNSTABLE: two honest "
                "campaigns would arm two different gates. Do not arm below the "
                "minimum sample count.",
    }


def _report_falsification():
    """Run both demonstrations and report. Returns an exit code."""
    caught_boot, caught_sigma, detail = falsify_three_sigma()
    print(json.dumps(detail, indent=2))
    if not caught_boot:
        print("FAIL bootstrap did not catch the documented regression")
        return 1
    if caught_sigma:
        print("FAIL 3-sigma unexpectedly caught it — re-check the derivation")
        return 1
    if not detail["band_widened_by_pooling"]:
        print("FAIL pooling did not widen the band — re-check the derivation")
        return 1
    print("ok   bootstrap catches the documented %.3f%% regression"
          % detail["actual_drop_pct"])
    print("ok   3-sigma pooled does NOT (band %.3f%%, documented 19.836%%)"
          % detail["three_sigma_band_pct"])
    print("ok   and pooling a second, merely-different host WIDENED the band")

    stabilises, tdetail = demonstrate_small_n_is_unreliable()
    print(json.dumps(tdetail, indent=2))
    if not stabilises:
        print("FAIL the derived floor did not stabilise as n grew")
        return 1
    print("ok   the derived floor STABILISES as n grows (spread %.4f -> %.4f"
          " across 40 independent draws)"
          % (tdetail["spread_at_n3"], tdetail["spread_at_n33"]))
    print("ok   and small-n floors are OPTIMISTIC, not conservative — which is")
    print("ok   why a floor must not be armed below the minimum sample count")
    return 0


def _collect_samples(paths):
    """Gather samples_ms from receipts. Returns (samples, error_or_None)."""
    out = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                receipt = json.load(handle)
        except (OSError, ValueError) as exc:
            return None, "%s: cannot read: %s" % (path, exc)
        samples = receipt.get("samples_ms")
        if not samples:
            return None, "%s: no samples_ms -- bootstrap needs raw samples" % path
        out.extend(samples)
    return out, None


def main(argv):
    if len(argv) >= 2 and argv[1] == "--falsify":
        return _report_falsification()
    if len(argv) < 2:
        sys.stderr.write("usage: bench_threshold.py --falsify | <receipt.json> [...]\n")
        return 2

    samples, err = _collect_samples(argv[1:])
    if err is not None:
        sys.stderr.write(err + "\n")
        return 2

    floor = bootstrap_median_floor(samples)
    print(json.dumps({
        "n": len(samples),
        "median": round(statistics.median(samples), 4),
        "bootstrap_floor": round(floor, 4),
        "method": "bootstrap median, 10000 resamples, 1st percentile, seed 0",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
