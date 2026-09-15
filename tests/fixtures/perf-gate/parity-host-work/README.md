# `parity-host-work` — a synthetic `$WORK` in the P1 executor layout

This directory is a miniature of what `scripts/parity_host_receipt.sh` leaves in
its `$WORK` after one interleaved run: two bands (c=1 and c=4) of one lane
(`cpu`), five replicates per band per lane, the comparator's `GET /props`, the
subject's `GET /v1/effective-config`, and the device record either side of each
band.

**It is SYNTHETIC and it is not a measurement.** Every number in it was typed,
not observed, and nothing in the tree may cite a digit from it as a performance
result. Its one job is to give `scripts/lib/parity_block.py` and
`scripts/lib/perf_receipt.py` an input in the layout their only producer
actually writes, so the P1 → P2 → P3 chain has a test that fails when the
producer and the consumers drift apart again. The chain had no such test, and
the drift it exists to catch had already happened: the consumers read
`$WORK/{apr,llama}-<class>-c<N>.json`, a layout the executor stopped writing.

Two properties are deliberate, because the selftest rows in
`perf_receipt.py --selftest` mutate them:

* **it is conformant.** `interleaved: true`, five replicates, `stream_mode:
  "live"`, a `PASS` batch-invariance witness on the c=4 band, zero timeouts,
  every request generating exactly `protocol.n_predict` tokens, and an
  isolation record naming no foreign compute pid. So both bands reach
  `MEASURED` and both ratios carry a 95% lower bound. Mutate any one of those
  and the band must leave `MEASURED` — that is what the four `p1_chain_*` rows
  check.
* **the subject wins by about 1.1x, with jitter.** A constant ratio would give
  a zero-variance replicate estimator and a lower bound equal to the point
  estimate, which would pass the bound check without exercising it.

## Regenerating it

The files below were written by exactly this snippet, run from the repository
root. Re-run it after changing the executor's output contract, and say so in
the commit message.

```python
import json
import os

OUT = "tests/fixtures/perf-gate/parity-host-work"
KLASS = "cpu"
BANDS = (1, 4)
REPLICATES = 5
REQUESTS = 6
N_PREDICT = 128          # scripts/perf-matrix.yaml protocol.n_predict
PROMPT_TOKENS = 512      # scripts/perf-matrix.yaml protocol.prompt_tokens
# Per-replicate jitter, in per-mille, one sequence PER LANE. Typed and
# deterministic. The two sequences DIFFER on purpose: a single shared sequence
# cancels in the quotient, every per-replicate log-ratio comes out identical,
# the sample standard deviation is zero and the 95% lower bound lands exactly on
# the point estimate -- a bound check that passes without ever being exercised.
JITTER = {"subject": (0, 15, -8, 6, 3), "comparator": (0, -6, 11, -3, 8)}
# Subject over comparator, per band. Both sit inside [1 - arms.L3.delta.agg_ratio,
# derivation.sanity_ceiling], so the fixture's own band verdict is PASS.
AGG = {1: (100.0, 90.0), 4: (380.0, 345.0)}
DEC = {1: (110.0, 100.0), 4: (28.0, 25.5)}
PRE = {1: (1300.0, 1180.0), 4: (4200.0, 3860.0)}


def scaled(base, k, lane):
    return round(base * (1000 + JITTER[lane][k]) / 1000.0, 4)


def request_rows(k, lane, c):
    """REQUESTS raw per-request rows. The decode span carries the same ratio as
    the window-level decode rate, so the request-unit bootstrap and the
    replicate estimator do not disagree about which side is faster."""
    base = 1500.0 if lane == "subject" else 1650.0
    ttft = 40.0 if lane == "subject" else 45.0
    rows = []
    for i in range(REQUESTS):
        latency = round(base * (1000 + JITTER[lane][k]) / 1000.0 + i * 17.0 + c * 3.0, 4)
        rows.append({"latency_ms": latency,
                     "ttft_ms": round(ttft + i * 1.5, 4),
                     "completion_tokens": N_PREDICT,
                     "prompt_tokens": PROMPT_TOKENS,
                     "itl_ms": round((latency - ttft) / (N_PREDICT - 1), 6),
                     "finish_reason": "length"})
    return rows


def run(k, lane, c):
    agg, dec, pre = AGG[c], DEC[c], PRE[c]
    i = 0 if lane == "subject" else 1
    rows = request_rows(k, lane, c)
    return {
        "concurrency": c,
        "tokens_per_sec": scaled(agg[i], k, lane),
        "decode_tok_per_sec": scaled(dec[i], k, lane),
        "prefill_tok_per_sec": scaled(pre[i], k, lane),
        "ttft_p50_ms": rows[len(rows) // 2]["ttft_ms"],
        "total_requests": REQUESTS,
        "successful": REQUESTS,
        "failed": 0,
        "completion_tokens_total": REQUESTS * N_PREDICT,
        "prompt_tokens_total": REQUESTS * PROMPT_TOKENS,
        "elapsed_secs": 60.0,
        "runtime_name": "%s-%s-c%d" % ("apr" if lane == "subject" else "llamacpp", KLASS, c),
        # NOT WRITTEN BY TODAY'S HARNESS. §7.4 needs them to decide a band's
        # status, and an absent one is treated exactly as a failing one, so the
        # fixture states them and the mutation rows remove them.
        "stream_mode": "live",
        "timeouts": 0,
        "drain_ms": 120.0,
        "witness": ({"batch_invariance": "PASS", "divergence_at": None,
                     "declared_min": N_PREDICT, "m_formed": c,
                     "source": "server"} if c > 1 else None),
        "request_details": rows,
    }


def write(name, doc):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")


os.makedirs(OUT, exist_ok=True)
for c in BANDS:
    tag = "%s-c%d" % (KLASS, c)
    for k in range(REPLICATES):
        write("apr-%s-r%d.json" % (tag, k + 1),
              {"runs": [run(k, "subject", c)], "aggregate": {}, "regressions": []})
        write("llama-%s-r%d.json" % (tag, k + 1),
              {"runs": [run(k, "comparator", c)], "aggregate": {}, "regressions": []})
    write("llama-%s.props.json" % tag,
          {"total_slots": c, "n_ctx": 1024 * c,
           "default_generation_settings": {"n_ctx": 1024},
           "build_info": "7746 (39173bcac)"})
    write("apr-%s.config.json" % tag,
          {"schema_version": 1, "compute_class": "cpu",
           "build_features": [], "backend_loaded": ["cpu"],
           "server": {"version": "0.65.0", "pid": 4242,
                      "started_utc": "2026-09-02T10:11:12.345Z",
                      "clock_source": "chrono::Utc::now (CLOCK_REALTIME)"},
           "scheduler": {"kind": "iteration", "slots_admitted": 16,
                         "window_ms": 60000},
           "cuda": None, "lock_contended": False})
    for when in ("before", "after"):
        write("iso-%s-%s.json" % (tag, when),
              {"host": "fixture", "when": when, "probe": "nvidia-smi",
               "compute_pids": [{"pid": 4242, "used_memory_mib": 5120}],
               "foreign_pids": [], "memory_used_mib": 5480})
    write("band-%s.json" % tag,
          {"class": KLASS, "concurrency": c, "interleaved": True,
           "replicates": REPLICATES,
           "client_concurrency": {"subject": c, "comparator": c},
           "comparator_flags": "-ngl 0 -c %d -t 8 -np %d -fa auto -ub 512 "
                               "-ctk f16 -ctv f16 -cb --no-warmup" % (1024 * c, c),
           "comparator_slots_admitted": c, "comparator_n_ctx_slot": 1024,
           "comparator_n_batch": 2048,
           "comparator_props_file": "llama-%s.props.json" % tag,
           "subject_compute_class": "cpu",
           "subject_effective_config": "present",
           "subject_effective_config_file": "apr-%s.config.json" % tag,
           "gpu_layers_requested": "0", "gpu_layers_resolved": 0,
           "gpu_layers_total": 28,
           "isolation_before_file": "iso-%s-before.json" % tag,
           "isolation_after_file": "iso-%s-after.json" % tag,
           "replicate_files": {"subject": "apr-%s-r{k}.json" % tag,
                               "comparator": "llama-%s-r{k}.json" % tag}})
with open(os.path.join(OUT, "lanes.txt"), "w", encoding="utf-8") as fh:
    fh.write("%s 0\n" % KLASS)
```

The `lanes.txt` line has TWO fields on purpose: `<class> <gpu-layers>` is what
`run_lane()` appends, and the second field is a quantity handed to the loader,
not a compute class. The class the subject actually took is read from
`band-<tag>.json`'s `subject_compute_class`, which the executor fills from the
line the server printed about itself.
