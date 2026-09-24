# PMAT-4126 receipt: a route with no response keeps the serve record (branch fix/4126-probe-timeout-keeps-evidence)

Stacked on fix/4090-lock-wait-not-a-stall (5080786c4, itself on #4055). The probe code is shared.

## The defect, measured
lambda, #4034 A/B arm A (tree caecfe377, apr 0.69.1 fc942f6be sha256 2ba2d77e…, qwen35-9b-q4km,
cpu lane). The serve verdict was `probed:false`, "the serve probe produced no parseable object for
backend cpu -- it exited (status 1) rather than returned", with the server's own log showing it UP.
The same text is on the 27B cpu serve of the fc942f6be lambda sweep. Mechanism, confirmed:
`"http":$code` with curl's `000` gives `{"http":000}`, which json.loads rejects.

## The fix
- `crc` is curl's exit status. A 3-digit code with crc 0 is written as the number. Otherwise
  http:null and curl_error: `timeout: no response within <N>s` (curl exit 28), or `no HTTP
  response (curl exit <crc>)`. `code` stays `000` internally, so `rc=1` / `ok:false` are
  unchanged.
- LADDER_ROUTE_MAX_TIME (default 60) is a test seam for curl's --max-time. The default is
  unchanged.
- The row's RED reason prints the curl_error for a route without an http code.
- Per the cop's ruling, the /api/chat token bound and the timeout are NOT touched until the CPU 9B
  /api/chat duration is measured.

## Evidence
- NEW scripts/check_ladder_serve_probe_timeout.sh (guard_tree-dispatched): the SHIPPED
  ladder_serve_probe is lifted with its callees and run against a fake `apr serve` whose /api/chat
  hangs past LADDER_ROUTE_MAX_TIME=2. Results: parses (one object, rc 1); timeout-named (both
  /api/chat routes: http null plus a timeout reason); others-kept (4 routes, http 200). The
  --self-test plant, a bare `"http":$code`, is killed by `parses`.
- check_model_ladder.sh --self-test 302/0. serve_verdict, serve_probe_evidence,
  serve_backend_record, output_judged, serve_teardown, write_errors, apr_bin_pinned and
  guards_are_wired: all rc 0. check_bashrs_gate.sh: the base's 8 findings over 413 files, none
  in this diff.
