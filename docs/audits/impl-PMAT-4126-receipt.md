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

## Round-1 quorum finding (lane 1, claude-sonnet-5, PASS with a measured non-blocking gap), fixed
- On curl exit 18 (a 200 status line and then a body cut short), curl still reports the code. The
  first version nulled it as "no HTTP response". Now the real code is KEPT and curl_error says
  `transfer failed after HTTP <code> (curl exit <n>)`. The route is not ok (code forced to 000).
- Keeping a 200 required the row builder to stop calling it green: serve_ok now also requires no
  curl_error. The RED reason lists those routes separately ("serve routes without a complete
  response: <route> (<curl_error>)"), and the non-200 list excludes them.
- check_ladder_serve_verdict.sh: +route-timeout and +route-200-cut table rows (both red), and
  +reason:cut. Ad-hoc mutation: deleting `and not r.get("curl_error")` turns route-200-cut RED
  (green=true, expected false).
- check_ladder_serve_probe_timeout.sh: +cut-kept. The fake /v1/completions sends a 200 and cuts its
  body; the record keeps http 200, curl_error names curl exit 18, and ok is false.

## Trigger measured (see #4126 comment 5805617522)
The unbounded /api/chat is NOT the cause: it generated eval_count 7, then EOS. What pushes it past
60 s is per-request CPU latency under host load (100-125 s at load 155/48). The timeout policy is
left to the cop.

## Round-2 quorum finding (lane 2, claude-sonnet-5, PASS, measured), fixed
A 200 whose body stalled past --max-time (curl exit 28 AFTER a status line) was labelled
"timeout: no response", contradicting its own `http:200`. A received status now wins, and the error
reads `timeout after HTTP <code>: body incomplete within <N>s`. New case stall-named: the fake
/v1/chat/completions stream sends a 200 and stalls. A mutant restoring the old branch order turns
stall-named RED with exactly that contradiction.

## Follow-up: the cop's timeout-policy ruling (branch fix/4126-timeout-policy, stacked on a7b14b0cb)
- (a) TIMEOUT: a curl-28 route is flagged `timeout:true`. The row reason reads `serve TIMEOUT
  (harness/host condition, not a model verdict; route bound <N>s): <routes>`, and such a route is
  NOT also listed as failed. The row stays not-green (proven or RED; TIMEOUT is a named RED class).
- (b) Per-backend bound: contracts/model-capability-ladder-v1.yaml `serve_health.route_timeout_s:
  {cpu: 300, cuda: 60}` (`gpu` reads cuda). ladder_route_timeout reads it. An undeclared backend
  gets a decline by name. The bound is stamped per route (timeout_s) and per record
  (route_timeout_s). LADDER_ROUTE_MAX_TIME remains a test override only.
- (c) Host load: ladder_host_load reads the 1-min loadavg and core count (seams:
  LADDER_LOADAVG_FILE, LADDER_NPROC), stamped as `host_load`. A cpu serve with load > cores gets a
  decline by name (rc 2, carried out of $( ) by #4090's carry-out).
- Guards: check_ladder_serve_probe_timeout.sh +stamped, +load-decline, +contract-timeout (8 cases).
  Its self-test has 3 plants: bare code -> parses, deleted load check -> load-decline, hard-coded
  60 -> contract-timeout; all killed. check_ladder_serve_verdict.sh +reason:timeout; an ad-hoc
  mutation emptying the TIMEOUT list turns it RED ("unknown"). check_ladder_serve_teardown.sh lifts
  the two new functions. pv validate / pv lint on the contract: valid / PASS.
- check_model_ladder.sh --self-test 302/0, and the sibling guards all rc 0. check_bashrs_gate.sh:
  the base's 8, none added.
- Quorum lane 2 (claude-sonnet-5, filling a Gemini seat that hit its quota) FAILED 96ec637dd,
  measured: the load check failed OPEN. An unreadable /proc/loadavg or nproc gave "unknown";
  python's float() raised, and the exception's exit 1 read as "under the core count", so the cpu
  serve proceeded unchecked. Fixed: the comparison returns 0 (over), 1 (under), or 2
  (unmeasurable), and 2 declines by name ("host load unmeasurable"). New case load-unmeasurable
  (a missing loadavg file declines). Self-test plants: no-load-check -> load-decline, fail-open
  (unmeasurable read as fine) -> load-unmeasurable; 4 plants, all killed.
