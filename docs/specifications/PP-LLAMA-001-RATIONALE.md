# PP-LLAMA-001 — RATIONALE

**Companion to** `docs/specifications/PP-LLAMA-001-MASTER.md`. **Date:** 2026-09-02.

The master states rules and contains no sentence about its own history (§0.5 of that document).
This file is where the reason lives. Every rule ID in the master has a paragraph here, keyed by
the ID, and §0.5 makes that paragraph a requirement rather than a courtesy: a pull request that
changes a rule adds an Appendix D row to the master and edits the paragraph here.

Most paragraphs open with the finding they answer, from
`docs/audits/parity-spec-audit-2026-09-02.md` (`C-nn`, `CO-nn`, and the `B-nn` standing table) or
from `docs/postmortems/perf-parity-review-2026-09.md`. This is a design document: it argues, it
does not measure, and every number it repeats belongs to a receipt named beside it.

---

## Parity principles

### P-1 — Paired

The comparator's value from the *same run* is the target, so parity is `1.0` by definition and
needs no threshold. Every previous version of this gate set a literal floor, and every literal
floor was derived from a measurement taken on the architecture the fix was about to remove — the
comparator's decode value, then `apr`'s own prior decode, then a band the device could not admit.
A paired target has no such history to be wrong about. It is also the only form under which
`agg` and `dec` can be read together (§2.2 of the master): both lanes move under the same thermal
and admission conditions, so the quotient is about the servers rather than about the hour.

### P-2 — Named

`C-11` found three statements of the c=1 rule in one document, none matching the others, because
none of them said which cell, band and metric they were about. A claim that names no band is not
falsifiable by any receipt, so it cannot be wrong and cannot be checked. PP-17 makes it
schema-fatal instead of a style note.

### P-3 — Asymmetric by band

`C-13` and §2.2. Aggregate and per-user decode move in opposite directions under the change that
matters most — sharing a device across `c` requests — so a gate demanding both at or above
parity on every band demands a *beat*, and a receipt carrying only one of them cannot be read at
all. The split is: decode and prefill at c=1, aggregate above it. Prefill and decode at c=1 do
not trade against each other, which is why gating both there is not the same trap (`C-13`).

### P-4 — Correct before fast

`C-15`, the largest missing control the audit found. The predecessor published aggregate figures
at `c>1` from a build the tracker says emits a constant token for every `m>1`, in the same
document whose §9 forbade exactly that. Prose could not stop it because prose is not a schema
field. `INVALID-CORRECTNESS` is a status, not an adjective: it removes the band's throughput from
the receipt, from the gate and from the baseline set at once.

### P-5 — Decision rule

`C-9`. The predecessor's rule was "PASS iff the one-sided 95% lower bound is at least `1.0 − ε`",
with `ε` the receipt's own MDE. The confidence bound already *is* the noise allowance; subtracting
an MDE on top spends the same σ twice, and at `n=5` that makes PASS materially easier than the
stated confidence implies. The non-inferiority form fixes it: `δ` is a *policy* margin with a
named author, `δ = 0` is parity, and any other value is a recorded concession rather than an
arithmetic accident. The MDE stays, as the cell's resolving power — it decides whether the cell
can answer the question, never what the answer is.

### P-6 — Seeded at achieved

`C-11`, and round 7 of the review (`33a25cc1b`). Three rules in the predecessor outlawed their own
fix. The class-level cure is that a gate is REPORTING until the first receipt that passes it, and
ARMED from that receipt on: a gate cannot reject a change it has never accepted, because it has no
opinion yet. It also removes the last date-driven arming from the document — nothing arms because
a calendar advanced.

### P-7 — Must-not-fire

Round 7 again, and `C-20`. A guard that has only a must-fire case has never been shown to be
*silent* on correct input; PP-23 is the worked example, since it shipped in a form that declared
the epic's own correctly-batching receipts a harness bug, and a must-not-fire fixture would have
caught that in the same commit. Both cases land together or neither lands.

### P-8 — Latency

`C-14`. The predecessor refused TTFT under W1 outright, on a convoy argument — at every round
boundary all `c` prefills collide. At c=1 there is one request and no convoy, so TTFT(c=1) under
W1 is a clean measurement and the cheapest prefill witness available today. Above c=1 the argument
holds, which is what W3 is for; until W3 exists those numbers are REPORTING and no latency bound
is set.

---

## Invariants

### PP-1

Carried from `I-1`. The gate must enumerate the cells it expects from `perf-matrix.yaml` and
assert a status for each, because the alternative — verdicts over whatever receipts happen to be
present — cannot distinguish "this cell passed" from "nobody ran this cell". `C-7` and `C-8` are
the same defect one level up: a scope section listing four expiry dates while the obligation
section says expiry is derived. One enumeration, one denominator.

### PP-2

`B-8`, `C-16` and `C-19`. The predecessor asserted in one paragraph that `max_batch` is auto-sized
from free VRAM and in the next that it was set by an environment variable, and both sentences were
live. A field that the server alone knows must come from the server; `GET /v1/effective-config`
is that boundary, stored verbatim so a reader can re-derive rather than re-trust. `C-16` adds the
memory half: memory is measured, causal for admission, and was scheduled nowhere.

### PP-3

A ratio is a *relation between two lanes of one run*, and the wire had no way to say so — a
scalar `agg_ratio` field could hold a number whose denominator came from any run, any host, any
month. The `0.591/0.395/0.544/0.401` series in the master's §2.1 is precisely that failure: a
subject from one commit divided by a comparator from another week. Requiring the comparator's whole
band object, sharing `run_id`, makes the P-1 violation unrepresentable instead of merely
forbidden.

### PP-4

`C-13`. A server at decode and aggregate parity with four-times-slower prefill passes a gate that
carries no prefill row, forever. Requiring all three metrics on every band is what makes the §2.2
argument checkable: you cannot read one without the other two.

### PP-5

Carried from `I-5`. A timed-out request is a request whose latency is unbounded and whose tokens
are uncounted; averaging over the survivors reports the throughput of the requests that happened
to finish. Fatal to the band's ratio, not to the run, so the other bands still report.

### PP-6

The predecessor put a comparator wall-clock ratio in the merge phase, where four such assertions
have already failed in this repository and one blocked every open pull request. The phase is
declared per arm in `perf-matrix.yaml` and `run_gate` must obey it; that it did not was invisible
because the arms simply never received the argument. The must-fire case promotes the parity arm to
merge and expects RED.

### PP-7

`I-4`, and `CO-2`'s shape. Retaining the raw per-request rows is what makes every derived statistic
re-derivable after the fact — the 2026-09-01 cells owe a bootstrap interval that was never
computed, and it is recoverable only because the samples were kept. A receipt that reports a median
and discards the sample is asking to be trusted.

### PP-8

`B-5` and `C-4`. If the *client* drives the comparator at concurrency 1 while the band is 16, the
comparator's aggregate is a single-stream number wearing a band label, and the ratio is
manufactured. The rule binds on a two-lane receipt; until the JOIN lands (§12 row 7) there is only
one lane to check, which is why the master's §6 says so on that row rather than claiming more.

### PP-9

`I-9`, with `CO-1` and `CO-2` behind it. A cell that may be re-run until it goes green measures the
experimenter, not the server. The key includes the commit — a later commit is a legitimately new
row — and `interleaved`, because a non-interleaved sweep is a different experiment. `CO-2` is why
the ledger has two conformance tiers: PP-9 binds on `RECORDED`, the tier a row can actually reach
today, so the rule applies from the first row rather than after five missing producers land.

### PP-10

`I-14`. A request issued after the window closes has its prefill inside the window and its decode
outside it, or the reverse; counting it inflates or deflates the aggregate depending on which side
it lands. Drain the in-flight ones, record `drain_ms`, count nothing issued at or after close.

### PP-11

`I-13`. Two servers that count tokens differently produce a ratio of two different quantities.
`tokenization.method` has no default because a default is exactly the value nobody checks; absence
is schema-fatal.

### PP-12

`C-1`'s class, applied to numbers rather than to prose. A figure a user reads is the defect; the
receipt is what makes it re-derivable. The rule now reaches `docs/specifications/` because a
specification is also read, and because the predecessor's own §2 published `c>1` ratios over
tokens its §9 called garbage. The exception the guard keeps for `docs/archive/` is deliberate:
archiving is how this repository retires a document without deleting the record, and a guard that
punishes the archive teaches people to delete instead.

### PP-13

`B-8`. A harness-declared value passes any rule keyed on it, whatever the build did — the
2026-09-01 receipts carry `feature_set: ["cuda"]` because a flag said so, not because the binary
did. A field about the server that the harness fills in is not evidence; it is a restatement of the
command line.

### PP-14

`I-17`. Auto-fit exists to fill in what the operator left unsaid. The moment it overrides something
the operator *did* say, the receipt describes a configuration nobody asked for and the argument
that produced it is unrecoverable. `autofit_applied[] ∩ explicit_args = ∅` states it as a set
identity so the check is mechanical.

### PP-15

`I-18` and §9 #9. A boolean accelerator flag cannot express "how much", so it cannot be checked
against what the loader resolved — and the published binary once accepted such a flag and ran on
CPU anyway. A quantity has a resolved counterpart the server can report, which is what makes PP-2
able to falsify it.

### PP-16

Round 4's finding, in its second form. `mini` declared a compute class no build can reach, so its
cells were permanently `UNMEASURED`, and §12's expiry rule turns permanent `UNMEASURED` into a
release FAIL — a cell blocking every release with no legal move. `NA{decided_by}` is the move: out
of the denominator, with a name and a date on the decision.

### PP-17

P-2 as a schema rule, so that an unnameable claim cannot be serialised. A `ratios` object is only
representable inside a band, and a band carries its concurrency; the two together are the cell,
band and metric P-2 demands.

### PP-18

The mechanism-engaged rule, applied to provenance. A receipt names three binaries; if any of them
was built from a commit that is not an ancestor of the commit under test, the receipt is about a
tree nobody can check out. `git merge-base --is-ancestor` is the whole check, and the
`PERF_GATE_GIT_DIR` seam is what lets the case table exercise it without a real repository.

### PP-19

`I-7`, and the reason it is not bookkeeping: the largest dispersion figure in the 2026-09-01 gx10
cell was traced to a device-wide stall, which is exactly what an isolation rule exists to prevent.
One global concurrency group per host with `cancel-in-progress: false`, plus a foreign-PID check
around every band, because a shared device is a confound the statistics cannot remove.

### PP-20

`I-8`. A comparator without an expiry drifts silently: the same pin name resolves to a different
build months later, and every cross-time comparison in the ledger becomes incomparable with no diff
in this repository to show for it. A stale pin marks the ratio `COMPARATOR_STALE` rather than
failing the run, because the measurement is fine — its *comparability* is what expired.

### PP-21

`CO-2`. The release gate already printed a signature failure while the ledger's own criteria
claimed signatures existed. A receipt is a document that travels; without a signature over its
bytes, "the receipt says" means "some file said". The signature covers the body and the commit
under test.

### PP-22

`I-11`, extended by `C-4`. Two bands, two window lengths, two KV layouts or two batch settings are
not comparable, and the failure is silent: the arithmetic works and the answer is meaningless. The
join key carries `n_batch` and `n_ctx_slot` precisely because §5.3's decision changes them, so a
comparator crippled with a batch size of one cannot be joined even by accident.

### PP-23

`B-9`, and round 5's correction of it. Bandwidth over bytes-per-token is a *per-sequence decode*
ceiling; under batching `N` sequences share one weight read per step, so aggregate legitimately
exceeds it. Stated without naming a metric, the rule declared the epic's own correctly-batching
gx10 receipts a harness bug. It is now decode-only, which leaves it with no applicable input until
a measured bandwidth is committed under `evidence/bandwidth/` — and that is the honest position
rather than one it can apply wrongly.

### PP-24

`B-14`, and round 4's "unpassable band". Comparing a band that one lane cannot admit measures
queueing on one side and service on the other. Both lanes report `slots_admitted`; the ladder is
derived from the minimum; a deliberate, server-reported ceiling yields `NA{decided_by, budget}`
rather than a permanent `UNMEASURED` that expires into a FAIL.

### PP-25

`I-15`. Two clients differ in connection reuse, header set, streaming parser and timing origin, and
every one of those differences lands in the ratio. One binary drives both lanes, and its sha256 is
in the receipt so the claim is checkable rather than asserted.

### PP-26

`C-15`, the audit's largest design finding. Nothing in the receipt schema witnessed that the tokens
counted were correct, and the probe that would have caught it was in the tree and wired to nothing.
The witness is cheap and specific: at `temperature 0` the token sequence produced at `m=1` and
inside an `m=c` batch must agree to a declared divergence point. The divergence point is `[U]`
until floating-point non-determinism is measured, and it lives in `perf-matrix.yaml` with an author
so that the `[U]` is visible rather than baked into a script.

### PP-27

`C-12`. Streaming is where TTFT, inter-token latency and decode come from; without it those three
metrics are undefined, which the 2026-09-01 producer said in plain text and then continued. Worse,
a server can *replay* a completed generation as a stream, and every client-side timing then
describes the replay. Hence the dual witness: the server declares `stream_mode`, the client
independently computes `ttft/e2e` (which approaches 1.0 on replay), and disagreement sends the
three metrics to `unproduced_fields` with a reason instead of publishing them.

### PP-28

`C-12`'s other half. A sampler that is not pinned on the wire makes two lanes run two experiments,
and a generation budget that is not enforced makes the band's token count a property of the prompt.
The free must-fire fixture was already in the tree: retained samples that stopped short of the
budget while the receipt's `truncated` counter read zero, because that counter counts
drain-abandoned requests and nothing witnessed the shortfall. `short_of_n_predict` is the missing
counter.

### PP-29

`C-20` — gates-or-theater, applied to the specification itself. Nothing verified that an `ARMED`
row had a check, so a row could say ARMED and mean documentation. The §6 table carries the selftest
names and the surface each lives on, which turns the check into a *join* rather than a grep, and
`scripts/spec_conformance.sh` runs it in the merge phase. It replaces the v2.2 mutation registry,
whose scanner found its input by a glob over a filename that no longer exists.

### PP-30

The ledger's own closing note: receipts carry no clock, so both 2026-09-01 rows are dated by the
commit that added their evidence. Dating a run by when someone committed it is not dating the run,
and every freshness rule (PP-20's expiry, PP-9's key, the ratchet's "last MEASURED") needs a real
timestamp with a named source.

### PP-31

`B-10` and `C-3`, the fourth gate in the predecessor to outlaw its own fix.
`scaling_efficiency(c) = agg(c) / (c · agg(1))` carries single-stream throughput in its
denominator, so ratcheting it up-only *rejects* the single-stream improvement the epic's speed rows
exist to deliver, and *rewards* halving it. The replacement ratchets the quantities users
experience — `agg(c)`, `dec(c)`, and `prefill` at c=1 — each seeded at what the last `MEASURED`
receipt on protected `main` achieved. `scaling_efficiency` stays, as a reported diagnostic.

### PP-32

The engine track needs to move today, without a comparator and without a matrix run, and it must
not be able to smuggle a parity claim out through the back door. `AbRecord` is therefore typed so
that a comparator cannot be expressed: no field can hold a second runtime's name or a parity
verdict, `serde(deny_unknown_fields)` refuses one added later, and the two arms are interleaved
because an A/B whose arms ran an hour apart measures the hour.

### PP-33

`C-17`. Thresholds lived in three files, and a number in a script is unauthored — nobody can say
who chose it or against what. One file owns every bound, each with a `threshold_class` and an
author, and a numeric comparison in a gate script that is not read from that file is RED.
Definitional comparisons (a count against zero, a band against one) are not thresholds and are
explicitly out of scope, because widening the rule to cover them would make it unusable and
therefore ignored.

---

## §5.1 — the protocol decision

Two bounds, not one. A sample-count bound alone lets a fast host finish in a few seconds, inside
the window where clocks, boost states and graph capture are still settling; a wall-clock bound
alone lets a slow host report a band from a handful of requests. The band therefore ends when
**both** are satisfied: at least `max(30, 8·c)` retained samples and at least `window_ms` of wall
clock. The `8·c` term keeps the per-worker sample count roughly constant as the band widens, so the
per-request statistics of §4.3 do not thin out at c=16.

Warmup is counted in *requests per worker*, not seconds, because what needs warming is per-worker:
the connection, the slot, the captured graph. Two requests per worker is the smallest number that
retires a cold connection and a first capture. The quiesce after warmup, and the cooldown between
the two lanes of a replicate, exist for the same reason interleaving does — thermal and clock state
carries across a lane boundary, and a comparator measured immediately after the subject's window is
measured on a hotter device. The cooldown is new in v3.0: interleaving cancels a *linear* drift
across a sweep, but not a step change at each hand-over.

Every one of those quantities lives in `scripts/perf-matrix.yaml` under `protocol:` rather than in
`protocol.rs`, because PP-33 applies to the protocol as much as to the arms: a constant in Rust is
a bound nobody authored. The Rust keeps the values only as test fallbacks.

Adding `ignore_eos` to the W1 corpus rotates its sha256, which is a component of both the PP-22
join key and the PP-9 cell key. That is intended and is stated in the master so nobody reads the
resulting join refusal as a bug: receipts taken before the rotation are keyed to the old digest and
do not join to receipts taken after it. It is also why the rotation happens exactly once, in the
same change that lands PP-28.

---

## §5.3 — the comparator decision, and the dissent

`C-4` found the predecessor asserting the comparator contract in §5.2 while §12.3 recorded the same
question as open with no owner. Both could not be the rule. The decision, `decided_by: spec-owner`,
is that **the comparator is `llama-server` configured to serve the band**: `-np c`,
`-c c·n_ctx_slot`. The argument is that parity is a claim about serving the same offered load, and
a comparator that queues twelve of sixteen requests is not serving the band; `-np c` is the
documented way to serve `c` users.

`C-4` also falsified the premise the opposing position rested on. The predecessor said the
comparator "serves 4 slots by design" at c=8 and c=16; its own withdrawn table shows 3.93, 7.84 and
15.74 sequences in flight at c = 4, 8 and 16 `[C]`, i.e. `c` sequences, not four. That is why §12
row 3's first action is to read `/props` at the withdrawn run's argv rather than to argue further.

### Dissent (`scripts/llama_pin.toml`, 2026-08-28) — moved verbatim

The pin is a declaration the gate reads (§0.1 of the master). After §5.3 decided the comparator,
about ninety lines of the pin's comments instructed the opposite of the decided rule, so a reader
of the pin alone learned the dissent as the rule. The prose is preserved here, verbatim, and the
pin now carries a short pointer to the decided rule and to this section.

**Block A — the batch-size history (`llama_pin.toml:100-125`, with the gx10 table at `:104-108`):**

```toml
# IT WAS `1`, WHICH IS WHY THIS COMMENT IS LONG (#2737). `-b 1` is not a tidy
# control; it switches llama.cpp's batching off. Measured on gx10 against the
# pinned comparator, medians of 2 runs at `-ngl 999`:
#
#     band          agg tok/s   decode tok/s
#     c=1   -b 1        185.4          186.5
#     c=1   default     184.9          186.0
#     c=16  -b 1        181.9          181.8
#     c=16  default     434.2          109.0
#
# THE c=1 ROW IS THE FALSIFIER. There `-b 1` changes nothing (0.3%), so this was
# never a general slowdown — it was specifically the batching path being turned
# off. Under `-b 1`, `agg == decode`. Under the default, `agg ~= 4 x decode`,
# and that 4 is the server's own `n_parallel` showing up in the arithmetic
# (see comparator_parallel below).
#
# Effect on the number this epic exists to compute: the c=16 aggregate ratio
# moved 2.03x -> 4.85x, a 2.39x overstatement in apr's favour, manufactured by
# handicapping the baseline rather than by apr getting faster.
#
# WHY "AS A USER RUNS IT" IS THE RIGHT DENOMINATOR. Arm B1's floor is policy and
# §4.6.1 states that policy in words: *below 0.80x a user is better served by
# llama.cpp.* That question only means something against the llama.cpp a user
# actually gets. Imposing apr's own documented non-batching defect (PERF-000) on
# the comparator answers a different question — "is apr better than a llama.cpp
# we broke the same way as apr" — and answers it flatteringly.
```

**Block C — `comparator_parallel` kept at "default" on purpose (`llama_pin.toml:128-164`):**

```toml

# The comparator's server slot count (`-np` / `--parallel`), declared for the
# same reason as batch_size and resolved the same way: "default" means pass no
# flag (#2737).
#
# THE PREMISE THAT THIS IS HOST-DERIVED IS WRONG, and the correction matters.
# Read from the pinned source rather than inferred from two hosts agreeing:
#
#   common/arg.cpp:991          `params.n_parallel = -1;  // auto by default`
#                               reached only under `ex == LLAMA_EXAMPLE_SERVER`
#   tools/server/server.cpp:86  `if (params.n_parallel < 0) { ... n_parallel = 4;
#                                params.kv_unified = true; }`
#
# So the auto value is the CONSTANT 4, baked into the pinned build. It does not
# consult core count, VRAM, or anything else about the host. Both servers picked
# 4 on gx10 because 4 is what 39173bcac picks everywhere, not because gx10 has
# some property. That also explains the `agg ~= 4 x decode` in the table above.
#
# It is therefore pin-determined and reproducible — but only by someone who
# reads llama.cpp's source, and §4.4's test is that a skeptical outsider can
# reproduce both invocations FROM THIS FILE. Two further reasons to write it
# down rather than leave it implied:
#
#   · it is entangled. The same branch flips `kv_unified = true`. A reader who
#     knows only "slots = 4" does not know the KV cache layout changed with it.
#   · it moves on a pin bump, silently. A future llama.cpp that auto-picks 8
#     would change every band above c=4 with no diff in this repo, which is the
#     exact cross-time drift build_commit exists to prevent.
#
# KEPT AT "default" ON PURPOSE. Pinning a number here would be the -b 1 mistake
# again in a second costume: at c=8 and c=16 the comparator runs 4 slots against
# 8 and 16 closed-loop clients, and that is a real constraint a real user hits.
# §4.6.1 asks whether the user is better served by llama.cpp, so the comparator
# must be the one the user gets. The resolved value is not left to inference —
# the server prints `n_parallel is set to auto, using n_parallel = 4` about
# itself, which is the §4.4.9 requested-vs-resolved pattern, and
# parity_host_receipt.sh REPORTs that line per lane.
```

A third block, the withdrawn 2026-08-25 lambda band table (`llama_pin.toml:218-237`), belongs to
the ledger rather than to this file and is recorded at `evidence/parity/LEDGER.md` beside row 0b. A
fourth, a bootstrap paragraph describing a pin value that has not been current since 2026-08-24, is
deleted; its deletion is reviewable in the pull request diff.

### Where the dissent is wrong, measured at the pin

Block C asserts the auto slot count is the constant 4 and that `agg ~= 4 x decode` follows from it.
The `/props` reads committed as `evidence/parity/props-39173bcac-template.json` and
`evidence/parity/props-39173bcac-np16.json` (both `lambda`, both build `39173bcac`, 2026-09-02)
confirm the slot count and settle the `n_ctx` question the block left implied. At the template argv
(`-c 4096`, no `-np`) the server reports `total_slots = 4` with
`default_generation_settings.n_ctx = 4096` — each of the four slots is given the FULL context.
Under the decided configuration (`-np 16 -c 16384`) it reports `total_slots = 16` with
`n_ctx = 1024` per slot. So the two configurations differ in per-slot KV budget as well as in slot
count, which is the entanglement Block C itself warned about and then used as an argument for
leaving the value implied.

The `kv_unified` half of Block C's argument is **not** witnessed by these files: neither `/props`
body carries a `kv_unified` field at all, so its value at either argv is `[U]` from the evidence
and `[A]` from the source lines Block C cites. The correction is therefore about *provenance*, not
about the claim: reading a flag out of `tools/server/server.cpp` and reading it back off the
running server are different acts, and only the second is a receipt.

The dissent is kept anyway, and kept verbatim, for the reason the audit gives for keeping the
withdrawn §2.1 table: the batch-size episode is the strongest available argument against
configuring a comparator at all, and a decision that hides its strongest counter-argument is how a
premise survives that nobody actually believes. The must-not-fire fixture for §5.3 is a run with a
batch size of one, which must be **refused** as a comparator configuration — the PP-22 join key
carries `n_batch` so that refusal is mechanical rather than editorial.

---

## §10 — the falsifiers

The master's §10 registers five predictions. A prediction with no kill criterion is a hope, so each
carries the observation that retires it. They are restated here with the reason each kill criterion
is the right one.

**§9 #1 probe (gx10).** Predicted: prefill wall time linear in prompt length at about 32.6 ms per
prompt token on the default arm, collapsing to roughly 0.35 s at 512 prompt tokens under the
batched-prefill arm. **Kill if** the wall time is flat in prompt length, or does not collapse.
Flatness would mean the cost is a fixed setup rather than a per-token loop, which is a different
defect with a different fix; no collapse would mean the serial loop is not the mechanism at all,
and §12 row 21 is scoped wrong.

**§9 #5 fix (lambda).** Predicted: batched decode at most 3.5 ms per token once the per-token host
work leaves the loop, and `agg(2)` above one client. **Kill if** it stays above 3.5 ms per token
after that work is removed — then the penalty is not host-side bookkeeping, and the named lever is
the wrong one.

**§9 #3 fix (lambda).** Predicted: synchronous copies and device allocations fall to under a tenth
of CUDA API time and prefill more than doubles. **Kill if** the share stays above three tenths. The
profile in `evidence/parity-http/` puts kernel launches at well under one part in a hundred of API
time, so if removing the copies does not move the share, the profile was measuring the wrong
process.

**Effective-config endpoint.** Predicted: `max_batch = 11` reconstructs from the reported inputs.
**Kill if** it does not — then the KV budget is not the mechanism that sets admission, and §9 #7's
memory finding is scoped wrong. This falsifier is currently *untestable*, and the master says so:
free VRAM at the sizing instant is recorded nowhere, and the two recorded residency figures imply a
value one and a half to two and a half gigabytes away from the one the arithmetic needs. Making it
testable is precisely what §12 row 6 delivers.

**JOIN fixture.** Predicted: the new Rust reproduces the eight committed ratios to four decimals
with no GPU. **Kill if** any digit differs. The kill criterion is strict on purpose: the fixture is
the only end-to-end test of the ratio path that needs no hardware, and a JOIN that is *close* is a
JOIN with an undiagnosed difference. The statistic must be named to reproduce it — median over the
two runs per lane of the run-level fields, not the master's §3 per-request estimator, which the
fixture cannot express — and that is a property of a 2026-08 artifact, so the fixture mode is
explicitly historical.

---

## PP-12 — the widening, and what it cost

`check_no_claim_literals.sh` excluded `docs/specifications/` on an argument its own header states:
a specification is where a measured number belongs *with its provenance*, and sweeping the
directory in wholesale would force a document to describe its own subject matter in euphemism —
including the figures a spec quotes in order to ban them. The counter-argument won: the master is a
specification that publishes ratios, and a rule whose universe excludes the document that states it
is not a rule.

The widening deletes the `docs/specifications/` exclusion and keeps the `docs/archive/` one, for
the reason above: nobody installs `apr` and reads an archive, and a guard that punishes the archive
teaches people to delete rather than to archive. Its cost is recorded in
`scripts/claim_literal_baseline.txt` under a dated `# PP-12 widening 2026-09-02` comment — 327
recorded locations after the widening, of which 202 are the specification prose the exclusion had
been hiding. They are *recorded, not blessed*: the file is shrink-only and compared against
protected `main`, so an append is refused rather than discouraged, and every one of those 202 lines
leaves the file by being receipted or deleted.

Two of them are named in the master's §12 row 10 because they are live product claims rather than
spec prose, and one document — `docs/benchmarking-gate-spec.md` — is archived instead of
baselined, because it republished the withdrawn comparator table that the master's §2.1 exists to
refuse.
