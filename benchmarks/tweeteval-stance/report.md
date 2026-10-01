SetFit - benchmark claims report
contract: setfit-benchmark-claims-v1
design:   10 seeds per cell, df = 9, 95% CI uses the frozen t = 2.262157162798205
verified: every cell of the contracted matrix. A missing, substituted, unmatched or
          post-test-selected cell would have REFUSED this report rather than
          shrunk it, and the evidence below was RECOMPUTED from
          the committed lock bytes, the committed selection manifest, and
          the row's own confusion matrix
          rather than read off the rows.
residual: every published accuracy figure is recomputed from the row's OWN confusion
          matrix, so a doctored figure is refused — but the matrix itself is
          producer-written, and a producer who edits it and recomputes the
          figures from it emits a set this report cannot distinguish from a
          measurement. The two calibration diagnostics are not recomputable at
          all: no committed file carries the per-row probabilities they need.
          `evidence_table_hash` and `apr_artifact_sha256` stay claims about
          artifacts this index does not carry.

SCOPE - ONE METHOD WAS MEASURED. This report covers SetFit alone. A second method was planned for this matrix and was not run; the reason is recorded as decision D-19 and the restoration path as ticket D-ITEM-05-15. Nothing here states or implies any result about a second method. Read the absence as absence.

QUALITY - official F_avg, mean +/- (n-1) std [min, max]
method   shots       mean       std        min        max      95% CI (seeds)   (macro F1 mean / MCC mean)
setfit       8     0.4746    0.0456     0.4057     0.5318   [0.4420, 0.5072]       0.4643   0.2656
setfit      16     0.5115    0.0383     0.4437     0.5678   [0.4841, 0.5389]       0.4988   0.3059
setfit      32     0.5346    0.0208     0.4996     0.5678   [0.5197, 0.5495]       0.5133   0.3198
setfit      64     0.5607    0.0270     0.5185     0.6026   [0.5414, 0.5800]       0.5343   0.3397
F_avg = (F1_against + F1_favor) / 2, the official TweetEval stance metric. Macro F1
is published BESIDE it and never instead of it.
95% CI over the ten contracted seeds at fixed data and protocol - a seed-dispersion interval, not a population interval.

RESOURCE - per method and host, every figure with its measurement boundary
as-deployed costs on ONE host for ONE method; every figure carries its measurement boundary, and no figure here is averaged across hosts

setfit / s8  host: MacBook-Pro-7.local (macos/aarch64)  backend: cpu:setfit-core:autograd-trueno-matmul
  train wall                                 31236.6 ms
  cold latency (fresh child)                   738.4 ms
  warm latency (median of 10 after 3)           32.4 ms
  throughput                                    81.9 rows/s @ batch 32
  train peak RSS (training process)       5560488755 B   mechanism: sysinfo_sampled_19 [sampled_lower_bound]  LOWER BOUND (sampled; can only understate)
  inference peak RSS (cold child)         1553154048 B   mechanism: child_max_rss_time_l [exact_kernel_high_water_mark]
  ^ The two figures above are TWO metrics from TWO processes, and here they come from two different mechanism CLASSES. They are never added, never averaged, and never reduced to a single figure: a kernel high-water mark is process-cumulative, so the training process's peak is not the inference peak.
  artifact bytes (what this method wrote)    90777156 B

setfit / s16  host: MacBook-Pro-7.local (macos/aarch64)  backend: cpu:setfit-core:autograd-trueno-matmul
  train wall                                112993.7 ms
  cold latency (fresh child)                   716.9 ms
  warm latency (median of 10 after 3)           32.0 ms
  throughput                                    83.3 rows/s @ batch 32
  train peak RSS (training process)       5824020480 B   mechanism: sysinfo_sampled_19 [sampled_lower_bound]  LOWER BOUND (sampled; can only understate)
  inference peak RSS (cold child)         1555839386 B   mechanism: child_max_rss_time_l [exact_kernel_high_water_mark]
  ^ The two figures above are TWO metrics from TWO processes, and here they come from two different mechanism CLASSES. They are never added, never averaged, and never reduced to a single figure: a kernel high-water mark is process-cumulative, so the training process's peak is not the inference peak.
  artifact bytes (what this method wrote)    90777156 B

setfit / s32  host: MacBook-Pro-7.local (macos/aarch64)  backend: cpu:setfit-core:autograd-trueno-matmul
  train wall                                443426.6 ms
  cold latency (fresh child)                   743.5 ms
  warm latency (median of 10 after 3)           34.2 ms
  throughput                                    79.0 rows/s @ batch 32
  train peak RSS (training process)       5828229530 B   mechanism: sysinfo_sampled_17, sysinfo_sampled_18, sysinfo_sampled_19 [sampled_lower_bound]  LOWER BOUND (sampled; can only understate)
  inference peak RSS (cold child)         1553026253 B   mechanism: child_max_rss_time_l [exact_kernel_high_water_mark]
  ^ The two figures above are TWO metrics from TWO processes, and here they come from two different mechanism CLASSES. They are never added, never averaged, and never reduced to a single figure: a kernel high-water mark is process-cumulative, so the training process's peak is not the inference peak.
  artifact bytes (what this method wrote)    90777098 B

setfit / s64  host: MacBook-Pro-7.local (macos/aarch64)  backend: cpu:setfit-core:autograd-trueno-matmul
  train wall                               1739087.8 ms
  cold latency (fresh child)                   721.2 ms
  warm latency (median of 10 after 3)           33.9 ms
  throughput                                    79.8 rows/s @ batch 32
  train peak RSS (training process)       5238762701 B   mechanism: sysinfo_sampled_17, sysinfo_sampled_18, sysinfo_sampled_19 [sampled_lower_bound]  LOWER BOUND (sampled; can only understate)
  inference peak RSS (cold child)         1554453299 B   mechanism: child_max_rss_time_l [exact_kernel_high_water_mark]
  ^ The two figures above are TWO metrics from TWO processes, and here they come from two different mechanism CLASSES. They are never added, never averaged, and never reduced to a single figure: a kernel high-water mark is process-cumulative, so the training process's peak is not the inference peak.
  artifact bytes (what this method wrote)    90777143 B

MODEL SIZE - deployable_total_bytes, what a user must ship to serve this
method   shots   deployable_total_bytes
setfit       8                 90777156
setfit      16                 90777156
setfit      32                 90777098
setfit      64                 90777143
A size claim uses THIS column and no other. The artifact bytes in the per-method
detail above are what this method WROTE, which is a different question from what a
user must ship to serve it.

Estimation-first (D-08): point estimates, dispersion and 95% seed-dispersion intervals only. No binary verdict is printed.
