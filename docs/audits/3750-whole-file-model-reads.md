# #3750 — every whole-file read in apr-cli, aprender-serve and aprender-core

Measured on `PMAT-3750-qa-header-only-reads` (base `52f43da71`) with
`git grep -n -E '(std::)?fs::read\(|read_to_end\(' -- 'crates/apr-cli/src/**/*.rs' 'crates/aprender-serve/src/**/*.rs' 'crates/aprender-core/src/**/*.rs'`:
**231 hits** after this PR: **99 production**, **132 test-only**, plus the **13 `apr qa` reads this PR converted** (they no longer match).

Verdicts:
- **CONVERTED** — read only the header or magic now (this PR).
- **PR-B …** — needs only the magic / header / metadata and is converted by the 0.69.1 sub-issue #3761 (#3750 PR B), which puts the APR and SafeTensors prefix readers in their format crates.
- **PR-B STREAMED** — needs every byte, but never all of them at once; #3761 streams it.
- **WHOLE-DATA** — consumes the tensor data or every byte (loaders, converters, validators, copies, uploads); "could stream" notes a whole-file buffer that is not needed all at once.
- **NOT-MODEL** — reads a file that is not a model. **DOC** — a doc example, not executed code.

Counts over the 99 production sites: DOC 9, NOT-MODEL 11, PR-B CONDITIONAL 1, PR-B HEADER-ONLY 11, PR-B MAGIC-ONLY 4, PR-B STREAMED 1, WHOLE-DATA 62.

## Converted by this PR (the `apr qa` path, line numbers on `52f43da71`)

| site | function | was | now |
|---|---|---|---|
| apr-cli/src/commands/qa_capability.rs:46 | run_capability_gate | whole file for a 4-byte magic + the header | `read_prefix(4)` + `gguf_arch_and_tensors` |
| apr-cli/src/commands/qa_capability.rs:377 | cpu_only_architecture | whole file + a `to_vec` copy for the header | `gguf_arch_and_tensors` |
| apr-cli/src/commands/qa_capability.rs:406 | hybrid_loader_architecture | whole file + a `to_vec` copy for the header | `gguf_arch_and_tensors` |
| apr-cli/src/commands/qa_capability.rs:431 | extract_gguf_arch_and_tensors | `data.to_vec()` of the whole file | removed |
| apr-cli/src/commands/qa_gguf.rs:6 | is_gguf_format | whole file for 8 bytes | `read_prefix(8)` |
| apr-cli/src/commands/golden_output.rs:277 | run_golden_output_gate | whole file for the magic + a GGUFModel beside the map | `read_prefix(8)` + `mapped.model` |
| apr-cli/src/commands/golden_output.rs:410 | throughput_gguf | parsed the whole-file bytes read at speedup.rs:49, for the tokenizer | `mapped.model.encode`; the bytes parameter is gone |
| apr-cli/src/commands/output_verification.rs:19 | run_metadata_plausibility_gate | whole file for metadata | magic + GGUF header prefix / APR header+metadata+index |
| apr-cli/src/commands/forward_error.rs:526 | run_format_parity_gate | whole file for 8 bytes, under a comment saying "cheap — no full-file read" | `read_prefix(8)` |
| apr-cli/src/commands/forward_error.rs:549 | run_format_parity_gate | whole file for the tokenizer | `mapped.model.encode` |
| apr-cli/src/commands/gpu_isolation_result.rs:210 | run_gpu_isolation_test | whole file for the tokenizer | one map, `mapped.model` |
| apr-cli/src/commands/speedup.rs:49 | run_throughput_gate | whole file for the magic | `read_prefix(8)` |
| apr-cli/src/commands/speedup.rs:248 | measure_our_gguf_tps | whole file for the tokenizer | one map, `mapped.model` |
| apr-cli/src/commands/speedup.rs:419 | measure_gpu_cpu_tps | whole file for the tokenizer | one map, `mapped.model` |
| apr-cli/src/commands/speedup.rs:504 | run_gpu_speedup_gate | whole file for the magic | `read_prefix(8)` |

(13 whole-file reads and one whole-file copy.)

## Measured on gx10 (done_when 3)

`apr qa` on `~/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (18,556,689,568 bytes), on gx10
(GB10, aarch64, unified memory), both binaries built there with `--features cuda`, each run under
`flock /tmp/apr-gpu.lock choom -n 1000`, with `--json --offline --skip-golden --skip-throughput
--skip-ollama --skip-gpu-speedup --skip-ptx-parity --skip-gpu-state --skip-format-parity`. Peak RSS
is `/usr/bin/time -v`'s "Maximum resident set size". A second pair samples `/proc/<pid>/status`
every 0.2 s for RssAnon (heap) and RssFile (mapped pages) apart, because the maximum sums them.
The `--skip-contract` rows each combine two runs of the same binary and flags: max RSS from a
`/usr/bin/time -v` run, RssAnon and RssFile from a sampled run.

| binary (`apr --version`) | gates | max RSS (KiB) | peak RssAnon (KiB) | peak RssFile (KiB) | qa |
|---|---|---|---|---|---|
| before: `apr 0.69.0 (52f43da71)` | as above | 36,278,440 | 36,112,744 | 11,160 | passed, rc 0, 4:36 |
| after: `apr 0.69.0 (ff89fb6aa)` | as above | 19,367,800 | 19,357,084 | 11,236 | passed, rc 0, 3:57 |
| before: `apr 0.69.0 (52f43da71)` | as above + `--skip-contract` | 36,273,092 | 36,247,676 | 10,596 | passed, rc 0, 1:10 |
| after: `apr 0.69.0 (ff89fb6aa)` | as above + `--skip-contract` | 46,396 | — (the run took 0.14 s, under one 0.2 s sample) | | passed, rc 0, 0:00.14 |

What the numbers say:

- **The capability predicates alone** (the `--skip-contract` pair, where the gates that run are
  capability_match, metadata_plausibility and performance_regression) went from a 36.2 GB heap
  peak to a 46,396 KiB (45.3 MiB) process peak, 1:10 to 0.14 s: two whole-file reads and a `to_vec` copy of an 18.6 GB model, then
  none. That is done_when 1 on the real 30B, beside the sparse-fixture row in `model_header.rs`.
- **The 19.36 GB left in the full run is heap, not a mapping** (RssFile 11 MB), and it belongs to
  the `tensor_contract` gate: `RosettaStone::validate` → `validate_gguf` → `GgufReader::from_file`,
  a `read_to_end` of the whole model, then a dequantization of every tensor. That gate reads every
  tensor value (NaN/Inf/all-zero checks), so it is WHOLE-DATA by reading, but it does not need the
  whole file resident at once. The row `gguf/reader_parsing.rs:6` below says so, and #3790
  tracks streaming it from a map.
- 36,278,440 − 19,367,800 = 16,910,640 KiB (16.1 GiB) less peak per `apr qa` run on this model.

## Every production site

| site | function | verdict | what the bytes are used for |
|---|---|---|---|
| apr-cli/src/commands/audio_inspect.rs:108 | inspect | NOT-MODEL | a WAV file; parses its fmt/data chunks |
| apr-cli/src/commands/benchmark.rs:104 | run_realizar_benchmark | PR-B MAGIC-ONLY | bytes used only for detect_format on the first 8 |
| apr-cli/src/commands/benchmark.rs:167 | run_gguf_benchmark | PR-B HEADER-ONLY | GGUFModel::from_bytes only for the tokenizer; the model is mapped separately |
| apr-cli/src/commands/canary.rs:166 | load_tensor_data_gguf | WHOLE-DATA | loads every GGUF tensor as f32 for the canary |
| apr-cli/src/commands/canary.rs:186 | load_tensor_data_apr | WHOLE-DATA | loads every APR tensor as f32 for the canary |
| apr-cli/src/commands/chat_session_02.rs:21 | new | WHOLE-DATA | the chat session keeps the model bytes and runs inference from them |
| apr-cli/src/commands/debug.rs:522 | run_strings_mode | WHOLE-DATA | `strings` mode scans every byte (could stream) |
| apr-cli/src/commands/diff_quant_roundtrip.rs:166 | load_tensors_f32 | WHOLE-DATA | loads every tensor as f32 to diff |
| apr-cli/src/commands/distill.rs:695 | run_cuda_backend | WHOLE-DATA | teacher model weights for distillation |
| apr-cli/src/commands/distill.rs:738 | run_cuda_backend | WHOLE-DATA | student model weights for distillation |
| apr-cli/src/commands/embed.rs:245 | run | WHOLE-DATA | APR v2 reader; reads the embedding weights |
| apr-cli/src/commands/embed_viz.rs:382 | gguf_vocab | PR-B HEADER-ONLY | LlamaTokenizer::from_gguf_bytes needs only the header vocabulary |
| apr-cli/src/commands/embed_viz_lint.rs:35 | run | NOT-MODEL | an output artifact whose determinism is classified |
| apr-cli/src/commands/eval/mod.rs:746 | count_safetensors_keys | PR-B HEADER-ONLY | SafeTensors: 8-byte length + JSON header, counts keys |
| apr-cli/src/commands/eval/mod.rs:922 | verify_single_file | PR-B STREAMED | SafeTensors header checks (bounded), plus an FNV-1a hash of EVERY byte: it needs the whole file, never all of it at once, so #3761 streams the hash in 1 MiB chunks |
| apr-cli/src/commands/eval/mod.rs:1314 | run_encrypt | WHOLE-DATA | encrypts every byte (could stream) |
| apr-cli/src/commands/eval/mod.rs:1399 | run_decrypt | WHOLE-DATA | decrypts every byte (could stream) |
| apr-cli/src/commands/eval/mod.rs:1478 | derive_encryption_key | NOT-MODEL | an encryption key file |
| apr-cli/src/commands/export.rs:346 | run_export_to_stdout | WHOLE-DATA | copies the exported model to stdout (could stream) |
| apr-cli/src/commands/finetune_display_next_validate.rs:455 | verify_merged_runnable | WHOLE-DATA | post-merge gate validates the written artifact's structure, then loads it |
| apr-cli/src/commands/finetune_display_next_validate.rs:514 | run_merge | WHOLE-DATA | merge reads the base model's tensors |
| apr-cli/src/commands/hex.rs:133 | run | WHOLE-DATA | hex dump of arbitrary offsets (could seek) |
| apr-cli/src/commands/hex.rs:176 | run_apr | WHOLE-DATA | APR hex dump of tensor bytes |
| apr-cli/src/commands/ppl.rs:22 | run | NOT-MODEL | a log-probs file |
| apr-cli/src/commands/publish.rs:255 | upload_to_hub_extended | WHOLE-DATA | uploads every byte (could stream) |
| apr-cli/src/commands/publish.rs:374 | upload_to_hub | WHOLE-DATA | uploads every byte (could stream) |
| apr-cli/src/commands/publish.rs:797 | emit_safetensors_alias | WHOLE-DATA | copies the file to its alias (could stream) |
| apr-cli/src/commands/pull_remove_resolve_model.rs:264 | fetch_safetensors_companions | NOT-MODEL | an HTTP response body (companion files) |
| apr-cli/src/commands/pull_remove_resolve_model.rs:441 | resolve_sharded_safetensors | NOT-MODEL | a sharded-SafeTensors index JSON |
| apr-cli/src/commands/rerank.rs:297 | run | WHOLE-DATA | APR v2 reader; reranker weights |
| apr-cli/src/commands/rosetta_validate.rs:308 | load_apr_tensors_direct | WHOLE-DATA | loads every APR tensor |
| apr-cli/src/commands/safetensors.rs:104 | execute_safetensors_inference | WHOLE-DATA | SafeTensors inference from the bytes |
| apr-cli/src/commands/serve/safetensors.rs:63 | start_safetensors_server | WHOLE-DATA | serves SafeTensors inference from the bytes |
| apr-cli/src/commands/shard/sharder.rs:164 | shard_safetensors_file | WHOLE-DATA | splits every tensor into shards |
| apr-cli/src/commands/shard/unsharder.rs:213 | unshard_safetensors_dir | WHOLE-DATA | reassembles every shard's tensors |
| apr-cli/src/commands/showcase/pipeline.rs:427 | run_apr_inference | WHOLE-DATA | APR inference from the bytes |
| apr-cli/src/commands/slice.rs:146 | slice_apr | WHOLE-DATA | slices tensor data |
| apr-cli/src/commands/stamp.rs:97 | run | WHOLE-DATA | rewrites the file with a new header |
| apr-cli/src/commands/tokenize.rs:2240 | sha256_file | NOT-MODEL | hashes the input corpus for source_sha256 (could stream) |
| apr-cli/src/commands/trace_likely_has_repeated.rs:77 | run_traced_inference_safetensors | WHOLE-DATA | traced SafeTensors inference |
| apr-cli/src/commands/trace_likely_has_repeated.rs:265 | trace_gguf | WHOLE-DATA | traced GGUF inference |
| apr-cli/src/commands/train.rs:1044 | copy_checkpoint_files | WHOLE-DATA | copies checkpoint files (could stream) |
| apr-cli/src/commands/tui.rs:187 | load_model | WHOLE-DATA | AprValidator::validate_bytes checks tensor content |
| apr-cli/src/commands/validate.rs:117 | run_apr_validation | WHOLE-DATA | AprValidator + fail-closed content gates on the tensors |
| aprender-core/src/bundle/mmap.rs:204 | open | WHOLE-DATA | loads the whole bundle (named mmap, reads; could map) |
| aprender-core/src/cluster/kmeans_impl.rs:110 | load | WHOLE-DATA | deserializes a small classical model |
| aprender-core/src/ensemble/moe.rs:323 | load | WHOLE-DATA | deserializes the ensemble |
| aprender-core/src/format/converter/apr_export_fn.rs:133 | detect_apr_architecture_for_completeness | PR-B HEADER-ONLY | APR metadata architecture only |
| aprender-core/src/format/converter/convert_report.rs:173 | load_apr_tensors_f32 | WHOLE-DATA | loads every APR tensor as f32 |
| aprender-core/src/format/converter/gguf_export_config.rs:516 | export_to_gguf | WHOLE-DATA | exports every tensor to GGUF |
| aprender-core/src/format/converter/metadata.rs:563 | export_apr_to_gguf_raw | WHOLE-DATA | raw APR -> GGUF export of every tensor |
| aprender-core/src/format/converter/tensor.rs:202 | extract_apr_tokenizer_hint | PR-B HEADER-ONLY | tokenizer hint from the APR metadata section |
| aprender-core/src/format/converter/tensor.rs:226 | read_apr_metadata | PR-B HEADER-ONLY | APR metadata only |
| aprender-core/src/format/converter/tensor.rs:382 | extract_user_metadata | PR-B HEADER-ONLY | user metadata from the APR metadata section |
| aprender-core/src/format/converter/tensor.rs:439 | detect_apr_quantization | PR-B HEADER-ONLY | counts tensor dtypes from the APR tensor index |
| aprender-core/src/format/converter/tokenizer_loader.rs:482 | load_tokenizer_from_sentencepiece | NOT-MODEL | a SentencePiece tokenizer.model |
| aprender-core/src/format/core_io.rs:173 | read_file_content | WHOLE-DATA | generic whole-content reader (callers decide) |
| aprender-core/src/format/gguf/reader_parsing.rs:6 | from_file | WHOLE-DATA, could stream | GgufReader::from_file owns the whole file by API (importers read tensors). `apr qa`'s tensor_contract gate reaches it through `RosettaStone::validate`: the 19.36 GB heap peak measured above. It reads every tensor, but never needs them all at once (#3790) |
| aprender-core/src/format/gguf/reader_parsing.rs:20 | from_file_full | WHOLE-DATA | GgufReader::from_file_full, the shard merge reads tensors |
| aprender-core/src/format/lint/lint.rs:187 | lint_safetensors_file | PR-B HEADER-ONLY | SafeTensors metadata from the header; tensors come from the existing map |
| aprender-core/src/format/lint/lint.rs:324 | lint_apr_v2_file | PR-B HEADER-ONLY | lints APR metadata fields |
| aprender-core/src/format/onnx/reader.rs:5 | from_file | WHOLE-DATA | parses the ONNX protobuf including initializers |
| aprender-core/src/format/onnx/reader.rs:467 | is_onnx_file | PR-B MAGIC-ONLY | checks data[0] == 0x08 only |
| aprender-core/src/format/rosetta/validate_inspect.rs:7 | validate_apr | WHOLE-DATA | rosetta validate reads tensor data |
| aprender-core/src/format/rosetta/validate_inspect.rs:552 | inspect_apr | PR-B HEADER-ONLY | rosetta inspect: metadata + tensor index entries |
| aprender-core/src/format/safetensors.rs:448 | list_tensors | PR-B CONDITIONAL | GGUF/APR v1 listing: data only with --stats |
| aprender-core/src/index/persistent_hnsw.rs:110 | open | NOT-MODEL | an HNSW index file |
| aprender-core/src/inspect/safetensors.rs:174 | from_file | WHOLE-DATA | keeps the bytes for later tensor reads |
| aprender-core/src/linear_model/elastic_net.rs:165 | load | WHOLE-DATA | deserializes a small classical model |
| aprender-core/src/linear_model/lasso.rs:73 | load | WHOLE-DATA | deserializes a small classical model |
| aprender-core/src/linear_model/lasso_impl.rs:100 | load | WHOLE-DATA | deserializes a small classical model |
| aprender-core/src/linear_model/mod.rs:129 | load | WHOLE-DATA | deserializes a small classical model |
| aprender-core/src/serialization/apr/mod.rs:156 | open | WHOLE-DATA | AprReader::open loads the model |
| aprender-core/src/serialization/apr/mod.rs:220 | open_filtered | WHOLE-DATA | AprReader::open_filtered loads selected tensors |
| aprender-core/src/serialization/safetensors.rs:505 | load_safetensors | WHOLE-DATA | loads every tensor |
| aprender-core/src/setfit/artifact.rs:1945 | read_setfit_apr_bytes_bounded_within | WHOLE-DATA | already bounded by `_within` (a size cap) |
| aprender-core/src/setfit/import.rs:605 | open_slice_fixture | WHOLE-DATA | a SetFit fixture slice |
| aprender-core/src/setfit/import.rs:699 | read_required | NOT-MODEL | a required SetFit sidecar file |
| aprender-core/src/tree/classifier.rs:156 | load | WHOLE-DATA | deserializes a small classical model |
| aprender-core/src/verify/ground_truth.rs:88 | from_bin_file | NOT-MODEL | a ground-truth .bin |
| aprender-serve/src/apr/helpers.rs:309 | is_apr_file | PR-B MAGIC-ONLY | data[0..4] == MAGIC only |
| aprender-serve/src/apr/helpers.rs:325 | format_from_magic | PR-B MAGIC-ONLY | format from the first 4 bytes only |
| aprender-serve/src/apr/loading_mmap.rs:48 | load | WHOLE-DATA | a COMPRESSED .apr must be decompressed whole |
| aprender-serve/src/apr/loading_mmap.rs:69 | load | WHOLE-DATA | the wasm32 fallback (no mmap) |
| aprender-serve/src/apr_transformer/from_apr_file.rs:32 | from_apr_file | WHOLE-DATA | loads the transformer's weights |
| aprender-serve/src/apr_transformer/mod.rs:19 | ? | DOC | a module doc example |
| aprender-serve/src/cli/mod.rs:290 | run_model_command | WHOLE-DATA | `realizar run` loads the model from the bytes |
| aprender-serve/src/cli/mod.rs:417 | run_chat_command | WHOLE-DATA | `realizar chat` loads the model from the bytes |
| aprender-serve/src/convert/mod.rs:11 | ? | DOC | a module doc example |
| aprender-serve/src/convert/q4k_conversion_stats.rs:98 | convert | WHOLE-DATA | converts every tensor to Q4K APR |
| aprender-serve/src/convert/q4k_converter_helpers.rs:223 | convert | WHOLE-DATA | converts every tensor to Q4K APR |
| aprender-serve/src/fixtures/mod_gguf_try_model.rs:225 | read_bytes | WHOLE-DATA | test-fixture harness helper reading a fixture model |
| aprender-serve/src/gguf/loader_parse.rs:18 | ? | DOC | a doc example |
| aprender-serve/src/gguf/metadata.rs:24 | ? | DOC | a doc example |
| aprender-serve/src/gguf/reading.rs:18 | ? | DOC | a doc example |
| aprender-serve/src/layers/mod.rs:28 | ? | DOC | a module doc example |
| aprender-serve/src/layers/mod.rs:55 | ? | DOC | a module doc example |
| aprender-serve/src/safetensors/safetensors_config.rs:19 | ? | DOC | a doc example |
| aprender-serve/src/safetensors/safetensors_parser.rs:19 | ? | DOC | a doc example |

## Test-only sites (132)

Reads inside `#[test]` functions, `#[cfg(test)]` modules or test files: fixtures a test wrote or reads back. Not on any production path.

| file | reads |
|---|---|
| apr-cli/src/commands/distill_include_01.rs | 5 |
| apr-cli/src/commands/probar_tests_export_png.rs | 2 |
| apr-cli/src/commands/pull.rs | 1 |
| apr-cli/src/commands/shard/tests.rs | 6 |
| apr-cli/src/commands/stamp.rs | 6 |
| apr-cli/src/commands/tokenize.rs | 3 |
| apr-cli/src/commands/train_tests.rs | 1 |
| aprender-core/src/citl/pattern/tests_pattern_persistence.rs | 1 |
| aprender-core/src/ensemble/mod.rs | 1 |
| aprender-core/src/format/converter/tests/convert_tied_lmhead.rs | 2 |
| aprender-core/src/format/converter/tests/core_conversion.rs | 1 |
| aprender-core/src/format/converter/tests/core_convert.rs | 2 |
| aprender-core/src/format/converter/tests/core_rosetta_gqa.rs | 6 |
| aprender-core/src/format/converter/tests/coverage_falsification.rs | 1 |
| aprender-core/src/format/converter/tests/coverage_gap_quantized_save.rs | 5 |
| aprender-core/src/format/converter/tests/coverage_rosetta_inspect.rs | 1 |
| aprender-core/src/format/converter/tests/coverage_types_pygmy.rs | 3 |
| aprender-core/src/format/converter/tests/coverage_write_apr_file.rs | 4 |
| aprender-core/src/format/converter/tests/dogfood_2392.rs | 4 |
| aprender-core/src/format/converter/tests/pmat.rs | 1 |
| aprender-core/src/format/converter/tests/pmat_round19.rs | 1 |
| aprender-core/src/format/converter/tests/pure_functions_infer_q4k.rs | 5 |
| aprender-core/src/format/converter/tests/streaming_quantize_test.rs | 3 |
| aprender-core/src/format/rosetta/minimal.rs | 2 |
| aprender-core/src/format/rosetta/tests_pygmy.rs | 1 |
| aprender-core/src/format/rosetta/tests_tokenizer_stress.rs | 2 |
| aprender-core/src/format/test_factory/collection.rs | 2 |
| aprender-core/src/format/test_factory/gguf_pygmy_config.rs | 2 |
| aprender-core/src/format/test_factory/harness_impl.rs | 1 |
| aprender-core/src/format/test_factory/harness_roundtrip_tests.rs | 1 |
| aprender-core/src/format/test_factory/harness_strict_tests.rs | 1 |
| aprender-core/src/format/test_model.rs | 4 |
| aprender-core/src/format/tests/encryption_proptests.rs | 1 |
| aprender-core/src/format/tests/error_proptests.rs | 2 |
| aprender-core/src/format/tests/signing_proptests.rs | 1 |
| aprender-core/src/format/tests/unit.rs | 4 |
| aprender-core/src/format/tests/unit_compression_signing_encryption.rs | 3 |
| aprender-core/src/serialization/safetensors_tests_core.rs | 3 |
| aprender-core/src/setfit/encoder_tests.rs | 1 |
| aprender-core/src/setfit/import_tests.rs | 1 |
| aprender-core/src/setfit/model_tests.rs | 1 |
| aprender-core/src/setfit/tokenizer_tests.rs | 2 |
| aprender-serve/src/apr/cuda_tests.rs | 1 |
| aprender-serve/src/apr_transformer/traced_save_tensor.rs | 2 |
| aprender-serve/src/cli/inference_tests.rs | 2 |
| aprender-serve/src/cli/tests_12.rs | 2 |
| aprender-serve/src/convert/tests_q4k_converter.rs | 9 |
| aprender-serve/src/convert/writing.rs | 9 |
| aprender-serve/src/gguf/loader_vocab_tests.rs | 1 |
| aprender-serve/src/gguf/qwen35_load.rs | 2 |
| aprender-serve/src/inference_trace/gpu_stage_dump.rs | 2 |
| aprender-serve/src/inference_trace/save_tensor_compose.rs | 1 |
| aprender-serve/src/inference_trace/save_tensor_emit.rs | 1 |
