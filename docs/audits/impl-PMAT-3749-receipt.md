# impl receipt — PMAT-3749 (#3745 S1)

## Identity
- ticket: PMAT-3749 (GitHub #3749, part S1 of #3745), kind: code
- branch: `PMAT-3745-s1-surface-emitter`, cut from origin/main `52f43da71`
- owner: aprender-fc; routed by cop aprender-04
- consumers who agreed the schema before any code: S2 aprender-97, S3 aprender-62, #3739 aprender-76. Their acks are on #3745: design note issuecomment-5764787123, amendment 1 5764810360, amendment 2 5764962501.

## What S1 owes, verbatim (cop's per-part comment on #3745, issuecomment-5764661626)
- S1.1 A hidden `apr` subcommand prints the full clap tree as JSON: every subcommand path; every arg with long/short/positional, num_args, value_enum values, default and **role** ∈ {model, prompt, input-file, backend, mode, other}. A test adds a dummy flag to the clap tree and finds it in the JSON.
- S1.2 The role is read from the arg's typed constructor, not its name. Every existing model-taking command is migrated onto the markers.
- S1.3 Marker guard (issue mutant 3): a command that loads a model through a plain `PathBuf` arg is RED, with a case row and a mutant.
- S1.4 The JSON schema is agreed with S2/S3/#3739 and versioned inside the JSON. It's generated from the binary under test at gate time and never committed.

Cop rulings on the design are quoted verbatim in `docs/roadmaps/entries/PMAT-3749.yaml` `notes:`: totality instead of loader detection, KEEP the String half, the foreign model args typed in S1, role `unknown`, and dispatch_run.rs/serve handlers adapted at dispatch.rs.

## What landed, criterion by criterion
| criterion | where | evidence |
|---|---|---|
| S1.1 emitter | `crates/apr-cli/src/surface.rs`: `pub fn emit() -> Surface` (walk on a 64 MiB thread) and `emit_from(&clap::Command)`. The hidden `apr surface` verb (`Commands::Surface`, dispatched first in `dispatch_core_command`) prints `emit().to_json()` via `std::io::Write`, so `--quiet` cannot suppress it | `surface::tests::the_emitter_sees_a_flag_added_to_the_clap_tree`: `--zz-dummy` (SetTrue) is injected into `run` with `mut_subcommand`, is absent before and present after, value_type `flag`, role `mode` |
| S1.1 fields | `schema, binary{name,version,git_sha}, roles, global_args, commands[]{path,key,aliases,hidden,foreign,leaf,subcommands,args[]{id,long,short,long_aliases,short_aliases,positional,index,required,takes_value,num_args,value_type,values,default,role,marker,hidden,conflicts_with}}` | `the_surface_is_versioned_and_deterministic`, `globals_are_emitted_once_at_the_root`, `clap_generated_help_and_version_are_not_emitted` |
| S1.2 role from construction | `batuta_common::cli_roles` (aprender-common, feature `cli-roles`): `ModelPath`, `ModelRef`, `PromptText`, `InputFile`, `OutputPath`, `DirPath`, `ConfigPath`, `FreeText`. Each is a newtype whose clap parser is the raw type's own parser (`PathBufValueParser` / `StringValueParser`), `.map`ped. The walker compares `ValueParser::type_id()` with `TypeId::of::<Marker>()`. Backend is membership in `<BackendArg as clap::Args>::group_id()`. Mode is structural (flag action, or a finite value set) | `classify_case_table`: 18 rows, one per way an arg can be built. `cli_roles::tests`: every marker parser reports its own TypeId, accepts exactly what the raw parser accepts, and prints `{:?}` as the raw type |
| S1.2 migration | all 428 free-form args of apr-cli's own tree (242 path, 186 text). 77 of them are model args (`ModelPath` 70, `ModelRef` 7) and 14 are prompts. Each role was read from the arg's help and use, not its name: `tune --model` is a model SIZE → FreeText; `test llm bench --model` is a request-body name → FreeText; `publish DIR` and `encrypt FILE` never load a model → DirPath / InputFile. Per the cop's ruling, the two foreign model args are typed too: `pv verify-structure --model` → ModelPath, `rag transcribe --model` → ModelPath (its help: "Path to Whisper .apr model file"; it was a `String`, and is handed on as `to_string_lossy()`) | `surface::guard::no_free_form_arg_is_undeclared_outside_foreign_subtrees` (0 unknown); `the_leak_verbs_declare_model_prompt_and_input_roles` (run SOURCE/--prompt/PROMPT/-i, chat, serve run, code, pv, rag) |
| S1.3 marker guard | `surface::unknown_outside_foreign()` must be empty. There is no loader list: an undeclared role is inexpressible outside a foreign subtree. Foreign mounts are found by TYPE (each foreign `Subcommand` type's child-name SET), and the list fails closed both ways (`the_foreign_list_matches_the_tree`) | Mutants planted in the real tree on every run: `inspect` file → PathBuf, a new verb with raw `--model: PathBuf`, `run --prompt` → String, nested `serve run file` → PathBuf, a raw global. Each is RED and names exactly `<path> <arg>`. Live source mutant (below) |
| S1.4 versioned, not committed, binary == library | `"schema": "apr-cli-surface/v1"`. Versioning rule (amendment 2): additive fields allowed; renames/removals/meaning changes bump. No surface JSON is committed | `tests/surface_binary_pin.rs`: the spawned `apr surface`, `--json` and `--quiet` all print byte-exactly `emit().to_json() + "\n"`, and `surface` is absent from `apr --help`. Wired: `ci/explicit-test-commands.d/447-apr-cli-surface-binary-pin.cmd` |

## Verification (all re-run by the orchestrator on lambda-vector, CARGO_TARGET_DIR per worktree)
| command | result |
|---|---|
| gate record: `gate-reduce.sh -- bash -c 'cargo test -p apr-cli --lib surface:: ; cargo test -p apr-cli --test surface_binary_pin ; cargo test -p aprender-common --features cli-roles --lib cli_roles ; cargo test -p apr-cli --lib'` | run 2: exit 0, sha256 `823e378f4793334f…`, 574587 bytes. 19 / 2 / 5 / 7317 passed. Run 1: exit 101, 7316 passed and 1 failed (`commands::qa::tests::detect_ollama_model_file_size_heuristic_tiny`). That is a pre-existing flake, root-caused and filed as **#3765**: a random `NamedTempFile` name ending in `[a-z]7b.gguf` passes `detect_size_from_filename`'s boundary rule. This diff does not touch `commands/qa*` or `forward_error.rs` |
| `cargo test -p aprender-contracts --lib` | 1689 passed |
| `cargo test -p aprender-contracts-cli -p aprender-rag-cli` | all targets pass |
| `cargo test -p apr-cli --test cli_commands` | 15 passed (hidden `surface` is absent from `--help`, and the registry test is unaffected) |
| `cargo check -p apr-cli --features cuda --lib --tests` / `--features dev` / `-p aprender` | rc 0 / rc 0 / rc 0 |
| `cargo clippy -p aprender-common --features cli-roles --lib --tests -- -D warnings`, `-p aprender-contracts-cli -p aprender-rag-cli --lib -- -D warnings` | clean |
| `cargo fmt --all -- --check` · `cargo deny check advisories` | clean · advisories ok |
| `scripts/guard_tree.sh --no-cargo` | 76 checks, 0 failed (after the ledger re-resolution below) |
| cargo-side guards touched by the diff: `check_explicit_test_commands`, `check_lockfile_current`, `check_lockfile_no_registry_siblings`, `check_cascade_covers_all_crates`, `check_hermetic_stdin_tests`, `check_duplicate_bin_names`, `check_contract_test_binding`, `check_model_tests_wired` | all rc 0 |

### v1.1 `generates` (cop ruling after S2 found `serve run` takes its prompts over HTTP)
| what | where | evidence |
|---|---|---|
| `commands[].generates: bool`, schema bumped to `apr-cli-surface/v1.1` | `surface::generates()`: true iff the command has a `PromptText` arg or carries the command-level `batuta_common::cli_roles::ServesGeneration` marker. The marker is an argument-less `ArgGroup` attached by `#[command(group(ServesGeneration::group()))]`, so it adds no field and changes neither parsing nor `--help` (`cli_roles::tests::serves_generation_marks_a_command_without_changing_it`), and it is matched by `ServesGeneration::ID`, never by name | `generates_case_table` has 5 rows: a PromptText arg (true); the marker (true); model-only (false); a look-alike group literally named `ServesGeneration` (false); a raw String arg named `prompt` (false) |
| marker placement | `serve run` (HTTP), `chat` (stdin; its only PromptText is `--system`), `mcp` (JSON-RPC `apr.run`/`apr.serve` tools) | `the_generators_are_declared`: serve run, chat, mcp, run and code generate; inspect, tensors, serve plan and surface do not. `mutant_serve_run_without_its_marker_does_not_generate`: serve run's real argument set without the marker reads `generates: false` |
| encoders are not generators (aprender-97) | `EncodeText` (role `other`) is a new role type for text a model encodes or scores: `embed --text`, `rerank --query/--passage/--passages/--input-ids/--token-type-ids`, `eval --text` (perplexity). Thinking × context-rung cells mean nothing for them | classify case row `encoded` → (text, other, EncodeText). `the_generators_are_declared`: embed, rerank and eval do not generate. The binary now reports 8 generators: bench chat code mcp parity "rosetta compare-inference" run "serve run" |
| role `sampling` + `sampling_kind` (cop ruling, for S2.5; kind per aprender-97) | `SamplingArg` / `SamplingKind`: each generation sampling control joins the argument-less `multiple(true)` group of its kind (`#[arg(group = SamplingKind::Temperature.id())]` plus `#[command(groups(SamplingArg::groups()))]`), so these numeric args keep their types. Membership, and so `role: sampling` and `sampling_kind` ∈ {seed, temperature, top_k, top_p, min_p, repeat_penalty, repeat_last_n}, is read from the BUILT command by group id, never by name. On: run temperature/top_k/top_p/seed/repeat_penalty/repeat_last_n, chat temperature/top_p, rosetta compare-inference temperature. Not on: distill's KD temperature, data seeds, rerank top_k | case rows `temperature`/`top_k` → sampling, `ungrouped_temperature` → other. `the_sampling_controls_are_declared`. `mutant_temperature_without_its_marker_is_not_sampling`. Live mutant: the marker removed from run --temperature → RED `run --temperature left: "other" right: "sampling"`; restored → GREEN. apr-cli lib: 7321 passed, and 1 unrelated wall-clock flake filed as #3788 (python child 5 s timeout under load ~50; passes alone) |
| live source mutant | the marker line removed from `serve_commands.rs:40` | `the_generators_are_declared` rc 101: `["serve", "run"] must be a generator`. Restored: rc 0. Suites after the change: apr-cli lib 7320 passed, surface_binary_pin 2, cli_roles 6, clippy clean |

### Live source mutant (S1.3)
`crates/apr-cli/src/commands_enum.rs:227` `file: ModelPath` → `file: PathBuf`. Then `cargo test -p apr-cli --lib surface::guard::no_free_form` gives rc 101: `1 free-form argument(s) with no declared role:\n  inspect file`. After restoring, rc 0.

## Owned-file edits, notified before
`crates/apr-cli/src/dispatch_run.rs`: 7 conversion edits in the `serve run` and rosetta arms only. The `dispatch_run` fn and its `run` args are untouched; conversion happens at the `dispatch.rs` call site. I checked the edits against #3707's hunk (`run_prompt_and_chat`, lines 41-100); they are disjoint. The cop approved them before commit. The serve handlers are untouched.

## docs/audits/surface_audit.csv (check_dogfood_coverage G2.1)
This branch changes 15 files the ledger cites. Each row citing one of them was re-resolved by line identity (difflib base..HEAD), and 39 pointers followed their line. 12 rows cited a line this branch rewrote in place. Each was checked by hand, and all 12 already pointed at the WRONG declaration before this branch (e.g. `apr devices` → a hex `offset: String`), so they now point at the feature's own variant or field. Quality scores are unchanged: the migration changes argument types, not behaviour.

## Gaps, stated
- A model arg DECLARED as a non-model role type (`OtherPath`-class) is a typed lie that is visible in the diff but not machine-caught. The backstops are review and S2's runs-or-refuses cell.
- Foreign subtrees (pv, alimentar `data x`, sim, rag, zram, cgp) still carry raw args other than the two typed ones. They are emitted as role `unknown` and counted by S2 under a shrink-only ratchet (cop ruling).
- `validate.rs::extract_model_paths` (PMAT-237) is itself a hand-typed list of which commands load models. It could now be derived from `ModelPath`/`ModelRef` markers, and is left for a follow-up.
- v1 declares no stdin form. S2 records `stdin: undeclared` under its ratchet (97's ruling on #3745).

## Routing
All phases ran direct (the orchestrator implemented; no worker subagent), because the migration was compile-driven over one crate's clap tree. `route.sh --phase-class review` gave `route=agy-quorum w=1.00 basis=absent effort=1[U]`.

## Dispatch ledger
| # | executor | outcome |
|---|---|---|
| 1 | `paiml-agy-delegate`, ph4 quorum width 3 | **DENIED by the spawn hook**: `kind-gate refused PMAT-3749 (exit 2) … not filed in /home/noah/src/aprender/docs/roadmaps/roadmap.yaml`. The hook keys the gate on the session's launch cwd (the shared checkout), while the ticket lives in this worktree. Not retried as-is. The session moved into this worktree (`EnterWorktree path=`, per the cop's own-worktree rule), and the gate then read the tree the ticket lives in |
| 2 | `paiml-agy-delegate` (opus), agent a31b9c07abd7d6029, same brief | `quorum-review.sh --base 52f43da71 --ticket PMAT-3749 --lane-model gemini-3.1-pro-high ×2 --lane-model gemini-3.1-pro-low --fallback-model gpt-oss-120b-medium` on head 0d417f07b. It hit the 30-turn cap and was resumed once (the only resume allowed) to write its receipt. It disclosed that it had launched the round 3 times and sent 6 manual gpt-oss probes |

## Quorum
- **Round 1 (head 0d417f07b, diff_sha256 20bfaec0…): NO-VERDICT ×3 in each of 3 attempts. This was a quota/capacity outage, not a review. No lane read the diff.**
  - Attempt 1: gemini returned 429 at the pre-check. All 3 lanes fell back to gpt-oss-120b-medium, and each got 503 "No capacity" twice.
  - Attempt 2: gemini 429, gpt-oss 503 at the pre-check, and no lane launched.
  - Attempt 3: gemini 429 ("Resets in 2h44m53s") and gpt-oss 429 ("Resets in 2h20m24s"), and no lane launched. `receipt-lint` refused each artifact (no `model_measured`).
  - All three artifacts are archived outside the tree, under the delegate's out_dir. A quota round is archived and never committed.
  - agy 1.2.7 offers no other non-author family (claude-* is the author's). The seat-fill was requested from the cop (never park), and the round is relaunched when a family resets.
- **Round 2 (cop slot 3, head 17dfb1291, base = merge-base 52f43da71): NO-VERDICT ×3 in each of 2 attempts, again on quota. No lane read the diff.** Run directly by the orchestrator (`quorum-review.sh`), with no delegate.
  - 22:16Z, `--lane-model gemini-3.1-pro-high ×2 --lane-model gemini-3.1-pro-low --fallback-model gpt-oss-120b-medium`: the gemini pre-check returned 429 "Individual quota reached … Resets in 30h14m6s", and gpt-oss returned 503 "No capacity". Every lane was skipped as quota-exhausted.
  - 22:22Z, `--lane-model gpt-oss-120b-medium ×3`: the pre-check returned 429 "Individual quota reached … Resets in 3h51m25s".
  - Both artifacts are archived outside the tree, never committed. A second seat-fill was requested from the cop at 22:23Z.

verdict: PARTIAL — gates green, quorum pending (round 1 NO-VERDICT on quota)
