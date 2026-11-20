// #7331, lane-2 finding 1 (PR #7356): every entry into `apr run` refuses the 2.4T checkpoint
// with exit 12 before anything else. The dispatch arm ran `run_accel_plan` first (exit 5 /
// 14 on a forced backend) and sent `--batch-jsonl` to a path with no refusal at all.

fn dispatch_run_argv(argv: &[&str]) -> Option<Result<(), CliError>> {
    let args: Vec<String> = argv.iter().map(|s| (*s).to_string()).collect();
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let cli = Cli::try_parse_from(args).expect("argv parses");
            dispatch_run_command(&cli)
        })
        .expect("spawn thread")
        .join()
        .expect("join thread")
}

#[test]
fn run_dispatch_refuses_known_checkpoint_on_every_entry() {
    let src = "hf://Qwen/Qwen3.8-2.4T-A95B/model-00001-of-00500.safetensors";
    for extra in [
        &["--batch-jsonl", "/nonexistent/prompts.jsonl"][..],
        &["--backend", "cuda"][..],
        &["--gpu"][..],
        &["--prompt", "hi"][..],
    ] {
        let mut argv = vec!["apr", "run", src];
        argv.extend_from_slice(extra);
        let err = dispatch_run_argv(&argv)
            .expect("the run arm handles `apr run`")
            .expect_err("the 2.4T checkpoint must be refused");
        assert_eq!(err.exit_code_value(), 12, "{extra:?}: {err}");
        assert!(err.to_string().contains("#7331"), "{extra:?}: {err}");
    }
}
