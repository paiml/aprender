// #3754: `apr run --temperature T` given alone samples; `--top-k` defaults to the sampling
// top-k (realizar's DEFAULT_TOP_K), and --help shows that constant, not a second copy.

/// Parse argv on a 16 MB stack: clap's recursive destructuring of the full `Commands`
/// enum overflows the default 2 MiB test-thread stack in debug builds (as in data.rs).
fn parse_run_top_k(args: &[&'static str]) -> (f32, usize) {
    let args = args.to_vec();
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            use clap::Parser;
            let cli = crate::Cli::try_parse_from(args).expect("parse");
            match *cli.command {
                crate::Commands::Run {
                    temperature, top_k, ..
                } => (temperature, top_k),
                _ => panic!("expected `run`"),
            }
        })
        .expect("spawn parse thread")
        .join()
        .expect("join parse thread")
}

/// done_when 1 at the flag: `--temperature 0.8` alone leaves `--top-k` at a value that
/// samples. At 0.69.0 this parsed to 1, which every decode loop treats as greedy.
#[cfg(feature = "inference")]
#[test]
fn run_temperature_alone_parses_to_a_sampling_top_k() {
    let (temperature, top_k) = parse_run_top_k(&["apr", "run", "m.gguf", "--temperature", "0.8"]);
    assert!((temperature - 0.8).abs() < f32::EPSILON);
    assert_eq!(top_k, realizar::infer::DEFAULT_TOP_K);
    assert!(top_k > 1, "top_k 1 is greedy: the flag would do nothing");
}

/// done_when 2 at the flag: an explicit `--top-k 1` is kept, and the default temperature
/// is 0, so a bare `apr run` stays greedy.
#[test]
fn run_explicit_top_k_one_and_default_temperature_are_greedy() {
    let (_, top_k) = parse_run_top_k(&[
        "apr",
        "run",
        "m.gguf",
        "--temperature",
        "0.8",
        "--top-k",
        "1",
    ]);
    assert_eq!(top_k, 1);
    let (temperature, _) = parse_run_top_k(&["apr", "run", "m.gguf"]);
    assert_eq!(temperature, 0.0, "bare `apr run` must stay greedy");
}

/// done_when 3: `apr run --help` documents the default from the same constant the code
/// uses, and `RunOptions::default()` agrees with the flag.
#[test]
fn run_help_default_top_k_is_the_constant() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(run_help_default_top_k_body)
        .expect("spawn")
        .join()
        .expect("--help default row panicked");
}

fn run_help_default_top_k_body() {
    use clap::CommandFactory;
    let cmd = crate::Cli::command();
    let run = cmd.find_subcommand("run").expect("run subcommand");
    let arg = run
        .get_arguments()
        .find(|a| a.get_id() == "top_k")
        .expect("--top-k");
    let defaults: Vec<String> = arg
        .get_default_values()
        .iter()
        .map(|v| v.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        defaults,
        vec![crate::commands::run::DEFAULT_TOP_K.to_string()]
    );
    assert_eq!(
        RunOptions::default().top_k,
        crate::commands::run::DEFAULT_TOP_K
    );
}
