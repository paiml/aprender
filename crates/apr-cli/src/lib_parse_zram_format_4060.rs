// #4060: `apr zram` must reach every output format `trueno-zram` reaches, or
// retiring the standalone binary deletes a capability. It took only the
// subcommand, so `trueno-zram --format raw` had no `apr` equivalent — the
// sibling-reach test compared SUBCOMMANDS and could not see a root flag.

fn zram_format(args: Vec<&'static str>) -> Option<aprender_zram_cli::output::OutputFormat> {
    match *parse_cli(args).expect("apr zram argv parses").command {
        Commands::Zram(zram) => zram.format,
        other => panic!("parsed as {other:?}, not `apr zram`"),
    }
}

#[test]
fn apr_zram_accepts_every_format_trueno_zram_accepts() {
    use aprender_zram_cli::output::OutputFormat;
    for (word, want) in [
        ("table", OutputFormat::Table),
        ("json", OutputFormat::Json),
        ("raw", OutputFormat::Raw),
    ] {
        let got = zram_format(vec!["apr", "zram", "--format", word, "status"]).unwrap_or_else(|| panic!("--format {word} was dropped"));
        assert_eq!(
            std::mem::discriminant(&got),
            std::mem::discriminant(&want),
            "--format {word}"
        );
    }
}

#[test]
fn apr_zram_without_format_defers_to_the_global_json_flag() {
    assert!(zram_format(vec!["apr", "zram", "status"]).is_none());
    assert!(parse_cli(vec!["apr", "zram", "--format", "bogus", "status"]).is_err());
}
