//! #4947 F2: the host-RAM admission rule, pinned by a case table built from the
//! measured passing-run peaks, the small-device rows and the exact boundary.

use super::*;
use crate::gguf::test_factory::GGUFBuilder;

const KIB: u64 = 1024;
const VOCAB: u64 = 248_320;

/// f32 `token_embd.weight` bytes of a Qwen3.5 with this hidden size.
const fn embd(hidden: u64) -> u64 {
    VOCAB * hidden * 4
}

struct Measured {
    model: &'static str,
    file_bytes: u64,
    hidden: u64,
    maxrss_kb: u64,
}

/// Every passing run the rule was set from, after #4991: `/usr/bin/time -v` maximum
/// resident set, `apr run <gguf> --no-gpu -p "What is 2+2?" -n 8`, apr 0.70.3
/// (1ec5fe3b31). Each row is the larger of x86_64 (noah-Lambda-Vector) and aarch64
/// (GB10); the 4B rows ran on GB10 only, and the 9B row is the x86_64 16-token run
/// #4991 recorded. File bytes and kB exactly as recorded.
const MEASURED: [Measured; 6] = [
    Measured {
        model: "0.8B Q4_K_M",
        file_bytes: 532_517_120,
        hidden: 1024,
        maxrss_kb: 1_031_204,
    },
    Measured {
        model: "2B Q4_K_M",
        file_bytes: 1_280_835_840,
        hidden: 2048,
        maxrss_kb: 2_170_920,
    },
    Measured {
        model: "4B Q4_K_M",
        file_bytes: 2_740_937_888,
        hidden: 2560,
        maxrss_kb: 3_796_724,
    },
    Measured {
        model: "9B Q4_K_M",
        file_bytes: 5_680_522_464,
        hidden: 4096,
        maxrss_kb: 7_019_736,
    },
    Measured {
        model: "0.8B UD-IQ2_XXS",
        file_bytes: 338_227_456,
        hidden: 1024,
        maxrss_kb: 776_700,
    },
    Measured {
        model: "4B UD-Q4_K_XL",
        file_bytes: 2_912_109_728,
        hidden: 2560,
        maxrss_kb: 3_963_208,
    },
];

/// The 27B, measured on GB10 under this rule after 1.10 and 0.60 were set: the
/// out-of-sample row.
const MEASURED_27B: Measured = Measured {
    model: "27B Q4_K_M",
    file_bytes: 16_740_812_704,
    hidden: 5120,
    maxrss_kb: 18_167_088,
};

fn measured(model: &str) -> &'static Measured {
    MEASURED
        .iter()
        .chain(std::iter::once(&MEASURED_27B))
        .find(|m| m.model == model)
        .expect("a measured model")
}

fn verdict(m: &Measured, memory: HostMemory) -> Result<()> {
    host_build_verdict(m.file_bytes, embd(m.hidden), memory)
}

fn refusal(result: Result<()>) -> String {
    match result {
        Err(RealizarError::HostRamRefused(msg)) => msg,
        other => panic!("expected HostRamRefused, got {other:?}"),
    }
}

/// The Jetson Orin's `MemAvailable` when the 4B was OOM-killed, and a later reading.
const JETSON_AVAILABLE_MIB: [u64; 2] = [3910, 4789];

/// GB10's reference `/proc/meminfo` (`capacity.rs`): `MemAvailable` 94,661,632 kB.
const GB10_AVAILABLE: u64 = 94_661_632 * KIB;

#[test]
fn need_covers_every_measured_peak_with_at_least_ten_percent() {
    for m in MEASURED.iter().chain(std::iter::once(&MEASURED_27B)) {
        let need = host_build_need(m.file_bytes, embd(m.hidden));
        let peak = m.maxrss_kb * KIB;
        assert!(
            u128::from(need) * 100 > u128::from(peak) * 110,
            "{}: need {need} B is within 10% of the measured peak {peak} B",
            m.model
        );
    }
}

#[test]
fn a_host_whose_limit_is_the_measured_peak_is_refused() {
    // 90% of this memory is (at most) the peak the run really reached, so the
    // rule must not admit it: every admitted run has headroom above its peak.
    for m in MEASURED.iter().chain(std::iter::once(&MEASURED_27B)) {
        let memory = HostMemory::Available(m.maxrss_kb * KIB * 100 / 90);
        assert!(
            matches!(verdict(m, memory), Err(RealizarError::HostRamRefused(_))),
            "{}: admitted on a host whose 90% limit is its own measured peak",
            m.model
        );
    }
}

/// C314 on #5124: the 2.75 × file + f32 embedding rule refused all of these, though
/// each host holds the measured peak with room to spare.
#[test]
fn hosts_that_hold_the_post_4991_peak_are_not_refused() {
    let admitted = [
        // The 0.8B under the 2174 MiB of cgroup headroom the serve tests had.
        ("0.8B Q4_K_M", HostMemory::CgroupHeadroom(2174 * MIB)),
        ("2B Q4_K_M", HostMemory::Available(3 * 1024 * MIB)),
        ("4B Q4_K_M", HostMemory::Available(5 * 1024 * MIB)),
        ("9B Q4_K_M", HostMemory::Available(10 * 1024 * MIB)),
        ("9B Q4_K_M", HostMemory::CgroupHeadroom(16 * 1024 * MIB)),
        ("27B Q4_K_M", HostMemory::Available(24 * 1024 * MIB)),
    ];
    for (model, memory) in admitted {
        let m = measured(model);
        let bytes = match memory {
            HostMemory::Available(b) | HostMemory::CgroupHeadroom(b) | HostMemory::TotalOnly(b) => {
                b
            },
            HostMemory::Unknown => 0,
        };
        assert!(
            bytes > m.maxrss_kb * KIB,
            "{model}: the host must hold the measured peak"
        );
        assert!(
            verdict(m, memory).is_ok(),
            "{model} refused with {memory:?}, which holds its measured peak"
        );
    }
}

#[test]
fn jetson_admits_the_08b_and_2b_models_and_refuses_4b_and_9b() {
    for mib in JETSON_AVAILABLE_MIB {
        let jetson = HostMemory::Available(mib * MIB);
        for model in ["0.8B Q4_K_M", "0.8B UD-IQ2_XXS", "2B Q4_K_M"] {
            assert!(
                verdict(measured(model), jetson).is_ok(),
                "{model} refused on a Jetson with {mib} MiB available"
            );
        }
        for model in ["4B Q4_K_M", "4B UD-Q4_K_XL", "9B Q4_K_M"] {
            refusal(verdict(measured(model), jetson));
        }
    }
}

#[test]
fn gb10_admits_every_measured_model_as_before() {
    for m in MEASURED.iter().chain(std::iter::once(&MEASURED_27B)) {
        assert!(
            verdict(m, HostMemory::Available(GB10_AVAILABLE)).is_ok(),
            "{} refused on GB10, whose result must not change",
            m.model
        );
    }
}

#[test]
fn the_refusal_states_its_arithmetic() {
    let msg = refusal(verdict(
        measured("4B Q4_K_M"),
        HostMemory::Available(3910 * MIB),
    ));
    for part in [
        "~4330 MiB",
        "1.10 x file 2613 MiB",
        "0.60 x f32 token-embedding size 2425 MiB",
        "3910 MiB available",
        "(3519 MiB)",
        "#4947",
    ] {
        assert!(msg.contains(part), "{part:?} missing from: {msg}");
    }
    let shown = RealizarError::HostRamRefused(msg).to_string();
    assert!(shown.starts_with("host RAM refused: "), "{shown}");
}

/// Three ways to need exactly 990 MiB: all file, all embedding, and both. 90% of
/// 1100 MiB is exactly 990 MiB, so one byte less memory turns admit into refuse.
/// These rows kill a changed 110, 60 or 90, `>` as `>=`, and a dropped term.
#[test]
fn the_boundary_is_exact() {
    let needs = [(900 * MIB, 0), (0, 1650 * MIB), (300 * MIB, 1100 * MIB)];
    for (file, embedding) in needs {
        assert_eq!(host_build_need(file, embedding), 990 * MIB);
        let kinds: [fn(u64) -> HostMemory; 3] = [
            HostMemory::Available,
            HostMemory::CgroupHeadroom,
            HostMemory::TotalOnly,
        ];
        for memory in kinds {
            assert!(
                host_build_verdict(file, embedding, memory(1100 * MIB)).is_ok(),
                "need == limit must be admitted ({file}, {embedding})"
            );
            refusal(host_build_verdict(file, embedding, memory(1100 * MIB - 1)));
        }
    }
}

#[test]
fn the_need_rounds_down_and_saturates_instead_of_wrapping() {
    assert_eq!(host_build_need(10, 0), 11);
    assert_eq!(host_build_need(9, 0), 9);
    assert_eq!(host_build_need(0, 10), 6);
    assert_eq!(host_build_need(0, 7), 4);
    assert_eq!(host_build_need(9, 7), 13);
    assert_eq!(host_build_need(u64::MAX, 0), u64::MAX);
    assert_eq!(host_build_need(u64::MAX / 2, u64::MAX), u64::MAX);
    refusal(host_build_verdict(
        u64::MAX,
        u64::MAX,
        HostMemory::Available(u64::MAX),
    ));
}

#[test]
fn total_ram_is_the_fallback_bound_and_says_so() {
    let sixteen_gib = HostMemory::TotalOnly(16 * 1024 * MIB);
    assert!(verdict(measured("9B Q4_K_M"), sixteen_gib).is_ok());
    let msg = refusal(verdict(measured("27B Q4_K_M"), sixteen_gib));
    assert!(msg.contains("of total RAM"), "{msg}");
}

#[test]
fn unknown_memory_refuses_even_an_empty_build() {
    let msg = refusal(host_build_verdict(0, 0, HostMemory::Unknown));
    assert!(msg.contains("#2568"), "{msg}");
    assert!(msg.contains(std::env::consts::OS), "{msg}");
}

#[cfg(target_os = "linux")]
#[test]
fn linux_measures_mem_available() {
    assert!(
        matches!(HostMemory::measure(), HostMemory::Available(b) | HostMemory::CgroupHeadroom(b) if b > 0),
        "/proc/meminfo has MemAvailable on Linux"
    );
}

fn model_with_embedding(dims: &[u64]) -> GGUFModel {
    let n = dims.iter().product::<u64>();
    let data = vec![0.0f32; usize::try_from(n).expect("small test tensor")];
    let file = GGUFBuilder::new()
        .architecture("qwen35")
        .add_f32_tensor("token_embd.weight", dims, &data)
        .add_f32_tensor("output_norm.weight", &[8], &[1.0; 8])
        .build();
    GGUFModel::from_bytes(&file).expect("parse the synthetic GGUF")
}

#[test]
fn embedding_bytes_are_the_f32_size_of_token_embd() {
    assert_eq!(
        qwen35_embedding_f32_bytes(&model_with_embedding(&[8, 16])),
        512
    );

    let mut no_embd = model_with_embedding(&[8, 16]);
    no_embd.tensors.retain(|t| t.name != "token_embd.weight");
    assert_eq!(qwen35_embedding_f32_bytes(&no_embd), 0);

    let mut overflow = model_with_embedding(&[8, 16]);
    for t in &mut overflow.tensors {
        if t.name == "token_embd.weight" {
            t.dims = vec![u64::MAX / 2, 3];
        }
    }
    assert_eq!(qwen35_embedding_f32_bytes(&overflow), u64::MAX);
}

#[test]
fn admission_counts_the_models_embedding() {
    // token_embd [10, 10] f32 = 400 bytes, and no file term: the need is 60% of it, 240.
    let model = model_with_embedding(&[10, 10]);
    assert!(admit_qwen35_host_build(&model, 0, HostMemory::Available(267)).is_ok());
    refusal(admit_qwen35_host_build(
        &model,
        0,
        HostMemory::Available(266),
    ));
    refusal(admit_qwen35_host_build(&model, 0, HostMemory::Unknown));
}

/// One cgroup v2 level under `root`: its `memory.max` and `memory.current`.
fn cgroup_level(root: &std::path::Path, relative: &str, max: &str, current: &str) {
    let dir = root.join(relative);
    std::fs::create_dir_all(&dir).expect("cgroup dir");
    std::fs::write(dir.join("memory.max"), format!("{max}\n")).expect("memory.max");
    std::fs::write(dir.join("memory.current"), format!("{current}\n")).expect("memory.current");
}

#[test]
fn a_cgroup_level_bounds_by_max_minus_current_and_max_means_unlimited() {
    assert_eq!(
        cgroup_level_headroom("8589934592\n", "1073741824\n"),
        Some(7_516_192_768)
    );
    assert_eq!(cgroup_level_headroom("max\n", "1073741824\n"), None);
    // Over its limit (reclaim pending): no headroom, never a wrapped huge one.
    assert_eq!(cgroup_level_headroom("100", "150"), Some(0));
    assert_eq!(cgroup_level_headroom("garbage", "1"), None);
    assert_eq!(cgroup_level_headroom("100", "garbage"), None);
}

/// #4992 done_when, case 1: a limited cgroup (the container root itself, as a
/// container sees `/sys/fs/cgroup/memory.max`) bounds below `MemAvailable`.
#[test]
fn a_capped_cgroup_lowers_the_bound_below_mem_available() {
    let root = tempfile::tempdir().expect("tempdir");
    cgroup_level(
        root.path(),
        "",
        &(4096 * MIB).to_string(),
        &(1024 * MIB).to_string(),
    );
    let headroom = cgroup_headroom_under(root.path(), "/");
    assert_eq!(headroom, Some(3072 * MIB));
    let memory = HostMemory::bound(60 * 1024 * MIB, headroom);
    assert_eq!(memory, HostMemory::CgroupHeadroom(3072 * MIB));
    // The 4B fits a 60 GiB host and is refused in this 4 GiB container.
    assert!(verdict(
        measured("4B Q4_K_M"),
        HostMemory::Available(60 * 1024 * MIB)
    )
    .is_ok());
    let msg = refusal(verdict(measured("4B Q4_K_M"), memory));
    assert!(msg.contains("cgroup memory.max"), "{msg}");
    assert!(msg.contains("3072 MiB"), "{msg}");
}

/// #4992 done_when, case 2: `memory.max` is "max" at every level, so
/// `MemAvailable` alone bounds the build exactly as before.
#[test]
fn an_unlimited_cgroup_leaves_mem_available_as_the_bound() {
    let root = tempfile::tempdir().expect("tempdir");
    cgroup_level(root.path(), "", "max", "5");
    cgroup_level(root.path(), "user.slice", "max", "5");
    assert_eq!(cgroup_headroom_under(root.path(), "/user.slice"), None);
    assert_eq!(
        HostMemory::bound(3910 * MIB, None),
        HostMemory::Available(3910 * MIB)
    );
    // A cgroup with more headroom than MemAvailable does not loosen the bound.
    assert_eq!(
        HostMemory::bound(3910 * MIB, Some(8000 * MIB)),
        HostMemory::Available(3910 * MIB)
    );
}

/// The limit can sit on an ancestor (this dev host caps `agent.slice`, not the
/// scope the process runs in): every level on the path counts, the tightest wins,
/// and a level whose files are missing is skipped.
#[test]
fn the_tightest_level_on_the_path_wins_even_on_an_ancestor() {
    let root = tempfile::tempdir().expect("tempdir");
    cgroup_level(
        root.path(),
        "a",
        &(2000 * MIB).to_string(),
        &(500 * MIB).to_string(),
    );
    std::fs::create_dir_all(root.path().join("a/b")).expect("level without files");
    cgroup_level(root.path(), "a/b/c", "max", "1");
    assert_eq!(
        cgroup_headroom_under(root.path(), "/a/b/c"),
        Some(1500 * MIB)
    );
    cgroup_level(
        root.path(),
        "a/b/c",
        &(900 * MIB).to_string(),
        &(100 * MIB).to_string(),
    );
    assert_eq!(
        cgroup_headroom_under(root.path(), "/a/b/c"),
        Some(800 * MIB)
    );
    // A sibling's limit is not on the path.
    cgroup_level(root.path(), "a/x", "1", "0");
    assert_eq!(
        cgroup_headroom_under(root.path(), "/a/b/c"),
        Some(800 * MIB)
    );
}
