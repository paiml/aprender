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

/// Every passing run the rule was set from: `/usr/bin/time -v` maximum resident
/// set, `apr run <gguf> --no-gpu -p "What is 2+2?" -n 8`, apr 0.70.3 (95f64ba14),
/// x86_64. File bytes and kB exactly as recorded.
const MEASURED: [Measured; 7] = [
    Measured {
        model: "0.8B Q4_K_M",
        file_bytes: 532_517_120,
        hidden: 1024,
        maxrss_kb: 2_171_716,
    },
    Measured {
        model: "2B Q4_K_M",
        file_bytes: 1_280_835_840,
        hidden: 2048,
        maxrss_kb: 4_626_876,
    },
    Measured {
        model: "4B Q4_K_M",
        file_bytes: 2_740_937_888,
        hidden: 2560,
        maxrss_kb: 8_061_068,
    },
    Measured {
        model: "9B Q4_K_M",
        file_bytes: 5_680_522_464,
        hidden: 4096,
        maxrss_kb: 17_694_984,
    },
    Measured {
        model: "0.8B UD-IQ2_XXS",
        file_bytes: 338_227_456,
        hidden: 1024,
        maxrss_kb: 1_806_964,
    },
    Measured {
        model: "4B UD-Q4_K_XL",
        file_bytes: 2_912_109_728,
        hidden: 2560,
        maxrss_kb: 8_395_756,
    },
    Measured {
        model: "27B Q4_K_M",
        file_bytes: 16_740_812_704,
        hidden: 5120,
        maxrss_kb: 42_322_916,
    },
];

fn measured(model: &str) -> &'static Measured {
    MEASURED
        .iter()
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
fn need_covers_every_measured_peak_with_at_least_five_percent() {
    for m in &MEASURED {
        let need = host_build_need(m.file_bytes, embd(m.hidden));
        let peak = m.maxrss_kb * KIB;
        assert!(
            u128::from(need) * 100 > u128::from(peak) * 105,
            "{}: need {need} B is within 5% of the measured peak {peak} B",
            m.model
        );
    }
}

#[test]
fn a_host_whose_limit_is_the_measured_peak_is_refused() {
    // 90% of this memory is (at most) the peak the run really reached, so the
    // rule must not admit it: every admitted run has headroom above its peak.
    for m in &MEASURED {
        let memory = HostMemory::Available(m.maxrss_kb * KIB * 100 / 90);
        assert!(
            matches!(verdict(m, memory), Err(RealizarError::HostRamRefused(_))),
            "{}: admitted on a host whose 90% limit is its own measured peak",
            m.model
        );
    }
}

#[test]
fn jetson_admits_the_08b_models_and_refuses_2b_and_4b() {
    for mib in JETSON_AVAILABLE_MIB {
        let jetson = HostMemory::Available(mib * MIB);
        for model in ["0.8B Q4_K_M", "0.8B UD-IQ2_XXS"] {
            assert!(
                verdict(measured(model), jetson).is_ok(),
                "{model} refused on a Jetson with {mib} MiB available"
            );
        }
        for model in ["2B Q4_K_M", "4B Q4_K_M", "4B UD-Q4_K_XL", "9B Q4_K_M"] {
            refusal(verdict(measured(model), jetson));
        }
    }
}

#[test]
fn gb10_admits_every_measured_model_as_before() {
    for m in &MEASURED {
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
        measured("2B Q4_K_M"),
        HostMemory::Available(3910 * MIB),
    ));
    for part in [
        "~5299 MiB",
        "file 1221 MiB",
        "f32 token embedding 1940 MiB",
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
/// These rows kill a changed 275 or 90, `>` as `>=`, and a dropped term.
#[test]
fn the_boundary_is_exact() {
    let needs = [(360 * MIB, 0), (0, 990 * MIB), (200 * MIB, 440 * MIB)];
    for (file, embedding) in needs {
        assert_eq!(host_build_need(file, embedding), 990 * MIB);
        let kinds: [fn(u64) -> HostMemory; 2] = [HostMemory::Available, HostMemory::TotalOnly];
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
    assert_eq!(host_build_need(4, 0), 11);
    assert_eq!(host_build_need(1, 0), 2);
    assert_eq!(host_build_need(0, 7), 7);
    assert_eq!(host_build_need(u64::MAX, 0), u64::MAX);
    assert_eq!(host_build_need(u64::MAX / 2, u64::MAX / 2), u64::MAX);
    refusal(host_build_verdict(
        u64::MAX,
        u64::MAX,
        HostMemory::Available(u64::MAX),
    ));
}

#[test]
fn total_ram_is_the_fallback_bound_and_says_so() {
    let sixteen_gib = HostMemory::TotalOnly(16 * 1024 * MIB);
    assert!(verdict(measured("0.8B Q4_K_M"), sixteen_gib).is_ok());
    let msg = refusal(verdict(measured("9B Q4_K_M"), sixteen_gib));
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
        matches!(HostMemory::measure(), HostMemory::Available(b) if b > 0),
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
    // token_embd [4, 8] f32 = 128 bytes, and no file term: the need is 128.
    let model = model_with_embedding(&[4, 8]);
    assert!(admit_qwen35_host_build(&model, 0, HostMemory::Available(143)).is_ok());
    refusal(admit_qwen35_host_build(
        &model,
        0,
        HostMemory::Available(142),
    ));
    refusal(admit_qwen35_host_build(&model, 0, HostMemory::Unknown));
}
