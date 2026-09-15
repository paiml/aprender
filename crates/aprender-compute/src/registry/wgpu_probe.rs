//! wgpu factory: every adapter is an entry; a software rasteriser is listed as
//! `NoBackend`, never silently used as a GPU; the Metal adapter on Apple silicon
//! appears here with `transport = metal` (lane 2: a transport, not a peer kind).
use super::{Api, BackendEntry, BackendFactory, BackendKind, MemKind, Reason, Source, Status};

/// Discovers wgpu adapters.
pub struct WgpuFactory;

impl BackendFactory for WgpuFactory {
    fn kind(&self) -> BackendKind {
        BackendKind::Wgpu
    }

    fn discover(&self) -> Vec<BackendEntry> {
        let adapters = crate::backends::gpu::runtime::block_on(async {
            let instance = crate::backends::gpu::shared_instance();
            instance
                .enumerate_adapters(crate::backends::gpu::gpu_backends())
                .iter()
                .map(|a| (a.get_info(), a.limits()))
                .collect::<Vec<_>>()
        });
        if adapters.is_empty() {
            return vec![BackendEntry::unavailable(
                BackendKind::Wgpu,
                Api::Wgpu,
                Source::CompiledIn,
                Reason::NoDevice,
            )];
        }
        adapters.iter().enumerate().map(|(i, (info, limits))| entry(i, info, limits)).collect()
    }
}

/// PCI vendor id -> vendor name. `"unknown"` for anything not in the table.
fn vendor_name(vendor_id: u32) -> &'static str {
    match vendor_id {
        0x10de => "NVIDIA",
        0x1002 => "AMD",
        0x8086 => "Intel",
        0x106b => "Apple",
        _ => "unknown",
    }
}

/// wgpu backend -> the transport string `apr devices` prints (lane 2).
fn transport_name(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Gl => "gl",
        wgpu::Backend::BrowserWebGpu => "webgpu",
        wgpu::Backend::Noop => "noop",
    }
}

/// Device type -> (`device_type` string, status). A software rasteriser is
/// `Unavailable(NoBackend)`: listed, never used as a GPU.
fn classify(device_type: wgpu::DeviceType, name: &str) -> (&'static str, Status) {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => ("discrete-gpu", Status::Ready),
        wgpu::DeviceType::IntegratedGpu => ("integrated-gpu", Status::Ready),
        wgpu::DeviceType::VirtualGpu => ("virtual-gpu", Status::Ready),
        wgpu::DeviceType::Cpu => (
            "software",
            Status::Unavailable(Reason::NoBackend {
                vendor: format!("software rasteriser ({name}) is not a GPU"),
            }),
        ),
        wgpu::DeviceType::Other => ("other", Status::Ready),
    }
}

/// Unified memory: Metal, or any integrated part. wgpu exposes no VRAM figure;
/// `max_buffer_size` is an allocation cap, so it is the working-set limit here
/// and `mem_total` stays unknown.
fn mem_kind(
    backend: wgpu::Backend,
    device_type: wgpu::DeviceType,
    limits: &wgpu::Limits,
) -> MemKind {
    let unified = matches!(backend, wgpu::Backend::Metal)
        || matches!(device_type, wgpu::DeviceType::IntegratedGpu);
    if unified {
        MemKind::Unified { working_set_limit: Some(limits.max_buffer_size) }
    } else {
        MemKind::Discrete
    }
}

#[allow(clippy::cast_possible_truncation)]
fn entry(i: usize, info: &wgpu::AdapterInfo, limits: &wgpu::Limits) -> BackendEntry {
    let vendor = vendor_name(info.vendor);
    let (device_type, status) = classify(info.device_type, &info.name);
    BackendEntry {
        kind: BackendKind::Wgpu,
        api: Api::Wgpu,
        device_index: Some(i as u32),
        device_uid: Some(super::device_uid(vendor, &info.name)),
        device_name: info.name.clone(),
        vendor: vendor.to_string(),
        vendor_id: if vendor == "unknown" { None } else { Some(info.vendor) },
        device_type: device_type.to_string(),
        mem_total: None,
        mem_free: None,
        mem_kind: mem_kind(info.backend, info.device_type, limits),
        compute_class: None,
        caps: vec![format!("max_buffer_size={}", limits.max_buffer_size)],
        source: Source::CompiledIn,
        status,
        transport: Some(transport_name(info.backend).to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_table_names_the_four_and_unknown_for_the_rest() {
        assert_eq!(vendor_name(0x10de), "NVIDIA");
        assert_eq!(vendor_name(0x1002), "AMD");
        assert_eq!(vendor_name(0x8086), "Intel");
        assert_eq!(vendor_name(0x106b), "Apple");
        assert_eq!(vendor_name(0x0000), "unknown");
        assert_eq!(vendor_name(0xffff), "unknown");
    }

    #[test]
    fn transport_table_is_total_over_wgpu_backends() {
        assert_eq!(transport_name(wgpu::Backend::Vulkan), "vulkan");
        assert_eq!(transport_name(wgpu::Backend::Metal), "metal");
        assert_eq!(transport_name(wgpu::Backend::Dx12), "dx12");
        assert_eq!(transport_name(wgpu::Backend::Gl), "gl");
        assert_eq!(transport_name(wgpu::Backend::BrowserWebGpu), "webgpu");
        assert_eq!(transport_name(wgpu::Backend::Noop), "noop");
    }

    #[test]
    fn software_rasteriser_is_listed_but_never_ready() {
        let (ty, status) = classify(wgpu::DeviceType::Cpu, "llvmpipe");
        assert_eq!(ty, "software");
        match status {
            Status::Unavailable(Reason::NoBackend { vendor }) => {
                assert!(vendor.contains("llvmpipe"), "names the adapter: {vendor}");
            }
            other => panic!("software rasteriser must be Unavailable(NoBackend), got {other:?}"),
        }
        for (dt, want) in [
            (wgpu::DeviceType::DiscreteGpu, "discrete-gpu"),
            (wgpu::DeviceType::IntegratedGpu, "integrated-gpu"),
            (wgpu::DeviceType::VirtualGpu, "virtual-gpu"),
            (wgpu::DeviceType::Other, "other"),
        ] {
            let (ty, status) = classify(dt, "x");
            assert_eq!(ty, want);
            assert!(matches!(status, Status::Ready), "{want} is Ready");
        }
    }

    #[test]
    fn unified_memory_is_metal_or_integrated_and_carries_the_buffer_cap() {
        let mut limits = wgpu::Limits::default();
        limits.max_buffer_size = 4096;
        for (b, dt) in [
            (wgpu::Backend::Metal, wgpu::DeviceType::DiscreteGpu),
            (wgpu::Backend::Vulkan, wgpu::DeviceType::IntegratedGpu),
        ] {
            match mem_kind(b, dt, &limits) {
                MemKind::Unified { working_set_limit } => assert_eq!(working_set_limit, Some(4096)),
                other => panic!("{b:?}/{dt:?} must be Unified, got {other:?}"),
            }
        }
        assert!(matches!(
            mem_kind(wgpu::Backend::Vulkan, wgpu::DeviceType::DiscreteGpu, &limits),
            MemKind::Discrete
        ));
    }
}
