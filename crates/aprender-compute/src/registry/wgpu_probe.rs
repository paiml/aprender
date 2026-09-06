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

#[allow(clippy::cast_possible_truncation)]
fn entry(i: usize, info: &wgpu::AdapterInfo, limits: &wgpu::Limits) -> BackendEntry {
    let vendor = match info.vendor {
        0x10de => "NVIDIA",
        0x1002 => "AMD",
        0x8086 => "Intel",
        0x106b => "Apple",
        _ => "unknown",
    };
    let transport = match info.backend {
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Gl => "gl",
        wgpu::Backend::BrowserWebGpu => "webgpu",
        wgpu::Backend::Noop => "noop",
    };
    let (device_type, status) = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => ("discrete-gpu", Status::Ready),
        wgpu::DeviceType::IntegratedGpu => ("integrated-gpu", Status::Ready),
        wgpu::DeviceType::VirtualGpu => ("virtual-gpu", Status::Ready),
        wgpu::DeviceType::Cpu => (
            "software",
            Status::Unavailable(Reason::NoBackend {
                vendor: format!("software rasteriser ({}) is not a GPU", info.name),
            }),
        ),
        wgpu::DeviceType::Other => ("other", Status::Ready),
    };
    let unified = matches!(info.backend, wgpu::Backend::Metal)
        || matches!(info.device_type, wgpu::DeviceType::IntegratedGpu);
    BackendEntry {
        kind: BackendKind::Wgpu,
        api: Api::Wgpu,
        device_index: Some(i as u32),
        device_uid: Some(super::device_uid(vendor, &info.name)),
        device_name: info.name.clone(),
        vendor: vendor.to_string(),
        vendor_id: if vendor == "unknown" { None } else { Some(info.vendor) },
        device_type: device_type.to_string(),
        mem_total: Some(limits.max_buffer_size),
        mem_free: None,
        mem_kind: if unified {
            MemKind::Unified { working_set_limit: Some(limits.max_buffer_size) }
        } else {
            MemKind::Discrete
        },
        compute_class: None,
        caps: Vec::new(),
        source: Source::CompiledIn,
        status,
        transport: Some(transport.to_string()),
    }
}
