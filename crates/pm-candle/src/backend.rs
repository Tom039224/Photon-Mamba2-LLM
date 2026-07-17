use candle_core::Device;
use pm_backend::{Backend, DeviceKind};

#[cfg(feature = "cuda")]
use crate::Error;

/// Candle-backed implementation of `pm_core::Ops` and `pm_backend::Backend`.
///
/// Cheap to clone (just holds a `Device`).
#[derive(Clone)]
pub struct CandleBackend {
    pub(crate) device: Device,
    pub(crate) device_kind: DeviceKind,
}

impl CandleBackend {
    /// CPU backend. Always available.
    pub fn new_cpu() -> Self {
        Self {
            device: Device::Cpu,
            device_kind: DeviceKind::Cpu,
        }
    }

    /// CUDA backend on the given ordinal.
    ///
    /// Requires the `cuda` feature and a working CUDA install. Used by
    /// `examples/smoke.rs` to exercise Blackwell (sm_120) + CUDA 13.3.
    #[cfg(feature = "cuda")]
    pub fn new_cuda(ordinal: usize) -> Result<Self, Error> {
        let device = Device::new_cuda(ordinal)?;
        Ok(Self {
            device,
            device_kind: DeviceKind::Cuda { ordinal },
        })
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    /// Block until every queued operation on the backing device has
    /// completed. CPU device is a no-op. Mostly useful for
    /// micro-benchmarks that need to time GPU work end-to-end.
    pub fn synchronize(&self) -> Result<(), candle_core::Error> {
        self.device.synchronize()
    }
}

impl Backend for CandleBackend {
    fn device_kind(&self) -> DeviceKind {
        self.device_kind
    }
}
