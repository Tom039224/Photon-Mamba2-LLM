use crate::DeviceKind;
use pm_core::Ops;

/// Marker trait that bundles `Ops` with device metadata.
///
/// `Ops` alone is enough for model code; `Backend` adds the lightweight
/// information that trainers / CLIs need for logging and device routing.
pub trait Backend: Ops {
    fn device_kind(&self) -> DeviceKind;
}
