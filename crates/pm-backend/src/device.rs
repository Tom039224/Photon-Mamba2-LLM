/// Coarse device classification used for diagnostics and routing.
///
/// Backends report this from `Backend::device_kind`. New device families
/// (Tenstorrent, etc.) are added as variants; existing code matches on
/// it for logging only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Cuda { ordinal: usize },
    Tenstorrent { ordinal: usize },
}
