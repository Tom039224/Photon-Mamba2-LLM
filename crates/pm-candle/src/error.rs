use pm_core::Dtype;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("unsupported dtype on this backend: {0:?}")]
    UnsupportedDtype(Dtype),

    #[error("{0} is not yet implemented in the Candle backend")]
    NotImplemented(&'static str),
}
