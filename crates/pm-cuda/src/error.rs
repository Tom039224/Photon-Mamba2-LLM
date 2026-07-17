use cudarc::cublas::result::CublasError;
use cudarc::driver::DriverError;

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("cudarc driver error: {0}")]
    Driver(#[from] DriverError),

    #[error("cuBLAS error: {0}")]
    Cublas(#[from] CublasError),

    #[error("kernel `{0}` not found in loaded PTX module")]
    KernelNotFound(&'static str),

    #[error("unsupported operation or dtype: {0}")]
    Unsupported(&'static str),

    #[error("shape error: {0}")]
    Shape(String),

    #[error("internal error: {0}")]
    Internal(String),
}
