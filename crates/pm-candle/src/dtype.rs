use crate::Error;
use candle_core::DType;
use pm_core::Dtype;

pub(crate) fn to_candle(d: Dtype) -> Result<DType, Error> {
    Ok(match d {
        Dtype::F32 => DType::F32,
        Dtype::F16 => DType::F16,
        Dtype::BF16 => DType::BF16,
        Dtype::I64 => DType::I64,
        Dtype::U32 => DType::U32,
    })
}

pub(crate) fn from_candle(d: DType) -> Result<Dtype, Error> {
    Ok(match d {
        DType::F32 => Dtype::F32,
        DType::F16 => Dtype::F16,
        DType::BF16 => Dtype::BF16,
        DType::I64 => Dtype::I64,
        DType::U32 => Dtype::U32,
        other => return Err(Error::NotImplemented(unsupported_label(other))),
    })
}

const fn unsupported_label(d: DType) -> &'static str {
    match d {
        DType::U8 => "u8 dtype mapping",
        DType::F64 => "f64 dtype mapping",
        DType::F8E4M3 => "f8e4m3 dtype mapping",
        _ => "candle dtype mapping",
    }
}
