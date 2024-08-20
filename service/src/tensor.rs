use std::cmp::Ordering;

use anyhow::Result;
use axum::http::StatusCode;
use safetensors::{tensor::TensorView, Dtype};
use tch::{Device, Kind, Tensor};

pub const NUM_FRAMES: i64 = 1024;
pub const NUM_MEL_BINS: i64 = 128;

pub fn to_tensor(view: TensorView) -> Result<Tensor, (StatusCode, String)> {
    let size: Vec<i64> = view.shape().iter().map(|&x| x as i64).collect();
    let kind = match view.dtype() {
        Dtype::BOOL => Kind::Bool,
        Dtype::U8 => Kind::Uint8,
        Dtype::I8 => Kind::Int8,
        Dtype::I16 => Kind::Int16,
        Dtype::I32 => Kind::Int,
        Dtype::I64 => Kind::Int64,
        Dtype::BF16 => Kind::BFloat16,
        Dtype::F16 => Kind::Half,
        Dtype::F32 => Kind::Float,
        Dtype::F64 => Kind::Double,
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("unsupported dtype: {:?}", view.dtype()),
            ))
        }
    };
    let mut tensor = Tensor::f_from_data_size(view.data(), &size, kind)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    check(&tensor)?;
    tensor = tensor
        .f_to(Device::cuda_if_available())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(tensor)
}

pub fn check(tensor: &Tensor) -> Result<(), (StatusCode, String)> {
    let kind = tensor
        .f_kind()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    if kind != Kind::Float {
        return Err((
            StatusCode::BAD_REQUEST,
            "input must be a 32 bit float".to_string(),
        ));
    }

    let (_, num_mel_bins) = tensor
        .size2()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    if num_mel_bins != NUM_MEL_BINS {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Expected {} mel bins but got {}",
                NUM_MEL_BINS, num_mel_bins
            ),
        ));
    }
    Ok(())
}

pub fn fit(tensor: &mut Tensor) -> Result<(), (StatusCode, String)> {
    let (num_frames, _) = tensor
        .size2()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    match NUM_FRAMES.cmp(&num_frames) {
        Ordering::Less => {
            *tensor = tensor
                .f_narrow(0, 0, NUM_FRAMES)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        }
        Ordering::Greater => {
            *tensor = tensor
                .f_pad([0, 0, 0, NUM_FRAMES - num_frames], "constant", Some(0.0))
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        }
        _ => {}
    }
    Ok(())
}

pub fn normalize(tensor: &mut Tensor) -> Result<(), (StatusCode, String)> {
    *tensor = tensor
        .f_subtract_scalar(-4.2677393)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .f_divide_scalar(4.5689974 * 2.0)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(())
}
