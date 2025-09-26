use crate::cuda_funcs::CUDAError;
use std::fmt::Display;

#[derive(Debug)]
pub enum MonitorError {
    CUDAError(CUDAError),
    Internal(anyhow::Error),
}

impl Display for MonitorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MonitorError::CUDAError(err) => write!(f, "CUDA error: {}", err),
            MonitorError::Internal(err) => write!(f, "Internal error: {}", err),
        }
    }
}
