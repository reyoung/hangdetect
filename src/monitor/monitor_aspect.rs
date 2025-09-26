use crate::monitor::LaunchCUDAKernel;
use crate::monitor::error::MonitorError;

pub trait MonitorAspect: Send + Sync {
    fn before_call(&self, launch: &LaunchCUDAKernel) -> Result<(), MonitorError>;

    fn after_call(&self, launch: &LaunchCUDAKernel) -> Result<(), MonitorError>;
}
