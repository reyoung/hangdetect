use super::monitor_aspect::MonitorAspect;

pub struct LoggingAspect {}

impl MonitorAspect for LoggingAspect {
    fn before_call(
        &self,
        launch: &crate::monitor::LaunchCUDAKernel,
    ) -> Result<(), crate::monitor::error::MonitorError> {
        log::info!("Launching CUDA kernel: {}", launch);
        Ok(())
    }

    fn after_call(
        &self,
        _launch: &crate::monitor::LaunchCUDAKernel,
    ) -> Result<(), crate::monitor::error::MonitorError> {
        Ok(())
    }
}
