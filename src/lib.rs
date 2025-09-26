use crate::cuda_funcs::{CuLaunchConfig, CudaLaunchConfig};
use libc::{c_char, c_uint};
use monitor::{LaunchCUDAKernel, monitor_launch_cuda_kernel};
use std::ffi::{c_int, c_void};

mod cuda_funcs;
mod init;
mod logger;

mod monitor;

#[unsafe(no_mangle)]
pub extern "C" fn cudaLaunchKernel(
    func: *const c_void,
    grid_dim: cuda_funcs::Dim3,
    block_dim: cuda_funcs::Dim3,
    args: *const *const c_void,
    shared_mem: usize,
    stream: *mut c_void,
) -> c_int {
    monitor_launch_cuda_kernel(LaunchCUDAKernel::Runtime { func, stream }, || {
        cuda_funcs::launch_cuda_kernel(func, grid_dim, block_dim, args, shared_mem, stream)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn cudaLaunchKernelExC(
    config: &CudaLaunchConfig,
    func: *const c_void,
    args: *mut *const c_void,
) -> c_int {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Runtime {
            func,
            stream: config.stream,
        },
        || cuda_funcs::launch_cuda_kernel_ex_c(config, func, args),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn cuLaunchKernel(
    func: *const c_void,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem: c_uint,
    stream: *const c_void,
    kernel_params: *mut *const c_void,
    extra: *mut *const c_void,
) -> c_int {
    monitor_launch_cuda_kernel(LaunchCUDAKernel::Driver { func, stream }, || {
        cuda_funcs::launch_cu_kernel(
            func,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            block_dim_x,
            block_dim_y,
            block_dim_z,
            shared_mem,
            stream,
            kernel_params,
            extra,
        )
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn cuLaunchKernelEx(
    config: &CuLaunchConfig,
    func: *const c_void,
    args: *mut *const c_void,
) -> c_int {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Driver {
            func,
            stream: config.stream,
        },
        || cuda_funcs::launch_cu_kernel_ex(config, func, args),
    )
}

// Settings APIs
#[unsafe(no_mangle)]
pub extern "C" fn hangdetect_set_enable(enabled: bool) {
    monitor::set_hang_detection_enabled(enabled);
}

#[unsafe(no_mangle)]
pub extern "C" fn hangdetect_set_kernel_exec_label(label: *const c_char) {
    if label.is_null() {
        monitor::set_kernel_exec_time_user_label("");
    } else {
        let c_str = unsafe { std::ffi::CStr::from_ptr(label) };
        if let Ok(str_slice) = c_str.to_str() {
            monitor::set_kernel_exec_time_user_label(str_slice);
        } else {
            log::warn!("hangdetect_set_kernel_exec_label: invalid UTF-8 string");
        }
    }
}
