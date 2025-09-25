use std::ffi::c_int;

mod cuda_funcs;

trait CudaErrorExt {
    fn as_return_value(&self) -> c_int;
}

impl CudaErrorExt for Result<(), cuda_funcs::CUDAError> {
    fn as_return_value(&self) -> c_int {
        match self {
            Ok(_) => 0,
            Err(e) => {
                eprintln!("CUDA Error: {}", e);
                e.code
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cudaLaunchKernel(
    func: *const std::ffi::c_void,
    grid_dim: cuda_funcs::Dim3,
    block_dim: cuda_funcs::Dim3,
    args: *const *const std::ffi::c_void,
    shared_mem: usize,
    stream: *mut std::ffi::c_void,
) -> c_int {
    cuda_funcs::get_cuda_func_name(func)
        .and_then(|func_name| {
            eprintln!("launch function {}", func_name);
            cuda_funcs::launch_cuda_kernel(func, grid_dim, block_dim, args, shared_mem, stream)
        })
        .as_return_value()
}
