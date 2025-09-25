use libc::c_int;
use std::ffi::c_void;
use std::ptr::null;
use std::sync::Once;

#[repr(C)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

type CudaFuncGetNameFunc = unsafe extern "C" fn(
    name: *mut *const std::ffi::c_char,
    func: *const c_void,
) -> std::ffi::c_int;

type CudaFuncLaunchKernel = unsafe extern "C" fn(
    func: *const c_void,
    grid_dim: Dim3,
    block_dim: Dim3,
    args: *const *const c_void,
    shared_mem: usize,
    stream: *mut c_void,
) -> std::ffi::c_int;

static CUDA_FUNCS_INIT_ONCE: Once = Once::new();
static mut CUDA_GET_NAME_FUNC: Option<CudaFuncGetNameFunc> = None;
static mut CUDA_LAUNCH_KERNEL_FUNC: Option<CudaFuncLaunchKernel> = None;

fn init_cuda_funcs() {
    CUDA_FUNCS_INIT_ONCE.call_once(|| unsafe {
        let sym = std::ffi::CString::new("cudaFuncGetName").unwrap();

        let fn_ptr = libc::dlsym(libc::RTLD_NEXT, sym.as_ptr());
        if fn_ptr.is_null() {
            panic!("failed to load cudaFuncGetName")
        }
        CUDA_GET_NAME_FUNC = Some(std::mem::transmute(fn_ptr));

        let sym = std::ffi::CString::new("cudaLaunchKernel").unwrap();
        let fn_ptr = libc::dlsym(libc::RTLD_NEXT, sym.as_ptr());
        if fn_ptr.is_null() {
            panic!("failed to load cudaLaunchKernel")
        }
        CUDA_LAUNCH_KERNEL_FUNC = Some(std::mem::transmute(fn_ptr));
    })
}
#[derive(Debug)]
pub struct CUDAError {
    pub code: c_int,
}
impl std::fmt::Display for CUDAError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUDA error code: {}", self.code)
    }
}

pub fn get_cuda_func_name(func: *const c_void) -> Result<String, CUDAError> {
    init_cuda_funcs();
    unsafe {
        let mut name_ptr: *const std::ffi::c_char = null();
        let cuda_status = CUDA_GET_NAME_FUNC.unwrap()(&mut name_ptr, func);
        if cuda_status != 0 {
            Err(CUDAError { code: cuda_status })
        } else {
            let cstr = std::ffi::CStr::from_ptr(name_ptr);
            Ok(cstr.to_str().unwrap().to_string())
        }
    }
}

pub fn launch_cuda_kernel(
    func: *const c_void,
    grid_dim: Dim3,
    block_dim: Dim3,
    args: *const *const c_void,
    shared_mem: usize,
    stream: *mut c_void,
) -> Result<(), CUDAError> {
    init_cuda_funcs();
    unsafe {
        let cuda_status =
            CUDA_LAUNCH_KERNEL_FUNC.unwrap()(func, grid_dim, block_dim, args, shared_mem, stream);
        if cuda_status != 0 {
            Err(CUDAError { code: cuda_status })
        } else {
            Ok(())
        }
    }
}
