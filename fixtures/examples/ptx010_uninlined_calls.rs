//! PTX010: libdevice maths are real ABI calls unless they get inlined, and an
//! ABI call costs a stack frame and blocks cross-function register allocation.
#![no_std]
#![feature(abi_gpu_kernel)]

use ptxlint_fixtures::tid;

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn libdevice_calls(x: *const f32, y: *mut f32, n: u32) {
    let i = tid();
    if i < n as usize {
        let v = unsafe { *x.add(i) };
        unsafe { *y.add(i) = sinf(v) * sqrtf(v.abs()) };
    }
}

unsafe extern "C" {
    #[link_name = "__nv_sinf"]
    fn sinf(x: f32) -> f32;
    #[link_name = "__nv_sqrtf"]
    fn sqrtf(x: f32) -> f32;
}
