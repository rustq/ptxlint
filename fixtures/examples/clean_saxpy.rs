//! The control case: coalesced f32 work, no local memory, cheap math.
//! ptxlint must report nothing on this one.
#![no_std]
#![feature(abi_gpu_kernel)]

use ptxlint_fixtures::tid;

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn clean_saxpy(a: f32, x: *const f32, y: *mut f32, n: u32) {
    let i = tid();
    if i < n as usize {
        unsafe { *y.add(i) = a * *x.add(i) + *y.add(i) };
    }
}
