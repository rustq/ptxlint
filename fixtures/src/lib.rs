//! Deliberately good and bad kernels, used to generate real PTX test fixtures.
#![no_std]
#![feature(abi_gpu_kernel)]
#![feature(stdarch_nvptx)]

use core::arch::nvptx::*;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { trap() }
}

#[inline(always)]
fn tid() -> usize {
    unsafe { (_block_idx_x() * _block_dim_x() + _thread_idx_x()) as usize }
}

/// Clean f32 kernel: coalesced, no local memory, cheap math.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn good_saxpy(a: f32, x: *const f32, y: *mut f32, n: u32) {
    let i = tid();
    if i < n as usize {
        unsafe { *y.add(i) = a * *x.add(i) + *y.add(i) };
    }
}

/// The classic Rust footgun: float literals default to f64, so this whole
/// kernel runs on the FP64 units (1/32 rate on consumer GeForce cards).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bad_f64_literals(x: *const f32, y: *mut f32, n: u32) {
    let i = tid();
    if i < n as usize {
        let v = unsafe { *x.add(i) } as f64;
        // `0.5`, `1.0` etc. are f64 -> the whole expression is promoted.
        let r = v * 0.5 + 1.0 / (1.0 + v * v);
        unsafe { *y.add(i) = r as f32 };
    }
}

/// Dynamically indexed local array -> spilled to local memory (DRAM-backed).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn bad_local_array(x: *const f32, y: *mut f32, n: u32) {
    let i = tid();
    if i >= n as usize {
        return;
    }
    let mut scratch = [0.0f32; 64];
    for (k, s) in scratch.iter_mut().enumerate() {
        *s = unsafe { *x.add(i + k) };
    }
    // Data-dependent index defeats register promotion.
    let j = (unsafe { *x.add(i) } as usize) & 63;
    let mut acc = 0.0f32;
    for k in 0..8 {
        acc += scratch[(j + k * 7) & 63];
    }
    unsafe { *y.add(i) = acc };
}

/// Transcendental-heavy kernel: fine, but worth reporting in the instruction mix.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn transcendental(x: *const f32, y: *mut f32, n: u32) {
    let i = tid();
    if i < n as usize {
        let v = unsafe { *x.add(i) };
        unsafe { *y.add(i) = libm_sinf(v) * libm_sqrtf(v.abs()) };
    }
}

unsafe extern "C" {
    #[link_name = "__nv_sinf"]
    fn libm_sinf(x: f32) -> f32;
    #[link_name = "__nv_sqrtf"]
    fn libm_sqrtf(x: f32) -> f32;
}
