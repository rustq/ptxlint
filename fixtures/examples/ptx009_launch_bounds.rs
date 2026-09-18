//! PTX009: no `.maxntid` in the PTX. Without launch bounds ptxas has to budget
//! registers for the largest block the kernel could ever be launched with,
//! which can cost occupancy. Rust has no attribute for this yet, so every
//! kernel built from Rust currently lands here.
//!
//! The loads are vectorised on purpose, so this case does not also trip PTX008.
#![no_std]
#![feature(abi_gpu_kernel)]

use ptxlint_fixtures::tid;

/// `[f32; 4]` alone is only 4-byte aligned, which is not enough for a `.v4`
/// access; the alignment has to be stated explicitly.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct F32x4([f32; 4]);

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn launch_bounds(x: *const F32x4, y: *mut f32, n: u32) {
    let i = tid();
    if i >= n as usize {
        return;
    }
    let a = unsafe { (*x.add(i)).0 };
    let b = unsafe { (*x.add(i + 1024)).0 };
    // Enough live values to keep occupancy under 100%.
    let p = [a[0] * b[3], a[1] * b[2], a[2] * b[1], a[3] * b[0]];
    let q = [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
    let r = p[0] * q[3] + p[1] * q[2] + p[2] * q[1] + p[3] * q[0];
    unsafe { *y.add(i) = r + p[0] * q[0] + p[3] * q[3] };
}
