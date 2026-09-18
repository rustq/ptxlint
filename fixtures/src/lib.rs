//! Shared scaffolding for the fixture kernels in `examples/`.
#![no_std]
#![feature(stdarch_nvptx)]

use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { core::arch::nvptx::trap() }
}

/// Global thread index.
#[inline(always)]
pub fn tid() -> usize {
    unsafe { (_block_idx_x() * _block_dim_x() + _thread_idx_x()) as usize }
}
