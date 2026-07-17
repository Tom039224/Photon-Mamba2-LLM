//! Build script: compile `kernel/` to a PTX module and expose its path
//! to the host crate via the `KERNEL_PTX` env var (consumed by
//! `include_str!(env!("KERNEL_PTX"))` in `src/lib.rs`).
//!
//! Runs only when the `cuda` feature is enabled. CPU-only builds are
//! a no-op so contributors without an nvptx64 toolchain or CUDA
//! installation can still build the workspace.

use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        // No CUDA backend requested — leave KERNEL_PTX unset.
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = manifest_dir.join("kernel");
    let kernel_target = manifest_dir.join("target-nvptx");

    println!(
        "cargo:rerun-if-changed={}",
        kernel_dir.join("src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        kernel_dir.join("Cargo.toml").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        kernel_dir.join("rust-toolchain.toml").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        kernel_dir.join(".cargo/config.toml").display()
    );

    // The parent cargo passes its resolved binary path via $CARGO,
    // which bypasses the rustup shim and pins the child to the parent
    // (stable) toolchain. We need rustup to re-read
    // kernel/rust-toolchain.toml and switch to nightly + nvptx64, so
    // invoke `cargo` from PATH (the rustup shim) and drop CARGO from
    // the child's env.
    let status = Command::new("cargo")
        .args(["build", "--release", "--target", "nvptx64-nvidia-cuda"])
        .current_dir(&kernel_dir)
        .env("CARGO_TARGET_DIR", &kernel_target)
        // Strip parent-build context so the kernel's own .cargo/config.toml
        // and rust-toolchain.toml take effect. RUSTUP_TOOLCHAIN in
        // particular pins the rustup shim to the parent's (stable)
        // toolchain and prevents the nvptx64 target from resolving.
        .env_remove("CARGO")
        .env_remove("CARGO_BUILD_TARGET")
        .env_remove("RUSTC")
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("RUSTUP_TOOLCHAIN")
        .status()
        .expect("failed to spawn cargo for pm-cuda kernel build");

    assert!(status.success(), "pm-cuda kernel build failed");

    let ptx_src = kernel_target.join("nvptx64-nvidia-cuda/release/pm_cuda_kernel.ptx");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_dest = out_dir.join("pm_cuda_kernel.ptx");
    fs::copy(&ptx_src, &ptx_dest)
        .unwrap_or_else(|e| panic!("failed to copy PTX {ptx_src:?} -> {ptx_dest:?}: {e}"));

    println!("cargo:rustc-env=KERNEL_PTX={}", ptx_dest.display());
}
