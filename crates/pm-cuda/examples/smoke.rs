//! pm-cuda smoke example (PLAN I.3).
//!
//! Loads the PTX bundled into `pm_cuda::KERNEL_PTX`, launches the
//! `mul` kernel on RTX 5070, and verifies the result against a host
//! reference. Run with:
//!
//! ```text
//! cargo run -p pm-cuda --example smoke --features cuda --release
//! ```

use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

fn main() -> anyhow::Result<()> {
    let n: usize = 1024;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = ctx.load_module(Ptx::from_src(pm_cuda::KERNEL_PTX))?;
    let mul = module.load_function("mul")?;

    let a_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    let a_dev = stream.clone_htod(&a_host)?;
    let b_dev = stream.clone_htod(&b_host)?;
    let mut c_dev = stream.alloc_zeros::<f32>(n)?;

    let n_u32 = n as u32;
    let cfg = LaunchConfig::for_num_elems(n_u32);

    let mut launch = stream.launch_builder(&mul);
    launch.arg(&a_dev);
    launch.arg(&b_dev);
    launch.arg(&mut c_dev);
    launch.arg(&n_u32);
    unsafe { launch.launch(cfg) }?;

    let c_host = stream.clone_dtoh(&c_dev)?;

    for i in 0..n {
        let expected = a_host[i] * b_host[i];
        let got = c_host[i];
        anyhow::ensure!(
            (got - expected).abs() < 1e-4,
            "mismatch at {i}: got {got}, expected {expected}",
        );
    }

    println!("OK: pm-cuda smoke — {n} elements multiplied on GPU");
    println!("  c[0]   = {} * {} = {}", a_host[0], b_host[0], c_host[0]);
    println!(
        "  c[{}] = {} * {} = {}",
        n - 1,
        a_host[n - 1],
        b_host[n - 1],
        c_host[n - 1]
    );

    Ok(())
}
