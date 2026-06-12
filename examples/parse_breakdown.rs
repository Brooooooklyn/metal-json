//! Per-phase wall/GPU breakdown of full GPU parses (the M5 `timing`
//! feature's consumer):
//!
//! ```text
//! cargo run --release --features timing --example parse_breakdown -- \
//!     data/bench/twitter_512m.json [iters] [copy|aligned|file|mmap]
//! ```
//!
//! Parses the file `iters` times (default 9, after 3 warmups) on the GPU
//! backend and prints, per pipeline phase, the **median** wall time, the
//! GPU execution time of its command buffer, and the CPU-side gap
//! (`wall − gpu`) — medians per spike C (low-occupancy wall times jitter up
//! to 4×). This answers "where does the wall time go" at command-buffer
//! granularity; per-kernel counters can layer on once a GPU-bound CB is the
//! bottleneck.
//!
//! The third argument selects the input path: `copy` (default —
//! `Parser::parse`, one memcpy into a pooled buffer), `aligned`
//! (`Parser::parse_aligned`, zero copy), `file` (`Parser::parse_file`,
//! one read straight into a pooled buffer) or `mmap` (the unsafe
//! `Parser::parse_file_mmap`, mmap zero copy — sound here because nothing
//! modifies the benchmark file mid-run).

use std::time::Instant;

use metal_json::gpu::timing::{take_kernel_timings, take_parse_timings};
use metal_json::{AlignedInput, Backend, Document, Parser, ParserOptions};

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    values[values.len() / 2]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        eprintln!("usage: parse_breakdown <file.json> [iters]");
        std::process::exit(2);
    });
    let iters: usize = args
        .next()
        .map(|s| s.parse().expect("iters must be a number"))
        .unwrap_or(9);
    let mode = args.next().unwrap_or_else(|| "copy".to_owned());

    let bytes = std::fs::read(&path).expect("read input file");
    let aligned = AlignedInput::from_slice(&bytes);
    let mut opts = ParserOptions::default();
    opts.backend = Backend::Gpu;
    let parser = Parser::with_options(opts).expect("GPU parser (Metal device required)");

    let run = |label: &str| -> Document {
        match label {
            "copy" => parser.parse(&bytes).expect("parse failed"),
            "aligned" => parser.parse_aligned(&aligned).expect("parse_aligned failed"),
            "file" => parser.parse_file(&path).expect("parse_file failed"),
            // SAFETY: the benchmark input file is not modified while this
            // example runs (the caller's contract for the mmap path).
            "mmap" => unsafe { parser.parse_file_mmap(&path) }.expect("parse_file_mmap failed"),
            other => {
                eprintln!("unknown mode {other:?} (want copy|aligned|file|mmap)");
                std::process::exit(2);
            }
        }
    };

    // Warmups: PSO creation, page faults, GPU power ramp.
    for _ in 0..3 {
        let doc = run(&mode);
        drop(doc);
        let _ = take_parse_timings();
        let _ = take_kernel_timings();
    }

    // Phase samples in first-seen order: (name, wall samples, gpu samples).
    let mut order: Vec<&'static str> = Vec::new();
    let mut walls: std::collections::HashMap<&'static str, Vec<f64>> =
        std::collections::HashMap::new();
    let mut gpus: std::collections::HashMap<&'static str, Vec<f64>> =
        std::collections::HashMap::new();
    let mut totals: Vec<f64> = Vec::new();

    // Per-kernel samples keyed by dispatch-order position (sort passes
    // repeat their kernel names): (name, samples per iteration).
    let mut kernel_rows: Vec<(String, Vec<f64>)> = Vec::new();

    for _ in 0..iters {
        let t = Instant::now();
        let doc = run(&mode);
        totals.push(t.elapsed().as_secs_f64());
        drop(doc); // untimed, like the criterion harness
        let timings = take_parse_timings().expect("timing feature recorded the parse");
        for phase in timings.phases {
            if !walls.contains_key(phase.name) {
                order.push(phase.name);
            }
            walls.entry(phase.name).or_default().push(phase.wall_seconds);
            gpus.entry(phase.name).or_default().push(phase.gpu_seconds);
        }
        for (i, (name, gpu)) in take_kernel_timings().into_iter().enumerate() {
            if kernel_rows.len() <= i {
                kernel_rows.push((name.clone(), Vec::new()));
            }
            assert_eq!(kernel_rows[i].0, name, "dispatch order stable across iters");
            kernel_rows[i].1.push(gpu);
        }
    }

    let total = median(&mut totals);
    let gb = bytes.len() as f64 / 1e9;
    println!(
        "\n{} — {} bytes, {iters} iters, input mode {mode} (medians)\n",
        path,
        bytes.len()
    );
    println!(
        "{:<42} {:>10} {:>10} {:>10} {:>7}",
        "phase", "wall ms", "gpu ms", "gap ms", "% wall"
    );
    let mut wall_sum = 0.0;
    let mut gpu_sum = 0.0;
    for name in &order {
        let w = median(walls.get_mut(name).expect("phase recorded"));
        let g = median(gpus.get_mut(name).expect("phase recorded"));
        wall_sum += w;
        gpu_sum += g;
        println!(
            "{:<42} {:>10.3} {:>10.3} {:>10.3} {:>6.1}%",
            name,
            w * 1e3,
            g * 1e3,
            (w - g) * 1e3,
            w / total * 100.0
        );
    }
    println!(
        "{:<42} {:>10.3} {:>10.3} {:>10.3} {:>6.1}%",
        "(unaccounted: encode-call gaps etc.)",
        (total - wall_sum) * 1e3,
        0.0,
        (total - wall_sum) * 1e3,
        (total - wall_sum) / total * 100.0
    );
    println!(
        "{:<42} {:>10.3} {:>10.3} {:>10.3} {:>6.1}%",
        "TOTAL parse wall",
        total * 1e3,
        gpu_sum * 1e3,
        (total - gpu_sum) * 1e3,
        100.0
    );
    println!(
        "\nthroughput: {:.3} GB/s wall | GPU-execution-only bound: {:.3} GB/s",
        gb / total,
        gb / gpu_sum
    );

    if !kernel_rows.is_empty() {
        println!(
            "\nper-kernel GPU times (METAL_JSON_SPLIT_KERNELS=1, medians of {iters}):\n"
        );
        println!("{:<4} {:<32} {:>10} {:>8}", "#", "kernel", "gpu ms", "% gpu");
        let kernel_sum: f64 = kernel_rows
            .iter_mut()
            .map(|(_, samples)| median(samples))
            .sum();
        for (i, (name, samples)) in kernel_rows.iter_mut().enumerate() {
            let g = median(samples);
            println!(
                "{:<4} {:<32} {:>10.3} {:>7.1}%",
                i,
                name,
                g * 1e3,
                g / kernel_sum * 100.0
            );
        }
        println!("{:<4} {:<32} {:>10.3} {:>7.1}%", "", "SUM", kernel_sum * 1e3, 100.0);
    }
}
