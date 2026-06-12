//! Parse-to-tape comparison: metal-json vs C++ simdjson (FFI shim) vs Rust
//! simd-json vs serde_json, one criterion group per dataset found under
//! `<workspace>/data/bench` (populate with `cargo run -p xtask -- fetch-data`
//! / `gen-data`; gitignored).
//!
//! # Methodology
//!
//! - All contenders parse byte-identical documents. Input preparation is
//!   never timed:
//!   - **metal-json** parses from a caller-held [`AlignedInput`] (16 KiB
//!     page-aligned, space-padded tail), prepared once, via
//!     `Parser::parse_aligned` — the zero-copy `bytesNoCopy` input path:
//!     no input byte is copied inside the timed region, mirroring how
//!     simdjson parses from its caller-held padded buffer. The backend is
//!     a runtime parameter: `METAL_JSON_BENCH_BACKEND=gpu|cpu-reference`
//!     (default `gpu`). Without a usable Metal device the contender is
//!     skipped with a message.
//!   - **simdjson (C++)** parses from a `SIMDJSON_PADDING`-padded buffer
//!     prepared once; the timed call is `sj_parse_tape` (parse + a linear
//!     tape walk filling checksum stats, defeating dead-code elimination).
//!     The `dom::parser` is reused, so its tape allocation is warm —
//!     mirroring metal-json's reused parser/buffer pool.
//!   - **Symmetric tape walk**: metal-json's timed closure performs the
//!     equivalent work — parse, then `metal_stats` (the same shallow
//!     tape walk, defined once in `metal_json_bench`) — and an UNTIMED
//!     once-per-dataset check asserts both parsers produce identical stats
//!     (node count / unescaped string bytes / number-payload XOR), proving
//!     both did the same work. Document drop stays outside timing (the
//!     reused simdjson parser's tape drop is also outside its timed calls).
//!   - **simd-json (Rust)** uses `to_tape`, which parses in place and
//!     mutates its input, so each iteration needs a fresh copy; the copy is
//!     made in `iter_batched_ref` setup and is NOT timed (the API offers no
//!     non-destructive tape parse). Output drop is also outside timing.
//!   - **serde_json** is the orientation floor: `from_slice` to
//!     `serde_json::Value` (a DOM, not a tape — more allocation by design).
//!     Drop of the DOM is excluded via `iter_with_large_drop`.
//! - `Throughput::Bytes(document_len)`; criterion reports medians.
//! - Sample sizes scale down with input size (>= 100 MB: 10 samples) to
//!   keep wall-clock time sane on GB inputs.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use metal_json::{AlignedInput, Backend, Parser, ParserOptions};
use metal_json_bench::{PaddedBuf, SjParser, data_dir, list_datasets, load, metal_stats};

const MIB: u64 = 1024 * 1024;

/// Runtime backend selection for the metal-json contender.
fn metal_backend() -> Backend {
    match std::env::var("METAL_JSON_BENCH_BACKEND").as_deref() {
        Ok("cpu-reference" | "cpu_reference" | "cpu") => Backend::CpuReference,
        Ok("gpu") | Err(_) => Backend::Gpu,
        Ok(other) => {
            eprintln!(
                "METAL_JSON_BENCH_BACKEND={other:?} not recognized (want gpu|cpu-reference); using gpu"
            );
            Backend::Gpu
        }
    }
}

/// Build the metal-json parser for the selected backend, or explain why the
/// contender is skipped.
fn metal_parser() -> Option<Parser> {
    let backend = metal_backend();
    // ParserOptions is #[non_exhaustive]: construct via Default, then set.
    let mut opts = ParserOptions::default();
    opts.backend = backend;
    let parser = match Parser::with_options(opts) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("metal-json contender skipped: parser construction failed: {err}");
            return None;
        }
    };
    // Probe parse: while the GPU backend is a stub (pre-M4) this fails and
    // the contender is skipped gracefully.
    match parser.parse(b"[0]") {
        Ok(_) => {
            eprintln!("metal-json contender active (backend: {backend:?})");
            Some(parser)
        }
        Err(err) => {
            eprintln!(
                "metal-json contender skipped (backend {backend:?} unavailable: {err}); \
                 set METAL_JSON_BENCH_BACKEND=cpu-reference to bench the CPU oracle"
            );
            None
        }
    }
}

fn bench_datasets(c: &mut Criterion) {
    let datasets = list_datasets();
    if datasets.is_empty() {
        eprintln!(
            "no datasets under {} — run `cargo run -p xtask -- fetch-data` (and optionally \
             `gen-data --template twitter --size 100m`); nothing to bench",
            data_dir().display()
        );
        return;
    }

    let metal = metal_parser();
    let sj = SjParser::new();

    for (name, path) in datasets {
        let bytes = match load(&path) {
            Ok(b) => b,
            Err(err) => {
                eprintln!("skipping {}: {err}", path.display());
                continue;
            }
        };
        let len = bytes.len() as u64;

        let mut group = c.benchmark_group(&name);
        group.throughput(Throughput::Bytes(len));
        // Scale sampling effort down with input size.
        if len >= 100 * MIB {
            group.sample_size(10);
            group.measurement_time(Duration::from_secs(20));
            group.warm_up_time(Duration::from_secs(2));
        } else if len >= 10 * MIB {
            group.sample_size(20);
            group.measurement_time(Duration::from_secs(10));
        } else {
            group.sample_size(60);
        }

        // metal-json: zero-copy aligned input, reused parser (and warm
        // buffer pool). The timed closure is parse + the symmetric stats
        // walk; Document drop is untimed (iter_with_large_drop) and
        // returns the tape/string buffers to the pool, mirroring the shim
        // whose reused tape is never dropped inside its timed call.
        if let Some(parser) = &metal {
            let aligned = AlignedInput::from_slice(&bytes);

            // UNTIMED verification, once per dataset: both parsers must
            // produce identical stats — same node count, same unescaped
            // string bytes, same number-payload XOR — proving the timed
            // contenders do equivalent work on this document.
            {
                let doc = parser
                    .parse_aligned(&aligned)
                    .expect("metal-json parse failed during verification");
                let ours = metal_stats(&doc);
                let padded = PaddedBuf::from_slice(&bytes);
                let theirs = sj
                    .parse(&padded)
                    .expect("simdjson parse failed during verification");
                assert_eq!(
                    ours, theirs,
                    "{name}: metal-json and simdjson tape stats diverge — \
                     the contenders are not doing the same work"
                );
                eprintln!(
                    "{name}: stats verified vs simdjson (nodes {}, string bytes {}, \
                     number xor {:#018x})",
                    ours.node_count, ours.string_bytes, ours.number_xor
                );
            }

            group.bench_function("metal-json", |b| {
                let routine = || {
                    let doc = parser
                        .parse_aligned(black_box(&aligned))
                        .expect("metal-json parse failed");
                    let stats = metal_stats(&doc);
                    black_box(stats.node_count ^ stats.string_bytes ^ stats.number_xor);
                    doc // dropped by criterion, outside the timed region
                };
                if len >= 100 * MIB {
                    // Large inputs: BatchSize::PerIteration drops each
                    // Document between timed iterations, returning its
                    // tape/string buffers to the scratch pool so every
                    // parse runs warm — mirroring the shim, whose reused
                    // tape is warm across iterations. With
                    // iter_with_large_drop the batch ACCUMULATES documents,
                    // holding the pooled buffers alive and forcing every
                    // parse onto fresh cold allocations (measured +25%
                    // parse time at 256 MB) — a harness artifact, not
                    // steady-state behavior. Same policy as simd-json-rust
                    // below.
                    b.iter_batched(|| (), |()| routine(), BatchSize::PerIteration);
                } else {
                    b.iter_with_large_drop(routine);
                }
            });
        }

        // C++ simdjson via the FFI shim: pre-padded input, reused parser.
        {
            let padded = PaddedBuf::from_slice(&bytes);
            group.bench_function("simdjson-cpp", |b| {
                b.iter(|| {
                    let stats = sj
                        .parse(black_box(&padded))
                        .expect("simdjson parse failed");
                    black_box(stats.node_count ^ stats.string_bytes ^ stats.number_xor)
                });
            });
        }

        // Rust simd-json: to_tape mutates its input, so each iteration gets
        // a fresh copy in (untimed) setup.
        {
            let batch = if len >= 100 * MIB {
                BatchSize::PerIteration
            } else {
                BatchSize::LargeInput
            };
            group.bench_function("simd-json-rust", |b| {
                b.iter_batched_ref(
                    || bytes.clone(),
                    |buf| {
                        let tape = simd_json::to_tape(buf).expect("simd-json parse failed");
                        black_box(&tape);
                    },
                    batch,
                );
            });
        }

        // serde_json DOM: the orientation floor.
        group.bench_function("serde-json", |b| {
            b.iter_with_large_drop(|| {
                serde_json::from_slice::<serde_json::Value>(black_box(&bytes))
                    .expect("serde_json parse failed")
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_datasets);
criterion_main!(benches);
