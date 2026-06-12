//! serde deserialization comparison: metal-json (parse-to-tape + tape-walking
//! `Deserializer`) vs serde_json, into the same typed Rust struct.
//!
//! Typed deserialization needs a fixed schema, so unlike `compare.rs` this
//! bench does not walk the datasets under `data/bench` — it generates one
//! deterministic synthetic corpus in code (a ~4 MiB JSON array of objects;
//! see [`generate_corpus`]) and parses byte-identical input four ways:
//!
//! - **metal-json / borrowed**: `parse_aligned` from a caller-held
//!   [`AlignedInput`] (the zero-copy input path, as in `compare.rs`), then
//!   `Document::deserialize` into a struct with `&str` fields borrowed
//!   straight from the document's string buffer — no per-string allocation.
//!   The timed closure covers parse + deserialize; the records (cheap: no
//!   owned strings) are dropped inside, the `Document` outside
//!   (`iter_with_large_drop`), mirroring `compare.rs`.
//! - **metal-json / owned**: `Parser::parse_deserialize` into the same
//!   struct shape with owned `String` fields — the convenience one-call API.
//! - **serde_json / owned**: `from_slice` into the owned struct — the
//!   conventional baseline.
//! - **serde_json / borrowed**: `from_str` into the borrowed struct
//!   (serde_json can borrow escape-free strings from `&str` input; the
//!   generated strings contain no escapes).
//!
//! The backend follows `compare.rs`: `METAL_JSON_BENCH_BACKEND=gpu|cpu-
//! reference` when set; otherwise the GPU when a probe parse succeeds, with
//! a fallback to `Backend::CpuReference` so the bench always runs.
//!
//! An UNTIMED check asserts all four contenders produce identical record
//! checksums (id XOR / string bytes / score-bits XOR), proving the timed
//! closures do equivalent work. `Throughput::Bytes(document_len)`.

use std::fmt::Write as _;
use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use metal_json::{AlignedInput, Backend, Parser, ParserOptions};
use serde::Deserialize;

/// Target corpus size in bytes (~4 MiB; generation overshoots by at most
/// one record).
const TARGET_BYTES: usize = 4 * 1024 * 1024;

/// The owned struct shape: every string field allocates.
#[derive(Debug, Deserialize)]
struct EventOwned {
    id: u64,
    user: String,
    message: String,
    score: f64,
    active: bool,
    tags: Vec<String>,
}

/// The borrowed struct shape: `&str` fields point into the deserializer's
/// backing buffer (metal-json's string buffer / serde_json's `&str` input).
#[derive(Debug, Deserialize)]
struct EventBorrowed<'a> {
    id: u64,
    #[serde(borrow)]
    user: &'a str,
    #[serde(borrow)]
    message: &'a str,
    score: f64,
    active: bool,
    #[serde(borrow)]
    tags: Vec<&'a str>,
}

/// Order-independent record checksum used by the untimed equivalence check:
/// XOR of ids (each offset by the `active` flag), total string bytes, XOR of
/// raw score bits — every field of the structs feeds the checksum.
#[derive(Debug, Default, PartialEq, Eq)]
struct Checksum {
    id_xor: u64,
    string_bytes: u64,
    score_xor: u64,
}

fn checksum_owned(events: &[EventOwned]) -> Checksum {
    let mut c = Checksum::default();
    for e in events {
        c.id_xor ^= e.id + u64::from(e.active);
        c.string_bytes += (e.user.len() + e.message.len()) as u64;
        c.string_bytes += e.tags.iter().map(|t| t.len() as u64).sum::<u64>();
        c.score_xor ^= e.score.to_bits();
    }
    c
}

fn checksum_borrowed(events: &[EventBorrowed<'_>]) -> Checksum {
    let mut c = Checksum::default();
    for e in events {
        c.id_xor ^= e.id + u64::from(e.active);
        c.string_bytes += (e.user.len() + e.message.len()) as u64;
        c.string_bytes += e.tags.iter().map(|t| t.len() as u64).sum::<u64>();
        c.score_xor ^= e.score.to_bits();
    }
    c
}

/// Deterministic xorshift64* PRNG — no dependency, identical corpus on every
/// run and every machine.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Generate a deterministic JSON array of objects, at least [`TARGET_BYTES`]
/// long. Strings are alphanumeric words (no escapes), so serde_json's
/// borrowed path can borrow every string from the input.
fn generate_corpus() -> String {
    const WORDS: &[&str] = &[
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
        "juliett", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo",
        "sierra", "tango",
    ];
    let mut rng = Rng(0x6d65_7461_6c2d_6a73); // "metal-js"
    let mut out = String::with_capacity(TARGET_BYTES + 256);
    out.push('[');
    let mut id: u64 = 0;
    while out.len() < TARGET_BYTES {
        if id > 0 {
            out.push(',');
        }
        let user = WORDS[(rng.next() % WORDS.len() as u64) as usize];
        write!(
            out,
            r#"{{"id":{id},"user":"{user}_{}","message":""#,
            rng.next() % 10_000
        )
        .expect("write to String");
        // 4–11 space-separated words; spaces are legal unescaped in JSON.
        for w in 0..4 + rng.next() % 8 {
            if w > 0 {
                out.push(' ');
            }
            out.push_str(WORDS[(rng.next() % WORDS.len() as u64) as usize]);
        }
        write!(
            out,
            r#"","score":{}.{:02},"active":{},"tags":["#,
            rng.next() % 1_000,
            rng.next() % 100,
            rng.next().is_multiple_of(2),
        )
        .expect("write to String");
        for t in 0..1 + rng.next() % 4 {
            if t > 0 {
                out.push(',');
            }
            write!(
                out,
                r#""{}""#,
                WORDS[(rng.next() % WORDS.len() as u64) as usize]
            )
            .expect("write to String");
        }
        out.push_str("]}");
        id += 1;
    }
    out.push(']');
    out
}

/// Build the metal-json parser: honor `METAL_JSON_BENCH_BACKEND` when set
/// (as in `compare.rs`); otherwise probe the GPU and fall back to the CPU
/// oracle so the bench runs everywhere.
fn metal_parser() -> Parser {
    let requested = match std::env::var("METAL_JSON_BENCH_BACKEND").as_deref() {
        Ok("cpu-reference" | "cpu_reference" | "cpu") => Some(Backend::CpuReference),
        Ok("gpu") => Some(Backend::Gpu),
        Err(_) => None,
        Ok(other) => {
            eprintln!(
                "METAL_JSON_BENCH_BACKEND={other:?} not recognized (want gpu|cpu-reference); \
                 auto-selecting"
            );
            None
        }
    };
    for backend in requested.map_or(vec![Backend::Gpu, Backend::CpuReference], |b| vec![b]) {
        // ParserOptions is #[non_exhaustive]: construct via Default, then set.
        let mut opts = ParserOptions::default();
        opts.backend = backend;
        // Probe parse: catches a missing/paravirtual Metal device.
        match Parser::with_options(opts).and_then(|p| p.parse(b"[0]").map(|_| p)) {
            Ok(parser) => {
                eprintln!("metal-json contenders active (backend: {backend:?})");
                return parser;
            }
            Err(err) => eprintln!("metal-json backend {backend:?} unavailable: {err}"),
        }
    }
    panic!("no usable metal-json backend (GPU probe failed and CpuReference unavailable)");
}

fn bench_serde(c: &mut Criterion) {
    let corpus = generate_corpus();
    let bytes = corpus.as_bytes();
    let len = bytes.len() as u64;
    let parser = metal_parser();
    let aligned = AlignedInput::from_slice(bytes);

    // UNTIMED verification, once: all four contenders must produce the same
    // record count and checksum — proof the timed closures do the same work.
    let expected = {
        let owned: Vec<EventOwned> =
            serde_json::from_slice(bytes).expect("serde_json owned parse failed");
        let expected = checksum_owned(&owned);
        let borrowed: Vec<EventBorrowed> =
            serde_json::from_str(&corpus).expect("serde_json borrowed parse failed");
        assert_eq!(checksum_borrowed(&borrowed), expected);
        let owned: Vec<EventOwned> = parser
            .parse_deserialize(bytes)
            .expect("metal-json owned parse failed");
        assert_eq!(checksum_owned(&owned), expected);
        let doc = parser
            .parse_aligned(&aligned)
            .expect("metal-json parse failed");
        let borrowed: Vec<EventBorrowed> =
            doc.deserialize().expect("metal-json borrowed parse failed");
        assert_eq!(checksum_borrowed(&borrowed), expected);
        eprintln!(
            "synthetic corpus: {} bytes, {} records; checksums verified across all contenders",
            bytes.len(),
            borrowed.len()
        );
        expected
    };

    let mut group = c.benchmark_group("serde_synthetic_4m");
    group.throughput(Throughput::Bytes(len));
    group.sample_size(60);

    // metal-json, zero-copy: parse from the aligned input, deserialize with
    // borrowed strings. Dropping the records is cheap (no owned strings) and
    // stays inside the timed region; the Document drop (returning pooled
    // buffers) is outside, as in compare.rs.
    group.bench_function("metal-json-borrowed", |b| {
        b.iter_with_large_drop(|| {
            let doc = parser
                .parse_aligned(black_box(&aligned))
                .expect("metal-json parse failed");
            let events: Vec<EventBorrowed> =
                doc.deserialize().expect("metal-json deserialize failed");
            black_box(events.len() ^ events[0].id as usize);
            drop(events);
            doc // dropped by criterion, outside the timed region
        });
    });

    // metal-json, owned: the one-call convenience API (parse + deserialize
    // with String fields). Vec drop (many Strings) is outside timing,
    // matching serde-json-owned below.
    group.bench_function("metal-json-owned", |b| {
        b.iter_with_large_drop(|| {
            parser
                .parse_deserialize::<Vec<EventOwned>>(black_box(bytes))
                .expect("metal-json parse_deserialize failed")
        });
    });

    // serde_json into the owned struct: the conventional baseline.
    group.bench_function("serde-json-owned", |b| {
        b.iter_with_large_drop(|| {
            serde_json::from_slice::<Vec<EventOwned>>(black_box(bytes))
                .expect("serde_json parse failed")
        });
    });

    // serde_json borrowing from &str input (possible here: no escapes).
    group.bench_function("serde-json-borrowed", |b| {
        b.iter_with_large_drop(|| {
            serde_json::from_str::<Vec<EventBorrowed>>(black_box(corpus.as_str()))
                .expect("serde_json parse failed")
        });
    });

    group.finish();
    black_box(expected);
}

criterion_group!(benches, bench_serde);
criterion_main!(benches);
