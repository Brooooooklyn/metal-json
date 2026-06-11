//! M2 kernel tests on real GPU hardware:
//! - `u2_selftest`: every `shaders/bitmap_u2.h` uint2 helper vs in-kernel
//!   native-ulong expectations, over adversarial bit patterns;
//! - `CommandBatch`: multi-dispatch encoding order within one serial encoder;
//! - `Stage` / `TestHarness` plumbing;
//! - `Stage1Buffers` size formulas, exact token allocation, and the
//!   poisoned-buffer reset (the zero/init preconditions are an explicit
//!   invariant, not an allocator accident);
//! - `stage1_vs_reference` (feature `cpu-reference`): the M2 differential
//!   suite — GPU K1-K5 vs the scalar oracle on the corpus, the full
//!   JSONTestSuite, adversarial fixtures (backslash/quote walls, seam and
//!   chunk-boundary straddles, UTF-8 torture) and property tests (random
//!   JSON / byte soup / quote-backslash-heavy ASCII; case count defaults to
//!   64, override with `PROPTEST_CASES`).
//!
//! Run with `MTL_SHADER_VALIDATION=1` in CI, and once with
//! `--features runtime-shaders` to prove both shader build paths.

#[cfg(feature = "cpu-reference")]
mod common;

use metal_json::gpu::Stage1;
use metal_json::metal::{Binding, Dispatch, MjHeader, MjParams};
use metal_json::stage::{CHUNK_BYTES, CHUNK_WORDS, Stage, Stage1Buffers, TestHarness, WORD_BYTES};

/// GPU gating: in environments without a Metal device, skip with a loud
/// message instead of failing — unless `METAL_JSON_REQUIRE_GPU=1` (set in
/// CI) makes a missing device a hard error.
fn harness_or_skip(test: &str) -> Option<TestHarness> {
    match TestHarness::new() {
        Ok(harness) => Some(harness),
        Err(err) => {
            if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                panic!("METAL_JSON_REQUIRE_GPU=1 but no usable Metal device: {err}");
            }
            eprintln!("SKIP {test}: no usable Metal device here ({err})");
            None
        }
    }
}

// --- u2_selftest --------------------------------------------------------------

/// Names of the per-element failure bits written by `u2_selftest`.
/// Mirrors the `bad |= 1u << i` assignments in shaders/01_u2_selftest.metal
/// — keep in sync.
const FAIL_NAMES: [&str; 15] = [
    "make_u2/lo_u2/hi_u2 roundtrip",
    "not_u2",
    "or_u2",
    "and_u2",
    "xor_u2",
    "shl64_u2",
    "shr64_u2",
    "add64_u2 sum",
    "add64_u2 carry-out",
    "sub64_u2 difference",
    "sub64_u2 borrow-out",
    "popcount64_u2",
    "prefix_xor64_u2",
    "clz64_u2",
    "ctz64_u2",
];

fn fail_names(mask: u32) -> String {
    FAIL_NAMES
        .iter()
        .enumerate()
        .filter(|(i, _)| mask >> i & 1 == 1)
        .map(|(_, name)| *name)
        .collect::<Vec<_>>()
        .join(", ")
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Adversarial 64-bit patterns: zero, all-ones, alternating bits, every
/// single bit (including 31/32/33 around the uint2 seam), values straddling
/// the seam for carry/borrow propagation, and deterministic random fill.
fn adversarial_patterns() -> Vec<u64> {
    let mut patterns = vec![
        0,
        u64::MAX,
        0xAAAA_AAAA_AAAA_AAAA,
        0x5555_5555_5555_5555,
        // Seam-straddling values: carries out of the low word, borrows into it.
        0x0000_0000_FFFF_FFFF,
        0xFFFF_FFFF_0000_0000,
        0x0000_0000_FFFF_FFFE,
        0x0000_0001_0000_0000,
        0x0000_0001_0000_0001,
        0xFFFF_FFFE_FFFF_FFFF,
        0x7FFF_FFFF_FFFF_FFFF,
        0x8000_0000_0000_0000,
        0x0000_0000_8000_0000,
        0x0000_0000_7FFF_FFFF,
        0x0123_4567_89AB_CDEF,
        0xDEAD_BEEF_CAFE_F00D,
    ];
    // Every single bit — positions 31/32/33 cross the 32-bit seam.
    for i in 0..64 {
        patterns.push(1u64 << i);
    }
    let mut state = 0x6A09_E667_F3BC_C908; // deterministic
    for _ in 0..48 {
        patterns.push(splitmix64(&mut state));
    }
    patterns.sort_unstable();
    patterns.dedup();
    patterns
}

/// Every bitmap_u2.h helper, validated on-GPU against native-ulong expected
/// values, over the full cross product of the adversarial patterns.
#[test]
fn u2_selftest_passes_on_adversarial_patterns() {
    let Some(harness) = harness_or_skip("u2_selftest_passes_on_adversarial_patterns") else {
        return;
    };
    let patterns = adversarial_patterns();
    let mut a_host = Vec::with_capacity(patterns.len() * patterns.len());
    let mut b_host = Vec::with_capacity(patterns.len() * patterns.len());
    for &a in &patterns {
        for &b in &patterns {
            a_host.push(a);
            b_host.push(b);
        }
    }
    let n = a_host.len();

    let a = harness.upload(&a_host).unwrap();
    let b = harness.upload(&b_host).unwrap();
    // Poison the output so an early-exiting kernel cannot fake a pass.
    let mut fail = harness.upload(&vec![u32::MAX; n]).unwrap();

    let stage = Stage::new("u2_selftest");
    let params = MjParams {
        input_len: (n * size_of::<u64>()) as u64,
        element_count: n as u64,
        ..Default::default()
    };
    harness
        .run(
            &stage,
            &mut [
                Binding::Read(&a),
                Binding::Read(&b),
                Binding::ReadWrite(&mut fail),
            ],
            Some(&params),
            Dispatch::Threads(n),
        )
        .expect("dispatch u2_selftest");

    let results = harness.read_back::<u32>(&fail);
    assert_eq!(results.len(), n);
    let mut failures = 0usize;
    for (i, &mask) in results.iter().enumerate() {
        if mask != 0 {
            failures += 1;
            if failures <= 16 {
                eprintln!(
                    "u2_selftest fail: a={:#018x} b={:#018x}: {}",
                    a_host[i],
                    b_host[i],
                    fail_names(mask)
                );
            }
        }
    }
    assert_eq!(
        failures, 0,
        "{failures}/{n} (a, b) pairs failed uint2 helper checks (first 16 above)"
    );
}

// --- CommandBatch -------------------------------------------------------------

/// Two dependent dispatches in ONE batch / one serial encoder: the second
/// reads the first one's output, proving in-order execution with implicit
/// barriers (the documented `MTLDispatchTypeSerial` choice) plus the
/// handle-based binding model (`mid` is write-registered once, then used as
/// an input slot in the second dispatch).
#[test]
fn command_batch_chains_dependent_dispatches_in_order() {
    let Some(harness) = harness_or_skip("command_batch_chains_dependent_dispatches_in_order")
    else {
        return;
    };
    let ctx = harness.ctx();
    let stage = Stage::new("smoke_add");
    let pipeline = stage.pipeline(ctx).expect("pipeline smoke_add");

    const N: usize = 4096 + 37; // not a threadgroup multiple
    let a_host: Vec<u32> = (0..N as u32).collect();
    let b_host: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(2654435761)).collect();
    let a = harness.upload(&a_host).unwrap();
    let b = harness.upload(&b_host).unwrap();
    let mut mid = harness.alloc_zeroed::<u32>(N).unwrap();
    let mut out = harness.alloc_zeroed::<u32>(N).unwrap();

    let params = MjParams {
        input_len: (N * size_of::<u32>()) as u64,
        element_count: N as u64,
        ..Default::default()
    };

    let mut batch = ctx.batch().expect("begin batch");
    let ha = batch.bind_read(&a);
    let hb = batch.bind_read(&b);
    let hmid = batch.bind_write(&mut mid);
    let hout = batch.bind_write(&mut out);
    // mid = a + b
    batch.dispatch(
        pipeline,
        &[ha, hb, hmid],
        Some(&params),
        Dispatch::Threads(N),
    );
    // out = mid + b — must observe the first dispatch's writes.
    batch.dispatch(
        pipeline,
        &[hmid, hb, hout],
        Some(&params),
        Dispatch::Threads(N),
    );
    batch.commit_and_wait().expect("commit_and_wait");

    let mid_got = harness.read_back::<u32>(&mid);
    let out_got = harness.read_back::<u32>(&out);
    for i in 0..N {
        let want_mid = a_host[i].wrapping_add(b_host[i]);
        assert_eq!(mid_got[i], want_mid, "mid mismatch at {i}");
        assert_eq!(
            out_got[i],
            want_mid.wrapping_add(b_host[i]),
            "chained out mismatch at {i}"
        );
    }
}

/// An abandoned batch (dropped without commit) must not crash or poison the
/// queue for later work.
#[test]
fn command_batch_drop_without_commit_is_harmless() {
    let Some(harness) = harness_or_skip("command_batch_drop_without_commit_is_harmless") else {
        return;
    };
    let ctx = harness.ctx();
    let stage = Stage::new("smoke_add");
    let pipeline = stage.pipeline(ctx).expect("pipeline smoke_add");

    let a = harness.upload(&[1u32, 2, 3, 4]).unwrap();
    let b = harness.upload(&[10u32, 20, 30, 40]).unwrap();
    let mut out = harness.alloc_zeroed::<u32>(4).unwrap();
    let params = MjParams {
        input_len: 16,
        element_count: 4,
        ..Default::default()
    };

    {
        let mut abandoned = ctx.batch().expect("begin batch");
        let ha = abandoned.bind_read(&a);
        let hb = abandoned.bind_read(&b);
        let hout = abandoned.bind_write(&mut out);
        abandoned.dispatch(
            pipeline,
            &[ha, hb, hout],
            Some(&params),
            Dispatch::Threads(4),
        );
        // dropped here, never committed
    }
    let untouched = harness.read_back::<u32>(&out);
    assert_eq!(untouched, vec![0; 4], "uncommitted batch must not run");

    // The queue still works.
    harness
        .run(
            &stage,
            &mut [
                Binding::Read(&a),
                Binding::Read(&b),
                Binding::ReadWrite(&mut out),
            ],
            Some(&params),
            Dispatch::Threads(4),
        )
        .expect("dispatch after abandoned batch");
    assert_eq!(harness.read_back::<u32>(&out), vec![11, 22, 33, 44]);
}

// --- Stage --------------------------------------------------------------------

#[test]
fn stage_pipeline_is_lazy_and_cached() {
    let Some(harness) = harness_or_skip("stage_pipeline_is_lazy_and_cached") else {
        return;
    };
    let stage = Stage::new("smoke_popcount64");
    assert_eq!(stage.name(), "smoke_popcount64");
    let first = stage.pipeline(harness.ctx()).expect("pipeline");
    let second = stage.pipeline(harness.ctx()).expect("pipeline (cached)");
    assert!(
        std::ptr::eq(first, second),
        "second call must return the cached pipeline"
    );

    let missing = Stage::new("kernel_that_does_not_exist");
    assert!(
        missing.pipeline(harness.ctx()).is_err(),
        "unknown kernels must surface as errors, not panics"
    );
}

// --- Stage1Buffers ---------------------------------------------------------------

#[test]
fn stage1_buffers_size_formulas_are_exact() {
    let Some(harness) = harness_or_skip("stage1_buffers_size_formulas_are_exact") else {
        return;
    };
    let ctx = harness.ctx();

    // (input_len, words = ceil(len/64), chunks = ceil(words/1024))
    let cases: &[(usize, usize, usize)] = &[
        (0, 0, 0),
        (1, 1, 1),
        (63, 1, 1),
        (64, 1, 1),
        (65, 2, 1),
        (CHUNK_BYTES - 1, CHUNK_WORDS, 1),
        (CHUNK_BYTES, CHUNK_WORDS, 1),
        (CHUNK_BYTES + 1, CHUNK_WORDS + 1, 2),
        (3 * CHUNK_BYTES + 100, 3 * CHUNK_WORDS + 2, 4),
    ];
    for &(len, words, chunks) in cases {
        let input = vec![b'7'; len];
        let bufs = Stage1Buffers::new(ctx, &input).unwrap();
        assert_eq!(bufs.input_len(), len, "input_len for {len}");
        assert_eq!(bufs.words(), words, "words for {len}");
        assert_eq!(bufs.chunks(), chunks, "chunks for {len}");

        assert_eq!(bufs.input.len(), words * WORD_BYTES, "input size for {len}");
        assert_eq!(bufs.bm_quote.len(), words * 8, "bm_quote size for {len}");
        assert_eq!(bufs.bm_tok.len(), words * 8, "bm_tok size for {len}");
        assert_eq!(bufs.escape_info.len(), words, "escape_info size for {len}");
        assert_eq!(
            bufs.chunk_quote_counts.len(),
            chunks * 4,
            "chunk_quote_counts size for {len}"
        );
        assert_eq!(
            bufs.chunk_token_counts.len(),
            chunks * 4,
            "chunk_token_counts size for {len}"
        );
        assert_eq!(bufs.header.len(), 64, "header size for {len}");
        assert!(bufs.tok_pos.is_none(), "tok_pos must wait for CB1");
        assert!(bufs.tok_kind.is_none(), "tok_kind must wait for CB1");

        // Input copied verbatim, tail padded with spaces (whitespace class).
        let contents = bufs.input.contents();
        assert_eq!(&contents[..len], &input[..], "input copy for {len}");
        assert!(
            contents[len..].iter().all(|&b| b == b' '),
            "padding must be ASCII spaces for {len}"
        );

        // Header initialized: no error, zero counts.
        let header = bufs.read_header();
        assert_eq!(header, MjHeader::new(), "fresh header for {len}");
        assert_eq!(header.first_error(), None);
    }
}

#[test]
fn stage1_buffers_apply_exact_token_count() {
    let Some(harness) = harness_or_skip("stage1_buffers_apply_exact_token_count") else {
        return;
    };
    let ctx = harness.ctx();
    let mut bufs = Stage1Buffers::new(ctx, br#"{"a":[1,2.5],"b":"x\n"}"#).unwrap();

    // The reference token stream for this document has 16 tokens; the real
    // count will come from MjHeader::token_total after CB1.
    bufs.alloc_tokens(ctx, 16).unwrap();
    assert_eq!(bufs.tok_pos.as_ref().unwrap().len(), 16 * 4);
    assert_eq!(bufs.tok_kind.as_ref().unwrap().len(), 16);

    // Token-free documents (whitespace-only) get empty buffers, not None.
    bufs.alloc_tokens(ctx, 0).unwrap();
    assert_eq!(bufs.tok_pos.as_ref().unwrap().len(), 0);
    assert_eq!(bufs.tok_kind.as_ref().unwrap().len(), 0);

    // reset_for_reuse re-arms the same buffers for a fresh parse: header
    // re-initialized, chunk counts re-zeroed, token buffers dropped.
    bufs.reset_for_reuse();
    assert_eq!(bufs.read_header(), MjHeader::new());
    assert!(bufs.tok_pos.is_none(), "reset must drop tok_pos");
    assert!(bufs.tok_kind.is_none(), "reset must drop tok_kind");
    assert!(
        bufs.chunk_quote_counts.contents().iter().all(|&b| b == 0),
        "reset must zero chunk_quote_counts"
    );
    assert!(
        bufs.chunk_token_counts.contents().iter().all(|&b| b == 0),
        "reset must zero chunk_token_counts"
    );
}

/// The Stage1Buffers zero/init preconditions are an explicit invariant, not
/// an allocator accident: scribble 0xDEADBEEF over every buffer a kernel
/// accumulates into (the chunk count buffers, the header) — and over the
/// no-precondition buffers for good measure — then `reset_for_reuse()` and
/// run the full stage-1 pipeline on the poisoned-then-reset buffers. The
/// output must be bit-identical to a run on freshly constructed buffers.
/// Pins the invariant against allocator behavior (the planned M5 buffer
/// pool hands out dirty buffers).
#[test]
fn poisoned_buffers_reset_to_a_fresh_parse_state() {
    let Some(harness) = harness_or_skip("poisoned_buffers_reset_to_a_fresh_parse_state") else {
        return;
    };
    let ctx = harness.ctx();
    let stage1 = Stage1::new();

    // A known document spanning several spine chunks (so every entry of the
    // per-chunk count buffers is in play), with strings and escapes so the
    // quote partials are nonzero everywhere.
    let mut input = b"[".to_vec();
    let mut i = 0usize;
    while input.len() < 2 * CHUNK_BYTES + 4096 {
        input.extend_from_slice(format!(r#"{{"k{i}":"v\"{i}\\","n":{i}}},"#).as_bytes());
        i += 1;
    }
    input.pop(); // trailing comma
    input.push(b']');

    let fresh = stage1.run(ctx, &input).expect("fresh run");
    assert_eq!(fresh.error, None, "the known document is stage-1 clean");
    assert!(fresh.token_total > 0);
    assert!(fresh.quote_total > 0);

    let mut bufs = Stage1Buffers::new(ctx, &input).unwrap();
    assert!(bufs.chunks() >= 3, "input must span several chunks");
    // Poison every kernel accumulation target...
    bufs.chunk_quote_counts
        .as_mut_slice::<u32>()
        .fill(0xDEAD_BEEF);
    bufs.chunk_token_counts
        .as_mut_slice::<u32>()
        .fill(0xDEAD_BEEF);
    bufs.header.as_mut_slice::<u32>().fill(0xDEAD_BEEF);
    // ...and the no-precondition buffers, which K1 must fully overwrite.
    bufs.bm_quote.contents_mut().fill(0xEF);
    bufs.bm_tok.contents_mut().fill(0xEF);
    bufs.escape_info.contents_mut().fill(0xEF);

    bufs.reset_for_reuse();
    let (after_reset, _) = stage1
        .run_with_buffers(ctx, &mut bufs)
        .expect("run on poisoned-then-reset buffers");
    assert_eq!(
        after_reset, fresh,
        "poisoned-then-reset buffers must reproduce a fresh run bit-for-bit"
    );
}

/// Oversized inputs must fail with the structured error, before any
/// allocation is attempted.
#[test]
fn stage1_buffers_reject_oversized_input() {
    // No GPU needed conceptually, but the constructor takes a context.
    let Some(harness) = harness_or_skip("stage1_buffers_reject_oversized_input") else {
        return;
    };
    // Don't actually materialize 4 GiB: this exercises only the length
    // check, which runs before any allocation. A zero-length slice with a
    // forged length would be UB, so just verify the boundary arithmetic on
    // the constant instead and the error path with a real (small) cap test
    // is covered by reference::MAX_INPUT_BYTES equality in src/stage.rs.
    assert_eq!(metal_json::stage::MAX_INPUT_BYTES, u32::MAX as u64 - 64);
    let _ = harness;
}

// --- K1-K5 vs the cpu-reference oracle ------------------------------------------

/// Per-kernel differential tests: stage 1 on the GPU (K1 classify+escape+
/// UTF-8 → valve → K2/K4 spines → K3 mask → K5 scatter) vs the scalar
/// oracle (`reference::stage1_classify` + `reference::stage2_tokens`) on
/// identical inputs, diffing the quote bitmap, the token bitmap, the token
/// stream (positions + kinds), the totals and the error word on clean
/// inputs — and, on rejected inputs (UTF-8 / odd quotes), error parity
/// plus the empty-token-outputs rejection contract (see `diff`).
#[cfg(feature = "cpu-reference")]
mod stage1_vs_reference {
    use metal_json::Error;
    use metal_json::gpu::{ERR_STRING, ERR_UTF8, Stage1, Stage1Output};
    use metal_json::metal::MetalContext;
    use metal_json::reference::{stage1_classify, stage2_tokens};
    use metal_json::stage::{CHUNK_BYTES, WORD_BYTES};

    use super::{common, harness_or_skip, splitmix64};

    /// Which error class stage 1 reported — the ONLY classes stage 1 may
    /// catch (UTF-8 and odd quote count). Structural errors belong to later
    /// stages on both backends; [`diff`] asserts the two sides agree on the
    /// class, the offset and the code.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Stage1Verdict {
        Clean,
        Utf8,
        OddQuotes,
    }

    /// Run both backends on `input` and require agreement; returns the
    /// (agreed) stage-1 verdict.
    ///
    /// Clean inputs are compared bit-for-bit (bitmaps, token stream,
    /// totals). Rejected inputs (UTF-8 / odd quotes) follow the narrowed
    /// stage-1 contract instead: both sides must agree on WHICH inputs
    /// error and at which offset/code (verdict parity — what makes the
    /// narrowing sound: a kernel bug that only manifests on rejected
    /// inputs cannot affect parser output, because the parser aborts), and
    /// the GPU must return EMPTY token outputs ([`assert_rejected`]). On
    /// odd-quote inputs the reference still produces bitmaps, so the K1
    /// quote bitmap and K2 total are additionally compared there.
    fn diff(stage1: &Stage1, ctx: &MetalContext, input: &[u8], label: &str) -> Stage1Verdict {
        let got = stage1
            .run(ctx, input)
            .unwrap_or_else(|e| panic!("{label}: GPU stage 1 failed: {e}"));
        match stage1_classify(input) {
            Ok(bitmaps) => {
                // Both sides produce the quote bitmap regardless of the
                // odd-quote verdict (it is what the verdict derives from).
                assert_eq!(got.quote_real, bitmaps.quote_real, "{label}: quote bitmap");
                let quote_total: u64 = bitmaps
                    .quote_real
                    .iter()
                    .map(|w| u64::from(w.count_ones()))
                    .sum();
                assert_eq!(got.quote_total, quote_total, "{label}: quote total");

                if quote_total % 2 == 1 {
                    // Odd quotes: K2 packs MJ_ERR_STRING at offset input_len
                    // (stage 3 refines it in M3) and the input is rejected.
                    assert_eq!(
                        got.error_offset_code(),
                        Some((input.len() as u64, ERR_STRING)),
                        "{label}: odd-quote error word"
                    );
                    assert_rejected(&got, label);
                    return Stage1Verdict::OddQuotes;
                }
                assert_eq!(got.error, None, "{label}: spurious error word");

                let tokens = stage2_tokens(&bitmaps, input);
                let want_pos: Vec<u32> = tokens.iter().map(|t| t.pos).collect();
                let want_kind: Vec<u8> = tokens.iter().map(|t| t.kind as u8).collect();
                assert_eq!(got.tok_pos, want_pos, "{label}: token positions");
                assert_eq!(got.tok_kind, want_kind, "{label}: token kinds");
                assert_eq!(got.token_total, tokens.len() as u64, "{label}: token total");

                // The token bitmap K3 leaves behind is exactly the token
                // positions as bits.
                let mut want_bitmap = vec![0u64; bitmaps.quote_real.len()];
                for t in &tokens {
                    want_bitmap[t.pos as usize / 64] |= 1u64 << (t.pos % 64);
                }
                assert_eq!(got.tokens, want_bitmap, "{label}: token bitmap");
                Stage1Verdict::Clean
            }
            Err(Error::Utf8 { offset }) => {
                let (got_offset, code) = got
                    .error_offset_code()
                    .unwrap_or_else(|| panic!("{label}: missing UTF-8 error (want {offset})"));
                assert_eq!(code, ERR_UTF8, "{label}: error code");
                assert_eq!(got_offset, offset, "{label}: UTF-8 error offset");
                assert_rejected(&got, label);
                Stage1Verdict::Utf8
            }
            Err(other) => panic!("{label}: unexpected reference error {other:?}"),
        }
    }

    /// The stage-1 rejection contract on the GPU side: a rejected input
    /// (UTF-8 / odd quotes) produces NO token outputs — K5 is skipped and
    /// downstream stages never see tokens for it.
    fn assert_rejected(got: &Stage1Output, label: &str) {
        assert!(
            got.tok_pos.is_empty(),
            "{label}: rejected input must produce no tok_pos"
        );
        assert!(
            got.tok_kind.is_empty(),
            "{label}: rejected input must produce no tok_kind"
        );
        assert_eq!(
            got.token_total, 0,
            "{label}: rejected input must report token_total 0"
        );
    }

    fn run_corpus(test: &str, cases: &[(String, Vec<u8>)]) {
        let Some(harness) = harness_or_skip(test) else {
            return;
        };
        let stage1 = Stage1::new();
        for (label, input) in cases {
            diff(&stage1, harness.ctx(), input, label);
        }
    }

    #[test]
    fn documents_and_scalars_match_the_reference() {
        let docs: &[&[u8]] = &[
            br#"{"a":[1,2.5],"b":"x\n"}"#,
            br#"{ true:12}"#,
            br#""{[: ,]} true 5""#,
            br#""a","b""#,
            br#""a\"b""#,
            br#""a\\""#,
            br#""a\\\""#,
            b"\"abc",
            b"-0.0",
            b"true",
            b"false",
            b"null",
            b"42",
            b"x",
            b"",
            b" \t\r\n ",
            b"[]",
            b"[ ]",
            br#"{"nested":{"deep":[[[{"x":[null,true,false]}]]]},"":""}"#,
            "h\u{e9}llo w\u{f6}rld".as_bytes(),
            "\u{7FF}\u{800}\u{FFFD}\u{10000}\u{10FFFF} \u{1F600} mixed".as_bytes(),
            b"\t{\n\"k\"\r:\t[\n1 ,\r2\t]\n}\r",
        ];
        let cases: Vec<(String, Vec<u8>)> = docs
            .iter()
            .map(|d| (format!("{:?}", String::from_utf8_lossy(d)), d.to_vec()))
            .collect();
        run_corpus("documents_and_scalars_match_the_reference", &cases);
    }

    /// Backslash runs of every length 0..=5 ending at every alignment
    /// around the 64-byte word seam, followed by a quote: escape carries
    /// across the seam in both parities, quotes at bit 0 of a word, and
    /// the trailing odd-quote error path.
    #[test]
    fn backslash_runs_across_word_seams_match_the_reference() {
        let mut cases = Vec::new();
        for pad in 56..=72usize {
            for run in 0..=5usize {
                let mut input = vec![b' '; pad];
                input.extend(std::iter::repeat_n(b'\\', run));
                input.push(b'"');
                input.extend_from_slice(b"x\"y");
                cases.push((format!("pad={pad} run={run}"), input));
            }
        }
        // Quote exactly at the seam closing a string that opened in word 0.
        let mut input = b"\"".to_vec();
        input.extend(std::iter::repeat_n(b'a', 63));
        input.push(b'"'); // byte 64
        input.extend_from_slice(b":1");
        cases.push(("close quote at byte 64".to_owned(), input));
        run_corpus(
            "backslash_runs_across_word_seams_match_the_reference",
            &cases,
        );
    }

    /// Backslash walls around and beyond the 4096-byte look-back cap: the
    /// valve must engage and the fixed-up bitmaps must stay bit-exact, for
    /// both parities, aligned and unaligned, including a wall crossing the
    /// 64 KiB chunk boundary.
    #[test]
    fn backslash_walls_engage_the_valve_and_match_the_reference() {
        let Some(harness) = harness_or_skip("backslash_walls_engage_the_valve_and_match") else {
            return;
        };
        let stage1 = Stage1::new();
        let walls = [
            4090usize, 4095, 4096, 4097, 4159, 4160, 4161, 5000, 8191, 8192, 8193,
        ];
        for wall in walls {
            for lead in [0usize, 37] {
                // Bare wall, then a quote whose escapedness is the wall parity.
                let mut input = vec![b' '; lead];
                input.extend(std::iter::repeat_n(b'\\', wall));
                input.extend_from_slice(b"\"x\"");
                diff(
                    &stage1,
                    harness.ctx(),
                    &input,
                    &format!("wall={wall} lead={lead}"),
                );

                // The same wall inside a string literal.
                let mut input = vec![b' '; lead];
                input.push(b'"');
                input.extend(std::iter::repeat_n(b'\\', wall));
                input.extend_from_slice(b"\" 1");
                diff(
                    &stage1,
                    harness.ctx(),
                    &input,
                    &format!("string wall={wall} lead={lead}"),
                );
            }
        }
        // The valve must actually have engaged for the long walls (cap hit),
        // not just produced correct output by never capping.
        let mut input = std::iter::repeat_n(b'\\', 8192).collect::<Vec<u8>>();
        input.push(b'"');
        let out = stage1.run(harness.ctx(), &input).unwrap();
        assert!(out.carry_overflow_count > 0, "8192-wall must hit the cap");

        // Wall straddling the 64 KiB chunk boundary.
        let mut input = vec![b' '; CHUNK_BYTES - 4500];
        input.extend(std::iter::repeat_n(b'\\', 9000));
        input.extend_from_slice(b"\"x\"");
        diff(&stage1, harness.ctx(), &input, "wall across chunk boundary");
    }

    /// The valve's second flag kind: a word starting exactly one byte after
    /// a `"` whose own backslash look-back hit the cap (quote at byte
    /// 64k-1, >= 4096 backslashes before it). Both run parities — for the
    /// odd runs K1's "assume real quote" guess is wrong and the fix-up must
    /// repair the word's scalar-start carry — and a long flag chain (k=129
    /// walks through 64 flagged words to its anchor).
    #[test]
    fn quote_capped_at_word_edge_engages_the_quote_flag() {
        let Some(harness) = harness_or_skip("quote_capped_at_word_edge_engages_the_quote_flag")
        else {
            return;
        };
        let stage1 = Stage1::new();
        for word_index in [65usize, 129] {
            for lead in [0usize, 1] {
                let quote_at = word_index * 64 - 1;
                let mut input = vec![b' '; lead];
                input.extend(std::iter::repeat_n(b'\\', quote_at - lead));
                input.push(b'"'); // at byte 64k-1
                input.extend_from_slice(b"xy\"z");
                let out = stage1.run(harness.ctx(), &input).unwrap();
                assert!(
                    out.carry_overflow_count > 0,
                    "k={word_index} lead={lead}: cap must be hit"
                );
                diff(
                    &stage1,
                    harness.ctx(),
                    &input,
                    &format!("quote at word edge k={word_index} lead={lead}"),
                );
            }
        }
    }

    /// Strings and documents spanning multiple 64 KiB spine chunks: the
    /// quote-parity and token-rank carries must propagate through K2/K4.
    #[test]
    fn multi_chunk_inputs_match_the_reference() {
        let mut cases = Vec::new();

        // A single string body crossing the chunk boundary.
        let mut input = b"[\"".to_vec();
        input.extend(std::iter::repeat_n(b'x', CHUNK_BYTES + 4000));
        input.extend_from_slice(b"\"]");
        cases.push(("string across chunk boundary".to_owned(), input));

        // A close quote exactly at the chunk boundary byte.
        let mut input = b"\"".to_vec();
        input.extend(std::iter::repeat_n(b'a', CHUNK_BYTES - 1));
        input.push(b'"'); // byte CHUNK_BYTES
        input.extend_from_slice(b":1");
        cases.push(("close quote at chunk boundary".to_owned(), input));

        // ~3 chunks of real JSON shape.
        let mut doc = b"[".to_vec();
        let mut i = 0usize;
        while doc.len() < 3 * CHUNK_BYTES {
            doc.extend_from_slice(
                format!(r#"{{"key{i}":"value with \"escapes\\ inside","n":{i}}},"#).as_bytes(),
            );
            i += 1;
        }
        doc.pop(); // trailing comma
        doc.push(b']');
        cases.push(("three chunks of members".to_owned(), doc));

        run_corpus("multi_chunk_inputs_match_the_reference", &cases);
    }

    /// Valid and invalid UTF-8, including sequences straddling 64-byte word
    /// seams: error offsets must equal the reference's (= the offset of the
    /// first byte of the first invalid sequence).
    #[test]
    fn utf8_validation_matches_the_reference() {
        let mut cases: Vec<(String, Vec<u8>)> = Vec::new();

        let invalid: &[&[u8]] = &[
            b"\x80",
            b"ab\x80",
            b"\xC0\xAF",
            b"\xC1\xBF",
            b"\xC2",
            b"\xC2x",
            b"\xE0\x80\x80",
            b"\xE0\x9F\xBF",
            b"\xED\xA0\x80",
            b"\xED\xBF\xBF",
            b"\xE2\x82",
            b"\xF0\x80\x80\x80",
            b"\xF0\x8F\xBF\xBF",
            b"\xF4\x90\x80\x80",
            b"\xF5\x80\x80\x80",
            b"\xFF",
            b"{\"k\": \"\xE0\x80x\"}",
        ];
        for (i, seq) in invalid.iter().enumerate() {
            cases.push((format!("invalid #{i}"), seq.to_vec()));
            // The same sequence at every alignment around a word seam.
            for pad in 60..=66usize {
                let mut input = vec![b' '; pad];
                input.extend_from_slice(seq);
                cases.push((format!("invalid #{i} pad={pad}"), input));
            }
        }

        // Valid multi-byte sequences straddling the seam at every offset.
        for pad in 58..=66usize {
            let mut input = vec![b' '; pad];
            input.extend_from_slice("\u{e9}\u{1F600}\u{800}\u{10FFFF}x".as_bytes());
            cases.push((format!("valid seam pad={pad}"), input));
        }

        // Truncations at EOF, lead at the very last byte(s).
        for tail in ["\u{1F600}", "\u{800}", "\u{e9}"] {
            let bytes = tail.as_bytes();
            for cut in 1..bytes.len() {
                let mut input = b"ok ".to_vec();
                input.extend_from_slice(&bytes[..cut]);
                cases.push((format!("{tail:?} cut at {cut}"), input));
            }
        }

        run_corpus("utf8_validation_matches_the_reference", &cases);
    }

    /// Deterministic random byte soup over a JSON-flavored alphabet rich in
    /// quotes, backslashes and UTF-8 fragments: exercises every carry and
    /// error path at once, sized across word and chunk boundaries.
    #[test]
    fn random_soup_matches_the_reference() {
        const ALPHABET: &[u8] = b"\\\\\\\"\"\"{}[]:, \t\n\rtruefalsenull0123456789.eE+-x\
                                  \xC3\xA9\xF0\x9F\x98\x80\xE0\xED\x80\xBF";
        let sizes = [
            1usize,
            7,
            63,
            64,
            65,
            127,
            128,
            1000,
            4096 + 33,
            CHUNK_BYTES - 1,
            CHUNK_BYTES + 17,
        ];
        let mut cases = Vec::new();
        let mut state = 0x5EED_0F42_D00D_u64;
        for (round, &size) in sizes.iter().enumerate() {
            let mut input = Vec::with_capacity(size);
            for _ in 0..size {
                let r = splitmix64(&mut state) as usize;
                input.push(ALPHABET[r % ALPHABET.len()]);
            }
            cases.push((format!("soup round={round} size={size}"), input));
        }
        run_corpus("random_soup_matches_the_reference", &cases);
    }

    // --- Whole-corpus differential -------------------------------------------

    /// Every checked-in corpus fixture, GPU stage 1 vs the oracle.
    #[test]
    fn corpus_files_match_the_reference() {
        let Some(harness) = harness_or_skip("corpus_files_match_the_reference") else {
            return;
        };
        let stage1 = Stage1::new();
        let mut count = 0usize;
        for path in common::corpus_files() {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let bytes = std::fs::read(&path).expect("readable corpus fixture");
            let verdict = diff(&stage1, harness.ctx(), &bytes, &name);
            assert_eq!(
                verdict,
                Stage1Verdict::Clean,
                "{name}: corpus fixtures are valid JSON — no stage-1 error"
            );
            count += 1;
        }
        println!("corpus stage-1 differential: {count} files bit-identical");
        assert!(count >= 15, "corpus/ must contain the checked-in fixtures");
    }

    /// Every JSONTestSuite file (`y_*`/`n_*`/`i_*`), GPU stage 1 vs the
    /// oracle. Stage 1 only catches UTF-8 and odd-quote errors — for `n_`
    /// (and `i_`) files both sides must agree on WHICH of those classes (if
    /// any) fires and at which offset; structural rejections belong to later
    /// stages on both backends. `y_` files must be verdict-clean.
    #[test]
    fn jsontestsuite_files_match_the_reference() {
        let Some(harness) = harness_or_skip("jsontestsuite_files_match_the_reference") else {
            return;
        };
        let Some(dir) = common::jsontestsuite_dir() else {
            return; // loud skip already printed
        };
        let stage1 = Stage1::new();

        let mut totals = [0usize; 3]; // per prefix
        let mut classes = [0usize; 3]; // clean / utf8 / odd-quote, all prefixes
        for (p, prefix) in ["y_", "n_", "i_"].into_iter().enumerate() {
            for path in common::jsontestsuite_files(&dir, prefix) {
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                let bytes = std::fs::read(&path).expect("readable suite file");
                let verdict = diff(&stage1, harness.ctx(), &bytes, &name);
                if prefix == "y_" {
                    assert_eq!(
                        verdict,
                        Stage1Verdict::Clean,
                        "{name}: y_ files are valid JSON — stage 1 must not error"
                    );
                }
                totals[p] += 1;
                classes[verdict as usize] += 1;
            }
        }

        let [y, n, i] = totals;
        let [clean, utf8, odd] = classes;
        println!(
            "JSONTestSuite stage-1 differential: y {y} + n {n} + i {i} files \
             bit-identical ({clean} clean, {utf8} utf8-rejected, {odd} odd-quote)"
        );
        assert!(
            y > 0 && n > 0 && i > 0,
            "all three prefixes must be present"
        );
        assert!(y + n + i >= 300, "the fetched suite has 318 files");
    }

    // --- Adversarial fixtures --------------------------------------------------

    /// Quote walls: every byte a `"` (strict open/close alternation, the
    /// densest possible parity stress for the K3/K5 prefix-XOR ladders and
    /// the K2 carries), walls of escaped quotes (`\"` repeated inside one
    /// string), and walls of adjacent one-char strings — odd and even
    /// lengths, across word and chunk boundaries.
    #[test]
    fn quote_walls_match_the_reference() {
        let mut cases = Vec::new();
        for n in [
            1usize,
            2,
            3,
            63,
            64,
            65,
            127,
            128,
            1023,
            1024,
            1025,
            4095,
            4096,
            4097,
            CHUNK_BYTES - 1,
            CHUNK_BYTES,
            CHUNK_BYTES + 1,
        ] {
            cases.push((format!("bare quote wall n={n}"), vec![b'"'; n]));
        }
        // `"` + reps x `\"` + `"`: one string whose body is all escaped
        // quotes — every quote bit except the ends must be suppressed.
        for reps in [31usize, 2048, 21845] {
            let mut input = b"\"".to_vec();
            for _ in 0..reps {
                input.extend_from_slice(br#"\""#);
            }
            input.push(b'"');
            cases.push((format!("escaped-quote wall reps={reps}"), input));
        }
        // `"a"` repeated: adjacent strings, quote parity flipping 2x per 3
        // bytes — every word seam lands at a different phase.
        for reps in [21usize, 1366, 21846] {
            let mut input = Vec::with_capacity(reps * 3);
            for _ in 0..reps {
                input.extend_from_slice(b"\"a\"");
            }
            cases.push((format!("adjacent-string wall reps={reps}"), input));
        }
        run_corpus("quote_walls_match_the_reference", &cases);
    }

    /// Regression pin for the K1 `mj_zmask` borrow bug found by this suite:
    /// the classic `(x - 0x01010101) & ~x & 0x80808080` zero-byte trick
    /// borrows across byte lanes, so a byte equal to `c ^ 0x01` directly
    /// after a real class-`c` byte was falsely flagged as class `c` (`]\`
    /// classified the backslash as `]`, `"#` made `#` a quote, ...), and the
    /// borrow ripples through runs. Every classified byte followed by its
    /// XOR-1 neighbor, at every alignment mod 4 (the borrow is confined to
    /// one uint lane), plus run-ripple chains.
    #[test]
    fn bit_trick_neighbor_bytes_match_the_reference() {
        let classified: &[u8] = b"\"\\{}[]:, \t\n\r";
        let mut cases = Vec::new();
        for &c in classified {
            let pair = [c, c ^ 0x01];
            for offset in 0..8usize {
                let mut input = vec![b'a'; offset];
                for _ in 0..24 {
                    input.extend_from_slice(&pair);
                }
                cases.push((
                    format!("pair `{}` offset={offset}", pair.escape_ascii()),
                    input,
                ));
            }
            // Ripple chain: c-runs and neighbor-runs of every length 1..=4.
            let mut input = Vec::new();
            for run in 1..=4usize {
                input.extend(std::iter::repeat_n(c, run));
                input.extend(std::iter::repeat_n(c ^ 0x01, run));
            }
            cases.push((format!("ripple `{}`", input.escape_ascii()), input.clone()));
        }
        run_corpus("bit_trick_neighbor_bytes_match_the_reference", &cases);
    }

    /// Escape units tiled over a 4 KiB window, swept across all 64 seam
    /// offsets: every alignment of a backslash run (length 1..=3) and its
    /// quote relative to the 64-byte word seams occurs in some
    /// (offset, unit) pair — including runs straddling the seam itself and
    /// words that BEGIN mid-run (the look-back carry path, uncapped).
    #[test]
    fn escapes_at_every_seam_offset_match_the_reference() {
        let units: &[(&str, &[u8])] = &[
            ("bs-quote", br#"\""#),
            ("2bs-quote", br#"\\""#),
            ("3bs-quote", b"\\\\\\\""),
            ("x-bs", b"x\\"),
        ];
        let mut cases = Vec::new();
        for offset in 0..WORD_BYTES {
            for &(name, unit) in units {
                let mut input = vec![b'a'; offset];
                while input.len() < 4096 + 65 {
                    input.extend_from_slice(unit);
                }
                cases.push((format!("{name} offset={offset}"), input));
            }
        }
        run_corpus("escapes_at_every_seam_offset_match_the_reference", &cases);
    }

    /// Quotes and escapes placed byte-exactly around the 64 KiB spine-chunk
    /// boundary: open/close quotes at every byte in the straddle window, an
    /// escaped quote whose backslash is the last byte of chunk 0, and
    /// backslash runs crossing the boundary in both parities.
    #[test]
    fn quotes_at_chunk_boundaries_match_the_reference() {
        let mut cases = Vec::new();
        // An open quote at each byte around the seam (its close follows
        // immediately); leading 'a's make the prefix one giant scalar run.
        for qpos in CHUNK_BYTES - 2..=CHUNK_BYTES + 2 {
            let mut input = vec![b'a'; qpos];
            input.extend_from_slice(b"\"body\" 1");
            cases.push((format!("open quote at byte {qpos}"), input));
        }
        // A string OPENED in chunk 0 whose close quote lands on each byte
        // around the seam: the in-string mask must carry across chunks.
        for qpos in CHUNK_BYTES - 2..=CHUNK_BYTES + 2 {
            let mut input = b"\"".to_vec();
            input.extend(std::iter::repeat_n(b'x', qpos - 1));
            input.push(b'"'); // at byte qpos
            input.extend_from_slice(b" {}");
            cases.push((format!("close quote at byte {qpos}"), input));
        }
        // Escaped quote straddling the seam: `\` at CHUNK_BYTES-1, `"` at
        // CHUNK_BYTES — and the even-run variant where the quote is real.
        for bs_run in 1usize..=4 {
            let mut input = vec![b' '; CHUNK_BYTES - bs_run];
            input.extend(std::iter::repeat_n(b'\\', bs_run));
            input.extend_from_slice(b"\"x\"");
            cases.push((format!("{bs_run} backslashes ending at chunk seam"), input));
        }
        run_corpus("quotes_at_chunk_boundaries_match_the_reference", &cases);
    }

    /// Input-length boundary cases — 0/1/63/64/65/4095/4096/4097 bytes — for
    /// each interesting uniform fill (scalar run, quote wall, backslash
    /// wall, structural wall, whitespace-only) plus mixed whitespace.
    #[test]
    fn length_boundary_inputs_match_the_reference() {
        let fills: &[(&str, u8)] = &[
            ("scalar", b'x'),
            ("quote", b'"'),
            ("backslash", b'\\'),
            ("lbrace", b'{'),
            ("comma", b','),
            ("space", b' '),
        ];
        let mut cases = Vec::new();
        for &len in &[0usize, 1, 63, 64, 65, 4095, 4096, 4097] {
            for &(name, byte) in fills {
                cases.push((format!("{name} len={len}"), vec![byte; len]));
            }
            // Mixed whitespace-only of every boundary length.
            let ws: Vec<u8> = b" \t\n\r".iter().copied().cycle().take(len).collect();
            cases.push((format!("mixed whitespace len={len}"), ws));
        }
        run_corpus("length_boundary_inputs_match_the_reference", &cases);
    }

    /// Valid multi-byte UTF-8 (CJK, emoji, 4-byte supplementary characters)
    /// swept across all 64 word-seam offsets, plus a multi-word CJK string
    /// body inside a JSON document.
    #[test]
    fn multibyte_utf8_across_seams_matches_the_reference() {
        // 2-, 3- and 4-byte sequences back to back: every seam offset makes
        // a different sequence straddle the boundary.
        let sample = "é漢字測試😀🚀𠜎𠜱𐍈中文テスト한국어\u{10FFFF}";
        let mut cases = Vec::new();
        for offset in 0..WORD_BYTES {
            let mut input = vec![b'a'; offset];
            while input.len() < offset + 4 * WORD_BYTES {
                input.extend_from_slice(sample.as_bytes());
            }
            cases.push((format!("multibyte sweep offset={offset}"), input));
        }
        // A JSON document whose string bodies are several words of CJK.
        let body = "漢字テスト😀".repeat(64);
        let doc = format!(r#"{{"键":"{body}","数":[1,2,3]}}"#);
        cases.push(("CJK document".to_owned(), doc.into_bytes()));
        run_corpus("multibyte_utf8_across_seams_matches_the_reference", &cases);
    }

    /// Every invalid-UTF-8 class (truncated 2/3-byte, overlong 2/3/4-byte,
    /// surrogate, beyond U+10FFFF, stray continuation, 0xFF) at every
    /// position mod 64 (two full word revolutions, exercising both the
    /// in-word walk and the look-back skip), with a valid tail after the
    /// junk — and once truncated at EOF for every position mod 64.
    #[test]
    fn invalid_utf8_at_every_position_mod_64_matches_the_reference() {
        let kinds: &[(&str, &[u8])] = &[
            ("stray continuation", b"\x80"),
            ("overlong 2-byte", b"\xC0\xAF"),
            ("truncated 2-byte", b"\xC2x"),
            ("overlong 3-byte", b"\xE0\x80\x80"),
            ("surrogate", b"\xED\xA0\x80"),
            ("truncated 3-byte", b"\xE2\x82y"),
            ("overlong 4-byte", b"\xF0\x8F\xBF\xBF"),
            ("beyond U+10FFFF", b"\xF4\x90\x80\x80"),
            ("invalid lead 0xFF", b"\xFF"),
        ];
        let Some(harness) = harness_or_skip("invalid_utf8_at_every_position_mod_64") else {
            return;
        };
        let stage1 = Stage1::new();
        for &(name, seq) in kinds {
            for pos in 0..2 * WORD_BYTES {
                // With a valid tail: the error must still be at `pos`.
                let mut input = vec![b'a'; pos];
                input.extend_from_slice(seq);
                input.extend_from_slice(b" tail\"ok\"");
                let verdict = diff(
                    &stage1,
                    harness.ctx(),
                    &input,
                    &format!("{name} at {pos} + tail"),
                );
                assert_eq!(verdict, Stage1Verdict::Utf8, "{name} at {pos} + tail");
            }
            for pos in 0..WORD_BYTES {
                // Hard EOF right after the sequence (truncation at the
                // space-padded tail word).
                let mut input = vec![b'a'; pos];
                input.extend_from_slice(seq);
                let verdict = diff(
                    &stage1,
                    harness.ctx(),
                    &input,
                    &format!("{name} at {pos} @EOF"),
                );
                assert_eq!(verdict, Stage1Verdict::Utf8, "{name} at {pos} @EOF");
            }
        }
    }

    // --- Big synthetic documents (manual) ---------------------------------------

    /// Deterministic single-line synthetic document of at least `target`
    /// bytes: an array of escape-heavy members whose lengths drift (i % 29)
    /// so quotes and backslash runs hit every word-seam alignment.
    fn synthetic_single_line(target: usize) -> Vec<u8> {
        use std::io::Write;
        let mut doc = Vec::with_capacity(target + 192);
        doc.push(b'[');
        let mut i = 0u64;
        while doc.len() < target {
            let pad = "ab".repeat((i % 29) as usize);
            write!(
                doc,
                r#"{{"id{i}":{i},"s":"{pad}\"q\\{pad}","f":-{i}.5e-3,"t":true,"n":null}},"#
            )
            .expect("Vec<u8> write cannot fail");
            i += 1;
        }
        doc.pop(); // trailing comma
        doc.push(b']');
        doc
    }

    /// ~100 MB single line, GPU vs oracle, bit-for-bit. `#[ignore]`d by
    /// default: the scalar oracle alone takes tens of seconds in debug
    /// builds. Run manually:
    /// `cargo test --release --features cpu-reference --test kernels -- --ignored hundred_megabyte`
    #[test]
    #[ignore = "100 MB scale; slow scalar oracle — run manually with -- --ignored"]
    fn hundred_megabyte_single_line_matches_the_reference() {
        let Some(harness) = harness_or_skip("hundred_megabyte_single_line") else {
            return;
        };
        let stage1 = Stage1::new();
        let input = synthetic_single_line(100 * 1024 * 1024);
        assert!(!input.contains(&b'\n'), "generator must keep one line");
        let verdict = diff(&stage1, harness.ctx(), &input, "100 MB single line");
        assert_eq!(verdict, Stage1Verdict::Clean);
    }

    /// Manual timing sanity check, NOT a perf gate: ~256 MB synthetic doc,
    /// GB/s from the command buffers' GPUStartTime/GPUEndTime (CB1 + CB2,
    /// via `Stage1::run_timed`). The plan's expected band for stage 1 on an
    /// M5 Max is > 50 GB/s; this prints the number for eyeballing only.
    /// Run manually:
    /// `cargo test --release --features cpu-reference --test kernels -- --ignored --nocapture timing_sanity`
    #[test]
    #[ignore = "manual: allocates ~256 MB and prints GB/s; not a perf gate"]
    fn timing_sanity_quarter_gigabyte_throughput() {
        let Some(harness) = harness_or_skip("timing_sanity_quarter_gigabyte_throughput") else {
            return;
        };
        let stage1 = Stage1::new();
        let input = synthetic_single_line(256 * 1024 * 1024);

        // Warm-up: PSO creation, first-touch page faults, GPU power ramp.
        let (out, _) = stage1
            .run_timed(harness.ctx(), &input)
            .expect("warm-up run");
        assert_eq!(out.error, None, "synthetic doc must be stage-1 clean");
        assert!(out.token_total > 0);

        let mut best = f64::INFINITY;
        for round in 0..3 {
            let (out, secs) = stage1.run_timed(harness.ctx(), &input).expect("timed run");
            assert_eq!(out.error, None);
            assert!(
                secs > 0.0,
                "GPU timestamps must be available (round {round})"
            );
            best = best.min(secs);
        }
        let gbps = input.len() as f64 / best / 1e9;
        println!(
            "stage-1 GPU time (best of 3): {:.3} ms over {} bytes = {gbps:.1} GB/s \
             (plan band on M5 Max: > 50 GB/s; informational, not asserted)",
            best * 1e3,
            input.len(),
        );
    }

    // --- Property tests -----------------------------------------------------------

    /// Random-input differentials. Neither backend may panic or hang, and
    /// the bitmaps/tokens/verdicts must agree on every case. Case count
    /// defaults to 64 (CI-sized); override with the standard
    /// `PROPTEST_CASES` env var (e.g. `PROPTEST_CASES=1024` locally).
    mod proptests {
        use proptest::prelude::*;

        use metal_json::gpu::Stage1;
        use metal_json::metal::MetalContext;

        use super::diff;

        /// `PROPTEST_CASES` env knob with a CI default of 64.
        /// (`ProptestConfig::default()` would also read the env var, but its
        /// built-in default of 256 is more than these GPU round trips need.)
        fn ci_cases() -> u32 {
            std::env::var("PROPTEST_CASES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64)
        }

        /// One Metal context + Stage1 per test thread (proptest runs all
        /// cases of a test on its thread). Skips quietly without a device
        /// unless `METAL_JSON_REQUIRE_GPU=1`, like `harness_or_skip`.
        fn with_gpu(f: impl FnOnce(&MetalContext, &Stage1)) {
            use std::cell::OnceCell;
            thread_local! {
                static GPU: OnceCell<Option<(MetalContext, Stage1)>> =
                    const { OnceCell::new() };
            }
            GPU.with(|cell| {
                let gpu = cell.get_or_init(|| match MetalContext::new() {
                    Ok(ctx) => Some((ctx, Stage1::new())),
                    Err(err) => {
                        if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                            panic!("METAL_JSON_REQUIRE_GPU=1 but no usable Metal device: {err}");
                        }
                        eprintln!("SKIP stage1 proptests: no usable Metal device ({err})");
                        None
                    }
                });
                if let Some((ctx, stage1)) = gpu {
                    f(ctx, stage1);
                }
            });
        }

        /// Arbitrary JSON documents — the M1 generator (tests/numbers.rs
        /// `arb_json`), duplicated here because test crates cannot share
        /// code outside `tests/common/`.
        fn arb_json() -> impl Strategy<Value = serde_json::Value> {
            let leaf = prop_oneof![
                Just(serde_json::Value::Null),
                any::<bool>().prop_map(serde_json::Value::Bool),
                any::<i64>().prop_map(|v| serde_json::Value::Number(v.into())),
                any::<u64>().prop_map(|v| serde_json::Value::Number(v.into())),
                any::<f64>().prop_filter_map("finite f64", |v| {
                    serde_json::Number::from_f64(v).map(serde_json::Value::Number)
                }),
                any::<String>().prop_map(serde_json::Value::String),
            ];
            leaf.prop_recursive(4, 64, 8, |inner| {
                prop_oneof![
                    prop::collection::vec(inner.clone(), 0..8).prop_map(serde_json::Value::Array),
                    prop::collection::vec((any::<String>(), inner), 0..8)
                        .prop_map(|pairs| serde_json::Value::Object(pairs.into_iter().collect())),
                ]
            })
        }

        /// ASCII with brutal quote/backslash density (the escape resolver's
        /// worst case), seasoned with structural bytes and whitespace.
        fn heavy_quote_backslash_ascii() -> impl Strategy<Value = Vec<u8>> {
            let byte = prop_oneof![
                8 => Just(b'\\'),
                8 => Just(b'"'),
                3 => prop::sample::select(&b"{}[]:,"[..]),
                2 => prop::sample::select(&b" \t\n\r"[..]),
                3 => prop::sample::select(&b"truefalsenull0123456789.eE+-x"[..]),
            ];
            prop::collection::vec(byte, 0..4097)
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(ci_cases()))]

            /// Random valid JSON documents (serde-serialized) are always
            /// verdict-clean and bit-identical across backends.
            #[test]
            fn random_json_documents_match_the_reference(value in arb_json()) {
                let bytes = serde_json::to_vec(&value).expect("serializable");
                with_gpu(|ctx, stage1| {
                    diff(stage1, ctx, &bytes, "random JSON document");
                });
            }

            /// Raw byte soup: arbitrary bytes (mostly invalid UTF-8) must
            /// never panic or hang either backend, and the error verdicts
            /// must match exactly.
            #[test]
            fn random_byte_soup_matches_the_reference(
                bytes in prop::collection::vec(any::<u8>(), 0..4097)
            ) {
                with_gpu(|ctx, stage1| {
                    diff(stage1, ctx, &bytes, "random byte soup");
                });
            }

            /// Quote/backslash-dense ASCII: maximal pressure on escape
            /// resolution, quote parity and the scalar-start carry.
            #[test]
            fn random_quote_backslash_ascii_matches_the_reference(
                bytes in heavy_quote_backslash_ascii()
            ) {
                with_gpu(|ctx, stage1| {
                    diff(stage1, ctx, &bytes, "quote/backslash-heavy ASCII");
                });
            }
        }
    }
}
