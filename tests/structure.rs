//! M3 structure tests on real GPU hardware: the full pipeline through CB3
//! (K1–K9 + K12/K13) vs the scalar oracle, mirroring tests/kernels.rs:
//!
//! - `diff`: whole-pipeline structural differential — GPU stages 1–3 vs
//!   reference stages 1–4 (+ `emit_tape` for the container/root tape words)
//!   on identical inputs, under the **M3 error-class contract**: structural
//!   error classes (UTF-8, odd-quote, Layer-1 syntax, depth/balance,
//!   Layer-2 context, EmptyInput, TrailingContent) must match code AND
//!   offset; scalar-content-error inputs (number grammar, string escapes,
//!   control chars — M4's classes) form an explicit three-way split: the
//!   GPU must ACCEPT through M3, reference stages 1–4 must also accept,
//!   and the FULL reference parse must reject (`Verdict::ScalarPending`).
//!   (The split is a property of the structure-only `Stage3` runner under
//!   test here; the full M4 pipeline collapses it to two-way — see
//!   `jsontestsuite_gpu_backend_matches_the_reference` in
//!   tests/jsontestsuite.rs.)
//!   Accepted inputs compare tape length, the tape_ofs map, pair maps,
//!   child counts and the container/root tape words bit-exactly at the
//!   reference-designated positions (everything else is a zero-word hole).
//! - corpus + JSONTestSuite sweeps (all 318 files) through `diff`;
//! - the plan's **model check**: every token sequence of length ≤ 4 over a
//!   12-symbol alphabet (exhaustive, 22,620 inputs) plus a deterministic
//!   stride sample of lengths 5–6 (~30k inputs) — the Layer-1/Layer-2
//!   completeness net;
//! - adversarial structure fixtures: depth walls at/past the limit (default
//!   and custom `max_depth`), chunk-spanning flat arrays, bracket garbage,
//!   brackets inside strings, deep empty containers, multi-root and
//!   whitespace-padded roots, and (`#[ignore]`) a >0xFFFFFF-element array
//!   pinning the K12 child-count saturation;
//! - property tests: random JSON / byte soup / structural token soup /
//!   mutated JSON (case count defaults to 64; override with
//!   `PROPTEST_CASES`);
//! - (`#[ignore]`) a manual full-pipeline GB/s timing probe on a ~256 MB
//!   synthetic document.
//!
//! Run with `MTL_SHADER_VALIDATION=1` in CI, and once with
//! `--features runtime-shaders` to prove both shader build paths.
#![cfg(feature = "cpu-reference")]

mod common;

use metal_json::gpu::{
    ERR_DEPTH_LIMIT, ERR_EMPTY_INPUT, ERR_INVALID_LITERAL, ERR_MISSING_COLON, ERR_MISSING_COMMA,
    ERR_STRING, ERR_TRAILING_CONTENT, ERR_UNBALANCED, ERR_UNEXPECTED_TOKEN,
    ERR_UNTERMINATED_STRING, ERR_UTF8, Stage3, Stage3Output,
};
use metal_json::metal::MetalContext;
use metal_json::parser::DEFAULT_MAX_DEPTH;
use metal_json::reference::{
    Stage3Output as RefStage3Output, Stage4Output, Token, emit_tape, stage1_classify,
    stage2_tokens, stage3_validate_local, stage4_structure, stage5_scalars, stage6_strings,
};
use metal_json::tape::{make_close, make_final_root, make_open, make_root};
use metal_json::{Error, ParserOptions, SyntaxErrorKind};

/// GPU gating: in environments without a Metal device, skip with a loud
/// message instead of failing — unless `METAL_JSON_REQUIRE_GPU=1` (set in
/// CI) makes a missing device a hard error.
fn ctx_or_skip(test: &str) -> Option<MetalContext> {
    match MetalContext::new() {
        Ok(ctx) => Some(ctx),
        Err(err) => {
            if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                panic!("METAL_JSON_REQUIRE_GPU=1 but no usable Metal device: {err}");
            }
            eprintln!("SKIP {test}: no usable Metal device here ({err})");
            None
        }
    }
}

/// Where (and whether) the pipelines rejected an input — the agreed
/// verdict `diff` returns. The M3 error-class contract is the split
/// between [`Structural`](Verdict::Structural) (and everything above it,
/// caught by the GPU in M3) and [`ScalarPending`](Verdict::ScalarPending)
/// (M4's classes: GPU accepts through M3, full reference rejects).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// Stage 1: invalid UTF-8 (offset parity asserted).
    Utf8,
    /// Stage 1: odd quote count (class parity only — the GPU reports
    /// offset `input_len`, the documented provisional offset).
    OddQuotes,
    /// CPU verdict at sync 1/2: no tokens at all.
    Empty,
    /// CB2 / reference stage 3: a Layer-1 syntax rule (offset + code
    /// parity asserted).
    Layer1,
    /// CB3 / reference stage 4: depth, balance, Layer-2 context or
    /// trailing content (offset + code parity asserted).
    Structural,
    /// THE THREE-WAY SPLIT: the GPU accepted through M3 AND reference
    /// stages 1–4 accepted (structure outputs bit-identical) AND the full
    /// reference parse rejects with an M4 scalar-content class.
    ScalarPending,
    /// Everything accepts: structure outputs and the designated tape words
    /// are bit-identical, and the full reference parse succeeds.
    Clean,
}

/// The GPU code for each Layer-1 `SyntaxErrorKind` (the same mapping the
/// in-module stage-2/3 suites pin).
fn layer1_code(kind: SyntaxErrorKind) -> u32 {
    match kind {
        SyntaxErrorKind::MissingColon => ERR_MISSING_COLON,
        SyntaxErrorKind::MissingComma => ERR_MISSING_COMMA,
        SyntaxErrorKind::UnexpectedToken => ERR_UNEXPECTED_TOKEN,
        SyntaxErrorKind::InvalidLiteral => ERR_INVALID_LITERAL,
        SyntaxErrorKind::UnbalancedBrackets => ERR_UNBALANCED,
        SyntaxErrorKind::UnterminatedString => ERR_UNTERMINATED_STRING,
        SyntaxErrorKind::EmptyInput => ERR_EMPTY_INPUT,
        other => panic!("reference stage 3 cannot produce {other:?}"),
    }
}

/// The GPU `(offset, code)` for a reference stage-4 error.
fn stage4_code(err: &Error) -> (u64, u32) {
    match err {
        Error::Syntax { offset, kind } => {
            let code = match kind {
                SyntaxErrorKind::UnbalancedBrackets => ERR_UNBALANCED,
                SyntaxErrorKind::UnexpectedToken => ERR_UNEXPECTED_TOKEN,
                SyntaxErrorKind::MissingColon => ERR_MISSING_COLON,
                other => panic!("reference stage 4 cannot produce {other:?}"),
            };
            (*offset, code)
        }
        Error::DepthLimit { offset, .. } => (*offset, ERR_DEPTH_LIMIT),
        Error::TrailingContent { offset } => (*offset, ERR_TRAILING_CONTENT),
        other => panic!("reference stage 4 cannot produce {other:?}"),
    }
}

/// The rejection contract on the GPU side: a rejected input produces no
/// stage-3 outputs (including the tape).
fn assert_stage3_empty(got: &Stage3Output, label: &str) {
    assert!(got.depths.is_empty(), "{label}: no depths on rejection");
    assert!(got.sorted_by_depth.is_empty(), "{label}: no sort output");
    assert!(got.match_index.is_empty(), "{label}: no pair map");
    assert!(got.context_opener.is_empty(), "{label}: no context");
    assert!(got.child_counts.is_empty(), "{label}: no child counts");
    assert!(got.tape.is_empty(), "{label}: no tape on rejection");
}

/// Inputs at most this long additionally re-run the FULL reference parse
/// (`metal_json::reference::parse`) as a tripwire that the staged oracle
/// calls in `diff` agree with the assembled pipeline. Bounded so the
/// multi-megabyte adversarial fixtures don't pay the whole scalar pipeline
/// twice; the staged calls already decide the verdict.
const FULL_PARSE_CAP: usize = 64 * 1024;

/// Run both backends on `input` with the same `max_depth` and require
/// agreement per the M3 error-class contract; returns the agreed verdict.
///
/// Rejected inputs compare the packed `(offset, code)` verdict (odd quotes:
/// class parity only, the documented exception) plus the rejection
/// contract. Inputs that pass reference stages 1–4 compare every structure
/// vector, the skeleton, the tape_ofs map and the tape (container/root
/// words at the reference-designated positions, zero-word holes everywhere
/// else) bit-for-bit, then split on the scalar stages: clean inputs must
/// match `emit_tape`'s words and pass the full reference parse;
/// scalar-content rejects (M4's classes) must fail the full reference
/// parse — the three-way split.
fn diff(stage3: &Stage3, ctx: &MetalContext, input: &[u8], max_depth: u32, label: &str) -> Verdict {
    let got = stage3
        .run_with_max_depth(ctx, input, max_depth)
        .unwrap_or_else(|e| panic!("{label}: GPU stage 3 failed: {e}"));

    // Reference stage 1: UTF-8.
    let bitmaps = match stage1_classify(input) {
        Ok(bitmaps) => bitmaps,
        Err(Error::Utf8 { offset }) => {
            assert_eq!(
                got.error_offset_code(),
                Some((offset, ERR_UTF8)),
                "{label}: UTF-8 verdict"
            );
            assert_stage3_empty(&got, label);
            assert!(
                got.stage2.skeleton_byte.is_empty(),
                "{label}: stage-1 rejection leaves no stage-2 outputs"
            );
            return Verdict::Utf8;
        }
        Err(other) => panic!("{label}: unexpected reference stage-1 error {other:?}"),
    };

    // Odd quote count: rejected in CB1 at offset input_len (documented
    // provisional offset — class parity only; the reference reports the
    // open quote from its stage 3).
    let quote_total: u64 = bitmaps
        .quote_real
        .iter()
        .map(|w| u64::from(w.count_ones()))
        .sum();
    if quote_total % 2 == 1 {
        assert_eq!(
            got.error_offset_code(),
            Some((input.len() as u64, ERR_STRING)),
            "{label}: odd-quote verdict"
        );
        assert_stage3_empty(&got, label);
        // Verdict parity: the reference must also reject — an odd quote
        // count leaves a dangling QuoteOpen no rule table can accept. The
        // CLASS may legitimately differ on multi-error inputs (an earlier
        // token-order violation can outrank the unterminated string in the
        // reference's token-order iteration; the documented error policy in
        // src/reference/mod.rs: backends may disagree about WHICH error,
        // never about WHETHER parsing fails).
        let tokens = stage2_tokens(&bitmaps, input);
        assert!(
            stage3_validate_local(&tokens, input).is_err(),
            "{label}: reference must also reject an odd-quote input"
        );
        return Verdict::OddQuotes;
    }

    // Reference stages 2–3: tokens + Layer 1.
    let tokens = stage2_tokens(&bitmaps, input);
    let s3 = match stage3_validate_local(&tokens, input) {
        Err(Error::Syntax { offset, kind }) => {
            assert_eq!(
                got.error_offset_code(),
                Some((offset, layer1_code(kind))),
                "{label}: Layer-1 verdict for reference {kind:?}"
            );
            assert_stage3_empty(&got, label);
            // CB2's rejection contract: K6b never ran.
            assert!(
                got.stage2.tape_ofs.is_empty(),
                "{label}: no tape_ofs on a Layer-1 rejection"
            );
            assert!(
                got.stage2.skeleton_byte.is_empty(),
                "{label}: no skeleton on a Layer-1 rejection"
            );
            return if kind == SyntaxErrorKind::EmptyInput {
                Verdict::Empty
            } else {
                Verdict::Layer1
            };
        }
        Err(other) => panic!("{label}: unexpected reference stage-3 error {other:?}"),
        Ok(s3) => s3,
    };

    // The skeleton CB2b produced (field-for-field the reference records).
    assert_eq!(
        got.stage2.skeleton_byte.len(),
        s3.skeleton.len(),
        "{label}: skeleton length"
    );
    for (si, rec) in s3.skeleton.iter().enumerate() {
        assert_eq!(
            (
                got.stage2.skeleton_token_index[si],
                got.stage2.skeleton_pos[si],
                got.stage2.skeleton_byte[si],
            ),
            (rec.token_index, rec.pos, rec.byte),
            "{label}: skeleton record {si}"
        );
    }

    // Reference stage 4 — the CB3 spec.
    let want = match stage4_structure(&s3.skeleton, max_depth) {
        Err(err) => {
            let (offset, code) = stage4_code(&err);
            assert_eq!(
                got.error_offset_code(),
                Some((offset, code)),
                "{label}: structural verdict for reference {err:?}"
            );
            // Rejection contract: stage-2 outputs kept (asserted above),
            // stage-3 outputs never produced.
            assert_stage3_empty(&got, label);
            return Verdict::Structural;
        }
        Ok(want) => want,
    };

    // Accepted through stage 4: every structure vector bit-for-bit.
    assert_eq!(got.error, None, "{label}: spurious GPU error");
    assert_eq!(got.depths, want.depths, "{label}: depths");
    assert_eq!(
        got.sorted_by_depth, want.sorted_by_depth,
        "{label}: sorted_by_depth"
    );
    assert_eq!(got.match_index, want.match_index, "{label}: match_index");
    assert_eq!(
        got.context_opener, want.context_opener,
        "{label}: context_opener"
    );
    assert_eq!(got.child_counts, want.child_counts, "{label}: child_counts");

    // The M3 tape: tape_ofs map, length, container/root words, holes.
    let tape_pos = diff_tape(&got, &tokens, &s3, &want, label);

    // THE THREE-WAY SPLIT: structure accepted everywhere; now the scalar
    // stages (M4's classes) decide Clean vs ScalarPending.
    let opts = {
        let mut opts = ParserOptions::default();
        opts.max_depth = max_depth;
        opts
    };
    match (stage5_scalars(&tokens, input), stage6_strings(&tokens, input)) {
        (Ok(scalars), Ok(strings)) => {
            // Scalar content clean: the GPU words must equal the REAL
            // emitter's words at every designated position (and the holes
            // are exactly the emitter's scalar/string slots).
            let (ref_tape, _) = emit_tape(&tokens, &s3, &want, &scalars, &strings);
            assert_eq!(ref_tape.len(), got.tape.len(), "{label}: emit length");
            for (t, tok) in tokens.iter().enumerate() {
                use metal_json::reference::TokenKind;
                if matches!(
                    tok.kind,
                    TokenKind::LBrace | TokenKind::LBracket | TokenKind::RBrace | TokenKind::RBracket
                ) {
                    let pos = tape_pos[t] as usize;
                    assert_eq!(
                        got.tape[pos],
                        ref_tape.as_words()[pos],
                        "{label}: container word vs reference emit at tape[{pos}]"
                    );
                }
            }
            assert_eq!(
                got.tape[0],
                ref_tape.as_words()[0],
                "{label}: root prologue vs reference emit"
            );
            assert_eq!(
                got.tape[got.tape.len() - 1],
                ref_tape.as_words()[ref_tape.len() - 1],
                "{label}: final root vs reference emit"
            );
            if input.len() <= FULL_PARSE_CAP {
                metal_json::reference::parse(input, &opts).unwrap_or_else(|e| {
                    panic!("{label}: full reference parse must accept a clean input: {e}")
                });
            }
            Verdict::Clean
        }
        (scalars, strings) => {
            // M3/M4 split: the GPU accepted through M3 (asserted above) and
            // reference stages 1–4 accepted; the failing class must be a
            // scalar-content one (the only classes M4 still owns)...
            let err = match (scalars, strings) {
                (Err(e), _) | (Ok(_), Err(e)) => e,
                (Ok(_), Ok(_)) => unreachable!("handled by the first match arm"),
            };
            match &err {
                Error::Syntax { kind, .. } => assert!(
                    matches!(
                        kind,
                        SyntaxErrorKind::InvalidNumber
                            | SyntaxErrorKind::InvalidStringEscape
                            | SyntaxErrorKind::ControlCharacterInString
                    ),
                    "{label}: stage-5/6 rejection {kind:?} is not a scalar-content class \
                     (it would belong to M3's contract)"
                ),
                other => panic!("{label}: unexpected scalar-stage error {other:?}"),
            }
            // ... and the full reference parse must reject the input.
            if input.len() <= FULL_PARSE_CAP {
                assert!(
                    metal_json::reference::parse(input, &opts).is_err(),
                    "{label}: full reference parse must reject a scalar-content error"
                );
            }
            Verdict::ScalarPending
        }
    }
}

/// The M3 tape oracle for an input that passed reference stages 1–4:
/// tape_ofs == 1 + the exclusive footprint prefix sum, tape length ==
/// footprint total + 2, every container/root position holds the bit-exact
/// word rebuilt from the reference stage-3/4 outputs (open: one-past-close
/// plus saturated count; close: open index), and every other position is a
/// zero-word HOLE (the documented M3 convention). Returns the per-token
/// tape positions for the emit cross-check.
fn diff_tape(
    got: &Stage3Output,
    tokens: &[Token],
    s3: &RefStage3Output,
    want: &Stage4Output,
    label: &str,
) -> Vec<u32> {
    let mut tape_pos = vec![0u32; tokens.len()];
    let mut running = 1u32;
    for (t, fp) in s3.footprints.iter().enumerate() {
        tape_pos[t] = running;
        running += fp;
    }
    assert_eq!(got.stage2.tape_ofs, tape_pos, "{label}: tape_ofs map");

    let len = running as usize + 1;
    assert_eq!(got.tape.len(), len, "{label}: tape length");

    let mut expected: Vec<Option<u64>> = vec![None; len];
    expected[0] = Some(make_root(u64::from(running)));
    expected[len - 1] = Some(make_final_root());
    for (si, rec) in s3.skeleton.iter().enumerate() {
        if rec.byte == b':' || rec.byte == b',' {
            continue;
        }
        let partner_si = want.match_index[si] as usize;
        let partner_pos = tape_pos[s3.skeleton[partner_si].token_index as usize];
        let own = tape_pos[rec.token_index as usize] as usize;
        let word = if rec.byte == b'{' || rec.byte == b'[' {
            // make_open saturates the count at 24 bits exactly like K12.
            make_open(rec.byte, partner_pos + 1, want.child_counts[si])
        } else {
            make_close(rec.byte, partner_pos)
        };
        expected[own] = Some(word);
    }
    for (i, want_word) in expected.iter().enumerate() {
        match want_word {
            Some(word) => assert_eq!(got.tape[i], *word, "{label}: tape[{i}]"),
            None => assert_eq!(got.tape[i], 0, "{label}: hole at tape[{i}]"),
        }
    }
    tape_pos
}

/// Tallies of the verdicts a sweep produced, for the printed summaries and
/// the contract assertions.
#[derive(Debug, Default)]
struct VerdictCounts {
    utf8: usize,
    odd_quotes: usize,
    empty: usize,
    layer1: usize,
    structural: usize,
    scalar_pending: usize,
    clean: usize,
}

impl VerdictCounts {
    fn record(&mut self, verdict: Verdict) {
        match verdict {
            Verdict::Utf8 => self.utf8 += 1,
            Verdict::OddQuotes => self.odd_quotes += 1,
            Verdict::Empty => self.empty += 1,
            Verdict::Layer1 => self.layer1 += 1,
            Verdict::Structural => self.structural += 1,
            Verdict::ScalarPending => self.scalar_pending += 1,
            Verdict::Clean => self.clean += 1,
        }
    }

    fn total(&self) -> usize {
        self.utf8
            + self.odd_quotes
            + self.empty
            + self.layer1
            + self.structural
            + self.scalar_pending
            + self.clean
    }
}

// --- 1. Whole-pipeline structural diff: corpus + JSONTestSuite ------------------

/// Every checked-in corpus fixture is valid JSON: the whole pipeline must
/// agree with the reference on every intermediate and the tape words.
#[test]
fn corpus_files_match_the_reference_through_stage4() {
    let Some(ctx) = ctx_or_skip("corpus_files_match_the_reference_through_stage4") else {
        return;
    };
    let stage3 = Stage3::new();
    let mut count = 0usize;
    for path in common::corpus_files() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let bytes = std::fs::read(&path).expect("readable corpus fixture");
        let verdict = diff(&stage3, &ctx, &bytes, DEFAULT_MAX_DEPTH, &name);
        assert_eq!(
            verdict,
            Verdict::Clean,
            "{name}: corpus fixtures are valid JSON"
        );
        count += 1;
    }
    println!("corpus M3 structural differential: {count} files bit-identical");
    assert!(count >= 15, "corpus/ must contain the checked-in fixtures");
}

/// Every JSONTestSuite file (`y_*`/`n_*`/`i_*`), GPU stages 1–3 vs
/// reference stages 1–4 + emit: the M3 error-class contract pinned across
/// the whole suite. `y_` files must be fully Clean; `n_` files must reject
/// somewhere (M3 structurally, or M4 via the explicit ScalarPending
/// split); `i_` files may land anywhere — but every file must agree
/// between the backends.
#[test]
fn jsontestsuite_files_match_the_reference_through_stage4() {
    let Some(ctx) = ctx_or_skip("jsontestsuite_files_match_the_reference_through_stage4") else {
        return;
    };
    let Some(dir) = common::jsontestsuite_dir() else {
        return; // loud skip already printed
    };
    let stage3 = Stage3::new();

    let mut totals = [0usize; 3];
    let mut counts = VerdictCounts::default();
    for (p, prefix) in ["y_", "n_", "i_"].into_iter().enumerate() {
        for path in common::jsontestsuite_files(&dir, prefix) {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let bytes = std::fs::read(&path).expect("readable suite file");
            let verdict = diff(&stage3, &ctx, &bytes, DEFAULT_MAX_DEPTH, &name);
            match prefix {
                "y_" => assert_eq!(
                    verdict,
                    Verdict::Clean,
                    "{name}: y_ files are valid JSON — the pipeline must accept"
                ),
                "n_" => assert_ne!(
                    verdict,
                    Verdict::Clean,
                    "{name}: n_ files are invalid JSON — some stage must reject"
                ),
                _ => {}
            }
            totals[p] += 1;
            counts.record(verdict);
        }
    }

    let [y, n, i] = totals;
    println!(
        "JSONTestSuite M3 structural differential: y {y} + n {n} + i {i} files agree \
         ({clean} clean, {scalar} scalar-pending [M4's classes], {layer1} Layer-1, \
         {structural} structural, {empty} empty, {utf8} utf8, {odd} odd-quote)",
        clean = counts.clean,
        scalar = counts.scalar_pending,
        layer1 = counts.layer1,
        structural = counts.structural,
        empty = counts.empty,
        utf8 = counts.utf8,
        odd = counts.odd_quotes,
    );
    assert!(y > 0 && n > 0 && i > 0, "all three prefixes must be present");
    assert!(y + n + i >= 300, "the fetched suite has 318 files");
    // The error-class split must actually appear in the suite: structural
    // rejects (M3's job), scalar-content rejects (M4's job, the three-way
    // split), and clean accepts.
    assert!(counts.structural > 0, "suite has structural n_ cases");
    assert!(
        counts.scalar_pending > 0,
        "suite has scalar-content n_ cases (the M4 split)"
    );
    assert!(counts.clean >= y, "every y_ file is clean");
}

// --- 2. The model check: exhaustive short token sequences -----------------------

/// The 12-symbol token alphabet of the plan's model check: each symbol is
/// a complete token spelling (strings as `"x"`, numbers as `1`, literals
/// in full, `x` as the bad-scalar-byte probe).
const SYMBOLS: [&[u8]; 12] = [
    b"{",
    b"}",
    b"[",
    b"]",
    b":",
    b",",
    b"\"x\"",
    b"1",
    b"true",
    b"false",
    b"null",
    b"x",
];

/// Decode sequence index `k` of length `len` into its base-12 digits and
/// render the space-joined input (spaces keep adjacent symbols distinct
/// tokens, so the input realizes exactly the intended token sequence).
fn sequence_input(mut k: usize, len: usize) -> Vec<u8> {
    let mut input = Vec::with_capacity(len * 6);
    for i in 0..len {
        if i > 0 {
            input.push(b' ');
        }
        input.extend_from_slice(SYMBOLS[k % SYMBOLS.len()]);
        k /= SYMBOLS.len();
    }
    input
}

fn run_sequences(test: &str, len: usize, stride: usize) -> Option<VerdictCounts> {
    let ctx = ctx_or_skip(test)?;
    let stage3 = Stage3::new();
    let mut counts = VerdictCounts::default();
    let total = SYMBOLS.len().pow(u32::try_from(len).unwrap());
    let mut k = 0usize;
    while k < total {
        let input = sequence_input(k, len);
        let verdict = diff(
            &stage3,
            &ctx,
            &input,
            DEFAULT_MAX_DEPTH,
            &format!("len={len} k={k} {:?}", String::from_utf8_lossy(&input)),
        );
        counts.record(verdict);
        k += stride;
    }
    Some(counts)
}

/// EVERY token sequence of length 1..=4 over the alphabet (12 + 144 +
/// 1728 + 20736 = 22,620 inputs), GPU verdict vs reference verdict on
/// every one — the Layer-1/Layer-2 completeness net the plan requires
/// before trusting the K6/K9 rule tables.
#[test]
fn exhaustive_token_sequences_up_to_length_4() {
    let mut grand = VerdictCounts::default();
    for len in 1..=4usize {
        let Some(counts) = run_sequences("exhaustive_token_sequences_up_to_length_4", len, 1)
        else {
            return;
        };
        assert_eq!(counts.total(), SYMBOLS.len().pow(u32::try_from(len).unwrap()));
        assert_eq!(counts.utf8, 0, "the alphabet is pure ASCII");
        assert_eq!(counts.odd_quotes, 0, "strings are complete `\"x\"` units");
        assert_eq!(counts.empty, 0, "every sequence has at least one token");
        grand.utf8 += counts.utf8;
        grand.layer1 += counts.layer1;
        grand.structural += counts.structural;
        grand.scalar_pending += counts.scalar_pending;
        grand.clean += counts.clean;
    }
    println!(
        "exhaustive token sequences (len 1..=4, 22620 inputs): {} Layer-1, {} structural, \
         {} scalar-pending, {} clean — all verdicts agree",
        grand.layer1, grand.structural, grand.scalar_pending, grand.clean
    );
    // Sanity that the net actually has all the interesting buckets.
    assert!(grand.layer1 > 0 && grand.structural > 0 && grand.clean > 0);
}

/// A deterministic stride sample of the length-5 and length-6 sequence
/// spaces (~15k each, ~30k total; strides coprime to 12 so every digit
/// position cycles through the whole alphabet).
#[test]
fn sampled_token_sequences_length_5_and_6() {
    let Some(counts5) = run_sequences("sampled_token_sequences_length_5_and_6", 5, 17) else {
        return;
    };
    let Some(counts6) = run_sequences("sampled_token_sequences_length_5_and_6", 6, 199) else {
        return;
    };
    println!(
        "sampled token sequences: len 5 = {} inputs ({} clean), len 6 = {} inputs ({} clean) \
         — all verdicts agree",
        counts5.total(),
        counts5.clean,
        counts6.total(),
        counts6.clean
    );
    assert!(counts5.total() > 14_000 && counts6.total() > 14_000);
    // Clean length-5 sequences exist in the stride-17 sample; valid
    // length-6 sequences are so rare that the stride-199 sample holds none
    // — assert the rejection buckets are exercised instead.
    assert!(counts5.clean > 0);
    assert!(counts6.layer1 > 0 && counts6.structural > 0);
}

// --- 3. Adversarial structure fixtures -------------------------------------------

/// Bracket/object nesting walls at exactly `max_depth` and one past it,
/// at the simdjson-parity default (1024/1025) and at small custom limits
/// on both sort paths (1-pass ≤ 32, 2-pass above) — plus a 20,000-deep
/// wall (depth ≈ n/2) whose rejection offset is the 1025th open bracket.
#[test]
fn depth_walls_at_and_past_the_limit() {
    let Some(ctx) = ctx_or_skip("depth_walls_at_and_past_the_limit") else {
        return;
    };
    let stage3 = Stage3::new();

    let array_nest = |depth: usize| {
        let mut v = b"[".repeat(depth);
        v.extend_from_slice(&b"]".repeat(depth));
        v
    };
    let object_nest = |depth: usize| {
        let mut v = Vec::with_capacity(depth * 6 + 1);
        for _ in 0..depth {
            v.extend_from_slice(br#"{"a":"#);
        }
        v.push(b'1');
        v.extend_from_slice(&b"}".repeat(depth));
        v
    };

    // Default limit: 1024 exactly passes, 1025 fails, both shapes.
    for (name, nest) in [
        ("array", &array_nest as &dyn Fn(usize) -> Vec<u8>),
        ("object", &object_nest),
    ] {
        let verdict = diff(
            &stage3,
            &ctx,
            &nest(DEFAULT_MAX_DEPTH as usize),
            DEFAULT_MAX_DEPTH,
            &format!("{name} wall at 1024"),
        );
        assert_eq!(verdict, Verdict::Clean, "{name} wall at the limit");
        let verdict = diff(
            &stage3,
            &ctx,
            &nest(DEFAULT_MAX_DEPTH as usize + 1),
            DEFAULT_MAX_DEPTH,
            &format!("{name} wall at 1025"),
        );
        assert_eq!(verdict, Verdict::Structural, "{name} wall one past");
    }

    // A 20,000-deep wall (n/2 depth for a 40 KB input): the GPU must
    // reject at the 1025th open bracket exactly like the reference.
    let verdict = diff(
        &stage3,
        &ctx,
        &array_nest(20_000),
        DEFAULT_MAX_DEPTH,
        "20000-deep array wall",
    );
    assert_eq!(verdict, Verdict::Structural);
    let verdict = diff(
        &stage3,
        &ctx,
        &object_nest(20_000),
        DEFAULT_MAX_DEPTH,
        "20000-deep object wall",
    );
    assert_eq!(verdict, Verdict::Structural);

    // Custom limits (the plumbed ParserOptions::max_depth equivalent), on
    // both the 1-pass (≤ 32) and 2-pass sort paths, at the boundary and
    // one past it.
    for max_depth in [1u32, 2, 3, 31, 32, 33, 100] {
        for shape in [&array_nest as &dyn Fn(usize) -> Vec<u8>, &object_nest] {
            let at = diff(
                &stage3,
                &ctx,
                &shape(max_depth as usize),
                max_depth,
                &format!("wall at custom limit {max_depth}"),
            );
            assert_eq!(at, Verdict::Clean, "depth == max_depth {max_depth} passes");
            let past = diff(
                &stage3,
                &ctx,
                &shape(max_depth as usize + 1),
                max_depth,
                &format!("wall past custom limit {max_depth}"),
            );
            assert_eq!(past, Verdict::Structural, "depth == {max_depth}+1 fails");
        }
    }
}

/// Overflow depths (> max_depth) share the clamped key_max sort bucket
/// with the legal max_depth group but must stay INERT in K9 (the
/// `mj_sort_key` contract in shaders/common.h): at max_depth=1 the `[[1]`
/// close must NOT adjacent-pair with the inner (overflow) open — that
/// would suppress the outer open's leftover error and report DepthLimit@1
/// where the reference's first error is UnbalancedBrackets@0. Pins the
/// reference's own verdict explicitly, then sweeps max_depth ∈ {1, 2, 3}
/// over the nesting shapes asserting GPU verdict + code + offset ==
/// reference via `diff`.
#[test]
fn overflow_depth_groups_match_the_reference() {
    let Some(ctx) = ctx_or_skip("overflow_depth_groups_match_the_reference") else {
        return;
    };
    let stage3 = Stage3::new();

    // The reference is the spec: verify it really puts the unclosed-open
    // error (offset 0) ahead of the DepthLimit (offset 1) for `[[1]`.
    let input: &[u8] = b"[[1]";
    let tokens = stage2_tokens(&stage1_classify(input).unwrap(), input);
    let s3 = stage3_validate_local(&tokens, input).expect("[[1] passes Layer 1");
    match stage4_structure(&s3.skeleton, 1) {
        Err(Error::Syntax {
            offset: 0,
            kind: SyntaxErrorKind::UnbalancedBrackets,
        }) => {}
        other => panic!("reference verdict for max_depth=1 [[1] moved: {other:?}"),
    }
    // ... and the GPU agrees exactly.
    let got = stage3.run_with_max_depth(&ctx, input, 1).unwrap();
    assert_eq!(
        got.error_offset_code(),
        Some((0, ERR_UNBALANCED)),
        "max_depth=1 [[1]: the unclosed OUTER open wins, not DepthLimit@1"
    );

    for max_depth in [1u32, 2, 3] {
        for input in [&b"[[1]"[..], b"[[1]]", b"[[[1]]]", br#"{"a":[1]}"#] {
            let verdict = diff(
                &stage3,
                &ctx,
                input,
                max_depth,
                &format!(
                    "overflow sweep max_depth={max_depth} {:?}",
                    String::from_utf8_lossy(input)
                ),
            );
            // Sanity on the sweep's shape: `[[1]` always rejects
            // structurally; the rest reject iff the nesting exceeds the
            // limit (`diff` already asserted offset/code parity).
            let depth = input.iter().filter(|&&b| b == b'[').count() as u32
                + u32::from(input[0] == b'{');
            let want = if input == &b"[[1]"[..] || depth > max_depth {
                Verdict::Structural
            } else {
                Verdict::Clean
            };
            assert_eq!(
                verdict,
                want,
                "max_depth={max_depth} {:?}",
                String::from_utf8_lossy(input)
            );
        }
    }
}

/// A 100,000-element flat array: the skeleton (1 + 99,999 commas + 1)
/// spans ~98 sort/scan chunks at depth 1 — the chunk-carry torture for
/// the depth scan, the sort and the segmented context fill.
#[test]
fn flat_100k_element_array_spans_chunks() {
    let Some(ctx) = ctx_or_skip("flat_100k_element_array_spans_chunks") else {
        return;
    };
    let stage3 = Stage3::new();
    let n = 100_000usize;
    let mut input = Vec::with_capacity(2 * n + 1);
    input.push(b'[');
    for i in 0..n {
        if i > 0 {
            input.push(b',');
        }
        input.push(b'1');
    }
    input.push(b']');
    let verdict = diff(&stage3, &ctx, &input, DEFAULT_MAX_DEPTH, "flat 100k array");
    assert_eq!(verdict, Verdict::Clean);

    // The same array with objects sprinkled so depth groups interleave
    // across chunk seams.
    let mut input = Vec::with_capacity(4 * n);
    input.push(b'[');
    for i in 0..n {
        if i > 0 {
            input.push(b',');
        }
        if i % 7 == 0 {
            input.extend_from_slice(br#"{"k":1}"#);
        } else {
            input.push(b'1');
        }
    }
    input.push(b']');
    let verdict = diff(
        &stage3,
        &ctx,
        &input,
        DEFAULT_MAX_DEPTH,
        "flat 100k array with objects",
    );
    assert_eq!(verdict, Verdict::Clean);
}

/// More than 0xFFFFFF direct children: K12 saturates the open word's
/// 24-bit count field (the Rust `make_open` rebuild saturates identically,
/// so `diff`'s tape oracle pins the exact word) while the `child_counts`
/// vector keeps the true count. ~17M elements ≈ 34 MB; `#[ignore]`d for time
/// (the scalar oracle dominates) — run manually:
/// `cargo test --release --features cpu-reference --test structure -- --ignored count_saturation`
#[test]
#[ignore = "~34 MB input, slow scalar oracle — run manually with --release -- --ignored"]
fn count_saturation_past_24_bits_matches_the_reference() {
    let Some(ctx) = ctx_or_skip("count_saturation_past_24_bits_matches_the_reference") else {
        return;
    };
    let stage3 = Stage3::new();
    let n = 0x100_0001usize; // 16,777,217 = 0xFFFFFF + 2 children
    let mut input = Vec::with_capacity(2 * n + 1);
    input.push(b'[');
    for i in 0..n {
        if i > 0 {
            input.push(b',');
        }
        input.push(b'1');
    }
    input.push(b']');
    let verdict = diff(
        &stage3,
        &ctx,
        &input,
        DEFAULT_MAX_DEPTH,
        "17M-element saturating array",
    );
    assert_eq!(verdict, Verdict::Clean);

    // Belt and braces: the open word on the GPU tape really is saturated.
    let got = stage3.run(&ctx, &input).unwrap();
    assert_eq!(got.error, None);
    assert_eq!(got.child_counts[0], n as u32, "true count in the vector");
    assert_eq!(
        got.tape[1],
        make_open(b'[', u32::try_from(got.tape.len() - 1).unwrap(), n as u32),
        "open word saturates via make_open parity"
    );
}

/// Alternating `][` garbage and friends: every variant must reject with
/// the reference's exact class and offset (Layer 1 for starter bans,
/// CB3 for underflow/leftovers).
#[test]
fn alternating_bracket_garbage_matches_the_reference() {
    let Some(ctx) = ctx_or_skip("alternating_bracket_garbage_matches_the_reference") else {
        return;
    };
    let stage3 = Stage3::new();
    let mut cases: Vec<Vec<u8>> = Vec::new();
    for n in [1usize, 2, 3, 64, 1023, 1024, 1025, 5000] {
        cases.push(b"][".repeat(n));
        if n >= 2 {
            cases.push(b"[]".repeat(n)); // `[][]...` — MissingComma at 2
        }
        cases.push(b"}{".repeat(n));
        cases.push(b"]".repeat(n));
        cases.push(b"[".repeat(n));
        let mut v = b"[]".repeat(n);
        v.extend_from_slice(&b"]".repeat(n)); // balanced prefix, then closes
        cases.push(v);
    }
    // Underflow buried mid-document (the depth scan's negative excursion).
    cases.push(b"[[]]]]][[".to_vec());
    cases.push(br#"[1,2]],3"#.to_vec());
    for input in &cases {
        let verdict = diff(
            &stage3,
            &ctx,
            input,
            DEFAULT_MAX_DEPTH,
            &format!("{:?}", String::from_utf8_lossy(&input[..input.len().min(32)])),
        );
        assert_ne!(verdict, Verdict::Clean, "garbage must reject");
    }
}

/// Brackets, colons and commas inside string literals must NOT become
/// skeleton elements or pair with real brackets.
#[test]
fn brackets_inside_strings_do_not_pair() {
    let Some(ctx) = ctx_or_skip("brackets_inside_strings_do_not_pair") else {
        return;
    };
    let stage3 = Stage3::new();
    let cases: &[&[u8]] = &[
        br#"["{", "}"]"#,
        br#"{"]": "["}"#,
        br#"["a}b", {"k[": "]v"}]"#,
        br#"{"{[: ,]}": [1, "][", {"}": "{"}]}"#,
        br#""[[[[[[[[""#,
        br#"["\"[", "]\""]"#,
    ];
    for &input in cases {
        let verdict = diff(
            &stage3,
            &ctx,
            input,
            DEFAULT_MAX_DEPTH,
            &format!("{:?}", String::from_utf8_lossy(input)),
        );
        assert_eq!(
            verdict,
            Verdict::Clean,
            "{:?}: in-string brackets are content, not structure",
            String::from_utf8_lossy(input)
        );
    }

    // A string body of brackets spanning several 64 KiB chunks: the
    // in-string mask must suppress every one of them.
    let mut input = b"[\"".to_vec();
    let unit = b"[]{}:,";
    while input.len() < 200_000 {
        input.extend_from_slice(unit);
    }
    input.extend_from_slice(b"\",[1]]");
    let verdict = diff(
        &stage3,
        &ctx,
        &input,
        DEFAULT_MAX_DEPTH,
        "multi-chunk bracket-soup string",
    );
    assert_eq!(verdict, Verdict::Clean);
}

/// Empty containers nested deep, in both shapes, at several depths
/// including the 1024 boundary (the close token immediately follows the
/// open token — the `[]` child-count special case — at every depth).
#[test]
fn empty_containers_nested_deep_match_the_reference() {
    let Some(ctx) = ctx_or_skip("empty_containers_nested_deep_match_the_reference") else {
        return;
    };
    let stage3 = Stage3::new();
    for depth in [1usize, 2, 31, 32, 33, 700, 1023, 1024] {
        // [[[...{}...]]] — an empty object at the innermost level.
        let mut input = b"[".repeat(depth - 1);
        input.extend_from_slice(b"{}");
        input.extend_from_slice(&b"]".repeat(depth - 1));
        let verdict = diff(
            &stage3,
            &ctx,
            &input,
            DEFAULT_MAX_DEPTH,
            &format!("empty object at depth {depth}"),
        );
        assert_eq!(verdict, Verdict::Clean, "empty object at depth {depth}");

        // {"a":{"a":...[]...}} — an empty array under object nesting.
        let mut input = Vec::new();
        for _ in 0..depth - 1 {
            input.extend_from_slice(br#"{"a":"#);
        }
        input.extend_from_slice(b"[]");
        input.extend_from_slice(&b"}".repeat(depth - 1));
        let verdict = diff(
            &stage3,
            &ctx,
            &input,
            DEFAULT_MAX_DEPTH,
            &format!("empty array at depth {depth}"),
        );
        assert_eq!(verdict, Verdict::Clean, "empty array at depth {depth}");
    }

    // Siblings of empty containers (counting-sort stability + the
    // adjacent-token empty check, many times in one depth group).
    let mut input = b"[".to_vec();
    for i in 0..2000usize {
        if i > 0 {
            input.push(b',');
        }
        input.extend_from_slice(if i % 2 == 0 { b"{}" } else { b"[]" });
    }
    input.push(b']');
    let verdict = diff(
        &stage3,
        &ctx,
        &input,
        DEFAULT_MAX_DEPTH,
        "2000 empty siblings",
    );
    assert_eq!(verdict, Verdict::Clean);
}

/// Multi-root inputs and whitespace-padded roots: starter-led trailing
/// content dies in Layer 1 (MissingComma), separator-led trailing content
/// in CB3 (TrailingContent) — and whitespace alone never changes a
/// verdict.
#[test]
fn multi_root_and_whitespace_padded_roots() {
    let Some(ctx) = ctx_or_skip("multi_root_and_whitespace_padded_roots") else {
        return;
    };
    let stage3 = Stage3::new();

    let multi_root: &[(&[u8], Verdict)] = &[
        (b"{} {}", Verdict::Layer1),
        (b"[] []", Verdict::Layer1),
        (b"1 2", Verdict::Layer1),
        (b"null null", Verdict::Layer1),
        (br#""a" "b""#, Verdict::Layer1),
        (b"{},{}", Verdict::Structural),
        (b"1,2", Verdict::Structural),
        (br#"[1],"x""#, Verdict::Structural),
        (br#"{"a":1}:2"#, Verdict::Layer1), // `}` then `:` — adjacency ban
        (b"[] , []", Verdict::Structural),
        (b"true,", Verdict::Layer1), // separator then end of input
    ];
    for &(input, want) in multi_root {
        let verdict = diff(
            &stage3,
            &ctx,
            input,
            DEFAULT_MAX_DEPTH,
            &format!("{:?}", String::from_utf8_lossy(input)),
        );
        assert_eq!(verdict, want, "{:?}", String::from_utf8_lossy(input));
    }

    // Whitespace-padded roots parse identically to their trimmed forms.
    let padded: &[&[u8]] = &[
        b"   {}   ",
        b"\t[1,2]\n",
        b"\r\n  {\"a\": [true, {}]}  \r\n",
        b" 42 ",
        b"\nnull\t",
        b"  \"s\"  ",
    ];
    for &input in padded {
        let verdict = diff(
            &stage3,
            &ctx,
            input,
            DEFAULT_MAX_DEPTH,
            &format!("{:?}", String::from_utf8_lossy(input)),
        );
        assert_eq!(
            verdict,
            Verdict::Clean,
            "{:?}: whitespace padding is harmless",
            String::from_utf8_lossy(input)
        );
    }
}

// --- 4. Property tests -----------------------------------------------------------

mod proptests {
    use proptest::prelude::*;

    use metal_json::gpu::Stage3;
    use metal_json::metal::MetalContext;
    use metal_json::parser::DEFAULT_MAX_DEPTH;

    use super::{Verdict, diff};

    /// `PROPTEST_CASES` env knob with a CI default of 64.
    fn ci_cases() -> u32 {
        std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64)
    }

    /// One Metal context + Stage3 per test thread (proptest runs all cases
    /// of a test on its thread). Skips quietly without a device unless
    /// `METAL_JSON_REQUIRE_GPU=1`.
    fn with_gpu(f: impl FnOnce(&MetalContext, &Stage3)) {
        use std::cell::OnceCell;
        thread_local! {
            static GPU: OnceCell<Option<(MetalContext, Stage3)>> =
                const { OnceCell::new() };
        }
        GPU.with(|cell| {
            let gpu = cell.get_or_init(|| match MetalContext::new() {
                Ok(ctx) => Some((ctx, Stage3::new())),
                Err(err) => {
                    if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                        panic!("METAL_JSON_REQUIRE_GPU=1 but no usable Metal device: {err}");
                    }
                    eprintln!("SKIP structure proptests: no usable Metal device ({err})");
                    None
                }
            });
            if let Some((ctx, stage3)) = gpu {
                f(ctx, stage3);
            }
        });
    }

    /// Arbitrary JSON documents — the M1/M2 generator, duplicated here
    /// because test crates cannot share code outside `tests/common/`.
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

    /// Random structural token soup: bracket/comma/colon-heavy symbol
    /// streams (balanced and unbalanced alike) with strings, scalars and
    /// whitespace sprinkled — the structural-stage analogue of kernels.rs's
    /// quote/backslash soup. Symbols concatenate WITHOUT forced separators,
    /// so literal smashes (`true1`, `1x`) probe the literal/number token
    /// boundaries too.
    fn structural_soup() -> impl Strategy<Value = Vec<u8>> {
        let sym = prop_oneof![
            6 => Just(&b"{"[..]),
            6 => Just(&b"}"[..]),
            6 => Just(&b"["[..]),
            6 => Just(&b"]"[..]),
            5 => Just(&b":"[..]),
            5 => Just(&b","[..]),
            4 => Just(&b"\"x\""[..]),
            4 => Just(&b"1"[..]),
            2 => Just(&b"true"[..]),
            2 => Just(&b"null"[..]),
            1 => Just(&b"x"[..]),
            4 => Just(&b" "[..]),
            1 => Just(&b"\n"[..]),
        ];
        prop::collection::vec(sym, 0..96).prop_map(|syms| syms.concat())
    }

    /// Serialize a random valid document, then apply a few random
    /// byte-level edits (delete / insert / replace with structural bytes):
    /// near-miss documents whose first error can land in any stage.
    fn mutated_json() -> impl Strategy<Value = Vec<u8>> {
        const EDIT_BYTES: &[u8] = b"{}[]:,\"\\x10 ";
        let edit = (
            any::<prop::sample::Index>(),
            0..3u8,
            prop::sample::select(EDIT_BYTES),
        );
        (arb_json(), prop::collection::vec(edit, 0..4)).prop_map(|(value, edits)| {
            let mut bytes = serde_json::to_vec(&value).expect("serializable");
            for (index, kind, byte) in edits {
                match kind {
                    0 if !bytes.is_empty() => {
                        let i = index.index(bytes.len());
                        bytes.remove(i);
                    }
                    1 => {
                        let i = index.index(bytes.len() + 1);
                        bytes.insert(i, byte);
                    }
                    _ if !bytes.is_empty() => {
                        let i = index.index(bytes.len());
                        bytes[i] = byte;
                    }
                    _ => {}
                }
            }
            bytes
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(ci_cases()))]

        /// Random valid JSON documents must come out fully Clean: verdict
        /// AND every structure output AND the designated tape words.
        #[test]
        fn random_json_documents_are_clean_through_stage4(value in arb_json()) {
            let bytes = serde_json::to_vec(&value).expect("serializable");
            with_gpu(|ctx, stage3| {
                let verdict = diff(stage3, ctx, &bytes, DEFAULT_MAX_DEPTH, "random JSON");
                assert_eq!(verdict, Verdict::Clean, "valid JSON must be clean");
            });
        }

        /// Raw byte soup: arbitrary bytes must never panic or hang either
        /// backend, and the verdicts/outputs must agree exactly.
        #[test]
        fn random_byte_soup_matches_the_reference(
            bytes in prop::collection::vec(any::<u8>(), 0..2049)
        ) {
            with_gpu(|ctx, stage3| {
                diff(stage3, ctx, &bytes, DEFAULT_MAX_DEPTH, "random byte soup");
            });
        }

        /// Structural token soup: maximal pressure on the Layer-1 table,
        /// the depth scan, pairing and the Layer-2 context checks.
        #[test]
        fn random_structural_soup_matches_the_reference(bytes in structural_soup()) {
            with_gpu(|ctx, stage3| {
                diff(stage3, ctx, &bytes, DEFAULT_MAX_DEPTH, "structural soup");
            });
        }

        /// Near-miss documents (valid JSON with a few structural edits):
        /// whatever stage rejects first must agree across backends.
        #[test]
        fn mutated_json_documents_match_the_reference(bytes in mutated_json()) {
            with_gpu(|ctx, stage3| {
                diff(stage3, ctx, &bytes, DEFAULT_MAX_DEPTH, "mutated JSON");
            });
        }
    }
}

// --- 5. Manual timing sanity ------------------------------------------------------

/// Deterministic single-line synthetic document of at least `target`
/// bytes (the tests/kernels.rs generator): escape-heavy members whose
/// lengths drift so quotes and runs hit every seam alignment.
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

/// Manual timing sanity check, NOT a perf gate: ~256 MB synthetic doc
/// through the FULL M3 pipeline (CB1 + CB2 + CB2b + CB3), GB/s from the
/// command buffers' GPU timestamps via `Stage3::run_timed` (CPU sync gaps
/// and the test-only Vec readback excluded; wall clock printed alongside
/// for context). Expect lower than stage-1's ~52.9 GB/s — this prints the
/// number for eyeballing only. Run manually:
/// `cargo test --release --features cpu-reference --test structure -- --ignored --nocapture timing_sanity`
#[test]
#[ignore = "manual: allocates ~256 MB and prints GB/s; not a perf gate"]
fn timing_sanity_full_pipeline_throughput() {
    let Some(ctx) = ctx_or_skip("timing_sanity_full_pipeline_throughput") else {
        return;
    };
    let stage3 = Stage3::new();
    let input = synthetic_single_line(256 * 1024 * 1024);

    // Warm-up: PSO creation, first-touch page faults, GPU power ramp.
    let (out, _) = stage3.run_timed(&ctx, &input).expect("warm-up run");
    assert_eq!(out.error, None, "synthetic doc must be structure-clean");
    assert!(!out.tape.is_empty());

    let mut best_gpu = f64::INFINITY;
    let mut best_wall = f64::INFINITY;
    for round in 0..3 {
        let wall = std::time::Instant::now();
        let (out, gpu_secs) = stage3.run_timed(&ctx, &input).expect("timed run");
        let wall_secs = wall.elapsed().as_secs_f64();
        assert_eq!(out.error, None);
        assert!(
            gpu_secs > 0.0,
            "GPU timestamps must be available (round {round})"
        );
        best_gpu = best_gpu.min(gpu_secs);
        best_wall = best_wall.min(wall_secs);
    }
    let gpu_gbps = input.len() as f64 / best_gpu / 1e9;
    let wall_gbps = input.len() as f64 / best_wall / 1e9;
    println!(
        "full M3 pipeline (best of 3): GPU {:.3} ms = {gpu_gbps:.1} GB/s over {} bytes \
         (wall {:.3} ms = {wall_gbps:.1} GB/s incl. CPU syncs + test readback; \
         stage-1 alone measured ~52.9 GB/s — informational, not asserted)",
        best_gpu * 1e3,
        input.len(),
        best_wall * 1e3,
    );
}
