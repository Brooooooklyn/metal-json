//! M4 milestone gate: the GPU backend end-to-end against the CPU reference
//! oracle AND serde_json — `GPU == reference == serde_json`.
//!
//! What this suite pins (and `tests/jsontestsuite.rs` /
//! `tests/differential.rs` pin for the reference backend):
//!
//! 1. **JSONTestSuite conformance on the GPU**: every `y_` file parses,
//!    every `n_` file is rejected, every `i_` file lands on the SAME
//!    pinned verdict as the CPU suite (`common::I_FILE_VERDICTS`, shared),
//!    and the GPU verdict equals the reference verdict on every file
//!    (WHETHER always; per-class code+offset parity is pinned in
//!    `src/parser.rs` / `src/gpu/pipeline.rs`, with the documented
//!    multi-error WHETHER-not-WHICH relaxation).
//! 2. **Differential vs serde_json through the GPU**: every corpus fixture
//!    and every serde-accepted `y_` file walks identically
//!    (`common::assert_doc_eq`: kinds, document order, exact strings,
//!    bit-exact doubles); duplicate-key tape semantics asserted on the raw
//!    GPU tape.
//! 3. **The number torture table** (`tests/common/numbers.rs`, shared with
//!    `tests/numbers.rs`) bit-exact through the GPU backend, plus
//!    fixup-path literals embedded inside full documents.
//! 4. **Whole-tape bit-exactness**: for every accepted corpus + `y_` file
//!    the GPU tape words equal the reference tape words AND the whole
//!    string buffer is byte-equal — records and zero-filled gap bytes
//!    alike (the pinned policy in `docs/tape-format.md`).
//! 5. **Property tests**: random documents (tape-exact), raw byte soup
//!    (verdict parity, never panic), escape-dense documents (tape-exact +
//!    serde agreement).
//! 6. `#[ignore]` informational timing prints (M5 owns perf).
#![cfg(feature = "cpu-reference")]

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use metal_json::tape::{
    STRING_RECORD_HEADER_BYTES, TAG_DOUBLE, TAG_INT64, TAG_STRING, TAG_UINT64, string_offset, tag,
};
use metal_json::{Document, Parser, ValueKind};

fn file_name(path: &Path) -> &str {
    path.file_name().and_then(|n| n.to_str()).unwrap_or("???")
}

/// Whole-artifact equality, the apples-to-apples contract of
/// `docs/tape-format.md`: tape words bit-identical (string offsets
/// included — the raw-length prefix-sum allocation makes them equal), the
/// **whole string buffer byte-equal** (gap bytes are zero-filled on both
/// backends, so even the slack of escape-shrunk slots must match), and at
/// every `"` word's offset a well-formed `[u32 LE len][content][NUL]`
/// record. Returns the number of string records compared.
fn assert_tape_and_records_eq(gpu: &Document, cpu: &Document, label: &str) -> usize {
    assert_raw_tape_and_records_eq(gpu.tape(), gpu.strings().as_bytes(), cpu, label)
}

/// [`assert_tape_and_records_eq`] over raw GPU-side slices — usable
/// straight off a `GpuParseOutput` (the pipeline-level pool/poison tests)
/// as well as a `Document`.
fn assert_raw_tape_and_records_eq(
    gpu_tape: &[u64],
    gpu_bytes: &[u8],
    cpu: &Document,
    label: &str,
) -> usize {
    assert_eq!(
        gpu_tape,
        cpu.tape(),
        "{label}: tape words must be bit-identical"
    );
    let cpu_bytes = cpu.strings().as_bytes();
    assert_eq!(
        gpu_bytes.len(),
        cpu_bytes.len(),
        "{label}: string buffer size (raw-length prefix-sum total)"
    );
    // THE whole-buffer pin: gap bytes are zero-filled on both backends
    // (docs/tape-format.md), so the raw buffers — records AND gaps — must
    // be byte-identical, not merely record-equal.
    assert_eq!(
        gpu_bytes, cpu_bytes,
        "{label}: whole string buffer must be byte-identical (gaps included)"
    );
    let mut records = 0usize;
    let tape = gpu_tape;
    let mut i = 0;
    while i < tape.len() {
        let word = tape[i];
        // Number entries are TWO words; the raw value word can hold any
        // bit pattern (its high byte may collide with a tag), so skip it.
        if matches!(tag(word), TAG_INT64 | TAG_UINT64 | TAG_DOUBLE) {
            i += 2;
            continue;
        }
        if tag(word) != TAG_STRING {
            i += 1;
            continue;
        }
        let offset = usize::try_from(string_offset(word)).expect("offset fits usize");
        let header_end = offset + STRING_RECORD_HEADER_BYTES;
        assert!(
            header_end <= gpu_bytes.len(),
            "{label}: tape[{i}] record header at {offset} exceeds the buffer"
        );
        let header: [u8; STRING_RECORD_HEADER_BYTES] =
            gpu_bytes[offset..header_end].try_into().expect("4 bytes");
        let len = u32::from_le_bytes(header) as usize;
        let record_end = header_end + len + 1; // content + NUL
        assert!(
            record_end <= gpu_bytes.len(),
            "{label}: tape[{i}] record at {offset} (len {len}) exceeds the buffer"
        );
        assert_eq!(
            &gpu_bytes[offset..record_end],
            &cpu_bytes[offset..record_end],
            "{label}: tape[{i}] string record at offset {offset}"
        );
        assert_eq!(
            gpu_bytes[record_end - 1],
            0,
            "{label}: tape[{i}] record at {offset} must end in NUL"
        );
        records += 1;
        i += 1;
    }
    records
}

/// Assert every byte of `bytes` NOT covered by a record reachable from a
/// `"` tape word reads back **zero** — the slot gaps escape-shrunk records
/// leave. Independent of the reference backend on purpose: the poison test
/// uses it to prove a pooled buffer's previous contents (0xDB) never
/// survive into the reachable gap bytes of an accepted parse. Returns the
/// number of gap bytes checked.
fn assert_string_gaps_zero(tape: &[u64], bytes: &[u8], label: &str) -> usize {
    // Collect each record's [start, end-of-NUL) extent off the tape.
    let mut records: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < tape.len() {
        let word = tape[i];
        if matches!(tag(word), TAG_INT64 | TAG_UINT64 | TAG_DOUBLE) {
            i += 2; // skip the raw value word (any bit pattern)
            continue;
        }
        if tag(word) == TAG_STRING {
            let offset = usize::try_from(string_offset(word)).expect("offset fits usize");
            let header: [u8; STRING_RECORD_HEADER_BYTES] = bytes
                [offset..offset + STRING_RECORD_HEADER_BYTES]
                .try_into()
                .expect("4 header bytes");
            let len = u32::from_le_bytes(header) as usize;
            records.push((offset, offset + STRING_RECORD_HEADER_BYTES + len + 1));
        }
        i += 1;
    }
    // Slots are allocated back-to-back in document order, so sorting by
    // start offset makes every uncovered range a slot-tail gap.
    records.sort_unstable();
    let mut gap_bytes = 0usize;
    let mut cursor = 0usize;
    for &(start, end) in &records {
        assert!(
            bytes[cursor..start].iter().all(|&b| b == 0),
            "{label}: gap bytes {cursor}..{start} must read back zero"
        );
        gap_bytes += start - cursor;
        cursor = end;
    }
    assert!(
        bytes[cursor..].iter().all(|&b| b == 0),
        "{label}: trailing gap bytes {cursor}..{} must read back zero",
        bytes.len()
    );
    gap_bytes + (bytes.len() - cursor)
}

/// Both backends on one input: same verdict; on acceptance the
/// tape/record artifacts are bit-identical and the value walks agree.
fn diff_backends(gpu: &Parser, cpu: &Parser, input: &[u8], label: &str) {
    match (gpu.parse(input), cpu.parse(input)) {
        (Ok(gpu_doc), Ok(cpu_doc)) => {
            assert_tape_and_records_eq(&gpu_doc, &cpu_doc, label);
            common::assert_docs_eq(gpu_doc.root(), cpu_doc.root(), label);
        }
        (Err(_), Err(_)) => {} // verdict parity; WHICH may differ by design
        (Ok(_), Err(e)) => panic!("{label}: GPU accepted, reference rejected ({e})"),
        (Err(e), Ok(_)) => panic!("{label}: GPU rejected ({e}), reference accepted"),
    }
}

// --- 1. JSONTestSuite through Parser(Gpu) ----------------------------------------

/// The full conformance suite through the public `Parser` on the GPU
/// backend: y 95/95 parse, n 188/188 reject, i 35/35 on the pinned
/// verdicts shared with the CPU suite — plus verdict parity against the
/// reference on every single file.
#[test]
fn jsontestsuite_full_suite_through_the_gpu_parser() {
    let Some(dir) = common::jsontestsuite_dir() else {
        return; // loud skip already printed
    };
    let Some(gpu) = common::gpu_parser_or_skip("jsontestsuite_full_suite_through_the_gpu_parser")
    else {
        return;
    };
    let cpu = common::cpu_parser();

    let mut y_pass = 0usize;
    let mut n_pass = 0usize;
    let mut i_pinned = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for prefix in ["y_", "n_", "i_"] {
        for path in common::jsontestsuite_files(&dir, prefix) {
            let name = file_name(&path).to_owned();
            let bytes = std::fs::read(&path).expect("readable corpus file");
            let Ok(gpu_result) = catch_unwind(AssertUnwindSafe(|| gpu.parse(&bytes))) else {
                failures.push(format!("{name}: GPU backend PANICKED (must never happen)"));
                continue;
            };
            let accepted = gpu_result.is_ok();

            // Two-way verdict parity vs the reference, on EVERY file.
            let cpu_accepted = cpu.parse(&bytes).is_ok();
            if accepted != cpu_accepted {
                failures.push(format!(
                    "{name}: GPU accept={accepted}, reference accept={cpu_accepted}"
                ));
            }

            match prefix {
                "y_" if accepted => y_pass += 1,
                "y_" => failures.push(format!("{name}: must parse on the GPU, got Err")),
                "n_" if !accepted => n_pass += 1,
                "n_" => failures.push(format!("{name}: must be rejected on the GPU, got Ok")),
                _ => match common::I_FILE_VERDICTS.iter().find(|(n, _, _)| *n == name) {
                    None => failures.push(format!(
                        "{name}: new i_ file — add a pinned verdict to common::I_FILE_VERDICTS"
                    )),
                    Some((_, want_accept, reason)) if *want_accept != accepted => {
                        failures.push(format!(
                            "{name}: pinned verdict accept={want_accept} ({reason}), \
                             but the GPU got accept={accepted}"
                        ));
                    }
                    Some(_) => i_pinned += 1,
                },
            }
        }
    }

    let y_total = common::jsontestsuite_files(&dir, "y_").len();
    let n_total = common::jsontestsuite_files(&dir, "n_").len();
    let i_total = common::jsontestsuite_files(&dir, "i_").len();
    println!(
        "GPU JSONTestSuite: y {y_pass}/{y_total} parsed, n {n_pass}/{n_total} rejected, \
         i {i_pinned}/{i_total} on pinned verdicts, {} failures",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{} GPU JSONTestSuite failures:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
    assert_eq!(y_pass, y_total, "every y_ file must parse on the GPU");
    assert_eq!(n_pass, n_total, "every n_ file must be rejected on the GPU");
    assert_eq!(i_pinned, i_total, "every i_ file must match its pinned verdict");
    // The fetched corpus must be complete (95/188/35 at the time of pinning).
    assert!(
        y_total >= 95 && n_total >= 188 && i_total >= 35,
        "suite incomplete? y {y_total} n {n_total} i {i_total} — re-fetch with \
         scripts/fetch_jsontestsuite.sh"
    );
}

// --- 2. Differential vs serde_json through the GPU backend ------------------------

/// Every corpus fixture and every serde-accepted `y_` file, parsed on the
/// GPU and walked against serde_json (`preserve_order` +
/// `arbitrary_precision`: ordered objects, by-kind numbers, bit-exact
/// doubles vs `str::parse::<f64>` of the raw literal).
#[test]
fn corpus_and_y_files_match_serde_json_through_the_gpu() {
    let Some(gpu) =
        common::gpu_parser_or_skip("corpus_and_y_files_match_serde_json_through_the_gpu")
    else {
        return;
    };

    let mut files: Vec<PathBuf> = common::corpus_files();
    let suite_present = common::jsontestsuite_dir();
    if let Some(dir) = &suite_present {
        files.extend(common::jsontestsuite_files(dir, "y_"));
    }

    let mut compared = 0usize;
    let mut dup_key_files = Vec::new();
    for path in &files {
        let name = file_name(path).to_owned();
        let bytes = std::fs::read(path).expect("readable fixture");
        let doc = gpu
            .parse(&bytes)
            .unwrap_or_else(|e| panic!("{name}: must parse on the GPU, got {e}"));

        if common::has_duplicate_keys(doc.root()) {
            dup_key_files.push(name); // covered by the duplicate-key test below
            continue;
        }
        let serde: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            panic!(
                "{name}: serde_json (arbitrary_precision) accepts every corpus fixture \
                 and y_ file today; a new rejection needs investigating: {e}"
            )
        });
        common::assert_doc_eq(doc.root(), &serde, &name);
        compared += 1;
    }

    println!(
        "GPU vs serde_json: {compared} files compared, {} duplicate-key files \
         (tape-tested separately): {dup_key_files:?}",
        dup_key_files.len()
    );
    let want_dups: &[&str] = if suite_present.is_some() {
        &[
            "duplicate_keys.json",
            "y_object_duplicated_key.json",
            "y_object_duplicated_key_and_value.json",
        ]
    } else {
        &["duplicate_keys.json"]
    };
    assert_eq!(dup_key_files, want_dups, "the known duplicate-key fixtures");
    assert!(
        compared >= files.len() - want_dups.len(),
        "every non-duplicate-key file must be serde-comparable"
    );
}

/// Duplicate object keys ride the GPU tape verbatim, in document order
/// (simdjson parity) — where serde_json's map would collapse them.
#[test]
fn duplicate_key_tape_semantics_on_the_gpu() {
    let Some(gpu) = common::gpu_parser_or_skip("duplicate_key_tape_semantics_on_the_gpu") else {
        return;
    };
    let cpu = common::cpu_parser();

    // The corpus fixture, against the raw tape walk (mirrors
    // tests/differential.rs's reference-backend assertions).
    let bytes =
        std::fs::read(common::corpus_dir().join("duplicate_keys.json")).expect("checked in");
    diff_backends(&gpu, &cpu, &bytes, "duplicate_keys.json");
    let doc = gpu.parse(&bytes).expect("fixture parses on the GPU");
    let root = doc.root();
    let entries: Vec<&str> = root.entries().map(|(k, _)| k).collect();
    assert_eq!(entries, ["k", "k", "k", "other", "arr"]);
    let k_values: Vec<i64> = root
        .entries()
        .filter(|(k, _)| *k == "k")
        .map(|(_, v)| v.as_i64().expect("k values are ints"))
        .collect();
    assert_eq!(k_values, [1, 2, 3]);
    // get() resolves duplicates to the FIRST match (simdjson at_key).
    assert_eq!(root.get("k").unwrap().as_i64(), Some(1));
    // And serde really does disagree — guarding the premise of the split.
    let serde: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(serde.as_object().unwrap().len(), 3);

    // The suite's two duplicate-key y_ files, when fetched.
    if let Some(dir) = common::jsontestsuite_dir() {
        // y_object_duplicated_key.json: {"a":"b","a":"c"}
        let bytes = std::fs::read(dir.join("y_object_duplicated_key.json")).unwrap();
        diff_backends(&gpu, &cpu, &bytes, "y_object_duplicated_key.json");
        let doc = gpu.parse(&bytes).expect("parses on the GPU");
        let root = doc.root();
        assert_eq!(root.kind(), ValueKind::Object);
        assert_eq!(root.len(), Some(2));
        let entries: Vec<(&str, &str)> = root
            .entries()
            .map(|(k, v)| (k, v.as_str().unwrap()))
            .collect();
        assert_eq!(entries, [("a", "b"), ("a", "c")]);
        assert_eq!(root.get("a").unwrap().as_str(), Some("b"), "first wins");

        // y_object_duplicated_key_and_value.json: {"a":"b","a":"b"}
        let bytes = std::fs::read(dir.join("y_object_duplicated_key_and_value.json")).unwrap();
        diff_backends(&gpu, &cpu, &bytes, "y_object_duplicated_key_and_value.json");
        let doc = gpu.parse(&bytes).expect("parses on the GPU");
        let entries: Vec<(&str, &str)> = doc
            .root()
            .entries()
            .map(|(k, v)| (k, v.as_str().unwrap()))
            .collect();
        assert_eq!(entries, [("a", "b"), ("a", "b")]);
    }
}

// --- 3. The number torture table through the GPU backend --------------------------

/// The full deterministic torture table (shared with `tests/numbers.rs`)
/// through the GPU backend: bit-exact doubles vs the `str::parse::<f64>`
/// oracle, type-selection boundaries, overflow/underflow policy, grammar
/// rejections — including the Eisel-Lemire hard cases that route through
/// the CPU fixup patch.
#[test]
fn numbers_torture_table_through_the_gpu() {
    let Some(gpu) = common::gpu_parser_or_skip("numbers_torture_table_through_the_gpu") else {
        return;
    };
    common::numbers::full_torture_table(&gpu);
}

/// Fixup-path literals INSIDE full documents: hard roundings the GPU
/// kernel punts to the CPU patch (truncated ≥ 20-digit mantissas,
/// halfway ties, the subnormal boundary) embedded among containers,
/// strings and easy numbers — tape-exact vs the reference AND bit-exact
/// vs the oracle.
#[test]
fn fixup_path_numbers_inside_documents_are_bit_exact() {
    let Some(gpu) =
        common::gpu_parser_or_skip("fixup_path_numbers_inside_documents_are_bit_exact")
    else {
        return;
    };
    let cpu = common::cpu_parser();

    // Double-path literals only (so the oracle check below is uniform);
    // the first three are pinned fixup-path cases in src/gpu/numbers.rs.
    const HARD: &[&str] = &[
        // halfway(1.0, 1.0 + ulp): 54 digits, ties to even -> exactly 1.0.
        "1.00000000000000011102230246251565404236316680908203125",
        "-1.00000000000000011102230246251565404236316680908203125",
        // The 2.2250738585072011e-308 long form (subnormal/normal boundary).
        "2.22507385850720113605740979670913197593481954635164564e-308",
        // Truncated >= 20-digit mantissas.
        "0.99999999999999999999999999999999999999",
        "12345678901234567890123456789012345678901234567890e-25",
        "1234567890123456789.012345678901234567890123456789",
        // Extreme-but-finite exponents and range edges.
        "5e-324",
        "2.4703282292062328e-324",
        "1.7976931348623157e308",
        "9007199254740993.0", // 2^53 + 1: not exactly representable
        "1e23",               // famous halfway case
    ];
    let json = format!(
        r#"{{"hard":[{}],"pad":[true,null,"sA\n",42,2.5]}}"#,
        HARD.join(",")
    );

    diff_backends(&gpu, &cpu, json.as_bytes(), "fixup document");

    let doc = gpu.parse(json.as_bytes()).expect("fixup document parses");
    let hard = doc.root().get("hard").expect("hard array present");
    for (i, text) in HARD.iter().enumerate() {
        let oracle: f64 = text.parse().expect("oracle accepts every HARD literal");
        let value = hard.at(i).unwrap_or_else(|| panic!("hard[{i}] missing"));
        assert_eq!(value.kind(), ValueKind::Double, "hard[{i}] = {text:?}");
        assert_eq!(
            value.as_f64().map(f64::to_bits),
            Some(oracle.to_bits()),
            "hard[{i}] = {text:?}: bits must equal str::parse::<f64>"
        );
    }
    assert_eq!(doc.root().get("pad").unwrap().at(3).unwrap().as_i64(), Some(42));
}

// --- 4. Whole-tape bit-exactness ---------------------------------------------------

/// THE apples-to-apples artifact: for every accepted corpus + `y_` file,
/// the GPU tape words equal the reference tape words bit-for-bit AND the
/// whole string buffer is byte-equal — records and zero-filled gaps alike
/// (the pinned policy in docs/tape-format.md).
#[test]
fn whole_tape_bit_exact_on_corpus_and_y_files() {
    let Some(gpu) = common::gpu_parser_or_skip("whole_tape_bit_exact_on_corpus_and_y_files")
    else {
        return;
    };
    let cpu = common::cpu_parser();

    let mut files: Vec<PathBuf> = common::corpus_files();
    if let Some(dir) = common::jsontestsuite_dir() {
        files.extend(common::jsontestsuite_files(&dir, "y_"));
    }

    let mut checked = 0usize;
    let mut string_records = 0usize;
    for path in &files {
        let name = file_name(path).to_owned();
        let bytes = std::fs::read(path).expect("readable fixture");
        let cpu_doc = cpu
            .parse(&bytes)
            .unwrap_or_else(|e| panic!("{name}: accepted set must parse on the reference: {e}"));
        let gpu_doc = gpu
            .parse(&bytes)
            .unwrap_or_else(|e| panic!("{name}: must parse on the GPU, got {e}"));
        string_records += assert_tape_and_records_eq(&gpu_doc, &cpu_doc, &name);
        checked += 1;
    }
    println!(
        "whole-tape bit-exactness: {checked} files, {string_records} string records compared"
    );
    assert_eq!(checked, files.len());
    assert!(
        string_records > 1000,
        "the corpus must exercise string records heavily (got {string_records})"
    );
}

// --- 5. Property tests -------------------------------------------------------------

mod proptests {
    use std::cell::OnceCell;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use metal_json::Parser;
    use proptest::prelude::*;

    use super::{assert_tape_and_records_eq, common, diff_backends};

    /// `PROPTEST_CASES` env knob with a CI default of 64.
    fn ci_cases() -> u32 {
        std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64)
    }

    /// One GPU + one reference parser per test thread (proptest runs all
    /// cases of a test on its thread). Skips quietly without a device
    /// unless `METAL_JSON_REQUIRE_GPU=1`.
    fn with_parsers(f: impl FnOnce(&Parser, &Parser)) {
        thread_local! {
            static PARSERS: OnceCell<Option<(Parser, Parser)>> = const { OnceCell::new() };
        }
        PARSERS.with(|cell| {
            let parsers = cell.get_or_init(|| match Parser::new() {
                Ok(gpu) => Some((gpu, common::cpu_parser())),
                Err(err) => {
                    if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                        panic!("METAL_JSON_REQUIRE_GPU=1 but no usable Metal device: {err}");
                    }
                    eprintln!("SKIP gpu_e2e proptests: no usable Metal device ({err})");
                    None
                }
            });
            if let Some((gpu, cpu)) = parsers {
                f(gpu, cpu);
            }
        });
    }

    /// Arbitrary JSON documents — the shared M1/M2 generator shape.
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

    /// JSON string-literal SOURCE text dense with escapes: every simple
    /// escape, `\uXXXX` BMP escapes (controls and NUL included), surrogate
    /// pairs for astral code points, raw multi-byte UTF-8 and plain runs,
    /// concatenated so escapes collide with each other and with chunk
    /// seams. Always grammatically valid.
    fn escape_dense_literal() -> impl Strategy<Value = String> {
        let fragment = prop_oneof![
            3 => prop::sample::select(vec![
                r#"\""#, r"\\", r"\/", r"\b", r"\f", r"\n", r"\r", r"\t",
            ])
            .prop_map(str::to_owned),
            // BMP escapes below the surrogate range (NUL + controls included).
            2 => (0x0000u32..0xD800).prop_map(|c| format!("\\u{c:04x}")),
            // BMP escapes above the surrogate range, uppercase hex.
            1 => (0xE000u32..0x1_0000).prop_map(|c| format!("\\u{c:04X}")),
            // Astral code points as surrogate pairs.
            1 => (0x1_0000u32..=0x10_FFFF).prop_map(|c| {
                let v = c - 0x1_0000;
                format!("\\u{:04x}\\u{:04x}", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF))
            }),
            // Plain ASCII runs (alphabet free of `"` and `\`).
            2 => "[a-zA-Z0-9 .,:;{}\\[\\]_-]{0,8}",
            // Raw multi-byte UTF-8 between the escapes.
            1 => Just("é😀\u{800}\u{10FFFF}".to_owned()),
        ];
        prop::collection::vec(fragment, 1..24).prop_map(|v| v.concat())
    }

    /// A document whose every key and value is an escape-dense string.
    fn escape_dense_document() -> impl Strategy<Value = String> {
        prop::collection::vec((escape_dense_literal(), escape_dense_literal()), 1..8).prop_map(
            |pairs| {
                let members: Vec<String> = pairs
                    .iter()
                    .enumerate()
                    .map(|(i, (k, v))| format!(r#""k{i}{k}":["{v}",1.5,"{k}"]"#))
                    .collect();
                format!("{{{}}}", members.join(","))
            },
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(ci_cases()))]

        /// Random valid documents: GPU output is tape-exact (words +
        /// string records) against the reference.
        #[test]
        fn random_documents_are_tape_exact_on_the_gpu(value in arb_json()) {
            let json = serde_json::to_vec(&value).expect("serializable");
            with_parsers(|gpu, cpu| {
                let gpu_doc = gpu
                    .parse(&json)
                    .unwrap_or_else(|e| panic!(
                        "valid JSON must parse on the GPU: {e}\n{}",
                        String::from_utf8_lossy(&json)
                    ));
                let cpu_doc = cpu.parse(&json).expect("valid JSON parses on the reference");
                assert_tape_and_records_eq(&gpu_doc, &cpu_doc, "random document");
            });
        }

        /// Raw byte soup: the GPU backend must never panic, and the
        /// accept/reject verdict must match the reference on every input
        /// (accepted soup — rare but possible — must be tape-exact too).
        #[test]
        fn random_byte_soup_has_verdict_parity_and_never_panics(
            bytes in prop::collection::vec(any::<u8>(), 0..2049)
        ) {
            with_parsers(|gpu, cpu| {
                let gpu_result = catch_unwind(AssertUnwindSafe(|| gpu.parse(&bytes)));
                let Ok(gpu_result) = gpu_result else {
                    panic!("GPU backend PANICKED on byte soup {bytes:?}");
                };
                match (gpu_result, cpu.parse(&bytes)) {
                    (Ok(gpu_doc), Ok(cpu_doc)) => {
                        assert_tape_and_records_eq(&gpu_doc, &cpu_doc, "byte soup");
                    }
                    (Err(_), Err(_)) => {}
                    (gpu_v, cpu_v) => panic!(
                        "verdict split on byte soup {:?}: GPU ok={}, reference ok={}",
                        String::from_utf8_lossy(&bytes),
                        gpu_v.is_ok(),
                        cpu_v.is_ok()
                    ),
                }
            });
        }

        /// Escape-dense documents (every K11 escape form, surrogate pairs,
        /// interior NULs): tape-exact vs the reference AND serde-identical.
        #[test]
        fn escape_dense_documents_are_tape_exact_on_the_gpu(json in escape_dense_document()) {
            with_parsers(|gpu, cpu| {
                diff_backends(gpu, cpu, json.as_bytes(), "escape-dense document");
                let gpu_doc = gpu
                    .parse(json.as_bytes())
                    .unwrap_or_else(|e| panic!("escape-dense doc must parse on the GPU: {e}\n{json}"));
                let serde: serde_json::Value =
                    serde_json::from_str(&json).expect("serde accepts every generated doc");
                common::assert_doc_eq(gpu_doc.root(), &serde, "escape-dense document");
            });
        }
    }
}

// --- 6. The K11 long-string valve ---------------------------------------------------

/// Regressions for the long-string availability valve: strings with
/// `raw_len > LONG_STRING_THRESHOLD` are unescaped by the CPU patch pass
/// (`gpu::strings::patch_long_strings`) instead of one GPU lane, mirroring
/// the K10 number-fixup pattern. Every accepted case asserts the fixup
/// list was actually exercised AND bit-matches the reference
/// (tape + records); see `src/gpu/strings.rs` for the runner-level
/// boundary/rejection twins.
mod long_strings {
    use metal_json::gpu::{GpuParse, GpuPipeline, LONG_STRING_THRESHOLD};
    use metal_json::metal::MetalContext;
    use metal_json::{Error, SyntaxErrorKind};

    use super::{common, diff_backends};

    /// `\u` + `hex` escape text, built at runtime (the literal sequence
    /// must not appear in source — the src/gpu/strings.rs convention).
    fn u_esc(hex: &str) -> String {
        format!("{}u{hex}", '\\')
    }

    /// The raw pipeline (for fixup-list visibility), gated like everything
    /// else.
    fn pipeline_or_skip(test: &str) -> Option<(MetalContext, GpuPipeline)> {
        match MetalContext::new() {
            Ok(ctx) => Some((ctx, GpuPipeline::new())),
            Err(err) => {
                if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                    panic!("METAL_JSON_REQUIRE_GPU=1 but no usable Metal device: {err}");
                }
                eprintln!("SKIP {test}: no usable Metal device here ({err})");
                None
            }
        }
    }

    /// Run the raw pipeline; the input must parse. Returns the sorted
    /// long-string fixup list.
    fn accepted_fixups(ctx: &MetalContext, pipeline: &GpuPipeline, input: &[u8]) -> Vec<u32> {
        match pipeline
            .run(ctx, input, metal_json::parser::DEFAULT_MAX_DEPTH)
            .expect("GPU plumbing")
        {
            GpuParse::Accepted(out) => out.long_string_fixups,
            GpuParse::Rejected(packed) => panic!(
                "input must parse, got packed error {:?}",
                (packed >> 32, packed as u32)
            ),
        }
    }

    /// (a) ~8 MB CLEAN string: the fast-path valve. One lane never owns
    /// the 8 MB walk; the CPU patch produces reference-identical output.
    #[test]
    fn valve_8mb_clean_string_is_bit_exact() {
        let Some(gpu) = common::gpu_parser_or_skip("valve_8mb_clean_string_is_bit_exact") else {
            return;
        };
        let Some((ctx, pipeline)) = pipeline_or_skip("valve_8mb_clean_string_is_bit_exact")
        else {
            return;
        };
        let cpu = common::cpu_parser();

        let mut body = String::new();
        while body.len() < 8 * 1024 * 1024 {
            body.push_str("abcdefgh é→😀 0123");
        }
        let json = format!("\"{body}\"");

        assert_eq!(
            accepted_fixups(&ctx, &pipeline, json.as_bytes()),
            vec![0],
            "an 8MB string MUST take the long-string fixup path"
        );

        let started = std::time::Instant::now();
        let gpu_doc = gpu.parse(json.as_bytes()).expect("clean 8MB string parses");
        eprintln!(
            "valve_8mb_clean_string: {} raw bytes in {:?} (whole Parser::parse)",
            body.len(),
            started.elapsed()
        );
        assert_eq!(gpu_doc.root().as_str(), Some(body.as_str()));
        let cpu_doc = cpu.parse(json.as_bytes()).expect("reference parses");
        super::assert_tape_and_records_eq(&gpu_doc, &cpu_doc, "8MB clean string");
    }

    /// (b) ~8 MB HEAVILY-ESCAPED string: the escape valve (the worst case
    /// of the old thread-per-string cliff). serde_json is the content
    /// oracle; the reference is the artifact oracle.
    #[test]
    fn valve_8mb_heavily_escaped_string_is_bit_exact() {
        let Some(gpu) =
            common::gpu_parser_or_skip("valve_8mb_heavily_escaped_string_is_bit_exact")
        else {
            return;
        };
        let Some((ctx, pipeline)) =
            pipeline_or_skip("valve_8mb_heavily_escaped_string_is_bit_exact")
        else {
            return;
        };
        let cpu = common::cpu_parser();

        let piece = format!(
            "{}{}{}{}{}{}",
            u_esc("D83D"),
            u_esc("DE00"),
            r"\n\t\\",
            "\\\"", // the \" escape (a raw string cannot hold a quote)
            u_esc("0000"),
            r"\/"
        );
        let mut body = String::new();
        while body.len() < 8 * 1024 * 1024 {
            body.push_str(&piece);
        }
        let json = format!("\"{body}\"");
        let want: String = serde_json::from_str(&json).expect("valid JSON string");

        assert_eq!(
            accepted_fixups(&ctx, &pipeline, json.as_bytes()),
            vec![0],
            "an 8MB escaped string MUST take the long-string fixup path"
        );

        let started = std::time::Instant::now();
        let gpu_doc = gpu.parse(json.as_bytes()).expect("escaped 8MB string parses");
        eprintln!(
            "valve_8mb_heavily_escaped_string: {} raw bytes in {:?} (whole Parser::parse)",
            body.len(),
            started.elapsed()
        );
        assert_eq!(gpu_doc.root().as_str(), Some(want.as_str()));
        let cpu_doc = cpu.parse(json.as_bytes()).expect("reference parses");
        super::assert_tape_and_records_eq(&gpu_doc, &cpu_doc, "8MB escaped string");
    }

    /// (c) Long + short strings + numbers in one document: offsets and
    /// gaps stay correct around the CPU-patched records (whole-tape +
    /// whole-record bit-match against the reference).
    #[test]
    fn valve_mixed_document_keeps_offsets_and_gaps_correct() {
        let Some(gpu) =
            common::gpu_parser_or_skip("valve_mixed_document_keeps_offsets_and_gaps_correct")
        else {
            return;
        };
        let Some((ctx, pipeline)) =
            pipeline_or_skip("valve_mixed_document_keeps_offsets_and_gaps_correct")
        else {
            return;
        };
        let cpu = common::cpu_parser();

        let t = LONG_STRING_THRESHOLD as usize;
        let long_clean = "x".repeat(4 * t);
        let long_escaped = format!("{}tail", r"\t".repeat(t)); // raw 2t + 4, shrinks to t + 4
        let json = format!(
            r#"{{"a":"short","b":"{long_clean}","n":[1,2.5e10,"{long_escaped}"],"z":"k\n"}}"#
        );

        // Document string order: "a", "short", "b", long_clean, "n",
        // long_escaped, "z", "k\n" — exactly the two long ones flagged.
        assert_eq!(
            accepted_fixups(&ctx, &pipeline, json.as_bytes()),
            vec![3, 5],
            "exactly the two long strings take the fixup path"
        );

        diff_backends(&gpu, &cpu, json.as_bytes(), "mixed long/short document");
        let doc = gpu.parse(json.as_bytes()).expect("mixed document parses");
        let root = doc.root();
        assert_eq!(root.get("a").unwrap().as_str(), Some("short"));
        assert_eq!(root.get("b").unwrap().as_str(), Some(long_clean.as_str()));
        let n = root.get("n").unwrap();
        assert_eq!(n.at(0).unwrap().as_i64(), Some(1));
        assert_eq!(n.at(1).unwrap().as_f64(), Some(2.5e10));
        let want_escaped = format!("{}tail", "\t".repeat(t));
        assert_eq!(n.at(2).unwrap().as_str(), Some(want_escaped.as_str()));
        assert_eq!(root.get("z").unwrap().as_str(), Some("k\n"));
    }

    /// (d) A long string whose escape ERROR sits past the threshold: the
    /// CPU patch pass rejects with the reference's exact (offset, kind).
    #[test]
    fn valve_long_string_error_rejects_at_the_reference_offset() {
        let Some(gpu) =
            common::gpu_parser_or_skip("valve_long_string_error_rejects_at_the_reference_offset")
        else {
            return;
        };
        let cpu = common::cpu_parser();

        let n = LONG_STRING_THRESHOLD as usize + 1234;
        let json = format!("[\"ok\",\"{}{}b\"]", "a".repeat(n), r"\q");
        // Backslash absolute offset: `["ok","` is 7 bytes, then n `a`s.
        let want_offset = 7 + n as u64;
        match gpu.parse(json.as_bytes()) {
            Err(Error::Syntax { offset, kind }) => {
                assert_eq!(kind, SyntaxErrorKind::InvalidStringEscape);
                assert_eq!(offset, want_offset, "CPU-side rejection at the backslash");
            }
            other => panic!("expected InvalidStringEscape, got {other:?}"),
        }
        // Reference parity: single-error document, code AND offset.
        let gpu_err = gpu.parse(json.as_bytes()).expect_err("rejects on GPU");
        let cpu_err = cpu.parse(json.as_bytes()).expect_err("rejects on reference");
        assert_eq!(format!("{gpu_err:?}"), format!("{cpu_err:?}"));
    }

    /// (e) The threshold boundary through the full pipeline: raw_len ==
    /// threshold stays on the GPU, threshold + 1 takes the valve; both
    /// bit-match the reference.
    #[test]
    fn valve_threshold_boundary_routes_and_bit_matches() {
        let Some(gpu) =
            common::gpu_parser_or_skip("valve_threshold_boundary_routes_and_bit_matches")
        else {
            return;
        };
        let Some((ctx, pipeline)) =
            pipeline_or_skip("valve_threshold_boundary_routes_and_bit_matches")
        else {
            return;
        };
        let cpu = common::cpu_parser();

        let at = LONG_STRING_THRESHOLD as usize;
        let exact = format!("\"{}\"", "a".repeat(at));
        assert!(
            accepted_fixups(&ctx, &pipeline, exact.as_bytes()).is_empty(),
            "raw_len == threshold stays on the GPU"
        );
        diff_backends(&gpu, &cpu, exact.as_bytes(), "exactly at threshold");

        let over = format!("\"{}\"", "a".repeat(at + 1));
        assert_eq!(
            accepted_fixups(&ctx, &pipeline, over.as_bytes()),
            vec![0],
            "raw_len == threshold + 1 takes the valve"
        );
        diff_backends(&gpu, &cpu, over.as_bytes(), "one over threshold");
    }
}

// --- 6b. M5: pool reuse, poison, zero-copy input -----------------------------------

/// The M5 structural-overhead work changed three contracts this module
/// pins: (a) pooled buffers are reused **without** whole-buffer zero
/// fills — a parse must be bit-exact over arbitrary garbage (poison) in
/// every checked-out buffer, with reachable slot gaps zero-filled by the
/// producers; (b) `Document`s read the shared GPU buffers zero-copy and
/// own them until drop (on any thread, parser dead or alive); (c) the
/// file/aligned input paths (`parse_file` read-into-pool, `parse_aligned`
/// bytesNoCopy, unsafe `parse_file_mmap`) are verdict- and tape-equal to
/// the copied path on every seam shape.
mod m5_reuse_and_zero_copy {
    use std::io::Write;

    use metal_json::gpu::{GpuInput, GpuParse, GpuPipeline};
    use metal_json::metal::MetalContext;
    use metal_json::pool::ScratchPool;
    use metal_json::AlignedInput;

    use super::{assert_raw_tape_and_records_eq, assert_tape_and_records_eq, common, file_name};

    /// GPU gating for the pipeline-level tests, as in `long_strings`.
    fn pipeline_or_skip(test: &str) -> Option<(MetalContext, GpuPipeline)> {
        match MetalContext::new() {
            Ok(ctx) => Some((ctx, GpuPipeline::new())),
            Err(err) => {
                if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                    panic!("METAL_JSON_REQUIRE_GPU=1 but no usable Metal device: {err}");
                }
                eprintln!("SKIP {test}: no usable Metal device here ({err})");
                None
            }
        }
    }

    /// Small inputs covering every CB3 producer (containers, fast/escaped
    /// strings, numbers incl. a fixup-path literal, literals, root
    /// scalars) plus rejections from every stage.
    fn fixtures() -> Vec<(&'static str, Vec<u8>)> {
        vec![
            ("worked example", br#"{"a":[1,2.5],"b":"x\n"}"#.to_vec()),
            ("root int", b"42".to_vec()),
            ("root string", br#""xA""#.to_vec()),
            ("root true", b"true".to_vec()),
            ("literals", b"[true,false,null]".to_vec()),
            (
                "fixup number",
                format!(
                    r#"{{"k":[{},2],"s":"v"}}"#,
                    "1.00000000000000011102230246251565404236316680908203125"
                )
                .into_bytes(),
            ),
            (
                "escape dense",
                r#"["A\n\\","😀",{"\"":"\t"}]"#.as_bytes().to_vec(),
            ),
            ("utf8 reject", b"ab\x80".to_vec()),
            ("layer1 reject", b"[1 true]".to_vec()),
            ("structure reject", b"[1".to_vec()),
            ("number reject", b"[01]".to_vec()),
            ("string reject", br#"["\q"]"#.to_vec()),
            ("empty", Vec::new()),
        ]
    }

    /// THE poison test: one shared pool across the whole corpus + fixture
    /// sweep, free buffers filled with 0xDB before every parse. Bit-exact
    /// tapes and **whole string buffers** (and identical verdicts on
    /// rejects) prove no kernel or CPU step reads a byte it did not write
    /// AND that every reachable byte of an accepted parse — slot gaps
    /// included — is written: the 0xDB poison must never survive into
    /// gap bytes (asserted explicitly, independent of the reference).
    /// This is the contract that let the M5 work drop the whole-buffer
    /// tape/stringbuf zero fills and pool buffers without resets.
    #[test]
    fn pooled_parses_over_poisoned_buffers_are_bit_exact() {
        let Some((ctx, pipeline)) =
            pipeline_or_skip("pooled_parses_over_poisoned_buffers_are_bit_exact")
        else {
            return;
        };
        let cpu = common::cpu_parser();
        let pool = ScratchPool::new();

        let mut cases: Vec<(String, Vec<u8>)> = fixtures()
            .into_iter()
            .map(|(label, bytes)| (label.to_owned(), bytes))
            .collect();
        for path in common::corpus_files() {
            let bytes = std::fs::read(&path).expect("readable corpus fixture");
            cases.push((file_name(&path).to_owned(), bytes));
        }

        // Two sweeps over everything: the first grows the pool (different
        // sizes contaminate each other's buffers), the second re-checks
        // every input over a fully warmed, poisoned pool.
        for sweep in 0..2 {
            for (label, input) in &cases {
                let label = format!("{label} (sweep {sweep})");
                pool.poison_free_buffers(0xDB);
                let parse = pipeline
                    .run_pooled(&ctx, &pool, GpuInput::Bytes(input), 1024)
                    .unwrap_or_else(|e| panic!("{label}: pipeline failed: {e}"));
                match (parse, cpu.parse(input)) {
                    (GpuParse::Accepted(out), Ok(cpu_doc)) => {
                        let strings = out.stringbuf.as_ref().map_or(&[][..], |b| b.contents());
                        assert_raw_tape_and_records_eq(
                            out.tape.as_slice::<u64>(),
                            strings,
                            &cpu_doc,
                            &label,
                        );
                        // Poison must not survive into reachable gap bytes
                        // (asserted off the GPU artifacts alone, on top of
                        // the whole-buffer reference equality above).
                        super::assert_string_gaps_zero(
                            out.tape.as_slice::<u64>(),
                            strings,
                            &label,
                        );
                        // Hand the document buffers back so the next parse
                        // reuses (and re-poisons) them.
                        pool.put_back(out.tape);
                        if let Some(buf) = out.stringbuf {
                            pool.put_back(buf);
                        }
                    }
                    (GpuParse::Rejected(_), Err(_)) => {} // verdict parity; WHICH may differ
                    (GpuParse::Accepted(_), Err(e)) => {
                        panic!("{label}: GPU accepted, reference rejected ({e})")
                    }
                    (GpuParse::Rejected(packed), Ok(_)) => panic!(
                        "{label}: GPU rejected ({:?}), reference accepted",
                        (packed >> 32, packed as u32)
                    ),
                }
            }
        }
        assert!(pool.free_len() > 0, "the pool must actually be reused");
    }

    /// Pool reuse through the public `Parser`: one parser, every corpus
    /// file parsed twice with size-mixed interleaving (so checked-out
    /// buffers carry other documents' garbage), every result bit-exact vs
    /// the reference — and steady state allocates nothing new (the pool's
    /// free list stops growing).
    #[test]
    fn parser_buffer_reuse_across_sizes_is_bit_exact() {
        let Some(gpu) = common::gpu_parser_or_skip("parser_buffer_reuse_across_sizes_is_bit_exact")
        else {
            return;
        };
        let cpu = common::cpu_parser();
        let files: Vec<_> = common::corpus_files();
        assert!(!files.is_empty());

        for round in 0..3 {
            // Alternate sweep direction so sizes interleave both ways.
            let order: Vec<_> = if round % 2 == 0 {
                files.iter().collect()
            } else {
                files.iter().rev().collect()
            };
            for path in order {
                let bytes = std::fs::read(path).expect("readable corpus fixture");
                let label = format!("{} (round {round})", file_name(path));
                let gpu_doc = gpu
                    .parse(&bytes)
                    .unwrap_or_else(|e| panic!("{label}: GPU parse failed: {e}"));
                let cpu_doc = cpu.parse(&bytes).expect("corpus parses on the reference");
                assert_tape_and_records_eq(&gpu_doc, &cpu_doc, &label);
                common::assert_docs_eq(gpu_doc.root(), cpu_doc.root(), &label);
            }
        }
    }

    /// `parse_aligned` (bytesNoCopy over a caller-held `AlignedInput`) is
    /// tape- and verdict-equal to the copied path on the corpus and on
    /// every padding seam shape: lengths ≡ 0/1/63 (mod 64), an exact
    /// page-multiple document, a trailing root scalar at EOF, and a
    /// multi-byte UTF-8 sequence truncated exactly at EOF.
    #[test]
    fn parse_aligned_matches_parse_on_corpus_and_seams() {
        let Some(gpu) = common::gpu_parser_or_skip("parse_aligned_matches_parse_on_corpus_and_seams")
        else {
            return;
        };

        let mut cases: Vec<(String, Vec<u8>)> = Vec::new();
        for path in common::corpus_files() {
            cases.push((
                file_name(&path).to_owned(),
                std::fs::read(&path).expect("readable corpus fixture"),
            ));
        }
        // Seam shapes: pad with trailing spaces (legal JSON whitespace) to
        // hit exact word/page boundaries.
        let pad_to = |json: &[u8], target: usize| {
            let mut v = json.to_vec();
            assert!(v.len() <= target);
            v.resize(target, b' ');
            v
        };
        cases.push(("len 64k".into(), pad_to(br#"[1,2,3]"#, 64)));
        cases.push(("len 65".into(), pad_to(br#"[1,2,3]"#, 65)));
        cases.push(("len 63".into(), pad_to(br#"[1,2,3]"#, 63)));
        cases.push(("len page".into(), pad_to(br#"{"k":"v"}"#, 16384)));
        cases.push(("scalar at EOF".into(), b"12345".to_vec()));
        cases.push(("truncated utf8 at EOF".into(), b"[\"a\xC3".to_vec()));
        cases.push(("empty".into(), Vec::new()));

        for (label, bytes) in &cases {
            let aligned = AlignedInput::from_slice(bytes);
            match (gpu.parse_aligned(&aligned), gpu.parse(bytes)) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a.tape(), b.tape(), "{label}: tape (aligned vs copied)");
                    common::assert_docs_eq(a.root(), b.root(), label);
                }
                (Err(a), Err(b)) => assert_eq!(
                    format!("{a:?}"),
                    format!("{b:?}"),
                    "{label}: error parity (aligned vs copied)"
                ),
                (a, b) => panic!(
                    "{label}: verdicts diverge — aligned {:?} vs copied {:?}",
                    a.map(|d| d.tape().len()),
                    b.map(|d| d.tape().len())
                ),
            }
        }
    }

    /// `parse_file` (the safe read-into-pooled-buffer path) matches
    /// `parse` on every corpus file plus the tail edge cases: a document
    /// of exactly one page (no tail to pad), a 64-multiple length, a
    /// truncated UTF-8 sequence at EOF (the space padding must keep the
    /// reference offset), empty, and a trailing root scalar at EOF.
    #[test]
    fn parse_file_matches_parse_on_corpus_and_edges() {
        let Some(gpu) = common::gpu_parser_or_skip("parse_file_matches_parse_on_corpus_and_edges")
        else {
            return;
        };

        for path in common::corpus_files() {
            let bytes = std::fs::read(&path).expect("readable corpus fixture");
            let label = file_name(&path).to_owned();
            let from_file = gpu
                .parse_file(&path)
                .unwrap_or_else(|e| panic!("{label}: parse_file failed: {e}"));
            let from_bytes = gpu.parse(&bytes).expect("corpus parses");
            assert_eq!(from_file.tape(), from_bytes.tape(), "{label}: tape");
            common::assert_docs_eq(from_file.root(), from_bytes.root(), &label);
        }

        // Edge files in a temp dir: exact-page length (no tail to pad), a
        // 64-multiple length, truncated UTF-8 at EOF, and empty.
        let dir = std::env::temp_dir().join(format!("metal-json-m5-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let mut page_doc = br#"{"k":"v"}"#.to_vec();
        page_doc.resize(16384, b' ');
        let mut word_doc = br#"[1,2,3]"#.to_vec();
        word_doc.resize(128, b' ');
        let edge_cases: Vec<(&str, Vec<u8>)> = vec![
            ("page.json", page_doc),
            ("word.json", word_doc),
            ("trunc-utf8.json", b"[\"a\xC3".to_vec()),
            ("empty.json", Vec::new()),
            ("scalar.json", b"12345".to_vec()),
        ];
        for (name, bytes) in &edge_cases {
            let path = dir.join(name);
            let mut f = std::fs::File::create(&path).expect("temp file");
            f.write_all(bytes).expect("write temp file");
            drop(f);
            match (gpu.parse_file(&path), gpu.parse(bytes)) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a.tape(), b.tape(), "{name}: tape (file vs bytes)");
                    common::assert_docs_eq(a.root(), b.root(), name);
                }
                (Err(a), Err(b)) => assert_eq!(
                    format!("{a:?}"),
                    format!("{b:?}"),
                    "{name}: error parity (file vs bytes)"
                ),
                (a, b) => panic!(
                    "{name}: verdicts diverge — file {:?} vs bytes {:?}",
                    a.map(|d| d.tape().len()),
                    b.map(|d| d.tape().len())
                ),
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The unsafe mmap variant, `parse_file_mmap` (zero-copy, COW tail
    /// padding): tape- and verdict-equal to both `parse_file` and `parse`
    /// on temp files covering the same tail edge cases. The temp files are
    /// private to this test, so the # Safety contract (no concurrent
    /// truncation/modification) holds trivially.
    #[test]
    fn parse_file_mmap_matches_parse_file_on_edges() {
        let Some(gpu) = common::gpu_parser_or_skip("parse_file_mmap_matches_parse_file_on_edges")
        else {
            return;
        };

        let dir =
            std::env::temp_dir().join(format!("metal-json-m5-mmap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let mut page_doc = br#"{"k":"v"}"#.to_vec();
        page_doc.resize(16384, b' ');
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("worked.json", br#"{"a":[1,2.5],"b":"x\n"}"#.to_vec()),
            ("page.json", page_doc),
            ("trunc-utf8.json", b"[\"a\xC3".to_vec()),
            ("empty.json", Vec::new()),
            ("scalar.json", b"12345".to_vec()),
            ("reject.json", br#"["\q"]"#.to_vec()),
        ];
        for (name, bytes) in &cases {
            let path = dir.join(name);
            let mut f = std::fs::File::create(&path).expect("temp file");
            f.write_all(bytes).expect("write temp file");
            drop(f);
            // SAFETY: the file is private to this test and not touched
            // again until the call returns.
            let mmap_result = unsafe { gpu.parse_file_mmap(&path) };
            match (mmap_result, gpu.parse_file(&path), gpu.parse(bytes)) {
                (Ok(m), Ok(f), Ok(b)) => {
                    assert_eq!(m.tape(), b.tape(), "{name}: tape (mmap vs bytes)");
                    assert_eq!(m.tape(), f.tape(), "{name}: tape (mmap vs parse_file)");
                    assert_eq!(
                        m.strings().as_bytes(),
                        b.strings().as_bytes(),
                        "{name}: string buffer (mmap vs bytes)"
                    );
                    common::assert_docs_eq(m.root(), b.root(), name);
                }
                (Err(m), Err(f), Err(b)) => {
                    assert_eq!(format!("{m:?}"), format!("{b:?}"), "{name}: mmap vs bytes");
                    assert_eq!(format!("{m:?}"), format!("{f:?}"), "{name}: mmap vs file");
                }
                (m, f, b) => panic!(
                    "{name}: verdicts diverge — mmap ok={}, parse_file ok={}, bytes ok={}",
                    m.is_ok(),
                    f.is_ok(),
                    b.is_ok()
                ),
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Zero-copy `Document`s are self-contained `Send + Sync` values: they
    /// outlive the parser that produced them, move to other threads, read
    /// identically there, and drop there (returning their buffers to the
    /// still-alive pool through the `Arc` handle).
    #[test]
    fn documents_outlive_the_parser_and_drop_on_other_threads() {
        let doc = {
            let Some(gpu) =
                common::gpu_parser_or_skip("documents_outlive_the_parser_and_drop_on_other_threads")
            else {
                return;
            };
            let doc = gpu
                .parse(br#"{"k":[1,"s",2.5],"x":"y"}"#)
                .expect("document parses");
            // Parser (device, pipelines, pool handle) drops HERE; the
            // document must stay fully readable.
            doc
        };
        assert_eq!(doc.root().get("k").unwrap().at(1).unwrap().as_str(), Some("s"));

        let handle = std::thread::spawn(move || {
            let k = doc.root().get("k").unwrap();
            assert_eq!(k.at(0).unwrap().as_i64(), Some(1));
            assert_eq!(k.at(2).unwrap().as_f64(), Some(2.5));
            assert_eq!(doc.root().get("x").unwrap().as_str(), Some("y"));
            drop(doc); // pool return on a foreign thread
        });
        handle.join().expect("cross-thread document use must not panic");
    }
}

// --- 7. Manual timing prints (informational only; M5 owns perf) --------------------

/// Deterministic single-line synthetic document of at least `target`
/// bytes (the tests/kernels.rs / tests/structure.rs generator shape):
/// escape-heavy members whose lengths drift so quotes and runs hit every
/// seam alignment.
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

/// Full `Parser::parse` wall-time GB/s on a ~256 MB synthetic document —
/// informational only (M5 owns perf and the GPU-timestamp breakdown). Run:
/// `cargo test --release --features cpu-reference --test gpu_e2e -- --ignored --nocapture timing`
#[test]
#[ignore = "manual: allocates ~256 MB and prints GB/s; informational only — run with --release"]
fn timing_full_gpu_parse_on_the_256mb_synthetic() {
    let Some(gpu) = common::gpu_parser_or_skip("timing_full_gpu_parse_on_the_256mb_synthetic")
    else {
        return;
    };
    let input = synthetic_single_line(256 * 1024 * 1024);

    // Warm-up: PSO creation, first-touch page faults, GPU power ramp.
    let doc = gpu.parse(&input).expect("synthetic doc must parse");
    assert!(doc.root().len().is_some_and(|n| n > 100_000));
    drop(doc);

    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let start = std::time::Instant::now();
        let doc = gpu.parse(&input).expect("synthetic doc must parse");
        let secs = start.elapsed().as_secs_f64();
        std::hint::black_box(&doc);
        best = best.min(secs);
    }
    println!(
        "full GPU parse (Parser::parse wall, best of 3): {:.1} ms = {:.2} GB/s over {} bytes \
         (includes CPU syncs, exact allocations, fixup patch and the M5-pending tape/string \
         copy-out — informational only, M5 owns perf)",
        best * 1e3,
        input.len() as f64 / best / 1e9,
        input.len(),
    );
}

/// Parser-level wall time on the checked-in twitter-like 100 KB corpus
/// file — the small-document sanity print (fixed pipeline overhead
/// dominates at this size; see docs/spikes.md spike C).
#[test]
#[ignore = "manual: prints wall time; informational only"]
fn timing_twitter_like_100kb_corpus_file() {
    let Some(gpu) = common::gpu_parser_or_skip("timing_twitter_like_100kb_corpus_file") else {
        return;
    };
    let bytes = std::fs::read(common::corpus_dir().join("twitter_like_100kb.json"))
        .expect("corpus fixture is checked in");

    let doc = gpu.parse(&bytes).expect("fixture parses"); // warm-up
    drop(doc);

    let mut best = f64::INFINITY;
    for _ in 0..20 {
        let start = std::time::Instant::now();
        let doc = gpu.parse(&bytes).expect("fixture parses");
        let secs = start.elapsed().as_secs_f64();
        std::hint::black_box(&doc);
        best = best.min(secs);
    }
    println!(
        "twitter_like_100kb.json (Parser::parse wall, best of 20): {:.0} µs over {} bytes \
         = {:.3} GB/s (fixed ~0.5 ms pipeline overhead dominates at this size — \
         informational only)",
        best * 1e6,
        bytes.len(),
        bytes.len() as f64 / best / 1e9,
    );
}
