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
//!    the GPU tape words equal the reference tape words AND every string
//!    record (`[u32 LE len][content][NUL]`) is byte-equal at its tape
//!    offset — gap bytes excluded per the pinned policy in
//!    `docs/tape-format.md`.
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
/// included — the raw-length prefix-sum allocation makes them equal), and
/// at every `"` word's offset the full `[u32 LE len][content][NUL]` record
/// byte-equal. Bytes BETWEEN records (gaps) are unspecified on the GPU and
/// never compared. Returns the number of string records compared.
fn assert_tape_and_records_eq(gpu: &Document, cpu: &Document, label: &str) -> usize {
    assert_eq!(
        gpu.tape(),
        cpu.tape(),
        "{label}: tape words must be bit-identical"
    );
    let gpu_bytes = gpu.strings().as_bytes();
    let cpu_bytes = cpu.strings().as_bytes();
    assert_eq!(
        gpu_bytes.len(),
        cpu_bytes.len(),
        "{label}: string buffer size (raw-length prefix-sum total)"
    );
    let mut records = 0usize;
    let tape = gpu.tape();
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
/// the GPU tape words equal the reference tape words bit-for-bit AND every
/// string record is byte-equal at its tape offset (gap bytes excluded per
/// the pinned policy in docs/tape-format.md).
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

// --- 6. Manual timing prints (informational only; M5 owns perf) --------------------

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
