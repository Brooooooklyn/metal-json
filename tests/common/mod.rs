//! Shared helpers for the M1 correctness suites (compiled into each test
//! crate via `mod common;`). Everything here assumes the `cpu-reference`
//! feature: the suites drive [`Backend::CpuReference`], the scalar oracle
//! the GPU backend will be diffed against in M2-M4.
#![allow(dead_code)] // each test crate uses a different subset

pub mod numbers;

use std::path::{Path, PathBuf};

use metal_json::{Backend, Parser, ParserOptions, Value, ValueKind};

/// A parser running the CPU reference pipeline with default options.
pub fn cpu_parser() -> Parser {
    let mut opts = ParserOptions::default();
    opts.backend = Backend::CpuReference;
    Parser::with_options(opts).expect("CPU reference parser construction cannot fail")
}

/// A parser on the default GPU backend, or `None` (with a loud skip
/// message) when no usable Metal device exists — unless
/// `METAL_JSON_REQUIRE_GPU=1` (set in CI) makes that a hard failure.
pub fn gpu_parser_or_skip(test: &str) -> Option<Parser> {
    match Parser::new() {
        Ok(parser) => Some(parser),
        Err(err) => {
            if std::env::var_os("METAL_JSON_REQUIRE_GPU").is_some_and(|v| v == "1") {
                panic!("METAL_JSON_REQUIRE_GPU=1 but no usable Metal device: {err}");
            }
            eprintln!("SKIP {test}: no usable Metal device here ({err})");
            None
        }
    }
}

/// Recursively assert two parsed documents agree: kinds, scalar values
/// (doubles bit-for-bit), string contents, array elements in order, object
/// members in order (keys included — duplicates and all). The GPU-vs-
/// reference counterpart of [`assert_doc_eq`].
pub fn assert_docs_eq(got: Value<'_>, want: Value<'_>, path: &str) {
    assert_eq!(got.kind(), want.kind(), "{path}: kind");
    match want.kind() {
        ValueKind::Null => assert!(got.is_null(), "{path}: null"),
        ValueKind::Bool => assert_eq!(got.as_bool(), want.as_bool(), "{path}: bool"),
        ValueKind::Int64 => assert_eq!(got.as_i64(), want.as_i64(), "{path}: i64"),
        ValueKind::UInt64 => assert_eq!(got.as_u64(), want.as_u64(), "{path}: u64"),
        ValueKind::Double => assert_eq!(
            got.as_f64().map(f64::to_bits),
            want.as_f64().map(f64::to_bits),
            "{path}: f64 bits"
        ),
        ValueKind::String => assert_eq!(got.as_str(), want.as_str(), "{path}: string"),
        ValueKind::Array => {
            assert_eq!(got.len(), want.len(), "{path}: array length");
            for (i, (g, w)) in got.elements().zip(want.elements()).enumerate() {
                assert_docs_eq(g, w, &format!("{path}[{i}]"));
            }
        }
        ValueKind::Object => {
            assert_eq!(got.len(), want.len(), "{path}: member count");
            for (i, ((gk, gv), (wk, wv))) in got.entries().zip(want.entries()).enumerate() {
                assert_eq!(gk, wk, "{path}: key #{i}");
                assert_docs_eq(gv, wv, &format!("{path}.{}", wk.escape_debug()));
            }
        }
    }
}

/// The checked-in corpus directory (always present in the repo).
pub fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus")
}

/// All corpus fixture paths, sorted by file name for deterministic order.
pub fn corpus_files() -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(corpus_dir())
        .expect("corpus/ is checked in")
        .map(|e| e.expect("readable corpus entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "corpus/ must contain fixtures");
    files
}

/// `data/JSONTestSuite/test_parsing`, or `None` (with a loud warning) when
/// the gitignored corpus has not been fetched. Tests that return early here
/// still pass — run `scripts/fetch_jsontestsuite.sh` for full coverage.
pub fn jsontestsuite_dir() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/JSONTestSuite/test_parsing");
    if dir.is_dir() {
        Some(dir)
    } else {
        eprintln!(
            "\n==============================================================\n\
             WARNING: JSONTestSuite corpus not found at\n  {}\n\
             SKIPPING the conformance run. Fetch it with:\n  \
             bash scripts/fetch_jsontestsuite.sh\n\
             ==============================================================\n",
            dir.display()
        );
        None
    }
}

/// JSONTestSuite files whose name starts with `prefix`, sorted.
pub fn jsontestsuite_files(dir: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("readable test_parsing dir")
        .map(|e| e.expect("readable entry").path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "json")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(prefix))
        })
        .collect();
    files.sort();
    files
}

/// True if any object anywhere in `value` has two members with the same
/// key. Such documents are excluded from serde_json comparison: serde's map
/// keeps only the last duplicate, while our tape keeps all members verbatim
/// (simdjson parity); the tape behavior is asserted separately.
pub fn has_duplicate_keys(value: Value<'_>) -> bool {
    match value.kind() {
        ValueKind::Object => {
            let keys: Vec<&str> = value.entries().map(|(k, _)| k).collect();
            for (i, k) in keys.iter().enumerate() {
                if keys[..i].contains(k) {
                    return true;
                }
            }
            value.entries().any(|(_, v)| has_duplicate_keys(v))
        }
        ValueKind::Array => value.elements().any(has_duplicate_keys),
        _ => false,
    }
}

/// Pinned verdict (`true` = accept) and rationale for every `i_*`
/// JSONTestSuite file. These are implementation-defined by JSONTestSuite;
/// ours mirror simdjson — and they are SHARED between the CPU conformance
/// suite (`tests/jsontestsuite.rs`) and the GPU milestone gate
/// (`tests/gpu_e2e.rs`), so the two backends cannot drift apart:
///
/// - **number precision/overflow**: grammar-valid numbers whose value
///   overflows to ±inf are REJECTED (simdjson rejects infinities);
///   underflow to 0.0 and precision loss on huge mantissas are ACCEPTED
///   (value = `str::parse::<f64>()`, correctly rounded);
/// - **integers beyond u64**: fall to the double path and are ACCEPTED
///   with the correctly rounded f64 value;
/// - **invalid UTF-8 / UTF-16 input**: REJECTED — the contract validates
///   full UTF-8 (Error::Utf8);
/// - **lone/inverted `\uXXXX` surrogate escapes**: REJECTED
///   (InvalidStringEscape), like simdjson;
/// - **depth 500**: ACCEPTED (limit is 1024, simdjson parity);
/// - **UTF-8 BOM**: REJECTED (0xEF cannot start a scalar), like simdjson
///   without its BOM-stripping front end.
pub const I_FILE_VERDICTS: &[(&str, bool, &str)] = &[
    (
        "i_number_double_huge_neg_exp.json",
        true,
        "123.456e-789 underflows to 0.0; underflow is accepted",
    ),
    (
        "i_number_huge_exp.json",
        false,
        "0.4e+00669...9 overflows to inf; overflow is rejected (InvalidNumber)",
    ),
    (
        "i_number_neg_int_huge_exp.json",
        false,
        "-1e+9999 overflows to -inf; rejected",
    ),
    (
        "i_number_pos_double_huge_exp.json",
        false,
        "1.5e+9999 overflows to inf; rejected",
    ),
    (
        "i_number_real_neg_overflow.json",
        false,
        "-123123e100000 overflows to -inf; rejected",
    ),
    (
        "i_number_real_pos_overflow.json",
        false,
        "123123e100000 overflows to inf; rejected",
    ),
    (
        "i_number_real_underflow.json",
        true,
        "123e-10000000 underflows to 0.0; accepted",
    ),
    (
        "i_number_too_big_neg_int.json",
        true,
        "30-digit negative integer: beyond i64, parses as correctly rounded f64",
    ),
    (
        "i_number_too_big_pos_int.json",
        true,
        "21-digit integer: beyond u64, parses as correctly rounded f64",
    ),
    (
        "i_number_very_big_negative_int.json",
        true,
        "27-digit negative integer: parses as correctly rounded f64",
    ),
    (
        "i_object_key_lone_2nd_surrogate.json",
        false,
        "lone low-surrogate escape in a key: InvalidStringEscape",
    ),
    (
        "i_string_1st_surrogate_but_2nd_missing.json",
        false,
        "high-surrogate escape with no low surrogate: InvalidStringEscape",
    ),
    (
        "i_string_1st_valid_surrogate_2nd_invalid.json",
        false,
        "high surrogate chased by a non-surrogate escape: InvalidStringEscape",
    ),
    (
        "i_string_incomplete_surrogate_and_escape_valid.json",
        false,
        "high surrogate chased by \\n: InvalidStringEscape",
    ),
    (
        "i_string_incomplete_surrogate_pair.json",
        false,
        "lone low-surrogate escape: InvalidStringEscape",
    ),
    (
        "i_string_incomplete_surrogates_escape_valid.json",
        false,
        "two high surrogates in a row: InvalidStringEscape",
    ),
    (
        "i_string_invalid_lonely_surrogate.json",
        false,
        "lone high-surrogate escape: InvalidStringEscape",
    ),
    (
        "i_string_invalid_surrogate.json",
        false,
        "high surrogate followed by plain characters: InvalidStringEscape",
    ),
    (
        "i_string_invalid_utf-8.json",
        false,
        "raw 0xFF byte: Error::Utf8",
    ),
    (
        "i_string_inverted_surrogates_U+1D11E.json",
        false,
        "low surrogate before high surrogate: InvalidStringEscape",
    ),
    (
        "i_string_iso_latin_1.json",
        false,
        "Latin-1 (non-UTF-8) byte: Error::Utf8",
    ),
    (
        "i_string_lone_second_surrogate.json",
        false,
        "lone low-surrogate escape: InvalidStringEscape",
    ),
    (
        "i_string_lone_utf8_continuation_byte.json",
        false,
        "stray continuation byte: Error::Utf8",
    ),
    (
        "i_string_not_in_unicode_range.json",
        false,
        "encodes a code point above U+10FFFF: Error::Utf8",
    ),
    (
        "i_string_overlong_sequence_2_bytes.json",
        false,
        "overlong 2-byte encoding: Error::Utf8",
    ),
    (
        "i_string_overlong_sequence_6_bytes.json",
        false,
        "6-byte (pre-2003) UTF-8 sequence: Error::Utf8",
    ),
    (
        "i_string_overlong_sequence_6_bytes_null.json",
        false,
        "overlong 6-byte encoding of NUL: Error::Utf8",
    ),
    (
        "i_string_truncated-utf-8.json",
        false,
        "truncated multibyte sequence: Error::Utf8",
    ),
    (
        "i_string_UTF-16LE_with_BOM.json",
        false,
        "UTF-16 input: not valid UTF-8, Error::Utf8",
    ),
    (
        "i_string_UTF-8_invalid_sequence.json",
        false,
        "invalid UTF-8 sequence: Error::Utf8",
    ),
    (
        "i_string_utf16BE_no_BOM.json",
        false,
        "UTF-16BE input: not valid UTF-8, Error::Utf8",
    ),
    (
        "i_string_utf16LE_no_BOM.json",
        false,
        "UTF-16LE input: not valid UTF-8, Error::Utf8",
    ),
    (
        "i_string_UTF8_surrogate_U+D800.json",
        false,
        "raw UTF-8-encoded surrogate: Error::Utf8",
    ),
    (
        "i_structure_500_nested_arrays.json",
        true,
        "500 < the 1024 depth limit (simdjson parity); parses fine",
    ),
    (
        "i_structure_UTF-8_BOM_empty_object.json",
        false,
        "UTF-8 BOM before {}: 0xEF cannot start a scalar (UnexpectedToken), \
         matching simdjson's no-BOM stance",
    ),
];

/// Recursively assert that our tape walk of `ours` matches serde_json's
/// parse of the same document.
///
/// - objects compare as **ordered** entry sequences (the serde_json
///   `preserve_order` feature is enabled, so both sides are in document
///   order); the caller must exclude duplicate-key documents first;
/// - arrays compare element-wise;
/// - strings compare exactly (both sides fully unescaped);
/// - numbers compare **by kind**: serde's `arbitrary_precision` feature
///   keeps the raw literal, and its `as_i64`/`as_u64` parse that literal —
///   which matches our tape's type-selection contract (integer literal that
///   fits `i64` → `l`, else fits `u64` → `u`, else `d`) exactly. `i64`/`u64`
///   compare for equality; doubles compare **bit-for-bit** against
///   `str::parse::<f64>()` of the raw literal (the oracle the tape contract
///   names).
pub fn assert_doc_eq(ours: Value<'_>, serde: &serde_json::Value, path: &str) {
    match serde {
        serde_json::Value::Null => {
            assert!(ours.is_null(), "{path}: expected null, got {ours:?}");
        }
        serde_json::Value::Bool(b) => {
            assert_eq!(ours.as_bool(), Some(*b), "{path}: bool mismatch");
        }
        serde_json::Value::Number(n) => {
            // arbitrary_precision: `n` holds the raw literal text; as_i64 /
            // as_u64 are str::parse on it, mirroring our tape type rules.
            if let Some(i) = n.as_i64() {
                assert_eq!(
                    ours.kind(),
                    ValueKind::Int64,
                    "{path}: kind for integer literal {n}"
                );
                assert_eq!(ours.as_i64(), Some(i), "{path}: i64 value");
            } else if let Some(u) = n.as_u64() {
                assert_eq!(
                    ours.kind(),
                    ValueKind::UInt64,
                    "{path}: kind for big integer literal {n}"
                );
                assert_eq!(ours.as_u64(), Some(u), "{path}: u64 value");
            } else {
                assert_eq!(
                    ours.kind(),
                    ValueKind::Double,
                    "{path}: kind for double literal {n}"
                );
                let raw = n.to_string(); // the raw literal under arbitrary_precision
                let oracle: f64 = raw
                    .parse()
                    .unwrap_or_else(|e| panic!("{path}: oracle parse of {raw:?} failed: {e}"));
                assert_eq!(
                    ours.as_f64().map(f64::to_bits),
                    Some(oracle.to_bits()),
                    "{path}: f64 bits for literal {raw:?} (got {:?}, want {oracle:?})",
                    ours.as_f64(),
                );
            }
        }
        serde_json::Value::String(s) => {
            assert_eq!(ours.as_str(), Some(s.as_str()), "{path}: string mismatch");
        }
        serde_json::Value::Array(items) => {
            assert_eq!(ours.kind(), ValueKind::Array, "{path}: expected array");
            assert_eq!(ours.len(), Some(items.len()), "{path}: array length");
            let ours_items: Vec<Value<'_>> = ours.elements().collect();
            assert_eq!(ours_items.len(), items.len(), "{path}: element count");
            for (i, (v, s)) in ours_items.iter().zip(items).enumerate() {
                assert_doc_eq(*v, s, &format!("{path}[{i}]"));
            }
        }
        serde_json::Value::Object(members) => {
            assert_eq!(ours.kind(), ValueKind::Object, "{path}: expected object");
            assert_eq!(ours.len(), Some(members.len()), "{path}: member count");
            let entries: Vec<(&str, Value<'_>)> = ours.entries().collect();
            assert_eq!(entries.len(), members.len(), "{path}: entry count");
            // preserve_order: serde's map iterates in document order too.
            for ((our_key, our_value), (serde_key, serde_value)) in entries.iter().zip(members) {
                assert_eq!(our_key, serde_key, "{path}: key order/spelling");
                assert_doc_eq(
                    *our_value,
                    serde_value,
                    &format!("{path}.{}", serde_key.escape_debug()),
                );
            }
        }
    }
}
