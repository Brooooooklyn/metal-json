//! Shared helpers for the M1 correctness suites (compiled into each test
//! crate via `mod common;`). Everything here assumes the `cpu-reference`
//! feature: the suites drive [`Backend::CpuReference`], the scalar oracle
//! the GPU backend will be diffed against in M2-M4.
#![allow(dead_code)] // each test crate uses a different subset

use std::path::{Path, PathBuf};

use metal_json::{Backend, Parser, ParserOptions, Value, ValueKind};

/// A parser running the CPU reference pipeline with default options.
pub fn cpu_parser() -> Parser {
    let mut opts = ParserOptions::default();
    opts.backend = Backend::CpuReference;
    Parser::with_options(opts).expect("CPU reference parser construction cannot fail")
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
