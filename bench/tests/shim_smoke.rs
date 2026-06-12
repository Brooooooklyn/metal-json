//! FFI smoke test: parse twitter.json through the C++ simdjson shim and
//! check its tape-walk stats against an independent serde_json count.

use metal_json_bench::{SjParser, data_dir, load, load_padded};

/// serde_json oracle for the shim's stats: (node_count, string_bytes).
///
/// node_count: every scalar, every object key, and every container counts 1.
/// string_bytes: unescaped byte length of every string (keys and values).
fn count(v: &serde_json::Value) -> (u64, u64) {
    match v {
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
            (1, 0)
        }
        serde_json::Value::String(s) => (1, s.len() as u64),
        serde_json::Value::Array(items) => items.iter().map(count).fold((1, 0), |acc, x| {
            (acc.0 + x.0, acc.1 + x.1)
        }),
        serde_json::Value::Object(map) => map.iter().map(|(k, val)| {
            let (n, s) = count(val);
            (n + 1, s + k.len() as u64)
        })
        .fold((1, 0), |acc, x| (acc.0 + x.0, acc.1 + x.1)),
    }
}

#[test]
fn shim_stats_match_serde_json_on_twitter() {
    let path = data_dir().join("twitter.json");
    if !path.exists() {
        // Keep CI green before data is fetched; the harness prints how to fix.
        eprintln!(
            "SKIP shim_stats_match_serde_json_on_twitter: {} missing — run `cargo run -p xtask -- fetch-data`",
            path.display()
        );
        return;
    }

    let parser = SjParser::new();
    let padded = load_padded(&path).expect("read twitter.json");
    let stats = parser.parse(&padded).expect("twitter.json must parse");

    assert!(stats.node_count > 0, "node_count must be nonzero");
    assert!(stats.string_bytes > 0, "string_bytes must be nonzero");
    assert_ne!(stats.number_xor, 0, "number_xor must be nonzero");

    let raw = load(&path).expect("read twitter.json");
    let doc: serde_json::Value = serde_json::from_slice(&raw).expect("serde_json parse");
    let (node_count, string_bytes) = count(&doc);

    assert_eq!(stats.node_count, node_count, "node_count vs serde_json");
    assert_eq!(stats.string_bytes, string_bytes, "string_bytes vs serde_json");
}
