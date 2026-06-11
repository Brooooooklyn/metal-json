//! Number torture tests for the CPU reference backend.
//!
//! The deterministic tables live in `tests/common/numbers.rs`,
//! parameterized over the backend: this suite drives them against
//! `Backend::CpuReference`; `tests/gpu_e2e.rs` drives the SAME tables
//! against the GPU backend (the M4 bit-exactness gate). See that module
//! for the pinned policy decisions (overflow rejected, underflow to signed
//! zero accepted, `-0.0` sign preserved, simdjson type selection).
#![cfg(feature = "cpu-reference")]

mod common;

use metal_json::ValueKind;
use proptest::prelude::*;

#[test]
fn subnormal_normal_boundary_and_range_extremes() {
    common::numbers::subnormal_normal_boundary_and_range_extremes(&common::cpu_parser());
}

#[test]
fn overflow_is_rejected_like_simdjson() {
    common::numbers::overflow_is_rejected_like_simdjson(&common::cpu_parser());
}

#[test]
fn underflow_collapses_to_signed_zero() {
    common::numbers::underflow_collapses_to_signed_zero(&common::cpu_parser());
}

#[test]
fn negative_zero_keeps_its_sign_bit() {
    common::numbers::negative_zero_keeps_its_sign_bit(&common::cpu_parser());
}

#[test]
fn seventeen_digit_round_trips() {
    common::numbers::seventeen_digit_round_trips(&common::cpu_parser());
}

#[test]
fn hundred_plus_digit_mantissas() {
    common::numbers::hundred_plus_digit_mantissas(&common::cpu_parser());
}

#[test]
fn integer_type_selection_boundaries() {
    common::numbers::integer_type_selection_boundaries(&common::cpu_parser());
}

#[test]
fn grammar_rejections() {
    common::numbers::grammar_rejections(&common::cpu_parser());
}

#[test]
fn exponent_edge_forms_parse() {
    common::numbers::exponent_edge_forms_parse(&common::cpu_parser());
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    /// (a) Any finite f64, printed in shortest-round-trip exponential form
    /// (Rust's `LowerExp` is shortest-round-trip and always valid JSON),
    /// parses back to the exact same bit pattern.
    #[test]
    fn random_f64_round_trips_bit_exactly(value in any::<f64>()) {
        prop_assume!(value.is_finite());
        let text = format!("{value:e}");
        // Sanity: the oracle round-trips (Rust formatting guarantee).
        prop_assert_eq!(text.parse::<f64>().unwrap().to_bits(), value.to_bits());

        let parser = common::cpu_parser();
        let doc = common::numbers::parse_root(&parser, &text).expect("shortest form must parse");
        prop_assert_eq!(doc.root().kind(), ValueKind::Double);
        prop_assert_eq!(
            doc.root().as_f64().map(f64::to_bits),
            Some(value.to_bits()),
            "round trip of {}", text
        );
    }
}

/// Strategy for arbitrary JSON documents as `serde_json::Value` (objects in
/// insertion order via `preserve_order`; duplicate keys impossible because
/// maps collapse before serialization).
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
                .prop_map(|pairs| { serde_json::Value::Object(pairs.into_iter().collect()) }),
        ]
    })
}

proptest! {
    /// (b) Any serde-serializable JSON document round-trips through our
    /// parser and matches serde structurally (kinds, order, exact strings,
    /// bit-exact doubles).
    #[test]
    fn random_documents_match_serde(value in arb_json()) {
        let json = serde_json::to_string(&value).expect("serializable");
        let parser = common::cpu_parser();
        let doc = parser
            .parse(json.as_bytes())
            .unwrap_or_else(|e| panic!("serde-serialized JSON must parse: {e}\n{json}"));
        common::assert_doc_eq(doc.root(), &value, "$");
    }
}
