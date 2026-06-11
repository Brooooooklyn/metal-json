//! Number torture tests for the CPU reference backend.
//!
//! Oracle: `str::parse::<f64>()` (correctly rounded), compared
//! **bit-for-bit**. Policy decisions (documented + pinned here):
//!
//! - **overflow**: grammar-valid numbers whose value rounds to ±inf
//!   (`1e309`, ...) are REJECTED as `InvalidNumber` — simdjson rejects
//!   values that overflow to infinity;
//! - **underflow**: values that round to (signed) zero (`1e-400`, ...) are
//!   ACCEPTED — simdjson parity again;
//! - **`-0.0`** keeps its sign bit; integer literal `-0` is `Int64(0)`
//!   (simdjson's integer fast path applies: no `.`/`e`);
//! - type selection: integer literal fitting `i64` → `Int64`, else `u64` →
//!   `UInt64`, else (fraction/exponent/out of range) → `Double`.
#![cfg(feature = "cpu-reference")]

mod common;

use metal_json::{Document, Error, SyntaxErrorKind, ValueKind};
use proptest::prelude::*;

fn parse_root(json: &str) -> Result<Document, Error> {
    common::cpu_parser().parse(json.as_bytes())
}

/// Parse `text` as a root number; it must be a `Double` whose bits equal
/// the `str::parse::<f64>()` oracle.
fn assert_double_matches_oracle(text: &str) {
    let oracle: f64 = text
        .parse()
        .unwrap_or_else(|e| panic!("oracle rejects fixture {text:?}: {e}"));
    assert!(
        oracle.is_finite(),
        "fixture bug: {text:?} is not finite — belongs in the reject table"
    );
    let doc = parse_root(text).unwrap_or_else(|e| panic!("{text:?} must parse: {e}"));
    assert_eq!(
        doc.root().kind(),
        ValueKind::Double,
        "{text:?} must select the double tape kind"
    );
    assert_eq!(
        doc.root().as_f64().map(f64::to_bits),
        Some(oracle.to_bits()),
        "{text:?}: bits must equal str::parse::<f64> (got {:?}, want {oracle:?})",
        doc.root().as_f64()
    );
}

fn assert_rejected_as(text: &str, want: SyntaxErrorKind) {
    match parse_root(text) {
        Err(Error::Syntax { kind, .. }) => {
            assert_eq!(kind, want, "{text:?}: rejection kind");
        }
        other => panic!("{text:?} must be rejected as {want:?}, got {other:?}"),
    }
}

#[test]
fn subnormal_normal_boundary_and_range_extremes() {
    for text in [
        "2.2250738585072011e-308", // largest subnormal-rounding literal (PHP/Java hang bug)
        "2.2250738585072014e-308", // smallest normal
        "2.2250738585072012e-308", // between: rounds to min normal
        "5e-324",                  // smallest subnormal
        "4.9e-324",
        "2.4703282292062327e-324", // just below half the smallest subnormal: 0.0
        "2.4703282292062328e-324", // just above: rounds up to 5e-324
        "1.7976931348623157e308",  // largest finite f64
        "1.7976931348623158e308",  // rounds back down to f64::MAX
        "1e308",
        "8.98846567431158e307", // 2^1023
        "1e-308",               // subnormal territory via plain exponent
    ] {
        assert_double_matches_oracle(text);
    }
}

#[test]
fn overflow_is_rejected_like_simdjson() {
    // Values that round to ±inf: simdjson rejects infinities, so do we.
    for text in [
        "1e309",
        "-1e309",
        "1e400",
        "-1e400",
        "2e308",
        "1.7976931348623159e308", // first literal that rounds to inf
        "1e99999999",
        "123123e100000",
        "0.4e00669999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999969999999006",
    ] {
        assert_rejected_as(text, SyntaxErrorKind::InvalidNumber);
        // The oracle agrees these are infinite (documenting the policy).
        assert!(text.parse::<f64>().unwrap().is_infinite(), "{text:?}");
    }
}

#[test]
fn underflow_collapses_to_signed_zero() {
    for (text, want) in [
        ("1e-400", 0.0f64),
        ("-1e-400", -0.0),
        ("1e-99999999", 0.0),
        ("123e-10000000", 0.0),
        ("-123.456e-789", -0.0),
    ] {
        let doc = parse_root(text).unwrap_or_else(|e| panic!("{text:?} must parse: {e}"));
        assert_eq!(
            doc.root().as_f64().map(f64::to_bits),
            Some(want.to_bits()),
            "{text:?}"
        );
        assert_double_matches_oracle(text); // and the oracle agrees
    }
    // Just above the underflow cliff: subnormal, NOT zero.
    let subnormal = parse_root("1e-310").unwrap().root().as_f64().unwrap();
    assert_ne!(subnormal, 0.0, "1e-310 is representable as a subnormal");
    assert_double_matches_oracle("1e-310");
}

#[test]
fn negative_zero_keeps_its_sign_bit() {
    for text in ["-0.0", "-0e0", "-0.0e5", "-0E-2"] {
        let doc = parse_root(text).unwrap();
        assert_eq!(doc.root().kind(), ValueKind::Double, "{text:?}");
        assert_eq!(
            doc.root().as_f64().map(f64::to_bits),
            Some((-0.0f64).to_bits()),
            "{text:?} must keep the sign bit"
        );
    }
    // Integer "-0" takes the integer fast path: Int64(0), like simdjson.
    let doc = parse_root("-0").unwrap();
    assert_eq!(doc.root().kind(), ValueKind::Int64);
    assert_eq!(doc.root().as_i64(), Some(0));
    // And "0.0" is plain positive zero.
    let doc = parse_root("0.0").unwrap();
    assert_eq!(
        doc.root().as_f64().map(f64::to_bits),
        Some(0.0f64.to_bits())
    );
}

#[test]
fn seventeen_digit_round_trips() {
    // 17 significant digits uniquely identify any f64.
    for text in [
        "0.1234567890123456",
        "0.12345678901234567",
        "1.7976931348623157",
        "17.976931348623157",
        "1797.6931348623157",
        "2.2250738585072014",
        "9007199254740993.0", // 2^53 + 1: not exactly representable
        "9007199254740992.0", // 2^53
        "0.3000000000000000444089209850062616169452667236328125", // exact 0.3
        "0.1",
        "0.2",
        "0.3",
        "123.456",
        "1e23", // famous half-way case
        "9.109383632e-31",
        "6.02214085774e23",
        "7.2057594037927933e16",
    ] {
        assert_double_matches_oracle(text);
    }
}

#[test]
fn hundred_plus_digit_mantissas() {
    // >100-digit integer part.
    let big_int = "1".repeat(120);
    assert_double_matches_oracle(&big_int);
    let big_int_frac = format!("{}.{}", "9".repeat(105), "9".repeat(40));
    assert_double_matches_oracle(&big_int_frac);
    // >100-digit fraction.
    let long_frac = format!("0.{}1", "0".repeat(100));
    assert_double_matches_oracle(&long_frac);
    let pi_ish = format!("3.{}", "1415926535897932384626433832795028841971".repeat(3));
    assert_double_matches_oracle(&pi_ish);
    // Long mantissa + exponent, still finite.
    let mixed = format!("{}e-200", "123456789".repeat(12));
    assert_double_matches_oracle(&mixed);
    // 19 digits of garbage after a truncation-sensitive prefix.
    assert_double_matches_oracle(
        "0.99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999",
    );
}

#[test]
fn integer_type_selection_boundaries() {
    let cases: &[(&str, ValueKind)] = &[
        ("9223372036854775807", ValueKind::Int64),   // i64::MAX
        ("-9223372036854775808", ValueKind::Int64),  // i64::MIN
        ("9223372036854775808", ValueKind::UInt64),  // i64::MAX + 1
        ("18446744073709551615", ValueKind::UInt64), // u64::MAX
        ("18446744073709551616", ValueKind::Double), // u64::MAX + 1
        ("-9223372036854775809", ValueKind::Double), // i64::MIN - 1
        ("0", ValueKind::Int64),
        ("-0", ValueKind::Int64),
        ("0e0", ValueKind::Double),
        ("9223372036854775807.0", ValueKind::Double), // fraction forces double
    ];
    for &(text, kind) in cases {
        let doc = parse_root(text).unwrap_or_else(|e| panic!("{text:?} must parse: {e}"));
        assert_eq!(doc.root().kind(), kind, "{text:?}");
    }
    assert_eq!(
        parse_root("18446744073709551615").unwrap().root().as_u64(),
        Some(u64::MAX)
    );
    assert_eq!(
        parse_root("-9223372036854775808").unwrap().root().as_i64(),
        Some(i64::MIN)
    );
}

#[test]
fn grammar_rejections() {
    use SyntaxErrorKind::*;
    // (text, kind): InvalidNumber for digit-led garbage; UnexpectedToken
    // when the first byte cannot start a scalar; InvalidLiteral when it
    // looks like true/false/null.
    let cases: &[(&str, SyntaxErrorKind)] = &[
        // Leading zeros.
        ("01", InvalidNumber),
        ("-01", InvalidNumber),
        ("00", InvalidNumber),
        ("012", InvalidNumber),
        ("0.e1", InvalidNumber), // empty fraction
        // Incomplete forms.
        ("-", InvalidNumber), // lone minus
        ("1.", InvalidNumber),
        ("-.", InvalidNumber),
        ("1e", InvalidNumber),
        ("1e+", InvalidNumber),
        ("1e-", InvalidNumber),
        ("0e", InvalidNumber),
        ("-x", InvalidNumber),
        ("--1", InvalidNumber),
        ("1eE2", InvalidNumber),
        ("1e1.5", InvalidNumber),
        ("1.2.3", InvalidNumber),
        // Hex / junk.
        ("0x1", InvalidNumber),
        ("0X42", InvalidNumber),
        ("-0x1", InvalidNumber),
        ("1x", InvalidNumber),
        ("123abc", InvalidNumber),
        // inf / nan in every spelling.
        ("-Infinity", InvalidNumber), // '-' starts the number grammar
        ("-inf", InvalidNumber),
        ("-NaN", InvalidNumber),
        ("nan", InvalidLiteral),  // 'n' looks like null
        ("inf", UnexpectedToken), // 'i' cannot start a scalar
        ("Infinity", UnexpectedToken),
        ("NaN", UnexpectedToken),
        // Not numbers at all.
        (".5", UnexpectedToken),
        ("+1", UnexpectedToken),
        ("+0", UnexpectedToken),
    ];
    for &(text, kind) in cases {
        assert_rejected_as(text, kind);
    }
    // The same rejections hold nested in a container.
    assert!(parse_root("[01]").is_err());
    assert!(parse_root("{\"k\": 1e}").is_err());
}

#[test]
fn exponent_edge_forms_parse() {
    for text in [
        "1e0", "1E0", "1e+0", "1e-0", "1e01", "1e+01",
        "1e-01", // leading zeros in the EXPONENT are legal
        "0e000", "1.5e3", "1.5E+3", "1.5e-3", "100e-2",
    ] {
        assert_double_matches_oracle(text);
    }
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

        let doc = parse_root(&text).expect("shortest form must parse");
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
