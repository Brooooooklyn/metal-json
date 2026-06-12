#![cfg(all(feature = "cpu-reference", feature = "serde"))]

mod common;

use std::collections::BTreeMap;

use serde::Deserialize;

#[derive(Debug, Deserialize, PartialEq)]
struct BorrowedUser<'a> {
    id: u64,
    name: &'a str,
    active: bool,
    score: Option<f64>,
    #[serde(borrow)]
    tags: Vec<&'a str>,
    #[serde(borrow)]
    nested: Nested<'a>,
}

#[derive(Debug, Deserialize, PartialEq)]
struct Nested<'a> {
    label: &'a str,
}

#[derive(Debug, Deserialize, PartialEq)]
struct OwnedConfig {
    name: String,
    retries: u8,
    enabled: bool,
    limits: Vec<u16>,
    metadata: BTreeMap<String, i64>,
}

#[derive(Debug, Deserialize, PartialEq)]
enum Event<'a> {
    Unit,
    Newtype(u64),
    Tuple(u8, &'a str),
    Struct { ok: bool },
}

#[test]
fn document_deserializes_borrowed_struct() {
    let parser = common::cpu_parser();
    let doc = parser
        .parse(
            br#"{
                "id": 18446744073709551615,
                "name": "Ada\nLovelace",
                "active": true,
                "score": null,
                "tags": ["gpu", "json"],
                "nested": { "label": "borrowed" }
            }"#,
        )
        .expect("valid JSON");

    let user: BorrowedUser<'_> = doc.deserialize().expect("serde struct");
    assert_eq!(
        user,
        BorrowedUser {
            id: u64::MAX,
            name: "Ada\nLovelace",
            active: true,
            score: None,
            tags: vec!["gpu", "json"],
            nested: Nested { label: "borrowed" },
        }
    );
}

#[test]
fn parser_parse_deserialize_returns_owned_struct() {
    let parser = common::cpu_parser();
    let config: OwnedConfig = parser
        .parse_deserialize(
            br#"{
                "name": "worker",
                "retries": 3,
                "enabled": false,
                "limits": [1, 255, 1024],
                "metadata": { "a": -1, "b": 2 }
            }"#,
        )
        .expect("owned serde struct");

    assert_eq!(
        config,
        OwnedConfig {
            name: "worker".to_owned(),
            retries: 3,
            enabled: false,
            limits: vec![1, 255, 1024],
            metadata: BTreeMap::from([("a".to_owned(), -1), ("b".to_owned(), 2)]),
        }
    );
}

#[test]
fn value_deserializes_subtrees_and_enums() {
    let parser = common::cpu_parser();
    let doc = parser
        .parse(
            br#"{
                "limits": [8, 16, 32],
                "unit": "Unit",
                "newtype": { "Newtype": 42 },
                "tuple": { "Tuple": [7, "seven"] },
                "struct": { "Struct": { "ok": true } }
            }"#,
        )
        .expect("valid JSON");
    let root = doc.root();

    let limits: Vec<u16> = root
        .get("limits")
        .expect("limits")
        .deserialize()
        .expect("limits subtree");
    assert_eq!(limits, vec![8, 16, 32]);

    let unit: Event<'_> = root
        .get("unit")
        .expect("unit")
        .deserialize()
        .expect("unit enum");
    let newtype: Event<'_> = root
        .get("newtype")
        .expect("newtype")
        .deserialize()
        .expect("newtype enum");
    let tuple: Event<'_> = root
        .get("tuple")
        .expect("tuple")
        .deserialize()
        .expect("tuple enum");
    let structure: Event<'_> = root
        .get("struct")
        .expect("struct")
        .deserialize()
        .expect("struct enum");

    assert_eq!(unit, Event::Unit);
    assert_eq!(newtype, Event::Newtype(42));
    assert_eq!(tuple, Event::Tuple(7, "seven"));
    assert_eq!(structure, Event::Struct { ok: true });
}

#[test]
fn root_scalars_and_interior_nul_strings_deserialize() {
    let parser = common::cpu_parser();

    let none: Option<u8> = parser
        .parse(b"null")
        .expect("null parses")
        .deserialize()
        .expect("option");
    assert_eq!(none, None);

    let flag: bool = parser
        .parse(b"true")
        .expect("bool parses")
        .deserialize()
        .expect("bool");
    assert!(flag);

    let signed: i64 = parser
        .parse(b"-42")
        .expect("i64 parses")
        .deserialize()
        .expect("i64");
    assert_eq!(signed, -42);

    let unsigned: u64 = parser
        .parse(b"18446744073709551615")
        .expect("u64 parses")
        .deserialize()
        .expect("u64");
    assert_eq!(unsigned, u64::MAX);

    let negative_zero: f64 = parser
        .parse(b"-0.0")
        .expect("double parses")
        .deserialize()
        .expect("f64");
    assert_eq!(negative_zero.to_bits(), (-0.0f64).to_bits());

    let doc = parser
        .parse(br#""a\u0000b""#)
        .expect("interior NUL string parses");
    let borrowed: &str = doc.deserialize().expect("borrowed string");
    assert_eq!(borrowed.as_bytes(), b"a\0b");
}

#[test]
fn deserialization_errors_preserve_serde_semantics() {
    let parser = common::cpu_parser();

    let duplicate = parser.parse(br#"{"k":1,"k":2}"#).expect("valid JSON");
    let err = duplicate
        .deserialize::<BTreeMap<String, i64>>()
        .expect("maps may consume duplicate keys with last-write-wins");
    assert_eq!(err.get("k"), Some(&2));

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct Single {
        k: i64,
    }

    let err = duplicate
        .deserialize::<Single>()
        .expect_err("duplicate field");
    assert!(
        err.to_string().contains("duplicate field"),
        "unexpected duplicate-field error: {err}"
    );

    let too_large = parser.parse(b"256").expect("valid JSON");
    assert!(
        too_large.deserialize::<u8>().is_err(),
        "serde must reject out-of-range numbers"
    );
}

#[test]
fn map_keys_parse_for_integer_bool_and_newtype_targets() {
    let parser = common::cpu_parser();

    let by_id: BTreeMap<u32, String> = parser
        .parse_deserialize(br#"{"1":"a","2":"b"}"#)
        .expect("integer-keyed map");
    assert_eq!(
        by_id,
        BTreeMap::from([(1, "a".to_owned()), (2, "b".to_owned())])
    );

    let negative: BTreeMap<i64, u8> = parser
        .parse_deserialize(br#"{"-7":1}"#)
        .expect("signed integer key");
    assert_eq!(negative, BTreeMap::from([(-7, 1)]));

    let by_flag: BTreeMap<bool, i64> = parser
        .parse_deserialize(br#"{"true":1,"false":0}"#)
        .expect("bool-keyed map");
    assert_eq!(by_flag, BTreeMap::from([(true, 1), (false, 0)]));

    #[derive(Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
    struct Id(String);

    let by_newtype: BTreeMap<Id, u8> = parser
        .parse_deserialize(br#"{"a":1}"#)
        .expect("newtype-keyed map");
    assert_eq!(by_newtype, BTreeMap::from([(Id("a".to_owned()), 1)]));

    // Unparseable keys still hit the target type's own error.
    assert!(
        parser
            .parse_deserialize::<BTreeMap<u32, u8>>(br#"{"x":1}"#)
            .is_err(),
        "non-numeric key must not deserialize as u32"
    );
}

#[test]
fn structs_accept_positional_arrays() {
    #[derive(Debug, Deserialize, PartialEq)]
    struct Point {
        x: u8,
        y: u8,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    enum Shape {
        Vertex { x: u8, y: u8 },
    }

    let parser = common::cpu_parser();

    let point: Point = parser
        .parse_deserialize(b"[1,2]")
        .expect("positional struct");
    assert_eq!(point, Point { x: 1, y: 2 });

    let vertex: Shape = parser
        .parse_deserialize(br#"{"Vertex":[3,4]}"#)
        .expect("positional struct variant");
    assert_eq!(vertex, Shape::Vertex { x: 3, y: 4 });

    let err = parser
        .parse_deserialize::<Point>(br#""nope""#)
        .expect_err("string is not a struct");
    assert!(
        err.to_string().contains("expected object or array"),
        "unexpected struct error: {err}"
    );

    // Positional inputs are exact-arity, like tuples: extras are rejected,
    // not silently dropped.
    let err = parser
        .parse_deserialize::<Point>(b"[1,2,3]")
        .expect_err("extra positional struct field");
    assert!(
        err.to_string()
            .contains("invalid length 3, expected fewer elements in array"),
        "unexpected extra-element error: {err}"
    );
    let err = parser
        .parse_deserialize::<Shape>(br#"{"Vertex":[3,4,5]}"#)
        .expect_err("extra positional struct-variant field");
    assert!(
        err.to_string()
            .contains("invalid length 3, expected fewer elements in array"),
        "unexpected extra-element error: {err}"
    );
}

#[test]
fn tuples_reject_arrays_with_extra_elements() {
    let parser = common::cpu_parser();

    let pair: (i64, i64) = parser.parse_deserialize(b"[1,2]").expect("exact tuple");
    assert_eq!(pair, (1, 2));

    let err = parser
        .parse_deserialize::<(i64, i64)>(b"[1,2,3]")
        .expect_err("extra elements must not be dropped");
    assert!(
        err.to_string()
            .contains("invalid length 3, expected fewer elements in array"),
        "unexpected tuple error: {err}"
    );

    #[derive(Debug, Deserialize, PartialEq)]
    struct Pair(i64, i64);

    assert!(
        parser.parse_deserialize::<Pair>(b"[1,2,3]").is_err(),
        "tuple structs must also reject extra elements"
    );

    #[derive(Debug, Deserialize, PartialEq)]
    enum E {
        T(i64, i64),
    }

    assert!(
        parser.parse_deserialize::<E>(br#"{"T":[1,2,3]}"#).is_err(),
        "tuple variants must also reject extra elements"
    );
}

#[test]
fn i128_u128_exact_within_u64_range_and_error_beyond() {
    let parser = common::cpu_parser();

    // Exact through u64::MAX — no precision loss above 2^53.
    let above_2_53: u128 = parser
        .parse_deserialize(b"9007199254740995")
        .expect("u128 above 2^53");
    assert_eq!(above_2_53, 9_007_199_254_740_995);

    let max: u128 = parser
        .parse_deserialize(b"18446744073709551615")
        .expect("u128 at u64::MAX");
    assert_eq!(max, u64::MAX as u128);

    let min: i128 = parser
        .parse_deserialize(b"-9223372036854775808")
        .expect("i128 at i64::MIN");
    assert_eq!(min, i64::MIN as i128);

    // Beyond u64::MAX the tape holds an f64, which 128-bit targets reject.
    let err = parser
        .parse_deserialize::<u128>(b"18446744073709551616")
        .expect_err("beyond-u64 integers were stored as f64");
    assert!(
        err.to_string().contains("invalid type: floating point"),
        "unexpected u128 error: {err}"
    );
}
