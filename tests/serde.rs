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
