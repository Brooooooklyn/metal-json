//! Differential test: the CPU reference backend vs serde_json on every
//! corpus fixture and every `y_*` JSONTestSuite file.
//!
//! serde_json is built with `preserve_order` (objects compare as ordered
//! entry sequences) and `arbitrary_precision` (numbers compare by tape
//! kind, doubles bit-for-bit against `str::parse::<f64>` of the raw
//! literal) — see `common::assert_doc_eq`.
//!
//! Duplicate-key documents are excluded from the serde comparison (serde's
//! map keeps the last duplicate; our tape keeps all members, simdjson
//! parity) and asserted separately against the raw tape walk.
#![cfg(feature = "cpu-reference")]

mod common;

use metal_json::ValueKind;

#[test]
fn corpus_matches_serde_json() {
    let parser = common::cpu_parser();
    let mut compared = 0usize;
    let mut dup_key_files = 0usize;

    for path in common::corpus_files() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let bytes = std::fs::read(&path).expect("readable corpus fixture");
        let doc = parser
            .parse(&bytes)
            .unwrap_or_else(|e| panic!("{name}: corpus fixture must parse, got {e}"));

        if common::has_duplicate_keys(doc.root()) {
            dup_key_files += 1; // covered by corpus_duplicate_keys_on_the_raw_tape
            continue;
        }
        let serde: serde_json::Value = serde_json::from_slice(&bytes)
            .unwrap_or_else(|e| panic!("{name}: serde_json must accept corpus fixture: {e}"));
        common::assert_doc_eq(doc.root(), &serde, &name);
        compared += 1;
    }

    println!("corpus differential: {compared} files compared, {dup_key_files} duplicate-key files");
    assert!(compared >= 13, "most corpus fixtures must be comparable");
    assert_eq!(
        dup_key_files, 1,
        "exactly corpus/duplicate_keys.json carries duplicates"
    );
}

/// The duplicate-key fixture, checked against the raw tape walk: every
/// member is present, verbatim, in document order — where serde_json's map
/// would have collapsed them to the last value.
#[test]
fn corpus_duplicate_keys_on_the_raw_tape() {
    let parser = common::cpu_parser();
    let path = common::corpus_dir().join("duplicate_keys.json");
    let bytes = std::fs::read(&path).expect("duplicate_keys.json is checked in");
    let doc = parser.parse(&bytes).expect("fixture parses");
    let root = doc.root();

    // serde would report 3 members; the tape keeps all 5.
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

    let other = root.get("other").unwrap();
    let d_values: Vec<bool> = other
        .entries()
        .map(|(k, v)| {
            assert_eq!(k, "d");
            v.as_bool().expect("d values are bools")
        })
        .collect();
    assert_eq!(d_values, [true, false]);

    let x = root.get("arr").unwrap().at(0).unwrap();
    let x_values: Vec<&str> = x.entries().map(|(_, v)| v.as_str().unwrap()).collect();
    assert_eq!(x_values, ["first", "second"]);

    // And serde really does disagree — guarding the premise of the split.
    let serde: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(serde.as_object().unwrap().len(), 3);
}

#[test]
fn jsontestsuite_y_files_match_serde_json() {
    let Some(dir) = common::jsontestsuite_dir() else {
        return; // loud skip already printed
    };
    let parser = common::cpu_parser();
    let mut compared = 0usize;
    let mut dup_keys = Vec::new();
    let mut serde_rejected = Vec::new();

    for path in common::jsontestsuite_files(&dir, "y_") {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let bytes = std::fs::read(&path).expect("readable y_ file");
        let doc = parser
            .parse(&bytes)
            .unwrap_or_else(|e| panic!("{name}: y_ file must parse, got {e}"));

        if common::has_duplicate_keys(doc.root()) {
            dup_keys.push(name); // tape behavior asserted below
            continue;
        }
        let Ok(serde) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            serde_rejected.push(name); // outside the comparable set
            continue;
        };
        common::assert_doc_eq(doc.root(), &serde, &name);
        compared += 1;
    }

    println!(
        "y_ differential: {compared} compared, {} duplicate-key (tape-tested), \
         {} rejected by serde: {serde_rejected:?}",
        dup_keys.len(),
        serde_rejected.len()
    );
    assert!(compared >= 90, "nearly all y_ files must be comparable");
    assert_eq!(
        dup_keys,
        vec![
            "y_object_duplicated_key.json".to_owned(),
            "y_object_duplicated_key_and_value.json".to_owned(),
        ],
        "the suite's duplicate-key y_ files"
    );
    assert!(
        serde_rejected.is_empty(),
        "serde_json (arbitrary_precision) accepts every y_ file today; \
         a new rejection needs investigating: {serde_rejected:?}"
    );
}

/// The two duplicate-key `y_` files, against the raw tape walk.
#[test]
fn jsontestsuite_duplicate_key_files_keep_all_members() {
    let Some(dir) = common::jsontestsuite_dir() else {
        return; // loud skip already printed
    };
    let parser = common::cpu_parser();

    // y_object_duplicated_key.json: {"a":"b","a":"c"}
    let doc = parser
        .parse(&std::fs::read(dir.join("y_object_duplicated_key.json")).unwrap())
        .expect("y_object_duplicated_key.json parses");
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
    let doc = parser
        .parse(&std::fs::read(dir.join("y_object_duplicated_key_and_value.json")).unwrap())
        .expect("y_object_duplicated_key_and_value.json parses");
    let root = doc.root();
    assert_eq!(root.len(), Some(2));
    let entries: Vec<(&str, &str)> = root
        .entries()
        .map(|(k, v)| (k, v.as_str().unwrap()))
        .collect();
    assert_eq!(entries, [("a", "b"), ("a", "b")]);
}
