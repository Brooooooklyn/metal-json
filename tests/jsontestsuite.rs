//! JSONTestSuite (nst/JSONTestSuite) conformance run for the CPU reference
//! backend — the M1 oracle the GPU pipeline is diffed against in M2-M4.
//!
//! Verdict contract (non-negotiable):
//! - `y_*.json` MUST parse (`Ok`);
//! - `n_*.json` MUST return `Err` — and the library must never panic;
//! - `i_*.json` (implementation-defined) may go either way but must not
//!   crash; our choice for every `i_*` file is pinned in
//!   [`I_FILE_VERDICTS`] with a reason, so any behavior change is noticed.
//!
//! The corpus lives in the gitignored `data/JSONTestSuite` (fetch with
//! `scripts/fetch_jsontestsuite.sh`); the run auto-skips loudly when it is
//! missing.
#![cfg(feature = "cpu-reference")]

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

/// Expected-divergence notes for `n_*` files: cases where we reject (as
/// required) but for a *different reason* than the file name advertises, or
/// where naive parsers crash. Documentation, kept honest by asserting each
/// listed file exists and is rejected. Verdicts themselves never diverge.
const N_FILE_NOTES: &[(&str, &str)] = &[
    (
        "n_structure_100000_opening_arrays.json",
        "100000 unclosed '[': recursive parsers stack-overflow here. Our \
         Layer-1 open-bracket-then-EOF ban rejects it as UnbalancedBrackets \
         before stage 4's DepthLimit (1024, simdjson parity) would; either \
         error counts as correct rejection.",
    ),
    (
        "n_structure_open_array_object.json",
        "50000 repetitions of `[{\"\":`: rejected as a Layer-1 \
         separator-then-EOF violation (UnexpectedToken), again before the \
         DepthLimit could fire. No recursion anywhere in the pipeline.",
    ),
    (
        "n_single_space.json",
        "whitespace-only input is rejected as EmptyInput (offset 0), \
         matching simdjson's EMPTY error.",
    ),
    (
        "n_multidigit_number_then_00.json",
        "`123\\0`: the NUL byte extends the scalar run, so this is rejected \
         by the number grammar (InvalidNumber), not as a stray token.",
    ),
];

/// Pinned verdict (`true` = accept) and rationale for every `i_*` file.
/// These are implementation-defined by JSONTestSuite; ours mirror simdjson:
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
const I_FILE_VERDICTS: &[(&str, bool, &str)] = &[
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

/// Parse `bytes` with the CPU reference backend, catching any panic.
/// Returns `Ok(parsed_ok)` or `Err(())` if the library panicked.
fn parse_no_panic(parser: &metal_json::Parser, bytes: &[u8]) -> Result<bool, ()> {
    catch_unwind(AssertUnwindSafe(|| parser.parse(bytes).is_ok())).map_err(|_| ())
}

fn file_name(path: &Path) -> &str {
    path.file_name().and_then(|n| n.to_str()).unwrap_or("???")
}

#[test]
fn jsontestsuite_verdicts() {
    let Some(dir) = common::jsontestsuite_dir() else {
        return; // loud skip already printed
    };
    let parser = common::cpu_parser();

    let mut y_pass = 0usize;
    let mut n_pass = 0usize;
    let mut i_accept = 0usize;
    let mut i_reject = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for prefix in ["y_", "n_", "i_"] {
        for path in common::jsontestsuite_files(&dir, prefix) {
            let name = file_name(&path).to_owned();
            let bytes = std::fs::read(&path).expect("readable corpus file");
            match parse_no_panic(&parser, &bytes) {
                Err(()) => failures.push(format!("{name}: PANICKED (must never happen)")),
                Ok(parsed) => match prefix {
                    "y_" if parsed => y_pass += 1,
                    "y_" => failures.push(format!("{name}: must parse, got Err")),
                    "n_" if !parsed => n_pass += 1,
                    "n_" => failures.push(format!("{name}: must be rejected, got Ok")),
                    _ => {
                        // i_: either verdict is conformant; pin ours.
                        if parsed {
                            i_accept += 1;
                        } else {
                            i_reject += 1;
                        }
                        match I_FILE_VERDICTS.iter().find(|(n, _, _)| *n == name) {
                            None => failures.push(format!(
                                "{name}: new i_ file — add a pinned verdict to I_FILE_VERDICTS"
                            )),
                            Some((_, want_accept, reason)) if *want_accept != parsed => {
                                failures.push(format!(
                                    "{name}: pinned verdict accept={want_accept} ({reason}), \
                                     but got accept={parsed}"
                                ));
                            }
                            Some(_) => {}
                        }
                    }
                },
            }
        }
    }

    // The documentation lists must reference real, correctly-handled files.
    for (name, _) in N_FILE_NOTES {
        let path = dir.join(name);
        assert!(path.is_file(), "N_FILE_NOTES lists missing file {name}");
        let bytes = std::fs::read(&path).expect("readable corpus file");
        assert_eq!(
            parse_no_panic(&parser, &bytes),
            Ok(false),
            "{name} is documented as rejected"
        );
    }
    for (name, _, _) in I_FILE_VERDICTS {
        assert!(
            dir.join(name).is_file(),
            "I_FILE_VERDICTS lists missing file {name} — corpus updated?"
        );
    }

    let y_total = common::jsontestsuite_files(&dir, "y_").len();
    let n_total = common::jsontestsuite_files(&dir, "n_").len();
    let i_total = common::jsontestsuite_files(&dir, "i_").len();
    println!(
        "JSONTestSuite summary: y {y_pass}/{y_total} parsed, \
         n {n_pass}/{n_total} rejected, \
         i {i_total} no-crash ({i_accept} accepted / {i_reject} rejected), \
         {} failures",
        failures.len()
    );

    assert!(
        failures.is_empty(),
        "{} JSONTestSuite failures:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
    assert_eq!(y_pass, y_total, "every y_ file must parse");
    assert_eq!(n_pass, n_total, "every n_ file must be rejected");
}
