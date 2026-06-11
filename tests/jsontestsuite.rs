//! JSONTestSuite (nst/JSONTestSuite) conformance run for the CPU reference
//! backend — the M1 oracle the GPU pipeline is diffed against in M2-M4.
//!
//! Verdict contract (non-negotiable):
//! - `y_*.json` MUST parse (`Ok`);
//! - `n_*.json` MUST return `Err` — and the library must never panic;
//! - `i_*.json` (implementation-defined) may go either way but must not
//!   crash; our choice for every `i_*` file is pinned in
//!   [`common::I_FILE_VERDICTS`] with a reason (shared with the GPU gate in
//!   `tests/gpu_e2e.rs`), so any behavior change is noticed.
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
                        match common::I_FILE_VERDICTS.iter().find(|(n, _, _)| *n == name) {
                            None => failures.push(format!(
                                "{name}: new i_ file — add a pinned verdict to common::I_FILE_VERDICTS"
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
    for (name, _, _) in common::I_FILE_VERDICTS {
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

/// The M4 error-contract completion: the **GPU backend** against the CPU
/// reference on every file in the suite — the M3 three-way split (GPU
/// accepts what only the scalar stages reject) collapsed to **two-way**.
/// Every file must agree on the verdict; accepted files must produce
/// bit-identical tapes and identical value trees. Rejected files compare
/// WHETHER only: multi-error documents may legitimately differ in WHICH
/// error is reported (GPU = earliest byte offset, reference = stage order
/// — the documented relaxation; single-error code+offset parity is pinned
/// in src/parser.rs / src/gpu/pipeline.rs).
#[test]
fn jsontestsuite_gpu_backend_matches_the_reference() {
    let Some(dir) = common::jsontestsuite_dir() else {
        return; // loud skip already printed
    };
    let Some(gpu) = common::gpu_parser_or_skip("jsontestsuite_gpu_backend_matches_the_reference")
    else {
        return;
    };
    let cpu = common::cpu_parser();

    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for prefix in ["y_", "n_", "i_"] {
        for path in common::jsontestsuite_files(&dir, prefix) {
            let name = file_name(&path).to_owned();
            let bytes = std::fs::read(&path).expect("readable corpus file");
            let gpu_result = catch_unwind(AssertUnwindSafe(|| gpu.parse(&bytes)));
            let Ok(gpu_result) = gpu_result else {
                failures.push(format!("{name}: GPU backend PANICKED (must never happen)"));
                continue;
            };
            match (gpu_result, cpu.parse(&bytes)) {
                (Ok(gpu_doc), Ok(cpu_doc)) => {
                    accepted += 1;
                    assert_eq!(
                        gpu_doc.tape(),
                        cpu_doc.tape(),
                        "{name}: tape words must be bit-identical"
                    );
                    common::assert_docs_eq(gpu_doc.root(), cpu_doc.root(), &name);
                }
                (Err(_), Err(_)) => rejected += 1, // verdict parity
                (Ok(_), Err(e)) => {
                    failures.push(format!("{name}: GPU accepted, reference rejected ({e})"));
                }
                (Err(e), Ok(_)) => {
                    failures.push(format!("{name}: GPU rejected ({e}), reference accepted"));
                }
            }
        }
    }

    println!(
        "GPU two-way parity: {accepted} accepted on both, {rejected} rejected on both, \
         {} disagreements",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{} GPU/reference disagreements:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
    // Sanity: both halves of the two-way split are well represented (the
    // suite has ~100 accepts incl. the accepted i_ files and ~215 rejects).
    assert!(accepted >= 90, "only {accepted} files accepted — suite incomplete?");
    assert!(rejected >= 180, "only {rejected} files rejected — suite incomplete?");
}
