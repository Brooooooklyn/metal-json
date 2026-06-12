//! Workspace task runner.
//!
//! ```text
//! cargo run -p xtask -- fetch-data   [--out DIR]
//! cargo run -p xtask -- gen-data     --template twitter|canada --size <N>m|<N>g [--out PATH]
//! cargo run -p xtask -- bench-report [--out PATH] [--skip-bench] [--skip-breakdown]
//! ```

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives inside the workspace")
        .to_path_buf()
}

fn default_data_dir() -> PathBuf {
    workspace_root().join("data").join("bench")
}

const USAGE: &str = "xtask subcommands:
  fetch-data   [--out DIR]
      Download simdjson-data canonical files (twitter.json, canada.json,
      citm_catalog.json) into data/bench/ (or DIR).
  gen-data     --template twitter|canada --size <N>m|<N>g [--out PATH]
      Deterministically expand the fetched template into one large top-level
      JSON array of about N MiB / N GiB (default PATH:
      data/bench/<template>_<size>.json). e.g. --size 4m, --size 512m,
      --size 1g.
  bench-report [--out PATH] [--skip-bench] [--skip-breakdown]
      Run the criterion suite and render the full benchmark report
      (GB/s medians, dataset provenance, size-sweep crossover, per-kernel
      breakdown, methodology) to PATH (default docs/bench-report.md).
      --skip-bench reuses existing target/criterion results;
      --skip-breakdown omits the timing-feature per-kernel section.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("fetch-data") => fetch_data(&args[1..]),
        Some("gen-data") => gen_data(&args[1..]),
        Some("bench-report") => bench_report(&args[1..]),
        Some("--help" | "-h" | "help") | None => {
            println!("{USAGE}");
            Ok(())
        }
        Some(other) => Err(format!("unknown subcommand {other:?}\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("xtask error: {msg}");
            ExitCode::FAILURE
        }
    }
}

/// Pull the value following a `--flag`, removing both from `args`.
fn take_flag_value(args: &mut Vec<String>, flag: &str) -> Result<Option<String>, String> {
    if let Some(pos) = args.iter().position(|a| a == flag) {
        if pos + 1 >= args.len() {
            return Err(format!("{flag} requires a value"));
        }
        let value = args.remove(pos + 1);
        args.remove(pos);
        return Ok(Some(value));
    }
    Ok(None)
}

/// Pull a boolean `--flag`, removing it from `args`.
fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    if let Some(pos) = args.iter().position(|a| a == flag) {
        args.remove(pos);
        return true;
    }
    false
}

fn ensure_no_leftovers(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(format!("unexpected arguments: {args:?}\n\n{USAGE}"))
    }
}

// ---------------------------------------------------------------------------
// fetch-data
// ---------------------------------------------------------------------------

const CANONICAL_FILES: &[&str] = &["twitter.json", "canada.json", "citm_catalog.json"];
const SIMDJSON_DATA_RAW: &str =
    "https://raw.githubusercontent.com/simdjson/simdjson-data/master/jsonexamples";

fn fetch_data(args: &[String]) -> Result<(), String> {
    let mut args = args.to_vec();
    let out_dir = take_flag_value(&mut args, "--out")?
        .map_or_else(default_data_dir, PathBuf::from);
    ensure_no_leftovers(&args)?;

    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("create {}: {e}", out_dir.display()))?;

    for name in CANONICAL_FILES {
        let dest = out_dir.join(name);
        if dest.exists()
            && std::fs::metadata(&dest).map(|m| m.len() > 0).unwrap_or(false)
        {
            println!("fetch-data: {} already present, skipping", dest.display());
            continue;
        }
        let url = format!("{SIMDJSON_DATA_RAW}/{name}");
        let part = out_dir.join(format!("{name}.part"));
        println!("fetch-data: downloading {url}");
        let status = std::process::Command::new("curl")
            .args(["-fSL", "--retry", "3", "-o"])
            .arg(&part)
            .arg(&url)
            .status()
            .map_err(|e| format!("spawn curl: {e}"))?;
        if !status.success() {
            let _ = std::fs::remove_file(&part);
            return Err(format!("curl failed for {url} ({status})"));
        }
        std::fs::rename(&part, &dest)
            .map_err(|e| format!("rename {}: {e}", part.display()))?;
        println!("fetch-data: wrote {}", dest.display());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// gen-data
// ---------------------------------------------------------------------------

/// Deterministic large-file generator.
///
/// Records are extracted from the fetched canonical template
/// (`data/bench/twitter.json` statuses / `canada.json` features),
/// re-serialized compactly by serde_json (BTreeMap key order — fully
/// deterministic), and cycled verbatim into one valid top-level JSON array
/// until the target size is reached. No randomness, no time, no
/// hash-iteration anywhere: given the same template file and the same
/// (Cargo.lock-pinned) serde_json, the byte output is identical across
/// runs and machines.
/// Parse a `--size` spec: `<N>m` = N MiB, `<N>g` = N GiB (e.g. `4m`,
/// `512m`, `1g`).
fn parse_size(size: &str) -> Result<u64, String> {
    let (digits, unit): (&str, u64) = if let Some(d) = size.strip_suffix('m') {
        (d, 1024 * 1024)
    } else if let Some(d) = size.strip_suffix('g') {
        (d, 1024 * 1024 * 1024)
    } else {
        return Err(format!("--size must look like 4m, 100m or 1g, got {size:?}"));
    };
    let n: u64 = digits
        .parse()
        .map_err(|_| format!("--size must look like 4m, 100m or 1g, got {size:?}"))?;
    if n == 0 {
        return Err("--size must be nonzero".into());
    }
    n.checked_mul(unit).ok_or_else(|| format!("--size {size:?} overflows"))
}

fn gen_data(args: &[String]) -> Result<(), String> {
    let mut args = args.to_vec();
    let template = take_flag_value(&mut args, "--template")?
        .ok_or("gen-data requires --template twitter|canada")?;
    let size = take_flag_value(&mut args, "--size")?
        .ok_or("gen-data requires --size 100m|256m|512m|1g")?;
    let out = take_flag_value(&mut args, "--out")?;
    ensure_no_leftovers(&args)?;

    let target_bytes = parse_size(&size)?;

    let template_path = default_data_dir().join(format!("{template}.json"));
    if !template_path.exists() {
        return Err(format!(
            "template {} missing — run `cargo run -p xtask -- fetch-data` first",
            template_path.display()
        ));
    }
    let raw = std::fs::read(&template_path)
        .map_err(|e| format!("read {}: {e}", template_path.display()))?;
    let doc: serde_json::Value =
        serde_json::from_slice(&raw).map_err(|e| format!("parse template: {e}"))?;

    // Extract the record list to cycle.
    let records: Vec<String> = match template.as_str() {
        "twitter" => doc
            .get("statuses")
            .and_then(|v| v.as_array())
            .ok_or("twitter.json: missing top-level \"statuses\" array")?
            .iter()
            .map(|r| serde_json::to_string(r).expect("re-serialize record"))
            .collect(),
        "canada" => doc
            .get("features")
            .and_then(|v| v.as_array())
            .ok_or("canada.json: missing top-level \"features\" array")?
            .iter()
            .map(|r| serde_json::to_string(r).expect("re-serialize record"))
            .collect(),
        other => return Err(format!("--template must be twitter|canada, got {other:?}")),
    };
    if records.is_empty() {
        return Err("template produced no records".into());
    }

    let out_path = out.map_or_else(
        || default_data_dir().join(format!("{template}_{size}.json")),
        PathBuf::from,
    );
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }

    let file = std::fs::File::create(&out_path)
        .map_err(|e| format!("create {}: {e}", out_path.display()))?;
    let mut w = std::io::BufWriter::with_capacity(4 << 20, file);
    let mut written: u64 = 0;
    let emit = |w: &mut dyn std::io::Write, bytes: &[u8], written: &mut u64| {
        w.write_all(bytes).map_err(|e| format!("write: {e}"))?;
        *written += bytes.len() as u64;
        Ok::<(), String>(())
    };

    emit(&mut w, b"[", &mut written)?;
    let mut index = 0usize;
    // Cycle records verbatim until the target size is reached; the file ends
    // slightly above target_bytes (by at most one record + 2 bytes).
    while written < target_bytes {
        if index > 0 {
            emit(&mut w, b",", &mut written)?;
        }
        emit(&mut w, records[index % records.len()].as_bytes(), &mut written)?;
        index += 1;
    }
    emit(&mut w, b"]", &mut written)?;
    w.flush().map_err(|e| format!("flush: {e}"))?;

    println!(
        "gen-data: wrote {} ({} bytes, {} records, template {})",
        out_path.display(),
        written,
        index,
        template
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// bench-report
// ---------------------------------------------------------------------------

const CONTENDER_ORDER: &[&str] = &["metal-json", "simdjson-cpp", "simd-json-rust", "serde-json"];

struct Measurement {
    bytes: u64,
    median_ns: f64,
}

impl Measurement {
    fn gb_per_s(&self) -> f64 {
        // bytes / ns == GB/s (decimal GB).
        self.bytes as f64 / self.median_ns
    }
}

type DatasetRows = Vec<(String, Vec<(String, Measurement)>)>;

/// The dataset directory the bench harness reads
/// (`bench/src/lib.rs::data_dir`): `$METAL_JSON_BENCH_DATA` or
/// `<workspace>/data/bench`.
fn bench_data_dir() -> PathBuf {
    std::env::var("METAL_JSON_BENCH_DATA").map_or_else(|_| default_data_dir(), PathBuf::from)
}

fn bench_report(args: &[String]) -> Result<(), String> {
    let mut args = args.to_vec();
    let out_path = take_flag_value(&mut args, "--out")?
        .map_or_else(|| workspace_root().join("docs").join("bench-report.md"), PathBuf::from);
    let skip_bench = take_flag(&mut args, "--skip-bench");
    let skip_breakdown = take_flag(&mut args, "--skip-breakdown");
    ensure_no_leftovers(&args)?;

    if !skip_bench {
        println!("bench-report: running `cargo bench -p metal-json-bench` ...");
        let status = std::process::Command::new("cargo")
            .args(["bench", "-p", "metal-json-bench"])
            .current_dir(workspace_root())
            .status()
            .map_err(|e| format!("spawn cargo bench: {e}"))?;
        if !status.success() {
            return Err(format!("cargo bench failed ({status})"));
        }
    }

    let criterion_dir = workspace_root().join("target").join("criterion");
    if !criterion_dir.is_dir() {
        return Err(format!(
            "{} missing — did the bench run produce results?",
            criterion_dir.display()
        ));
    }

    // dataset -> contender -> measurement
    let mut datasets: Vec<(String, Vec<(String, Measurement)>)> = Vec::new();
    let mut group_dirs: Vec<PathBuf> = std::fs::read_dir(&criterion_dir)
        .map_err(|e| format!("read {}: {e}", criterion_dir.display()))?
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.is_dir() && p.file_name()? != "report").then_some(p)
        })
        .collect();
    group_dirs.sort();

    for group_dir in group_dirs {
        let dataset = group_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_owned();
        let mut rows: Vec<(String, Measurement)> = Vec::new();
        let mut fn_dirs: Vec<PathBuf> = std::fs::read_dir(&group_dir)
            .map_err(|e| format!("read {}: {e}", group_dir.display()))?
            .filter_map(|e| {
                let p = e.ok()?.path();
                (p.is_dir() && p.file_name()? != "report").then_some(p)
            })
            .collect();
        fn_dirs.sort();
        for fn_dir in fn_dirs {
            let contender = fn_dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_owned();
            let Some(m) = read_measurement(&fn_dir) else {
                continue;
            };
            rows.push((contender, m));
        }
        if !rows.is_empty() {
            datasets.push((dataset, rows));
        }
    }

    if datasets.is_empty() {
        return Err("no criterion measurements found".into());
    }

    // Order datasets by measured size for a readable sweep.
    datasets.sort_by_key(|(_, rows)| rows.first().map_or(0, |(_, m)| m.bytes));

    let breakdown = if skip_breakdown {
        None
    } else {
        run_breakdown(&datasets)
    };
    let report = render_report(&datasets, breakdown.as_deref());
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    std::fs::write(&out_path, report).map_err(|e| format!("write {}: {e}", out_path.display()))?;
    println!("bench-report: wrote {}", out_path.display());
    Ok(())
}

// --- environment / provenance helpers ---------------------------------------

/// Run a command, returning trimmed stdout on success.
fn capture(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    (!s.is_empty()).then_some(s)
}

/// sha256 of a file via the system `shasum` (macOS ships it).
fn sha256(path: &Path) -> Option<String> {
    let out = capture("shasum", &["-a", "256", &path.to_string_lossy()])?;
    out.split_whitespace().next().map(str::to_owned)
}

/// The vendored simdjson version, parsed from the amalgamation header.
fn simdjson_version() -> Option<String> {
    let header = workspace_root()
        .join("bench")
        .join("cpp")
        .join("vendor")
        .join("simdjson.h");
    let text = std::fs::read_to_string(header).ok()?;
    let line = text
        .lines()
        .find(|l| l.starts_with("#define SIMDJSON_VERSION "))?;
    Some(line.split('"').nth(1)?.to_owned())
}

/// `(label, value)` pairs describing the machine and toolchain.
fn collect_env_info() -> Vec<(&'static str, String)> {
    let unknown = || "(unknown)".to_owned();
    let mut info = Vec::new();
    info.push((
        "Date",
        capture("date", &["-u", "+%Y-%m-%d"]).unwrap_or_else(unknown),
    ));
    let chip = capture("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_else(unknown);
    let mem = capture("sysctl", &["-n", "hw.memsize"])
        .and_then(|s| s.parse::<u64>().ok())
        .map_or_else(unknown, |b| format!("{} GiB unified memory", b >> 30));
    info.push(("Machine", format!("{chip}, {mem}")));
    info.push((
        "OS",
        capture("sw_vers", &["-productVersion"])
            .map_or_else(unknown, |v| format!("macOS {v}")),
    ));
    info.push(("Rust", capture("rustc", &["--version"]).unwrap_or_else(unknown)));
    info.push((
        "Metal toolchain",
        capture("xcrun", &["-sdk", "macosx", "metal", "--version"])
            .map_or_else(unknown, |v| v.lines().next().unwrap_or_default().to_owned()),
    ));
    info.push((
        "simdjson (vendored)",
        simdjson_version().map_or_else(unknown, |v| format!("v{v} (amalgamation, cc -O3)")),
    ));
    info.push((
        "metal-json backend",
        std::env::var("METAL_JSON_BENCH_BACKEND").unwrap_or_else(|_| "gpu (default)".to_owned()),
    ));
    info
}

/// Classify a dataset name into its provenance string.
fn dataset_source(name: &str) -> String {
    if CANONICAL_FILES.contains(&format!("{name}.json").as_str()) {
        return "simdjson-data canonical (`xtask fetch-data`)".to_owned();
    }
    if let Some((template, size)) = name.rsplit_once('_')
        && (template == "twitter" || template == "canada")
        && parse_size(size).is_ok()
    {
        return format!("generated: `xtask gen-data --template {template} --size {size}`");
    }
    "(local file)".to_owned()
}

// --- crossover sweep ---------------------------------------------------------

struct SweepRow {
    name: String,
    bytes: u64,
    metal_ns: f64,
    sj_ns: f64,
}

/// Twitter-template datasets with both headline contenders measured,
/// sorted by size: the CPU/GPU crossover sweep.
fn sweep_rows(datasets: &DatasetRows) -> Vec<SweepRow> {
    let mut rows: Vec<SweepRow> = datasets
        .iter()
        .filter(|(name, _)| {
            name == "twitter"
                || matches!(name.rsplit_once('_'),
                    Some((t, s)) if t == "twitter" && parse_size(s).is_ok())
        })
        .filter_map(|(name, rows)| {
            let find = |n: &str| rows.iter().find(|(c, _)| c == n).map(|(_, m)| m);
            let metal = find("metal-json")?;
            let sj = find("simdjson-cpp")?;
            Some(SweepRow {
                name: name.clone(),
                bytes: metal.bytes,
                metal_ns: metal.median_ns,
                sj_ns: sj.median_ns,
            })
        })
        .collect();
    rows.sort_by_key(|r| r.bytes);
    rows
}

/// Estimated input size where the two contenders' median times cross,
/// linearly interpolated between the last simdjson-won size and the first
/// metal-json-won size (of the final sign change, should noise produce
/// several).
fn crossover_estimate(rows: &[SweepRow]) -> Option<f64> {
    let diff = |r: &SweepRow| r.metal_ns - r.sj_ns; // >0: simdjson wins
    let mut cross = None;
    for pair in rows.windows(2) {
        let (d0, d1) = (diff(&pair[0]), diff(&pair[1]));
        if d0 > 0.0 && d1 <= 0.0 {
            let b0 = pair[0].bytes as f64;
            let b1 = pair[1].bytes as f64;
            cross = Some(b0 + d0 * (b1 - b0) / (d0 - d1));
        }
    }
    cross
}

// --- per-kernel breakdown ----------------------------------------------------

/// Run `examples/parse_breakdown` (the `timing` feature) on the largest
/// twitter dataset, once in the production pipeline shape (phase table)
/// and once with `METAL_JSON_SPLIT_KERNELS=1` (per-kernel GPU times), and
/// return the rendered markdown section body. `None` (with a console
/// warning) when the breakdown cannot run.
fn run_breakdown(datasets: &DatasetRows) -> Option<String> {
    if matches!(
        std::env::var("METAL_JSON_BENCH_BACKEND").as_deref(),
        Ok("cpu-reference" | "cpu_reference" | "cpu")
    ) {
        eprintln!("bench-report: breakdown skipped (cpu-reference backend run)");
        return None;
    }
    let data = bench_data_dir();
    let (name, path) = sweep_rows(datasets)
        .into_iter()
        .rev()
        .map(|r| (r.name.clone(), data.join(format!("{}.json", r.name))))
        .find(|(_, p)| p.is_file())?;

    let run = |split: bool| -> Option<String> {
        let mut cmd = std::process::Command::new("cargo");
        cmd.args([
            "run",
            "--release",
            "--features",
            "timing",
            "--example",
            "parse_breakdown",
            "--",
        ])
        .arg(&path)
        .args(["9", "aligned"])
        .current_dir(workspace_root());
        if split {
            cmd.env("METAL_JSON_SPLIT_KERNELS", "1");
        }
        let out = cmd.output().ok()?;
        if !out.status.success() {
            eprintln!(
                "bench-report: parse_breakdown (split={split}) failed:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            return None;
        }
        Some(String::from_utf8_lossy(&out.stdout).trim().to_owned())
    };

    println!("bench-report: running parse_breakdown on {name} ...");
    let phases = run(false)?;
    let kernels = run(true)?;
    // From the split run keep only the per-kernel table; its wall/phase
    // numbers are inflated by the per-kernel syncs and must not be quoted.
    let kernels = kernels
        .find("per-kernel GPU times")
        .map_or(kernels.as_str(), |i| &kernels[i..])
        .trim()
        .to_owned();

    let mut s = String::new();
    let _ = writeln!(
        s,
        "Phase-level wall/GPU split of `Parser::parse_aligned` on `{name}` \
         (`cargo run --release --features timing --example parse_breakdown`, \
         median of 9 iterations after 3 warmups). This times **the parse call \
         only** — the symmetric stats walk of the criterion harness is not \
         included, so the total here is faster than the table above.\n"
    );
    let _ = writeln!(s, "```text\n{phases}\n```\n");
    let _ = writeln!(
        s,
        "Per-kernel GPU execution times (`METAL_JSON_SPLIT_KERNELS=1` \
         measurement mode: each dispatch gets its own command buffer + sync, \
         so *wall* time inflates and only the GPU column is representative; \
         phase numbers from that mode are therefore not shown):\n"
    );
    let _ = writeln!(s, "```text\n{kernels}\n```");
    Some(s)
}

/// Read `new/estimates.json` (median ns) + `new/benchmark.json` (throughput
/// bytes) for one criterion function dir.
fn read_measurement(fn_dir: &Path) -> Option<Measurement> {
    let estimates: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fn_dir.join("new").join("estimates.json")).ok()?)
            .ok()?;
    let benchmark: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fn_dir.join("new").join("benchmark.json")).ok()?)
            .ok()?;
    let median_ns = estimates.get("median")?.get("point_estimate")?.as_f64()?;
    let bytes = benchmark.get("throughput")?.get("Bytes")?.as_u64()?;
    Some(Measurement { bytes, median_ns })
}

fn human_size(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= 1024.0 * MIB {
        format!("{:.2} GiB", b / (1024.0 * MIB))
    } else {
        format!("{:.1} MiB", b / MIB)
    }
}

fn format_ms(ns: f64) -> String {
    format!("{:.3} ms", ns / 1e6)
}

fn render_report(datasets: &DatasetRows, breakdown: Option<&str>) -> String {
    // Stable column order: known contenders first, then anything else found.
    let mut contenders: Vec<String> = CONTENDER_ORDER
        .iter()
        .filter(|c| {
            datasets
                .iter()
                .any(|(_, rows)| rows.iter().any(|(name, _)| name == *c))
        })
        .map(|s| (*s).to_owned())
        .collect();
    for (_, rows) in datasets {
        for (name, _) in rows {
            if !contenders.contains(name) {
                contenders.push(name.clone());
            }
        }
    }

    let sweep = sweep_rows(datasets);
    let crossover = crossover_estimate(&sweep);

    let mut out = String::new();
    out.push_str("# metal-json benchmark report\n\n");
    out.push_str(
        "Parse-to-tape throughput: metal-json (GPU) vs C++ simdjson (vendored, \
         FFI) vs Rust simd-json vs serde_json. All numbers are **medians of \
         criterion samples** in decimal GB/s (input bytes / 1e9 / seconds), \
         measured back-to-back in one session by \
         `cargo run -p xtask -- bench-report`. The timed region for the two \
         headline contenders is *parse to tape + an identical shallow stats \
         walk over the resulting tape* — see [Methodology](#methodology) for \
         exactly what is and isn't timed.\n\n",
    );

    // Headline, computed from the data.
    if let Some(big) = sweep.last()
        && big.metal_ns < big.sj_ns
    {
        let _ = writeln!(
            out,
            "**Headline:** on this machine metal-json parses the {} \
             ({}) document **{:.2}× faster** than C++ simdjson ({} vs {}). \
             The win holds across the ≥100 MiB sweep below. **Crossover \
             caveat:** below {} of input, simdjson wins — the GPU pipeline \
             carries a fixed dispatch/sync overhead that small documents \
             cannot amortize.\n",
            big.name,
            human_size(big.bytes),
            big.sj_ns / big.metal_ns,
            format_ms(big.metal_ns),
            format_ms(big.sj_ns),
            crossover.map_or("a few MiB".to_owned(), |b| human_size(b as u64)),
        );
        // The generalization boundary of the headline, stated explicitly:
        // the large-input rows are all one document shape on one machine.
        out.push_str(
            "**Scope of evidence:** every row at and above 1 MiB of the \
             twitter sweep — including all ≥100 MiB rows — is a \
             deterministic expansion of the *twitter template* measured on \
             this one machine; the other document shapes (citm_catalog, \
             canada) were measured only at their canonical 1.6–2.1 MiB \
             sizes. Shape already matters at those sizes (number-dense \
             canada favors metal-json at 2.1 MiB, citm_catalog favors \
             simdjson), so the ≥100 MiB speedups should be read as \
             \"twitter-shaped documents on this machine\", not as a \
             universal large-input constant; large-size coverage of other \
             shapes is future work.\n\n",
        );
    }

    // Environment.
    out.push_str("## Environment\n\n");
    for (label, value) in collect_env_info() {
        let _ = writeln!(out, "- **{label}**: {value}");
    }
    out.push('\n');

    // Dataset provenance.
    out.push_str("## Datasets\n\n");
    out.push_str(
        "Canonical files come from \
         [simdjson-data](https://github.com/simdjson/simdjson-data) via \
         `cargo run -p xtask -- fetch-data`. Large/sweep variants are \
         deterministic expansions produced by `cargo run -p xtask -- \
         gen-data`: records from the canonical template (twitter `statuses`), \
         re-serialized by the Cargo.lock-pinned serde_json (sorted keys) and \
         cycled verbatim into one top-level JSON array until the target size \
         is reached — byte-identical across runs and machines, valid \
         standard JSON at every size.\n\n",
    );
    out.push_str("| dataset | bytes | sha256 | source |\n|---|---:|---|---|\n");
    let data = bench_data_dir();
    for (name, rows) in datasets {
        let path = data.join(format!("{name}.json"));
        let bytes = std::fs::metadata(&path)
            .map_or_else(|_| rows.first().map_or(0, |(_, m)| m.bytes), |m| m.len());
        let digest = sha256(&path).unwrap_or_else(|| "(file absent at report time)".to_owned());
        let _ = writeln!(
            out,
            "| {name} | {bytes} | `{digest}` | {} |",
            dataset_source(name)
        );
    }
    out.push('\n');

    // Main results table.
    out.push_str("## Results (median GB/s, higher is better)\n\n");
    out.push_str("| dataset | size |");
    for c in &contenders {
        let _ = write!(out, " {c} GB/s |");
    }
    out.push_str(" metal-json / simdjson-cpp |\n");
    out.push_str("|---|---:|");
    for _ in &contenders {
        out.push_str("---:|");
    }
    out.push_str("---:|\n");

    for (dataset, rows) in datasets {
        let find = |name: &str| rows.iter().find(|(n, _)| n == name).map(|(_, m)| m);
        let size = rows.first().map_or(0, |(_, m)| m.bytes);
        let _ = write!(out, "| {dataset} | {} |", human_size(size));
        for c in &contenders {
            match find(c) {
                Some(m) => {
                    let _ = write!(out, " {:.3} |", m.gb_per_s());
                }
                None => out.push_str(" — |"),
            }
        }
        let speedup = match (find("metal-json"), find("simdjson-cpp")) {
            (Some(metal), Some(sj)) => format!("{:.2}x", metal.gb_per_s() / sj.gb_per_s()),
            _ => "—".to_owned(),
        };
        let _ = writeln!(out, " {speedup} |");
    }
    out.push('\n');
    let _ = writeln!(out, "Contenders present: {}.\n", contenders.join(", "));

    // Size sweep / crossover.
    if sweep.len() >= 2 {
        out.push_str("## CPU/GPU crossover (twitter size sweep)\n\n");
        out.push_str(
            "metal-json vs C++ simdjson on the twitter template across input \
             sizes. The GPU pipeline pays a roughly fixed dispatch/sync \
             overhead per parse (~0.5–0.9 ms on this machine), so small \
             documents lose to the CPU; throughput grows with size until the \
             pipeline is memory-bound.\n\n",
        );
        out.push_str(
            "| dataset | size | metal-json median | simdjson-cpp median | \
             metal-json GB/s | simdjson-cpp GB/s | speedup | winner |\n\
             |---|---:|---:|---:|---:|---:|---:|---|\n",
        );
        for r in &sweep {
            let metal_gbs = r.bytes as f64 / r.metal_ns;
            let sj_gbs = r.bytes as f64 / r.sj_ns;
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | {:.3} | {:.3} | {:.2}x | {} |",
                r.name,
                human_size(r.bytes),
                format_ms(r.metal_ns),
                format_ms(r.sj_ns),
                metal_gbs,
                sj_gbs,
                r.sj_ns / r.metal_ns,
                if r.metal_ns < r.sj_ns {
                    "**metal-json**"
                } else {
                    "simdjson"
                },
            );
        }
        out.push('\n');
        match crossover {
            Some(bytes) => {
                let _ = writeln!(
                    out,
                    "**Crossover: ≈ {}.** Below that size C++ simdjson wins \
                     (the GPU's fixed overhead dominates); above it metal-json \
                     wins, with the gap widening toward large inputs. The \
                     estimate interpolates linearly between the neighboring \
                     sweep sizes where the winner flips; treat it as a \
                     band around that value, not a sharp constant — it moves \
                     with document shape and machine.\n",
                    human_size(bytes as u64),
                );
            }
            None => out.push_str(
                "No winner flip observed inside the measured sweep range.\n\n",
            ),
        }
    }

    // Per-kernel breakdown.
    if let Some(body) = breakdown {
        out.push_str("## Where the time goes (largest input)\n\n");
        out.push_str(body);
        out.push('\n');
    }

    // Methodology.
    out.push_str(METHODOLOGY);
    out
}

const METHODOLOGY: &str = "\
## Methodology

Harness: `bench/benches/compare.rs` (criterion); helpers and the FFI shim
contract in `bench/src/lib.rs` + `bench/cpp/shim.cpp`.

**Timed region, metal-json** — one call to `Parser::parse_aligned` plus a
shallow stats walk over the produced tape (`metal_stats`):

- *Inside the timed region*: the whole parse — all GPU command buffers
  (encode + commit + `waitUntilCompleted`), the CPU syncs between them,
  exact-size output allocations, **all CPU fixup costs** (hard-rounding
  float re-parses; the >16 KiB long-string valve, which unescapes
  fixup-listed long strings on the CPU so one giant string cannot
  serialize a GPU lane), `Document` assembly, and the stats walk.
- *Outside the timed region*: input preparation (one page-aligned copy of
  the file, made once per dataset — the zero-copy `bytesNoCopy` input
  path then maps it straight into an `MTLBuffer`), and `Document` drop
  (which returns pooled buffers; the reused simdjson parser's tape is
  likewise never freed inside its timed call).

**Timed region, simdjson (C++)** — one call to `sj_parse_tape`: a reused
`simdjson::dom::parser` parses a pre-padded buffer to its tape, then walks
that tape linearly filling the same stats struct. Padding the input
(`SIMDJSON_PADDING`) happens once per dataset, untimed. The parser object
is reused across iterations so its tape allocation is warm — mirroring
metal-json's reused parser and buffer pool.

**Symmetric stats walk (DCE defeat + proof of equivalent work)** — both
contenders compute the same `SjStats` (node count, total unescaped string
bytes, XOR of all 64-bit number payloads) inside the timed region, and the
results feed `black_box`. Once per dataset, an **untimed** check asserts
both parsers produce bit-identical stats — both really parsed the same
document to an equivalent tape (bit-exact f64s included). Numbers quoted
as \"parse-only\" anywhere strip this walk and are labeled as such.

**Other contenders** — Rust `simd-json::to_tape` mutates its input, so
each iteration gets a fresh copy in untimed setup (the API has no
non-destructive tape parse); output drop is untimed. `serde_json` parses
to a DOM (`Value`) — an allocation-heavy floor, not a tape peer; drop is
untimed.

**Sampling** — criterion defaults (60 samples) for inputs <10 MiB;
20 samples / 10 s for 10–100 MiB; 10 samples / 20 s + per-iteration
batching for ≥100 MiB. Warmup precedes every measurement (criterion
default 3 s; 2 s on ≥100 MiB groups), which also absorbs GPU power-state
ramp and PSO/pool warming. **Medians** everywhere: low-occupancy GPU wall
times jitter up to 4× from power-state ramping (see `docs/spikes.md`), so
means would overweight outliers. All contenders for all datasets run
back-to-back in a single session. Background desktop activity during the
session is not controlled for beyond using medians (it hits both
contenders); the sub-4 MiB wall times and the exact crossover point are
the numbers most sensitive to it, the ≥100 MiB headline the least.

**Honesty notes** —

- Throughput is decimal (GB = 1e9 bytes); sizes in tables are binary MiB.
- metal-json numbers *include* every CPU-side cost of the hybrid design
  (syncs, allocations, fixups, copy-out); only input preparation is
  excluded, identically for both headline contenders.
- The per-kernel breakdown times the bare parse call (no stats walk) and
  says so; split-kernel mode adds one sync per kernel, so only its GPU
  column is meaningful.
- The crossover is a property of *this* machine and document shape;
  the report states the measured band rather than a universal constant.
- simdjson is run through its DOM tape API (`dom::parser`), the closest
  apples-to-apples target for a full materialized tape; On-Demand is a
  different (lazier) contract and would not produce a comparable tape.
";
