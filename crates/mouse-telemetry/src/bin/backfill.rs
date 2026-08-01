//! Backfills CSV files recorded under the pre-`SCHEMA_VERSION` 2 layout with
//! the columns schema version 2 added (`monotonic_ns`, `batch_id`,
//! `event_seq`, `viewport_width`, `viewport_height`, `coord_space`,
//! `schema_version`).
//!
//! The 23 original columns are never modified: this tool appends, it does
//! not rewrite. In particular `time_between_movements` and the position
//! columns are copied through byte-for-byte. Grouping coalesced pointer
//! events by a burst threshold is a reconstruction, and reconstructions
//! belong in preprocessing where they can be tuned or swapped, not baked
//! irreversibly into stored data — the same reasoning that keeps the raw
//! delta in the live collector once `batch_id` makes it interpretable.
//!
//! Usage:
//!   backfill [--out <dir>] [--infer-viewport] [<input>...]
//!
//! `<input>` defaults to every `*.csv` directly under `data/`. `--out`
//! defaults to `data/backfilled/`. Files already at the current schema are
//! left untouched (not an error, so re-running is idempotent even if the
//! output directory is later included as an input by accident).

use std::error::Error;
use std::ffi::OsStr;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use mouse_telemetry::mouse_telemetry::{CSV_COLUMNS, LEGACY_CSV_COLUMNS, SCHEMA_VERSION};

/// Column indices within [`LEGACY_CSV_COLUMNS`] this tool reads.
mod col {
    pub const TIMESTAMP: usize = 0;
    pub const TOTAL_DURATION: usize = 1;
    pub const TIME_BETWEEN_MOVEMENTS: usize = 2;
    pub const POSITION_X: usize = 4;
    pub const POSITION_Y: usize = 5;
    pub const CLICK_EVENTS: usize = 10;
}

/// Events drained together in one delivery batch sit microseconds apart,
/// while real frame gaps are 11-17 ms — four orders of magnitude apart. This
/// splits comfortably below the shortest plausible frame interval (~11 ms at
/// 91 Hz), so it cannot merge two real frames together. Matches the
/// collector's own `push`-path semantics: a gap at or above this is real
/// motion, not drain overhead.
const BURST_THRESHOLD_SECONDS: f64 = 0.001;

fn main() {
    let args = match Args::parse(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}\n\n{}", Args::usage());
            std::process::exit(2);
        }
    };

    let inputs = match resolve_inputs(&args.inputs) {
        Ok(inputs) => inputs,
        Err(e) => {
            eprintln!("Failed to resolve input files: {e}");
            std::process::exit(1);
        }
    };

    if inputs.is_empty() {
        eprintln!("No input CSV files found.");
        std::process::exit(1);
    }

    let mut reports = Vec::new();
    let mut had_error = false;

    for input in &inputs {
        match backfill_file(input, &args.out_dir, args.infer_viewport) {
            Ok(Some(report)) => {
                println!(
                    "{}: {} rows, {} batches (mean size {:.2}){}",
                    input.display(),
                    report.rows,
                    report.batches,
                    report.mean_batch_size,
                    report
                        .implied_refresh_hz
                        .map(|hz| format!(", ~{hz:.1} Hz"))
                        .unwrap_or_default(),
                );
                reports.push(report);
            }
            Ok(None) => {
                println!("{}: already at schema version {SCHEMA_VERSION}, skipped", input.display());
            }
            Err(e) => {
                eprintln!("{}: {e}", input.display());
                had_error = true;
            }
        }
    }

    if !reports.is_empty() {
        if let Err(e) = write_manifest(&reports, &args.out_dir, args.infer_viewport) {
            eprintln!("Failed to write manifest: {e}");
            had_error = true;
        }
        if let Err(e) = write_readme(&args.out_dir) {
            eprintln!("Failed to write README: {e}");
            had_error = true;
        }
    }

    if had_error {
        std::process::exit(1);
    }
}

struct Args {
    inputs: Vec<PathBuf>,
    out_dir: PathBuf,
    infer_viewport: bool,
}

impl Args {
    fn usage() -> &'static str {
        "Usage: backfill [--out <dir>] [--infer-viewport] [<input>...]"
    }

    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut inputs = Vec::new();
        let mut out_dir = PathBuf::from("data/backfilled");
        let mut infer_viewport = false;

        let mut args = args.peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "--out requires a directory argument".to_string())?;
                    out_dir = PathBuf::from(value);
                }
                "--infer-viewport" => infer_viewport = true,
                "-h" | "--help" => return Err(Self::usage().to_string()),
                other => inputs.push(PathBuf::from(other)),
            }
        }

        Ok(Self {
            inputs,
            out_dir,
            infer_viewport,
        })
    }
}

/// Expands explicit inputs (files pass through, directories are scanned
/// non-recursively for `*.csv`) or, if none were given, scans `data/`.
fn resolve_inputs(inputs: &[PathBuf]) -> std::io::Result<Vec<PathBuf>> {
    if inputs.is_empty() {
        return scan_csv_dir(Path::new("data"));
    }
    let mut resolved = Vec::new();
    for input in inputs {
        if input.is_dir() {
            resolved.extend(scan_csv_dir(input)?);
        } else {
            resolved.push(input.clone());
        }
    }
    Ok(resolved)
}

fn scan_csv_dir(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && path.extension().and_then(OsStr::to_str) == Some("csv")
        })
        .collect();
    files.sort();
    Ok(files)
}

#[derive(Debug)]
enum BackfillError {
    Io(std::io::Error),
    Csv(csv::Error),
    HeaderMismatch { columns: usize },
    ParseFloat { row: usize, field: &'static str },
}

impl fmt::Display for BackfillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackfillError::Io(e) => write!(f, "I/O error: {e}"),
            BackfillError::Csv(e) => write!(f, "CSV error: {e}"),
            BackfillError::HeaderMismatch { columns } => write!(
                f,
                "header has {columns} columns, expected {} (legacy) or {} (current) — skipping, refusing to guess",
                LEGACY_CSV_COLUMNS.len(),
                CSV_COLUMNS.len()
            ),
            BackfillError::ParseFloat { row, field } => {
                write!(f, "row {row}: could not parse '{field}' as a number")
            }
        }
    }
}

impl Error for BackfillError {}

impl From<std::io::Error> for BackfillError {
    fn from(e: std::io::Error) -> Self {
        BackfillError::Io(e)
    }
}

impl From<csv::Error> for BackfillError {
    fn from(e: csv::Error) -> Self {
        BackfillError::Csv(e)
    }
}

/// Per-file statistics, written into the migration manifest.
struct FileReport {
    source: PathBuf,
    rows: usize,
    batches: usize,
    mean_batch_size: f64,
    median_batch_interval_seconds: Option<f64>,
    implied_refresh_hz: Option<f64>,
    viewport_estimate: (f64, f64),
    viewport_written: bool,
    click_count: usize,
    click_extent: Option<(f64, f64)>,
}

/// One parsed row, retained in memory so the per-file viewport estimate
/// (which needs the whole file) can be computed before rows are written.
struct Row {
    raw: csv::StringRecord,
    monotonic_ns: u128,
    batch_id: u64,
    event_seq: u64,
}

/// Backfills one file. Returns `Ok(None)` if the file is already current
/// (not an error — lets a directory scan include already-migrated files
/// without failing), `Ok(Some(report))` on success.
fn backfill_file(
    input: &Path,
    out_dir: &Path,
    infer_viewport: bool,
) -> Result<Option<FileReport>, BackfillError> {
    let mut reader = csv::ReaderBuilder::new().from_path(input)?;
    let header = reader.headers()?.clone();

    if header_equals(&header, &CSV_COLUMNS) {
        return Ok(None);
    }
    if !header_equals(&header, &LEGACY_CSV_COLUMNS) {
        return Err(BackfillError::HeaderMismatch {
            columns: header.len(),
        });
    }

    // Written here rather than left to the caller: `main` happens to create
    // `out_dir` up front, but that's an accident of call order, not a
    // contract this function can rely on.
    fs::create_dir_all(out_dir)?;

    let mut rows: Vec<Row> = Vec::new();
    let mut batch_start_ns: Vec<u128> = Vec::new();
    let mut prev_monotonic_ns: u128 = 0;
    let mut batch_id: u64 = 0;
    let mut first_timestamp: Option<f64> = None;
    let mut segment_base = 0.0_f64;

    let mut max_x = 0.0_f64;
    let mut max_y = 0.0_f64;
    let mut click_count = 0usize;
    let mut click_max: Option<(f64, f64)> = None;

    for (row_index, result) in reader.records().enumerate() {
        let record = result?;

        let timestamp = parse_field(&record, col::TIMESTAMP, "timestamp", row_index)?;
        let total_duration =
            parse_field(&record, col::TOTAL_DURATION, "total_duration", row_index)?;
        let time_between_movements = parse_field(
            &record,
            col::TIME_BETWEEN_MOVEMENTS,
            "time_between_movements",
            row_index,
        )?;
        let position_x = parse_field(&record, col::POSITION_X, "position_x", row_index)?;
        let position_y = parse_field(&record, col::POSITION_Y, "position_y", row_index)?;

        if row_index == 0 {
            first_timestamp = Some(timestamp);
            segment_base = 0.0;
        } else {
            // A new recorder segment always starts with total_duration == 0
            // (its first row also carries dt == 0, but the segment boundary
            // is what actually matters — checking both is a safety net).
            let is_new_segment = total_duration == 0.0;
            let is_burst_boundary = time_between_movements > BURST_THRESHOLD_SECONDS;
            if is_new_segment {
                segment_base = timestamp - first_timestamp.unwrap_or(timestamp);
            }
            if is_new_segment || is_burst_boundary {
                batch_id += 1;
            }
        }

        let candidate_seconds = segment_base + total_duration;
        let candidate_ns = if candidate_seconds.is_finite() && candidate_seconds >= 0.0 {
            (candidate_seconds * 1e9).round() as u128
        } else {
            0
        };
        // total_duration is monotonic within a segment (verified against the
        // real data), but segment_base is derived from 100us-resolution
        // timestamps and can occasionally place a new segment's start at or
        // just before the previous segment's end. Clamp so monotonic_ns
        // never goes backwards across that boundary.
        let monotonic_ns = candidate_ns.max(prev_monotonic_ns);

        // Record the start of each batch (by monotonic_ns) so inter-batch
        // intervals can be derived afterwards.
        if row_index == 0 || rows.last().map(|r| r.batch_id) != Some(batch_id) {
            batch_start_ns.push(monotonic_ns);
        }

        max_x = max_x.max(position_x);
        max_y = max_y.max(position_y);
        if record.get(col::CLICK_EVENTS) == Some("Left") {
            click_count += 1;
            let (cx, cy) = click_max.unwrap_or((0.0, 0.0));
            click_max = Some((cx.max(position_x), cy.max(position_y)));
        }

        rows.push(Row {
            raw: record,
            monotonic_ns,
            batch_id,
            event_seq: row_index as u64,
        });
        prev_monotonic_ns = monotonic_ns;
    }

    let row_count = rows.len();
    let batches = batch_id as usize + 1;
    let mean_batch_size = row_count as f64 / batches as f64;

    let mut intervals: Vec<f64> = batch_start_ns
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64 / 1e9)
        .filter(|dt| *dt > 0.0)
        .collect();
    let median_batch_interval_seconds = median(&mut intervals);
    let implied_refresh_hz = median_batch_interval_seconds
        .filter(|dt| *dt > 0.0)
        .map(|dt| 1.0 / dt);

    let viewport_estimate = (max_x.ceil(), max_y.ceil());

    let out_path = out_dir.join(input.file_name().unwrap_or_else(|| OsStr::new("output.csv")));
    write_backfilled_csv(&out_path, &rows, viewport_estimate, infer_viewport)?;

    Ok(Some(FileReport {
        source: input.to_path_buf(),
        rows: row_count,
        batches,
        mean_batch_size,
        median_batch_interval_seconds,
        implied_refresh_hz,
        viewport_estimate,
        viewport_written: infer_viewport,
        click_count,
        click_extent: click_max,
    }))
}

fn write_backfilled_csv(
    out_path: &Path,
    rows: &[Row],
    viewport_estimate: (f64, f64),
    infer_viewport: bool,
) -> Result<(), BackfillError> {
    let mut writer = csv::Writer::from_path(out_path)?;
    writer.write_record(CSV_COLUMNS)?;

    let (viewport_width, viewport_height) = if infer_viewport {
        (viewport_estimate.0.to_string(), viewport_estimate.1.to_string())
    } else {
        (String::new(), String::new())
    };

    for row in rows {
        let new_fields: [String; CSV_COLUMNS.len() - LEGACY_CSV_COLUMNS.len()] = [
            row.monotonic_ns.to_string(),
            row.batch_id.to_string(),
            row.event_seq.to_string(),
            viewport_width.clone(),
            viewport_height.clone(),
            "canvas".to_string(),
            SCHEMA_VERSION.to_string(),
        ];
        let out_row: Vec<String> = row
            .raw
            .iter()
            .map(String::from)
            .chain(new_fields)
            .collect();
        assert_eq!(
            out_row.len(),
            CSV_COLUMNS.len(),
            "backfilled row width must match the current schema"
        );
        writer.write_record(&out_row)?;
    }
    writer.flush()?;
    Ok(())
}

fn header_equals(header: &csv::StringRecord, expected: &[&str]) -> bool {
    header.len() == expected.len() && header.iter().eq(expected.iter().copied())
}

fn parse_field(
    record: &csv::StringRecord,
    index: usize,
    field: &'static str,
    row_index: usize,
) -> Result<f64, BackfillError> {
    record
        .get(index)
        .and_then(|s| s.parse::<f64>().ok())
        .ok_or(BackfillError::ParseFloat {
            row: row_index,
            field,
        })
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    })
}

fn write_manifest(
    reports: &[FileReport],
    out_dir: &Path,
    infer_viewport: bool,
) -> std::io::Result<()> {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"schema_version\": {SCHEMA_VERSION},\n"));
    json.push_str(&format!(
        "  \"burst_threshold_seconds\": {BURST_THRESHOLD_SECONDS},\n"
    ));
    json.push_str(&format!("  \"infer_viewport\": {infer_viewport},\n"));
    json.push_str("  \"files\": [\n");
    for (i, report) in reports.iter().enumerate() {
        json.push_str("    {\n");
        json.push_str(&format!(
            "      \"source\": \"{}\",\n",
            json_escape(&report.source.display().to_string())
        ));
        json.push_str(&format!("      \"rows\": {},\n", report.rows));
        json.push_str(&format!("      \"batches\": {},\n", report.batches));
        json.push_str(&format!(
            "      \"mean_batch_size\": {:.4},\n",
            report.mean_batch_size
        ));
        json.push_str(&format!(
            "      \"median_batch_interval_seconds\": {},\n",
            opt_num(report.median_batch_interval_seconds, 6)
        ));
        json.push_str(&format!(
            "      \"implied_refresh_hz\": {},\n",
            opt_num(report.implied_refresh_hz, 2)
        ));
        json.push_str("      \"viewport_estimate\": {\n");
        json.push_str(&format!(
            "        \"width\": {},\n",
            report.viewport_estimate.0
        ));
        json.push_str(&format!(
            "        \"height\": {}\n",
            report.viewport_estimate.1
        ));
        json.push_str("      },\n");
        json.push_str(&format!(
            "      \"viewport_written\": {},\n",
            report.viewport_written
        ));
        json.push_str(
            "      \"viewport_estimate_note\": \"lower bound from observed cursor extent; window size was never recorded\",\n"
        );
        json.push_str("      \"click_cross_check\": ");
        match report.click_extent {
            Some((cx, cy)) => {
                json.push_str("{\n");
                json.push_str(&format!("        \"click_count\": {},\n", report.click_count));
                json.push_str(&format!("        \"click_max_x\": {cx},\n"));
                json.push_str(&format!("        \"click_max_y\": {cy}\n"));
                json.push_str("      }\n");
            }
            None => json.push_str("null\n"),
        }
        json.push_str(if i + 1 == reports.len() {
            "    }\n"
        } else {
            "    },\n"
        });
    }
    json.push_str("  ]\n");
    json.push_str("}\n");

    fs::write(out_dir.join("manifest.json"), json)
}

fn opt_num(value: Option<f64>, precision: usize) -> String {
    match value {
        Some(v) => format!("{v:.precision$}"),
        None => "null".to_string(),
    }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn write_readme(out_dir: &Path) -> std::io::Result<()> {
    let readme = format!(
        "# Backfilled mouse trajectory data\n\n\
         These CSVs were migrated from the pre-schema-version-{prev} layout to \
         schema version {cur} by `backfill` (crates/mouse-telemetry/src/bin/backfill.rs).\n\n\
         ## What was and was not changed\n\n\
         The original 23 columns are copied through unchanged, including \
         `time_between_movements` — it still reports the raw drain-time delta. \
         Only the 7 schema-version-{cur} columns are appended:\n\
         `monotonic_ns`, `batch_id`, `event_seq`, `viewport_width`, `viewport_height`, \
         `coord_space`, `schema_version`.\n\n\
         ## batch_id is inferred, not recorded\n\n\
         These files predate the collector writing `batch_id` directly. It is \
         reconstructed here from a burst threshold ({threshold} s): consecutive rows \
         closer together than that, within the same recorder segment, are treated as \
         one delivery batch. `manifest.json` records the implied refresh rate this \
         produces per file — a value near a real display rate (60 / 70 / 75 / 90 Hz) is \
         the evidence the reconstruction is trustworthy for that file; treat any file \
         without a value in that range with suspicion.\n\n\
         ## viewport is an estimate\n\n\
         The window/canvas size was never recorded in these files and cannot be \
         recovered. `viewport_width`/`viewport_height` are left empty unless \
         `--infer-viewport` was passed, in which case they hold the observed cursor \
         extent — a biased lower bound, not the true viewport. `manifest.json` records \
         this estimate for every file regardless, alongside a click-position \
         cross-check, so the choice to trust it is visible rather than silent.\n\n\
         ## Source data is untouched\n\n\
         Nothing under `data/` outside this directory was modified.\n",
        prev = SCHEMA_VERSION - 1,
        cur = SCHEMA_VERSION,
        threshold = BURST_THRESHOLD_SECONDS,
    );
    fs::write(out_dir.join("README.md"), readme)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds one legacy (23-column) row. Columns backfill doesn't read are
    /// filled with a placeholder — only timestamp, total_duration,
    /// time_between_movements, position_x/y and click_events matter here.
    fn legacy_row(timestamp: f64, total_duration: f64, dt: f64, x: f64, y: f64, click: &str) -> Vec<String> {
        vec![
            format!("{timestamp:.4}"),
            total_duration.to_string(),
            dt.to_string(),
            "0".to_string(),
            x.to_string(),
            y.to_string(),
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
            click.to_string(),
            "None".to_string(),
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
        ]
    }

    fn write_csv(path: &Path, header: &[&str], rows: &[Vec<String>]) {
        let mut writer = csv::Writer::from_path(path).expect("create test csv");
        writer.write_record(header).expect("write header");
        for row in rows {
            writer.write_record(row).expect("write row");
        }
        writer.flush().expect("flush test csv");
    }

    /// Reads a backfilled CSV back as (header, rows) for assertions.
    fn read_csv(path: &Path) -> (csv::StringRecord, Vec<csv::StringRecord>) {
        let mut reader = csv::Reader::from_path(path).expect("read backfilled csv");
        let header = reader.headers().expect("header").clone();
        let rows = reader.records().collect::<Result<Vec<_>, _>>().expect("rows");
        (header, rows)
    }

    fn column<'a>(header: &csv::StringRecord, row: &'a csv::StringRecord, name: &str) -> &'a str {
        let idx = header.iter().position(|c| c == name).unwrap_or_else(|| panic!("missing column {name}"));
        &row[idx]
    }

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!("backfill_test_{name}"));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// A burst of events drained together (dt <= 1ms) must share a batch_id;
    /// a real gap (dt > 1ms) must start a new one. Mirrors the pattern
    /// observed in the actual recordings: bursts at the display refresh
    /// interval, with near-zero dt inside each burst.
    #[test]
    fn batch_id_groups_bursts_and_splits_on_real_gaps() {
        let dir = TempDir::new("burst_pattern");
        let input = dir.path().join("in.csv");
        let out = dir.path().join("out");

        let rows = vec![
            legacy_row(1000.0000, 0.0000, 0.0000, 0.0, 0.0, "None"),      // batch 0
            legacy_row(1000.0005, 0.0005, 0.0005, 1.0, 0.0, "None"),      // batch 0 (500us < 1ms)
            legacy_row(1000.0180, 0.0180, 0.0175, 2.0, 0.0, "None"),      // batch 1 (17.5ms real gap)
            legacy_row(1000.0185, 0.0185, 0.0005, 3.0, 0.0, "None"),      // batch 1 (500us < 1ms)
            legacy_row(1000.0360, 0.0360, 0.0175, 4.0, 0.0, "None"),      // batch 2 (17.5ms real gap)
        ];
        write_csv(&input, &LEGACY_CSV_COLUMNS, &rows);

        let report = backfill_file(&input, &dir.path().join("out"), false)
            .expect("backfill should succeed")
            .expect("legacy file should produce a report");
        assert_eq!(report.batches, 3);
        assert_eq!(report.rows, 5);

        let (header, out_rows) = read_csv(&out.join("in.csv"));
        let batch_ids: Vec<&str> = out_rows.iter().map(|r| column(&header, r, "batch_id")).collect();
        assert_eq!(batch_ids, vec!["0", "0", "1", "1", "2"]);
    }

    /// A new recorder segment (total_duration == 0) must start a new batch
    /// even though its own dt is exactly 0 — the dt-only rule alone would
    /// wrongly merge it into the trailing batch of the previous segment.
    #[test]
    fn segment_boundary_forces_new_batch_despite_zero_dt() {
        let dir = TempDir::new("segment_boundary");
        let input = dir.path().join("in.csv");
        let out = dir.path().join("out");

        let rows = vec![
            legacy_row(1000.0000, 0.0000, 0.0000, 0.0, 0.0, "None"), // segment 1, batch 0
            legacy_row(1000.0200, 0.0200, 0.0200, 5.0, 0.0, "Left"), // segment 1, batch 1
            legacy_row(1005.0000, 0.0000, 0.0000, 0.0, 0.0, "None"), // segment 2 start: dt==0, but must still be a new batch
            legacy_row(1005.0005, 0.0005, 0.0005, 1.0, 0.0, "None"), // segment 2, same batch as previous
        ];
        write_csv(&input, &LEGACY_CSV_COLUMNS, &rows);

        let report = backfill_file(&input, &out, false)
            .expect("backfill should succeed")
            .expect("legacy file should produce a report");
        assert_eq!(report.batches, 3);

        let (header, out_rows) = read_csv(&out.join("in.csv"));
        let batch_ids: Vec<&str> = out_rows.iter().map(|r| column(&header, r, "batch_id")).collect();
        assert_eq!(batch_ids, vec!["0", "1", "2", "2"]);
    }

    /// monotonic_ns must never decrease, including across a segment boundary
    /// where the new segment's low-precision anchor timestamp could place it
    /// at or before the previous segment's last recorded instant.
    #[test]
    fn monotonic_ns_is_clamped_non_decreasing_across_segments() {
        let dir = TempDir::new("monotonic_clamp");
        let input = dir.path().join("in.csv");
        let out = dir.path().join("out");

        let rows = vec![
            // segment 1: 50ms of total_duration recorded under one anchor timestamp.
            legacy_row(1000.0000, 0.0000, 0.0000, 0.0, 0.0, "None"),
            legacy_row(1000.0000, 0.0500, 0.0500, 1.0, 0.0, "None"),
            // segment 2 restarts at the SAME anchor timestamp, so its naive
            // (segment_base + total_duration) would compute to 0 -- less
            // than segment 1's last value. It must be clamped forward instead.
            legacy_row(1000.0000, 0.0000, 0.0000, 0.0, 0.0, "None"),
        ];
        write_csv(&input, &LEGACY_CSV_COLUMNS, &rows);

        backfill_file(&input, &out, false)
            .expect("backfill should succeed")
            .expect("legacy file should produce a report");

        let (header, out_rows) = read_csv(&out.join("in.csv"));
        let ns: Vec<u128> = out_rows
            .iter()
            .map(|r| column(&header, r, "monotonic_ns").parse().unwrap())
            .collect();
        assert!(ns[1] > ns[0], "monotonic_ns must advance within a segment");
        assert!(
            ns[2] >= ns[1],
            "monotonic_ns must not go backwards across a segment boundary: {ns:?}"
        );
        // The clamp should hold it at the previous value exactly, since the
        // naive computation for row 2 is 0.
        assert_eq!(ns[2], ns[1]);
    }

    /// A file already at the current schema is left alone, not treated as an
    /// error -- so a directory scan can include already-migrated output
    /// without failing the whole run.
    #[test]
    fn already_current_schema_file_is_skipped() {
        let dir = TempDir::new("already_current");
        let input = dir.path().join("in.csv");
        let out = dir.path().join("out");

        let mut row: Vec<String> = legacy_row(1000.0, 0.0, 0.0, 0.0, 0.0, "None");
        row.extend(["0".to_string(), "0".to_string(), "0".to_string(), "".to_string(), "".to_string(), "canvas".to_string(), "2".to_string()]);
        write_csv(&input, &CSV_COLUMNS, &[row]);

        let result = backfill_file(&input, &out, false).expect("should not error");
        assert!(result.is_none(), "already-current file should be skipped, not reprocessed");
    }

    /// A header that is neither the legacy nor the current layout is a hard
    /// error: guessing at an unknown schema would risk silently mislabeling
    /// columns.
    #[test]
    fn unrecognized_header_is_rejected() {
        let dir = TempDir::new("bad_header");
        let input = dir.path().join("in.csv");
        write_csv(&input, &["a", "b", "c"], &[vec!["1".into(), "2".into(), "3".into()]]);

        let result = backfill_file(&input, &dir.path().join("out"), false);
        assert!(matches!(result, Err(BackfillError::HeaderMismatch { columns: 3 })));
    }

    /// viewport_width/height are withheld by default -- the true viewport
    /// was never recorded and the observed extent is only a lower bound.
    /// --infer-viewport opts into writing that estimate explicitly.
    #[test]
    fn viewport_columns_are_empty_unless_inferred() {
        let dir = TempDir::new("viewport");
        let input = dir.path().join("in.csv");
        let rows = vec![
            legacy_row(1000.0, 0.0, 0.0, 10.0, 20.0, "None"),
            legacy_row(1000.1, 0.1, 0.1, 900.0, 600.0, "Left"),
        ];
        write_csv(&input, &LEGACY_CSV_COLUMNS, &rows);

        // Default: withheld.
        let out_default = dir.path().join("out_default");
        backfill_file(&input, &out_default, false).unwrap().unwrap();
        let (header, out_rows) = read_csv(&out_default.join("in.csv"));
        for row in &out_rows {
            assert_eq!(column(&header, row, "viewport_width"), "");
            assert_eq!(column(&header, row, "viewport_height"), "");
        }

        // --infer-viewport: filled with the observed extent (ceiling of the
        // max cursor position seen anywhere in the file).
        let out_inferred = dir.path().join("out_inferred");
        let report = backfill_file(&input, &out_inferred, true).unwrap().unwrap();
        assert_eq!(report.viewport_estimate, (900.0, 600.0));
        let (header, out_rows) = read_csv(&out_inferred.join("in.csv"));
        for row in &out_rows {
            assert_eq!(column(&header, row, "viewport_width"), "900");
            assert_eq!(column(&header, row, "viewport_height"), "600");
        }
    }

    /// Every original column must survive the migration unchanged -- this is
    /// the load-bearing promise of the whole tool.
    #[test]
    fn original_columns_are_preserved_byte_for_byte() {
        let dir = TempDir::new("preserve_original");
        let input = dir.path().join("in.csv");
        let out = dir.path().join("out");
        let rows = vec![legacy_row(1000.1234, 0.0, 0.0, 12.5, 34.75, "Right")];
        write_csv(&input, &LEGACY_CSV_COLUMNS, &rows);

        backfill_file(&input, &out, false).unwrap().unwrap();

        let (_, in_rows) = read_csv(&input);
        let (_, out_rows) = read_csv(&out.join("in.csv"));
        for i in 0..LEGACY_CSV_COLUMNS.len() {
            assert_eq!(in_rows[0][i], out_rows[0][i], "column {i} must be unchanged");
        }
    }

    /// CLI parsing: positional inputs, --out, and --infer-viewport.
    #[test]
    fn args_parse_flags_and_positionals() {
        let args = Args::parse(
            ["--out", "/tmp/x", "--infer-viewport", "a.csv", "b.csv"]
                .into_iter()
                .map(String::from),
        )
        .expect("parse should succeed");
        assert_eq!(args.out_dir, PathBuf::from("/tmp/x"));
        assert!(args.infer_viewport);
        assert_eq!(args.inputs, vec![PathBuf::from("a.csv"), PathBuf::from("b.csv")]);
    }

    #[test]
    fn args_out_without_value_is_an_error() {
        let result = Args::parse(["--out"].into_iter().map(String::from));
        assert!(result.is_err());
    }
}
