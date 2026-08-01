# Backfilled mouse trajectory data

These CSVs were migrated from the pre-schema-version-1 layout to schema version 2 by `backfill` (crates/mouse-telemetry/src/bin/backfill.rs).

## What was and was not changed

The original 23 columns are copied through unchanged, including `time_between_movements` — it still reports the raw drain-time delta. Only the 7 schema-version-2 columns are appended:
`monotonic_ns`, `batch_id`, `event_seq`, `viewport_width`, `viewport_height`, `coord_space`, `schema_version`.

## batch_id is inferred, not recorded

These files predate the collector writing `batch_id` directly. It is reconstructed here from a burst threshold (0.001 s): consecutive rows closer together than that, within the same recorder segment, are treated as one delivery batch. `manifest.json` records the implied refresh rate this produces per file — a value near a real display rate (60 / 70 / 75 / 90 Hz) is the evidence the reconstruction is trustworthy for that file; treat any file without a value in that range with suspicion.

## viewport is an estimate

The window/canvas size was never recorded in these files and cannot be recovered. `viewport_width`/`viewport_height` are left empty unless `--infer-viewport` was passed, in which case they hold the observed cursor extent — a biased lower bound, not the true viewport. `manifest.json` records this estimate for every file regardless, alongside a click-position cross-check, so the choice to trust it is visible rather than silent.

## Source data is untouched

Nothing under `data/` outside this directory was modified.
