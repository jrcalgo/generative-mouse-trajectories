# Mouse Collection Environment

A desktop GUI for collecting, processing, and exporting mouse movement data for ML/analytics. It uses Iced for UI and Winit for event capture. Data are batched by left-clicks, derived features are computed, and results are saved to timestamped CSV files.

## Features
- Start/Stop collection with a simple game-like button layout (randomized clickable targets)
- Batch segmentation by Left click (each batch written on click)
- Real-time derived stats display (3-column grid, scrollable)
  - Average/Peak velocity, Average/Peak acceleration
  - Average/Peak momentum (unit-mass momentum = speed)
  - Smoothness (1 / average |jerk|), Deviation from ideal path
  - Idle time, Fitts ID, Fitts movement time
- CSV output with unique filename: `mouse_data_{YYYYMMDD_HHMMSS}[_n].csv`
- Append-only writing with header written once
- Tunable smoothing and derivative clamp to tame spikes (from OS micro-intervals)
- Settings panel for:
  - Button count range (min/max)
  - Data directory (where CSV is stored)
  - Min dt (s) clamp and EMA alpha
  - Reset to defaults
- Collapsible hotbar (▼ to collapse, ▲ to reopen)

## Build & Run
Requirements: Rust toolchain (stable), a desktop environment supported by `iced`/`winit`.

```bash
cd mouse-collection-environment
cargo run --release
```

## Run as Library
```toml
[dependencies]
mouse-collection-environment = { path = "path/to/mouse-collection-environment" }
once_cell = "1"
tokio = { version = "1", features = ["full"] }
```

- Recorder thread only (feed events from your own UI):
```rust
use once_cell::sync::Lazy;
use std::sync::Arc;
use tokio::runtime::Runtime;
use mouse_collection_environment::mouse_collector::MouseCollector;

static RT: Lazy<Runtime> = Lazy::new(|| Runtime::new().expect("Failed to create runtime"));
static COLLECTOR: Lazy<Arc<MouseCollector>> =
    Lazy::new(|| RT.block_on(MouseCollector::new(false, false)));

fn main() {
    // Enter the Tokio runtime so the collector's internals can use it
    let _enter = RT.enter();

    // Start the recorder thread (no built-in listener window)
    let _recorder = MouseCollector::start_recording_thread(&COLLECTOR);

    // Optional tunables
    COLLECTOR.set_min_derivative_dt(0.006);
    COLLECTOR.set_smoothing_alpha(0.25);
    COLLECTOR.set_data_dir("data".to_string());

    // Push events from your own UI framework
    COLLECTOR.push_move((100.0, 200.0));
    COLLECTOR.push_left_click((100.0, 200.0));

    // Pull stats for your UI
    println!("{:?}", COLLECTOR.get_latest_derived_stats());
}
```

- Full in-window capture (listener + recorder):
```rust
use std::sync::Arc;
use mouse_collection_environment::mouse_collector::MouseCollector;

#[tokio::main]
async fn main() {
    let collector: Arc<MouseCollector> = MouseCollector::new(false, false).await;

    // Start the winit listener (window) + recorder; set `report` to true to print stats periodically
    let (_listen, _record, _report) = collector.clone().start_collecting(false).await;

    // Your app can continue doing work here. Drop the process to exit, or add your own shutdown logic.
}
```

**On macOS you may need to allow input monitoring (System Settings → Privacy & Security → Input Monitoring) if you extend this to global capture. This app currently captures inside its own window.**

## Usage
- Click "Start Collecting" to begin.
- Move the mouse and click the green buttons; each Left click delimits a batch that gets processed and appended to the CSV.
- The stats panel shows real-time derived metrics. If the window is small, scroll to view.
- Open Settings to adjust:
  - Min/Max buttons (spawns between min..=max)
  - Data dir (click "Apply Dir" to change)
  - Min dt (s) and Alpha (click "Apply Tunables")
  - Reset restores defaults for the button counts
- Collapse the hotbar with ▼; when collapsed, a bottom-right ▲ restores it.

## CSV Output
Files are written to the configured data directory (default: `data/`).
Each run generates a unique file name: `mouse_data_{YYYYMMDD_HHMMSS}.csv`; if it exists, `_n` is appended.

Columns:
- timestamp, total_duration, time_between_movements, hover_time
- position_x, position_y, velocity, acceleration, jerk, path_length
- click_events, scroll_events
- average_velocity, peak_velocity, average_acceleration, peak_acceleration
- average_momentum, peak_momentum, smoothness, deviation_from_ideal_path, idle_time
- fitts_index_of_difficulty, fitts_movement_time

Notes:
- time_between_movements logs the raw dt for transparency.
- Derivative calculations use `effective_dt = max(raw_dt, min_dt)`; velocity/acceleration are EMA-smoothed via `alpha`.
- Momentum is computed as unit-mass momentum (i.e., `|velocity|`).

## Tuning
- Min dt (s): Increase to limit spikes from micro-interval events (typical 0.004–0.010).
- Alpha (0–1): Higher smooths less (more responsive), lower smooths more (less noisy). Try 0.2–0.5.

## Architecture Overview
- `mouse_collector.rs` implements collection, derivation, batching, and CSV writing; tunables are stored as `RwLock<f64>` and read at compute time.
- `mouse_gui.rs` builds the Iced UI: canvas game area, stats grid, settings, and hotbar controls.
- Batching: events are buffered in memory; once a Left click arrives, the batch up to and including that click is processed and appended to CSV.

## License
[MIT](LICENSE)