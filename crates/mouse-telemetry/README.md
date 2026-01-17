## mouse-telemetry

**DISCLAIMER: This is intended for benevolent analytics and machine learning tasks; Do NOT use this for nefarious purposes**

High-fidelity mouse telemetry collector for Rust. Captures cursor movement, clicks, and scrolls; computes derived kinematics (velocity, acceleration, jerk, smoothness, Fitts-style indices); and writes everything to CSV for analysis.

### Highlights
- **Event capture**: Cursor moves, left/middle/right clicks, wheel up/down
- **Derived metrics**: Velocity, acceleration, jerk, momentum, smoothness, path deviation, idle time, simple Fitts metrics
- **CSV export**: Human-readable schema with a stable header
- **Runtime tuning**: Smoothing and derivative clamping controls
- **Live stats**: Summary strings for easy HUD/overlay display

## Quickstart

You can run the collector in two ways:

### 1) Spawn an internal listener window
The library will open a minimal window and record mouse activity within it. A left-click marks the end of a batch; batches are processed and appended to CSV.

```rustgit 
use std::time::Duration;
use std::sync::Arc;
use mouse_collector::mouse_collector::MouseCollector;

#[tokio::main]
async fn main() {
    // Start immediately; disable CLI reporting (stats available via APIs below)
    let collector: Arc<MouseCollector> = MouseCollector::new(true, false).await;

    // Optional: change where CSV is written
    collector.set_data_dir("data".to_string());

    // Keep the app alive; interact with the spawned window
    // Each left-click will finalize the current batch and write rows to CSV.
    loop { tokio::time::sleep(Duration::from_secs(1)).await; }
}
```

Notes:
- The listener records events that occur within the spawned window (focus matters).
- A left-click finalizes the current batch and triggers CSV writes.

### 2) Integrate with your own app (push events programmatically)
Use your own event source, push events to the collector, and run only the recorder thread.

```rust
use std::{sync::Arc, thread, time::Duration};
use mouse_collector::mouse_collector::MouseCollector;

#[tokio::main]
async fn main() {
    let collector: Arc<MouseCollector> = MouseCollector::new(false, false).await;

    // Write output to a custom directory/filename
    collector.set_data_dir("data".to_string());
    if let Ok(mut name) = collector.record_filename.write() {
        *name = "session_01.csv".to_string();
    }

    // Start only the CSV recording thread
    let _recorder = collector.start_recording_thread();

    // Feed your own events
    collector.push_move((100.0, 100.0));
    collector.push_move((120.0, 110.0));
    collector.push_move((180.0, 140.0));

    // A left-click marks the end of the current batch and triggers CSV write
    collector.push_left_click((180.0, 140.0));

    thread::sleep(Duration::from_millis(500));
}
```

## Live stats (for overlays/HUDs)

- `get_latest_derived_summary()` → single compact string
- `get_latest_derived_stats()` → Vec<String> for column/row UI layouts

```rust
let summary = collector.get_latest_derived_summary();
let stats = collector.get_latest_derived_stats();
println!("{summary}");
for s in stats { println!("{s}"); }
```

## Configuration

### Output location
- Default directory: `data/`
- Default filename: `mouse_data_YYYYMMDD_HHMMSS.csv` (auto-increments if the file exists)

```rust
collector.set_data_dir("my_data".to_string());
if let Ok(mut name) = collector.record_filename.write() {
    *name = "experiment_a.csv".to_string();
}
```

### Smoothing and derivative clamp
These influence stability of velocity/acceleration estimates:

```rust
// Clamp very small dt (seconds) to avoid huge derivatives (default ~0.004)
collector.set_min_derivative_dt(0.003);

// Exponential moving average alpha in (0, 1]; higher = more responsive
collector.set_smoothing_alpha(0.35);
```

## How batching and writing work

- Events are buffered in memory.
- A left-click acts as a delimiter that closes the current batch.
- When a batch is closed and has >1 event, the collector:
  - computes kinematics and derived metrics
  - appends rows to the CSV file

Tip: If you drive the collector programmatically, ensure you call `push_left_click` periodically to persist data in well-defined segments (e.g., per task, per gesture, per trial).

## CSV schema

All numeric values are floating-point unless noted. Units: time in seconds, positions in pixels, speeds in pixels/second.

| column | description |
|---|---|
| `timestamp` | Event time as seconds since UNIX epoch (stringified for consistency) |
| `total_duration` | Elapsed time since the first event in the batch |
| `time_between_movements` | Delta time since the previous event (raw dt) |
| `hover_time` | Accumulated time below small-movement threshold |
| `position_x` | Cursor x position |
| `position_y` | Cursor y position |
| `velocity` | Smoothed speed |
| `acceleration` | Smoothed acceleration |
| `jerk` | Rate of change of acceleration |
| `path_length` | Cumulative path length since batch start |
| `click_events` | `Left`, `Right`, `Middle`, or `None` |
| `scroll_events` | `Up`, `Down`, or `None` |
| `average_velocity` | Running average speed up to this event |
| `peak_velocity` | Max speed observed so far in the batch |
| `average_acceleration` | Running average acceleration |
| `peak_acceleration` | Max acceleration observed so far |
| `average_momentum` | Running average of unit-mass momentum (equals speed) |
| `peak_momentum` | Peak momentum observed so far |
| `smoothness` | Inverse of average absolute jerk (higher = smoother) |
| `deviation_from_ideal_path` | Ratio: path_length / straight-line distance |
| `idle_time` | `total_duration - active_time` approximation |
| `fitts_index_of_difficulty` | Simple ID estimate based on progress |
| `fitts_movement_time` | Same as `total_duration` (per-event running value) |

## Platform notes

- Uses `winit` for portable window/input handling (Linux, Windows, macOS).
- Listener mode records events within the library’s window. For global OS-level hooks, integrate with your own input source and push events via `push_move` / `push_left_click`.

## Troubleshooting

- **No rows appear in the CSV**: Ensure a left-click occurs to close a batch. If you push events programmatically, call `push_left_click` to finalize.
- **No CSV file created**: Verify the `data` directory exists or call `set_data_dir`. Also confirm your process has write permissions.
- **Stats look noisy**: Lower `set_smoothing_alpha` or increase `set_min_derivative_dt`.
- **Nothing is recorded in listener mode**: Make sure the spawned window is focused and your cursor is inside it.

## License
[MIT](LICENSE)

