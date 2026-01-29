/*!
# Mouse Collector Module

This module implements a comprehensive mouse event collector for capturing, processing,
and recording mouse movements, clicks, and scroll events. It gathers raw data,
computes derived metrics (such as velocity, acceleration, jerk, smoothness, and Fitts's indices),
and exports the results to a CSV file for further analysis. The design leverages both
asynchronous tasks (via Tokio) and multi-threading (via std::thread) and handles window events using the `winit` crate.
*/

use chrono::{DateTime, Local};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::SystemTime;
use std::{thread, time};
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::RwLock as TokioRwLock;
use tokio::task;
use winit::application::ApplicationHandler;
use winit::event::{MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

/// Constant for the directory in which data files (CSV) will be stored.
const DATA_DIR: &str = "data";

// Clamp very small time deltas to avoid astronomical derivatives (seconds)
const MIN_DERIVATIVE_DT: f64 = 0.004;
// Exponential moving average smoothing factor for velocity/acceleration (0..1]
const SMOOTHING_ALPHA: f64 = 0.3;

// Persistent buffer sizing for winit listener
const MAX_BUFFER_SIZE: usize = 50000;
const MIN_PROCESSING_SIZE: usize = MAX_BUFFER_SIZE / 2;

/// Enum representing the various mouse click events that can be detected.
#[derive(Clone)]
enum ClickEvent {
    Left,
    Middle,
    Right,
    None,
}

/// Implements the Display trait for `ClickEvent` for human-readable formatting.
impl fmt::Display for ClickEvent {
    /// Formats the click event into a human-readable string.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let output = match self {
            ClickEvent::Left => "Left",
            ClickEvent::Middle => "Middle",
            ClickEvent::Right => "Right",
            ClickEvent::None => "None",
        };
        write!(f, "{output}")
    }
}

/// Enum representing the scroll events (up or down) that may occur.
#[derive(Clone)]
enum ScrollEvent {
    Up,
    Down,
    None,
}

/// Implements the Display trait for `ScrollEvent` for human-readable formatting.
impl fmt::Display for ScrollEvent {
    /// Formats the scroll event into a human-readable string.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let output = match self {
            ScrollEvent::Up => "Up",
            ScrollEvent::Down => "Down",
            ScrollEvent::None => "None",
        };
        write!(f, "{output}")
    }
}

/// Type alias for a coordinate represented as a tuple (x, y) of f64 values.
type Coordinate = (f64, f64);

/// Structure capturing attributes directly related to mouse movement.
#[derive(Clone)]
struct MouseMovementAttributes {
    current_position: Coordinate,
    movement_delta: Coordinate,
    distance_traveled: f64,
    path_length: f64,
    direction_angles: f64,
    velocity: f64,
    acceleration: f64,
    jerk: f64,
}

/// Structure capturing temporal aspects of mouse events.
#[derive(Clone)]
struct TemporalAttributes {
    timestamp: SystemTime,
    time_between_movements: f64,
    total_duration: f64,
    hover_time: f64,
}

/// Structure capturing behavioral aspects such as click and scroll events.
#[derive(Clone)]
struct BehavioralAttributes {
    click_events: ClickEvent,
    scroll_events: ScrollEvent,
}

/// Structure aggregating raw collected data from mouse events, including temporal, movement, and behavioral attributes.
#[derive(Clone)]
struct CollectedData {
    temporal_attributes: TemporalAttributes,
    mouse_attributes: MouseMovementAttributes,
    behavioral_attributes: BehavioralAttributes,
}

/// Structure holding derived attributes computed from the raw mouse event data.
#[derive(Clone, Default)]
struct DerivedAttributes {
    average_velocity: f64,
    peak_velocity: f64,
    average_acceleration: f64,
    peak_acceleration: f64,
    average_momentum: f64,
    peak_momentum: f64,
    smoothness: f64,
    deviation_from_ideal_path: f64,
    idle_time: f64,
    fitts_index_of_difficulty: f64,
    fitts_movement_time: f64,
}

/// Structure acting as a buffer for storing collected mouse event data.
#[derive(Default)]
struct CollectionBuffer {
    collected_data: Vec<CollectedData>,
}

/// Structure for holding the latest derivation report containing computed mouse metrics.
#[derive(Default)]
struct DerivationReport {
    report: DerivedAttributes,
}

type SyncHandle = Option<JoinHandle<()>>;
type TokioHandle = Option<task::JoinHandle<()>>;

type SyncHandleRef<'a> = Option<&'a JoinHandle<()>>;
type TokioHandleRef<'a> = Option<&'a task::JoinHandle<()>>;

/// The main struct managing mouse event collection, processing, and recording.
/// It holds buffers, configuration, and handles to asynchronous and threaded tasks.
pub struct MouseCollector {
    pub record_filename: Arc<RwLock<String>>,
    collection_buffer: Arc<TokioRwLock<CollectionBuffer>>,
    derivation_report: Arc<TokioRwLock<DerivationReport>>,
    pub start_timestamp: Option<time::SystemTime>,
    pub last_left_click_timestamp: Option<time::SystemTime>,
    pub listening_handle: TokioMutex<TokioHandle>,
    pub reporting_handle: TokioMutex<TokioHandle>,
    pub recording_handle: Mutex<SyncHandle>,
    pub data_dir: Arc<RwLock<String>>,
    // Tunables
    min_derivative_dt: Arc<RwLock<f64>>,
    smoothing_alpha: Arc<RwLock<f64>>,
}

impl MouseCollector {
    /// Creates a new `MouseCollector` instance and, if requested, starts the collection processes.
    ///
    /// # Parameters
    /// - `start`: Indicates whether collection should begin immediately.
    /// - `init_cmd_reporting`: Specifies if command-line reporting should be initialized.
    ///
    /// # Returns
    /// An `Arc`-wrapped instance of `MouseCollector`.
    pub async fn new(start: bool, init_cmd_reporting: bool) -> Arc<Self> {
        // Ensure default data dir exists
        let default_dir = PathBuf::from(DATA_DIR);
        let _ = std::fs::create_dir_all(&default_dir);
        // Generate a unique default filename: mouse_data_{datetime}.csv, with _{n} if exists
        let default_filename = Self::generate_unique_record_filename(&default_dir);
        let record_filename = Arc::new(RwLock::new(default_filename));

        let collection_buffer: Arc<tokio::sync::RwLock<CollectionBuffer>> =
            Arc::new(TokioRwLock::new(CollectionBuffer {
                collected_data: Vec::new(),
            }));

        let mouse_collector = Arc::new(MouseCollector {
            record_filename,
            collection_buffer,
            derivation_report: Default::default(),
            start_timestamp: None,
            last_left_click_timestamp: None,
            listening_handle: TokioMutex::new(None),
            reporting_handle: TokioMutex::new(None),
            recording_handle: Mutex::new(None),
            data_dir: Arc::new(RwLock::new(DATA_DIR.to_string())),
            min_derivative_dt: Arc::new(RwLock::new(MIN_DERIVATIVE_DT)),
            smoothing_alpha: Arc::new(RwLock::new(SMOOTHING_ALPHA)),
        });

        if start {
            let collector_clone: Arc<MouseCollector> = mouse_collector.clone();
            let handlers: (
                TokioHandle,
                SyncHandle,
                TokioHandle,
            ) = Self::start_collecting(collector_clone, init_cmd_reporting).await;

            let mut _listening_handle: TokioHandleRef =
                mouse_collector.listening_handle.lock().await.as_ref();
            let mut _recording_handle: SyncHandleRef =
                mouse_collector.recording_handle.lock().unwrap().as_ref();
            let mut _reporting_handle: TokioHandleRef =
                mouse_collector.reporting_handle.lock().await.as_ref();

            _listening_handle = handlers.0.as_ref();
            _recording_handle = handlers.1.as_ref();
            _reporting_handle = handlers.2.as_ref();
        }

        mouse_collector
    }

    /// Build a unique default record filename in the provided directory.
    fn generate_unique_record_filename(base_dir: &std::path::Path) -> String {
        let now: DateTime<Local> = Local::now();
        let dt = now.format("%Y%m%d_%H%M%S").to_string();
        let mut name = format!("mouse_data_{dt}.csv");
        let mut counter: usize = 1;
        while base_dir.join(&name).exists() {
            name = format!("mouse_data_{dt}_{counter}.csv");
            counter += 1;
        }
        name
    }

    /// Initiates the collection process by starting event listening, recording, and (optionally) command-line reporting.
    ///
    /// # Parameters
    /// - `report`: Determines whether command-line reporting should be started.
    ///
    /// # Returns
    /// A tuple containing optional handles for the listening task, recording thread, and reporting task.
    pub async fn start_collecting(
        self: Arc<Self>,
        report: bool,
    ) -> (
        Option<task::JoinHandle<()>>,
        Option<thread::JoinHandle<()>>,
        Option<task::JoinHandle<()>>,
    ) {
        let listen_clone: Arc<MouseCollector> = self.clone();
        let listening_handle: tokio::task::JoinHandle<()> = tokio::spawn(async move {
            listen_clone.mouse_listener().await;
        });

        let record_clone: Arc<MouseCollector> = self.clone();
        let recording_handle: JoinHandle<()> = thread::spawn(move || {
            record_clone.mouse_recorder();
        });

        let reporting_handle: Option<tokio::task::JoinHandle<()>> = if report {
            let report_clone: Arc<MouseCollector> = self.clone();
            Some(tokio::spawn(async move {
                report_clone.cmd_report().await;
            }))
        } else {
            None
        };

        (
            Some(listening_handle),
            Some(recording_handle),
            reporting_handle,
        )
    }

    /// Stops all collection processes by aborting asynchronous tasks and joining the recording thread.
    pub async fn stop_collecting(&mut self) {
        if let Some(handle) = &self.listening_handle.lock().await.take() {
            handle.abort();
        }
        let _ = self.recording_handle.lock().unwrap().take().unwrap().join();
        if let Some(handle) = &self.reporting_handle.lock().await.take() {
            handle.abort();
        }
    }

    /// Appends new mouse event data to the collection buffer.
    ///
    /// # Parameters
    /// - `data`: A vector of `CollectedData` to be added.
    fn append_new_collection(&self, data: &[CollectedData]) {
        tokio::runtime::Handle::current().block_on(async {
            self.collection_buffer
                .write()
                .await
                .collected_data
                .append(&mut data.to_owned());
        });
    }

    /// Returns a formatted, human-readable snapshot of the latest derived metrics.
    /// Intended for real-time GUI display in the hotbar.
    pub fn get_latest_derived_summary(&self) -> String {
        let report = tokio::runtime::Handle::current()
            .block_on(async { self.derivation_report.read().await.report.clone() });
        format!(
            "Avg vel: {:.2}  |  Peak vel: {:.2}  |  Avg acc: {:.2}  |  Peak acc: {:.2}  |  Smoothness: {:.3}  |  Idle: {:.2}s  |  Fitts ID: {:.2}",
            report.average_velocity,
            report.peak_velocity,
            report.average_acceleration,
            report.peak_acceleration,
            report.smoothness,
            report.idle_time,
            report.fitts_index_of_difficulty,
        )
    }

    /// Returns individual, formatted statistics as separate strings for GUI layout.
    pub fn get_latest_derived_stats(&self) -> Vec<String> {
        let report = tokio::runtime::Handle::current()
            .block_on(async { self.derivation_report.read().await.report.clone() });
        vec![
            format!("Avg vel: {:.2}", report.average_velocity),
            format!("Peak vel: {:.2}", report.peak_velocity),
            format!("Avg acc: {:.2}", report.average_acceleration),
            format!("Peak acc: {:.2}", report.peak_acceleration),
            format!("Avg mom: {:.2}", report.average_momentum),
            format!("Peak mom: {:.2}", report.peak_momentum),
            format!("Smoothness: {:.3}", report.smoothness),
            format!("Dev path: {:.3}", report.deviation_from_ideal_path),
            format!("Idle: {:.2}s", report.idle_time),
            format!("Fitts ID: {:.2}", report.fitts_index_of_difficulty),
            format!("Fitts MT: {:.3}", report.fitts_movement_time),
        ]
    }

    // --- Public helper APIs for GUI integration ---
    /// Pushes a single cursor movement event into the internal buffer.
    pub fn push_move(&self, position: (f64, f64)) {
        let current_timestamp: SystemTime = SystemTime::now();
        let new_data = CollectedData {
            temporal_attributes: TemporalAttributes {
                timestamp: current_timestamp,
                time_between_movements: 0.0,
                total_duration: 0.0,
                hover_time: 0.0,
            },
            mouse_attributes: MouseMovementAttributes {
                current_position: position,
                movement_delta: (0.0, 0.0),
                distance_traveled: 0.0,
                path_length: 0.0,
                direction_angles: 0.0,
                velocity: 0.0,
                acceleration: 0.0,
                jerk: 0.0,
            },
            behavioral_attributes: BehavioralAttributes {
                click_events: ClickEvent::None,
                scroll_events: ScrollEvent::None,
            },
        };
        self.append_new_collection(&[new_data]);
    }

    /// Pushes a single left-click event into the internal buffer.
    pub fn push_left_click(&self, position: (f64, f64)) {
        let current_timestamp: SystemTime = SystemTime::now();
        let new_data = CollectedData {
            temporal_attributes: TemporalAttributes {
                timestamp: current_timestamp,
                time_between_movements: 0.0,
                total_duration: 0.0,
                hover_time: 0.0,
            },
            mouse_attributes: MouseMovementAttributes {
                current_position: position,
                movement_delta: (0.0, 0.0),
                distance_traveled: 0.0,
                path_length: 0.0,
                direction_angles: 0.0,
                velocity: 0.0,
                acceleration: 0.0,
                jerk: 0.0,
            },
            behavioral_attributes: BehavioralAttributes {
                click_events: ClickEvent::Left,
                scroll_events: ScrollEvent::None,
            },
        };
        self.append_new_collection(&[new_data]);
    }

    /// Starts only the recording thread, without spawning a winit listener window.
    pub fn start_recording_thread(self: &Arc<Self>) -> JoinHandle<()> {
        let record_clone: Arc<MouseCollector> = self.clone();
        let handle = tokio::runtime::Handle::current();
        thread::spawn(move || {
            let _enter = handle.enter();
            record_clone.mouse_recorder();
        })
    }

    /// Public setter for the data directory used to store CSV output.
    pub fn set_data_dir(&self, dir: String) {
        if dir.trim().is_empty() {
            return;
        }
        if let Ok(mut w) = self.data_dir.write() {
            *w = dir;
        }
    }

    /// Public getter for the current data directory path.
    pub fn get_data_dir(&self) -> String {
        self.data_dir
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| DATA_DIR.to_string())
    }

    // Tunable getters/setters
    pub fn set_min_derivative_dt(&self, dt: f64) {
        if dt > 0.0 {
            if let Ok(mut w) = self.min_derivative_dt.write() {
                *w = dt;
            }
        }
    }
    pub fn get_min_derivative_dt(&self) -> f64 {
        self.min_derivative_dt
            .read()
            .map(|g| *g)
            .unwrap_or(MIN_DERIVATIVE_DT)
    }
    pub fn set_smoothing_alpha(&self, alpha: f64) {
        let a = if alpha <= 0.0 {
            0.0001
        } else if alpha > 1.0 {
            1.0
        } else {
            alpha
        };
        if let Ok(mut w) = self.smoothing_alpha.write() {
            *w = a;
        }
    }
    pub fn get_smoothing_alpha(&self) -> f64 {
        self.smoothing_alpha
            .read()
            .map(|g| *g)
            .unwrap_or(SMOOTHING_ALPHA)
    }
}

/// Trait defining the interface for recording mouse events and processing the collected data.
trait MouseRecorder {
    /// Continuously records mouse events and processes the data.
    fn mouse_recorder(&self);
    /// Calculates movement attributes (e.g., velocity, acceleration) from raw collected data.
    ///
    /// # Parameters
    /// - `data`: A slice of raw `CollectedData`.
    ///
    /// # Returns
    /// A vector of `CollectedData` with updated movement metrics.
    fn calculate_collection_attributes(&self, data: &[CollectedData]) -> Vec<CollectedData>;
    /// Computes derived attributes such as averages, peaks, and smoothness metrics.
    ///
    /// # Parameters
    /// - `data`: A slice of processed `CollectedData`.
    ///
    /// # Returns
    /// A vector of `DerivedAttributes` corresponding to each processed event.
    fn calculate_derived_attributes(&self, data: &[CollectedData]) -> Vec<DerivedAttributes>;
    /// Writes the collected and computed mouse event data to a CSV file.
    ///
    /// # Parameters
    /// - `collection`: Processed mouse event data.
    /// - `derivations`: Derived metrics corresponding to the mouse data.
    fn write_data_to_csv(
        &self,
        collection: Vec<CollectedData>,
        derivations: Vec<DerivedAttributes>,
    );
}

impl MouseRecorder for MouseCollector {
    /// Implements continuous mouse data recording. When a threshold is met, processes the data,
    /// writes results to CSV, and clears processed events from the buffer.
    fn mouse_recorder(&self) {
        loop {
            // Drain a batch ending with a Left click from the shared buffer
            let batch: Option<Vec<CollectedData>> =
                tokio::runtime::Handle::current().block_on(async {
                    let mut buffer_write = self.collection_buffer.write().await;
                    if buffer_write.collected_data.is_empty() {
                        return None;
                    }
                    // Find first Left click to delimit the batch
                    let maybe_left_idx = buffer_write.collected_data.iter().position(|event| {
                        matches!(event.behavioral_attributes.click_events, ClickEvent::Left)
                    });
                    if let Some(left_idx) = maybe_left_idx {
                        // Drain inclusive range [0..=left_idx] as one batch
                        let drained: Vec<CollectedData> =
                            buffer_write.collected_data.drain(0..=left_idx).collect();
                        Some(drained)
                    } else {
                        None
                    }
                });

            if let Some(batch_events) = batch {
                if batch_events.len() > 1 {
                    let collection_data_vec =
                        Self::calculate_collection_attributes(self, &batch_events);
                    let derived_data =
                        Self::calculate_derived_attributes(self, &collection_data_vec);
                    Self::write_data_to_csv(self, collection_data_vec, derived_data);
                } else {
                    // If a batch had only a single event, skip writing as no movement can be derived
                }
            } else {
                // No complete batch yet; avoid busy loop
                thread::sleep(time::Duration::from_millis(5));
            }
        }
    }

    /// Processes raw mouse data to calculate movement metrics such as delta, distance, velocity,
    /// acceleration, jerk, and hover time.
    ///
    /// # Parameters
    /// - `data`: A slice of raw `CollectedData`.
    ///
    /// # Returns
    /// A vector of updated `CollectedData` with computed movement attributes.
    fn calculate_collection_attributes(&self, data: &[CollectedData]) -> Vec<CollectedData> {
        let length: i32 = data.iter().len() as i32;
        if length <= 1 {
            return vec![];
        }

        let mut updated_data: Vec<CollectedData> = Vec::with_capacity(length as usize);
        let mut cumulative_path_length: f64 = 0.0;
        // Smoothed state
        let mut prev_velocity_smoothed: f64 = 0.0;
        let mut prev_acceleration_smoothed: f64 = 0.0;
        // Read current tunables
        let min_dt = self.get_min_derivative_dt();
        let alpha = self.get_smoothing_alpha();

        for (i, event) in data.iter().enumerate() {
            let mut new_event: CollectedData = event.clone();
            let current_timestamp: SystemTime = event.temporal_attributes.timestamp;

            let first_timestamp: SystemTime = data[0].temporal_attributes.timestamp;
            let total_duration: f64 = current_timestamp
                .duration_since(first_timestamp)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            new_event.temporal_attributes.total_duration = total_duration;

            if i == 0 {
                new_event.mouse_attributes.path_length = cumulative_path_length;
            } else {
                let previous_event = &updated_data[i - 1];
                let raw_dt = current_timestamp
                    .duration_since(previous_event.temporal_attributes.timestamp)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);
                // store raw dt for logging
                new_event.temporal_attributes.time_between_movements = raw_dt;
                // use effective dt for derivatives
                let effective_dt = if raw_dt > min_dt { raw_dt } else { min_dt };

                let previous_coord: Coordinate = previous_event.mouse_attributes.current_position;
                let current_coord: Coordinate = new_event.mouse_attributes.current_position;
                let delta_coord: Coordinate = (
                    current_coord.0 - previous_coord.0,
                    current_coord.1 - previous_coord.1,
                );
                new_event.mouse_attributes.movement_delta = delta_coord;

                let distance_traveled: f64 = (delta_coord.0.powi(2) + delta_coord.1.powi(2)).sqrt();
                new_event.mouse_attributes.distance_traveled = distance_traveled;

                cumulative_path_length += distance_traveled;
                new_event.mouse_attributes.path_length = cumulative_path_length;

                let velocity_raw: f64 = if effective_dt > 0.0 {
                    distance_traveled / effective_dt
                } else {
                    0.0
                };
                // EMA smoothing for velocity
                let velocity_smoothed = if i == 1 {
                    velocity_raw
                } else {
                    alpha * velocity_raw + (1.0 - alpha) * prev_velocity_smoothed
                };
                new_event.mouse_attributes.velocity = velocity_smoothed;

                let acceleration_raw: f64 = if i > 1 && effective_dt > 0.0 {
                    (velocity_smoothed - prev_velocity_smoothed) / effective_dt
                } else {
                    0.0
                };
                // EMA smoothing for acceleration
                let acceleration_smoothed = if i <= 1 {
                    acceleration_raw
                } else {
                    alpha * acceleration_raw + (1.0 - alpha) * prev_acceleration_smoothed
                };
                new_event.mouse_attributes.acceleration = acceleration_smoothed;

                let jerk: f64 = if i > 1 && effective_dt > 0.0 {
                    (acceleration_smoothed - prev_acceleration_smoothed) / effective_dt
                } else {
                    0.0
                };
                new_event.mouse_attributes.jerk = jerk;

                new_event.mouse_attributes.direction_angles = delta_coord.1.atan2(delta_coord.0);

                let hover_threshold: f64 = 1.0;
                if distance_traveled < hover_threshold {
                    new_event.temporal_attributes.hover_time =
                        previous_event.temporal_attributes.hover_time + raw_dt;
                } else {
                    new_event.temporal_attributes.hover_time = 0.0;
                }

                prev_velocity_smoothed = velocity_smoothed;
                prev_acceleration_smoothed = acceleration_smoothed;
            }

            updated_data.push(new_event);
        }

        updated_data
    }

    /// Computes derived attributes including average and peak velocity, acceleration,
    /// smoothness, and Fitts's metrics based on the processed mouse event data.
    ///
    /// # Parameters
    /// - `data`: A slice of processed `CollectedData`.
    ///
    /// # Returns
    /// A vector of `DerivedAttributes` for each event.
    fn calculate_derived_attributes(&self, data: &[CollectedData]) -> Vec<DerivedAttributes> {
        let length = data.len();
        let mut derived_results: Vec<DerivedAttributes> = Vec::with_capacity(length);

        if length == 0 {
            return derived_results;
        }

        let first_position: Coordinate = data[0].mouse_attributes.current_position;

        let mut cumulative_velocity: f64 = 0.0;
        let mut max_velocity: f64 = 0.0;
        let mut cumulative_acceleration: f64 = 0.0;
        let mut max_acceleration: f64 = 0.0;
        let mut cumulative_jerk: f64 = 0.0;
        let mut jerk_count: i32 = 0;
        let mut cumulative_active_time: f64 = 0.0;
        // momentum (unit mass -> equals speed)
        let mut cumulative_momentum: f64 = 0.0;
        let mut max_momentum: f64 = 0.0;

        for (i, data_element) in data.iter().enumerate().take(length) {
            let current_total_duration: f64 = data[i].temporal_attributes.total_duration;
            let mut derived: DerivedAttributes = DerivedAttributes::default();

            if i >= 1 {
                let velocity: f64 = data_element.mouse_attributes.velocity;
                cumulative_velocity += velocity;
                if velocity > max_velocity {
                    max_velocity = velocity;
                }
                derived.average_velocity = cumulative_velocity / (i as f64);
                derived.peak_velocity = max_velocity;

                // momentum as unit-mass momentum (speed)
                let momentum = velocity.abs();
                cumulative_momentum += momentum;
                if momentum > max_momentum {
                    max_momentum = momentum;
                }
                derived.average_momentum = cumulative_momentum / (i as f64);
                derived.peak_momentum = max_momentum;

                cumulative_active_time += data_element.temporal_attributes.time_between_movements;
                derived.idle_time = current_total_duration - cumulative_active_time;
            }

            if i >= 2 {
                let acceleration: f64 = data_element.mouse_attributes.acceleration;
                cumulative_acceleration += acceleration;
                if acceleration > max_acceleration {
                    max_acceleration = acceleration;
                }
                derived.average_acceleration = cumulative_acceleration / ((i - 1) as f64);
                derived.peak_acceleration = max_acceleration;
            }

            if i >= 3 {
                cumulative_jerk += data_element.mouse_attributes.jerk.abs();
                jerk_count += 1;
                let avg_jerk = if jerk_count > 0 {
                    cumulative_jerk / (jerk_count as f64)
                } else {
                    0.0
                };
                derived.smoothness = if avg_jerk > 0.0 { 1.0 / avg_jerk } else { 0.0 };
            }

            let current_position: Coordinate = data_element.mouse_attributes.current_position;
            let ideal_dx: f64 = current_position.0 - first_position.0;
            let ideal_dy: f64 = current_position.1 - first_position.1;
            let ideal_path_length: f64 = (ideal_dx.powi(2) + ideal_dy.powi(2)).sqrt();

            let current_path_length: f64 = data_element.mouse_attributes.path_length;
            derived.deviation_from_ideal_path = if ideal_path_length > 0.0 {
                current_path_length / ideal_path_length
            } else {
                0.0
            };

            derived.fitts_index_of_difficulty = ((ideal_path_length / 100.0) + 1.0).log2();
            derived.fitts_movement_time = current_total_duration;

            derived_results.push(derived);
        }

        tokio::runtime::Handle::current().block_on(async {
            let mut report_write = self.derivation_report.write().await;
            report_write.report = derived_results[length - 1].clone();
        });

        derived_results
    }

    /// Writes the processed mouse event data and its derived attributes to a CSV file.
    ///
    /// # Parameters
    /// - `collection`: The processed mouse event data.
    /// - `derivations`: The corresponding derived metrics.
    fn write_data_to_csv(
        &self,
        collection: Vec<CollectedData>,
        derivations: Vec<DerivedAttributes>,
    ) {
        fn system_time_to_string(system_time: SystemTime) -> String {
            use std::time::UNIX_EPOCH;
            let secs = match system_time.duration_since(UNIX_EPOCH) {
                Ok(d) => d.as_secs_f64(),
                Err(_) => 0.0,
            };
            format!("{secs:.4}")
        }

        let dir_string = self.get_data_dir();
        let data_dir = PathBuf::from(&dir_string);
        if let Err(e) = std::fs::create_dir_all(&data_dir) {
            eprintln!("Failed to create data dir '{}': {e}", data_dir.display());
            return;
        }
        // Use the configured record filename
        let filename = self
            .record_filename
            .read()
            .map(|s| s.clone())
            .unwrap_or_else(|_| "mouse_data.csv".to_string());
        let file_path = data_dir.join(&filename);
        let file_result = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path);
        let file = match file_result {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to open data file '{}': {e}", file_path.display());
                return;
            }
        };
        let mut writer: csv::Writer<File> = csv::Writer::from_writer(file);

        // Write header only if file is empty
        let need_header = std::fs::metadata(&file_path)
            .map(|m| m.len() == 0)
            .unwrap_or(true);
        if need_header && let Err(e) = writer.write_record([
                "timestamp",
                "total_duration",
                "time_between_movements",
                "hover_time",
                "position_x",
                "position_y",
                "velocity",
                "acceleration",
                "jerk",
                "path_length",
                "click_events",
                "scroll_events",
                "average_velocity",
                "peak_velocity",
                "average_acceleration",
                "peak_acceleration",
                "average_momentum",
                "peak_momentum",
                "smoothness",
                "deviation_from_ideal_path",
                "idle_time",
                "fitts_index_of_difficulty",
                "fitts_movement_time",
            ]) {
                eprintln!(
                    "Failed to write CSV header to '{}': {e}",
                    file_path.display()
                );
                return;
            }
        

        for (data, derived) in collection.iter().zip(derivations.iter()) {
            if let Err(e) = writer.write_record(&[
                system_time_to_string(data.temporal_attributes.timestamp),
                data.temporal_attributes.total_duration.to_string(),
                data.temporal_attributes.time_between_movements.to_string(),
                data.temporal_attributes.hover_time.to_string(),
                data.mouse_attributes.current_position.0.to_string(),
                data.mouse_attributes.current_position.1.to_string(),
                data.mouse_attributes.velocity.to_string(),
                data.mouse_attributes.acceleration.to_string(),
                data.mouse_attributes.jerk.to_string(),
                data.mouse_attributes.path_length.to_string(),
                data.behavioral_attributes.click_events.to_string(),
                data.behavioral_attributes.scroll_events.to_string(),
                derived.average_velocity.to_string(),
                derived.peak_velocity.to_string(),
                derived.average_acceleration.to_string(),
                derived.peak_acceleration.to_string(),
                derived.average_momentum.to_string(),
                derived.peak_momentum.to_string(),
                derived.smoothness.to_string(),
                derived.deviation_from_ideal_path.to_string(),
                derived.idle_time.to_string(),
                derived.fitts_index_of_difficulty.to_string(),
                derived.fitts_movement_time.to_string(),
            ]) {
                eprintln!("Failed to write CSV row to '{}': {e}", file_path.display());
                break;
            }
        }

        if let Err(e) = writer.flush() {
            eprintln!("Failed to flush CSV writer '{}': {e}", file_path.display());
        }
    }
}

/// Trait for outputting derived mouse metrics to the command line.
trait CommandlineOutput {
    /// Outputs a report of the derived attributes to the command line.
    async fn cmd_report(&self);
}

impl CommandlineOutput for MouseCollector {
    /// Prints computed mouse metrics (velocity, acceleration, smoothness, etc.) to standard output.
    async fn cmd_report(&self) {
        let derived_attributes = &self.derivation_report.read().await;
        let derived_report = &derived_attributes.report;

        println!("Average velocity: {}", derived_report.average_velocity);
        println!("Peak velocity: {}", derived_report.peak_velocity);
        println!(
            "Average acceleration: {}",
            derived_report.average_acceleration
        );
        println!("Peak acceleration: {}", derived_report.peak_acceleration);
        println!("Average momentum: {}", derived_report.average_momentum);
        println!("Peak momentum: {}", derived_report.peak_momentum);
        println!("Smoothness: {}", derived_report.smoothness);
        println!(
            "Deviation from ideal path: {}",
            derived_report.deviation_from_ideal_path
        );
        println!("Idle time: {}", derived_report.idle_time);
        println!(
            "Fitts's index of difficulty: {}",
            derived_report.fitts_index_of_difficulty
        );
        println!(
            "Fitts's movement time: {}",
            derived_report.fitts_movement_time
        );
    }
}

/// Trait for asynchronously listening to mouse events.
trait MouseListener {
    /// Asynchronously listens for mouse events and routes them to the collector.
    async fn mouse_listener(self: Arc<Self>);
}

impl MouseListener for MouseCollector {
    /// Spawns a blocking task to run the mouse listener using the `winit` event loop.
    async fn mouse_listener(self: Arc<Self>) {
        let collector = self.clone();

        task::spawn_blocking(move || {
            let event_loop = EventLoop::new().expect("Failed to create event loop");
            let mut app = MouseApp {
                collector: &collector,
                window: None,
                buffer: Vec::with_capacity(MAX_BUFFER_SIZE),
                last_position: None,
            };
            event_loop.run_app(&mut app).expect("Failed to run app");
        })
        .await
        .expect("The mouse listener task panicked");
    }
}

/// Structure representing the application that handles mouse events.
/// Contains a reference to the main mouse collector and an optional window.
struct MouseApp<'a> {
    collector: &'a MouseCollector,
    window: Option<Window>,
    buffer: Vec<CollectedData>,
    last_position: Option<Coordinate>,
}

/// Implements the `ApplicationHandler` trait for `MouseApp` to handle window-related events.
impl ApplicationHandler<()> for MouseApp<'_> {
    /// Called when the application is resumed; creates a window for event capture.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.window = Some(
            event_loop
                .create_window(Window::default_attributes())
                .expect("Failed to create window"),
        );
    }

    /// Handles window events (cursor movements, mouse clicks, mouse wheel) and buffers the collected data.
    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let current_timestamp: SystemTime = SystemTime::now();
                let coord: Coordinate = (position.x, position.y);
                let previous_coord: Coordinate = self.last_position.unwrap_or(coord);
                let movement_delta: Coordinate =
                    (coord.0 - previous_coord.0, coord.1 - previous_coord.1);
                self.last_position = Some(coord);

                let new_data = CollectedData {
                    temporal_attributes: TemporalAttributes {
                        timestamp: current_timestamp,
                        time_between_movements: 0.0,
                        total_duration: 0.0,
                        hover_time: 0.0,
                    },
                    mouse_attributes: MouseMovementAttributes {
                        current_position: coord,
                        movement_delta,
                        distance_traveled: 0.0,
                        path_length: 0.0,
                        direction_angles: 0.0,
                        velocity: 0.0,
                        acceleration: 0.0,
                        jerk: 0.0,
                    },
                    behavioral_attributes: BehavioralAttributes {
                        click_events: ClickEvent::None,
                        scroll_events: ScrollEvent::None,
                    },
                };
                self.buffer.push(new_data);
                if self.buffer.len() >= MIN_PROCESSING_SIZE {
                    self.collector.append_new_collection(&self.buffer);
                    self.buffer.clear();
                }
            }
            WindowEvent::MouseInput {
                state: _state,
                button,
                ..
            } => {
                let current_timestamp: SystemTime = SystemTime::now();
                let click_event: ClickEvent = match button {
                    MouseButton::Left => ClickEvent::Left,
                    MouseButton::Right => ClickEvent::Right,
                    MouseButton::Middle => ClickEvent::Middle,
                    _ => ClickEvent::None,
                };
                let current_coord: Coordinate = self.last_position.unwrap_or((0.0, 0.0));

                let new_data = CollectedData {
                    temporal_attributes: TemporalAttributes {
                        timestamp: current_timestamp,
                        time_between_movements: 0.0,
                        total_duration: 0.0,
                        hover_time: 0.0,
                    },
                    mouse_attributes: MouseMovementAttributes {
                        current_position: current_coord,
                        movement_delta: (0.0, 0.0),
                        distance_traveled: 0.0,
                        path_length: 0.0,
                        direction_angles: 0.0,
                        velocity: 0.0,
                        acceleration: 0.0,
                        jerk: 0.0,
                    },
                    behavioral_attributes: BehavioralAttributes {
                        click_events: click_event,
                        scroll_events: ScrollEvent::None,
                    },
                };
                self.buffer.push(new_data);
                if self.buffer.len() >= MIN_PROCESSING_SIZE {
                    self.collector.append_new_collection(&self.buffer);
                    self.buffer.clear();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let current_timestamp: SystemTime = SystemTime::now();
                let scroll_event: ScrollEvent = match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        if y > 0.0 {
                            ScrollEvent::Up
                        } else {
                            ScrollEvent::Down
                        }
                    }
                    _ => ScrollEvent::None,
                };
                let current_coord: Coordinate = self.last_position.unwrap_or((0.0, 0.0));
                let new_data = CollectedData {
                    temporal_attributes: TemporalAttributes {
                        timestamp: current_timestamp,
                        time_between_movements: 0.0,
                        total_duration: 0.0,
                        hover_time: 0.0,
                    },
                    mouse_attributes: MouseMovementAttributes {
                        current_position: current_coord,
                        movement_delta: (0.0, 0.0),
                        distance_traveled: 0.0,
                        path_length: 0.0,
                        direction_angles: 0.0,
                        velocity: 0.0,
                        acceleration: 0.0,
                        jerk: 0.0,
                    },
                    behavioral_attributes: BehavioralAttributes {
                        click_events: ClickEvent::None,
                        scroll_events: scroll_event,
                    },
                };
                self.buffer.push(new_data);
                if self.buffer.len() >= MIN_PROCESSING_SIZE {
                    self.collector.append_new_collection(&self.buffer);
                    self.buffer.clear();
                }
            }
            _ => {}
        }
    }
}
