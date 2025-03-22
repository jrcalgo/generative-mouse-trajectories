//!
//!
//!

use chrono::{DateTime, Local};
use std::fmt;
use std::fs::File;
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

const DATA_DIR: &str = "data";

#[derive(Clone)]
enum ClickEvent {
    Left,
    Middle,
    Right,
    None,
}

impl fmt::Display for ClickEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let output = match self {
            ClickEvent::Left => "Left",
            ClickEvent::Middle => "Middle",
            ClickEvent::Right => "Right",
            ClickEvent::None => "None",
        };
        write!(f, "{}", output)
    }
}

#[derive(Clone)]
enum ScrollEvent {
    Up,
    Down,
    None,
}

impl fmt::Display for ScrollEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let output = match self {
            ScrollEvent::Up => "Up",
            ScrollEvent::Down => "Down",
            ScrollEvent::None => "None",
        };
        write!(f, "{}", output)
    }
}

type Coordinate = (f64, f64);

#[derive(Clone)]
struct MouseMovementAttributes {
    mouse_dpi: usize,
    current_position: Coordinate,
    movement_delta: Coordinate,
    distance_traveled: f64,
    path_length: f64,
    direction_angles: f64,
    velocity: f64,
    acceleration: f64,
    curvature: f64,
    jerk: f64,
}

#[derive(Clone)]
struct TemporalAttributes {
    timestamp: SystemTime,
    time_between_movements: f64,
    total_duration: f64,
    hover_time: f64,
}

#[derive(Clone)]
struct BehavioralAttributes {
    click_events: ClickEvent,
    scroll_events: ScrollEvent,
}

#[derive(Clone)]
struct CollectedData {
    temporal_attributes: TemporalAttributes,
    mouse_attributes: MouseMovementAttributes,
    behavioral_attributes: BehavioralAttributes,
}

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

#[derive(Default)]
struct CollectionBuffer {
    collected_data: Vec<CollectedData>,
}

#[derive(Default)]
struct DerivationReport {
    report: DerivedAttributes,
}

pub struct MouseCollector {
    pub record_filename: Arc<RwLock<String>>,
    collection_buffer: Arc<TokioRwLock<CollectionBuffer>>,
    derivation_report: Arc<TokioRwLock<DerivationReport>>,
    pub start_timestamp: Option<time::SystemTime>,
    pub last_left_click_timestamp: Option<time::SystemTime>,
    pub listening_handle: TokioMutex<Option<task::JoinHandle<()>>>,
    pub reporting_handle: TokioMutex<Option<task::JoinHandle<()>>>,
    pub recording_handle: Mutex<Option<JoinHandle<()>>>,
}

impl MouseCollector {
    pub async fn new(start: bool, init_cmd_reporting: bool) -> Arc<Self> {
        let record_filename = Arc::new(RwLock::new("data.csv".to_string()));

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
        });

        if start {
            let collector_clone: Arc<MouseCollector> = mouse_collector.clone();
            let handlers: (
                Option<task::JoinHandle<()>>,
                Option<JoinHandle<()>>,
                Option<task::JoinHandle<()>>,
            ) = Self::start_collecting(collector_clone, init_cmd_reporting).await;

            let mut listening_handle: Option<&tokio::task::JoinHandle<()>> =
                mouse_collector.listening_handle.lock().await.as_ref();
            let mut recording_handle: Option<&JoinHandle<()>> =
                mouse_collector.recording_handle.lock().unwrap().as_ref();
            let mut reporting_handle: Option<&tokio::task::JoinHandle<()>> =
                mouse_collector.reporting_handle.lock().await.as_ref();

            listening_handle = handlers.0.as_ref();
            recording_handle = handlers.1.as_ref();
            reporting_handle = handlers.2.as_ref();
        }

        mouse_collector
    }

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

    pub async fn stop_collecting(&mut self) {
        if let Some(handle) = &self.listening_handle.lock().await.take() {
            handle.abort();
        }
        let _ = self.recording_handle.lock().unwrap().take().unwrap().join();
        if let Some(handle) = &self.reporting_handle.lock().await.take() {
            handle.abort();
        }
    }

    fn append_new_collection(&self, data: &Vec<CollectedData>) {
        tokio::runtime::Handle::current().block_on(async {
            self.collection_buffer
                .write()
                .await
                .collected_data
                .append(&mut data.clone());
        });
    }

    fn get_current_collection(&self) -> Option<Vec<CollectedData>> {
        Some(
            tokio::runtime::Handle::current()
                .block_on(async { self.collection_buffer.read().await.collected_data.clone() }),
        )
    }
}

trait MouseRecorder {
    fn mouse_recorder(&self);
    fn calculate_collection_attributes(&self, data: &[CollectedData]) -> Vec<CollectedData>;
    fn calculate_derived_attributes(&self, data: &[CollectedData]) -> Vec<DerivedAttributes>;
    fn write_data_to_csv(
        &self,
        collection: Vec<CollectedData>,
        derivations: Vec<DerivedAttributes>,
    );
}

impl MouseRecorder for MouseCollector {
    fn mouse_recorder(&self) {
        loop {
            if let Some(data) = self.get_current_collection() {
                let current_timestamp: SystemTime = SystemTime::now();
                if data.len() > 1000 {
                    // calculate data values
                    let collection_data_vec = Self::calculate_collection_attributes(self, &data);
                    let derived_data =
                        Self::calculate_derived_attributes(self, &collection_data_vec);
                    // write data to csv
                    Self::write_data_to_csv(self, collection_data_vec, derived_data);
                    // clear all data older than current_timestamp
                    tokio::runtime::Handle::current().block_on(async {
                        self.collection_buffer
                            .write()
                            .await
                            .collected_data
                            .retain(|event| {
                                event.temporal_attributes.timestamp >= current_timestamp
                            });
                    });
                }
            } else {
                thread::sleep(time::Duration::from_secs(1));
            }
        }
    }

    fn calculate_collection_attributes(&self, data: &[CollectedData]) -> Vec<CollectedData> {
        let length: i32 = data.iter().len() as i32;
        if length <= 1 {
            return vec![];
        }

        let mut updated_data: Vec<CollectedData> = Vec::with_capacity(length as usize);
        let mut cumulative_path_length: f64 = 0.0;
        let mut previous_velocity: f64 = 0.0;
        let mut previous_acceleration: f64 = 0.0;

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
                let time_delta = current_timestamp
                    .duration_since(previous_event.temporal_attributes.timestamp)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);
                new_event.temporal_attributes.time_between_movements = time_delta;

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

                let velocity: f64 = if time_delta > 0.0 {
                    distance_traveled / time_delta
                } else {
                    0.0
                };
                new_event.mouse_attributes.velocity = velocity;

                let acceleration: f64 = if i > 1 && time_delta > 0.0 {
                    (velocity - previous_velocity) / time_delta
                } else {
                    0.0
                };
                new_event.mouse_attributes.acceleration = acceleration;

                let jerk: f64 = if i > 1 && time_delta > 0.0 {
                    (acceleration - previous_acceleration) / time_delta
                } else {
                    0.0
                };
                new_event.mouse_attributes.jerk = jerk;

                new_event.mouse_attributes.direction_angles = delta_coord.1.atan2(delta_coord.0);

                let hover_threshold: f64 = 1.0;
                if distance_traveled < hover_threshold {
                    new_event.temporal_attributes.hover_time =
                        previous_event.temporal_attributes.hover_time + time_delta;
                } else {
                    new_event.temporal_attributes.hover_time = 0.0;
                }

                previous_velocity = velocity;
                previous_acceleration = acceleration;
            }

            updated_data.push(new_event);
        }

        updated_data
    }

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

        for i in 0..length {
            let current_total_duration: f64 = data[i].temporal_attributes.total_duration;
            let mut derived: DerivedAttributes = DerivedAttributes::default();

            if i >= 1 {
                let velocity: f64 = data[i].mouse_attributes.velocity;
                cumulative_velocity += velocity;
                if velocity > max_velocity {
                    max_velocity = velocity;
                }
                derived.average_velocity = cumulative_velocity / (i as f64);
                derived.peak_velocity = max_velocity;

                cumulative_active_time += data[i].temporal_attributes.time_between_movements;
                derived.idle_time = current_total_duration - cumulative_active_time;
            }

            if i >= 2 {
                let acceleration: f64 = data[i].mouse_attributes.acceleration;
                cumulative_acceleration += acceleration;
                if acceleration > max_acceleration {
                    max_acceleration = acceleration;
                }
                derived.average_acceleration = cumulative_acceleration / ((i - 1) as f64);
                derived.peak_acceleration = max_acceleration;
            }

            if i >= 3 {
                cumulative_jerk += data[i].mouse_attributes.jerk.abs();
                jerk_count += 1;
                let avg_jerk = if jerk_count > 0 {
                    cumulative_jerk / (jerk_count as f64)
                } else {
                    0.0
                };
                derived.smoothness = if avg_jerk > 0.0 { 1.0 / avg_jerk } else { 0.0 };
            }

            let current_position: Coordinate = data[i].mouse_attributes.current_position;
            let ideal_dx: f64 = current_position.0 - first_position.0;
            let ideal_dy: f64 = current_position.1 - first_position.1;
            let ideal_path_length: f64 = (ideal_dx.powi(2) + ideal_dy.powi(2)).sqrt();

            let current_path_length: f64 = data[i].mouse_attributes.path_length;
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

    fn write_data_to_csv(
        &self,
        collection: Vec<CollectedData>,
        derivations: Vec<DerivedAttributes>,
    ) {
        fn system_time_to_string(system_time: SystemTime) -> String {
            let datetime: DateTime<Local> = system_time.into();
            datetime.format("%H:%M:%S%.4f").to_string()
        }

        let mut writer: csv::Writer<File> = csv::Writer::from_writer(
            File::create(PathBuf::from(DATA_DIR).join("data.csv")).unwrap(),
        );
        // append header
        writer
            .write_record([
                "timestamp",
                "total_duration",
                "time_between_movements",
                "hover_time",
                "position_x",
                "position_y",
                "velocity",
                "acceleration",
                "momentum",
                "path_length",
                "jerk",
                "click_events",
                "scroll_events",
                "average_velocity",
                "peak_velocity",
                "average_acceleration",
                "peak_acceleration",
                "average_momentum",
                "peak_momentum",
                "smoothness",
            ])
            .unwrap();

        // write all data arguments
        for (data, derived) in collection.iter().zip(derivations.iter()) {
            writer
                .write_record(&[
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
                ])
                .unwrap();
        }
    }
}

trait CommandlineOutput {
    async fn cmd_report(&self);
}

impl CommandlineOutput for MouseCollector {
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

trait MouseListener {
    async fn mouse_listener(self: Arc<Self>);
}

impl MouseListener for MouseCollector {
    async fn mouse_listener(self: Arc<Self>) {
        let collector = self.clone();

        task::spawn_blocking(move || {
            let event_loop = EventLoop::new().expect("Failed to create event loop");
            let mut app = MouseApp {
                collector: &collector,
                window: None,
            };
            event_loop.run_app(&mut app).expect("Failed to run app");
        })
        .await
        .expect("The mouse listener task panicked");
    }
}

struct MouseApp<'a> {
    collector: &'a MouseCollector,
    window: Option<Window>,
}

impl ApplicationHandler<()> for MouseApp<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.window = Some(
            event_loop
                .create_window(Window::default_attributes())
                .expect("Failed to create window"),
        );
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        const MAX_BUFFER_SIZE: usize = 50000;
        const MIN_PROCESSING_SIZE: usize = MAX_BUFFER_SIZE / 2;
        let mut buffer: Vec<CollectedData> = Vec::with_capacity(MAX_BUFFER_SIZE);
        let mut last_position: Option<Coordinate> = None;
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let current_timestamp = SystemTime::now();
                let coord: Coordinate = (position.x, position.y);
                last_position = Some(coord);
                let movement_delta: Coordinate = (
                    last_position.unwrap().0 - coord.0,
                    last_position.unwrap().1 - coord.1,
                );

                let new_data = CollectedData {
                    temporal_attributes: TemporalAttributes {
                        timestamp: current_timestamp,
                        time_between_movements: 0.0,
                        total_duration: 0.0,
                        hover_time: 0.0,
                    },
                    mouse_attributes: MouseMovementAttributes {
                        mouse_dpi: 0,
                        current_position: coord,
                        movement_delta,
                        distance_traveled: 0.0,
                        path_length: 0.0,
                        direction_angles: 0.0,
                        velocity: 0.0,
                        acceleration: 0.0,
                        curvature: 0.0,
                        jerk: 0.0,
                    },
                    behavioral_attributes: BehavioralAttributes {
                        click_events: ClickEvent::None,
                        scroll_events: ScrollEvent::None,
                    },
                };
                buffer.push(new_data);
                if buffer.len() >= MIN_PROCESSING_SIZE {
                    self.collector.append_new_collection(&buffer);
                    buffer.clear();
                }
            }
            WindowEvent::MouseInput {
                state: _state,
                button,
                ..
            } => {
                let current_timestamp = SystemTime::now();
                let click_event = match button {
                    MouseButton::Left => ClickEvent::Left,
                    MouseButton::Right => ClickEvent::Right,
                    MouseButton::Middle => ClickEvent::Middle,
                    _ => ClickEvent::None,
                };

                let new_data = CollectedData {
                    temporal_attributes: TemporalAttributes {
                        timestamp: current_timestamp,
                        time_between_movements: 0.0,
                        total_duration: 0.0,
                        hover_time: 0.0,
                    },
                    mouse_attributes: MouseMovementAttributes {
                        mouse_dpi: 0,
                        current_position: last_position.unwrap(),
                        movement_delta: (0.0, 0.0),
                        distance_traveled: 0.0,
                        path_length: 0.0,
                        direction_angles: 0.0,
                        velocity: 0.0,
                        acceleration: 0.0,
                        curvature: 0.0,
                        jerk: 0.0,
                    },
                    behavioral_attributes: BehavioralAttributes {
                        click_events: click_event,
                        scroll_events: ScrollEvent::None,
                    },
                };
                buffer.push(new_data);
                if buffer.len() >= MIN_PROCESSING_SIZE {
                    self.collector.append_new_collection(&buffer);
                    buffer.clear();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let current_timestamp: time::SystemTime = time::SystemTime::now();
                let scroll_event = match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        if y > 0.0 {
                            ScrollEvent::Up
                        } else {
                            ScrollEvent::Down
                        }
                    }
                    _ => ScrollEvent::None,
                };
                let new_data = CollectedData {
                    temporal_attributes: TemporalAttributes {
                        timestamp: current_timestamp,
                        time_between_movements: 0.0,
                        total_duration: 0.0,
                        hover_time: 0.0,
                    },
                    mouse_attributes: MouseMovementAttributes {
                        mouse_dpi: 0,
                        current_position: last_position.unwrap(),
                        movement_delta: (0.0, 0.0),
                        distance_traveled: 0.0,
                        path_length: 0.0,
                        direction_angles: 0.0,
                        velocity: 0.0,
                        acceleration: 0.0,
                        curvature: 0.0,
                        jerk: 0.0,
                    },
                    behavioral_attributes: BehavioralAttributes {
                        click_events: ClickEvent::None,
                        scroll_events: scroll_event,
                    },
                };
                buffer.push(new_data);
                if buffer.len() >= MIN_PROCESSING_SIZE {
                    self.collector.append_new_collection(&buffer);
                    buffer.clear();
                }
            }
            _ => {}
        }
    }
}
