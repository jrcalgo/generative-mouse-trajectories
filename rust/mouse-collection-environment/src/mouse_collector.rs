//!
//!
//!

use std::fmt;
use std::fs;
use std::io::Error;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::{JoinHandle, current};
use std::{thread, time};
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::{MutexGuard, RwLock as TokioRwLock, RwLockWriteGuard};
use tokio::task;
use winit::application::ApplicationHandler;
use winit::event::{MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};
use winit::*;

#[derive(Clone)]
pub enum ClickEvent {
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
pub enum ScrollEvent {
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
pub struct MouseMovementAttributes {
    pub mouse_dpi: usize,
    pub current_position: Coordinate,
    pub movement_delta: Coordinate,
    pub distance_traveled: f64,
    pub path_length: f64,
    pub direction_angles: f64,
    pub velocity: f64,
    pub acceleration: f64,
    pub momentum: f64,
    pub curvature: f64,
    pub jerk: f64,
}

#[derive(Clone)]
pub struct TemporalAttributes {
    pub timestamp: time::SystemTime,
    pub time_between_movements: f64,
    pub total_duration: f64,
    pub hover_time: f64,
}

#[derive(Clone)]
pub struct BehavioralAttributes {
    pub click_events: ClickEvent,
    pub scroll_events: ScrollEvent,
}

#[derive(Clone)]
pub struct CollectedData {
    pub temporal_attributes: TemporalAttributes,
    pub mouse_attributes: MouseMovementAttributes,
    pub behavioral_attributes: BehavioralAttributes,
}

#[derive(Clone, Default)]
pub struct DerivedAttributes {
    pub average_velocity: f64,
    pub peak_velocity: f64,
    pub average_acceleration: f64,
    pub peak_acceleration: f64,
    pub average_momentum: f64,
    pub peak_momentum: f64,
    pub smoothness: f64,
    pub deviation_from_ideal_path: f64,
    pub idle_time: f64,
    pub fitts_index_of_difficulty: f64,
    pub fitts_movement_time: f64,
}

#[derive(Clone, Default)]
pub struct SystemAttributes {
    pub screen_width: usize,
    pub screen_height: usize,
    pub window_width: usize,
    pub window_height: usize,
    pub os_device_info: String,
}

#[derive(Default)]
pub struct CollectionBuffer {
    pub collected_data: Vec<CollectedData>,
}

#[derive(Default)]
pub struct DerivedBuffer {
    pub derived_data: Vec<DerivedAttributes>,
}

pub struct MouseCollector {
    pub record_file_path: PathBuf,
    pub system_attributes: SystemAttributes,
    pub collection_buffer: Arc<TokioRwLock<CollectionBuffer>>,
    pub derivation_buffer: Arc<TokioRwLock<DerivedBuffer>>,
    pub start_timestamp: Option<time::SystemTime>,
    pub last_left_click_timestamp: Option<time::SystemTime>,
    pub listening_handle: TokioMutex<Option<task::JoinHandle<()>>>,
    pub reporting_handle: TokioMutex<Option<task::JoinHandle<()>>>,
    pub recording_handle: Mutex<Option<JoinHandle<()>>>,
}

impl MouseCollector {
    pub async fn new(start: bool, init_cmd_reporting: bool) -> Arc<Self> {
        let record_file_path = PathBuf::from("./mouse_data.txt");

        let collection_buffer = Arc::new(TokioRwLock::new(CollectionBuffer {
            collected_data: Vec::new(),
        }));

        let mouse_collector = Arc::new(MouseCollector {
            record_file_path,
            system_attributes: Default::default(),
            collection_buffer,
            derivation_buffer: Default::default(),
            start_timestamp: None,
            last_left_click_timestamp: None,
            listening_handle: TokioMutex::new(None),
            reporting_handle: TokioMutex::new(None),
            recording_handle: Mutex::new(None),
        });

        if start {
            let collector_clone = mouse_collector.clone();
            let handlers = Self::start_collecting(collector_clone, init_cmd_reporting).await;

            let mut listening_handle = mouse_collector.listening_handle.lock().await.as_ref();
            let mut recording_handle = mouse_collector.recording_handle.lock().unwrap().as_ref();
            let mut reporting_handle = mouse_collector.reporting_handle.lock().await.as_ref();

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
        let listen_clone = self.clone();
        let listening_handle = tokio::spawn(async move {
            listen_clone.mouse_listener().await;
        });

        let record_clone = self.clone();
        let recording_handle = thread::spawn(move || {
            record_clone.mouse_recorder();
        });

        let reporting_handle = if report {
            let report_clone = self.clone();
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

    fn append_new_data(&self, new_data: CollectedData) -> Result<(), Error> {
        let mut buffer: RwLockWriteGuard<CollectionBuffer> =
            tokio::runtime::Handle::current().block_on(self.collection_buffer.write());
        buffer.collected_data.push(new_data);
        Ok(())
    }

    fn append_new_derivations(&self, new_derivations: DerivedAttributes) -> Result<(), Error> {
        let mut buffer: RwLockWriteGuard<DerivedBuffer> = tokio::runtime::Handle::current().block_on(self.derivation_buffer.write());
        buffer.derived_data.push(new_derivations);
        Ok(())
    }

    fn get_current_collection(&self) -> Option<Vec<CollectedData>> {
        Some(
            tokio::runtime::Handle::current()
                .block_on(async { self.collection_buffer.read().await.collected_data.clone() }),
        )
    }

    fn get_current_derivations(&self) -> Option<Vec<DerivedAttributes>> {
        Some(
            tokio::runtime::Handle::current()
                .block_on(async { self.derivation_buffer.read().await.derived_data.clone() }),
        )
    }

    fn set_updated_collection(&self, updated_data: Vec<CollectedData>) {

    }
}

trait MouseListener {
    async fn mouse_listener(self: Arc<Self>);
}

impl MouseListener for MouseCollector {
    async fn mouse_listener(self: Arc<Self>) {
        let collector = self.clone();

        tokio::task::spawn_blocking(move || {
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

trait MouseRecorder {
    fn mouse_recorder(&self);
    fn calculate_collection_attributes(&self, data: &Vec<CollectedData>) -> Vec<CollectedData>;
    fn calculate_derived_attributes(&self, data: &Vec<CollectedData>) -> DerivedAttributes;
    fn write_collection_data(&self, data: &Vec<CollectedData>);
    fn write_derivation_data(&self, data: &Vec<DerivedAttributes>);
}

impl MouseRecorder for MouseCollector {
    fn mouse_recorder(&self) {
        let new_file = fs::File::create_new(&self.record_file_path).expect("Unable to create file");

        loop {
            if let Some(data) = self.get_current_collection() {
                let current_timestamp = time::SystemTime::now();
                if data.len() > 1000 {
                    let collection_data_vec = Self::calculate_collection_attributes(self, &data);
                    let derived_data = Self::calculate_derived_attributes(self, &data);
                    self.append_new_derivations(self, derived_data);
                }
            } else {
                thread::sleep(time::Duration::from_secs(1));
            }
        }
    }

    fn calculate_collection_attributes(&self, data: &Vec<CollectedData>) -> Vec<CollectedData> {
        todo!();
    }

    fn calculate_derived_attributes(&self, data: &Vec<CollectedData>) -> DerivedAttributes {
        let length = data.iter().len() as i32;
        if length <= 1 {
            return Default::default();
        }

        let sum_velocity: f64 = data
            .iter()
            .map(|d| d.mouse_attributes.velocity)
            .sum::<f64>();
        let average_velocity: f64 = sum_velocity / (length - 1) as f64;
        let peak_velocity: f64 = data
            .iter()
            .map(|d| d.mouse_attributes.velocity)
            .reduce(f64::max)
            .unwrap();

        let sum_acceleration: f64 = data[2..]
            .iter()
            .map(|d| d.mouse_attributes.acceleration)
            .sum::<f64>();
        let average_acceleration: f64 = sum_acceleration / (length - 2) as f64;
        let peak_acceleration: f64 = data
            .iter()
            .map(|d| d.mouse_attributes.acceleration)
            .reduce(f64::max)
            .unwrap();

        let sum_momentum: f64 = data
            .iter()
            .map(|d| d.mouse_attributes.momentum)
            .sum::<f64>();
        let average_momentum: f64 = sum_momentum / (length - 1) as f64;
        let peak_momentum: f64 = data
            .iter()
            .map(|d| d.mouse_attributes.momentum)
            .reduce(f64::max)
            .unwrap();

        let active_time: f64 = data
            .iter()
            .map(|d| d.temporal_attributes.time_between_movements)
            .sum::<f64>();
        let idle_time: f64 = data
            .iter()
            .map(|d| d.temporal_attributes.total_duration - active_time)
            .sum::<f64>();

        let sum_jerk: f64 = data[3..]
            .iter()
            .map(|d| d.mouse_attributes.jerk.abs())
            .sum::<f64>();
        let smoothness: f64 = {
            if length > 3 {
                1.0 / (sum_jerk / (length - 3) as f64)
            } else {
                0.0
            }
        };

        let position_history: Vec<Coordinate> = data
            .iter()
            .map(|d| d.mouse_attributes.current_position)
            .collect::<Vec<Coordinate>>();
        let ideal_dx: f64 =
            position_history.last().unwrap().0 - position_history.first().unwrap().0;
        let ideal_dy: f64 =
            position_history.last().unwrap().1 - position_history.first().unwrap().1;
        let ideal_path_length: f64 = (ideal_dx.powi(2) + ideal_dy.powi(2)).sqrt();
        let deviation_from_ideal_path: f64 = data
            .iter()
            .map(|d| d.mouse_attributes.path_length)
            .sum::<f64>()
            / ideal_path_length;

        let fitts_index_of_difficulty: f64 = ((ideal_path_length / 100.0) + 1.0).log2();
        let fitts_movement_time: f64 = data
            .iter()
            .map(|d| d.temporal_attributes.total_duration)
            .sum::<f64>();

        DerivedAttributes {
            average_velocity,
            peak_velocity,
            average_acceleration,
            peak_acceleration,
            average_momentum,
            peak_momentum,
            smoothness,
            deviation_from_ideal_path,
            idle_time,
            fitts_index_of_difficulty,
            fitts_movement_time,
        }
    }

    fn write_collection_data(&self, data: &Vec<CollectedData>) {}
    fn write_derivation_data(&self, data: &Vec<DerivedAttributes>) {}
}

trait CommandlineOutput {
    async fn cmd_report(&self);
}

impl CommandlineOutput for MouseCollector {
    async fn cmd_report(&self) {
        let derivation_buffer = self.derivation_buffer.read().await;
        let derived_attributes = derivation_buffer.derived_data.last().unwrap();

        println!("Average velocity: {}", derived_attributes.average_velocity);
        println!("Peak velocity: {}", derived_attributes.peak_velocity);
        println!(
            "Average acceleration: {}",
            derived_attributes.average_acceleration
        );
        println!(
            "Peak acceleration: {}",
            derived_attributes.peak_acceleration
        );
        println!("Average momentum: {}", derived_attributes.average_momentum);
        println!("Peak momentum: {}", derived_attributes.peak_momentum);
        println!("Smoothness: {}", derived_attributes.smoothness);
        println!(
            "Deviation from ideal path: {}",
            derived_attributes.deviation_from_ideal_path
        );
        println!("Idle time: {}", derived_attributes.idle_time);
        println!(
            "Fitts's index of difficulty: {}",
            derived_attributes.fitts_index_of_difficulty
        );
        println!(
            "Fitts's movement time: {}",
            derived_attributes.fitts_movement_time
        );
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
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let current_timestamp = time::SystemTime::now();
                let coord = (position.x, position.y);
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
                        movement_delta: (0.0, 0.0),
                        distance_traveled: 0.0,
                        path_length: 0.0,
                        direction_angles: 0.0,
                        velocity: 0.0,
                        acceleration: 0.0,
                        momentum: 0.0,
                        curvature: 0.0,
                        jerk: 0.0,
                    },
                    behavioral_attributes: BehavioralAttributes {
                        click_events: ClickEvent::None,
                        scroll_events: ScrollEvent::None,
                    },
                };
                self.collector.append_new_data(new_data);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let current_timestamp = time::SystemTime::now();
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
                        current_position: (0.0, 0.0),
                        movement_delta: (0.0, 0.0),
                        distance_traveled: 0.0,
                        path_length: 0.0,
                        direction_angles: 0.0,
                        velocity: 0.0,
                        acceleration: 0.0,
                        momentum: 0.0,
                        curvature: 0.0,
                        jerk: 0.0,
                    },
                    behavioral_attributes: BehavioralAttributes {
                        click_events: click_event,
                        scroll_events: ScrollEvent::None,
                    },
                };
                self.collector.append_new_data(new_data);
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
                        current_position: (0.0, 0.0),
                        movement_delta: (0.0, 0.0),
                        distance_traveled: 0.0,
                        path_length: 0.0,
                        direction_angles: 0.0,
                        velocity: 0.0,
                        acceleration: 0.0,
                        momentum: 0.0,
                        curvature: 0.0,
                        jerk: 0.0,
                    },
                    behavioral_attributes: BehavioralAttributes {
                        click_events: ClickEvent::None,
                        scroll_events: scroll_event,
                    },
                };
                self.collector.append_new_data(new_data);
            }
            _ => {}
        }
    }
}
