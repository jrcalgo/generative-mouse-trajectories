use iced::Theme;
use mouse_telemetry::mouse_telemetry::MouseCollector;

mod gui {
    pub(crate) mod mouse_gui;
    mod stylesheet;
}

static COLLECTOR_RT: once_cell::sync::Lazy<tokio::runtime::Runtime> =
    once_cell::sync::Lazy::new(|| {
        tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime")
    });

static COLLECTOR: once_cell::sync::Lazy<std::sync::Arc<MouseCollector>> =
    once_cell::sync::Lazy::new(|| {
        COLLECTOR_RT.block_on(MouseCollector::new(false, false))
    });

fn main() {
    let _enter = COLLECTOR_RT.enter();

    let _recording_handle = MouseCollector::start_recording_thread(&COLLECTOR);

    iced::application(
        "Mouse Collection Environment",
        gui::mouse_gui::update,
        gui::mouse_gui::view,
    )
    .theme(|_| Theme::Dark)
    .centered()
    .run()
    .expect("Failed to run application");
}
