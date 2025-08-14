use iced::Theme;

pub mod mouse_collector;
mod gui {
    pub(crate) mod mouse_gui;
    mod stylesheet;
}

static COLLECTOR_RT: once_cell::sync::Lazy<tokio::runtime::Runtime> =
    once_cell::sync::Lazy::new(|| {
        tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime")
    });

static COLLECTOR: once_cell::sync::Lazy<std::sync::Arc<mouse_collector::MouseCollector>> =
    once_cell::sync::Lazy::new(|| {
        COLLECTOR_RT.block_on(mouse_collector::MouseCollector::new(false, false))
    });

fn main() {
    let _enter = COLLECTOR_RT.enter();

    let _recording_handle = mouse_collector::MouseCollector::start_recording_thread(&COLLECTOR);

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
