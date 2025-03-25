use iced::Theme;

mod mouse_collector;
mod gui {
    pub(crate) mod mouse_gui;
    mod stylesheet;
}

fn main() {
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
