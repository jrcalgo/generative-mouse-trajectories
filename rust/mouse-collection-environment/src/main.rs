use iced::Theme;

mod mouse_collector;
mod mouse_gui;

fn main() {
    iced::application(
        "Mouse Collection Environment",
        mouse_gui::update,
        mouse_gui::view,
    )
    .theme(|_| Theme::Dark)
    .centered()
    .run()
    .expect("Failed to run application");
}
