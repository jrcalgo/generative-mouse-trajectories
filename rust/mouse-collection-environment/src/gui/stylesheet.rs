use iced::border::Radius;
use iced::theme::palette::Extended;
use iced::widget::button::{Status, Style as ButtonStyle};
use iced::widget::container::Style as ContainerStyle;
use iced::{Background, Border, Color, Shadow, Theme, Vector};
use iced_style::container::StyleSheet;

#[derive(Debug)]
pub struct HotbarButton<'a> {
    pub(crate) theme: &'a Theme,
}

impl<'a> HotbarButton<'a> {
    /// Returns a closure that maps a button status to a style.
    pub fn style(self, _status: Status) -> impl Fn(&Theme, Status) -> ButtonStyle + use<'a> {
        move |_theme: &Theme, _status: Status| -> ButtonStyle {
            match _status {
                Status::Active | Status::Pressed => ButtonStyle {
                    background: Some(Background::Color(Color::from_rgb(0.6, 0.6, 0.6))),
                    text_color: Color::BLACK,
                    border: Border {
                        width: 1.0,
                        color: Color::BLACK,
                        radius: Radius::default(),
                    },
                    shadow: Shadow {
                        color: Color::from_rgba(0.0, 0.0, 0.0, 0.25),
                        offset: Vector::new(1.0, 1.0),
                        blur_radius: 2.0,
                    },
                },
                Status::Hovered => ButtonStyle {
                    background: Some(Background::Color(Color::from_rgb(0.7, 0.7, 0.7))),
                    text_color: Color::BLACK,
                    border: Border {
                        width: 1.0,
                        color: Color::BLACK,
                        radius: Radius::default(),
                    },
                    shadow: Shadow::default(),
                },
                Status::Disabled => ButtonStyle {
                    background: Some(Background::Color(Color::from_rgb(0.5, 0.5, 0.5))),
                    text_color: Color::from_rgb(0.4, 0.4, 0.4),
                    border: Border::default(),
                    shadow: Shadow::default(),
                },
            }
        }
    }
}

/// Custom container style for the hotbar.
#[derive(Debug, Clone, Copy)]
pub struct HotbarStyle<'b> {
    pub(crate) theme: &'b Theme,
}

impl<'b> HotbarStyle<'b> {
    pub fn style(self) -> impl Fn(&Theme) -> ContainerStyle + use<'b> {
        move |_theme: &Theme| -> ContainerStyle {
            ContainerStyle {
                text_color: None,
                background: Some(Background::Color(Color::from_rgb(0.85, 0.85, 0.85))),
                border: Border {
                    width: 1.0,
                    color: Color::BLACK,
                    radius: Radius::default(),
                },
                shadow: Shadow::default(),
            }
        }
    }
}

/// Custom container style for dividers (both horizontal and vertical).
#[derive(Debug, Clone, Copy)]
pub struct DividerStyle<'c> {
    pub(crate) theme: &'c Theme,
}

impl<'c> DividerStyle<'c> {
    pub fn style(self) -> impl Fn(&Theme) -> ContainerStyle + use<'c> {
        move |_theme: &Theme| -> ContainerStyle {
            ContainerStyle {
                text_color: None,
                background: Some(Background::Color(Color::BLACK)),
                border: Border::default(),
                shadow: Shadow::default(),
            }
        }
    }
}
