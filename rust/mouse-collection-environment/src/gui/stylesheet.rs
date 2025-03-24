use iced::Theme;
use iced::theme::palette::Extended;
use iced::widget::button::{Status, Style as ButtonStyle};
use iced::widget::container::Style as ContainerStyle;
use iced_style::container::{Appearance, StyleSheet};

#[derive(Debug)]
pub struct HotbarButton<'a> {
    pub(crate) theme: &'a Theme,
}

impl<'a> HotbarButton<'a> {
    /// Returns a closure that maps a button status to a style.
    pub fn style(self, _status: Status) -> impl Fn(&Theme, Status) -> ButtonStyle + use<'a> {
        let theme: &Theme = self.theme;
        let palette: &Extended = theme.extended_palette();

        move |_theme: &Theme, _status: Status| -> ButtonStyle {
            match _status {
                Status::Active | Status::Pressed => ButtonStyle {
                    background: None,
                    text_color: Default::default(),
                    border: Default::default(),
                    shadow: Default::default(),
                },
                Status::Hovered => ButtonStyle {
                    background: None,
                    text_color: palette.background.base.text.scale_alpha(0.8),
                    border: Default::default(),
                    shadow: Default::default(),
                },
                Status::Disabled => ButtonStyle {
                    background: None,
                    text_color: palette.background.base.text.scale_alpha(0.5),
                    border: Default::default(),
                    shadow: Default::default(),
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
                background: None,
                border: Default::default(),
                shadow: Default::default(),
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
                background: None,
                border: Default::default(),
                shadow: Default::default(),
            }
        }
    }
}
