use iced::event::Status;
use iced::widget::button::Status as buttonStatus;
use iced::widget::canvas::{Cache, Frame, Path, Text};
use iced::widget::{Canvas, Space, button, canvas, column, container, row, text};
use iced::{Alignment, Color, Element, Length, Pixels, Point, Rectangle, Size, Theme, mouse};

use crate::gui::stylesheet::*;
use mouse_telemetry::mouse_telemetry::MouseCollector;
use iced_graphics::geometry;
use rand::Rng;
use rand::prelude::ThreadRng;
use std::rc::Rc;
use std::sync::Arc;

pub(crate) const THEME: Theme = Theme::Dark;

const CANVAS_WIDTH: u32 = 900;
const CANVAS_HEIGHT: u32 = 600;
const BUTTON_WIDTH: f32 = 120.0;
const BUTTON_HEIGHT: f32 = 56.0;

#[derive(Debug, Clone)]
pub(crate) enum Message {
    ToggleCollecting,
    EditSettings,
    ViewAnalytics,
    CanvasClicked(Point),
    CanvasClickedWithSize(Point, Size),
    CursorMovedWithSize(Point, Size),
    SettingsMinChanged(String),
    SettingsMaxChanged(String),
    SettingsApply,
    SettingsCancel,
    SettingsReset,
    SettingsDataDirChanged(String),
    ToggleHotbarCollapse,
    SettingsMinDtChanged(String),
    SettingsAlphaChanged(String),
}

pub(crate) struct GuiEnvironment {
    collecting: bool,
    game_buttons: Vec<GameButton>,
    canvas_cache: Rc<Cache>,
    collector: Option<Arc<MouseCollector>>,
    canvas_size: Size,
    show_settings: bool,
    min_buttons: i32,
    max_buttons: i32,
    min_input: String,
    max_input: String,
    data_dir_input: String,
    hotbar_collapsed: bool,
    min_dt_input: String,
    alpha_input: String,
}

#[derive(Clone)]
struct GameButton {
    rect: Rectangle,
    is_real: bool,
}

impl GameButton {
    fn contains(&self, point: Point) -> bool {
        point.x >= self.rect.x
            && point.x <= self.rect.x + self.rect.width
            && point.y >= self.rect.y
            && point.y <= self.rect.y + self.rect.height
    }
}

impl Default for GuiEnvironment {
    fn default() -> Self {
        Self {
            collecting: false,
            game_buttons: Vec::new(),
            canvas_cache: Rc::new(Cache::new()),
            collector: Some(crate::COLLECTOR.clone()),
            canvas_size: Size::new(CANVAS_WIDTH as f32, CANVAS_HEIGHT as f32),
            show_settings: false,
            min_buttons: 3,
            max_buttons: 6,
            min_input: String::from("3"),
            max_input: String::from("6"),
            data_dir_input: crate::COLLECTOR.get_data_dir(),
            hotbar_collapsed: false,
            min_dt_input: format!("{:.4}", crate::COLLECTOR.get_min_derivative_dt()),
            alpha_input: format!("{:.3}", crate::COLLECTOR.get_smoothing_alpha()),
        }
    }
}

impl GuiEnvironment {
    fn spawn_buttons(&mut self) {
        self.game_buttons.clear();
        let mut rng: ThreadRng = rand::rng();
        let range_start = self.min_buttons.max(1);
        let range_end = self.max_buttons.max(range_start);
        let num_buttons: i32 = rng.random_range(range_start..=range_end);
        let real_index: i32 = rng.random_range(0..num_buttons);

        let available_width = self.canvas_size.width.max(BUTTON_WIDTH);
        let available_height = self.canvas_size.height.max(BUTTON_HEIGHT);

        for i in 0..num_buttons {
            let x = rng.random_range(0.0..(available_width - BUTTON_WIDTH));
            let y = rng.random_range(0.0..(available_height - BUTTON_HEIGHT));
            let rect = Rectangle {
                x,
                y,
                width: BUTTON_WIDTH,
                height: BUTTON_HEIGHT,
            };
            self.game_buttons.push(GameButton {
                rect,
                is_real: i == real_index,
            });
        }
        self.canvas_cache.clear();
    }

    fn respawn_if_out_of_bounds(&mut self) {
        let w = self.canvas_size.width;
        let h = self.canvas_size.height;
        let mut out_of_bounds = false;
        for b in &self.game_buttons {
            if b.rect.x + b.rect.width > w || b.rect.y + b.rect.height > h {
                out_of_bounds = true;
                break;
            }
        }
        if out_of_bounds {
            self.spawn_buttons();
        }
    }
}

pub(crate) fn update(gui: &mut GuiEnvironment, message: Message) {
    match message {
        Message::ToggleCollecting => {
            gui.collecting = !gui.collecting;
            if gui.collecting {
                gui.show_settings = false;
                gui.spawn_buttons();
            } else {
                gui.game_buttons.clear();
                gui.canvas_cache.clear();
            }
        }
        Message::EditSettings => {
            if gui.collecting {
                gui.collecting = false;
                gui.game_buttons.clear();
                gui.canvas_cache.clear();
            }
            gui.show_settings = true;
            gui.min_input = gui.min_buttons.to_string();
            gui.max_input = gui.max_buttons.to_string();
            if let Some(c) = &gui.collector {
                gui.data_dir_input = c.get_data_dir();
            }
        }
        Message::ViewAnalytics => {}
        Message::SettingsMinChanged(s) => {
            gui.min_input = s;
        }
        Message::SettingsMaxChanged(s) => {
            gui.max_input = s;
        }
        Message::SettingsApply => {
            let parsed_min = gui
                .min_input
                .trim()
                .parse::<i32>()
                .unwrap_or(gui.min_buttons);
            let parsed_max = gui
                .max_input
                .trim()
                .parse::<i32>()
                .unwrap_or(gui.max_buttons);
            let minv = parsed_min.max(1);
            let maxv = parsed_max.max(minv);
            gui.min_buttons = minv;
            gui.max_buttons = maxv;

            if let Some(c) = &gui.collector {
                c.set_data_dir(gui.data_dir_input.clone());
                if let Ok(v) = gui.min_dt_input.trim().parse::<f64>() {
                    c.set_min_derivative_dt(v.max(0.0001));
                }
                if let Ok(a) = gui.alpha_input.trim().parse::<f64>() {
                    c.set_smoothing_alpha(a);
                }
            }

            gui.show_settings = false;
        }
        Message::SettingsCancel => {
            gui.show_settings = false;
        }
        Message::SettingsReset => {
            gui.min_buttons = 3;
            gui.max_buttons = 6;
            gui.min_input = String::from("3");
            gui.max_input = String::from("6");
        }
        Message::SettingsDataDirChanged(s) => {
            gui.data_dir_input = s;
        }
        Message::ToggleHotbarCollapse => {
            gui.hotbar_collapsed = !gui.hotbar_collapsed;
        }
        Message::SettingsMinDtChanged(s) => {
            gui.min_dt_input = s;
        }
        Message::SettingsAlphaChanged(s) => {
            gui.alpha_input = s;
        }
        Message::CanvasClicked(point) => {
            if !gui.collecting {
                return;
            }
            if let Some(collector) = &gui.collector {
                collector.push_left_click((point.x as f64, point.y as f64));
            }
            for button in &gui.game_buttons {
                if button.contains(point) {
                    if button.is_real {
                        gui.spawn_buttons();
                    }
                    break;
                }
            }
        }
        Message::CanvasClickedWithSize(point, size) => {
            gui.canvas_size = size;
            if !gui.collecting {
                return;
            }
            gui.respawn_if_out_of_bounds();
            if let Some(collector) = &gui.collector {
                collector.push_left_click((point.x as f64, point.y as f64));
            }
            for button in &gui.game_buttons {
                if button.contains(point) {
                    if button.is_real {
                        gui.spawn_buttons();
                    }
                    break;
                }
            }
        }
        Message::CursorMovedWithSize(point, size) => {
            gui.canvas_size = size;
            if !gui.collecting {
                return;
            }
            gui.respawn_if_out_of_bounds();
            if let Some(collector) = &gui.collector {
                collector.push_move((point.x as f64, point.y as f64));
            }
        }
    }
}

pub(crate) fn view(gui: &GuiEnvironment) -> Element<'_, Message> {
    let canvas = Canvas::new(GameCanvas {
        game_buttons: gui.game_buttons.clone(),
        cache: gui.canvas_cache.clone(),
    })
    .width(Length::Fill)
    .height(Length::Fill);

    let start_stop_label = if gui.collecting {
        "Stop Collecting"
    } else {
        "Start Collecting"
    };

    let start_stop_button = button(start_stop_label)
        .on_press(Message::ToggleCollecting)
        .style(HotbarButton::style(
            HotbarButton { theme: &THEME },
            buttonStatus::Active,
        ));

    let settings_button =
        button("Settings")
            .on_press(Message::EditSettings)
            .style(HotbarButton::style(
                HotbarButton { theme: &THEME },
                buttonStatus::Active,
            ));

    // Live statistics box
    let stats_text = if let Some(collector) = &gui.collector {
        collector.get_latest_derived_stats()
    } else {
        vec![String::from("No data yet")]
    };

    // Build a dynamic 3-column grid with equal column widths
    let num_cols: usize = 3;
    let num_rows: usize = (stats_text.len() + num_cols - 1) / num_cols;

    let mut cols: Vec<Element<'_, Message>> = Vec::new();
    for col_index in 0..num_cols {
        let mut col = column![]
            .spacing(6)
            .width(Length::FillPortion(1))
            .height(Length::Shrink)
            .align_x(Alignment::Start);
        for row_index in 0..num_rows {
            let idx = row_index * num_cols + col_index;
            let label = stats_text
                .get(idx)
                .cloned()
                .unwrap_or_else(|| String::from(""));
            let stat =
                container(
                    text(label)
                        .size(20)
                        .style(|_theme: &Theme| iced::widget::text::Style {
                            color: Some(Color::BLACK),
                        }),
                )
                .padding([2, 6])
                .width(Length::Fill)
                .height(Length::Shrink);
            col = col.push(stat);
        }
        cols.push(col.into());
    }

    let mut iter = cols.into_iter();
    let c1 = iter
        .next()
        .unwrap_or_else(|| column![].width(Length::FillPortion(1)).into());
    let c2 = iter
        .next()
        .unwrap_or_else(|| column![].width(Length::FillPortion(1)).into());
    let c3 = iter
        .next()
        .unwrap_or_else(|| column![].width(Length::FillPortion(1)).into());

    let stats_row = row![
        c1,
        Space::with_width(Length::Fixed(16.0)),
        c2,
        Space::with_width(Length::Fixed(16.0)),
        c3,
    ]
    .spacing(0)
    .align_y(Alignment::Start)
    .width(Length::Fill)
    .height(Length::Shrink);

    // Make stats area scrollable vertically
    let stats_scroll_v = iced::widget::scrollable(
        container(stats_row)
            .padding([8, 8])
            .width(Length::Fill)
            .height(Length::Shrink),
    )
    .width(Length::FillPortion(8))
    .height(Length::Fill);

    // Collapse/expand small button
    let collapse_btn =
        button("▼")
            .on_press(Message::ToggleHotbarCollapse)
            .style(HotbarButton::style(
                HotbarButton { theme: &THEME },
                buttonStatus::Active,
            ));

    // Constrain stats area to a portion so spacers can equalize distances
    let stats_box = container(stats_scroll_v)
        .width(Length::FillPortion(8))
        .height(Length::Fill)
        .style(HotbarStyle::style(HotbarStyle { theme: &THEME }));

    // Left buttons in a vertical column
    let buttons_column = column![
        start_stop_button,
        Space::with_height(Length::Fixed(10.0)),
        settings_button,
    ]
    .spacing(10)
    .align_x(Alignment::Start)
    .width(Length::Shrink)
    .height(Length::Shrink);

    // Hotbar row including collapse button at far right
    let hotbar_row = row![
        Space::with_width(Length::Fixed(10.0)),
        buttons_column,
        Space::with_width(Length::Fixed(10.0)),
        stats_box,
        Space::with_width(Length::Fixed(10.0)),
        collapse_btn,
        Space::with_width(Length::Fixed(10.0)),
    ]
    .spacing(0)
    .padding(10)
    .align_y(Alignment::Center);

    let hotbar = container(hotbar_row)
        .width(Length::Fill)
        .height(Length::FillPortion(1))
        .style(HotbarStyle::style(HotbarStyle { theme: &THEME }));

    let horizontal_divider = container("")
        .height(Length::Fixed(1.0))
        .width(Length::Fill)
        .style(DividerStyle::style(DividerStyle { theme: &THEME }));

    let game_area = container(canvas)
        .width(Length::Fill)
        .height(Length::FillPortion(5))
        .style(GameAreaStyle::style(GameAreaStyle { theme: &THEME }));

    // Settings panel (visible only when show_settings)
    let settings_ui: Option<Element<'_, Message>> = if gui.show_settings {
        let label_style = |_theme: &Theme| iced::widget::text::Style {
            color: Some(Color::BLACK),
        };
        let min_row = row![
            container(text("Min buttons:").style(label_style)).width(Length::Shrink),
            Space::with_width(Length::Fixed(10.0)),
            iced::widget::text_input("3", &gui.min_input)
                .on_input(Message::SettingsMinChanged)
                .size(18)
                .padding(6)
                .width(Length::Fixed(120.0)),
        ]
        .spacing(8)
        .align_y(Alignment::Center);

        let max_row = row![
            container(text("Max buttons:").style(label_style)).width(Length::Shrink),
            Space::with_width(Length::Fixed(10.0)),
            iced::widget::text_input("6", &gui.max_input)
                .on_input(Message::SettingsMaxChanged)
                .size(18)
                .padding(6)
                .width(Length::Fixed(120.0)),
        ]
        .spacing(8)
        .align_y(Alignment::Center);

        // SECOND COLUMN: tunables and data dir
        let second_col = column![
            row![
                container(text("Min dt (s):").style(label_style)).width(Length::Shrink),
                Space::with_width(Length::Fixed(10.0)),
                iced::widget::text_input("0.004", &gui.min_dt_input)
                    .on_input(Message::SettingsMinDtChanged)
                    .size(18)
                    .padding(6)
                    .width(Length::Fixed(120.0)),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
            Space::with_height(Length::Fixed(8.0)),
            row![
                container(text("Alpha (0-1):").style(label_style)).width(Length::Shrink),
                Space::with_width(Length::Fixed(10.0)),
                iced::widget::text_input("0.3", &gui.alpha_input)
                    .on_input(Message::SettingsAlphaChanged)
                    .size(18)
                    .padding(6)
                    .width(Length::Fixed(120.0)),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
            Space::with_height(Length::Fixed(8.0)),
            row![
                container(text("Data dir:").style(label_style)).width(Length::Shrink),
                Space::with_width(Length::Fixed(10.0)),
                iced::widget::text_input("data", &gui.data_dir_input)
                    .on_input(Message::SettingsDataDirChanged)
                    .size(18)
                    .padding(6)
                    .width(Length::Fixed(240.0)),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
        ]
        .width(Length::Fill);

        let actions = row![
            button("Apply")
                .on_press(Message::SettingsApply)
                .style(HotbarButton::style(
                    HotbarButton { theme: &THEME },
                    buttonStatus::Active
                )),
            Space::with_width(Length::Fixed(10.0)),
            button("Cancel")
                .on_press(Message::SettingsCancel)
                .style(HotbarButton::style(
                    HotbarButton { theme: &THEME },
                    buttonStatus::Active
                )),
            Space::with_width(Length::Fixed(10.0)),
            button("Reset")
                .on_press(Message::SettingsReset)
                .style(HotbarButton::style(
                    HotbarButton { theme: &THEME },
                    buttonStatus::Active
                )),
        ]
        .spacing(10)
        .align_y(Alignment::Center);

        // Layout two columns in settings
        let settings_grid = row![
            column![min_row, Space::with_height(Length::Fixed(8.0)), max_row]
                .width(Length::FillPortion(1)),
            Space::with_width(Length::Fixed(20.0)),
            second_col.width(Length::FillPortion(1)),
        ]
        .width(Length::Fill);

        let panel = column![
            settings_grid,
            Space::with_height(Length::Fixed(12.0)),
            actions,
        ]
        .spacing(6)
        .align_x(Alignment::Start);

        Some(
            container(panel)
                .padding(12)
                .width(Length::Fill)
                .height(Length::Shrink)
                .style(HotbarStyle::style(HotbarStyle { theme: &THEME }))
                .into(),
        )
    } else {
        None
    };

    // Reopen button when hotbar is collapsed
    let reopen_row = row![
        Space::with_width(Length::Fill),
        button("▲")
            .on_press(Message::ToggleHotbarCollapse)
            .style(HotbarButton::style(
                HotbarButton { theme: &THEME },
                buttonStatus::Active
            )),
        Space::with_width(Length::Fixed(8.0)),
    ]
    .padding([6, 8])
    .width(Length::Fill)
    .align_y(Alignment::End);

    let content: Element<'_, Message> = if gui.hotbar_collapsed {
        // collapsed: only game area and bottom-right reopen
        column![game_area, reopen_row,]
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    } else if let Some(settings_panel) = settings_ui {
        column![game_area, horizontal_divider, settings_panel, hotbar]
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    } else {
        column![game_area, horizontal_divider, hotbar]
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    };

    container(content)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

struct GameCanvas<R>
where
    R: geometry::Renderer,
{
    game_buttons: Vec<GameButton>,
    cache: Rc<Cache<R>>,
}

impl<Theme, R> canvas::Program<Message, Theme, R> for GameCanvas<R>
where
    R: geometry::Renderer,
{
    type State = ();

    fn update(
        &self,
        _state: &mut Self::State,
        event: canvas::Event,
        _bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> (iced::event::Status, Option<Message>) {
        match event {
            canvas::Event::Mouse(mouse::Event::CursorMoved { position }) => (
                Status::Captured,
                Some(Message::CursorMovedWithSize(position, _bounds.size())),
            ),
            canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if let Some(pos) = cursor.position() {
                    (
                        Status::Captured,
                        Some(Message::CanvasClickedWithSize(pos, _bounds.size())),
                    )
                } else {
                    (Status::Ignored, None)
                }
            }
            _ => (Status::Ignored, None),
        }
    }

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &R,
        _theme: &Theme,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<R::Geometry> {
        let geometry = self.cache.draw(
            renderer,
            Size::new(_bounds.width, _bounds.height),
            |frame: &mut Frame<R>| {
                for button in &self.game_buttons {
                    let path = Path::rectangle(
                        Point::new(button.rect.x, button.rect.y),
                        Size::new(button.rect.width, button.rect.height),
                    );
                    frame.fill(&path, Color::from_rgb(0.2, 0.7, 0.3));

                    let center_x = button.rect.x + button.rect.width / 2.0;
                    let center_y = button.rect.y + button.rect.height / 2.0;
                    let text = {
                        let mut t = Text {
                            content: "Click me!".into(),
                            position: Point::new(center_x, center_y),
                            color: Color::WHITE,
                            size: Pixels::from(18.0),
                            ..Default::default()
                        };
                        use iced::alignment::{Horizontal, Vertical};
                        t.horizontal_alignment = Horizontal::Center;
                        t.vertical_alignment = Vertical::Center;
                        t
                    };
                    frame.fill_text(text);
                }
            },
        );
        vec![geometry]
    }

    fn mouse_interaction(
        &self,
        _state: &Self::State,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        mouse::Interaction::Pointer
    }
}
