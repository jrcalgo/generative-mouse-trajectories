use iced::event::Status;
use iced::widget::button::Status as buttonStatus;
use iced::widget::canvas::{Cache, Frame, Path, Text};
use iced::widget::{Canvas, button, canvas, column, container, row};
use iced::{Alignment, Color, Element, Length, Pixels, Point, Rectangle, Size, Theme, mouse};

use crate::gui::stylesheet::*;
use crate::mouse_collector::MouseCollector;
use iced_graphics::core::Widget;
use iced_graphics::geometry;
use rand::Rng;
use rand::prelude::ThreadRng;
use std::rc::Rc;

pub(crate) const THEME: Theme = Theme::Dark;

const CANVAS_WIDTH: u32 = 900;
const CANVAS_HEIGHT: u32 = 900;
const BUTTON_WIDTH: f32 = 100.0;
const BUTTON_HEIGHT: f32 = 50.0;


#[derive(Debug, Clone)]
pub(crate) enum Message {
    ToggleCollecting,
    EditSettings,
    ViewAnalytics,
    CanvasClicked(Point),
}

/// The application’s state.
#[derive(Default)]
pub(crate) struct GuiEnvironment {
    collecting: bool,
    game_buttons: Vec<GameButton>,
    canvas_cache: Rc<Cache>,
}

/// A game button with a rectangular area and a flag indicating the "real" button.
#[derive(Clone)]
struct GameButton {
    rect: Rectangle,
    is_real: bool,
}

/// Returns whether the given point is inside the game button.
impl GameButton {
    fn contains(&self, point: Point) -> bool {
        point.x >= self.rect.x
            && point.x <= self.rect.x + self.rect.width
            && point.y >= self.rect.y
            && point.y <= self.rect.y + self.rect.height
    }
}

/// The application's environment.
impl GuiEnvironment {
    fn new() -> Self {
        Self {
            collecting: false,
            game_buttons: Vec::new(),
            canvas_cache: Rc::new(Cache::new()),
        }
    }

    fn spawn_buttons(&mut self) {
        self.game_buttons.clear();
        let mut rng: ThreadRng = rand::rng();
        let num_buttons: i32 = rng.random_range(3..=6);
        let real_index: i32 = rng.random_range(0..num_buttons);

        for i in 0..num_buttons {
            let x = rng.random_range(0.0..(CANVAS_WIDTH as f32 - BUTTON_WIDTH));
            let y = rng.random_range(0.0..(CANVAS_HEIGHT as f32 - BUTTON_HEIGHT));
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
}

/// Update function: processes messages and updates the state.
pub(crate) fn update(gui: &mut GuiEnvironment, message: Message) {
    match message {
        Message::ToggleCollecting => {
            gui.collecting = !gui.collecting;
            if gui.collecting {
                gui.spawn_buttons();
            } else {
                gui.game_buttons.clear();
                gui.canvas_cache.clear();
            }
        }
        Message::EditSettings => {
            todo!();
        }
        Message::ViewAnalytics => {
            todo!();
        }
        Message::CanvasClicked(point) => {
            if gui.collecting {
                for button in &gui.game_buttons {
                    if button.contains(point) {
                        if button.is_real {
                            gui.spawn_buttons();
                        }
                        break;
                    }
                }
            }
        }
    }
}

/// View function: produces the widget tree from the current state.
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

    let analytics_button = button("View Analytics")
        .on_press(Message::ViewAnalytics)
        .style(HotbarButton::style(
            HotbarButton { theme: &THEME },
            buttonStatus::Active,
        ));

    let vertical_divider = container("")
        .width(Length::Fill)
        .height(Length::Fill)
        .style(DividerStyle::style(DividerStyle { theme: &THEME }));

    let hotbar_row = row![start_stop_button, vertical_divider, settings_button, analytics_button]
        .spacing(10)
        .padding(10)
        .align_y(Alignment::Center);

    let hotbar = container(hotbar_row)
        .width(Length::Fill)
        .style(HotbarStyle::style(HotbarStyle { theme: &THEME }));

    let horizontal_divider = container("")
        .height(Length::Fill)
        .width(Length::Fill)
        .style(DividerStyle::style(DividerStyle { theme: &THEME }));

    let content = column![canvas, horizontal_divider, hotbar]
        .width(Length::Fill)
        .height(Length::Fill);

    container(content)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

/// The canvas for the game buttons.
struct GameCanvas<R>
where
    R: geometry::Renderer,
{
    game_buttons: Vec<GameButton>,
    cache: Rc<Cache<R>>,
}

/// The canvas program.
impl<Message, Theme, R> canvas::Program<Message, Theme, R> for GameCanvas<R>
where
    R: geometry::Renderer,
{
    type State = ();

    /// Initializes the canvas.
    fn update(
        &self,
        _state: &mut Self::State,
        _event: canvas::Event,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> (Status, Option<Message>) {
        (Status::Ignored, None)
    }

    /// Draws the canvas.
    fn draw(
        &self,
        _state: &Self::State,
        renderer: &R,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<R::Geometry> {
        // The cache draws using the provided renderer and closure.
        let geometry = self
            .cache
            .draw(renderer, bounds.size(), |frame: &mut Frame<R>| {
                for button in &self.game_buttons {
                    let path = Path::rectangle(
                        Point::new(button.rect.x, button.rect.y),
                        Size::new(button.rect.width, button.rect.height),
                    );
                    frame.fill(&path, Color::from_rgb(0.2, 0.7, 0.3));
                    let text_position = Point::new(
                        button.rect.x + button.rect.width / 2.0,
                        button.rect.y + button.rect.height / 2.0,
                    );
                    frame.fill_text(Text {
                        content: "Click me!".into(),
                        position: text_position,
                        color: Color::WHITE,
                        size: Pixels::from(20.0),
                        ..Default::default()
                    });
                }
            });
        vec![geometry]
    }

    /// Returns the mouse interaction for the canvas.
    fn mouse_interaction(
        &self,
        _state: &Self::State,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        mouse::Interaction::default()
    }
}
