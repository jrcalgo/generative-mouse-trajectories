use iced::event::Status;
use iced::widget::canvas::{Cache, Frame, Geometry, Path, Text};
use iced::widget::{Canvas, button, canvas, column, container, row};
use iced::{
    Alignment, Color, Element, Event, Length, Pixels, Point, Rectangle, Settings, Size, Theme,
    application, event, mouse,
};
use iced_graphics::core::Widget;
use iced_graphics::geometry;
use rand::Rng;

const CANVAS_WIDTH: u32 = 900;
const CANVAS_HEIGHT: u32 = 900;
const BUTTON_WIDTH: f32 = 100.0;
const BUTTON_HEIGHT: f32 = 50.0;

#[derive(Debug, Clone)]
enum Message {
    ToggleCollecting,
    EditSettings,
    CanvasClicked(Point),
}

/// The application’s state.
#[derive(Default)]
struct GuiEnvironment {
    collecting: bool,
    game_buttons: Vec<GameButton>,
    canvas_cache: Cache,
}

/// A game button with a rectangular area and a flag indicating the "real" button.
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

impl GuiEnvironment {
    fn new() -> Self {
        Self {
            collecting: false,
            game_buttons: Vec::new(),
            canvas_cache: Cache::new(),
        }
    }

    fn spawn_buttons(&mut self) {
        self.game_buttons.clear();
        let mut rng = rand::thread_rng();
        let num_buttons = rng.gen_range(3..=6);
        let real_index = rng.gen_range(0..num_buttons);

        for i in 0..num_buttons {
            let x = rng.gen_range(0.0..(CANVAS_WIDTH as f32 - BUTTON_WIDTH));
            let y = rng.gen_range(0.0..(CANVAS_HEIGHT as f32 - BUTTON_HEIGHT));
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
            println!("Edit Settings clicked!");
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
pub(crate) fn view(gui: &GuiEnvironment) -> Element<Message> {
    let canvas = Canvas::new(GameCanvas {
        game_buttons: gui.game_buttons.clone(),
        cache: gui.canvas_cache,
    })
    .width(Length::Fixed(CANVAS_WIDTH as f32))
    .height(Length::Fixed(CANVAS_HEIGHT as f32))
    .on_event(|event, _bounds, cursor| match event {
        Event::Mouse(iced::mouse::Event::ButtonPressed(_Left)) => {
            if let Some(cursor_position) = cursor {
                Status::Captured(Message::CanvasClicked(cursor_position))
            } else {
                Status::Ignored
            }
        }
        _ => Status::Ignored,
    });

    let start_stop_label = if gui.collecting {
        "Stop Collecting"
    } else {
        "Start Collecting"
    };
    let hotbar = row![
        button(start_stop_label).on_press(Message::ToggleCollecting),
        button("Edit Settings").on_press(Message::EditSettings)
    ]
    .spacing(20)
    .padding(10)
    .align_y(Alignment::Center);

    let content = column![canvas, hotbar]
        .spacing(10)
        .align_x(Alignment::Center);

    container(content)
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x()
        .center_y()
        .into()
}

/// Custom canvas program for rendering game buttons.
struct GameCanvas {
    game_buttons: Vec<GameButton>,
    cache: Cache,
}

impl<Message, Theme, Renderer> canvas::Program<Message, Theme, Renderer> for GameCanvas
where
    Renderer: geometry::Renderer,
{
    type State = ();

    fn update(
        &self,
        _state: &mut Self::State,
        _event: canvas::Event,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> (Status, Option<Message>) {
        (Status::Ignored, None)
    }

    fn draw(
        &self,
        _state: &Self::State,
        _renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry<Renderer>> {
        let geometry = self
            .cache
            .draw(_renderer, bounds.size(), |frame: &mut Frame| {
                for button in &self.game_buttons {
                    let path = Path::rectangle(
                        Point::new(button.rect.x, button.rect.y),
                        Size::new(button.rect.width, button.rect.height),
                    );
                    frame.fill(&path, Color::from_rgb(0.2, 0.7, 0.3));
                    let text_position = Point::new(
                        button.rect.x + button.rect.width / 4.0,
                        button.rect.y + button.rect.height / 2.0,
                    );
                    frame.fill_text(Text {
                        content: "Button".into(),
                        position: text_position,
                        color: Color::WHITE,
                        size: Pixels::from(20.0),
                        ..Default::default()
                    });
                }
            });
        vec![geometry]
    }

    fn mouse_interaction(
        &self,
        _state: &Self::State,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        mouse::Interaction::default()
    }
}
