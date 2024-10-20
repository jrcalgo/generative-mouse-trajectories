import sdl2
import sdl2.ext
import math
import time
import csv

FEATURES = {
    "Spatial Features": [
        "x_y_coordinates",           # Position of the mouse on the screen
        "movement_delta",            # Change in position from previous point (Δx, Δy)
        "distance_traveled",         # Euclidean distance between consecutive points
        "path_length",               # Cumulative distance mouse pointer has traveled
        "direction_angle",           # Angle of movement between two consecutive points
        "velocity",                  # Speed of the mouse movement
        "acceleration",              # Rate of change in velocity
        "curvature",                 # Smoothness/sharpness of movement
        "jerk"                       # Rate of change in acceleration
    ],
    "Temporal Features": [
        "timestamps",                # Time of each recorded position
        "time_between_movements",    # Time difference between consecutive points (Δt)
        "total_duration",            # Total time from start to end of movement
        "hover_time"                 # Time spent stationary at a specific point
    ],
    "Behavioral Features": [
        "click_events",              # Click time, position, type (single/double/right-click), duration
        "scroll_events",             # Scroll distance, direction (up/down, left/right), velocity
        "pause_time",                # Time the mouse stays stationary between movements
        "reversal_points",           # Points where the mouse movement changes direction
        "target_interactions"        # Mouse-over time, distance from interactive targets
    ],
    "Contextual Features": [
        "screen_resolution",         # Resolution of the screen during tracking
        "window_size_and_position",  # Size and position of the active window
        "mouse_dpi_settings",        # Sensitivity settings of the mouse
        "os_and_device_info"         # Details about the system and device used
    ]
}

DERIVED_FEATURES = [
    "average_speed",                 # Average velocity over the movement session
    "peak_speed",                    # Maximum speed during movement
    "smoothness",                    # Consistency of movement in terms of speed and direction
    "deviation_from_ideal_path",     # Deviation from a straight-line path
    "idle_time",                     # Total time spent stationary
    "fitts_law_parameters"           # Fitts's law-based features: Index of Difficulty (ID), Movement Time (MT)
]


class MouseActivityRecorder:
    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width  # Screen width
        self.screen_height = screen_height  # Screen height

        self.positions = []  # To store (x, y) positions of the mouse
        self.timestamps = []  # To store timestamps of each recorded position
        self.distances = []  # To store distance traveled between points
        self.speeds = []  # To store calculated speeds
        self.click_events = []  # To log click events

        self.total_path_length = 0  # Total path length
        self.total_duration = 0  # Total time spent moving
        self.start_time = None  # Track start time of the session
        self.last_position = None  # Track last mouse position
        self.last_time = None  # Track last time for delta calculations

    def start_recording(self):
        sdl2.ext.init()
        window = sdl2.ext.Window("Mouse Activity Recorder", size=(self.screen_width, self.screen_height))
        window.show()

        self.start_time = time.time()  # Record the start time
        running = True
        while running:
            events = sdl2.ext.get_events()
            for event in events:
                if event.type == sdl2.SDL_QUIT:
                    running = False
                    break
                elif event.type == sdl2.SDL_MOUSEMOTION:
                    self.record_motion(event.motion.x, event.motion.y)
                elif event.type == sdl2.SDL_MOUSEBUTTONDOWN:
                    self.record_click(event.button.button, event.button.x, event.button.y)
            sdl2.SDL_Delay(16)

        sdl2.ext.quit()

    def record_motion(self, x, y):
        timestamp = time.time()  # Get the current timestamp

        if self.last_position is not None:
            # Calculate distance traveled
            dx = x - self.last_position[0]
            dy = y - self.last_position[1]
            distance = math.sqrt(dx**2 + dy**2)

            # Calculate time difference
            dt = timestamp - self.last_time

            # Calculate speed
            speed = distance / dt if dt > 0 else 0

            # Update totals
            self.total_path_length += distance
            self.total_duration += dt

            # Store calculated values
            self.distances.append(distance)
            self.speeds.append(speed)

        # Log current position and time
        self.positions.append((x, y))
        self.timestamps.append(timestamp)

        # Update last position and time
        self.last_position = (x, y)
        self.last_time = timestamp

    def record_click(self, button, x, y):
        click_info = {
            "button": button,
            "position": (x, y),
            "time": time.time()
        }
        self.click_events.append(click_info)

    def calculate_derived_features(self):
        if len(self.distances) == 0 or len(self.speeds) == 0:
            return {}

        avg_speed = sum(self.speeds) / len(self.speeds)
        peak_speed = max(self.speeds)
        idle_time = self.total_duration - sum(self.distances) / peak_speed if peak_speed > 0 else 0

        return {
            "average_speed": avg_speed,
            "peak_speed": peak_speed,
            "total_path_length": self.total_path_length,
            "total_duration": self.total_duration,
            "idle_time": idle_time
        }

    def report(self):
        print("Mouse Activity Report:")
        print("Total path length:", self.total_path_length)
        print("Total duration:", self.total_duration)
        print("Click events:", len(self.click_events))
        derived = self.calculate_derived_features()
        for feature, value in derived.items():
            print(f"{feature}: {value}")


if __name__ == "__main__":
    recorder = MouseActivityRecorder()
    recorder.start_recording()
    recorder.report()
