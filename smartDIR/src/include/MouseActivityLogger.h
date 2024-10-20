// MouseActivityLogger.h

#ifndef MOUSE_ACTIVITY_LOGGER_H
#define MOUSE_ACTIVITY_LOGGER_H

#include <SDL2-2.30.8/include/SDL.h>


#define MAX_EVENTS 100000

typedef struct {
    int screen_width;
    int screen_height;
    int window_width;
    int window_height;
    int mouse_dpi;
    char os_device_info;

    // Spatial Features
    int positions[MAX_EVENTS][2];  // positions[n][0] = x, positions[n][1] = y
    double movement_delta[MAX_EVENTS][2];
    double distance_traveled[MAX_EVENTS];
    double path_length;
    double direction_angles[MAX_EVENTS];
    double velocity[MAX_EVENTS];
    double acceleration[MAX_EVENTS];
    double momentum[MAX_EVENTS];
    double curvature[MAX_EVENTS];
    double jerk[MAX_EVENTS];

    // Temporal Features
    double timestamps[MAX_EVENTS]; // timestamps[n]
    double time_between_movements[MAX_EVENTS];
    double total_duration;
    double hover_time[MAX_EVENTS];

    // Behavioral Features
    int click_count;
    struct {
        Uint8 button;
        int x;
        int y;
        double time;
    } click_events[MAX_EVENTS];
    int scroll_count;
    double pause_time;
    int reversal_points[MAX_EVENTS][2];
    int scroll_events[MAX_EVENTS][3];

    // Contextual Features
    int screen_resolution[2];
    int window_position[2];  // x, y of window's top-left corner

    // Derived Features
    double average_velocity;
    double peak_velocity;
    double average_acceleration;
    double peak_acceleration;
    double average_momentum;
    double peak_momentum;
    double smoothness;
    double deviation_from_ideal_path;
    double idle_time;
    double fitts_index_of_difficulty;
    double fitts_movement_time;

    // Internal State
    double start_time;
    int last_position[2];
    double last_time;
    int position_count;

} MouseActivityLogger;

void init_mouse_activity_recorder(MouseActivityLogger *recorder, int screen_width, int screen_height);
void start_recording(MouseActivityLogger *recorder);
void record_motion(MouseActivityLogger *recorder, int x, int y);
void record_click(MouseActivityLogger *recorder, Uint8 button, int x, int y);
void calculate_derived_features(MouseActivityLogger *recorder);
void report(const MouseActivityLogger *recorder);
void log_mouse_activity(const MouseActivityLogger *recorder, const char *filename);

#endif // MOUSE_ACTIVITY_LOGGER_H
