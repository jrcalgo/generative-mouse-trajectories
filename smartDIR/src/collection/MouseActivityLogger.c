// MouseActivityLogger.c

#include "../include/MouseActivityLogger.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

void init_mouse_activity_recorder(MouseActivityLogger* recorder, const int screen_width, const int screen_height)
{
    recorder->screen_width = screen_width;
    recorder->screen_height = screen_height;
    recorder->window_width = screen_width;
    recorder->window_height = screen_height;
    recorder->mouse_dpi = 800;
    recorder->os_device_info = '\0';

    recorder->path_length = 0.0;
    memset(recorder->positions, 0, sizeof(recorder->positions));
    memset(recorder->movement_delta, 0, sizeof(recorder->movement_delta));
    memset(recorder->distance_traveled, 0, sizeof(recorder->distance_traveled));
    memset(recorder->direction_angles, 0, sizeof(recorder->direction_angles));
    memset(recorder->velocity, 0, sizeof(recorder->velocity));
    memset(recorder->acceleration, 0, sizeof(recorder->acceleration));
    memset(recorder->momentum, 0, sizeof(recorder->momentum));
    memset(recorder->curvature, 0, sizeof(recorder->curvature));
    memset(recorder->jerk, 0, sizeof(recorder->jerk));

    recorder->total_duration = 0.0;
    memset(recorder->timestamps, 0, sizeof(recorder->timestamps));
    memset(recorder->time_between_movements, 0, sizeof(recorder->time_between_movements));
    memset(recorder->hover_time, 0, sizeof(recorder->hover_time));

    recorder->click_count = 0;
    recorder->scroll_count = 0;
    recorder->pause_time = 0.0;
    memset(recorder->click_events, 0, sizeof(recorder->click_events));
    memset(recorder->reversal_points, 0, sizeof(recorder->reversal_points));
    memset(recorder->scroll_events, 0, sizeof(recorder->scroll_events));

    recorder->screen_resolution[0] = screen_width;
    recorder->screen_resolution[1] = screen_height;
    recorder->window_position[0] = 0;
    recorder->window_position[1] = 0;

    recorder->average_velocity = 0.0;
    recorder->peak_velocity = 0.0;
    recorder->average_acceleration = 0.0;
    recorder->peak_acceleration = 0.0;
    recorder->average_momentum = 0.0;
    recorder->peak_momentum = 0.0;
    recorder->smoothness = 0.0;
    recorder->deviation_from_ideal_path = 0.0;
    recorder->idle_time = 0.0;
    recorder->fitts_index_of_difficulty = 0.0;
    recorder->fitts_movement_time = 0.0;

    recorder->start_time = 0.0;
    recorder->last_position[0] = -1;
    recorder->last_position[1] = -1;
    recorder->last_time = 0.0;
    recorder->position_count = 0;
}

void start_recording(MouseActivityLogger* recorder)
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        printf("SDL_Init Error: %s\n", SDL_GetError());
        return;
    }
    SDL_Window* window = SDL_CreateWindow("Mouse Activity Logger", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          recorder->window_width, recorder->window_height, SDL_WINDOW_SHOWN);
    if (window == NULL)
    {
        printf("SDL_CreateWindow Error: %s\n", SDL_GetError());
        SDL_Quit();
        return;
    }

    recorder->start_time = SDL_GetTicks() / 1000.0;
    recorder->last_time = recorder->start_time;

    int running = 1;
    SDL_Event event;
    while (running)
    {
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                running = 0;
                break;
            }
            else if (event.type == SDL_MOUSEMOTION)
            {
                record_motion(recorder, event.motion.x, event.motion.y);
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN)
            {
                record_click(recorder, event.button.button, event.button.x, event.button.y);
            }
        }
        SDL_Delay(16);
    }
    SDL_DestroyWindow(window);
    SDL_Quit();
    calculate_derived_features(recorder);
}

void record_motion(MouseActivityLogger* recorder, const int x, const int y)
{
    if (recorder->position_count >= MAX_EVENTS)
    {
        printf("Exceeded maximum number of events.\n");
        return;
    }

    const double timestamp = SDL_GetTicks() / 1000.0;
    const int index = recorder->position_count;

    recorder->positions[index][0] = x;
    recorder->positions[index][1] = y;
    recorder->timestamps[index] = timestamp;

    if (recorder->last_position[0] != -1)
    {
        const double dx = x - recorder->last_position[0];
        const double dy = y - recorder->last_position[1];
        recorder->movement_delta[index][0] = dx;
        recorder->movement_delta[index][1] = dy;

        const double distance = sqrt(dx * dx + dy * dy);
        recorder->distance_traveled[index] = distance;
        recorder->path_length += distance;

        const double dt = timestamp - recorder->last_time;
        recorder->time_between_movements[index] = dt;
        recorder->total_duration += dt;

        const double speed = (dt > 0) ? distance / dt : 0.0;
        recorder->velocity[index] = speed;

        if (index > 0)
        {
            const double dv = speed - recorder->velocity[index - 1];
            const double dt_prev = recorder->time_between_movements[index];
            const double accel = (dt_prev > 0) ? dv / dt_prev : 0.0;
            recorder->acceleration[index] = accel;

            if (index > 1)
            {
                const double da = accel - recorder->acceleration[index - 1];
                const double dt_prev_prev = recorder->time_between_movements[index - 1];
                const double jerk = (dt_prev_prev > 0) ? da / dt_prev_prev : 0.0;
                recorder->jerk[index] = jerk;
            }

            const double angle = atan2(dy, dx);
            recorder->direction_angles[index] = angle;

            const double d_angle = angle - recorder->direction_angles[index - 1];
            recorder->curvature[index] = d_angle / distance;
        }
    }
    else
    {
        recorder->movement_delta[index][0] = 0.0;
        recorder->movement_delta[index][1] = 0.0;
        recorder->distance_traveled[index] = 0.0;
        recorder->velocity[index] = 0.0;
        recorder->acceleration[index] = 0.0;
        recorder->jerk[index] = 0.0;
        recorder->direction_angles[index] = 0.0;
        recorder->curvature[index] = 0.0;
    }

    recorder->last_position[0] = x;
    recorder->last_position[1] = y;
    recorder->last_time = timestamp;
    recorder->position_count++;
}

void record_click(MouseActivityLogger* recorder, const Uint8 button, const int x, const int y)
{
    if (recorder->click_count >= MAX_EVENTS)
    {
        printf("Exceeded maximum number of click events.\n");
        return;
    }

    const int index = recorder->click_count;
    recorder->click_events[index].button = button;
    recorder->click_events[index].x = x;
    recorder->click_events[index].y = y;
    recorder->click_events[index].time = SDL_GetTicks() / 1000.0;

    recorder->click_count++;
}

void calculate_derived_features(MouseActivityLogger* recorder)
{
    const int n = recorder->position_count;
    if (n <= 1)
    {
        return;
    }

    double sum_velocity = 0.0;
    double max_velocity = 0.0;
    for (int i = 1; i < n; i++)
    {
        sum_velocity += recorder->velocity[i];
        if (recorder->velocity[i] > max_velocity)
            max_velocity = recorder->velocity[i];
    }
    recorder->average_velocity = sum_velocity / (n - 1);
    recorder->peak_velocity = max_velocity;

    double sum_acceleration = 0.0;
    double max_acceleration = 0.0;
    for (int i = 2; i < n; i++)
    {
        sum_acceleration += recorder->acceleration[i];
        if (fabs(recorder->acceleration[i]) > max_acceleration)
            max_acceleration = fabs(recorder->acceleration[i]);
    }
    recorder->average_acceleration = sum_acceleration / (n - 2);
    recorder->peak_acceleration = max_acceleration;

    double sum_momentum = 0.0;
    double max_momentum = 0.0;
    for (int i = 1; i < n; i++)
    {
        const double momentum = recorder->velocity[i];
        sum_momentum += momentum;
        if (momentum > max_momentum)
            max_momentum = momentum;
    }
    recorder->average_momentum = sum_momentum / (n - 1);
    recorder->peak_momentum = max_momentum;

    double active_time = 0.0;
    for (int i = 1; i < n; i++)
    {
        if (recorder->velocity[i] > 0.0)
            active_time += recorder->time_between_movements[i];
    }
    recorder->idle_time = recorder->total_duration - active_time;

    double sum_jerk = 0.0;
    for (int i = 3; i < n; i++)
    {
        sum_jerk += fabs(recorder->jerk[i]);
    }
    if (n > 3 && sum_jerk != 0)
        recorder->smoothness = 1.0 / (sum_jerk / (n - 3));
    else
        recorder->smoothness = 0.0;

    const double ideal_dx = recorder->positions[n - 1][0] - recorder->positions[0][0];
    const double ideal_dy = recorder->positions[n - 1][1] - recorder->positions[0][1];
    const double ideal_distance = sqrt(ideal_dx * ideal_dx + ideal_dy * ideal_dy);

    recorder->deviation_from_ideal_path = recorder->path_length - ideal_distance;

    double W = 100.0;
    double D = ideal_distance;
    if (W > 0.0)
    {
        recorder->fitts_index_of_difficulty = log2((D / W) + 1);
        recorder->fitts_movement_time = recorder->total_duration;
    }
}

void report(const MouseActivityLogger* recorder)
{
    printf("Mouse Activity Report:\n");
    printf("Total Path Length: %f pixels\n", recorder->path_length);
    printf("Total Duration: %f seconds\n", recorder->total_duration);
    printf("Click Events: %d\n", recorder->click_count);
    printf("Average Velocity: %f pixels/second\n", recorder->average_velocity);
    printf("Peak Velocity: %f pixels/second\n", recorder->peak_velocity);
    printf("Average Acceleration: %f pixels/second^2\n", recorder->average_acceleration);
    printf("Peak Acceleration: %f pixels/second^2\n", recorder->peak_acceleration);
    printf("Average Momentum: %f\n", recorder->average_momentum);
    printf("Peak Momentum: %f\n", recorder->peak_momentum);
    printf("Smoothness: %f\n", recorder->smoothness);
    printf("Deviation from Ideal Path: %f pixels\n", recorder->deviation_from_ideal_path);
    printf("Idle Time: %f seconds\n", recorder->idle_time);
    printf("Fitts's Index of Difficulty: %f\n", recorder->fitts_index_of_difficulty);
    printf("Fitts's Movement Time: %f seconds\n", recorder->fitts_movement_time);
}

void log_mouse_activity(const MouseActivityLogger* recorder, const char* filename)
{
    FILE* file = fopen(filename, "a");
    if (file == NULL)
    {
        printf("Error opening file.\n");
        return;
    }

    fseek(file, 0, SEEK_END);
    if (ftell(file) == 0)
    {
        fprintf(file, "timestamp,x_position,y_position,delta_x,delta_y,distance_traveled,velocity,acceleration,jerk,"
                      "direction_angle,curvature,button_click,click_x,click_y,click_time\n");
    }

    for (int i = 0; i < recorder->position_count; i++)
    {
        const double delta_x = recorder->movement_delta[i][0];
        const double delta_y = recorder->movement_delta[i][1];
        const double distance = recorder->distance_traveled[i];
        const double speed = recorder->velocity[i];
        const double accel = recorder->acceleration[i];
        const double jerk = recorder->jerk[i];
        const double angle = recorder->direction_angles[i];
        const double curvature = recorder->curvature[i];

        Uint8 button = 0;
        int click_x = -1, click_y = -1;
        double click_time = 0.0;
        for (int j = 0; j < recorder->click_count; j++)
        {
            if (fabs(recorder->click_events[j].time - recorder->timestamps[i]) < 0.01)
            {
                button = recorder->click_events[j].button;
                click_x = recorder->click_events[j].x;
                click_y = recorder->click_events[j].y;
                click_time = recorder->click_events[j].time;
                break;
            }
        }

        fprintf(file, "%f,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%u,%d,%d,%f\n",
                recorder->timestamps[i],
                recorder->positions[i][0], recorder->positions[i][1],
                delta_x, delta_y, distance, speed, accel, jerk,
                angle, curvature, button, click_x, click_y, click_time);
    }

    fclose(file);
    printf("Mouse activity successfully logged to %s\n", filename);
}
