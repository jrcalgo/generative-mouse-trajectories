import os
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pynput import keyboard as kb

class Collect_Keyboard_Data:
    def __init__(self, kb_layout="QWERTY"):
        if kb_layout is None:
            raise ValueError("No keyboard layout specified.")

        self.MAX_PROCESSES = 5
        self.running_processes = 0

        self.kb_layout = kb_layout

        self.FEATURES = ['timestamp', 'key', 'state', 'press_duration', 'key_press_delay', 'special_key']
        self.data = pd.DataFrame(columns=self.FEATURES)
        TYPES = {'timestamp': 'datetime64[ns]', 'key': str, 'state': str, 'press_duration': float, 'key_press_delay': float, 'special_key': bool}
        self.data = self.data.astype(TYPES)
        # self.data.set_index('timestamp', inplace=True)
    
    def get_data(self):
        return self.data

    def collect_data(self, activation:bool=True):
        if activation is None:
            raise ValueError("No activation specified.")

        if activation:
            self.collecting = True
        else:
            self.collecting = False
            try:
                return self.listener.stop()
            except:
                pass

        def _on_key_press(key:kb.Key):
            timestamp = datetime.datetime.now()
            key = str(key)
            state = 'press'
            press_duration = 0.00
            key_press_delay = 0.00
            special_key = False

            if self.data.shape[0] > 0:
                previous_timestamp = self.data.index[-1]
                key_press_delay = timestamp - previous_timestamp - self.data['press_duration'].iloc[-1]

            if key is kb.HotKey:
                special_key = True

            new_row = {'timestamp': timestamp, 'key': key, 'state': state, 'press_duration': press_duration, 'key_press_delay': key_press_delay, 'special_key': special_key}
            self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)

        def _on_key_release(key:kb.Key):
            timestamp = datetime.datetime.now()
            key = str(key)
            state = 'release'
            press_duration = 0.00
            key_press_delay = 0.00
            special_key = False

            if self.data.shape[0] > 0:
                previous_timestamp = self.data.index[-1]
                press_duration = timestamp - previous_timestamp
                key_press_delay = timestamp - previous_timestamp - self.data['press_duration'].iloc[-1]

            if key is kb.HotKey:
                special_key = True

            new_row = {'timestamp': timestamp, 'key': key, 'state': state, 'press_duration': press_duration, 'key_press_delay': key_press_delay, 'special_key': special_key}
            self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)

        while self.collecting:
            listener = kb.Listener(on_press=_on_key_press, on_release=_on_key_release)
            listener.start()

            if self.data.shape[0] > 0 and self.data.iloc[-1]['key'] == 'Key.esc':
                self.collecting = False
                listener.join()
                listener.stop()

    def plot_data(self, x_axis, y_axis):
        if x_axis is None or y_axis is None:
            raise ValueError("No axis specified.")
        elif x_axis not in self.data.columns or y_axis not in self.data.columns:
            raise ValueError("Invalid axis specified.")

        

    def save_data(self, remove_stored_examples:bool=False):
        with open(self.kb_layout +'_keyboardLog.csv', 'w', newline='') as file:
            writer = csv.Writer(file, fieldNames=_process_data(self.data))

        if remove_stored_examples:
            self.data = pd.DataFrame(columns=self.FEATURES)


def main():
    keyboard_data = Collect_Keyboard_Data()
    print("Keyboard data collector initialized.....")
    print(type(keyboard_data.data))
    print("Keyboard data collector started.....")
    keyboard_data.collect_data(activation=True)
    # keyboard_data.plot_data(x_axis=keyboard_data.data.index, y_axis=keyboard_data.data['press_duration'], real_time=True)


if __name__ == '__main__':
    main()