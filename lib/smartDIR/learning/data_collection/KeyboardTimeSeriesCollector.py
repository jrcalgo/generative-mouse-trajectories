import os
import csv
import datetime
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pynput import keyboard as kb

class Collect_Keyboard_Data:
    def __init__(self, kb_layout="QWERTY"):
        if kb_layout is None:
            raise ValueError("No keyboard layout specified.")

        self.MAX_PROCESSES = 3
        self.running_processes = 0

        self.kb_layout = kb_layout

        self.PREPROCESSED_FEATURES = ['timestamp', 'key', 'state', 'press_duration', 'key_press_delay', 'special_key']
        self.data = pd.DataFrame(columns=self.PREPROCESSED_FEATURES)
        TYPES = {'timestamp': 'datetime64[ns]', 'key': str, 'state': str, 'press_duration': float, 'key_press_delay': float, 'special_key': bool}
        self.data = self.data.astype(TYPES)
        self.data = self.data.set_index('timestamp')

    
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
            press_duration = 0.0
            key_press_delay = 0.0
            special_key = False

            if self.data.shape[0] > 0:
                previous_timestamp = self.data.index[-1]
                key_press_delay = timestamp - previous_timestamp - self.data['press_duration'].iloc[-1]

            if key is kb.HotKey:
                special_key = True

            self.data.loc[timestamp] = [key, state, press_duration, key_press_delay, special_key]

        def _on_key_release(key:kb.Key):
            timestamp = datetime.datetime.now()
            key = str(key)
            state = 'release'
            press_duration = 0.0
            key_press_delay = 0.0
            special_key = False

            if self.data.shape[0] > 0:
                previous_timestamp = self.data.index[-1]
                press_duration = timestamp - previous_timestamp
                key_press_delay = timestamp - previous_timestamp - self.data['press_duration'].iloc[-1]

            if key is kb.HotKey:
                special_key = True

            self.data.loc[timestamp] = [key, state, press_duration, key_press_delay, special_key]

            if key == kb.Key.esc:
                self.collecting = False
                return self.listener.stop()

        if self.collecting:
            self.listener = kb.Listener(on_press=_on_key_press, on_release=_on_key_release)
            self.listener.start()

    def plot_data(self, real_time:bool=True, x_axis:str='timestamp', y_axis:str='press_duration'):
        def _live_plotting(x_axis, y_axis):
            plt.figure()
            while self.live_plotting:
                plt.clf()
                self.data.plot(x=x_axis, y=y_axis, ax=plt.gca())
                plt.pause(.100)

                if not self.collecting:
                    break
            
            _stop_live_plotting()

        def _start_live_plotting(x_axis, y_axis):
            if self.running_processes < self.MAX_PROCESSES and self.collecting is True:
                try:
                    self.plotting_process = multiprocessing.Process(target=_live_plotting(x_axis, y_axis), args=('timestamp', y_axis))
                    self.live_plotting = True
                    self.running_processes += 1
                except:
                    print("Cannot start live plotting.")
                    pass
            else:
                print("Either no processes running or no live data to plot.")
                pass
                

        def _stop_live_plotting():
            if self.running_processes > 0:
                try:
                    self.live_plotting = False
                    self.running_processes -= 1
                    self.plotting_process.terminate()
                    self.plotting_process.join()
                except:
                    print("Cannot stop live plotting.")
                    pass
            else:
                print("No live plotting processes running.")
                pass

        if real_time:
            _start_live_plotting(x_axis, y_axis)
        else:
            _stop_live_plotting()


    def get_data(self):
        return self.data
    
    def save_data(self, remove_stored_examples:bool=False):
        def _process_data(preprocessed_data:pd.DataFrame):
            processed_data = pd.DataFrame(columns=self.PREPROCESSED_FEATURES)

            return processed_data

        with open(self.kb_layout +'_keyboard.csv', 'w', newline='') as file:
            writer = csv.Writer(file, fieldNames=_process_data(self.data))

        if remove_stored_examples:
            self.data = pd.DataFrame(columns=self.PREPROCESSED_FEATURES)


keyboard_data = Collect_Keyboard_Data()
print("Keyboard data collector initialized.")
keyboard_data.collect_data(activation=True)
print("Keyboard data collector started.....")
keyboard_data.plot_data(real_time=True, x_axis='timestamp', y_axis='key_press_delay')