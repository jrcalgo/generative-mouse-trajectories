import os
import csv
import time
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
        
        self.kb_layout = kb_layout
        
        self.MAX_PROCESSES = 3
        self.running_processes = 0

        self.FEATURES = ['timestamp', 'key', 'state', 'press_duration', 'key_press_delay', 'special_key']
        self.data = pd.DataFrame(columns=self.FEATURES)
        TYPES = {'timestamp': float, 'key': str, 'state': str, 'press_duration': float, 'key_press_delay': float, 'special_key': bool}
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
            timestamp = time.time()
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
            timestamp = time.time()
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
            with kb.Listener(on_press=_on_key_press(kb.Key), on_release=_on_key_release(kb.Key)) as listener:
                listener.join()

    def plot_data(self, real_time:bool=True, x_axis:str='timestamp', y_axis:str='press_duration'):
        def _start_live_plotting(self):
            if self.running_processes < self.MAX_PROCESSES and self.collecting is True:
                self.live_plotting = True
                self.running_processes += 1
                self.plotting_process = multiprocessing.Process(target=self._live_plotting, args=('timestamp', y_axis))
                
        def _live_plotting(self, x_axis, y_axis):
            plt.figure()
            while self.live_plotting:
                plt.clf()
                self.data.plot(x=x_axis, y=y_axis, ax=plt.gca())
                plt.pause(.250)

                if not self.collecting:
                    break
            
            self._stop_live_plotting()

        def _stop_live_plotting(self):
            if self.running_processes > 0:
                try:
                    self.live_plotting = False
                    self.running_processes -= 1
                    self.plotting_process.termiante()
                    self.plotting_process.join()
                except:
                    pass

        if real_time:
            self._start_live_plotting(x_axis, y_axis)
        else:
            self._stop_live_plotting()


    def get_data(self):
        return self.data
    
    def save_data(self, remove_stored_examples:bool=False):
        with open(self.kb_layout +'_keyboard.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldNames=self.FEATURES)

        if remove_stored_examples:
            self.data = pd.DataFrame(columns=self.FEATURES)


keyboard_data = Collect_Keyboard_Data()
keyboard_data.collect_data(activation=True)