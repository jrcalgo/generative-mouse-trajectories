import os
import csv
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pynput import mouse as ms

class Collect_Mouse_Data:
    def __init__(self):
        self.MAX_PROCESSES = 5
        self.running_processes = 0

        self.mouse_dpi = os.system()
        self.os_mouse_sensitivity = os.system()
        self.FEATURES = ['timestamp', 'button', 'x', 'y', 'x_velocity', 'y_velocity'] 
        self.TYPES:dict = {'timestamp': float, 'button': str, 'x': int, 'y': int, 'x_velocity': float, 'y_velocity': float}
        self.data = pd.DataFrame(columns=self.FEATURES)
        self.data = self.data.astype(self.TYPES)

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

    def plot_data(self, x_axis, y_axis, real_time:bool=False)

    def save_data(self):
        with open('mouseLog.csv', 'w', newline='') as file:
            wr