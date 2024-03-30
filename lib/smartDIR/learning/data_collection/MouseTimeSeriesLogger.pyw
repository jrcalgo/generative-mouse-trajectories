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
        self.os_mouse_sensitivity = os_mouse_sensitivity
        self.features = ['timestamp', 'x', 'y', 'x_velocity', 'y_velocity', 'button'] 
        self.data = pd.DataFrame(columns=self.features)

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

    def plot_data(self, x_axis, y_axis, real_time:bool=False):

    def get_data(self):
        return self.data

    def save_data(self):
        with open('mouseLog.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldNames=self.features)    