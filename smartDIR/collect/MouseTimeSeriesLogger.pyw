import os
import csv
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pynput import mouse as ms

class Collect_Mouse_Data:
    def __init__(self, driver:str="logitech", os_dpi:int=1, driver_dpi:int=1500, os_acceleration:int=0, driver_acceleration:int=0,):
        self.MAX_PROCESSES = 5
        self.running_processes = 0

        self.PREPROCESSED_FEATURES = ['timestamp','x', 'y', 'x_velocity', 'y_velocity', 'button', 'press_duration'] 
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
            
        def _on_move(x:int, y:int):
            
        def _on_scroll(x:int, y:int, dx:int, dy:int):
        
        def _on_click(button:str, click_duration:float):

    def plot_data(self, x_axis, y_axis, real_time:bool=False)

    def process_data(self):
        self.processed_data 

    def save_data(self):
        with open('mouseLog.csv', 'w', newline='') as file:
            
            
            