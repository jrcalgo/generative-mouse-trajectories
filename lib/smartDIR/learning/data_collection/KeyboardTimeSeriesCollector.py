import os
import csv
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pynput import keyboard as kb

class Collect_Keyboard_Data:
    def __init__(self, kb_layout="QWERTY"):
        if kb_layout is None:
            raise ValueError("No keyboard layout specified.")
        
        self.kb_layout = kb_layout
        self.features = ['timestamp', 'key', 'state', 'press_duration', 'key_press_delay', 'special_key']
        self.temp_stored_examples = pd.DataFrame(columns=self.features)

    def get_stored_examples(self):
        return self.stored_examples

    def _collect_data(self):
        while (self.collecting):
            

    def _plot_current_datastream(self):
        while (self.plotting):

    def activate_collector(self):
        self.collecting = True
        self.collecting_thread = threading.Thread(target=self._collect_data).start()

    def deactivate_collector(self):
        self.collecting = False
        self.collecting_thread.join()  
        
    def start_live_plotting(self):
        self.plotting = True
        self.plotting_thread = threading.Thread(target=self._plot_current_datastream).start()

    def stop_live_plotting(self):
        self.plotting = False
        self.plotting_thread.join()

    def plot_stored_examples(self):
        plot = plt 

    def save_data(self, remove_stored_examples:bool=False):
        with open(self.kb_layout +'_keyboard.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldNames=self.features)

        if remove_stored_examples:
            self.temp_stored_examples = pd.DataFrame(columns=self.features)
