import os
import csv
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pynput import mouse as ms

class Collect_Mouse_Data:
    def __init__(self, mouse_dpi, os_mouse_sensitivity):
        self.mouse_dpi = mouse_dpi
        self.os_mouse_sensitivity = os_mouse_sensitivity
        self.features = ['timestamp', 'x', 'y', 'x_velocity', 'y_velocity', 'button'] 
        self.stored_data = pd.DataFrame(columns=self.features)

    def activate_collector(self):

    def deactivate_collector(self, save:bool=False):

    def save_data(self):
        with open('mouse.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldNames=self.features)

    def get_current_data(self):

    def plot_current_data(self):
        
    