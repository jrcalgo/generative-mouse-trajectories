import os
import csv
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from pynput import keyboard as kb

class Collect_Keyboard_Data:
    def __init__(self, kb_layout="QWERTY"):
        if kb_layout is None:
            raise ValueError("No keyboard layout specified.")

        self.kb_layout:str = kb_layout
        self.character_count:int = 0

        self.FEATURES:list = ['timestamp', 'key', 'state', 'press_duration', 'key_press_delay', 'special_key', 'wpm']
        self.TYPES:dict = {'timestamp': float, 'key': str, 'state': str, 'press_duration': float, 'key_press_delay': float, 'special_key': bool, 'wpm': float}
        self.data:pd = pd.DataFrame(columns=self.FEATURES)
        self.data:pd = self.data.astype(self.TYPES)
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
            # print('Key pressed: {0}'.format(key))
            if key == kb.Key.esc:
                self.collecting = False
                print("Keyboard data collection stopped.....")
                return False

            if key == kb.Key.backspace:
                if self.character_count > 0:
                    self.character_count -= 1
            else:
                self.character_count += 1

            timestamp = datetime.datetime.now().timestamp()
            key = str(key)
            state = 'press'
            press_duration = 0.00
            key_press_delay = 0.00
            special_key = False
            wpm = (self.character_count / 5) / float(timestamp / 60)

            if self.data.shape[0] > 0:
                previous_timestamp = self.data.index[-1]
                key_press_delay = timestamp + previous_timestamp

            if key is kb.HotKey:
                special_key = True

            self.new_press_row = pd.DataFrame([{'timestamp': timestamp, 'key': key, 'state': state, 'press_duration': press_duration, 'key_press_delay': key_press_delay, 'special_key': special_key, 'wpm': wpm}])

        def _on_key_release(key:kb.Key):
            # print('Key released: {0}'.format(key))
            timestamp = datetime.datetime.now().timestamp()
            key = str(key)
            state = 'release'
            press_duration = 0.00
            key_press_delay = 0.00
            special_key = False
            wpm = 0.00

            if self.data.shape[0] > 0:
                previous_timestamp = self.data.index[-1]
                press_duration = timestamp - previous_timestamp

            if key is kb.HotKey:
                special_key = True

            self.new_release_row = pd.DataFrame([{'timestamp': timestamp, 'key': key, 'state': state, 'press_duration': press_duration, 'key_press_delay': key_press_delay, 'special_key': special_key}])
            new_row_set = pd.concat([self.new_press_row, self.new_release_row], ignore_index=True)
            self.data = pd.concat([self.data, new_row_set], ignore_index=True)
            print("New set stored for {0}".format(key))

        if self.collecting:
            listener = kb.Listener(on_press=_on_key_press, on_release=_on_key_release)
            listener.start()
            listener.join()

    def plot_data(self, x_axis, y_axis):
        if x_axis is None or y_axis is None:
            raise ValueError("No axis specified.")
        elif x_axis not in self.data.columns or y_axis not in self.data.columns:
            raise ValueError("Invalid axis specified.")
        
        plt.bar(self.data[x_axis], self.data[y_axis])
        plt.show()
        
    def save_data(self, save_words:bool=False, remove_stored_examples:bool=False):
        file = self.kb_layout +'_keyStrokeLog.csv'
        file_exists = {'existence': os.path.isfile(file), 'mode': 'a' if file_exists['existence'] else 'w', 'newline': '~~~New Session~~~\n' if file_exists['existence'] else ''}
        with open(file, file_exists['mode'], newline=file_exists['newline']) as file:
            writer = csv.Writer(file, fieldNames=self.FEATURES)
            if file_exists['existence'] is False:
                writer.writeheader()
            writer.writerows(self.data)
                
        if save_words:
            file = self.kb_layout + '_wordLog.csv'
            file_exists = {'existence': os.path.isfile(file), 'mode': 'a' if file_exists['existence'] else 'w', 'newline': '~~~New Session~~~\n' if file_exists['existence'] else ''}
            with open(file, file_exists['mode'], newline=file_exists['newline']) as file:
                writer = csv.Writer(file)
                for key in self.data['key']:
                    if key.len() == 1:
                        writer.writecolumn(key)
                    elif key == 'Key.space':
                        writer.writecolumn(' ')
                    elif key == 'Key.enter':
                        writer.writecolumn('\n')
            
        if remove_stored_examples:
            self.data = pd.DataFrame(columns=self.FEATURES)
            self.data = self.data.astype(self.TYPES)
 

keyboard_data = Collect_Keyboard_Data()
print("Keyboard data collector initialized.....")
print(type(keyboard_data.data))
print("Keyboard data collector started.....")
while True:
    keyboard_data.collect_data(True)
    if keyboard_data.collecting is False:
        break
print("Plotting data.....")
keyboard_data.plot_data('key', 'press_duration')
