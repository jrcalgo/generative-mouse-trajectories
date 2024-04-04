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

            if key != kb.Key.backspace:
                self.character_count += 1
            else:
                if self.character_count > 0:
                    self.character_count -= 1

            timestamp = datetime.datetime.now().timestamp() * 1000 # in milliseconds
            key = str(key)
            state = 'press'
            press_duration = 0.00
            key_press_delay = 0.00
            special_key = False
            wpm = (self.character_count / 5) / float(timestamp / 60)

            if self.data.shape[0] > 0:
                previous_timestamp = self.data.index[-1]
                key_press_delay = timestamp + previous_timestamp

            if 'Key.' in key:
                special_key = True

            self.new_press_row = pd.DataFrame([{'timestamp': timestamp, 'key': key, 'state': state, 'press_duration': press_duration, 'key_press_delay': key_press_delay, 'special_key': special_key, 'wpm': wpm}])

        def _on_key_release(key:kb.Key):
            # print('Key released: {0}'.format(key))
            timestamp = datetime.datetime.now().timestamp() * 1000 # in milliseconds
            key = str(key)
            state = 'release'
            press_duration = 0.00
            key_press_delay = 0.00
            special_key = False
            wpm = self.new_press_row['wpm'].iloc[-1]

            if self.data.shape[0] > 0:
                previous_timestamp = self.data.index[-1]
                press_duration = timestamp - previous_timestamp

            if 'Key.' in key:
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
        file_exists = {'existence': os.path.isfile(file)}
        file_exists['mode'] = 'a' if file_exists['existence'] else 'w'
        with open(file, file_exists['mode'], newline='\n' if file_exists['existence'] else '') as file:
            if file_exists['existence'] is False:
                file.write(','.join(self.FEATURES) + '\n')

            self.data.to_csv(file, mode=file_exists['mode'], header=False, index=False)
                
        if save_words:
            file = self.kb_layout + '_wordLog.csv'
            file_exists['existence'] = os.path.isfile(file)
            next_file_num = 1
            while file_exists['existence'] is True:
                file = self.kb_layout + '_wordLog_' + str(next_file_num) + '.csv'
                file_exists['existence'] = os.path.isfile(file)
                next_file_num += 1
            file_exists['mode'] = 'a' if file_exists['existence'] else 'w'
            with open(file, file_exists['mode'], newline='\n' if file_exists['existence'] else '') as file:
                writer = csv.writer(file)
                new_row = []
                for key in self.data['key']:
                    if len(key) == 1:
                        new_row.append(key)
                    elif key == 'Key.space':
                        new_row.append(' ')
                    elif key == 'Key.enter':
                        writer.writerow(np.array2string(np.array(new_row), separator='').replace('[', '').replace(']', ''))
                        new_row = []
            
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
if input("plot data? (y/n): ") == 'y':
    print("Plotting data.....")
    keyboard_data.plot_data('timestamp', 'wpm')
save_words = False
remove_examples = False
save_data: str = input("Save data? (y/n): ")
if save_data == 'y':
    save_data = input("Save words? (y/n): ")
    if save_data == 'y':
        save_words = True
    remove_examples = input("Remove stored examples? (y/n): ")
    if remove_examples == 'y':
        remove_examples = True
    keyboard_data.save_data(save_words, remove_examples)
else:
    print("Data not saved.....")
