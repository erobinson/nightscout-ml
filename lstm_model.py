from nightscout_ml_base import NightscoutMlBase
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers
from tensorflow.keras.layers import PReLU
import time

class LstmModel(NightscoutMlBase):
    current_cols = [
                    "hourOfDay","hour0_2","hour3_5","hour6_8","hour9_11","hour12_14","hour15_17","hour18_20","hour21_23","weekend",
                    "bg","targetBg","iob","cob","lastCarbAgeMin","futureCarbs","delta","shortAvgDelta","longAvgDelta",
                    "accelerating_up","deccelerating_up","accelerating_down","deccelerating_down","stable",
                    "tdd7Days","tdd7DaysPerHour","tddDaily","tddDailyPerHour","tdd24Hrs","tdd24HrsPerHour",
                    "recentSteps5Minutes","recentSteps10Minutes","recentSteps15Minutes","recentSteps30Minutes","recentSteps60Minutes",
                    "sleep","sedentary",
                    "smbToGive"]

    def build_lstm_model(self):
        df = pd.read_excel('data.xlsx','training_data')

        # print(df.describe().transpose())

        df = self.convert_to_frames(df, 6)

        print(f'{df.shape}')


    def convert_to_frames(self, df, window_size):
        windows = []
        for index, row in df.iterrows():
            start_index = index - window_size
            if start_index >= 0:
                window = df[start_index:index]
                first_date = self.str_to_time(window.iloc[0]['dateStr'])
                row_date = self.str_to_time(row['dateStr'])
                if first_date > row_date - pd.DateOffset(minutes=5*(window_size+1)):
                    windows.append(window)

        return np.array(windows)
