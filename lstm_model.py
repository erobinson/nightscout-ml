from nightscout_ml_base import NightscoutMlBase
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers
from tensorflow.keras.layers import PReLU
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint


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

        df = self.normalize(df)

        # print(df.describe().transpose())

        features, labels = self.convert_to_frames(df, 6)

        print(f'{features.shape} - {len(labels)}')
        
        model = Sequential()
        model.add(LSTM(20, input_shape=(features.shape[1:])))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='relu'))

        model.compile(
            optimizer='adam',
            loss='mean_absolute_error')
        
        model.fit(
            features,
            labels,
            epochs=10,
            verbose=1)

        prediction = model.predict(features[0])
        print(f"prediction: {prediction} vs {labels[0]}")


    def convert_to_frames(self, df, window_size):
        windows = []
        labels = []
        for index, row in df.iterrows():
            start_index = index - window_size
            if start_index >= 0:
                window = df[start_index:index]
                first_date = self.str_to_time(window.iloc[0]['dateStr'])
                row_date = self.str_to_time(row['dateStr'])
                if first_date > row_date - pd.DateOffset(minutes=5*(window_size+1)):
                    window = window[self.current_cols]
                    window.pop('smbToGive')
                    windows.append(window.to_numpy())
                    labels.append(row['smbToGive'])
                    # windows.append(window)


        return np.array(windows), labels

    def normalize(self, df):
        df['hourOfDay'] = df['hourOfDay'] / 23
        df['bg'] = df['bg'] / 400
        df['targetBg'] = df['targetBg'] / 400
        df['iob'] = df['iob'] / 20


        df['cob'] = df['cob'] / 200
        df['lastCarbAgeMin'] = df['lastCarbAgeMin'] / 10000
        df['futureCarbs'] = df['futureCarbs'] / 200
        df['delta'] = df['delta'] + 50
        df['delta'] = df['delta'] / 100
        df['shortAvgDelta'] = df['shortAvgDelta'] + 50
        df['shortAvgDelta'] = df['shortAvgDelta'] / 100
        df['longAvgDelta'] = df['longAvgDelta'] + 50
        df['longAvgDelta'] = df['longAvgDelta'] / 100
        df['tdd7Days'] = df['tdd7Days'] / 100
        df['tdd7DaysPerHour'] = df['tdd7DaysPerHour'] / 5
        df['tddDaily'] = df['tddDaily'] / 100
        df['tddDailyPerHour'] = df['tddDailyPerHour'] / 5
        df['tdd24Hrs'] = df['tdd24Hrs'] / 100
        df['tdd24HrsPerHour'] = df['tdd24HrsPerHour'] / 5
        df['recentSteps5Minutes'] = df['recentSteps5Minutes'] / 2000
        df['recentSteps10Minutes'] = df['recentSteps10Minutes'] / 2000
        df['recentSteps15Minutes'] = df['recentSteps15Minutes'] / 2000
        df['recentSteps30Minutes'] = df['recentSteps30Minutes'] / 20000
        df['recentSteps60Minutes'] = df['recentSteps60Minutes'] / 40000
        return df
