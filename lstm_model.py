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
from tensorboard.plugins.hparams import api as hp
from tensorboard import program
import os
import shutil

class LstmModel(NightscoutMlBase):
    current_cols = [
                    "hourOfDay","hour0_2","hour3_5","hour6_8","hour9_11","hour12_14","hour15_17","hour18_20","hour21_23","weekend",
                    "bg","targetBg","iob","cob","lastCarbAgeMin","futureCarbs","delta","shortAvgDelta","longAvgDelta",
                    "accelerating_up","deccelerating_up","accelerating_down","deccelerating_down","stable",
                    "tdd7Days","tdd7DaysPerHour","tddDaily","tddDailyPerHour","tdd24Hrs","tdd24HrsPerHour",
                    "recentSteps5Minutes","recentSteps10Minutes","recentSteps15Minutes","recentSteps30Minutes","recentSteps60Minutes",
                    "sleep","sedentary",
                    "smbToGive"]

    log_folder = 'logs'

    # HP_WINDOW_LEN = hp.HParam('window_len', hp.Discrete([6, 12, 18, 24]))
    HP_WINDOW_LEN = hp.HParam('window_len', hp.Discrete([6]))
    HP_NUM_NODES_L1 = hp.HParam('num_nodes_l1', hp.Discrete([37, 100]))
    # HP_NUM_NODES_L2 = hp.HParam('num_nodes_l2', hp.Discrete([0, 6, 25]))
    HP_NUM_NODES_L2 = hp.HParam('num_nodes_l2', hp.Discrete([0]))
    HP_NUM_NODES_L3 = hp.HParam('num_nodes_l3', hp.Discrete([0, 5, 10]))
    # HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([10, 20, 30])) # 30 was consistently better
    # HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([10, 20, 30, 40])) # 40 was consistently better
    # HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([40, 60, 80])) # 80 consitently the best
    HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([80, 100, 150]))
    # HP_SPLIT_END = hp.HParam('split_end', hp.Discrete([1, 0])) # testing with a random sample of the data was consistently better
    HP_SPLIT_END = hp.HParam('split_end', hp.Discrete([0]))
    METRIC_LOSS = 'loss'

    def build_lstm_model(self):
        self.clear_log_folder()

        df = pd.read_excel('data.xlsx','training_data')
        df = self.normalize(df)

        # print(df.describe().transpose())

        

        # self.load_tensorboard()

        with tf.summary.create_file_writer(f'{self.log_folder}/hparam_tuning').as_default():
            hp.hparams_config(
                hparams=[self.HP_WINDOW_LEN, self.HP_SPLIT_END, self.HP_NUM_NODES_L1, self.HP_NUM_NODES_L2, self.HP_NUM_NODES_L3, self.HP_NUM_EPOCHS],
                metrics=[hp.Metric(self.METRIC_LOSS, display_name='loss')],
            )

        session_num = 0

        for split_end in self.HP_SPLIT_END.domain.values:
            for window_len in self.HP_WINDOW_LEN.domain.values:
                features, labels, \
                    train_features, train_labels, \
                    test_features, test_labels = self.prep_data(df, window_len, split_end)
                for num_nodes_l1 in self.HP_NUM_NODES_L1.domain.values:
                    for num_nodes_l2 in self.HP_NUM_NODES_L2.domain.values:
                        for num_nodes_l3 in self.HP_NUM_NODES_L3.domain.values:
                            for num_epochs in self.HP_NUM_EPOCHS.domain.values:
                                hparams = {
                                    self.HP_WINDOW_LEN: window_len,
                                    self.HP_SPLIT_END: split_end,
                                    self.HP_NUM_NODES_L1: num_nodes_l1,
                                    self.HP_NUM_NODES_L2: num_nodes_l2,
                                    self.HP_NUM_NODES_L3: num_nodes_l3,
                                    self.HP_NUM_EPOCHS: num_epochs
                                }
                                run_name = "run-%d" % session_num
                                print('--- Starting trial: %s' % run_name)
                                print({h.name: hparams[h] for h in hparams})
                                self.run('logs/hparam_tuning/' + run_name, hparams, train_features, train_labels, test_features, test_labels)
                                session_num += 1


        # loss = model.evaluate(test_features, test_labels)
        # print(f"loss: {loss}")
        # line_to_predict = 88
        # prediction = model.predict(np.array([features[line_to_predict]]))[0][0]
        # # print(f"{features[line_to_predict]}")
        # print(f"prediction: {prediction} vs {labels[line_to_predict]}")

    def run(self, run_dir, hparams, train_features, train_labels, test_features, test_labels):
        loss = 1
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            loss = self.train_test_model(hparams, train_features, train_labels, test_features, test_labels)
            tf.summary.scalar(self.METRIC_LOSS, loss, step=1)
        return loss

    def train_test_model(self, hparams, train_features, train_labels, test_features, test_labels):
        model = Sequential()
        model.add(layers.Input(shape=(train_features.shape[1:])))
        return_seq = True if hparams[self.HP_NUM_NODES_L2] > 0 else False
        model.add(LSTM(hparams[self.HP_NUM_NODES_L1], input_shape=(train_features.shape[1:]), return_sequences=return_seq))
        model.add(BatchNormalization())

        if hparams[self.HP_NUM_NODES_L2] > 0:
            model.add(LSTM(hparams[self.HP_NUM_NODES_L2], input_shape=(train_features.shape[1:])))
            model.add(BatchNormalization())

        if hparams[self.HP_NUM_NODES_L3] > 0:
            model.add(Dense(hparams[self.HP_NUM_NODES_L3], activation='relu'))
        
        model.add(Dense(1, activation='relu'))

        model.compile(
            optimizer='adam',
            loss='mean_absolute_error')
        
        model.fit(
            train_features,
            train_labels,
            epochs=hparams[self.HP_NUM_EPOCHS],
            verbose=0,
            # callbacks=[
            #     tf.keras.callbacks.TensorBoard(self.log_folder),  # log metrics
            #     hp.KerasCallback(self.log_folder, hparams),  # log hparams
            # ]
            )

        loss = model.evaluate(test_features, test_labels)
        return loss

    def prep_data(self, df, window_len, split_end):
        features, labels = self.convert_to_frames(df, window_len)

        if split_end != 1:
            features, labels = self.shuffle_in_unison(features, labels)

        split_index = round(len(labels) * .8)
        train_features = features[:split_index]
        test_features = features[split_index:]
        train_labels = labels[:split_index]
        test_labels = labels[split_index:]

        train_features, train_labels = self.shuffle_in_unison(train_features, train_labels)
        test_features, test_labels = self.shuffle_in_unison(test_features, test_labels)

        print(f'TRAIN: {train_features.shape} - {len(train_labels)}')
        print(f'TEST: {test_features.shape} - {len(test_labels)}')

        return features, labels, train_features, train_labels, test_features, test_labels

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


        return np.array(windows), np.array(labels)

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

    def shuffle_in_unison(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(b))
        return a[p], b[p]

    def clear_log_folder(self):
        if os.path.exists(self.log_folder):
            shutil.rmtree(self.log_folder)
        os.mkdir(self.log_folder)

    def load_tensorboard(self):
        tracking_address = self.log_folder # the path of your log file.

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tracking_address])
        url = tb.launch()
        print(f"Tensorflow listening on {url}")