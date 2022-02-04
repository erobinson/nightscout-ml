from nightscout_ml_base import NightscoutMlBase
import tensorflow as tf
import pandas
import numpy as np
import tensorflow.keras.losses as losses
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split



class BuildModel(NightscoutMlBase):

    def build_model(self, data_file, feature_cols, label_cols):
        print('tensorflow version:', tf.__version__)
        df = pandas.read_csv(self.data_folder+'/'+data_file)

        # shuffle
        df = df.sample(frac=1).reset_index(drop=True)
        
        train, test = train_test_split(df, test_size=0.2)

        label_cols = ['30_min_block_0_above_120', '30_min_block_0_above_160']

        x_train = train[feature_cols].values
        y_train = train[label_cols].values
        x_test = test[feature_cols].values
        y_test = test[label_cols].values

        x_train_tf = tf.convert_to_tensor(x_train)
        y_train_tf = tf.convert_to_tensor(y_train)
        x_test_tf = tf.convert_to_tensor(x_test)
        y_test_tf = tf.convert_to_tensor(y_test)

        model = models.Sequential()
        model.add(layers.Dense(x_train.shape[1], input_dim=x_train.shape[1], kernel_initializer='he_uniform', activation='relu'))
        # model.add(layers.Dense(5))
        model.add(layers.Dense(y_train.shape[1], activation='sigmoid'))
        model.compile(loss=losses.BinaryCrossentropy(), optimizer='adam')

        predictions = model(x_train_tf[:1]).numpy()
        print(predictions)

        model.fit(x_train_tf, y_train_tf, epochs=5)

        # model.summary()

        score = model.evaluate(x_test_tf,  y_test_tf, verbose=5)
        print(score)
        predictions = model(x_test_tf[100:105])
        print(x_test_tf[100:105])
        print(y_test_tf[100:105])
        print(predictions)