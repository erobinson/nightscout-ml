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
        df = df.sample(frac=1).reset_index(drop=True)
        features = df[['sgv']]
        full_fields = ['30_min_block_0_below_50','30_min_block_0_below_60','30_min_block_0_above_120','30_min_block_0_above_140','30_min_block_0_above_160','30_min_block_0_above_180','30_min_block_0_above_200','30_min_block_0_above_220','30_min_block_0_above_240','30_min_block_0_above_260','30_min_block_0_above_280','30_min_block_0_above_300']
        label_cols = ['30_min_block_0_above_180', '30_min_block_0_above_240']
        labels = df[label_cols]
        # labels['temp'] = labels['30_min_block_0_above_180']
        # labels['30_min_block_0_above_180'] = np.where(labels['temp'], 1, 0)
        # labels.drop(columns=['temp'])
        
        print(features.head())
        train_split_size = round(df.shape[0] * .75)
        # tf_train_features = tf.convert_to_tensor(features.iloc[:, :train_split_size])
        # tf_test_features  = tf.convert_to_tensor(features.iloc[:, train_split_size:])

        # tf_train_labels = tf.convert_to_tensor(labels.iloc[:, :train_split_size])
        # tf_test_labels  = tf.convert_to_tensor(labels.iloc[:, train_split_size:])
        limited_df = df[['sgv', '30_min_block_0_above_180', '30_min_block_0_above_240']]
        train, test = train_test_split(limited_df, test_size=0.2)

        X_train = train.drop(columns=label_cols, axis=1).values
        Y_train = train[label_cols].values
        X_test = test.drop(columns=label_cols).values
        Y_test = test[label_cols].values

        X_train_tf = tf.convert_to_tensor(X_train)
        Y_train_tf = tf.convert_to_tensor(Y_train)
        X_test_tf = tf.convert_to_tensor(X_test)
        Y_test_tf = tf.convert_to_tensor(Y_test)


        # model = models.Sequential([
        #     layers.Flatten(input_shape=(X_train.shape[1], 1)),
        #     layers.Dense(128, activation='relu'),
        #     layers.Dropout(0.2),
        #     layers.Dense(Y_train.shape[1])
        # ])

        model = models.Sequential()
        model.add(layers.Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='he_uniform', activation='relu'))
        model.add(layers.Dense(10))
        model.add(layers.Dense(Y_train.shape[1], activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')

        predictions = model(X_train_tf[:1]).numpy()
        print(predictions)

        # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # loss_fn = losses.SparseCategoricalCrossentropy()
        # initial_loss = loss_fn(Y_train_tf[:1], predictions).numpy()
        # print(initial_loss)

        # model.compile(optimizer='adam',
        #       loss=loss_fn,
        #       metrics=['accuracy'])

        model.fit(X_train_tf, Y_train_tf, epochs=2)

        model.summary()

        model.evaluate(X_test_tf,  Y_test_tf, verbose=5)
        result = model.predict([[160]])
        print(result)