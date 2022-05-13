from nightscout_ml_base import NightscoutMlBase
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers


class TFModel(NightscoutMlBase):
    def build_tf_regression(self):
        # https://www.tensorflow.org/tutorials/keras/regression#regression_with_a_deep_neural_network_dnn
        df_base = pd.read_csv(self.data_folder+'/simple_data_generated.csv')
        df = pd.read_csv(self.data_folder+'/aiSMB_records_adjusted.csv')
        df = self.adjust_smbs_based_on_outcomes(df)

        df = pd.concat([df, df_base])
        current_cols = ["bg", "iob", "cob", "smbToGive"]
        df = df[current_cols]
        
        train_dataset = df.sample(frac=0.8, random_state=0)
        test_dataset = df.drop(train_dataset.index)

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = train_features.pop('smbToGive')
        test_labels = test_features.pop('smbToGive')

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        linear_model = tf.keras.Sequential([
            layers.Input(shape=(3,)),
            normalizer,
            layers.Dense(units=1)
        ])

        linear_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.1),
            loss='mean_absolute_error')

        linear_model.fit(
            train_features,
            train_labels,
            epochs=10,
            # Suppress logging.
            verbose=0,
            # Calculate validation results on 20% of the training data.
            validation_split = 0.2)

        config = linear_model.get_config() # Returns pretty much every information about your model
        print(config["layers"][0]["config"]["batch_input_shape"])

        test_results = linear_model.evaluate(
            test_features, test_labels, verbose=0)
        print(test_results)

        self.run_predictions(linear_model, "TF Linear Regression")

        # https://www.tensorflow.org/tutorials/keras/save_and_load#savedmodel_format
        now = datetime.now()
        date_str = "{}-{}-{}_{}-{}".format(now.year, now.month, now.day, now.hour, now.minute)
        linear_model.save('models/tf_linear_model_'+date_str)

        # https://medium.com/analytics-vidhya/running-ml-models-in-android-using-tensorflow-lite-e549209287f0
        converter = tf.lite.TFLiteConverter.from_keras_model(model=linear_model)
        lite_model = converter.convert()
        open('models/tf_linear_model_'+date_str+'.tflite', "wb").write(lite_model)

    def adjust_smbs_based_on_outcomes(self, df):
        # df['smbToGive'] = np.where(df['smbToGive'] > .2, df['smbToGive'] - .2, 0)
        return df

    def run_predictions(self, model, model_description):
        print("\n == "+model_description+" ==")
        low = model.predict([[50.0,0.0,0.0]])
        low_w_iob = model.predict([[50.0,1.0,0.0]])
        normal_w_iob = model.predict([[100.0,1.0,0.0]])
        normal_wo_iob = model.predict([[100.0,0.0,0.0]])
        print(f"    low: {low}    low_w_iob: {low_w_iob}    normal_w_iob: {normal_w_iob}    normal_wo_iob: {normal_wo_iob}")
        high_bg = model.predict([[200.0,0.0,0.0]])
        high_cob = model.predict([[100.0,1.0,30.0]])
        high_both = model.predict([[200.0,1.0,30.0]])
        print(f"    high_bg: {high_bg}    high_cob: {high_cob}    high_both: {high_both}")