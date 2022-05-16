from nightscout_ml_base import NightscoutMlBase
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers


class TFModel(NightscoutMlBase):

    now = datetime.now()
    date_str = "{}-{}-{}_{}-{}".format(now.year, now.month, now.day, now.hour, now.minute)

    def build_tf_regression(self):
        # https://www.tensorflow.org/tutorials/keras/regression#regression_with_a_deep_neural_network_dnn
        df = pd.read_csv('aiSMB_records_adjusted.csv')
        df = self.adjust_smbs_based_on_outcomes(df)
 
        df_gen = pd.read_csv(self.data_folder+'/simple_data_generated.csv')
        df = pd.concat([df, df_gen])

        current_cols = [
                        "hour0_2","hour3_5","hour6_8","hour9_11","hour12_14","hour15_17","hour18_20","hour21_23","weekend",
                        "bg","iob","cob","delta","shortAvgDelta","longAvgDelta",
                        "tdd7Days","tddDaily","tdd24Hrs",
                        "recentSteps5Minutes","recentSteps10Minutes","recentSteps15Minutes","recentSteps30Minutes","recentSteps60Minutes",
                        "maxSMB","maxIob","smbToGive"]
        df = df[current_cols]
        
        train_dataset = df.sample(frac=0.8, random_state=0)
        test_dataset = df.drop(train_dataset.index)

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = train_features.pop('smbToGive')
        test_labels = test_features.pop('smbToGive')

        best_results = 1
        best_model = 1
        best_epochs = 0
        for num_hidden_nodes_l1 in range(0,15):
            for num_hidden_nodes_l2 in range(0,6):
                for num_hidden_nodes_l3 in range(0,6):
                    for num_epochs in range(2, 10):
                        model = self.train_model(train_features, train_labels, num_hidden_nodes_l1, num_hidden_nodes_l2, num_hidden_nodes_l3, num_epochs)
                        results = model.evaluate(test_features, test_labels)
                        if results < best_results:
                            best_model = model
                            best_results = results
                            best_epochs = num_epochs

        
        self.save_model_info(best_model, best_results, best_epochs, current_cols, len(df))

        self.save_model(best_model)



    def train_model(self, train_features, train_labels, num_hidden_nodes_l1, num_hidden_nodes_l2, num_hidden_nodes_l3, num_epochs):
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        linear_model = tf.keras.Sequential()
        linear_model.add(layers.Input(shape=(train_features.shape[1],)))
        linear_model.add(normalizer)
        if num_hidden_nodes_l1 > 0:
            linear_model.add(layers.Dense(units=num_hidden_nodes_l1))
        if num_hidden_nodes_l2 > 0:
            linear_model.add(layers.Dense(units=num_hidden_nodes_l2))
        if num_hidden_nodes_l3 > 0:
            linear_model.add(layers.Dense(units=num_hidden_nodes_l3))

        linear_model.add(layers.Dense(units=1))
        

        linear_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.1),
            loss='mean_absolute_error')

        linear_model.fit(
            train_features,
            train_labels,
            epochs=num_epochs,
            # Suppress logging.
            verbose=0,
            # Calculate validation results on 20% of the training data.
            validation_split = 0.2)
        return linear_model

    def save_model_info(self, linear_model, test_results, num_epochs, current_cols, data_row_count):
        model_info = "\n\n------------\n"
        model_info += f"Model {self.date_str}\n"
        model_info += f"{len(linear_model.layers)} Layers:\n"
        for i in range (len(linear_model.layers)):
            layer = linear_model.layers[i]
            layer_info = f"    - Layer {i}  - {layer.name}"
            layer_info += f" ({layer.units})" if 'dense' in layer.name else ""
            model_info += layer_info + "\n"
            
        model_info += f"Model Loss: {test_results} \n"
        model_info += f"Number of Epochs: {num_epochs} \n"
        model_info += f"Columns ({len(current_cols)}): {current_cols} \n"
        model_info += f"Training Data Size: {data_row_count} \n"
        open('models/tf_linear_model_results.txt', "a").write(model_info)


    def save_model(self, linear_model):
        # https://www.tensorflow.org/tutorials/keras/save_and_load#savedmodel_format
        linear_model.save('models/tf_linear_model_'+self.date_str)

        # https://medium.com/analytics-vidhya/running-ml-models-in-android-using-tensorflow-lite-e549209287f0
        converter = tf.lite.TFLiteConverter.from_keras_model(model=linear_model)
        lite_model = converter.convert()
        open('models/tf_linear_model_'+self.date_str+'.tflite', "wb").write(lite_model)
        open('models/model.tflite', "wb").write(lite_model)


    def adjust_smbs_based_on_outcomes(self, df):
        # df['smbToGive'] = np.where(df['smbToGive'] > .2, df['smbToGive'] - .2, 0)
        return df