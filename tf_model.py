from sklearn import metrics
from nightscout_ml_base import NightscoutMlBase
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers
import time


class TFModel(NightscoutMlBase):

    now = datetime.now()
    date_str = "{}-{}-{}_{}-{}".format(now.year, now.month, now.day, now.hour, now.minute)

    def build_tf_regression(self):
        # https://www.tensorflow.org/tutorials/keras/regression#regression_with_a_deep_neural_network_dnn
        df = pd.read_csv('aiSMB_records_adjusted.csv')
 
        # df_gen = pd.read_csv(self.data_folder+'/simple_data_generated.csv')
        # df = pd.concat([df, df_gen])
        # df = df_gen

        current_cols = [
                        "hour0_2","hour3_5","hour6_8","hour9_11","hour12_14","hour15_17","hour18_20","hour21_23","weekend",
                        "bg","iob","cob","delta","shortAvgDelta","longAvgDelta",
                        "tdd7Days","tddDaily","tdd24Hrs",
                        "recentSteps5Minutes","recentSteps10Minutes","recentSteps15Minutes","recentSteps30Minutes","recentSteps60Minutes",
                        "smbToGive"]
        df = df[current_cols]
        
        train_dataset = df.sample(frac=0.8, random_state=0)
        test_dataset = df.drop(train_dataset.index)

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = train_features.pop('smbToGive')
        test_labels = test_features.pop('smbToGive')

        best_loss = 1
        best_accuracy = 0
        best_model = 1
        best_epochs = 0


        start = time.time()

        for dropout_rate_l1 in range(0, 6, 4):
            for num_hidden_nodes_l1 in range(0, 12, 4):
                for dropout_rate_l2 in range(0, 6, 2):
                    for num_hidden_nodes_l2 in range(0, 6, 2):
                        for num_hidden_nodes_l3 in range(0, 3, 2):
                            # for num_epochs in range(0, 30, 5):
                                num_epochs = 10
                                model = self.train_model(train_features, train_labels, dropout_rate_l1/10, num_hidden_nodes_l1, dropout_rate_l2/10, num_hidden_nodes_l2, num_hidden_nodes_l3, num_epochs)
                                results = model.evaluate(test_features, test_labels)
                                if results[0] < best_loss:
                                    best_model = model
                                    best_loss = results[0]
                                    best_accuracy = results[1]
                                    best_epochs = num_epochs


        # model = self.train_model(train_features, train_labels, 0, 6, .4, 4, 2, 10)
        # results = model.evaluate(test_features, test_labels)
        # if results < best_results:
        #     best_model = model
        #     best_results = results
        #     best_epochs = 10

        training_time = time.time() - start
        
        self.save_model_info(best_model, best_loss, best_accuracy, best_epochs, current_cols, len(df), training_time)

        self.save_model(best_model)



    def train_model(self, train_features, train_labels, dropout_rate_l1, num_hidden_nodes_l1, dropout_rate_l2, num_hidden_nodes_l2, num_hidden_nodes_l3, num_epochs):
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(train_features.shape[1],)))
        model.add(normalizer)
        if dropout_rate_l1 > 0:
            model.add(layers.Dropout(dropout_rate_l1))
        if num_hidden_nodes_l1 > 0:
            model.add(layers.Dense(units=num_hidden_nodes_l1, activation="relu"))
        if dropout_rate_l2 > 0:
            model.add(layers.Dropout(dropout_rate_l2))
        if num_hidden_nodes_l2 > 0 and num_hidden_nodes_l1 > 0:
            model.add(layers.Dense(units=num_hidden_nodes_l2, activation="relu"))
        if num_hidden_nodes_l3 > 0 and num_hidden_nodes_l2 > 0 and num_hidden_nodes_l1 > 0:
            model.add(layers.Dense(units=num_hidden_nodes_l3, activation="relu"))

        model.add(layers.Dense(units=1, activation='relu'))
        

        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.1),
            loss='mean_squared_error',
            metrics=["accuracy"])

        model.fit(
            train_features,
            train_labels,
            epochs=num_epochs,
            # Suppress logging.
            verbose=0,
            # Calculate validation results on 20% of the training data.
            validation_split = 0.2)
        return model

    def save_model_info(self, model, best_loss, best_accuracy, num_epochs, current_cols, data_row_count, training_time):
        model_info = "\n\n------------\n"
        model_info += f"Model {self.date_str}\n"
        model_info += f"{len(model.layers)} Layers:\n"
        for i in range (len(model.layers)):
            layer = model.layers[i]
            layer_info = f"    - Layer {i}  - {layer.name}"
            layer_info += f" ({layer.units})" if 'dense' in layer.name else ""
            layer_info += f" ({layer.rate})" if 'dropout' in layer.name else ""
            model_info += layer_info + "\n"

        model_info += f"Model Loss & Accuracy: {best_loss} - {best_accuracy} \n"
        model_info += f"Number of Epochs: {num_epochs} \n"
        model_info += f"Columns ({len(current_cols)}): {current_cols} \n"
        model_info += f"Training Data Size: {data_row_count} \n"
        model_info += self.basic_predictions(model, current_cols) + "\n"
        model_info += f"Took {time.strftime('%H:%M:%S', time.gmtime(training_time))} to train\n"
        model_info += "NOTES: \n"
        open('models/tf_model_results.txt', "a").write(model_info)

    def basic_predictions(self, model, current_cols):
        if len(current_cols) != 24:
            return f"ERROR: incorrect number of columns ({len(current_cols)})"

        low = self.basic_predict(model,50.0,0.0,0.0,0)
        low_w_iob = self.basic_predict(model,50.0,1.0,0.0,0)
        normal_w_iob = self.basic_predict(model,100.0,1.0,0.0,0)
        normal_wo_iob = self.basic_predict(model,100.0,0.0,0.0,0)
        high_bg = self.basic_predict(model,200.0,0.0,0.0,0)
        high_cob = self.basic_predict(model,100.0,1.0,30.0,0)
        high_both = self.basic_predict(model,200.0,1.0,30.0,0)
        line =  f"    low: {low}    low_w_iob: {low_w_iob}    normal_w_iob: {normal_w_iob}    normal_wo_iob: {normal_wo_iob}\n"
        line += f"    high_bg: {high_bg}    high_cob: {high_cob}    high_both: {high_both}\n"
        line += f"    low_rising  : {self.basic_predict(model, 70, 0, 20, 10)}    normal_rising  : {self.basic_predict(model, 100, 0, 20, 10)}    high_rising  : {self.basic_predict(model, 180, 0, 20, 10)}\n"
        line += f"    low_dropping: {self.basic_predict(model, 70, 0, 20, -7)}    normal_dropping: {self.basic_predict(model, 100, 0, 20, -7)}    high_dropping: {self.basic_predict(model, 180, 0, 20, -7)}\n"
        return line
        
    def basic_predict(self, model, bg, iob, cob, delta):
        prediction = model.predict([0,0,0, 0,0,0, 0,0,0, \
                        bg,iob,cob, delta,delta,delta, \
                        40,40,40, \
                        0,0,0, 0,0])
                        # "hour0_2","hour3_5","hour6_8", "hour9_11","hour12_14","hour15_17", "hour18_20","hour21_23","weekend",
                        # "bg","iob","cob", "delta","shortAvgDelta","longAvgDelta",
                        # "tdd7Days","tddDaily","tdd24Hrs",
                        # "recentSteps5Minutes","recentSteps10Minutes","recentSteps15Minutes","recentSteps30Minutes","recentSteps60Minutes",
                        
        return str(round(prediction[0][0], 3))
    

    def save_model(self, model):
        # https://www.tensorflow.org/tutorials/keras/save_and_load#savedmodel_format
        model.save('models/backup/tf_model_'+self.date_str)

        # https://medium.com/analytics-vidhya/running-ml-models-in-android-using-tensorflow-lite-e549209287f0
        converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
        lite_model = converter.convert()
        open('models/backup/tf_model_'+self.date_str+'.tflite', "wb").write(lite_model)
        open('models/model.tflite', "wb").write(lite_model)

