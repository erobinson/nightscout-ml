from sklearn import metrics
from nightscout_ml_base import NightscoutMlBase
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers
from tensorflow.keras.layers import PReLU
import time


class TFModel(NightscoutMlBase):

    now = datetime.now()
    date_str = "{}-{}-{}_{}-{}".format(now.year, now.month, now.day, now.hour, now.minute)

    def build_tf_regression(self):
        # https://www.tensorflow.org/tutorials/keras/regression#regression_with_a_deep_neural_network_dnn
        # df = pd.read_csv('aiSMB_records_adjusted.csv')
        df = pd.read_excel('data.xlsx','training_data')
 
        # df_gen = pd.read_csv(self.data_folder+'/simple_data_generated.csv')
        # df = pd.concat([df, df_gen])
        # df = df_gen

        current_cols = [
                        "hourOfDay","hour0_2","hour3_5","hour6_8","hour9_11","hour12_14","hour15_17","hour18_20","hour21_23","weekend",
                        "bg","targetBg","iob","cob","lastCarbAgeMin","futureCarbs","delta","shortAvgDelta","longAvgDelta",
                        "accelerating_up","deccelerating_up","accelerating_down","deccelerating_down","stable",
                        "tdd7Days","tddDaily","tddPerHour","tdd24Hrs",
                        "recentSteps5Minutes","recentSteps10Minutes","recentSteps15Minutes","recentSteps30Minutes","recentSteps60Minutes",
                        "sleep","sedentary",
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
        best_last_activation = ''
        best_learning_rate = .1
        best_loss_function = ''

        start = time.time()

        # loss_functions = ['mean_squared_error', 'mean_absolute_error']
        # loss_functions = ['mean_absolute_error']
        loss_functions = ['mean_squared_error']
        # last_activation_functions = [None, 'relu', 'prelu']
        # last_activation_functions = ['relu', 'prelu']
        last_activation_functions = ['prelu']
        # learning_rates = [.01, .05, .1, .2]
        # tried .001 -> .2 and .01 seems to consistently work the best
        learning_rates = [.01]

        for dropout_rate_l1 in range(0, 6, 8):
            for num_hidden_nodes_l1 in range(10, 20, 3):
                for dropout_rate_l2 in range(0, 6, 8):
                    for num_hidden_nodes_l2 in range(0, 8, 3):
                        for num_hidden_nodes_l3 in range(0, 8, 3):
                            for num_hidden_nodes_l4 in range(0, 8, 3):
                                for num_epochs in range(10, 11, 3):
                                    for loss_function in loss_functions:
                                        for last_activation in last_activation_functions:
                                            for learning_rate in learning_rates:
                                                model = self.train_model(train_features, train_labels, dropout_rate_l1/10, num_hidden_nodes_l1, dropout_rate_l2/10, num_hidden_nodes_l2, num_hidden_nodes_l3, num_hidden_nodes_l4, num_epochs, last_activation, loss_function, learning_rate)
                                                results = model.evaluate(test_features, test_labels)
                                                meets_min_requirements = self.model_meets_min_requirements(model)
                                                if meets_min_requirements and results[0] < best_loss:
                                                    best_model = model
                                                    best_loss = results[0]
                                                    best_accuracy = results[1]
                                                    best_epochs = num_epochs
                                                    best_last_activation = last_activation
                                                    best_learning_rate = learning_rate
                                                    best_loss_function = loss_function


        # model = self.train_model(train_features, train_labels, 0, 0, 0, 0, 0, 1)
        # results = model.evaluate(test_features, test_labels)
        # best_model = model
        # best_loss = results[0]
        # best_accuracy = results[1]
        # best_epochs = 10

        # for i in range(10):
        #     learning_rate = .01
        #     model = self.train_model(train_features, train_labels, 0, 10, 0, 3, 0, 10, 'relu', 'mean_squared_error', learning_rate)
        #     results = model.evaluate(test_features, test_labels)
        #     meets_min_requirements = self.model_meets_min_requirements(model)
        #     if meets_min_requirements and results[0] < best_loss:
        #         best_model = model
        #         best_loss = results[0]
        #         best_accuracy = results[1]
        #         best_epochs = 10
        #         best_last_activation = 'relu'
        #         best_learning_rate = learning_rate
        #         best_loss_function = 'mean_squared_error'

        training_time = time.time() - start
        
        self.save_model_info(best_model, best_loss, best_accuracy, best_epochs, current_cols, len(df), training_time, best_last_activation, best_learning_rate, best_loss_function)

        self.save_model(best_model)



    def train_model(self, train_features, train_labels, dropout_rate_l1, num_hidden_nodes_l1, dropout_rate_l2, num_hidden_nodes_l2, num_hidden_nodes_l3, num_hidden_nodes_l4, num_epochs, last_activation, loss_function, learning_rate):
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
        if num_hidden_nodes_l4 > 0 and num_hidden_nodes_l3 > 0 and num_hidden_nodes_l2 > 0 and num_hidden_nodes_l1 > 0:
            model.add(layers.Dense(units=num_hidden_nodes_l3, activation="relu"))

        if last_activation == 'prelu':
            prelu = PReLU()
            model.add(layers.Dense(units=1, activation=prelu))
        else:
            model.add(layers.Dense(units=1, activation=last_activation))

        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_function,
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

    def save_model_info(self, model, best_loss, best_accuracy, num_epochs, current_cols, data_row_count, training_time, best_last_activation, best_learning_rate, best_loss_function):
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
        model_info += f"Learning Rate: {best_learning_rate} \n"
        model_info += f"Loss function: {best_loss_function} \n"
        model_info += f"Activation: {best_last_activation} \n" if best_last_activation is not None else "Activation: None\n"
        model_info += self.basic_predictions(model, current_cols) + "\n"
        model_info += f"Took {time.strftime('%H:%M:%S', time.gmtime(training_time))} to train\n"
        model_info += "NOTES: \n"
        open('models/tf_model_results.txt', "a").write(model_info)


    def model_meets_min_requirements(self, model):
        # return True
        high_rising_and_no_iob = self.basic_predict(model, 160, 0, 0, 8)
        low_dropping = self.basic_predict(model, 70, 3, 0, -10)
        return .5 < float(high_rising_and_no_iob) and .024 > float(low_dropping)

    def basic_predictions(self, model, current_cols):
        if len(current_cols) != 36:
            return f"ERROR: incorrect number of columns ({len(current_cols)})"

        low = self.basic_predict(model,50,0.0,0.0,0)
        low_w_iob = self.basic_predict(model,50,1.0,0.0,0)
        normal_w_iob = self.basic_predict(model,100,1.0,0.0,0)
        normal_wo_iob = self.basic_predict(model,100,0.0,0.0,0)
        high_bg = self.basic_predict(model,200,0.0,0.0,0)
        high_cob = self.basic_predict(model,100,1.0,30.0,0)
        high_both = self.basic_predict(model,200,1.0,30.0,0)
        line =  f"    low: {low}    low_w_iob: {low_w_iob}    normal_w_iob: {normal_w_iob}    normal_wo_iob: {normal_wo_iob}\n"
        line += f"    high_bg: {high_bg}    high_cob: {high_cob}    high_both: {high_both}    high_both_and_rising {self.basic_predict(model, 200, 1, 60, 10)}\n"
        line += f"    low_rising  : {self.basic_predict(model, 70, 0, 20, 10)}    normal_rising  : {self.basic_predict(model, 100, 0, 20, 10)}    high_rising  : {self.basic_predict(model, 180, 0, 20, 10)}\n"
        line += f"    low_dropping: {self.basic_predict(model, 70, 2, 0, -7)}    normal_dropping: {self.basic_predict(model, 100, 2, 0, -7)}    high_dropping: {self.basic_predict(model, 180, 2, 0, -7)}\n"
        return line
        
    def basic_predict(self, model, bg, iob, cob, delta):
        last_cob_min = 0 if cob == 0 else 5
        accelerating_up = 1 if delta > 3 else 0
        deccelerating_down = 1 if delta < -3 else 0
        stable= 1 if delta > -3 and delta < 3 else 0
        prediction = model.predict([[11,1,0,0, 0,0,0, 0,0,0, 
                        bg,100,iob,cob,last_cob_min,0,delta,delta,delta,
                        accelerating_up, 0, deccelerating_down, 0, stable, 
                        33,33,1,33, 
                        0,0,0, 0,0,
                        0,1]])
                        # "hourOfDay","hour0_2","hour3_5","hour6_8", "hour9_11","hour12_14","hour15_17", "hour18_20","hour21_23","weekend",
                        # "bg","targetBg","iob","cob","lastCarbAgeMin","futureCarbs","delta","shortAvgDelta","longAvgDelta",
                        # "accelerating_up","deccelerating_up","accelerating_down","deccelerating_down","stable",
                        # "tdd7Days","tddDaily","tddPerHour","tdd24Hrs",
                        # "recentSteps5Minutes","recentSteps10Minutes","recentSteps15Minutes", "recentSteps30Minutes","recentSteps60Minutes",
                        # "sleep","sedintary",
                        
        return str(round(prediction[0][0], 3))
    

    def save_model(self, model):
        # https://www.tensorflow.org/tutorials/keras/save_and_load#savedmodel_format
        model.save('models/backup/tf_model_'+self.date_str)

        # https://medium.com/analytics-vidhya/running-ml-models-in-android-using-tensorflow-lite-e549209287f0
        converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
        lite_model = converter.convert()
        open('models/backup/tf_model_'+self.date_str+'.tflite', "wb").write(lite_model)
        open('models/model.tflite', "wb").write(lite_model)

