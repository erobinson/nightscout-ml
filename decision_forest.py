from sklearn import metrics
from nightscout_ml_base import NightscoutMlBase
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers
from tensorflow.keras.layers import PReLU
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

class DecisionForest(NightscoutMlBase):

    now = datetime.now()
    date_str = "{}-{}-{}_{}-{}".format(now.year, now.month, now.day, now.hour, now.minute)

    def build_model(self):
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

        #Create a Gaussian Classifier
        clf=RandomForestRegressor(n_estimators=100, verbose=1)

        #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(train_features,train_labels)

        predictions=clf.predict(test_features)
        errors = abs(predictions - test_labels)

        # Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
        # Calculate mean absolute percentage error (MAPE)
        mape = 100 * (errors / test_labels)
        # Calculate and display accuracy
        accuracy = 100 - np.mean(mape)
        print('Accuracy:', round(accuracy, 2), '%.')

        predict_features = test_features[15:20]
        predict_labels = test_labels[15:20]
        simple_predictions = clf.predict(predict_features)
        print(simple_predictions)
        print(predict_labels)

        # print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))