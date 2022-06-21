# from get_svgs import GetSgvs
# from label_data import LabelData
# from build_model import BuildModel
# from nightscout_python_client.rest import ApiException
from tf_model import TFModel
from generate_simple_data import GenerateSimpleData
from adjust_smbs import AdjustSmbs
from pull_notes import PullNotes
from lstm_model import LstmModel
import numpy as np
import tensorflow as tf
import pandas as pd


# GenerateSimpleData().generate_data(500)

# start_date_time = '6/17/22 04:22PM'
# notes_file = PullNotes().pull_notes_to_csv(start_date_time)
# PullNotes().add_adjustment_flags('data/aiSMB_records.csv', notes_file)
# AdjustSmbs().adjust_smbs(start_date_time)

# TFModel().build_tf_regression()
# TFModel().compare_two_models('2022-6-11_21-18', '2022-6-12_8-10', '6/12/22 03:10AM')
# TFModel().compare_two_models('2022-6-18_0-31', '2022-6-19_7-28', '6/17/22 10:20AM')
LstmModel().build_lstm_model()

# modelMAE = tf.keras.models.load_model('models/backup/tf_model_2022-6-7_21-23')
# modelMSE = tf.keras.models.load_model('models/backup/tf_model_2022-6-7_13-4')
# bg = 120
# cob = 60
# last_cob_min = 5
# delta = 4
# shortDelta = 1
# longDelta = 1
# stable = 1
# accelerating_up = 0
# deccelerating_up = 0
# accelerating_down = 0
# deccelerating_down = 0
# values = [11,0,0,0, 1,0,0, 0,0,0, 
#                     bg,100,1,cob,last_cob_min,0,delta,shortDelta,longDelta,
#                     accelerating_up, deccelerating_up, accelerating_down, deccelerating_down, stable, 
#                     33,33,1,33, 
#                     0,0,0, 0,0,
#                     0,1]
# maeVal = modelMAE.predict([values])[0][0]
# mseVal = modelMSE.predict([values])[0][0]
# print(f"MAE: {maeVal}  MSE: {mseVal}")

# "hourOfDay","hour0_2","hour3_5","hour6_8", "hour9_11","hour12_14","hour15_17", "hour18_20","hour21_23","weekend",
# "bg","targetBg","iob","cob","lastCarbAgeMin","futureCarbs","delta","shortAvgDelta","longAvgDelta",
# "accelerating_up","deccelerating_up","accelerating_down","deccelerating_down","stable",
# "tdd7Days","tddDaily","tddPerHour","tdd24Hrs",
# "recentSteps5Minutes","recentSteps10Minutes","recentSteps15Minutes", "recentSteps30Minutes","recentSteps60Minutes",
# "sleep","sedintary",


# Notes:
# TODO: build LSTM model
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# https://www.youtube.com/watch?v=ne-dpRdNReI
# https://pythonprogramming.net/cryptocurrency-recurrent-neural-network-deep-learning-python-tensorflow-keras/

# TODO: get exercise/step count data from fit API
## https://github.com/android/fit-samples/blob/main/StepCounterKotlin/app/src/main/java/com/google/android/gms/fit/samples/stepcounterkotlin/MainActivity.kt
## credentials id - 564617406014-8kkm14657o81ancp2okpo9oga0078v2j.apps.googleusercontent.com
# TODO: meal tags
# TODO: extended high, low earlier today
# TODO: sensor age, site age, site placement
# TODO: measure & optimize model for battery/app
    # https://www.tensorflow.org/model_optimization/guide/quantization/post_training
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
# DONE: parameters search - https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
# found that 8-10 epochs works well, number of nodes doesn't make a big difference
# DONE: support time since last meal
# DONE: add target, futureCob, minSinceLastCarb to model/training
# DONE: support temp target - done in AAPS
# DONE: account for future cob - right now doesn't show up in COB - done in AAPS
# DONE: tag low preventions - include as adjustments in records
# DONE: tag low preventions - download
# DONE: move smb adjustments into code rather than excel
# DONE: remove maxSMB/Iob from model
# DONE: add safety preventions in android
# DONE: expand date data
# DONE: add more layers to model
# DONE: figure out & use total IOB
# DONE: get exercise/step count data from phone sensor data
# DONE: get max IOB & max SMB from preferences
# DONE: Record data in app
# DONE: generate some data based on current settings
# DONE: implement safety - max IOB, max Bolus
# DONE: Round pump request to .05
# DONE: Header in csv
# DONE: Load model from file for faster feedback loop


# features to add:
# DONE: cob
# Done - glucose, delta, short delta, long delta, noise
# Partially done - day of week, hour of day, AM/PM, holiday
# targetBG
# Done - ttd 7d, 1d, last 24

# Excel formulas: 
# dynamic ISF - =IF(C2 > 200, 60, IF(C2>150, 80, 100)) * (1-F2/20)
# smb recommended - =C2/V2 + E2/15 - D2 +1
# future high/low =IF(C9>200,"HIGH",IF(C9>150,"high",IF(C9>120,"medium",IF(C9<70,"LOW",IF(C9<80,"low","normal")))))
# future rise/drop - =IF(F9>6,"RISE",IF(F9>3,"rise",IF(F9<-6,"DROP",IF(F9<-3,"drop","stable"))))



# OLD REFERENCES

# Try Logistic regression with TF
# GOOD START: https://www.tensorflow.org/tutorials/keras/regression#regression_with_a_deep_neural_network_dnn
# can build regression further based on docs here

# AAPS 3ef758d4 commit has a bunch of the stuff needed for implementing algorithm in AAPS
# autoML - https://medium.com/analytics-vidhya/6-open-source-automated-machine-learning-tools-every-data-scientist-should-know-49960c1397c9
# https://www.marktechpost.com/2021/04/08/logistic-regression-with-keras/
# https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781789132212/1/ch01lvl1sec15/logistic-regression-with-keras
# https://medium.com/@luwei.io/logistic-regression-with-keras-d75d640d175e

