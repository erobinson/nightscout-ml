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




# start_date_time = '7/9/22 10:16PM'
# notes_file = PullNotes().pull_notes_to_csv(start_date_time)
# PullNotes().add_adjustment_flags('data/aiSMB_records.csv', notes_file)
# AdjustSmbs().adjust_smbs(start_date_time)

model_date = TFModel().build_tf_regression()
# TFModel().compare_two_models('2022-6-22_15-32', model_date, 
#     ['6/12/22 03:10AM', '6/17/22 10:20AM', '6/18/22 09:40PM', '6/19/22 04:45AM', '6/27/22 05:25PM', '6/28/22 07:21PM', '7/2/22 02:26AM', '7/6/22 03:11AM', '7/7/22 09:51AM'])
# TFModel().compare_two_models('2022-7-7_10-12', model_date, 
#     ['6/12/22 03:10AM', '6/17/22 10:20AM', '6/18/22 09:40PM', '6/19/22 04:45AM', '6/27/22 05:25PM', '6/28/22 07:21PM', '7/2/22 02:26AM', '7/6/22 03:11AM', '7/7/22 09:51AM'])

# TFModel().compare_two_models('2022-6-11_21-18', '2022-6-12_8-10', '6/12/22 03:10AM')
# TFModel().compare_two_models('2022-6-18_0-31', '2022-6-19_7-28', '6/17/22 10:20AM')

# LstmModel().build_lstm_model() # currently only gets .08 loss - so worse than NN
# tensorboard --logdir logs/




# Notes:
# TODO: Add carb delta, last meal
# TODO: train w/ meal tags
# TODO: update & leverage more TDD values - added 2 days - possibly add recent hours
# TODO: refactor Android code to be cleaner - split factors into methods
# TODO: get exercise/step count data from fit API
## https://github.com/android/fit-samples/blob/main/StepCounterKotlin/app/src/main/java/com/google/android/gms/fit/samples/stepcounterkotlin/MainActivity.kt
## credentials id - 564617406014-8kkm14657o81ancp2okpo9oga0078v2j.apps.googleusercontent.com
# TODO: extended high, low earlier today
# TODO: sensor age, site age, site placement
# TODO: measure & optimize model for battery/app
    # https://www.tensorflow.org/model_optimization/guide/quantization/post_training
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
# TODO: retest LSTM w/ more training


# PullNotes().pull_old_meal_notes() # NS Notes doesn't have carb entry notes from AAPS
# select * from userEntry where note is not null and note != '' and note not like '%Additional Carbs Required%' and source not in ('ConfigBuilder', 'Automation', 'Aaps', 'Food') limit 10
# Notes on AAPS - notes is not null and source in ('CarbsDialog', 'Exercise', 'Insulin Dialog', 'Note', 'Question', 'Wizard Dialog') and note not in ('low treatment', 'more aggressive', 'less aggressive')

# GenerateSimpleData().generate_data(500)

# DONE: Compare current feature preformance
# results:
    # TDD per hour is the best, can probably include other TDDs - major impact
    # TDD - can drop other TDD values - only have normalized to per/hour
    # recent steps - maybe get heart rate for better data - minor impact
    # recent steps - probably fine w/ just steps
    # accellerating does appear to help - could probably refine calcs
    # drop hour breakdowns - no major impact

# DONE: Populate meal tags for past meals - PullNotes().temp_add_tags()
# DONE: log meal tags in Android
# DONE: build LSTM model - Loss was not better than regular NN - but did better w/ more epochs, may be better w/ more data
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# https://www.youtube.com/watch?v=yWkpRdpOiPY
# https://pythonprogramming.net/cryptocurrency-recurrent-neural-network-deep-learning-python-tensorflow-keras/

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

