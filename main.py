# from get_svgs import GetSgvs
# from label_data import LabelData
# from build_model import BuildModel
# from nightscout_python_client.rest import ApiException
from tf_model import TFModel
from generate_simple_data import GenerateSimpleData
from adjust_smbs import AdjustSmbs
from pull_notes import PullNotes

# GenerateSimpleData().generate_data(500)

# start_date_time = '6/4/22 09:47PM'
# notes_file = PullNotes().pull_notes_to_csv(start_date_time)
# PullNotes().add_adjustment_flags('data/aiSMB_records.csv', notes_file)
# AdjustSmbs().adjust_smbs(start_date_time)

TFModel().build_tf_regression()

# Notes:
# TODO: parameters search - https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
# TODO: get exercise/step count data from fit API
## https://github.com/android/fit-samples/blob/main/StepCounterKotlin/app/src/main/java/com/google/android/gms/fit/samples/stepcounterkotlin/MainActivity.kt
## credentials id - 564617406014-8kkm14657o81ancp2okpo9oga0078v2j.apps.googleusercontent.com
# TODO: meal tags
# TODO: extended high, low earlier today
# TODO: sensor age, site age, site placement
# TODO: measure & optimize model for battery/app
    # https://www.tensorflow.org/model_optimization/guide/quantization/post_training
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
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

