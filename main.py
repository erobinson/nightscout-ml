import sys
# sys.path.append('../nightscout-python-client')
# from get_svgs import GetSgvs
# from label_data import LabelData
# from build_model import BuildModel
# from nightscout_python_client.rest import ApiException
from simple_model import SimpleModel
from tf_model import TFModel
from generate_simple_data import GenerateSimpleData


# Notes:
# DONE: get max IOB & max SMB from preferences
# TODO: figure out & use total IOB
# TODO: move smb adjustments into code rather than excel
# TODO: use more features when building model
# TODO: add more layers to model
# TODO: get exercise/step count data
# TODO: account for future cob - right now doesn't show up in COB
# TODO: support temp target
# DONE: Record data in app
# DONE: generate some data based on current settings
# DONE: implement safety - max IOB, max Bolus
# DONE: Round pump request to .05
# DONE: Header in csv
# DONE: Load model from file for faster feedback loop

# features to add:
# cob, last meal time & carbs
# Done - glucose, delta, short delta, long delta, noise
# Partially done - day of week, hour of day, AM/PM, holiday
# targetBG
# Done - ttd 7d, 1d, last 24
# Excel formulas: 
# dynamic ISF - =IF(C2 > 200, 60, IF(C2>150, 80, 100)) * (1-F2/20)
# smb recommended - =C2/V2 + E2/15 - D2 +1
# future high/low =IF(C11>200,"HIGH",IF(C11>150,"high",IF(C11>120,"medium",IF(C11<70,"LOW",IF(C11<80,"low","normal")))))
# future rise/drop - =IF(F11>6,"RISE",IF(F11>3,"rise",IF(F11<-6,"DROP",IF(F11<-3,"drop","stable"))))





# GenerateSimpleData().generate_data(200)
TFModel().build_tf_regression()



# try:
    # number_of_days = 5
    # GetSgvs().get_sgvs_day_to_day(number_of_days)
    # GetSgvs().get_all_sgvs_last_x_days(number_of_days)
    # file_name, feature_cols, label_cols = LabelData().label_sgv_data('nightscout_1445_sgvs_starting_2022-1-26.csv')
    # BuildModel().build_model(file_name, feature_cols, label_cols)
    # SimpleModel().build_model()
    # SimpleModel().sklearn_linear_regression_model()
    # SimpleModel().build_tf_regression()
    # TFModel().build_tf_regression()


    
# except ApiException as e:
#     print("Exception when calling EntriesApi->entries_get: %s\n" % e)


# Try Logistic regression with TF
# GOOD START: https://www.tensorflow.org/tutorials/keras/regression#regression_with_a_deep_neural_network_dnn
# can build regression further based on docs here


# 3ef758d4 commit has a bunch of the stuff needed for implementing algorithm in AAPS
# autoML - https://medium.com/analytics-vidhya/6-open-source-automated-machine-learning-tools-every-data-scientist-should-know-49960c1397c9
# https://www.marktechpost.com/2021/04/08/logistic-regression-with-keras/
# https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781789132212/1/ch01lvl1sec15/logistic-regression-with-keras
# https://medium.com/@luwei.io/logistic-regression-with-keras-d75d640d175e

