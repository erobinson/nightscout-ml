import sys
# sys.path.append('../nightscout-python-client')
# from get_svgs import GetSgvs
# from label_data import LabelData
# from build_model import BuildModel
# from nightscout_python_client.rest import ApiException
from simple_model import SimpleModel


# Notes:

# Try Logistic regression with TF
# GOOD START: https://www.tensorflow.org/tutorials/keras/regression#regression_with_a_deep_neural_network_dnn
# can build regression further based on docs here





try:
    # number_of_days = 5
    # GetSgvs().get_sgvs_day_to_day(number_of_days)
    # GetSgvs().get_all_sgvs_last_x_days(number_of_days)
    # file_name, feature_cols, label_cols = LabelData().label_sgv_data('nightscout_1445_sgvs_starting_2022-1-26.csv')
    # BuildModel().build_model(file_name, feature_cols, label_cols)
    # SimpleModel().build_model()
    # SimpleModel().sklearn_linear_regression_model()
    SimpleModel().build_tf_regression()


    
except ApiException as e:
    print("Exception when calling EntriesApi->entries_get: %s\n" % e)


# 3ef758d4 commit has a bunch of the stuff needed for implementing algorithm in AAPS
# autoML - https://medium.com/analytics-vidhya/6-open-source-automated-machine-learning-tools-every-data-scientist-should-know-49960c1397c9
# https://www.marktechpost.com/2021/04/08/logistic-regression-with-keras/
# https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781789132212/1/ch01lvl1sec15/logistic-regression-with-keras
# https://medium.com/@luwei.io/logistic-regression-with-keras-d75d640d175e

