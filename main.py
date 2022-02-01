import sys
sys.path.append('../nightscout-python-client')
from get_svgs import GetSgvs
from label_data import LabelData
from nightscout_python_client.rest import ApiException

try:
    # number_of_days = 5
    # GetSgvs().get_sgvs_day_to_day(number_of_days)
    # GetSgvs().get_all_sgvs_last_x_days(number_of_days)
    LabelData().label_sgv_data('nightscout_1445_sgvs_starting_2022-1-26.csv')

    
except ApiException as e:
    print("Exception when calling EntriesApi->entries_get: %s\n" % e)
