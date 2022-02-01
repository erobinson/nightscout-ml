from __future__ import print_function
import sys
sys.path.append('nightscout-python-client')

import nightscout_python_client
from nightscout_python_client.models.entries import Entries  # noqa: E501
from nightscout_python_client.rest import ApiException

import time
from nightscout_python_client.rest import ApiException
from pprint import pprint
import json
import csv
from dateutil import rrule
from datetime import datetime, timedelta

# Configure API key authorization: api_secret
configuration = nightscout_python_client.Configuration()
configuration.host = "erjr-nightscout.herokuapp.com/api/v1"
# configuration.api_key['api_secret'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['api_secret'] = 'Bearer'
# Configure API key authorization: token_in_url
# configuration = nightscout_python_client.Configuration()
# configuration.api_key['token'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['token'] = 'Bearer'

# create an instance of the API class
api_instance = nightscout_python_client.EntriesApi(nightscout_python_client.ApiClient(configuration))
find = 'find_example' # str | The query used to find entries, support nested query syntax, for example `find[dateString][$gte]=2015-08-27`.  All find parameters are interpreted as strings. (optional)
count = 1.2 # float | Number of entries to return. (optional)
folder = 'data'

def get_sgvs_for_day_and_write_to_file(start_date, end_date):

    api_response = api_instance.entries_spec_get("sgv", findDateLte=end_date.isoformat(), findDateGte=start_date.isoformat(), count=count)
    
    start_date_str = "{}-{}-{}".format(start_date.year, start_date.month, start_date.day)
    sgv_output_file = open('{}/nightscout_{}_svgs_starting_{}.csv'.format(folder, count, start_date_str), 'w')
    csv_writer = csv.writer(sgv_output_file)
    csv_writer.writerow(['_id', 'device', 'date', 'dateString', 'isValid', 'sgv', 'direction',  'type', 'created_at'])
    for sgv in api_response:
        csv_writer.writerow(sgv.values())
    sgv_output_file.close()
    

try:
    # All Entries matching query
    # api_response = api_instance.entries_get(find=find, count=count)
    # spec = {}
    # spec['type'] = 'sgv'
    # spec['find[date][$gte]'] = "2022-01-30T05:00:00.000Z"
    # spec['count'] = 100
    # api_response = api_instance.entries_spec_get(spec)
    # api_response = api_instance.entries_get(spec)

    # now = datetime.now()
    # start_date = now - timedelta(days=90)
    # api_response = api_instance.entries_spec_get("sgv", findDateGte='1642482000000', findDateLte='1642568400000', count="100")
    # api_response = api_instance.entries_spec_get("sgv", findDateGte="2021-01-01", count="100")
    # api_response = api_instance.entries_spec_get("sgv", findDateGte="2021-11-01", count="30000")
    # api_response = api_instance.entries_get(find="find[date][$gte]=1643518800000", count=100.0)
    # pprint(api_response)
    # start_date = "2021-11-01"

    count = 300
    number_of_days = 5
    now = datetime.now()
    now = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = now - timedelta(days=number_of_days)
    for dt in rrule.rrule(rrule.DAILY, dtstart=start_date, until=now):
        end_date = dt + timedelta(days=1)
        get_sgvs_for_day_and_write_to_file(dt, end_date)
    
except ApiException as e:
    print("Exception when calling EntriesApi->entries_get: %s\n" % e)

