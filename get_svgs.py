from __future__ import print_function
import nightscout_python_client
import csv
from dateutil import rrule
from datetime import datetime, timedelta
from nightscout_ml_base import NightscoutMlBase

configuration = nightscout_python_client.Configuration()
configuration.host = "erjr-nightscout.herokuapp.com/api/v1"
api_instance = nightscout_python_client.EntriesApi(nightscout_python_client.ApiClient(configuration))

class GetSgvs(NightscoutMlBase):

    def get_sgvs_for_date_range_and_write_to_file(self, start_date, end_date, count):

        api_response = api_instance.entries_spec_get("sgv", findDateLte=end_date.isoformat(), findDateGte=start_date.isoformat(), count=count)
        
        start_date_str = "{}-{}-{}".format(start_date.year, start_date.month, start_date.day)
        sgv_output_file = open('{}/nightscout_{}_sgvs_starting_{}.csv'.format(self.data_folder, count, start_date_str), 'w')
        csv_writer = csv.writer(sgv_output_file)
        csv_writer.writerow(['_id', 'device', 'date', 'dateString', 'isValid', 'sgv', 'direction',  'type', 'created_at'])
        for sgv in api_response:
            csv_writer.writerow(sgv.values())
        sgv_output_file.close()
        
    def get_sgvs_day_to_day(self, number_of_days):
        now = self.get_today_without_time()
        start_date = now - timedelta(days=number_of_days)

        for date in rrule.rrule(rrule.DAILY, dtstart=start_date, until=now):
            end_date = date + timedelta(days=1)
            self.get_sgvs_for_date_range_and_write_to_file(date, end_date, 300)
    
    def get_all_sgvs_last_x_days(self, number_of_days):
        now = self.get_today_without_time()
        start_date = now - timedelta(days=number_of_days)
        count = number_of_days * 289
        self.get_sgvs_for_date_range_and_write_to_file(start_date, now, count)

    def get_today_without_time(self):
        now = datetime.now()
        now = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return now


