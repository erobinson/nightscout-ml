import sys
sys.path.append('../nightscout-python-client')
import nightscout_python_client
import csv
from datetime import datetime, timedelta
from nightscout_ml_base import NightscoutMlBase
import pandas as pd

class PullNotes(NightscoutMlBase):
    configuration = nightscout_python_client.Configuration()
    configuration.host = "erjr-nightscout.herokuapp.com/api/v1"
    treatment_api = nightscout_python_client.TreatmentsApi(nightscout_python_client.ApiClient(configuration))
    output_file_path = 'data/recent_notes.csv'
    timezone_offset_from_zulu = 4

    def pull_notes_to_csv(self, start_date_time):
        count = 10000
        now = datetime.now()
        now = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = now + timedelta(days=1)
        start_date = self.str_to_time(start_date_time)
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        api_response = self.treatment_api.treatments_get("note", createdAtGte=start_date.isoformat(), createdAtLte=end_date.isoformat(), count=count)
        
        sgv_output_file = open(self.output_file_path, 'w')
        csv_writer = csv.writer(sgv_output_file)
        csv_writer.writerow(['dateStr', 'note'])
        for line in api_response:
            if 'notes' in line and ('low' in line['notes'].lower() or 'aggressive' in line['notes'].lower()):
                created_at_dt = datetime.strptime(line['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')
                created_at_dt = created_at_dt - timedelta(hours=self.timezone_offset_from_zulu)
                created_at_str = self.time_to_str(created_at_dt)
                csv_writer.writerow([created_at_str,line['notes']])
        
        return self.output_file_path

    def add_adjustment_flags(self, aismb_records_file_str, notes_file_str):
        notes_pd = pd.read_csv(notes_file_str)
        aismb_df = pd.read_csv(aismb_records_file_str)
        
        if 'low_treatment' not in aismb_df:
            aismb_df['low_treatment'] = 0
        
        if 'more_aggressive' not in aismb_df:
            aismb_df['more_aggressive'] = 0

        for index, row in notes_pd.iterrows():
            note_date = self.str_to_time(row['dateStr'])
            column = 'low_treatment' if 'low' in row['note'].lower() else 'more_aggressive'
            self.apply_adjustment_flags_to_row(aismb_df, note_date, column)

        aismb_df.to_csv(aismb_records_file_str, index=False)
    
    def apply_adjustment_flags_to_row(self, aismb_df, note_date, column):
        # search for multiple lines to find an existing matching timestamp
        for offset in range(10):
            note_date_plus_offset = note_date + timedelta(minutes=offset)
            note_date_str = self.time_to_str(note_date_plus_offset)
            if note_date_str in aismb_df['dateStr'].values:
                aismb_df.loc[aismb_df['dateStr'] == note_date_str, column] = 1
                return