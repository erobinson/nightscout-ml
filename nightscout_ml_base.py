from datetime import datetime
import os
import shutil

class NightscoutMlBase():
    data_folder = 'data'
    date_format_str = '%m/%d/%y %I:%M%p'
    log_folder = 'logs'

    def time_to_str(self, date):
        date_str = datetime.strftime(date, self.date_format_str)
        
        # drop leading 0 on month
        date_str = date_str[1:] if date_str.startswith('0') else date_str
        
        date_str = date_str.replace('/0', '/')
        return date_str
    
    def str_to_time(self, date_str):
        return datetime.strptime(date_str, self.date_format_str)

    def clear_log_folder(self):
        if os.path.exists(self.log_folder):
            shutil.rmtree(self.log_folder)
        os.mkdir(self.log_folder)
