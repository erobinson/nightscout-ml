from datetime import datetime
class NightscoutMlBase():
    data_folder = 'data'
    date_format_str = '%m/%d/%y %I:%M%p'

    def time_to_str(self, date):
        date_str = datetime.strftime(date, self.date_format_str)
        
        # drop leading 0 on month
        date_str = date_str[1:] if date_str.startswith('0') else date_str
        
        # switch midnight 00:XXAM to 12:XXAM
        # date_str = date_str.replace('00:', '12:')
        return date_str
    
    def str_to_time(self, date_str):
        return datetime.strptime(date_str, self.date_format_str)

