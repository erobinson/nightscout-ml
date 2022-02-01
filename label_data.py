import pandas
from nightscout_ml_base import NightscoutMlBase

class LabelData(NightscoutMlBase):

    def label_sgv_data(self, file):
        data = pandas.read_csv(self.data_folder+'/'+file)
        data.head()
        data['sgvIn5'] = data['sgv']
        data['sgvIn5'] = data.sgvIn5.shift(1)
        sgvs_only = data[['dateString','sgv','sgvIn5']]
        data.head()


