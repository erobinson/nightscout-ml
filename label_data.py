import pandas
from nightscout_ml_base import NightscoutMlBase
import numpy as np

class LabelData(NightscoutMlBase):
    data: pandas.DataFrame

    def label_sgv_data(self, file):
        self.data = pandas.read_csv(self.data_folder+'/'+file)
        print(self.data.head())
        
        number_of_future_sgvs_to_use = 36
        self.add_future_sgvs_to_each_row(number_of_future_sgvs_to_use)
        self.add_high_low_labels(number_of_future_sgvs_to_use)
        self.data.to_csv(self.data_folder+'/'+file+'-labeled.csv')


    def add_future_sgvs_to_each_row(self, number_of_futre_sgvs):

        for i in range(1, number_of_futre_sgvs+1):
            self.data['temp'] = self.data['sgv']
            self.data['temp'] = self.data.temp.shift(i)
            self.data.rename(columns={'temp': 'sgv_in_{}'.format(i*5)}, inplace=True)

        sgvs_only = self.data[['dateString','sgv','sgv_in_5', 'sgv_in_10', 'sgv_in_15', 'sgv_in_20', 'sgv_in_25', 'sgv_in_30']]
        print(sgvs_only.tail())

    def add_high_low_labels(self, number_of_futre_sgvs):
        number_of_30_min_blocks = number_of_futre_sgvs / 6
        for i in range(int(number_of_30_min_blocks)):
            columns = []
            for j in range(6):
                column_name = 'sgv_in_{}'.format((i * 30) + ((j + 1) * 5))
                columns.append(column_name)
            block = self.data[columns]
            self.data['min_value'] = block[columns].min(axis=1)
            self.data['max_value'] = block[columns].max(axis=1)
            self.data['30_min_block_{}_below_50'.format(i)] = np.where(self.data['min_value'] < 50, True, False)
            self.data['30_min_block_{}_below_60'.format(i)] = np.where(self.data['min_value'] < 60, True, False)
            self.data['30_min_block_{}_above_120'.format(i)] = np.where(self.data['max_value'] > 120, True, False)
            self.data['30_min_block_{}_above_140'.format(i)] = np.where(self.data['max_value'] > 140, True, False)
            self.data['30_min_block_{}_above_160'.format(i)] = np.where(self.data['max_value'] > 160, True, False)
            self.data['30_min_block_{}_above_180'.format(i)] = np.where(self.data['max_value'] > 180, True, False)
            self.data['30_min_block_{}_above_200'.format(i)] = np.where(self.data['max_value'] > 200, True, False)
            self.data['30_min_block_{}_above_220'.format(i)] = np.where(self.data['max_value'] > 220, True, False)
            self.data['30_min_block_{}_above_240'.format(i)] = np.where(self.data['max_value'] > 240, True, False)
            self.data['30_min_block_{}_above_260'.format(i)] = np.where(self.data['max_value'] > 260, True, False)
            self.data['30_min_block_{}_above_280'.format(i)] = np.where(self.data['max_value'] > 280, True, False)
            self.data['30_min_block_{}_above_300'.format(i)] = np.where(self.data['max_value'] > 300, True, False)

        low_values = self.data[((self.data['sgv_in_5']<50) | (self.data['sgv_in_10']<50) | (self.data['sgv_in_15']<50) | (self.data['sgv_in_20']<50) | (self.data['sgv_in_25']<50) | (self.data['sgv_in_30']<50))]
        print(low_values[['sgv', 'sgv_in_5', 'sgv_in_10', 'sgv_in_15', 'sgv_in_20', 'sgv_in_25', 'sgv_in_30', '30_min_block_0_below_50']].head())
        # self.data['30_min_block_{}_below_60'.format(i)] = min_value[0].where(min_value[0] <= 60, 'True')
        self.data.drop(columns=['min_value','max_value'])
