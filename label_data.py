import pandas
from nightscout_ml_base import NightscoutMlBase
import numpy as np

class LabelData(NightscoutMlBase):
    data: pandas.DataFrame
    label_cols = []
    feature_cols = []
    temp_cols = []

    def label_sgv_data(self, file):
        self.data = pandas.read_csv(self.data_folder+'/'+file)
        print(self.data.head())
        
        number_of_future_sgvs_to_use = 24
        self.add_future_sgvs_to_each_row(number_of_future_sgvs_to_use)
        self.add_high_low_labels(number_of_future_sgvs_to_use)
        self.drop_temp_columns()
        self.add_sgvs_for_past_hour()
        self.add_past_low_feature()
        self.add_past_sgv_changes_features()
        self.drop_temp_columns()

        output_file = file+'-labeled.csv'
        self.data.to_csv(self.data_folder+'/'+output_file)
        return output_file, self.feature_cols, self.label_cols

    def add_past_sgv_changes_features(self):
        for i in range(12):
            first_label = 'sgv' if i==0 else 'sgv_{}_min_ago'.format(i*5)
            second_label = 'sgv_{}_min_ago'.format((i+1)*5)
            feature_label = 'delta_{}_min_ago'.format(i*5)
            self.data[feature_label] = self.data[first_label] - self.data[second_label]
            self.feature_cols.append(feature_label)

    def add_sgvs_for_past_hour(self):
        for i in range(1, 13):
            self.data['temp'] = self.data['sgv']
            self.data['temp'] = self.data.temp.shift(-i)
            col_name = 'sgv_{}_min_ago'.format(i*5)
            self.data.rename(columns={'temp': col_name}, inplace=True)
            self.temp_cols.append(col_name)

    def add_past_low_feature(self):
        block = self.data[self.temp_cols]
        self.data['min_value'] = block[self.temp_cols].min(axis=1)
        label = 'low_in_last_hour'
        self.data[label] = np.where(self.data['min_value'] < 70, True, False)
        self.feature_cols.append(label)
        self.data.drop(columns=['min_value'])
        

    def add_future_sgvs_to_each_row(self, number_of_futre_sgvs):
        for i in range(1, number_of_futre_sgvs+1):
            self.data['temp'] = self.data['sgv']
            self.data['temp'] = self.data.temp.shift(i)
            col_name = 'sgv_in_{}'.format(i*5)
            self.data.rename(columns={'temp': col_name}, inplace=True)
            self.temp_cols.append(col_name)
        # sgvs_only = self.data[['dateString','sgv','sgv_in_5', 'sgv_in_10', 'sgv_in_15', 'sgv_in_20', 'sgv_in_25', 'sgv_in_30']]
        # print(sgvs_only.tail())

    def drop_temp_columns(self):
        self.data.drop(columns=self.temp_cols)

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
            for k in [50, 60, 70]:
                label = '30_min_block_{}_below_{}'.format(i, k)
                self.data[label] = np.where(self.data['min_value'] < k, True, False)
                self.label_cols.append(label)
            for k in [120, 140, 160, 180, 200, 220, 240, 260, 280, 300]:
                label = '30_min_block_{}_below_{}'.format(i, k)
                self.data[label] = np.where(self.data['min_value'] < k, True, False)
                self.label_cols.append(label)

        low_values = self.data[((self.data['sgv_in_5']<50) | (self.data['sgv_in_10']<50) | (self.data['sgv_in_15']<50) | (self.data['sgv_in_20']<50) | (self.data['sgv_in_25']<50) | (self.data['sgv_in_30']<50))]
        print(low_values[['sgv', 'sgv_in_5', 'sgv_in_10', 'sgv_in_15', 'sgv_in_20', 'sgv_in_25', 'sgv_in_30', '30_min_block_0_below_50']].head())
        # self.data['30_min_block_{}_below_60'.format(i)] = min_value[0].where(min_value[0] <= 60, 'True')
        self.data.drop(columns=['min_value','max_value'])
