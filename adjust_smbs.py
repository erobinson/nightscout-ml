from nightscout_ml_base import NightscoutMlBase
import os
import random
import pandas as pd
import numpy as np
from datetime import datetime


class AdjustSmbs(NightscoutMlBase):
    date_format_str = '%m/%d/%y %H:%M%p'
    no_insulin_threshold = 70
    no_insulin_when_dropping_threshold = 100
    min_target = 75
    max_target = 125
    base_isf = 100

    def adjust_smbs(self):
        df = pd.read_csv('adjustment_test.csv')
        
        

        # when below 70 then don't give insulin
        df.loc[df['bg'] < self.no_insulin_threshold, 'smbToGive'] = 0

        # when below 100 & dropping, don't give insulin
        df.loc[(df['bg'] < self.no_insulin_when_dropping_threshold) & (df['delta'] < 0), 'smbToGive'] = 0

        for index, row in df.iterrows():
            # if low, calculate insulin suplus (delta/isf) - go back 30+ min & delete
            if row['bg'] < self.min_target and row['delta'] < 0:
                u_to_remove = -1 * row['delta'] / self.get_isf(row['bg'])
                u_to_remove = self.roundToPt05(u_to_remove)
                # still want to remove some insulin if .02 should be removed
                u_to_remove = .05 if u_to_remove == 0 else u_to_remove
                self.remove_prior_insulin(index, row, df, u_to_remove)
            
            # if low treatment - go back 30+ min & delete 1u
            if row['low_treatment'] == 1:
                u_to_remove = 1
                self.remove_prior_insulin(index, row, df, u_to_remove)


        # if high, calculate u needed (delta/isf) - go back 15+ min & add insulin - check for if/when COB came into play
        # # # do more after COB show up rather than more before COB show up
        # # # don't give insulin if low
        # factor in breaks in time & sensor readings
        # mark as adjusted
        df['adjusted'] = 1

        df.to_csv('data/adjustment_test_updated.csv', index=False)

    def remove_prior_insulin(self, index, row, df, u_to_remove):
        orig_date = datetime.strptime(row['dateStr'], self.date_format_str)

        for i in range(6, 36):
            expected_date = orig_date - pd.DateOffset(minutes=i*5)
            prior_row = df.iloc[index - i]
            if expected_date == datetime.strptime(prior_row['dateStr'], self.date_format_str):
                smb_given = prior_row['smbToGive']
                if smb_given >= u_to_remove:
                    new_smb = smb_given - u_to_remove
                    prior_row['smbToGive'] = new_smb
                    df.at[index-i, 'smbToGive'] = new_smb
                    return
                if smb_given > 0 and smb_given < u_to_remove:
                    prior_row['smbToGive'] = 0
                    u_to_remove -= smb_given

    def roundToPt05(self, units):
        return round(units * 20) / 20

    def get_isf(self, bg):
        if bg > 150:
            return self.base_isf * .8
        if bg > 200:
            return self.base_isf * .6
        return self.base_isf