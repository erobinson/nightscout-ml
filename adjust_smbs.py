from nightscout_ml_base import NightscoutMlBase
import pandas as pd
from datetime import datetime


class AdjustSmbs(NightscoutMlBase):
    date_format_str = '%m/%d/%y %H:%M%p'
    no_insulin_threshold = 70
    no_insulin_when_dropping_threshold = 100
    min_target = 75
    max_target = 125
    base_isf = 100
    max_smb = 2

    def adjust_smbs(self):
        # df = pd.read_csv('adjustment_test.csv')
        df = pd.read_csv('aiSMB_records_adjusted.csv')

        df.loc[df['adjusted'] != 1, 'smbToGive'] = df['smbGiven']

        # when below 70 then don't give insulin
        df.loc[df['bg'] < self.no_insulin_threshold, 'smbToGive'] = 0

        # when below 100 & dropping, don't give insulin
        df.loc[(df['bg'] < self.no_insulin_when_dropping_threshold) & (df['delta'] < 0), 'smbToGive'] = 0

        for index, row in df.iterrows():
            if row['adjusted'] != 1:

                self.adjust_for_lows(index, row, df)

                self.adjust_for_highs(index, row, df)

                df.at[index, 'adjusted'] = 1

        # df.to_csv('data/adjustment_test_updated.csv', index=False)
        df.to_csv('data/aiSMB_records_adjusted.code.csv', index=False)

    def adjust_for_lows(self, index, row, df):
        # if low, calculate insulin suplus (delta/isf) - go back 30+ min & delete
        if row['bg'] < self.min_target and row['delta'] < 0:
            u_to_remove = self.u_to_adjust_based_on_delta(row)
            self.remove_prior_insulin(index, row, df, u_to_remove)
        
        # if low treatment - go back 30+ min & delete 1u
        if row['low_treatment'] == 1:
            u_to_remove = 1
            self.remove_prior_insulin(index, row, df, u_to_remove)

    def adjust_for_highs(self, index, row, df):
        if row['bg'] > self.max_target and row['delta'] > 0:
            u_to_add = self.u_to_adjust_based_on_delta(row)
            self.add_prior_insulin(index, row, df, u_to_add)
        
    def add_prior_insulin(self, index, row, df, u_to_add):
        one_hour_ago = index - 12 if index > 12 else 0
        df_last_hour = df[one_hour_ago:index]
        orig_date = datetime.strptime(row['dateStr'], self.date_format_str)
        min_date = orig_date - pd.DateOffset(minutes=13*5)
        for i, recent_row in df_last_hour.iterrows():
            # don't add insulin if carbs are added later, add after carbs are added
            more_carbs_added_later = recent_row['cob'] == 0 and self.more_carbs_added_later(recent_row['cob'], i, df_last_hour, index)
            # don't add insulin if pending low
            upcoming_low = self.check_for_upcoming_low(i, df_last_hour)
            # don't add if old date
            out_dated = datetime.strptime(recent_row['dateStr'], self.date_format_str) < min_date

            if not more_carbs_added_later and not upcoming_low and not out_dated:
                current_smb = recent_row['smbToGive']
                if u_to_add + current_smb < self.max_smb:
                    df.at[i, 'smbToGive'] = current_smb + u_to_add
                    return
                if u_to_add < self.max_smb:
                    df.at[i, 'smbToGive'] = 2
                    u_to_add -= self.max_smb - current_smb

    def more_carbs_added_later(self, current_cob, recent_index, df_last_hour, index):
        for i in range(recent_index, index):
            if current_cob < df_last_hour.at[i, 'cob']:
                return True
        return False

    def check_for_upcoming_low(self, recent_index, df_last_hour):
        for i in range(recent_index, len(df_last_hour)):
            row = df_last_hour[i]
            if row['bg'] < self.no_insulin_threshold \
                or (row['bg'] < self.no_insulin_when_dropping_threshold and row['delta'] < 0):
                return True
        return False

    def remove_prior_insulin(self, index, row, df, u_to_remove):
        orig_date = datetime.strptime(row['dateStr'], self.date_format_str)
        min_date = orig_date - pd.DateOffset(minutes=37*5)

        for i in range(6, 36):
            prior_row = df.iloc[index - i]
            row_date = datetime.strptime(prior_row['dateStr'], self.date_format_str)
            if row_date > min_date:
                smb_given = prior_row['smbToGive']
                if smb_given >= u_to_remove:
                    new_smb = smb_given - u_to_remove
                    df.at[index-i, 'smbToGive'] = new_smb
                    return
                if smb_given > 0 and smb_given < u_to_remove:
                    df.at[index-i, 'smbToGive'] = 0
                    u_to_remove -= smb_given

    def u_to_adjust_based_on_delta(self, row):
        units = abs(row['delta']) / self.get_isf(row['bg'])
        units = self.roundToPt05(units)
        # still want to remove some insulin if .02 should be removed
        units = .05 if units == 0 else units
        return units

    def roundToPt05(self, units):
        return round(units * 20) / 20

    def get_isf(self, bg):
        if bg > 150:
            return self.base_isf * .8
        if bg > 200:
            return self.base_isf * .6
        return self.base_isf