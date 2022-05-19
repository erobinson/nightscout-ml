from nightscout_ml_base import NightscoutMlBase
import os
import random
import pandas as pd
import numpy as np
from datetime import datetime


class AdjustSmbs(NightscoutMlBase):

    def adjust_smbs(self):
        df = pd.read_csv('data/adjustment_test.csv')
        df["maxIob"] = 7
        df["maxSMB"] = 2

        


        df.to_csv('data/adjustment_test_updated.csv', index=False)
