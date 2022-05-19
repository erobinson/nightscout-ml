from nightscout_ml_base import NightscoutMlBase
import os
import random


class GenerateSimpleData(NightscoutMlBase):
    isf = 100
    cr = 13
    target = 100
    maxIob = 7
    maxSMB = 2
    baseIob = 1
    weekend_ratio = .6

    def generate_data(self, count_to_generate):
        file_name = self.data_folder+'/simple_data_generated.csv'
        if(os.path.exists(file_name)):
            os.remove(file_name)
        file = open(file_name, 'a')
        current_cols =  "hour0_2,hour3_5,hour6_8,hour9_11,hour12_14,hour15_17,hour18_20,hour21_23,weekend," + \
                        "bg,iob,cob,delta,shortAvgDelta,longAvgDelta," + \
                        "tdd7Days,tddDaily,tdd24Hrs," + \
                        "recentSteps5Minutes,recentSteps10Minutes,recentSteps15Minutes,recentSteps30Minutes,recentSteps60Minutes," + \
                        "smbToGive"
        file.write(f"{current_cols}\n")
        for i in range(count_to_generate):
            bg = random.randint(60, 180)
            # iob = random.randint(0, 6)
            iob = round(random.gauss(1.0, 2), 2)
            iob = iob if iob > 0 else 0
            cob = random.randint(0, 60)
            cob = 0 if random.randint(0,7) == 0 else cob
            iob = round(iob + 2 if cob > 20 else iob, 2)
            # delta = random.randint(-10, 10)
            delta = round(random.gauss(0, 5), 0)
            shortAvgDelta = random.randint(-10, 10)
            longAvgDelta = random.randint(-10, 10)
            tdd7Days = random.randint(48, 55)
            tdd24Hrs = random.randint(48, 55)
            tddDaily = random.randint(0, 55)
            recentSteps5Minutes = 0
            recentSteps10Minutes = 0
            recentSteps15Minutes = 0
            recentSteps30Minutes = 0
            recentSteps60Minutes = 0
            hour = random.randint(0,23)
            hour0_2 =   1 if hour >= 0 and hour <= 2 else 0
            hour3_5 =   1 if hour >= 3 and hour <= 5 else 0
            hour6_8 =   1 if hour >= 6 and hour <= 8 else 0
            hour9_11 =  1 if hour >= 9 and hour <= 11 else 0
            hour12_14 = 1 if hour >= 12 and hour <= 14 else 0
            hour15_17 = 1 if hour >= 15 and hour <= 17 else 0
            hour18_20 = 1 if hour >= 18 and hour <= 20 else 0
            hour21_23 = 1 if hour >= 21 and hour <= 23 else 0
            weekend = random.randint(0,1)

            dynamicISF = self.isf
            if bg > 150:
                dynamicISF = self.isf * .8
            if bg > 200:
                dynamicISF = self.isf * .6
            insulin_for_bg = (bg - self.target) / dynamicISF
            insulin_for_cob = cob / self.cr
            target_iob = self.baseIob if weekend == 0 else self.baseIob * self.weekend_ratio
            
            smbToGive = insulin_for_bg + insulin_for_cob - iob + target_iob
            smbToGive += delta / dynamicISF

            # don't give if low or dropping
            if (bg < 70) or (bg < 100 and delta < 0) or (delta < -8):
                smbToGive = 0
            
            if smbToGive + iob > self.maxIob:
                smbToGive = self.maxIob - iob
            if smbToGive > self.maxSMB:
                smbToGive = self.maxSMB
            if smbToGive < 0:
                smbToGive = 0
            
            line = f"{hour0_2},{hour3_5},{hour6_8},{hour9_11},{hour12_14},{hour15_17},{hour18_20},{hour21_23},{weekend}," + \
                        f"{bg},{iob},{cob},{delta},{shortAvgDelta},{longAvgDelta}," + \
                        f"{tdd7Days},{tddDaily},{tdd24Hrs}," + \
                        f"{recentSteps5Minutes},{recentSteps10Minutes},{recentSteps15Minutes},{recentSteps30Minutes},{recentSteps60Minutes}," + \
                        f"{smbToGive}"
            file.write(f"{line}\n")
        file.close()



