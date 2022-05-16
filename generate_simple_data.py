from nightscout_ml_base import NightscoutMlBase
import os
import random


class GenerateSimpleData(NightscoutMlBase):
    isf = 100
    cr = 13
    target = 100
    maxIob = 7
    maxSMB = 2
    baseIob = .6

    def generate_data(self, count_to_generate):
        file_name = self.data_folder+'/simple_data_generated.csv'
        if(os.path.exists(file_name)):
            os.remove(file_name)
        file = open(file_name, 'a')
        current_cols = "bg,iob,cob,delta,shortAvgDelta,longAvgDelta,"+ \
                        "tdd7Days,tddDaily,tdd24Hrs," + \
                        "recentSteps5Minutes,recentSteps10Minutes,recentSteps15Minutes," + \
                        "maxSMB,maxIob,smbToGive"
        file.write(f"{current_cols}\n")
        for i in range(count_to_generate):
            bg = random.randint(40, 350)
            iob = random.randint(0, 10)
            cob = random.randint(0, 100)
            delta = random.randint(-30, 30)
            shortAvgDelta = random.randint(-30, 30)
            longAvgDelta = random.randint(-30, 30)
            tdd7Days = random.randint(48, 55)
            tdd24Hrs = random.randint(48, 55)
            tddDaily = random.randint(0, 55)
            recentSteps5Minutes = 0
            recentSteps10Minutes = 0
            recentSteps15Minutes = 0

            dynamicISF = self.isf
            if bg > 150:
                dynamicISF = self.isf * .8
            if bg > 200:
                dynamicISF = self.isf * .6
            insulin_for_bg = (bg - self.target) / dynamicISF
            insulin_for_cob = cob / self.cr
            smbToGive = insulin_for_bg + insulin_for_cob - iob + self.baseIob
            smbToGive += delta / dynamicISF
            if bg < 100 and delta < 1:
                smbToGive = 0
            if bg < 70:
                smbToGive = 0
            if smbToGive + iob > self.maxIob:
                smbToGive = self.maxIob - iob
            if smbToGive > self.maxSMB:
                smbToGive = self.maxSMB
            if smbToGive < 0:
                smbToGive = 0
            line = f"{bg},{iob},{cob},{delta},{shortAvgDelta},{longAvgDelta},"+ \
                        f"{tdd7Days},{tddDaily},{tdd24Hrs}," + \
                        f"{recentSteps5Minutes},{recentSteps10Minutes},{recentSteps15Minutes}," + \
                        f"{self.maxSMB},{self.maxIob},{smbToGive}"
            file.write(f"{line}\n")
        file.close()



