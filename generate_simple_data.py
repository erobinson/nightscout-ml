from nightscout_ml_base import NightscoutMlBase
import os
import random


class GenerateSimpleData(NightscoutMlBase):
    isf = 100
    cr = 13
    target = 100
    maxIOB = 7
    maxSMB = 2

    def generate_data(self, count_to_generate):
        file_name = self.data_folder+'/simple_data_generated.csv'
        if(os.path.exists(file_name)):
            os.remove(file_name)
        file = open(file_name, 'a')
        file.write("bg,iob,cob,smbToGive\n")
        for i in range(count_to_generate):
            bg = random.randint(40, 350)
            iob = random.randint(0, 20)
            cob = random.randint(0, 100)
            dynamicISF = self.isf
            if bg > 150:
                dynamicISF = self.isf * .8
            if bg > 200:
                dynamicISF = self.isf * .6
            insulin_for_bg = (bg - self.target) / dynamicISF
            insulin_for_cob = cob / self.cr
            smbToGive = insulin_for_bg + insulin_for_cob - iob + 1
            if smbToGive + iob > self.maxIOB:
                smbToGive = self.maxIOB - iob
            if smbToGive > self.maxSMB:
                smbToGive = self.maxSMB
            file.write(f"{bg},{iob},{cob},{smbToGive}\n")
        file.close()



