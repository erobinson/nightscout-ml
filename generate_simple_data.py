from nightscout_ml_base import NightscoutMlBase
import os
import random


class GenerateSimpleData(NightscoutMlBase):
    isf = 80
    cr = 13
    target = 100

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
            insulin_for_bg = (bg - self.target) / self.isf
            insulin_for_cob = cob / self.cr
            smbToGive = insulin_for_bg + insulin_for_cob - iob + 1
            file.write(f"{bg},{iob},{cob},{smbToGive}\n")
        file.close()



