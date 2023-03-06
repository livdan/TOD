import myutils.MarshallHoare as MH
import numpy as np
import random 
import matplotlib.pyplot as plt
import os
import pandas as pd
class Datagenerator:
    
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    
    def round_to_nearest_half_int(self,num):
        return round(num * 2) / 2

    def GenerateData(self,train_count,maindata_dir,scale):
        mh = MH.MarshallHoare()

        times = np.arange(1,18,0.5)#[ti for ti in range(1,24,1)]
        T_ambient = np.arange(-10,35,0.5)#[ta for ta in range(-10,36,1)]
        corr_factor = [0.7,0.9,1.0,1.1,1.2,1.3,1.4]
        m_random = np.random.normal(loc=70,scale=scale,size=train_count)   
        
        # TODO histogram különböző scale-ekre, 1000 adattal
        # TODO az összes többi adatra is
        m = []
        for k in range(len(m_random)):
            if m_random[k] > 50 and m_random[k] < 101:
                m.append(m_random[k])
                
        length_m = len(m)   
        all_data = np.zeros((length_m,5))
        
        plt.rcParams["figure.figsize"] = [16.00, 8.0]
        plt.rcParams["figure.autolayout"] = True
        counts, _, patches = plt.hist(m_random,bins = range(0,160,10))
      
        for count, patch in zip(counts,patches):
            plt.annotate(str(int(count)), xy=(patch.get_x(), patch.get_height()))
            
        if not os.path.exists(maindata_dir):
            os.makedirs(maindata_dir)
            
        plt.savefig(str(maindata_dir)+'/m_random_all_'+str(length_m)+'_'+str(scale) +'_'+'.png')
        
        i=0
        while i < length_m:
            t = times[random.randint(0,len(times)-1)] #time [h]
            cf = corr_factor[random.randint(0,len(corr_factor)-1)]
            T_a = T_ambient[random.randint(0,len(T_ambient)-1)] #ambient temp. [°C]
            T_r = mh.MH_Tb(t,T_a,cf,m[i]) #rectal temp. [°C]
            
            all_data[i,0] = format(self.round_to_nearest_half_int(m[i]),'.1f')
            all_data[i,1] = cf
            all_data[i,2] = t
            all_data[i,3] = format(T_r, '.1f')
            all_data[i,4] = T_a
            i+=1
        
        filename = str(maindata_dir)+'/generateddata_'+str(format(train_count, '.0f'))+'.csv'
       
        fieldnames = ['m','corrfactor','delta_t','T_r', 'Ta']
        df = pd.DataFrame(all_data, columns = fieldnames)    
        df.to_csv(filename,index=False) 
       
        return all_data
