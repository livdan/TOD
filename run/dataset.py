import os

class LessOrderedData(object):
      def __init__(self,weight,corrfactor,measured_temp,ambient_temp, delta_time):
        self.weight = weight
        self.corrfactor = corrfactor
        self.measured_temp = measured_temp
        self.ambient_temp = ambient_temp
        self.delta_time = delta_time

class Dataset:   

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def CreateDataset(self,dir,svmpath,generated_data):      
        tempdata =[]         
        
        if not os.path.exists(dir):
            os.makedirs(dir)
            
        f = open(svmpath, "w")
                 
        for item in generated_data:          
            n = LessOrderedData(weight=item[0] ,corrfactor=item[1] ,measured_temp=item[3], ambient_temp=item[4], delta_time=item[2])
            tempdata.append(str(n.__dict__))

        for tdata in tempdata:            
            f.write(str(tdata)+"\n")

        f.close()

    def LoadDataset(self, path):
        f = open(path,"r")
        lines = f.readlines()
        f.close()              
        lines = [l.split("\n")[0] for l in lines]
        return lines
        