import math

class MarshallHoare:
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    
    def B(self,m,k) -> float:
        return -1.2815*(math.pow(k*m,-0.625))+0.0284

    def MH_Tb(self, t,T_a,k,m)-> float:
        B_temp = self.B(m,k)
        if T_a <= 23.2:
            return (37.2-T_a)*(1.25*math.exp(B_temp*t)-0.25*math.exp(5*B_temp*t))+T_a
        if T_a >23.3:          
            return (37.2-T_a)*(1.11*math.exp(B_temp*t)-0.11*math.exp(10*B_temp*t))+T_a 
    
    # def MH(self,t, *values):
    #     T_b,T_a,k,m = values
    #     if T_a <= 23.2:
    #         return 1.25*math.exp(self.B(m,k)*t)-0.25*math.exp(5*self.B(m,k)*t)-((T_b-T_a)/(37.2-T_a)) 
    #     if T_a >=23.3:
    #         return 1.11*math.exp(self.B(m,k)*t)-0.11*math.exp(10*self.B(m,k)*t)-((T_b-T_a)/(37.2-T_a)) 

    #     return -1