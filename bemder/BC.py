import numpy as np
from material import PorousAbsorber

class ApplyBC():
    def __init__(self,AC,AP):
        self.AP = AP
        self.AC = AC
        self.mu = {}
    
    def impedance(domain_index, impedance):
        if len(impedance) == self.AC.freq:
            for i in domain_index:
                self.mu[i] = np.array(1/impedance[:,i])
        else:
            print("Impedance must match number of defined frequencies. Impedance takes the for [freqs,domain_index]")
            
    def porous(domain_index)
            