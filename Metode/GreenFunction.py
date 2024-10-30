import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse import diags
sys.path.append('/workspaces/Tugas_Grafik/')
from Potential import Potential
 
class GreenFunction():
    def __init__(self, m, h, E) -> None:
        self.m = m
        self.h = h
        self.E = E
    
    def Hamiltonian(self,V):
        t = self.h**2/(2*self.m)*np.ones(V)
        H = diags([-t, V+2*t, -t], [-1, 0, 1], shape=(V.N,V.N)).toarray()
        return H
    
