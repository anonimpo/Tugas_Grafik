import sys
sys.path.append('/workspaces/Tugas_Grafik/')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
 
from numpy import ones,imag,pi
from scipy.sparse import diags, eye
from scipy.sparse.linalg import inv as inverse
from Potential import Potential


class GreenFunction_V2:
    def __init__(self, m, h, E):
        __name__ = "Metode Green's Function "
        self.m = m
        self.h = h
        self.E = E
    
    def Hamiltonian(self, V):
        N= V.parent.N
        dx = V.parent.x[1] - V.parent.x[0]
        self.t = self.h**2/(2*self.m)/dx**2
        V = V.get_potential()
        t = self.t*ones(N+1)
        H = diags([-t[:-1], V+2*t, -t[:-1]], [-1, 0, 1], shape=(N,N))
        return H
    
    def HL_HC_HR(self, V):
        N = V.parent.N
        H = self.Hamiltonian(V)
        HL = H[(N-2)/2,(N-2)/2]
        HR = H[(N+2)/2:-1,(N+2)/2:-1]
        HC = H[49:51,49:51]
        return HL,HC,HR

    def GreenFunction_for_finding_Self_Energy(self, V, Green_surfaceL=0,Green_surfaceR=0):
        self.Green_surfaceL = Green_surfaceL
        self.Green_surfaceR = Green_surfaceR
        N = V.parent.N
        H = self.Hamiltonian(V)
        Sigma=eye(N) * 0
        Sigma=Sigma.tocsc()
        Sigma[0,0],Sigma[-1,-1] = self.t*Green_surfaceL*self.t, self.t*Green_surfaceR*self.t
        G = inverse(self.E*eye(N) - H -Sigma)                     
        return G
    
    #full green's function, masih belum yakin benar ??
    def GreenFunction(self, V):
        GreenFunction_0 = self.GreenFunction_for_finding_Self_Energy(V,0,0)
        gL,gR = GreenFunction_0[0,0],GreenFunction_0[-1,-1]
        G= self.GreenFunction_for_finding_Self_Energy(V,gL,gR)
        return G

    def GreenFunction_for_finding_center_green_function(self, V):
        GreenFunction_0 = self.GreenFunction_for_finding_Self_Energy(V,0,0)
        gL,gR = GreenFunction_0[0,0],GreenFunction_0[-1,-1]
        Gc = self.GreenFunction_for_finding_Self_Energy(V,gL,gR)
        Gc = Gc[Gc.shape[0] // 2, Gc.shape[1] // 2]
        return Gc
    
    def DensityOfState_in_center(self, V):
        G = self.GreenFunction_for_finding_center_green_function(V)
        DOS = -1/pi*imag(G)
        return DOS