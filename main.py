import numpy as np
import matplotlib.pyplot as plt 
from Potential import Potential
from Metode.AnalyticalMethod import AnalyticalMethod_V2 
from Metode.MatrixTransfer import TransferMatrixMethod_V2
from Metode.GreenFunction import GreenFunction 

m,h= 1,1
N = 100 
a = 6 
E = 8 
Vmax = 10

call = Potential(N=N,a=a,E=E,m=m,h=h)
Potential_Barrier= call.PotentialBarrier(call,V_max=Vmax, V_min=0, width=1/3)
Step_Potential= call.StepPotential(call,V_max=Vmax,V_min=0)
Morse_Feshbach= call.MorseFeshbach(call,V_0=Vmax,d=1,x_0=0,mu=0.2)

Metode_Analitik = AnalyticalMethod_V2(m,h,E)            # masih kurang benar di plot_wavefunction 
Metode_MatrikTransfer =TransferMatrixMethod_V2(m,h,E)   # masih tidak yakin
Metode_GreenFunction = GreenFunction()                  # belum seleai


#plot bentuk potensial V dan E  
def plot(potential):
    potential.plot_VE()
    plt.savefig(f"grafik-{potential.__class__.__name__}.png")
#plot(Potential_Barrier)

#menghitung koefisien Transmisi dan Refleksi 
T_Analitik,R_Analitik=Metode_Analitik.transmission_reflection_coefficients(Potential_Barrier)  # baru potential penghalang yang sudah di cek
T_MatrikTransfer,R_MatrikTransfer=Metode_MatrikTransfer.transmission_reflection_coefficients(Potential_Barrier)[1:]
#T_GF,R_GF= Metode_GreenFunction.transmission_reflection_coefficients() belum ada

print (T_Analitik)
