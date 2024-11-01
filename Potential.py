import numpy as np
import matplotlib.pyplot as plt

class Potential:
    def __init__(self, N=100, a=6, E=8, m=1, h=1):
        self.N = N  # Number of points
        self.a = a  # Range for x
        self.E = E  # Energy
        self.m = m  # Mass
        self.h = h  # Planck's constant
        self.x = np.linspace(-self.a, self.a, self.N + 1)

    class StepPotential:
        def __init__(self, parent, V_max,V_min=0):
            self.parent = parent
            self.V_max = V_max
            self.V_min = V_min

        def boundary_condition(self, n):
            if self.parent.x[n] < 0:
                return self.V_min
            else:
                return self.V_max

        def get_potential(self):
            return np.array([self.boundary_condition(n) for n in range(self.parent.N + 1)])

        def get_wavevector(self):
            V = self.get_potential()
            k_array = np.zeros_like(V, dtype=np.complex128)  # Fix: use complex128 instead of complex
            for i in range(len(V)):
                # Calculate k = sqrt(2m(E-V))/ℏ
                k_array[i] = np.sqrt(2.0 * self.parent.m * (self.parent.E - V[i]) + 0j) / self.parent.h
            return k_array

        def plot(self):
            plt.plot(self.parent.x, self.get_potential())
            plt.title("Step Potential vs x")
            plt.xlabel("x")
            plt.ylabel("V(x)")

        def plot_VE(self):
            plt.plot(self.parent.x, self.get_potential())
            plt.plot(self.parent.x, self.parent.E*np.ones(len(self.parent.x)))
            plt.title("Step Potential vs x")
            plt.xlabel("x")
            plt.ylabel("V(x)")
            plt.legend(["V(x)","E(x)"])
        
        def plot_subplots(self,ax):
            ax.plot(self.parent.x, self.get_potential())
            ax.plot(self.parent.x, self.parent.E*np.ones(len(self.parent.x)))
            ax.set_title("Step Potential vs x")
            ax.set_xlabel("x")
            ax.set_ylabel("V(x)")

    class PotentialBarrier:
        def __init__(self, parent, V_max =1, V_min=-1, width=1/3):
            self.parent = parent
            self.V_min = V_min
            self.V_max = V_max
            self.width = width

        def boundary_condition(self, n):
          if abs(self.parent.x[n]) <= self.parent.a *self.width:
            return self.V_max
          else:
            return self.V_min

        def get_potential(self):
            return np.array([self.boundary_condition(n) for n in range(self.parent.N + 1)])

        # In Potential.py, update the get_wavevector method:

        def get_wavevector(self):
            V = self.get_potential()
            k_array = np.zeros_like(V, dtype=np.complex128)  # Fix: use complex128 instead of complex
            for i in range(len(V)):
                # Calculate k = sqrt(2m(E-V))/ℏ
                k_array[i] = np.sqrt(2.0 * self.parent.m * (self.parent.E - V[i]) + 0j) / self.parent.h
            return k_array

        def plot(self):
            plt.plot(self.parent.x, self.get_potential())
            plt.title("Potential Barrier vs x")
            plt.xlabel("x")
            plt.ylabel("V(x)")

        def plot_VE(self):
            plt.plot(self.parent.x, self.get_potential())
            plt.plot(self.parent.x, self.parent.E*np.ones(len(self.parent.x)))
            plt.title("Potential Barrier vs x")
            plt.xlabel("x")
            plt.ylabel("V(x)")
            plt.legend(["V(x)","E(x)"])

        def plot_k(self):
            plt.plot(self.parent.x, self.get_wavevector())
            plt.title("wavevector vs x")
            plt.xlabel("x")
            plt.ylabel("V(x)")

        def plot_subplots(self,ax):
            ax.plot(self.parent.x, self.get_potential())
            ax.plot(self.parent.x, self.parent.E*np.ones(len(self.parent.x)))
            ax.set_title("Potential Barrier vs x")
            ax.set_xlabel("x")
            ax.set_ylabel("V(x)")

    class MorseFeshbach:
        def __init__(self, parent, V_0=2.5, d=1, x_0=0, mu=0.2,V_off=0):
            self.parent = parent
            self.V_0 = V_0
            self.d = d
            self.x_0 = x_0
            self.mu = mu
            self.V_off = V_off

        def get_potential(self):
          sinh2 = np.sinh((self.parent.x - self.x_0) /self.d)
          cosh2 = np.cosh((self.parent.x - self.x_0) / self.d - self.mu)
          return -self.V_0*(sinh2**2) / (cosh2**2)+self.V_off

        def get_wavevector(self):
            V = self.get_potential()
            k_array = np.zeros_like(V, dtype=np.complex128)  # Fix: use complex128 instead of complex
            for i in range(len(V)):
                # Calculate k = sqrt(2m(E-V))/ℏ
                k_array[i] = np.sqrt(2.0 * self.parent.m * (self.parent.E - V[i]) + 0j) / self.parent.h
            return k_array

        def plot(self):
            plt.plot(self.parent.x, self.get_potential())
            plt.title("Morse-Feshbach Potential vs x")
            plt.xlabel("x")
            plt.ylabel("V(x)")

        def plot_VE(self):
            plt.plot(self.parent.x, self.get_potential())
            plt.plot(self.parent.x, self.parent.E*np.ones(len(self.parent.x)))
            plt.title("Morse-Feshbach Potential vs x")
            plt.xlabel("x")
            plt.ylabel("V(x)")
            plt.legend(["V(x)","E(x)"])

        def plot_subplots(self,ax):
            ax.plot(self.parent.x, self.get_potential())
            ax.plot(self.parent.x, self.parent.E*np.ones(len(self.parent.x)))
            ax.set_title("Morse-Feshbach Potential vs x")
            ax.set_xlabel("x")
            ax.set_ylabel("V(x)")


# initiate the wavefunction 
from numpy import exp,sum,abs,sqrt,real,imag

class intial_wavefunction:
    def __init__(self,N=100,a=6,E=8):
        self.N = N
        self.a = a

    def gasussian_wavepacket(self,V,x0=0,variance=0.02):
        """the initial wavefunction have a form of gaussian wave packet"""
        def len_x0(x0)->int:
            if abs(x0) > self.a:raise ValueError("x0 must be in the range of the system")
            else:len_x0 = round((1+x0/self.a)*self.N/2)
            return len_x0

        k0= V.get_wavevector()[len_x0(x0)]
        x = V.parent.x
        dx= x[1]-x[0]

        Psi_0 = exp(-(x[1:-1]-x0)**2/variance**2 )*exp(1j*k0*x[1:-1])
        normalize = sqrt(sum(abs(Psi_0)**2*dx))
        Psi_0 = Psi_0/normalize
        return Psi_0
    
    def plot(self,V,Psi_0):
        x = V.parent.x
        plt.plot(x[1:-1],real(Psi_0))
        plt.plot(x[1:-1],imag(Psi_0),"--")
        plt.plot(x[1:-1],abs(Psi_0)**2)
        plt.xlabel("x")
        plt.ylabel("Psi")
        plt.legend(["real","imaginary","probability"])