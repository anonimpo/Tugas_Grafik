import numpy as np
import matplotlib.pyplot as plt

class Potential:
    def __init__(self, N=100, a=6, E=80, m=1, h=1):
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
            k_array = np.lib.scimath.sqrt(2 * self.parent.m * (self.parent.E - V) / self.parent.h**2)
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

        def get_wavevector(self):
            V = self.get_potential()
            k_array = np.lib.scimath.sqrt(2 * self.parent.m * (self.parent.E - V) / self.parent.h**2)
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

        def plot_VEk(self):
            plt.plot(self.parent.x, self.get_potential())
            plt.plot(self.parent.x, self.get_wavevector())
            plt.plot(self.parent.x, self.parent.E*np.ones(len(self.parent.x)))
            plt.title("Potential Barrier")
            plt.xlabel("x")
            plt.ylabel("V(x)")
            plt.legend(["V(x)","k(x)","E(x)"])

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
            k_array = np.lib.scimath.sqrt(2 * self.parent.m * (self.parent.E - V) / self.parent.h**2)
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


    class Other:
        def __init__(self, parent,f,*arg):
            self.parent = parent
            self.f = f

        def get_potential(self,x):
            return self.f
        def get_wavevector(self):
            V = self.get_potential()
            k_array = np.array([
                np.lib.scimath.sqrt(2 * self.parent.m * (self.parent.E - V) / self.parent.h**2)
                for x in self.parent.x ])
            return k_array