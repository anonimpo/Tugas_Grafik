import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/workspaces/Tugas_Grafik/')
from Potential import Potential

class AnalyticalMethod_V2:
    def __init__(self, m=1, h=1, E=80):
        self.m = m
        self.h = h
        self.E = E

    def wave_numbers(self, V):
        k1 = np.lib.scimath.sqrt(2 * self.m * self.E) / self.h
        k2 = np.lib.scimath.sqrt(2 * self.m * (self.E - V)) / self.h
        return k1, k2

    def transmission_reflection_coefficients(self, potential, **kwargs):
        V = potential.get_potential()
        L = potential.parent.a * potential.width
        if isinstance(potential, Potential.PotentialBarrier):
            k1, k2 = self.wave_numbers(V)
            kappa = 1j*k2/k1
            T = 4/((kappa+1/kappa)**2*np.sinh(1j*k2*L)**2+1)
            R = 1-T
        elif isinstance(potential, Potential.StepPotential):
            k1, k2 = self.wave_numbers(V)
            T = 4 * k1 * k2 / ((k1 + k2)**2)
            R = ((k1 - k2) / (k1 + k2))**2
        elif isinstance(potential, Potential.MorseFeshbach):
            k1, k2 = self.wave_numbers(V)
            T = 4 * k1 * k2 / ((k1 + k2)**2)
            R = ((k1 - k2) / (k1 + k2))**2
        else:
            raise ValueError("Unknown potential type")
        
        # Ensure T and R are probabilities
        T = np.abs(T)
        R = np.abs(R)
        
        # Ensure the relation T + R = 1
        if not np.all(np.isclose(T + R, 1)):
            T = T / (T + R)
            R = 1 - T
        
        return T, R

    def wavefunction(self, potential, x):
        """
        Calculate the wavefunction for a given potential.
        """
        V = potential.get_potential()
        A,B= np.sqrt(self.transmission_reflection_coefficients(potential))
        psi = np.zeros_like(x, dtype=complex)
        if isinstance(potential, Potential.PotentialBarrier):
            a = potential.width
            for i, xi in enumerate(x):
                k1, k2 = self.wave_numbers(V[i])
                if xi < -a:
                    psi[i] = 1*np.exp(1j * k1 * xi) + B[i]*np.exp(-1j * k1 * xi)
                elif -a <= xi <= a:
                    psi[i] = A[i]*np.exp(1j * k2 * xi) + B[i]*np.exp(-1j * k2 * xi)
                else:
                    psi[i] = A[i]*np.exp(1j * k1 * xi)
        elif isinstance(potential, Potential.StepPotential):
            for i, xi in enumerate(x):
                k1, k2 = self.wave_numbers(V[i])
                if xi < 0:
                    psi[i] = np.exp(1j * k1 * xi) + np.exp(-1j * k1 * xi)
                else:
                    psi[i] = np.exp(1j * k2 * xi)
        elif isinstance(potential, Potential.MorseFeshbach):
            for i, xi in enumerate(x):
                k1, k2 = self.wave_numbers(V[i])
                psi[i] = np.exp(1j * k2 * xi)
        else:
            raise ValueError("Unknown potential type")
        return psi

    def plot_wavefunction(self, potential, x):
        """
        Plot the wavefunction vs. x.
        """
        psi = self.wavefunction(potential, x)
        plt.plot(x, np.real(psi), label='Real part')
        plt.plot(x, np.imag(psi), label='Imaginary part')
        plt.xlabel('x')
        plt.ylabel('Wavefunction')
        plt.title('Wavefunction vs. x')
        plt.legend()
        plt.show()