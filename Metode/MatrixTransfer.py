import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/workspaces/Tugas_Grafik/')
from Potential import Potential

class TransferMatrixMethod_V2:
    def __init__(self, m=1, h=1, E=8):
        self.m = m
        self.h = h
        self.E = E

    def wave_number(self, V):
        return np.lib.scimath.sqrt(2 * self.m * (self.E - V)) / self.h

    def transfer_matrix(self, k1, k2, a):
        M11 = 0.5 * (1 + k2 / k1) * np.exp(1j * (k1 - k2) * a)
        M12 = 0.5 * (1 - k2 / k1) * np.exp(1j * (k1 + k2) * a)
        M21 = 0.5 * (1 - k2 / k1) * np.exp(-1j * (k1 + k2) * a)
        M22 = 0.5 * (1 + k2 / k1) * np.exp(-1j * (k1 - k2) * a)
        return np.array([[M11, M12], [M21, M22]])

    def total_transfer_matrix(self, potential):
        V = potential.get_potential()
        x = potential.parent.x
        M_total = np.identity(2, dtype=complex)
        for i in range(len(V) - 1):
            k1 = self.wave_number(V[i])
            k2 = self.wave_number(V[i + 1])
            M = self.transfer_matrix(k1, k2, x[i])
            M_total = np.dot(M_total, M)
        return M_total

    def compute_Aj_Bj(self, A_jp1, B_jp1, j, k, x):
        """
        Calculate the A and B components at discretized x.
        """
        k1 = k[j]
        k2 = k[j + 1]
        a = x[j]

        M = self.transfer_matrix(k1, k2, a)
        
        A_j = (A_jp1 * M[0, 0] + B_jp1 * M[0, 1])
        B_j = (A_jp1 * M[1, 0] + B_jp1 * M[1, 1])

        return A_j, B_j
    
    def wavefunction(self, potential, x, A_n=1, B_n=0):
        V = potential.get_potential()
        x = potential.parent.x
        A = np.zeros(len(V), dtype=complex)
        B = np.zeros(len(V), dtype=complex)
        psi = np.zeros(len(V), dtype=complex)
        k = self.wave_number(V)
        
        A[-1] = A_n
        B[-1] = B_n
        
        for j in range(len(V) - 2, -1, -1):
            A[j], B[j] = self.compute_Aj_Bj(A[j+1], B[j+1], j, k, x)
            psi[j] = A[j] * np.exp(1j * k[j] * x[j]) + B[j] * np.exp(-1j * k[j] * x[j])
        
        return psi, A, B

    def transmission_reflection_coefficients(self, potential):
        M_total = self.total_transfer_matrix(potential)
        T1 = 1 / np.abs(M_total[0, 0])**2
        R1 = np.abs(M_total[1, 0] / M_total[0, 0])**2

        V = potential.get_potential()
        x = potential.parent.x
        k = self.wave_number(V)
        A = np.zeros(len(V), dtype=complex)
        B = np.zeros(len(V), dtype=complex)
        
        A[-1] = 1  # Assuming A_n = 1 at the last point
        B[-1] = 0  # Assuming B_n = 0 at the last point
        
        for j in range(len(V) - 2, -1, -1):
            A[j], B[j] = self.compute_Aj_Bj(A[j+1], B[j+1], j, k, x)
        
        A_array = np.array(A)
        B_array = np.array(B)

        # Calculate transmission (T) and reflection (R)
        T = np.fromiter((np.abs(A_array[i])**2 / np.abs(A_array[i-1])**2 for i in range(len(V)-1, 0, -1)), dtype=complex)
        R = np.fromiter((np.abs(B_array[i])**2 / np.abs(A_array[i])**2 for i in range(len(V)-1)), dtype=complex)
        
        return T, R, T1, R1
    
    def plot_wavefunction(self, potential, x):
        """
        Plot the wavefunction vs. x.
        """
        psi = self.wavefunction(potential, x)[0]
        plt.plot(x, np.real(psi), label='Real part')
        plt.plot(x, np.imag(psi), label='Imaginary part')
        plt.xlabel('x')
        plt.ylabel('Wavefunction')
        plt.title('Wavefunction vs. x')
        plt.legend()
        plt.show()