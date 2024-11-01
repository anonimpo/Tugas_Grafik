import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation, PillowWriter
import os


class Gaussian_Wave:
        def __init__(self, N, L, x0, sigma, k0, V0, a, b, t):
            self.N = N
            self.L = L
            self.x0 = x0
            self.sigma = sigma
            self.k0 = k0
            self.V0 = V0
            self.a = a
            self.b = b
            self.t = t
            self.x = np.linspace(0, L, N)
            self.V_flat = self.V0 * np.exp(-((self.x - self.a) ** 2) / (2 * self.b ** 2))

        def Psi(self, t):
            # Define the wave function Psi here
            pass

        def animation(self, output_path='animations'):
            """Create and save animation as GIF"""
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Setup animation same as before
            fig = plt.figure(figsize=(20,12))
            ax = plt.axes(xlim=(0, self.L), ylim=(-0.25, 0.25))
            line, = ax.plot([], [], lw=2)
            ax.plot(self.x[1:-1], self.V_flat, label='$V(x)$')
            ax.set_title('Gaussian wave packet with a potential barrier', fontsize=20)
            line1, = ax.plot(self.x[1:-1], np.zeros(self.N-1), lw=2, color="red", label='$\Re(\psi)$')
            line2, = ax.plot(self.x[1:-1], np.zeros(self.N-1), lw=2, color="blue", label='$\Im(\psi)$')
            ax.legend(fontsize=15)
            ax.set_xlabel('$x$', fontsize=15)

            def animate(t):
                y1 = np.real(self.Psi(t))
                y2 = np.imag(self.Psi(t))
                line1.set_data(self.x[1:-1], y1)  
                line2.set_data(self.x[1:-1], y2)
                return (line1, line2,)

            def init():
                line1.set_data([], [])  
                line2.set_data([], [])
                return (line1, line2,)

            # Create animation
            ani = FuncAnimation(fig, animate, len(self.t), 
                              init_func=init,
                              interval=20, blit=True)

            # Save as GIF
            writer = PillowWriter(fps=30)
            ani.save(os.path.join(output_path, 'wavepacket.gif'), writer=writer)
            plt.close()

    # Usage
wavepacket = Gaussian_Wave(750, 750, 420, 0.1, 50, 100, 0.4, 15, np.linspace(0., 2500, 1000))
wavepacket.animation()  # Saves as 'animations/wavepacket.gif'