import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def shoot2(y, x, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]  


tol = 1e-6 # define a tolerance level 
col = ['r', 'b', 'g', 'c', 'm']  # eigenfunc colors
L = 4  
xshoot = np.arange(-L, L + 0.1, 0.1)  

A1 = [] 
A2 = []

epsilon_start = 0.1  # Beginning value of epsilon
for modes in range(1, 6):  
    epsilon = epsilon_start  # Initial value of eigenvalue
    depsilon = 0.2  # Deefault step size for epsilon
    for _ in range(1000):  
        Y0 = [1, np.sqrt(L**2 - epsilon)]
        y = odeint(shoot2, Y0, xshoot, args=(epsilon,))

        if abs(y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0]) < tol: # Check for convergence
            A2.append(epsilon) 
            break
        
        if ((-1) ** (modes + 1) * (y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0])) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon
            depsilon = depsilon / 2

    epsilon_start = epsilon + 0.1  
    norm = np.trapz(y[:, 0] * y[:, 0], xshoot)  
    func = abs(y[:, 0] / np.sqrt(norm))
    A1.append(func)
    
plt.plot(xshoot, func, col[modes - 1])  
A1 = np.array(A1).T
print(A1)
print(A2)
