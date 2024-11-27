import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def hw3_rhs_a(y, x, eps):
    return [y[1], (x**2 - eps) * y[0]] 

tol = 1e-4
L = 4
xspan = np.arange(-L, L + 0.1, 0.1)
N = len(xspan)
eps_start = 0.1

A1 = []
A2 = []

col = ['r', 'b', 'g', 'c', 'm']  

for modes in range(5):  
    eps = eps_start  
    deps = 0.2  
    
    for _ in range(1000):  
        y0 = [1, np.sqrt(L**2 - eps)]
        sol = solve_ivp(lambda x, y: hw3_rhs_a(y, x, eps), [xspan[0], xspan[-1]], y0, t_eval=xspan)
        y = sol.y.T

        if abs(y[-1, 1] + np.sqrt(L**2 - eps) * y[-1, 0]) < tol:
            A2.append(eps)
            break
        
        if ((-1) ** modes * (y[-1, 1] + np.sqrt(L**2 - eps) * y[-1, 0])) > 0:
            eps += deps
        else:
            eps -= deps
            deps = deps / 2
    
    eps_start = eps + 0.1
    norm = np.trapz(y[:, 0] * y[:, 0], xspan) 
    eigenfunction = abs(y[:, 0] / np.sqrt(norm))
    A1.append(eigenfunction)
    
A1 = np.column_stack(A1)
A2 = np.array(A2)

print("Eigenfunctions (A1):", A1)
print("Eigenvalues (A2):", A2)


# PART B
import numpy as np
from scipy.sparse.linalg import eigs

L = 4
dx=0.1
xspan = np.arange(-L, L + dx, dx)
n = len(xspan)
tol=1e-4

A = np.zeros((n - 2, n - 2))

for j in range(n - 2):
    A[j, j] = -2 - (dx**2) * xspan[j + 1]**2
    if j < n - 3:
        A[j + 1, j] = 1
        A[j, j + 1] = 1

A[0, 0] = A[0, 0] + 4 / 3
A[0, 1] = A[0, 1] - 1 / 3
A[-1, -1] = A[-1, -1] + 4 / 3
A[-1, -2] = A[-1, -2] - 1 / 3

eigenvalues, eigenvectors = eigs(-A, k=5, which='SM')

V2 = np.vstack([4/3 * eigenvectors[0, :] - 1/3 * eigenvectors[1, :], eigenvectors, 
                4/3 * eigenvectors[-1, :] - 1/3 * eigenvectors[-2, :]])
ysolb = np.zeros((n, 5))
Esolb = np.zeros(5)

for j in range(5):
    norm = np.sqrt(np.trapz(V2[:, j]**2, xspan))
    ysolb[:, j] = np.abs(V2[:, j] / norm)

Esolb = np.sort(eigenvalues[:5] / dx**2)
A3 = ysolb
A4 = Esolb

print(A3)
print(A4)


# PART C
import numpy as np
from scipy.integrate import solve_ivp

def hw3_rhs_c(x, y, E, gamma):
    return [y[1], (gamma * y[0]**2 + x**2 - E) * y[0]]

L = 2
x = np.arange(-L, L + 0.1, 0.1)
n = len(x)
Esolcpos, Esolcneg = np.zeros(2), np.zeros(2)
ysolcpos, ysolcneg = np.zeros((n, 2)), np.zeros((n, 2))

for gamma in [0.05, -0.05]:
    E0, A = 0.1, 1e-6
    for jmode in range(2):
        dA = 0.01
        for jj in range(100):
            E, dE = E0, 0.2
            for j in range(100):
                y0 = [A, np.sqrt(L**2 - E) * A]
                sol = solve_ivp(lambda x, y: hw3_rhs_c(x, y, E, gamma), [x[0], x[-1]], y0, t_eval=x)
                ys = sol.y.T
                xs = sol.t
                
                boundary_cond = ys[-1, 1] + np.sqrt(L**2 - E) * ys[-1, 0]
                if abs(boundary_cond) < tol:
                    break
                
                if (-1)**(jmode) * boundary_cond > 0:
                    E += dE
                else:
                    E -= dE
                    dE /= 2

            area = np.trapz(ys[:, 0]**2, xs)
            if abs(area - 1) < tol:
                break
            if area < 1:
                A += dA
            else:
                A -= dA / 2
                dA /= 2
        
        E0 = E + 0.2
        if gamma > 0:
            Esolcpos[jmode] = E
            ysolcpos[:, jmode] = np.abs(ys[:, 0])
        else:
            Esolcneg[jmode] = E
            ysolcneg[:, jmode] = np.abs(ys[:, 0])

A5 = ysolcpos
A6 = Esolcpos
A7 = ysolcneg
A8 = Esolcneg


print(A5)
print(A6)
print(A7)
print(A8)



# PART D
def hw1_rhs_a(x, y, E):
    return [y[1], (x**2 - E) * y[0]]

L = 2
x_span = [-L, L]  
E = 1
A = 1
y0 = [A, np.sqrt(L**2 - E) * A]  
tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]  

dt45, dt23, dtRadau, dtBDF = [], [], [], []


for tol in tols:
    options = {'rtol': tol, 'atol': tol}

    sol45 = solve_ivp(hw1_rhs_a, x_span, y0, method='RK45', args=(E,), **options)
    sol23 = solve_ivp(hw1_rhs_a, x_span, y0, method='RK23', args=(E,), **options)
    solRadau = solve_ivp(hw1_rhs_a, x_span, y0, method='Radau', args=(E,), **options)
    solBDF = solve_ivp(hw1_rhs_a, x_span, y0, method='BDF', args=(E,), **options)

    dt45.append(np.mean(np.diff(sol45.t)))
    dt23.append(np.mean(np.diff(sol23.t)))
    dtRadau.append(np.mean(np.diff(solRadau.t)))
    dtBDF.append(np.mean(np.diff(solBDF.t)))

fit45 = np.polyfit(np.log(dt45), np.log(tols), 1)
fit23 = np.polyfit(np.log(dt23), np.log(tols), 1)
fitRadau = np.polyfit(np.log(dtRadau), np.log(tols), 1)
fitBDF = np.polyfit(np.log(dtBDF), np.log(tols), 1)


slopes = np.array([fit45[0], fit23[0], fitRadau[0], fitBDF[0]])
A9 = slopes
print(A9)



#PART E
import numpy as np

L = 4
x = np.arange(-L, L + 0.1, 0.1)
n = len(x)

h = np.array([np.ones_like(x),
              2 * x,
              4 * x**2 - 2,
              8 * x**3 - 12 * x,
              16 * x**4 - 48 * x**2 + 12])

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

phi = np.zeros((n, 5))
for i in range(5):
    phi[:,i] = (np.exp(-x**2 / 2) * h[i,:] / 
                (np.sqrt(factorial(i) * 2**i * np.sqrt(np.pi)))).T

A10 = np.zeros(5)
A11 = np.zeros(5)
A12 = np.zeros(5)
A13 = np.zeros(5)

for j in range(5):
    A10[j] = np.trapz((np.abs(A1[:, j]) - np.abs(phi[:, j]))**2, x)
    A12[j] = np.trapz((np.abs(A3[:, j]) - np.abs(phi[:, j]))**2, x)
    A11[j] = 100 * abs(A2[j] - (2 * (j + 1) - 1)) / (2 * (j + 1) - 1)
    A13[j] = 100 * abs(A4[j] - (2 * (j + 1) - 1)) / (2 * (j + 1) - 1)

print(A10)
print(A11)
print(A12)
print(A13)

