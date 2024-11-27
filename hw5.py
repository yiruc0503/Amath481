import numpy as np
from numpy.fft import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags
from scipy.linalg import lu, solve_triangular
import matplotlib.pyplot as plt
from scipy.sparse.linalg import bicgstab
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import os

# Part a
# Parameters
tspan = np.linspace(0, 4, 9)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

# Initial condition
w = np.exp(-X**2 - Y**2/20)
w2 = w.reshape(N)

kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
kx[0], ky[0] = 1e-6, 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

# Set up matrices
m = 64
n = m * m
x = np.linspace(-10, 10, m+1)
x = x[:m]
dx = x[1] - x[0]

# Create vectors
e0 = np.zeros((n, 1))
e1 = np.ones((n, 1))
e2 = np.copy(e1)
e4 = np.copy(e0)

for j in range(1, m+1):
    e2[m*j-1] = 0
    e4[m*j-1] = 1

e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Build matrices
diagonals_A = [
    e1.flatten(), e1.flatten(), e5.flatten(),
    e2.flatten(), -4 * e1.flatten(), e3.flatten(),
    e4.flatten(), e1.flatten(), e1.flatten()
]
offsets_A = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]
matA = spdiags(diagonals_A, offsets_A, n, n).toarray()
A = matA/(dx**2)

diagonals_B = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets_B = [-(n-m), -m, m, (n-m)]
matB = spdiags(diagonals_B, offsets_B, n, n).toarray()
B = matB/(2*dx)

diagonals_C = [e5.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets_C = [-m+1, -1, 1, m-1]
matC = spdiags(diagonals_C, offsets_C, n, n).toarray()
C = matC/(2*dx)

A[0,0] = 2/(dx**2)


def spc_rhs(t, w2, nx, ny, N, KX, KY, K, nu):
    
    w = w2.reshape((nx, ny))
    
    wt = fft2(w)
    psi_hat = -wt / K
    psi = np.real(ifft2(psi_hat)).flatten() 
    
    psix = np.dot(B, psi)
    psiy = np.dot(C, psi)
    wx = np.dot(B, w2)
    wy = np.dot(C, w2)
    
    rhs = nu * np.dot(A, w2) - psix * wy + psiy * wx
    
    return rhs

start_time = time.time()

sol = solve_ivp(
    spc_rhs,
    [0, 4],
    w2,
    t_eval=tspan,
    args=(nx, ny, N, KX, KY, K, nu),
    method='RK45'
)

A1 = sol.y

print(f"A1 first value: {A1[0,0]:.16e} last value: {A1[-1,-1]:.16e}")
print(f"A1 shape: {A1.shape}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Part b
def Ab_rhs(t, w2, nx, ny, N, A, B, C, K, nu):
    psi = np.linalg.solve(A, w2)
    psix = np.dot(B, psi)
    psiy = np.dot(C, psi)
    wx = np.dot(B, w2)
    wy = np.dot(C, w2)
    
    rhs = nu * np.dot(A, w2) - psix * wy + psiy * wx
    
    return rhs

start_time = time.time() 

Ab_sol = solve_ivp(
    Ab_rhs, 
    [0, 4],
    w2, 
    t_eval=tspan,
    args=(nx, ny, N, A, B, C, K, nu), 
    method="RK45"
)

A2 = Ab_sol.y
print(f"A2 first value: {A2[0,0]:.16e} last value: {A2[-1,-1]:.16e}")
print(f"A2 shape: {A2.shape}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")



def LU_rhs(t, w2, nx, ny, N, A, B, C, K, nu):
    Pb = np.dot(P, w2)
    y = solve_triangular(L, Pb, lower=True)
    psi = solve_triangular(U, y)
    psix = np.dot(B, psi)
    psiy = np.dot(C, psi)
    wx = np.dot(B, w2)
    wy = np.dot(C, w2)

    rhs = nu * np.dot(A, w2) - psix * wy + psiy * wx
    
    return rhs

start_time = time.time() 

P, L, U = lu(A)
LU_sol = solve_ivp(
    LU_rhs,
    [0, 4], 
    w2, 
    t_eval=tspan,
    args=(nx, ny, N, A, B, C, K, nu), 
    method="RK45"
)
A3 = LU_sol.y

print(f"A3 first value: {A3[0,0]:.16e} last value: {A3[-1,-1]:.16e}")
print(f"A3 shape: {A3.shape}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")


def BICGSTAB_rhs(t, w2, nx, ny, N, A, B, C, K, nu):
    A_sparse = csr_matrix(A)
    psi, info = bicgstab(A_sparse, w2)
    
    psix = np.dot(B, psi)
    psiy = np.dot(C, psi)
    wx = np.dot(B, w2)
    wy = np.dot(C, w2)
    
    rhs = nu * np.dot(A, w2) - psix * wy + psiy * wx
    return rhs

start_time = time.time() 

BICGSTAB_sol = solve_ivp(
    BICGSTAB_rhs, 
    [0, 4],
    w2, 
    t_eval=tspan,
    args=(nx, ny, N, A, B, C, K, nu), 
    method="RK45"
)

A4 = BICGSTAB_sol.y
print(f"A4 first value: {A4[0,0]:.16e} last value: {A4[-1,-1]:.16e}")
print(f"A4 shape: {A4.shape}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")


def GMRES_rhs(t, w2, nx, ny, N, A, B, C, K, nu):
    A_sparse = csr_matrix(A)
    psi, info = gmres(A_sparse, w2)
    
    psix = np.dot(B, psi)
    psiy = np.dot(C, psi)
    wx = np.dot(B, w2)
    wy = np.dot(C, w2)
    
    rhs = nu * np.dot(A, w2) - psix * wy + psiy * wx
    return rhs

start_time = time.time() 

GMRES_sol = solve_ivp(
    GMRES_rhs, 
    [0, 4],
    w2, 
    t_eval=tspan,
    args=(nx, ny, N, A, B, C, K, nu), 
    method="RK45"
)

A5 = GMRES_sol.y
print(f"A5 first value: {A5[0,0]:.16e} last value: {A5[-1,-1]:.16e}")
print(f"A5 shape: {A5.shape}")
end_time = time.time() 
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Part C

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2

# Domain setup
Lx, Ly = 20, 20  # Domain size
nx, ny = 64, 64  # Grid points
x = np.linspace(-Lx/2, Lx/2, nx, endpoint=False)
y = np.linspace(-Ly/2, Ly/2, ny, endpoint=False)
X, Y = np.meshgrid(x, y)
dx = Lx / nx
dy = Ly / ny
N = nx * ny

# Solver setup
nu = 0.001
tspan_c = np.linspace(0, 10, 21)  # Time steps for question (c)

# Define RHS function for vorticity evolution using FFT
def spc_rhs(t, w2, nx, ny, N, K, nu):
    w = w2.reshape((ny, nx))
    wt = fft2(w)
    psi_hat = -wt / K
    psi = np.real(ifft2(psi_hat)).flatten()
    
    # Compute gradients using central differences
    wx = (np.roll(w, -1, axis=1) - np.roll(w, 1, axis=1)) / (2 * dx)
    wy = (np.roll(w, -1, axis=0) - np.roll(w, 1, axis=0)) / (2 * dy)
    psix = (np.roll(psi.reshape((ny, nx)), -1, axis=1) - np.roll(psi.reshape((ny, nx)), 1, axis=1)) / (2 * dx)
    psiy = (np.roll(psi.reshape((ny, nx)), -1, axis=0) - np.roll(psi.reshape((ny, nx)), 1, axis=0)) / (2 * dy)
    
    # Compute the RHS of the vorticity equation
    diffusion = nu * ((np.roll(w, -1, axis=1) - 2 * w + np.roll(w, 1, axis=1)) / dx**2 +
                      (np.roll(w, -1, axis=0) - 2 * w + np.roll(w, 1, axis=0)) / dy**2)
    advection = - (psix * wy - psiy * wx)
    rhs = diffusion + advection

    # Flatten and return
    return rhs.flatten()

# Wave numbers for FFT
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx//2), np.arange(-nx//2, 0)))
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny//2), np.arange(-ny//2, 0)))
kx[0], ky[0] = 1e-6, 1e-6  # Avoid division by zero
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

# Scenario 1: Two Oppositely Charged Gaussian Vortices
tspan1 = np.linspace(0, 10, 21)
w1 = (np.exp(-((X - 2) ** 2) - Y ** 2 / 5) - np.exp(-((X + 2) ** 2) - Y ** 2 / 5)).flatten()
solution1 = solve_ivp(
    spc_rhs,
    [0, 10],
    w1,
    t_eval=tspan1,
    args=(nx, ny, N, K, nu),
    method="RK45"
)

plt.figure(figsize=(15, 15))
for idx, t in enumerate(tspan1):
    w = solution1.y[:, idx].reshape((ny, nx))
    plt.subplot(7, 3, idx + 1)
    plt.pcolormesh(x, y, w, shading='auto', cmap='plasma')
    plt.title(f'Time: {t:.2f}')
    plt.colorbar()
plt.tight_layout()
plt.show()

# Scenario 2: Two Same Charged Gaussian Vortices
tspan2 = np.linspace(0, 4, 9)
w2 = (np.exp(-((X - 2) ** 2) - Y ** 2 / 5) + np.exp(-((X + 2) ** 2) - Y ** 2 / 5)).flatten()
solution2 = solve_ivp(
    spc_rhs,
    [0, 4],
    w2,
    t_eval=tspan2,
    args=(nx, ny, N, K, nu),
    method="RK45"
)

plt.figure(figsize=(10, 10))
for idx, t in enumerate(tspan2):
    w = solution2.y[:, idx].reshape((ny, nx))
    plt.subplot(3, 3, idx + 1)
    plt.pcolormesh(x, y, w, shading='auto', cmap='viridis')
    plt.title(f'Time: {t:.2f}')
    plt.colorbar()
plt.tight_layout()
plt.show()

# Scenario 3: Two Pairs of Oppositely Charged Vortices
tspan3 = np.linspace(0, 4, 9)
# Defining the initial vorticity for colliding pairs
w3 = (np.exp(-((X + 5)**2 + (Y + 5)**2)) - np.exp(-((X - 5)**2 + (Y + 5)**2))
      - np.exp(-((X + 5)**2 + (Y - 5)**2)) + np.exp(-((X - 5)**2 + (Y - 5)**2))).flatten()
solution3 = solve_ivp(
    spc_rhs,
    [0, 4],
    w3,
    t_eval=tspan3,
    args=(nx, ny, N, K, nu),
    method="RK45"
)

plt.figure(figsize=(10, 10))
for idx, t in enumerate(tspan3):
    w = solution3.y[:, idx].reshape((ny, nx))
    plt.subplot(3, 3, idx + 1)
    plt.pcolormesh(x, y, w, shading='auto', cmap='inferno')
    plt.title(f'Time: {t:.2f}')
    plt.colorbar()
plt.tight_layout()
plt.show()

# Scenario 4: Random Assortment of Vortices
tspan4 = np.linspace(0, 4, 9)
num_vortices = 15
omega_total = np.zeros((ny, nx))

np.random.seed(42)
for i in range(num_vortices):
    charge = np.random.choice([-1, 1])
    intensity = np.random.uniform(0.5, 1.5)
    x0 = np.random.uniform(-Lx/2, Lx/2)
    y0 = np.random.uniform(-Ly/2, Ly/2)
    sigma_x = np.random.uniform(0.5, 1.5)
    sigma_y = np.random.uniform(0.5, 1.5)
    
    r_squared = ((X - x0) / sigma_x) ** 2 + ((Y - y0) / sigma_y) ** 2
    omega = charge * intensity * np.exp(-r_squared / 2)
    
    omega_total += omega

w4 = omega_total.flatten()
solution4 = solve_ivp(
    spc_rhs,
    [0, 4],
    w4,
    t_eval=tspan4,
    args=(nx, ny, N, K, nu),
    method="RK45"
)

plt.figure(figsize=(10, 10))
for idx, t in enumerate(tspan4):
    w = solution4.y[:, idx].reshape((ny, nx))
    plt.subplot(3, 3, idx + 1)
    plt.pcolormesh(x, y, w, shading='auto', cmap='cividis')
    plt.title(f'Time: {t:.2f}')
    plt.colorbar()
plt.tight_layout()
plt.show()








# Part d

# Function to create animations
def create_animation(solution, X, Y, tspan, title_prefix, cmap, filename):
    # Reshape the solution array
    W = solution.y.reshape((ny, nx, -1))  # W[:,:,i] is the vorticity at time tspan[i]
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Set up the initial plot
    levels = np.linspace(np.min(W), np.max(W), 50)
    contour = ax.contourf(X, Y, W[:, :, 0], levels=levels, cmap=cmap)
    fig.colorbar(contour)
    ax.set_title(f'{title_prefix} at t = {tspan[0]:.2f}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Animation function
    def animate(i):
        ax.clear()
        contour = ax.contourf(X, Y, W[:, :, i], levels=levels, cmap=cmap)
        ax.set_title(f'{title_prefix} at t = {tspan[i]:.2f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return contour.collections
    
    # Create the animation
    anim = animation.FuncAnimation(fig, animate, frames=W.shape[2], interval=200)
    
    # Save the animation as a GIF
    anim.save(filename, writer='pillow')
    
    # Display the animation
    plt.close(fig)  # Close the figure to prevent it from displaying in static environments
    print(f'Animation saved as {filename}')

# Example 1: Two Oppositely Charged Gaussian Vortices
create_animation(
    solution=solution1,
    X=X,
    Y=Y,
    tspan=tspan1,
    title_prefix='Opposite Charged Vortices',
    cmap='plasma',
    filename='opposite_charged_vortices.gif'
)

# Example 2: Two Same Charged Gaussian Vortices
create_animation(
    solution=solution2,
    X=X,
    Y=Y,
    tspan=tspan2,
    title_prefix='Same Charged Vortices',
    cmap='viridis',
    filename='same_charged_vortices.gif'
)

# Example 3: Two Pairs of Oppositely Charged Vortices
create_animation(
    solution=solution3,
    X=X,
    Y=Y,
    tspan=tspan3,
    title_prefix='Colliding Vortex Pairs',
    cmap='inferno',
    filename='colliding_vortex_pairs.gif'
)

# Example 4: Random Assortment of Vortices
create_animation(
    solution=solution4,
    X=X,
    Y=Y,
    tspan=tspan4,
    title_prefix='Random Vortices',
    cmap='cividis',
    filename='random_vortices.gif'
)
