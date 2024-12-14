import numpy as np
from numpy.fft import fft2, ifft2
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.linalg import kron
from numpy import *

tspan = np.linspace(0, 4, 9)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

D1 = 0.1
D2 = 0.1
beta = 1
T = 4

# Initial condition
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

m = 1; #number of spirals
u=tanh(sqrt(X**2+Y**2))*cos(m*angle(X+1j*Y)-(sqrt(X**2+Y**2)))
v=tanh(sqrt(X**2+Y**2))*sin(m*angle(X+1j*Y)-(sqrt(X**2+Y**2)))
ut = fft2(u)
vt = fft2(v)
uvt0 = np.hstack([(ut.reshape(N)),(vt.reshape(N))])

def spc_rhs(t, uvt, nx, ny, N, D1, D2, beta, K):
    utc = uvt[0:N]
    vtc = uvt[N:]
    
    utc = utc.reshape((nx, ny))
    vtc = vtc.reshape((nx, ny))
    
    u = ifft2(utc)
    v = ifft2(vtc)
    
    A = u**2 + v**2
    lam = 1 - A
    ome = -beta * A
    
    rhs_u = (-D1 * K * utc + fft2(lam * u - ome * v)).reshape(N)
    rhs_v = (-D2 * K * vtc + fft2(ome * u + lam * v)).reshape(N)
    rhs = np.hstack((rhs_u, rhs_v))
    
    return rhs

uvtsol = solve_ivp(
    spc_rhs,
    [0, T],
    uvt0,
    t_eval=tspan,
    args=(nx, ny, N, D1, D2, beta, K),
    method='RK45'
)
A1 = uvtsol.y
print(A1)

def cheb(N):
    if N == 0: 
        D = 0
        x = 1
    else:
        n = np.arange(0, N+1)
        x = np.cos(np.pi * n/N).reshape(N+1, 1) 
        c = (np.hstack(([2.], np.ones(N-1), [2.]))*(-1)**n).reshape(N+1, 1)
        X = np.tile(x, (1, N+1))
        dX = X - X.T
        D = np.dot(c,1/c.T)/(dX + np.eye(N+1))
        D -= np.diag(sum(D.T, axis=0))
    
    return D, x.reshape(N+1)

# Initial condition
N = 30
D,x = cheb(N)
D[N,:] = 0
D[0,:] = 0
Dxx = np.dot(D,D)/((20 / 2)**2)
y = x
N2 = (N + 1) * (N + 1)
I = np.eye(len(Dxx))
L = kron(I, Dxx) + kron(Dxx, I)
X, Y = np.meshgrid(x, y)
X = X * (20 / 2)
Y = Y * (20 / 2)


m=1 #number of spirals
u=np.tanh(sqrt(X**2+Y**2))*cos(m*angle(X+1j*Y)-(sqrt(X**2+Y**2)))
v=tanh(sqrt(X**2+Y**2))*sin(m*angle(X+1j*Y)-(sqrt(X**2+Y**2)))
uv0 = np.hstack([(u.reshape(N2)), (v.reshape(N2))])


def RD_2D(t, uv, L, N2, N, D1, D2, beta):
    u = uv[0:N2]
    v = uv[N2:]
    A2 = u * u + v * v
    la = 1 - A2
    om = -beta * A2
    rhs_u = D1 * np.dot(L,u) + la * u - om * v
    rhs_v = D2 * np.dot(L,v) + om * u + la * v
    rhs = np.hstack([rhs_u,rhs_v])
    
    return rhs

uvsol = solve_ivp(
    RD_2D,
    [0, T],
    uv0,
    t_eval=tspan,
    args=(L, N2, N, D1, D2, beta),
    method='RK45'
)

A2 = uvsol.y
print(A2)