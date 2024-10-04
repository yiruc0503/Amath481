import numpy as np

#Question I
#Part one:  Newton-Raphson method

x = np.array([-1.6]) # initial guess
for j in range(1000):
    x = np.append(
        x, x[j]-(x[j]*np.sin(3 * x[j])-np.exp(x[j]))
        / (np.sin(3 * x[j])+3 * x[j] * np.cos(3 * x[j]) - np.exp(x[j])))
    fc = x[j + 1] * np.sin(3 * x[j + 1]) - np.exp(x[j + 1])

    if abs(fc) < 1e-6:
        break
A1 = x
iter_n = j+1
print(A1)

#Part two: bisection
def f(x):
    return x * np.sin(3 * x) - np.exp(x)
xr = -0.4
xl = -0.7
mid_value = []

for j in range(100):
    xc = (xr + xl) / 2
    mid_value.append(xc)
    fc = f(xc)
    
    if abs(fc) < 1e-6:
        break
    if f(xl) * fc < 0:
        xr = xc
    else:
        xl = xc

A2 = mid_value
iter_b = j+1
print(A2)

A3 = np.array([iter_n, iter_b])
print(A3)



##Question II
A = np.array([[1,2],[-1,1]])
B = np.array([[2,0],[0,2]])
C = np.array([[2,0,-3],[0,0,-1]])
D = np.array([[1,2],[2,3],[-1,0]])
x = np.array([1,0])
y = np.array([0,1])
z = np.array([1,2,-1])

A4 = A+B
A5 = 3*x-4*y
A6 = np.matmul(A,x)
A7 = np.matmul(B,x-y)
A8 = np.matmul(D,x)
A9 = np.matmul(D,y)+z
A10 = np.matmul(A,B)
A11 = np.matmul(B,C)
A12 = np.matmul(C,D)



