import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

#PART A
m = 8    # N value in x and y directions
n = m * m  # total size of matrix
dx=20/m

e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements
diagonals_A = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets_A = [-(n - m), -m, -m + 1, -1, 0, 1, m - 1, m, (n - m)]
matA = spdiags(diagonals_A, offsets_A, n, n).toarray()
A1 = matA/(dx**2)

diagonals_B = [e1.flatten(), -e1.flatten(),e1.flatten(), -e1.flatten()]
offsets_B = [-(n-m), -m, m, (n-m)]
matB = (spdiags(diagonals_B, offsets_B, n, n).toarray())
A2 = matB/(2*dx)

diagonals_C = [e5.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets_C = [-m+1,-1,1,m-1]
matC = (spdiags(diagonals_C, offsets_C, n, n).toarray())
A3 = matC/(2*dx)

print(f"A: {A1}")
print(f"B: {A2}")
print(f"C: {A3}")
