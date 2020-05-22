# linear_systems.py
"""Volume 1A: Linear Systems.
Darren Lund
Lab 1
10/4/16
"""

from matplotlib import pyplot as plt
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla
import numpy as np
import random
import time

# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.
    """
    A = A.astype(float)
    n = A[0,:].size
    for j in xrange(n) :
        row1 = A[j,j:]
        for i in xrange(j+1,n) :
            row2 = A[i,j:]
            if row2[0] != 0 and row1[0] != 0:
                A[i,j:] = row2 - row2[0]/float(row1[0])*row1
                A[i,j] = 0
    return A
    #raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may assume the
    decomposition exists and requires no row swaps.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    U = A.astype(float)
    n = U[0,:].size
    L = np.identity(n)
    L = L.astype(float)
    for j in xrange(n) :
        u1 = U[j,j:]
        for i in xrange(j+1,n) :
            u2 = U[i,j:]
            l = L[i,:j+1]
            l = l.astype(float)
            if u2[0] != 0 and u1[0] != 0:
                num = u2[0]
                den = u1[0]
                U[i,j:] = u2 - u2[0]/float(u1[0])*u1
                U[i,j] = 0
                L[i:i+1,j:j+1] = num/float(den)
    return L , U
    #raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may assume that A is invertible (hence square).
    """
    L,U = lu(A)
    b = b.astype(float)
    y = np.zeros_like(b)
    x = np.zeros_like(b)
    n = L[0,:].size
    for i in xrange(n) :
        y[i] = b[i] - sum([y[j]*float(L[i:i+1,j:j+1]) for j in xrange(i)])
    for i in xrange(n-1,-1,-1) :
        x[i] =  (y[i] - sum([x[j]*float(U[i:i+1,j:j+1])
                            for j in xrange(i,n)]))/float(U[i:i+1,i:i+1])
    return x
    #raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.
    Plot the system size versus the execution times. Use log scales if needed.
    """
    N = np.linspace(1,32,32)
    Inv = []
    Solve = []
    LU_factor = []
    LU_Solve = []
    for n in xrange(1,33) :
        A = np.random.random((n,n))
        B = np.random.random(n)

        start = time.time()
        Ainv = la.inv(A)
        np.dot(Ainv,B)
        end = time.time()
        Inv.append(end-start)

        start = time.time()
        solve = la.solve(A,B)
        end = time.time()
        Solve.append(end-start)

        start = time.time()
        lu,piv = la.lu_factor(A)
        start1 = time.time()
        la.lu_solve((lu,piv),B)
        end = time.time()
        end1 = time.time()
        LU_factor.append(end-start)
        LU_Solve.append(end1-start1)

    plt.loglog(N,Inv,'b.-', basex=2, basey=2, label="la.inv")
    plt.loglog(N,Solve,'g.-', basex=2, basey=2, label="la.solve")
    plt.loglog(N,LU_factor,'r.-', basex=2, basey=2, label="lu-factor")
    plt.loglog(N,LU_Solve,'k.-', basex=2, basey=2, label="lu-solve")
    plt.xlabel("n")
    plt.ylabel("Time")
    plt.axis([0,32,0,2**(-8)])
    plt.legend(loc="upper left")
    plt.show()
    #raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def prob5(n):
    """Return a sparse n x n tridiagonal matrix with 2's along the main
    diagonal and -1's along the first sub- and super-diagonals.
    """
    diagonals = [-1,2,-1]
    offsets = [-1,0,1]
    return sparse.diags(diagonals,offsets,shape=(n,n))
    #raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """Time regular and sparse linear system solvers. Plot the system size
    versus the execution times. As always, use log scales where appropriate.
    """
    N = np.linspace(3,32,30)
    csr = []
    npsolve = []
    for n in xrange(3,33) :
        A = prob5(n)
        B = np.random.random(n)

        start = time.time()
        Acsr = A.tocsr()
        spla.spsolve(Acsr,B)
        end = time.time()
        csr.append(end-start)

        start = time.time()
        numarray = A.toarray()
        la.solve(numarray,B)
        end = time.time()
        npsolve.append(end-start)

    plt.loglog(N,csr,'b.-',basex=2,basey=2,label="SPsolve")
    plt.loglog(N,npsolve,'g.-',basex=2,basey=2,label="Solve")
    plt.xlabel("n")
    plt.ylabel("Time")
    plt.axis([0,32,0,2**(-8)])
    plt.legend(loc="upper left")
    plt.show()
    #raise NotImplementedError("Problem 6 Incomplete")

if __name__ == "__main__" :
    """
    A = np.random.rand(5,5)
    print A
    ref(A)
    print A
    L,U = lu(A)
    print L
    print U
    prob4()
    A = prob5(10)
    print(A.toarray())
    """
    prob4()
