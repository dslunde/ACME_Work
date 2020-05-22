# iterative_solvers.py
"""Volume 1A: Iterative Solvers.
Darren Lund
Math 321
10/25/16
"""

from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as ssla
from random import randint
from matplotlib import pyplot as plt
import numpy as np
import time

# Helper function
def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant nxn matrix.

    Inputs:
        n (int): the dimension of the system.
        num_entries (int): the number of nonzero values. Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): An nxn strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in xrange(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in xrange(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1
    return A


# Problems 1 and 2
def jacobi_method(A, b, tol=1e-8, maxiters=100, plot=False):
    """Calculate the solution to the system Ax = b voa the Jacobi Method.

    Inputs:
        A ((n,n) ndarray): A square matrix.
        b ((n,) ndarray): A vector of length n.
        tol (float, opt): the convergence tolerance.
        maxiters (int, opt): the maximum number of iterations to perform.
        plot (bool, opt): if True, plot the convergence rate of the algorithm.
            (this is for Problem 2).

    Returns:
        x ((n,) ndarray): the solution to system Ax = b.
    """
    #raise NotImplementedError("Problem 1 Incomplete")
    D = np.diag(A)
    if 0 in D :
        raise ValueError("A cannot have 0s on the diagonal.")
    else :
        d = float(1)/D
        xnew = np.zeros_like(d)
        xold = 1000*d
        N = 0
        error = []
        while la.norm(xold-xnew,ord=np.inf) >= tol and N < maxiters :
            xold = xnew
            xnew = xold + d*(b-np.dot(A,xold))
            error.append(la.norm(xold-xnew,ord=np.inf))
            N += 1
        if plot :
            n = np.linspace(1,N,N)
            plt.semilogy(n,error)
            plt.axis([0,N+1,0,10])
            plt.xlabel("# of Iterations")
            plt.ylabel("Error of Approximation")
            plt.title("Convergence of the Jacobi Method")
            plt.show()
        if np.allclose(np.dot(A,xnew),b) :
            return xnew
        else :
            raise RuntimeError("Solution did not converge.")


# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiters=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Inputs:
        A ((n,n) ndarray): A square matrix.
        b ((n,) ndarray): A vector of length n.
        tol (float, opt): the convergence tolerance.
        maxiters (int, opt): the maximum number of iterations to perform.
        plot (bool, opt): if True, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): the solution to system Ax = b.
    """
    #raise NotImplementedError("Problem 3 Incomplete")
    D = np.diag(A)
    if 0 in D :
        raise ValueError("A cannot have 0s on the diagonal.")
    else :
        d = float(1)/D
        xnew = np.zeros_like(d)
        xold = 1000*d
        N = 0
        error = []
        while la.norm(xold-xnew,ord=np.inf) >= tol and N < maxiters :
            xold = np.copy(xnew)
            for i in xrange(len(xnew)) :
                xnew[i] = xold[i] + d[i]*(b[i] - np.dot(A[i,:].T,xnew))
            error.append(la.norm(xold-xnew,ord=np.inf))
            N += 1
        if plot :
            n = np.linspace(1,N,N)
            plt.semilogy(n,error)
            plt.axis([0,N+1,0,10])
            plt.xlabel("# of Iterations")
            plt.ylabel("Error of Approximation")
            plt.title("Convergence of the Gauss-Seidel Method")
            plt.show()
        if np.allclose(np.dot(A,xnew),b) :
            return xnew
        else :
            raise RuntimeError("Solution did not converge.")


# Problem 4
def prob4():
    """For a 5000 parameter system, compare the runtimes of the Gauss-Seidel
    method and la.solve(). Print an explanation of why Gauss-Seidel is so much
    faster.
    """
    #raise NotImplementedError("Problem 4 Incomplete")
    N = [5,6,7,8,9,10,11]
    solve_times = []
    gs_times = []
    for n in N :
        A = diag_dom(2**n)
        b = np.random.random(2**n)

        start = time.time()
        la.solve(A,b)
        end = time.time()
        solve_times.append(end-start)

        start = time.time()
        gauss_seidel(A,b)
        end = time.time()
        gs_times.append(end-start)

    for x in xrange(len(N)) :
        N[x] = 2**N[x]
    plt.loglog(N,solve_times,'b.-',label="la.solve",basex=2,basey=2)
    plt.loglog(N,gs_times,'r.-',label="Gauss-Seidel",basex=2,basey=2)
    plt.axis([2**4,2**12,0,2])
    plt.legend(loc="upper left")
    plt.show()

# Problem 5
def sparse_gauss_seidel(A, b, tol=1e-8, maxiters=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Inputs:
        A ((n,n) csr_matrix): An nxn sparse CSR matrix.
        b ((n,) ndarray): A vector of length n.
        tol (float, opt): the convergence tolerance.
        maxiters (int, opt): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): the solution to system Ax = b.
    """
    #raise NotImplementedError("Problem 5 Incomplete")
    D = A.diagonal()
    if 0 in D :
        raise ValueError("A cannot have 0s on the diagonal.")
    else :
        d = float(1)/D
        xnew = np.zeros_like(d)
        xold = 1000*d
        N = 0
        error = []
        while la.norm(xold-xnew,ord=np.inf) >= tol and N < maxiters :
            xold = np.copy(xnew)
            for i in xrange(len(xnew)) :
                rowstart = A.indptr[i]
                rowend = A.indptr[i+1]
                Aix = np.dot(A.data[rowstart:rowend], xnew[A.indices[rowstart:rowend]])
                xnew[i] = xold[i] + d[i]*(b[i] - Aix)
            error.append(la.norm(xold-xnew,ord=np.inf))
            N += 1
        return xnew


# Problem 6
def sparse_sor(A, b, omega, tol=1e-8, maxiters=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Inputs:
        A ((n,n) csr_matrix): An nxn sparse matrix.
        b ((n,) ndarray): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float, opt): the convergence tolerance.
        maxiters (int, opt): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): the solution to system Ax = b.
    """
    #raise NotImplementedError("Problem 6 Incomplete")
    D = A.diagonal()
    if 0 in D :
        raise ValueError("A cannot have 0s on the diagonal.")
    else :
        d = float(1)/D
        xnew = np.zeros_like(d)
        xold = 1000*d
        N = 0
        error = []
        while la.norm(xold-xnew,ord=np.inf) >= tol and N < maxiters :
            xold = np.copy(xnew)
            for i in xrange(len(xnew)) :
                rowstart = A.indptr[i]
                rowend = A.indptr[i+1]
                Aix = np.dot(A.data[rowstart:rowend], xnew[A.indices[rowstart:rowend]])
                xnew[i] = xold[i] + omega*d[i]*(b[i] - Aix)
            error.append(la.norm(xold-xnew,ord=np.inf))
            N += 1
        return xnew


# Problem 7
def finite_difference(n):
    """Return the A and b described in the finite difference problem that
    solves Laplace's equation.
    """
    #raise NotImplementedError("Problem 7 Incomplete")
    diags = [1,1,-4,1,1]
    offsets = [-3,-1,0,1,3]
    A = sparse.diags(diags,offsets,shape=(n**2,n**2))
    b = np.zeros((n**2,1))
    b[0] = -100
    for i in xrange(n,n**2,n) :
        b[i-1] = -100
        b[i] = -100
    b[n**2-1] = -100
    return A,b.reshape(1,-1)[0]


# Problem 8
def compare_omega():
    """Time sparse_sor() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiters = 1000 using the A and b generated by finite_difference()
    with n = 20. Plot the times as a function of omega.
    """
    #raise NotImplementedError("Problem 8 Incomplete")
    A,b = finite_difference(20)
    omega = np.linspace(1,1.95,20)
    times = []
    tol = 1e-2
    maxiters = 1000
    for w in omega :
        start = time.time()
        sparse_sor(A,b,tol,maxiters)
        end = time.time()
        times.append(end-start)
    plt.semilogy(omega,times,'g.-')
    plt.axis([.9,2,0,10**(-2.5)])
    plt.xlabel('Omega')
    plt.ylabel('Time')
    plt.title("SOR Solving Times")
    plt.show()



# Problem 9
def hot_plate(n):
    """Use finite_difference() to generate the system Au = b, then solve the
    system using SciPy's sparse system solver, scipy.sparse.linalg.spsolve().
    Visualize the solution using a heatmap using np.meshgrid() and
    plt.pcolormesh() ("seismic" is a good color map in this case).
    """
    #raise NotImplementedError("Problem 9 Incomplete")
    x = np.linspace(1,n,n)
    y = x
    X,Y = np.meshgrid(x,y)
    A,b = finite_difference(n)
    u = ssla.spsolve(A,b)
    u1 = u.reshape((n,n))
    plt.pcolormesh(X,Y,u1,cmap='seismic')
    plt.show()
    return u

if __name__ == "__main__" :
    #A = np.array([[2,0,-1],[-1,3,2],[0,1,3]])
    #b = np.array([3,3,-1])
    #n = randint(1,100)
    #A = diag_dom(n)
    #b = np.random.random(n)
    #jacobi_method(A,b,plot=True)
    #gauss_seidel(A,b,plot=True)
    #prob4()
    """
    B = diag_dom(n)
    A = sparse.csr_matrix(B)
    b = np.random.random(n)
    print np.allclose(gauss_seidel(B,b),sparse_gauss_seidel(A,b))
    """
    #finite_difference(4)
    #compare_omega()
    hot_plate(20)
