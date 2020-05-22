# profiling.py
"""Python Essentials: Profiling.
    Bruce Wayne
    League of Shadows
    Today
    """

import numpy as np
from scipy import linalg as la
import time
from numba import jit


# Problem 1
def compare_timings(f, g, *args):
    """Compare the timings of 'f' and 'g' with arguments '*args'.
        
        Inputs:
        f (func): first function to compare.
        g (func): second function to compare.
        *args: arguments to use when callings functions 'f' and 'g',
        i.e., call f with f(*args).
        Returns:
        comparison (str): The comparison of the runtimes of functions
        'f' and 'g' in the following format:
        Timing for <f>: <time>
        Timing for <g>: <time>
        """
#raise NotImplementedError("Problem 1 Incomplete")
    start = time.time()
    f(*args)
    end = time.time()
    f_time = end-start
    start = time.time()
    g(*args)
    end = time.time()
    g_time = end-start
    print "Timing for " + str(f) + ": " + str(f_time)
    print "Timing for " + str(g) + ": " + str(g_time)


# Problem 2
def LU(A):
    """Return the LU decomposition of a square matrix."""
    n = A.shape[0]
    U = np.array(np.copy(A), dtype=float)
    L = np.eye(n)
    for i in range(1, n):
        for j in range(i):
            L[i,j] = U[i,j] / U[j,j]
            for k in range(j, n):
                U[i,k] -= L[i,j] * U[j,k]
    return L, U

def LU_opt(A):
    """Return the LU decomposition of a square matrix."""
#raise NotImplementedError("Problem 2 Incomplete")
    n = A.shape[0]
    A = A.astype(float)
    L = np.eye(n)
    for i in xrange(n-1) :
        for j in xrange(i+1,n) :
            L[j,i] = A[j,i]/A[i,i]
            A[j,i:] = A[j,i:]-L[j,i]*A[i,i:]
            A[j,i] = 0
    """
    for i in range(1, n):
        for j in range(i):
            L[i,j] = A[i,j] / A[j,j]
            for k in range(j, n):
                A[i,k] -= L[i,j] * A[j,k]
    """
    return L, A

def compare_LU(A):
    """Prints a comparison of LU and LU_opt with input of a square matrix A."""
#raise NotImplementedError("Problem 2 Incomplete")
    print "Comparison of LU and LU_opt:"
    compare_timings(LU,LU_opt,A)

# Problem 3
def mysum(x):
    """Return the sum of the elements of X without using a built-in function.
        
        Inputs:
        x (iterable): a list, set, 1-d NumPy array, or another iterable.
        """
#raise NotImplementedError("Problem 3 Incomplete")
    tot = 0.
    for i in x :
        tot += i
    return float(tot)

def compare_sum(X):
    """
        Inputs:
        x (iterable): a list, set, 1-d NumPy array, or another iterable.
        
        Prints a comparison of mysum and sum
        Prints a comparison of mysum and np.sum
        """
#raise NotImplementedError("Problem 3 Incomplete")
    print "Comparison of mysum and sum:"
    compare_timings(mysum,sum,X)
    print "\nComparison of mysum and np.sum:"
    compare_timings(mysum,np.sum,X)

# Problem 4
def fibonacci(n):
    """Yield the first n Fibonacci numbers."""
#raise NotImplementedError("Problem 4 Incomplete")
    if n <= 0 or int(n) != n :
        raise ValueError("n must be a natural number (integer s.t. n>0).")
    else :
        yield 1
        if n > 1 :
            n1 = 0
            n2 = 0
            n3 = 1
            for i in xrange(n-1) :
                n1 = n2
                n2 = n3
                n3 = n1+n2
                yield n3


# Problem 5
def foo(n):
    my_list = []
    for i in range(n):
        num = np.random.randint(-9,9)
        my_list.append(num)
    evens = 0
    for j in range(n):
        if my_list[j] % 2 == 0:
            evens += my_list[j]
    return my_list, evens

def foo_opt(n):
    """An optimized version of 'foo'"""
#raise NotImplementedError("Problem 5 Incomplete")
    randint = np.random.randint
    my_list = [randint(-9,9) for i in xrange(n)]
    evens = sum([my_list[i]*(my_list[i]%2) for i in xrange(n)])
    return my_list,evens

def compare_foo(n):
    """Prints a comparison of foo and foo_opt"""
#raise NotImplementedError("Problem 5 Incomplete")
    print "Comparison of foo and foo_opt:"
    compare_timings(foo,foo_opt,n)


# Problem 6
def pymatpow(X, power):
    """Return X^{power}, the matrix product XX...X, 'power' times.
        
        Inputs:
        X ((n,n) ndarray): A square matrix.
        power (int): The power to which to raise X.
        """
    prod = X.copy()
    temparr = np.empty_like(X[0])
    size = X.shape[0]
    for n in xrange(1, power):
        for i in xrange(size):
            for j in xrange(size):
                tot = 0.
                for k in xrange(size):
                    tot += prod[i,k] * X[k,j]
                temparr[j] = tot
            prod[i] = temparr
    return prod

@jit(nopython=True)
def numba_matpow(X, power):
    """ Return X^{power}.
        
        Inputs:
        X (ndarray):  A square 2-D NumPy array
        power (int):  The power to which to raise X.
        Returns:
        prod (ndarray):  X^{power}
        """
#raise NotImplementedError("Problem 6 Incomplete")
    prod = X.copy()
    temparr = np.empty_like(X[0])
    size = X.shape[0]
    for n in xrange(1, power):
        for i in xrange(size):
            for j in xrange(size):
                tot = 0.
                for k in xrange(size):
                    tot += prod[i,k] * X[k,j]
                temparr[j] = tot
            prod[i] = temparr
    return prod

def numpy_matpow(X, power):
    """ Return X^{power}.
        
        Inputs:
        X (ndarray):  A square 2-D NumPy array
        power (int):  The power to which to raise X.
        Returns:
        prod (ndarray):  X^{power}
        """
#raise NotImplementedError("Problem 6 Incomplete")
    Y = np.copy(X)
    for i in xrange(power) :
        Y = np.dot(Y,X)
    return Y

def compare_matpow(X, power):
    """
        Inputs:
        X (ndarray):  A square 2-D NumPy array
        power (int):  The power to which to raise X.
        
        Prints a comparison of pymatpow and numba_matpow
        Prints a comparison of pymatpow and numpy_matpow
        """
#raise NotImplementedError("Problem 6 Incomplete")
    numba_matpow(X,power)
    print "Comparison of pymatpow and numba_matpow:"
    compare_timings(pymatpow,numba_matpow,X,power)
    print "\nComparison of pymatpow and numpy_matpow:"
    compare_timings(pymatpow,numpy_matpow,X,power)


# Problem 7
def init_tridiag(n):
    """Construct a random nxn tridiagonal matrix A by diagonals.
        
        Inputs:
        n (int): The number of rows / columns of A.
        
        Returns:
        a ((n-1,) ndarray): first subdiagonal of A.
        b ((n,) ndarray): main diagonal of A.
        c ((n-1,) ndarray): first superdiagonal of A.
        A ((n,n) ndarray): the tridiagonal matrix.
        """
    a = np.random.random_integers(-9, 9, n-1).astype("float")
    b = np.random.random_integers(-9 ,9, n  ).astype("float")
    c = np.random.random_integers(-9, 9, n-1).astype("float")
    
    # Replace any zeros with ones.
    a[a==0] = 1
    b[b==0] = 1
    c[c==0] = 1
    
    # Construct the matrix A.
    A = np.zeros((b.size,b.size))
    np.fill_diagonal(A, b)
    np.fill_diagonal(A[1:,:-1], a)
    np.fill_diagonal(A[:-1,1:], c)
    
    return a, b, c, A

def pytridiag(a, b, c, d):
    """Solve the tridiagonal system Ax = d where A has diagonals a, b, and c.
        
        Inputs:
        a ((n-1,) ndarray): first subdiagonal of A.
        b ((n,) ndarray): main diagonal of A.
        c ((n-1,) ndarray): first superdiagonal of A.
        d ((n,) ndarray): the right side of the linear system.
        
        Returns:
        x ((n,) ndarray): solution to the tridiagonal system Ax = d.
        """
    n = len(b)
    
    # Make copies so the original arrays remain unchanged.
    aa = np.copy(a)
    bb = np.copy(b)
    cc = np.copy(c)
    dd = np.copy(d)
    
    # Forward sweep.
    for i in xrange(1, n):
        temp = aa[i-1] / bb[i-1]
        bb[i] = bb[i] - temp*cc[i-1]
        dd[i] = dd[i] - temp*dd[i-1]
    
    # Back substitution.
    x = np.zeros_like(b)
    x[-1] = dd[-1] / bb[-1]
    for i in reversed(xrange(n-1)):
        x[i] = (dd[i] - cc[i]*x[i+1]) / bb[i]
    
    return x

@jit(nopython=True, locals=dict(a=double[:,1],b=double[:,1],c=double[:,1],d=double[:,1],
                                n=int32,temp=double,x=double[:,1]))
def numba_tridiag(a, b, c, d):
    """Solve the tridiagonal system Ax = d where A has diagonals a, b, and c.
        
        Inputs:
        a ((n-1,) ndarray): first subdiagonal of A.
        b ((n,) ndarray): main diagonal of A.
        c ((n-1,) ndarray): first superdiagonal of A.
        d ((n,) ndarray): the right side of the linear system.
        
        Returns:
        x ((n,) ndarray): solution to the tridiagonal system Ax = d.
        """
#raise NotImplementedError("Problem 7 Incomplete")
    n = len(b)
    
    # Forward sweep.
    for i in xrange(1, n):
        temp = a[i-1] / b[i-1]
        b[i] = b[i] - temp*c[i-1]
        d[i] = d[i] - temp*d[i-1]

    # Back substitution.
    x = np.zeros_like(b)
    x[-1] = d[-1] / b[-1]
    for i in reversed(xrange(n-1)):
        x[i] = (d[i] - c[i]*x[i+1]) / b[i]

    return x

def compare_tridiag():
    """Prints a comparison of numba_tridiag and pytridiag
        prints a comparison of numba_tridiag and scipy.linalg.solve."""
#raise NotImplementedError("Problem 7 Incomplete")
    print "Comparison of numba_tridiag and pytriadiag:"
    compare_timings(numba_tridiage,pytridiag,a,b,c,d)
    """
    print "Comparison of numba_tridiag and scipylinalg.solve:"
    compare_timings(numba_tridiag,la.solve,a,b,c,d)
    """
