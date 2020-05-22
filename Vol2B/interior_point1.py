# Name this file 'solutions.py'.
"""Volume 2 Lab 19: Interior Point 1 (Linear Programs)
    I'm a potatoe
    This is a title
    Why? I don't know.
    """

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt
import pandas as pd


# Auxiliary Functions ---------------------------------------------------------
def startingPoint(A, b, c):
    """Calculate an initial guess to the solution of the linear program
        min c^T x, Ax = b, x>=0.
        Reference: Nocedal and Wright, p. 410.
        """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A.dot(A.T))
    x = A.T.dot(B.dot(b))
    lam = B.dot(A.dot(c))
    mu = c - A.T.dot(lam)
    
    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)
    
    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)
    
    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(m):
    """Generate a 'square' linear program min c^T x s.t. Ax = b, x>=0.
        First generate m feasible constraints, then add slack variables.
        Inputs:
        m -- positive integer: the number of desired constraints
        and the dimension of space in which to optimize.
        Outputs:
        A -- array of shape (m,n).
        b -- array of shape (m,).
        c -- array of shape (n,).
        x -- the solution to the LP.
        """
    n = m
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(n)*10
    b = A.dot(x)
    c = A.sum(axis=0)/float(n)
    return A, b, -c, x

# This random linear program generator is more general than the first.
def randomLP2(m,n):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
        First generate m feasible constraints, then add
        slack variables to convert it into the above form.
        Inputs:
        m -- positive integer >= n, number of desired constraints
        n -- dimension of space in which to optimize
        Outputs:
        A -- array of shape (m,n+m)
        b -- array of shape (m,)
        c -- array of shape (n+m,), with m trailing 0s
        v -- the solution to the LP
        """
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    v = np.random.random(n)*10
    k = n
    b = np.zeros(m)
    b[:k] = A[:k,:].dot(v)
    b[k:] = A[k:,:].dot(v) + np.random.random(m-k)*10
    c = np.zeros(n+m)
    c[:n] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(m)))
    return A, b, -c, v


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
        using an Interior Point method.
        
        Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.
        
        Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
        """
#raise NotImplementedError("Problems 1-4 Incomplete")
    #Problem 1
    m,n = A.shape
    iters = 0
    sigma = 1./10
    x,lam,mu = startingPoint(A,b,c)
    nu = np.dot(x.T,mu)/float(n)
    F = lambda x,l,u : np.hstack((np.dot(A.T,l)+u-c,np.dot(A,x)-b,np.dot(np.diag(u),x)))
    while iters < niter and np.abs(nu) >= tol :
        #Problem 2
        D1 = np.hstack((np.zeros((n,n)),A.T,np.eye(n,n)))
        D2 = np.hstack((A,np.zeros((m,m)),np.zeros((m,n))))
        D3 = np.hstack((np.diag(mu),np.zeros((n,m)),np.diag(x)))
        
        DF = np.vstack((D1,D2,D3))
        z = -F(x,lam,mu) + np.hstack((np.zeros(n),np.zeros(m),sigma*nu*np.ones(n)))
        
        Delta = np.linalg.solve(DF,z)
        dx = Delta[:n]
        dlam = Delta[n:n+m]
        dmu = Delta[n+m:]
        
        #Problem 3
        mudmu = -mu.astype(float)/dmu
        alpha_mask = dmu < 0
        
        xdx = -x.astype(float)/dx
        delta_mask = dx < 0
        
        alpha_max = min(1,min(mudmu[alpha_mask]))
        delta_max = min(1,min(xdx[delta_mask]))
        
        alpha = min(1, .95*alpha_max)
        delta = min(1, .95*delta_max)
        
        x += delta*dx.reshape(x.shape)
        lam += alpha*dlam.reshape(lam.shape)
        mu += alpha*dmu.reshape(mu.shape)
        
        iters += 1
        nu = np.dot(x.T,mu)/float(n)
    return x,np.dot(c.T,x)

def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
#raise NotImplementedError("Problem 5 Incomplete")
    df = pd.read_csv(filename,delimiter=' ',header=None)
    data = df.values

    m,n = data.shape
    n -= 1
    c = np.zeros(3*m+2*(n+1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:,0]
    y[1::2] = data[:,0]
    x = data[:,1:]

    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)

    sol = interiorPoint(A,y,c,niter=10)[0]
    beta = float(sol[m:m+n] - sol[m+n:m+2*n])
    b = float(sol[m+2*n]-sol[m+2*n+1])

    first_col = data[:,0]
    sec_col = data[:,1]
    slope, intercept = linregress(sec_col,first_col)[:2]
    sl,int = np.polyfit(sec_col,first_col,1)
    domain = np.linspace(min(sec_col),max(sec_col),200)

    plt.subplot(211)
    plt.plot(sec_col,first_col, 'ko', alpha=0.8)
    plt.plot(domain,domain*beta+b, 'b',label="LAD")
    plt.legend(loc="upper right")
    plt.subplot(212)
    plt.plot(sec_col,first_col, 'ko', alpha=0.8)
    plt.plot(domain,domain*sl+int, 'r',label="Least Squares")
    plt.legend(loc="upper right")
    plt.show()
