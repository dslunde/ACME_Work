# cvxopt_intro.py
"""Volume 2B: CVXOPT
Darren Lund
Math 323
1/2/17
"""

from cvxopt import matrix, solvers
import numpy as np
from scipy import linalg as la


def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x + 2y          >= 3
                    2x + 10y + 3z   >= 10
                    x               >= 0
                    y               >= 0
                    z               >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    #raise NotImplementedError("Introductory problem not started.")
    c = matrix([2.,1.,3.])
    G = matrix([[-1.,-2.,-1.,0.,0.],[-2.,-10.,0.,-1.,0.],[0.,-3.,0.,0.,-1.]])
    h = matrix([-3.,-10.,0.,0.,0.])
    solvers.options['show_progress'] = False
    sol = solvers.lp(c,G,h)
    return np.ravel(sol['x']), sol['primal objective']


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray), without any slack variables u
        The optimal value (sol['primal objective'])
    """
    #raise NotImplementedError("L1 problem not started.")
    n = A.shape[1]
    G1 = np.hstack((-np.eye(n),np.eye(n)))
    G2 = np.hstack((-np.eye(n),-np.eye(n)))
    G = np.vstack((G1,G2))
    G = matrix(G.astype(float))
    c = matrix(np.hstack((np.ones(n),np.zeros(n))).astype(float))
    h = matrix(np.zeros_like(c).astype(float))
    A_pad = matrix(np.hstack((np.zeros_like(A),A)).astype(float))
    sol = solvers.lp(c,G,h,A_pad,matrix(b.astype(float)))
    return np.ravel(sol['x'][n:]),sol['primal objective']


def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    #raise NotImplementedError("Transportation problem not started.")
    c = matrix([4., 7., 6., 8., 8., 9])
    G = matrix(np.vstack((-1*np.eye(6),np.array([[0.,1.,0.,1.,0.,1.],[0.,-1.,0.,-1.,0.,-1.]]))))
    A = matrix(np.array([[1.,1.,0.,0.,0.,0.],
                         [0.,0.,1.,1.,0.,0.],
                         [0.,0.,0.,0.,1.,1.],
                         [1.,0.,1.,0.,1.,0.]]))
    h = matrix(np.hstack((np.zeros(6),np.array([8.,-8.]))))
    b = matrix([7., 2., 4., 5.])
    sol = solvers.lp(c, G, h, A, b)
    return np.ravel(sol['x']),sol['primal objective']


def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    #raise NotImplementedError("Quadratic minimization problem not started.")
    #1/2(ax^2+(b+d)xy+(c+h)xz+fy^2+(g+l)yz+mz^2)+sx+ry+tz
    #g(x, y, z) = 3/2x^2 + 2xy + xz + 2y^2 + 2yz + 3/2z^2 + 3x + z
    #a=3,b=d=2,c=h=1,f=4,g=l=2,m=3
    #s=3,r=0,t=1
    P = matrix(np.array([[3.,2.,1.],[2.,4.,2.],[1.,2.,3.]]))
    q = matrix(np.array([3.,0.,1.]).reshape(3,1))
    sol = solvers.qp(P,q)
    return np.ravel(sol['x']),sol['primal objective']

# Problem 5
def l2Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_2
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    #raise NotImplementedError("L2 problem not started.")
    n = A.shape[1]
    P = matrix(2*np.eye(n).astype(float))
    q = matrix(np.zeros(n).astype(float))
    sol = solvers.qp(P,q,A=matrix(A.astype(float)),b=matrix(b.astype(float)))
    return np.ravel(sol['x']),sol['primal objective']


def prob6():
    """Solve the allocation model problem in 'ForestData.npy'.
    Note that the first three rows of the data correspond to the first
    analysis area, the second group of three rows correspond to the second
    analysis area, and so on.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective']*-1000)
    """
    #raise NotImplementedError("Forest problem not started.")
    F = np.load('ForestData.npy')
    c = -matrix(F[:,3])
    A = np.zeros((7,21)).astype(float)
    for i in xrange(7) :
        A[i,i*3:(i+1)*3] = 1.
    A = matrix(A)
    b = F[:,1]
    b = matrix(b[(b!=0)].reshape(7,1).astype(float))
    G = matrix(-np.vstack((np.vstack((np.vstack((F[:,4].T,F[:,5].T)),F[:,6].T)),np.eye(21))))
    h = matrix(np.hstack(([-40000.,-5.,-70.*788.],np.zeros(21))))
    solvers.options['show_progress'] = False
    sol = solvers.lp(c,G,h,A,b)
    return np.ravel(sol['x']),-1000*sol['primal objective']
