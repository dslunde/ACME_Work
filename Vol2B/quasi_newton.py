# quasi_newton.py
"""Volume 2: Quasi-Newton Methods
Darren Lund
How to be BATMAN
2/28/17
"""

import time
import numpy as np
from math import exp
from scipy.optimize import leastsq
from matplotlib import pyplot as plt

years1 = np.arange(8)
pop1 = np.array([3.929, 5.308, 7.240, 9.638, 12.866,17.069, 23.192, 31.443])
years2 = np.arange(16)
pop2 = np.array([3.929, 5.308, 7.240, 9.638, 12.866,17.069, 23.192, 31.443, 38.558, 50.156,
                     62.948, 75.996, 91.972, 105.711, 122.775,131.669])

def newton_ND(J, H, x0, niter=10, tol=1e-5):
    """
    Perform Newton's method in N dimensions.

    Inputs:
        J (function): Jacobian of the function f for which we are finding roots.
        H (function): Hessian of f.
        x0 (float): The initial guess.
        niter (int): Number of iterations to compute.
        tol (float): Stopping criteria for iterations.

    Returns:
        The approximated root and the number of iterations it took.
    """
    #return NotImplementedError("Problem 1 Incomplete")
    iter = niter
    while iter > 0 and np.linalg.norm(J(x0)) >= tol :
        x0 -= np.dot(np.linalg.inv(H(x0)),J(x0).T)
        iter -= 1
    return (x0,niter-iter)

def broyden_ND(J, H, x0, niter=20, tol=1e-5):
    """
    Perform Broyden's method in N dimensions.

    Inputs:
        J (function): Jacobian of the function f for which we are finding roots.
        H (function): Hessian of f.
        x0 (float): The initial guess.
        niter (int): Number of iterations to compute.
        tol (float): Stopping criteria for iterations.

    Returns:
        The approximated root and the number of iterations it took.
    """
#return NotImplementedError("Problem 2 Incomplete")
    A = H(x0)
    iter = niter
    while iter > 0 and np.linalg.norm(J(x0)) >= tol :
        if iter != niter :
            yk = J(x1).T - J(x0).T
            sk = x1-x0
            A += np.dot((yk-np.dot(A,sk))/np.linalg.norm(sk)**2,sk.T)
            x0 = x1
        x1 = x0 - np.dot(np.linalg.inv(A),J(x0).T)
        iter -= 1
    return (x1,niter-iter)


def BFGS(J, H, x0, niter=10, tol=1e-6):
    """
    Perform BFGS in N dimensions.

    Inputs:
        J (function): Jacobian of objective function.
        H (function): Hessian of objective function.
        x0 (float): The initial guess.
        niter (int): Number of iterations to compute.
        tol (float): Stopping criteria for iterations.

    Returns:
        The approximated root and the number of iterations it took.
    """
#return NotImplementedError("Problem 3 Incomplete")
    A = H(x0)
    iter = niter
    while iter > 0 and np.linalg.norm(J(x0)) >= tol :
        if iter != niter :
            yk = J(x1).T - J(x0).T
            sk = x1-x0
            A += np.dot(yk,yk.T)/np.dot(yk.T,sk) - np.dot(np.dot(A,sk),np.dot(sk.T,A))/(np.dot(sk.T,np.dot(A,sk)))
            x0 = x1
        x1 = x0 - np.dot(np.linalg.inv(A),J(x0).T)
        iter -= 1
    return (x1,niter-iter)

def prob4():
    """
    Compare the performance of Newton's, Broyden's, and modified Broyden's
    methods on the following functions:
        f(x,y) = 0.26(x^2 + y^2) - 0.48xy
        f(x,y) = sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1
    """
#return NotImplementedError("Problem 4 Incomplete")
    f = lambda x : 0.26*(x[0]**2+x[1]**2)-0.48*x[0]*x[1]
    g = lambda x : np.sin(x[0]+x[1]) + (x[0]-x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1
    Jf = lambda x : np.array([0.52*x[0]-0.48*x[1],0.52*x[1]-0.48*x[0]])
    Jg = lambda x : np.array([np.cos(x[0]+x[1])+2*(x[0]-x[1])-1.5,np.cos(x[0]+x[1])-2*(x[0]-x[1])+2.5])
    Hf = lambda x : np.array([[0.52,-0.48],[-0.48,0.52]])
    Hg = lambda x : np.array([[-np.sin(x[0]+x[1])+2,-np.sin(x[0]+x[1])-2],[-np.sin(x[0]+x[1])-2,-np.sin(x[0]+x[1])+2]])
    funcs = [(f,Jf,Hf),(g,Jg,Hg)]
    func_names = ['0.26(x^2 + y^2) - 0.48xy','sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1']
    times = []
    iters = []
    for z in funcs :
        start1 = time.time()
        newt = newton_ND(z[1],z[2],np.ones(2))
        end1 = time.time()

        start2 = time.time()
        broy = broyden_ND(z[1],z[2],np.ones(2))
        end2 = time.time()

        start3 = time.time()
        bfgs = BFGS(z[1],z[2],np.ones(2))
        end3 = time.time()

        times.append((end1-start1,end2-start2,end3-start3))
        iters.append((newt[1],broy[1],bfgs[1]))
    for i in xrange(2) :
        print func_names[i] + " :"
        print "Newton  - Time/iter: " + str(times[i][0]/iters[i][0]) + " ; Iters: " + str(iters[i][0])
        print "Broyden - Time/iter: " + str(times[i][1]/iters[i][1]) + " ; Iters: " + str(iters[i][1])
        print "BFGS    - Time/iter: " + str(times[i][2]/iters[i][2]) + " ; Iters: " + str(iters[i][2]) + "\n"


def gauss_newton(J, r, x0, niter=10):
    """
    Solve a nonlinear least squares problem with Gauss-Newton method.

    Inputs:
        J (function): Jacobian of the objective function.
        r (function): Residual vector.
        x0 (float): The initial guess.
        niter (int): Number of iterations to compute.

    Returns:
        The approximated root.
    """
#return NotImplementedError("Problem 5 Incomplete")
    iter = niter
    while iter > 0 and np.linalg.norm(J(x0)) > 0 :
        x0 -= np.dot(np.linalg.inv(np.dot(J(x0).T,J(x0))),np.dot(J(x0).T,r(x0)))
        iter -= 1
    return x0

def model(x,t) :
    return x[0]*np.exp(x[1]*(t+x[2]))

def model2(x,t) :
    return float(x[0])/(1+np.exp(-x[1]*(t+x[2])))

def res(x) :
    return model(x,years1) - pop1

def res2(x) :
    return model2(x,years2) - pop2

def jac(x) :
    ans = np.empty((t.shape[0],3))
    ans[:,0] = exp(x[1]*(t+x[2]))
    ans[:,1] = (t+x[2])*x[0]*exp(x[1]*(t+x[2]))
    ans[:,2] = x[1]*x[0]*exp(x[1]*(t+x[2]))
    return ans

def jac2(x) :
    ans = np.empty((t.shape[0],3))
    ans[:,0] = 1./(1+nd_exp(-x[1]*(t+x[2])))
    ans[:,1] = float(x[0])/(1+nd_exp(-x[1]*(t+x[2])))**2*(t+x[2])*nd_exp(-x[1]*(t+x[2]))
    ans[:,2] = float(x[0])/(1+nd_exp(-x[1]*(t+x[2])))**2*x[1]*nd_exp(-x[1]*(t+x[2]))
    return ans

def objective(x) :
    return .5*(res(x)**2).sum()

def objective2(x) :
    return .5*(res2(x)**2).sum()

def grad(x) :
    return jac(x).T.dot(res(x))

def grad2(x) :
    return jac2(x).T.dot(res(x))

def prob6():
    """
    Compare the least squares regression with 8 years of population data and 16
    years of population data.
    """
#return NotImplementedError("Problem 6 Incomplete")
    pops = [pop1,pop2]
    years = [years1,years2]
    x0 = [np.array([150,.4,2.5]),np.array([150,.4,-15])]
        
    t = years[0]
    y = pops[0]
    x = leastsq(res,x0[0])
    plt.plot(t,model(x[0],t))
    plt.plot(t,y,'r+',linewidth=5)
    plt.title("First 8")
    plt.show()

    t = years[1]
    y = pops[1]
    plt.plot(t,model(x[0],t))
    plt.plot(t,y,'r+',linewidth=5)
    plt.title("All 16")
    plt.show()

    x = leastsq(res2,x0[1])
    plt.plot(t,model2(x[0],t),'b')
    plt.plot(t,y,'r+',linewidth=5)
    plt.title("Logistic Model")
    plt.show()

