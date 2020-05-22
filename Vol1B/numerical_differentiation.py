# numerical_differentiation.py
"""Volume 1B: Numerical Differentiation.
Darren Lund
Math Analysis Lab
01/17/17
"""

from scipy import linalg as la
import numpy as np
import math

# Problem 1
def centered_difference_quotient(f, pts, h=1e-5):
    """Compute the centered difference quotient for function (f)
        given points (pts).
        
        Inputs:
        f (function): the function for which the derivative will be
        approximated.
        pts (array): array of values to calculate the derivative.
        
        Returns:
        An array of the centered difference quotient.
        """
    #raise NotImplementedError("Problem 1 Incomplete")
    return (f(pts+h)/2.-f(pts-h)/2.)/float(h)

# Problem 2
def calculate_errors(f,df,pts,h = 1e-5):
    """Compute the errors using the centered difference quotient approximation.
        
        Inputs:
        f (function): the function for which the derivative will be
        approximated.
        df (function): the function of the derivative
        pts (array): array of values to calculate the derivative
        
        Returns:
        an array of the errors for the centered difference quotient
        approximation.
        """
    #raise NotImplementedError("Problem 2 Incomplete")
    return abs(centered_difference_quotient(f,pts,h)-df(pts))

# Problem 3
def prob3():
    """Use the centered difference quotient to approximate the derivative of
        f(x)=(sin(x)+1)^x at x= pi/3, pi/4, and pi/6.
        Then compute the error of each approximation
        
        Returns:
        an array of the derivative approximations
        an array of the errors of the approximations
        """
    #raise NotImplementedError("Problem 3 Incomplete")
    pts = np.array([np.pi/3.,np.pi/4.,np.pi/6.])
    f = lambda x : (np.sin(x)+1)**x
    g = lambda x : f(x)*(np.log(np.sin(x)+1.)+x*np.cos(x)/(np.sin(x)+1.))
    return g(pts),calculate_errors(f,g,pts)

# Problem 4
def prob4():
    """Use centered difference quotients to calculate the speed v of the plane
        at t = 10 s
        
        Returns:
        (float) speed v of plane
        """
    #raise NotImplementedError("Problem 4 Incomplete")
    t = [0,1,2]
    rad = lambda x : x*np.pi/180
    a = rad(np.array([54.80,54.06,53.34]))
    b = rad(np.array([65.59,64.59,63.62]))
    dadt = (a[2]-a[0])/2.
    dbdt = (b[2]-b[0])/2.
    dxdt = 250./(np.sin(a[1]-b[1]))**2*(dadt*np.sin(2*b[1])-np.sin(2*a[1])*dbdt)
    dydt = 250./(np.sin(a[1]-b[1]))**2*(dadt*(-np.cos(2*b[1]))+dadt+np.cos(2*a[1])*dbdt-dbdt)
    print dadt
    print dbdt
    print dxdt
    print dydt
    return math.sqrt(dydt**2+dxdt**2)
#return dydt/dxdt


# Problem 5
def jacobian(f, n, m, pt, h=1e-5):
    """Compute the approximate Jacobian matrix of f at pt using the centered
        difference quotient.
        
        Inputs:
        f (function): the multidimensional function for which the derivative
        will be approximated.
        n (int): dimension of the domain of f.
        m (int): dimension of the range of f.
        pt (array): an n-dimensional array representing a point in R^n.
        h (float): a float to use in the centered difference approximation.
        
        Returns:
        (ndarray) Jacobian matrix of f at pt using the centered difference
        quotient.
        """
    #raise NotImplementedError("Problem 5 Incomplete")
    J = np.zeros((m,n))
    for i in xrange(n) :
        J[:,i] = (f(pt+h*np.eye(1.,n,i)[0])-f(pt-h*np.eye(1.,n,i)[0]))/(2.*h)
    return J


# Problem 6
def findError():
    """Compute the maximum error of jacobian() for the function
        f(x,y)=[(e^x)sin(y) + y^3, 3y - cos(x)] on the square [-1,1]x[-1,1].
        
        Returns:
        Maximum error of your jacobian function.
        """
    #raise NotImplementedError("Problem 6 Incomplete")
    x = np.linspace(-1,1,100)
    y = np.copy(x)
    max = -1.
    J = lambda x : np.array([[np.exp(x[0])*np.sin(x[1]),np.exp(x[0])*np.cos(x[1])+3*x[1]**2],[np.sin(x[0]),3]])
    f = lambda x : np.array([np.exp(x[0])*np.sin(x[1])+x[1]**3,3*x[1]-np.cos(x[0])])
    for i in xrange(len(x)) :
        for j in xrange(len(y)) :
            point = np.array([x[i],y[j]])
            Jeval = jacobian(f,2,2,point)
            error = la.norm(J(point)-Jeval)
            if error > max :
                max = error
    return max
