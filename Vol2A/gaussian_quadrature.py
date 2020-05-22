# gaussian_quadrature.py
"""Volume 2 Lab 12: Gaussian Quadrature.
Darren Lund
AD&Opt Lab
12/08/16
"""

from math import sqrt
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as las
import scipy
import numpy as np

# Problem 1
def shift(f, a, b, plot=False):
    """Shift the function f on [a, b] to a new function g on [-1, 1] such that
    the integral of f from a to b is equal to the integral of g from -1 to 1.

    Inputs:
        f (function): a scalar-valued function on the reals.
        a (int): the left endpoint of the interval of integration.
        b (int): the right endpoint of the interval of integration.
        plot (bool): if True, plot f over [a,b] and g over [-1,1] in separate
            subplots.

    Returns:
        The new, shifted function.
    """
    #raise NotImplementedError("Problem 1 Incomplete")
    g = lambda u : f((b-a)/2.*u+(b+a)/2.)
    if plot :
        x = np.linspace(a,b,(a+b)/2.*100)
        u = np.linspace(-1,1,len(x))
        plt.subplot(121)
        plt.plot(x,f(x))
        plt.title('f')
        plt.subplot(122)
        plt.plot(u,g(u),label='g')
        plt.title('g')
        plt.show()
    return g


# Problem 2
def estimate_integral(f, a, b, points, weights):
    """Estimate the value of the integral of the function f over [a,b].

    Inputs:
        f (function): a scalar-valued function on the reals.
        a (int): the left endpoint of the interval of integration.
        b (int): the right endpoint of the interval of integration.
        points ((n,) ndarray): an array of n sample points.
        weights ((n,) ndarray): an array of n weights.

    Returns:
        The approximate integral of f over [a,b].
    """
    #raise NotImplementedError("Problem 2 Incomplete")
    g = shift(f,a,b)
    g_points = g(points)
    return (b-a)/2.*np.dot(weights.T,g_points)


# Problem 3
def construct_jacobi(gamma, alpha, beta):
    """Construct the Jacobi matrix."""
    #raise NotImplementedError("Problem 3 Incomplete")
    n = len(alpha)
    a = [-float(beta[i])/alpha[i] for i in xrange(n-1)]
    b = [sqrt(float(gamma[i+1])/(alpha[i]*alpha[i+1])) for i in xrange(n-1)]
    a.append(-float(beta[n-1])/alpha[n-1])
    b.insert(0,0)
    b.append(0)
    offsets = [-1,0,1]
    return sparse.spdiags([b[1:],a,b[:-1]],offsets,n,n).toarray()


# Problem 4
def points_and_weights(n):
    """Calculate the points and weights for a quadrature over [a,b] with n
    points.

    Returns:
        points ((n,) ndarray): an array of n sample points.
        weights ((n,) ndarray): an array of n weights.
    """
    #raise NotImplementedError("Problem 4 Incomplete")
    alpha = [float(2*i-1)/i for i in xrange(1,n+1)]
    beta = [0 for r in xrange(1,n+1)]
    gamma = [float(i-1)/i for i in xrange(1,n+1)]
    A = construct_jacobi(gamma,alpha,beta)
    points,e_vects = np.linalg.eig(A)
    weights = np.array([2*(e_vects[0,i]**2) for i in xrange(e_vects.shape[1])])
    weights2 = np.copy(weights)
    a = np.argsort(points)
    for i in xrange(weights.shape[0]) :
        weights2[a[i]] = weights[i]
    return np.sort(points),weights2



# Problem 5
def gaussian_quadrature(f, a, b, n):
    """Using the functions from the previous problems, integrate the function
    'f' over the domain [a,b] using 'n' points in the quadrature.
    """
    #raise NotImplementedError("Problem 5 Incomplete")
    p,w = points_and_weights(n)
    return estimate_integral(f,a,b,p,w)


# Problem 6
def normal_cdf(x):
    """Use scipy.integrate.quad() to compute the CDF of the standard normal
    distribution at the point 'x'. That is, compute P(X <= x), where X is a
    normally distributed random variable with mean = 0 and std deviation = 1.
    """
    #raise NotImplementedError("Problem 6 Incomplete")
    f = lambda x : np.exp(-x**2/2.)/(2*np.pi)**.5
    return scipy.integrate.quadrature(f,-5,x)[0]

def test() :
    s1 = 2 * sqrt(10. / 7.)
    points = np.array([-sqrt(5 + s1) / 3.,
                       -sqrt(5 - s1) / 3.,
                                       0.,
                        sqrt(5 - s1) / 3.,
                        sqrt(5 + s1) / 3.])
    s2 = 13 * sqrt(70)
    weights = np.array([(322 - s2) / 900.,
                        (322 + s2) / 900.,
                               128 / 225.,
                        (322 + s2) / 900.,
                        (322 - s2) / 900.])

    """
    f = lambda x : np.sin(x)
    g = lambda x : np.cos(x)
    print estimate_integral(f,-np.pi,np.pi,points,weights)
    print estimate_integral(g,-np.pi,np.pi,points,weights)

    p,w = points_and_weights(5)
    print np.allclose(p,points)
    print np.allclose(w,weights)
    """
