# montecarlo_integration.py
"""Volume 1B: Monte Carlo Integration.
    Darren Lund
    I'M BATMAN!
    2/14/17
    """

from matplotlib import pyplot as plt
import numpy as np
from math import exp
from math import sqrt
from scipy import stats

# Problem 1
def prob1(N=10000):
    """Return an estimate of the volume of the unit sphere using Monte
        Carlo Integration.
        
        Input:
        N (int, optional) - The number of points to sample. Defaults
        to 10000.
        
        """
#raise NotImplementedError("Problem 1 Incomplete")
    points = np.random.rand(3,N)
    points = points*2-1
    pDist = np.linalg.norm(points,axis=0)
    inCircle = np.count_nonzero(pDist < 1)
    return 8*inCircle/float(N)


# Problem 2
def prob2(f, a, b, N=10000):
    """Use Monte-Carlo integration to approximate the integral of
        1-D function f on the interval [a,b].
        
        Inputs:
        f (function) - Function to integrate. Should take scalar input.
        a (float) - Left-hand side of interval.
        b (float) - Right-hand side of interval.
        N (int, optional) - The number of points to sample in
        the Monte-Carlo method. Defaults to 10000.
        
        Returns:
        estimate (float) - The result of the Monte-Carlo algorithm.
        
        Example:
        >>> f = lambda x: x**2
        >>> # Integral from 0 to 1. True value is 1/3.
        >>> prob2(f, 0, 1)
        0.3333057231764805
        """
#raise NotImplementedError("Problem 2 Incomplete")
    points = np.random.rand(1,N)[0]*(b-a)+a
    return (b-a)*sum(f(points))/float(N)


# Problem 3
def prob3(f, mins, maxs, N=10000):
    """Use Monte-Carlo integration to approximate the integral of f
        on the box defined by mins and maxs.
        
        Inputs:
        f (function) - The function to integrate. This function should
        accept a 1-D NumPy array as input.
        mins (1-D np.ndarray) - Minimum bounds on integration.
        maxs (1-D np.ndarray) - Maximum bounds on integration.
        N (int, optional) - The number of points to sample in
        the Monte-Carlo method. Defaults to 10000.
        
        Returns:
        estimate (float) - The result of the Monte-Carlo algorithm.
        
        Example:
        >>> f = lambda x: np.hypot(x[0], x[1]) <= 1
        >>> # Integral over the square [-1,1] x [-1,1]. True value is pi.
        >>> mc_int(f, np.array([-1,-1]), np.array([1,1]))
        3.1290400000000007
        """
#raise NotImplementedError("Problem 3 Incomplete")
    points = np.multiply(np.random.rand(mins.shape[0],N).T,(maxs-mins))+mins
    values = np.apply_along_axis(f,1,points)
    vol = np.prod(maxs-mins)
    return vol*sum(values)/float(N)


# Problem 4
def prob4():
    """Integrate the joint normal distribution.
        
        Return your Monte Carlo estimate, SciPy's answer, and (assuming SciPy is
        correct) the relative error of your Monte Carlo estimate.
        """
#raise NotImplementedError("Problem 4 Incomplete")
    mins = np.array([-1.5,0,0,0])
    maxs = np.array([0.75,1,0.5,1])
    f = lambda x : exp(-np.dot(x.T,x)/2.)/(2*np.pi)**2
    mine = prob3(f,mins,maxs,50000)
    means = np.zeros(4)
    covs = np.eye(4)
    value,inform = stats.mvn.mvnun(mins,maxs,means,covs)
    return mine,value,abs(mine-value)


# Problem 5
def prob5(numEstimates=50):
    """Plot the error of Monte Carlo Integration."""
   #raise NotImplementedError("Problem 5 Incomplete")
    V = 4./3*np.pi
    N = [50,100,500]
    err = []
    for i in xrange(50) :
        N.append((i+1)*1000)
    for n in N :
        vol = []
        for i in xrange(numEstimates):
            vol.append(prob1(n))
        err.append(abs(V-sum(vol)/float(numEstimates)))
    f = [1./sqrt(x) for x in N]
    plt.plot(N,err,'b-')    
    plt.plot(N,f,'r-')
    plt.show()
