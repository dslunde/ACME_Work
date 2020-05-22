# montecarlo_sampling.py
"""Volume 1B: Monte Carlo 2 (Importance Sampling).
    Darren Lund
    Batman Returns
    3/7/17
    """

from matplotlib import pyplot as plt
from scipy import stats
from math import sqrt
import numpy as np

# Problem 1
def prob1(n):
    """Approximate the probability that a random draw from the standard
        normal distribution will be greater than 3."""
#raise NotImplementedError("Problem 1 Incomplete")
    if n == 0 :
        raise ValueError("Sampling 0 points is not defined.")
    total = 0
    for i in xrange(n) :
        if np.random.normal() > 3 :
            total += 1
    return float(total)/n

# Problem 2
def prob2():
    """Answer the following question using importance sampling:
        A tech support hotline receives an average of 2 calls per
        minute. What is the probability that they will have to wait
        at least 10 minutes to receive 9 calls?
        Returns:
        IS (array) - an array of estimates using
        [5000, 10000, 15000, ..., 500000] as number of
        sample points."""
    #raise NotImplementedError("Problem 2 Incomplete")
    h = lambda x : x > 10
    f = lambda x : stats.gamma(a=9,scale=.5).pdf(x)
    g = lambda x : stats.norm(loc=11,scale=1).pdf(x)
    MC_est = []
    for N in xrange(5000,505000,5000) :
        X = np.random.normal(11,scale=1,size=N)
        MC_est.append(1./N*np.sum(h(X)*f(X)/g(X)))
    return np.array(MC_est)

# Problem 3
def prob3():
    """Plot the errors of Monte Carlo Simulation vs Importance Sampling
        for the prob2()."""
#raise NotImplementedError("Problem 3 Incomplete")
    n = np.linspace(5000,500000,100)
    h = lambda x : x > 10
    MC_estimates = []
    for N in xrange(5000,505000,5000):
        X = np.random.gamma(9,scale=0.5,size=N)
        MC = 1./N*np.sum(h(X))
        MC_estimates.append(MC)
    MC_estimates = np.array(MC_estimates)
    MC_est = prob2()
    exact = np.array([1-stats.gamma(a=9,scale=0.5).cdf(10)]*n.shape[0])
    print n.shape,MC_estimates.shape
    plt.plot(n,abs(MC_estimates-exact),'r-',label="Monte Carlo")
    plt.plot(n,abs(MC_est-exact),'b-',label="Importance")
    plt.title("Error or Approx.")
    plt.legend(loc=1)
    plt.show()

# Problem 4
def prob4():
    """Approximate the probability that a random draw from the
        multivariate standard normal distribution will be less than -1 in
        the x-direction and greater than 1 in the y-direction."""
#raise NotImplementedError("Problem 4 Incomplete")
    h = lambda x : x[0] < -1 and x[1] > 1
    f = lambda x : stats.multivariate_normal.pdf(x,mean=np.array([0,0]),cov=np.eye(2))
    g = lambda x : stats.multivariate_normal.pdf(x,mean=np.array([-1,1]),cov=np.eye(2))
    X = np.random.multivariate_normal(mean=np.array([-1,1]),cov=np.eye(2),size=10000)
    return 1./10000*np.sum(np.apply_along_axis(h,1,X)*np.apply_along_axis(f,1,X)/np.apply_along_axis(g,1,X))
