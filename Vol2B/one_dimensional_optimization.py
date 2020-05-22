# one_dimensional_optimization.py
"""Volume 2B: 1-D Optimization.
    <Name>
    <Class>
    <Date>
    """

import numpy as np
from math import sqrt

# Problem 1
def golden_section(f, a, b, niter=10):
    """Find the minimizer of the unimodal function f on the interval [a,b]
        using the golden section search method.
        
        Inputs:
        f (function): unimodal scalar-valued function on R.
        a (float): left bound of the interval of interest.
        b (float): right bound of the interval of interest.
        niter (int): number of iterations to compute.
        
        Returns:
        the approximated minimizer (the midpoint of the final interval).
        """
#raise NotImplementedError("Problem 1 Incomplete")
    if niter == 0 :
        return (b+a)/2.
    else :
        p = (3-sqrt(5))/2.
        a_prime = a + p*(b-a)
        b_prime = a + (1-p)*(b-a)
        if f(a_prime) >= f(b_prime) :
            return golden_section(f,a_prime,b,niter-1)
        else :
            return golden_section(f,a,b_prime,niter-1)

# Problem 2
def bisection(df, a, b, niter=10):
    """Find the minimizer of the unimodal function with derivative df on the
        interval [a,b] using the bisection algorithm.
        
        Inputs:
        df (function): derivative of a unimodal scalar-valued function on R.
        a (float): left bound of the interval of interest.
        b (float): right bound of the interval of interest.
        niter (int): number of iterations to compute.
        """
#raise NotImplementedError("Problem 2 Incomplete")
    mid = (b+a)/2.
    if niter == 0 :
        return mid
    else :
        if df(mid) > 0 :
            return bisection(df,a,mid,niter-1)
        elif df(mid) < 0 :
            return bisection(df,mid,b,niter-1)
        else :
            return mid



# Problem 3
def newton1d(f, df, ddf, x, niter=10):
    """Minimize the scalar function f with derivative df and second derivative
        df using Newton's method.
        
        Parameters
        f (function): A twice-differentiable scalar-valued function on R.
        df (function): The first derivative of f.
        ddf (function): The second derivative of f.
        x (float): The initial guess.
        niter (int): number of iterations to compute.
        
        Returns:
        The approximated minimizer.
        """
#raise NotImplementedError("Problem 3 Incomplete")
    if np.isclose(df(x),0) or niter == 0:
        return x
    elif np.isclose(ddf(x),0) :
        raise ValueError("Division by zero occured.")
    else :
        return newton1d(f,df,ddf,x-df(x)/float(ddf(x)),niter-1)

# Problem 4
def secant1d(f, df, x0, x1, niter=10):
    """Minimize the scalar function f using the secant method.
        
        Inputs:
        f (function): A differentiable scalar-valued function on R.
        df (function): The first derivative of f.
        x0 (float): A first initial guess.
        x1 (float): A second initial guess.
        niter (int): number of iterations to compute.
        
        Returns:
        The approximated minimizer.
        """
#raise NotImplementedError("Problem 4 Incomplete")
    if np.isclose(df(x1),0) or niter == 0 :
        return x1
    else :
        r = float(df(x1)-df(x0))
        if np.isclose(r,0) :
            raise ValueError("Division by zero occured.")
        else :
            return secant1d(f,df,x1,x1-df(x1)*(x1-x0)/r,niter-1)


# Problem 5
def backtracking(f, slope, x, p, a=1, rho=.9, c=10e-4):
    """Do a backtracking line search to satisfy the Wolfe Conditions.
        Return the step length.
        
        Inputs:
        f (function): A scalar-valued function on R.
        slope (float): The derivative of f at x.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        a (float): Initial step length (set to 1 in Newton and quasi-Newton
        methods).
        rho (float): Parameter in (0,1).
        c (float): Parameter in (0,1).
        
        Returns:
        The computed step size.
        """
    raise NotImplementedError("Problem 5 Incomplete")
