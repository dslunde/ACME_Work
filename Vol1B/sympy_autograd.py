# sympy_autograd.py
"""Volume 1B: Differentiation 2 (SymPy and Autograd).
    Darren Lund
    White Cohort
    1/24/17
    """

from autograd import grad
from autograd import jacobian
from math import sqrt
import sympy as sy
import numerical_differentiation as ndif
import numpy as np
import autograd.numpy as anp
import time

# Problem 1
def myexp(n):
    """Compute e to the nth digit.
        
        Inputs:
        n (integer): n decimal places to calculate e.
        
        Returns:
        approximation (float): approximation of e.
        """
#raise NotImplementedError("Problem 1 Incomplete")
    total = sy.Rational(0,1)
    term = 1
    bound = sy.Rational(1,10)**(n+1)
    i = 0
    while bound <= term :
        term = sy.Rational(1,sy.factorial(i))
        total += term
        i += 1
    return sy.Float(total,n)


# Problem 2
def prob2():
    """Solve y = e^x + x for x.
        
        Returns:
        the solution (list).
        """
#raise NotImplementedError("Problem 2 Incomplete")
    x,y = sy.symbols('x,y')
    expr = sy.exp(x)+x-y
    return sy.solve(expr,x)


# Problem 3
def prob3():
    """Compute the integral of sin(x^2) from 0 to infinity.
        
        Returns:
        the integral value (float).
        """
#raise NotImplementedError("Problem 3 Incomplete")
    x = sy.Symbol('x')
    return sy.integrate(sy.sin(x**2), (x,0,sy.oo))


# Problem 4
def prob4():
    """Calculate the derivative of e^sin(cos(x)) at x = 1.
        Time how long it takes to compute the derivative using SymPy as well as
        centered difference quotients.
        Calculate the error for each approximation.
        
        Print the time it takes to compute and the error for both SymPy and
        centered difference quotients.
        
        Returns:
        SymPy approximation (float)
        """
#raise NotImplementedError("Problem 4 Incomplete")
    x = sy.Symbol('x')
    start = time.time()
    sy_eval = sy.Derivative(sy.exp(sy.sin(sy.cos(x)))).doit().subs({x:1}).evalf()
    end = time.time()
    sy_time = end-start
    f = lambda x : np.exp(np.sin(np.cos(x)))
    start = time.time()
    cdif_eval = ndif.centered_difference_quotient(f,1)
    end = time.time()
    cdif_time = end-start
    approx = np.sin(1)*np.cos(np.cos(1))*(-np.exp(np.sin(np.cos(1))))
    print "Sympy    | Time: %f ; Error: %f" % (sy_time,abs(sy_eval-approx))
    print "Cent.Dif.| Time: %f ; Error: %f" % (cdif_time,abs(cdif_eval-approx))
    return sy_eval


# Problem 5
def prob5():
    """Solve the differential equation when x = 1.
        
        Returns:
        Solution when x = 1.
        """
#raise NotImplementedError("Problem 5 Incomplete")
    x = sy.Symbol('x')
    y = sy.Function('y')
    expr = sy.Eq(y(x).diff(x,6)+3*y(x).diff(x,4)+3*y(x).diff(x,2)+y(x) , x**10*sy.exp(x)+x**11*sy.sin(x)+x**12*sy.exp(x)*sy.sin(x)-x**13*sy.cos(2*x)+x**14*sy.exp(x)*sy.cos(3*x))
    return sy.dsolve(expr).subs({x:1}).evalf


# Problem 6
def prob6():
    """Compute the derivative of ln(sqrt(sin(sqrt(x)))) at x = pi/4.
        Times how long it take to compute using SymPy, autograd, and centered
        difference quotients. Compute the error of each approximation.
        
        Print the time
        Print the error
        
        Returns:
        derviative (float): the derivative computed using autograd.
        """
#raise NotImplementedError("Problem 6 Incomplete")
    x = sy.Symbol('x')
    start = time.time()
    sy_eval = sy.Derivative(sy.log(sy.sqrt(sy.sin(sy.sqrt(x))))).doit().subs({x:1}).evalf()
    end = time.time()
    sy_time = end-start
    g = lambda x : anp.log(anp.sqrt(anp.sin(anp.sqrt(x))))
    start = time.time()
    auto_eval = grad(g)(anp.pi/4)
    end = time.time()
    auto_time = end-start
    f = lambda x : np.log(sqrt(np.sin(sqrt(x))))
    start = time.time()
    cdif_eval = ndif.centered_difference_quotient(f,1)
    end = time.time()
    cdif_time = end-start
    approx = 1./(np.tan(sqrt(np.pi)/2.)*2.*sqrt(np.pi))
    print "Sympy    | Time: %f ; Error: %f" % (sy_time,abs(sy_eval-approx))
    print "Autograd | Time: %f ; Error: %f" % (auto_time,abs(auto_eval-approx))
    print "Cent.Dif.| Time: %f ; Error: %f" % (cdif_time,abs(cdif_eval-approx))
    return auto_eval

# Problem 7
def prob7():
    """Computes Jacobian for the function
        f(x,y)=[(e^x)sin(y) + y^3, 3y - cos(x)]
        Time how long it takes to compute the Jacobian using SymPy and autograd.
        
        Print the times.
        
        Returns:
        Jacobian (array): jacobian found using autograd at (x,y) = (1,1)
        """
#raise NotImplementedError("Problem 7 Incomplete")
    f = lambda x : anp.array([sy.exp(x[0])*sy.sin(x[1])+x[1]**3,3*x[1]-sy.cos(x[0])])
    start = time.time()
    auto_eval = jacobian(f)
    #return auto_eval
    #raise ValueError("I want to stop.")
    auto_eval = auto_eval(anp.array([1.,1.]))
    end = time.time()
    auto_time = end-start
    X = sy.Matrix([sy.exp(x[0])*sy.sin(x[1])+x[1]**3,3*x[1]-sy.cos(x[0])])
    Y = sy.Matrix([x,y])
    start = time.time()
    jX = X.jacobian(Y).subs({x:1,y:1}).evalf()
    end = time.time()
    sy_time = end-start
    print "Sympy    | Time: %f" % (sy_time)
    print "Autograd | Time: %f" % (auto_time)
    return auto_eval
