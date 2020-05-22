# newtons_method.py
"""Volume 1B: Newton's Method.
    Darren Lund
    Math. Analysis
    1/31/17
    """

from matplotlib import pyplot as plt
from numpy import linalg as la
import numpy as np
from math import sqrt

# Problem 1
def Newtons_method(f, x0, Df, iters=15, tol=1e-5, alpha=1):
    """Use Newton's method to approximate a zero of a function.
        
        Inputs:
        f (function): A function handle. Should represent a function
        from R to R.
        x0 (float): Initial guess.
        Df (function): A function handle. Should represent the derivative
        of f.
        iters (int): Maximum number of iterations before the function
        returns. Defaults to 15.
        tol (float): The function returns when the difference between
        successive approximations is less than tol.
        
        Returns:
        A tuple (x, converged, numiters) with
        x (float): the approximation for a zero of f.
        converged (bool): a Boolean telling whether Newton's method
        converged.
        numiters (int): the number of iterations computed.
        """
#raise NotImplementedError("Problem 1 Incomplete")
    count = 0
    conv = True
    while abs(f(x0)) >= tol and count < iters :
        count += 1
        try :
            x0 -= alpha*float(f(x0))/Df(x0)
        except ZeroDivisionError:
            conv = False
            break
    if count == iters :
        conv = False
    return (x0,conv,count)


# Problem 2.1
def prob2_1():
    """Plot f(x) = sin(x)/x - x on [-4,4].
        Return the zero of this function to 7 digits of accuracy.
        """
#raise NotImplementedError("Problem 2.1 Incomplete")
    f = lambda x : np.sin(x)/float(x) - x
    g = lambda x : np.cos(x)/float(x) - np.sin(x)/float(x**2) - 1
    x = np.linspace(-4,4,1000)
    y = np.zeros_like(x)
    for i in xrange(x.shape[0]) :
        try :
            y[i] = f(x[i])
        except ZeroDivisionError :
            y[i] = 1
    zero = Newtons_method(f,1.,g)
    plt.plot(x,y)
    plt.show()
    return zero[0]

# Problem 2.2
def prob2_2():
    """Return a string as to what happens and why during Newton's Method for
        the function f(x) = x^(1/3) where x_0 = .01.
        """
#raise NotImplementedError("Problem 2.2 Incomplete")
    f = lambda x : np.sign(x)*np.power(np.abs(x),1./3)
    Df = lambda x : 1./3/(x**2)**(1./3)
    zero = Newtons_method(f,.01,Df)
    print zero
    return "It diverged spectacularly, because the derivative is discontinuous at 0, and so it shoots x off towards infinity."


# Problem 3
def prob3():
    """Given P1[(1+r)**N1-1] = P2[1-(1+r)**(-N2)], if N1 = 30, N2 = 20,
        P1 = 2000, and P2 = 8000, use Newton's method to determine r.
        Return r.
        """
#raise NotImplementedError("Problem 3 Incomplete")
    N1 = 30
    N2 = 20
    P1 = 2000
    P2 = 8000
    f = lambda r : P1*((1+r)**N1-1)-P2*(1-(1+r)**(-N2))
    Df = lambda r : N1*P1*(1+r)**(N1-1)-N2*P2*(1+r)**(-N2-1)
    zero = Newtons_method(f,0.1,Df)
    return zero[0]


# Problem 4: Modify Newtons_method and implement this function.
def prob4():
    """Find an alpha < 1 so that running Newtons_method() on f(x) = x**(1/3)
        with x0 = .01 converges. Return the complete results of Newtons_method().
        """
#raise NotImplementedError("Problem 4 Incomplete")
    f = lambda x : np.sign(x)*np.power(np.abs(x),1./3)
    Df = lambda x : 1./3/(x**2)**(1./3)
    zero = Newtons_method(f,.01,Df,alpha=0.3333)
    return zero



# Problem 5: Implement Newtons_vector() to solve Bioremediation problem
def Newtons_vector(f, x0, Df, iters = 15, tol = 1e-5, alpha = 1):
    """Use Newton's method to approximate a zero of a vector valued function.
        
        Inputs:
        f (function): A function handle.
        x0 (list): Initial guess.
        Df (function): A function handle. Should represent the derivative
        of f.
        iters (int): Maximum number of iterations before the function
        returns. Defaults to 15.
        tol (float): The function returns when the difference between
        successive approximations is less than tol.
        alpha (float): Defaults to 1.  Allows backstepping.
        
        Returns:
        A tuple (x_values, y_values) where x_values and y_values are lists that contain the x and y value from each iteration of Newton's method
        """
#raise NotImplementedError("Problem 5.1 Incomplete")
    count = 0
    x = [np.copy(x0)]
    conv = True
    while la.norm(f(x0)) >= tol and count < iters :
        count += 1
        try :
            x0 -= alpha*np.dot(la.inv(Df(x0)),f(x0))
            x.append(np.copy(x0))
        except ZeroDivisionError:
            conv = False
            break
    if count != iters and conv:
        return x
    else :
        raise ValueError("Did not converge.")


def prob5():
    """Solve the system using Newton's method and Newton's method with
        backtracking
        """
#raise NotImplementedError("Problem 5.2 Incomplete")
    f = lambda x : np.array([5*x[0]*x[1]-x[0]*(1+x[1]),-x[0]*x[1]+(1-x[1])*(1+x[1])])
    Df = lambda x : np.array([[5*x[1]-(1+x[1]),5*x[0]-x[0]],[-x[1],-x[0]-2*x[1]]])
    x0 = [np.array([-0.2,-0.25]),np.array([0.2,0.25]),np.array([0.1,0.1])]
    colors = ['r','b','g']
    for j in xrange(len(x0)) :
        sol_list = Newtons_vector(f,x0[j],Df)
        x = [sol_list[i][0] for i in xrange(len(sol_list))]
        y = [sol_list[i][1] for i in xrange(len(sol_list))]
        plt.plot(x,y,colors[j]+"o-",linewidth=2.0)
    x = np.linspace(-6,8,1000)
    y = np.linspace(-6,6,1000)
    X,Y = np.meshgrid(x,y)
    z1 = 5*X*Y-X*(1+Y)
    z2 = -X*Y+(1-Y)*(1+Y)
#plt.contour(x,y,z1)
    plt.contour(x,y,z2)
    plt.axis([-6,8,-6,6])
    plt.show()


# Problem 6
def plot_basins(f, Df, roots, xmin, xmax, ymin, ymax, numpoints=1000, iters=15, colormap='brg'):
    """Plot the basins of attraction of f.
        
        INPUTS:
        f (function): Should represent a function from C to C.
        Df (function): Should be the derivative of f.
        roots (array): An array of the zeros of f.
        xmin, xmax, ymin, ymax (float,float,float,float): Scalars that define the domain
        for the plot.
        numpoints (int): A scalar that determines the resolution of the plot. Defaults to 100.
        iters (int): Number of times to iterate Newton's method. Defaults to 15.
        colormap (str): A colormap to use in the plot. Defaults to 'brg'.
        """
#raise NotImplementedError("Problem 6 Incomplete")
    plt.axis([xmin,xmax,ymin,ymax])
    x_real = np.linspace(xmin,xmax,numpoints)
    x_imag = np.linspace(xmin,xmax,numpoints)
    X_real, X_imag = np.meshgrid(x_real,x_imag)
    X_old = X_real + 1j*X_imag
    for i in xrange(iters) :
        X_new = X_old - f(X_old)/Df(X_old)
        X_old = X_new
    for i in xrange(numpoints) :
        for j in xrange(numpoints) :
            if X_new[i][j] not in roots :
                for k in xrange(len(roots)) :
                    if abs(X_new[i][j]-roots[k]) < 1e-5 :
                        X_new[i][j] = k
                        break
    plt.pcolormesh(X_real,X_imag,X_new,cmap=colormap)
    plt.show()

# Problem 7
def prob7():
    """Run plot_basins() on the function f(x) = x^3 - 1 on the domain
        [-1.5,1.5]x[-1.5,1.5].
        """
#raise NotImplementedError("Problem 7 Incomplete")
    roots = np.array([1,-1./2+sqrt(3)/2j,-1./2-sqrt(3)/2j])
    f = lambda x : x**3-1
    Df = lambda x : 3*x**2
    plot_basins(f,Df,roots,-1.5,1.5,-1.5,1.5,colormap='brg')
