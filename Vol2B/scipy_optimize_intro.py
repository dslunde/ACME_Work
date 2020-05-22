# scipy_optimize_intro.py
"""Volume 2B: Optimization with Scipy
    <Name>
    <Class>
    <Date>
    """
import scipy.optimize as opt
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from blackbox_function import blackbox

# Problem 1
def prob1():
    """Use the minimize() function in the scipy.optimize package to find the
        minimum of the Rosenbrock function (scipy.optimize.rosen) using the
        following methods:
        Nelder-Mead
        CG
        BFGS
        Use x0 = np.array([4., -2.5]) for the initial guess for each test.
        
        For each method, print whether it converged, and if so, print how many
        iterations it took.
        """
#return NotImplementedError("Problem 1 not implemented")
    x0 = np.array([4.,-2.5])
    NelderMead = opt.minimize(opt.rosen, x0, method='Nelder-Mead', hess=opt.rosen_hess,jac=opt.rosen_der)
    CG = opt.minimize(opt.rosen, x0, method='CG', hess=opt.rosen_hess,jac=opt.rosen_der)
    BFGS = opt.minimize(opt.rosen, x0, method='BFGS', hess=opt.rosen_hess,jac=opt.rosen_der)
    solutions = [NelderMead,CG,BFGS]
    sols = ["Nelder-Mead","CG","BFGS"]
    for i in xrange(len(solutions)) :
        if solutions[i]['success'] :
            print str(sols[i]) + " Converged; " + str(solutions[i]['nit'])
        else :
            print "Didn't converge."

# Problem 2
def prob2():
    """Minimize the function blackbox() in the blackbox_function module,
        selecting the appropriate method of scipy.optimize.minimize() for this
        problem.  Do not pass your method a derivative. You may need to test
        several methods and determine which is most appropriate.
        
        The blackbox() function returns the length of a piecewise-linear curve
        between two fixed points: the origin, and the point (40,30).
        It accepts a one-dimensional ndarray} of length m of y-values, where m
        is the number of points of the piecewise curve excluding endpoints.
        These points are spaced evenly along the x-axis, so only the y-values
        of each point are passed into blackbox().
        
        Once you have selected a method, select an initial point with the
        provided code.
        
        Plot your initial curve and minimizing curve together on the same
        plot, including endpoints. Note that this will require padding your
        array of internal y-values with the y-values of the endpoints, so
        that you plot a total of 20 points for each curve.
        """
#return NotImplementedError("Problem 2 not implemented")
    y_initial = 30*np.random.random_sample(18)
    """
    methods = ['Nelder-Mead','Powell','CG','BFGS','L-BFGS-B','TNC','COBYLA','SLSQP']
    for x in methods :
        sol = opt.minimize(blackbox,y_initial,method=x)
        if sol['success'] :
            print x, sol['fun'], sol['nit']
    """
    sol = opt.minimize(blackbox,y_initial,method='BFGS')
    x = np.linspace(0,40,20)
    y = np.hstack((np.array([0]),sol['x'],np.array([30])))
    plt.plot(x,np.hstack((np.array([0]),y_initial,np.array([30]))),'.-b')
    plt.plot(x,y,'.-g')
    plt.show()

# Problem 3
def prob3():
    """Explore the documentation on the function scipy.optimize.basinhopping()
        online or via IPython. Use it to find the global minimum of the multmin()
        function given in the lab, with initial point x0 = np.array([-2, -2]) and
        the Nelder-Mead algorithm. Try it first with stepsize=0.5, then with
        stepsize=0.2.
        
        Plot the multimin function and minima found using the code provided.
        Print statements answering the following questions:
        Which algorithms fail to find the global minimum?
        Why do these algorithms fail?
        
        Finally, return the global minimum.
        """
#return NotImplementedError("Problem 3 not implemented")
    x0 = np.array([-2,-2])
    r_eq = lambda x : np.sqrt((x[0]+1)**2 + x[1]**2)
    multimin = lambda x : r_eq(x)**2*(1+np.sin(4*r_eq(x))**2)
    sol = opt.basinhopping(multimin, x0, stepsize=0.5, minimizer_kwargs={'method':'nelder-mead'})
    xdomain = np.linspace(-3.5,1.5,70)
    ydomain = np.linspace(-2.5,2.5,60)
    X,Y = np.meshgrid(xdomain,ydomain)
    Z = multimin((X,Y))
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_wireframe(X, Y, Z, linewidth=.5, color='c')
    ax1.scatter((-2,sol['x'][0]), (-2,sol['x'][1]), (multimin(x0),sol['fun']))
    plt.show()
    print "Step sizes of 0.2 don't work because it isn't far enough to jump out of the basin it's currently stuck in."
    return sol['fun']


# Problem 4
def prob4():
    """Find the roots of the function
        [       -x + y + z     ]
        f(x,y,z) = [  1 + x^3 - y^2 + z^3 ]
        [ -2 - x^2 + y^2 + z^2 ]
        
        Returns the values of x,y,z as an array.
        """
# return NotImplementedError("Problem 4 not implemented")
    f = lambda x : np.array([-x[0]+x[1]+x[2],1+x[0]**3-x[1]**2+x[2]**3,-2-x[0]**2+x[1]**2+x[2]**2])
    jac = lambda x : np.array([[-1,1,1],[3*x[0]**2,-2*x[1],3*x[2]**2],[-2*x[0],2*x[1],2*x[2]]])
    sol = opt.root(f, [0, 0, 0], jac=jac, method='hybr')
    return np.array(sol['x'])


# Problem 5
def prob5():
    """Use the scipy.optimize.curve_fit() function to fit a curve to
        the data found in `convection.npy`. The first column of this file is R,
        the Rayleigh number, and the second column is Nu, the Nusselt number.
        
        The fitting parameters should be c and beta, as given in the convection
        equations.
        
        Plot the data from `convection.npy` and the curve generated by curve_fit.
        Return the values c and beta as an array.
        """
#return NotImplementedError("Problem 5 not implemented")
    data = np.load('convection.npy')
    f = lambda R,c,b : c*R**b
    fitted = opt.curve_fit(f,data[4:,0],data[4:,1])
    x = np.linspace(data[4,0],data[-1,0],1000)
    y = f(x,fitted[0][0],fitted[0][1])
    plt.loglog(data[:,0],data[:,1],'.k')
    plt.loglog(x,y,'-b')
    plt.show()
    return fitted[0]
